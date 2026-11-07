"""2-D WCSPH numerical wave tank: makers, sponge layer, probes.

Scope. This module adds the *wave* object to the solver: a rectangular tank of
still water of depth ``h`` is driven by a piston or flap paddle at the left end,
the train propagates along the WCSPH free surface, and the right end is closed by
a sponge layer that damps it instead of reflecting it. Wave height is measured by
column-integral probes, so the elevation is a material-volume integral rather
than the ``y`` of the topmost particle.

The bulk-flow solvers are the existing ones --
``tensorsph.droplet_impact.rhs`` (density summation, Tait equation of state,
symmetrised pressure force, regularised laminar viscosity) and the static-wall
model with extrapolated wall pressure. What this module adds is the moving paddle
(rigid translation/rotation of wall particles between steps), the sponge damping,
the probe estimators, and the analytical wave anchors they are measured against.

Linear wave anchors
-------------------
``angular_frequency`` / ``wave_celerity`` / ``wave_number`` are the finite-depth
dispersion relation ``omega^2 = g k tanh(k h)`` and its inverse.

``piston_stroke_for_height`` and ``flap_stroke_for_height`` invert the classical
wavemaker transfer functions. Both follow from one integral: for a paddle whose
horizontal displacement is ``X(z) sin(omega t)`` on ``-h <= z <= 0``, matching the
paddle velocity to the propagating mode gives

    H = (4 sinh(kh) / (2kh + sinh 2kh)) * k * int_{-h}^{0} X(z) cosh(k(z+h)) dz

With ``X = S/2`` (piston, stroke ``S`` peak-to-peak) this is Havelock's
``H/S = 4 sinh^2(kh)/(2kh + sinh 2kh)``, which tends to ``kh`` for ``kh -> 0``.
With ``X = (S/2)(1 + z/h)`` (flap hinged on the floor) it is
``H/S = (4 sinh(kh)/(2kh + sinh 2kh)) (sinh(kh) - (cosh(kh) - 1)/kh)``.

What is not claimed: the transfer function is used only to *size* the stroke. The
measured wave height is reported against it, so a systematic offset is visible
instead of assumed away.

Probe estimator
---------------
The elevation at a probe is the material volume of the particles in a column
window of width ``W``:

    eta_p(t) = sum_{|x_i - x_p| < W/2} (m_i / rho0) / W - h

For a lattice this is exact in the mean (each particle carries ``dx^2`` of
volume), it is smooth in time -- unlike the elevation of the topmost particle,
which jumps by one particle spacing -- and it is insensitive to a single particle
leaving the surface. ``W`` is a few particle spacings, far below the shortest
wavelength used here (``lambda >= 2h``).

Sponge layer
------------
Velocity damping ``v <- (1 - a(x) dt) v`` with ``a(x) = A omega ((x - x_s)/(L -
x_s))^2`` over the last ``sponge_length`` of the tank. Velocity only: the density
is obtained by summation, so it cannot be relaxed towards a target without
breaking mass conservation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .droplet_impact import (
    Config,
    hydrostatic_density,
    lattice_mass_unit,
    resolve_device,
    rhs,
    tait_pressure,
)
from .neighbor_search import PairListCache

GRAVITY = 9.81


# --------------------------------------------------------------------------- #
# Linear wave anchors
# --------------------------------------------------------------------------- #

def angular_frequency(k: float, depth: float, gravity: float = GRAVITY) -> float:
    """Linear dispersion relation ``omega = sqrt(g k tanh(k h))``."""
    if k <= 0.0 or depth <= 0.0 or gravity <= 0.0:
        raise ValueError("k, depth and gravity must be positive")
    return math.sqrt(gravity * k * math.tanh(k * depth))


def wave_celerity(k: float, depth: float, gravity: float = GRAVITY) -> float:
    """Phase speed ``c = omega / k = sqrt(g tanh(kh) / k)``."""
    return angular_frequency(k, depth, gravity) / k


def wave_number(omega: float, depth: float, gravity: float = GRAVITY,
                tolerance: float = 1.0e-14) -> float:
    """Wave number of the linear wave of angular frequency ``omega``.

    Bisection on the monotone function ``g k tanh(kh) - omega^2``. The bracket is
    grown until it contains the root, so the answer does not depend on a guess.
    """
    if omega <= 0.0 or depth <= 0.0 or gravity <= 0.0:
        raise ValueError("omega, depth and gravity must be positive")
    target = omega * omega

    def residual(k: float) -> float:
        return gravity * k * math.tanh(k * depth) - target

    low = 1.0e-9 / depth
    high = 4.0 * low
    while residual(high) < 0.0:
        high *= 2.0
        if high > 1.0e6 / depth:
            raise ValueError("could not bracket the wave number")
    for _ in range(300):
        mid = 0.5 * (low + high)
        if residual(mid) < 0.0:
            low = mid
        else:
            high = mid
        if high - low <= tolerance * high:
            break
    return 0.5 * (low + high)


def piston_transfer(k: float, depth: float) -> float:
    """``H/S`` of the piston wavemaker (Havelock)."""
    kh = k * depth
    return 4.0 * math.sinh(kh) ** 2 / (2.0 * kh + math.sinh(2.0 * kh))


def flap_transfer(k: float, depth: float) -> float:
    """``H/S`` of the floor-hinged flap wavemaker."""
    kh = k * depth
    integral = math.sinh(kh) - (math.cosh(kh) - 1.0) / kh
    return 4.0 * math.sinh(kh) * integral / (2.0 * kh + math.sinh(2.0 * kh))


def piston_stroke_for_height(height: float, k: float, depth: float) -> float:
    """Piston stroke ``S`` (peak to peak) that generates wave height ``H``."""
    ratio = piston_transfer(k, depth)
    if ratio <= 0.0:
        raise ValueError("degenerate piston transfer function")
    return height / ratio


def flap_stroke_for_height(height: float, k: float, depth: float) -> float:
    """Flap stroke ``S`` at the still-water line for wave height ``H``."""
    ratio = flap_transfer(k, depth)
    if ratio <= 0.0:
        raise ValueError("degenerate flap transfer function")
    return height / ratio


# --------------------------------------------------------------------------- #
# Solitary wave (Boussinesq, weakly non-linear)
# --------------------------------------------------------------------------- #

def solitary_beta(height: float, depth: float) -> float:
    """Inverse width ``beta = sqrt(3H/(4h^3))`` of the Boussinesq solitary wave."""
    if height <= 0.0 or depth <= 0.0:
        raise ValueError("height and depth must be positive")
    return math.sqrt(3.0 * height / (4.0 * depth ** 3))


def solitary_celerity(height: float, depth: float, gravity: float = GRAVITY) -> float:
    """``c = sqrt(g (h + H))``: the Boussinesq and KdV forms agree to O(H/h)."""
    return math.sqrt(gravity * (depth + height))


def solitary_elevation(x, height: float, depth: float, crest: float = 0.0):
    """``eta(x) = H sech^2(beta (x - crest))``."""
    beta = solitary_beta(height, depth)
    return height / torch.cosh(beta * (x - crest)).square()


def solitary_volume(height: float, depth: float) -> float:
    """Excess volume per unit width, ``int eta dx = 2H/beta``."""
    return 2.0 * height / solitary_beta(height, depth)


def solitary_velocity(x, height: float, depth: float, crest: float = 0.0,
                      gravity: float = GRAVITY):
    """Depth-uniform horizontal velocity ``u = c eta / (h + eta)``.

    The leading-order shallow-water solitary wave has a nearly uniform horizontal
    velocity; the vertical component is ``O((h/lambda)^2)`` smaller and is left at
    zero, the standard initial condition for this verification.
    """
    c = solitary_celerity(height, depth, gravity)
    eta = solitary_elevation(x, height, depth, crest)
    return c * eta / (depth + eta)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class WaveTankConfig(Config):
    """Wave-tank configuration.

    ``depth`` is the still-water depth ``h`` and ``length`` the tank length; the
    particle spacing is ``dx``, normally set as a fraction of the depth (the
    resolution ladder uses ``dx/h`` in ``{0.02, 0.015, 0.01, 0.0075}``).

    The artificial sound speed is ``c0 = c0_factor * sqrt(g h)``. The
    weak-compressibility bias of the phase speed is about ``(c/c0)^2 / 2``, so
    ``c0_factor = 10`` holds it under 0.5 % for every ``kh`` used here, and the
    artificial-sound-speed sweep measures the bias directly.

    ``grav`` is the gravitational acceleration; ``Config.gravity`` is reused so
    the RHS applies it unchanged.
    """

    depth: float = 0.5
    length: float = 5.5
    c0_factor: float = 10.0
    tank_top: float | None = None
    maker: str = "relaxation"              # relaxation | piston | flap | none
    wave_kind: str = "regular"             # regular | solitary
    kh: float = 3.0
    wave_height: float = 0.05
    ramp_periods: float = 1.0
    inlet_wavelengths: float = 0.3         # length of the inlet relaxation zone
    sponge_wavelengths: float = 1.0
    sponge_strength: float = 3.0
    probe_offsets: tuple = (0.4, 0.9, 1.15, 1.4)   # wavelengths from the paddle
    probe_width: float | None = None              # column window, default 2 dx
    end_time: float | None = None                 # default: 3 periods
    snapshot_offsets: tuple = ()                  # fractions of end_time
    pressure_bands: int = 0        # record band-averaged pressure history
    hydrostatic_lattice: bool = True   # lattice mass from the hydrostatic profile

    # ------------------------------------------------------------------ setup
    def __post_init__(self):
        self.pool_width = float(self.length)
        self.pool_depth = float(self.depth)
        if self.c0_factor > 0.0:
            self.sound_speed = self.c0_factor * math.sqrt(self.gravity * self.depth)
        self.dt = self.courant_dt()
        if self.end_time is None:
            self.end_time = 3.0 * self.period()
        self.steps = int(math.ceil(self.end_time / self.dt))
        if self.probe_width is None:
            self.probe_width = 2.0 * self.dx
        # A tank initialised at uniform density is not in equilibrium: it falls
        # until the pressure builds up, which rings the tank at 0.5-0.8 m/s for
        # about a second -- several times the orbital velocity of a 0.1 h wave.
        # ``hydrostatic_lattice`` is therefore on by default.
        if not self.density_calibration:
            # A wave is a pressure signal of a few hundred pascals; the 2.4 kPa
            # lattice bias of the uncalibrated mass would swamp it.
            self.density_calibration = True

    # ---------------------------------------------------------------- derived
    def h_smoothing(self) -> float:
        return self.h_factor * self.dx

    def courant_dt(self) -> float:
        """Fixed step from the acoustic Courant condition at rest."""
        return self.max_courant * self.h_factor * self.dx / self.sound_speed

    def tank_top_height(self) -> float:
        return float(1.6 * self.depth if self.tank_top is None else self.tank_top)

    def wave_number_k(self) -> float:
        return self.kh / self.depth

    def wavelength(self) -> float:
        return 2.0 * math.pi / self.wave_number_k()

    def omega(self) -> float:
        return angular_frequency(self.wave_number_k(), self.depth, self.gravity)

    def period(self) -> float:
        if self.wave_kind == "solitary":
            return 6.0 * self.depth / solitary_celerity(
                self.wave_height, self.depth, self.gravity)
        return 2.0 * math.pi / self.omega()

    def celerity(self) -> float:
        if self.wave_kind == "solitary":
            return solitary_celerity(self.wave_height, self.depth, self.gravity)
        return wave_celerity(self.wave_number_k(), self.depth, self.gravity)

    def stroke(self) -> float:
        """Paddle stroke (peak to peak) for the configured wave height."""
        if self.maker == "none":
            return 0.0
        if self.wave_kind == "solitary":
            # Single push of amplitude 2H/(beta h): the water-particle
            # displacement of the wave, spread over the paddle face depth.
            return 2.0 * solitary_volume(self.wave_height, self.depth) / self.depth
        k = self.wave_number_k()
        if self.maker == "piston":
            return piston_stroke_for_height(self.wave_height, k, self.depth)
        return flap_stroke_for_height(self.wave_height, k, self.depth)

    def transfer_ratio(self) -> float:
        """Theoretical ``H/S`` of the configured maker."""
        if self.maker == "none":
            return float("nan")
        k = self.wave_number_k()
        if self.maker == "piston":
            return piston_transfer(k, self.depth)
        return flap_transfer(k, self.depth)

    def inlet_length(self) -> float:
        """Length of the inlet relaxation zone (m)."""
        if self.wave_kind == "solitary":
            return min(0.25 * self.length, 2.0 / solitary_beta(self.wave_height, self.depth))
        return self.inlet_wavelengths * self.wavelength()

    def sponge_start(self) -> float:
        if self.wave_kind == "solitary":
            return self.length - self.sponge_wavelengths * 0.25 * self.length
        return self.length - self.sponge_wavelengths * self.wavelength()

    def probe_positions(self) -> list[float]:
        if self.wave_kind == "solitary":
            return [float(x) for x in self.probe_offsets]
        lam = self.wavelength()
        return [float(off * lam) for off in self.probe_offsets]

    def validate(self) -> None:
        if self.depth <= 0.0 or self.length <= 0.0 or self.dx <= 0.0:
            raise ValueError("depth, length and dx must be positive")
        if self.maker not in ("relaxation", "piston", "flap", "none"):
            raise ValueError(
                "maker must be 'relaxation', 'piston', 'flap' or 'none'")
        if self.wave_kind not in ("regular", "solitary"):
            raise ValueError("wave_kind must be 'regular' or 'solitary'")
        if self.wave_kind == "regular" and self.kh <= 0.0:
            raise ValueError("kh must be positive")
        if not 0.0 < self.wave_height < 0.5 * self.depth:
            raise ValueError("wave_height must be a small fraction of the depth")
        if self.max_courant * self.h_factor * self.dx / self.sound_speed > 0.6:
            raise ValueError("courant_dt gives an acoustic Courant above 0.6")
        if self.length <= 2.0 * self.dx:
            raise ValueError("tank must be longer than two particle spacings")
        if self.dx > 0.04 * self.depth:
            raise ValueError("dx above 4 % of the depth cannot carry a wave")
        if self.wave_kind == "regular" and self.sponge_strength > 0.0:
            if self.sponge_start() <= max(self.probe_positions()):
                raise ValueError("probes must stay clear of the sponge layer")


# --------------------------------------------------------------------------- #
# Tank construction
# --------------------------------------------------------------------------- #

def _column(x0: float, y_lo: float, y_hi: float, dx: float) -> torch.Tensor:
    """A vertical column of wall particles centred at ``x0``."""
    count = int(math.floor((y_hi - y_lo) / dx))
    y = y_lo + (torch.arange(count, dtype=torch.float64) + 0.5) * dx
    return torch.stack((torch.full_like(y, x0), y), dim=-1)


def make_tank(cfg: WaveTankConfig, device=None):
    """Build the tank: ``(fluid_positions, fluid_mass, paddle, wall)``.

    ``paddle`` is the moving part of the left boundary, stored separately so the
    integrator can translate (piston) or rotate (flap) it; ``wall`` is static.

    For a solitary-wave run with ``maker="none"`` the lattice is adapted column by
    column to the still-water surface ``h + eta(x)``: every column keeps the
    spacing ``dx`` (so its summation density stays at ``rho0``) and only the
    number of particles changes. The wave then lives in the geometry instead of in
    an unbalanced velocity field, which is what an initial-value solitary-wave
    verification needs.
    """
    dv = resolve_device(cfg.device) if device is None else device
    dx = cfg.dx
    dtype = torch.float64

    count_x = int(math.floor(cfg.length / dx))
    count_y = int(math.floor(cfg.depth / dx))
    if count_x < 2 or count_y < 2:
        raise ValueError("dx is too coarse for the tank dimensions")
    xs = (torch.arange(count_x, dtype=dtype) + 0.5) * dx

    if cfg.wave_kind == "solitary" and cfg.maker == "none":
        crest = solitary_crest_offset(cfg)
        surface = cfg.depth + solitary_elevation(xs, cfg.wave_height, cfg.depth, crest)
        counts = torch.floor(surface / dx).to(torch.long).clamp_min(2)
    else:
        counts = torch.full((count_x,), count_y, dtype=torch.long)

    positions = []
    for index in range(count_x):
        n = int(counts[index])
        y = (torch.arange(n, dtype=dtype) + 0.5) * dx
        positions.append(torch.stack((torch.full_like(y, float(xs[index])), y), dim=-1))
    pos = torch.cat(positions, dim=0)
    if cfg.hydrostatic_lattice:
        # Mass from the analytical hydrostatic density profile: the column then
        # starts in discrete equilibrium (the pressure gradient balances gravity)
        # instead of falling until the pressure builds up. Without it the tank
        # rings for ~1 s at 0.5-0.8 m/s, which is several times the orbital
        # velocity of a 0.1 h wave.
        mass = hydrostatic_density(pos[:, 1], cfg, surface=cfg.depth) * cfg.dx * cfg.dx
        mass = mass.to(device=dv, dtype=dtype)
    else:
        mass = torch.full((pos.shape[0],), lattice_mass_unit(cfg), device=dv, dtype=dtype)

    top = cfg.tank_top_height()
    layers = max(1, int(cfg.wall_layers))

    paddle = torch.cat([_column(-0.5 * dx - k * dx, -0.5 * dx, top, dx)
                        for k in range(layers)], dim=0)
    floor = []
    x_span = torch.arange(-0.5, count_x + 1.5) * dx
    for k in range(1, layers + 1):
        yy = torch.full_like(x_span, 0.5 * dx - k * dx)
        floor.append(torch.stack((x_span, yy), dim=-1))
    right = [_column(0.5 * dx + cfg.length + k * dx, -0.5 * dx, top, dx)
             for k in range(layers)]
    wall = torch.cat(floor + right, dim=0)

    return (pos.to(device=dv, dtype=dtype), mass,
            paddle.to(device=dv, dtype=dtype), wall.to(device=dv, dtype=dtype))


def initial_condition(cfg: WaveTankConfig, pos: torch.Tensor):
    """Initial velocity for a solitary-wave run (zeros for a still tank).

    A regular-wave run starts from still water and lets the paddle build the
    train, so only the solitary case needs an initial velocity. The solitary wave
    is placed ``solitary_crest`` metres from the paddle; the default is two wave
    widths, far enough that the paddle does not disturb it.
    """
    if cfg.wave_kind != "solitary":
        return torch.zeros_like(pos)
    crest = solitary_crest_offset(cfg)
    eta = solitary_elevation(pos[:, 0], cfg.wave_height, cfg.depth, crest)
    # Impose the profile by shifting the free surface: particles keep their
    # lattice position, so the initial elevation is carried by the velocity.
    u = cfg.celerity() * eta / (cfg.depth + eta)
    vel = torch.zeros_like(pos)
    vel[:, 0] = u
    return vel


def solitary_crest_offset(cfg: WaveTankConfig) -> float:
    """Initial crest position (m) of the solitary wave."""
    width = 1.0 / solitary_beta(cfg.wave_height, cfg.depth)
    return min(0.35 * cfg.length, 4.0 * width + 2.0 * cfg.depth)


# --------------------------------------------------------------------------- #
# Paddle kinematics, sponge, boundary
# --------------------------------------------------------------------------- #

def ramp_factor(cfg: WaveTankConfig, t: float) -> float:
    """Smooth 0 -> 1 ramp over ``ramp_periods`` wave periods."""
    if cfg.ramp_periods <= 0.0:
        return 1.0
    ramp_time = cfg.ramp_periods * cfg.period()
    if ramp_time <= 0.0:
        return 1.0
    fraction = min(1.0, max(0.0, t / ramp_time))
    return 0.5 - 0.5 * math.cos(math.pi * fraction)


def analytic_velocity(x: torch.Tensor, y: torch.Tensor, cfg: WaveTankConfig,
                      t: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic (Airy, or shallow-water solitary) velocity field at time ``t``.

    For the regular wave the depth dependence is the finite-depth Airy profile
    ``cosh(k y)/sinh(kh)`` with ``y`` measured from the *bed* (so the surface value
    is ``coth(kh)`` and the bed value ``1/sinh(kh)``); ``v`` follows from
    continuity. The solitary wave
    uses the depth-uniform ``u = c eta/(h+eta)`` with ``eta`` travelling at ``c``.
    """
    if cfg.wave_kind == "solitary":
        crest = solitary_crest_offset(cfg) + cfg.celerity() * t
        eta = solitary_elevation(x, cfg.wave_height, cfg.depth, crest)
        u = cfg.celerity() * eta / (cfg.depth + eta)
        return u, torch.zeros_like(u)
    k = cfg.wave_number_k()
    omega = cfg.omega()
    amplitude = 0.5 * cfg.wave_height
    phase = omega * t - k * x
    kh = k * cfg.depth
    depth_ratio = amplitude * omega / math.sinh(kh)
    arg = (k * y).clamp(max=30.0)
    u = depth_ratio * torch.cosh(arg) * torch.cos(phase)
    v = depth_ratio * torch.sinh(arg) * torch.sin(phase)
    return u, v


def relax_inlet(pos: torch.Tensor, vel: torch.Tensor, cfg: WaveTankConfig,
                t: float) -> float:
    """Blend the velocity in the inlet zone towards the analytic wave field.

    Weights ``w(x) = ((x_r - x)/x_r)^2`` for ``x < x_r`` and zero elsewhere, with
    ``w(0) = 1``. This is the relaxation-zone wavemaker (waves2Foam style): the
    fluid is driven by a velocity field instead of being pushed by a moving wall,
    so it cannot open a void on the backstroke or jam particles on the forward
    stroke, and the same zone absorbs whatever comes back from the tank.

    Returns the largest weight applied (0 when the zone is empty, which is what
    the ``maker`` check uses).
    """
    x_r = cfg.inlet_length()
    if x_r <= 0.0 or cfg.maker != "relaxation":
        return 0.0
    weight = (1.0 - pos[:, 0] / x_r).clamp(min=0.0, max=1.0).square()
    if not bool((weight > 0.0).any()):
        return 0.0
    u, v = analytic_velocity(pos[:, 0], pos[:, 1], cfg, t)
    scale = ramp_factor(cfg, t)
    weight = weight * scale
    vel[:, 0] = (1.0 - weight) * vel[:, 0] + weight * u
    vel[:, 1] = (1.0 - weight) * vel[:, 1] + weight * v
    return float(weight.max())


def paddle_offset(cfg: WaveTankConfig, t: float) -> float:
    """Horizontal displacement of the paddle face (m) at time ``t``."""
    if cfg.maker not in ("piston", "flap"):
        return 0.0
    if cfg.wave_kind == "solitary":
        beta = solitary_beta(cfg.wave_height, cfg.depth)
        c = cfg.celerity()
        width = 2.0 / beta
        return cfg.stroke() * 0.5 * (1.0 + math.tanh(beta * c * (t - width / c)))
    omega = cfg.omega()
    ramp = 1.0
    if cfg.ramp_periods > 0.0:
        ramp_time = cfg.ramp_periods * cfg.period()
        fraction = min(1.0, max(0.0, t / ramp_time)) if ramp_time > 0.0 else 1.0
        ramp = 0.5 - 0.5 * math.cos(math.pi * fraction)     # smooth start
    return 0.5 * cfg.stroke() * math.sin(omega * t) * ramp


def paddle_face(cfg: WaveTankConfig, t: float) -> float:
    """x coordinate of the paddle face at time ``t``."""
    return paddle_offset(cfg, t)


def sponge_coefficient(x: torch.Tensor, cfg: WaveTankConfig) -> torch.Tensor:
    """Damping rate ``a(x)`` of the sponge layer (1/s)."""
    x_s = cfg.sponge_start()
    span = cfg.length - x_s
    if span <= 0.0:
        return torch.zeros_like(x)
    omega = 2.0 * math.pi / cfg.period()
    ramp = ((x - x_s) / span).clamp_min(0.0)
    return cfg.sponge_strength * omega * ramp.square()


def clamp_domain(pos: torch.Tensor, vel: torch.Tensor, cfg: WaveTankConfig,
                 face: float) -> int:
    """Keep particles inside the tank; return the number of clamped particles.

    The left limit is the paddle face, which moves: the fluid may follow the
    paddle back, but it may not pass through it. The normal velocity component is
    zeroed on contact (free slip tangentially), which is what
    ``droplet_impact.apply_boundary`` does for the static box.
    """
    dx = cfg.dx
    x_min = face - 1.5 * dx
    x_max = cfg.length
    y_min = 0.0
    y_max = cfg.tank_top_height()
    tolerance = 0.25 * dx
    low = pos[:, 0] < x_min - tolerance
    high = pos[:, 0] > x_max + tolerance
    bottom = pos[:, 1] < y_min - tolerance
    top = pos[:, 1] > y_max + tolerance
    pos[:, 0] = pos[:, 0].clamp(x_min, x_max)
    pos[:, 1] = pos[:, 1].clamp(y_min, y_max)
    zero = torch.zeros_like(vel[:, 0])
    vel[:, 0] = torch.where(low | high, zero, vel[:, 0])
    vel[:, 1] = torch.where(bottom | top, zero, vel[:, 1])
    return int((low | high | bottom | top).sum().item())


def displace_paddle(paddle_base: torch.Tensor, paddle: torch.Tensor,
                    cfg: WaveTankConfig, t: float) -> float:
    """Move ``paddle`` (in place) to the time-``t`` position; return the face x.

    Piston: rigid translation. Flap: rotation about the floor hinge, with the
    horizontal displacement growing linearly from zero at the floor to the full
    stroke at the still-water line.
    """
    offset = paddle_offset(cfg, t)
    if cfg.maker == "flap":
        lever = (paddle_base[:, 1] / cfg.depth).clamp(min=0.0, max=1.0)
        paddle[:, 0] = paddle_base[:, 0] + offset * lever
    else:
        paddle[:, 0] = paddle_base[:, 0] + offset
    return offset


# --------------------------------------------------------------------------- #
# Probes and post-processing
# --------------------------------------------------------------------------- #

def still_water_volume(cfg: WaveTankConfig) -> float:
    """Column height the elevation estimator reads for still water.

    With the hydrostatic-consistent lattice mass the particles are heavier at the
    bottom (``rho(y)/rho0 = (1 + rho0 g (h-y)/B)^(1/gamma)``), so the volume of a
    column is ``h`` times a factor of about ``1 + 5e-3``; subtracting ``h``
    instead would bias every elevation by that amount (2.5 mm, i.e. 5 % of a
    0.1 h wave). With a uniform lattice mass the factor is exactly one.
    """
    if not cfg.hydrostatic_lattice:
        return float(cfg.depth)
    ys = torch.linspace(0.0, cfg.depth, 401, dtype=torch.float64)
    density = hydrostatic_density(ys, cfg, surface=cfg.depth)
    return float(torch.trapezoid(density / cfg.rho0, ys))


def column_elevation(pos: torch.Tensor, mass: torch.Tensor, x_probe: float,
                     width: float, cfg: WaveTankConfig) -> float:
    """Material-volume free-surface elevation at a probe (m).

    The estimator is the material volume per unit width of the neighbourhood of
    ``x_probe``, weighted by a normalised Gaussian of standard deviation
    ``width``::

        eta_p = sum_i (m_i / rho0) G(x_i - x_p) - h ,   G(s) = exp(-s^2/2w^2)/(w sqrt(2 pi))

    For a lattice in a column of height ``h`` the sum returns ``h`` because
    ``sum_j G(x_j - x_p) dx = 1``. A *box* window of a couple of spacings is not
    usable here: particles sit on a lattice, so a box window gains or loses whole
    columns when the surface moves sideways, which quantises the elevation in
    steps of ``dp / (W/dp)`` -- for ``W = 2dp`` that is half a column, i.e. 20 % of
    a wave of height ``5 dp``. The smooth window has no such jumps.

    ``width`` is therefore the *smoothing length* of the estimator, not a box
    width; the default ``2 dp`` smears the elevation over ~2 % of the shortest
    wavelength used here.
    """
    sigma = float(width)
    if sigma <= 0.0:
        raise ValueError("probe width must be positive")
    dx = pos[:, 0] - x_probe
    weight = torch.exp(-0.5 * (dx / sigma).square()) / (sigma * math.sqrt(2.0 * math.pi))
    volume = float((mass * weight).sum()) / cfg.rho0
    return volume - still_water_volume(cfg)


def elevation_profile(pos: torch.Tensor, mass: torch.Tensor, cfg: WaveTankConfig,
                      bins: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Elevation on a uniform grid of column windows spanning the tank."""
    edges = torch.linspace(0.0, cfg.length, bins + 1, dtype=pos.dtype, device=pos.device)
    width = float(edges[1] - edges[0])
    index = torch.bucketize(pos[:, 0].contiguous(), edges) - 1
    valid = (index >= 0) & (index < bins)
    volume = torch.zeros(bins, dtype=pos.dtype, device=pos.device)
    volume = volume.index_add(0, index[valid], mass[valid]) / cfg.rho0
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, volume / width - still_water_volume(cfg)


@dataclass
class HarmonicFit:
    """Least-squares fit ``y = offset + a cos(omega t) + b sin(omega t) + drift*t``."""

    amplitude: float
    phase: float
    offset: float
    drift: float
    residual: float
    samples: int

    @property
    def height(self) -> float:
        """Wave height ``H = 2A``."""
        return 2.0 * self.amplitude

    def as_dict(self) -> dict:
        return {"amplitude": self.amplitude, "height": self.height,
                "phase": self.phase, "offset": self.offset, "drift": self.drift,
                "residual": self.residual, "samples": self.samples}


def harmonic_fit(times, values, omega: float) -> HarmonicFit:
    """Fit a single frequency to a sampled series (lock-in detection).

    Used instead of a peak-to-peak read-out because the series is noisy at the
    particle scale: a least-squares projection on the known forcing frequency
    rejects everything that is not the wave.
    """
    t = torch.as_tensor(times, dtype=torch.float64)
    y = torch.as_tensor(values, dtype=torch.float64)
    good = torch.isfinite(y)
    if int(good.sum()) < 4:
        raise ValueError("not enough samples for a harmonic fit")
    t, y = t[good], y[good]
    design = torch.stack((torch.ones_like(t), torch.cos(omega * t),
                          torch.sin(omega * t), t - t.mean()), dim=1)
    solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
    residual = (y - design @ solution)
    rms = float(torch.sqrt((residual.square().mean())))
    a, b = float(solution[1]), float(solution[2])
    # y = a cos(wt) + b sin(wt) = A cos(wt - phi) with A = hypot(a, b) and
    # phi = atan2(b, a). For a wave travelling in +x, eta = A cos(wt - kx), the
    # fitted phase therefore grows with x as phi = +k x.
    return HarmonicFit(amplitude=math.hypot(a, b), phase=math.atan2(b, a),
                       offset=float(solution[0]), drift=float(solution[3]),
                       residual=rms, samples=int(t.numel()))


def wrap_angle(angle: float) -> float:
    """Wrap to ``(-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ProbePair:
    """Two-probe measurement of the wave train."""

    k_measured: float
    celerity_measured: float
    celerity_theory: float
    celerity_error: float
    wavelength_measured: float
    heights: tuple
    phases: tuple
    residuals: tuple
    omega: float

    def as_dict(self) -> dict:
        return {
            "k_measured": self.k_measured,
            "celerity_measured": self.celerity_measured,
            "celerity_theory": self.celerity_theory,
            "celerity_error": self.celerity_error,
            "wavelength_measured": self.wavelength_measured,
            "heights": list(self.heights),
            "phases": list(self.phases),
            "residuals": list(self.residuals),
            "omega": self.omega,
        }


def two_probe_dispersion(times, eta_a, eta_b, separation: float, omega: float,
                         depth: float, gravity: float = GRAVITY) -> ProbePair:
    """Wave number and phase speed from the phase lag between two probes.

    ``eta(x, t) = A cos(omega t - k x + phi0)`` gives ``phi(x) = k x - phi0``, so
    ``k = (phi_b - phi_a) / separation``: the phase of the downstream probe leads
    for a wave travelling in +x. ``separation`` must stay below ``lambda/2`` for
    the wrapped difference to be unambiguous. The phase speed is compared with the
    linear value *at the measured wave number*, which is the self-consistent
    comparison: a theory evaluated at the imposed ``k`` would hide an error in
    ``k`` behind the very quantity being tested.
    """
    fit_a = harmonic_fit(times, eta_a, omega)
    fit_b = harmonic_fit(times, eta_b, omega)
    delta_phase = wrap_angle(fit_b.phase - fit_a.phase)
    if delta_phase == 0.0:
        raise ValueError("degenerate phase difference between the probes")
    k_measured = delta_phase / separation
    c_measured = omega / k_measured
    c_theory = wave_celerity(abs(k_measured), depth, gravity)
    return ProbePair(k_measured=k_measured, celerity_measured=c_measured,
                     celerity_theory=c_theory,
                     celerity_error=abs(c_measured - c_theory) / c_theory,
                     wavelength_measured=2.0 * math.pi / abs(k_measured),
                     heights=(fit_a.height, fit_b.height),
                     phases=(fit_a.phase, fit_b.phase),
                     residuals=(fit_a.residual, fit_b.residual),
                     omega=omega)


def reflection_coefficient(heights) -> tuple[float, float, float]:
    """``R = (H_max - H_min)/(H_max + H_min)`` over a probe array.

    A partially standing wave has a spatial envelope; sampling it at several
    probes and taking the extremes gives the classic two-probe decomposition
    without needing the phase at a node. Returns ``(R, H_max, H_min)``.
    """
    values = [float(h) for h in heights if math.isfinite(h)]
    if len(values) < 2:
        raise ValueError("need at least two probe heights")
    high, low = max(values), min(values)
    if high <= 0.0:
        raise ValueError("degenerate envelope")
    return (high - low) / (high + low), high, low


def spatial_wave_number(positions, heights, phases, omega: float,
                        depth: float, gravity: float = GRAVITY) -> dict:
    """Wave number from the *spatial* phase slope over a probe array.

    Each probe contributes a complex amplitude ``A_j = (H_j/2) exp(i phi_j)`` of
    ``eta = Re[A exp(i(omega t - phi(x)))]``. For a single incident wave
    ``phi(x) = k x`` and ``ln H`` is linear in ``x``; a two-probe difference uses
    two points only, so any reflected component (which makes both phase and
    amplitude wiggle in ``x`` with period ``lambda/2``) is folded straight into
    the answer. Fitting both ``phi`` and ``ln H`` against ``x`` over the whole
    array averages part of that wiggle out and returns the decay exponent
    ``alpha`` from the same fit.

    Returns ``{"k", "celerity", "celerity_theory", "celerity_error", "alpha",
    "decay_ratio_per_wavelength", "samples"}``.
    """
    values = [(float(x), float(h), float(p))
              for x, h, p in zip(positions, heights, phases)
              if math.isfinite(h) and h > 0.0 and math.isfinite(p)]
    if len(values) < 3:
        raise ValueError("need at least three usable probes")
    xs = [item[0] for item in values]
    mean_x = sum(xs) / len(xs)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    # Unwrap the phases in x order before the linear fit.
    ordered = sorted(values)
    unwrapped, previous, offset = [], None, 0.0
    for _, _, phase in ordered:
        if previous is not None:
            step = (phase + offset) - previous
            while step > math.pi:
                offset -= 2.0 * math.pi
                step -= 2.0 * math.pi
            while step < -math.pi:
                offset += 2.0 * math.pi
                step += 2.0 * math.pi
        unwrapped.append(phase + offset)
        previous = phase + offset
    phase_by_x = {item[0]: value for item, value in zip(ordered, unwrapped)}
    logs = {item[0]: math.log(item[1]) for item in ordered}
    phases = [phase_by_x[x] for x in xs]
    slopes = []
    for series in (phases, [logs[x] for x in xs]):
        mean_y = sum(series) / len(series)
        slopes.append(sum((x - mean_x) * (y - mean_y)
                          for x, y in zip(xs, series)) / denominator)
    k, alpha = slopes[0], -slopes[1]
    celerity = omega / k if k else float("inf")
    theory = wave_celerity(abs(k), depth, gravity) if k else float("nan")
    wavelength = 2.0 * math.pi / abs(k) if k else float("inf")
    return {"k": k, "celerity": celerity, "celerity_theory": theory,
            "celerity_error": abs(celerity - theory) / theory if theory else float("nan"),
            "alpha": alpha,
            "decay_ratio_per_wavelength": math.exp(-alpha * wavelength),
            "samples": len(values)}


def detrended_reflection(positions, heights, wavelength: float) -> tuple[float, float | None]:
    """Reflection from the envelope of ``heights`` after removing the decay trend.

    ``(H_max - H_min)/(H_max + H_min)`` is the campaign formula, but it reads a
    spatially decaying incident wave as reflection: a 30 % drop between the first
    and last probe becomes R = 0.16 with no reflector at all. Fitting
    ``ln H = a + b x`` and taking the envelope of ``H / exp(a + bx)`` removes the
    trend and leaves only what varies about it, which is the standing wave
    pattern. Returns ``(R_detrended, height_ratio_per_wavelength)``; the second
    entry is ``exp(b * wavelength)``, the measured amplitude ratio per wavelength.
    """
    values = [float(h) for h in heights]
    if len(values) < 3 or any(value <= 0.0 for value in values):
        return float("nan"), None
    xs = [float(x) for x in positions]
    logs = [math.log(value) for value in values]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(logs) / len(logs)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, logs)) / denominator
             if denominator else 0.0)
    ratios = [value / math.exp(mean_y + slope * (x - mean_x))
              for x, value in zip(xs, values)]
    reflection = (max(ratios) - min(ratios)) / (max(ratios) + min(ratios))
    return reflection, math.exp(slope * wavelength)


def crest_time(times, eta) -> tuple[float, float]:
    """Time and elevation of the maximum of a solitary-wave series.

    Parabolic refinement on the three samples around the discrete maximum, so the
    arrival time is not quantised by the sampling interval (which would bias the
    measured phase speed).
    """
    y = torch.as_tensor(eta, dtype=torch.float64)
    t = torch.as_tensor(times, dtype=torch.float64)
    index = int(torch.argmax(y))
    if 0 < index < y.numel() - 1:
        y0, y1, y2 = float(y[index - 1]), float(y[index]), float(y[index + 1])
        denominator = (y0 - 2.0 * y1 + y2)
        shift = 0.5 * (y0 - y2) / denominator if denominator != 0.0 else 0.0
        dt = float(t[index + 1] - t[index])
        return float(t[index]) + shift * dt, y1 - 0.25 * (y0 - y2) * shift
    return float(t[index]), float(y[index])


def waveform_error(x, eta_sim, height: float, depth: float, crest: float,
                   halfwidth: float | None = None) -> tuple[float, int]:
    """Relative L2 error of a solitary-wave profile against ``H sech^2``.

    The window is ``|x - crest| < halfwidth`` (default ``2.5/beta``), which holds
    98.7 % of the wave's volume -- the volume outside scales as ``1 - tanh(2.5)``.
    A wider window adds nothing to the comparison and only accumulates the noise
    of the undisturbed surface, where the analytic profile is already below 0.7 %
    of ``H``.
    """
    beta = solitary_beta(height, depth)
    window = 2.5 / beta if halfwidth is None else halfwidth
    mask = (x - crest).abs() < window
    if int(mask.sum()) < 3:
        raise ValueError("profile window too narrow")
    analytic = solitary_elevation(x[mask], height, depth, crest)
    error = eta_sim[mask] - analytic
    denominator = torch.sqrt(analytic.square().sum())
    return float(torch.sqrt(error.square().sum()) / denominator), int(mask.sum())


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #

@dataclass
class WaveTankRun:
    """Result of a wave-tank integration."""

    times: list = field(default_factory=list)
    probe_x: list = field(default_factory=list)
    probes: list = field(default_factory=list)       # probes[i][k] at times[k]
    paddle_track: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)    # (t, centres, eta, pos, vel)
    band_history: list = field(default_factory=list)  # (t, [band mean rel. pressure error])
    dt: float = 0.0
    steps_taken: int = 0
    simulated_time: float = 0.0
    mass_initial: float = 0.0
    mass_final: float = 0.0
    volume_initial: float = 0.0
    volume_final: float = 0.0
    max_compression: float = 0.0
    max_dilation: float = 0.0
    max_speed: float = 0.0
    max_courant: float = 0.0
    escaped: int = 0
    rebuilds: int = 0
    wall_seconds: float = 0.0

    @property
    def mass_error(self) -> float:
        if self.mass_initial == 0.0:
            return float("nan")
        return abs(self.mass_final - self.mass_initial) / self.mass_initial

    @property
    def volume_error(self) -> float:
        if self.volume_initial == 0.0:
            return float("nan")
        return abs(self.volume_final - self.volume_initial) / self.volume_initial

    def probe_series(self, index: int):
        return self.times, self.probes[index]

    def as_dict(self) -> dict:
        return {
            "dt": self.dt, "steps_taken": self.steps_taken,
            "simulated_time": self.simulated_time,
            "mass_initial": self.mass_initial, "mass_final": self.mass_final,
            "mass_error": self.mass_error, "volume_error": self.volume_error,
            "max_compression": self.max_compression,
            "max_dilation": self.max_dilation, "max_speed": self.max_speed,
            "max_courant": self.max_courant, "escaped": self.escaped,
            "rebuilds": self.rebuilds, "wall_seconds": self.wall_seconds,
            "probe_x": list(self.probe_x),
        }


def run_tank(cfg: WaveTankConfig, snapshot_times=()) -> WaveTankRun:
    """Integrate the wave tank and record probes, masses and snapshots."""
    import time as _time

    cfg.validate()
    dv = resolve_device(cfg.device)
    started = _time.time()

    positions, mass, paddle_base, wall = make_tank(cfg, dv)
    paddle = paddle_base.clone()
    velocity = initial_condition(cfg, positions)
    fluid_count = positions.shape[0]
    initial_mass = float(mass.sum())

    h = cfg.h_smoothing()
    skin = cfg.pair_skin_factor * h
    cache = (PairListCache(2.0 * h, skin, fluid_count=fluid_count) if skin > 0 else None)

    probe_x = cfg.probe_positions()
    run = WaveTankRun(probe_x=probe_x, dt=cfg.dt)
    run.probes = [[] for _ in probe_x]
    run.mass_initial = initial_mass

    snap_targets = sorted(float(t * cfg.end_time) for t in snapshot_times)
    snap_next = 0
    max_compression = -float("inf")
    max_dilation = float("inf")
    max_speed = 0.0
    courant_peak = 0.0
    escaped = 0
    face = paddle_offset(cfg, 0.0)

    dt = cfg.dt
    elapsed = 0.0
    step = 0
    total_steps = cfg.steps

    def _boundary_pairs():
        if cache is None:
            return None
        joined = torch.cat((positions, paddle, wall), dim=0)
        pairs, _ = cache.get(joined)
        return pairs

    while True:
        final = elapsed >= cfg.end_time - 1.0e-12
        if final:
            break
        face = displace_paddle(paddle_base, paddle, cfg, elapsed)
        pairs = _boundary_pairs()
        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs,
                              wall=torch.cat((paddle, wall), dim=0), dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")

        if step == 0:
            # Reference material volume, measured with the same summation
            # density as the final value so that the two are comparable.
            run.volume_initial = float((mass / density).sum())
        compression = float(density.max().double() / cfg.rho0 - 1.0)
        dilation = float(density.min().double() / cfg.rho0 - 1.0)
        max_compression = max(max_compression, compression)
        max_dilation = min(max_dilation, dilation)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        speed_scalar = float(speed.max()) if speed.numel() else 0.0
        max_speed = max(max_speed, speed_scalar)
        courant_peak = max(courant_peak, dt * (cfg.sound_speed + speed_scalar) / h)

        if step % cfg.output_every == 0 or final:
            run.times.append(elapsed)
            run.paddle_track.append(face)
            for index, x_p in enumerate(probe_x):
                run.probes[index].append(
                    column_elevation(positions, mass, x_p, cfg.probe_width, cfg))
            if cfg.pressure_bands > 0:
                pressure = tait_pressure(density, cfg)
                edges = torch.linspace(0.0, cfg.depth, cfg.pressure_bands + 1,
                                       dtype=positions.dtype)
                ratio = []
                for low, high in zip(edges[:-1].tolist(), edges[1:].tolist()):
                    band = (positions[:, 1] >= low) & (positions[:, 1] < high)
                    if not bool(band.any()):
                        ratio.append(float("nan"))
                        continue
                    reference = float(cfg.rho0 * cfg.gravity
                                      * (cfg.depth - 0.5 * (low + high)))
                    ratio.append(float(pressure[band].mean()) / reference - 1.0)
                run.band_history.append((elapsed, ratio))
            while snap_next < len(snap_targets) and elapsed >= snap_targets[snap_next] - 1e-12:
                centres, eta = elevation_profile(positions, mass, cfg)
                run.snapshots.append((elapsed, centres, eta,
                                      positions.detach().clone(),
                                      velocity.detach().clone()))
                snap_next += 1

        velocity = velocity + dt * acc
        # Inlet relaxation zone: drives the wave (and absorbs what returns).
        relax_inlet(positions, velocity, cfg, elapsed + dt)
        # Sponge: velocity damping over the last part of the tank.
        damping = sponge_coefficient(positions[:, 0], cfg)
        if bool((damping > 0.0).any()):
            velocity = velocity * (1.0 - damping * dt).clamp_min(0.0).unsqueeze(-1)
        positions = positions + dt * velocity
        escaped += clamp_domain(positions, velocity, cfg, face)
        elapsed += dt
        step += 1
        if step > total_steps + 2:
            break

    run.steps_taken = step
    run.simulated_time = elapsed
    run.mass_final = float(mass.sum())
    run.volume_final = float((mass / density.clamp_min(0.01 * cfg.rho0)).sum())
    run.max_compression = max_compression
    run.max_dilation = max_dilation
    run.max_speed = max_speed
    run.max_courant = courant_peak
    run.escaped = escaped
    run.rebuilds = 0 if cache is None else cache.rebuilds
    run.wall_seconds = _time.time() - started
    run.final_positions = positions.detach().clone()
    run.final_velocity = velocity.detach().clone()
    run.final_density = density.detach().clone()
    return run


__all__ = [
    "GRAVITY", "WaveTankConfig", "WaveTankRun", "HarmonicFit", "ProbePair",
    "angular_frequency", "wave_celerity", "wave_number", "piston_transfer",
    "flap_transfer", "piston_stroke_for_height", "flap_stroke_for_height",
    "solitary_beta", "solitary_celerity", "solitary_elevation",
    "solitary_velocity", "make_tank", "initial_condition", "paddle_offset",
    "analytic_velocity", "relax_inlet", "ramp_factor", "still_water_volume",
    "sponge_coefficient", "clamp_domain", "column_elevation",
    "elevation_profile", "harmonic_fit", "wrap_angle", "two_probe_dispersion",
    "reflection_coefficient", "detrended_reflection", "spatial_wave_number",
    "crest_time",
    "waveform_error", "run_tank",
]
