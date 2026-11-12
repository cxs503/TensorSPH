"""Stokes' second problem in a bounded gap (oscillatory Couette flow).

Physics
-------
A viscous fluid fills a gap ``0 <= y <= H``.  The lower plate oscillates in its
own plane with ``u_wall(t) = U0 cos(omega t)``; the upper plate is held at rest.
The flow is x-invariant, so the Navier-Stokes equations reduce to the linear
diffusion problem

    du/dt = nu d2u/dy2,   u(0, t) = U0 cos(omega t),   u(H, t) = 0.

Its periodic (fully developed) solution is the bounded-gap form of Stokes'
second problem,

    u(y, t) = Re{ U0 sinh(beta (H - y)) / sinh(beta H) e^{i omega t} },
    beta = (1 + i) / delta,   delta = sqrt(2 nu / omega),

with ``delta`` the viscous penetration depth: the distance over which the wall
motion decays by ``1/e`` in the semi-infinite limit ``H >> delta``.  At
``y = 0`` the ratio is exactly one (no-slip on the moving plate); at ``y = H``
it is exactly zero (no-slip on the fixed plate).

Why this case exists
--------------------
The repository verifies the viscous operator in three other ways, and none of
them is an *unsteady* test of it:

* the steady plane-Poiseuille channel (``benchmarks/poiseuille``) measures the
  curvature of a *steady* profile, i.e. a momentum balance;
* ``tests/test_boundary.py`` measures a transient **quarter-wave shear decay**
  ``exp(-nu (pi / 2H)^2 t)`` and records a measured rate of 0.84 of the analytic
  one at ``dx = 0.02`` -- a real number, but a single decaying mode with a
  single time constant, so it cannot separate "the right amount of viscosity"
  from "the right phase";
* the capillary-wave and droplet cases exercise viscosity only through a
  damping rate, where the surface-tension operator dominates the error budget.

Oscillatory shear separates the two error mechanisms, because the analytic
solution carries *both* an amplitude envelope and a phase lag at every depth:
one number for viscosity (`nu_eff / nu`) and one for unsteadiness (the phase
error in degrees).  It is also the canonical setting for an SPH-specific worry:
a fluid-wall viscosity blend (``cfg.wall_friction``) imposes no-slip through the
kernel weight ``W_ij`` of the first fluid row rather than at the wall face, so
the effective wall position is not the nominal one.  In a *steady* channel that
offset changes the profile level; in an *oscillatory* one it changes the phase,
and the phase is where the extra parameter shows up.

Discretisation notes
--------------------
Density stays summation-based (there is no continuity equation here), so the
density is whatever the neighbour kernel sum gives; the pressure is the Tait law
of that density.  Gravity is set to zero by this case on purpose: the problem is
a shear problem, a hydrostatic gradient would only add a compressibility budget
on top of it and force a large ``c0``.  With ``gravity = 0`` and a genuinely
small wall amplitude (``U0 / c0 <= 0.01`` in every case shipped here) the flow
is incompressible to ``O(Mach^2) ~ 1e-4``, and ``c0`` can be chosen small enough
for the run to be affordable -- the step count of this case is
``pi delta^2 c0 / (0.3 h nu)`` per period, which is proportional to ``c0``.

The initial condition is the analytic field at ``t = 0``, so the periodic state
is present from the first step and no startup transient has to decay (in a gap
of height ``4 delta`` that transient decays on a time ``~5`` periods).

Nothing in this module knows about ships, ice or free surfaces: it is a
verification case for the viscous operator and the moving-wall condition.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .droplet_impact import Config, resolve_device, rhs
from .initial_density import calibrate_mass

__all__ = [
    "OscillatoryCouette",
    "build_fluid",
    "build_walls",
    "wall_velocity",
    "run_case",
    "fit_effective_viscosity",
    "crank_nicolson_reference",
]


# --------------------------------------------------------------------------- #
# Analytic solution
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OscillatoryCouette:
    """Exact solution of the bounded-gap Stokes second problem.

    ``H`` gap height, ``omega`` angular frequency, ``nu`` kinematic viscosity,
    ``U0`` wall amplitude.  ``delta`` and ``beta`` are derived, never supplied,
    so a caller cannot quote an inconsistent penetration depth.
    """

    gap: float
    omega: float
    viscosity: float
    amplitude: float

    def __post_init__(self):
        for name in ("gap", "omega", "viscosity"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not float(self.amplitude) > 0.0:
            raise ValueError(
                f"amplitude must be positive, got {self.amplitude}")

    @property
    def delta(self) -> float:
        """Viscous penetration depth ``sqrt(2 nu / omega)``."""
        return math.sqrt(2.0 * self.viscosity / self.omega)

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    def _sinh_1pi(self, a):
        """``sinh((1+i) a)`` split into real and imaginary tensors."""
        return torch.sinh(a) * torch.cos(a), torch.cosh(a) * torch.sin(a)

    def shape(self, y):
        """Complex amplitude ``U_hat(y)`` as ``(real, imag)`` tensors.

        ``u(y, t) = Re{ U_hat(y) e^{i omega t} }``
                  ``= Re U_hat cos(omega t) - Im U_hat sin(omega t)``.
        """
        y = torch.as_tensor(y)
        re_n, im_n = self._sinh_1pi((self.gap - y) / self.delta)
        re_d, im_d = self._sinh_1pi(torch.as_tensor(self.gap / self.delta,
                                                   dtype=y.dtype if torch.is_tensor(y) else None))
        den = re_d * re_d + im_d * im_d
        re = self.amplitude * (re_n * re_d + im_n * im_d) / den
        im = self.amplitude * (im_n * re_d - re_n * im_d) / den
        return re, im

    def amplitude_profile(self, y):
        re, im = self.shape(y)
        return torch.sqrt(re * re + im * im)

    def phase_profile(self, y):
        re, im = self.shape(y)
        return torch.atan2(im, re)

    def velocity(self, y, t: float):
        """Real velocity field at time ``t``."""
        re, im = self.shape(y)
        return re * math.cos(self.omega * t) - im * math.sin(self.omega * t)


def crank_nicolson_reference(case: OscillatoryCouette, rows: int = 401,
                             steps_per_period: int = 4000,
                             periods: float = 3.0, t_offset: float = 0.0):
    """Independent Crank-Nicolson solution of the same diffusion problem.

    This exists so that the *closed form* is verified rather than trusted: it is
    a second, unrelated discretisation (tridiagonal implicit in time, central in
    space) of ``du/dt = nu d2u/dy2`` with the moving-wall Dirichlet condition.
    A test compares the two and requires agreement far below the SPH error that
    the benchmark reports, so an algebra slip in :meth:`shape` cannot pass.

    Returns ``(t, y, u)`` with ``u`` of shape ``(steps + 1, rows)``.  The initial
    condition is the analytic field at ``t = t_offset`` so the comparison is
    against the periodic state, not a startup transient.
    """
    nu = case.viscosity
    H = case.gap
    dy = H / (rows - 1)
    t_end = periods * case.period
    dt = case.period / steps_per_period
    n_steps = int(round(t_end / dt))

    y = torch.linspace(0.0, H, rows, dtype=torch.float64)
    u = case.velocity(y, t_offset).to(torch.float64)
    u[0] = case.amplitude * math.cos(case.omega * t_offset)
    u[-1] = 0.0

    r = nu * dt / (2.0 * dy * dy)
    # Tridiagonal factors are constant in time; the implicit system for the
    # interior nodes is (1 + 2r) u^{n+1}_i - r (u^{n+1}_{i-1} + u^{n+1}_{i+1})
    # = (1 - 2r) u^n_i + r (u^n_{i-1} + u^n_{i+1}), with the boundary values
    # moved to the right-hand side.
    m = rows - 2
    # Constant-coefficient implicit operator: factorise once, solve per step.
    band = torch.zeros((m, m), dtype=torch.float64)
    idx = torch.arange(m)
    band[idx, idx] = 1.0 + 2.0 * r
    band[idx[:-1], idx[:-1] + 1] = -r
    band[idx[1:], idx[1:] - 1] = -r
    lu, piv = torch.linalg.lu_factor(band)

    out = torch.empty((n_steps + 1, rows), dtype=torch.float64)
    out[0] = u
    for step in range(1, n_steps + 1):
        t_new = t_offset + step * dt
        t_old = t_new - dt
        interior = u[1:-1]
        rhs_vec = ((1.0 - 2.0 * r) * interior
                   + r * (u[:-2] + u[2:]))
        w0 = case.amplitude * math.cos(case.omega * t_new)
        w1 = 0.0
        rhs_vec = rhs_vec.clone()
        rhs_vec[0] += r * w0
        rhs_vec[-1] += r * w1
        u_new = torch.linalg.lu_solve(lu, piv, rhs_vec.reshape(-1, 1)).reshape(-1)
        u = torch.cat((torch.tensor([w0], dtype=torch.float64), u_new,
                       torch.tensor([w1], dtype=torch.float64)))
        out[step] = u
    t = t_offset + torch.arange(n_steps + 1, dtype=torch.float64) * dt
    return t, y, out


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def build_fluid(gap: float, dx: float, length: float, device) -> torch.Tensor:
    """Fluid lattice with rows at ``y = dx/2, 3dx/2, ... < gap``."""
    xs = torch.arange(dx * 0.5, length, dx, device=device)
    ys = torch.arange(dx * 0.5, gap, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def build_walls(gap: float, dx: float, length: float, layers: int,
                device) -> torch.Tensor:
    """Static particle bands one spacing outside both plates.

    The lower band sits at ``y = -dx/2``, the upper at ``y = gap + dx/2``: one
    spacing from the first fluid row, the arrangement the rest of the solver
    uses.  The bands extend three spacings past both periodic ends so the image
    clouds always see full support.
    """
    margin = 3.0 * dx
    xs = torch.arange(-margin, length + margin, dx, device=device)
    parts: List[torch.Tensor] = []
    for layer in range(1, layers + 1):
        yb = torch.full_like(xs, -dx * 0.5 - (layer - 1) * dx)
        parts.append(torch.stack((xs, yb), dim=-1))
        yt = torch.full_like(xs, gap + dx * 0.5 + (layer - 1) * dx)
        parts.append(torch.stack((xs, yt), dim=-1))
    return torch.cat(parts, dim=0)


def wall_velocity(pos_wall: torch.Tensor, case: OscillatoryCouette, t: float,
                  lower_indices: torch.Tensor):
    """Prescribed wall particle velocities at time ``t``.

    Only the lower band moves (in its own plane); the upper band is at rest.
    ``lower_indices`` is a boolean mask over the wall particles.
    """
    vel = torch.zeros_like(pos_wall)
    speed = case.amplitude * math.cos(case.omega * t)
    vel[:, 0] = torch.where(lower_indices, torch.full_like(vel[:, 0], speed),
                            torch.zeros_like(vel[:, 0]))
    return vel


def wall_acceleration(pos_wall: torch.Tensor, case: OscillatoryCouette, t: float,
                      lower_indices: torch.Tensor):
    """Wall particle accelerations ``a_w = dU_wall/dt`` at time ``t``."""
    accel = torch.zeros_like(pos_wall)
    value = -case.amplitude * case.omega * math.sin(case.omega * t)
    accel[:, 0] = torch.where(lower_indices, torch.full_like(accel[:, 0], value),
                              torch.zeros_like(accel[:, 0]))
    return accel


# --------------------------------------------------------------------------- #
# Solver driver
# --------------------------------------------------------------------------- #

@dataclass
class CaseResult:
    """Everything one resolution contributes, plus its analytic comparison."""

    delta_over_dx: float
    dx: float
    gap: float
    steps: int
    dt: float
    steps_per_period: float
    periods: float
    cases: Dict[str, Any] = field(default_factory=dict)


def run_case(gap: float, dx: float, viscosity: float, omega: float,
             amplitude: float, length: float = None, periods: float = 2.0,
             settle_periods: float = 0.0, sound_speed: float = None,
             wall_layers: int = 3, wall_friction: float = 1.0,
             device: str = "cpu", sample_from_period: float = 0.0,
             progress: bool = True, store_history: bool = False,
             history_every: int = 1) -> Dict[str, Any]:
    """Run one oscillatory-Couette case and return binned measurements.

    ``periods`` counts periods of *sampling* after ``settle_periods`` of spin-up
    (the initial condition is already the periodic field, so the default of zero
    settle periods is intentional; a caller that wants the startup transient
    gone can ask for more).

    The returned dictionary carries per-row complex amplitudes and the
    error metrics, all computed inside the solver loop (the field is never
    written to disk and re-read, so no number can be typed by hand).
    """
    case = OscillatoryCouette(gap=gap, omega=omega, viscosity=viscosity,
                              amplitude=amplitude)
    if sound_speed is None:
        # c0 = 25 U0 puts the wall Mach number at 0.04, i.e. a compressibility
        # error O(Mach^2) <= 0.16 %, while keeping the acoustic step -- and with
        # it the step count, which is (2 pi / omega) c0 / (0.3 h) per period --
        # as large as the accuracy budget allows.
        sound_speed = 25.0 * amplitude
    delta = case.delta
    ratio = delta / dx
    if ratio < 3.0:
        raise ValueError(
            f"penetration depth is only {ratio:.2f} dx; needs >= 3 (below that "
            "the wall layer is not resolved and the case measures nothing)")
    if gap / delta < 2.0:
        raise ValueError(
            f"gap is only {gap / delta:.2f} penetration depths; the analytic "
            "bounded-gap solution degenerates and the phase information dies")
    if amplitude / sound_speed > 0.05:
        raise ValueError(
            f"U0/c0 = {amplitude / sound_speed:.3f} exceeds 0.05: the acoustic "
            "error is no longer negligible against the effect being measured")

    if length is None:
        length = 8.0 * dx

    cfg = Config(dx=dx, h_factor=1.3, rho0=1000.0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 wall_layers=wall_layers, wall_friction=wall_friction,
                 density_calibration=True, adaptive_dt=False)
    dev = resolve_device(device)

    h = cfg.h_factor * dx
    dt_acoustic = cfg.max_courant * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps_per_period = case.period / dt
    if steps_per_period < 400.0:
        raise ValueError(
            f"only {steps_per_period:.0f} steps per period; the acoustic step "
            "does not resolve the oscillation (need >= 400)")

    pos = build_fluid(gap, dx, length, dev)
    n = pos.shape[0]
    mass = torch.full((n,), cfg.rho0 * dx * dx, device=dev,
                      dtype=pos.dtype)
    wall = build_walls(gap, dx, length, wall_layers, dev)
    lower = (wall[:, 1] < 0.0)

    if cfg.density_calibration:
        # Divide the lattice kernel-sum bias out of the mass so the interior
        # starts at exactly rho0 and the Tait pressure at exactly zero; the
        # spurious background pressure would otherwise bias the wall
        # extrapolation (docs/verification_initial_density_calibration.md).
        mass = calibrate_mass(mass, dx, h)

    # Start from the periodic state: no startup transient to wait out.
    vel = torch.zeros_like(pos)
    vel[:, 0] = case.velocity(pos[:, 1], 0.0)

    total_periods = settle_periods + periods
    t_end = total_periods * case.period
    t_sample_from = settle_periods * case.period
    steps = int(round(t_end / dt))

    row_index = torch.clamp((pos[:, 1] / dx).long(), 0, int(round(gap / dx)) - 1)
    n_rows = int(row_index.max()) + 1

    # Fourier projection of u_x on {cos, sin} plus the cross terms of the L2
    # error, all accumulated with index_add so the cost is O(particles).
    s_c = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
    s_s = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
    s_uu = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
    s_ua = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
    s_aa = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
    counts = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
    samples = 0

    history_t: List[float] = []
    history_u: List[List[float]] = []
    history_a: List[List[float]] = []
    max_speed = 0.0
    wall_clock = time.time()
    for step in range(steps + 1):
        t = step * dt
        if t >= t_sample_from:
            y = pos[:, 1]
            u = vel[:, 0]
            ua = case.velocity(y, t)
            c = math.cos(case.omega * t)
            s_ = math.sin(case.omega * t)
            s_c.index_add_(0, row_index, u * c)
            s_s.index_add_(0, row_index, u * s_)
            s_uu.index_add_(0, row_index, u * u)
            s_ua.index_add_(0, row_index, u * ua)
            s_aa.index_add_(0, row_index, ua * ua)
            counts.index_add_(0, row_index,
                              torch.ones(n, dtype=pos.dtype, device=dev))
            samples += 1
            if store_history and step % history_every == 0:
                # Row-mean velocity at this instant, for the u(y, t) figure.
                row_u = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
                row_a = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
                row_n = torch.zeros(n_rows, dtype=pos.dtype, device=dev)
                row_u.index_add_(0, row_index, u)
                row_a.index_add_(0, row_index, ua)
                row_n.index_add_(0, row_index,
                                 torch.ones(n, dtype=pos.dtype, device=dev))
                history_t.append(t)
                history_u.append((row_u / row_n.clamp_min(1.0)).tolist())
                history_a.append((row_a / row_n.clamp_min(1.0)).tolist())
        if step == steps:
            break

        wv = wall_velocity(wall, case, t, lower)
        wa = wall_acceleration(wall, case, t, lower)
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall, dt=dt,
                        wall_vel=wv, wall_accel=wa)
        vel = vel + dt * acc
        pos = pos + dt * vel
        # The flow is x-invariant; wrapping keeps a stray drift from leaving the
        # periodic window without disturbing the statistics.
        pos[:, 0] = torch.remainder(pos[:, 0], length)
        max_speed = max(max_speed, float(vel.abs().max()))

        if progress and steps > 0 and step % max(1, steps // 10) == 0:
            print(f"  step {step}/{steps} t={t:.4f}s "
                  f"clock={time.time() - wall_clock:.1f}s", flush=True)

    inv = 1.0 / counts.clamp_min(1.0)
    mean_c = s_c * inv
    mean_s = s_s * inv
    # u = Re U_hat cos - Im U_hat sin, so over whole periods
    # mean(u cos) = Re U_hat / 2 and mean(u sin) = -Im U_hat / 2.
    re_meas = 2.0 * mean_c
    im_meas = -2.0 * mean_s

    rows = torch.nonzero(counts > 0).reshape(-1)
    y_rows = (rows.to(pos.dtype) + 0.5) * dx
    re_ana, im_ana = case.shape(y_rows)
    amp_meas = torch.sqrt(re_meas[rows] ** 2 + im_meas[rows] ** 2)
    amp_ana = torch.sqrt(re_ana ** 2 + im_ana ** 2)
    phase_err = torch.atan2(im_meas[rows] * re_ana - re_meas[rows] * im_ana,
                            re_meas[rows] * re_ana + im_meas[rows] * im_ana)
    # Error against the analytic shape with imparted amplitude U0.
    amp_err_rel = (amp_meas - amp_ana) / case.amplitude
    mean_uu = s_uu * inv
    mean_ua = s_ua * inv
    mean_aa = s_aa * inv
    row_err2 = mean_uu[rows] - 2.0 * mean_ua[rows] + mean_aa[rows]

    rms_err = float(torch.sqrt(row_err2.mean()))
    rms_ref = float(torch.sqrt(mean_aa[rows].mean()))
    l2_rel = float(torch.sqrt(row_err2.sum() / mean_aa[rows].sum()))

    nu_eff = fit_effective_viscosity(
        y_rows, re_meas[rows], im_meas[rows], case)

    result = {
        "geometry": {
            "gap": gap, "dx": dx, "length": length, "delta": delta,
            "delta_over_dx": ratio, "gap_over_delta": gap / delta,
            "rows": int(n_rows), "n_fluid": int(n), "n_wall": int(wall.shape[0]),
        },
        "parameters": {
            "omega": omega, "period": case.period, "viscosity": viscosity,
            "amplitude": amplitude, "sound_speed": sound_speed,
            "mach_wall": amplitude / sound_speed, "wall_friction": wall_friction,
            "wall_layers": wall_layers, "gravity": cfg.gravity,
        },
        "integration": {
            "dt": dt, "steps": steps, "periods": periods,
            "settle_periods": settle_periods,
            "steps_per_period": steps_per_period, "samples": samples,
            "wall_clock_s": time.time() - wall_clock,
            "max_speed": max_speed,
        },
        "rows": {
            "y": y_rows.tolist(),
            "y_over_delta": (y_rows / delta).tolist(),
            "amplitude_measured": amp_meas.tolist(),
            "amplitude_analytic": amp_ana.tolist(),
            "amplitude_error_over_U0": amp_err_rel.tolist(),
            "phase_measured_deg": (torch.atan2(im_meas[rows], re_meas[rows])
                                   * 180.0 / math.pi).tolist(),
            "phase_analytic_deg": (torch.atan2(im_ana, re_ana)
                                   * 180.0 / math.pi).tolist(),
            "phase_error_deg": (phase_err * 180.0 / math.pi).tolist(),
            "rms_error_over_U0": torch.sqrt(row_err2).div(case.amplitude).tolist(),
        },
        "history": {
            "t": history_t,
            "u_measured_rows": history_u,
            "u_analytic_rows": history_a,
        } if store_history else None,
        "metrics": {
            "rms_error_over_U0": rms_err / case.amplitude,
            "rms_error_over_rms_reference": rms_err / rms_ref if rms_ref else float("nan"),
            "relative_l2": l2_rel,
            "max_amplitude_error_over_U0": float(amp_err_rel.abs().max()),
            "max_phase_error_deg": float(phase_err.abs().max() * 180.0 / math.pi),
            "nu_effective": nu_eff["nu_effective"],
            "nu_effective_over_nu": nu_eff["nu_effective"] / viscosity,
            "relative_l2_at_nu_effective": nu_eff["relative_l2"],
        },
    }
    return result


def _fourier_l2(case: OscillatoryCouette, y, re_meas, im_meas) -> float:
    """Relative L2 distance between measured complex amplitudes and a candidate
    analytic solution with the same ``U0`` but possibly a different viscosity."""
    re_ana, im_ana = case.shape(y)
    num = ((re_meas - re_ana) ** 2 + (im_meas - im_ana) ** 2).sum()
    den = (re_ana ** 2 + im_ana ** 2).sum()
    return float(torch.sqrt(num / den))


def fit_effective_viscosity(y, re_meas, im_meas, case: OscillatoryCouette,
                            bracket=(0.2, 5.0), iterations: int = 60,
                            probe=None) -> Dict[str, float]:
    """Fit the viscosity that the *measured* field is consistent with.

    A single multiplicative error in the viscous operator shows up as
    ``nu_eff != nu`` with the residual collapsing at the fitted value; an error
    in the wall condition instead leaves a residual at every ``nu``.  The fit
    therefore separates "wrong viscosity" from "wrong wall", which the raw
    amplitude and phase errors cannot.

    Golden-section search on a unimodal assumption in ``[0.2 nu, 5 nu]``; the
    objective is evaluated on the analytic formula only, so no extra simulation
    is needed.  ``probe`` optionally receives ``(ratio, l2)`` for every
    evaluation, which the benchmark uses to record the objective curve.
    """
    lo, hi = bracket[0] * case.viscosity, bracket[1] * case.viscosity
    phi = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(nu: float) -> float:
        candidate = OscillatoryCouette(gap=case.gap, omega=case.omega,
                                       viscosity=nu, amplitude=case.amplitude)
        value = _fourier_l2(candidate, y, re_meas, im_meas)
        if probe is not None:
            probe(nu, value)
        return value

    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = objective(d)
    nu_eff = 0.5 * (a + b)
    return {
        "nu_effective": nu_eff,
        "relative_l2": objective(nu_eff),
    }