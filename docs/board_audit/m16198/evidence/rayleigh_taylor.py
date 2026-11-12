"""Two-phase (density-contrast) WCSPH case: Rayleigh-Taylor linear growth.

Why this module exists
----------------------
The shipped 2-D solver (:mod:`tensorsph.droplet_impact`) is **single phase**:
one ``rho0``, one Tait law, no interface.  ``benchmarks/two_phase_laplace``
builds a two-phase *cloud* to verify the CSF operators, but it is a static
diagnostic, not a time integrator, and it was registered as L0 (tests + docs
only, no solver change) in ``negative_knowledge_ledger`` entry ``negk:gate:93``.
So before this module TensorSPH had **no multiphase time integrator** — the gap
recorded for SPHinXsys as ``GAP-SPH-MULTIPHASE`` ("per-particle reference
density field + interface force", ``oss_solutions``) had no local counterpart.

What is implemented
-------------------
A two-layer, variable-density weakly-compressible SPH integrator that reuses the
shared machinery (cubic-spline kernel, ``pair_kernels``,
``lattice_kernel_sum``, cell-linked pair search, the Morris viscous Laplacian
with ``VISCOUS_LAPLACIAN_FACTOR``) and adds only what a density jump needs:

1. **Per-particle reference density** ``rho0_i`` with its own Tait stiffness
   ``B_i = rho0_i c0^2 / gamma``.
2. **Number-density (volume) continuity form** instead of mass summation::

       delta_i = sum_j W_ij (+ W_ii),      rho_i = rho0_i * delta_i / delta0

   ``delta`` counts particle volumes, so ``delta_i / delta0`` is the *local
   volume compression* ``V0 / V_i`` of that particle's own phase, and
   ``rho_i = m_i / V_i`` follows.  This is the density-contrast formulation of
   Solenthaler & Pajarola (2008, *Density contrast SPH*, SCA): the smoothed
   quantity is the *compression*, which is continuous across the interface,
   instead of the *density*, which jumps 3x here.  Summing masses instead (the
   obvious change to the single-phase code) gives the particle next to the
   interface a mixture density; the magnitude of that failure is measured by
   :func:`spurious_interface_pressure` and pinned in the tests, so the rejected
   alternative is a number, not a preference.
3. **Symmetric pressure force** ``-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W``
   with per-particle masses ``m_j = rho0_j V0``: momentum-conserving, and with
   a continuous pressure and a per-phase ``rho`` it is the discrete version of
   ``-grad p / rho``, so each layer feels its own buoyancy.
4. **Analytic hydrostatic lattice initialisation** of the two-layer column:
   particles are placed with the local vertical spacing implied by the
   hydrostatic compression ``s(y) = (1 + gamma p(y)/(rho0_i c0^2))^(1/gamma)``,
   so ``delta/delta0 = s`` and the Tait law reproduces ``p(y)`` at t = 0
   without an acoustic transient.  Static wall layers (bottom + lid, free slip)
   restore the truncated kernel support and extrapolate their pressure with the
   Adami et al. (2012) eq. (11) body-force term.
5. **Exponential eigenfunction initialisation**: the single-mode kick is applied
   as ``eta0 cos(k x) exp(-k |y - y_int|)`` in *position* and the matching
   ``sigma`` velocity field in *velocity*, i.e. the linearised incompressible
   mode, so the run starts inside the growing mode instead of radiating an
   acoustic transient from a discontinuous kick.
6. **Two independent amplitude gauges**, both ``O(N)``: (a) the vertical centroid
   of the heavy phase per x column, ``ybar(x_b)``, which for a layer of uniform
   thickness gives ``y_int = 2 ybar - H`` (exact for the volume-weighted mean);
   (b) the interface-band projection of ``v_y``.  The centroid gauge is
   calibrated against the *known* initial amplitude, so a constant attenuation
   factor cannot masquerade as a growth-rate error.

Physics / reference
-------------------
Rayleigh (1883) / Taylor (1950, Proc. R. Soc. A 201:192,
doi:10.1098/rspa.1950.0052): a perturbed interface between ``rho_h > rho_l`` is
unstable under gravity when the heavier fluid rests on the lighter, with the
linear growth rate ``sigma = sqrt(A g k)`` for two deep layers,
``A = (rho_h - rho_l)/(rho_h + rho_l)``.  The finite-depth and viscous forms
used as references here are

    sigma^2 = k g (rho_h - rho_l) / (rho_h coth(k d_h) + rho_l coth(k d_l))
    sigma   = sqrt(sigma_0^2 + nu^2 k^4) - nu k^2            (viscous correction)

obtained by inserting ``exp(i k x + sigma t)`` into the linearised Euler
equations; see ``docs/benchmarks/rayleigh-taylor.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .droplet_impact import (
    VISCOUS_LAPLACIAN_FACTOR,
    cubic_spline_kernel,
    lattice_kernel_sum,
    pair_kernels,
)
from .neighbor_search import cell_linked_pairs

__all__ = [
    "RTConfig", "build_column", "hydrostatic_profile", "periodic_pairs",
    "number_density", "two_phase_rhs", "h1_kernel", "interface_centroid",
    "modal_amplitude", "velocity_amplitude", "spurious_interface_pressure",
    "theory_growth_rate", "fit_growth_rate", "fit_window", "stable_dt", "reference_state",
    "run_case", "initial_velocity",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class RTConfig:
    """Two-layer Rayleigh-Taylor column, one wavelength wide, periodic in x."""

    wavelength: float = 1.0
    depth_light: float = 0.75      # light layer below the interface
    depth_heavy: float = 0.75      # heavy layer above the interface
    dx: float = 0.01
    h_factor: float = 1.3
    rho_light: float = 500.0
    rho_heavy: float = 1500.0      # A = 0.5
    gamma: float = 7.0
    sound_speed: float = 40.0
    gravity: float = 9.81
    viscosity: float = 1.0e-3      # kinematic, same for both phases
    amplitude: float = 0.01        # 0.01 lambda
    end_time: float = 0.55
    output_every: int = 10
    pair_rebuild_every: int = 8
    max_courant: float = 0.25
    dt_cap: float = 2.0e-4
    safety_dt: float = 0.9
    wall_layers: int = 3
    wall_friction: float = 0.0     # free slip: the coth(kd) reference assumes it
    device: str = "cpu"
    dtype: str = "float64"

    @property
    def height(self) -> float:
        return self.depth_light + self.depth_heavy

    @property
    def wave_number(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def h(self) -> float:
        return self.h_factor * self.dx

    @property
    def atwood(self) -> float:
        return (self.rho_heavy - self.rho_light) / (self.rho_heavy + self.rho_light)

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


# --------------------------------------------------------------------------- #
# Analytic hydrostatic state
# --------------------------------------------------------------------------- #

def hydrostatic_profile(y: torch.Tensor, cfg: RTConfig,
                        interface: float | None = None):
    """Analytic pressure, compression ratio and phase of the column.

    Heavy fluid above ``interface`` (default ``depth_light``), light below.
    ``ratio = (1 + gamma p / (rho0 c0^2))^(1/gamma)`` is the local volume
    compression ``V0 / V`` of that phase -- exactly what ``delta/delta0``
    measures, which is why the initialisation below is consistent.
    """
    y_int = cfg.depth_light if interface is None else interface
    rho_h, rho_l, g = cfg.rho_heavy, cfg.rho_light, cfg.gravity
    p_int = rho_h * g * cfg.depth_heavy
    is_heavy = y >= y_int
    p = torch.where(is_heavy,
                    (p_int - rho_h * g * (y - y_int)).clamp_min(0.0),
                    p_int + rho_l * g * (y_int - y))
    rho0 = torch.where(is_heavy, torch.full_like(y, rho_h), torch.full_like(y, rho_l))
    ratio = (1.0 + cfg.gamma * p / (rho0 * cfg.sound_speed ** 2)).pow(1.0 / cfg.gamma)
    return p, ratio, is_heavy


def eigenfunction_decay(y: torch.Tensor, cfg: RTConfig) -> torch.Tensor:
    """``exp(-k |y - y_int|)``: the linearised two-layer mode shape."""
    return torch.exp(-cfg.wave_number * (y - cfg.depth_light).abs())


def build_column(cfg: RTConfig, perturb: bool = True):
    """Hydrostatically compressed two-layer lattice plus static wall layers.

    Returns a dict with ``pos``, ``mass``, ``rho0``, ``is_heavy`` (material
    tags), ``wall_pos``, ``wall_rho0``, ``volume0``, ``compression``, ``y_ref``.
    """
    dt = cfg.torch_dtype()
    dx = cfg.dx
    length, height = cfg.wavelength, cfg.height
    n_x = max(2, int(round(length / dx)))
    x = (torch.arange(n_x, dtype=dt) + 0.5) * (length / n_x)

    # integrate the vertical spacing dy = dx / s(y) of the hydrostatic state
    ys, y = [], 0.0
    probe = torch.zeros(1, dtype=dt)
    while y < height - 0.5 * dx:          # keep half a spacing for the lid layer
        probe[0] = y + 0.5 * dx
        _, ratio, _ = hydrostatic_profile(probe, cfg)
        step = dx / float(ratio[0])
        ys.append(y + 0.5 * step)
        y += step
    ys = torch.tensor(ys, dtype=dt) * ((height - 0.5 * dx) / ys[-1])

    xx, yy = torch.meshgrid(x, ys, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    y_ref = pos[:, 1].clone()
    if perturb and cfg.amplitude != 0.0:
        # full linearised mode displacement, not just its vertical part:
        #   xi_y = eta0 cos(kx) e^{-k|d|},  xi_x = -+ eta0 sin(kx) e^{-k|d|}
        # (sign flips across the interface).  div(xi) = 0, so the kick does not
        # compress the lattice -- a vertical-only kick would, by k eta0 (6 %),
        # which shows up directly in delta/delta0 and hence in the pressure.
        k = cfg.wave_number
        decay = eigenfunction_decay(y_ref, cfg)
        sign = torch.where(y_ref >= cfg.depth_light,
                           torch.ones_like(y_ref), -torch.ones_like(y_ref))
        pos[:, 0] = pos[:, 0] + sign * cfg.amplitude * torch.sin(
            k * pos[:, 0]) * decay
        pos[:, 1] = y_ref + cfg.amplitude * torch.cos(k * pos[:, 0]) * decay
    _, ratio0, is_heavy = hydrostatic_profile(y_ref, cfg)
    rho0 = torch.where(is_heavy, torch.full_like(pos[:, 1], cfg.rho_heavy),
                       torch.full_like(pos[:, 1], cfg.rho_light))
    volume0 = torch.full((pos.shape[0],), dx * dx, dtype=dt)
    mass = rho0 * volume0

    wpos, wrho = [], []
    for layer in range(cfg.wall_layers):
        off = (0.5 + layer) * dx
        wpos.append(torch.stack([x, torch.full_like(x, -off)], dim=1))
        wrho.append(torch.full((n_x,), cfg.rho_light, dtype=dt))
        wpos.append(torch.stack([x, torch.full_like(x, height + off)], dim=1))
        wrho.append(torch.full((n_x,), cfg.rho_heavy, dtype=dt))
    return {
        "pos": pos, "mass": mass, "rho0": rho0, "is_heavy": is_heavy,
        "wall_pos": torch.cat(wpos, dim=0), "wall_rho0": torch.cat(wrho),
        "volume0": volume0, "compression": ratio0, "y_ref": y_ref,
    }


def initial_velocity(pos, cfg: RTConfig, sigma: float | None = None) -> torch.Tensor:
    """Velocity field of the growing linear mode at ``t = 0``.

    ``u_y = sigma eta0 cos(kx) e^{-k|y-y_int|}`` and
    ``u_x = -+ sigma eta0 sin(kx) e^{-k|y-y_int|}`` (the sign flips across the
    interface, as continuity requires for a mode decaying away from it).
    """
    if sigma is None:
        sigma = theory_growth_rate(cfg)["sigma_ideal_Agk"]
    k = cfg.wave_number
    decay = eigenfunction_decay(pos[:, 1], cfg)
    x, y = pos[:, 0], pos[:, 1]
    vy = sigma * cfg.amplitude * torch.cos(k * x) * decay
    sign = torch.where(y >= cfg.depth_light, torch.ones_like(y), -torch.ones_like(y))
    vx = sign * sigma * cfg.amplitude * torch.sin(k * x) * decay
    return torch.stack([vx, vy], dim=1)


# --------------------------------------------------------------------------- #
# Periodic pair list (x only)
# --------------------------------------------------------------------------- #

def periodic_pairs(all_pos: torch.Tensor, n_fluid: int, cutoff: float,
                   length: float) -> torch.Tensor:
    """Pairs of a cloud that is periodic in x, over fluid *and* wall particles.

    The cloud is duplicated at ``x + length`` and only ``i < n, j >= n`` cross
    pairs are kept (every wrap-neighbour appears exactly once, because
    ``length`` is far larger than ``cutoff``), with the image index folded back.
    Wall-wall pairs are dropped: both are static, so they cannot reach any
    retained quantity (same argument as ``drop_wall_wall_pairs``).
    """
    n = all_pos.shape[0]
    shift = torch.tensor([length], dtype=all_pos.dtype, device=all_pos.device)
    zero = torch.zeros(n, dtype=all_pos.dtype, device=all_pos.device)
    big = torch.cat([all_pos, all_pos + torch.stack([shift.expand(n), zero], dim=1)],
                    dim=0)
    pairs = cell_linked_pairs(big, cutoff)
    if pairs.numel() == 0:
        return pairs
    i, j = pairs.unbind(dim=1)
    # keep pairs whose first member is a "real" particle; fold an image partner
    # (j >= n) back into the real cloud.  Pairs with i >= n are image-image and
    # would duplicate the real-real pairs, so they are dropped here.
    keep = i < n
    i = i[keep]
    j = j[keep]
    j = torch.where(j >= n, j - n, j)
    keep = (i < n_fluid) | (j < n_fluid)
    return torch.stack([i[keep], j[keep]], dim=1)


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #

def min_image(sep: torch.Tensor, length: float) -> torch.Tensor:
    """Minimum image of a separation in a box that is periodic in x."""
    out = sep.clone()
    out[:, 0] = out[:, 0] - length * torch.round(out[:, 0] / length)
    return out


def number_density(all_pos: torch.Tensor, pairs: torch.Tensor, h: float,
                   length: float | None = None) -> torch.Tensor:
    """Dimensionless kernel sum ``delta_i = sum_j W_ij + W_ii`` (self term).

    ``W_ij`` is added to *both* end points (the kernel is symmetric); adding it
    to one end only halves every kernel sum, which is a silent factor 2 in the
    compression -- pinned by ``tests/test_rayleigh_taylor.py``.
    """
    n = all_pos.shape[0]
    dt, device = all_pos.dtype, all_pos.device
    self_w = float(cubic_spline_kernel(torch.zeros(1, dtype=dt, device=device), h)[0])
    delta = torch.full((n,), self_w, dtype=dt, device=device)
    if pairs.numel() == 0:
        return delta
    i, j = pairs.unbind(dim=1)
    sep = all_pos[i] - all_pos[j]
    if length is not None:
        sep = min_image(sep, length)
    w = cubic_spline_kernel(sep.norm(dim=1), h)
    delta = delta.index_add(0, i, w)
    return delta.index_add(0, j, w)


def reference_state(pos, wall_pos, wall_rho0, cfg: RTConfig, pairs=None,
                    y_ref=None):
    """Per-particle reference number density and rest compression.

    ``delta_ref`` is the number density of the *initial* lattice and ``s0`` the
    analytic hydrostatic compression ``(1 + gamma p/(rho0 c0^2))^(1/gamma)`` at
    the particle's initial height.  Dividing by ``delta_ref`` instead of by the
    uniform ``delta0`` removes the lattice-sum bias of each particle (the same
    idea as ``Config.calibrate_initial_mass`` in the single-phase solver, whose
    ``+5.9e-3`` bias is documented in
    ``docs/verification_initial_density_calibration.md``), so the pressure at
    t = 0 is the analytic hydrostatic profile *exactly* instead of being a
    discretisation-error pattern amplified by ``gamma B``.
    """
    all_pos = torch.cat((pos, wall_pos)) if wall_pos is not None else pos
    if pairs is None:
        n = pos.shape[0]
        pairs = periodic_pairs(all_pos, n, 2.0 * cfg.h, cfg.wavelength)
    delta_ref = number_density(all_pos, pairs, cfg.h, cfg.wavelength)
    y = all_pos[:, 1]
    # the rest compression is a *material* label: it belongs to the particle's
    # undisplaced height (the perturbed kick would otherwise move interface
    # particles into the other phase's reference and break pressure continuity)
    y_fluid = y[:pos.shape[0]] if y_ref is None else y_ref
    _, s0_fluid, _ = hydrostatic_profile(
        torch.clamp(y_fluid, 0.0, cfg.height), cfg)
    if wall_pos is not None:
        s0 = torch.cat((s0_fluid, torch.ones_like(y[pos.shape[0]:])))
    else:
        s0 = s0_fluid
    return delta_ref, s0


def two_phase_rhs(pos, vel, mass, rho0, cfg: RTConfig, wall_pos=None,
                  wall_rho0=None, pairs=None, ref=None):
    """Acceleration and state of the two-phase column.

    Returns ``(rho, pressure, acc, ratio)`` for the fluid particles, where
    ``ratio = delta / delta0`` is each particle's own-phase compression.
    """
    h = cfg.h
    n = pos.shape[0]
    device, dt = pos.device, pos.dtype
    if wall_pos is None:
        all_pos, all_mass, all_rho0, all_vel, nw = pos, mass, rho0, vel, 0
    else:
        nw = wall_pos.shape[0]
        all_pos = torch.cat((pos, wall_pos), dim=0)
        all_mass = torch.cat((mass, wall_rho0 * (cfg.dx * cfg.dx)))
        all_rho0 = torch.cat((rho0, wall_rho0))
        all_vel = torch.cat((vel, torch.zeros_like(wall_pos)))
    if pairs is None:
        pairs = periodic_pairs(all_pos, n, 2.0 * h, cfg.wavelength)

    delta0 = lattice_kernel_sum(h, cfg.dx)
    if ref is None:
        delta_ref, s0 = reference_state(
            pos, wall_pos, wall_rho0 if wall_rho0 is not None
            else torch.ones(0, dtype=dt), cfg, pairs)
    else:
        delta_ref, s0 = ref
    delta = number_density(all_pos, pairs, h, cfg.wavelength)
    ratio = ((delta / delta_ref).clamp_min(0.5) * s0).clamp_min(0.5)
    rho_all = all_rho0 * ratio
    bulk = all_rho0 * cfg.sound_speed ** 2 / cfg.gamma
    pressure_all = (bulk * (ratio.pow(cfg.gamma) - 1.0)).clamp_min(0.0)

    gravity = torch.zeros_like(pos)
    gravity[:, 1] = -cfg.gravity

    if pairs.numel() == 0:
        return rho_all[:n], pressure_all[:n], gravity, ratio[:n]

    i, j = pairs.unbind(dim=1)
    sep = min_image(all_pos[i] - all_pos[j], cfg.wavelength)
    wij, grad_w = pair_kernels(sep, h)

    if nw:
        # Adami et al. (2012) eq. (11), body term with g = (0, -gravity):
        #   p_w = [sum_f p_f W + sum_f rho_f (g - a_w).(r_w - r_f) W] / sum_f W
        i_fluid = i < n
        crossing = i_fluid != (j < n)
        fluid_idx = torch.where(i_fluid, i, j)
        wall_idx = (torch.where(i_fluid, j, i) - n).clamp_(0, nw - 1)
        weight = torch.where(crossing, wij, torch.zeros_like(wij))
        dy = torch.where(i_fluid, -sep[:, 1], sep[:, 1])
        source = pressure_all[fluid_idx] + rho_all[fluid_idx] * (-cfg.gravity * dy)
        num = torch.zeros(nw, dtype=dt, device=device).index_add(0, wall_idx,
                                                                 source * weight)
        den = torch.zeros(nw, dtype=dt, device=device).index_add(0, wall_idx, weight)
        wall_p = (num / den.clamp_min(1.0e-30)).clamp_min(0.0)
        wall_ratio = (wall_p / (wall_rho0 * cfg.sound_speed ** 2 / cfg.gamma)
                      + 1.0).pow(1.0 / cfg.gamma)
        pressure_all = torch.cat((pressure_all[:n], wall_p))
        rho_all = torch.cat((rho_all[:n], wall_rho0 * wall_ratio))

    # Density-contrast symmetric pressure force (Grenier et al. 2009,
    # Hamiltonian interface SPH):
    #     a_i = -sum_j m_j (p_i + p_j)/(rho_i rho_j) grad_i W_ij
    # For rho_i = rho_j this is *identically* the single-phase form
    # m_j (p_i/rho_i^2 + p_j/rho_j^2) already used by tensorsph.droplet_impact;
    # for a density jump it is the form that keeps the discrete hydrostatic
    # balance phase-blind (a light particle's heavy neighbour contributes
    # V_j 2p/rho_i, the same as a light one).  The pair term is symmetric, so
    # momentum is still conserved exactly.
    pressure_term = ((pressure_all[i] + pressure_all[j])
                     / (rho_all[i] * rho_all[j]))
    pair_pressure = pressure_term[:, None] * grad_w
    acc = torch.zeros((n + nw, pos.shape[1]), dtype=dt, device=device)
    acc = acc.index_add(0, i, -all_mass[j, None] * pair_pressure)
    acc = acc.index_add(0, j, all_mass[i, None] * pair_pressure)
    acc_pressure = acc[:n]

    # Morris et al. (1997) Laplacian with the calibrated factor 2, exactly as
    # tensorsph.droplet_impact applies it (test_viscosity_is_calibrated pins it).
    fluid_pair = (i < n) & (j < n)
    if nw and cfg.wall_friction > 0.0:
        cross = torch.logical_xor(i < n, j < n)
        blend = fluid_pair.to(dt) + cfg.wall_friction * cross.to(dt)
        visc_scale = (VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity * blend)[:, None]
    else:
        visc_scale = (VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity
                      * fluid_pair.to(dt))[:, None]
    dv = all_vel[i] - all_vel[j]
    denom = sep.square().sum(dim=1) + 0.01 * h * h
    laplace_weight = torch.relu(-(sep * grad_w).sum(dim=1)) / denom
    visc_i = (-visc_scale * dv * (all_mass[j] / rho_all[j] * laplace_weight)[:, None])
    visc_j = (visc_scale * dv * (all_mass[i] / rho_all[i] * laplace_weight)[:, None])
    acc_visc = torch.zeros((n + nw, pos.shape[1]), dtype=dt, device=device)
    acc_visc = acc_visc.index_add(0, i, visc_i)
    acc_visc = acc_visc.index_add(0, j, visc_j)

    return (rho_all[:n], pressure_all[:n],
            acc_pressure + acc_visc[:n] + gravity, ratio[:n])


# --------------------------------------------------------------------------- #
# Interface gauges
# --------------------------------------------------------------------------- #

def h1_kernel(s: torch.Tensor, h: float) -> torch.Tensor:
    """Normalised 1-D cubic-spline kernel, ``int K ds = 1``, support ``2h``."""
    q = (s.abs() / h).clamp(max=2.0)
    inner = 2.0 / 3.0 - q.square() + 0.5 * q.pow(3)
    outer = (2.0 - q).clamp_min(0.0).pow(3) / 6.0
    return torch.where(q < 1.0, inner, outer) / h


def interface_centroid(pos, volume, is_heavy, cfg: RTConfig,
                       n_bins: int | None = None, h_smooth: float | None = None):
    """Vertical centroid ``ybar(x_b)`` of the heavy phase, per x column.

    For a layer of near-uniform thickness between ``y_int(x)`` and ``H`` the
    centroid is ``(y_int(x) + H)/2``, so ``y_int = 2 ybar - H``: the interface
    height is read off a volume-weighted first moment instead of a contour or a
    particle tag.  The x smoothing is a kernel regression over the column, whose
    attenuation of a smooth mode is a constant factor removed by the
    calibration in :func:`run_case`.
    """
    h_s = cfg.h if h_smooth is None else h_smooth
    n_bins = int(round(cfg.wavelength / cfg.dx)) if n_bins is None else n_bins
    x_b = (torch.arange(n_bins, dtype=pos.dtype, device=pos.device) + 0.5) \
        * (cfg.wavelength / n_bins)
    xj = pos[is_heavy, 0]
    yj = pos[is_heavy, 1]
    vj = volume[is_heavy]
    dxp = xj[None, :] - x_b[:, None]
    dxp = dxp - cfg.wavelength * torch.round(dxp / cfg.wavelength)
    w = vj[None, :] * h1_kernel(dxp, h_s)
    return (w * yj[None, :]).sum(dim=1) / w.sum(dim=1).clamp_min(1e-300), x_b


def modal_amplitude(pos, volume, is_heavy, cfg: RTConfig, scale: float = 1.0,
                    n_bins: int | None = None) -> tuple[float, float]:
    """Amplitude and phase of the interface at ``k = 2 pi / lambda``.

    ``eta(x) = 2 (ybar(x) - mean ybar)`` is the interface elevation (mean
    removed, so a volume drift cannot leak in) and the returned amplitude is its
    cosine coefficient ``2 |<eta exp(-i k x)>|`` times ``scale``.
    """
    ybar, x_b = interface_centroid(pos, volume, is_heavy, cfg, n_bins=n_bins)
    eta = 2.0 * (ybar - ybar.mean())
    k = cfg.wave_number
    ck = (eta.to(torch.complex128) * torch.exp(
        -1j * k * x_b.to(torch.float64))).mean()
    return float(2.0 * ck.abs()) * scale, float(torch.angle(ck))


def velocity_amplitude(pos, vel, volume, cfg: RTConfig) -> float:
    """Independent gauge: modal amplitude of ``v_y`` in the interface band."""
    band = (pos[:, 1] - cfg.depth_light).abs() <= 1.5 * cfg.h
    xj, vyj, vj = pos[band, 0], vel[band, 1], volume[band]
    if xj.numel() == 0:
        return float("nan")
    k = cfg.wave_number
    ck = (vyj * vj * torch.exp(-1j * k * xj.to(torch.float64))).sum() / vj.sum()
    return float(2.0 * ck.abs())


# --------------------------------------------------------------------------- #
# Reference values
# --------------------------------------------------------------------------- #

def theory_growth_rate(cfg: RTConfig, viscosity: float | None = None,
                       finite_depth: bool = True) -> dict:
    """Linear growth-rate references: ideal, finite-depth, viscous-corrected."""
    k = cfg.wave_number
    g = cfg.gravity
    rho_h, rho_l = cfg.rho_heavy, cfg.rho_light
    nu = cfg.viscosity if viscosity is None else viscosity
    sigma_ideal = math.sqrt(cfg.atwood * g * k)
    if finite_depth:
        denom = (rho_h / math.tanh(k * cfg.depth_heavy)
                 + rho_l / math.tanh(k * cfg.depth_light))
        s2 = k * g * (rho_h - rho_l) / denom
    else:
        s2 = cfg.atwood * g * k
    sigma_fd = math.sqrt(s2)
    sigma_visc = math.sqrt(s2 + (nu * k * k) ** 2) - nu * k * k
    return {"sigma_ideal_Agk": sigma_ideal, "sigma_finite_depth": sigma_fd,
            "sigma_viscous": sigma_visc,
            "finite_depth_correction": (sigma_fd / sigma_ideal
                                        if sigma_ideal > 0 else float("nan")),
            "viscous_correction": (sigma_visc / sigma_ideal
                                   if sigma_ideal > 0 else float("nan")),
            "k": k, "atwood": cfg.atwood}


def fit_window(times, values, t0: float, t1: float) -> dict:
    """Least-squares slope of ``ln values`` over the time window ``[t0, t1]``."""
    import numpy as _np
    t = _np.asarray(times, dtype=float)
    v = _np.asarray(values, dtype=float)
    mask = (t >= t0) & (t <= t1) & (v > 0)
    if mask.sum() < 6:
        return {"sigma": float("nan"), "r2": float("nan"), "n_fit": int(mask.sum())}
    tt, lv = t[mask], _np.log(v[mask])
    slope, intercept = _np.polyfit(tt, lv, 1)
    pred = slope * tt + intercept
    ss_res = float(((lv - pred) ** 2).sum())
    ss_tot = float(((lv - lv.mean()) ** 2).sum())
    return {"sigma": float(slope), "r2": 1.0 - ss_res / max(ss_tot, 1e-300),
            "n_fit": int(mask.sum()), "t_start": float(tt[0]),
            "t_end": float(tt[-1]), "v_start": float(v[mask][0]),
            "v_end": float(v[mask][-1])}


def fit_growth_rate(times, amplitudes, lo: float, hi: float) -> dict:
    """Least-squares slope of ``ln a`` over ``lo <= a <= hi`` (single mode)."""
    import numpy as _np
    t = _np.asarray(times, dtype=float)
    a = _np.asarray(amplitudes, dtype=float)
    mask = (a >= lo) & (a <= hi)
    if mask.sum() < 6:
        return {"sigma": float("nan"), "r2": float("nan"),
                "n_fit": int(mask.sum()), "window": [lo, hi]}
    tt, la = t[mask], _np.log(a[mask])
    slope, intercept = _np.polyfit(tt, la, 1)
    pred = slope * tt + intercept
    ss_res = float(((la - pred) ** 2).sum())
    ss_tot = float(((la - la.mean()) ** 2).sum())
    return {"sigma": float(slope), "r2": 1.0 - ss_res / max(ss_tot, 1e-300),
            "n_fit": int(mask.sum()), "window": [lo, hi],
            "t_start": float(tt[0]), "t_end": float(tt[-1]),
            "a_start": float(a[mask][0]), "a_end": float(a[mask][-1])}


# --------------------------------------------------------------------------- #
# Quantifying the rejected alternative
# --------------------------------------------------------------------------- #

def spurious_interface_pressure(cfg: RTConfig) -> dict:
    """Pressure a *mass-summation* Tait law would give at the interface.

    The obvious way to make the single-phase solver two-phase is to keep
    ``rho_i = sum_j m_j W_ij`` and give each particle its own ``rho0_i`` in the
    Tait law.  Mass summation smooths the *density*, which jumps 3x, over ~2h,
    so a particle within 2h of the interface is assigned a mixture density and
    the Tait law fires on the ratio to *its own* reference.  This function
    reports that pressure next to the hydrostatic value it should have had.
    """
    dt = cfg.torch_dtype()
    dx, h = cfg.dx, cfg.h
    n_x = int(round(cfg.wavelength / dx))
    x = (torch.arange(n_x, dtype=dt) + 0.5) * dx
    y_int = cfg.depth_light
    y_rows = torch.tensor([y_int - 0.5 * dx, y_int + 0.5 * dx], dtype=dt)
    depth = torch.arange(1, 8, dtype=dt) * dx
    grid_y = torch.cat([y_rows, y_int - depth, y_int + depth])
    xx, yy = torch.meshgrid(x, grid_y, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    is_heavy = pos[:, 1] >= y_int
    rho0 = torch.where(is_heavy, torch.full_like(pos[:, 1], cfg.rho_heavy),
                       torch.full_like(pos[:, 1], cfg.rho_light))
    mass = rho0 * dx * dx
    zero = torch.zeros_like(pos[:, 0])
    big = torch.cat([pos, pos + torch.stack([torch.full_like(zero, cfg.wavelength),
                                             zero], dim=1)], dim=0)
    pairs = cell_linked_pairs(big, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    keep = i < pos.shape[0]
    i, j = i[keep], torch.where(j[keep] >= pos.shape[0],
                               j[keep] - pos.shape[0], j[keep])
    keep = (i < pos.shape[0]) & (j < pos.shape[0])
    i, j = i[keep], j[keep]
    r = min_image(pos[i] - pos[j], cfg.wavelength).norm(dim=1)
    w = cubic_spline_kernel(r, h)
    self_w = float(cubic_spline_kernel(torch.zeros(1, dtype=dt), h)[0])
    rho = mass * self_w
    rho = rho.index_add(0, i, mass[j] * w)
    rho = rho.index_add(0, j, mass[i] * w)
    bulk = rho0 * cfg.sound_speed ** 2 / cfg.gamma
    p = (bulk * ((rho / rho0).clamp_min(0.01).pow(cfg.gamma) - 1.0)).clamp_min(0.0)
    p_ref, _, _ = hydrostatic_profile(y_rows, cfg)
    light_row, heavy_row = float(p[~is_heavy].mean()), float(p[is_heavy].mean())
    return {
        "p_light_at_interface": light_row,
        "p_heavy_at_interface": heavy_row,
        "p_hydrostatic_light_row": float(p_ref[0]),
        "p_hydrostatic_heavy_row": float(p_ref[1]),
        "error_ratio_light": light_row / max(float(p_ref[0]), 1e-30),
        "error_ratio_heavy": heavy_row / max(float(p_ref[1]), 1e-30),
        "rho_mass_summation_light": float(rho[~is_heavy].mean()),
        "rho_mass_summation_heavy": float(rho[is_heavy].mean()),
        "rho0_light": cfg.rho_light, "rho0_heavy": cfg.rho_heavy,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def stable_dt(cfg: RTConfig, max_speed: float) -> float:
    """Acoustic Courant + gravity + viscous limits, capped by ``dt_cap``."""
    h = cfg.h
    dt_c = cfg.max_courant * h / (cfg.sound_speed + max_speed)
    dt_g = 0.25 * math.sqrt(h / cfg.gravity)
    dt_v = 0.125 * h * h / max(cfg.viscosity, 1e-30)
    return cfg.safety_dt * min(cfg.dt_cap, dt_c, dt_g, dt_v)


def run_case(cfg: RTConfig, snapshot: bool = True, progress=None) -> dict:
    """Integrate the perturbed two-layer column and fit the growth rate."""
    torch.manual_seed(0)
    state = build_column(cfg, perturb=True)
    pos = state["pos"].clone()
    vel = initial_velocity(pos, cfg)
    mass, rho0 = state["mass"], state["rho0"]
    is_heavy, volume = state["is_heavy"], state["volume0"]
    wall_pos, wall_rho0 = state["wall_pos"], state["wall_rho0"]
    ref = theory_growth_rate(cfg)

    # calibrate the centroid gauge against the known initial amplitude
    a0_raw, _ = modal_amplitude(pos, volume, is_heavy, cfg, scale=1.0)
    scale = cfg.amplitude / a0_raw if a0_raw > 0 else 1.0

    times, amps, amps_v, phases, ek, mach = [], [], [], [], [], []
    rho_min, rho_max = [], []

    def sample(t, v, rho=None):
        a, ph = modal_amplitude(pos, volume, is_heavy, cfg, scale=scale)
        times.append(t)
        amps.append(a)
        amps_v.append(velocity_amplitude(pos, v, volume, cfg))
        phases.append(ph)
        ek.append(0.5 * float((mass * v.square().sum(dim=1)).sum()))
        mach.append(float(v.norm(dim=1).max()) / cfg.sound_speed)
        if rho is not None:
            rho_min.append(float(rho.min()))
            rho_max.append(float(rho.max()))

    pairs = periodic_pairs(torch.cat((pos, wall_pos)), pos.shape[0],
                           2.0 * cfg.h, cfg.wavelength)
    ref_state = reference_state(pos, wall_pos, wall_rho0, cfg, pairs,
                                y_ref=state["y_ref"])
    t, dt, steps, pair_age = 0.0, cfg.dt_cap, 0, 0
    sample(0.0, vel)
    snap_rho = snap_p = snap_pos = None
    while t < cfg.end_time:
        rho, pressure, acc, ratio = two_phase_rhs(
            pos, vel, mass, rho0, cfg, wall_pos, wall_rho0, pairs, ref_state)
        vel = vel + dt * acc
        pos = pos + dt * vel
        t += dt
        steps += 1
        pair_age += 1
        speed = float(vel.norm(dim=1).max())
        if pair_age >= cfg.pair_rebuild_every:
            pairs = periodic_pairs(torch.cat((pos, wall_pos)), pos.shape[0],
                                   2.0 * cfg.h, cfg.wavelength)
            pair_age = 0
        if steps % cfg.output_every == 0:
            sample(t, vel, rho)
            snap_rho, snap_p = rho, pressure
            if snapshot:
                snap_pos = pos.clone()
            if progress is not None and steps % (cfg.output_every * 20) == 0:
                progress(t, amps[-1], speed)
        dt = stable_dt(cfg, speed)

    lam = cfg.wavelength
    fit = fit_growth_rate(times, amps, lo=0.02 * lam, hi=0.1 * lam)
    fit_v = {}
    if fit.get("n_fit", 0) >= 6:
        fit_v = fit_window(times, amps_v, fit["t_start"], fit["t_end"])
    result = {
        "config": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                   for k, v in cfg.__dict__.items()},
        "derived": {"k": cfg.wave_number, "atwood": cfg.atwood,
                    "delta0": lattice_kernel_sum(cfg.h, cfg.dx),
                    "particles": int(pos.shape[0]),
                    "wall_particles": int(wall_pos.shape[0]),
                    "steps": steps, "final_dt": dt, "end_time": t,
                    "gauge_scale": scale,
                    "amplitude_t0_raw": a0_raw},
        "reference": ref,
        "fit": fit,
        "fit_velocity_gauge": fit_v,
        "sigma_over_ideal": fit["sigma"] / ref["sigma_ideal_Agk"],
        "sigma_over_finite_depth": fit["sigma"] / ref["sigma_finite_depth"],
        "sigma_over_viscous": fit["sigma"] / ref["sigma_viscous"],
        "sigma_velocity_over_ideal": (fit_v.get("sigma", float("nan"))
                                      / ref["sigma_ideal_Agk"]),
        "max_amplitude": max(amps) if amps else float("nan"),
        "mass_drift": float((mass * 1.0).sum()) * 0.0,  # mass is conserved by construction
        "series": {"t": times, "amplitude": amps, "amplitude_velocity": amps_v,
                   "phase": phases, "kinetic_energy": ek, "mach": mach,
                   "rho_min": rho_min, "rho_max": rho_max},
    }
    if snapshot and snap_pos is not None:
        result["fields"] = _field_snapshot(snap_pos, snap_rho, snap_p, rho0, cfg)
    return result


def _field_snapshot(pos, rho, pressure, rho0, cfg: RTConfig) -> dict:
    """Shepard-interpolated fields on a regular grid (60 x 90) for the figure."""
    nx, ny = 60, 90
    x = (torch.arange(nx, dtype=pos.dtype) + 0.5) * (cfg.wavelength / nx)
    y = (torch.arange(ny, dtype=pos.dtype) + 0.5) * (cfg.height / ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    h = cfg.h
    vol = torch.full((pos.shape[0],), cfg.dx * cfg.dx, dtype=pos.dtype)
    out = {}
    chunk = 2048
    names = ("rho", "pressure", "phase_index")
    acc = {k: [] for k in names}
    for start in range(0, grid.shape[0], chunk):
        g = grid[start:start + chunk]
        d = g[:, None, :] - pos[None, :, :]
        d[:, :, 0] = d[:, :, 0] - cfg.wavelength * torch.round(d[:, :, 0] / cfg.wavelength)
        w = cubic_spline_kernel(d.norm(dim=2), h)
        wsum = w.sum(dim=1).clamp_min(1e-30)
        for name, field in (("rho", rho), ("pressure", pressure),
                            ("phase_index", rho0 / cfg.rho_heavy)):
            acc[name].append(((w * (field * vol)[None, :]).sum(dim=1) / wsum))
    for name in names:
        out[name] = torch.cat(acc[name]).reshape(nx, ny).tolist()
    out["x"] = x.tolist()
    out["y"] = y.tolist()
    return out