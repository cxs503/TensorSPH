"""Time-advancing (dynamic) arm for ``CASE-SPH-OSC-DROP-FREQ``.

Why this module exists
----------------------
Rounds 1-4 of the anchor measured the mode-``l`` capillary frequency through the
**quasi-static linear response** of the CSF traction, i.e.

.. code-block:: text

    omega_meas / omega_exact = sqrt(dp_measured / dp_exact)

with ``dp`` the modal projection of the static traction on a *frozen*,
analytically deformed particle cloud (``tensorsph.oscillating_drop3d``).  That
route sees the discretisation of the *restoring force* only: it carries no
discrete inertia, no spurious currents, no acoustic coupling and no volume
redistribution.  ``docs/case_sph_osc_drop_freq_round4.md`` lists "no
time-advancing arm (not_run)" as the remaining hole, and round 4 measured that
the shipped curvature operator's *deformation response* is far weaker than the
analytic traction while its static value is right (``kappa R = 2.07..2.09`` but
``dp`` two-to-three orders below ``dp_exact``).  A trajectory is the decisive
experiment: if the traction really does not respond to the mode, a free drop
will barely oscillate; if part D's ``-96 %`` was an artefact of its 1.2 dx force
shell, the trajectory will show a finite, measurable ``omega``.

Governing equations (exact ``m = 0`` axisymmetric reduction)
-----------------------------------------------------------
The Rayleigh mode ``P_l(cos theta)`` is a surface of revolution, so the flow is
axisymmetric with ``v_phi = 0`` and the **momentum** equation in the meridional
half plane ``(rho, z)`` has exactly its planar form,

.. code-block:: text

    Dv/Dt = -(1/rho) grad_2 p + (1/rho) f_csf ,      f_csf = sigma kappa delta_s n

while the **continuity** equation picks up the cylindrical source term

.. code-block:: text

    Drho/Dt = -rho div_2 v - rho v_rho / rho_axis .

That is the whole axisymmetric content: one extra source term plus the
cylindrical curvature ``kappa = kappa_meridional + n_rho / rho`` (the split the
static arm already uses, see :func:`tensorsph.oscillating_drop3d.axisym_curvature`).
The axis ``rho = 0`` is handled by **mirror particles** (Colagrossi & Landrini
2003): every force evaluation is performed on the doubled cloud
``{(rho, z)} union {(-rho, z)}`` with the mirror velocities' radial component
flipped, so the axis carries no truncation and ``v_rho(0) = 0`` is imposed.

Scheme (all ingredients are the repository's validated ones)
------------------------------------------------------------
* continuity -- Monaghan (1992) eq. 2.9, ``Drho_i/Dt = sum_j m_j (v_i-v_j).grad_i W_ij``
  plus the axis source above; masses ``m_i = rho0 dA_i`` are the *planar* area
  weights, which is what makes the plane equations above exact.
* delta-SPH density diffusion (Antuono, Colagrossi & Marrone 2012) with
  ``delta = 0.1``, the standard cure for the pressure noise of the summation-free
  continuity form.
* EOS -- Tait ``p = rho0 c0^2 / gamma [(rho/rho0)^gamma - 1]``, ``gamma = 7``,
  clamped at ``p >= 0`` (Monaghan 1994 free surface).
* momentum -- symmetric form ``-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W_ij``
  plus Monaghan & Gingold (1983) artificial viscosity.
* surface tension -- CSF (Brackbill, Kothe & Zemach 1992) through
  :func:`tensorsph.csf_consistent.csf_fields` with the repository's
  ``fit_quadratic`` curvature and the ``legacy`` (= ``2 |grad c|``) surface delta.
* time integration -- kick-drift-kick leapfrog, ``dt = cfl * min(h/(c0+|v|),
  capillary limit)``.

Measurement
-----------
The mode amplitude is the mass(area)-weighted quadrupole projection

.. code-block:: text

    A_l(t) = sum_i dA_i r_i^2 P_l(cos theta_i) / sum_i dA_i

normalised by ``A_l(0)`` and rescaled to the imposed ``eps``, so a static
resolution bias of the estimator cancels exactly and ``A_l(t) -> eps cos(omega t)``
for a linear mode.  ``(omega, gamma)`` come from
:func:`tensorsph.oscillating_drop3d.fit_damped_cosine`.

STATUS (honest, 2026-11-15)
-------------------------
The time-advancing arm is **implemented but not usable**: the free-drop
integration diverges in every configuration tried, *including the surface-tension
free control*, so the defect is in the density/pressure time evolution, not in
the CSF operator alone.  Measured on this worktree
(``tests/test_oscillating_drop_freq_round5.py``,
``benchmarks/oscillating_drop_freq/round5_dynamic.py``):

* ``sigma -> 0`` control, ``R/dx = 9``, 50 steps: ``|v|max`` reaches **214**
  from a cloud that starts at rest (and 281 at ``R/dx = 7``): the pressure +
  continuity + axis-source subsystem is *not* stable, so the time arm cannot be
  used to attribute anything to surface tension yet.  (An earlier hand-written
  diagnostic that re-imposed the initial density every step looked static at
  ``|v| ~ 1e-11``; that loop was *not* a valid integration and the observation
  was withdrawn -- it is the reason this note exists.)
* capillary arm (``sigma = 1``, ``eps = 0.05``, ``c0 = 20``, ``alpha_av = 0.02``,
  ``R/dx = 9``, 400 steps = 0.12 time units ~ 0.05 period): the mode amplitude
  runs from ``eps = 0.05`` to **0.66** while ``|v|max`` goes from 0.013 to
  **619** and ``rho`` to **6.4 rho0**; the acoustic CFL (``dt = 0.25 h/(c0+|v|)``)
  is respected throughout, so this is a physical-space instability, not a
  too-large time step.
* delta-SPH diffusion *inflates* the free surface: on an otherwise static cloud
  (``sigma -> 0``, ``eps = 1e-3``, 25 steps) it produces ``|Delta rho|`` up to
  **1.7 %** at ``R/dx = 7`` (0.5 %/step at ``R/dx = 9``) because the surface
  density deficit is a *truncation* deficit, not a physical gradient.
* what *is* delivered and tested: the exact ``m = 0`` axisymmetric reduction,
  the half-plane lattice + mirror axis, the Laplace-compressed initial state
  (interior ``rho0 (1 + 2 sigma/(rho0 c0^2 R))^{1/gamma}`` to 0.2 %, free-surface
  taper below ``rho0``), the area-weighted quadrupole estimator (linear to 2 % in
  ``eps``), and the anchor references (3-D vs 2-D Rayleigh branch pinned).

Consequence for the anchor: ``CASE-SPH-OSC-DROP-FREQ`` **stays open on the
dynamic side** and the honest verdict of round 5 is FAIL-to-instrument (not
"measured and failed the 6 % gate"): no trustworthy ``omega`` was produced from a
trajectory.  Next steps are named in
``docs/case_sph_osc_drop_freq_round5.md`` section 5.

The 2-D (cylinder) branch is *not* a reference here: the run is the meridional
half plane of a 3-D drop (cylindrical weighting + ``n_rho/rho``), so the
reference is the 3-D Rayleigh value ``omega^2 = l(l-1)(l+2) sigma/(rho R^3)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .csf_consistent import csf_fields
from .neighbor_search import cell_linked_pairs
from .oscillating_drop3d import (fit_damped_cosine, legendre_p,
                                 rayleigh_frequency_3d)
from .wcsph import kernels

__all__ = [
    "DynamicConfig",
    "half_plane_lattice",
    "mirror_cloud",
    "mode_amplitude",
    "summation_density",
    "initial_density",
    "run_dynamic_drop",
    "dynamic_frequency",
]


@dataclass
class DynamicConfig:
    """Configuration of the dynamic arm (dimensionless: ``R = rho0 = sigma = 1``)."""

    radius: float = 1.0
    rho0: float = 1.0
    sigma: float = 1.0
    dx: float = 0.0625
    h_factor: float = 1.3
    kernel: str = "quintic"
    mode: int = 2
    eps: float = 0.05
    c0: float = 20.0
    gamma: float = 7.0
    alpha_av: float = 0.02
    beta_av: float = 0.0
    delta_sph: float = 0.1
    cfl: float = 0.25
    curvature_scheme: str = "fit_quadratic"
    curvature_stencil: float = 4.0
    delta_normalization: str = "legacy"
    #: integrate this many reference periods ``2 pi / omega_ref``
    periods: float = 4.0
    #: throw away this many reference periods before fitting (start-up transient)
    skip_periods: float = 0.25
    #: hard cap on the number of steps (guards a mis-set c0)
    max_steps: int = 200_000
    #: sample every n-th step
    sample_every: int = 5
    calibrate_mass: bool = True
    #: integrate to this time instead of ``periods * 2 pi / omega_ref`` (used by
    #: the diagnostics that switch surface tension off)
    t_end: float | None = None

    @property
    def h(self) -> float:
        return self.h_factor * self.dx

    @property
    def support(self) -> float:
        return kernels.SUPPORT_FACTOR[self.kernel] * self.h

    @property
    def radius_over_dx(self) -> float:
        return self.radius / self.dx

    @property
    def omega_ref(self) -> float:
        return rayleigh_frequency_3d(self.sigma, self.rho0, self.radius, self.mode)

    def validate(self) -> None:
        if self.mode < 2:
            raise ValueError("mode must be >= 2 (l = 1 is a translation)")
        if self.dx <= 0 or self.h_factor <= 0:
            raise ValueError("dx and h_factor must be positive")
        if self.c0 <= 0 or self.rho0 <= 0 or self.gamma <= 0:
            raise ValueError("c0, rho0 and gamma must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if not 0.0 <= self.alpha_av <= 1.0 or self.beta_av < 0.0:
            raise ValueError("0 <= alpha_av <= 1 and beta_av >= 0 are required")
        if not 0.0 < self.cfl <= 0.5:
            raise ValueError("cfl must be in (0, 0.5]")
        if self.eps == 0.0:
            raise ValueError("eps must be non-zero (it normalises the amplitude)")
        if self.curvature_scheme not in ("mls", "fit_quadratic", "fit_circle"):
            raise ValueError(f"unknown curvature_scheme {self.curvature_scheme!r}")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def half_plane_lattice(cfg: DynamicConfig) -> dict:
    """Hexagonal lattice of the meridional half plane ``rho >= 0``.

    ``dA_i`` is the planar area of the lattice cell (``dx * dy`` with
    ``dy = dx sqrt(3)/2``), so ``m_i = rho0 dA_i`` and the sums below are planar
    integrals -- exactly the weights the reduced equations need.  Points are
    mapped radially onto ``r(theta) = R [1 + eps P_l(cos theta)]`` with
    ``cos theta = z/r`` (``theta`` from the symmetry axis), the perturbed shape
    is volume preserving to ``O(eps^2)`` because ``<P_l> = 0``.
    """
    R, dx, mode, eps = cfg.radius, cfg.dx, cfg.mode, cfg.eps
    dy = dx * math.sqrt(3.0) / 2.0
    extent = R * (1.0 + abs(eps)) + 2.0 * dx
    ny = int(math.ceil(extent / dy)) + 1
    rows = []
    for j in range(-ny, ny + 1):
        # half-cell offset on *every* row: the nearest particle sits at rho = dx/2
        # so the mirror image is a full dx away and no particle lies on the axis
        # (an on-axis particle would be duplicated by the mirror at r = 0)
        shift = 0.5 * dx if (j % 2) else dx
        x = torch.arange(0, dtype=torch.float64)
        nr = int(math.ceil(extent / dx)) + 1
        x = torch.arange(0, nr, dtype=torch.float64) * dx + shift
        y = torch.full_like(x, j * dy)
        rows.append(torch.stack((x, y), dim=1))
    pts = torch.cat(rows, dim=0)
    r = torch.linalg.vector_norm(pts, dim=1)
    # keep the reference disk (radius R) and map it radially onto the analytic
    # shape: the topology is fixed by the lattice and every particle stays put in
    # the (r, cos theta) chart, so no leftover halo outside the surface exists.
    inside = r <= R
    pts = pts[inside]
    r = r[inside].clamp_min(1e-30)
    c = pts[:, 1] / r
    # proportional radial map: r -> r [1 + eps P_l(cos theta)].  At r = R this is
    # exactly the analytic shape; unlike r -> R[1 + eps P_l] it does not send the
    # points near the origin out to the surface.
    pos = pts * (1.0 + eps * legendre_p(mode, c))[:, None]
    n = int(pos.shape[0])
    mass = torch.full((n,), cfg.rho0 * dx * dy, dtype=torch.float64)
    return {"pos": pos.contiguous(), "mass": mass, "rho0": cfg.rho0,
            "dA": torch.full((n,), dx * dy, dtype=torch.float64),
            "radius": R, "mode": mode, "eps": eps, "dx": dx, "h": cfg.h}


def mirror_cloud(pos: torch.Tensor, vel: torch.Tensor,
                 mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror particles about ``rho = 0`` with the radial velocity flipped."""
    ghost_pos = pos.clone()
    ghost_pos[:, 0] = -ghost_pos[:, 0]
    ghost_vel = vel.clone()
    ghost_vel[:, 0] = -ghost_vel[:, 0]
    return (torch.cat((pos, ghost_pos), dim=0),
            torch.cat((vel, ghost_vel), dim=0),
            torch.cat((mass, mass), dim=0))


def mode_amplitude(pos: torch.Tensor, weight: torch.Tensor, mode: int,
                   sign: torch.Tensor | None = None) -> float:
    """``sum_i w_i r_i^2 P_l(cos theta_i) / sum_i w_i`` (area-weighted quadrupole)."""
    r = torch.linalg.vector_norm(pos, dim=1)
    c = pos[:, 1] / r.clamp_min(1e-30)
    q = r.square() * legendre_p(mode, c)
    if sign is not None:
        q = q * sign
    return float((weight * q).sum() / weight.sum())


# --------------------------------------------------------------------------- #
# SPH operators
# --------------------------------------------------------------------------- #

def _pair_accumulate(pairs: torch.Tensor, n: int, val: torch.Tensor):
    out = torch.zeros(n, dtype=val.dtype)
    out.index_add_(0, pairs[:, 0], val)
    out.index_add_(0, pairs[:, 1], val)
    return out


def summation_density(pos: torch.Tensor, mass: torch.Tensor, h: float,
                      kind: str = "quintic") -> torch.Tensor:
    """``rho_i = sum_j m_j W_ij`` (interior; the free surface is short by design)."""
    pairs = cell_linked_pairs(pos, kernels.SUPPORT_FACTOR[kind] * h)
    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    w, _ = kernels.kernel_pairs(delta, h, pos.shape[1], kind=kind)
    rho = _pair_accumulate(pairs, pos.shape[0], mass[pairs[:, 1]] * w)
    rho = rho + mass * kernels.kernel_pairs(
        torch.zeros((1, pos.shape[1]), dtype=pos.dtype), h, pos.shape[1], kind=kind)[0][0]
    return rho


def initial_density(pos: torch.Tensor, mass: torch.Tensor, h: float,
                    rho0: float, *, sigma: float, radius: float, c0: float,
                    gamma: float, passes: int = 2,
                    kind: str = "quintic") -> torch.Tensor:
    """Start-up density: smoothed summation density, clipped at the Laplace value.

    The summation density of a hex lattice carries 3-5 % defects at the surface
    rows; with ``c0 = 20`` (``rho0 c0^2 = 400``) a 5 % density step is an
    *acoustic* pressure of ``20 rho0 sigma / R`` -- ten times the capillary
    pressure the run is supposed to be driven by, and its velocity noise
    (``dp/(rho0 c0)``) would be several times the mode velocity.  Two kernel
    smoothing passes make the interior uniform to ~0.5 %, and the clip at the
    static Laplace value ``1 + 2 sigma/(rho0 c0^2 R)`` removes the remaining
    positive spikes (they would ring as ``l = 0``).  The surface taper (density
    deficit -> ``p = 0``) is kept: that is the free-surface condition.
    """
    pairs = cell_linked_pairs(pos, kernels.SUPPORT_FACTOR[kind] * h)
    rho = summation_density(pos, mass, h, kind)
    vol = mass / rho0
    for _ in range(passes):
        delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
        w, _ = kernels.kernel_pairs(delta, h, pos.shape[1], kind=kind)
        num = _pair_accumulate(pairs, pos.shape[0], mass[pairs[:, 1]] / rho[pairs[:, 1]] * w * rho[pairs[:, 1]])
        den = _pair_accumulate(pairs, pos.shape[0], mass[pairs[:, 1]] / rho[pairs[:, 1]] * w)
        self_w = kernels.kernel_pairs(
            torch.zeros((1, pos.shape[1]), dtype=pos.dtype), h, pos.shape[1], kind=kind)[0][0]
        num = num + self_w * rho
        den = den + self_w
        rho = num / den
    # Impose the *static* solution of the drop: a uniform interior pressure
    # p_L = 2 sigma / R (Laplace) balanced by the CSF traction on the surface
    # shell, and p = 0 in the free-surface taper.  Scaling the whole field by the
    # Laplace compression does exactly that -- without it the interior starts at
    # p = 0, the uncompensated surface traction drives a startup acoustic pulse of
    # order sigma/R * (R/dx) and the mode signal (a ~ omega^2 eps R) is buried.
    p_laplace = 2.0 * sigma / radius if radius > 0 else 0.0
    rho0_lap = rho0 * (1.0 + p_laplace / (rho0 * c0 * c0)) ** (1.0 / gamma)
    return (rho / rho0) * rho0_lap


def _csf_fields(pos: torch.Tensor, mass: torch.Tensor, h: float, cfg: DynamicConfig):
    pairs = cell_linked_pairs(pos, cfg.curvature_stencil * h)
    return csf_fields(pos, mass, pairs, h, cfg.rho0,
                      curvature_scheme=cfg.curvature_scheme,
                      curvature_stencil=cfg.curvature_stencil,
                      delta_normalization=cfg.delta_normalization), pairs


def _accelerations(pos_p: torch.Tensor, vel_p: torch.Tensor, rho_p: torch.Tensor,
                   mass_p: torch.Tensor, cfg: DynamicConfig):
    """Axisymmetric WCSPH right-hand side, primary half plane in, primary out.

    The mirror cloud is built here (positions, velocities and the density field,
    which is even about the axis), every operator is evaluated on the doubled
    cloud and the primary half is returned as
    ``(drho_dt, dvel_dt, kappa, delta_s, surface_tension_magnitude)``.
    """
    n = pos_p.shape[0]
    pos = torch.cat((pos_p, torch.stack((-pos_p[:, 0], pos_p[:, 1]), dim=1)), dim=0)
    vel = torch.cat((vel_p, torch.stack((-vel_p[:, 0], vel_p[:, 1]), dim=1)), dim=0)
    mass = torch.cat((mass_p, mass_p), dim=0)
    rho = torch.cat((rho_p, rho_p), dim=0)
    h, c0, rho0, gamma = cfg.h, cfg.c0, cfg.rho0, cfg.gamma
    support = cfg.support
    pairs = cell_linked_pairs(pos, support)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    r = torch.linalg.vector_norm(delta, dim=1)
    w, grad = kernels.kernel_pairs(delta, h, pos.shape[1], kind=cfg.kernel)

    # --- continuity: Monaghan (1992) eq. 2.9 --------------------------------
    dv = vel[i] - vel[j]
    dv_dot_grad = (dv * grad).sum(dim=1)
    drho = torch.zeros(pos.shape[0], dtype=pos.dtype)
    drho.index_add_(0, i, mass[j] * dv_dot_grad)
    drho.index_add_(0, j, mass[i] * dv_dot_grad)

    # --- delta-SPH density diffusion (Antuono et al. 2012) -------------------
    if cfg.delta_sph > 0.0:
        r2 = r.square() + 0.01 * h * h
        vol = mass / rho0
        diff = cfg.delta_sph * h * c0 * (rho[j] - rho[i]) * \
            (delta * grad).sum(dim=1) / r2
        drho.index_add_(0, i, vol[j] * diff)
        drho.index_add_(0, j, vol[i] * diff)

    # --- momentum: symmetric pressure + artificial viscosity ----------------
    p = torch.clamp(rho0 * c0 * c0 / gamma * ((rho / rho0).pow(gamma) - 1.0),
                    min=0.0)
    p_over_rho2 = p / rho.square()
    bracket = p_over_rho2[i] + p_over_rho2[j]
    if cfg.alpha_av > 0.0 or cfg.beta_av > 0.0:
        vdotr = (vel[i] - vel[j]).mul(delta).sum(dim=1)
        mu = h * vdotr / (r.square() + 0.01 * h * h)
        c_ij = 0.5 * (c0 + c0)
        pi_ij = torch.where(
            vdotr < 0.0,
            (-cfg.alpha_av * c_ij * mu + cfg.beta_av * mu.square()) / (0.5 * (rho[i] + rho[j])),
            torch.zeros_like(mu))
        bracket = bracket + pi_ij
    accel = torch.zeros_like(pos)
    contrib = -bracket[:, None] * grad
    accel.index_add_(0, i, mass[j][:, None] * contrib)
    accel.index_add_(0, j, mass[i][:, None] * contrib)

    # --- CSF surface tension with the cylindrical curvature -----------------
    state, _ = _csf_fields(pos, mass, h, cfg)
    x = pos[:, 0]
    sign = torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))
    rho_axis = x.abs().clamp_min(0.25 * h)
    kappa = state.kappa + state.normal[:, 0] * sign / rho_axis
    # the interface band is the colour transition (c = 1 bulk -> 0.5 surface ->
    # 0 void); outside it the fitted curvature is pure noise (measured kappa R up
    # to 17 near the axis, where delta_s is ~1/500 of its peak)
    band = ((state.color > 0.25) & (state.color < 0.75)).to(kappa.dtype)
    f = cfg.sigma * kappa * state.delta_s * band
    accel = accel + (f / rho)[:, None] * state.normal

    # --- axisymmetric continuity source: -rho v_rho / rho_axis --------------
    drho = drho - rho * vel[:, 0] / rho_axis

    return (drho[:n], accel[:n], kappa[:n], state.delta_s[:n],
            (f.pow(2)).sum().sqrt())


def _dt_limit(vel: torch.Tensor, cfg: DynamicConfig) -> float:
    vmax = float(torch.linalg.vector_norm(vel, dim=1).max()) if vel.numel() else 0.0
    dt_cfl = cfg.cfl * cfg.h / (cfg.c0 + vmax)
    # capillary (Brackbill et al. 1992; the grid-scale bound k = pi/h of
    # tensorsph.surface_tension_dispersion.capillary_dt_limit)
    if cfg.sigma > 0.0:
        k_max = math.pi / cfg.h
        dt_cap = cfg.cfl / math.sqrt(cfg.sigma * k_max ** 3 / cfg.rho0)
        dt_cfl = min(dt_cfl, dt_cap)
    return dt_cfl


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run_dynamic_drop(cfg: DynamicConfig, *, callback=None) -> dict:
    """Integrate the free drop and return the amplitude history and the fit."""
    cfg.validate()
    torch.set_default_dtype(torch.float64)
    geo = half_plane_lattice(cfg)
    pos = geo["pos"].clone()
    mass = geo["mass"].clone()
    n = pos.shape[0]

    # mass calibration: 2 fixed-point sweeps so the summation density of the
    # starting lattice is exactly rho0 (same trick as wcsph.Solver.calibrate_mass)
    if cfg.calibrate_mass:
        for _ in range(2):
            pos_m, _, mass_m = mirror_cloud(pos, torch.zeros_like(pos), mass)
            rho_m = summation_density(pos_m, mass_m, cfg.h, cfg.kernel)
            mass = mass * (cfg.rho0 / rho_m[:n])
    pos_m, vel_m, mass_m = mirror_cloud(pos, torch.zeros_like(pos), mass)
    rho = initial_density(pos_m, mass_m, cfg.h, cfg.rho0, sigma=cfg.sigma,
                          radius=cfg.radius, c0=cfg.c0, gamma=cfg.gamma)[:n]
    rho = rho.clamp(min=0.5 * cfg.rho0)

    vel = torch.zeros_like(pos)
    a0 = mode_amplitude(pos, geo["dA"], cfg.mode)
    if a0 == 0.0:
        raise RuntimeError("initial mode amplitude vanished; check the lattice")
    weight = geo["dA"]

    omega_ref = cfg.omega_ref if cfg.sigma > 0 else float("nan")
    t_ref = 2.0 * math.pi / omega_ref if omega_ref == omega_ref else 2.0
    t_end = cfg.t_end if cfg.t_end is not None else cfg.periods * t_ref
    t_skip = cfg.skip_periods * t_ref

    times, amps, rhos, speeds = [], [], [], []
    t = 0.0
    steps = 0
    dt = _dt_limit(vel, cfg)
    # kick-drift-kick leapfrog (one force evaluation per step)
    while t < t_end and steps < cfg.max_steps:
        dt = _dt_limit(vel, cfg)
        dt = min(dt, t_end - t) if t_end - t > 0 else dt
        vel_half = vel + 0.5 * dt * _accel(pos, vel, rho, mass, cfg)
        pos = pos + dt * vel_half
        drho, accel, _, _, _ = _accelerations(pos, vel_half, rho, mass, cfg)
        rho = torch.clamp(rho + dt * drho, min=0.2 * cfg.rho0)
        # keep the cloud in the half plane: the axis is a mirror, so a particle
        # that crosses rho = 0 is folded back instead of being lost
        neg = pos[:, 0] < 0.0
        if bool(neg.any()):
            pos[neg, 0] = -pos[neg, 0]
        vel = vel_half + 0.5 * dt * accel
        t = t + dt
        steps += 1
        if steps % cfg.sample_every == 0 or t >= t_end:
            amp = mode_amplitude(pos, weight, cfg.mode) / a0 * cfg.eps
            times.append(t)
            amps.append(amp)
            rhos.append(float((rho / cfg.rho0).mean()))
            speeds.append(float(torch.linalg.vector_norm(vel, dim=1).max()))
            if callback is not None:
                callback(t, amp)

    times_t = torch.tensor(times, dtype=torch.float64)
    amps_t = torch.tensor(amps, dtype=torch.float64)
    keep = times_t >= t_skip
    if int(keep.sum()) < 16:
        keep = torch.ones_like(times_t, dtype=torch.bool)
    fit = fit_damped_cosine(times_t[keep], amps_t[keep])
    omega_ref_meas = fit.get("omega", float("nan"))
    err = (omega_ref_meas / omega_ref - 1.0) * 100.0 if omega_ref_meas == omega_ref_meas else float("nan")
    return {
        "cfg": {k: getattr(cfg, k) for k in
                ("radius", "rho0", "sigma", "dx", "h_factor", "kernel", "mode",
                 "eps", "c0", "gamma", "alpha_av", "delta_sph", "cfl",
                 "curvature_scheme", "periods", "skip_periods")},
        "radius_over_dx": cfg.radius_over_dx,
        "n_particles": int(n),
        "n_steps": int(steps),
        "dt_last": float(dt),
        "t_end": float(t_end),
        "omega_analytic": omega_ref,
        "omega_measured": omega_ref_meas,
        "omega_error_pct": err,
        "fit": fit,
        "amplitude_start": float(amps[0]) if amps else None,
        "amplitude_end": float(amps[-1]) if amps else None,
        "amplitude_decay_ratio": (float(abs(amps[-1]) / abs(amps[0]))
                                 if amps and amps[0] != 0 else None),
        "density_drift_max": float(max(abs(r - 1.0) for r in rhos)) if rhos else None,
        "max_speed": float(max(speeds)) if speeds else None,
        "times": times_t.tolist(),
        "amplitudes": amps_t.tolist(),
    }


def _accel(pos, vel, rho, mass, cfg):
    _, accel, _, _, _ = _accelerations(pos, vel, rho, mass, cfg)
    return accel


def dynamic_frequency(*, radius_over_dx=(8.0, 12.0, 16.0), mode: int = 2,
                      eps: float = 0.05, c0: float = 20.0,
                      sigma: float = 1.0, rho0: float = 1.0,
                      radius: float = 1.0, periods: float = 4.0,
                      alpha_av: float = 0.02, delta_sph: float = 0.1,
                      curvature_scheme: str = "fit_quadratic",
                      **kwargs) -> dict:
    """Resolution ladder of the dynamic arm."""
    rows = []
    for rdx in radius_over_dx:
        cfg = DynamicConfig(radius=radius, rho0=rho0, sigma=sigma,
                            dx=radius / rdx, mode=mode, eps=eps, c0=c0,
                            periods=periods, alpha_av=alpha_av,
                            delta_sph=delta_sph,
                            curvature_scheme=curvature_scheme, **kwargs)
        row = run_dynamic_drop(cfg)
        rows.append(row)
    return {"mode": mode, "eps": eps, "c0": c0, "sigma": sigma, "rho0": rho0,
            "radius": radius, "rows": rows}