"""2-D drop capillary oscillation: exact reference, initial state, readings.

Why this module exists
----------------------
The surface-tension capability of TensorSPH has been verified *statically* only:
the colour gradient, the curvature and the stress divergence have all been
integrated across a fixed interface and compared with the Young-Laplace jump
(``docs/verification_surface_tension_*.md``, ``benchmarks/surface_tension/``).
None of those readings can tell whether the force that is actually *integrated
in time* converts surface energy into kinetic energy at the right rate: a force
field may have the correct normal integral and still give the wrong dynamics.

The exact, parameter-free dynamic reference for a closed interface is the
inviscid oscillation of a drop about its circular equilibrium (Rayleigh 1879;
Lamb, *Hydrodynamics*, 6th ed., arts. 313-315)::

    omega_m^2 = sigma * m * (m^2 - 1) / (rho * R^3)         (2-D drop, mode m)

The ``m = 1`` branch is the neutral translation mode (``omega = 0``), which
anchors the formula independently of any measurement; ``m = 2`` is used here
because it preserves the drop area to first order, so the reference involves
surface energy only and no bulk compression.

What is here
------------
* :func:`rayleigh_2d` -- the exact frequency above.
* :func:`make_droplet` -- a particle cloud filling ``r <= R (1 + eps cos m theta)``.
* :func:`mode_amplitude` -- the order parameter
  ``q_m = sum m_i r_i^2 cos(m theta_i) / sum m_i r_i^2``, which for a shape
  ``r(theta) = R (1 + eps cos m theta)`` equals ``2 eps / (1 + 3 eps^2)`` to
  leading order, i.e. it is *linear* in the perturbation and its zero crossings
  are spaced ``T_m / 2`` independently of the damping envelope.
* :func:`interior_pressure` -- the EOS-resolved mean pressure of ``r < R/2``,
  the dynamic counterpart of the static Laplace reading.
* :func:`run_point` -- one time-integrated ladder point with the production
  CSF force (:func:`tensorsph.surface_tension.compute_surface_tension_force`),
  returning every reading against its exact reference.

The science driver is ``benchmarks/surface_tension/droplet_oscillation.py``;
``tests/test_droplet_oscillation.py`` locks the invariants.
"""
from __future__ import annotations

import math
import time

import torch

from .droplet_impact import Config, rhs, suggested_dt
from .neighbor_search import PairListCache
from .particle_shifting import shift_positions
from .transport_velocity_correction import transport_velocity_correction

__all__ = [
    "rayleigh_2d",
    "make_droplet",
    "mode_amplitude",
    "interior_pressure",
    "period_from_crossings",
    "run_point",
]

SIGMA_WATER_AIR = 0.072
RHO_WATER = 1000.0
RADIUS_DEFAULT = 0.05
MODE_DEFAULT = 2


def rayleigh_2d(sigma: float, rho: float, radius: float, mode: int) -> float:
    """Exact 2-D drop (liquid cylinder) oscillation frequency, in rad/s."""
    if mode < 2:
        raise ValueError("mode 0 is the breathing mode and mode 1 the neutral "
                         "translation mode of the Rayleigh relation")
    return math.sqrt(sigma * mode * (mode * mode - 1.0) / (rho * radius ** 3))


def make_droplet(dx: float, eps: float, device, radius: float = RADIUS_DEFAULT,
                 mode: int = MODE_DEFAULT, rho0: float = RHO_WATER):
    """Particle cloud filling ``r <= R (1 + eps cos(mode*theta))``, uniform dx² mass."""
    n = int(math.ceil(2.4 * radius / dx))
    xs = torch.arange(-n, n + 1, dtype=torch.float64, device=device) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    r = torch.linalg.norm(pos, dim=1)
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    inside = r <= radius * (1.0 + eps * torch.cos(mode * theta))
    pos = pos[inside]
    mass = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=torch.float64,
                      device=device)
    return pos, mass


def mode_amplitude(pos: torch.Tensor, mass: torch.Tensor, mode: int) -> float:
    """``q_mode = sum m r^2 cos(mode theta) / sum m r^2`` (linear in the perturbation)."""
    r2 = (pos ** 2).sum(dim=1)
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    num = float((mass * r2 * torch.cos(mode * theta)).sum())
    den = float((mass * r2).sum())
    return num / den


def interior_pressure(pos: torch.Tensor, pressure: torch.Tensor,
                      radius: float = RADIUS_DEFAULT) -> float:
    """Mean pressure of the particles inside ``r < R/2`` (free surface outside)."""
    r = torch.linalg.norm(pos, dim=1)
    sel = r < 0.5 * radius
    return float(pressure[sel].mean())


def period_from_crossings(times, values) -> tuple[float, int]:
    """Period from the spacing of the zero crossings of ``values`` (linear in eps).

    Returns ``(period, n_crossings)``; ``period`` is ``nan`` when fewer than two
    crossings are found.  The envelope does not enter, so this reading survives
    numerical damping.
    """
    crossings = []
    for k in range(1, len(values)):
        a, b = values[k - 1], values[k]
        if a * b < 0.0:
            crossings.append(times[k - 1] + (times[k] - times[k - 1])
                             * abs(a) / (abs(a) + abs(b)))
    half = [crossings[k + 1] - crossings[k] for k in range(len(crossings) - 1)]
    half = [hp for hp in half if hp > 0.0]
    if not half:
        return float("nan"), len(crossings)
    return 2.0 * sum(half) / len(half), len(crossings)


def run_point(radius_over_dx: int, sigma: float, periods: float,
              device="cpu", eps: float = 0.05,
              radius: float = RADIUS_DEFAULT, mode: int = MODE_DEFAULT,
              rho0: float = RHO_WATER, sound_speed: float = 0.5,
              viscosity: float = 1.0e-6, density_diffusion: float = 0.0,
              sample_every: int = 1,
              t_end_s: float | None = None,
              tvc_alpha: float = 0.0,
              tvc_every: int = 1,
              tvc_max_shift: float | None = None,
              tvc_limiter=None,
              shift_strength: float = 0.0,
              shift_every: int = 1,
              shift_limit: float = 0.25,
              shift_free_surface: float = 0.95,
              shift_v_ref: float | None = None) -> dict:
    """Integrate one drop-oscillation point and return every reading + reference.

    ``tvc_*`` is the ported SPHinXsys transport-velocity correction and
    ``shift_*`` the Lind et al. (2012) particle shifting
    (:func:`tensorsph.particle_shifting.shift_positions`); both default to off
    (``alpha = 0`` / ``strength = 0``), which reproduces the historical
    trajectory bit for bit.  They are different operators with different
    free-surface behaviour: TVC moves particles along ``-grad(1)`` (outward at a
    free surface), shifting projects the displacement onto the *tangential*
    plane there, so the interface is redistributed rather than pushed.
    ``shift_v_ref`` defaults to Lind's instantaneous maximum speed with the
    ``0.05 * c0`` floor that ``dam_break`` uses, so a drop relaxing at rest
    still redistributes.
    """
    device = torch.device(device)
    dx = radius / radius_over_dx
    cfg = Config(
        dx=dx,
        h_factor=1.3,
        rho0=rho0,
        sound_speed=sound_speed,
        gamma=7.0,
        gravity=0.0,
        viscosity=viscosity,
        pool_width=2.4 * radius,
        pool_depth=2.4 * radius,
        surface_tension_coefficient=sigma,
        device=str(device),
        boundary="clamp",
        adaptive_dt=True,
        max_courant=0.25,
        density_calibration=True,
        density_diffusion=density_diffusion,
    )
    cfg.dt = suggested_dt(cfg)
    pos, mass = make_droplet(dx, eps, device, radius=radius, mode=mode, rho0=rho0)
    vel = torch.zeros_like(pos)
    h = cfg.h_factor * dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=pos.shape[0])

    omega = rayleigh_2d(sigma, rho0, radius, mode) if sigma > 0.0 else float("nan")
    t_exact = 2.0 * math.pi / omega if sigma > 0.0 else float("nan")
    t_end = t_end_s if t_end_s is not None else periods * t_exact
    dp_exact = sigma / radius

    t, steps = 0.0, 0
    times, qs, ps, rmaxes = [], [], [], []
    max_r_over_R, escaped = 0.0, 0
    t_first_escape = float("nan")
    # transport-velocity-correction bookkeeping (all zero when tvc_alpha = 0.0)
    tvc_steps, tvc_shift_peak, tvc_h_grad1_sum = 0, 0.0, 0.0
    # particle-shifting bookkeeping (all zero when shift_strength = 0.0)
    shift_steps, shift_peak_over_h = 0, 0.0
    shift_clipped_sum, shift_fs_sum = 0.0, 0.0
    shift_v_ref_last = float("nan")
    t0 = time.time()
    while t < t_end:
        pairs, _ = cache.get(pos)
        density, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                     dt=cfg.dt)
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        if tvc_alpha > 0.0 and steps > 0 and steps % tvc_every == 0:
            # The ported SPHinXsys ``TransportVelocityCorrection``, applied the
            # way ``dam_break`` applies it: volumes from this step's summation
            # density, positions moved, mass and velocity untouched.  There is
            # no wall in the drop case, so ``wall_count = 0``.
            pairs_now, _ = cache.get(pos)
            tvc = transport_velocity_correction(
                pos, mass / density, h, alpha=tvc_alpha, pairs=pairs_now,
                max_shift=(None if tvc_max_shift is None
                           else tvc_max_shift * h),
                limiter=tvc_limiter)
            pos = pos + tvc.shift
            tvc_shift_peak = max(tvc_shift_peak, tvc.max_shift)
            tvc_h_grad1_sum += h * float(
                torch.linalg.vector_norm(tvc.gradient_integral, dim=1).mean())
            tvc_steps += 1
        if shift_strength > 0.0 and steps > 0 and steps % shift_every == 0:
            # Lind et al. (2012) shifting, wired the way ``dam_break`` wires it:
            # volumes from this step's summation density, positions moved, mass
            # and velocity untouched.  At the free surface the displacement is
            # projected onto the tangent plane, so the interface is redistributed
            # instead of being pushed along the normal (the failure mode of TVC
            # documented in docs/xval_method_selection_csf_dynamic_m22526.md).
            pairs_now, _ = cache.get(pos)
            v_max = float(torch.linalg.vector_norm(vel, dim=1).max()) \
                if vel.numel() else 0.0
            v_ref = (max(v_max, 0.05 * sound_speed) if shift_v_ref is None
                     else float(shift_v_ref))
            pos, st = shift_positions(
                pos, mass / density, h, dt=cfg.dt, v_ref=v_ref,
                strength=shift_strength, limit=shift_limit,
                free_surface=shift_free_surface, pairs=pairs_now,
                return_state=True)
            shift_steps += 1
            shift_v_ref_last = v_ref
            shift_peak_over_h = max(
                shift_peak_over_h,
                float(torch.linalg.vector_norm(st.shift, dim=1).max()) / h)
            shift_clipped_sum += float(st.clipped)
            shift_fs_sum += float(st.free_surface.double().mean())
        t += cfg.dt
        steps += 1
        cfg.dt = suggested_dt(cfg)
        r_now = torch.linalg.norm(pos, dim=1)
        max_r_over_R = max(max_r_over_R, float(r_now.max()) / radius)
        if escaped == 0 and bool((r_now > 1.6 * radius).any()):
            t_first_escape = t
        escaped = max(escaped, int((r_now > 1.6 * radius).sum()))
        if steps % sample_every == 0:
            times.append(t)
            qs.append(mode_amplitude(pos, mass, mode))
            ps.append(interior_pressure(pos, pressure, radius))
            rmaxes.append(float(r_now.max()) / radius)
    wall_s = time.time() - t0

    period, n_cross = period_from_crossings(times, qs)
    err_period = (100.0 * (period - t_exact) / t_exact
                  if (period == period and t_exact == t_exact) else float("nan"))
    tail = ps[len(ps) // 2:]
    p_in = sum(tail) / len(tail) if tail else float("nan")
    err_dp = (100.0 * (p_in - dp_exact) / dp_exact
              if dp_exact > 0.0 else float("nan"))

    return {
        "radius_over_dx": radius_over_dx,
        "dx_m": dx,
        "particles": int(pos.shape[0]),
        "sigma": sigma,
        "steps": steps,
        "dt_last_s": cfg.dt,
        "t_end_s": t_end,
        "wall_s": round(wall_s, 2),
        "omega_exact_rad_s": omega,
        "period_exact_s": t_exact,
        "period_measured_s": period,
        "period_err_pct": err_period,
        "zero_crossings": n_cross,
        "q_initial": qs[0],
        "q_final": qs[-1],
        "q_half_range": 0.5 * (max(qs) - min(qs)),
        "p_interior_mean_pa": p_in,
        "dp_exact_pa": dp_exact,
        "dp_err_pct": err_dp,
        "tvc_alpha": float(tvc_alpha),
        "tvc_every": int(tvc_every),
        "tvc_steps": int(tvc_steps),
        "tvc_shift_peak_over_h": (tvc_shift_peak / h if h > 0.0 else 0.0),
        "tvc_mean_h_grad1": (tvc_h_grad1_sum / tvc_steps if tvc_steps else 0.0),
        "shift_strength": float(shift_strength),
        "shift_every": int(shift_every),
        "shift_limit": float(shift_limit),
        "shift_free_surface": float(shift_free_surface),
        "shift_steps": int(shift_steps),
        "shift_peak_over_h": float(shift_peak_over_h),
        "shift_clipped_fraction_mean": (shift_clipped_sum / shift_steps
                                        if shift_steps else 0.0),
        "shift_free_surface_fraction_mean": (shift_fs_sum / shift_steps
                                             if shift_steps else 0.0),
        "shift_v_ref_last": float(shift_v_ref_last),
        "r_mean_over_R": float(torch.linalg.norm(pos, dim=1).mean()) / radius,
        "containment_max_r_over_R": max_r_over_R,
        "t_first_escape_s": t_first_escape,
        "escaped_particles": escaped,
        "series": {"t": [round(x, 6) for x in times],
                   "q": [round(x, 8) for x in qs],
                   "p_interior": [round(x, 6) for x in ps],
                   "max_r_over_R": [round(x, 6) for x in rmaxes]},
    }