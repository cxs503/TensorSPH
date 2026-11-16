"""Far-field (control-volume) force from fluid: an independent second estimator.

Why this module exists
----------------------
Every load channel the repository actually uses integrates something **on the
body surface**:

``pair``    the reaction of the fluid/wall kernel pairs,
``panel``   an extrapolated wall-face pressure integrated over the face.

Both are *near-field* estimators: they must reconstruct a surface quantity from
particles that sit one or two spacings away, and both were measured to carry a
geometry-dependent bias -- the OSS-style kernel average is ``+24..28 %`` on a
flat wall and ``+31 %`` on a curved body at the coupled ``fsi2`` resolution
(``h/R = 0.52``), see ``docs/xval_fsi2/PRESSURE_MOMENT_NORMALISATION.md`` and
``docs/xval_fsi2/CYLINDER_LOAD_AT_BENCHMARK_SCALE.md``.

A near-field bias cannot be detected from inside the near-field channel: any
diagnostic built out of the same wall station sees the same bias.  The
control-volume (far-field) identity of the fluid inside an arbitrary closed
surface ``S`` *never touches the body*:

    F_on_body = -d/dt int_CV rho u dV
                - oint_S [ rho u (u.n) + (p - p_inf) n ] dS
                + oint_S (tau.n) dS
                + int_CV rho g dV                                    (1)

so its error mechanism (momentum flux through ``S``) is independent of the
error mechanism of the surface arms (kernel truncation at the body).  This is
the classical near-field / far-field pair used for added resistance in ship
hydrodynamics, and it is what this module provides for the ``fsi2``
force-from-fluid work.

Scope of this port (read before using it)
-----------------------------------------
* Two-dimensional, per unit thickness.
* The steady form of (1) is implemented and verified (``control_volume_force``,
  ``wake_plane_drag``).  The unsteady accumulation term
  ``-d/dt int rho u dV`` is **not** implemented: in 2-D it is exactly the term
  whose uniform part grows like ``R^2`` and cancels against the pressure flux,
  so shipping it without that cancellation would be worse than not shipping it.
  ``cv_momentum`` returns the raw volume integral so the cancellation can be
  studied later; no force is claimed from it.
* Viscous flux is a hook: the caller supplies the surface traction per bin
  (``traction=``).  Only the inviscid part is exercised by the audit.
* ``S`` is a circle (``control_volume_force``) or a straight vertical plane
  (``wake_plane_drag``).
* The surface value on each arc/plane bin is **not** a plain bin average: a
  plain average samples the quantities at radii ``R +- band``, which on a closed
  surface leaves a first-order residue ``sum_b ds_b delta_b (grad p . n_b) n_b``
  (measured: 6.6 % of the load scale at ``dx/R = 0.05``, halving with ``dx``).
  ``reconstruct="fit"`` (default) fits a weighted plane through the particles
  within ``1.6 dx`` of the bin centre and evaluates it *on* the surface, which
  is exact for a linear field and drops that residue to round-off.
  ``reconstruct="mean"`` reproduces the raw average as a control arm.

Sign convention
---------------
Normals point **out of the control volume**.  ``control_volume_force`` returns
the force **on the body from the fluid**, so a drag-producing body has
``F.x > 0`` when the free stream runs along ``+x``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

__all__ = [
    "circular_surface_bins",
    "reconstruct_surface_values",
    "plane_surface_bins",
    "control_volume_force",
    "wake_plane_drag",
    "cv_momentum",
    "potential_flow_cylinder",
]

_TWO_PI = 2.0 * math.pi


def _bin_average(bin_idx: torch.Tensor, weights: torch.Tensor,
                 values: torch.Tensor, n_bins: int):
    """Volume-weighted per-bin average of ``values`` (``(N,)`` or ``(N, k)``).

    Returns ``(mean, filled, wsum)``.  Empty bins are reported through
    ``filled=False`` and carry a zero mean; callers must drop them from the
    surface integral instead of inventing a value.
    """
    dtype = values.dtype
    wsum = torch.zeros(n_bins, dtype=dtype, device=values.device)
    wsum.index_add_(0, bin_idx, weights)
    filled = wsum > 0.0
    safe = torch.where(filled, wsum, torch.ones_like(wsum))
    if values.dim() == 1:
        num = torch.zeros(n_bins, dtype=dtype, device=values.device)
        num.index_add_(0, bin_idx, weights * values)
        mean = num / safe
    else:
        cols = []
        for c in range(values.shape[1]):
            num = torch.zeros(n_bins, dtype=dtype, device=values.device)
            num.index_add_(0, bin_idx, weights * values[:, c])
            cols.append(num / safe)
        mean = torch.stack(cols, dim=-1)
    return mean, filled, wsum


def _fit_window(d: torch.Tensor, h_win: float) -> torch.Tensor:
    """Compact C2 weight for the surface-plane fit (1 at d=0, 0 at d=h)."""
    q = (d / h_win) ** 2
    return torch.where(q < 1.0, (1.0 - q) ** 2, torch.zeros_like(q))


def reconstruct_surface_values(bin_points: torch.Tensor, pos: torch.Tensor,
                               values: torch.Tensor, h_win: float,
                               volumes: Optional[torch.Tensor] = None):
    """Linear-exact surface reconstruction of ``values`` on surface bins.

    For every bin centre ``x_b`` (a point *on* the surface) all particles within
    ``h_win`` are gathered and a weighted least-squares plane
    ``q(x) = a + b . (x - x_b)`` is solved; the surface value is ``a``.  Returns
    ``(fit, mean, filled)`` where ``mean`` is the plain kernel-free bin average
    (the control arm) and ``filled`` marks bins that got a fit.
    """
    d = torch.cdist(bin_points.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
    w = _fit_window(d, h_win)
    if volumes is not None:
        w = w * volumes.unsqueeze(0)
    dx_rel = (pos.unsqueeze(0) - bin_points.unsqueeze(1))          # (B, N, 2)
    nb, n = d.shape
    k = values.shape[1] if values.dim() > 1 else 1
    vals = values if values.dim() > 1 else values.unsqueeze(-1)
    # moments: [1, dx, dy]
    basis = torch.cat((torch.ones(nb, n, 1, dtype=d.dtype, device=d.device),
                       dx_rel), dim=-1)
    lhs = torch.einsum("bn,bni,bnj->bij", w, basis, basis)
    rhs = torch.einsum("bn,bni,bnk->bik", w, basis, vals.unsqueeze(0).expand(nb, n, k))
    count = (w > 0).sum(dim=1)
    eye = torch.eye(3, dtype=d.dtype, device=d.device).unsqueeze(0) * 1e-12
    lhs = lhs + eye
    filled = count >= 3
    if bool(filled.any()):
        sol = torch.linalg.solve(lhs, rhs)                          # (B, 3, k)
        fit = sol[:, 0, :]
    else:  # pragma: no cover - defensive
        fit = torch.zeros(nb, k, dtype=d.dtype, device=d.device)
    finite = torch.isfinite(fit).all(dim=-1)
    filled = filled & finite
    wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-30)
    mean = torch.einsum("bn,bnk->bk", w, vals.unsqueeze(0).expand(nb, n, k)) / wsum
    if values.dim() == 1:
        fit, mean = fit[:, 0], mean[:, 0]
    return fit, mean, filled


def circular_surface_bins(pos: torch.Tensor, centre: torch.Tensor, radius: float,
                          dx: float, n_bins: Optional[int] = None,
                          band: Optional[float] = None,
                          volumes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Bin the particles that straddle the circle ``|x - centre| = radius``.

    The particles within ``+-band`` of the circle are projected onto it and
    grouped into ``n_bins`` equal arc sectors; a sector value is the
    volume-weighted average of the particles in it.  ``band`` defaults to
    ``0.75 dx`` (roughly one lattice row) and ``n_bins`` to ``2 pi R / dx``, so
    every sector sees at least one particle on a regular lattice.
    """
    if band is None:
        band = 1.0 * dx
    if n_bins is None:
        n_bins = max(8, int(round(_TWO_PI * radius / dx)))
    rel = pos - centre
    r = torch.linalg.norm(rel, dim=1)
    sel = (r >= radius - band) & (r <= radius + band)
    if not bool(sel.any()):
        raise ValueError("no particles within +-band of the control surface")
    theta = torch.atan2(rel[sel, 1], rel[sel, 0]) % _TWO_PI
    idx = torch.clamp((theta / _TWO_PI * n_bins).floor().long(), max=n_bins - 1)
    if volumes is None:
        weights = torch.full((int(sel.sum()),), dx * dx, dtype=pos.dtype,
                             device=pos.device)
    else:
        weights = volumes[sel]
    theta_c = (torch.arange(n_bins, dtype=pos.dtype, device=pos.device) + 0.5) \
        * _TWO_PI / n_bins
    normals = torch.stack((torch.cos(theta_c), torch.sin(theta_c)), dim=-1)
    return {
        "sel": sel,
        "bin_idx": idx,
        "weights": weights,
        "normals": normals,
        "bin_points": (centre.unsqueeze(0)
                       + radius * normals),
        "ds": radius * _TWO_PI / n_bins,
        "n_bins": n_bins,
        "radius": radius,
    }


def plane_surface_bins(pos: torch.Tensor, x_plane: float, y_range: float,
                       dx: float, n_bins: Optional[int] = None,
                       band: Optional[float] = None,
                       volumes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Bin the particles straddling the vertical plane ``x = x_plane``.

    ``y_range`` is the half-height that must be covered; the bin height defaults
    to ``dx``, which keeps ``n_bins`` comparable to the circular case.
    """
    if band is None:
        band = 1.0 * dx
    if n_bins is None:
        n_bins = max(8, int(round(2.0 * y_range / dx)))
    sel = (pos[:, 0] >= x_plane - band) & (pos[:, 0] <= x_plane + band) \
        & (pos[:, 1].abs() <= y_range)
    if not bool(sel.any()):
        raise ValueError("no particles within +-band of the wake plane")
    y = pos[sel, 1]
    # bins: [0, 2*n_bins) from -y_range to +y_range, centred on dy/2
    dy = 2.0 * y_range / n_bins
    idx = torch.clamp(((y + y_range) / dy).floor().long(), max=n_bins - 1)
    if volumes is None:
        weights = torch.full((int(sel.sum()),), dx * dx, dtype=pos.dtype,
                             device=pos.device)
    else:
        weights = volumes[sel]
    y_c = -y_range + (torch.arange(n_bins, dtype=pos.dtype, device=pos.device) + 0.5) * dy
    return {
        "sel": sel,
        "bin_idx": idx,
        "weights": weights,
        "y_c": y_c,
        "bin_points": torch.stack((torch.full_like(y_c, x_plane), y_c), dim=-1),
        "ds": dy,
        "n_bins": n_bins,
    }


def control_volume_force(pos: torch.Tensor, vel: torch.Tensor,
                         pressure: torch.Tensor, rho: float,
                         centre: torch.Tensor, radius: float, dx: float,
                         p_ref: float = 0.0,
                         n_bins: Optional[int] = None,
                         band: Optional[float] = None,
                         volumes: Optional[torch.Tensor] = None,
                         traction: Optional[torch.Tensor] = None,
                         reconstruct: str = "fit",
                         h_window: Optional[float] = None) -> Dict[str, object]:
    """Steady control-volume force on the body inside a circular surface.

    Implements the steady part of (1) in the module docstring:
    ``F = -oint [rho u (u.n) + (p - p_ref) n] dS + oint tau.n dS``.
    Pressure and momentum-flux contributions are returned separately so a
    caller can see which term carries the answer.  ``reconstruct="fit"``
    (default) evaluates the surface state with :func:`reconstruct_surface_values`;
    ``"mean"`` uses the raw bin average.
    """
    if reconstruct not in ("fit", "mean"):
        raise ValueError("reconstruct must be 'fit' or 'mean'")
    surf = circular_surface_bins(pos, centre, radius, dx, n_bins=n_bins,
                                 band=band, volumes=volumes)
    sel, idx = surf["sel"], surf["bin_idx"]
    n_bins = int(surf["n_bins"])
    sub_pos = pos[sel]
    sub_vol = volumes[sel] if volumes is not None else None
    h_win = 1.6 * dx if h_window is None else h_window
    fit_u, mean_u, ok_u = reconstruct_surface_values(
        surf["bin_points"], sub_pos, vel[sel], h_win, sub_vol)
    fit_p, mean_p, ok_p = reconstruct_surface_values(
        surf["bin_points"], sub_pos, pressure[sel], h_win, sub_vol)
    filled = ok_u & ok_p
    if reconstruct == "fit":
        u_b, p_b = fit_u, fit_p
    else:
        u_b, p_b = mean_u, mean_p
    n = surf["normals"]
    ds = surf["ds"]
    un = (u_b * n).sum(dim=-1)
    f_mom_bins = rho * u_b * un[:, None]
    f_pres_bins = (p_b - p_ref)[:, None] * n
    mask = filled.to(f_mom_bins.dtype)[:, None]
    f_mom = -(f_mom_bins * mask * ds).sum(dim=0)
    f_pres = -(f_pres_bins * mask * ds).sum(dim=0)
    f_visc = None
    if traction is not None:
        f_visc = (traction * mask * ds).sum(dim=0)
    total = f_mom + f_pres
    if f_visc is not None:
        total = total + f_visc
    # control arm: same integral with the plain bin average
    un_m = (mean_u * n).sum(dim=-1)
    f_mean = -((rho * mean_u * un_m[:, None]
                + (mean_p - p_ref)[:, None] * n) * mask * ds).sum(dim=0)
    return {
        "force": total,
        "force_momentum_flux": f_mom,
        "force_pressure": f_pres,
        "force_viscous": f_visc,
        "force_mean_reconstruction": f_mean,
        "coverage": float(filled.to(torch.float64).mean()),
        "n_bins": n_bins,
        "n_surface_particles": int(sel.sum()),
        "ds": float(ds),
        "surface_speed": u_b.detach(),
        "surface_pressure": p_b.detach(),
    }


def cv_momentum(pos: torch.Tensor, vel: torch.Tensor, rho: float,
                centre: torch.Tensor, radius: float, dx: float,
                volumes: Optional[torch.Tensor] = None,
                frame_vel: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``int_CV rho (u - frame_vel) dV`` per unit thickness.

    The control volume is the fluid inside the circle; ``volumes`` defaults to a
    uniform ``dx**2`` per particle.  Supplying ``frame_vel`` removes the uniform
    part of the flow, which is what makes the time derivative of this integral
    usable as the accumulation term of (1): the raw integral carries the ``R**2``
    uniform contribution that must be cancelled against the pressure flux (see
    :func:`control_volume_force_unsteady`, which performs that cancellation with
    analytic areas).  With ``frame_vel=None`` this is the raw momentum, kept for
    the unsplit control arm and for bookkeeping.
    """
    rel = pos - centre
    inside = torch.linalg.norm(rel, dim=1) <= radius
    vol = (volumes if volumes is not None
           else torch.full((pos.shape[0],), dx * dx, dtype=pos.dtype,
                           device=pos.device))[inside]
    u = vel[inside]
    if frame_vel is not None:
        u = u - torch.as_tensor(frame_vel, dtype=pos.dtype).reshape(-1)
    return rho * (vol.unsqueeze(-1) * u).sum(dim=0)


def wake_plane_drag(pos: torch.Tensor, vel: torch.Tensor, pressure: torch.Tensor,
                    rho: float, u_inf: float, p_ref: float, x_plane: float,
                    y_range: float, dx: float, n_bins: Optional[int] = None,
                    band: Optional[float] = None,
                    volumes: Optional[torch.Tensor] = None,
                    reconstruct: str = "fit",
                    h_window: Optional[float] = None) -> Dict[str, object]:
    """Momentum-deficit (far-field wake) drag on a plane downstream of a body.

    ``D = int [ rho u_x (u_inf - u_x) + (p_ref - p) ] dy`` per unit thickness,
    positive downstream (+x).  This is the standard far-field form; it assumes
    a steady wake whose lateral boundaries (``+-y_range``) carry the free
    stream, which the caller must ensure.
    """
    if reconstruct not in ("fit", "mean"):
        raise ValueError("reconstruct must be 'fit' or 'mean'")
    surf = plane_surface_bins(pos, x_plane, y_range, dx, n_bins=n_bins,
                              band=band, volumes=volumes)
    sel = surf["sel"]
    sub_pos = pos[sel]
    sub_vol = volumes[sel] if volumes is not None else None
    h_win = 1.6 * dx if h_window is None else h_window
    fit_u, mean_u, ok_u = reconstruct_surface_values(
        surf["bin_points"], sub_pos, vel[sel], h_win, sub_vol)
    fit_p, mean_p, ok_p = reconstruct_surface_values(
        surf["bin_points"], sub_pos, pressure[sel], h_win, sub_vol)
    filled = ok_u & ok_p
    u_b, p_b = (fit_u, fit_p) if reconstruct == "fit" else (mean_u, mean_p)
    mask = filled.to(u_b.dtype)
    ds = surf["ds"]
    integrand = rho * u_b[:, 0] * (u_inf - u_b[:, 0]) + (p_ref - p_b)
    drag = (integrand * mask * ds).sum()
    integrand_m = rho * mean_u[:, 0] * (u_inf - mean_u[:, 0]) + (p_ref - mean_p)
    drag_mean = (integrand_m * mask * ds).sum()
    return {
        "drag": drag,
        "drag_mean_reconstruction": drag_mean,
        "coverage": float(filled.to(torch.float64).mean()),
        "n_bins": int(surf["n_bins"]),
        "velocity_profile": u_b.detach(),
        "y": surf["y_c"],
    }


def potential_flow_cylinder(pos: torch.Tensor, a: float, u_inf: float,
                            circulation: float, rho: float,
                            centre: Optional[torch.Tensor] = None,
                            p_inf: float = 0.0):
    """Steady inviscid flow past a circular cylinder, cylinder at rest.

    ``u_r = U (1 - a^2/r^2) cos t``, ``u_t = -U (1 + a^2/r^2) sin t + Gamma/(2 pi r)``
    with ``Gamma`` taken **counter-clockwise**; ``p`` from steady Bernoulli,
    ``p - p_inf = rho U^2 / 2 - rho |u|^2 / 2``.  Evaluated outside ``r > a``.

    The exact far-field consequence (what the audit checks) is d'Alembert drag
    ``F_x = 0`` and Kutta-Joukowski lift ``F_y = -rho U Gamma`` (the minus sign
    is fixed by the counter-clockwise convention; the audit prints the sign it
    measures instead of assuming it).
    """
    if centre is None:
        centre = torch.zeros(2, dtype=pos.dtype, device=pos.device)
    rel = pos - centre
    r = torch.linalg.norm(rel, dim=1).clamp_min(1e-12)
    cos_t = rel[:, 0] / r
    sin_t = rel[:, 1] / r
    u_r = u_inf * (1.0 - a * a / r ** 2) * cos_t
    u_t = -u_inf * (1.0 + a * a / r ** 2) * sin_t + circulation / (_TWO_PI * r)
    vel = torch.stack((u_r * cos_t - u_t * sin_t, u_r * sin_t + u_t * cos_t), dim=-1)
    speed2 = (vel * vel).sum(dim=-1)
    pressure = p_inf + 0.5 * rho * u_inf ** 2 - 0.5 * rho * speed2
    return vel, pressure


def potential_flow_tip_speed(a: float, u_inf: float, circulation: float) -> float:
    """Maximum surface speed of :func:`potential_flow_cylinder` (for CFL notes)."""
    return abs(2.0 * u_inf) + abs(circulation) / (_TWO_PI * a)

# --------------------------------------------------------------------------- #
# Unsteady (accumulation) term of identity (1)
# --------------------------------------------------------------------------- #
# Why the naive form is unusable
# ------------------------------
# Written literally on the *total* field, ``-d/dt int rho u dV`` and the
# pressure flux carry a spatially uniform part that scales like ``R**2`` (the
# momentum of the uniform stream and the uniform pressure gradient
# ``p_unif = -rho a_f . (x - c)`` implied by an accelerating free stream).  For
# a body whose own inertia load scales like ``rho A_body a_f`` the two big terms
# are ``R**2 / A_body`` times the answer and only their difference survives, so
# the estimator inherits the *relative* error of the big terms amplified by that
# factor.  The cure is the standard perturbation split (the CV form of the
# Froude-Krylov / added-mass decomposition):
#
#   u = u_f(t) + u',      p = p_f(t, x) + p',   p_f = -rho a_f . (x - c)
#
#   F_on_body = +rho A_body a_f                       displaced mass (Froude-Krylov)
#               - d/dt int_CV rho u' dV               fluid perturbation momentum
#               - oint_S [ rho u (u.n) + (p' - p_ref) n ] dS
#               + oint_S tau.n dS                     caller hook
#
# Derivation of the middle line: the uniform parts cancel as
# ``-rho A_fluid a_f + rho pi R**2 a_f = rho A_body a_f`` with
# ``A_fluid = pi R**2 - A_body`` the fluid area inside the control surface
# (``oint (x-c) n dS = pi R**2 I``).  Both areas are analytic, so no lattice
# jaggedness enters the cancellation.  For a circular cylinder in an oscillating
# stream this line reproduces the textbook split ``C_M = 1 + C_a = 2``:
# ``rho A_body a_f`` is the ``1`` and the perturbation terms supply the ``C_a``.

__all__.extend([
    "d_dt_periodic",
    "cv_fluid_area",
    "control_volume_force_unsteady",
    "unsteady_potential_flow_cylinder",
])


def d_dt_periodic(values: torch.Tensor, dt: float,
                  periodic: bool = True) -> torch.Tensor:
    """Second-order accurate ``d/dt`` of a uniformly sampled series.

    ``values`` is ``(N_t, ...)``; the difference runs along dim 0.  With
    ``periodic=True`` the series is wrapped (appropriate for a periodic sampling
    of a harmonic field) and the truncation error is ``-(dt**2/6) d3/dt3``;
    with ``periodic=False`` the two ends fall back to a second-order one-sided
    stencil.  A first-order backward difference is *not* offered on purpose: on
    a harmonic signal it biases the amplitude by ``O(Omega dt)`` and that bias
    would masquerade as a force error.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if values.shape[0] < 3:
        raise ValueError("need at least 3 samples for a second-order difference")
    if periodic:
        return (torch.roll(values, -1, 0) - torch.roll(values, 1, 0)) / (2.0 * dt)
    out = torch.empty_like(values)
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * dt)
    out[-1] = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * dt)
    return out


def cv_fluid_area(radius: float, body_area: float = 0.0) -> float:
    """``pi R**2 - A_body``: the fluid cross-section inside the control surface."""
    a = math.pi * radius ** 2 - body_area
    if a <= 0.0:
        raise ValueError("body_area must be smaller than the control surface")
    return a


def control_volume_force_unsteady(
    pos: torch.Tensor,
    vel: torch.Tensor,
    pressure: torch.Tensor,
    rho: float,
    centre: torch.Tensor,
    radius: float,
    dx: float,
    *,
    momentum_rate: torch.Tensor,
    frame_vel: torch.Tensor,
    frame_acc: torch.Tensor,
    body_area: float = 0.0,
    p_ref: float = 0.0,
    n_bins: Optional[int] = None,
    band: Optional[float] = None,
    volumes: Optional[torch.Tensor] = None,
    traction: Optional[torch.Tensor] = None,
    reconstruct: str = "fit",
    h_window: Optional[float] = None,
    momentum_rate_total: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Full unsteady control-volume force on the body inside a circular surface.

    Implements identity (1) of the module docstring in the perturbation form
    documented above the function.  Inputs:

    ``momentum_rate``     ``d/dt int_CV rho (u - frame_vel) dV`` (N, per unit
                          thickness), i.e. the derivative of
                          :func:`cv_momentum` with ``frame_vel`` supplied.  The
                          caller forms it from a time series with
                          :func:`d_dt_periodic`; the same ``pos`` lattice and
                          ``radius`` must be used for every sample.
    ``frame_vel``         undisturbed (free-stream) velocity ``u_f(t)``.
    ``frame_acc``         its time derivative ``a_f(t)`` -- *the caller is
                          responsible for consistency*; the displaced-mass term
                          is built from it.
    ``body_area``         cross-sectional area of the body inside the surface;
                          it is the only place the body enters (the panel/pair
                          arms need the body *surface*, this channel needs only
                          its area).
    ``momentum_rate_total`` optional control arm: the derivative of the *raw*
                          ``int rho u dV``.  When supplied, the returned
                          ``force_naive`` is the literal (unsplit) form of (1),
                          which is the estimator the previous round could not
                          ship because of the ``R**2`` cancellation.
    """
    if reconstruct not in ("fit", "mean"):
        raise ValueError("reconstruct must be 'fit' or 'mean'")
    frame_vel = torch.as_tensor(frame_vel, dtype=pos.dtype).reshape(-1)
    frame_acc = torch.as_tensor(frame_acc, dtype=pos.dtype).reshape(-1)
    if frame_vel.numel() != pos.shape[1] or frame_acc.numel() != pos.shape[1]:
        raise ValueError("frame_vel / frame_acc must match the spatial dimension")

    surf = circular_surface_bins(pos, centre, radius, dx, n_bins=n_bins,
                                 band=band, volumes=volumes)
    sel = surf["sel"]
    sub_pos = pos[sel]
    sub_vol = volumes[sel] if volumes is not None else None
    h_win = 1.6 * dx if h_window is None else h_window

    # perturbation pressure p' = p + rho a_f . (x - c)
    p_pert = pressure[sel] + rho * ((sub_pos - centre) @ frame_acc)
    fit_u, mean_u, ok_u = reconstruct_surface_values(
        surf["bin_points"], sub_pos, vel[sel], h_win, sub_vol)
    fit_p, mean_p, ok_p = reconstruct_surface_values(
        surf["bin_points"], sub_pos, p_pert, h_win, sub_vol)
    filled = ok_u & ok_p
    u_b, p_b = (fit_u, fit_p) if reconstruct == "fit" else (mean_u, mean_p)
    n = surf["normals"]
    ds = surf["ds"]
    mask = filled.to(u_b.dtype)
    mask2 = mask[:, None]

    un = (u_b * n).sum(dim=-1)
    f_mom = -((rho * u_b * un[:, None]) * mask2 * ds).sum(dim=0)
    f_pres = -(((p_b - p_ref)[:, None] * n) * mask2 * ds).sum(dim=0)
    f_visc = None
    if traction is not None:
        f_visc = (traction * mask2 * ds).sum(dim=0)

    displaced = rho * float(body_area) * frame_acc          # Froude-Krylov
    accum = -momentum_rate
    total = displaced + accum + f_mom + f_pres
    if f_visc is not None:
        total = total + f_visc

    out: Dict[str, object] = {
        "force": total,
        "force_displaced_mass": displaced.detach(),
        "force_unsteady": accum.detach(),
        "force_momentum_flux": f_mom.detach(),
        "force_pressure": f_pres.detach(),
        "force_viscous": None if f_visc is None else f_visc.detach(),
        "fluid_area": cv_fluid_area(float(radius), float(body_area)),
        "coverage": float(filled.to(torch.float64).mean()),
        "n_bins": int(surf["n_bins"]),
        "n_surface_particles": int(sel.sum()),
    }
    if reconstruct == "mean":
        un_m = (mean_u * n).sum(dim=-1)
        out["force_mean_reconstruction"] = -(
            rho * mean_u * un_m[:, None] + (mean_p - p_ref)[:, None] * n
        ).mul(mask2 * ds).sum(dim=0)
    if momentum_rate_total is not None:
        # literal, unsplit (1): -d/dt int rho u dV - oint [rho u (u.n) + p n] dS
        pa = pressure[sel]
        fit_pn, mean_pn, _ = reconstruct_surface_values(
            surf["bin_points"], sub_pos, pa, h_win, sub_vol)
        p_nt = fit_pn if reconstruct == "fit" else mean_pn
        f_pres_t = -(((p_nt - p_ref)[:, None] * n) * mask2 * ds).sum(dim=0)
        naive = -momentum_rate_total + f_mom + f_pres_t
        if f_visc is not None:
            naive = naive + f_visc
        out["force_naive"] = naive.detach()
        out["force_naive_pressure"] = f_pres_t.detach()
    return out


def unsteady_potential_flow_cylinder(pos: torch.Tensor, a: float,
                                     u_inf: float, u_dot: float, rho: float,
                                     centre: Optional[torch.Tensor] = None,
                                     p_inf: float = 0.0):
    """Potential flow past a fixed circular cylinder in an *accelerating* stream.

    ``u_inf`` / ``u_dot`` are the instantaneous uniform stream and its time
    derivative along ``+x``; the stream is evaluated as a snapshot, so the pair
    ``(u_inf, u_dot)`` is all the field needs.  Velocity from the stream
    function ``phi = U(t) (r + a**2/r) cos(theta)``:

        u_r = U (1 - a**2/r**2) cos t,   u_t = -U (1 + a**2/r**2) sin t

    and pressure from the *unsteady* Bernoulli equation,
    ``p = -rho dphi/dt - rho|u|**2/2 + rho U**2/2 + p_inf``:

        p - p_inf = -rho u_dot (r + a**2/r) cos t + rho (U**2 - |u|**2)/2

    so that ``p -> p_inf - rho u_dot x`` as ``r -> inf``, which is the pressure
    field that accelerates the uniform stream.  The exact load on the cylinder
    is ``F_x = 2 rho pi a**2 u_dot`` (``C_M = 2``: displaced mass ``rho pi a**2``
    plus added mass ``C_a = 1``) and ``F_y = 0`` -- the reference this module is
    audited against.
    """
    if centre is None:
        centre = torch.zeros(2, dtype=pos.dtype, device=pos.device)
    rel = pos - centre
    r = torch.linalg.norm(rel, dim=1).clamp_min(1e-12)
    cos_t = rel[:, 0] / r
    sin_t = rel[:, 1] / r
    eps = a * a / (r * r)
    u_r = u_inf * (1.0 - eps) * cos_t
    u_t = -u_inf * (1.0 + eps) * sin_t
    vel = torch.stack((u_r * cos_t - u_t * sin_t,
                       u_r * sin_t + u_t * cos_t), dim=-1)
    speed2 = (vel * vel).sum(dim=-1)
    pressure = (p_inf + 0.5 * rho * (u_inf ** 2 - speed2)
                - rho * u_dot * (r + a * a / r) * cos_t)
    return vel, pressure
