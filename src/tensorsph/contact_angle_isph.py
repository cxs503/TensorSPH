"""Wall-adhesion contact-angle force for 2-D SPH: what Young's equation actually asks for.

Why this module exists
----------------------
``csf_consistent.apply_contact_angle_boundary`` -- the shipped contact-angle BC,
the "interface normal" form of Adami et al. (2012) -- *rotates* the CSF force
direction near a wall while preserving ``|grad c|``.  Its amplitude therefore
stays ``sigma kappa |grad c|``, a curvature force, and no per-unit-contact-line
traction is ever applied.  The measured consequence is in
``docs/verification_contact_angle_bc.md``: set angles 60/90/120 deg relax to
151.4/152.6/154.1 deg with a response slope ``d theta_meas / d theta_set =
-0.025``.  The drop stays a cap tangent to the wall (``theta -> pi``) whatever
angle is set, because nothing in that BC pushes the contact line sideways.

This module supplies the missing *traction*, acting on the interface particles
next to the wall (Breinlinger, Polfer & Kraft 2014; Akinci et al. 2013 for the
wall delta function):

    f_i = C sigma cos(theta) * s_i * delta_s,i * w_i

with ``s_i`` the unit wall-tangential direction pointing **away from the
liquid** (the tangential part of the outward interface normal), ``delta_s,i``
the CSF surface delta (``delta_s`` integrates to 1 across the interface, so the
per-particle force scales as ``1/dx``, as a per-unit-length traction must), and
``w_i`` the wall proximity weight built from a wall surface delta

    delta_w,i = sum_{k in wall} W(|r_i - r_k|, h) A_k / (1/2),  w = clip(delta_w/delta_w_ref, 0, 1)

(the ``1/2`` is exact by isotropy: ``int_{half plane} W dA = 1/2``, so
``int delta_w dA = sum_k A_k`` -- the delta carries a surface measure instead of
a grid-dependent one).

Per unit contact line this is ``sigma cos theta`` along the wall: outward for a
wetting wall (``theta < pi/2``), inward for a non-wetting one, zero at
``theta = pi/2``.  That is Young's law ``gamma_sv - gamma_sl = sigma cos theta``
read as the wall traction the liquid feels, and it is what
:func:`traction_audit` measures.

``C`` is the *traction calibration*: one O(1) amplitude fixed on **analytic**
contact-line geometry (never on the relaxation it is later used to predict).
The alternative amplitude form from the same literature,
``f = C sigma (n_wall - cos theta n) delta_s w``, is kept as ``model="brackbill"``
and audited side by side, because its wall-tangential resultant is
``-sigma cos theta sin theta`` (it peaks at 45 deg instead of at the two
limits); the audit table decides, not this docstring.

Sign convention: ``theta`` is measured inside the liquid from the wall,
``theta = pi/2`` neutral, ``theta -> 0`` perfectly wetting.
"""
from __future__ import annotations

import math

import torch

from .csf_consistent import nearest_wall
from .droplet_impact import cubic_spline_kernel

#: ``int_{half plane} W dA`` for a normalised isotropic 2-D kernel.
HALF_PLANE_KERNEL_INTEGRAL = 0.5

#: ``delta_w`` of a fluid particle half a spacing above a flat lattice wall at
#: ``h = 1.3 dx`` (cubic spline, 2-D, unit wall area per particle).  Used only to
#: turn the wall delta into the dimensionless weight ``w in [0, 1]``; it is a
#: property of the kernel + lattice, measured by :func:`wall_delta_reference`.
DEFAULT_WALL_DELTA_REF = 0.4734

#: Traction calibration of the ``"young"`` model, fixed on the analytic audit
#: (:func:`traction_audit_ladder`: per-angle ratio 1.071/1.005/1.040/1.048 over
#: ``theta = 30/60/90/120`` deg, spread 6.4 %), i.e. ``1 / 1.0405``.  Measured
#: on analytic contact-line geometry only -- never fitted to a relaxation.
DEFAULT_TRACTION_CALIBRATION = 0.9611


def _wall_spacing(wall_pos: torch.Tensor) -> float:
    """Median nearest-neighbour distance of the wall particles."""
    if wall_pos.shape[0] < 2:
        return 1.0
    d = torch.cdist(wall_pos, wall_pos)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.median())


def wall_delta(
    pos: torch.Tensor,
    wall_pos: torch.Tensor,
    h: float,
    wall_area: float | None = None,
    chunk: int = 2048,
) -> torch.Tensor:
    """Wall surface delta function seen by each fluid particle.

    ``delta_w,i = sum_{k in wall} W(|r_i - r_k|, h) * A_k / (1/2)``: the wall area
    density available to particle ``i`` in units where the exact half-plane
    integral of the kernel normalises it to one ``A_k`` per wall particle.
    """
    n = pos.shape[0]
    out = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    if wall_pos is None or wall_pos.numel() == 0 or n == 0:
        return out
    if wall_area is None:
        wall_area = float(_wall_spacing(wall_pos) ** pos.shape[1])
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(pos[start:stop], wall_pos)
        out[start:stop] = cubic_spline_kernel(d, h).sum(dim=1)
    return out * (float(wall_area) / HALF_PLANE_KERNEL_INTEGRAL)


def wall_delta_reference(dx: float, h_factor: float = 1.3,
                         wall_layers: int = 3) -> float:
    """``delta_w`` of a particle half a spacing above a flat lattice wall.

    Measured, not assumed: this is the normalisation reference of the wall
    weight ``w``, and every audit row reports the value it used.
    """
    wx = torch.arange(-10, 11, dtype=torch.float64) * dx
    wall = torch.cat([torch.stack((wx, torch.full_like(wx, 0.5 * dx - k * dx)),
                                 dim=-1) for k in range(1, wall_layers + 1)], dim=0)
    probe = torch.tensor([[0.0, 0.5 * dx]], dtype=torch.float64)
    return float(wall_delta(probe, wall, h_factor * dx)[0])


def wall_weight(delta_w: torch.Tensor,
                reference: float = DEFAULT_WALL_DELTA_REF) -> torch.Tensor:
    """Dimensionless wall proximity weight ``clip(delta_w / reference, 0, 1)``."""
    return (delta_w / float(reference)).clamp(0.0, 1.0)


def wall_tangent(normal: torch.Tensor, n_wall: torch.Tensor) -> torch.Tensor:
    """Unit wall tangent pointing **away from the liquid**.

    The outward interface normal points away from the liquid, so its tangential
    part does too.  The tangential part vanishes where the interface meets the
    wall perpendicularly; there ``cos theta = 0`` as well, so the fallback value
    never carries force and only keeps the arithmetic finite.
    """
    tangent = normal - (normal * n_wall).sum(dim=1, keepdim=True) * n_wall
    mag = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    fallback = torch.zeros_like(tangent)
    fallback[:, 0] = 1.0
    fallback = fallback - (fallback * n_wall).sum(dim=1, keepdim=True) * n_wall
    fallback = fallback / torch.linalg.vector_norm(
        fallback, dim=1, keepdim=True).clamp_min(1e-30)
    return torch.where(mag < 1e-9, fallback, tangent / mag.clamp_min(1e-30))


def wall_adhesion_force(
    pos: torch.Tensor,
    wall_pos: torch.Tensor,
    normal: torch.Tensor,
    theta: float,
    sigma: float,
    delta_s: torch.Tensor,
    h: float,
    wall_area: float | None = None,
    calibration: float = 1.0,
    model: str = "young",
    delta_w: torch.Tensor | None = None,
    wall_delta_ref: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Wall-adhesion force on the interface particles next to a wall.

    Args:
        pos: (N, D) fluid positions.
        wall_pos: (M, D) wall particle positions.
        normal: (N, D) outward interface normal ``n`` (zero rows in the bulk).
        theta: contact angle in radians, measured inside the liquid.
        sigma: surface tension coefficient (N/m).
        delta_s: (N,) CSF surface delta, ``int delta_s dn = 1`` across the
            interface -- what makes the force a per-unit-length traction.
        h: smoothing length.
        calibration: the O(1) traction calibration ``C`` (see
            :func:`traction_audit`).
        model: ``"young"`` (default): ``sigma cos theta`` along the wall away
            from the liquid; ``"brackbill"``: ``sigma (n_wall - cos theta n)``.
        delta_w: optional precomputed wall delta.
        wall_delta_ref: optional normalisation reference for ``w``.

    Returns:
        ``(force, diagnostics)``: (N, D) per-particle force in the same units the
        CSF force uses (divide by mass for an acceleration), plus the scalars the
        audit needs.
    """
    diag = {"model": model, "theta_deg": math.degrees(float(theta))}
    n = pos.shape[0]
    if sigma <= 0.0 or n == 0:
        return torch.zeros_like(pos), diag
    ref = DEFAULT_WALL_DELTA_REF if wall_delta_ref is None else float(wall_delta_ref)
    delta_w = wall_delta(pos, wall_pos, h, wall_area) if delta_w is None else delta_w
    w = wall_weight(delta_w, ref)
    _, n_wall = nearest_wall(pos, wall_pos)
    cos_t = math.cos(float(theta))
    if model == "young":
        direction = wall_tangent(normal, n_wall) * cos_t
    elif model == "brackbill":
        direction = n_wall - cos_t * normal
    else:
        raise ValueError(f"unknown contact-angle force model {model!r}")
    coef = (float(calibration) * float(sigma)) * delta_s * w
    force = coef[:, None] * direction
    diag.update({
        "delta_w_max": float(delta_w.max()),
        "delta_w_ref": ref,
        "wall_weight_mean_active": (float(w[w > 0].mean())
                                    if bool((w > 0).any()) else 0.0),
        "n_wall_weighted": int((w > 0).sum()),
        "cos_theta": cos_t,
    })
    return force, diag


# --------------------------------------------------------------------------- #
# Analytic audit: traction per unit contact line on a known patch
# --------------------------------------------------------------------------- #
def contact_line_patch(
    dx: float,
    theta_deg: float,
    arc_radius_over_dx: float = 40.0,
    wall_layers: int = 3,
    pad_layers: int = 6,
    profile_width_dx: float = 0.75,
    band_dx: float = 5.0,
) -> dict:
    """Lattice patch around one contact line, with analytic geometry and delta_s.

    The liquid is the part of the disc ``|r - C| <= R_c`` above the wall
    ``y = 0`` with ``C = (0, R_c cos theta)`` -- the cap whose contact angle is
    ``theta`` by construction.  Two fields are analytic and therefore carry no
    colour/curvature operator error:

    * ``n = (r - C)/|r - C|``, the outward interface normal;
    * ``delta_s = |grad c|`` of the smooth colour field
      ``c = clip(1/2 - (|r - C| - R_c)/(2 w_prof dx), 0, 1)``, differenced on the
      lattice.  ``int delta_s dn = 1`` across the interface is checked separately
      by :func:`delta_s_normalisation`.

    Returns a dict with ``pos``, ``wall``, ``normal``, ``delta_s``, ``contact_x``,
    ``tangent`` (+x, away from the liquid), ``arc_radius_m`` and the colour grid.
    """
    theta = math.radians(float(theta_deg))
    rc = arc_radius_over_dx * dx
    cy = rc * math.cos(theta)
    x_c = rc * math.sin(theta)
    # only the near-wall part of the interface is needed: a band of ``band_dx``
    # spacings above the wall.  (With a large arc radius the normal barely
    # rotates across that band, so the patch is a local contact-line probe and
    # not a probe of the whole cap.)
    height = band_dx * dx
    # one contact line only: from a few spacings *inside* the liquid to a few
    # spacings past the contact point.  Including the opposite contact line of the
    # cap would sum two opposite tractions to zero and measure nothing.
    xs = torch.arange(x_c - 4.0 * dx, x_c + 4.0 * dx, dx, dtype=torch.float64)
    ys = torch.arange(0.5 * dx, height + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    centre = torch.tensor([0.0, cy], dtype=torch.float64)

    r_grid = (torch.stack((gx, gy), dim=-1) - centre).norm(dim=-1)
    c = (0.5 - (r_grid - rc) / (2.0 * profile_width_dx * dx)).clamp(0.0, 1.0)
    dcx = torch.zeros_like(c)
    dcy = torch.zeros_like(c)
    dcx[:, 1:-1] = (c[:, 2:] - c[:, :-2]) / (2.0 * dx)
    dcy[1:-1, :] = (c[2:, :] - c[:-2, :]) / (2.0 * dx)
    delta_s_grid = torch.sqrt(dcx ** 2 + dcy ** 2)

    coords = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    r = (coords - centre).norm(dim=1)
    h = 1.3 * dx
    # the wall-adjacent interface band: within 1.5 dx of the interface AND within
    # 2 h of the wall.  No wall BC can act anywhere else.
    interface = (r - rc).abs() <= 1.5 * dx
    near_wall = coords[:, 1] <= 2.0 * h + 0.5 * dx
    keep = interface & near_wall
    pos = coords[keep]
    normal = (pos - centre) / (pos - centre).norm(dim=1, keepdim=True)
    delta_s = delta_s_grid.reshape(-1)[keep]

    wx = torch.arange(x_c - 8.0 * dx, x_c + 8.0 * dx, dx,
                      dtype=torch.float64)
    wall = torch.cat([torch.stack((wx, torch.full_like(wx, 0.5 * dx - k * dx)),
                                 dim=-1) for k in range(1, wall_layers + 1)], dim=0)
    return {
        "pos": pos, "wall": wall, "normal": normal, "delta_s": delta_s,
        "dx": float(dx), "contact_x": x_c, "arc_radius_m": rc,
        "theta_deg": float(theta_deg),
        "tangent": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "colour_grid": (gx, gy, c),
    }


def delta_s_normalisation(dx: float, theta_deg: float = 90.0,
                          arc_radius_over_dx: float = 8.0) -> dict:
    """``int delta_s dn`` across the interface of the audit patch (must be 1)."""
    patch = contact_line_patch(dx, theta_deg, arc_radius_over_dx=arc_radius_over_dx)
    gx, gy, c = patch["colour_grid"]
    dcx = torch.zeros_like(c)
    dcy = torch.zeros_like(c)
    dcx[:, 1:-1] = (c[:, 2:] - c[:, :-2]) / 2.0
    dcy[1:-1, :] = (c[2:, :] - c[:-2, :]) / 2.0
    ds = torch.sqrt(dcx ** 2 + dcy ** 2)
    # every column whose interface lies inside the band carries the same
    # integral; the spread over the middle half of the patch is the honest
    # single number to report
    lo = c.shape[1] // 4
    hi = max(lo + 1, (3 * c.shape[1]) // 4)
    integrals = [float(ds[:, col].sum()) for col in range(lo, hi)]
    integrals.sort()
    integral = integrals[len(integrals) // 2]
    return {"dx_m": float(dx), "column": int(lo + len(integrals) // 2),
            "integral_delta_s_dn": integral,
            "integral_delta_s_dn_min": integrals[0],
            "integral_delta_s_dn_max": integrals[-1],
            "theta_deg": float(theta_deg)}


def traction_audit(
    dx: float,
    theta_deg: float,
    model: str = "young",
    sigma: float = 1.0,
    h_factor: float = 1.3,
    calibration: float = 1.0,
    arc_radius_over_dx: float = 40.0,
    wall_delta_ref: float | None = None,
) -> dict:
    """Per-unit-contact-line traction of the discrete adhesion force.

    In 2-D there is one contact line per side, so the wall-tangential resultant
    ``T = sum_i f_i.t dx`` **is** the per-unit-length traction.  Young's law
    requires ``T = sigma cos theta`` for the ``"young"`` model (outward for a
    wetting wall); ``"brackbill"`` produces whatever it produces and the table
    says so.
    """
    patch = contact_line_patch(dx, theta_deg, arc_radius_over_dx=arc_radius_over_dx)
    h = h_factor * dx
    force, diag = wall_adhesion_force(
        patch["pos"], patch["wall"], patch["normal"], math.radians(theta_deg),
        sigma, patch["delta_s"], h, calibration=calibration, model=model,
        wall_delta_ref=wall_delta_ref)
    f_t = (force * patch["tangent"]).sum(dim=1)
    total_t = float(f_t.sum()) * dx
    target = float(sigma) * math.cos(math.radians(theta_deg))
    cal = (target / total_t) if total_t != 0.0 else float("nan")
    return {
        "theta_deg": float(theta_deg),
        "dx_m": float(dx),
        "R_over_dx": float(arc_radius_over_dx),
        "model": model,
        "traction_per_line": total_t,
        "traction_target": target,
        "traction_ratio": (total_t / target) if target else float("nan"),
        "calibration_measured": cal,
        "n_patch_particles": int(patch["pos"].shape[0]),
        "delta_s_max": float(patch["delta_s"].max()),
        "delta_w_max": diag.get("delta_w_max", float("nan")),
        "wall_weight_mean_active": diag.get("wall_weight_mean_active", float("nan")),
    }


def traction_audit_ladder(
    dx: float,
    angles_deg=(30.0, 60.0, 90.0, 120.0),
    model: str = "young",
    sigma: float = 1.0,
    h_factor: float = 1.3,
    arc_radius_over_dx: float = 40.0,
    wall_delta_ref: float | None = None,
) -> dict:
    """Per-angle audit rows plus the spread of the measured calibration.

    ``calibration_spread_rel`` is the model's internal-consistency number: a
    single amplitude ``C`` is only legitimate if the *measured* calibration is
    angle-independent, i.e. if the discrete traction really does follow
    ``sigma cos theta``.
    """
    rows = [traction_audit(dx, a, model=model, sigma=sigma, h_factor=h_factor,
                           arc_radius_over_dx=arc_radius_over_dx,
                           wall_delta_ref=wall_delta_ref) for a in angles_deg]
    cal = [r["calibration_measured"] for r in rows
           if abs(r["traction_target"]) > 0.05 * float(sigma)
           and r["calibration_measured"] == r["calibration_measured"]]
    mean = sum(cal) / len(cal) if cal else float("nan")
    spread = (max(cal) - min(cal)) / abs(mean) if cal and mean else float("nan")
    return {
        "rows": rows,
        "model": model,
        "calibration_mean": mean,
        "calibration_spread_rel": spread,
        "dx_m": float(dx),
        "h_factor": float(h_factor),
        "arc_radius_over_dx": float(arc_radius_over_dx),
    }