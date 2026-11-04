"""Verification of the contact-angle wall BC on exact 2-D sessile droplets.

Why this benchmark exists
-------------------------
``apply_contact_angle_boundary`` (merged in m2090, ``docs/contact_angle_boundary.md``)
is documented as "enforcing Young's contact angle on the colour gradient near a
wall".  The evidence shipped with it is ten unit tests, all of which feed a
*synthetic* gradient into the rotation and check that the rotation happened --
e.g. ``test_hydrophilic_angle_tilts_toward_wall`` asserts the code does what the
code does.  None of them measures the contact angle of an actual interface, so
the model has never been shown to reproduce the angle it is given, nor to leave
the curvature operator usable at the contact line.

This benchmark supplies the missing instrument: an *analytic* sessile droplet.

Geometry
--------
A 2-D droplet (infinite cylinder cross-section) resting on a flat wall at
``y = 0`` with contact angle ``theta`` measured *inside* the liquid occupies the
region inside a circle of radius

    R = a / sin(theta),      centre at  (0, y_c),   y_c = -a / tan(theta) ,

intersected with ``y > 0``.  The circle meets the wall at ``x = +-a``, and the
angle between the wall and the interface tangent there is exactly ``theta``
(checked numerically by :func:`cap_geometry_check`, not asserted by hand).  The
interface curvature is the constant ``kappa = 1/R``, so this field is a complete
analytic reference for both the normal direction and the curvature.

What is measured
----------------
For each ``(theta, R/dx)`` case the colour field is built by the *production*
pipeline (``kernel_sum_color`` + MLS gradient, i.e. exactly what
``csf_fields`` does), optionally with the contact-angle BC, and then

* ``theta_fit``      geometric contact angle of the ``c = 0.5`` shell
                     (Kasa circle fit -> ``arccos(-y_c/R)``).  The BC only
                     touches normals, never positions, so this is the *shape*
                     bias the BC cannot repair;
* normal error       angle between the model normal and the analytic normal,
                     over the interface shell and over the wall-adjacent band
                     (``d < h``), with the BC off and on;
* curvature error    ``kappa = div n`` against ``1/R``, both as a
                     ``delta_s``-weighted mean (the quantity that sets the CSF
                     force) and as a median over the interface shell.

Controls (the instrument must be held to its own standard)
----------------------------------------------------------
* ``C1 exactness``   feeding the *analytic* normals through the BC must return
                     them unchanged (the target ``cos(theta) n_wall +
                     sin(theta) t`` equals the input when the input already
                     satisfies ``n . n_wall = cos(theta)``);
* ``C2 locality``    particles further than ``2h`` from the wall must be bitwise
                     identical with and without the BC;
* ``C3 assembly``    the locally assembled fields must equal ``csf_fields``
                     (proves the benchmark measures the production path);
* ``C4 wall rows``   the production BC is applied to wall particles too
                     (``d = 0`` there, so ``blend = 1``).  Their rotated normals
                     enter ``div n`` of the fluid particles next to the wall;
                     the effect is measured by also evaluating a fluid-only
                     variant.

Everything is static: no time integration, no EOS, so the numbers isolate the
operator.  A dynamic sessile-droplet relaxation is deliberately *not* used --
at ``c0 = 20 m/s`` the Laplace pressure lives at ``drho/rho ~ 1e-6``, below SPH
density noise (``benchmarks/laplace_droplet/run.py`` records that argument).

Usage
-----
    PYTHONPATH=src python benchmarks/contact_angle_geometry/run.py [--plot]

Writes ``results.json`` (and ``contact_angle_geometry.png`` with ``--plot``)
next to this file.  ``make_doc.py`` injects the numbers into
``docs/verification_contact_angle_geometry.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from tensorsph.csf_consistent import (apply_contact_angle_boundary, csf_fields,
                                      kernel_sum_color, mls_divergence,
                                      mls_gradient, nearest_wall, shepard_smooth)

HERE = Path(__file__).resolve().parent
RHO0 = 1000.0
SIGMA = 0.072
THETAS_DEG = (60.0, 90.0, 120.0)
RATIOS = (10, 20, 40)
R_ARC = 0.010                      # analytic interface radius [m]
TOL_ANGLE_DEG = 5.0                # contact-angle acceptance (experiments: +-3..5 deg)
TOL_CURVATURE = 0.15              # repo-wide relative gate on a curvature/force quantity
TINY = 1e-12


# --------------------------------------------------------------------------- #
# Analytic geometry
# --------------------------------------------------------------------------- #

def cap_geometry(a: float, theta: float) -> tuple[float, float]:
    """``(R, y_c)`` of the circular interface through ``(+-a, 0)`` at angle ``theta``."""
    return a / math.sin(theta), -a / math.tan(theta)


def cap_geometry_check() -> dict:
    """Numeric proof that the closed form above gives a contact angle ``theta``.

    The angle between the wall tangent ``(-1, 0)`` and the interface tangent is
    measured at the right contact point ``(a, 0)`` for a sweep of ``theta``.
    """
    worst = 0.0
    for deg in (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 165.0):
        theta = math.radians(deg)
        a = 1.0
        R, yc = cap_geometry(a, theta)
        # interface point at the contact line and the derivative of the arc there
        p = torch.tensor([a, 0.0], dtype=torch.float64)
        c = torch.tensor([0.0, yc], dtype=torch.float64)
        radial = (p - c) / (p - c).norm()
        # tangent along the arc away from the contact line, toward the apex
        tangent = torch.tensor([-radial[1], radial[0]], dtype=torch.float64)
        wall_tangent = torch.tensor([-1.0, 0.0], dtype=torch.float64)
        measured = float(torch.arccos((tangent * wall_tangent).sum().clamp(-1.0, 1.0)))
        worst = max(worst, abs(measured - theta))
    return {"max_abs_error_rad": worst, "max_abs_error_deg": math.degrees(worst),
            "result": "pass" if math.degrees(worst) < 1e-6 else "fail"}


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #

def cap_lattice(theta: float, dx: float, wall_layers: int = 3):
    """Fluid particles on a square lattice inside the cap, plus a flat wall.

    Fluid rows sit at ``y = (k + 0.5) dx`` and wall rows at ``y = -(k + 0.5) dx``,
    the convention of :func:`tensorsph.droplet_impact.wall_lattice`.
    Returns ``(pos_all, mass_all, wall, n_fluid, geom)`` with the fluid particles
    first, so ``pos_all[:n_fluid]`` is the fluid.
    """
    a = R_ARC * math.sin(theta)
    R, yc = cap_geometry(a, theta)
    centre = torch.tensor([0.0, yc], dtype=torch.float64)

    # fluid: square lattice, rows above the wall, kept inside the analytic arc
    height = yc + R                                # apex of the cap
    xs = torch.arange(-R - 2 * dx, R + 2 * dx + 0.5 * dx, dx, dtype=torch.float64)
    ys = torch.arange(0.5 * dx, height + 2 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    keep = (pts - centre).norm(dim=1) <= R
    fluid = pts[keep].clone()
    n_fluid = fluid.shape[0]
    m_fluid = torch.full((n_fluid,), RHO0 * dx * dx, dtype=torch.float64)

    # wall: `wall_layers` rows at the fluid spacing, one spacing wide margin
    wx = torch.arange(-a - (wall_layers + 1) * dx, a + (wall_layers + 1) * dx,
                      dx, dtype=torch.float64)
    rows = []
    for layer in range(1, wall_layers + 1):
        y = torch.full_like(wx, 0.5 * dx - layer * dx)
        rows.append(torch.stack((wx, y), dim=1))
    wall = torch.cat(rows, dim=0)
    m_wall = torch.full((wall.shape[0],), RHO0 * dx * dx, dtype=torch.float64)

    pos_all = torch.cat((fluid, wall), dim=0)
    mass_all = torch.cat((m_fluid, m_wall), dim=0)
    geom = {"a": a, "R": R, "y_c": yc, "height": height,
            "n_fluid": n_fluid, "n_wall": wall.shape[0], "dx": dx}
    return pos_all, mass_all, wall, n_fluid, geom


def prune_wall_wall(pairs: torch.Tensor, n_fluid: int) -> torch.Tensor:
    """Same pruning as ``droplet_impact.drop_wall_wall_pairs`` (wall-wall is waste)."""
    if pairs.numel() == 0:
        return pairs
    i, j = pairs.unbind(dim=1)
    return pairs[(i < n_fluid) | (j < n_fluid)]


# --------------------------------------------------------------------------- #
# Field assembly (mirrors csf_fields, but with the gradient pluggable)
# --------------------------------------------------------------------------- #

def assemble(pos: torch.Tensor, mass: torch.Tensor, pairs: torch.Tensor, h: float,
             grad_c: torch.Tensor, normal_smoothing: int = 2):
    """Normal / curvature / delta_s from a given colour gradient (= ``csf_fields``)."""
    vol = mass / RHO0
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    normal = torch.where(mag > TINY, -grad_c / mag.clamp_min(TINY),
                         torch.zeros_like(grad_c))
    if normal_smoothing:
        sm = shepard_smooth(normal, pos, vol, pairs, h, passes=normal_smoothing)
        normal = torch.where(mag > TINY,
                             sm / torch.linalg.vector_norm(sm, dim=1, keepdim=True).clamp_min(TINY),
                             normal)
    kappa = mls_divergence(pos, normal, vol, pairs, h)
    delta_s = 2.0 * torch.linalg.vector_norm(grad_c, dim=1)
    return normal, kappa, delta_s


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #

def angle_deg(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    cos = (v * w).sum(dim=1) / (v.norm(dim=1) * w.norm(dim=1)).clamp_min(TINY)
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def kasa_fit(pos: torch.Tensor) -> tuple[float, float]:
    """Algebraic (Kasa) circle fit: returns ``(R, y_c)`` of the fitted arc."""
    x, y = pos[:, 0], pos[:, 1]
    big_a = torch.stack((x, y, torch.ones_like(x)), dim=1)
    rhs = x.square() + y.square()
    sol = torch.linalg.lstsq(big_a, rhs[:, None]).solution[:, 0]
    cx, cy = 0.5 * sol[0], 0.5 * sol[1]
    R = torch.sqrt(sol[2] + cx.square() + cy.square())
    return float(R), float(cy)


def run_case(theta_deg: float, ratio: int, normal_smoothing: int = 2) -> dict:
    theta = math.radians(theta_deg)
    dx = R_ARC / ratio
    h = 1.3 * dx
    pos, mass, wall, n_fluid, geom = cap_lattice(theta, dx)
    from tensorsph.neighbor_search import cell_linked_pairs
    pairs = prune_wall_wall(cell_linked_pairs(pos, 2.0 * h), n_fluid)

    fluid = pos[:n_fluid]
    vol = mass / RHO0
    colour = kernel_sum_color(pos, mass, pairs, h, RHO0)
    grad = mls_gradient(pos, colour, vol, pairs, h)

    # --- control C3: the manual assembly must equal the production csf_fields
    prod = csf_fields(pos, mass, pairs, h, RHO0, wall=wall, contact_angle=theta,
                      normal_smoothing=normal_smoothing)
    grad_prod = prod.grad_c
    n_ref, k_ref, ds_ref = assemble(pos, mass, pairs, h, grad_prod, normal_smoothing)
    assembly_error = float(max((n_ref - prod.normal).abs().max(),
                               (k_ref - prod.kappa).abs().max(),
                               (ds_ref - prod.delta_s).abs().max()))

    d, n_wall = nearest_wall(pos, wall)

    # --- BC variants -----------------------------------------------------
    grad_bc = apply_contact_angle_boundary(grad, pos, wall, h, theta)        # production
    grad_fl = grad.clone()
    grad_fl[:n_fluid] = apply_contact_angle_boundary(
        grad[:n_fluid], fluid, wall, h, theta)                              # fluid rows only
    variants = {"off": grad, "on": grad_bc, "fluid_only": grad_fl}

    centre = torch.tensor([0.0, geom["y_c"]], dtype=torch.float64)
    rel = fluid - centre
    n_exact = rel / rel.norm(dim=1, keepdim=True).clamp_min(TINY)           # outward normal
    shell = (colour[:n_fluid] > 0.25) & (colour[:n_fluid] < 0.75)
    near_wall = (d[:n_fluid] < h) & shell
    far = d[:n_fluid] > 2.0 * h

    out: dict = {
        "theta_deg": theta_deg, "ratio": ratio, "dx": dx, "h": h,
        "n_fluid": n_fluid, "n_wall": geom["n_wall"], "n_pairs": int(pairs.shape[0]),
        "assembly_max_abs_error_vs_csf_fields": assembly_error,
        "n_shell": int(shell.sum()), "n_shell_near_wall": int(near_wall.sum()),
    }

    # --- geometric contact angle: the shape the lattice actually has ------
    fit_mask = shell & (d[:n_fluid] > 1.2 * h) & (fluid[:, 1] > 0.0)
    if int(fit_mask.sum()) >= 6:
        R_fit, yc_fit = kasa_fit(fluid[fit_mask])
        theta_fit = math.degrees(math.acos(max(-1.0, min(1.0, -yc_fit / R_fit))))
        out["fit"] = {"n_points": int(fit_mask.sum()), "R_fit": R_fit,
                      "y_c_fit": yc_fit, "R_rel_error": (R_fit - geom["R"]) / geom["R"],
                      "theta_fit_deg": theta_fit,
                      "theta_error_deg": theta_fit - theta_deg,
                      "result": ("pass" if abs(theta_fit - theta_deg) <= TOL_ANGLE_DEG
                                 else "fail")}
    else:
        out["fit"] = {"n_points": int(fit_mask.sum()), "result": "not_evaluated",
                      "note": "fewer than 6 shell particles clear of the wall"}

    # --- per-variant normal and curvature errors --------------------------
    errs = {}
    for name, g in variants.items():
        n, k, ds = assemble(pos, mass, pairs, h, g, normal_smoothing)
        nf, kf = n[:n_fluid], k[:n_fluid]
        ang = angle_deg(nf, n_exact)
        k_target = 1.0 / geom["R"]
        w = ds[:n_fluid][shell]
        k_shell = kf[shell]
        k_weighted = float((k_shell * w).sum() / w.sum().clamp_min(TINY))
        errs[name] = {
            "normal_angle_deg_shell_mean": float(ang[shell].mean()),
            "normal_angle_deg_shell_median": float(ang[shell].median()),
            "normal_angle_deg_near_wall_mean": (float(ang[near_wall].mean())
                                                if int(near_wall.sum()) else None),
            "kappa_median_rel_shell": float(((k_shell / k_target) - 1.0).median()),
            "kappa_delta_s_weighted": k_weighted,
            "kappa_delta_s_weighted_rel_error": k_weighted * geom["R"] - 1.0,
            "n_finite": bool(torch.isfinite(nf).all() and torch.isfinite(kf).all()),
        }
    out["variants"] = errs

    # --- control C2: locality of the BC ----------------------------------
    far_all = d > 2.0 * h
    out["locality_max_abs_delta"] = float((grad_bc[far_all] - grad[far_all]).abs().max())
    out["locality_result"] = "pass" if out["locality_max_abs_delta"] == 0.0 else "fail"

    # --- control C4: how many wall particles the BC rotates ---------------
    rotated = (grad_bc[:n_fluid] - grad[:n_fluid]).abs().sum(dim=1) > 0.0
    rotated_wall = (grad_bc[n_fluid:] - grad[n_fluid:]).abs().sum(dim=1) > 0.0
    out["rotated_fluid_fraction"] = float(rotated.double().mean())
    out["rotated_wall_fraction"] = float(rotated_wall.double().mean())
    out["n_rotated_wall"] = int(rotated_wall.sum())

    out["acceptance"] = {
        "theta_within_5deg": out["fit"].get("result", "not_evaluated"),
        "normal_error_improves_with_bc":
            ("pass" if (errs["on"]["normal_angle_deg_shell_mean"]
                        <= errs["off"]["normal_angle_deg_shell_mean"]) else "fail"),
        "curvature_within_15pct":
            ("pass" if abs(errs["on"]["kappa_delta_s_weighted_rel_error"]) <= TOL_CURVATURE
             else "fail"),
        "no_non_finite_values":
            "pass" if all(e["n_finite"] for e in errs.values()) else "fail",
    }
    return out


# --------------------------------------------------------------------------- #
# Controls that do not need a lattice
# --------------------------------------------------------------------------- #

def control_exactness() -> dict:
    """C1: the BC must be a no-op on normals that already satisfy Young's law.

    Analytic setup: wall normal ``n_wall = (0, 1)``, a normal field tilted by
    exactly ``theta`` from it, at a distance from the wall where ``blend = 1``
    would be reached -- here the operator is called directly on a particle one
    wall spacing above the wall, so the test isolates the rotation direction
    from the blending.
    """
    rows = []
    for deg in (30.0, 60.0, 90.0, 120.0, 150.0):
        theta = math.radians(deg)
        # the BC works with n . n_wall = cos(theta), so tilt the normal from the
        # wall normal (0, 1) by theta: n = (sin theta, cos theta)
        n = torch.tensor([[math.sin(theta), math.cos(theta)]], dtype=torch.float64)
        pos = torch.tensor([[0.0, 1e-9]], dtype=torch.float64)
        wall = torch.tensor([[0.0, -1e-9]], dtype=torch.float64)
        out = apply_contact_angle_boundary(-n, pos, wall, h=1.0, theta=theta)
        got = (-out)[0]
        rows.append({"theta_deg": deg,
                     "deviation_deg": float(angle_deg(got[None, :], n)[0])})
    worst = max(r["deviation_deg"] for r in rows)
    return {"rows": rows, "max_deviation_deg": worst,
            "result": "pass" if worst < 1e-6 else "fail"}


def control_wall_normal_fallback() -> dict:
    """C4b: what direction does a wall row itself get?  (``d = 0`` -> ``blend = 1``.)"""
    wall = torch.stack(torch.meshgrid(
        torch.arange(-0.02, 0.0201, 0.0005, dtype=torch.float64),
        torch.tensor([-0.00025, -0.00075, -0.00125], dtype=torch.float64),
        indexing="xy"), dim=-1).reshape(-1, 2)
    gen = torch.Generator().manual_seed(0)
    fluid = torch.stack((torch.rand(40, generator=gen, dtype=torch.float64) * 0.02 - 0.01,
                         torch.rand(40, generator=gen, dtype=torch.float64) * 0.02 + 0.0002),
                        dim=1)
    pos = torch.cat((fluid, wall), dim=0)
    grad = torch.zeros_like(pos)
    grad[:, 1] = 1.0                      # arbitrary non-zero gradient everywhere
    theta = math.radians(90.0)
    out = apply_contact_angle_boundary(grad, pos, wall, h=0.00065, theta=theta)
    changed = (out - grad).abs().sum(dim=1) > 0.0
    wall_changed = int(changed[fluid.shape[0]:].sum())
    return {"n_wall": int(wall.shape[0]), "n_wall_changed": wall_changed,
            "n_fluid_changed": int(changed[:fluid.shape[0]].sum()),
            "result": "recorded"}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_all(thetas=THETAS_DEG, ratios=RATIOS) -> dict:
    cases = [run_case(t, r) for t in thetas for r in ratios]
    summary = {
        "theta_max_abs_error_deg": max(
            abs(c["fit"].get("theta_error_deg", float("nan"))) for c in cases
            if "theta_error_deg" in c["fit"]),
        "theta_all_within_5deg": all(c["acceptance"]["theta_within_5deg"] == "pass"
                                     for c in cases),
        "normal_error_improves_all": all(
            c["acceptance"]["normal_error_improves_with_bc"] == "pass" for c in cases),
        "curvature_all_within_15pct": all(
            c["acceptance"]["curvature_within_15pct"] == "pass" for c in cases),
        "assembly_max_abs_error": max(
            c["assembly_max_abs_error_vs_csf_fields"] for c in cases),
        "locality_all_pass": all(c["locality_result"] == "pass" for c in cases),
        "any_non_finite": not all(c["acceptance"]["no_non_finite_values"] == "pass"
                                 for c in cases),
    }
    return {
        "benchmark": "contact_angle_geometry",
        "reference": "analytic 2-D spherical cap (Young, 1805; Lamb 1932 art. 1)",
        "sigma": SIGMA, "rho0": RHO0, "R_arc": R_ARC,
        "tolerances": {"angle_deg": TOL_ANGLE_DEG, "curvature_rel": TOL_CURVATURE},
        "controls": {"cap_geometry": cap_geometry_check(),
                     "bc_exactness": control_exactness(),
                     "wall_normal_fallback": control_wall_normal_fallback()},
        "cases": cases,
        "summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    # Batched 2x2 ``inv`` (used by ``mls_divergence``) costs 2.7 s per call at
    # 128 threads versus 3.5 ms for the equivalent ``solve``; pinning one thread
    # keeps this static operator benchmark reproducible and fast.  See the
    # "findings" note in docs/verification_contact_angle_geometry.md.
    torch.set_num_threads(int(os.environ.get("TSPH_THREADS", "1")))
    res = run_all()
    out = HERE / "results.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    s = res["summary"]
    print(json.dumps({"summary": s,
                      "controls": {k: v.get("result") for k, v in res["controls"].items()}},
                     indent=2))
    print(f"wrote {out}")
    if args.plot:
        import plot
        plot.make(res, HERE / "contact_angle_geometry.png")
        print(f"wrote {HERE / 'contact_angle_geometry.png'}")


if __name__ == "__main__":
    main()