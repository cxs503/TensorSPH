"""Young-Laplace verification of the CSF surface tension model (case B5).

What is checked, and why these are the right observables
--------------------------------------------------------
Young-Laplace for a 2-D droplet of radius R is ``p_in - p_out = sigma / R``.
The equivalent statement in terms of forces, which can be tested without
running a time integration, is the **capillary force balance**: the net
surface-tension force on either half of a circular droplet must be

    |F_half| = 2 * sigma            (2-D, force per unit depth)

independently of R, of the sound speed and of the resolution. Cut the droplet
along any diameter and the free surface of the half consists of a semicircular
arc; integrating ``sigma * kappa * n`` along it gives exactly ``2 sigma``
directed across the cut. Two consequences make this the cleanest available
analytic gate for a CSF implementation:

* the target does not depend on R, so a *correct* implementation gives the same
  ratio at every radius -- any spread is discretisation error, not physics;
* it is a *global* integral of the traction, so a locally noisy curvature that
  averages out would be invisible here (this gate is therefore necessary but
  not sufficient).

Alongside it the harness measures two observables whose analytic value is
exactly zero for this geometry, which no amount of calibration can fudge:

* ``F_total``: the traction on a *closed* surface must sum to zero, i.e. the
  model must not create net momentum out of nothing;
* mirror asymmetry ``|F_half,y| / |F_half,x|``: the droplet and the lattice are
  both symmetric about y = 0 and x = 0, so the force on the right half must be
  purely radial (and must equal the force on the top half rotated by 90 deg).

Plus ``bulk traction``: inside the droplet ``grad c`` must vanish, otherwise the
model is pushing fluid around in the interior where there is no surface.

How to read the output
----------------------
``sigma_eff / sigma = |F_half| / (2 sigma)`` is the effective surface tension
the model actually applies. It is 1 for a correct implementation.

Usage
-----
    python benchmarks/laplace_pressure/run.py --json-out out.json
    python benchmarks/laplace_pressure/run.py --quick     # CI-sized

``--quick`` uses one radius at one resolution and only checks the structural
gates (symmetry, bulk, momentum); the quantitative balance needs a refinement
ladder and is reported ``not_evaluated`` there.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    compute_color_gradient,
    compute_curvature,
    compute_normal,
    compute_surface_tension_force,
)

# Accuracy standard of the platform: a physical quantity that misses its
# analytic reference by more than 15 % is not acceptable.
FORCE_TOLERANCE = 0.15
SYMMETRY_TOLERANCE = 0.05


def circular_droplet(radius: float, dx: float, rho0: float = 1000.0):
    """Particles on a square lattice clipped to a circle centred at the origin.

    The lattice is centred so the particle set is symmetric under ``x -> -x``
    and ``y -> -y``; any asymmetry in the measured force is then attributable to
    the surface tension model and not to the sampling.
    """
    count = int(round(2.0 * radius / dx))
    axis = (torch.arange(count, dtype=torch.float64) + 0.5) * dx - radius
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    pos = pos[(pos.square().sum(dim=1) <= radius * radius)]
    n = pos.shape[0]
    mass = torch.full((n,), rho0 * dx * dx, dtype=torch.float64)
    density = torch.full((n,), rho0, dtype=torch.float64)
    return pos, mass, density


def legacy_color_field(pos, mass, density, pairs, cfg):
    """Surface-tension acceleration exactly as shipped in 0.8.0 (m277).

    Reproduced here -- and nowhere else -- so that the comparison against the
    current formulation is reproducible from a clean checkout. It is a frozen
    copy, not an alternative implementation: the neighbour-count surface
    detection, the ``c = neighbours / mean(neighbours)`` colour field and the
    unnormalised ``delta_s`` are all as they were.
    """
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient
    i, j = pairs.unbind(dim=1)
    counts = torch.zeros(n, dtype=pos.dtype)
    counts = counts.index_add(0, i, torch.ones_like(i, dtype=pos.dtype))
    counts = counts.index_add(0, j, torch.ones_like(j, dtype=pos.dtype))
    avg = counts.mean()
    surface = counts < 0.7 * avg
    color = torch.ones(n, dtype=pos.dtype)
    color[surface] = counts[surface] / avg
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    normal = compute_normal(-grad_c)
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    force_mag = -sigma * kappa * delta_s
    return force_mag[:, None] * normal / density[:, None]


def measure(pos, mass, density, acc, radius):
    """Integrate the traction and its symmetry residuals.

    All sums are ``sum_i m_i a_i`` over a subset, i.e. the force on that subset
    per unit depth, which is what the analytic target ``2 sigma`` refers to.
    """
    force = mass[:, None] * acc
    right = pos[:, 0] > 0.0
    top = pos[:, 1] > 0.0
    f_right = force[right].sum(dim=0)
    f_top = force[top].sum(dim=0)
    f_total = force.sum(dim=0)

    # Radial component sign check: on the surface, a convex droplet must be
    # pulled towards the centre, so m * a . r_hat must be negative.
    radial = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True).clamp_min(1e-30)
    moving = torch.linalg.vector_norm(acc, dim=1) > 0.0
    radial_dot = (acc * radial).sum(dim=1)[moving]

    # Bulk traction: no surface there, so the acceleration must be at the level
    # of floating-point noise relative to its surface scale.
    bulk = torch.linalg.vector_norm(pos, dim=1) < 0.5 * radius
    acc_scale = float(torch.linalg.vector_norm(acc, dim=1).max())
    bulk_ratio = (float(torch.linalg.vector_norm(acc[bulk], dim=1).max()) / acc_scale
                  if bulk.any() and acc_scale > 0 else 0.0)

    return {
        "force_right_half": [float(v) for v in f_right],
        "force_top_half": [float(v) for v in f_top],
        "force_total": [float(v) for v in f_total],
        "mirror_asymmetry": (abs(float(f_right[1])) / abs(float(f_right[0]))
                             if float(f_right[0]) != 0.0 else float("inf")),
        "rotation_asymmetry": (abs(float(f_top[0])) / abs(float(f_top[1]))
                               if float(f_top[1]) != 0.0 else float("inf")),
        "bulk_traction_ratio": bulk_ratio,
        "inward_fraction": (float((radial_dot < 0).to(torch.float64).mean())
                            if radial_dot.numel() else 0.0),
    }


def run_case(radius: float, dx: float, sigma: float, scheme: str):
    cfg = Config(dx=dx, rho0=1000.0, sound_speed=20.0, gravity=0.0,
                 surface_tension_coefficient=sigma)
    pos, mass, density = circular_droplet(radius, dx)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx)
    if scheme == "legacy":
        acc = legacy_color_field(pos, mass, density, pairs, cfg)
    else:
        acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)

    h = cfg.h_factor * cfg.dx
    measured = measure(pos, mass, density, acc, radius)
    f = math.hypot(*measured["force_right_half"])
    return {
        "radius_m": radius,
        "dx_m": dx,
        "h_over_dx": cfg.h_factor,
        "particles": int(pos.shape[0]),
        "radius_over_h": radius / h,
        "target_force_2sigma": 2.0 * sigma,
        "measured_force_half": f,
        "sigma_eff_over_sigma": f / (2.0 * sigma),
        **measured,
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Observed order of accuracy from two levels, |e| ~ C dx^p."""
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", default="0.04,0.06,0.08",
                        help="droplet radii in metres, comma separated")
    parser.add_argument("--sizes", default="0.005,0.0025,0.00125",
                        help="particle spacings in metres, comma separated")
    parser.add_argument("--sigma", type=float, default=0.072,
                        help="surface tension coefficient, N/m (water/air)")
    parser.add_argument("--scheme", default="csf", choices=("csf", "legacy"),
                        help="csf = current formulation, legacy = 0.8.0 (m277)")
    parser.add_argument("--quick", action="store_true",
                        help="CI-sized: one radius, one resolution")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--strict", action="store_true",
                        help="also fail the exit code on the quantitative "
                             "gates. They are documented failures of the "
                             "current model (see docs), so CI runs the "
                             "structural gates and records the rest.")
    args = parser.parse_args(argv)

    if args.quick:
        radii = [0.04]
        sizes = [0.005]
    else:
        radii = [float(v) for v in args.radii.split(",")]
        sizes = [float(v) for v in args.sizes.split(",")]

    started = time.time()
    results = []
    for dx in sizes:
        for radius in radii:
            results.append(run_case(radius, dx, args.sigma, args.scheme))

    # ---- gates -----------------------------------------------------------
    def gate(metric, measured, limit, comparison, note=""):
        ok = (measured <= limit) if comparison == "<=" else (measured >= limit)
        return {"metric": metric, "measured": measured, "limit": limit,
                "comparison": comparison, "result": "pass" if ok else "fail",
                "note": note}

    gates = {}
    worst_mirror = max(r["mirror_asymmetry"] for r in results)
    worst_rot = max(r["rotation_asymmetry"] for r in results)
    worst_bulk = max(r["bulk_traction_ratio"] for r in results)
    worst_inward = min(r["inward_fraction"] for r in results)

    gates["mirror_symmetry"] = gate(
        "|F_y| / |F_x| of the right half (exact: 0)", worst_mirror,
        SYMMETRY_TOLERANCE, "<=",
        "the droplet and the lattice are mirror-symmetric, so any transverse "
        "force is manufactured by the model")

    gates["rotation_symmetry"] = gate(
        "|F_x| / |F_y| of the top half (exact: 0)", worst_rot,
        SYMMETRY_TOLERANCE, "<=",
        "90-degree rotation must map the right-half result onto the top half")

    total_force = max(math.hypot(*r["force_total"]) for r in results)
    gates["closed_surface_momentum"] = gate(
        "|F_total| / (2 sigma) (exact: 0)", total_force / (2.0 * args.sigma),
        SYMMETRY_TOLERANCE, "<=",
        "a closed surface carries no net capillary force; a non-zero sum means "
        "the scheme creates momentum")

    gates["bulk_traction_free"] = gate(
        "max|a| (r < R/2) / max|a| overall (exact: 0)", worst_bulk,
        1.0e-3, "<=",
        "grad c vanishes exactly for a complete stencil, so the interior must "
        "not be pushed")

    gates["force_points_inward"] = gate(
        "fraction of moving particles with a . r_hat < 0", worst_inward,
        1.0, ">=",
        "a convex droplet is pulled towards its centre")

    worst_error = max(abs(r["sigma_eff_over_sigma"] - 1.0) for r in results)

    if len(results) > len(radii) and len(radii) >= 1:
        # group by radius to build a per-radius refinement ladder
        ladders = {}
        for r in results:
            ladders.setdefault(r["radius_m"], []).append(r)
        orders = {}
        for radius, ladder in ladders.items():
            ladder = sorted(ladder, key=lambda item: -item["dx_m"])
            errs = [abs(item["sigma_eff_over_sigma"] - 1.0) for item in ladder]
            if len(errs) >= 2:
                orders[f"{radius:g}"] = {
                    "errors": errs,
                    "dx": [item["dx_m"] for item in ladder],
                    "observed_order": observed_order(
                        errs[0], errs[-1],
                        ratio=(ladder[0]["dx_m"] / ladder[-1]["dx_m"])
                        if len(ladder) == 2 else 2.0),
                    "worst_error": max(errs),
                }
        gates["capillary_force_balance"] = gate(
            "max |sigma_eff/sigma - 1| over the ladder",
            worst_error, FORCE_TOLERANCE, "<=",
            "|F_half| must equal 2 sigma; this is Young-Laplace expressed as a "
            "force balance. KNOWN FAILURE of the current model: not waived, "
            "recorded. See docs/benchmarks/surface-tension-verification.md")
        gates["refinement_convergence"] = gate(
            "worst error must not grow under refinement",
            max((v["observed_order"] for v in orders.values()), default=0.0),
            0.0, ">=",
            "a converging scheme needs a positive observed order; a negative "
            "one means the error is generated by the discretisation itself")
    else:
        orders = {}
        gates["capillary_force_balance"] = {
            "metric": "max |sigma_eff/sigma - 1|", "measured": None,
            "limit": FORCE_TOLERANCE, "result": "not_evaluated",
            "note": "needs a refinement ladder; --quick measures structure only"}
        gates["refinement_convergence"] = {
            "metric": "observed order", "measured": None, "limit": None,
            "result": "not_evaluated", "note": "--quick"}

    report = {
        "benchmark": "B5_young_laplace",
        "analytic_reference": {
            "laplace_pressure_2d": "p_in - p_out = sigma / R",
            "capillary_force_half_droplet_2d": "|F_half| = 2 sigma",
            "independence": "the 2 sigma target does not depend on R, dx or c0",
        },
        "scheme": args.scheme,
        "sigma_N_per_m": args.sigma,
        "force_tolerance": FORCE_TOLERANCE,
        "cases": results,
        "refinement": orders,
        "gates": gates,
        "notes": [
            "Static force integration: one evaluation of the surface tension "
            "acceleration field on an equilibrated (undeformed) circular "
            "droplet, then the traction is summed. No time integration is "
            "needed because the analytic target is a force balance.",
            "The dynamic counterpart (integrate until the EOS pressure inside "
            "the droplet reaches sigma/R) measures the same relation and would "
            "need of order 1/t_cap * 2 pi / dt steps, which is why the force "
            "balance form is the gate used in CI. See the docs for the "
            "equivalence.",
            "sigma_eff/sigma is the effective surface tension the model "
            "applies; a correct implementation gives 1 at every radius and "
            "every resolution.",
        ],
    }

    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
    print(text)

    # Structural gates are exact to round-off for any correct implementation:
    # they are the ones CI must enforce. The rest measure *how far* the amplitude
    # is from the analytic reference and are recorded, not waived.
    structural = ("mirror_symmetry", "rotation_symmetry", "closed_surface_momentum",
                  "bulk_traction_free")
    quantitative = ("capillary_force_balance", "refinement_convergence",
                    "force_points_inward")
    failed = [name for name, item in gates.items() if item["result"] == "fail"]
    structural_failed = [n for n in failed if n in structural]
    quantitative_failed = [n for n in failed if n in quantitative]

    elapsed = time.time() - started
    print(f"\n# {len(results)} cases in {elapsed:.1f}s; "
          f"{len(gates) - len(failed)}/{len(gates)} gates pass", file=sys.stderr)
    if quantitative_failed:
        print("# quantitative gate FAIL (recorded, not waived): "
              f"{', '.join(quantitative_failed)}", file=sys.stderr)
    if structural_failed:
        raise SystemExit(f"ERROR: structural gates failed: {', '.join(structural_failed)}")
    if args.strict and quantitative_failed:
        raise SystemExit(f"ERROR: Young-Laplace gates failed: {', '.join(quantitative_failed)}")


if __name__ == "__main__":
    main()
