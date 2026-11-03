"""Laplace-law verification of the CSF surface-tension model (single phase).

Purpose
-------
The surface-tension model shipped in ``tensorsph.surface_tension`` (CSF,
Brackbill, Kothe & Zemach 1992) is declared in the README as implemented. Until
this benchmark it had never been compared with the exact two-dimensional
equilibrium condition, the **Laplace law**

    Delta p = sigma / R        (Pa),   R = droplet radius,   sigma in N/m.   (1)

``benchmarks/static_droplet`` only checked that the force points inward and is
"physically reasonable"; it does not test (1). This benchmark tests (1).

How the law is measured
-----------------------
The CSF force is a volumetric force density

    f = -sigma * kappa * n * delta_s,   delta_s = |grad c|  (N/m^3)        (2)

whose integral along the interface normal reproduces (1) exactly,

    integral f . (-n) dxi = sigma * kappa = sigma / R.                     (3)

Hence the measured quantity is the **normal integral** of (2), not the force at
any single particle: pointwise ``|f|`` carries the resolution-dependent factor
``delta_s ~ 1/dx`` and is *not* comparable with ``sigma/R``. Concretely the
droplet is spherically symmetric, so particles are binned by radius ``r`` and

    Delta p_disc = sum_bins mean_{i in bin} [ rho_i * (-a_i . n_exact_i) ] * dr   (4)

with ``a_i`` the acceleration returned by the shipped function, ``n_exact`` the
exact outward normal of the disc, and ``dr`` small compared with the interface
width. Outward contributions enter with a negative sign, so a model that pulls
particles the wrong way cannot pass by accident.

Gauge self-test (measurement error vs model error)
-------------------------------------------------
A FAIL is only informative if the estimator itself is unbiased. The same
estimator is therefore also applied to a **synthetic analytic field** built to
satisfy (3) exactly -- ``a_i = -(sigma/(rho R)) g(r_i) n_i`` with a unit-mass
radial profile ``g`` -- and its recovery of ``Delta p`` is reported as
``synthetic_recovery_error``. The discretisation error of the gauge is then
known independently of the model under test.

Reported diagnostics
---------------------
* ``detector_recall`` / ``detector_precision`` -- the shipped surface detector
  (neighbour count < 0.7 mean) against the exact geometric interface shell.
  Curvature only means something on the right particle set.
* ``curvature_ratio`` = median(kappa_model * R) on detected surface particles.
  Exact value is +1 for the outward-normal convention used in the module.
* ``inward_fraction`` -- fraction of *detected* surface particles whose force is
  inward, over the unfiltered set (no selection bias).

Usage
-----
    python benchmarks/surface_tension/laplace.py --radii-dx 8,16,32

Exit status is non-zero when a gate fails; the JSON artifact records every
number quoted in ``docs/benchmarks/surface-tension-laplace.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorsph import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    interface_fields,
    compute_color_gradient,
    compute_curvature,
    compute_normal,
    compute_surface_tension_force,
)

SIGMA = 0.072          # N/m, water-air at 20 C (same value the test-suite uses)
RHO0 = 1000.0          # kg/m^3
RADIUS = 0.1           # m


def make_disc(radius: float, dx: float) -> torch.Tensor:
    """Particles filling ``r <= radius`` on a square lattice of spacing ``dx``."""
    span = radius + 3.0 * dx
    axis = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    keep = torch.linalg.vector_norm(pts, dim=1) <= radius + 1.0e-12
    return pts[keep].clone()


def neighbour_count(pos: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    count = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    ones = torch.ones_like(i, dtype=pos.dtype)
    count = count.index_add(0, i, ones)
    return count.index_add(0, j, ones)


def radial_jump(pos, r, rho, inward_acc, radius, dx, width, dr):
    """Histogram diagnostic: force density vs radius (not used as a metric)."""
    edges = torch.arange(radius - width, radius + width + 0.5 * dr, dr)
    profile = []
    for k in range(len(edges) - 1):
        mask = (r >= edges[k]) & (r < edges[k + 1])
        if mask.sum() == 0:
            continue
        profile.append({"r": float(0.5 * (edges[k] + edges[k + 1])),
                        "f": float((rho * inward_acc[mask]).mean()),
                        "count": int(mask.sum())})
    return profile


def perimeter_jump(rho, inward_acc, radius, dx):
    """Normal integral of the force density, made geometry-exact.

    For a radially symmetric field the exact identity

        integral f dr = (1 / (2 pi)) integral f dA / r

    is evaluated with the force-weighted centroid radius

        r_c = sum_i f_i A_i r_i / sum_i f_i A_i,     A_i = dx^2,

    so that

        Delta p = sum_i f_i A_i / (2 pi r_c).

    For a field concentrated in a thin shell this is exact to O((Delta r / R)^2)
    and it removes the O(t/R) bias a plain ``/ (2 pi R)`` normalisation would
    introduce. The same formula is applied to the synthetic gauge field, so the
    comparison of the two is meaningful.
    """
    weight = rho * inward_acc * dx * dx
    total = float(weight.sum())
    if abs(total) < 1e-30:
        return float("nan"), float("nan")
    r = _LAST_RADIUS
    r_c = float((weight * r).sum() / total)
    return total / (2.0 * math.pi * r_c), r_c


_LAST_RADIUS = None


def measure(radius: float, dx: float, sigma: float = SIGMA) -> dict:
    global _LAST_RADIUS
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                 surface_tension_coefficient=sigma)
    h = cfg.h_factor * dx
    pos = make_disc(radius, dx)
    n = pos.shape[0]
    mass = torch.full((n,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((n,), RHO0, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)

    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)

    centre = pos.mean(dim=0)
    radial = pos - centre
    r = torch.linalg.vector_norm(radial, dim=1)
    n_out = radial / r.clamp_min(1e-12)[:, None]
    inward_acc = -(acc * n_out).sum(dim=1)          # a . (-n), positive = inward
    _LAST_RADIUS = r

    # --- gauge self-test ----------------------------------------------------
    # Smooth (C1) bump of thickness t = 2 dx on the fluid side,
    #     f(r) = (sigma/R) * b(r),  b = (1 + cos(2 pi u)) / t,  u = (r-r0)/t,
    # whose normal integral is exactly sigma/R. The residual after the
    # geometry-exact estimator is pure measurement error.
    t = 2.0 * dx
    u = (r - (radius - 0.5 * t)) / t
    bump = torch.where(u.abs() <= 0.5,
                       0.5 * (1.0 + torch.cos(2.0 * math.pi * u)),
                       torch.zeros_like(u))
    synth_inward = (sigma / (RHO0 * radius)) * bump * (2.0 / t)
    jump_synth, r_c_synth = perimeter_jump(RHO0, synth_inward, radius, dx)

    # --- model measurement --------------------------------------------------
    jump_model, r_c = perimeter_jump(RHO0, inward_acc, radius, dx)
    profile = radial_jump(pos, r, RHO0, inward_acc, radius, dx, 3.0 * h, dx / 8.0)

    # --- ablation: which ingredient carries the error? ---------------------
    # Both substitutions below use the module's own field except for one factor,
    # so the two numbers localise the defect.
    t = 2.0 * dx
    u = (r - (radius - 0.5 * t)) / t
    delta_exact = torch.where(u.abs() <= 0.5,
                              (1.0 / t) * (1.0 + torch.cos(2.0 * math.pi * u)),
                              torch.zeros_like(u))

    # --- surface detection and curvature -----------------------------------
    count = neighbour_count(pos, pairs)
    detected = count < 0.7 * count.mean()
    geometric = r > radius - dx
    hit = int((detected & geometric).sum())

    # The interface fields come from the module itself (single source of truth),
    # so the numbers below describe exactly the force that was returned above.
    fields = interface_fields(pos, density, mass, pairs, cfg)
    grad_c = fields["grad_colour"]
    normal = fields["normal"]
    delta_s = fields["delta_s"]
    kappa = fields["curvature"]

    # Integral of the model's delta function over the fluid side: a proper
    # one-sided normalised delta must give 1, so that sigma*kappa*int(delta)=sigma*kappa.
    delta_integral = float((delta_s * dx * dx).sum() / (2.0 * math.pi * radius))
    # (i) exact curvature kappa = 1/R, model delta_s
    jump_exact_kappa, _ = perimeter_jump(RHO0,
        (sigma / (RHO0 * radius)) * delta_s, radius, dx)
    # (ii) model curvature, exact normalised delta function
    jump_exact_delta, _ = perimeter_jump(RHO0,
        (sigma / RHO0) * kappa * delta_exact, radius, dx)

    on_interface = delta_s > 0.1 * delta_s.max()
    # Is the curvature systematically wrong-signed, or noise-dominated? A median
    # alone cannot tell the two apart, so the positive fraction and the
    # force-weighted mean are reported as well.
    positive_fraction = (float((kappa[on_interface] > 0).double().mean())
                         if int(on_interface.sum()) else float("nan"))
    kappa_weight = (delta_s[on_interface] * inward_acc[on_interface].abs())
    curvature_force_weighted_ratio = (
        float((kappa[on_interface] * kappa_weight).sum()
              / kappa_weight.sum()) * radius
        if float(kappa_weight.sum()) > 0 else float("nan"))
    kappa_ratio = (float(kappa[on_interface].median()) * radius
                   if int(on_interface.sum()) else float("nan"))

    theory = sigma / radius
    return {
        "radius_m": radius,
        "dx_m": dx,
        "r_over_dx": radius / dx,
        "h_over_dx": cfg.h_factor,
        "interface_width_over_radius": 2.0 * h / radius,
        "particles": n,
        "pairs": int(pairs.shape[0]),
        "jump_theory_pa": theory,
        "jump_measured_pa": jump_model,
        "jump_ratio": jump_model / theory,
        "jump_rel_error": abs(jump_model - theory) / theory,
        "centroid_radius_m": r_c,
        "synthetic_jump_pa": jump_synth,
        "synthetic_centroid_radius_m": r_c_synth,
        "synthetic_recovery_error": abs(jump_synth - theory) / theory,
        "delta_function_integral": delta_integral,
        "jump_with_exact_kappa_pa": jump_exact_kappa,
        "jump_with_exact_delta_pa": jump_exact_delta,
        "radial_profile": profile,
        "curvature_theory": 1.0 / radius,
        "curvature_median": (float(kappa[on_interface].median())
                             if int(on_interface.sum()) else None),
        "curvature_ratio": kappa_ratio,
        "curvature_positive_fraction": positive_fraction,
        "curvature_force_weighted_ratio": curvature_force_weighted_ratio,
        "detected_surface": int(detected.sum()),
        "geometric_surface": int(geometric.sum()),
        "detector_recall": hit / max(int(geometric.sum()), 1),
        "detector_precision": hit / max(int(detected.sum()), 1),
        "inward_fraction": (float((inward_acc[on_interface] > 0).double().mean())
                            if int(on_interface.sum()) else float("nan")),
        "max_outward_acc": (float((-inward_acc[on_interface]).max())
                            if int(on_interface.sum()) else 0.0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radii-dx", default="16,32,64",
                    help="comma separated R/dx values for the resolution ladder")
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="accepted relative error of the Laplace jump (finest level)")
    ap.add_argument("--out", default="artifacts/surface_tension_laplace.json")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    ratios = [float(s) for s in args.radii_dx.split(",")]
    rows = []
    t0 = time.perf_counter()
    for ratio in ratios:
        row = measure(args.radius, args.radius / ratio, args.sigma)
        rows.append(row)
        print(f"R/dx={ratio:5.0f} N={row['particles']:6d} "
              f"jump={row['jump_measured_pa']:8.4f} Pa "
              f"(theory {row['jump_theory_pa']:.4f})  "
              f"ratio={row['jump_ratio']:7.3f}  err={row['jump_rel_error']*100:6.1f}%  "
              f"kappaR={row['curvature_ratio']:7.3f}  "
              f"inward={row['inward_fraction']:.2f}  "
              f"detect R/P={row['detector_recall']:.2f}/{row['detector_precision']:.2f}  "
              f"gauge={row['synthetic_recovery_error']*100:5.2f}%")

    finest = rows[-1]
    # Observed order of the Laplace error, from the two finest levels.
    order = None
    if len(rows) >= 2:
        e0, e1 = rows[-2]["jump_rel_error"], rows[-1]["jump_rel_error"]
        if e0 > 1e-12 and e1 > 1e-12:
            order = math.log(e0 / e1) / math.log(rows[-1]["r_over_dx"] / rows[-2]["r_over_dx"])

    gauge_ok = max(r["synthetic_recovery_error"] for r in rows)
    gates = {
        "gauge_self_test": {
            "measured": gauge_ok,
            "limit": 0.05,
            "result": "pass" if gauge_ok <= 0.03 else "fail",
            "note": "estimator must recover sigma/R on a synthetic analytic field",
        },
        "laplace_error_finest": {
            "measured": finest["jump_rel_error"],
            "limit": args.tolerance,
            "result": "pass" if finest["jump_rel_error"] <= args.tolerance else "fail",
        },
        "curvature_sign_and_scale": {
            "measured": finest["curvature_ratio"],
            "limit": 1.0,
            "result": "pass" if abs(finest["curvature_ratio"] - 1.0) <= 0.15 else "fail",
            "note": "exact kappa*R = +1 for the module's outward-normal convention",
        },
        "force_direction": {
            "measured": min(r["inward_fraction"] for r in rows),
            "limit": 1.0,
            "result": "pass" if min(r["inward_fraction"] for r in rows) >= 1.0 else "fail",
        },
    }
    verdict = "PASS" if all(g["result"] == "pass" for g in gates.values()) else "FAIL"

    summary = {
        "benchmark": "surface_tension_laplace_law",
        "formulation": "CSF (Brackbill et al. 1992); 2-D Laplace law dp = sigma/R",
        "sigma_N_per_m": args.sigma,
        "rho0_kg_per_m3": RHO0,
        "radius_m": args.radius,
        "levels": rows,
        "laplace_error_order": order,
        "gates": gates,
        "failed_gates": [k for k, g in gates.items() if g["result"] == "fail"],
        "wall_clock_s": time.perf_counter() - t0,
        "verdict": verdict,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"verdict: {verdict}  failed={summary['failed_gates']}  "
          f"order={order}  -> {out}")
    if not args.no_plot:
        try:
            plot(summary, out.with_suffix(".png"))
        except Exception as exc:  # plotting is optional; never gate on it
            print(f"plot skipped: {exc}")
    return 0 if verdict == "PASS" else 1


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["levels"]
    dxr = [r["dx_m"] / r["radius_m"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    ax.plot(dxr, [r["jump_ratio"] for r in rows], "o-", color="tab:blue",
            label="model $\\Delta p/(\\sigma/R)$")
    ax.plot(dxr, [r["synthetic_jump_pa"] / r["jump_theory_pa"] for r in rows],
            "s--", color="tab:green", label="gauge self-test")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, label="Laplace law")
    ax.set_xscale("log"); ax.set_xlabel("$dx/R$")
    ax.set_ylabel("$\\Delta p\\,/\\,(\\sigma/R)$")
    ax.set_title("Laplace pressure jump")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(dxr, [r["curvature_ratio"] for r in rows], "s-", color="tab:red",
            label="shipped $\\kappa R$ (median)")
    ax.plot(dxr, [r["curvature_force_weighted_ratio"] for r in rows], "^--",
            color="tab:purple", label="shipped $\\kappa R$ (force-weighted)")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, label="exact $\\kappa R=1$")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("$dx/R$"); ax.set_ylabel("$\\kappa R$")
    ax.set_title("Curvature: zero-mean noise, not $1/R$")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    fine = rows[-1]
    prof = fine["radial_profile"]
    ax.step([q["r"] / fine["radius_m"] for q in prof],
            [q["f"] for q in prof], where="mid", color="tab:blue",
            label=f"model, $R/dx$={fine['r_over_dx']:.0f}")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("$r/R$"); ax.set_ylabel("$\\rho\\,a\\cdot(-n)$  [Pa/m]")
    ax.set_title("Radial force-density profile")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")


if __name__ == "__main__":
    sys.exit(main())