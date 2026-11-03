"""Quantitative Laplace-law verification of the CSF surface-tension chain.

The model in ``tensorsph.surface_tension`` builds a colour gradient from the
particle field, takes its normal, and forms the CSF force

    f = -sigma * kappa * n * |grad(c)|          (Brackbill, Kothe & Zemach 1992)

For a static circular interface of radius ``R`` the model must reproduce the
Laplace pressure jump ``dp = sigma / R``.  For a static interface the CSF body
force balances the pressure gradient, so the jump is the radial line integral of
the volume force:

    dp = int_0^inf sigma * kappa * |grad(c)| dr = sigma * kappa * d(c)

where ``d(c) = int |grad(c)| dr`` is the colour jump actually resolved by the
discretisation.  Two clouds are used:

* **two-phase** (colour 1 inside, 0 outside, equal density): the full jump is
  resolved, ``d(c) -> 1``, so ``dp -> sigma / R`` is a real benchmark.
* **one-sided free surface** (the shipped droplet configuration, colour from
  neighbour counts): there are no particles outside the interface, ``d(c)`` stays
  well below 1, and the Laplace law cannot be closed.  Reported as a documented
  limitation, not a pass/fail gate.

Run::

    python benchmarks/surface_tension_laplace/run.py [--plot]

Writes ``benchmarks/surface_tension_laplace/validation_report.json`` and, with
``--plot``, ``laplace_convergence.png`` next to it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import cubic_spline_gradient, cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    compute_color_gradient,
    compute_curvature,
    compute_normal,
)

RADIUS = 0.1
SIGMA = 0.072
RHO0 = 1000.0
DX_LIST = (0.01, 0.005, 0.0025)
H_FACTOR = 1.3


def build_disc(radius: float, dx: float, shell: float = 0.0):
    """Square lattice of particles within ``radius + shell`` of the origin."""
    extent = 2.0 * (radius + shell) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n, dtype=torch.float64) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    keep = r <= radius + shell
    pos = pos[keep]
    count = pos.shape[0]
    mass = torch.full((count,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((count,), RHO0, dtype=torch.float64)
    return pos, mass, density


def legacy_color_gradient(pos, color, mass, density, pairs, h):
    """Pre-fix form kept only to reproduce the defect this change removes.

    The ``j`` side of each pair was updated with ``-V_j`` instead of ``+V_i``.
    For uniform particle volumes the two updates then cancelled almost exactly,
    so the returned "gradient" was antisymmetric residue: normals came out
    tangential and curvature became noise.  Do not use outside this comparison.
    """
    grad_c = torch.zeros((pos.shape[0], 2), dtype=pos.dtype)
    i, j = pairs.unbind(dim=1)
    grad_w = cubic_spline_gradient(pos[i] - pos[j], h)
    dc = color[j] - color[i]
    vol_j = mass[j] / density[j]
    grad_c = grad_c.index_add(0, i, vol_j[:, None] * dc[:, None] * grad_w)
    grad_c = grad_c.index_add(0, j, -vol_j[:, None] * dc[:, None] * grad_w)
    return grad_c


def laplace_jump(pos, mass, density, color, dx, h, sigma, gradient_fn=None):
    """Radial line integral of the CSF volume force (see module docstring)."""
    pairs = cell_linked_pairs(pos, 2.0 * h)
    gradient = gradient_fn or compute_color_gradient
    grad_c = gradient(pos, color, mass, density, pairs, h)
    normal = compute_normal(-grad_c)
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)
    d_s = grad_c.norm(dim=1)
    r = pos.norm(dim=1)
    r_hat = pos / r.clamp_min(1.0e-12)[:, None]
    q = sigma * kappa * d_s * (normal * r_hat).sum(dim=1)

    n_bins = int(math.ceil((r.max().item() + dx) / dx))
    idx = torch.clamp((r / dx).long(), 0, n_bins - 1)
    counts = torch.zeros(n_bins, dtype=q.dtype).index_add(0, idx, torch.ones_like(q))
    q_sum = torch.zeros(n_bins, dtype=q.dtype).index_add(0, idx, q)
    ds_sum = torch.zeros(n_bins, dtype=q.dtype).index_add(0, idx, d_s)
    safe = counts.clamp_min(1.0)

    shell = (r > RADIUS - 1.5 * h) & (r < RADIUS + 1.5 * h)
    return {
        "n_particles": int(pos.shape[0]),
        "dp_Pa": float(((q_sum / safe) * dx).sum()),
        "resolved_color_jump": float(((ds_sum / safe) * dx).sum()),
        "normal_alignment_mean": float((normal * r_hat).sum(dim=1)[shell].mean()),
        "curvature_shell_mean_per_m": float(kappa[shell].mean()),
        "curvature_shell_std_per_m": float(kappa[shell].std()),
        "normal_alignment_min": float((normal * r_hat).sum(dim=1)[shell].min()),
    }


def smooth_color(pos, mass, density, color, pairs, h):
    """Kernel-smoothed colour: c_i = sum_j V_j c_j W_ij / sum_j V_j W_ij.

    A raw lattice step (``c`` jumping by 1 between neighbouring particles) puts
    the whole jump inside one smoothing length, and its second derivative --
    the curvature -- is then noise of order ``1/h``.  Smoothing the colour with
    the same kernel is the standard remedy.  Measured effect at dx = 0.0025:
    curvature spread 115 -> 25 per m, Laplace error 29 % -> 4.6 %.
    """
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    vol = mass / density
    zero = torch.zeros(pos.shape[0], dtype=pos.dtype)
    num = zero.index_add(0, i, w * vol[j] * color[j]) + zero.index_add(
        0, j, w * vol[i] * color[i]
    )
    den = zero.index_add(0, i, w * vol[j]) + zero.index_add(0, j, w * vol[i])
    return num / den.clamp_min(1.0e-30)


def two_phase_case(dx: float, outer: float = 0.04, smoothed: bool = True):
    pos, mass, density = build_disc(RADIUS, dx, shell=outer)
    r = pos.norm(dim=1)
    color = (r < RADIUS).to(torch.float64)
    if smoothed:
        pairs = cell_linked_pairs(pos, 2.0 * H_FACTOR * dx)
        color = smooth_color(pos, mass, density, color, pairs, H_FACTOR * dx)
    return pos, mass, density, color, r


def one_sided_case(dx: float):
    """Shipped free-surface colour field: neighbour count deficit."""
    pos, mass, density = build_disc(RADIUS, dx)
    h = H_FACTOR * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    count = torch.zeros(pos.shape[0], dtype=torch.float64)
    i, j = pairs.unbind(dim=1)
    count = count.index_add(0, i, torch.ones_like(i, dtype=torch.float64))
    count = count.index_add(0, j, torch.ones_like(j, dtype=torch.float64))
    mean_count = count.mean()
    surface = count < 0.7 * mean_count
    color = torch.ones(pos.shape[0], dtype=torch.float64)
    color[surface] = count[surface] / mean_count
    return pos, mass, density, color


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="write laplace_convergence.png")
    parser.add_argument(
        "--out",
        default="benchmarks/surface_tension_laplace",
        help="output directory",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    theory = SIGMA / RADIUS

    print("=" * 78)
    print("Laplace law verification: dp = sigma / R")
    print(f"  sigma = {SIGMA} N/m, R = {RADIUS} m, rho0 = {RHO0} kg/m^3")
    print(f"  theory: dp = {theory:.6f} Pa")
    print("=" * 78)
    print(f"{'dx [m]':>8} {'N':>7} {'dp [Pa]':>9} {'err [%]':>8} {'d(c)':>7} {'n.r':>6} "
          f"{'kappa [1/m]':>16}")

    smoothed_sweep = []
    for dx in DX_LIST:
        h = H_FACTOR * dx
        pos, mass, density, color, _ = two_phase_case(dx, smoothed=True)
        res = laplace_jump(pos, mass, density, color, dx, h, SIGMA)
        res["dx_m"] = dx
        res["rel_error"] = abs(res["dp_Pa"] - theory) / theory
        smoothed_sweep.append(res)
        print(f"{dx:8.4f} {res['n_particles']:7d} {res['dp_Pa']:9.4f} "
              f"{100 * res['rel_error']:8.2f} {res['resolved_color_jump']:7.3f} "
              f"{res['normal_alignment_mean']:6.3f} "
              f"{res['curvature_shell_mean_per_m']:7.2f} +- "
              f"{res['curvature_shell_std_per_m']:5.2f}")
    two_phase = smoothed_sweep

    print("\nSharp (unsmoothed) lattice step colour, same chain -- sensitivity case:")
    sharp_sweep = []
    for dx in DX_LIST:
        h = H_FACTOR * dx
        pos, mass, density, color, _ = two_phase_case(dx, smoothed=False)
        res = laplace_jump(pos, mass, density, color, dx, h, SIGMA)
        res["dx_m"] = dx
        res["rel_error"] = abs(res["dp_Pa"] - theory) / theory
        sharp_sweep.append(res)
        print(f"  dx={dx:.4f} N={res['n_particles']:5d} dp={res['dp_Pa']:7.4f} Pa "
              f"(err {100 * res['rel_error']:5.1f} %) kappa spread "
              f"{res['curvature_shell_std_per_m']:6.2f} /m")

    orders = []
    for coarse, fine in zip(two_phase[:-1], two_phase[1:]):
        if fine["rel_error"] > 0 and coarse["rel_error"] > 0:
            orders.append(
                math.log2(coarse["rel_error"] / fine["rel_error"])
            )

    print("\nOne-sided free-surface cloud (shipped droplet configuration):")
    one_sided = []
    for dx in (0.01, 0.005):
        h = H_FACTOR * dx
        pos, mass, density, color = one_sided_case(dx)
        res = laplace_jump(pos, mass, density, color, dx, h, SIGMA)
        res["dx_m"] = dx
        res["rel_error"] = abs(res["dp_Pa"] - theory) / theory
        one_sided.append(res)
        print(f"  dx={dx:.4f} N={res['n_particles']:5d} dp={res['dp_Pa']:7.4f} Pa "
              f"(err {100 * res['rel_error']:5.1f} %) resolved jump d(c)={res['resolved_color_jump']:.3f}")

    legacy = None
    if True:
        dx = 0.005
        h = H_FACTOR * dx
        pos, mass, density, color, _ = two_phase_case(dx, smoothed=True)
        res = laplace_jump(
            pos, mass, density, color, dx, h, SIGMA, gradient_fn=legacy_color_gradient
        )
        res["dx_m"] = dx
        res["rel_error"] = abs(res["dp_Pa"] - theory) / theory
        legacy = res
        ref = next(r for r in smoothed_sweep if abs(r["dx_m"] - dx) < 1e-12)
        print("\nPre-fix colour gradient (j side updated with -V_j), dx=0.005:")
        print(f"  dp={res['dp_Pa']:.4f} Pa (err {100 * res['rel_error']:.1f} %), "
              f"n.r={res['normal_alignment_mean']:.4f}, kappa spread "
              f"{res['curvature_shell_std_per_m']:.1f} /m")
        print(f"  fixed form for reference: dp={ref['dp_Pa']:.4f} Pa "
              f"(err {100 * ref['rel_error']:.1f} %), n.r={ref['normal_alignment_mean']:.4f}, "
              f"kappa spread {ref['curvature_shell_std_per_m']:.1f} /m")

    criteria = {
        "two_phase_laplace_error_at_dx_0.0025_lt_15pct": {
            "measured": two_phase[-1]["rel_error"],
            "limit": 0.15,
            "result": "pass" if two_phase[-1]["rel_error"] < 0.15 else "fail",
        },
        "two_phase_laplace_error_at_dx_0.005_lt_15pct": {
            "measured": two_phase[-2]["rel_error"],
            "limit": 0.15,
            "result": "pass" if two_phase[-2]["rel_error"] < 0.15 else "fail",
        },
        "resolved_color_jump_at_dx_0.0025_within_5pct": {
            "measured": abs(two_phase[-1]["resolved_color_jump"] - 1.0),
            "limit": 0.05,
            "result": "pass"
            if abs(two_phase[-1]["resolved_color_jump"] - 1.0) < 0.05
            else "fail",
        },
        "error_shrinks_under_refinement": {
            "measured": two_phase[-1]["rel_error"] < two_phase[0]["rel_error"],
            "expected": True,
            "result": "pass"
            if two_phase[-1]["rel_error"] < two_phase[0]["rel_error"]
            else "fail",
        },
        "legacy_gradient_form_is_rejected": {
            "measured_normal_alignment": (
                legacy["normal_alignment_mean"] if legacy else None
            ),
            "limit": 0.95,
            "result": "pass"
            if legacy is not None and legacy["normal_alignment_mean"] < 0.95
            else "fail",
            "note": "the pre-fix form must fail the outward-normal gate; this "
                    "keeps the regression visible in the benchmark itself",
        },
        "outward_normal_alignment_gt_0.95": {
            "measured": min(r["normal_alignment_mean"] for r in two_phase),
            "limit": 0.95,
            "result": "pass"
            if min(r["normal_alignment_mean"] for r in two_phase) > 0.95
            else "fail",
        },
    }
    status = "PASS" if all(c["result"] == "pass" for c in criteria.values()) else "FAIL"

    report = {
        "benchmark": "surface_tension_laplace",
        "method": "radial line integral of the CSF volume force, dp = int sigma*kappa*|grad c| dr",
        "setup": {
            "radius_m": RADIUS,
            "sigma_N_per_m": SIGMA,
            "rho0_kg_per_m3": RHO0,
            "h_factor": H_FACTOR,
            "kernel": "2-D cubic spline, support 2h",
            "colour_field": "two-phase step (1 inside, 0 outside, equal density)",
        },
        "theory": {"laplace_dp_Pa": theory},
        "two_phase_sweep_smoothed_color": smoothed_sweep,
        "two_phase_sweep_sharp_step_color": sharp_sweep,
        "observed_convergence_order": orders,
        "interpretation": {
            "smoothed_color": "primary case: the colour field is resolved by the "
            "kernel, so the curvature is smooth and the Laplace error converges",
            "sharp_step_color": "sensitivity case: a lattice step puts the full "
            "jump inside one smoothing length; curvature noise scales like 1/h and "
            "the error is no longer monotone in dx",
        },
        "legacy_gradient_comparison_at_dx_0.005": legacy,
        "one_sided_free_surface": {
            "cases": one_sided,
            "verdict": "cannot close the Laplace law: the colour jump is only "
            "partly resolved because the interface has no particles on the "
            "outside; needs a two-phase or ghost-particle extension",
        },
        "criteria": criteria,
        "validation": {"status": status},
    }

    json_path = out / "validation_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {json_path}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        dxs = [r["dx_m"] for r in two_phase]
        errs = [100.0 * r["rel_error"] for r in two_phase]
        axes[0].loglog(dxs, errs, "o-", label="two-phase (CFS line integral)")
        axes[0].axhline(15.0, color="tab:red", ls="--", label="15 % acceptance bar")
        axes[0].axhline(3.0, color="tab:orange", ls=":", label="3 % target")
        axes[0].invert_xaxis()
        axes[0].set_xlabel("particle spacing dx [m]")
        axes[0].set_ylabel("|dp - sigma/R| / (sigma/R)  [%]")
        axes[0].set_title("Laplace-law convergence")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8)

        pressed = [r["resolved_color_jump"] for r in two_phase]
        axes[1].semilogx(dxs, pressed, "s-", label="two-phase")
        axes[1].semilogx(
            [r["dx_m"] for r in one_sided],
            [r["resolved_color_jump"] for r in one_sided],
            "v--",
            label="one-sided free surface",
        )
        axes[1].axhline(1.0, color="k", ls=":", label="full jump d(c) = 1")
        axes[1].invert_xaxis()
        axes[1].set_xlabel("particle spacing dx [m]")
        axes[1].set_ylabel("resolved colour jump  int |grad c| dr")
        axes[1].set_title("Why the one-sided cloud cannot close the law")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        png_path = out / "laplace_convergence.png"
        fig.savefig(png_path, dpi=160)
        print(f"Figure written to {png_path}")

    print(f"\nStatus: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())