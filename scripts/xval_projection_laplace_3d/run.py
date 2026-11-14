"""XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE, 3-D arm: run the ladder, write results.

Every number in ``docs/xval_projection_laplace_3d.md`` comes from the JSON this
script writes; nothing is typed by hand.  Deterministic: no random input, the
lattices are analytic and the CG stop criterion is fixed at ``rel_tol = 1e-10``.

Usage
    python scripts/xval_projection_laplace_3d/run.py [--quick]

``--quick`` runs the coarse level only (the pytest suite uses the library
directly; this flag exists for a fast smoke check on the worktree).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph3d import laplace_isph3d_case  # noqa: E402

RADIUS = 0.02
SIGMA = 0.072
RHO0 = 1000.0
OUTER = 0.012
H_FACTOR = 1.3
LEVELS = (0.004, 0.0025, 0.0016)      # R/dx = 5, 8, 12.5
ARMS = ("mls", "color_quadric")
DECOMP_LEVEL = 0.0025
OUT = Path(__file__).resolve().parent / "results"


def _timed(**overrides):
    kwargs = {"radius": RADIUS, "sigma": SIGMA, "rho0": RHO0, "outer": OUTER,
              "h_factor": H_FACTOR}
    kwargs.update(overrides)
    t0 = time.time()
    report = laplace_isph3d_case(**kwargs)
    report["wall_s"] = round(time.time() - t0, 2)
    return report


def ladder(levels=LEVELS, arms=ARMS):
    runs = []
    for dx in levels:
        for arm in arms:
            runs.append(_timed(dx=dx, curvature_method=arm, curvature_stencil=2.0))
            print(f"[ladder] dx={dx:.4f} arm={arm:14s} "
                  f"dp_err={runs[-1]['dp_error_pct']:+.2f}% "
                  f"kappaR={runs[-1]['kappaR_delta_weighted']:.3f} "
                  f"{runs[-1]['wall_s']}s", flush=True)
    return runs


def decomposition(dx=DECOMP_LEVEL):
    """Which term owns the residual error: curvature or delta function?"""
    out = []
    for variant, arm in (("model", "mls"), ("kappa_exact", "mls"),
                         ("delta_unit", "mls"), ("gauge", "mls")):
        rep = _timed(dx=dx, curvature_method=arm, curvature_stencil=2.0,
                     variant=variant)
        out.append(rep)
        print(f"[decomp] {variant:12s} dp_err={rep['dp_error_pct']:+.2f}% "
              f"{rep['wall_s']}s", flush=True)
    return out


def controls(dx=DECOMP_LEVEL):
    zero = _timed(dx=dx, curvature_method="mls", curvature_stencil=2.0,
                  sigma=0.0, variant="kappa_exact")
    zero["control"] = "sigma=0 with the analytic curvature (pure solve check)"
    print(f"[control] sigma=0 dp={zero['dp_measured_Pa']:.3e} Pa "
          f"{zero['wall_s']}s", flush=True)
    return [zero]


def sensitivity(dx=DECOMP_LEVEL):
    """Band / core-fraction sensitivity of the measured jump."""
    out = []
    for core, band in ((0.4, (1.25, 1.45)), (0.6, (1.25, 1.45)),
                       (0.5, (1.05, 1.15)), (0.5, (1.6, 2.0))):
        rep = _timed(dx=dx, curvature_method="mls", curvature_stencil=2.0,
                     core_frac=core, band=band)
        rep["core_frac"] = core
        rep["band"] = list(band)
        out.append(rep)
        print(f"[sensitivity] core={core} band={band} "
              f"dp_err={rep['dp_error_pct']:+.2f}% {rep['wall_s']}s", flush=True)
    return out


def make_figure(runs, decomp, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    colours = {"mls": "tab:blue", "color_quadric": "tab:orange"}

    ax = axes[0]
    for arm in ARMS:
        rows = [r for r in runs if r["curvature_method"] == arm]
        ax.plot([r["R_over_dx"] for r in rows],
                [r["kappaR_delta_weighted"] for r in rows], "o-",
                color=colours[arm], label=f"{arm} (measured)")
    ax.axhline(2.0, color="k", ls="--", lw=1, label=r"analytic $\kappa R=2$")
    ax.axhspan(1.9, 2.1, color="grey", alpha=0.15, label=r"$\pm5\%$")
    ax.set_xlabel(r"$R/\Delta x$")
    ax.set_ylabel(r"$\kappa R$ (delta-weighted)")
    ax.set_title("(a) 3-D curvature on the lattice sphere")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for arm in ARMS:
        rows = [r for r in runs if r["curvature_method"] == arm]
        ax.loglog([r["R_over_dx"] for r in rows],
                  [abs(r["dp_error_pct"]) for r in rows], "o-",
                  color=colours[arm], label=f"{arm}")
    ax.axhline(3.0, color="tab:green", ls=":", lw=1.2, label="3 % target")
    ax.axhline(15.0, color="tab:red", ls="-.", lw=1.2, label="15 % owner gate")
    ax.set_xlabel(r"$R/\Delta x$")
    ax.set_ylabel(r"$|\Delta p/\Delta p_{\rm theory}-1|$  [%]")
    ax.set_title(r"(b) Young-Laplace jump $\Delta p=2\sigma/R$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    labels = [r["variant"] for r in decomp]
    errs = [r["dp_error_pct"] for r in decomp]
    bars = ax.bar(range(len(labels)), errs,
                  color=["tab:blue", "tab:purple", "tab:brown", "tab:grey"])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-3.0, 3.0, color="tab:green", alpha=0.12, label="3 % band")
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, e,
                f"{e:+.1f}%", ha="center",
                va="bottom" if e >= 0 else "top", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\Delta p$ error [%]")
    ax.set_title(f"(c) error decomposition at $R/\\Delta x={RADIUS/DECOMP_LEVEL:.0f}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("XV_THREADS", "8")))

    levels = (LEVELS[0],) if args.quick else LEVELS
    runs = ladder(levels)
    decomp = decomposition() if not args.quick else decomposition(LEVELS[0])
    ctrl = controls(LEVELS[0] if args.quick else DECOMP_LEVEL)
    sens = sensitivity(LEVELS[0] if args.quick else DECOMP_LEVEL)

    verdict = {
        "gate_15pct": all(abs(r["dp_error_pct"]) < 15.0 for r in runs),
        "under_3pct_all_levels": {
            arm: [abs(r["dp_error_pct"]) for r in runs
                  if r["curvature_method"] == arm] for arm in ARMS},
        "kappaR_within_5pct": all(abs(r["kappaR_delta_weighted"] - 2.0) < 0.1
                                  for r in runs),
        "sigma0_control_dp_Pa": ctrl[0]["dp_measured_Pa"],
        "dp_theory_Pa": 2.0 * SIGMA / RADIUS,
    }
    payload = {
        "case": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE/3d-arm",
        "params": {"radius_m": RADIUS, "sigma_N_per_m": SIGMA,
                   "rho0_kg_per_m3": RHO0, "outer_m": OUTER,
                   "h_factor": H_FACTOR, "levels_m": list(levels),
                   "arms": list(ARMS), "dp_theory_Pa": 2.0 * SIGMA / RADIUS},
        "runs": runs, "decomposition": decomp, "controls": ctrl,
        "sensitivity": sens, "verdict": verdict,
        "library": "tensorsph.laplace_isph3d.laplace_isph3d_case",
        "figure": str(OUT / "xval_projection_laplace_3d.png"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xval_projection_laplace_3d.json").write_text(json.dumps(payload, indent=1))
    make_figure(runs, decomp, OUT / "xval_projection_laplace_3d.png")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())