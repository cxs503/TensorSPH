"""Resolution ladder of the TVC *distribution leg* (experiment-free).

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Round 3 closed the front-position ladder (``RESOLUTION_LADDER.md``) and left two
debts behind (§7.2 / §7.3): the bulk metrics were never turned into a ladder of
their own, and ``dx = 0.005`` was never run.  This script is that ladder.  It
scores the *particle distribution* alone against the exact analytic anchors of
:mod:`distribution` (perfect square lattice), so:

* no experimental reference enters -- the leg cannot inherit a digitisation or
  front-definition bias;
* the sample size is the whole cloud instead of the one-particle-wide front;
* "error < 15 %" is a statement about a quantity whose exact value is known.

Every level also reports the front score of ``benchmarks/xval_sphinx_tvc/run.py``
so the two legs can be compared level by level ("front-equivalent,
distribution-different" is then measurable, not anecdotal).

Usage
-----
    python benchmarks/xval_sphinx_tvc/distribution_ladder.py \
        --resolutions 0.04 0.03 0.02 0.01 0.005 \
        --variants baseline tvc_a010_all tvc_a015_all \
        --out docs/xval_sphinx_tvc/data/distribution_ladder.json

The initial state of every run is the lattice the case starts from, so the
metric set at ``t = 0`` must reproduce the analytic anchors to round-off; the
script asserts that (``initial_error_max``), which is what makes the metric a
*measurement* instead of a proxy.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from distribution import (distribution_error, lattice_metrics,  # noqa: E402
                          lattice_reference)
from run import VARIANTS, load_experiment, nondim_front, score  # noqa: E402
from tensorsph.dam_break import (DamBreakConfig, make_column,  # noqa: E402
                                 run_case)

#: Owner accuracy standard (errors above 15 % are not acceptable).
OWNER_GATE_REL = 0.15
#: Tolerance of the analytic-anchor check on the *initial* lattice.  The metric
#: accumulation is float64, but the solver places particles in float32, so a
#: coordinate is a rounded multiple of ``dx`` and the reconstructed lattice
#: differs from the ideal one at the float32 floor (measured 2.1e-7 here).  The
#: tolerance is set 50x above that floor and is still six orders of magnitude
#: below the smallest error the ladder measures, so it cannot mask anything.
ANCHOR_TOL = 1e-5


def _f(x) -> str:
    """Format an optional metric for the progress line."""
    return "  n/a " if x is None else f"{x:.4f}"


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip(),
    }


def _slope(xs, ys) -> float | None:
    """Least-squares exponent of ``y = C x^p`` (None if < 2 usable points)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
           if y is not None and y > 0 and x > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return sxy / sxx if sxx > 0 else None


def run_level(name: str, variant: dict, dx: float, seconds: float,
              output_samples: int = 60) -> dict:
    d0 = DamBreakConfig(dx=dx)
    dt = d0.dt_from_courant()
    steps = int(round(seconds / dt))
    cfg = DamBreakConfig(dx=dx, dt=dt, steps=steps,
                         output_every=max(1, steps // output_samples),
                         device="cpu", **variant)
    h = cfg.h_factor * dx
    lattice = lattice_reference(dx, h)
    t0 = time.time()
    run = run_case(cfg)
    wall = time.time() - t0

    tau, za = nondim_front(run.times, run.front, cfg.column_width, cfg.gravity)
    final = lattice_metrics(run.final_positions, dx, h, lattice=lattice)
    # Initial state: the solver starts on the lattice ``make_column`` builds,
    # so the metric set must reproduce the analytic anchors there for every
    # level and every variant.  Taken from the placement function itself, not
    # from a one-step run, so nothing but the lattice enters the check.
    init_pos, _ = make_column(cfg, torch.device("cpu"))
    initial = lattice_metrics(init_pos, dx, h, lattice=lattice)
    init_err = distribution_error(initial)["absolute"]

    return {
        "variant": name,
        "params": variant,
        "dx": dx,
        "h": h,
        "dt": dt,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall,
        "tau": tau,
        "Z_over_a": za,
        "front_score": score(tau, za, load_experiment()),
        "max_tvc_shift_over_h": run.max_tvc_shift / h,
        "max_density_compression": run.max_density_compression,
        "mass_drift": run.mass_drift,
        "escaped": run.escaped,
        "initial_metrics": initial,
        "initial_error_max": max(abs(v) for v in init_err.values()
                                 if v is not None),
        "metrics": final,
        "error": distribution_error(final),
    }


def analyse(records: list[dict], gate: float = OWNER_GATE_REL) -> dict:
    """Ladder-level analysis: gate status per level, decay exponent per metric."""
    out: dict = {"gate": gate, "anchor_tol": ANCHOR_TOL, "by_variant": {},
                 "slopes": {}, "verdict": {}}
    variants = sorted({r["variant"] for r in records})
    for name in variants:
        rows = sorted((r for r in records if r["variant"] == name),
                      key=lambda r: r["dx"])
        worst = [r["error"]["worst_relative"] for r in rows]
        grad1 = [r["metrics"].get("bulk_grad1_mean_h") for r in rows]
        close = [r["metrics"].get("close_pair_fraction") for r in rows]
        out["by_variant"][name] = {
            "dx": [r["dx"] for r in rows],
            "nn_median_over_dx": [r["metrics"].get("nn_median_over_dx")
                                  for r in rows],
            "nn_cv": [r["metrics"].get("nn_cv") for r in rows],
            "nn_relative_error": [r["error"]["relative"].get("nn_median_over_dx")
                                  for r in rows],
            "worst_relative": worst,
            "bulk_grad1_mean_h": grad1,
            "close_pair_fraction": close,
            "bulk_fraction": [r["metrics"].get("bulk_fraction") for r in rows],
            "front_mae": [r["front_score"].get("mae") for r in rows],
            "front_rel": [r["front_score"].get("mean_rel_error") for r in rows],
            "gate_pass_levels": [r["dx"] for r, w in zip(rows, worst)
                                 if w is not None and w <= gate],
            "gate_fail_levels": [r["dx"] for r, w in zip(rows, worst)
                                 if w is not None and w > gate],
        }
        out["slopes"][name] = {
            "worst_relative_vs_dx": _slope([r["dx"] for r in rows], worst),
            "bulk_grad1_vs_dx": _slope([r["dx"] for r in rows], grad1),
            "close_pair_vs_dx": _slope([r["dx"] for r in rows], close),
            "front_rel_vs_dx": _slope([r["dx"] for r in rows],
                                      [r["front_score"].get("mean_rel_error")
                                       for r in rows]),
        }
    return out


def figure(payload: dict, out: Path) -> None:
    """Three-panel ladder: nn error, grad-1 residual, close-pair fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    an = payload["analysis"]["by_variant"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    palette = {"baseline": ("tab:blue", "o-"),
               "tvc_a010_all": ("tab:green", "s-"),
               "tvc_a015_all": ("tab:orange", "^-")}

    def _curve(xs, ys, floor):
        pts = [(x, max(y, floor)) for x, y in zip(xs, ys) if y is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    ax = axes[0]
    for name, data in an.items():
        colour, marker = palette.get(name, ("tab:gray", "d-"))
        xs, ys = _curve(data["dx"], data["nn_relative_error"], 1.0e-12)
        ax.loglog(xs, ys, marker, color=colour, label=name)
    ax.axhline(OWNER_GATE_REL, color="tab:red", ls="--", lw=1.0,
               label="owner gate 15 %")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"relative error of median $r_{nn}/dx$ vs 1")
    ax.set_title("(a) lattice order vs analytic anchor")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, data in an.items():
        colour, marker = palette.get(name, ("tab:gray", "d-"))
        xs, ys = _curve(data["dx"], data["bulk_grad1_mean_h"], 1.0e-12)
        ax.loglog(xs, ys, marker, color=colour, label=name)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"mean $h\,|\sum_j \nabla_i W_{ij} V_j|$ (full-support)")
    ax.set_title("(b) grad-1 residual vs analytic anchor 0")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for name, data in an.items():
        colour, marker = palette.get(name, ("tab:gray", "d-"))
        xs, ys = _curve(data["dx"], data["close_pair_fraction"], 0.0)
        ax.semilogx(xs, ys, marker, color=colour, label=name)
        ax.set_xlabel("dx [m]")
    ax.set_ylabel("close-pair fraction (r < 0.7 dx), anchor 0")
    ax.set_title("(c) clumping vs analytic anchor 0")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("TVC distribution leg — resolution ladder (dam break, "
                 "Martin & Moyce geometry, no experimental input)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=float, nargs="+",
                    default=[0.04, 0.03, 0.02, 0.01])
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "tvc_a010_all"])
    ap.add_argument("--seconds", type=float, default=0.45)
    ap.add_argument("--output-samples", type=int, default=240)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "xval_sphinx_tvc" / "data"
                    / "distribution_ladder.json")
    ap.add_argument("--figure", type=Path,
                    default=ROOT / "docs" / "xval_sphinx_tvc"
                    / "figure_distribution_ladder.png")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    records = []
    for dx in args.resolutions:
        for name in args.variants:
            rec = run_level(name, VARIANTS[name], dx, args.seconds,
                            output_samples=args.output_samples)
            err = rec["error"]
            print(f"dx={dx:<6} {name:<14} "
                  f"nn_err={_f(err['relative'].get('nn_median_over_dx'))} "
                  f"grad1={_f(rec['metrics'].get('bulk_grad1_mean_h'))} "
                  f"close={_f(rec['metrics'].get('close_pair_fraction'))} "
                  f"front_rel={_f(rec['front_score'].get('mean_rel_error'))} "
                  f"wall={rec['wall_clock_s']:.1f}s", flush=True)
            assert rec["initial_error_max"] <= ANCHOR_TOL, (
                f"metric does not reproduce the analytic anchors on the "
                f"initial lattice: {rec['initial_error_max']}")
            records.append(rec)
            payload = {
                "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
                "leg": "distribution (experiment-free, analytic anchors)",
                "reference": {
                    "anchors": "perfect square lattice, see "
                               "benchmarks/xval_sphinx_tvc/distribution.py",
                    "front": load_experiment()["source"],
                },
                "environment": _environment(),
                "seconds": args.seconds,
                "gates": {"owner_rel": OWNER_GATE_REL,
                          "anchor_tol": ANCHOR_TOL},
                "records": records,
                "analysis": analyse(records),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2))
    if not args.no_figure:
        try:
            figure(payload, args.figure)
            print(f"figure -> {args.figure}")
        except Exception as exc:            # matplotlib is optional on the
            print(f"figure skipped ({type(exc).__name__}: {exc})")  # nodes
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())