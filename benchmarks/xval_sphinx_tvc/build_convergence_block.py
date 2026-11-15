#!/usr/bin/env python3
"""Fill the ``convergence`` block of ``benchmarks/xval_sphinx_tvc/validation_report.json``.

Why this exists
---------------
``validation_report.json`` used to record only the *best* rung of the resolution
ladder (``dx = 0.03``, MAE 0.1425, PASS) while the four-level ladder was already
committed in ``docs/xval_sphinx_tvc/data/ladder_dense.json``.  Quoting the best
rung alone reads like "the TVC closure converges onto the experiment"; the
ladder says the opposite: the fixed-``alpha = 0.10`` error is
17.1 / 10.6 / 12.5 / 16.4 % at ``dx`` 0.04 / 0.03 / 0.02 / 0.01, i.e. it is
**non-monotone in dx** and only the ``dx = 0.03`` rung meets both gates.

This script promotes the whole ladder into the machine-readable report so the
ledger can grade every rung instead of one hand-picked point.  Every number is
read from a committed data file at generation time -- nothing is typed here, so
"recompute" and "read the report" cannot disagree.

Usage
-----
    python benchmarks/xval_sphinx_tvc/build_convergence_block.py           # write
    python benchmarks/xval_sphinx_tvc/build_convergence_block.py --check   # drift only

``--check`` exits 1 when the committed block differs from the recomputation
(CI semantics; ``tests/test_tvc_convergence_analysis.py`` also guards it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "benchmarks" / "xval_sphinx_tvc" / "validation_report.json"
DATA = REPO / "docs" / "xval_sphinx_tvc" / "data"
LADDER = "docs/xval_sphinx_tvc/data/ladder_dense.json"
SELF_CONV = "docs/xval_sphinx_tvc/data/convergence_ladder.json"
ALPHA_SWEEP = "docs/xval_sphinx_tvc/data/sweep_dx001_alpha.json"

#: The closure the ladder was run with (round-1 calibration, see README.md).
REFERENCE_VARIANT = "tvc_a010_all"
ALPHA = 0.10


def _load(rel: str) -> dict:
    with (REPO / rel).open() as fh:
        return json.load(fh)


def _r(x, nd: int = 6):
    return None if x is None else round(float(x), nd)


def _rows_by_level(ladder: dict) -> dict[float, dict[str, dict]]:
    """``{dx: {variant: score-ish row}}`` from the committed ladder payload."""
    out: dict[float, dict[str, dict]] = {}
    for rec in ladder["records"]:
        out.setdefault(float(rec["dx"]), {})[rec["variant"]] = rec
    return out


def _score_row(rec: dict) -> dict:
    s = rec["score"]
    return {
        "mae": _r(s["mae"]),
        "mean_rel_error": _r(s["mean_rel_error"]),
        "max_abs_error": _r(s.get("max_abs_error")),
        "gate_abs_pass": bool(s["gate_abs_pass"]),
        "gate_rel_pass": bool(s["gate_rel_pass"]),
        "max_density_compression": _r(rec.get("max_density_compression")),
        "mass_drift": _r(rec.get("mass_drift")),
    }


def build() -> dict:
    """Recompute the convergence block from the committed data files."""
    ladder = _load(LADDER)
    gates = {"abs_mae": ladder["gates"]["ledger_abs"], "rel": ladder["gates"]["owner_rel"]}
    by_level = _rows_by_level(ladder)
    levels: list[dict] = []
    for dx in sorted(by_level, reverse=True):
        rows = by_level[dx]
        variants = {name: _score_row(rec) for name, rec in sorted(rows.items())}
        ref = variants[REFERENCE_VARIANT]
        base = variants.get("baseline")
        levels.append(
            {
                "dx": dx,
                "variants": variants,
                "tvc_over_baseline_mae_factor": (
                    _r(base["mae"] / ref["mae"], 3) if base else None
                ),
                "tvc_beats_baseline": bool(base and ref["mae"] < base["mae"]),
                "tvc_meets_both_gates": bool(ref["gate_abs_pass"] and ref["gate_rel_pass"]),
            }
        )

    level_by_dx = {lvl["dx"]: lvl for lvl in levels}
    ref_mae = {dx: lvl["variants"][REFERENCE_VARIANT]["mae"] for dx, lvl in level_by_dx.items()}
    ordered = sorted(ref_mae, reverse=True)         # coarse -> fine
    pairs = list(zip(ordered[:-1], ordered[1:]))
    increments_improve = [ref_mae[fine] < ref_mae[coarse] for coarse, fine in pairs]
    meeting = [dx for dx in ordered if level_by_dx[dx]["tvc_meets_both_gates"]]
    best_dx = min(ref_mae, key=lambda d: ref_mae[d])
    worst_dx = max(ref_mae, key=lambda d: ref_mae[d])

    self_conv = _load(SELF_CONV)
    self_block: dict[str, dict] = {}
    for name, v in sorted(self_conv["variants"].items()):
        rich = v.get("richardson")
        self_block[name] = {
            "cauchy_l2": [
                {"dx_coarse": p["dx_coarse"], "dx_fine": p["dx_fine"], "r": _r(p["r"], 4),
                 "cauchy_l2": _r(p["cauchy_l2"])}
                for p in v["pairs"]
            ],
            "observed_order_l2": [
                {"dx_fine_interval": o["dx_fine_interval"],
                 "observed_order_l2": _r(o["observed_order_l2"], 3)}
                for o in v.get("orders", [])
            ],
            "richardson_limit": (
                None if rich is None else
                {"p_used": _r(rich["p_used"], 3), "from_levels": rich["from_levels"],
                 "mae_vs_experiment": _r(rich["vs_experiment"]["mae"]),
                 "mean_rel_error_vs_experiment": _r(rich["vs_experiment"]["mean_rel_error"])}
            ),
            "in_asymptotic_range": bool(
                v.get("orders") and all(o["observed_order_l2"] > 0 for o in v["orders"])
            ),
        }

    sweep = _load(ALPHA_SWEEP)
    sweep_rows = [
        {"variant": rec["variant"], "dx": rec["dx"],
         **_score_row(rec)}
        for rec in sweep["records"]
    ]

    rel_pcts = {dx: level_by_dx[dx]["variants"][REFERENCE_VARIANT]["mean_rel_error"] * 100
                for dx in ordered}
    verdict = {
        "fixed_alpha_convergence": "falsified",
        "statement": (
            "At fixed alpha = 0.10 the error against Martin & Moyce (1952) is "
            + " / ".join(f"{rel_pcts[dx]:.1f} %" for dx in ordered)
            + " at dx = " + " / ".join(f"{dx:g}" for dx in ordered)
            + " m: the sequence is non-monotone, the minimum sits at "
            f"dx = {best_dx:g} m and the finest level (dx = {worst_dx:g} m, "
            f"MAE {ref_mae[worst_dx]:.4f}) is worse than the coarsest. "
            f"Only {len(meeting)} of {len(ordered)} levels meet both gates."
        ),
        "mechanism": (
            "the per-application shift is resolution independent "
            "(|dr| = alpha h^2 |grad-1|, h ~ dx, |grad-1| ~ 1/dx), while a fixed "
            "physical duration contains N = T/dt ~ 1/dx applications, so the "
            "cumulative regularisation rate ~ alpha dx weakens under refinement; "
            "the optimum therefore drifts up (0.10 at dx = 0.03 -> 0.15 at dx = 0.01)"
        ),
        "may_be_quoted_as": (
            f"a calibrated point at dx = {best_dx:g} m (MAE {ref_mae[best_dx]:.4f} Z/a, "
            f"{rel_pcts[best_dx]:.1f} % of the experiment), NOT as a converged limit"
        ),
        "data": LADDER,
    }

    return {
        "schema": "tvc-convergence-ladder/1",
        "analysis": (
            "four-level resolution ladder of the fixed-alpha TVC closure against the "
            "Martin & Moyce (1952) front-position digitisation, plus Cauchy "
            "self-convergence and the alpha sweep at the finest level"
        ),
        "sources": {"ladder": LADDER, "self_convergence": SELF_CONV,
                    "alpha_sweep": ALPHA_SWEEP},
        "gates": gates,
        "reference_variant": {"variant": REFERENCE_VARIANT, "alpha": ALPHA},
        "levels": levels,
        "ladder_summary": {
            "n_levels": len(ordered),
            "dx_levels": ordered,
            "refinement_pairs_coarse_to_fine": [[c, f] for c, f in pairs],
            "mae_improves_at_each_refinement": increments_improve,
            "mae_monotone_under_refinement": all(increments_improve),
            "best_dx": best_dx,
            "best_mae": ref_mae[best_dx],
            "worst_dx": worst_dx,
            "worst_mae": ref_mae[worst_dx],
            "finest_level_is_best": best_dx == min(ordered),
            "n_levels_meeting_both_gates": len(meeting),
            "levels_meeting_both_gates": meeting,
        },
        "self_convergence": self_block,
        "alpha_sweep_finest_level": sweep_rows,
        "convergence_verdict": verdict,
    }


def write() -> dict:
    report = _load("benchmarks/xval_sphinx_tvc/validation_report.json")
    block = build()
    report["convergence"] = block
    with REPORT.open("w") as fh:
        json.dump(report, fh, indent=2)
    return block


def check() -> int:
    report = _load("benchmarks/xval_sphinx_tvc/validation_report.json")
    committed = report.get("convergence")
    if committed != build():
        print("DRIFT: benchmarks/xval_sphinx_tvc/validation_report.json::convergence is stale "
              "(run build_convergence_block.py)")
        return 1
    print("convergence block up to date")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = ap.parse_args(argv)
    if args.check:
        return check()
    block = write()
    s = block["ladder_summary"]
    print(
        "wrote benchmarks/xval_sphinx_tvc/validation_report.json::convergence: "
        f"{s['n_levels']} levels, best dx={s['best_dx']:g} MAE={s['best_mae']:.4f}, "
        f"monotone={s['mae_monotone_under_refinement']}, "
        f"levels meeting both gates={s['n_levels_meeting_both_gates']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())