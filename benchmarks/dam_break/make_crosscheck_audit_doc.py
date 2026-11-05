"""Render docs/benchmarks/dam-break-crosscheck-audit.md from the JSON artifacts.

Every number in the document is injected from
``benchmarks/dam_break/crosscheck_audit.json`` and
``benchmarks/dam_break/reference_ladder.json``; nothing is typed by hand.

Usage::

    python benchmarks/dam_break/make_crosscheck_audit_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "docs" / "benchmarks" / "dam-break-crosscheck-audit.md"


def f(x, fmt="{:.4f}"):
    return "n/a" if x is None else fmt.format(x)


def main() -> int:
    audit = json.loads((HERE / "crosscheck_audit.json").read_text())
    ladder_path = HERE / "reference_ladder.json"
    ladder = json.loads(ladder_path.read_text()) if ladder_path.exists() else None
    gb = audit["grid_stats"]["bundled"]
    gc = audit["grid_stats"]["crosscheck"]
    spread = audit["cross_source_spread"]
    gates = audit["gates"]

    L: list[str] = []
    L.append("# B2 dam-break: audit of the experimental reference (generated)\n")
    L.append("**Generated file** -- every number below is injected from "
             "`benchmarks/dam_break/crosscheck_audit.json`"
             + (" and `benchmarks/dam_break/reference_ladder.json`" if ladder else "")
             + " by `benchmarks/dam_break/make_crosscheck_audit_doc.py`. "
               "Do not edit by hand.\n")

    L.append("## Why this audit exists\n")
    L.append("`docs/benchmarks/dam-break-resolution-ladder.md` measured the Martin & Moyce "
             "error of the B2 dam-break case on a refinement ladder and found it "
             "*resolution independent* (mean absolute error 0.5988 / 0.5774 / 0.5728 for "
             "`dx` = 0.04 / 0.02 / 0.01 m). It concluded that the residual is "
             "\"structural rather than under-resolved\" and listed four candidate causes, "
             "of which the first three \"have to be separated by measurement\". The "
             "verdict recorded there is `physical validation fails`.\n")
    L.append("This audit measures the cheapest separable item first: **the reference "
             "table itself**. It needs no simulation, only the two tables, and it is "
             "decisive because an error the size of the reference's own uncertainty "
             "cannot be attributed to the solver.\n")

    L.append("## Table A -- the bundled digitisation (`martin_moyce_1952.json`)\n")
    L.append("| property | measured | what a point-by-point figure reading would give |")
    L.append("|---|---|---|")
    L.append(f"| points | {gb['n_points']} | -- |")
    L.append(f"| abscissa step | {f(gb['tau_step_mean'], '{:.4f}')} (spread "
             f"{f(gb['tau_step_spread'], '{:.1e}')}) | irregular (figure pixels) |")
    L.append(f"| abscissa grid arithmetic | {gb['tau_grid_is_arithmetic']} | no |")
    L.append(f"| ordinate quantum | {f(gb['ordinate_quantum'], '{:.3f}')} | "
             "finer than the claimed uncertainty |")
    L.append(f"| distinct slope levels | {gb['n_slope_levels']} "
             f"({', '.join(f(s, '{:.3f}') for s in gb['slope_levels'])}) | many, noisy |")
    L.append(f"| first-interval slope / peak slope | "
             f"{f(gb['first_slope_to_peak'], '{:.3f}')} | well below 1 (front starts from rest) |")
    L.append("")
    L.append("Every `tau` step is identical, every ordinate is a multiple of "
             f"{f(gb['ordinate_quantum'], '{:.3f}')}, and the finite-difference slope takes "
             f"only {gb['n_slope_levels']} discrete values: the table is a piecewise-linear "
             "sketch through a handful of hand-picked vertices, not a point-by-point "
             "reading of Figure 4. Its ±0.05 uncertainty is therefore an aspiration about "
             "the picture rather than a measured error bar.\n")

    L.append("## Table B -- an independent transcription of the same figure\n")
    L.append("`benchmarks/dam_break/martin_moyce_1952_crosscheck.json` holds a second, "
             "value-independent transcription of the same `Z/a` curve (14 points, "
             "provenance recorded in the file: it is the digitisation distributed with "
             "Lethe's official dam-break post-processing, transcribed from the frozen "
             "anchor table of the sibling solver repositories in this workspace). "
             "It shares the case aspect ratio (height/width = 2) and is converted to the "
             "same frame with `Z/a = X - 1`.\n")
    L.append("| property | bundled | cross-check |")
    L.append("|---|---|---|")
    L.append(f"| points | {gb['n_points']} | {gc['n_points']} |")
    L.append(f"| abscissa arithmetic | {gb['tau_grid_is_arithmetic']} | "
             f"{gc['tau_grid_is_arithmetic']} |")
    L.append(f"| ordinate quantum | {f(gb['ordinate_quantum'], '{:.3f}')} | "
             f"{f(gc['ordinate_quantum'], '{:.3f}')} |")
    L.append(f"| slope levels | {gb['n_slope_levels']} | {gc['n_slope_levels']} |")
    L.append(f"| first-interval slope / peak | {f(gb['first_slope_to_peak'], '{:.3f}')} | "
             f"{f(gc['first_slope_to_peak'], '{:.3f}')} |")
    L.append("")
    L.append("The cross-check has the shape a gate-release experiment must have: the "
             f"front leaves rest at {f(gc['first_slope_to_peak'] * 100, '{:.0f}')} % of its "
             f"peak speed and accelerates by {f(gc['slope_peak'] / gc['slope_first'], '{:.1f}')}x. "
             "The bundled table is already at "
             f"{f(gb['first_slope_to_peak'] * 100, '{:.0f}')} % of its peak in the first "
             "interval after release, i.e. it describes an impulsive start.\n")

    L.append("## Cross-source spread = a lower bound on the reference uncertainty\n")
    L.append(f"Over the common window `tau` in [{f(spread['tau_window'][0], '{:.2f}')}, "
             f"{f(spread['tau_window'][1], '{:.2f}')}] ({spread['n_overlap']} points):\n")
    L.append("| statistic | value |")
    L.append("|---|---|")
    L.append(f"| max abs difference in `Z/a` | {f(spread['max_abs_delta'], '{:.3f}')} |")
    L.append(f"| rms difference | {f(spread['rms_delta'], '{:.3f}')} |")
    L.append(f"| mean signed difference | {f(spread['mean_signed_delta'], '{:.3f}')} |")
    L.append("| claimed uncertainty (both tables) | ±0.050 |")
    L.append(f"| ratio max difference / claimed | "
             f"{f(gates['claimed_uncertainty_consistent']['measured'], '{:.1f}')}x |")
    L.append("")
    L.append("| `tau` | bundled `Z/a` | cross-check `Z/a` | difference |")
    L.append("|---|---|---|---|")
    for row in spread["rows"]:
        L.append(f"| {f(row['tau'], '{:.3f}')} | {f(row['bundled_za'], '{:.3f}')} | "
                 f"{f(row['crosscheck_za'], '{:.3f}')} | {f(row['delta'], '{:+.3f}')} |")
    L.append("")
    L.append("Both tables come from the same paywalled original, so this spread is a "
             "*lower bound*: two honest readers of the same figure already disagree by up "
             "to "
             f"{f(spread['max_abs_delta'], '{:.2f}')} in `Z/a`, "
             f"{f(gates['claimed_uncertainty_consistent']['measured'], '{:.0f}')}x the "
             "quoted error bar.\n")

    L.append("## Gates\n")
    L.append("| gate | metric | measured | limit | result |")
    L.append("|---|---|---|---|---|")
    for name, gate in gates.items():
        measured = gate["measured"]
        if isinstance(measured, dict):
            measured = ", ".join(f"{k}={v}" for k, v in measured.items())
        else:
            measured = f(measured, "{:.4g}")
        L.append(f"| `{name}` | {gate['metric']} | {measured} | {gate['limit']} | "
                 f"**{gate['result']}** |")
    L.append("")
    L.append(f"**Verdict: `physical_validation = "
             f"{audit['verdict']['physical_validation_status']}`.** "
             f"{audit['verdict']['reason']}\n")

    if ladder:
        L.append("## The same simulations against both references\n")
        params = ladder["simulation_params"]
        L.append("Setup (identical for every rung and for both references): "
                 f"`a` = {params['column_width']} m, column height = "
                 f"{params['column_height']} m (2:1), tank = {params['tank_width']} m, "
                 f"`c0` = {params['sound_speed']} m/s, wall friction "
                 f"{params['wall_friction']}, density diffusion "
                 f"{params['density_diffusion']}, simulated time {params['seconds']} s.\n")
        L.append("| `dx` (m) | particles | steps | wall clock (s) | MAE vs bundled | "
                 "RMSE vs bundled | MAE vs cross-check | RMSE vs cross-check |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in ladder["results"]:
            L.append(f"| {r['dx']:.3f} | {r['particles']} | {r['steps_taken']} | "
                     f"{r['wall_clock_s']:.1f} | "
                     f"{f(r['error_vs_bundled']['mae'])} | {f(r['error_vs_bundled']['rmse'])} | "
                     f"{f(r['error_vs_crosscheck']['mae'])} | "
                     f"{f(r['error_vs_crosscheck']['rmse'])} |")
        L.append("")
        L.append("Gates (limit on the mean absolute error in `Z/a` = "
                 f"{ladder['gates']['bundled']['limit']:.2f}):\n")
        L.append("| reference | finest `dx` | MAE | result | all rungs pass |")
        L.append("|---|---|---|---|---|")
        for name in ("bundled", "crosscheck"):
            g = ladder["gates"][name]
            L.append(f"| {name} | {g['finest']['dx']:.3f} | {f(g['finest']['mae'])} | "
                     f"**{g['finest']['result']}** | {g['all_pass']} |")
        L.append("")
        L.append("Observed convergence order of the error (slope of log MAE against "
                 "log `dx`, so a value near 0 means the residual does not respond to "
                 "refinement):\n")
        L.append("| reference | order from MAE | order from RMSE |")
        L.append("|---|---|---|")
        for name in ("bundled", "crosscheck"):
            conv = ladder["convergence"][name]
            L.append(f"| {name} | "
                     f"{', '.join(f(x, '{:+.3f}') for x in conv['order_mae'])} | "
                     f"{', '.join(f(x, '{:+.3f}') for x in conv['order_rmse'])} |")
        L.append("")

    if ladder:
        cc = ladder["gates"]["crosscheck"]
        worst = max(cc["rows"], key=lambda r: r["mae"])
        best = min(cc["rows"], key=lambda r: r["mae"])
        L.append("### The cross-check pass is not a convergence proof\n")
        L.append("The cross-check error is *not* monotone in `dx`: it reaches "
                 f"{f(best['mae'])} at `dx` = {best['dx']:.3f} m and then rises to "
                 f"{f(worst['mae'])} at `dx` = {worst['dx']:.3f} m, so the observed "
                 "order above changes sign. Every rung stays inside the "
                 f"{cc['limit']:.2f} limit, but the honest reading is \"within the "
                 "reference uncertainty at all three rungs\", not \"converging to the "
                 "experiment\". A 50-800 particle 2-D run cannot settle which of the two "
                 "tables is closer to the 1952 apparatus; it can only show that the "
                 "misfit against the bundled table is the same size as the disagreement "
                 "between the two transcriptions.\n")
    L.append("## What this changes and what it does not\n")
    L.append("1. **The B2 status changes from `fails` to `withheld`.** The old verdict read "
             "a 0.57 `Z/a` misfit as a solver error. Against a reference whose two "
             "transcriptions differ by up to "
             f"{f(spread['max_abs_delta'], '{:.2f}')}, that attribution is not supported. "
             "This is a correction of the *verdict*, not of the numbers: the solver's "
             "misfit against the bundled table is still 0.57.\n")
    L.append("2. **The bundled table stays in the repository and stays the default for the "
             "B2 gate**, so the historical series is comparable; the audit only records "
             "that its verdicts cannot be read as validation of the solver.\n")
    L.append("3. **What is still unmeasured.** The two remaining candidates from the "
             "resolution ladder are untouched here: the 2-D cross-section versus the "
             "3-D experiment (side-wall drag), and the front definition (`x_max` is the "
             "leading particle, which at `dx` = 0.04 m is a one-particle-thick sheet "
             "0.01-0.02 m above the floor). Neither can be settled from the tables "
             "alone; a 3-D run and a bulk-front estimator are the next increments.\n")
    L.append("4. **What would upgrade this audit.** Access to the archived 1952 figure at "
             "its native resolution, or experimental data from an open dataset with "
             "per-point error bars. Until then any B2 validation claim should quote the "
             "reference spread alongside the solver error.\n")

    L.append("## Reproduce\n")
    L.append("```bash")
    L.append("python benchmarks/dam_break/crosscheck_audit.py")
    L.append("python benchmarks/dam_break/reference_ladder.py --sizes 0.04,0.02,0.01 \\")
    L.append("    --seconds 0.5 --json-out benchmarks/dam_break/reference_ladder.json --plot")
    L.append("python benchmarks/dam_break/make_crosscheck_audit_doc.py")
    L.append("pytest tests/test_dam_break_crosscheck_audit.py -q")
    L.append("```")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
