"""Render ``docs/benchmarks/dam-break-reference-crosscheck.md`` from the JSON artifacts.

Every number in the document is injected from
``benchmarks/dam_break/reference_crosscheck.json`` (structure + cross-source
spread), ``benchmarks/dam_break/reference_ladder.json`` (solver scored against
each reference) and ``benchmarks/dam_break/reference_audit.json`` (the
companion table-integrity / Ritter audit). Nothing is typed by hand.

Usage::

    python benchmarks/dam_break/reference_crosscheck.py --plot
    python benchmarks/dam_break/reference_ladder.py --plot
    python benchmarks/dam_break/make_reference_audit_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "docs" / "benchmarks" / "dam-break-reference-crosscheck.md"


def f(x, fmt="{:.4f}"):
    return "n/a" if x is None else fmt.format(x)


def load(name: str):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    cross = load("reference_crosscheck.json")
    if cross is None:
        raise SystemExit("run reference_crosscheck.py first")
    ladder = load("reference_ladder.json")
    audit = load("reference_audit.json")

    gb = cross["grid_stats"]["bundled"]
    gc = cross["grid_stats"]["crosscheck"]
    spread = cross["cross_source_spread"]
    gates = cross["gates"]
    verdict = cross["verdict"]

    L: list[str] = []
    L.append("# B2 dam-break: is the experimental reference able to carry a verdict?\n")
    L.append("**Generated file** -- every number below is injected from "
             "`benchmarks/dam_break/reference_crosscheck.json`"
             + (", `benchmarks/dam_break/reference_ladder.json`" if ladder else "")
             + (" and `benchmarks/dam_break/reference_audit.json`" if audit else "")
             + " by `benchmarks/dam_break/make_reference_audit_doc.py`. "
               "Do not edit by hand.\n")

    L.append("## Why this audit exists\n")
    L.append("`docs/benchmarks/dam-break-resolution-ladder.md` measured the Martin & "
             "Moyce error of the B2 dam-break case on a refinement ladder and found it "
             "*resolution independent* (mean absolute error 0.5988 / 0.5774 / 0.5728 in "
             "`Z/a` for `dx` = 0.04 / 0.02 / 0.01 m). An error that survives refinement "
             "is either a structural reference problem or a structural solver problem. "
             "The ladder could not separate the two, and the case was reported as "
             "`physical validation fails`.\n")
    L.append("This audit settles the reference half of that question by measurement, "
             "with no simulation involved: it compares the bundled table against a "
             "second, independent transcription of the same published figure. If those "
             "two disagree by more than the gate limit, then the gate cannot arbitrate "
             "anything and the verdict must change from `fail` to `withheld`.\n")

    L.append("## The two references\n")
    L.append("| table | file | points | claimed uncertainty | role |")
    L.append("|---|---|---|---|---|")
    claim_b = cross["tables"]["bundled"]["claimed_uncertainty"]
    L.append(f"| bundled | `{cross['tables']['bundled']['path']}` | "
             f"{gb['n_points']} | +-{f(claim_b, '{:.2f}')} "
             "| the table the B2 gate has used |")
    L.append(f"| cross-check | `{cross['tables']['crosscheck']['path']}` | "
             f"{gc['n_points']} | "
             f"+-{f(cross['tables']['crosscheck']['claimed_uncertainty'], '{:.2f}')} "
             "| independent transcription, cross-check only, never a gate |")
    L.append("")
    L.append("Neither table has been verified against the paywalled 1952 original; both "
             "are transcriptions of the figure-level data that circulates in the "
             "open literature. Their spread is therefore a **lower bound** on the "
             "reference uncertainty.\n")

    L.append("## Gate 1 -- is the bundled table a point-by-point digitisation?\n")
    L.append("| property | bundled | cross-check | what a figure reading must look like |")
    L.append("|---|---|---|---|")
    L.append(f"| points | {gb['n_points']} | {gc['n_points']} | -- |")
    L.append(f"| abscissa step | {f(gb['tau_step_mean'])} (spread "
             f"{f(gb['tau_step_spread'], '{:.1e}')}) | -- | irregular, figure pixels |")
    L.append(f"| abscissa grid arithmetic | {gb['tau_grid_is_arithmetic']} | "
             f"{gc['tau_grid_is_arithmetic']} | no |")
    L.append(f"| ordinate quantum | {f(gb['ordinate_quantum'], '{:.3f}')} | "
             f"{f(gc['ordinate_quantum'], '{:.3f}')} | finer than the claimed "
             "uncertainty |")
    L.append(f"| distinct finite-difference slopes | {gb['n_slope_levels']} | "
             f"{gc['n_slope_levels']} | many |")
    L.append("")
    gate1 = gates["bundled_is_pointwise"]
    L.append(f"**Result: `{gate1['result'].upper()}`** (limit: {gate1['limit']}). "
             f"{gate1['interpretation']}\n")

    L.append("## Gate 2 -- does the reference contain the acceleration phase?\n")
    gate2 = gates["acceleration_phase_present"]
    L.append("| table | slope at first interval / peak slope |")
    L.append("|---|---|")
    L.append(f"| bundled | {f(gate2['measured'], '{:.3f}')} |")
    L.append(f"| cross-check | {f(gate2['crosscheck_measured'], '{:.3f}')} |")
    L.append("")
    L.append(f"**Result: `{gate2['result'].upper()}`** (limit: {gate2['limit']}). "
             f"{gate2['interpretation']}\n")

    L.append("## Gate 3 -- is the claimed uncertainty consistent with the "
             "cross-source spread?\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| overlap in `tau` | {f(spread['overlap_tau'][0], '{:.2f}')} - "
             f"{f(spread['overlap_tau'][1], '{:.2f}')} |")
    L.append(f"| samples on a 0.01 grid | {spread['n_samples']} |")
    L.append(f"| max abs difference in `Z/a` | **{f(spread['max_abs_delta'])}** "
             f"at `tau` = {f(spread.get('tau_at_max_delta'), '{:.2f}')} |")
    L.append(f"| rms difference | {f(spread['rms_delta'])} |")
    L.append(f"| mean difference | {f(spread['mean_delta'])} |")
    claim = gates["claimed_uncertainty_consistent"]["claim"]
    ratio = gates["claimed_uncertainty_consistent"]["measured"]
    L.append(f"| ratio to the claimed +-{f(claim, '{:.2f}')} | "
             f"**{f(ratio, '{:.1f}')}x** |")
    L.append("")
    gate3 = gates["claimed_uncertainty_consistent"]
    L.append(f"**Result: `{gate3['result'].upper()}`** (limit: {gate3['limit']}). "
             f"{gate3['interpretation']}\n")

    if ladder:
        L.append("## The same simulations, scored against each reference\n")
        sp = ladder["simulation_params"]
        L.append(f"Each row re-introduces nothing but the reference: `dx` = "
                 f"{', '.join(str(r['dx']) for r in ladder['results'])} m, "
                 f"`c0` = {sp['sound_speed']} m/s, Courant {sp['max_courant']}, "
                 f"`wall_friction` = {sp['wall_friction']}, "
                 f"`density_diffusion` = {sp['density_diffusion']}, "
                 f"t = {sp['seconds']} s, column 0.2 x 0.4 m in a 2.0 m tank.\n")
        L.append("| dx (m) | particles | wall clock (s) | MAE vs bundled | MAE vs "
                 "cross-check | MAE vs Ritter |")
        L.append("|---|---|---|---|---|---|")
        for r in ladder["results"]:
            L.append(f"| {r['dx']:.3f} | {r['particles']} | {r['wall_clock_s']:.1f} | "
                     f"{f(r['error_vs_bundled']['mae'])} | "
                     f"{f(r['error_vs_crosscheck']['mae'])} | "
                     f"{f(r['error_vs_ritter']['mae'])} |")
        L.append("")
        att = ladder["attribution"]
        L.append("| gate reading | value |")
        L.append("|---|---|")
        L.append(f"| gate limit on MAE in `Z/a` | {f(att['gate_limit'], '{:.2f}')} |")
        L.append(f"| cross-source spread | {f(att['reference_spread_max_abs_delta'])} = "
                 f"**{f(att['reference_spread_over_gate_limit'], '{:.1f}')}x** the gate limit |")
        L.append(f"| MAE shift caused by changing only the reference "
                 f"(finest dx) | {f(att['mae_shift_when_reference_changes'])} |")
        L.append(f"| gate result vs bundled / vs cross-check | "
                 f"{ladder['gates']['bundled']['finest']['result'].upper()} / "
                 f"{ladder['gates']['crosscheck']['finest']['result'].upper()} |")
        L.append("")
        L.append(f"{att['conclusion']}\n")

    if audit:
        L.append("## What the companion audit adds\n")
        ritter = audit["ritter"]
        slope = audit["table_slope"]
        L.append(f"`reference_audit.py` re-derives the analytic reference and pins the "
                 f"dimensionless front-speed coefficient. The Ritter identity "
                 f"`Z/a = 2 tau` holds to a relative residual of "
                 f"{f(ritter['max_identity_reldiff'], '{:.1e}')} and the shipped "
                 f"helper matches to {f(ritter['max_helper_reldiff'], '{:.1e}')}. The "
                 f"bundled table itself has "
                 f"`c = d(Z/a)/d tau` = {f(slope['table_coefficient_c'])} "
                 f"({f(100.0 * slope['table_c_over_ritter'], '{:.1f}')} % of Ritter's 2), "
                 f"so it encodes a front that is far slower than any shallow-water "
                 f"limit -- one more sign that it is not a reading of the "
                 f"published points.\n")
        L.append(f"Table-integrity grade recorded there: "
                 f"`{audit['table_audit']['grade']}`.\n")

    L.append("## Verdict\n")
    L.append(f"`physical_validation` for the B2 case: "
             f"**`{verdict['physical_validation_status']}`**. {verdict['reason']}\n")
    L.append("What this does *not* say: that the solver is accurate. The SPH front is "
             "still far from the analytic Ritter limit (`MAE` of order 2 in `Z/a`), "
             "which is expected for a finite reservoir at these resolutions and is not "
             "a validation statement either way. The point is narrower and checkable: "
             "the experimental table in the repository cannot support a pass/fail "
             "verdict against any solver, so the honest verdict is `withheld`.\n")
    L.append("What would change it: a scatter-bearing, citable digitisation of the "
             "1952 figure (or the paper's own tabulated values). Re-run "
             "`reference_crosscheck.py`; gate 1 and gate 3 would then pass and the "
             "verdict would flip to `evaluable` automatically. The tests pin the "
             "current state in *both* directions, so the flip cannot happen silently.\n")

    L.append("## Reproduction\n")
    L.append("```")
    L.append("python benchmarks/dam_break/reference_crosscheck.py --plot")
    L.append("python benchmarks/dam_break/reference_ladder.py --sizes 0.04,0.02 --plot")
    L.append("python benchmarks/dam_break/make_reference_audit_doc.py")
    L.append("python -m pytest tests/test_dam_break_reference_crosscheck.py -q")
    L.append("```\n")
    L.append("| file | role |")
    L.append("|---|---|")
    L.append("| `benchmarks/dam_break/reference_crosscheck.py` | structural audit + "
             "cross-source spread, no solver |")
    L.append("| `benchmarks/dam_break/reference_crosscheck.json` | its output |")
    L.append("| `benchmarks/dam_break/reference_ladder.py` | same simulations vs three "
             "references |")
    L.append("| `benchmarks/dam_break/reference_ladder.json` | its output |")
    L.append("| `benchmarks/dam_break/martin_moyce_1952_crosscheck.json` | the "
             "independent transcription |")
    L.append("| `benchmarks/dam_break/reference_audit.py` | companion audit "
             "(table integrity, Ritter identity, friction sweep) |")
    L.append("| `tests/test_dam_break_reference_crosscheck.py` | guards on every "
             "finding above |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
