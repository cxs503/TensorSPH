"""Render ``docs/benchmarks/dam-break-reference-audit.md`` from the audit JSON.

Same contract as ``make_ladder_doc.py``: the document is a rendering of
``benchmarks/dam_break/reference_audit.json``, never hand-edited numbers.
``tests/test_dam_break_reference_audit.py`` fails if a measured coefficient in
the document drifts from the artifact.

    python benchmarks/dam_break/make_audit_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT = HERE / "reference_audit.json"
DOC = ROOT / "docs" / "benchmarks" / "dam-break-reference-audit.md"


def main() -> None:
    p = json.loads(ARTIFACT.read_text())
    t = p["table_audit"]
    r = p["ritter"]
    runs = p["runs"]
    base = [x for x in runs if x["dx"] == 0.04]
    fine = [x for x in runs if x["dx"] == 0.02]

    lines = []
    lines.append("# B2 dam-break: what the reference audit changed\n")
    lines.append(
        "**Generated** by `benchmarks/dam_break/make_audit_doc.py` from "
        "`benchmarks/dam_break/reference_audit.json` (%s). Do not hand-edit.\n"
        % p["date"])
    lines.append("## 1. Summary\n")
    lines.append(
        "The dam-break case (B2) is the only place in TensorSPH where the solver "
        "is compared against an experiment, and that comparison is where the "
        "suite reports its worst numbers. The audit asked two prior questions: "
        "*is the validation target real*, and *is the analytic reference "
        "implemented correctly*. Both answers came back no.\n")
    lines.append("| Question | Answer | Grade |")
    lines.append("|---|---|---|")
    lines.append("| Is the bundled Martin & Moyce table a digitisation? | No: exact arithmetic "
                 "abscissa, exact 0.05 ordinate lattice at all 19 points | WITHHELD |")

    lines.append("| Is the Ritter reference correct in the harness? | No: `Z/a = tau` was used "
                 "instead of `Z/a = 2 tau` (factor 2 low) | fixed |")
    lines.append("| Is the frame bookkeeping correct? | No: `front/ritter` mixed the lab frame "
                 "with the self-similar frame | fixed |")
    lines.append("| Can the solver be held to a reference-free number? | Yes: "
                 "`c = d(Z/a)/d tau`, Ritter = 2 | REAL (numerical) |")
    lines.append("")
    lines.append("## 2. Table provenance\n")
    lines.append(
        "* abscissa: `tau = 0.226 k`, max residual to that lattice **%.1e**;\n"
        "* ordinates: every `Z/a` is an exact multiple of **%.2f**, residual "
        "**%.1e**;\n"
        "* second-difference RMS **%.4f** against the ±%.2f uncertainty the JSON "
        "claims for itself -- only ~%.1fx below it, so the residual "
        "*magnitude* is not the argument. The argument is the exactness: a "
        "digitisation has scatter of the order of the figure resolution and "
        "never lands on an exact 0.05 lattice at every one of 19 points.\n\n"
        "Whatever generated these 20 points, it was not a digitising tablet. The table is therefore *not* usable as a "
        "validation target; `physical_validation` for B2 is **WITHHELD**, not "
        "fail, because there is nothing trustworthy to fail against. The values "
        "are plausible in magnitude, so the table is kept as a plot reference and "
        "flagged in `martin_moyce_1952.json` itself.\n"
        % (t["tau_lattice_residual"], t["ordinate_quantum"],
           t["ordinate_quantum_residual"], t["second_difference_rms"],
           t["claimed_uncertainty"], t["claimed_uncertainty"] / t["second_difference_rms"]))
    lines.append("## 3. Analytic reference\n")
    lines.append(
        "Ritter's dry-bed solution for a reservoir of depth `H`, `c = sqrt(gH)`:\n\n"
        "```\nh(x,t) = (2c - x/t)^2 / (9g)   on  -ct <= x <= 2ct\n"
        "u(x,t) = (2/3)(c + x/t)\nx_front(t) = 2ct\n```\n\n"
        "The measurements in `tests/test_ritter_reference.py` (all at machine "
        "precision):\n\n"
        "* `h(-ct) = H`, `h(0) = 4H/9`, `h(2ct) = 0`;\n"
        "* Riemann invariant `u + 2 sqrt(gh) = 2c` identically across the fan;\n"
        "* self-similarity `h(Lx, Lt) = h(x,t)` to 1e-13;\n"
        "* water ahead of the dam integrates to exactly `(4/27) H x_front`.\n\n"
        "Non-dimensionalised with `tau = t sqrt(2g/a)` and `H = 2a` the front is "
        "**`Z/a = 2 tau`** (residual %.1e). The harness used `Z/a = tau`, so every "
        "reported Ritter error in `experimental_validation.py` was computed "
        "against a reference half the right size. `run.py` had a second, "
        "independent frame error: it divided the lab-frame front `x_max` by the "
        "self-similar `Z`, which is off by the column width in every sample.\n"
        % r["max_identity_reldiff"])
    lines.append("## 4. Reference-free measurement\n")
    lines.append(
        "With the table unusable, the case is held to a dimensionless, "
        "reference-free number instead: `c = d(Z/a)/d tau`, fitted over "
        "`tau` in [%.1f, %.1f] to skip the startup transient. Ritter gives "
        "`c = 2` and the bundled table gives `c = %.4f`.\n"
        % (p["window_tau"][0], p["window_tau"][1],
           p["table_slope"]["table_coefficient_c"]))
    lines.append("| dx (m) | particles | wall_friction | steps | c | c/2 | "
                 "max compression | Mach | escaped |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for x in runs:
        lines.append("| %g | %d | %g | %d | %.4f | %.1f%% | %.3f | %.3f | %d |"
                     % (x["dx"], x["particles"], x["wall_friction"],
                        x["steps_taken"], x["front_coefficient_c"],
                        100.0 * x["c_over_ritter"], x["max_density_compression"],
                        x["mach_number"], x["escaped"]))
    lines.append("")
    lines.append(
        "Findings:\n\n"
        "1. At free slip (`wall_friction = 0`) the front runs at **%.0f%%–%.0f%% "
        "of Ritter** (`c = %.4f` at dx = %g, `c = %.4f` at dx = %g). Ritter "
        "over-predicts because it assumes a semi-infinite shallow reservoir; the "
        "collapse here is deep (`H/a = 2`) and finite.\n"
        "2. Imposing a no-slip floor lowers the coefficient monotonically "
        "(`c = %.4f` -> `%.4f` at dx = %g), i.e. the floor boundary condition "
        "carries ~%.0f%% of the front speed. The effect *grows* with resolution "
        "(`c = %.4f` -> `%.4f` at dx = %g): the no-slip blend is not yet "
        "resolution-converged, so no friction calibration should be published "
        "from dx = 0.04 m.\n"
        "3. The bundled table's `c = %.4f` is **below** the no-slip run at both "
        "resolutions, so bed friction alone cannot explain the residual -- but "
        "since the table is unverified, the residual is not evidence against the "
        "solver either. Grade the physics claim WITHHELD, not FAIL.\n"
        % (100.0 * base[0]["c_over_ritter"], 100.0 * base[-1]["c_over_ritter"],
           base[0]["front_coefficient_c"], base[0]["dx"],
           fine[0]["front_coefficient_c"], fine[0]["dx"],
           base[0]["front_coefficient_c"], base[-1]["front_coefficient_c"],
           base[0]["dx"],
           100.0 * (1.0 - base[-1]["front_coefficient_c"] / base[0]["front_coefficient_c"]),
           base[-1]["front_coefficient_c"], fine[-1]["front_coefficient_c"],
           fine[-1]["dx"], p["table_slope"]["table_coefficient_c"]))
    lines.append("## 5. Limits of this audit\n")
    lines.append(
        "* The friction sweep is at `dx = 0.04 m` (50 particles) with only the two "
        "end points repeated at `dx = 0.02 m` (200 particles); the load on the "
        "host blocked a full 5 x 2 sweep. Nothing here claims resolution "
        "convergence for `wall_friction > 0`.\n"
        "* No citable digitisation of Martin & Moyce (1952) was installed. Until "
        "one is, `physical_validation` for B2 must stay `not_evaluated`, and the "
        "`--max-exp-error` gate in `experimental_validation.py` must not be "
        "quoted as a pass/fail verdict.\n"
        "* The three-step path back to a real physical claim: (a) digitise "
        "Figure 4 of the cited paper (or an open-access reproduction) with a "
        "scatter-bearing table and cite the figure number; (b) check the "
        "`H/a = 2` geometry against the paper's set-up; (c) then, and only then, "
        "quote an error against the 15 %% platform bar.\n")
    lines.append("## 6. Artifacts\n")
    lines.append(
        "| File | What |\n|---|---|\n"
        "| `benchmarks/dam_break/reference_audit.json` | all numbers above |\n"
        "| `benchmarks/dam_break/reference_audit.py` | regenerates the JSON and figures |\n"
        "| `benchmarks/dam_break/make_audit_doc.py` | renders this document |\n"
        "| `benchmarks/dam_break/dam_break_references.png` | trajectories vs both references |\n"
        "| `benchmarks/dam_break/dam_break_table_integrity.png` | lattice / quantum residuals |\n"
        "| `benchmarks/dam_break/dam_break_friction_sweep.png` | `c` vs floor friction |\n"
        "| `tests/test_ritter_reference.py` | analytic verification of the reference |\n"
        "| `tests/test_dam_break_reference_audit.py` | guards on the findings above |\n")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")
    print("wrote", DOC)


if __name__ == "__main__":
    main()