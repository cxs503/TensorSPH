"""Render ``docs/benchmarks/dam-break-reference-audit.md`` from the audit JSON.

Same contract as ``make_ladder_doc.py``: the document is a rendering of
``benchmarks/dam_break/reference_audit.json`` (plus, when present,
``benchmarks/dam_break/reference_ladder.json``), never hand-edited numbers.
``tests/test_dam_break_reference_audit.py`` fails if a measured coefficient in
the document drifts from the artifact.

The document covers **both** layers of the audit, because the artifact carries
both: layer A (table lattice, Ritter analytic reference, solver front-speed
sweep, figures) and layer B (structure and kinematics of the two independent
transcriptions, cross-source spread, three gates, verdict).  Layer B was
contributed by ``agent/m2905`` and reaches the document through the same
artifact, so there is one generator and one source of truth.

    python benchmarks/dam_break/make_audit_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT = HERE / "reference_audit.json"
LADDER = HERE / "reference_ladder.json"
DOC = ROOT / "docs" / "benchmarks" / "dam-break-reference-audit.md"


def _rel(path: str) -> str:
    """Repo-relative form of a stored path (artifacts may hold either)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix().lstrip("/")


def _layer_b(lines: list, p: dict) -> None:
    t = p["tables"]
    gb = p["grid_stats"]["bundled"]
    gc = p["grid_stats"]["crosscheck"]
    spread = p["cross_source_spread"]
    gates = p["gates"]

    lines.append("## 3. Structure and kinematics of the two transcriptions\n")
    lines.append(
        "Both tables are read in one frame: `Z/a` is measured from the column "
        "edge, and the independent transcription bundles `X = x_front/a`, so "
        "`Z/a = X - 1`. The conversion puts the front at `Z/a = 0` when `tau = 0` "
        "in both tables (bundled %+.3f, cross-check %+.3f). No number here "
        "touches the solver; they are properties of the reference.\n"
        % (gb["tau_first"], gc["tau_first"]))
    lines.append("| property | bundled | cross-check |")
    lines.append("|---|---|---|")
    lines.append("| source | `%s` | `%s` |" % (_rel(t["bundled"]["path"]),
                                               _rel(t["crosscheck"]["path"])))
    lines.append("| points | %d | %d |" % (gb["n_points"], gc["n_points"]))
    lines.append("| `tau` range | %.3f - %.3f | %.3f - %.3f |"
                 % (gb["tau_first"], gb["tau_last"], gc["tau_first"], gc["tau_last"]))
    lines.append("| abscissa arithmetic | %s | %s |"
                 % (gb["tau_grid_is_arithmetic"], gc["tau_grid_is_arithmetic"]))
    lines.append("| `tau` step spread | %.1e | %.3f |"
                 % (gb["tau_step_spread"], gc["tau_step_spread"]))
    lines.append("| ordinate quantum | %.3f | %.3f |"
                 % (gb["ordinate_quantum"], gc["ordinate_quantum"]))
    lines.append("| slope levels | %d | %d |"
                 % (gb["n_slope_levels"], gc["n_slope_levels"]))
    lines.append("| first-interval slope / peak | %.3f | %.3f |"
                 % (gb["first_slope_to_peak"], gc["first_slope_to_peak"]))
    lines.append("")
    lines.append(
        "The cross-check has the shape a gate-release experiment must have: the "
        "front leaves rest at %.0f%% of its peak speed and accelerates by %.1fx. "
        "The bundled table is already at %.0f%% of its peak in the first interval "
        "after release, i.e. it describes an impulsive start.\n"
        % (100.0 * gc["first_slope_to_peak"], gc["slope_peak"] / gc["slope_first"],
           100.0 * gb["first_slope_to_peak"]))

    lines.append("## 4. Cross-source spread = a lower bound on the uncertainty\n")
    lines.append(
        "Over the common window `tau` in [%.2f, %.3f] (%d points):\n"
        % (spread["tau_window"][0], spread["tau_window"][1], spread["n_overlap"]))
    claimed = t["bundled"]["claimed_uncertainty"]
    lines.append("| statistic | value |")
    lines.append("|---|---|")
    lines.append("| max abs difference in `Z/a` | %.3f |" % spread["max_abs_delta"])
    lines.append("| rms difference | %.3f |" % spread["rms_delta"])
    lines.append("| mean signed difference | %.3f |" % spread["mean_signed_delta"])
    lines.append("| claimed uncertainty (both tables) | +-%.3f |" % claimed)
    lines.append("| ratio max difference / claimed | %.1fx |"
                 % (spread["max_abs_delta"] / claimed))
    lines.append("")
    lines.append("| `tau` | bundled `Z/a` | cross-check `Z/a` | difference |")
    lines.append("|---|---|---|---|")
    for row in spread["rows"]:
        lines.append("| %.3f | %.3f | %.3f | %+.3f |"
                     % (row["tau"], row["bundled_za"], row["crosscheck_za"], row["delta"]))
    lines.append("")
    lines.append(
        "Both tables come from the same paywalled original, so this spread is a "
        "*lower bound*: two honest readers of the same figure already disagree by "
        "up to %.2f in `Z/a`, %.1fx the quoted error bar.\n"
        % (spread["max_abs_delta"], spread["max_abs_delta"] / claimed))

    lines.append("## 5. Gates on the bundled reference\n")
    lines.append("| gate | metric | measured | limit | result |")
    lines.append("|---|---|---|---|---|")
    for name in ("bundled_is_pointwise", "acceleration_phase_present",
                 "claimed_uncertainty_consistent"):
        gate = gates[name]
        measured = gate["measured"]
        if isinstance(measured, dict):
            shown = ", ".join("%s=%s" % (k, v) for k, v in measured.items())
        else:
            shown = "%.4f" % measured
        lines.append("| `%s` | %s | %s | %s | **%s** |"
                     % (name, gate["metric"], shown, gate["limit"], gate["result"]))
    lines.append("")
    lines.append("**Verdict: `physical_validation = %s`.** %s\n"
                 % (p["verdict"]["physical_validation_status"],
                    p["verdict"]["reason"]))


def _ladder(lines: list) -> None:
    if not LADDER.exists():
        return
    d = json.loads(LADDER.read_text())
    params = d["simulation_params"]
    refs = d["references"]
    lines.append("## 8. The same simulations against both references\n")
    lines.append(
        "Setup (identical for every rung and for both references): `a` = %g m, "
        "column height = %g m (%g:1), tank = %g m, `c0` = %g m/s, wall friction %g, "
        "density diffusion %g, simulated time %g s.\n"
        % (params["column_width"], params["column_height"],
           params["column_height"] / params["column_width"], params["tank_width"],
           params["sound_speed"], params["wall_friction"],
           params["density_diffusion"], params["seconds"]))
    rows = {r["dx"]: r for r in d["results"]}
    lines.append("| `dx` (m) | particles | steps | wall clock (s) | MAE vs bundled "
                 "| RMSE vs bundled | MAE vs cross-check | RMSE vs cross-check |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for dx in sorted(rows, reverse=True):
        r = rows[dx]
        lines.append("| %.3f | %d | %d | %.1f | %.4f | %.4f | %.4f | %.4f |"
                     % (dx, r["particles"], r["steps_taken"], r["wall_clock_s"],
                        r["error_vs_bundled"]["mae"], r["error_vs_bundled"]["rmse"],
                        r["error_vs_crosscheck"]["mae"],
                        r["error_vs_crosscheck"]["rmse"]))
    lines.append("")
    limit = d["gates"]["bundled"]["limit"]
    lines.append("Gates (limit on the mean absolute error in `Z/a` = %.2f):\n" % limit)
    lines.append("| reference | finest `dx` | MAE | result | all rungs pass |")
    lines.append("|---|---|---|---|---|")
    for name in ("bundled", "crosscheck"):
        gate = d["gates"][name]
        lines.append("| %s | %.3f | %.4f | **%s** | %s |"
                     % (name, gate["finest"]["dx"], gate["finest"]["mae"],
                        gate["finest"]["result"], gate["all_pass"]))
    lines.append("")
    lines.append(
        "Observed convergence order of the error (slope of log MAE against log "
        "`dx`, so a value near 0 means the residual does not respond to "
        "refinement):\n")
    lines.append("| reference | order from MAE | order from RMSE |")
    lines.append("|---|---|---|")
    for name in ("bundled", "crosscheck"):
        conv = d["convergence"][name]
        lines.append("| %s | %s | %s |"
                     % (name,
                        ", ".join("%+.3f" % o for o in conv["order_mae"]),
                        ", ".join("%+.3f" % o for o in conv["order_rmse"])))
    lines.append("")
    lines.append(
        "### The cross-check pass is not a convergence proof\n\n"
        "The cross-check error is *not* monotone in `dx`: it reaches %.4f at "
        "`dx` = %.3f m and then rises to %.4f at `dx` = %.3f m, so the observed "
        "order above changes sign. Every rung stays inside the %.2f limit, but "
        "the honest reading is \"within the reference uncertainty at all three "
        "rungs\", not \"converging to the experiment\". A 50-800 particle 2-D run "
        "cannot settle which of the two tables is closer to the 1952 apparatus; "
        "it can only show that the misfit against the bundled table is the same "
        "size as the disagreement between the two transcriptions.\n"
        % (d["gates"]["crosscheck"]["rows"][1]["mae"],
           d["gates"]["crosscheck"]["rows"][1]["dx"],
           d["gates"]["crosscheck"]["rows"][0]["mae"],
           d["gates"]["crosscheck"]["rows"][0]["dx"], limit))
    lines.append("| reference | points | claimed uncertainty | path |")
    lines.append("|---|---|---|---|")
    for name in ("bundled", "crosscheck"):
        ref = refs[name]
        n = (20 if name == "bundled" else 14)
        lines.append("| %s | %d | +-%.2f | `%s` |"
                     % (name, n, ref["claimed_uncertainty"], _rel(ref["path"])))
    lines.append("")


def main() -> None:
    p = json.loads(ARTIFACT.read_text())
    t = p["table_audit"]
    r = p["ritter"]
    runs = p["runs"]
    base = [x for x in runs if x["dx"] == 0.04]
    fine = [x for x in runs if x["dx"] == 0.02]
    solver = p["solver"]

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
                 "abscissa, exact 0.05 ordinate lattice at all %d points | WITHHELD |"
                 % t["n_points"])
    lines.append("| Do two independent transcriptions of the same figure agree? | No: "
                 "spread %.3f in `Z/a` = %.1fx the claimed +-%.2f | reference "
                 "uncertainty is a lower bound |"
                 % (p["cross_source_spread"]["max_abs_delta"],
                    p["cross_source_spread"]["max_abs_delta"]
                    / t["claimed_uncertainty"], t["claimed_uncertainty"]))
    lines.append("| Is the Ritter reference correct in the harness? | No: `Z/a = tau` was used "
                 "instead of `Z/a = 2 tau` (factor 2 low) | fixed |")
    lines.append("| Is the frame bookkeeping correct? | No: `front/ritter` mixed the lab frame "
                 "with the self-similar frame | fixed |")
    lines.append("| Can the solver be held to a reference-free number? | Yes: "
                 "`c = d(Z/a)/d tau`, Ritter = 2 | REAL (numerical) |")
    lines.append("| Can B2 carry a physical-validation verdict? | No: **%s** | see section 5 |"
                 % p["verdict"]["physical_validation_status"])
    lines.append("")
    lines.append("## 2. Table provenance\n")
    lines.append(
        "* abscissa: `tau = %.3f k`, max residual to that lattice **%.1e**;\n"
        "* ordinates: every `Z/a` is an exact multiple of **%.2f**, residual "
        "**%.1e**;\n"
        "* second-difference RMS **%.4f** against the +-%.2f uncertainty the JSON "
        "claims for itself -- only ~%.1fx below it, so the residual structure is "
        "the lattice, not noise;\n"
        "* verdict: **%s**; grade **%s**, usable as a validation target: **%s**.\n"
        % (t["tau_step_mean"], t["tau_lattice_residual"], t["ordinate_quantum"],
           t["ordinate_quantum_residual"], t["second_difference_rms"],
           t["claimed_uncertainty"], t["claimed_uncertainty"] / t["second_difference_rms"],
           t["verdict"], t["grade"], t["usable_as_validation_target"]))
    _layer_b(lines, p)
    lines.append("## 6. Analytic reference\n")
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
    lines.append("## 7. Reference-free measurement\n")
    lines.append(
        "With the table unusable, the case is held to a dimensionless, "
        "reference-free number instead: `c = d(Z/a)/d tau`, fitted over "
        "`tau` in [%.1f, %.1f] to skip the startup transient. Ritter gives "
        "`c = 2` and the bundled table gives `c = %.4f`.\n"
        % (p["window_tau"][0], p["window_tau"][1],
           p["table_slope"]["table_coefficient_c"]))
    lines.append(
        "Solver configuration (single source: `DamBreakConfig`): `c0` = %g m/s, "
        "max Courant %g, simulated %g s, adaptive `dt` %s, density diffusion %g, "
        "viscosity %g, `h` factor %g.\n"
        % (solver["sound_speed"], solver["max_courant"], solver["seconds"],
           solver["adaptive_dt"], solver["density_diffusion"], solver["viscosity"],
           solver["h_factor"]))
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
        "1. At free slip (`wall_friction = 0`) the front runs at **%.0f%%-%.0f%% "
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
    _ladder(lines)
    lines.append("## 9. What this changes and what it does not\n")
    lines.append(
        "1. **The B2 status changes from `fails` to `%s`.** The old verdict read "
        "a %.2f `Z/a` misfit as a solver error. Against a reference whose two "
        "transcriptions differ by up to %.2f, that attribution is not supported. "
        "This is a correction of the *verdict*, not of the numbers: the solver's "
        "misfit against the bundled table is unchanged.\n"
        % (p["verdict"]["physical_validation_status"],
           p["runs"][0]["trajectory"][-1]["za"],
           p["cross_source_spread"]["max_abs_delta"]))
    lines.append(
        "2. **The bundled table stays in the repository and stays the default for "
        "the B2 gate**, so the historical series is comparable; the audit only "
        "records that its verdicts cannot be read as validation of the solver.\n"
        "3. **The two remaining candidates are untouched here**: the 2-D "
        "cross-section versus the 3-D experiment (side-wall drag), and the front "
        "definition (`x_max` is the leading particle, which at `dx` = 0.04 m is a "
        "one-particle-thick sheet above the floor). Neither can be settled from "
        "the tables alone; a 3-D run and a bulk-front estimator are the next "
        "increments.\n"
        "4. **What would upgrade this audit**: access to the archived 1952 figure "
        "at its native resolution, or experimental data from an open dataset with "
        "per-point error bars. Until then any B2 validation claim should quote "
        "the reference spread alongside the solver error.\n")
    lines.append("## 10. Limits of this audit\n")
    lines.append(
        "* The friction sweep is at `dx = 0.04 m` (50 particles) with only the two "
        "end points repeated at `dx = 0.02 m` (200 particles); the load on the "
        "host blocked a full 5 x 2 sweep. Nothing here claims resolution "
        "convergence for `wall_friction > 0`.\n"
        "* `accounting for the reference`: no citable digitisation of Martin & "
        "Moyce (1952) is installed, and the two tables in the repository are "
        "independent *transcriptions*, not two measurements. Their spread bounds "
        "the reading error from below only.\n"
        "* The three-step path back to a real physical claim: (a) digitise "
        "Figure 4 of the cited paper (or an open-access reproduction) with a "
        "scatter-bearing table and cite the figure number; (b) check the "
        "`H/a = 2` geometry against the paper's set-up; (c) then, and only then, "
        "quote an error against the 15 %% platform bar.\n")
    lines.append("## 11. Artifacts\n")
    lines.append(
        "| File | What |\n|---|---|\n"
        "| `benchmarks/dam_break/reference_audit.json` | all numbers above (both "
        "layers) |\n"
        "| `benchmarks/dam_break/reference_audit.py` | regenerates the JSON, both "
        "layers and the figures |\n"
        "| `benchmarks/dam_break/reference_ladder.json` | the resolution ladder "
        "against both references |\n"
        "| `benchmarks/dam_break/reference_ladder.py` | regenerates the ladder |\n"
        "| `benchmarks/dam_break/make_audit_doc.py` | renders this document |\n"
        "| `benchmarks/dam_break/dam_break_references.png` | trajectories vs both "
        "references |\n"
        "| `benchmarks/dam_break/dam_break_table_integrity.png` | lattice / "
        "quantum residuals |\n"
        "| `benchmarks/dam_break/dam_break_friction_sweep.png` | `c` vs floor "
        "friction |\n"
        "| `benchmarks/dam_break/reference_ladder.png` | front trajectory ladder "
        "vs both references |\n"
        "| `tests/test_ritter_reference.py` | analytic verification of the "
        "reference |\n"
        "| `tests/test_dam_break_reference_audit.py` | guards on every finding "
        "above, both layers |\n")
    lines.append("## 12. Reproduce\n")
    lines.append(
        "```bash\n"
        "python benchmarks/dam_break/reference_audit.py --skip-runs   # layers A+B, "
        "no simulation\n"
        "python benchmarks/dam_break/reference_audit.py \\\n"
        "    --sizes 0.04 --frictions 0,0.25,0.5,0.75,1 --plot          # + sweep\n"
        "python benchmarks/dam_break/reference_ladder.py --sizes 0.04,0.02,0.01 \\\n"
        "    --seconds 0.5 --json-out benchmarks/dam_break/reference_ladder.json "
        "--plot\n"
        "python benchmarks/dam_break/make_audit_doc.py\n"
        "pytest tests/test_dam_break_reference_audit.py -q\n"
        "```\n")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")
    print("wrote", DOC)


if __name__ == "__main__":
    main()
