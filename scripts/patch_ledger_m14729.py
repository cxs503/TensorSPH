"""Insert the m14729 coverage-closure claim block into build_ledger.py.

Idempotent: does nothing when the block is already present.
"""
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "benchmarks/verification_ledger/build_ledger.py"
MARK = "def coverage_closure_claims()"

BLOCK = '''
# ---------------------------------------------------------------------------
# Coverage closure (m14729)
# ---------------------------------------------------------------------------
#: Readings that lived in committed artifacts the ledger could not see, plus the
#: harnesses that have no artifact at all.  The point of registering them is that
#: a regression in any of these harnesses used to be invisible: nothing graded
#: them, so nothing could fail.
def coverage_closure_claims() -> list[dict]:
    out: list[dict] = []

    # ---- aeration benchmark (bubble rise / air entrainment / hydraulic jump) ----
    rel = "benchmarks/aeration/report.json"
    d = _load(rel)
    crit = d["criteria"]
    verdict = "PASS" if d["all_criteria_passed"] else "FAIL"
    out.append(
        _entry(
            claim_id="aeration.void_fraction.erf_anchor_l2",
            quantity_id="aeration_void_fraction_profile",
            quantity="steady void-fraction profile of the aerated hydraulic jump vs the erf anchor",
            artifact=rel,
            json_path="criteria.C2.measured",
            reference=(0.0, "-", "exact erf anchor profile (zero residual)"),
            measured=(
                crit["C2"]["measured"],
                "-",
                "relative L2 difference between the measured and the anchor profile",
            ),
            # The metric *is* a relative error against the anchor, so it is entered
            # directly instead of being normalised by its own acceptance limit.
            rel_error_pct=crit["C2"]["measured"] * 100.0,
            claim_type="gate",
            gate_result="pass" if crit["C2"]["passed"] else "fail",
            gate_limit=crit["C2"]["threshold"],
            guarding_test="tests/test_aeration.py",
            artifact_status=verdict,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Registered because the harness was invisible to the ledger "
                "(uncovered_benchmark). 9.5 % against a 15 % limit is MARGINAL, not a "
                "pass: the anchor is an erf fit of the void-fraction profile, not "
                "experimental data."
            ),
        )
    )
    out.append(
        _entry(
            claim_id="aeration.air_volume_drift",
            quantity_id="aeration_air_mass_conservation",
            quantity="air volume drift per characteristic time of the aerated jump",
            artifact=rel,
            json_path="criteria.C5.measured",
            reference=(0.0, "-", "exact conservation of entrained air volume"),
            measured=(
                crit["C5"]["measured"],
                "-",
                "relative air-volume drift per characteristic time",
            ),
            rel_error_pct=crit["C5"]["measured"] * 100.0,
            guarding_test="tests/test_aeration.py",
            artifact_status=verdict,
            backlog_ref="docs/verification_ledger.md",
            notes="0.37 % drift over the characteristic time; conservation reading only.",
        )
    )
    out.append(
        _entry(
            claim_id="aeration.inception_constant.self_referential",
            quantity_id="aeration_inception_froude",
            quantity="inception Froude constant of air entrainment vs Chanson",
            artifact=rel,
            json_path="inception.constant_rel_error",
            reference=(
                d["inception"]["empirical_constant"],
                "-",
                "Chanson empirical criterion constant (input to the criterion)",
            ),
            measured=(
                d["inception"]["derived_constant"],
                "-",
                "constant the harness derives from its own onset sweep",
            ),
            rel_error_pct=d["inception"]["constant_rel_error"] * 100.0,
            gauge_degenerate=True,
            guarding_test="tests/test_aeration.py",
            artifact_status=verdict,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Exactly 0.0 % because the derived constant is the empirical constant "
                "rearranged: the criterion is checked against itself, so the reading "
                "is WITHHELD regardless of the artifact's all_criteria_passed = true. "
                "The same degeneracy hits C4 (bubble rise velocity vs Mendelson), which "
                "reports 0.0 % because the module *implements* the Mendelson formula; "
                "that reading is deliberately not registered as a claim."
            ),
        )
    )

    # ---- Rayleigh-Plateau ladder (analytic dispersion + 3-D column run) ----
    rel = "benchmarks/rayleigh_plateau/report.json"
    d = _load(rel)
    an = d["analytic"]
    out.append(
        _entry(
            claim_id="rayleigh_plateau.analytic.kR_star",
            quantity_id="rayleigh_plateau_most_unstable_kR",
            quantity="most unstable wavenumber of the Rayleigh-Plateau instability (kR)*",
            artifact=rel,
            json_path="analytic.most_unstable_kR",
            reference=(
                an["most_unstable_kR_literature"],
                "-",
                "Rayleigh (1879) / literature value (kR)* = 0.697",
            ),
            measured=(
                an["most_unstable_kR"],
                "-",
                "root of the implemented dispersion relation with the Bessel term",
            ),
            rel_error_pct=an["most_unstable_kR_error"] * 100.0,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "This validates the *dispersion solver* (the Bessel term matters: the "
                "naive kR = 1/sqrt(3) = 0.5774 is -17.2 % off), not a simulation. The "
                "dynamic 3-D column run in the same artifact does not deliver a growth "
                "rate (see rayleigh_plateau.dynamic.volume_drift)."
            ),
        )
    )
    out.append(
        _entry(
            claim_id="rayleigh_plateau.analytic.lambda_over_R",
            quantity_id="rayleigh_plateau_most_unstable_wavelength",
            quantity="most unstable wavelength of the Rayleigh-Plateau instability, lambda/R",
            artifact=rel,
            json_path="analytic.most_unstable_lambda_over_R",
            reference=(an["most_unstable_lambda_over_R_literature"], "-", "literature lambda/R = 9.02"),
            measured=(
                an["most_unstable_lambda_over_R"],
                "-",
                "2 pi / (kR)* from the implemented dispersion relation",
            ),
            rel_error_pct=an["most_unstable_lambda_over_R_error"] * 100.0,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes="Same analytic solver as kR*; -0.06 % is a rounding-level agreement.",
        )
    )
    cur = next(
        row
        for row in d["curvature"]
        if row["estimator"] == "profile" and row["radius_over_dx"] == 3
    )
    mls = next(
        row for row in d["curvature"] if row["estimator"] == "mls" and row["radius_over_dx"] == 3
    )
    axisym = next(
        row for row in d["curvature"] if row["estimator"] == "axisym" and row["radius_over_dx"] == 3
    )
    out.append(
        _entry(
            claim_id="rayleigh_plateau.curvature.profile_self_referential",
            quantity_id="csf_curvature_3d_cylinder",
            quantity="3-D CSF curvature on a cylindrical column, R/dx = 3",
            artifact=rel,
            json_path="curvature[0].relative_error",
            reference=(0.0, "-", "exact curvature 1/R of the constructed cylinder"),
            measured=(cur["relative_error"], "-", "'profile' estimator: fits the radius of the surface profile"),
            rel_error_pct=cur["relative_error"] * 100.0,
            gauge_degenerate=True,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "2.2e-16 is machine zero, i.e. the estimator recovers the radius it was "
                "built from: self-referential. The independent estimators on the same "
                f"cloud are mls {mls['relative_error'] * 100:+.1f} % and axisym "
                f"{axisym['relative_error'] * 100:+.1f} % -- that spread is the real "
                "curvature evidence, and it is not paper grade."
            ),
        )
    )
    run = d["column_run"]
    out.append(
        _entry(
            claim_id="rayleigh_plateau.dynamic.volume_drift",
            quantity_id="rayleigh_plateau_volume_drift",
            quantity="volume drift of the 3-D Rayleigh-Plateau column run (80 steps, 864 particles)",
            artifact=rel,
            json_path="column_run.final_volume_error",
            reference=(0.0, "-", "exact volume conservation of the incompressible column"),
            measured=(run["final_volume_error"], "-", "final relative volume error of the run"),
            rel_error_pct=run["final_volume_error"] * 100.0,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "The growth-rate fit of the same run is NaN "
                f"(omega_measured = NaN, max density error {run['max_density_error'] * 100:.1f} %), "
                "so the dynamic Rayleigh-Plateau capability is NOT validated: this entry "
                "records only the volume bookkeeping of a run that could not be reduced "
                "to a growth rate. Closing it needs a longer/steadier run."
            ),
        )
    )

    # ---- CSF dispersion ladder (capillary wave + Lamb modes) ----
    rel = "benchmarks/surface_tension_dispersion/validation_report.json"
    d = _load(rel)
    g = d["gates"]["local_range"]
    out.append(
        _entry(
            claim_id="csf_dispersion.capillary_wave.omega_error.local_range",
            quantity_id="capillary_wave_omega",
            quantity="capillary-wave frequency error of the local-range CSF operator, k = 1..3",
            artifact=rel,
            json_path="gates.local_range.C1_max_omega_error_pct",
            reference=(0.05, "-", "harness criterion C1: within 5 % of omega^2 = sigma k^3 / rho (Lamb 1932)"),
            measured=(
                g["C1_max_omega_error_pct"] / 100.0,
                "-",
                "max relative omega error over the three wavenumbers",
            ),
            claim_type="gate",
            gate_result="pass" if g["C1_within_5pct"] else "fail",
            gate_limit=0.05,
            guarding_test="tests/test_csf_dispersion.py",
            artifact_status="PASS" if g["C1_within_5pct"] else "FAIL",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Registered because the harness was uncovered: its own criterion fails at "
                "55 % and nothing in the ledger said so."
            ),
        )
    )
    out.append(
        _entry(
            claim_id="csf_dispersion.lamb_mode.omega_error.local_range",
            quantity_id="lamb_mode_omega",
            quantity="Lamb-mode frequency error of the local-range CSF operator, n = 2..4",
            artifact=rel,
            json_path="gates.local_range.C2_max_omega_error_pct",
            reference=(0.03, "-", "harness criterion C2: within 3 % of omega^2 = n(n^2-1) sigma/(rho R^3)"),
            measured=(
                g["C2_max_omega_error_pct"] / 100.0,
                "-",
                "max relative omega error over the three modes",
            ),
            claim_type="gate",
            gate_result="pass" if g["C2_within_3pct"] else "fail",
            gate_limit=0.03,
            guarding_test="tests/test_csf_dispersion.py",
            artifact_status="PASS" if g["C2_within_3pct"] else "FAIL",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "9.7 % is better than the legacy operator (18.5 %) but still 3.2x its own "
                "3 % gate -- the CSF frequency error is the same defect family as the "
                "Young-Laplace deficit, so it is not averaged into a PASS."
            ),
        )
    )

    # ---- fluctuating hydrodynamics (MD thermostat ladder) ----
    rel = "benchmarks/fluctuating_hydro/report.json"
    d = _load(rel)
    row = d["case"]["runs"][0]
    out.append(
        _entry(
            claim_id="fluctuating_hydro.thermostat.mean_temperature.n128",
            quantity_id="fluctuating_hydro_temperature",
            quantity="mean temperature of the LJ fluid under the Langevin (BAOAB) thermostat, N = 128",
            artifact=rel,
            json_path="case.runs[0].mean_temperature",
            reference=(1.0, "T*", "target temperature T* = 1.0 (reduced LJ units)"),
            measured=(row["mean_temperature"], "T*", f"mean over {row['n_steps']} steps"),
            guarding_test="tests/test_fluctuating_hydro.py",
            artifact_status="PASS" if d["verdicts"]["L4_thermostat_holds_target_temperature"] else "FAIL",
            backlog_ref="docs/verified/fluctuating_hydrodynamics_verification.md",
            notes=(
                "Supersedes the legacy reading T = 0.0152 (error -98.5 %) quoted in "
                "docs/verified/fluctuating_hydrodynamics_verification.md: the legacy "
                "lattice overlapped particles (up to 81 overlapping pairs at N = 256), "
                "which is what collapsed the temperature. Registered here so the old "
                "number cannot be quoted as current."
            ),
        )
    )
    return out


'''

def main() -> None:
    src = TARGET.read_text()
    if MARK in src:
        print("already patched")
        return
    needle = "def build() -> dict:"
    assert needle in src, "anchor not found"
    src = src.replace(needle, BLOCK.lstrip("\n") + needle, 1)
    src = src.replace(
        "        + ice_thermal_claims()\n    )",
        "        + ice_thermal_claims()\n        + coverage_closure_claims()\n    )",
        1,
    )
    TARGET.write_text(src)
    print("patched")


if __name__ == "__main__":
    main()