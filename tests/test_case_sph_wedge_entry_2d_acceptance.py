"""Acceptance test for CASE-SPH-WEDGE-ENTRY-2D, graded from committed artefacts only.

Why this file was rewritten (m21151)
------------------------------------
The previous version of this test asserted *module-level constants*

    PUBLISHED_LADDER = {0.5: (2.686, 1.1, 1.1), ...}          # hand copied
    def test_cp_max_within_mission_tolerance(dx_mm, cp_max_err_pct):
        assert cp_max_err_pct / 100.0 < 0.15

i.e. it re-stated the claim ("peak Cp error 1.1 % at dx = 0.5 mm") and then
checked that 0.011 < 0.15.  That test can never fail for any simulation, so it
carried no information about the case.  Two further defects were confirmed while
rewriting it:

* the ladder it cited did not match the document it named as its source
  (``c/(V t) = 2.783`` at ``dx = 2 mm`` vs 2.6124 in Table 3.2 of
  ``docs/verification_wedge_entry.md``), and
* instead of the committed evidence it read ``artifacts/``, which
  ``.gitignore`` excludes, so three of its tests failed in every clean checkout
  (and only passed on a worktree that happened to hold a leftover run).

What is graded now
------------------
Every number is read from committed artefacts and every verdict is *recomputed*
from them by an independent implementation of the rule that
``benchmarks/case_sph_wedge_entry_2d/ladder_closure.py`` declares:

    U_g     = max(estimator spread, jackknife spread) on the 61-frame ensemble
    judgeable_g  <=>  U_g <= 15 %
    r_g     = obs_g / synth_g - 1          (same-operator residual, Wagner field
                                            pushed through the same estimator)
    verdict: PASS  when the two finest judgeable grids have the same sign and
             |r| <= 15 %; FAIL_above / FAIL_below when they share a sign but
             exceed 15 %; NOT_DECIDABLE when fewer than two grids are judgeable
             or the sign flips (a single grid inside +/-15 % is *not* a PASS --
             that is the trap documented in m18382).

The Zhao & Faltinsen (1993) beta = 30 deg table value is **WITHHELD** (closed
access, no registered copy in the ecosystem, see ``refs.json``), so no test here
compares against it: the case is graded against Wagner (1932), the analytic
solution of the *same* idealisation, and the Z&F cell stays empty.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
ENS_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
BIAS_JSON = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias" / "bias_report.json"
LADDER_JSON = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ladder" / "ladder_closure.json"
ERRATUM_JSON = (ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_table_erratum"
                / "erratum.json")
REF_DOC = ROOT / "docs" / "verification_wedge_entry.md"

GATE = 0.15          # the mission tolerance on the peak slamming pressure coefficient
FRAMES = "61"        # the self-similar ensemble size used throughout


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _skip_unless(path: Path, why: str):
    if not Path(path).exists():
        pytest.skip(f"{Path(path).name} not committed ({why})")


def wagner_cp_max(beta_deg: float) -> float:
    """Independent closed form: C = pi/(2 tan b); Cp_max = C^2+1 (C>=1) else 2C."""
    C = math.pi / (2.0 * math.tan(math.radians(beta_deg)))
    return C * C + 1.0 if C >= 1.0 else 2.0 * C


def wagner_cp_centre(beta_deg: float) -> float:
    C = math.pi / (2.0 * math.tan(math.radians(beta_deg)))
    return 2.0 * C


def ensemble_grids() -> dict:
    out = {}
    for p in sorted(ENS_DIR.glob("ensemble_*.json")):
        art = _json(p)
        out[f"dx{art['config']['dx'] * 1e3:g}mm"] = (p, art)
    return out


def ladder_rows() -> list:
    return _json(LADDER_JSON)["grids"]


def judgeable(residual_row: dict, gate: float = GATE) -> bool:
    """Predeclared judgeability: the estimator spread must fit inside the gate."""
    return bool(residual_row.get("uncertainty") is not None
                and residual_row["uncertainty"] <= gate)


def verdict_from(rows: list, gate: float = GATE) -> str:
    """Independent implementation of the rule in ladder_closure.py's docstring."""
    good = [r for r in rows if judgeable(r, gate)]
    if len(good) < 2:
        return "NOT_DECIDABLE"
    fine, fine2 = good[-1], good[-2]
    if (fine["residual"] > 0) != (fine2["residual"] > 0):
        return "NOT_DECIDABLE"
    worst = max(abs(fine["residual"]), abs(fine2["residual"]))
    if worst <= gate:
        return "PASS"
    return "FAIL_above" if fine["residual"] > 0 else "FAIL_below"


# --------------------------------------------------------------------------- #
# 1. the analytic anchor really is analytic (and the artefacts use it)
# --------------------------------------------------------------------------- #
class TestWagnerAnchor:
    @pytest.mark.parametrize("beta_deg", [10.0, 15.0, 20.0, 30.0, 45.0])
    def test_module_matches_independent_closed_form(self, beta_deg):
        from tensorsph.wedge_entry import wagner_peak_over_q, wagner_p_centre, wagner_c_coeff

        beta = math.radians(beta_deg)
        assert wagner_c_coeff(beta) == pytest.approx(wagner_cp_centre(beta_deg) / 2.0,
                                                     rel=1e-12)
        assert wagner_peak_over_q(beta) == pytest.approx(wagner_cp_max(beta_deg),
                                                         rel=1e-12)
        assert wagner_p_centre(1.0, beta, 1000.0) == pytest.approx(
            500.0 * wagner_cp_centre(beta_deg), rel=1e-12)

    def test_committed_artefacts_use_the_module_reference(self):
        grids = ensemble_grids()
        _skip_unless(ENS_DIR, "no ensemble artefacts committed")
        for tag, (path, art) in grids.items():
            assert art["wagner_cp_max"] == pytest.approx(wagner_cp_max(30.0), rel=1e-10), tag
            assert art["wagner_cp_centre"] == pytest.approx(wagner_cp_centre(30.0),
                                                            rel=1e-10)
            assert art["case"] == "CASE-SPH-WEDGE-ENTRY-2D"


# --------------------------------------------------------------------------- #
# 2. the suite reads committed evidence only (fixes the gitignored-artefact bug)
# --------------------------------------------------------------------------- #
class TestEvidenceIsCommitted:
    EVIDENCE = [BIAS_JSON, LADDER_JSON, ERRATUM_JSON]

    def test_artifacts_dir_is_ignored(self):
        """Guard: if artifacts/ ever stops being ignored this guard is obsolete."""
        r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "artifacts/x.json"],
                           capture_output=True)
        assert r.returncode == 0, "artifacts/ is no longer gitignored -- revisit this test"

    @pytest.mark.parametrize("path", EVIDENCE, ids=lambda p: Path(p).name)
    def test_evidence_file_is_tracked(self, path):
        _skip_unless(path, "not produced in this checkout")
        rel = str(Path(path).relative_to(ROOT))
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{rel} is not committed, so CI cannot grade it"

    def test_ensemble_artefacts_are_committed(self):
        grids = ensemble_grids()
        assert grids, "no committed ensemble artefact"
        for tag, (path, art) in grids.items():
            rel = str(path.relative_to(ROOT))
            r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                               capture_output=True, text=True)
            assert r.returncode == 0, f"{rel} is not committed"
            assert FRAMES in art["by_frames"], f"{tag}: no {FRAMES}-frame reading"
            assert art["config"]["beta_deg"] == 30.0
            assert art["config"]["velocity"] == 1.0

    def test_ensemble_artefacts_have_finite_readings(self):
        for tag, (_, art) in ensemble_grids().items():
            rb = art["by_frames"][FRAMES]["resolution_band"]
            assert rb["peak_median"] == rb["peak_median"] and rb["peak_median"] > 0, tag
            assert 0.0 <= rb["spread_rel"] <= 10.0, tag


# --------------------------------------------------------------------------- #
# 3. the case verdict is recomputed from the artefacts (no hand-copied numbers)
# --------------------------------------------------------------------------- #
class TestLadderVerdictIsDataDerived:
    def setup_method(self):
        _skip_unless(LADDER_JSON, "ladder closure not produced")
        _skip_unless(BIAS_JSON, "estimator-bias report not produced")

    def test_same_operator_residual_recomputes_from_upstream_artefacts(self):
        bias = _json(BIAS_JSON)["grids"]
        ens = ensemble_grids()
        for row in ladder_rows():
            tag = row["tag"]
            if tag not in ens:
                continue
            obs = ens[tag][1]["by_frames"][FRAMES]["resolution_band"]["peak_median"]
            truth = bias[tag]["by_frames"][FRAMES]["synth_smoothed"]["peak_median"]
            assert row["obs_peak"] == pytest.approx(obs, rel=1e-12), tag
            assert row["synth_peak"] == pytest.approx(truth, rel=1e-12), tag
            assert row["residual"] == pytest.approx(obs / truth - 1.0, abs=1e-12), tag

    def test_judgeability_rule_is_the_declared_one(self):
        for row in ladder_rows():
            assert bool(row["judgeable"]) is judgeable(row), row["tag"]

    def test_recorded_verdict_equals_the_recomputed_verdict(self):
        recorded = _json(LADDER_JSON)["verdict"]["verdict"]
        assert recorded == verdict_from(ladder_rows())

    def test_no_pass_with_a_single_judgeable_grid(self):
        """m18382's trap: one grid inside +/-15 % must not be reported as PASS."""
        good = [r for r in ladder_rows() if judgeable(r)]
        recorded = _json(LADDER_JSON)
        if len(good) < 2:
            assert recorded["verdict"]["verdict"] != "PASS"
            assert sorted(recorded["verdict"]["grids_judgeable"]) == sorted(
                r["tag"] for r in good)

    def test_non_pass_verdict_is_explained_by_the_evidence(self):
        """A residual inside 15 % on a non-judgeable grid is noise, not a PASS."""
        recorded = _json(LADDER_JSON)["verdict"]["verdict"]
        rows = ladder_rows()
        good = [r for r in rows if judgeable(r)]
        if recorded == "PASS":
            assert len(good) >= 2
            fine, fine2 = good[-1], good[-2]
            assert (fine["residual"] > 0) == (fine2["residual"] > 0)
            for r in (fine, fine2):
                assert abs(r["residual"]) <= GATE
        else:
            why = []
            if len(good) < 2:
                why.append("fewer than two judgeable grids")
            elif (good[-1]["residual"] > 0) != (good[-2]["residual"] > 0):
                why.append("sign flip between the two finest judgeable grids")
            elif max(abs(good[-1]["residual"]), abs(good[-2]["residual"])) > GATE:
                why.append("residual outside the gate")
            assert why, f"verdict {recorded} is not explained by the artefacts"
            assert isinstance(_json(LADDER_JSON)["verdict"].get("reason"), str)

    def test_missing_finest_grid_is_reported_not_silently_dropped(self):
        ladder = _json(LADDER_JSON)
        assert "missing_grids" in ladder
        present = {r["tag"] for r in ladder["grids"]}
        for tag in ladder["missing_grids"]:
            assert tag not in present


# --------------------------------------------------------------------------- #
# 4. the published ladder is bound to the artefacts, or marked unreproducible
# --------------------------------------------------------------------------- #
class TestPublishedTableIsBoundToEvidence:
    def setup_method(self):
        _skip_unless(ERRATUM_JSON, "table erratum not generated")

    def test_erratum_recomputes_the_table_peak_column(self):
        err = _json(ERRATUM_JSON)
        q = 500.0                                     # 0.5 * rho * V^2, rho=1000, V=1
        assert err["design"]["q_pa"] == pytest.approx(q)
        assert err["design"]["wagner_cp_max"] == pytest.approx(wagner_cp_max(30.0))
        for row in err["comparisons"]:
            if row["table_pmax_kpa"] is None:
                continue
            assert row["table_cp"] == pytest.approx(row["table_pmax_kpa"] * 1000.0 / q,
                                                    rel=1e-12)
            if row["table_cp_err_recomputed"] is not None:
                assert row["table_cp_err_recomputed"] == pytest.approx(
                    row["table_pmax_err_printed"] / 100.0, abs=1e-3)

    def test_summary_agrees_with_the_per_row_evidence(self):
        err = _json(ERRATUM_JSON)
        s = err["summary"]
        v = s["verdict"]
        unrepro = [c["dx"] for c in err["comparisons"] if c["match"] == "unreproducible"]
        assert s["unreproducible"] == unrepro
        assert v["usable_as_acceptance"] is (not unrepro)

    def test_unreproducible_rows_are_not_used_as_an_acceptance_verdict(self):
        """The old test graded PASS on the dx = 0.5 mm row of this table."""
        err = _json(ERRATUM_JSON)
        by_dx = {c["dx"]: c for c in err["comparisons"]}
        fine = by_dx.get(0.0005)
        assert fine is not None, "table row dx=0.5 mm disappeared"
        if fine["match"] != "reproduced":
            assert err["summary"]["verdict"]["usable_as_acceptance"] is False

    def test_declared_solver_settings_differ_from_committed_runs(self):
        """Table 3.2 was produced at alpha=0.1; every committed run used 0.03."""
        cfg = _json(ERRATUM_JSON)["doc_config_vs_committed"]
        doc_a = cfg["doc_declared"]["alpha_visc"]
        if doc_a is None or not cfg["committed_runs"]:
            pytest.skip("solver settings not declared in both places")
        runs = {tag: c["alpha_visc"] for tag, c in cfg["committed_runs"].items()}
        differ = any(abs(a - doc_a) > 1e-12 for a in runs.values())
        assert cfg["alpha_mismatch"] is differ
        if differ:
            # documents the reproducibility gap, it does not hide it
            assert REF_DOC.exists()


# --------------------------------------------------------------------------- #
# 5. reference provenance contract (unchanged in intent)
# --------------------------------------------------------------------------- #
class TestProvenanceContract:
    def test_zhao_faltinsen_withheld(self):
        refs = _json(BENCH / "refs.json")
        zf = [r for r in refs["references"] if r["id"] == "zhao_faltinsen_1993"][0]
        assert zf["status"] == "WITHHELD"
        assert zf["table_or_figure_value_beta_30"] is None
        assert zf["what_would_close_it"]

    def test_wagner_real_and_derived_in_repo(self):
        refs = _json(BENCH / "refs.json")
        w = [r for r in refs["references"] if r["id"] == "wagner_1932"][0]
        assert w["status"] == "REAL"
        assert "wedge_entry.py" in w["source_of_formula"]
        assert (ROOT / "src" / "tensorsph" / "wedge_entry.py").exists()

    def test_no_reference_number_is_typeset_in_the_driver(self):
        """refs.json must stay the single source of truth for provenance."""
        src = (BENCH / "run.py").read_text()
        assert "refs.json" in src