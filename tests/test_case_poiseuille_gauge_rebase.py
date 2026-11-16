"""Gauge ruling + evidence-integrity tests for CASE-POISEUILLE-SPH (R3 combined).

Guards two defects that were found in the committed R3 artifact:

D1 mixed C1 normaliser -- ``combined_Hdx8`` was graded on ``U_w`` while the finer
   rungs were graded on ``max|u_a|`` over the graded rows, so the published
   convergence trend mixed two gauges.  Fixed at root: one closed-form continuum
   gauge (``== U_w`` for this configuration) in ``combined.py``,
   ``pool_conformance.py`` and ``rebase_gauge.py``, and the committed summary was
   rebased from its own rows (no solver re-run, no measured row touched).

D2 leg rows carry the combined analytic column -- ``legs[*].rows[].u_analytic``
   holds the combined reference rather than the leg's own.  Gate-free, so it is
   quarantined with an audit artifact and a test instead of being repaired
   silently.

The tests are cheap: they read committed JSON and re-derive closed forms.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "case_poiseuille"
COMBINED = BENCH / "out" / "combined.json"
AUDIT = BENCH / "out" / "legs_row_audit.json"
REBASE = BENCH / "rebase_gauge.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(COMBINED.read_text())


@pytest.fixture(scope="module")
def rebase():
    return _load(REBASE, "case_poiseuille_rebase_gauge")


@pytest.fixture(scope="module")
def pool():
    return _load(BENCH / "pool_conformance.py", "case_poiseuille_pool_conformance_mod")


# --------------------------------------------------------------------------- #
# the gauge itself
# --------------------------------------------------------------------------- #
def test_gauge_equals_wall_speed_for_the_combined_configuration():
    """The predeclared C1 text says 'normalised by U_w' -- make that literal."""
    from_artifact = json.loads(COMBINED.read_text())["parameters"]
    g, uw = from_artifact["g"], from_artifact["wall_speed"]
    H, nu = from_artifact["height"], from_artifact["viscosity"]
    sys.path.insert(0, str(BENCH))
    try:
        rg = _load(REBASE, "rebase_for_gauge_test")
        assert rg.gauge_scale(g, uw, H, nu) == pytest.approx(abs(uw), rel=0, abs=1e-12)
        # static-wall Poiseuille leg: U_w = 0 must not divide by zero, and the
        # gauge must fall back to the closed-form centre-line speed
        assert rg.gauge_scale(g, 0.0, H, nu) == pytest.approx(
            abs(g) * H * H / (8.0 * nu), rel=1e-12)
        # Couette leg (g = 0): linear profile, maximum at the moving wall
        assert rg.gauge_scale(0.0, uw, H, nu) == pytest.approx(abs(uw), rel=1e-12)
    finally:
        sys.path.pop(0)


def test_gauge_is_resolution_independent(rebase, pool):
    """A sampled maximum moves with the row set; the declared gauge must not."""
    params = json.loads(COMBINED.read_text())["parameters"]
    g, uw, H, nu = (params["g"], params["wall_speed"], params["height"],
                    params["viscosity"])
    values = {rebase.gauge_scale(g, uw, H, nu), pool.gauge_scale(g, uw, H, nu)}
    assert len(values) == 1, f"gauge implementations disagree: {values}"
    assert values.pop() == pytest.approx(abs(uw), rel=0, abs=1e-12)


def test_solver_side_grader_uses_the_same_gauge():
    """``combined.gauge_scale`` must exist and agree (skipped without torch)."""
    pytest.importorskip("torch")
    mod = _load(BENCH / "combined.py", "case_poiseuille_combined_mod")
    params = json.loads(COMBINED.read_text())["parameters"]
    assert mod.gauge_scale(g=params["g"], wall_speed=params["wall_speed"],
                           height=params["height"],
                           nu=params["viscosity"]) == pytest.approx(
        abs(params["wall_speed"]), rel=0, abs=1e-12)


# --------------------------------------------------------------------------- #
# the rebased artifact
# --------------------------------------------------------------------------- #
def test_every_rung_is_graded_on_the_declared_gauge(artifact):
    uw = artifact["parameters"]["wall_speed"]
    assert artifact["summary_rebase"]["declared_gauge_m_s"] == pytest.approx(uw, rel=1e-12)
    for run in artifact["runs"]:
        assert run["profile_rms_gauge"] == pytest.approx(uw, rel=0, abs=1e-12), run["label"]


def test_rebased_summary_is_exactly_rederivable_from_its_own_rows(artifact, rebase):
    H = artifact["parameters"]["height"]
    nu = artifact["parameters"]["viscosity"]
    for run in artifact["runs"]:
        _s, rms, worst = rebase._profile_metrics(run, H, nu, "new")
        assert rms == pytest.approx(run["profile_rms_pct"], rel=1e-9)
        assert worst == pytest.approx(run["profile_worst_row_pct"], rel=1e-9)


def test_superseded_values_are_preserved_with_their_gauge(artifact):
    uw = artifact["parameters"]["wall_speed"]
    for run in artifact["runs"]:
        sup = run["summary_superseded"][-1]
        assert sup["gauge_rule"].startswith("max|u_analytic| over the graded rows")
        assert sup["profile_rms_pct"] != run["profile_rms_pct"] or \
            sup["gauge"] == pytest.approx(uw, rel=0, abs=1e-12)
    # the pre-rewrite hash makes "artifact edited without regeneration" detectable
    sr = artifact["summary_rebase"]
    assert len(sr["source_sha256_before_rebase"]) == 64
    assert sr["solver_rerun"] is False and sr["measured_rows_touched"] is False
    assert sr["stored_gauge_rule_by_rung"]["combined_Hdx8"] == "declared_U_w"
    assert sr["stored_gauge_rule_by_rung"]["combined_Hdx12"] == "sampled_max_abs_u_a"


def test_no_gate_flips_and_trend_is_on_one_gauge(artifact):
    thr = artifact["thresholds"]["C1_profile_rms_pct"]
    values = [r["profile_rms_pct"] for r in artifact["runs"]]
    assert all(v <= thr for v in values), values
    assert artifact["gates"]["C1_profile_rms_5pct"] is True
    assert artifact["summary_rebase"]["gate_flips"] == []
    # monotone improvement on a single gauge (before the rebase this sequence
    # mixed gauges: 0.884 | 0.466 | 0.175)
    assert values == sorted(values, reverse=True), values


def test_rebase_check_mode_is_idempotent():
    out = subprocess.run([sys.executable, str(REBASE), "--check"],
                         capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stdout + out.stderr


# --------------------------------------------------------------------------- #
# the quarantined leg-row defect
# --------------------------------------------------------------------------- #
def test_leg_rows_analytic_column_is_flagged_not_hidden(artifact):
    for name, rec in artifact["legs"].items():
        defect = rec["row_column_defect"]
        assert defect["repaired"] is False and defect["gate_bearing"] is False
        assert defect["max_abs_error_m_s"] > 0.1, (name, defect)
        assert defect["profile_rms_reproducible_from_rows"] is True
        assert defect["profile_worst_row_reproducible_from_rows"] is False


def test_leg_audit_artifact_quantifies_the_defect():
    audit = json.loads(AUDIT.read_text())
    assert len(audit["legs"]) == 2
    for name, a in audit["legs"].items():
        # the row column is wrong, the summary RMS is right (own closed form)
        assert a["max_abs_column_minus_own_reference"] > 0.1
        assert a["rms_reproducible_from_rows"] is True
        # the declared gauge is 1.47/own sampled gauge larger -> smaller percentage
        assert a["own_reference_declared_gauge_pct"] < a["own_reference_sampled_gauge_pct"]
    # and the repaired column would change nothing that a gate reads
    assert all(not a["gate_bearing"] for a in audit["legs"].values())


def test_pool_conformance_binding_no_longer_needs_an_exception(pool):
    assert pool.KNOWN_BINDING_EXCEPTIONS == {}
    payload = json.loads((BENCH / "out" / "pool_conformance.json").read_text())
    assert payload["summary"]["n_binding_violations"] == 0
    assert payload["summary"]["n_unknown_binding_violations"] == 0
    assert payload["binding_rtol"] == pytest.approx(1e-6)
    ids = {f["id"] for f in payload["functionals"]}
    assert {"CP_leg_poiseuille_diagnostics", "CP_leg_couette_diagnostics"} <= ids