"""Independent audit of the CASE-POISEUILLE-SPH evidence bundle.

These tests do not re-run the SPH solver.  They assert that the numbers the case
publishes *follow from* its committed raw data and from a reference derived
independently of the case code:

* ``out/result.json``     -- the case's self-report (analytic + measured + gates)
* ``out/audit_independent.json`` -- what ``audit_independent.py`` derives without
  importing ``tensorsph.womersley``, ``tensorsph.stokes_second``,
  ``benchmarks.case_poiseuille.run`` or ``ref.py``.

Six claims are binding:

1. the audit is computed from the exact bytes of the committed artifact
   (sha256 binding; a stale audit fails);
2. the committed analytic reference is the solution of the PDE -- reproduced by
   a complex boundary-value solve (never evaluates the closed form) and by a
   time-domain Crank-Nicolson march (shares no code with the case);
3. the published gate values are reproduced from the committed per-row and
   centre-line samples by an independently written harmonic fit;
4. the gate stack is *falsifiable*: amplitude/phase perturbations above the
   tolerance flip the gates, a 2 % perturbation below it does not;
5. the case genuinely discriminates unsteadiness -- a quasi-steady solver is
   rejected by >100 %, which is what makes the frequency response a measurement
   and not a tautology;
6. the honest scope is published: the <5 % acceptance holds on the predeclared
   interior band, and the wall-adjacent rows exceed it at every rung.

Regenerate with:
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/audit_independent.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "case_poiseuille"
RESULT = CASE / "out" / "result.json"
AUDIT_JSON = CASE / "out" / "audit_independent.json"
AUDIT_PY = CASE / "audit_independent.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("case_poiseuille_audit", AUDIT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def result():
    assert RESULT.exists(), f"missing committed artifact {RESULT}"
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def audit():
    assert AUDIT_JSON.exists(), (
        f"missing {AUDIT_JSON}; regenerate with "
        "PYTHONPATH=.:src python3 benchmarks/case_poiseuille/audit_independent.py")
    return json.loads(AUDIT_JSON.read_text())


# --------------------------------------------------------------------------- #
# 1. binding
# --------------------------------------------------------------------------- #

def test_audit_is_bound_to_the_committed_artifact(audit):
    digest = hashlib.sha256(RESULT.read_bytes()).hexdigest()
    assert audit["source_sha256"] == digest, (
        "audit_independent.json was computed from different bytes than the "
        f"committed result.json (audit {audit['source_sha256'][:12]}, "
        f"file {digest[:12]}) -- stale audit")


def test_audit_is_reproducible_from_the_current_code(audit):
    """Re-running the audit must give back the committed numbers."""
    fresh = _load_audit_module().audit()
    assert fresh["source_sha256"] == audit["source_sha256"]
    assert fresh["verdict"] == audit["verdict"]
    got = [r["recomputed_committed_analytic"]["amp_centre_pct"]
           for r in fresh["harmonic_gate_audit"]]
    want = [r["recomputed_committed_analytic"]["amp_centre_pct"]
            for r in audit["harmonic_gate_audit"]]
    assert got == pytest.approx(want, rel=0, abs=1e-12)


def test_audit_does_not_import_case_code():
    """Independence is an invariant: the audit may not lean on the case code."""
    import ast
    tree = ast.parse(AUDIT_PY.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {"tensorsph", "benchmarks.case_poiseuille", "ref"}
    for mod in imported:
        root = mod.split(".")[0]
        assert root not in forbidden and mod not in forbidden, (
            f"audit_independent.py imports {mod!r}: the audit must derive its "
            "reference from the PDE, not from the case code")
    assert imported & {"numpy", "math", "json", "hashlib"}, "sanity: imports parsed"


# --------------------------------------------------------------------------- #
# 2. the reference is the solution of the PDE
# --------------------------------------------------------------------------- #

def test_closed_form_agrees_with_the_boundary_value_solve(audit):
    for tag, block in audit["reference_provenance"].items():
        assert block["closed_form_vs_bvp_rel"] < 2e-6, (
            f"{tag}: the case's closed form is not the BVP solution "
            f"(rel {block['closed_form_vs_bvp_rel']:.2e})")


def test_committed_analytic_reference_is_confirmed_independently(audit):
    for tag, block in audit["reference_provenance"].items():
        assert block["committed_amp_vs_bvp_rel"] < 1e-6, (
            f"{tag}: committed amplitudes differ from the independent solve "
            f"by {block['committed_amp_vs_bvp_rel']:.2e}")
        assert block["committed_phase_vs_bvp_deg"] < 1e-4, (
            f"{tag}: committed phases differ from the independent solve by "
            f"{block['committed_phase_vs_bvp_deg']:.2e} deg")


def test_time_domain_march_reproduces_the_reference_and_its_phase_convention(audit):
    """A sign slip shared by the analytic form and the fit would cancel here."""
    for tag, block in audit["reference_provenance"].items():
        assert block["marched_fd_vs_bvp_amp_rel"] < 5e-3, (
            f"{tag}: Crank-Nicolson march amplitude differs by "
            f"{block['marched_fd_vs_bvp_amp_rel']:.2e}")
        assert block["marched_fd_vs_bvp_phase_deg"] < 0.05, (
            f"{tag}: Crank-Nicolson march phase differs by "
            f"{block['marched_fd_vs_bvp_phase_deg']:.3f} deg -- the phase "
            "convention is not established by the closed form alone")


# --------------------------------------------------------------------------- #
# 3. published gates follow from the committed samples
# --------------------------------------------------------------------------- #

def test_published_gate_values_are_reproduced_from_the_raw_rows(audit):
    keys = ("amp_centre_pct", "amp_profile_rms_pct",
            "phase_centre_pct_of_quarter_turn",
            "phase_profile_rms_pct_of_quarter_turn")
    for run in audit["harmonic_gate_audit"]:
        pub, rec = run["published"], run["recomputed_committed_analytic"]
        for key in keys:
            assert rec[key] == pytest.approx(pub[key], rel=1e-6, abs=1e-9), (
                f"H/dx={run['resolution_H_over_dx']} {key}: published "
                f"{pub[key]!r}, recomputed from the committed rows {rec[key]!r}")


def test_centre_line_refit_reproduces_the_published_fit(audit):
    """An independently written least-squares fit on the committed series."""
    for run in audit["harmonic_gate_audit"]:
        r = run["refit_from_centre_series"]
        assert r["residual"] < 0.02, (
            f"harmonic fit residual {r['residual']:.4f} too large to certify")
        # the window is *not* an exact integer number of periods (dt does not
        # divide the period), so orthogonality is only approximate and the fit
        # carries a constant column.  Both facts are asserted, not assumed.
        assert abs(r["window_span_periods"] - 2.0) < 5e-3, (
            f"window spans {r['window_span_periods']:.6f} periods")
        assert r["cos_sin_orthogonality"] < 1e-3, (
            "cos/sin columns are further from orthogonal than the sampling "
            "jitter explains; the fit would be biased")
        assert abs(r["dc_mean"]) < 5e-3, (
            f"unexplained mean {r['dc_mean']:.2e} in the centre-line fit")
        assert r["amplitude_vs_published_rel"] < 1e-6
        assert r["phase_vs_published_deg"] < 1e-6


def test_every_rung_passes_the_five_percent_acceptance(audit):
    for run in audit["harmonic_gate_audit"]:
        assert all(run["gates"].values()), (
            f"H/dx={run['resolution_H_over_dx']} fails {run['gates']}")


def test_independent_anchor_does_not_move_the_verdict(audit):
    """Swapping in the independent analytic values must not change any gate."""
    for run in audit["harmonic_gate_audit"]:
        m = run["recomputed_independent_analytic"]
        assert abs(m["amp_centre_pct"]) <= 5.0
        assert m["amp_profile_rms_pct"] <= 5.0
        assert m["phase_centre_pct_of_quarter_turn"] <= 5.0
        assert m["phase_profile_rms_pct_of_quarter_turn"] <= 5.0
        committed = run["recomputed_committed_analytic"]
        assert m["amp_centre_pct"] == pytest.approx(
            committed["amp_centre_pct"], rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4./5. the gates are falsifiable and the case discriminates
# --------------------------------------------------------------------------- #

def test_gate_stack_rejects_degraded_amplitudes_and_phases(audit):
    fals = audit["falsifiability"]
    assert fals["no_mutation_passes"], (
        "at least one over-tolerance mutation still passes every gate -- the "
        "gate stack is vacuous")
    for tag in ("amplitude_plus_6pct", "amplitude_minus_6pct"):
        assert not fals["mutations"][tag]["gates"]["G1_amp_centre_5pct"], tag
        assert not fals["mutations"][tag]["gates"]["G2_amp_profile_rms_5pct"], tag
    phase = fals["mutations"]["phase_shift_5.5pct_quarter_turn"]["gates"]
    assert not phase["G3_phase_centre_5pct_of_quarter_turn"]
    assert not phase["G4_phase_profile_rms_5pct_of_quarter_turn"]


def test_a_sub_tolerance_mutation_still_passes(audit):
    """Control: the gates must not reject a 2 % perturbation."""
    assert audit["falsifiability"]["control_passes"], (
        "a 2 % amplitude perturbation was rejected -- the 5 % tolerance is not "
        "the thing being applied")


def test_quasi_steady_solver_would_be_rejected(audit):
    disc = audit["frequency_discrimination"]
    assert disc["G6"] is True
    assert disc["quasi_steady_rejection_pct"] > 100.0, (
        "the two frequencies do not separate a quasi-steady solver from the "
        "unsteady one; the frequency response would be untestable")
    assert disc["steady_would_be_error_pct_of_analytic_harmonic"] > 60.0


# --------------------------------------------------------------------------- #
# 6. honest scope
# --------------------------------------------------------------------------- #

def test_wall_band_caveat_is_published_not_hidden(audit):
    disc = audit["wall_band_disclosure"]
    assert disc["statement"], "the caveat must be stated, not implied"
    for row in disc["rows"]:
        assert row["gate_band"] == [0.15, 0.85]
        assert row["worst_excluded_amp_pct"] > 5.0, (
            "the audit claims wall rows exceed 5 % -- if that changed, the "
            "disclosure has to be rewritten")
    coarsest = min(disc["rows"], key=lambda r: r["resolution_H_over_dx"])
    assert coarsest["worst_excluded_amp_pct"] > 10.0


def test_verdict_is_consistent_with_its_evidence(audit):
    v = audit["verdict"]
    checks = [
        v["reference_provenance_independently_confirmed"],
        v["gate_arithmetic_reproduced"],
        v["gates_reject_degraded_data"],
        v["acceptance_5pct_holds_interior_band"],
    ]
    assert all(checks), f"verdict recorded despite failing evidence: {v}"
    assert v["disposition"] == "approve with notes"


def test_audit_declares_what_it_did_not_run(audit):
    """A missing check is 'not run', never 'passed'."""
    assert audit["not_run"], "the audit must list what it did not execute"
    joined = " ".join(audit["not_run"]).lower()
    assert "solver" in joined