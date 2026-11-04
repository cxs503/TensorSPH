"""Tests for the claim-to-evidence audit (``tools/claim_audit.py``).

The audit is only worth having if it actually fires, so every test here is a
control: the repository ledger must pass, and each class of defect it exists to
catch (a stale "we do not support X" sentence, a published number that no longer
matches its artifact, an artifact that vanished) is injected into a throwaway
tree and must be caught.

No SPH is run: the audit reads JSON, source text and the README.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "claim_audit.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("claim_audit", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load_tool()


# --------------------------------------------------------------------------
# the real repository
# --------------------------------------------------------------------------

def test_repository_ledger_is_consistent():
    """No fatal finding: published prose agrees with the committed evidence."""
    report = A.audit(_ROOT)
    fatal = [f for f in report["findings"] if f["severity"] == "fatal"]
    assert fatal == [], [f["message"] for f in fatal]
    assert report["summary"]["pass"] is True
    assert report["summary"]["drift"] == 0


def test_repository_pins_the_two_capabilities_that_actually_exist():
    """The audit is not passing because it checked nothing."""
    report = A.audit(_ROOT)
    by_id = {c["id"]: c for c in report["claims"]}
    assert by_id["st-model-implemented"]["grade"] == "REAL"
    assert by_id["st-consistent-variant"]["grade"] == "REAL"
    # ... and the failures are recorded as debt, not hidden.
    assert by_id["st-laplace-verification"]["grade"] == "REAL"
    assert by_id["st-laplace-verification"]["detail"]["checks"][0]["actual"] == "FAIL"
    assert by_id["b2-experimental-comparison"]["grade"] == "REAL"
    assert by_id["b2-experimental-comparison"]["detail"]["checks"][0]["actual"] == pytest.approx(
        0.5727571197674608)


def test_repository_debt_list_is_exactly_the_unbacked_numbers():
    """The prose-only claims stay visible; the set is pinned so it cannot grow silently."""
    report = A.audit(_ROOT)
    withheld = sorted(c["id"] for c in report["claims"] if c["grade"] == "WITHHELD")
    assert withheld == [
        "b1-hydrostatic-interior",
        "b2-convergence-order",
        "b2-density-compression",
        "gpu-crossover",
    ]


def test_generated_markdown_and_json_are_current():
    """docs/claim_audit.md and docs/claims/report.json must equal a fresh run."""
    md_path = _ROOT / "docs" / "claim_audit.md"
    json_path = _ROOT / "docs" / "claims" / "report.json"
    assert md_path.is_file() and json_path.is_file(), "generated reports missing"
    report = A.audit(_ROOT)
    assert md_path.read_text(encoding="utf-8") == A.to_markdown(report)
    committed = json.loads(json_path.read_text(encoding="utf-8"))
    assert committed["summary"] == report["summary"]
    assert [c["id"] for c in committed["claims"]] == [c["id"] for c in report["claims"]]


# --------------------------------------------------------------------------
# controls: the tree is the fixture
# --------------------------------------------------------------------------

def _mini_tree(tmp_path, *, readme, ledger, artifacts=None):
    (tmp_path / "docs" / "claims").mkdir(parents=True, exist_ok=True)
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "docs" / "claims" / "ledger.json").write_text(
        json.dumps(ledger), encoding="utf-8")
    for rel, payload in (artifacts or {}).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _metric_ledger(expected=0.573, artifact="benchmarks/x/report.json"):
    return {
        "claims": [{
            "id": "experimental-error",
            "kind": "artifact_metric",
            "statement": "front error vs experiment",
            "evidence": {"artifact": artifact, "pointer": "/gates/error/measured",
                         "relation": "close_to", "expected": expected, "abs_tol": 0.005},
        }],
        "absence_claims": [],
    }


def test_matching_artifact_is_real(tmp_path):
    tree = _mini_tree(
        tmp_path, readme="nothing stale here\n", ledger=_metric_ledger(),
        artifacts={"benchmarks/x/report.json": {"gates": {"error": {"measured": 0.5728}}}})
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "REAL"
    assert report["summary"]["pass"] is True


def test_changed_artifact_is_flagged_as_drift(tmp_path):
    """If someone re-runs the benchmark, the pinned prose must be updated."""
    tree = _mini_tree(
        tmp_path, readme="nothing stale here\n", ledger=_metric_ledger(),
        artifacts={"benchmarks/x/report.json": {"gates": {"error": {"measured": 0.11}}}})
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "DRIFT"
    assert [f["type"] for f in report["findings"]] == ["artifact_drift"]
    assert report["summary"]["pass"] is False
    assert A.main(["--root", str(tree), "--check"]) == 1


def test_missing_artifact_is_debt_not_a_crash(tmp_path):
    tree = _mini_tree(tmp_path, readme="clean\n", ledger=_metric_ledger())
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "WITHHELD"
    assert report["claims"][0]["reason"] == "artifact_missing"
    assert report["summary"]["pass"] is True  # debt is reported, not fatal


def test_unparsable_artifact_is_withheld(tmp_path):
    tree = _mini_tree(tmp_path, readme="clean\n", ledger=_metric_ledger(),
                      artifacts={"benchmarks/x/report.json": {"gates": {}}})
    (tree / "benchmarks/x/report.json").write_text("{not json", encoding="utf-8")
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "WITHHELD"
    assert report["claims"][0]["reason"].startswith("artifact_unparsable")


def test_stale_absence_sentence_is_caught(tmp_path):
    """The regression this tool exists for: README says the module does not exist."""
    (tmp_path / "src/tensorsph").mkdir(parents=True)
    (tmp_path / "src/tensorsph/surface_tension.py").write_text(
        "def compute_surface_tension_force():\n    return 0\n", encoding="utf-8")
    ledger = {
        "claims": [{
            "id": "st-model-implemented",
            "kind": "code_present",
            "statement": "CSF surface tension is implemented",
            "evidence": {"file": "src/tensorsph/surface_tension.py",
                         "symbols": ["compute_surface_tension_force"]},
        }],
        "absence_claims": [{
            "id": "readme-no-surface-tension",
            "capability": "st-model-implemented",
            "pattern": "No surface-tension model",
            "note": "drop the sentence",
        }],
    }
    tree = _mini_tree(tmp_path, readme="- No surface-tension model.\n", ledger=ledger)
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "REAL"
    assert [f["type"] for f in report["findings"]] == ["stale_absence_claim"]
    assert report["summary"]["pass"] is False


def test_absence_sentence_is_fine_when_the_module_is_absent(tmp_path):
    ledger = {
        "claims": [{
            "id": "st-model-implemented",
            "kind": "code_present",
            "statement": "CSF surface tension is implemented",
            "evidence": {"file": "src/tensorsph/surface_tension.py",
                         "symbols": ["compute_surface_tension_force"]},
        }],
        "absence_claims": [{
            "id": "readme-no-surface-tension",
            "capability": "st-model-implemented",
            "pattern": "No surface-tension model",
        }],
    }
    tree = _mini_tree(tmp_path, readme="- No surface-tension model.\n", ledger=ledger)
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "WITHHELD"
    assert report["summary"]["pass"] is True


def test_missing_symbol_downgrades_a_code_claim(tmp_path):
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src/m.py").write_text("def other():\n    pass\n", encoding="utf-8")
    ledger = {"claims": [{"id": "c", "kind": "code_present", "statement": "x",
                          "evidence": {"file": "src/m.py", "symbols": ["wanted"]}}],
              "absence_claims": []}
    tree = _mini_tree(tmp_path, readme="clean\n", ledger=ledger)
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "WITHHELD"
    assert "src/m.py::wanted" in report["claims"][0]["reason"]


def test_missing_test_id_downgrades_a_test_claim(tmp_path):
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/t.py").write_text("def test_present():\n    pass\n", encoding="utf-8")
    ledger = {"claims": [{"id": "t", "kind": "test_present", "statement": "x",
                          "evidence": {"file": "tests/t.py",
                                       "tests": ["test_present", "test_gone"]}}],
              "absence_claims": []}
    tree = _mini_tree(tmp_path, readme="clean\n", ledger=ledger)
    report = A.audit(tree)
    assert report["claims"][0]["grade"] == "WITHHELD"
    assert "tests/t.py::test_gone" in report["claims"][0]["reason"]


# --------------------------------------------------------------------------
# pointer / comparison primitives
# --------------------------------------------------------------------------

def test_pointer_resolution_walks_dicts_and_lists():
    doc = {"a": {"b": [10, {"c": 42}]}}
    assert A.resolve_pointer(doc, "") is doc
    assert A.resolve_pointer(doc, "/a/b/1/c") == 42
    with pytest.raises(KeyError):
        A.resolve_pointer(doc, "/a/zzz")


def test_pointer_escapes_rfc6901():
    doc = {"a/b": {"c~d": 1}}
    assert A.resolve_pointer(doc, "/a~1b/c~0d") == 1


def test_close_to_rejects_non_numeric_actuals():
    assert A._matches("close_to", 1.0, 1.0, 0.0) is True
    assert A._matches("close_to", None, 1.0, 10.0) is False
    assert A._matches("close_to", True, 1.0, 10.0) is False
    assert A._matches("eq", "FAIL", "FAIL", 0.0) is True
