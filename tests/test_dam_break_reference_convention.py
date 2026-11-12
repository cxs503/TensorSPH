"""Guards for the dam-break experimental gate's frame convention and reference grade.

Context: ``benchmarks/verification_ledger/ledger.json`` graded
``dam_break.martin_moyce.mae`` FAIL against the bundled Martin & Moyce table.
Two instrument facts decide whether that FAIL may stand, and both are pinned
here:

* the bundled table is generated (exact ``tau`` lattice, ordinates exact
  multiples of 0.05) and is graded ``usable_as_validation_target: false`` by
  ``benchmarks/dam_break/reference_audit.json``;
* ``validate.py`` used to gate on the lab-frame front position ``z/a`` (``= 1``
  at ``t = 0``) while the reference is a displacement ``Z/a = (x_front - a)/a``.

See ``docs/verification_dam_break_reference_convention.md``.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "dam_break"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit_module():
    return _load("dam_break_convention_audit", "convention_audit.py")


@pytest.fixture(scope="module")
def validate_module():
    return _load("dam_break_validate", "validate.py")


@pytest.fixture(scope="module")
def bundled():
    return json.loads((BENCH / "martin_moyce_1952.json").read_text())


@pytest.fixture(scope="module")
def crosscheck():
    return json.loads((BENCH / "martin_moyce_1952_crosscheck.json").read_text())


def test_bundled_table_is_synthetic(audit_module, bundled):
    diag = audit_module.bundled_table_diagnostics(bundled)
    assert diag["n_points"] >= 10
    # exact arithmetic abscissa: constant step to round-off
    assert diag["tau_step_spread"] < 1e-12
    # every ordinate an exact multiple of the 0.05 quantum
    assert diag["ordinate_quantum_residual"] < 1e-9
    assert diag["usable_as_validation_target"] is False


def test_reference_audit_agrees_with_recomputation(audit_module, bundled):
    stored = json.loads((BENCH / "reference_audit.json").read_text())
    assert stored["table_audit"]["usable_as_validation_target"] is False
    recomputed = audit_module.bundled_table_diagnostics(bundled)
    assert recomputed["n_points"] == stored["table_audit"]["n_points"]


def test_crosscheck_table_uses_the_wall_frame(audit_module, crosscheck):
    diag = audit_module.crosscheck_table(crosscheck)
    assert diag["offset_is_one"] is True, (
        "the crosscheck table stores X = x_front/a measured from the wall, so "
        "Z/a = X - 1; a table whose first point is not 1.0 would need a "
        "different conversion")
    assert diag["n_points"] == len(crosscheck["data"])
    assert diag["tau_range"][0] == pytest.approx(0.0)


def test_validate_gate_is_withheld_on_an_inadmissible_reference(validate_module):
    # the shipped table is synthetic, so the gate must not emit pass/fail
    assert validate_module._reference_usable() is False


def test_validate_publishes_the_experiment_frame_ordinate():
    source = (BENCH / "validate.py").read_text()
    assert '"Z_over_a": (z - cfg.column_width) / cfg.column_width' in source
    assert "sample.get(\"Z_over_a\"" in source


def test_lab_frame_bias_is_one_at_zero_shift(audit_module, bundled):
    """A trajectory that equals the reference in the converted frame is off by
    exactly +1 in the lab frame: that is the defect the audit quantifies."""
    tau = np.array([d["tau"] for d in bundled["data"]], float)
    za = np.array([d["Z_over_a"] for d in bundled["data"]], float)
    unc = np.array([d.get("uncertainty", 0.05) for d in bundled["data"]], float)

    converted = audit_module.error_metrics(tau, za, tau, za, unc)
    lab = audit_module.error_metrics(tau, za + 1.0, tau, za, unc)

    assert converted["mean_absolute_error"] == pytest.approx(0.0, abs=1e-12)
    assert lab["mean_absolute_error"] == pytest.approx(1.0, abs=1e-12)
    assert lab["max_absolute_error"] == pytest.approx(1.0, abs=1e-12)


def test_audit_artifact_regrades_the_claim():
    path = BENCH / "convention_audit.json"
    if not path.exists():  # artifact is generated; the guard is skipped if absent
        pytest.skip("convention_audit.json not generated in this checkout")
    data = json.loads(path.read_text())
    assert data["shipped_gate"]["result"] == "fail"
    assert data["grading"]["regraded"] == "withheld"
    comp = data["comparison"]
    # the converted frame must be the smaller error against the same table
    assert (comp["displacement"]["vs_bundled_Z_over_a"]["mean_absolute_error"]
            < comp["lab"]["vs_bundled_Z_over_a"]["mean_absolute_error"])
    # and the burst is ~1 in Z/a per point, never negative
    assert data["frame_offset_mean_MAE_bundled"] > 0.0
    # the only provenance-bearing table gives a sub-gate diagnostic
    assert comp["displacement"]["vs_crosscheck_X_minus_1"]["points_compared"] >= 10
    assert math.isfinite(comp["displacement"]["vs_crosscheck_X_minus_1"]["rmse"])