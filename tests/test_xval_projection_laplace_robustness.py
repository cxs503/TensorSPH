"""Gates for round 2 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (error budget).

Round 1 (``docs/xval_projection_laplace_oss.md``) measured the projection-SPH
Young-Laplace jump at *one* set-up.  Round 2 (``docs/xval_projection_laplace_robustness.md``)
asks whether that number is a property of the method or of the set-up, by
varying, one at a time, every knob the physics cannot see: the interface
position inside the lattice cell, the lattice orientation, the Dirichlet-gauge
distance, the readout band, the kernel support, and the resolution.

The gates below were declared before the numbers were inspected (owner gate
15 % on the jump).  They are deliberately stated as *spreads* of a knob sweep,
not as single point values, so a change that improves the fitted number while
making it more knob-dependent cannot pass.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "xval_projection_laplace"))

from tensorsph.laplace_isph import hex_block, laplace_isph_case  # noqa: E402

import robustness as rb  # noqa: E402

torch.set_num_threads(2)

RESULTS = ROOT / "scripts" / "xval_projection_laplace" / "results"
FULL_JSON = RESULTS / "robustness.json"
ROUND1_JSON = RESULTS / "xval_projection_laplace.json"


@pytest.fixture(scope="module")
def audit():
    return rb.run_audit(quick=True)


# --------------------------------------------------------------------------- #
# API contract: the audits must not move the shipped baseline
# --------------------------------------------------------------------------- #
def test_default_center_is_bit_identical():
    a = hex_block(0.02, 0.002, 0.012)
    b = hex_block(0.02, 0.002, 0.012, center=(0.0, 0.0), angle=0.0)
    assert a[0].shape == b[0].shape
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[2], b[2])


def test_round1_baseline_reproduced():
    """The round-1 finest-level number must survive the centre/angle refactor."""
    if not ROUND1_JSON.exists():  # pragma: no cover
        pytest.skip("round-1 result file not present")
    r1 = json.loads(ROUND1_JSON.read_text())
    ref = [r for r in r1["runs"] if r["arm"].startswith("ours[") and r["dx_m"] == 0.001]
    assert ref, "round-1 ours row at dx = 1 mm not found"
    rep = laplace_isph_case(radius=0.02, dx=0.001, curvature_method="mls")
    assert abs(rep["dp_error_pct"] - ref[0]["dp_error_pct"]) < 1e-9, (
        rep["dp_error_pct"], ref[0]["dp_error_pct"])


def test_rotated_lattice_has_the_same_density():
    """A rotation must rebuild the same triangular lattice, not a sparser one."""
    n0 = hex_block(0.02, 0.002, 0.012)[0].shape[0]
    n30 = hex_block(0.02, 0.002, 0.012, angle=math.radians(30.0))[0].shape[0]
    assert abs(n30 - n0) / n0 < 0.05, (n0, n30)


# --------------------------------------------------------------------------- #
# invariance gates (quick sweep, ~10 s)
# --------------------------------------------------------------------------- #
def test_translation_invariance(audit):
    ax = audit["axes"]["translation"]
    assert ax["spread"]["spread_pct"] < rb.GATES["translation_spread_pct"], ax["spread"]
    # physics: a static drop is translation invariant, so no offset may be
    # outside the owner gate on its own
    assert ax["spread"]["max_abs_pct"] < rb.GATES["single_config_abs_err_pct"]


def test_orientation_invariance(audit):
    ax = audit["axes"]["orientation"]
    assert ax["spread"]["spread_pct"] < rb.GATES["orientation_spread_pct"], ax["spread"]


def test_gauge_distance_is_a_gauge(audit):
    ax = audit["axes"]["gauge"]
    assert ax["spread"]["spread_pct"] < rb.GATES["gauge_spread_pct"], ax["spread"]


def test_estimator_readout_is_stable(audit):
    ax = audit["axes"]["estimator"]
    assert ax["spread"]["spread_pct"] < rb.GATES["estimator_spread_pct"], ax["spread"]


def test_two_instruments_agree(audit):
    """Pressure jump and the pressure-free traction flux must tell one story."""
    rows = [r for ax in audit["axes"].values() for r in ax["rows"]]
    worst = max(abs(r["dp_pressure_err_pct"] - r["dp_flux_err_pct"]) for r in rows)
    assert worst < rb.GATES["single_config_abs_err_pct"], worst


def test_quick_gate_matrix_is_green(audit):
    assert all(audit["summary"]["gate_check"].values()), audit["summary"]["gate_check"]


# --------------------------------------------------------------------------- #
# the committed full audit, if present
# --------------------------------------------------------------------------- #
def test_full_audit_on_disk_is_green():
    if not FULL_JSON.exists():  # pragma: no cover
        pytest.skip("full audit not run in this checkout")
    payload = json.loads(FULL_JSON.read_text())
    summary = payload["summary"]
    assert summary["n_equilibria"] >= 50
    assert all(summary["gate_check"].values()), summary["gate_check"]
    # the honest error bar of the claim must stay inside the owner gate
    assert summary["max_abs_err_pct"] < 15.0, summary["max_abs_err_pct"]
    # and both instruments must agree at that worst configuration
    assert summary["max_abs_flux_err_pct"] < 15.0
    # every axis carries the two instruments
    for name, ax in payload["axes"].items():
        assert ax["rows"] and "dp_flux_err_pct" in ax["rows"][0], name