"""Gates for round 6 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (pressure readout).

These tests do **not** re-run the sweep (104 922 particles at the top level): they
gate the *record* the sweep wrote and the claims the doc makes from it, so that a
later change of the model, of the probes or of the doc cannot silently leave a
"converging readout" story behind.

The honest-FAIL is asserted as a FAIL: `bias_convergence_order_min` is recorded
below its declared threshold, and the owner 15 % gate at a:b = 3:1,
`R_eq/dx = 40` is recorded broken.  A future round that closes them must edit
this file deliberately, not by accident.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "scripts" / "xval_projection_laplace" / "results"
sys.path.insert(0, str(ROOT / "scripts" / "xval_projection_laplace"))


@pytest.fixture(scope="module")
def ladder():
    return json.loads((RES / "readout_convergence.json").read_text())


@pytest.fixture(scope="module")
def diag():
    return json.loads((RES / "diag_pressure_profile.json").read_text())


def levels(ladder, aspect):
    return sorted([l for l in ladder["levels"] if l["aspect_ratio"] == aspect],
                  key=lambda l: l["r_eq_over_dx"])


def test_ladder_shape(ladder):
    assert ladder["anchor"] == "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE"
    assert len(ladder["levels"]) == 6
    for aspect, dxs in ((2.0, [20.0, 40.0, 80.0]), (3.0, [20.0, 40.0, 80.0])):
        assert [l["r_eq_over_dx"] for l in levels(ladder, aspect)] == dxs


def test_declared_gates_are_the_module_constants(ladder):
    """The declared gates in the JSON must be the ones in the script (no post-hoc edit)."""
    import readout_convergence as rc

    for k, v in ladder["gates_declared"].items():
        assert rc.GATES[k] == v, k


def test_readout_bias_decays_with_tip_resolution(ladder):
    """Monotone decay of the *reported* jump bias with R_curv,min/dx, both aspects."""
    for aspect in (2.0, 3.0):
        lv = levels(ladder, aspect)
        b = [l["pressure_stats_shipped"]["mean_pct"] for l in lv]
        rc = [l["r_curv_min_over_dx"] for l in lv]
        assert all(x > y for x, y in zip(b, b[1:])), (aspect, b)
        assert b[-1] < 0.6 * b[0], (aspect, b)


def test_bias_ratio_gate_passes_and_order_gate_fails(ladder):
    """Predeclared: ratio(80/40) <= 0.8 (passes) and order >= 0.5 (honest FAIL)."""
    v = ladder["gate_verdicts"]
    assert v["bias_ratio_80_over_40_aspect_3"]["pass"] is True
    assert v["bias_ratio_80_over_40_aspect_3"]["measured"] < 0.8
    # the honest failure is part of the record
    assert v["bias_convergence_order_min"]["pass"] is False
    assert 0.3 < v["bias_convergence_order_min"]["measured"] < 0.5


def test_owner_gate_still_broken_at_aspect_3_r40(ladder):
    """The owner 15 % readout gate at a:b = 3:1, R_eq/dx = 40 is NOT retro-fitted."""
    l = [x for x in levels(ladder, 3.0) if x["r_eq_over_dx"] == 40.0][0]
    assert l["pressure_stats_shipped"]["mean_pct"] > 15.0
    assert l["pressure_stats_shipped"]["mean_pct"] == pytest.approx(15.94, abs=0.05)


def test_traction_integral_is_resolution_convergent(ladder):
    """The force model is fine at every level: |flux bias| <= declared 15 %, <= 5 % in fact."""
    gate = ladder["gates_declared"]["flux_bias_pct_all_levels"]
    for l in ladder["levels"]:
        assert abs(l["flux_err"]["mean"]) <= gate
        assert abs(l["flux_err"]["mean"]) <= 5.0, l["flux_err"]
        assert abs(l["whitney_graustein_err_pct"]) <= 1.0


def _plateau(level, lo=None, hi=None):
    for b in level["plateau_flatness"]:
        if lo is not None and b["s_lo_dx"] <= lo:
            return b["p_mean"]
        if hi is not None and b["s_lo_dx"] >= hi:
            return b["p_mean"]
    raise AssertionError("bin not found")


def test_vapour_plateau_is_not_zero_and_grows_with_resolution(ladder):
    """Finding 3 of the doc: the vapour reference is contaminated, and the
    contamination grows with resolution -- that is why the order is 0.44."""
    for aspect in (2.0, 3.0):
        far = [_plateau(l, hi=6.0) for l in levels(ladder, aspect)]
        assert all(p > 0.0 for p in far), (aspect, far)
        assert all(x < y for x, y in zip(far, far[1:])), (aspect, far)


def test_cg_converged_at_every_level(ladder):
    for l in ladder["levels"]:
        assert 50 < l["cg_iterations"] < 2000
        assert l["resolved"] is True or l["r_curv_min_over_dx"] < 5.0


def test_diag_marks_the_shell_arms_as_station_blind(diag):
    """The diagnostic's moved/wide readouts were global shells: they must read the
    same number at every station, which is why finding 5 records them as *not run*
    as a station-resolved test.  If a later round makes them per-station, this test
    fails and forces the doc claim to be revisited."""
    dp = {d["moved_probes"]["dp"] for d in diag["stations"]}
    assert len(dp) == 1
    assert diag["summary"]["moved"]["err_rms_pct"] > 100.0


def test_doc_reports_the_honest_fail():
    doc = (ROOT / "docs" / "xval_projection_laplace_readout_convergence.md").read_text()
    assert "Honest-FAIL" in doc
    assert "bias_convergence_order_min" in doc
    assert "+15.94" in doc
