"""R1 (constant-forcing) ladder for CASE-POISEUILLE-SPH.

Until this increment the R1 regime — steady parabola (G7) and Boussinesq
start-up (G8) — carried a **single** resolution (H/dx = 12), so its two anchors
could only be compared with the 5 % tolerance: one number supports no observed
order, no GCI and no uncertainty band.  R2 already had that machinery; R1 did
not, and the note said so ("status: insufficient").

These tests pin the increment, and they are written as *invariants* rather than
"does the script run":

1. the committed artifact really carries the ladder, and every rung passes;
2. the start-up rows committed by the driver reproduce the library's Boussinesq
   series and the driver's own per-rung RMS — i.e. the stored evidence is the
   quantity the gate grades (this is the assertion that would have caught the
   phase/unit defect of m21208);
3. the uncertainty layer's R1 functional, taken on each rung's own rows,
   reproduces the driver's graded number *exactly* — one definition, two
   consumers;
4. the published bands support the 5 % acceptance, and the start-up band declares
   its zero ``U_input`` instead of inventing one;
5. a legacy single-rung artifact still degrades to "insufficient" (honest
   fallback, no silent pass).

Run:  pytest tests/test_case_poiseuille_r1_ladder.py -v
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_poiseuille"))

from tensorsph import womersley as wm  # noqa: E402
import uncertainty as case_unc  # noqa: E402  (the uncertainty layer)

CASE = REPO / "benchmarks" / "case_poiseuille"
RESULT = CASE / "out" / "result.json"
UNC = CASE / "out" / "uncertainty.json"

pytestmark = pytest.mark.skipif(not RESULT.is_file(), reason="case artifact absent")


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def published() -> dict:
    if not UNC.is_file():
        pytest.skip("uncertainty artifact absent")
    return json.loads(UNC.read_text())


def _interior(y_over_H: float) -> bool:
    lo, hi = case_unc.INTERIOR
    return lo <= y_over_H <= hi


def _rms(values) -> float:
    vals = [float(v) for v in values]
    return math.sqrt(sum(v * v for v in vals) / len(vals))


# --------------------------------------------------------------------------- #
# 1. the ladder exists and passes
# --------------------------------------------------------------------------- #
def test_r1_ladder_has_three_rungs(result: dict) -> None:
    ladder = result["constant_ladder"]
    assert ladder["resolutions"] == [8, 12, 16]
    assert [r["resolution_H_over_dx"] for r in ladder["runs"]] == [8, 12, 16]
    # the anchor rung stays at H/dx=12 so G7/G8 remain comparable with the
    # artifact that was already published
    assert ladder["anchor_resolution_H_over_dx"] == 12
    assert result["constant_run"]["resolution_H_over_dx"] == 12


def test_every_r1_rung_passes_the_tolerance(result: dict) -> None:
    tol = result["thresholds"]["G7_steady_anchor_rms_pct"]
    for run in result["constant_ladder"]["runs"]:
        assert run["steady_profile_rms_pct"] <= tol, run["resolution_H_over_dx"]
        assert run["startup_rms_pct_at_040"] <= tol, run["resolution_H_over_dx"]
        assert run["gates"]["G7_steady_anchor_5pct"] is True
        assert run["gates"]["G8_startup_anchor_5pct"] is True
    gates = result["constant_ladder"]["gates"]
    assert gates["G7b_all_rungs_steady_5pct"] is True
    assert gates["G8b_all_rungs_startup_5pct"] is True
    assert result["gates"]["G7b_all_rungs_steady_5pct"] is True
    assert result["gates"]["G8b_all_rungs_startup_5pct"] is True


def test_ladder_is_finer_than_the_anchor_and_not_a_duplicate(result: dict) -> None:
    """The rungs must be genuinely different resolutions, not the same run relabelled."""
    runs = result["constant_ladder"]["runs"]
    n_fluid = {r["resolution_H_over_dx"]: r["n_fluid"] for r in runs}
    assert len(set(n_fluid.values())) == len(n_fluid)
    dts = {r["resolution_H_over_dx"]: r["dt"] for r in runs}
    assert dts[8] > dts[12] > dts[16]


# --------------------------------------------------------------------------- #
# 2. the stored start-up rows are the graded quantity
# --------------------------------------------------------------------------- #
def test_startup_rows_reproduce_the_boussinesq_series(result: dict) -> None:
    height = result["geometry"]["height"]
    nu = result["geometry"]["viscosity"]
    g0 = result["geometry"]["body_force"]
    for run in result["constant_ladder"]["runs"]:
        for start in run["startup"]:
            rows = start["rows"]
            # one measurement row per particle spacing, as in the steady profile
            assert len(rows) == run["resolution_H_over_dx"]
            y = [r["y_over_H"] * height for r in rows]
            ref = wm.startup_velocity(y, start["t_actual"], height, g0, nu)
            got = [r["u_analytic"] for r in rows]
            assert got == pytest.approx(ref, rel=1e-9, abs=1e-12)

            rel = [(r["u_measured"] - a) / a
                   for r, a in zip(rows, got) if _interior(r["y_over_H"])]
            assert start["profile_rms_pct"] == pytest.approx(
                100.0 * _rms(rel), rel=1e-9, abs=1e-9)


def test_steady_rows_reproduce_the_parabola_anchor(result: dict) -> None:
    height = result["geometry"]["height"]
    nu = result["geometry"]["viscosity"]
    g0 = result["geometry"]["body_force"]
    for run in result["constant_ladder"]["runs"]:
        rows = run["steady_rows"]
        y = [r["y_over_H"] * height for r in rows]
        ref = wm.steady_velocity(y, height, g0, nu)
        assert [r["u_analytic"] for r in rows] == pytest.approx(ref, rel=1e-9, abs=1e-12)
        rel = [(r["u_measured"] - a) / a
               for r, a in zip(rows, ref) if _interior(r["y_over_H"])]
        assert run["steady_profile_rms_pct"] == pytest.approx(
            100.0 * _rms(rel), rel=1e-9, abs=1e-9)


def test_per_sample_steady_functional_brackets_the_mean(result: dict) -> None:
    """The repeatability samples must be samples of the *same* functional."""
    for run in result["constant_ladder"]["runs"]:
        samples = run["steady_rms_samples_pct"]
        assert len(samples) == run["steady_rows_used"]
        # averaging samples reduces the error, but the mean-of-samples and the
        # functional-of-the-mean cannot disagree by more than the sample spread
        spread = max(samples) - min(samples)
        assert abs(sum(samples) / len(samples) - run["steady_profile_rms_pct"]) \
            <= spread + 1e-9


# --------------------------------------------------------------------------- #
# 3. one definition, two consumers
# --------------------------------------------------------------------------- #
def test_uncertainty_layer_reproduces_the_drivers_own_rows_number(result: dict) -> None:
    """On each rung's own rows the two layers must agree exactly.

    The published layer defaults to a common y grid (correct for *orders*), so the
    comparison is made on the ``common_grid=False`` variant: that variant is the
    driver's definition, and it must match to round-off.  A silent redefinition
    here is how a gate number and its published error drift apart.
    """
    q = case_unc.r1_quantities(result, common_grid=False)
    assert set(q) == {"r1_steady_profile_rms", "r1_startup_profile_rms_at_0.40s"}
    by_res = {r["resolution_H_over_dx"]: r for r in result["constant_ladder"]["runs"]}
    for key in q:
        for dx, val, lab in zip(q[key]["spacings"], q[key]["values"], q[key]["labels"]):
            res = int(lab.split("=")[1])
            run = by_res[res]
            assert dx == pytest.approx(result["geometry"]["height"] / res, rel=1e-12)
            if key == "r1_steady_profile_rms":
                expect = run["steady_profile_rms_pct"] / 100.0
            else:
                expect = run["startup_rms_pct_at_040"] / 100.0
            assert val == pytest.approx(expect, rel=1e-9, abs=1e-12)


def test_r1_quantities_carry_a_zero_truth_and_unit_scale(result: dict) -> None:
    q = case_unc.r1_quantities(result)
    for key, spec in q.items():
        assert spec["truth"] == [0.0] * len(spec["truth"]), key
        assert spec["scale"] == 1.0, key
        assert len(spec["values"]) == 3, key


# --------------------------------------------------------------------------- #
# 4. the published verdict
# --------------------------------------------------------------------------- #
def test_published_r1_verdict_covers_all_three_rungs(published: dict) -> None:
    r1 = published["r1_quantities"]
    assert set(r1) == {"r1_steady_profile_rms", "r1_startup_profile_rms_at_0.40s"}
    for key, q in r1.items():
        assert q["n_rungs"] == 3, key
        assert q["order_analysis"]["status"] in {"converging", "order_unresolved",
                                                 "oscillatory", "stalled"}, key
        assert q["U_num_percent"] is not None, key
        # the point error and every rung are inside the tolerance ...
        assert q["validation"]["E_percent"] <= published["tolerance_percent"], key
        assert q["worst_rung_within_tolerance"] is True, key
        assert q["validation"]["verdict"] == "validated", key
    # ... but the band only supports the tolerance for the start-up functional.
    # The steady band is dominated by the residual fill of the channel over the
    # 0.80..1.00 s window, which is reported instead of being tuned away.
    assert r1["r1_startup_profile_rms_at_0.40s"]["validation"]["supports_tolerance"] is True
    steady = r1["r1_steady_profile_rms"]["validation"]
    assert steady["supports_tolerance"] is False
    assert steady["U_V_percent"] > published["tolerance_percent"]


def test_published_r1_gates_are_graded_not_skipped(published: dict) -> None:
    gates = published["gates"]
    for key in ("U1c_R1_ladder_all_rungs_within_5pct",
                "U2b_R1_validation_E_leq_U_V",
                "U3b_R1_band_supports_5pct"):
        assert key in gates, key
        assert gates[key] is not None, key          # graded, not skipped
    assert gates["U1c_R1_ladder_all_rungs_within_5pct"] is True
    assert gates["U2b_R1_validation_E_leq_U_V"] is True
    # honest FAIL: the steady band is wider than the 5 % acceptance
    assert gates["U3b_R1_band_supports_5pct"] is False
    assert published["all_gates_pass"] is False
    # the case's own acceptance gates are untouched by that finding
    for key in ("U1_all_rungs_within_5pct", "U3_band_supports_5pct"):
        assert gates[key] is True, key
    # the G-gates live in the driver artifact, not in the uncertainty note
    driver_gates = json.loads(RESULT.read_text())["gates"]
    for key in ("G7_steady_anchor_5pct", "G8_startup_anchor_5pct",
                "G7b_all_rungs_steady_5pct", "G8b_all_rungs_startup_5pct"):
        assert driver_gates[key] is True, key
    assert published["single_rung"]["status"] == "ladder"
    assert published["single_rung"]["ladder_resolutions_H_over_dx"] == [8, 12, 16]


def test_startup_input_uncertainty_is_declared_zero(published: dict) -> None:
    """No invented U_input: a single instant has no repeatability."""
    u_in = published["r1_input_uncertainty"]
    assert u_in["startup"] == 0.0
    assert "single instant" in u_in["startup_reason"]
    assert u_in["steady_half_range_percent_of_mean"] >= 0.0
    assert u_in["steady_samples"]["n"] >= 2
    # the estimate must come from the finest rung (the coarsest one is noisier
    # and would misstate the run's own repeatability)
    assert u_in["resolution_H_over_dx"] == max(
        r["resolution_H_over_dx"] for r in json.loads(RESULT.read_text())
        ["constant_ladder"]["runs"])


# --------------------------------------------------------------------------- #
# 5. honest fallback for a legacy single-rung artifact
# --------------------------------------------------------------------------- #
def test_legacy_single_rung_artifact_degrades_to_insufficient(result: dict) -> None:
    legacy = copy.deepcopy(result)
    legacy.pop("constant_ladder", None)
    payload = case_unc.analyse(legacy)
    assert payload["r1_quantities"] == {}
    assert payload["single_rung"]["status"] == "insufficient"
    assert "single resolution" in payload["single_rung"]["reason"]
    # an ungraded gate must be None and must not make the case pass by accident
    assert payload["gates"]["U1c_R1_ladder_all_rungs_within_5pct"] is None
    assert payload["gates"]["U3b_R1_band_supports_5pct"] is None