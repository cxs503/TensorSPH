"""Invariants of the closed-grid-ladder judgement (m20161).

What these tests protect
------------------------
``benchmarks/case_sph_wedge_entry_2d/ladder_closure.py`` turns the peak-Cp
question of CASE-SPH-WEDGE-ENTRY-2D into a *predeclared* decision:

* a grid is judgeable only when the estimator spread (max of the resolution-band
  spread and the early/late-frame jackknife) is within the case tolerance;
* the case verdict comes from the two **finest judgeable** grids and demands a
  stable residual sign -- one grid that happens to land inside +/-15 % is not a
  PASS (that is the trap documented in m18382);
* a Richardson estimate is only labelled ``trustworthy`` when the observed order
  is positive, inside the plausible range and consistent between grid pairs.

The tests build synthetic artefacts (no SPH run) so each branch of the rule is
exercised, and pin the tolerance so a future edit cannot relax it silently.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"


def _load():
    spec = importlib.util.spec_from_file_location(
        "ladder_closure", BENCH / "ladder_closure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = _load()


# --- synthetic artefacts ---------------------------------------------------- #
def write_ensemble(dirpath: Path, tag: str, dx: float, peak: float,
                   spread: float, jackknife: float, frames: str = "61") -> None:
    payload = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "config": {"dx": dx},
        "steps": 100, "wall_seconds": 1.0,
        "by_frames": {frames: {
            "n_frames": int(frames),
            "pooled_samples": 100,
            "mean_wet_nodes": 30.0,
            "peak_median": peak / 0.99,
            "spread_rel": spread + 0.01,
            "resolution_band": {"n_bins_used": [8, 10, 12],
                                "peak_median": peak,
                                "spread_rel": spread},
            "jackknife_rel": jackknife,
            "err_vs_wagner_median": peak / 8.402203300817021 - 1.0,
            "judgeable": spread < 0.15,
        }},
    }
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"ensemble_{tag}_c0v20.json").write_text(json.dumps(payload))


def write_bias(dirpath: Path, truths: dict) -> Path:
    grids = {}
    for tag, truth in truths.items():
        grids[tag] = {"dx": 0.0, "by_frames": {"61": {
            "obs_peak_median": float("nan"),
            "synth_smoothed": {"peak_median": truth, "spread_rel": 0.05},
        }}}
    payload = {"wagner_cp_max_point": 8.402203300817021, "grids": grids}
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / "bias_report.json"
    path.write_text(json.dumps(payload))
    return path


# --- the rule --------------------------------------------------------------- #
def test_case_gate_pinned_to_15_percent():
    """The judgement must not quietly adopt a looser tolerance."""
    assert lc.GATE == pytest.approx(0.15)
    assert lc.GRID_TAGS == ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")


def _two_grid_case(tmp_path, r_coarse, r_fine, u_coarse=0.10, u_fine=0.10):
    art, biasd = tmp_path / "art", tmp_path / "bias"
    truth = 7.0
    write_ensemble(art, "dx2mm", 0.002, truth * (1 + r_coarse), u_coarse, u_coarse)
    write_ensemble(art, "dx1mm", 0.001, truth * (1 + r_fine), u_fine, u_fine)
    bias = write_bias(biasd, {"dx2mm": truth, "dx1mm": truth})
    return lc.build(art, bias)


def test_pass_requires_two_judgeable_same_sign_grids(tmp_path):
    rep = _two_grid_case(tmp_path, -0.22, -0.12)
    assert rep["verdict"]["verdict"] == "PASS", rep["verdict"]
    assert [r["tag"] for r in rep["grids"]] == ["dx2mm", "dx1mm"]


def test_single_grid_inside_tolerance_is_not_a_pass(tmp_path):
    """Coarse grid outside, fine grid inside -> sign flip -> not decidable."""
    rep = _two_grid_case(tmp_path, +0.30, -0.10)
    assert rep["verdict"]["verdict"] == "NOT_DECIDABLE"
    assert "sign" in rep["verdict"]["reason"]


def test_fail_below_reports_side(tmp_path):
    rep = _two_grid_case(tmp_path, -0.30, -0.22)
    assert rep["verdict"]["verdict"] == "FAIL_below"


def test_fail_above_reports_side(tmp_path):
    rep = _two_grid_case(tmp_path, +0.30, +0.22)
    assert rep["verdict"]["verdict"] == "FAIL_above"


def test_unjudgeable_spread_blocks_the_verdict(tmp_path):
    rep = _two_grid_case(tmp_path, -0.25, -0.10, u_coarse=0.10, u_fine=0.40)
    assert rep["verdict"]["verdict"] == "NOT_DECIDABLE"
    assert rep["grids"][-1]["judgeable"] is False


def test_one_judgeable_grid_only(tmp_path):
    art, biasd = tmp_path / "art", tmp_path / "bias"
    write_ensemble(art, "dx2mm", 0.002, 5.0, 0.30, 0.30)
    write_ensemble(art, "dx1mm", 0.001, 6.0, 0.10, 0.10)
    bias = write_bias(biasd, {"dx2mm": 7.0, "dx1mm": 7.0})
    rep = lc.build(art, bias)
    assert rep["verdict"]["verdict"] == "NOT_DECIDABLE"
    assert "only 1 judgeable" in rep["verdict"]["reason"]


def test_missing_grid_is_skipped_not_zeroed(tmp_path):
    art, biasd = tmp_path / "art", tmp_path / "bias"
    write_ensemble(art, "dx4mm", 0.004, 10.0, 0.10, 0.10)
    write_ensemble(art, "dx1mm", 0.001, 6.0, 0.10, 0.10)
    bias = write_bias(biasd, {"dx4mm": 7.0, "dx1mm": 7.0})
    rep = lc.build(art, bias)
    tags = [r["tag"] for r in rep["grids"]]
    assert tags == ["dx4mm", "dx1mm"]                     # coarse -> fine
    assert all(math.isfinite(r["dx"]) for r in rep["grids"])


def test_grid_without_same_operator_truth_is_not_judgeable(tmp_path):
    art, biasd = tmp_path / "art", tmp_path / "bias"
    write_ensemble(art, "dx2mm", 0.002, 6.0, 0.05, 0.05)
    write_ensemble(art, "dx1mm", 0.001, 6.0, 0.05, 0.05)
    bias = write_bias(biasd, {"dx1mm": 7.0})              # no dx2mm truth
    rep = lc.build(art, bias)
    by_tag = {r["tag"]: r for r in rep["grids"]}
    assert by_tag["dx2mm"]["judgeable"] is False
    assert math.isnan(by_tag["dx2mm"]["residual"])


# --- observed order / Richardson -------------------------------------------- #
def _order_rows(residuals, dxs):
    return [{"tag": f"g{i}", "dx": dx, "residual": r, "obs_peak": 6.0,
             "synth_peak": 7.0, "judgeable": True, "uncertainty": 0.05}
            for i, (r, dx) in enumerate(zip(residuals, dxs))]


def test_second_order_residual_sequence_is_recovered():
    """r ~ dx^2 -> observed order 2 and Richardson pulls to the truth."""
    dxs = [0.004, 0.002, 0.001]
    residuals = [-0.40, -0.10, -0.025]
    rows = _order_rows(residuals, dxs)
    out = lc.observed_order(rows)
    assert out["available"] and out["trustworthy"]
    assert out["order_mean"] == pytest.approx(2.0, abs=0.05)
    # r_inf = r_fine/(2^p-1) = -0.025/3
    assert out["residual_asymptotic"] == pytest.approx(-0.025 / 3.0, rel=1e-6)
    assert out["obs_peak_asymptotic"] == pytest.approx(6.0 - (-0.025 / 3.0) * 7.0)


def test_diverging_residuals_are_not_trustworthy():
    """Residual growing under refinement (negative order) must not be trusted."""
    rows = _order_rows([-0.05, -0.10, -0.30], [0.004, 0.002, 0.001])
    out = lc.observed_order(rows)
    assert out["available"]
    assert out["trustworthy"] is False
    assert math.isnan(out["residual_asymptotic"])


def test_inconsistent_orders_are_not_trustworthy():
    """Pair orders that disagree by >30 % are not a converged order."""
    rows = _order_rows([-0.40, -0.08, -0.001], [0.004, 0.002, 0.001])
    out = lc.observed_order(rows)
    assert out["available"]
    assert out["consistent"] is False
    assert out["trustworthy"] is False


def test_order_needs_three_same_sign_grids():
    rows = _order_rows([-0.20, +0.10], [0.002, 0.001])
    out = lc.observed_order(rows)
    assert out["available"] is False


# --- committed-artefact facts ------------------------------------------------ #
@pytest.mark.skipif(not ART.exists(), reason="artefacts not in tree")
def test_committed_ensembles_are_self_similar_ensemble_runs():
    """Whatever rungs are committed use the same case definition and 61 frames."""
    rungs = sorted(ART.glob("ensemble_dx*_c0v20.json"))
    assert rungs, "no committed ensemble artefacts"
    for path in rungs:
        art = json.loads(path.read_text())
        assert art["case"] == "CASE-SPH-WEDGE-ENTRY-2D"
        assert "61" in art["by_frames"]
        rb = art["by_frames"]["61"]["resolution_band"]
        assert rb["peak_median"] == rb["peak_median"]      # finite
        assert art["design"]["spread_tolerance"] == pytest.approx(0.15)


def test_zhao_faltinsen_cell_stays_withheld(tmp_path):
    """The Z&F beta=30 table value must not be filled in from a secondary source."""
    rep = lc.build(tmp_path / "empty_art", tmp_path / "empty_bias.json")
    assert rep["zhao_faltinsen_1993_beta30"]["status"] == "WITHHELD"
    assert rep["verdict"]["verdict"] == "NOT_DECIDABLE"   # nothing to judge yet