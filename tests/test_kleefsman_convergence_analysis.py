"""Tests for the Kleefsman resolution-ladder analysis and protocol guard.

The guard exists because the committed ``resolution_study.json`` once mixed a
``dx = 0.01`` run stopped at ``t_end = 1.0 s`` with ``dx = 0.02 / 0.04`` runs
advanced to ``1.2 s`` and still reported the column as a ladder.  The tests
below pin (a) the pure numerology on synthetic ladders with a known order,
(b) the guard rejecting such a mixture, and (c) the invariant that whatever is
committed under ``results/`` is single-protocol.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "kleefsman_obstacle"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import convergence_analysis as ca  # noqa: E402


def synthetic_rows(exact: float, coeff: float, p: float, dxs=(0.04, 0.02, 0.01)):
    """f(dx) = exact + coeff * dx**p, i.e. an exactly asymptotic ladder."""
    return [{"dx": dx, "t_end": 1.6, "sound_speed": 20.0, "probe_offset": 0.05,
             "probe_halfwidth": 1.0, "surface_particles": 3, "surface_gap": 2.5,
             "arrival_threshold": 0.02, "h_factor": 1.3,
             "value": exact + coeff * dx ** p, "arrival_time_s": 0.44,
             "peak_height_m": exact + coeff * dx ** p} for dx in dxs]


# --------------------------------------------------------------------------- #
# pure numerology
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0])
def test_observed_order_recovers_exact_order(p_true):
    rows = ca.ladder(synthetic_rows(0.30, -5.0, p_true))
    m = ca.analyze_metric(rows, "value")
    assert m["p_finest"] == pytest.approx(p_true, rel=1e-9)


def test_richardson_recovers_exact_value_and_shrinks_gci():
    rows = ca.ladder(synthetic_rows(0.30, 3.0, 1.2))
    m = ca.analyze_metric(rows, "value")
    assert m["richardson"] == pytest.approx(0.30, rel=1e-9)
    assert m["gci_fine"] < 0.10          # a clean ladder is "converged" at 10 %
    assert m["verdict"] == "CONVERGED"


def test_non_monotone_sequence_is_not_converged():
    rows = ca.ladder(synthetic_rows(0.30, 1.0, 1.0))
    rows[1]["value"] = rows[2]["value"] + 0.5   # break monotonicity
    m = ca.analyze_metric(rows, "value")
    assert m["verdict"] == "NOT_CONVERGED"
    assert math.isnan(m["p_finest"]) or not m["monotone"]


def test_observed_order_undefined_when_two_grids_agree():
    assert math.isnan(ca.observed_order(0.3, 0.3, 0.3, 2.0))


def test_gci_uses_safety_factor_linearly():
    a = ca.gci_fine(0.40, 0.35, 0.33, 1.0, 2.0, safety=1.25)
    b = ca.gci_fine(0.40, 0.35, 0.33, 1.0, 2.0, safety=2.50)
    assert b == pytest.approx(2.0 * a)


# --------------------------------------------------------------------------- #
# moving average (gauge-equivalent level)
# --------------------------------------------------------------------------- #
def test_moving_average_preserves_constant_and_damps_spike():
    times = [i * 0.01 for i in range(101)]
    flat = [0.25] * len(times)
    assert max(abs(v - 0.25) for v in ca.moving_average(times, flat, 0.03)) < 1e-12
    spiky = list(flat)
    spiky[50] = 0.90                     # one ballistic spray sample
    # +-0.015 s at 0.01 s spacing keeps the two neighbours: one 0.90 sample is
    # diluted 1/3 -> 0.467, versus the raw 0.90 the instantaneous peak reports.
    assert max(ca.moving_average(times, spiky, 0.03)) == pytest.approx((0.25 + 0.90 + 0.25) / 3)
    assert max(ca.moving_average(times, spiky, 0.15)) < 0.40
    assert max(spiky) == 0.90            # raw peak would report 0.90


def test_moving_average_window_is_time_based():
    coarse = [0.0, 0.01, 0.02, 0.03, 0.04]
    fine = [i * 0.0025 for i in range(17)]
    a = ca.moving_average(coarse, [0.1, 0.4, 0.1, 0.4, 0.1], 0.02)
    b = ca.moving_average(fine, [0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1] +
                          [0.1] * 8, 0.02)
    assert len(a) == len(coarse) and len(b) == len(fine)
    assert abs(max(a) - 0.28) < 0.02 and abs(max(b) - 0.28) < 0.05


# --------------------------------------------------------------------------- #
# protocol guard
# --------------------------------------------------------------------------- #
def test_check_protocol_accepts_identical_fingerprints():
    assert ca.check_protocol(synthetic_rows(1.0, 1.0, 1.0)) == []


def test_check_protocol_flags_mixed_t_end():
    rows = synthetic_rows(1.0, 1.0, 1.0)
    rows[2]["t_end"] = 1.0               # the historical defect
    diff = ca.check_protocol(rows)
    assert diff and diff[0]["dx"] == 0.01
    assert "t_end" in diff[0]["differs"]


def test_check_protocol_flags_probe_and_sound_speed_drift():
    rows = synthetic_rows(1.0, 1.0, 1.0)
    rows[1]["probe_offset"] = 0.10
    rows[2]["sound_speed"] = 40.0
    keys = [list(d["differs"]) for d in ca.check_protocol(rows)]
    assert ["probe_offset"] in keys and ["sound_speed"] in keys


def test_incomparable_ladder_is_refused_not_scored():
    rows = synthetic_rows(0.30, 3.0, 1.2)
    rows[2]["t_end"] = 1.0
    rep = ca.analyse_rows(rows, BENCH / "results")
    assert rep["verdict"] == "INCOMPARABLE"
    assert rep["metrics"] == {}


# --------------------------------------------------------------------------- #
# invariants on the committed evidence
# --------------------------------------------------------------------------- #
def test_committed_resolution_study_never_mixes_silently():
    """Either the committed ladder is single-protocol, or the mixture is declared.

    The historical defect was a *silent* mixture (dx=0.01 at t_end=1.0 next to
    dx=0.04/0.02 at 1.2).  Declared mixtures are allowed because
    ``convergence_analysis --window-t`` equalises the horizon from the traces.
    """
    study = BENCH / "results" / "resolution_study.json"
    if not study.exists():               # pragma: no cover - evidence not checked out
        pytest.skip("no committed resolution study")
    doc = json.loads(study.read_text())
    found = ca.check_protocol(doc["runs"])
    assert found == [] or doc.get("protocol_mismatch") == found, (
        "committed ladder mixes protocols without declaring it; see convergence.py")


def test_committed_study_is_monotone_in_dx():
    study = BENCH / "results" / "resolution_study.json"
    if not study.exists():               # pragma: no cover
        pytest.skip("no committed resolution study")
    dxs = [r["dx"] for r in json.loads(study.read_text())["runs"]]
    assert dxs == sorted(dxs, reverse=True) and len(set(dxs)) == len(dxs)

# --------------------------------------------------------------------------- #
# mass-based slab level (spray-invariant level estimator)
# --------------------------------------------------------------------------- #
def test_column_level_equals_depth_of_uniform_slab():
    import torch
    from tensorsph.kleefsman_obstacle import KleefsmanConfig, column_level
    cfg = KleefsmanConfig(dx=0.01)
    half = cfg.probe_halfwidth * cfg.dx
    # 2 columns of particles symmetrical about the probe station, height 0.30 m
    ys = torch.arange(0.005, 0.30, 0.01)
    xs = torch.tensor([cfg.probe_x() - 0.5 * half, cfg.probe_x() + 0.5 * half]).repeat(
        ys.numel()).reshape(2, -1).T.reshape(-1)
    pos = torch.stack((xs, ys.repeat(2)), dim=-1)
    level = column_level(pos, cfg)
    # 2 rows x 30 particles x dx^2 / (2*halfwidth) = slab mean depth
    expected = pos.shape[0] * cfg.dx ** 2 / (2.0 * half)
    assert level == pytest.approx(expected)
    assert level == pytest.approx(0.30, rel=0.05)   # 30 rows of dx = 0.30 m deep


def test_column_level_is_invariant_to_spray_redistribution():
    """Lifting one particle into ballistic spray must not raise the level."""
    import torch
    from tensorsph.kleefsman_obstacle import KleefsmanConfig, column_level
    cfg = KleefsmanConfig(dx=0.01)
    half = cfg.probe_halfwidth * cfg.dx
    ys = torch.arange(0.005, 0.20, 0.01)
    xs = torch.tensor([cfg.probe_x()]).repeat(ys.numel())
    base = torch.stack((xs, ys), dim=-1)
    lifted = base.clone()
    lifted[-1, 1] = 0.80                     # one particle thrown up 0.6 m
    assert column_level(lifted, cfg) == pytest.approx(column_level(base, cfg), rel=1e-12)


def test_column_level_ignores_fluid_outside_the_slab():
    import torch
    from tensorsph.kleefsman_obstacle import KleefsmanConfig, column_level
    cfg = KleefsmanConfig(dx=0.01)
    half = cfg.probe_halfwidth * cfg.dx
    inside = torch.tensor([[cfg.probe_x(), 0.05], [cfg.probe_x(), 0.06]])
    outside = torch.tensor([[cfg.probe_x() - 5 * half, 0.05]])
    assert column_level(torch.cat((inside, outside)), cfg) == pytest.approx(
        column_level(inside, cfg))
