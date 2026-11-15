"""Regression tests for round 7 (observed order + GCI + ASME V&V 20).

The round-7 machinery is *decision* code: it turns a ladder into a verdict about
whether the anchor's ``R/dx >= 24`` pass is asymptotic.  A decision code that is
only ever run on the real data cannot be falsified, so every test here feeds it
synthetic data whose answer is known in closed form -- including a dataset with a
*deliberate asymptotic bias*, which the verdict must reject.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "oscillating_drop_freq"
sys.path.insert(0, str(BENCH))

R7 = pytest.importorskip("round7_convergence")


def test_observed_order_recovers_a_known_exponent():
    for p_true in (1.0, 1.5, 2.0):
        rdx = [12.0, 16.0, 24.0, 32.0, 40.0]
        q = [1.0 - 0.6 * (1.0 / v) ** p_true for v in rdx]
        fit = R7.observed_order(rdx, q)
        assert fit["status"] == "ok"
        assert fit["p_observed"] == pytest.approx(p_true, abs=5e-3)
        assert fit["q_inf"] == pytest.approx(1.0, abs=1e-9)
        assert fit["monotone"]


def test_first_order_assumption_is_rejected_when_order_is_two():
    """A ``1/(R/dx)`` basis is the wrong basis for a second-order ladder."""
    rdx = [12.0, 16.0, 24.0, 32.0, 40.0]
    q = [1.0 - 3.0 * (1.0 / v) ** 2 for v in rdx]
    fit = R7.observed_order(rdx, q)
    assert fit["p_observed"] == pytest.approx(2.0, abs=5e-3)
    # the same data read through the assumed basis mis-places the limit, which is
    # exactly the round-4 exposure this round removes
    q_inf_assumed, _, rms_assumed = R7._linear_fit([1.0 / v for v in rdx], q, 1.0)
    assert abs(q_inf_assumed - 1.0) > 5e-3
    assert rms_assumed > 1e-4


def test_asymptotic_bias_is_detected_and_fails_validation():
    """``Q_inf = 0.96`` with a tight ladder must not validate against 1."""
    rdx = [16.0, 24.0, 32.0, 40.0]
    q = [0.96 - 0.8 * (1.0 / v) ** 1.5 for v in rdx]
    fit = R7.observed_order(rdx, q)
    assert fit["q_inf"] == pytest.approx(0.96, abs=1e-9)
    g = R7.gci(q[-2], q[-1], fit["p_observed"], refinement=40.0 / 32.0)
    v = R7.validation(q[-1], g["gci_rel"], 1e-3)
    assert v["pass"] is False
    assert v["validation_ratio"] > 1.0


def test_gci_matches_the_roache_formula():
    p, fs, r = 1.5, 1.25, 1.25
    g = R7.gci(0.98, 0.984, p, refinement=r, fs=fs)
    eps = abs((0.98 - 0.984) / 0.984)
    assert g["gci_rel"] == pytest.approx(fs * eps / (r ** p - 1.0), rel=1e-12)


def test_validation_boundary_is_inclusive_and_reported():
    v = R7.validation(1.01, 0.02, 0.0)
    assert v["pass"] and v["validation_ratio"] == pytest.approx(0.5)
    v = R7.validation(1.05, 0.02, 0.0)
    assert not v["pass"] and v["validation_ratio"] > 1.0


def test_order_not_established_with_two_grids():
    fit = R7.observed_order([24.0, 32.0], [0.95, 0.96])
    assert fit["status"] == "order_not_established"


# --------------------------------------------------------------------------- #
# integration of the analysis layer on a synthetic payload (no SPH, no contour)
# --------------------------------------------------------------------------- #
def _synthetic_payload(p_true=1.5, bias=0.0, control_bias=2e-4):
    """Rows in the schema ``transfer_ladder`` produces, with a known answer.

    ``bias`` shifts the asymptotic limit: ``Q_inf = 1 + bias``.
    """
    radii = [16.0, 24.0, 32.0, 40.0]
    rows, energy = [], []
    for mode in (2,):
        for rdx in radii:
            q = 1.0 + bias - 0.5 * (1.0 / rdx) ** p_true
            g = q * q
            rows.append({"mode": mode, "radius_over_dx": rdx, "eps": 0.0,
                         "transfer_sph": float("nan"),
                         "transfer_control": float("nan"),
                         "transfer_corrected": float("nan"),
                         "differential_transfer": float("nan")})
            for eps in (0.02, 0.05):
                rows.append({"mode": mode, "radius_over_dx": rdx, "eps": eps,
                             "transfer_sph": g,
                             "transfer_control": (1.0 + control_bias) ** 2,
                             "transfer_corrected": g / (1.0 + control_bias) ** 2,
                             "differential_transfer": g})
            energy.append({"mode": mode, "radius_over_dx": rdx,
                           "g_energy_corrected": g,
                           "c2_sph": None,
                           "c2_control": 1.0, "c2_analytic": 1.0})
    return {"rows": rows, "energy": energy, "control": []}


def test_analysis_passes_a_bias_free_ladder():
    payload = _synthetic_payload(bias=0.0, control_bias=2e-4)
    analysis = R7.analyse(payload)
    ok = [r for r in analysis["rows"] if r["status"] == "ok"]
    assert ok, analysis["rows"]
    # the control row divides the measured transfer by (1 + control_bias)**2,
    # so the *corrected* limit sits at 1 / (1 + control_bias) -- the estimator
    # bias is removed from the SPH error but is not zero by construction
    # channel A is divided by the control row, A' (differential) is not
    q_lim = {"A": 1.0 / (1.0 + 2e-4), "A_differential": 1.0,
             "B": 1.0}
    assert {r["channel"] for r in ok} == set(q_lim)
    for r in ok:
        assert r["p_observed"] == pytest.approx(1.5, abs=5e-3)
        assert r["q_inf"] == pytest.approx(q_lim[r["channel"]], abs=1e-6)
    vd = R7.verdict(analysis)
    assert vd["first_order_assumption_held"] is False      # p = 1.5, not 1
    assert vd["gate_pass_rows_at_24"] == vd["gate_rows"] > 0
    assert "asymptotic" in vd["statement"]


def test_analysis_rejects_a_biased_ladder():
    """A 1.5 % asymptotic bias must show up in ``Q_inf`` and fail validation."""
    payload = _synthetic_payload(bias=-0.015, control_bias=2e-4)
    analysis = R7.analyse(payload)
    ok = [r for r in analysis["rows"] if r["status"] == "ok"]
    assert ok
    q_lim = {"A": 0.985 / (1.0 + 2e-4), "A_differential": 0.985, "B": 0.985}
    for r in ok:
        assert r["q_inf"] == pytest.approx(q_lim[r["channel"]], abs=1e-6)
        # every row must *see* the bias, whether it also inherits the control
        # factor (A) or not (A', B)
        assert r["q_inf_error_pct"] < -1.0
    vd = R7.verdict(analysis)
    assert vd["validation_pass_rows"] < vd["validation_rows"]
    assert "NOT fully asymptotic" in vd["statement"]


def test_control_bias_enters_the_validation_uncertainty():
    """``U_input`` must be taken from the control row, not silently dropped."""
    payload = _synthetic_payload(bias=0.0, control_bias=5e-3)
    u = R7.channel_input_uncertainty(payload["rows"], payload["energy"], mode=2,
                                     eps=0.02, channel="A", finest=40.0)
    assert u == pytest.approx(5e-3, rel=1e-6)
    near_zero = R7.channel_input_uncertainty(payload["rows"], payload["energy"],
                                             mode=2, eps=0.02, channel="A",
                                             finest=24.0)
    assert near_zero == pytest.approx(5e-3, rel=1e-6)
    missing = R7.channel_input_uncertainty(payload["rows"], payload["energy"],
                                           mode=9, eps=0.02, channel="A",
                                           finest=40.0)
    assert math.isnan(missing)


def test_channel_history_ignores_non_positive_transfer():
    rows = [{"mode": 2, "radius_over_dx": 24.0, "eps": 0.02,
             "transfer_corrected": -0.5, "differential_transfer": 0.9},
            {"mode": 2, "radius_over_dx": 32.0, "eps": 0.02,
             "transfer_corrected": 0.95, "differential_transfer": 0.95}]
    assert [p[0] for p in R7.channel_history(rows, [], [], mode=2, eps=0.02,
                                             channel="A")] == [32.0]
    # the differential readout at 24 is still positive, so it is kept
    assert len(R7.channel_history(rows, [], [], mode=2, eps=0.02,
                                  channel="A_differential")) == 2