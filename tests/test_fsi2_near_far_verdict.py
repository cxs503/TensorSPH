"""Guard the pre-declared acceptance logic of the near/far-field drag verdict.

These are pure-function tests over synthetic series: they pin that the gates
*can fail* (a gate that always passes is not a gate), that the stationarity test
rejects a drifting record, and that the Strouhal estimate is only reported when
enough cycles are present.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorsph.fsi2_near_far_verdict import (
    NEAR_FAR_TOL, block_stats, near_far_gate, order_gate, stationary_window,
    strouhal_from_lift, verdict,
)


def test_block_stats_recovers_mean_and_is_not_zero_width():
    x = np.full(64, 2.5)
    s = block_stats(x, n_blocks=8)
    assert s["mean"] == pytest.approx(2.5, abs=1e-12)
    assert s["spread"] == pytest.approx(0.0, abs=1e-12)
    noisy = 2.5 + 0.5 * np.sin(np.linspace(0, 20 * math.pi, 400))
    s2 = block_stats(noisy, n_blocks=8)
    assert s2["spread"] > 0.0 and s2["sem"] > 0.0


def test_stationary_window_accepts_flat_and_rejects_drift():
    t = np.linspace(0.0, 10.0, 200)
    flat = stationary_window(t, np.full_like(t, 3.0))
    assert flat["accepted"] and flat["mean"] == pytest.approx(3.0)
    drift = stationary_window(t, 0.2 * t)      # keeps growing: not stationary
    assert not drift["accepted"]


def test_stationary_window_rejects_short_record():
    with pytest.raises(ValueError):
        stationary_window([0.0, 1.0], [1.0, 2.0])


def test_near_far_gate_both_directions():
    ok = near_far_gate(3.0, 3.2)
    assert ok["passed"] and abs(ok["rel_diff"]) < 0.07
    bad = near_far_gate(0.05, 3.0)             # the coupled-run situation
    assert not bad["passed"] and bad["status"] == "fail"
    sign = near_far_gate(3.0, -3.0)            # sign flip is never agreement
    assert not sign["passed"] and sign["same_sign"] is False
    assert near_far_gate(float("nan"), 3.0)["status"] == "not_run"


def test_near_far_gate_threshold_is_the_declared_one():
    # 14 % apart -> pass, 16 % -> fail, on the same symmetric denominator
    a = 1.0
    # b such that (b - a) / ((a + b) / 2) == +tol exactly
    b = a * (1.0 + 0.5 * NEAR_FAR_TOL) / (1.0 - 0.5 * NEAR_FAR_TOL)
    assert near_far_gate(a, b)["rel_diff"] == pytest.approx(NEAR_FAR_TOL)
    assert near_far_gate(a, b)["passed"]                 # boundary is inclusive
    c = a * (1.0 + 1.2 * 0.5 * NEAR_FAR_TOL) / (1.0 - 1.2 * 0.5 * NEAR_FAR_TOL)
    assert not near_far_gate(a, c)["passed"]             # 1.2 tol must fail


def test_order_gate_band():
    assert order_gate(3.0)["status"] == "fluid_capable"
    assert order_gate(0.05)["status"] == "fluid_defective"
    assert order_gate(100.0)["status"] == "fluid_defective"
    assert order_gate(-1.0)["status"] == "not_run"


def test_strouhal_needs_cycles_and_reads_them():
    u, d = 0.2, 0.1
    f = 0.6                                    # St = f d / u = 0.3
    t = np.linspace(0.0, 10.0, 2001)
    short = strouhal_from_lift(t[:200], np.sin(2 * math.pi * f * t[:200])[:200], u, d)
    assert short["status"] == "not_run"
    got = strouhal_from_lift(t, 0.3 * np.sin(2 * math.pi * f * t), u, d)
    assert got["status"] == "diagnostic"
    assert got["n_crossings"] >= 3
    assert got["st"] == pytest.approx(0.3, rel=1e-2)


def _artifact(drifted: bool) -> dict:
    t = np.linspace(0.0, 8.0, 400)
    cd_near = 3.1 + 0.1 * np.sin(2 * math.pi * 0.6 * t)
    cd_far = 3.3 + 0.1 * np.sin(2 * math.pi * 0.6 * t)
    if drifted:
        cd_far = 3.3 + 2.0 * t
    return {"case": {"u_mean": 0.2, "dx": 0.02, "structure": "rigid"},
            "history": {"t": t.tolist(), "cd_total": cd_near.tolist(),
                        "cd_farfield": cd_far.tolist(),
                        "cl_total": (0.2 * np.sin(2 * math.pi * 0.6 * t)).tolist()}}


def test_verdict_judges_when_stationary_and_withholds_when_not():
    good = verdict(_artifact(False))
    assert good["status"] == "judged"
    assert good["near_far_gate"]["passed"]
    assert good["order_gate"]["status"] == "fluid_capable"
    assert good["not_run"]                       # the honest scope is reported
    bad = verdict(_artifact(True))
    assert bad["status"] == "withheld"
    assert bad["reason"]
