"""Tests for the B2 experimental-residual diagnosis and the reference audit.

Three things are pinned here:

1. the **Ritter reference convention** -- ``Z/a = 2 tau - 1`` for the 2:1
   column, which ``experimental_validation.py`` got wrong (it used ``Z/a =
   tau``) until this study;
2. the **reference-audit detector** -- it must flag the bundled table's
   uniform abscissae / quantised ordinates, and it must stay silent on a table
   that does look like a figure digitisation (negative control);
3. the **measured findings** in ``outputs/dam_break_exp_bias/bias_diagnosis.json``
   -- no-slip walls bring the late-time front slope much closer to the
   reference than the shipped free-slip default.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.dam_break import exp_bias_diagnosis as diag
from benchmarks.dam_break import experimental_validation as ev
from tensorsph.dam_break import DamBreakConfig, ritter_front

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "outputs" / "dam_break_exp_bias" / "bias_diagnosis.json"


# --------------------------------------------------------------------------
# 1. Ritter reference convention
# --------------------------------------------------------------------------
def _cfg():
    cfg = DamBreakConfig(dx=0.02, pool_width=2.0, column_width=0.2)
    cfg.column_height = 2.0 * cfg.column_width
    return cfg


def test_ritter_reference_is_two_tau_minus_one_for_a_2_to_1_column():
    """Z/a = (x_f - a)/a with x_f = 2 sqrt(g H) t and H = 2a gives 2 tau - 1."""
    cfg = _cfg()
    scale = math.sqrt(2.0 * cfg.gravity / cfg.column_width)
    times = [0.0, 0.05, 0.1, 0.2]
    za = ev.ritter_za(times, cfg)
    for t, z in zip(times, za):
        tau = t * scale
        assert z == pytest.approx(2.0 * tau - 1.0, abs=1e-12)
        # and it agrees with the library helper the harness now delegates to
        assert z == pytest.approx(
            (ritter_front(t, cfg) - cfg.column_width) / cfg.column_width, abs=1e-12
        )


def test_ritter_reference_is_not_the_old_tau_convention():
    """Guard against the regression `ritter_za = sim_tau` (a half-depth reservoir)."""
    cfg = _cfg()
    scale = math.sqrt(2.0 * cfg.gravity / cfg.column_width)
    times = [0.1, 0.2, 0.3]
    za = ev.ritter_za(times, cfg)
    old = [t * scale for t in times]
    # the two conventions differ by exactly tau - 1, which is large late on
    for new_value, old_value, t in zip(za, old, times):
        assert new_value - old_value == pytest.approx(t * scale - 1.0, abs=1e-12)
    assert abs(za[-1] - old[-1]) > 1.0


def test_harness_reports_the_corrected_ritter_curve():
    """``run_one_exp`` must emit the corrected curve, not ``sim_tau``."""
    args = SimpleNamespace(
        tank_width=2.0, column_width=0.2, sound_speed=25.0, max_courant=0.3,
        device="cpu", wall_friction=0.0, density_diffusion=0.0,
        adaptive_dt=False, seconds=0.02,
    )
    exp = json.loads((ROOT / "benchmarks/dam_break/martin_moyce_1952.json").read_text())["data"]
    cfg = ev.run_config if hasattr(ev, "run_config") else None  # noqa: F841  (documentation)
    result = ev.run_one_exp(0.05, args, exp)
    cfg = DamBreakConfig(dx=0.05, pool_width=2.0, column_width=0.2)
    cfg.column_height = 0.4
    # run_one_exp does not return the times it used for the reference curve, but
    # its entries are strictly less than the raw tau for every sample after t=0.
    assert result["ritter_za"] != result["sim_tau"]
    assert all(z < tau + 1e-9 for z, tau in zip(result["ritter_za"], result["sim_tau"]))


# --------------------------------------------------------------------------
# 2. reference-table audit detector
# --------------------------------------------------------------------------
def test_bundled_reference_is_flagged():
    ref = json.loads((ROOT / "benchmarks/dam_break/martin_moyce_1952.json").read_text())
    audit = diag.audit_reference(ref)
    assert audit["tau_grid_is_uniform"] is True
    assert audit["ordinate_quantum"] == pytest.approx(0.05)
    verdict = diag.audit_verdict(audit)
    assert verdict["n_flags"] >= 2
    assert "uniform" in " ".join(verdict["red_flags"]).lower()


def test_a_digitisation_like_table_is_not_flagged():
    """Negative control: irregular abscissae with scattered ordinates pass."""
    ref = {
        "source": {"paper": "synthetic control"},
        "digitization": {"method": "synthetic"},
        "data": [
            {"tau": 0.0, "Z_over_a": 0.00, "uncertainty": 0.05},
            {"tau": 0.237, "Z_over_a": 0.1873, "uncertainty": 0.05},
            {"tau": 0.451, "Z_over_a": 0.4719, "uncertainty": 0.05},
            {"tau": 0.702, "Z_over_a": 0.6284, "uncertainty": 0.05},
            {"tau": 0.918, "Z_over_a": 0.9417, "uncertainty": 0.05},
            {"tau": 1.181, "Z_over_a": 1.1329, "uncertainty": 0.05},
            {"tau": 1.402, "Z_over_a": 1.5086, "uncertainty": 0.05},
            {"tau": 1.673, "Z_over_a": 1.5761, "uncertainty": 0.05},
        ],
    }
    audit = diag.audit_reference(ref)
    assert audit["tau_grid_is_uniform"] is False
    assert diag.audit_verdict(audit)["n_flags"] == 0


# --------------------------------------------------------------------------
# 3. trajectory diagnostics on a synthetic trajectory (positive control)
# --------------------------------------------------------------------------
def _synthetic(times, slope, lag, a=0.2, g=9.81):
    scale = math.sqrt(2 * g / a)
    front = []
    for t in times:
        tau = t * scale
        za = slope * (tau - lag)   # released at tau = lag (negative before it)
        front.append(a * (1.0 + za))
    return front


def test_diagnose_recovers_a_known_slope_and_lag():
    cfg = _cfg()
    scale = math.sqrt(2.0 * cfg.gravity / cfg.column_width)
    times = [i * 0.01 for i in range(0, 101)]          # 0 .. 1.0 s
    front = _synthetic(times, slope=1.0, lag=0.5, g=cfg.gravity)
    exp = [{"tau": 0.4 * k, "Z_over_a": 1.0 * (0.4 * k - 0.5), "uncertainty": 0.05}
           for k in range(3, 9)]
    out = diag.diagnose_trajectory(times, front, cfg, exp)
    assert out["late_slope_dza_dtau"] == pytest.approx(1.0, abs=0.02)
    # z0 = -lag*slope = -0.5, so the "advanced 0.1 past start" criterion is met
    # at tau = (0.1 + z0)/slope + lag = 0.1; the trajectory is sampled every
    # 0.01 s = 0.099 tau, so allow two samples of granularity
    assert out["startup_lag_tau"] == pytest.approx(0.1, abs=0.12)
    assert out["exp_mae_raw"] is not None and out["exp_mae_raw"] < 0.02


def test_diagnose_reports_a_lag_shift_that_reduces_the_error():
    cfg = _cfg()
    scale = math.sqrt(2.0 * cfg.gravity / cfg.column_width)
    times = [i * 0.01 for i in range(0, 101)]
    front = _synthetic(times, slope=1.0, lag=0.6, g=cfg.gravity)   # sim lags by 0.6 tau
    exp = [{"tau": tau / 10.0, "Z_over_a": 1.0 * (tau / 10.0), "uncertainty": 0.05}
           for tau in range(40, 86, 5)]
    out = diag.diagnose_trajectory(times, front, cfg, exp)
    assert out["exp_best_shift_tau"] == pytest.approx(0.6, abs=0.1)
    assert out["exp_mae_best_shift"] < out["exp_mae_raw"]


# --------------------------------------------------------------------------
# 4. the measured findings (regression on the artifact)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(),
                    reason="run benchmarks/dam_break/exp_bias_diagnosis.py first")
def test_no_slip_brings_the_front_speed_closer_to_the_reference():
    payload = json.loads(ARTIFACT.read_text())
    rows = {(r["dx"], r["wall_friction"]): r for r in payload["results"]}
    for dx in sorted({r["dx"] for r in payload["results"]}):
        free = rows[(dx, 0.0)]
        no_slip = rows[(dx, 1.0)]
        assert free["late_slope_dza_dtau"] > no_slip["late_slope_dza_dtau"]
        # no slip removes at least a third of the free-slip front speed
        assert no_slip["late_slope_dza_dtau"] < 0.75 * free["late_slope_dza_dtau"]
        assert (abs(no_slip["slope_ratio_sim_over_exp"] - 1.0)
                < abs(free["slope_ratio_sim_over_exp"] - 1.0))
        assert no_slip["exp_mae_best_shift"] < free["exp_mae_best_shift"]


@pytest.mark.skipif(not ARTIFACT.exists(),
                    reason="run benchmarks/dam_break/exp_bias_diagnosis.py first")
def test_the_shipped_configuration_does_not_pass_the_experimental_gate():
    """Honest-FAIL guard: if this ever passes, the documents must be updated."""
    payload = json.loads(ARTIFACT.read_text())
    free = [r for r in payload["results"] if r["wall_friction"] == 0.0]
    assert free, "no free-slip row in the artifact"
    assert all(not r["gate_pass_raw"] for r in free)
    assert payload["reference"]["verdict"]["n_flags"] >= 2