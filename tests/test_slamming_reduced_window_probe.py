"""Guards for the reduced-window slamming probe (m14940).

The probe exists because the campaign case does not fit a 120 s shell window;
its *measurement semantics* must stay identical to the campaign driver
(``benchmarks/slamming/run.py::metrics``) or a reduced-window reading could be
compared against a criterion it was never graded by.  These tests pin the
semantics on synthetic series, so they run in milliseconds and do not depend on
a simulation.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _probe():
    path = ROOT / "benchmarks" / "slamming" / "reduced_window_probe.py"
    spec = importlib.util.spec_from_file_location("slam_probe_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(**kw):
    mod = _probe()
    return mod, mod.SlammingConfig(beta_deg=20.0, velocity=2.0, dx=2.0e-3,
                                   c_end=0.02, frames=4, **kw)


def test_window_rule_selects_only_the_self_similar_stage():
    mod, cfg = _cfg()
    half = mod.C_WINDOW * cfg.c_end
    series = [
        {"t": 0.0, "c": 0.0, "c_s": 0.0, "force": 0.0, "noise": 0.0,
         "shape_l2": 0.0, "mass_drift": 0.0, "p_max": 0.0, "p_max_wagner": 1.0},
        {"t": 1e-4, "c": 0.4 * half, "c_s": 99.0, "force": 1.0, "noise": 0.0,
         "shape_l2": 0.0, "mass_drift": 0.0, "p_max": 0.0, "p_max_wagner": 1.0},
        {"t": 2e-4, "c": 0.9 * cfg.c_end, "c_s": math.pi ** 2 / math.tan(cfg.beta),
         "force": 2.0, "noise": 0.0, "shape_l2": 0.0, "mass_drift": 0.0,
         "p_max": 0.0, "p_max_wagner": 1.0},
        {"t": 3e-4, "c": cfg.c_end, "c_s": math.pi ** 2 / math.tan(cfg.beta),
         "force": 2.0, "noise": 0.0, "shape_l2": 0.0, "mass_drift": 0.0,
         "p_max": 0.0, "p_max_wagner": 1.0},
    ]
    m = mod.metrics(series, cfg)
    # the t = 0 and the sub-window samples are excluded, so the bogus 99.0
    # C_s cannot reach the maximum
    assert m["n_samples"] == 2
    assert m["c_s_max"] == pytest.approx(math.pi ** 2 / math.tan(cfg.beta))
    assert m["c_s_rel_err_mean"] == pytest.approx(0.0, abs=1e-12)


def test_metrics_reports_the_wagner_reference_and_relative_error():
    mod, cfg = _cfg()
    c = cfg.c_end
    series = [{"t": 5e-4 * (k + 1), "c": c, "c_s": 2.0 * math.pi ** 2
                                           / math.tan(cfg.beta),
               "force": 1.0, "noise": 0.1, "shape_l2": 0.2, "mass_drift": 1e-9,
               "p_max": 3.0, "p_max_wagner": 3.0} for k in range(3)]
    m = mod.metrics(series, cfg)
    assert m["c_s_wagner"] == pytest.approx(math.pi ** 2 / math.tan(cfg.beta))
    # twice the Wagner coefficient -> +100 % on both the mean and the maximum
    assert m["c_s_rel_err_mean"] == pytest.approx(1.0, rel=1e-12)
    assert m["c_s_rel_err_max"] == pytest.approx(1.0, rel=1e-12)
    assert m["noise_rms"] == pytest.approx(0.1, rel=1e-12)
    assert m["shape_l2_mean"] == pytest.approx(0.2, rel=1e-12)


def test_probe_config_reproduces_the_campaign_window_algebra():
    mod, cfg = _cfg()
    # t_end = 2 c_end tan(beta) / (pi V), the campaign's own definition
    assert cfg.resolved_t_end() == pytest.approx(
        2.0 * cfg.c_end * math.tan(cfg.beta) / (math.pi * cfg.velocity))
    # particles across the wetted half width is what the reduced window costs
    assert cfg.c_end / cfg.dx == pytest.approx(10.0)


def test_empty_window_is_reported_not_invented():
    mod, cfg = _cfg()
    assert mod.metrics([], cfg) == {}
