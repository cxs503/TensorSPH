"""Tests for the onset-attribution diagnostic of the method-selection anchor.

Two kinds of check:

* **synthetic** — a record built here with a known first-crossing time, a known
  exponential growth rate and a known wave-front arrival, so the analysis is
  verified against an answer instead of against itself;
* **archive** — the stored CASE-WAVE-FLUME record must show a piston that follows
  the first-order transfer-function motion exactly (amplitude ``S/2`` and the
  designed ``omega * settle_time`` phase lag), because that is what makes the
  remaining error attributable to the solver rather than to the wavemaker.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "xval_method_selection" / "attribute_onset.py"
ARCHIVE = REPO / "docs" / "reports" / "artifacts_wave_flume" / "wf_D20_unstable.json"


def _load():
    spec = importlib.util.spec_from_file_location("attribute_onset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic(speed_of_time, growth=1.0, settle=0.8):
    """Record whose peak speed is ``speed_of_time(t)`` on a fine time grid."""
    times = np.arange(0.0, 6.001, 0.01)
    speed = np.asarray([speed_of_time(t) for t in times], dtype=float)
    compression = speed / 30.0
    omega = 2.0 * math.pi
    stroke = 0.03
    paddle = 0.5 * stroke * np.sin(omega * (times - settle))
    return {
        "config": {
            "sound_speed": 30.0,
            "depth": 0.2,
            "wave_height": 0.03,
            "wave_period": 1.0,
            "particle_count": 100,
            "dt_requested": 1.0e-4,
            "dt_final": 1.0e-4,
            "flume_length": 3.2,
            "sponge_length": 0.9,
            "settle_time": settle,
        },
        "spec": {
            "omega": omega,
            "k": 5.18,
            "celerity": 1.21,
            "stroke": stroke,
            "wavelength": 1.21,
        },
        "probe_x": [1.0, 1.2],
        "times": times.tolist(),
        "speed_history": speed.tolist(),
        "compression_history": compression.tolist(),
        "paddle_offset": paddle.tolist(),
    }


def test_linear_orbital_velocity_matches_deep_and_shallow_limits():
    module = _load()
    # deep water: coth(kh) -> 1, so u_max -> (H/2) omega
    deep = module._orbital_velocity(0.2, 100.0 * 2.0 * math.pi, 1.0, 2.0)
    assert deep == pytest.approx(0.5 * 0.2 * 2.0, rel=1e-6)
    # shallow water: coth(kh) -> 1/(kh), so u_max -> (H/2) omega/(kh) = c (H/2h)
    k, h, omega = 0.02, 1.0, 2.0
    shallow = module._orbital_velocity(0.2, k, h, omega)
    assert shallow == pytest.approx(0.5 * 0.2 * omega / (k * h), rel=1e-3)


def test_first_crossing_and_growth_rate_recovered():
    module = _load()
    tau = 1.0
    born = 1.5

    def speed(t):
        return 0.0 if t < born else math.exp((t - born) / tau)

    record = _synthetic(speed)
    result = module.analyse(record)
    crossings = result["threshold_crossings_s"]
    # e^(t-born) = level  =>  t = born + tau ln(level)
    assert crossings["2"] == pytest.approx(born + tau * math.log(2.0), abs=0.02)
    assert crossings["10"] == pytest.approx(born + tau * math.log(10.0), abs=0.02)
    growth = result["peak_speed_growth"]
    assert growth["rate_per_s"] == pytest.approx(1.0 / tau, rel=0.02)
    assert growth["r2"] > 0.999


def test_wave_front_arrivals_are_linear_theory():
    module = _load()
    result = module.analyse(_synthetic(lambda t: 0.1))
    arrivals = result["wave_front_arrivals_s"]
    assert arrivals["probes_s"] == pytest.approx([1.0 / 1.21, 1.2 / 1.21])
    assert arrivals["sponge_entrance_s"] == pytest.approx((3.2 - 0.9) / 1.21)
    assert arrivals["downstream_wall_s"] == pytest.approx(3.2 / 1.21)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archive record not present")
def test_archive_piston_follows_the_transfer_function():
    module = _load()
    record = json.loads(ARCHIVE.read_text())
    result = module.analyse(record)
    fidelity = result["wavemaker_fidelity"]
    # S/2*ramp*sin(omega*(t - settle)) is the prescribed motion: after the ramp the
    # fitted amplitude must be S/2 and the phase must be the designed -omega*settle.
    assert fidelity["amplitude_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert fidelity["relative_residual"] < 1e-9
    assert fidelity["phase_matches_designed_lag"] is True
    assert fidelity["offset_bias_m"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archive record not present")
def test_archive_onset_precedes_the_sponge_arrival():
    """The instability starts in the propagation zone, so the sponge is exonerated."""
    module = _load()
    record = json.loads(ARCHIVE.read_text())
    result = module.analyse(record)
    onset = result["onset_vs_front"]["first_supersonic_s"]
    sponge = result["wave_front_arrivals_s"]["sponge_entrance_s"]
    assert onset is not None
    # sub-orbital speeds are already left behind before the front reaches the sponge
    assert result["threshold_crossings_s"]["0.5"] < sponge
    assert result["compression_crossings_s"]["0.01"] < sponge
    assert result["onset_vs_front"]["supersonic_before_sponge_arrival"] is False
    # observed peak speed is three orders of magnitude above linear wave kinematics
    assert result["speed_scale"]["ratio_observed_over_orbital"] > 100.0