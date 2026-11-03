"""Tests for dam-break validation infrastructure."""
import json
import math
import os

import pytest

from tensorsph.dam_break import DamBreakConfig, run_case


def _load_exp_data():
    """Load Martin & Moyce experimental data."""
    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "dam_break",
                        "martin_moyce_1952.json")
    with open(path) as f:
        return json.load(f)["data"]


def _interpolate(times, values, query):
    """Linear interpolation."""
    if query <= times[0]:
        return values[0]
    if query >= times[-1]:
        return values[-1]
    for i in range(1, len(times)):
        if times[i] >= query:
            t0, t1 = times[i - 1], times[i]
            v0, v1 = values[i - 1], values[i]
            w = 0.0 if t1 == t0 else (query - t0) / (t1 - t0)
            return v0 + w * (v1 - v0)
    return values[-1]


def test_experimental_data_loads():
    """Martin & Moyce data file loads with expected structure."""
    data = _load_exp_data()
    assert len(data) >= 10
    for d in data:
        assert "tau" in d
        assert "Z_over_a" in d
        assert d["tau"] >= 0
        assert d["Z_over_a"] >= 0


def test_experimental_data_monotonic():
    """Experimental front position should be monotonically increasing."""
    data = _load_exp_data()
    for i in range(1, len(data)):
        assert data[i]["tau"] > data[i - 1]["tau"]
        assert data[i]["Z_over_a"] >= data[i - 1]["Z_over_a"]


def test_interpolation_at_endpoints():
    """Interpolation should return exact values at endpoints."""
    times = [0.0, 1.0, 2.0]
    values = [0.0, 1.0, 2.0]
    assert _interpolate(times, values, 0.0) == pytest.approx(0.0)
    assert _interpolate(times, values, 2.0) == pytest.approx(2.0)


def test_interpolation_midpoint():
    """Interpolation at midpoint should be average."""
    times = [0.0, 2.0]
    values = [0.0, 4.0]
    assert _interpolate(times, values, 1.0) == pytest.approx(2.0)


def test_interpolation_extrapolation_clamped():
    """Interpolation outside range should clamp to endpoints."""
    times = [0.0, 1.0]
    values = [0.0, 1.0]
    assert _interpolate(times, values, -1.0) == pytest.approx(0.0)
    assert _interpolate(times, values, 2.0) == pytest.approx(1.0)


def _compute_relative_l2(sph_samples, exp_data, max_tau=None):
    """Compute relative L2 error (same as validate.py)."""
    sph_taus = [s["tau"] for s in sph_samples]
    sph_z = [s["front_over_a"] for s in sph_samples]

    if max_tau is not None:
        exp_filtered = [d for d in exp_data if d["tau"] <= max_tau]
    else:
        exp_filtered = list(exp_data)

    if not exp_filtered or not sph_taus:
        return None, 0

    exp_taus = [d["tau"] for d in exp_filtered]
    exp_z = [d["Z_over_a"] for d in exp_filtered]

    numerator = denominator = 0.0
    n_points = 0
    for tau, z_exp in zip(exp_taus, exp_z):
        z_sph = _interpolate(sph_taus, sph_z, tau)
        numerator += (z_sph - z_exp) ** 2
        denominator += z_exp ** 2
        n_points += 1

    if denominator <= 0 or n_points == 0:
        return None, n_points
    return math.sqrt(numerator / denominator), n_points


def test_relative_l2_perfect_match():
    """L2 error should be zero for identical data."""
    samples = [{"tau": t, "front_over_a": z} for t, z in [(0, 0), (1, 1), (2, 2)]]
    exp = [{"tau": t, "Z_over_a": z} for t, z in [(0, 0), (1, 1), (2, 2)]]
    error, n = _compute_relative_l2(samples, exp)
    assert error == pytest.approx(0.0, abs=1e-10)
    assert n == 3


def test_relative_l2_with_max_tau():
    """L2 error should only consider data up to max_tau."""
    samples = [{"tau": t, "front_over_a": t} for t in [0, 1, 2, 3, 4]]
    exp = [{"tau": t, "Z_over_a": t} for t in [0, 1, 2, 3, 4]]
    error, n = _compute_relative_l2(samples, exp, max_tau=2.0)
    assert n == 3  # tau = 0, 1, 2


def test_dam_break_run_produces_samples():
    """A short dam-break run should produce samples with tau and front_over_a."""
    cfg = DamBreakConfig(dx=0.04, pool_width=1.0, column_width=0.2,
                         sound_speed=25.0, max_courant=0.3, device="cpu",
                         steps=200, output_every=50)
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    run = run_case(cfg)

    tau, normalised = run.dimensionless(cfg)
    assert len(tau) == len(normalised)
    assert len(tau) >= 2
    assert tau[0] == pytest.approx(0.0, abs=1e-6)
    # Initial front is at the column edge (~column_width), so Z/a ≈ 1.0
    assert normalised[0] == pytest.approx(1.0, abs=0.15)
    assert tau[-1] > 0
    assert normalised[-1] > normalised[0], "front must advance"


def test_experimental_comparison_with_actual_run():
    """Compare a short SPH run against experimental data (sanity check)."""
    cfg = DamBreakConfig(dx=0.04, pool_width=1.0, column_width=0.2,
                         sound_speed=25.0, max_courant=0.3, device="cpu",
                         steps=200, output_every=50)
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    run = run_case(cfg)

    tau, normalised = run.dimensionless(cfg)
    samples = [{"tau": t, "front_over_a": z} for t, z in zip(tau, normalised)]
    exp_data = _load_exp_data()

    # Compute L2 error up to wall impact or max experimental tau
    max_tau = 4.24  # max experimental tau
    if run.wall_impact_time is not None:
        max_tau = min(max_tau, run.wall_impact_time * math.sqrt(
            2.0 * cfg.gravity / cfg.column_width))

    error, n = _compute_relative_l2(samples, exp_data, max_tau)
    assert error is not None
    assert n >= 5  # Should have at least 5 comparison points
    # Error will be large at coarse resolution, but should be finite
    assert 0 < error < 2.0
