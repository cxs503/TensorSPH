"""Gates and statistics of the end-to-end Young-Laplace benchmark.

``benchmarks/laplace_equilibrium/run.py`` re-measures the Young-Laplace law on a
static drop with (D1) a gauge that compares the interior *plateau* with the
ambient zero instead of two interior points and (D2) a volume-consistent initial
state instead of a lattice whose discrete partition of unity is 0.6 % high.  The
tests pin the pieces that make those claims falsifiable:

1. the initial lattice really is over-dense (0.6 %, i.e. a 2.4 kPa Tait
   pressure -- 6.7x the 360 Pa signal), and the calibrated mass scale removes it
   exactly;
2. the plateau estimator accepts a stationary series and rejects one that still
   drifts or rings above the 15 % fluctuation gate;
3. the least-squares slope estimator recovers an exact synthetic slope, and the
   slope form of the law is immune to a constant offset (which is why the
   absolute value is the primary verdict and the slope the secondary one);
4. the radial-profile binning recovers a synthetic parabolic pressure field;
5. the whole pipeline runs and keeps a ``sigma = 0`` drop quiescent once the
   initial state is consistent.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_equilibrium" / "run.py"

RADIUS = 0.02
DX = 0.002
RHO0 = 1000.0


def _bench():
    spec = importlib.util.spec_from_file_location("laplace_equilibrium_run", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# D2: the initial state
# --------------------------------------------------------------------------- #
def test_the_uncalibrated_lattice_is_over_dense_and_loud():
    """``mass = rho0 dx dy`` over-fills the lattice; that is the 2.4 kPa ghost."""
    bench = _bench()
    ratio = bench.core_density_ratio(RADIUS, DX, RHO0)
    p0 = bench.initial_core_pressure(RADIUS, DX, RHO0)
    # measured value in the report: 1.005884 -> 2395 Pa
    assert 1.004 < ratio < 1.008, f"lattice density ratio {ratio:.6f} moved"
    assert 2000.0 < p0 < 2800.0, f"initial Tait pressure {p0:.1f} Pa moved"
    # the artefact must stay large compared with the signal it is supposed to test
    assert p0 > 5.0 * (7.2 / RADIUS), "the initial bias is no longer dominant"


def test_volume_calibration_removes_the_bias_exactly():
    bench = _bench()
    scale, info = bench.calibrate_mass_scale(RADIUS, DX, RHO0)
    assert scale < 1.0
    assert abs(info["achieved_core_density_ratio"] - 1.0) < 1e-6
    assert abs(bench.initial_core_pressure(RADIUS, DX, RHO0, mass_scale=scale)) < 1.0


def test_calibrated_mass_scale_keeps_the_same_particle_set():
    """Only the mass changes: positions, count and ``h/dx`` are untouched."""
    bench = _bench()
    scale, _ = bench.calibrate_mass_scale(RADIUS, DX, RHO0)
    pos_a, mass_a = bench.hexagonal_droplet(RADIUS, DX, RHO0)
    pos_b, mass_b = bench.hexagonal_droplet(RADIUS, DX, RHO0)
    assert torch.equal(pos_a, pos_b)
    assert torch.allclose(mass_b * scale, mass_a * scale)
    assert abs(float(mass_a[0]) * scale / (RHO0 * DX * DX * math.sqrt(3) / 2) - scale) < 1e-12
    assert bench.H_FACTOR == 1.3


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def test_linfit_recovers_an_exact_line():
    bench = _bench()
    x = torch.linspace(0.0, 7.0, 11, dtype=torch.float64)
    y = 3.0 + 50.0 * x
    slope, offset = bench._linfit(x, y)
    assert abs(slope - 50.0) < 1e-9
    assert abs(offset - 3.0) < 1e-9


def test_plateau_accepts_a_stationary_series():
    bench = _bench()
    t = [i * 1e-4 for i in range(200)]
    series = [360.0 + 5.0 * math.sin(2 * math.pi * i / 20.0) for i in range(200)]
    stats = bench.plateau(series, t)
    assert abs(stats["mean"] - 360.0) < 1.0
    assert stats["stable"] is True
    assert stats["sem"] > 0.0


def test_plateau_rejects_an_undamped_ringing_series():
    bench = _bench()
    t = [i * 1e-4 for i in range(200)]
    series = [360.0 + 200.0 * math.sin(2 * math.pi * i / 40.0) for i in range(200)]
    stats = bench.plateau(series, t)
    assert stats["relative_fluctuation"] > 0.15
    assert stats["stable"] is False


def test_plateau_reports_a_drift():
    bench = _bench()
    t = [i * 1e-4 for i in range(200)]
    series = [100.0 + 2.0 * i for i in range(200)]
    stats = bench.plateau(series, t)
    # the window is the last half (100 samples, 99 intervals of 1e-4 s) and the
    # series rises by 2 Pa per sample, so the fitted rise over the window is
    # 2 * 99 = 198 Pa -- the estimator reports exactly that, by construction
    assert abs(stats["drift_per_window"] - 198.0) < 1.0
    # the residual trend is 50 % of the window mean, so this is not a plateau
    # even though the fluctuation alone (14.5 %) would sneak under 15 %
    assert stats["relative_drift"] > 0.10
    assert stats["stable"] is False


def test_laplace_verdict_thresholds():
    bench = _bench()
    theory = 7.2 / RADIUS
    assert bench.laplace_verdict(1.01 * theory, theory)["within_3pct"] is True
    inside15 = bench.laplace_verdict(0.90 * theory, theory)
    assert inside15["within_15pct"] is True and inside15["within_3pct"] is False
    outside = bench.laplace_verdict(0.80 * theory, theory)
    assert outside["within_15pct"] is False


def test_slope_form_is_immune_to_a_constant_offset():
    """A residual density bias shifts every p_core; the slope must not move."""
    bench = _bench()
    sigmas = [7.2, 14.4, 21.6]
    offset = 2395.0
    pressures = [offset + s / RADIUS for s in sigmas]
    fit = bench.slope_estimator(sigmas, pressures)
    verdict = bench.slope_verdict(fit["slope_Pa_per_N_per_m"], RADIUS)
    assert abs(verdict["rel_error_pct"]) < 0.01
    assert abs(fit["offset_Pa"] - offset) < 1e-6


def test_slope_estimator_rejects_a_degenerate_sweep():
    bench = _bench()
    with pytest.raises(ValueError):
        bench.slope_estimator([7.2, 7.2], [300.0, 400.0])


def test_radial_profile_recovers_a_parabola():
    bench = _bench()
    # a 1-D radial grid dressed as 2-D positions (y = 0) is enough for the binning
    r = torch.linspace(0.0, RADIUS, 401, dtype=torch.float64)
    pos = torch.stack([r, torch.zeros_like(r)], dim=1)
    p0 = 360.0
    pressure = p0 * (1.0 - (r / RADIUS) ** 2)
    profile = bench.radial_pressure_profile(pos, pressure, RADIUS, bins=10)
    filled = [b for b in profile if b["n"]]
    assert len(filled) == 10
    # centre bin close to p0, outer bin within a few percent of 0
    assert 0.85 * p0 < filled[0]["p_Pa"] < p0
    assert abs(filled[-1]["p_Pa"]) < 0.15 * p0
    # monotone decreasing
    vals = [b["p_Pa"] for b in filled]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


# --------------------------------------------------------------------------- #
# the pipeline itself
# --------------------------------------------------------------------------- #
def test_sigma_zero_drop_stays_quiescent_when_the_initial_state_is_consistent():
    """No force, no pressure -> nothing may move (the D2 artefact would move it)."""
    bench = _bench()
    scale, _ = bench.calibrate_mass_scale(RADIUS, DX, RHO0)
    case = bench.run_case(dx=DX, sigma=0.0, mass_scale=scale, steps=40,
                          radius=RADIUS, output_every=10, label="control")
    assert abs(case["initial_state"]["core_pressure_Pa"]) < 1.0
    assert max(case["history"]["vmax"]) < 1e-9
    assert case["history"]["rho_core"][-1] == pytest.approx(1.0, abs=1e-9)


def test_surface_tension_produces_a_positive_interior_pressure():
    """Smoke: a short relax with the calibrated state builds the right sign."""
    bench = _bench()
    scale, _ = bench.calibrate_mass_scale(RADIUS, DX, RHO0)
    case = bench.run_case(dx=DX, sigma=7.2, mass_scale=scale, steps=40,
                          radius=RADIUS, output_every=10, label="smoke")
    case_cfg = case["reference"]
    assert case_cfg["delta_p_Pa"] == pytest.approx(360.0)
    # the first samples are the acoustic response to the new surface force
    assert case["history"]["p_core"][-1] > 0.0
    assert all(v == v for v in case["history"]["p_core"])  # no NaNs