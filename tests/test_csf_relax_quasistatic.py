"""Tests for the quasi-static Laplace-jump harness (``relax_quasistatic.py``).

The point of this file is that the *estimand* is testable without running a
droplet: the committed ``run.py`` record scored an interior-to-interior pressure
difference against ``sigma/R``, and that is a property of the sampling bands, not
of the solver.  The tests below pin (a) that logic, (b) the Tait inversion used by
the density route, (c) the block statistics that keep a single noisy snapshot from
passing as a measurement, and (d) a very short end-to-end smoke run of the tool.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "csf_consistent"
JSON_PATH = BENCH / "relax_quasistatic.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load("relax_quasistatic", BENCH / "relax_quasistatic.py")


def _cfg(tool, **kw):
    from tensorsph.droplet_impact import Config
    base = dict(dx=0.002, h_factor=1.3, rho0=tool.RHO0, sound_speed=tool.C0,
                gamma=tool.GAMMA, gravity=0.0, viscosity=tool.VISCOSITY,
                surface_tension_coefficient=tool.SIGMA, csf_scheme="consistent")
    base.update(kw)
    return Config(**base)


# --------------------------------------------------------------------------- #
# (a) the estimand defect is structural, not numerical
# --------------------------------------------------------------------------- #
def test_legacy_estimand_is_zero_for_the_exact_young_laplace_solution(tool):
    """Exact solution p = sigma/R inside, 0 outside -> the legacy bands differ by 0.

    ``run.py`` measures ``mean(p, r<0.5R) - mean(p, 0.8R<r<0.95R)`` and compares it
    with ``sigma/R``.  Both bands are inside the liquid, so on the *exact* solution
    the difference vanishes while the reference is 360 Pa.  The reported -78.6 %
    is therefore an interior non-uniformity, not a model error.
    """
    radius, dp = tool.RADIUS, tool.SIGMA / tool.RADIUS
    r = torch.linspace(0.0, 1.4 * radius, 401)[1:]
    p_exact = torch.where(r < radius, torch.full_like(r, dp), torch.zeros_like(r))
    core = p_exact[r < 0.50 * radius].mean()
    shell = p_exact[(r > tool.LEGACY_SHELL[0] * radius)
                    & (r < tool.LEGACY_SHELL[1] * radius)].mean()
    legacy_dp = float(core - shell)
    assert abs(legacy_dp) < 1e-9, legacy_dp
    # ... yet the committed harness divides that zero by the 360 Pa reference.
    assert abs(legacy_dp - dp) / dp == pytest.approx(1.0)

    # The corrected estimand reads the plateau against vacuum and recovers sigma/R.
    bulk = p_exact[r < tool.BULK_FRACTION * radius].mean()
    assert float(bulk) == pytest.approx(dp, rel=1e-12)


def test_legacy_shell_is_inside_the_liquid(tool):
    """Guard against a future edit that moves the reference band outside the drop."""
    lo, hi = tool.LEGACY_SHELL
    assert 0.0 < lo < hi < 1.0
    assert tool.BULK_FRACTION < lo


# --------------------------------------------------------------------------- #
# (b) density route: Tait inversion, not the dimensionless pressure
# --------------------------------------------------------------------------- #
def test_required_compression_inverts_the_tait_law(tool):
    cfg = _cfg(tool)
    dp = tool.SIGMA / tool.RADIUS
    comp = tool.required_compression(dp, cfg)
    # round trip through the EOS
    assert tool.pressure_from_compression(comp, cfg) == pytest.approx(dp, rel=1e-12)
    # the *density* compression is 0.09 %, the dimensionless pressure is 0.63 %
    assert comp == pytest.approx(0.000897, abs=5e-6)
    dimensionless = dp * cfg.gamma / (cfg.sound_speed ** 2 * cfg.rho0)
    assert dimensionless == pytest.approx(0.0063, abs=1e-4)
    assert dimensionless > 6.0 * comp  # the two are not interchangeable
    # small-compression expansion: rho/rho0 - 1 ~ dp/(c0^2 rho0)
    assert comp == pytest.approx(dp / (cfg.sound_speed ** 2 * cfg.rho0), rel=0.01)


# --------------------------------------------------------------------------- #
# (c) window statistics: a noisy snapshot cannot pass as a measurement
# --------------------------------------------------------------------------- #
def test_block_stats_constant_series_has_no_error(tool):
    s = tool.block_stats([360.0] * 40)
    assert s["mean"] == pytest.approx(360.0)
    assert s["block_stderr"] == pytest.approx(0.0, abs=1e-12)
    assert s["half_drift"] == pytest.approx(0.0, abs=1e-12)


def test_block_stats_detects_a_window_that_has_not_relaxed(tool):
    ramp = [100.0 + 10.0 * i for i in range(40)]
    s = tool.block_stats(ramp)
    assert s["mean"] == pytest.approx(295.0)
    assert s["half_drift"] < -100.0     # first half far below the second half (ramp up)
    falling = tool.block_stats(list(reversed(ramp)))
    assert falling["half_drift"] > 100.0
    noisy = [360.0 + (50.0 if i % 2 else -50.0) for i in range(40)]
    s_noisy = tool.block_stats(noisy)
    assert s_noisy["block_stderr"] > 1.0
    assert s_noisy["n_samples"] == 40


# --------------------------------------------------------------------------- #
# (d) end-to-end smoke run of the tool (short, one core, a few hundred steps)
# --------------------------------------------------------------------------- #
def test_short_run_produces_a_consistent_record(tool):
    rep = tool.run(dx=0.005, t_sigma_end=0.05, t_sigma_avg_from=0.02,
                   sample_every=10, progress=False)
    for key in ("laplace", "legacy_estimator", "density_route", "spurious_current",
                "interior_uniformity", "setup", "reference", "history"):
        assert key in rep, key
    assert rep["setup"]["n_particles"] > 20
    assert rep["history"], "history must not be empty"
    assert rep["laplace"]["p_bulk_Pa"] == pytest.approx(
        sum(h["p_bulk"] for h in rep["history"] if h["in_window"])
        / max(1, sum(1 for h in rep["history"] if h["in_window"])), rel=1e-6)
    rel = rep["laplace"]["rel_error_pct"]
    assert math.isfinite(rel)
    assert rep["laplace"]["within_15pct"] == (abs(rel) < 15.0)
    assert rep["laplace"]["within_3pct"] == (abs(rel) < 3.0)
    d = rep["density_route"]
    assert d["compression_required"] == pytest.approx(
        tool.required_compression(rep["reference"]["delta_p_Pa"], _cfg(tool,
                                                                     dx=0.005)), rel=1e-12)
    # the two estimands are distinct quantities on the same run: the legacy one is
    # the bulk-minus-shell difference and its analytic value is zero, not sigma/R
    lg = rep["legacy_estimator"]
    assert lg["analytic_value_of_the_difference_Pa"] == 0.0
    assert lg["delta_p_measured_Pa"] == pytest.approx(
        lg["p_core_Pa"] - lg["p_shell_Pa"], rel=1e-12)
    assert rep["estimand"]["legacy_analytic_value_Pa"] == 0.0


# --------------------------------------------------------------------------- #
# (e) the committed artefact, when present, must agree with its own flags
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not JSON_PATH.exists(), reason="relax_quasistatic.json not generated")
def test_committed_artifact_is_self_consistent():
    payload = json.loads(JSON_PATH.read_text())
    cases = payload["cases"]
    assert cases, "artifact carries no resolutions"
    for rep in cases:
        la = rep["laplace"]
        assert la["within_15pct"] == (abs(la["rel_error_pct"]) < 15.0)
        assert la["within_3pct"] == (abs(la["rel_error_pct"]) < 3.0)
        assert la["delta_p_measured_Pa"] == pytest.approx(la["p_bulk_Pa"], rel=1e-12)
        # the averaging window is genuinely a window, and it is quasi-static
        lo, hi = rep["setup"]["averaging_window_t_over_t_sigma"]
        assert lo < hi
        assert hi >= 2.0, "a Laplace-jump record below 2 t_sigma is not quasi-static"
        window = [h for h in rep["history"] if h["in_window"]]
        assert len(window) >= 8, "too few samples in the averaging window"
        # drift check: the window mean must be small against the noise-scaled jump
        assert abs(la["p_bulk_half_drift_Pa"]) < 0.5 * abs(la["p_bulk_Pa"])
        d = rep["density_route"]
        # the ratio may sit on either side of 1 (the 3 mm rung is slightly
        # under-compressed); it only has to be a sane finite number
        ratio = d["compression_ratio_measured_over_required"]
        assert -0.5 < ratio < 2.0, ratio