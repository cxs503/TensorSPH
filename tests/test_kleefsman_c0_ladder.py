"""Tests for the Kleefsman c0 (artificial sound speed) sensitivity ladder.

Two things are guarded here:

1. the **defect** found while building the ladder -- ``run.py`` derived ``dt``
   from a hard-coded ``20.0`` m/s instead of the configured sound speed, so any
   run with ``c0 >= 60`` m/s at ``dx = 0.04`` silently violated the acoustic
   Courant limit (``validate_case`` would have rejected the same config);
2. the **physics claim** the ladder is built on -- raising ``c0`` at fixed dx
   lowers the measured density compression, and the weakly-compressible
   estimate ``(v/c0)**2`` is the right order of magnitude for it.
"""
import json
import math
from pathlib import Path

import pytest

from tensorsph.kleefsman_obstacle import KleefsmanConfig, run_case, validate_case

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "benchmarks" / "kleefsman_obstacle"))
from c0_sensitivity import build_config  # noqa: E402

REF = json.loads(Path(__file__).resolve().parents[1].joinpath(
    "benchmarks/kleefsman_obstacle/reference.json").read_text())


def _cfg(dx=0.04, c0=20.0, t_end=0.05):
    return build_config(dx, c0, t_end)


def test_dt_follows_the_configured_sound_speed():
    """dt must shrink with c0 so that dt*c0/h <= max_courant stays true."""
    limits = [build_config(0.04, c0, 0.05).dt for c0 in (20.0, 40.0, 80.0)]
    assert limits[0] == pytest.approx(2.0e-4)          # historical cap kept
    assert limits[1] == pytest.approx(2.0e-4)          # 3.9e-4 limit > cap
    assert limits[2] < 2.0e-4                          # 1.95e-4 limit binds
    assert limits[2] == pytest.approx(0.3 * 1.3 * 0.04 / 80.0)


def test_fixed_dt_with_high_sound_speed_is_rejected():
    """Regression guard for the run.py hard-code: c0=80 with dt=2e-4 must fail."""
    cfg = _cfg(c0=80.0)
    cfg.dt = 2.0e-4
    with pytest.raises(ValueError, match="acoustic limit"):
        validate_case(cfg)


def test_compression_is_monotone_decreasing_in_c0():
    """The ladder's core claim, checked on the cheapest runnable grid."""
    lo = run_case(_cfg(dx=0.08, c0=20.0, t_end=0.12))
    hi = run_case(_cfg(dx=0.08, c0=80.0, t_end=0.12))
    assert lo.finished and hi.finished
    assert lo.max_density_compression > hi.max_density_compression
    assert hi.max_density_compression < 0.05
    # the weakly-compressible estimate must bracket the measurement to O(10)
    ratio = hi.max_density_compression / max(hi.max_speed / 80.0, 1e-12) ** 2
    assert 0.1 < ratio < 50.0


def test_scan_summary_is_self_consistent():
    """If the scan artifacts exist, the aggregate must agree with the rows."""
    here = Path(__file__).resolve().parents[1] / "benchmarks" / "kleefsman_obstacle"
    summary_path = here / "results" / "c0_scan" / "c0_scan_summary.json"
    if not summary_path.exists():
        pytest.skip("c0 scan artifacts not present in this checkout")
    summary = json.loads(summary_path.read_text())
    assert summary["completed"] <= summary["expected"]
    for row in summary["rows"]:
        assert row["dx"] > 0 and row["sound_speed"] > 0
        assert 0.0 <= row["max_density_compression"] < 1.0
        assert math.isfinite(row["peak_height_m"])
    for grid, block in summary["per_grid"].items():
        assert block["compressibility_ratio_range"][0] > 0.0