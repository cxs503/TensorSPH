"""The flume stability verdict: a wave record vs a numerical event.

Backs ``docs/verification_wave_flume_stability.md``: the published D16 grade and
the local D20 record behind it are not a resolution limit -- the runs leave the
weakly-compressible regime.  ``stability_verdict`` is the machine-readable form of
that statement, so the thresholds are pinned by test rather than by prose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "wave_flume"))

from diagnose import stability_verdict  # noqa: E402


def test_a_dambreak_grade_record_is_stable():
    verdict = stability_verdict({"max_speed": 1.5, "max_compression": 0.008}, 30.0)
    assert verdict["stable"] is True
    assert verdict["mach"] == pytest.approx(0.05)


def test_the_published_d20_record_is_not_a_wave():
    """max|v| = 109 m/s against c0 = 30 m/s and 45 % density compression."""
    verdict = stability_verdict({"max_speed": 109.27555084228516,
                                 "max_compression": 0.4494273681640626}, 30.0)
    assert verdict["stable"] is False
    assert verdict["mach_flag"] is True
    assert verdict["compression_flag"] is True
    assert verdict["mach"] == pytest.approx(3.64, abs=0.01)


def test_a_transonic_record_is_flagged_even_if_the_density_looks_fine():
    verdict = stability_verdict({"max_speed": 20.0, "max_compression": 0.005}, 30.0)
    assert verdict["stable"] is False
    assert verdict["mach_flag"] is True
    assert verdict["compression_flag"] is False


def test_the_thresholds_are_inclusive_of_the_documented_limits():
    verdict = stability_verdict({"max_speed": 15.0, "max_compression": 0.01}, 30.0)
    assert verdict["stable"] is True
