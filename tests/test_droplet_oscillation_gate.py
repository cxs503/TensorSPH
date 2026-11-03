"""Regression net for the droplet-oscillation dynamic surface-tension gate.

Two layers:

1. **Measurement chain** -- the quadrupole observable and the period readers are
   exercised on *synthetic* signals with a known frequency, so a broken reader
   is caught without running the solver (this is cheap enough for CI).
2. **Recorded result** -- the committed ``results.json`` of the last full run is
   re-read and its verdict is pinned.  The gate currently FAILS (the oscillation
   is not a clean damped sinusoid, see ``docs/verification_droplet_oscillation.md``
   §3 and §7); the test asserts exactly that, so that a future fix cannot pass
   silently *and* a future regression cannot turn the recorded FAIL into a PASS
   without updating the verification note.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "droplet_oscillation"))

import torch  # noqa: E402

from run import (decay_rate, elliptically_deform, hexagonal_drop,  # noqa: E402
                 period_from_crossings, quadrupole_moment, radius_percentile,
                 rayleigh_omega)

RESULT_JSON = ROOT / "benchmarks" / "droplet_oscillation" / "results.json"


# --------------------------------------------------------------------------- #
# reference formula
# --------------------------------------------------------------------------- #
def test_rayleigh_2d_coefficient_is_six_not_eight():
    """2-D mode m=2 has omega^2 = 6 sigma / (rho R^3); 8 is the 3-D l=2 value."""
    sigma, rho, radius = 7.2, 1000.0, 0.02
    omega = rayleigh_omega(sigma, rho, radius, mode=2)
    assert omega == pytest.approx(math.sqrt(6.0 * sigma / (rho * radius ** 3)))
    assert omega != pytest.approx(math.sqrt(8.0 * sigma / (rho * radius ** 3)))


def test_rayleigh_scaling_with_radius():
    """omega ~ R^{-3/2}: doubling R must shrink the frequency by 2**1.5."""
    a = rayleigh_omega(7.2, 1000.0, 0.02)
    b = rayleigh_omega(7.2, 1000.0, 0.04)
    assert a / b == pytest.approx(2.0 ** 1.5, rel=1e-12)


# --------------------------------------------------------------------------- #
# the observable
# --------------------------------------------------------------------------- #
def test_quadrupole_moment_of_a_circle_is_zero():
    pos, mass = hexagonal_drop(0.02, 0.002, 1000.0)
    assert abs(quadrupole_moment(pos, mass)) < 2.0e-3


def test_quadrupole_moment_recovers_twice_epsilon():
    """For r(theta) = R(1 + eps cos 2 theta) the mass-weighted q is 2 eps."""
    pos, mass = hexagonal_drop(0.02, 0.002, 1000.0)
    for eps in (0.02, 0.10):
        q = quadrupole_moment(elliptically_deform(pos, eps), mass)
        assert q == pytest.approx(2.0 * eps, rel=0.10)


def test_affine_deformation_preserves_area():
    """det = 1, so the deformation injects no compressional energy.

    The exact invariant of ``x -> (1+eps) x, y -> y/(1+eps)`` is
    ``<x^2><y^2> = const``: the two moments move in opposite directions by
    ``(1+eps)^2`` and cancel.  A compressional map would break that.
    """
    pos, mass = hexagonal_drop(0.02, 0.002, 1000.0)
    x2 = float((mass * pos[:, 0] ** 2).sum())
    y2 = float((mass * pos[:, 1] ** 2).sum())
    for eps in (0.05, 0.15):
        out = elliptically_deform(pos, eps)
        nx2 = float((mass * out[:, 0] ** 2).sum())
        ny2 = float((mass * out[:, 1] ** 2).sum())
        assert nx2 / x2 == pytest.approx((1.0 + eps) ** 2, rel=1e-12)
        assert ny2 / y2 == pytest.approx(1.0 / (1.0 + eps) ** 2, rel=1e-12)
        assert nx2 * ny2 == pytest.approx(x2 * y2, rel=1e-12)


def test_radius_percentile_tracks_the_drop_radius():
    pos, mass = hexagonal_drop(0.02, 0.002, 1000.0)
    assert radius_percentile(pos, mass, 0.95) == pytest.approx(0.02, abs=0.001)


# --------------------------------------------------------------------------- #
# period readers
# --------------------------------------------------------------------------- #
def _synthetic(period: float, amplitude: float = 0.2, decay: float = 0.0,
               offset: float = 0.0, samples: int = 200, cycles: float = 2.5):
    times = [i * (cycles * period / samples) for i in range(samples + 1)]
    vals = [amplitude * math.exp(-decay * t) * math.cos(2 * math.pi * t / period) + offset
            for t in times]
    return times, vals


def test_period_reader_recovers_a_clean_sinusoid():
    times, vals = _synthetic(0.0855)
    fit = period_from_crossings(times, vals)
    assert fit["period"] == pytest.approx(0.0855, rel=1e-3)


def test_period_reader_survives_damping_and_offset():
    times, vals = _synthetic(0.0855, decay=6.0, offset=0.05)
    fit = period_from_crossings(times, vals)
    assert fit["period"] == pytest.approx(0.0855, rel=2e-2)


def test_period_reader_refuses_a_signal_without_enough_crossings():
    times, vals = _synthetic(0.0855, cycles=0.6)
    fit = period_from_crossings(times, vals)
    assert fit["period"] is None


def test_decay_rate_of_a_clean_exponential_envelope():
    """The envelope reader is a *crude* diagnostic (extrema-pair estimate).

    It is biased high on short records (the tail-level subtraction moves the
    extrema) and converges from above: 13.7 / 6.1 / 4.2 for 2 / 3 / 5 cycles
    against a true 5.0.  It is therefore only used to decide whether the
    oscillation decays at all and to quote a damping ratio to ~25 %; it is not
    used to accept or reject the gate.
    """
    times, vals = _synthetic(0.0855, decay=5.0, cycles=5.0, samples=600)
    lam = decay_rate(times, vals)
    assert lam == pytest.approx(5.0, rel=0.25)


# --------------------------------------------------------------------------- #
# recorded full run
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RESULT_JSON.exists(), reason="no recorded run")
def test_recorded_run_is_self_consistent():
    r = json.loads(RESULT_JSON.read_text())
    assert r["benchmark"] == "droplet_oscillation"
    assert r["setup"]["gravity"] == 0.0
    assert r["setup"]["sigma_N_per_m"] > 0.0
    m = r["measured"]
    assert m["particles"] > 100
    assert m["period_s"] is not None
    # the reference in the file must be the 2-D 6-sigma law, not the 3-D one
    assert r["reference"]["omega_rad_s"] == pytest.approx(
        math.sqrt(6.0 * r["setup"]["sigma_N_per_m"]
                  / (r["setup"]["rho0_kg_per_m3"] * r["setup"]["radius_m"] ** 3)))
    # every acceptance number must be present and traceable, never null-by-omission
    assert m["frequency_error"] is not None
    assert m["error_from_first_crossing"] is not None


@pytest.mark.skipif(not RESULT_JSON.exists(), reason="no recorded run")
def test_recorded_verdict_is_the_one_documented():
    """Pins the honest-FAIL: the zero-crossing period is outside 15 %."""
    r = json.loads(RESULT_JSON.read_text())
    assert r["pass"] is False, "the gate passed; the verification note must be updated"
    assert r["measured"]["frequency_error"] > 0.15
    assert r["threshold"] == 0.15
    # the independent (quarter-period) proxy must be reported too, and it must
    # also miss the threshold -- a single flattering metric is not evidence
    assert r["measured"]["error_from_first_crossing"] > 0.15