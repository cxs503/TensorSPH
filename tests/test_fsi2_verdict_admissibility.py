"""Admissibility guards for the FSI2 periodic-window verdict.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why these exist
---------------
The gate in ``verdict_body_force_transient.py`` answers *one* question: does the
analysis window hold a stationary cycle, and if so how far is the cycle from the
Turek-Hron reference.  It used to answer it unconditionally.  Two long Aitken
artefacts of the coupled run then made the defect visible:

* ``dx = 0.05`` (10000 steps, 26 s physical) has a stationary envelope -- the
  field has *saturated* by diverging: ``max|v| = 12.0 m/s`` against a declared
  artificial sound speed ``c0 = 5 m/s``, and the beam tip reaches 1.24 m on a
  0.35 m beam.  The old gate scored it ``FAIL`` with ``+109547154 %`` on
  ``lift_amplitude``, i.e. it reported a blow-up as an accuracy deficit of the
  force operator.
* ``dx = 0.02`` (6000 steps, 6.24 s) decays instead: ``Cd`` mean 0.055 against
  3.2.  Also scored off a "period" of 0.20 s, which is 8-19x below the VIV band.

So a stationary envelope is necessary but not sufficient.  These tests pin the
sufficiency part: the verdict may only be ``PASS``/``FAIL`` on an artefact that
is a solution of the FSI2 problem at all, and the screen must say which checks
it actually ran.

The check thresholds are *not* fitted to the artefacts: ``c0`` is read from the
payload, the beam length from the benchmark geometry (3.5 D) or the payload, and
the VIV band is the existing ``VIV_STROUHAL_BAND``.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

from verdict_body_force_transient import (  # noqa: E402
    BLOWUP_MACH_LIMIT,
    TIP_EXCURSION_LIMIT,
    build_verdict,
    physical_admissibility,
)

BENCH_PERIOD = 2.4          # s; inside D/(St U) = [1.67, 3.85] for U=0.2, D=0.1
BENCH_C0 = 5.0


def _artifact(period=BENCH_PERIOD, t_end=12.0, n=3000, amp_y=0.0082,
              cd_mean=3.2, cd_amp=0.96, cl_amp=0.22, max_speed=0.4,
              sound_speed=BENCH_C0, beam_length=None, tip_x_scale=0.34):
    """A clean periodic artefact in the schema the driver writes.

    Defaults are a *good* run: on the benchmark amplitudes, one period inside
    the VIV band, particle speeds well under the artificial sound speed.  Each
    test breaks exactly one thing.
    """
    dt = t_end / n
    t = dt * np.arange(n)
    scale = 0.5 * 1000.0 * 0.2 ** 2 * 0.1        # 0.5 rho U^2 D = 2.0
    phase = 2.0 * math.pi * t / period
    tip = amp_y * np.sin(phase)
    body_drag_fit = scale * (cd_mean + cd_amp * np.sin(phase + 0.4))
    body_lift_fit = scale * cl_amp * np.sin(phase + 1.1)
    hist = {
        "t": t.tolist(),
        "tip_x": (tip_x_scale * tip).tolist(),
        "tip_y": tip.tolist(),
        "body_drag": body_drag_fit.tolist(),
        "body_lift": body_lift_fit.tolist(),
        "body_drag_fit": body_drag_fit.tolist(),
        "body_lift_fit": body_lift_fit.tolist(),
        "drag_viscous": np.zeros(n).tolist(),
        "max_speed": np.full(n, float(max_speed)).tolist(),
        "coupling_converged": [True] * n,
    }
    case = {"rho0": 1000.0, "inlet_u_max": 0.3, "cylinder_d": 0.1}
    if sound_speed is not None:
        case["sound_speed"] = sound_speed
    if beam_length is not None:
        case["beam_length"] = beam_length
    return {"history": hist, "dt": dt, "dx": 0.05, "n_steps": n,
            "aborted": False, "iterations": [3] * n,
            "interface_solver": "aitken", "case": case}


# --------------------------------------------------------------------------- #
# 1. the screen exists and is enforced
# --------------------------------------------------------------------------- #
def test_clean_periodic_artifact_still_reaches_pass():
    """Guard against over-blocking: a sound run must not be withheld."""
    v = build_verdict(_artifact(), warmup_frac=0.3)
    assert v["admissible"] is True
    assert v["withheld_reason"] is None
    assert v["gates"]["tip"]["verdict"] == "PASS"
    assert v["gates"]["fit_arm"]["verdict"] == "PASS"
    assert v["gates"]["tip"]["comparable"] is True


def test_blowup_is_withheld_not_scored_as_an_accuracy_fail():
    """The dx=0.05 defect, reduced: saturated envelope + supersonic field."""
    v = build_verdict(_artifact(max_speed=2.5 * BENCH_C0,
                                amp_y=0.96, cd_mean=60.0), warmup_frac=0.3)
    # the statistical gate alone would call this a periodic regime
    assert v["regime_ok"] is True
    assert v["admissibility"]["max_speed_over_c0"] > BLOWUP_MACH_LIMIT
    assert v["admissible"] is False
    assert v["gates"]["tip"]["verdict"] == "WITHHELD"
    assert v["gates"]["fit_arm"]["verdict"] == "WITHHELD"
    # ... and the numbers are still recorded, only marked non-comparable
    assert v["gates"]["tip"]["comparable"] is False
    assert v["gates"]["tip"]["worst_rel_error"] > 10.0
    joined = "; ".join(v["admissibility"]["reasons"])
    assert "blow-up" in joined
    assert "c0" in joined


def test_body_leaving_its_own_length_is_withheld():
    """dx=0.05 defect, part two: the tip is 3.5 beam lengths from the hinge."""
    p = _artifact()
    t = np.asarray(p["history"]["t"], float)
    # 1.2 m excursion on a 0.35 m beam, still perfectly stationary
    p["history"]["tip_x"] = (1.2 * np.sin(2.0 * math.pi * t / BENCH_PERIOD)).tolist()
    del p["case"]["sound_speed"]            # isolate this screen
    v = build_verdict(p, warmup_frac=0.3)
    assert v["admissibility"]["tip_excursion_over_beam"] > TIP_EXCURSION_LIMIT
    assert v["admissible"] is False
    assert v["gates"]["tip"]["verdict"] == "WITHHELD"
    assert any("beam length" in r for r in v["admissibility"]["reasons"])


def test_beam_length_falls_back_to_the_benchmark_geometry():
    """A payload without ``beam_length`` must not silently skip the screen.

    Older artefacts (including the two long runs that exposed this defect) do
    not carry the field.  The benchmark fixes the beam at 3.5 D, so the screen
    can still run -- and the verdict must say where the number came from.
    """
    p = _artifact()
    assert "beam_length" not in p["case"]
    v = build_verdict(p, warmup_frac=0.3)
    assert v["admissibility"]["beam_length"] == pytest.approx(3.5 * 0.1)
    assert "body_containment" in v["admissibility"]["checked"]


def test_period_outside_the_viv_band_is_withheld():
    """A stationary but far-too-fast oscillation is ringing, not shedding."""
    v = build_verdict(_artifact(period=0.15), warmup_frac=0.3)
    assert v["regime_ok"] is True                      # statistically periodic
    assert v["window"]["amplitude_stationary"] is True
    assert v["admissibility"]["period_in_band"] is False
    assert v["admissible"] is False
    assert v["gates"]["tip"]["verdict"] == "WITHHELD"
    assert any("VIV band" in r for r in v["admissibility"]["reasons"])


# --------------------------------------------------------------------------- #
# 2. no check may pass by absence
# --------------------------------------------------------------------------- #
def test_absent_configuration_is_reported_as_skipped_not_passed():
    p = _artifact(sound_speed=None)
    del p["history"]["max_speed"]
    adm = physical_admissibility(p["history"], 0, p["case"], None)
    assert "blowup" in adm["skipped"]
    assert "period_band" in adm["skipped"]
    assert "finite" in adm["checked"]
    # nothing claims to have been verified that was not
    assert "blowup" not in adm["checked"]
    assert adm["max_speed_over_c0"] is None


def test_non_finite_samples_are_withheld():
    p = _artifact()
    p["history"]["body_drag_fit"][-10:] = [float("nan")] * 10
    v = build_verdict(p, warmup_frac=0.3)
    assert v["admissible"] is False
    assert "body_drag_fit" in v["admissibility"]["non_finite_fields"]


def test_screen_can_be_disabled_to_reproduce_a_historical_verdict():
    """The old behaviour stays reachable, explicitly and visibly."""
    v = build_verdict(_artifact(max_speed=2.5 * BENCH_C0),
                      warmup_frac=0.3, enforce_admissibility=False)
    assert v["admissibility"]["admissible"] is False      # still measured
    assert v["admissibility"]["enforced"] is False
    assert v["gates"]["tip"]["verdict"] in ("PASS", "FAIL")


# --------------------------------------------------------------------------- #
# 3. the driver records the beam resolution it actually used
# --------------------------------------------------------------------------- #
def test_driver_discloses_a_clamped_beam_discretisation(tmp_path):
    """``max(15, beam_length/dx)`` silently decoupled beam from fluid before.

    At dx = 0.1 the formula asks for 3 nodes and got 15, i.e. a beam 4x finer
    than the fluid.  That must be visible in the artefact, because such a run
    cannot be part of a dx ladder.
    """
    out = tmp_path / "probe.json"
    cmd = [sys.executable, str(ROOT / "benchmarks" / "xval_fsi2"
                              / "run_body_force_transient.py"),
           "--dx", "0.1", "--steps", "3", "--max-iter", "2",
           "--interface-solver", "aitken", "--threads", "2", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    payload = json.loads(out.read_text())
    assert payload["beam_nodes_clamped"] is True
    case = payload["case"]
    assert case["beam_length"] == pytest.approx(0.35)
    assert case["beam_node_spacing"] == pytest.approx(0.35 / 15)
    assert case["beam_spacing_over_dx"] < 1.0        # beam finer than fluid