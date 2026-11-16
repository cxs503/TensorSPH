"""Verdict machinery for the FSI2 transient body-force run.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

These are invariant tests on the *analysis* layer, not smoke tests: each one
asserts a property the verdict must have for the benchmark claim to be
trustworthy.

1. Known-answer recovery -- a synthetic signal built from exactly the
   Turek-Hron coefficients must come back at the same coefficients, and the
   measured VIV period must be the injected one.  If the analysis cannot
   recover a known answer it cannot be used to judge an unknown one.
2. Partial artefact -- a run checkpointed mid-flight has no ``summary`` block;
   the verdict must still be computable from the history alone (this is the
   defect that left the anchor's Cd/Cl WITHHELD).
3. Honest WITHHELD -- a monotone start-up ramp has no periodic regime, so the
   verdict must refuse to state PASS/FAIL and defer.
4. Gate boundary -- the owner threshold is applied at exactly 15 % (inclusive)
   and the decisive FAIL band starts above it.

The synthetic signals below are *analysis-layer* fixtures, not physical FSI2
runs: the injected ``PERIOD = 0.25 s`` is deliberately outside the physical
VIV Strouhal band for ``(U_mean, D) = (0.2, 0.1)`` (``[1.67, 3.85] s``), so the
``physical_admissibility`` precondition (added in ``m22030``) would downgrade
every one of them to ``WITHHELD``.  These tests therefore run with
``enforce_admissibility=False`` -- they assert the *analysis* recovers the
injected answer and applies the owner gate, not that the fixture is a physical
solution.  The physical screen is exercised separately in
``test_fsi2_verdict_admissibility.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401  (ensures the same import stack as the benchmark)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

from verdict_body_force_transient import build_verdict, dominant_period  # noqa: E402

RHO, U_MEAN, D = 1000.0, 0.2, 0.1
SCALE = 0.5 * RHO * U_MEAN ** 2 * D
PERIOD = 0.25


def _payload(cd_mean=3.2, cd_amp=0.96, cl_amp=0.22, n=4000, dt=1e-3,
             tip=(0.0028, 0.0082), ramp=False):
    t = np.arange(n) * dt
    if ramp:
        cd = cd_mean * (1.0 - np.exp(-t / 0.5))
        cl = np.zeros(n)
        tx = t * 1e-4
        ty = t * 1e-4
    else:
        cd = cd_mean + cd_amp * np.sin(2 * np.pi * t / PERIOD)
        cl = cl_amp * np.sin(2 * np.pi * t / PERIOD - 0.4)
        tx = tip[0] * np.sin(2 * np.pi * t / PERIOD)
        ty = tip[1] * np.sin(2 * np.pi * t / PERIOD)
    drag, lift = cd * SCALE, cl * SCALE
    keys = dict(
        t=t, tip_x=tx, tip_y=ty, body_drag=drag, body_lift=lift,
        cyl_drag=drag * 0.2, cyl_lift=lift * 0.3,
        cyl_drag_fit=drag * 0.2, cyl_lift_fit=lift * 0.3,
        drag_viscous=np.zeros(n), lift_viscous=np.zeros(n),
        max_speed=np.full(n, 0.2), drag=np.zeros(n), lift=np.zeros(n),
        body_drag_fit=drag, body_lift_fit=lift, cyl_coverage=np.ones(n),
        coupling_iterations=np.ones(n), coupling_converged=np.ones(n),
    )
    hist = {k: list(np.asarray(v, dtype=float)) for k, v in keys.items()}
    return dict(history=hist, dt=dt, dx=0.02, n_steps=n, aborted=False,
                case=dict(rho0=RHO, inlet_u_max=1.5 * U_MEAN, cylinder_d=D))


def test_known_answer_recovered_exactly():
    """Injected Turek-Hron coefficients must come back to <0.1 %."""
    v = build_verdict(_payload(), enforce_admissibility=False)
    fit = v["coefficients"]["fit_arm"]
    assert fit["cd_mean"] == pytest.approx(3.2, rel=1e-3)
    assert fit["cd_amplitude"] == pytest.approx(0.96, rel=1e-3)
    assert fit["cl_amplitude"] == pytest.approx(0.22, rel=1e-3)
    assert v["tip"]["tip_x_amplitude"] == pytest.approx(0.0028, rel=1e-3)
    assert v["tip"]["tip_y_amplitude"] == pytest.approx(0.0082, rel=1e-3)
    assert v["gates"]["fit_arm"]["verdict"] == "PASS"
    assert v["gates"]["tip"]["verdict"] == "PASS"


def test_viv_period_measured_not_assumed():
    """The window actually spans the periodic regime: period is recovered."""
    v = build_verdict(_payload(), enforce_admissibility=False)
    p = v["window"]["period_s"]
    assert p == pytest.approx(PERIOD, rel=0.02)
    assert v["window"]["periods_in_window"] >= 1.0
    assert v["regime_ok"] is True


def test_partial_artefact_without_summary_still_judged():
    """A checkpointed run has no summary block; the verdict must survive it."""
    p = _payload(n=1500)
    assert "summary" not in p
    v = build_verdict(p, enforce_admissibility=False)
    assert v["n_steps"] == 1500
    assert v["coefficients"]["fit_arm"]["cd_mean"] == pytest.approx(3.2, rel=1e-3)


def test_startup_ramp_is_withheld_not_failed():
    """No periodic regime -> defer; never claim PASS/FAIL on start-up data."""
    v = build_verdict(_payload(n=400, ramp=True), enforce_admissibility=False)
    assert v["regime_ok"] is False
    assert v["gates"]["fit_arm"]["verdict"] == "WITHHELD"
    assert v["gates"]["tip"]["verdict"] == "WITHHELD"


def test_gate_boundary_is_inclusive_at_15pct():
    """Exactly +15 % passes the owner gate; +16 % does not."""
    ok = build_verdict(_payload(cd_mean=3.2 * 1.15), enforce_admissibility=False)
    assert ok["gates"]["fit_arm"]["rows"][0]["pass_15"] is True
    assert ok["gates"]["fit_arm"]["rows"][0]["pass_3"] is False
    bad = build_verdict(_payload(cd_mean=3.2 * 1.16), enforce_admissibility=False)
    assert bad["gates"]["fit_arm"]["rows"][0]["pass_15"] is False
    assert bad["gates"]["fit_arm"]["verdict"] == "FAIL"


def test_dominant_period_rejects_non_oscillatory_signal():
    t = np.arange(200) * 1e-3
    assert dominant_period(t, 1.0 - np.exp(-t / 0.05)) is None
    assert dominant_period(np.arange(8) * 1e-3, np.zeros(8)) is None
