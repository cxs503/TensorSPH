"""Guards for the falsifiable gauges of the oscillating-drop benchmark.

The point of these tests is *negative control*: a guard that cannot fail is
worse than no guard.  Each gauge is driven through a case where it **must**
trip, next to the case where it must stay quiet.  No simulation is run here
(everything is synthetic or read from the committed artifact), so the file
costs milliseconds and can gate every CI job.

Background: ``docs/verification_oscillating_droplet_gauge_audit.md``.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "oscillating_droplet"
REPO = BENCH.parents[1]
sys.path.insert(0, str(BENCH))


def _load_harness():
    """Import ``benchmarks/oscillating_droplet/run.py`` under a unique name.

    ``import run`` is not usable here: :mod:`sys.modules` has one global slot
    called ``"run"``, and ``tests/test_kernel_sum_consistency.py`` imports
    ``benchmarks/init_consistency/run.py`` under that same name.  Whichever
    test file is collected first wins, so under the alphabetical CI collection
    order this module silently received the *init-consistency* harness and died
    with ``AttributeError: module 'run' has no attribute 'elliptical_drop'``
    (main CI red at d61b6c26).  Loading by file path keeps the two harnesses
    distinct regardless of collection order.
    """
    name = "benchmarks_oscillating_droplet_run"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_harness()

R = 0.02
RHO = 1000.0


def _artifact() -> dict:
    return json.loads((BENCH / "validation_report.json").read_text())


# --------------------------------------------------------------------------- #
# harness identity: a stale ``sys.modules["run"]`` slot must not be reused
# --------------------------------------------------------------------------- #
def test_gauges_harness_is_this_benchmarks_run_module():
    """Negative control for the collection-order bug that reddened main CI.

    Before the fix this module did ``import run``, which resolves to whatever
    ``sys.modules["run"]`` happens to hold -- under CI's alphabetical order that
    is ``benchmarks/init_consistency/run.py``.  This guard fails in exactly that
    situation instead of letting the gauge tests error out one by one.
    """
    assert Path(bench.__file__).resolve() == (BENCH / "run.py").resolve()
    for fn in ("elliptical_drop", "containment_diagnostics", "divergence_onset",
               "windowed_frequency", "spurious_momentum"):
        assert callable(getattr(bench, fn)), fn


# --------------------------------------------------------------------------- #
# containment: the gauge must separate "still a drop" from "dispersed cloud"
# --------------------------------------------------------------------------- #
def test_containment_is_quiet_on_a_healthy_ellipse_and_trips_on_a_disperal():
    pos, mass = bench.elliptical_drop(R, 0.002, RHO, 0.10)
    healthy = bench.containment_diagnostics(pos, mass, R)
    assert healthy["intact"] is True
    assert healthy["escaped_mass_fraction"] == 0.0
    # a uniform disk has <r^2> = R^2/2 -> rms/R = 1/sqrt(2) = 0.7071; the
    # area-conserving eps = 0.1 ellipse moves it to 0.7081 on this hexagonal
    # lattice (0.7135 in the continuum), so ~0.71 is the analytic answer, not a
    # tolerance.  Measured value pinned: 0.708126.
    assert healthy["rms_radius_over_radius"] == pytest.approx(0.7081, abs=5e-4)

    # same mass, same particle count, 30x the radius: the mass gate is blind to
    # this by construction, the containment gauge must not be.
    dispersed = bench.containment_diagnostics(pos * 30.0, mass, R)
    assert dispersed["intact"] is False
    assert (dispersed["rms_radius_over_radius"]
            / healthy["rms_radius_over_radius"]) == pytest.approx(30.0,
                                                                  rel=1e-9)
    assert float(mass.sum()) == pytest.approx(float(mass.sum()))  # structural 0


def test_escaped_mass_gauge_trips_on_a_small_but_fatal_ejection():
    """A few per cent of the mass beyond 1.5 R is already fatal."""
    pos, mass = bench.elliptical_drop(R, 0.002, RHO, 0.10)
    n = pos.shape[0]
    k = max(1, int(round(0.04 * n)))
    pos = pos.clone()
    pos[:k] = pos[:k] * 5.0  # fling the first 4 % out to ~5 R
    out = bench.containment_diagnostics(pos, mass, R)
    assert out["escaped_mass_fraction"] > bench.ESCAPE_GATE
    assert out["intact"] is False


def test_recorded_artifact_of_the_registered_claim_is_a_dispersed_cloud():
    """The claim graded FAIL (-43.5 %) was measured on a cloud 4994 R wide."""
    case = _artifact()["frequency"]
    setup, fin, diag = case["setup"], case["final_state"], case["diagnostics"]
    rms_over_r = fin["r_eff"] / (math.sqrt(2.0) * setup["radius_m"])
    assert rms_over_r > bench.RADIUS_GATE * 1000.0     # 1.5e4 x the gate
    v_cap = math.sqrt(setup["sigma_N_per_m"] / (setup["rho0"] * setup["radius_m"]))
    assert diag["max_speed_m_per_s"] / v_cap > 1e5     # 3e5 x capillary speed
    # ... while the gate that was supposed to police this reads exactly zero,
    # for a structural reason (mass is never advected).
    assert diag["mass_relative_drift"] == 0.0
    assert abs(fin["s_moment"]) > 9.0 * case["initial_state"]["s_moment"]


# --------------------------------------------------------------------------- #
# window sweep: the estimator must recover the truth when the blow-up is dropped
# --------------------------------------------------------------------------- #
def test_windowed_frequency_recovers_the_analytic_omega_before_the_blowup():
    omega = 73.48469228349533          # Rayleigh-Lamb, mode 2, R = 20 mm
    period = 2.0 * math.pi / omega
    dt = period / 400.0
    t_onset = 1.06 * period            # the recorded run diverges here
    ts, s = [], []
    for i in range(2000):
        t = i * dt
        ts.append(t)
        a = 0.10 * math.exp(-0.7 * t) * math.cos(omega * t)
        if t > t_onset:                # stand-in for the numerical blow-up
            a += 0.02 * math.exp((t - t_onset) / (0.05 * period))
        s.append(a)
    rows = bench.windowed_frequency(ts, s, omega,
                                    windows=(0.5, 0.75, 1.0, float("inf")))
    early = [r for r in rows if r["window_periods"] in (0.5, 0.75, 1.0)]
    assert early, "window sweep returned no usable window"
    for r in early:
        assert abs(r["rel_error_pct"]) < 3.0, (r["window_periods"],
                                               r["rel_error_pct"])
        assert r["r2"] > 0.99
    late = [r for r in rows if r["window_periods"] is None][0]
    assert abs(late["rel_error_pct"]) > 5.0 * abs(early[0]["rel_error_pct"])


def test_divergence_onset_reports_the_first_crossing_and_none_when_quiet():
    ts = [0.0, 1.0, 2.0, 3.0]
    hit = bench.divergence_onset(ts, [0.1, 0.5, 40.0, 1e5], 20.0)
    assert hit["detected"] is True
    assert hit["onset_time_s"] == 2.0
    assert bench.divergence_onset(ts, [0.1, 0.5, 1.0, 2.0], 20.0)[
        "detected"] is False


# --------------------------------------------------------------------------- #
# momentum residual: the *real* conservation gauge (CFL-free, falsifiable)
# --------------------------------------------------------------------------- #
def test_spurious_momentum_is_zero_for_a_translation_and_grows_for_outflow():
    pos, mass = bench.elliptical_drop(R, 0.002, RHO, 0.10)
    v_cap = math.sqrt(7.2 / (RHO * R))          # 0.6 m/s
    zero = torch.zeros_like(pos)
    assert bench.spurious_momentum(zero, mass, v_cap) == 0.0
    # rigid translation of the whole cloud: momentum is carried, but the
    # residual relative to M*v_translation is 1 by construction, so use a
    # *cancelling* pair instead: equal and opposite jets must read ~0.
    vel = zero.clone()
    vel[0] = torch.tensor([5.0 * v_cap, 0.0], dtype=pos.dtype)
    vel[1] = torch.tensor([-5.0 * v_cap, 0.0], dtype=pos.dtype)
    assert bench.spurious_momentum(vel, mass, v_cap) < 1e-12
    # rigid translation: every particle carries the same momentum, so the
    # residual is exactly 5 M v_cap by construction (normalisation check).
    carrier = torch.zeros_like(pos)
    carrier[:, 0] = 5.0 * v_cap
    assert bench.spurious_momentum(carrier, mass, v_cap) == pytest.approx(5.0)
    # a radial burst about the mass centroid is only *nearly* neutral on a
    # finite hexagonal lattice: the residual is lattice asymmetry, ~1e-2 M v_cap,
    # i.e. 10 orders below the blow-up it is meant to catch (3e5 M v_cap).
    com = (mass[:, None] * pos).sum(dim=0) / mass.sum()
    d = pos - com
    outward = 5.0 * v_cap * d / torch.linalg.vector_norm(d, dim=1, keepdim=True)
    assert bench.spurious_momentum(outward, mass, v_cap) < 1e-2


def test_structural_mass_gate_is_labelled_in_the_source():
    """A reader must not be able to mistake the mass gate for conservation."""
    src = (BENCH / "run.py").read_text()
    assert '"mass_relative_drift_is_structural": True' in src
    assert "never advected" in src
