"""Adaptive time stepping.

``validate_config`` checks the step against the acoustic, viscous, gravity and
diffusion limits *at rest*. A moving flow needs a smaller step: the acoustic
signal speed is ``c0 + |v|``, so a dam-break surge that reaches 20 % of ``c0``
puts the true Courant number 20 % above the nominal one. The dam-break ladder
measured exactly that: nominal 0.300, actual 0.347.

These tests pin the policy (shrink at once, grow slowly, never exceed the
configured step) and the two invariants a user cares about: the fixed-step path
is untouched, and an adaptive run honours the limit it claims to honour.
"""
import json
import math

import pytest

from tensorsph.dam_break import DamBreakConfig, run_case
from tensorsph.droplet_impact import (
    Config,
    acoustic_courant,
    acoustic_dt_limit,
    diffusion_dt_limit,
    gravity_dt_limit,
    next_dt,
    stable_dt,
    validate_config,
    viscous_dt_limit,
)

SMALL = dict(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
             drop_center_y=0.25, device="cpu")


def test_the_limit_is_the_smallest_of_every_operator():
    cfg = Config(**SMALL)
    without = stable_dt(cfg, 0.0)
    assert without == pytest.approx(acoustic_dt_limit(cfg, 0.0))
    # A viscous fluid must lower the acoustic step.
    sticky = Config(**SMALL, viscosity=5.0)
    assert stable_dt(sticky, 0.0) == pytest.approx(min(acoustic_dt_limit(sticky, 0.0),
                                                       viscous_dt_limit(sticky)))


def test_the_acoustic_limit_shrinks_with_the_flow_speed():
    cfg = Config(**SMALL)
    at_rest = acoustic_dt_limit(cfg, 0.0)
    moving = acoustic_dt_limit(cfg, cfg.sound_speed)
    # Twice the signal speed, half the step.
    assert moving == pytest.approx(0.5 * at_rest)
    assert acoustic_courant(moving, cfg, cfg.sound_speed) == pytest.approx(cfg.max_courant)


def test_a_fast_flow_shrinks_the_step_immediately():
    cfg = Config(**SMALL, adaptive_dt=True)
    dt = cfg.dt
    # ``dt_safety`` leaves headroom for the acceleration that happens inside the
    # step: the speed is the one at the start of it.
    target = cfg.dt_safety * stable_dt(cfg, 4.0 * cfg.sound_speed)
    assert next_dt(cfg, dt, 4.0 * cfg.sound_speed) == pytest.approx(target)
    # Not a fraction of the way: the very next step is already safe.
    assert next_dt(cfg, dt, 4.0 * cfg.sound_speed) < dt


def test_the_step_grows_by_the_growth_factor_and_no_faster():
    cfg = Config(**SMALL, adaptive_dt=True, dt_growth=1.05)
    tiny = 1.0e-9
    assert next_dt(cfg, tiny, 0.0) == pytest.approx(1.05 * tiny)
    # ... and never above the configured step, which is the validated cap.
    assert next_dt(cfg, cfg.dt, 0.0) == pytest.approx(cfg.dt)


def test_a_growth_factor_below_one_is_rejected():
    cfg = Config(**SMALL, adaptive_dt=True, dt_growth=0.9)
    with pytest.raises(ValueError, match="dt_growth"):
        validate_config(cfg)


def test_diffusion_lowers_the_adaptive_limit_too():
    """The adaptive step must respect the operator that is actually switched on."""
    cfg = Config(**SMALL, density_diffusion=1.0, adaptive_dt=True)
    assert stable_dt(cfg, 0.0) == pytest.approx(min(acoustic_dt_limit(cfg, 0.0),
                                                    diffusion_dt_limit(cfg),
                                                    gravity_dt_limit(cfg)))


def test_the_fixed_path_is_bit_identical_when_the_cap_is_never_reached(tmp_path):
    """The default impact case is limited by its own ``dt``, not by the flow.

    Its configured step is 41 % of what the acoustic criterion allows, so an
    adaptive run never leaves the cap and must reproduce the fixed run exactly.
    If that ever stops being true the test should fail loudly rather than
    quietly change the reference numbers.
    """
    base = dict(SMALL, steps=6, output_every=3, fail_fast=False)
    fixed = Config(**base)
    adaptive = Config(**base, adaptive_dt=True)
    run_fixed = run_with_manifest(fixed, tmp_path / "fixed")
    run_adaptive = run_with_manifest(adaptive, tmp_path / "adaptive")
    assert run_adaptive["derived"]["dt_min"] == pytest.approx(fixed.dt)
    assert run_adaptive["derived"]["dt_max"] == pytest.approx(fixed.dt)
    for a, b in zip(run_fixed["rows"], run_adaptive["rows"]):
        for key in ("kinetic_energy", "rho_max", "max_speed", "momentum_x"):
            assert a[key] == b[key]


def run_with_manifest(cfg, out):
    from tensorsph.droplet_impact import run
    rows = run(cfg, out, plot=False)
    return {"rows": rows, "derived": json.loads((out / "run_manifest.json")
                                                .read_text())["derived"]}


def test_an_adaptive_dam_break_respects_the_courant_limit():
    """The case that motivated the change: 0.347 actual at a nominal 0.300."""
    def build(adaptive):
        cfg = DamBreakConfig(dx=0.04, pool_width=2.0, column_width=0.2,
                             sound_speed=25.0, adaptive_dt=adaptive)
        cfg.column_height = 0.4
        cfg.dt = cfg.dt_from_courant()
        cfg.steps = int(round(0.25 / cfg.dt))
        cfg.output_every = max(1, int(round(0.05 / cfg.dt)))
        return cfg

    fixed = build(False)
    adaptive = build(True)
    fixed_run = run_case(fixed)
    adaptive_run = run_case(adaptive)

    nominal = fixed.dt * fixed.sound_speed / (fixed.h_factor * fixed.dx)
    assert nominal == pytest.approx(fixed.max_courant)
    # Fixed: the speed pushes the true number past the nominal one.
    assert fixed_run.max_acoustic_courant > nominal * 1.05
    # Adaptive: bounded, with only the growth ramp's rounding above the limit.
    assert adaptive_run.max_acoustic_courant <= adaptive.max_courant * 1.02
    assert adaptive_run.dt_min < adaptive_run.dt_max
    # Both reach the requested simulated time, and mass is still exact.
    assert adaptive_run.times[-1] == pytest.approx(fixed_run.times[-1], rel=1e-3)
    assert adaptive_run.mass_drift == 0.0


def test_the_adaptive_step_does_not_change_the_answer_much():
    """Accuracy check: capping the Courant number must not move the trajectory.

    A step chosen by stability is not a step chosen by accuracy, so this is a
    measured bound rather than a convergence claim: the relative L2 difference of
    ``front(t)/a`` over the whole run is 0.05 % at ``dx = 0.04`` and 0.03 % at
    ``dx = 0.02``. The bound below is that number with twenty times the margin.
    """
    def build(adaptive):
        cfg = DamBreakConfig(dx=0.04, pool_width=2.0, column_width=0.2,
                             sound_speed=25.0, adaptive_dt=adaptive)
        cfg.column_height = 0.4
        cfg.dt = cfg.dt_from_courant()
        cfg.steps = int(round(0.8 / cfg.dt))
        cfg.output_every = max(1, int(round(0.01 / cfg.dt)))
        return cfg

    fixed_run = run_case(build(False))
    adaptive_run = run_case(build(True))

    def interp(times, values, query):
        import bisect
        if query <= times[0]:
            return values[0]
        if query >= times[-1]:
            return values[-1]
        k = bisect.bisect_left(times, query)
        weight = (query - times[k - 1]) / (times[k] - times[k - 1])
        return values[k - 1] + weight * (values[k] - values[k - 1])

    limit = min(fixed_run.times[-1], adaptive_run.times[-1])
    numerator = denominator = 0.0
    for k in range(41):
        t = limit * k / 40
        x = interp(fixed_run.times, fixed_run.front, t)
        y = interp(adaptive_run.times, adaptive_run.front, t)
        numerator += (x - y) ** 2
        denominator += y * y
    assert math.sqrt(numerator / denominator) < 0.01


def test_an_accelerating_flow_cannot_exceed_the_limit(tmp_path):
    """The step is chosen from the speed at its start; a violent flow overshoots.

    Without ``dt_safety`` this case measured 0.979 against a 0.9 Courant limit:
    with gravity 20x the default, a particle can accelerate past the limit inside
    a step placed exactly on it. The headroom is what keeps the recorded number
    honest, and this test is the case that found it.
    """
    cfg = Config(dx=0.025, steps=300, output_every=300, device="cpu",
                 fail_fast=False, adaptive_dt=True, dt=1.2e-3,
                 max_courant=0.9, gravity=200.0)
    run_with_manifest(cfg, tmp_path / "stress")
    manifest = json.loads((tmp_path / "stress" / "run_manifest.json").read_text())
    extrema = manifest["sampling"]["extrema"]
    assert extrema["peak_courant"] <= cfg.max_courant
    # The step really was forced below the cap, i.e. adaptation did the work.
    assert manifest["derived"]["dt_min"] < cfg.dt
    # ... and the run still lands on the requested simulated time.
    assert manifest["derived"]["simulated_time"] == pytest.approx(300 * cfg.dt)


def test_the_recorded_time_is_the_sum_of_the_steps(tmp_path):
    """``time`` in the diagnostics is accumulated, not ``step * dt``."""
    cfg = Config(**SMALL, steps=4, output_every=1, fail_fast=False,
                 density_diffusion=1.0, adaptive_dt=True)
    rows = run_with_manifest(cfg, tmp_path / "t")["rows"]
    assert rows[0]["time"] == 0.0
    # Each row reports the step that produced it, so the time difference between
    # two rows is the later row's ``dt``.
    for previous, current in zip(rows, rows[1:]):
        assert current["time"] - previous["time"] == pytest.approx(current["dt"])
    # The run lands on the requested time instead of overshooting it by a step.
    assert rows[-1]["time"] == pytest.approx(4 * cfg.dt, rel=1e-12)
