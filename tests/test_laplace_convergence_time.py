"""The Laplace resolution ladder must hold *physical time* fixed (m3342).

At HEAD the committed ladder ``benchmarks/laplace_law/convergence/*.json`` was
produced with a fixed step count (380).  Because ``dt ~ h / c0``, every
refinement also shortened the relaxation: the three rows relaxed for
0.556 / 0.445 / 0.333 capillary times and reported -84 / -78 / -115 %, while the
coarse main run at t = 0.889 t_sigma reported +5.6 %.  The ladder was therefore
comparing three different physical problems, and the "within 15 %" headline was
a single-frame reading of an oscillating signal.

These tests pin the repaired methodology (fixed t/t_sigma + tail average) and
keep the old artifact on the record as the confound it is.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_law"
T_SIGMA = math.sqrt(1000.0 * 0.02 ** 3 / 7.2)


def _load():
    spec = importlib.util.spec_from_file_location("laplace_law_run",
                                                  BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(pattern: str):
    paths = sorted(BENCH.glob(pattern))
    assert paths, f"missing committed artifacts: {pattern}"
    return [(p.name, json.loads(p.read_text())) for p in paths]


@pytest.fixture(scope="module")
def ladder():
    return _json("convergence/*.json")


@pytest.fixture(scope="module")
def legacy():
    return _json("convergence/legacy_steps380/*.json")


def test_ladder_holds_physical_time_fixed(ladder):
    """t/t_sigma is the controlled variable, so it must not drift with dx."""
    times = [rep["setup"]["t_over_t_sigma"] for _, rep in ladder]
    assert min(times) >= 0.99
    assert max(times) - min(times) < 0.02 * max(times)
    # steps must follow dx, i.e. get larger on the finer grids
    steps = {rep["setup"]["dx_m"]: rep["setup"]["steps"] for _, rep in ladder}
    ordered = sorted(steps)
    assert [steps[d] for d in ordered] == sorted((steps[d] for d in ordered),
                                                 reverse=True)


def test_ladder_is_time_averaged(ladder):
    for name, rep in ladder:
        tail = rep["laplace"]["tail"]
        assert tail["readout"].startswith("time_average_of_last"), name
        assert tail["n_samples"] >= 2, name
        assert tail["t_from_s"] > 0.5 * rep["setup"]["physical_time_s"], name
        # the final frame is kept for comparison, not as the readout
        assert rep["laplace"]["readout"] == "final_frame"
        assert tail["final_frame_rel_error_pct"] == pytest.approx(
            rep["laplace"]["rel_error_pct"], rel=1e-12)


def test_tail_verdicts_are_reproducible_from_committed_numbers(ladder):
    for name, rep in ladder:
        tail, ref = rep["laplace"]["tail"], rep["reference"]["delta_p_Pa"]
        recomputed = (tail["p_center_Pa"] - tail["p_band_Pa"] - ref) / ref * 100.0
        assert recomputed == pytest.approx(tail["rel_error_pct"], abs=1e-9), name
        assert tail["within_15pct"] is (abs(tail["rel_error_pct"]) < 15.0), name


def test_single_frame_is_not_a_measurement(ladder):
    """The tail average must differ from the final frame by more than the bar."""
    diffs = [abs(rep["laplace"]["tail"]["rel_error_pct"]
                 - rep["laplace"]["rel_error_pct"]) for _, rep in ladder]
    assert max(diffs) > 15.0
    # the oscillation itself is comparable to the 360 Pa signal
    drift = [rep["laplace"]["tail"]["delta_p_std_Pa"] for _, rep in ladder]
    assert max(drift) > 10.0


def test_no_resolution_reaches_the_laplace_law(ladder):
    """Honest verdict: every fixed-time row misses, so the grade stays WITHHELD."""
    for name, rep in ladder:
        assert rep["laplace"]["tail"]["within_15pct"] is False, name
        assert rep["laplace"]["tail"]["within_3pct"] is False, name


def test_legacy_ladder_records_the_time_confound(legacy):
    """The 380-step artifact is kept, pinned as the confound it is."""
    for name, rep in legacy:
        assert rep["setup"]["t_over_t_sigma"] < 0.6, name
        assert rep["laplace"]["rel_error_pct"] < -50.0, name
    times = sorted(rep["setup"]["t_over_t_sigma"] for _, rep in legacy)
    # refining the grid shortened the physical time: the confound in one line
    assert times[0] < 0.7 * times[-1]


def test_mass_calibration_ladder_is_flagged():
    """If the calibrated ladder exists it must carry the bias it removed."""
    paths = sorted(BENCH.glob("convergence_calibrated/*.json"))
    if not paths:
        pytest.skip("calibrated ladder not generated")
    for path in paths:
        rep = json.loads(path.read_text())
        assert rep["setup"]["calibrate_mass"] is True
        # the field is the ratio rho/rho0 = 1 + b, so it lives just above one
        assert 1.0 < rep["setup"]["lattice_bias_rho_over_rho0"] < 1.02
        # same physical time as the uncalibrated ladder
        assert rep["setup"]["t_over_t_sigma"] >= 0.99


def test_steps_follow_the_requested_physical_time():
    module = _load()
    dxs = (0.004, 0.0025, 0.002, 0.0015)
    steps = [module.steps_for_physical_time(dx, 20.0, 0.02, 7.2, 1000.0, 1.0)
             for dx in dxs]
    assert steps == sorted(steps)                    # finer grid, more steps
    for dx, n in zip(dxs, steps):
        # steps * dt reproduces the requested time to within one step
        dt = None
        from tensorsph.droplet_impact import Config, stable_dt
        dt = float(stable_dt(Config(dx=dx, sound_speed=20.0, rho0=1000.0,
                                    gamma=7.0, gravity=0.0, viscosity=0.0,
                                    surface_tension_coefficient=7.2)))
        assert n * dt / T_SIGMA == pytest.approx(1.0, abs=dt / T_SIGMA)