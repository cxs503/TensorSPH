"""Regression tests for the Laplace *plateau* benchmark.

The end-to-end Young-Laplace check that this repository shipped compared the
core pressure with a shell *inside* the liquid (0.80-0.95 R).  Both sit on the
same equilibrium plateau, so that difference is ~0 for any correct
implementation and the reported "-88 %" was an artefact of the definition.
These tests pin three things:

1. the definition itself (the jump is plateau minus *ambient*);
2. the two static bounds that the pressure path must be compared against;
3. the honest verdict: the plateau misses ``sigma/R`` by far more than 15 % and
   does **not** converge with resolution, so the evidence grade stays WITHHELD.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_plateau"


def _bench():
    spec = importlib.util.spec_from_file_location("laplace_plateau_run",
                                                 BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench()


@pytest.fixture(scope="module")
def report():
    return json.loads((BENCH / "validation_report.json").read_text())


# --------------------------------------------------------------------------- #
# committed-artifact consistency (no compute)
# --------------------------------------------------------------------------- #
def test_definition_is_the_cross_interface_jump(report):
    """The scored quantity must be plateau - ambient, not core - inner shell."""
    assert report["reference"]["definition"].startswith("p_plateau")
    assert "ambient" in report["reference"]["definition"]
    lap = report["ladder"][0]["laplace"]
    recomputed = (lap["p_plateau_Pa"] - lap["p_ambient_Pa"]
                  - report["reference"]["delta_p_Pa"]) \
        / report["reference"]["delta_p_Pa"] * 100.0
    assert recomputed == pytest.approx(lap["rel_error_pct"], abs=1e-6)


def test_ambient_side_carries_no_pressure(report):
    """Outside the drop the EOS reference pressure is zero by construction."""
    for row in report["ladder"] + report["variants"]:
        assert abs(row["laplace"]["p_ambient_Pa"]) < 10.0


def test_verdict_stays_honest(report):
    v = report["verdict"]
    assert v["evidence_grade"] == "WITHHELD"
    assert v["within_15pct"] is False
    assert v["worst_rel_error_pct"] > 15.0
    # the failure is systematic: every resolution misses by a similar margin
    errors = [row["laplace"]["rel_error_pct"] for row in report["ladder"]]
    assert max(errors) - min(errors) < 10.0
    # ... while the global force identity is inside the band
    assert v["force_path_verified"] is True


def test_plateau_does_not_converge_with_resolution(report):
    ladder = sorted(report["ladder"], key=lambda r: -r["dx_m"])
    errors = [abs(r["laplace"]["rel_error_pct"]) for r in ladder]
    # refining dx must not look like convergence: the coarsest run is already
    # as good as the finest one
    assert errors[-1] > 0.5 * errors[0]


def test_kernel_bias_exceeds_the_density_change_the_law_needs(report):
    for row in report["static"]:
        assert row["bias_over_required_density_change"] > 5.0
        # the spurious pressure the bias alone produces is several sigma/R
        assert row["spurious_p_core_at_t0_Pa"] > 3.0 * report["reference"]["delta_p_Pa"]
        assert row["required_drho_over_rho0"] < 1e-3


def test_time_average_is_required(report):
    """A single frame is not a measurement: the plateau swings by >> 15 %."""
    hist = max(report["ladder"], key=lambda r: r["n_particles"])["history"]
    values = [h["p_plateau_Pa"] for h in hist]
    assert max(values) - min(values) > 100.0


def test_initial_state_does_not_decide_the_plateau(report):
    """Removing the t=0 kernel bias changes the transient, not the plateau."""
    biased, unbiased = report["variants"]
    diff = abs(biased["laplace"]["p_plateau_Pa"] - unbiased["laplace"]["p_plateau_Pa"])
    assert diff < 0.1 * report["reference"]["delta_p_Pa"]
    # but it does change the initial pressure by an order of magnitude
    assert biased["p_core_at_t0_Pa"] > 3.0 * report["reference"]["delta_p_Pa"]


# --------------------------------------------------------------------------- #
# live recomputation of the cheap parts
# --------------------------------------------------------------------------- #
def test_static_diagnostics_reproduce_committed_values(bench, report):
    live = bench.static_diagnostics(0.002)
    committed = next(r for r in report["static"] if r["dx_m"] == 0.002)
    assert live["kernel_sum_bias_rho_over_rho0"] == pytest.approx(
        committed["kernel_sum_bias_rho_over_rho0"], rel=1e-9)
    assert live["plateau_bounds"]["mls"]["cut_force_ratio_over_theory"] == \
        pytest.approx(committed["plateau_bounds"]["mls"]["cut_force_ratio_over_theory"],
                      rel=1e-9)
    assert live["required_drho_over_rho0"] == pytest.approx(
        committed["required_drho_over_rho0"], rel=1e-9)


def test_short_relaxation_runs_and_reports_ambient(bench):
    """Smoke: the measurement pipeline runs and sees a zero ambient pressure."""
    rep = bench.relax(dx=0.004, t_end_sigma=0.25)
    assert rep["laplace"]["n_plateau"] > 3
    assert abs(rep["laplace"]["p_ambient_Pa"]) < 20.0
    assert rep["laplace"]["delta_p_Pa"] > 0.0
    # the drop keeps its size (a uniform disc has mean radius 2R/3)
    assert 0.55 < rep["history"][-1]["mean_radius_over_R"] < 0.75
