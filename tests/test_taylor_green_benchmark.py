"""pytest cases for CASE-TAYLOR-GREEN-SPH (TensorSPH).

Two layers, on purpose:

* **reference layer** -- the closed form must satisfy the Navier-Stokes
  equations and its decay rate must be recovered from the field itself.  These
  tests are cheap and they protect the *reference*; if they fail the benchmark is
  comparing against the wrong number.
* **solver layer** -- the SPH run must reproduce that rate inside the 8 %
  tolerance, the Mach window must close above M ~ 0.05, and the committed run
  artefacts must agree with the closed form they claim to be compared against.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorsph.taylor_green import (CASE_KEY, CaseSpec, analytic_decay_rate_from_field,
                                    ns_residual, ref_vs_got, run_case)

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "runs" / "taylor_green" / "m15860"


@pytest.fixture(scope="module", autouse=True)
def _bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


# --------------------------------------------------------------------------- #
# Reference layer
# --------------------------------------------------------------------------- #
@pytest.mark.timeout(120)
def test_reference_satisfies_the_navier_stokes_equations():
    """The closed form must solve momentum + continuity to round-off."""
    spec = CaseSpec()
    for t in (0.0, 0.3, 0.6):
        check = ns_residual(spec, t=t)
        assert check["max_abs_momentum_residual_scaled"] < 1e-10, check
        assert check["max_abs_continuity_residual_scaled"] < 1e-10, check


@pytest.mark.timeout(120)
def test_reference_decay_rate_is_recovered_from_the_field():
    """4 nu k^2 must come out of the analytic field, not from a table."""
    spec = CaseSpec()
    check = analytic_decay_rate_from_field(spec)
    assert check["rel_error_pct"] < 1e-9, check
    # closed form sanity: 4 nu k^2 with nu = 1/(20 pi), k = 2 pi -> 0.8 pi
    assert check["rate_closed_form"] == pytest.approx(0.8 * math.pi, rel=1e-12)


@pytest.mark.timeout(60)
def test_parameter_table_is_self_consistent():
    """The non-dimensional table must be derived, not typed in."""
    spec = CaseSpec()
    table = spec.parameter_table(0.025)
    assert spec.reynolds == pytest.approx(2.0, rel=1e-9)
    assert spec.mach == pytest.approx(0.02, rel=1e-12)
    assert table["kL"] == pytest.approx(2.0 * math.pi, rel=1e-12)
    assert table["particles"] == 1600
    assert table["steps"] == int(math.ceil(spec.t_end / table["dt_s"]))
    assert table["energy_decay_rate_1_s"] == pytest.approx(0.8 * math.pi, rel=1e-12)
    with pytest.raises(ValueError):
        spec.validate(0.03)  # does not divide the periodic box


# --------------------------------------------------------------------------- #
# Solver layer
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_case_run_meets_the_eight_percent_tolerance():
    spec = CaseSpec()
    run = run_case(spec, 0.025)
    assert run["case_key"] == CASE_KEY
    assert run["error_pct"] < spec.tolerance_pct, run["error_pct"]
    rows = run["benchmark"]["rows"]
    assert all(r["pass"] for r in rows), rows
    # energy and amplitude ratios must carry the same decay, not just the fit
    assert run["measured"]["energy_ratio"] == pytest.approx(
        run["measured"]["energy_ratio_analytic"], rel=0.08)
    assert run["measured"]["energy_decay_rate_r2"] > 0.99


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_mach_window_breaks_above_0_05():
    """The error is flat for M <= 0.02 and the window collapses above M ~ 0.05."""
    spec = CaseSpec()
    low = run_case(spec, 0.05, u0=0.02)
    high = run_case(spec, 0.05, u0=0.5)
    assert low["error_pct"] < spec.tolerance_pct, low["error_pct"]
    assert high["error_pct"] > 50.0, high["error_pct"]


@pytest.mark.timeout(120)
def test_committed_run_artifacts_reproduce_the_reference():
    """The committed run must be internally consistent and self-checking."""
    results = json.loads((RUN_DIR / "results.json").read_text())
    evidence = json.loads((RUN_DIR / "evidence.json").read_text())
    spec = CaseSpec()

    # the reference block is the closed form, recomputed here
    assert results["reference"]["energy_decay_rate_1_s"] == pytest.approx(
        spec.energy_decay_rate, rel=1e-12)
    assert results["reference_checks"]["ns_residual"][
        "max_abs_momentum_residual_scaled"] < 1e-10
    assert results["reference_checks"]["analytic_field_decay_rate"][
        "rel_error_pct"] < 1e-9

    # every ref-vs-got row carries a number, a tolerance and a pass flag
    for row in results["benchmark"]["rows"]:
        assert set(row) >= {"quantity", "ref", "got", "rel_error_pct",
                            "tolerance_pct", "pass"}
        assert row["tolerance_pct"] == pytest.approx(8.0)
        assert row["pass"] is (row["rel_error_pct"] < row["tolerance_pct"])
    assert results["tolerance"]["pass"] is True

    # the evidence bundle points at files that exist and are non-empty
    assert evidence["verdict"] == "PASS", evidence["problems"]
    for rel in evidence["run_artifacts"]:
        path = REPO / rel
        assert path.is_file() and path.stat().st_size > 0, rel
    for rel in evidence["external_anchor"]:
        assert (REPO / rel).is_file(), rel

    # the sensitivity studies must be recorded, not asserted in prose only
    assert len(results["convergence"]) >= 3
    assert len(results["reynolds_window"]) >= 3
    assert results["negative_control"]["viscosity_m2_s"] == 0.0


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_package_cli_writes_results(tmp_path):
    """``python -m tensorsph --case taylor_green`` is a real, working entry."""
    out = tmp_path / "tg"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run(
        [sys.executable, "-m", "tensorsph", "--case", "taylor_green",
         "--output", str(out), "--no-plot"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=560)
    assert proc.returncode == 0, proc.stderr[-2000:]
    results = json.loads((out / "results.json").read_text())
    assert results["case"] == CASE_KEY
    assert set(results["benchmark"]["rows"][0]) >= {"ref", "got", "rel_error_pct"}
    assert (out / "run.log").stat().st_size > 0
    assert (out / "environment.json").exists()