"""Tests for ``CASE-HYDROSTATIC`` (still-water pressure profile).

Two layers:

* **Unit / live checks** (fast, no committed artifacts): the analytic reference,
  the gate arithmetic, the free-surface overlap rule, one live ``rhs``
  evaluation of the lattice (interior complete-support column is exact, the
  wall columns are not), and a coarse smoke run of the case.
* **Artifact checks** (read ``artifacts/case_hydrostatic/report.json``): the
  shipped gate value is the one in the run file, it is the pre-declared config,
  and it is inside the 1 % limit.  These assert an *invariant of the shipped
  evidence* (a report that disagrees with its own run files fails here), not
  that a simulation was executed; the simulation is reproducible with
  ``python benchmarks/case_hydrostatic/run.py --config window_8s``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_hydrostatic import (CASE_ID, DEPTH, G, RHO0,  # noqa: E402
                                       TOLERANCE_PCT, analytic_pressure,
                                       evaluate, run_case, run_initial_state,
                                       support_width)

REPORT = REPO / "runs" / "case_hydrostatic" / "report.json"
RUNS = REPORT.parent / "runs"


# --------------------------------------------------------------------------- #
# Analytic reference and gate arithmetic
# --------------------------------------------------------------------------- #

def test_analytic_reference_is_hydrostatic_equilibrium():
    y = torch.tensor([0.0, 0.1, 0.2, 0.3, DEPTH])
    p = analytic_pressure(y)
    assert torch.allclose(p, RHO0 * G * (DEPTH - y), rtol=0, atol=1e-9)
    assert float(p[0]) == pytest.approx(RHO0 * G * DEPTH)
    assert float(p[-1]) == 0.0
    # dp/dy = -rho g: finite-difference check of the closed form itself
    depths = torch.linspace(0.05, 0.35, 7, dtype=torch.float64)
    dp = (analytic_pressure(depths + 1e-4) - analytic_pressure(depths - 1e-4)) / 2e-4
    assert torch.allclose(dp, torch.full_like(dp, -RHO0 * G), rtol=1e-6, atol=1e-6)


def _row(band, y, p, p_ref):
    return {"band": band, "y": y, "y_low": y - 0.05, "y_high": y + 0.05,
            "particles": 10, "pressure": p, "pressure_analytic": p_ref,
            "rel_error_pct": (p - p_ref) / p_ref * 100.0, "free_surface": False}


def test_gate_takes_the_worst_interior_band_only():
    rows = [_row(0, 0.05, 3900.0, RHO0 * G * (DEPTH - 0.05)),
            _row(1, 0.15, 2400.0, RHO0 * G * (DEPTH - 0.15)),
            _row(2, 0.25, 1400.0, RHO0 * G * (DEPTH - 0.25))]
    gate = evaluate(rows)
    assert gate["limit_pct"] == TOLERANCE_PCT
    assert gate["max_band_error_pct"] == pytest.approx(
        max(abs(r["rel_error_pct"]) for r in rows))
    assert gate["worst_band"] == 0
    assert gate["pass"] is False          # the 0.5 %-low interior band is off
    rows[0]["pressure"] = rows[0]["pressure_analytic"] * 1.002   # +0.2 %
    rows[1]["pressure"] = rows[1]["pressure_analytic"] * 1.003
    rows[2]["pressure"] = rows[2]["pressure_analytic"] * 0.999
    gate = evaluate(rows)
    assert gate["pass"] is True and gate["max_band_error_pct"] < TOLERANCE_PCT


def test_free_surface_overlap_rule_excludes_the_shelf_band():
    """A band whose *centre* is deeper than 2h but whose *upper edge* is inside
    the kernel support of the surface must be excluded from the gate."""
    dx = 0.02
    rows = run_initial_state(dx=dx)["profile"]
    support = support_width(dx)
    shelf = [r for r in rows if r["y_high"] > DEPTH - support]
    assert shelf and all(r["free_surface"] and not r["gated"] for r in shelf)
    assert rows[-1]["free_surface"] is True              # topmost band always out
    assert all(not r["free_surface"] for r in rows if r["y_high"] <= DEPTH - support)
    # the exclusion band is 2h = 2.6 dx wide: at 8 bands over a 20-cell column
    # (2.5 cm bands) exactly the top two bands fall inside it
    assert len(shelf) == 2


# --------------------------------------------------------------------------- #
# Live solver checks
# --------------------------------------------------------------------------- #

def test_static_lattice_is_exact_away_from_the_walls():
    """One ``rhs`` evaluation: complete-support interior is < 0.1 %, the wall
    columns are an order of magnitude worse (that is the residual the tank mean
    carries)."""
    record = run_initial_state(dx=0.02)
    rows = [r for r in record["profile"] if r["gated"]]
    assert rows
    # interior: away from floor row, away from the side walls
    support = support_width(0.02)
    interior = rows[1:] if len(rows) > 1 else rows
    assert max(abs(r["rel_error_pct"]) for r in interior) < 5.0

    # complete-support columns only (the lateral truncation is the mechanism)
    from tensorsph.case_hydrostatic import _pool
    from tensorsph.droplet_impact import PairListCache, rhs, wall_lattice
    from tensorsph.vv.hydrostatic import HydrostaticConfig, band_table

    dx, width = 0.02, 1.0
    cfg = HydrostaticConfig(dx=dx, tank_width=width, depth=DEPTH)
    pos, mass, solver = _pool(cfg)
    wall = wall_lattice(solver, pos.device, top=DEPTH)
    h = solver.h_factor * dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=pos.shape[0])
    pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
    _, pressure, _ = rhs(pos, torch.zeros_like(pos), mass, solver, pairs=pairs,
                         wall=wall, dt=0.3 * h / 20.0)
    support = support_width(dx)
    keep = (pos[:, 0] >= support) & (pos[:, 0] <= width - support)
    rows = band_table(pos[keep], pressure[keep], mass[keep], cfg)
    inner = [abs(r["rel_error_pct"]) for r in rows[1:] if not r["free_surface"]]
    assert inner and max(inner) < 0.1

    outer = (pos[:, 0] < 1.5 * dx) | (pos[:, 0] > width - 1.5 * dx)
    mid = pos[:, 1] < DEPTH - support
    ref_all = analytic_pressure(pos[outer & mid, 1])
    wall_err = ((pressure[outer & mid] - ref_all) / ref_all).abs().median()
    assert float(wall_err) * 100.0 > 10.0 * max(inner)


@pytest.mark.slow
def test_negative_control_measures_gravity():
    """``g = 0`` in the solver, reference unchanged -> the case must fail hard."""
    record = run_case(dx=0.05, tank_width=0.5, seconds=1.0, bands=5,
                      sample_every=10, gravity_override=0.0)
    assert abs(record["gate"]["mean_band_error_pct"]) > 80.0
    assert record["gate"]["pass"] is False


@pytest.mark.slow
def test_smoke_case_produces_a_benchmark_table():
    record = run_case(dx=0.05, tank_width=0.5, seconds=1.0, bands=5, sample_every=10)
    assert record["case_id"] == CASE_ID
    assert record["reference"]["formula"] == "p(y) = rho0 * g * (H - y)"
    assert record["config"]["particle_requirement_met"] is False  # smoke size
    assert record["benchmark"], "no gated band -> nothing was measured"
    for row in record["benchmark"]:
        # every benchmark row carries ref and got and the error between them
        assert row["ref_Pa"] > 0 and row["got_Pa"] > 0
        assert row["rel_error_pct"] == pytest.approx(
            (row["got_Pa"] - row["ref_Pa"]) / row["ref_Pa"] * 100.0)
    assert record["gate"]["max_band_error_pct"] == pytest.approx(
        max(abs(r["rel_error_pct"]) for r in record["benchmark"]))


# --------------------------------------------------------------------------- #
# Shipped evidence
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def report():
    assert REPORT.exists(), (
        "runs/case_hydrostatic/report.json is missing; reproduce with "
        "`python benchmarks/case_hydrostatic/run.py --config all --static` "
        "then `--aggregate --figure`")
    return json.loads(REPORT.read_text())


def test_shipped_gate_is_the_predeclared_config_and_within_limit(report):
    verdict = report["verdict"]
    assert verdict["case_id"] == CASE_ID
    assert verdict["gate_limit_pct"] == TOLERANCE_PCT
    assert verdict["gate_config"] == "window_8s"
    run = json.loads((RUNS / f"{verdict['gate_config']}.json").read_text())
    assert verdict["gate_value_pct"] == run["gate"]["max_band_error_pct"]
    assert verdict["gate_pass"] is bool(verdict["gate_value_pct"] <= TOLERANCE_PCT)
    assert verdict["gate_pass"] is True


def test_every_gated_benchmark_row_has_ref_and_got(report):
    run = json.loads((RUNS / f"{report['verdict']['gate_config']}.json").read_text())
    assert len(run["benchmark"]) >= 4
    for row in run["benchmark"]:
        assert row["ref_Pa"] > 0
        assert row["within_tolerance"] is True
    assert run["config"]["fluid_particles"] >= run["config"]["min_particles"]


def test_report_quotes_the_run_files(report):
    for run in report["runs"]:
        stored = json.loads((RUNS / f"{run['config_name']}.json").read_text())
        assert run == stored
    for static in report["static_runs"]:
        stored = json.loads((RUNS / f"static_{static['config_name']}.json").read_text())
        assert static == stored


def test_repro_command_is_recorded(report):
    repro = report["repro"]
    assert "python benchmarks/case_hydrostatic/run.py --config window_8s" == repro["gate"]
    assert "runs/case_hydrostatic" in repro["one_shot"] + repro["grid"]

@pytest.mark.slow
def test_package_entry_dispatches_to_the_case(tmp_path):
    """``python -m tensorsph --case hydrostatic`` is a real entry, not a stub."""
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "1"}
    out = tmp_path / "record.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tensorsph", "--case", "hydrostatic", "--quick",
         "--json-out", str(out)],
        capture_output=True, text=True, timeout=600, env=env, cwd=str(REPO))
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    record = json.loads(out.read_text())
    assert record["case_id"] == CASE_ID
    assert record["benchmark"], "the CLI record carries no ref-vs-got rows"
    assert "max interior-band error" in proc.stdout
