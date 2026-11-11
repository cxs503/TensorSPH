"""Static-consistency audit of the 3-D hydrostatic column.

The numbers asserted here are not hand-typed: they are read back from
``benchmarks/hydrostatic_consistency/report.json`` (regenerate with
``PYTHONPATH=src python benchmarks/hydrostatic_consistency/run.py``), and the
first test recomputes the audit and requires the committed artifact to still
match.  What is pinned is therefore a *relationship*, not a magic constant:

* one wall layer is not enough for the kernel the solver ships;
* the ``wall_shell`` lid above the free surface is a spurious boundary;
* the static imbalance of the pressure operator is smaller than, and of the
  opposite sign to, the shipped ``threed_hydrostatic`` profile error -- i.e. it
  cannot be the explanation;
* the free-surface kernel-sum deficit, not the operator, owns the surface-layer
  pressure reading.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.threed import run_hydrostatic  # noqa: E402

BENCH = REPO / "benchmarks" / "hydrostatic_consistency"
REPORT_PATH = BENCH / "report.json"


def _audit():
    spec = importlib.util.spec_from_file_location("hydrostatic_consistency",
                                                  BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    assert REPORT_PATH.exists(), (
        "benchmarks/hydrostatic_consistency/report.json is missing; regenerate it")
    return json.loads(REPORT_PATH.read_text())


@pytest.fixture(scope="module")
def recomputed():
    return _audit().build_report()


def test_committed_artifact_is_current(report, recomputed):
    """Every committed number must be reproducible from the code in this tree."""
    assert report["verdict"] == recomputed["verdict"]
    assert report["criteria"] == recomputed["criteria"]
    for old, new in zip(report["grid"], recomputed["grid"]):
        assert old["h_factor"] == new["h_factor"]
        assert old["wall_layers"] == new["wall_layers"]
        assert old["lid"] == new["lid"]
        assert old["wall_field"] == new["wall_field"]
        for a, b in zip(old["az_over_g_mean"], new["az_over_g_mean"]):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12)


def test_all_criteria_pass(report):
    failed = [k for k, v in report["criteria"].items() if not v]
    assert not failed, f"audit criteria failed: {failed}"
    assert report["verdict"] == "pass"


def test_one_wall_layer_is_an_order_of_magnitude_worse(report):
    grid = {(r["h_factor"], r["wall_layers"], r["lid"], r["wall_field"]): r
            for r in report["grid"]}
    thin = grid[(1.3, 1, True, "adami")]["az_over_g_mean"][0]
    thick = grid[(1.3, 2, True, "adami")]["az_over_g_mean"][0]
    # a single wall layer leaves the wall-adjacent row without support inside
    # 2 h = 2.6 dx; the residual is ~90x the two-layer value at h = 1.3 dx
    assert abs(thin) > 50.0 * abs(thick)
    assert abs(thin := thin) > 0.01  # noqa: F841  (> 1 % g, not a rounding effect)


def test_open_free_surface_cuts_the_surface_residual(report):
    grid = {(r["h_factor"], r["wall_layers"], r["lid"], r["wall_field"]): r
            for r in report["grid"]}
    closed = grid[(1.3, 2, True, "adami")]["az_over_g_mean"][-1]
    opened = grid[(1.3, 2, False, "adami")]["az_over_g_mean"][-1]
    assert abs(opened) < 0.2 * abs(closed)
    # and at h = 1.5 dx the lid even flips the sign of the surface residual
    closed15 = grid[(1.5, 2, True, "adami")]["az_over_g_mean"][-1]
    opened15 = grid[(1.5, 2, False, "adami")]["az_over_g_mean"][-1]
    assert closed15 * opened15 < 0.0


def test_static_imbalance_cannot_explain_the_shipped_error(report):
    measured = report["measured_vs_static"]
    assert measured["shipped_verdict"] == "fail"
    # the operator's own floor is positive (the pressure must be *raised*) ...
    assert measured["static_mean_pct"] > 0.0
    # ... while every shipped layer reads low
    assert measured["measured_mean_pct"] < -10.0
    assert measured["sign_agreement_layers"] == 0
    # and it is an order of magnitude smaller
    assert abs(measured["static_mean_pct"]) < 0.2 * abs(measured["measured_mean_pct"])
    # the artifact itself says the window had not settled
    assert measured["settling_flags"][
        "column_settled_window_drift_below_2pct"] is False


def test_surface_layer_is_clamp_limited_not_operator_limited(report):
    floor = report["readability_floor"]
    at20 = next(r for r in floor["rows"] if r["sound_speed"] == 20.0)
    assert at20["clamp_over_signal"] >= 10.0
    assert floor["rows"][-1]["clamp_over_signal"] > floor["rows"][0]["clamp_over_signal"]


def test_kernel_sum_deficit_lives_only_on_the_free_surface(report):
    closed = report["kernel_sum"]["closed"]["kernel_sum_deficit_pct"]
    opened = report["kernel_sum"]["open"]["kernel_sum_deficit_pct"]
    # with the lid the top row has complete support (the lid fills the kernel)
    assert max(abs(v) for v in closed) < 1e-6
    # without it, only the top row is deficient
    assert opened[-1] > 0.5
    assert max(abs(v) for v in opened[:-1]) < 1e-6


@pytest.mark.xfail(strict=True, reason=(
    "Defect DW-1: wall_shell closes all six faces, so run_hydrostatic puts two "
    "wall layers at z = H and z = H + dx above the free surface of a column "
    "case, inside the 2h = 2.6 dx support. Removing them cuts the surface-layer "
    "static residual 10x. Fix run_hydrostatic to use lid=False and this test "
    "turns red so the ledger is updated."))
def test_free_surface_face_has_no_wall_particles():
    cfg = _audit().column_lattice(h_factor=1.3, wall_layers=2, lid=True)
    assert cfg["lid_particles"] == 0


def test_lid_flag_changes_the_wall_particle_count():
    common = dict(nx=8, ny=8, nz=16, dx=0.01, sound_speed=20.0, steps=1, dt=1e-4)
    torch.manual_seed(0)
    closed = run_hydrostatic(lid=True, **common)
    opened = run_hydrostatic(lid=False, **common)
    assert closed.wall_particles == 1856
    assert opened.wall_particles == 1712
    assert opened.fluid_particles == closed.fluid_particles == 1024
    assert set(opened.profile) & {"rows", "rho0"}