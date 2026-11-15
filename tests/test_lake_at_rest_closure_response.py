"""Gates for the anchor's tolerance: scheme property or identity?

Artefact: ``tests/data/lake_at_rest/closure_response.json`` (written by
``benchmarks/lake_at_rest/closure_response.py``).  The doc is
``docs/verification_lake_at_rest_closure_response.md``.

Two kinds of check:

* the **shipped artefact** gates (fast, no solver run), which are what the
  verdict is read from;
* **live** re-derivations at ``dx = 0.04`` of the two numbers the finding rests
  on (the tolerant closure is flat in the perturbation amplitude; the shipped
  operator is not), plus the regression for the ``build_state`` dtype fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorsph.hydrostatic_closure import (HydrostaticClosureOptions,
                                           closure_residual, shipped_residual)
from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "lake_at_rest"))
import closure_response as cr  # noqa: E402

RECORD = Path(__file__).resolve().parent / "data" / "lake_at_rest" / "closure_response.json"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_artefact_is_present_and_scoped(record):
    assert record["case"] == "CASE-SPH-LAKE-AT-REST"
    assert record["anchor_tolerance"] == pytest.approx(1.0e-9)
    assert len(record["sweeps"]) == 2


def test_all_gates_pass(record):
    failed = [row for row in record["gates"] if not row["pass"]]
    assert not failed, [(row["gate"], row["measured"]) for row in failed]


def test_tolerant_closure_is_an_identity(record):
    """The row that meets the anchor must be flat: same residual at A = 1 dx."""
    for sweep in record["sweeps"]:
        rows = [row for row in sweep["rows"] if row["closure"] == "incompressible"]
        flat = {row["amplitude_dx"]: row["a_max_over_g"] for row in rows}
        assert flat[0.0] < 1.0e-12
        for amplitude in (0.01, 0.25, 1.00):
            assert flat[amplitude] < 1.0e-12
        # flat to within two orders of magnitude across three decades of A
        assert flat[1.00] < 100.0 * max(flat[0.0], 1e-18)


def test_shipped_operator_responds_and_is_linear(record):
    rows = [row for row in record["sweeps"][0]["rows"]
            if row["closure"] == "shipped" and not row.get("unperturbed")]
    by_amplitude = {row["amplitude_dx"]: row["a_max_over_g"] for row in rows}
    assert by_amplitude[1.00] > 100.0 * by_amplitude[0.01]
    ratio = by_amplitude[0.10] / by_amplitude[0.01]
    assert 5.0 <= ratio <= 25.0, ratio


def test_barotropic_closure_is_responsive(record):
    for sweep in record["sweeps"]:
        rows = [row for row in sweep["rows"]
                if row["closure"] == "barotropic" and not row.get("unperturbed")]
        assert min(row["a_max_over_g"] for row in rows) > 0.0
        assert rows[-1]["a_max_over_g"] > rows[0]["a_max_over_g"]


def test_tolerant_closure_has_no_dynamics(record):
    dyn = record["dynamics"]
    assert dyn["closure"]["amplitude_over_H"] > 0.0
    assert dyn["closure"]["max_surface_move_over_H"] < 1.0e-12
    assert dyn["closure"]["max_v_over_c0"] < 1.0e-12
    # the control arm at the same amplitude does move
    assert dyn["shipped"]["max_surface_move_over_H"] > 1.0e-3


def test_slope_control_is_the_expected_one_percent(record):
    slope = record["slope_control"]
    assert slope["a_max_over_g_base"] < 1.0e-12
    assert slope["a_max_over_g_slope"] == pytest.approx(1.0e-2, rel=0.05)


def test_build_state_honours_declared_dtype_without_global_state():
    """Regression: the lattice must be built in ``LakeConfig.dtype``.

    ``pool_lattice`` reads the global default dtype, so a caller who does not set
    ``torch.set_default_dtype(float64)`` used to get a float32 lattice, which
    fails the mirror-pair map at *every* ``dx`` (~1.5e-8 mismatch).
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        for dx in (0.05, 0.04, 0.025, 0.02):
            state = build_state(LakeConfig(dx=dx))
            assert state["pos"].dtype == torch.float64
            assert state["wall"].dtype == torch.float64
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(previous)


def test_live_identity_and_response_at_dx004():
    """Re-derive the finding live: flat closure, responsive shipped operator."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cfg = LakeConfig(dx=0.04)
        state = build_state(cfg)
        mass = reconstruct(state, cfg)["mass"]
        options = dict(cr.CLOSURES)["incompressible"]
        base = closure_residual(state, cfg, mass, options)
        assert base["a_max_over_g"] < 1.0e-12
        residuals = []
        shipped = []
        for amplitude in (0.01, 0.25, 1.00):
            pert, _mask, amp, _k = cr.perturb_surface(state, cfg, amplitude)
            assert amp > 0.0
            residuals.append(closure_residual(pert, cfg, mass, options)["a_max_over_g"])
            shipped.append(shipped_residual(pert, cfg, mass)["a_max_over_g"])
        assert max(residuals) < 1.0e-12
        assert shipped[0] > 1.0e-3
        assert shipped[-1] > 100.0 * shipped[0]
    finally:
        torch.set_default_dtype(previous)