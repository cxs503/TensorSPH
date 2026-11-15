"""Gates for the hydrostatic-closure verdict of ``CASE-SPH-LAKE-AT-REST``.

Two layers, deliberately separate:

* the **shipped record** (``tests/data/lake_at_rest/hydrostatic_closure.json``) is
  re-graded here from its raw numbers -- the stored booleans are not trusted, the
  thresholds are re-applied;
* the **coarse re-derivation** reruns the physics at ``dx = 0.05`` on the audited
  lattice and checks that the record still describes this code (a stale record
  would otherwise pass forever).

The anchor tolerance is ``1e-9`` (machine-precision statement, analytic identity);
``TOLERANCE`` is imported from the runner so the gate and the document cannot
drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import importlib.util

_RUNNER = Path(__file__).resolve().parents[1] / "benchmarks" / "lake_at_rest" / \
    "hydrostatic_closure.py"
_spec = importlib.util.spec_from_file_location("lake_hydrostatic_closure_runner", _RUNNER)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
ANCHOR_TOLERANCE = _runner.ANCHOR_TOLERANCE
gates = _runner.gates
from tensorsph.hydrostatic_closure import (HydrostaticClosureOptions,
                                           closure_residual, controls,
                                           level_set)
from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct

DATA = Path(__file__).resolve().parent / "data" / "lake_at_rest" / \
    "hydrostatic_closure.json"
DOC = Path(__file__).resolve().parents[1] / "docs" / \
    "verification_lake_at_rest_hydrostatic_closure.md"


@pytest.fixture(scope="module")
def payload():
    if not DATA.is_file():
        pytest.skip(f"shipped record missing: {DATA}")
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def coarse():
    """The audited lattice at ``dx = 0.05``, rebuilt from source."""
    torch.set_default_dtype(torch.float64)
    cfg = LakeConfig(dx=0.05)
    state = build_state(cfg)
    mass = reconstruct(state, cfg)["mass"]
    return cfg, state, mass


# --------------------------------------------------------------------------- #
# layer 1: re-grade the shipped record
# --------------------------------------------------------------------------- #
def test_record_covers_the_anchor_resolutions(payload):
    assert payload["anchor"] == "CASE-SPH-LAKE-AT-REST"
    assert payload["tolerance"] == ANCHOR_TOLERANCE
    assert [r["dx"] for r in payload["records"]] == [0.05, 0.04, 0.025, 0.02]


def test_gate_table_holds_when_regraded(payload):
    """Re-derive the gate table from the record's own numbers and demand a pass."""
    regraded = gates(payload["records"])
    assert regraded["all_pass"], [
        (row["dx"], key) for row in regraded["rows"] for key, cell in row.items()
        if isinstance(cell, dict) and cell.get("pass") is False]
    assert regraded["H10_resolution_floor"]["pass"]


def test_anchor_tolerance_is_met_by_the_consistent_closure(payload):
    """The claim of the document: the consistent closure is below the tolerance."""
    for row in payload["gates"]["rows"]:
        assert row["H1_anchor_tolerance"]["value"] < ANCHOR_TOLERANCE, row["dx"]


def test_shipped_and_compressible_closures_stay_above_it(payload):
    """The honest FAILs: no resolution and no closure of those two reaches the tolerance."""
    for row in payload["gates"]["rows"]:
        assert row["H2_shipped_control"]["value"] > 1e-2, row["dx"]
        assert row["H4_compressible_floor"]["value"] > 1e-6, row["dx"]
    # ... and the compressible floor does not shrink with dx by the factor the
    # anchor tolerance would need: it moves by less than 10x over a 2.5x refinement.
    values = [row["H4_compressible_floor"]["value"] for row in payload["gates"]["rows"]]
    assert max(values) / min(values) < 10.0


def test_regularisation_ladder_is_monotone(payload):
    """The floor is the regulariser, so the ladder must track it monotonically."""
    for record in payload["records"]:
        ladder = [r for r in record["regularisation_ladder"] if r["regularization"] > 0]
        values = [r["a_max_over_g"] for r in ladder]
        assert all(values[i] < values[i - 1] for i in range(1, len(values))), record["dx"]
        # six decades of regularisation buy five decades of residual
        assert values[0] / values[-1] > 1e5, record["dx"]


def test_controls_are_reported_with_the_claim(payload):
    """A machine-precision residual is only admissible with its controls attached."""
    for record in payload["records"]:
        c = record["controls"]
        assert c["offset"] - c["base"] < 1e-15          # level-set shift annihilated
        assert c["slope"] / max(c["base"], 1e-300) > 1e6  # 1 % slope error breaks it
        assert c["clamped"] > 1e-2                       # the kink is the wall clamp
        assert 0.9 <= c["zero_pressure"] <= 1.1          # pressure-less state free-falls
        assert c["cross_check"] < 1e-10                  # operator == shipped rhs
    # the winning row must also declare the moment condition it was taken at
    for record in payload["records"]:
        base = record["best"]
        assert base["condition_max"] is not None
        assert base["fallbacks"] >= 0


def test_document_is_generated_from_the_record(payload):
    if not DOC.is_file():
        pytest.skip("document not generated in this checkout")
    text = DOC.read_text()
    best = payload["records"][-1]["best"]["a_max_over_g"]
    assert f"{best:.3e}" in text
    assert "CASE-SPH-LAKE-AT-REST" in text


# --------------------------------------------------------------------------- #
# layer 2: rerun the physics at dx = 0.05 and compare with the record
# --------------------------------------------------------------------------- #
def test_coarse_rederivation_matches_the_record(payload, coarse):
    cfg, state, mass = coarse
    record = next(r for r in payload["records"] if r["dx"] == 0.05)
    best = closure_residual(state, cfg, mass,
                            HydrostaticClosureOptions(pressure="incompressible",
                                                      wall="continuation",
                                                      formulation="moment"))
    assert best["a_max_over_g"] < ANCHOR_TOLERANCE
    assert best["a_max_over_g"] == pytest.approx(record["best"]["a_max_over_g"],
                                                 rel=1e-6)
    assert record["particles"] == int(state["pos"].shape[0])


def test_coarse_shipped_control_still_fails(coarse):
    """If the shipped operator ever reached the tolerance this whole verdict is void."""
    cfg, state, mass = coarse
    shipped = closure_residual(state, cfg, mass,
                               HydrostaticClosureOptions(pressure="eos",
                                                         wall="shipped_adami",
                                                         formulation="symmetric"))
    assert shipped["a_x_max_over_g"] > 1e-2


def test_coarse_controls(coarse):
    cfg, state, mass = coarse
    c = controls(state, cfg, mass)
    assert c["offset"] - c["base"] < 1e-15
    assert c["slope"] / max(c["base"], 1e-300) > 1e6
    assert c["clamped"] > 1e-2
    assert 0.9 <= c["zero_pressure"] <= 1.1
    assert c["cross_check"] < 1e-10


def test_level_set_is_read_from_the_data(coarse):
    """``eta`` must come off the particle set, not off ``cfg.depth``."""
    cfg, state, _ = coarse
    level = level_set(state, cfg)
    assert level["eta"] == pytest.approx(float(state["pos"][:, 1].max()) + 0.5 * cfg.dx)
    assert level["roughness"] < 1e-12
    # and where depth/dx is not an integer the lattice is shorter than the tank,
    # which the level set reports instead of repairing
    torch.set_default_dtype(torch.float64)
    cfg2 = LakeConfig(dx=0.04)
    level2 = level_set(build_state(cfg2), cfg2)
    assert level2["eta"] < level2["depth_reference"]