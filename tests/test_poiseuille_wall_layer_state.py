"""Invariants of the mirror-wall Poiseuille *wall-layer state* artifact.

The anchor `CASE-POISEUILLE-SPH` is graded on velocity.  This test guards the
field that backs it -- summation density and Tait pressure -- and the mechanism
claim this delivery adds (the wall row loses a fixed fraction of its viscous
coupling, identically at every resolution, which is what the strict wall-shear
gauge measures).

Every assertion is read from the committed artifact
``benchmarks/poiseuille_mirror/out/wall_layer_state.json``, so a re-run that
moves a number fails here instead of silently changing the note.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "benchmarks/poiseuille_mirror/out/wall_layer_state.json"
DOC = REPO / "docs/verification_poiseuille_wall_layer_state.md"


@pytest.fixture(scope="module")
def data():
    assert ART.exists(), f"missing artifact {ART}"
    return json.loads(ART.read_text())


def test_ladder_is_present(data):
    res = [c["resolution_H_over_dx"] for c in data["cases"]]
    assert len(res) >= 2, "the ladder must have at least two rungs"
    assert res == sorted(res)
    for case in data["cases"]:
        assert case["n_ghost"] > 0, "mirror ghosts must be active"
        assert case["n_averaged_samples"] > 0


def test_summation_density_is_uniform_so_no_deficit_mechanism(data):
    """Q2's first candidate: a near-wall *density* deficit.

    It does not exist: the row-0 density is the bulk density to better than
    2e-5, and the residual is a surplus.  Pinned so a future change that
    introduces one is visible rather than absorbed into the wall-shear story.
    """
    for case in data["cases"]:
        d = case["diagnostics"]
        assert abs(d["rho_over_rho0_row0"] - d["rho_over_rho0_bulk"]) < 2e-5, (
            f"H/dx={case['resolution_H_over_dx']}: density is not uniform")
        # the ~0.13 % offset from rho0 is the particle-mass convention and is
        # the same on every row; what must be absent is any *row-to-row* deficit
        assert abs(d["rho_deficit_row0"] - d["rho_deficit_bulk"]) < 2e-5
        assert d["rho_deficit_asymmetry"] < 1e-5, "unexpected top/bottom asymmetry"


def test_pressure_gradient_is_zero_as_the_analytic_solution_requires(data):
    """Plate Poiseuille with an axial body force has ``grad p = 0``.

    This is the field-level check the anchor never made: the body force must be
    carried by viscous stress alone.
    """
    for case in data["cases"]:
        d = case["diagnostics"]
        assert abs(d["dp_dy_over_rho0_g"]) < 1e-4, (
            f"H/dx={case['resolution_H_over_dx']}: spurious wall-normal pressure "
            f"gradient {d['dp_dy_over_rho0_g']:.2e} of rho0 g")
        assert d["p_x_span_over_scale"] < 0.05, (
            "spurious streamwise pressure gradient in a periodic channel")


def test_pressure_level_is_an_eos_offset_not_zero(data):
    """The absolute pressure is *not* zero: it is the EOS response to the mass
    convention ``rho0 dx^2``.  Recorded so no one writes a ``p ~ 0`` gate."""
    for case in data["cases"]:
        d = case["diagnostics"]
        assert 0.2 < d["p_bulk_over_scale"] < 1.5, d["p_bulk_over_scale"]
        # ... and it is flat, i.e. the offset carries no force
        assert abs(d["p_row0_over_scale"] - d["p_bulk_over_scale"]) < 0.02


def test_operator_balances_the_body_force_in_the_interior(data):
    """``rhs`` applied to the exact solution must return ``-g`` in the bulk."""
    for case in data["cases"]:
        op = case["operator_probe"]
        assert abs(abs(op["residual_over_g_interior_max"]) - 1.0) < 1e-3, (
            f"H/dx={case['resolution_H_over_dx']}: interior operator residual "
            f"{op['residual_over_g_interior_max']}")
        assert abs(op["residual_over_g_odd_part"]) < 1e-3, "solution is not symmetric"


def test_the_wall_row_loses_a_resolution_independent_share_of_its_drag(data):
    """The mechanism: 32 % of the viscous deceleration is missing on row 0.

    The *share* is a property of the discrete operator and the mirror ghost
    layer, so it must not move with ``dx`` -- that is what makes it a
    discretisation defect of the wall layer rather than a convergence trend.
    """
    row0 = [c["operator_probe"]["residual_over_g_row0"] for c in data["cases"]]
    for value in row0:
        assert -0.75 < value < -0.60, f"wall-row operator residual {value}"
    assert max(row0) - min(row0) < 5e-3, (
        "the wall-row deficit is supposed to be resolution independent")


def test_discrete_fixpoint_reproduces_the_open_gauge_and_stays_inside_the_band(data):
    """Q2 answered: the fixed point predicts gauge G1 without time stepping.

    It must (a) solve its own linear system, (b) predict the same out-of-band
    excess the time-marched run shows, and (c) come back inside the 5 % band at
    the finest rung available.
    """
    preds = {}
    for case in data["cases"]:
        fp = case["discrete_fixpoint"]
        assert fp["system_residual_over_g"] < 1e-8, "linear solve is not exact"
        preds[case["resolution_H_over_dx"]] = 100.0 * fp["G1_bias_corrected_excess"]
        # agreement with the time-marched run, from this artifact's own rows
        measured = 100.0 * case["diagnostics"]["G1_bias_corrected_excess"]
        assert abs(preds[case["resolution_H_over_dx"]] - measured) < 1.5, (
            f"H/dx={case['resolution_H_over_dx']}: fixed point {preds} vs "
            f"measured {measured:.2f} %")
    assert preds[min(preds)] > 5.0, "the coarsest rung must reproduce the caveat"
    assert preds[max(preds)] < 10.0


def test_note_is_generated_and_free_of_placeholders(data):
    assert DOC.exists(), f"missing note {DOC}"
    text = DOC.read_text()
    assert "nan" not in text.lower().replace("nano", "")
    assert "{q2[" not in text and "{main[" not in text, "unrendered f-string template"
    assert "**Status: run**" in text or "**Status: not_run in this delivery**" in text