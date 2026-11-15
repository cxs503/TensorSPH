"""Gates for the closed-tank rest gap (`CASE-SPH-LAKE-AT-REST`).

`docs/verification_lake_at_rest.md` records an honest FAIL for the wall-bounded
tank; `docs/verification_lake_at_rest_periodic.md` records that the same operator
meets the anchor tolerance once the side walls are replaced by periodicity.  This
module pins the *reason* the closed tank fails, so that the claim cannot rot:

* the residual does not refine away (measured over four resolutions, and
  re-derived live at `dx = 0.05` from the shipped operator);
* it lives in the two side-wall columns and is mirror-antisymmetric;
* at the worst particle it is entirely the neighbour-pressure sum (the particle's
  own pressure is clamped to zero at the free surface), so no mass field can
  cancel it;
* neither a different wall-face pressure (Adami / dry-clip / mirrored /
  analytic-hydrostatic) nor a different wall mass closes it;
* no shipped operator option closes it either.

The numbers are re-derived from `tests/data/lake_at_rest/wall_gap.json`, which
`benchmarks/lake_at_rest/wall_gap.py` writes programmatically.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct, residual_report

EVIDENCE = Path(__file__).parent / "data" / "lake_at_rest" / "wall_gap.json"
ANCHOR_TOLERANCE = 1.0e-9


@pytest.fixture(scope="module")
def evidence() -> dict:
    with open(EVIDENCE) as handle:
        return json.load(handle)


def test_evidence_channel_is_the_wall_bounded_case(evidence):
    assert evidence["case"] == "CASE-SPH-LAKE-AT-REST"
    assert evidence["geometry"]["periodic_x"] is False
    assert evidence["anchor_tolerance"] == ANCHOR_TOLERANCE
    assert [row["dx"] for row in evidence["ladder"]] == [0.050, 0.040, 0.025, 0.020]


def test_vertical_balance_is_closable_at_every_resolution(evidence):
    """The solve is a real solve: a_y reaches machine precision everywhere."""
    for row in evidence["ladder"]:
        assert row["reconstruct"]["converged"], row["dx"]
        assert row["reconstruct"]["a_y_max_over_g"] < ANCHOR_TOLERANCE, row["dx"]


def test_horizontal_residual_does_not_refine_away(evidence):
    """The gap is not a discretisation error: dx 0.05 -> 0.02 changes it by 1.2x
    with no trend, 8 orders of magnitude above the anchor tolerance."""
    values = [row["adami"]["a_x_max_over_g"] for row in evidence["ladder"]]
    assert min(values) > 1.0e-2
    assert max(values) / min(values) < 1.3, values
    shortfall = math.log10(min(values) / ANCHOR_TOLERANCE)
    assert shortfall > 6.0, shortfall
    # No monotone decay: the finest grid is not closer to zero than the coarsest.
    assert values[-1] > 0.9 * values[0], values


def test_residual_lives_in_the_two_wall_columns_and_is_antisymmetric(evidence):
    for row in evidence["ladder"]:
        columns = row["adami"]["per_column_max"]
        ordered = sorted(columns.items(), key=lambda kv: float(kv[0]))
        values = [float(v) for _k, v in ordered]
        wall_side = values[0]
        # the wall column carries the maximum, and the profile decays inward
        assert wall_side == pytest.approx(max(values))
        assert wall_side > 5.0 * min(values), (row["dx"], wall_side, min(values))
        assert values[0] > values[1] > values[2], (row["dx"], values[:3])
        # a_x(W - x) = -a_x(x) to roundoff: the defect is the side wall, not noise.
        assert row["adami"]["antisymmetry"] < 1.0e-9, row["dx"]
        # The worst particle sits on the free-surface row of the wall column.
        worst = row["adami"]["worst_particle"]
        top = max(float(k) for k in row["adami"]["per_row_max"])
        assert abs(worst["y"] - top) < 0.25 * row["dx"], (row["dx"], worst)


def test_worst_particle_is_pure_neighbour_pressure(evidence):
    """Its own pressure is clamped to zero, so no mass field has a handle on it."""
    for row in evidence["ladder"]:
        split = row["worst_split"]
        assert split["pressure_at_worst"] == 0.0
        assert abs(split["own_pressure_term_over_g"]) < 1.0e-12
        assert abs(split["total_over_g"]
                   - split["neighbour_pressure_term_over_g"]) < 1.0e-12


def test_no_wall_face_pressure_field_closes_the_gap(evidence):
    """Adami is already the best of the four; the artefact is not the wall
    pressure extrapolation."""
    for row in evidence["ladder"]:
        base = row["adami"]["a_x_max_over_g"]
        variants = row["wall_pressure_variants"]
        assert variants["dry_clip"]["a_x_max_over_g"] == pytest.approx(base)
        assert variants["mirror"]["a_x_max_over_g"] > 1.1 * base, row["dx"]
        assert variants["hydrostatic"]["a_x_max_over_g"] > 0.5 * base, row["dx"]


def test_no_shipped_closure_closes_the_gap(evidence):
    closures = evidence["closures"]
    assert len(closures) >= 4
    baseline = [row for row in closures if row["closure"] == "baseline"][0]
    for row in closures:
        if row["closure"] == "baseline":
            continue
        assert not (row["converged"] and row["a_x_max_over_g"]
                    < 0.9 * baseline["a_x_max_over_g"]), row


def test_recorded_numbers_come_from_the_shipped_operator():
    """Re-derive the dx = 0.05 entry live, from tensorsph.lake_at_rest."""
    with open(EVIDENCE) as handle:
        recorded = [row for row in json.load(handle)["ladder"]
                    if row["dx"] == 0.050][0]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cfg = LakeConfig(dx=0.050, solver_max_iter=40)
        state = build_state(cfg)
        solved = reconstruct(state, cfg)
        live = residual_report(build_state(cfg), cfg, solved["mass"])
    finally:
        torch.set_default_dtype(previous)
    assert live["a_max_over_g_horizontal"] == pytest.approx(
        recorded["adami"]["a_x_max_over_g"], rel=1e-12)
    assert live["a_max_over_g"] == pytest.approx(
        recorded["reconstruct"]["a_y_max_over_g"], rel=1e-9)
    assert live["a_max_over_g"] < ANCHOR_TOLERANCE
    assert live["a_max_over_g_horizontal"] > 1.0e7 * ANCHOR_TOLERANCE