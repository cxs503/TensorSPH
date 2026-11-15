"""Gates for the wavy-bed extension of `CASE-SPH-LAKE-AT-REST`.

The flat-bottom case is certified in ``tests/test_lake_at_rest.py``.  What is
asserted here is the *transfer* question and it is deliberately two-sided:

* **identities** that must hold exactly -- ``A = 0`` reduces to the shipped flat
  periodic lattice, the bed wall is the bed contour offset by ``(k - 1/2) dx``
  along the normal, every particle clears the bed by ``clearance * dx``.  If any
  of these drifted, the curved-bed numbers would be measuring geometry bugs.
* **the measured defect**, asserted as a fact so a future well-balanced operator
  turns the test red instead of silently invalidating the prose: with the
  translation-invariant mass field (``tie="row"``, the statement that certified the
  flat tank) a ripple of 8 % of the depth leaves an ``O(g)`` residual, and buying
  vertical roundoff with a free per-particle mass field costs a horizontal
  residual an order of magnitude worse than not fitting at all.
* **negative controls** that must stay ``O(1)`` -- uniform lattice mass free-falls
  at ``g`` -- otherwise the roundoff identities would be measured on a dead
  operator.

Kept at ``dx = 0.04`` (70 particles), ``t_end = 0.02 s`` so the module runs in
tens of seconds; the published artefact uses ``dx = 0.04`` and ``0.02``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.lake_at_rest import LakeConfig, build_state
from tensorsph.lake_at_rest_bed import (GATES, BedConfig, bed_height, bed_slope,
                                        bed_state, bed_wall_particles,
                                        evaluate_gates, fluid_lattice,
                                        max_bed_slope, run_case, tie_map)


@pytest.fixture(autouse=True)
def _float64():
    """The case is a float64 statement: run everything here in double."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def runs():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        out = {}
        for amplitude, ties, iterations in ((0.0, ("row", "particle"), 30),
                                            (0.005, ("row", "particle"), 40),
                                            (0.01, ("row",), 30)):
            cfg = BedConfig(dx=0.04, bed_amplitude=amplitude, t_end=0.02,
                            sample_every=5, solver_max_iter=iterations)
            out[amplitude] = run_case(cfg, ties=ties, t_end=0.02)
        return out
    finally:
        torch.set_default_dtype(previous)


# --------------------------------------------------------------------------
# geometry identities
# --------------------------------------------------------------------------
def test_flat_amplitude_reduces_to_the_shipped_lattice():
    """``A = 0`` is the shipped flat periodic lattice, not merely similar to it."""
    cfg = BedConfig(dx=0.04, bed_amplitude=0.0)
    mine = fluid_lattice(cfg).round(decimals=12)
    shipped = build_state(LakeConfig(width=0.40, depth=0.30, dx=0.04,
                                     periodic_x=True))["pos"]
    assert mine.shape == shipped.shape
    assert torch.allclose(mine, shipped, atol=1e-12), "A=0 lattice drifted"


def test_bed_geometry_is_periodic_and_bounded():
    cfg = BedConfig(dx=0.04, bed_amplitude=0.02, bed_waves=1)
    x = torch.linspace(0.0, cfg.width, 257, dtype=torch.float64)
    b = bed_height(x, cfg)
    assert abs(float(b.max())) < 1e-15, "crest must sit at y = 0"
    assert abs(float(b.min()) + cfg.bed_amplitude) < 1e-12
    assert abs(float(bed_height(torch.tensor([0.0], dtype=torch.float64),
                                cfg))) < 1e-15
    # seam closure: b(0) == b(W) and the slope is periodic too
    assert abs(float(bed_height(torch.tensor([cfg.width], dtype=torch.float64),
                                cfg))) < 1e-15
    assert abs(float(bed_slope(torch.tensor([0.0], dtype=torch.float64), cfg))) \
        < 1e-15
    assert max_bed_slope(cfg) == pytest.approx(
        0.5 * cfg.bed_amplitude * 2.0 * math.pi / cfg.width, rel=1e-15)


def test_wall_layers_follow_the_bed_normal():
    """First wall layer is ``dx/2`` below the bed along the normal, not straight
    down: on a slope those are different points and only one is the lattice."""
    cfg = BedConfig(dx=0.04, bed_amplitude=0.02)
    wall = bed_wall_particles(cfg)
    xs = fluid_lattice(cfg)[:, 0].unique()   # built in the same column order
    first = wall[:xs.numel()]
    b = bed_height(xs, cfg)
    s = bed_slope(xs, cfg)
    norm = torch.sqrt(1.0 + s.square())
    distance = torch.sqrt((first[:, 0] - xs).square()
                          + (first[:, 1] - b).square())
    assert torch.allclose(distance, torch.full_like(distance, 0.5 * cfg.dx),
                          atol=1e-12)
    # the offset really has a horizontal component away from the crest
    assert float((first[:, 0] - xs).abs().max()) > 0.0
    assert torch.allclose(first[:, 0], xs + 0.5 * cfg.dx * s / norm, atol=1e-12)
    assert torch.allclose(first[:, 1], b - 0.5 * cfg.dx / norm, atol=1e-12)


def test_every_particle_clears_the_bed():
    for amplitude in (0.0, 0.0025, 0.005, 0.01, 0.02):
        cfg = BedConfig(dx=0.04, bed_amplitude=amplitude)
        state = bed_state(cfg, "row")
        assert float(state["clearance"].min()) >= cfg.clearance * cfg.dx - 1e-12
        assert int(state["n_fluid"]) > 0
        # the pair list must contain every interaction inside one image
        pairs = state["pairs"]
        assert pairs.shape[0] > 0
        assert int(pairs.max()) < state["n_fluid"] + state["n_wall"]


def test_row_tie_groups_particles_by_height():
    cfg = BedConfig(dx=0.04, bed_amplitude=0.01)
    state = bed_state(cfg, "row")
    mapping = tie_map(state["pos"], cfg, "row")
    rows = state["pos"][:, 1]
    for value in torch.unique(rows):
        mask = (rows - value).abs() < 1e-12
        assert int(torch.unique(mapping[mask]).numel()) == 1
    assert int(mapping.max()) + 1 == int(torch.unique(rows).numel())


# --------------------------------------------------------------------------
# the flat case still passes through this code path
# --------------------------------------------------------------------------
def test_flat_bottom_is_certified_through_the_bed_module(runs):
    solution = runs[0.0]["solutions"]["row"]
    assert solution["converged"]
    assert solution["a_y_over_g"] < 1e-9
    assert solution["a_x_over_g"] < 1e-9
    rest = runs[0.0]["rest"]["row"]
    assert rest["max_v_over_c0"] < 1e-9
    assert rest["max_dy_over_H"] < 1e-9


def test_negative_control_uniform_mass_free_falls(runs):
    control = runs[0.01]["controls"]["uniform_lattice_mass"]
    assert control["a_y_over_g"] == pytest.approx(1.0, abs=1e-9)
    assert control["p_max"] < 1e-6, "uniform mass in still water has no pressure"


# --------------------------------------------------------------------------
# the transfer result (documented defects, asserted so they cannot go stale)
# --------------------------------------------------------------------------
def test_translation_invariant_mass_field_cannot_balance_a_curved_bed(runs):
    """The ``B2`` claim: roundoff at ``A = 0``, ``O(g)`` at ``A = 0.01``."""
    solution = runs[0.01]["solutions"]["row"]
    assert solution["a_y_over_g"] > 1e-3
    assert not solution["converged"]
    # and the price it pays is not small: it deforms the mass field by tens of %
    assert solution["mass_modulation_max_abs"] > 0.1


def test_fitting_vertical_balance_costs_the_horizontal_one(runs):
    """``B3``: the free per-particle fit buys ``a_y`` roundoff with a *worse* a_x"""
    result = runs[0.005]
    fit = result["solutions"]["particle"]
    prior = result["controls"]["analytic_hydrostatic_profile"]
    assert fit["a_y_over_g"] < 1e-9, "the fit is supposed to reach a_y roundoff here"
    assert fit["a_x_over_g"] > 5.0 * prior["a_x_over_g"]
    assert fit["mass_modulation_max_abs"] > 0.05


def test_curved_bed_defect_lives_near_the_bed(runs):
    """The curved bed injects its residual into the near-bed bands, not the top.

    Measured on the fitting-free prior: ``A = 0.01`` raises the near-bed
    ``max|a_y|/g`` by an order of magnitude while the band above two spacings is
    unchanged (that one is the free-surface truncation, flat or not).
    """
    flat = runs[0.0]["controls"]["analytic_hydrostatic_profile"]["bands"]
    curved = runs[0.01]["controls"]["analytic_hydrostatic_profile"]["bands"]
    assert curved["near_bed"]["a_y_over_g"] > 10.0 * flat["near_bed"]["a_y_over_g"]
    assert curved["interior"]["a_y_over_g"] == pytest.approx(
        flat["interior"]["a_y_over_g"], rel=0.05)


def test_spurious_current_grows_with_bed_slope(runs):
    """Fitting-free measurement: the still lake starts to move on a curved bed."""
    flat = runs[0.0]["controls"]["analytic_hydrostatic_profile"]
    curved = runs[0.01]["controls"]["analytic_hydrostatic_profile"]
    assert curved["a_y_over_g"] > 2.0 * flat["a_y_over_g"]
    assert curved["a_x_over_g"] > 10.0 * flat["a_x_over_g"]
    assert curved["rest"]["max_v_over_c0"] > flat["rest"]["max_v_over_c0"]


# --------------------------------------------------------------------------
# gate grader
# --------------------------------------------------------------------------
def test_gate_grader_marks_missing_runs_not_run():
    grades = evaluate_gates({"runs": []})
    assert len(grades["gates"]) == len(GATES)
    assert all(g["verdict"] == "not_run" for g in grades["gates"])
    assert grades["passed"] is False


def test_gate_grader_reads_the_sweep(runs):
    payload = {"runs": []}
    for amplitude, result in runs.items():
        payload["runs"].append({"dx": 0.04, "bed_amplitude": amplitude,
                                "result": result})
    grades = evaluate_gates(payload)
    verdicts = {g["gate"]: g["verdict"] for g in grades["gates"]}
    # B1/B2/B4/B6 only need dx=0.04 + A=0/0.01 and are covered by the fixture
    assert verdicts["B1"] == "pass"
    assert verdicts["B2"] == "pass"
    assert verdicts["B4"] == "pass"
    assert verdicts["B6"] == "pass"
    assert verdicts["B3"] == "pass"
    assert verdicts["B5"] == "pass"
    assert grades["passed"] is True