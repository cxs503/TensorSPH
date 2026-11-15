"""Gates for the boundary-integral wall treatment (CASE-SPH-LAKE-AT-REST).

Two kinds of assertions, kept apart on purpose:

* **invariants / implementation checks** -- geometry of the wall surface, the
  face-pressure closure, the surface-vs-volume identity, and the controls
  (pressure sensitivity, gravity off, no-wall).  These are what a reviewer should
  trust; they fail if the operator is ever mis-signed or mis-scaled.
* **pinned measurements** -- the numbers of the published study, recomputed from
  the library and compared with the frozen artefact
  ``tests/data/lake_at_rest_bi/bi_wall.json``.  They are *measurements*, so they
  are pinned with a relative tolerance, never asserted as desirable: the honest
  FAIL of the anchor is a value to keep, not a bug to fix.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct
from tensorsph.lake_at_rest_bi import (ANCHOR_TOLERANCE, band_stats, bi_accel,
                                       solve_mass, variant_spec, wall_face_pressure,
                                       wall_integral_check, wall_segments)
from tensorsph.lake_at_rest_wb import wb_accel, wb_fields

ARTEFACT = Path(__file__).resolve().parent / "data" / "lake_at_rest_bi" / "bi_wall.json"


@pytest.fixture(scope="module")
def torch64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def case(torch64):
    cfg = LakeConfig(dx=0.05)
    state = build_state(cfg)
    mass = reconstruct(state, cfg)["mass"]
    return cfg, state, mass


@pytest.fixture(scope="module")
def study():
    if not ARTEFACT.exists():  # pragma: no cover - artefact ships with the repo
        pytest.skip(f"missing study artefact {ARTEFACT}")
    return json.loads(ARTEFACT.read_text())


# --------------------------------------------------------------------------- #
# geometry and closure
# --------------------------------------------------------------------------- #
def test_wall_segments_cover_the_boundary(case):
    cfg, _, _ = case
    seg = wall_segments(cfg)
    nx = int(round(cfg.width / cfg.dx))
    ny = int(round(cfg.depth / cfg.dx))
    assert seg["pos"].shape[0] == nx + 2 * ny
    # the quadrature reproduces the exact perimeter to machine precision
    assert seg["quadrature_perimeter"] == pytest.approx(seg["perimeter"], rel=1e-14)
    assert seg["perimeter"] == pytest.approx(2.0 * cfg.depth + cfg.width, rel=1e-14)


def test_wall_normals_point_into_the_fluid(case):
    cfg, state, _ = case
    seg = wall_segments(cfg)
    centre = state["pos"].mean(dim=0)
    outward = seg["pos"] - centre          # from the tank centre to the face
    dot = (outward * seg["normal"]).sum(dim=1)
    # on the bottom face the normal has no x, on the side faces no y, so the
    # meaningful statement is that the normal is *anti*-parallel to the offset
    assert bool((dot <= 1e-12).all())
    assert torch.allclose(seg["length"], torch.full_like(seg["length"], cfg.dx))


def test_quadrature_refinement_moves_nothing_but_the_lengths(case):
    cfg, _, _ = case
    coarse, fine = wall_segments(cfg, sub=1), wall_segments(cfg, sub=2)
    assert fine["pos"].shape[0] == 2 * coarse["pos"].shape[0]
    assert fine["quadrature_perimeter"] == pytest.approx(coarse["perimeter"], rel=1e-14)
    assert torch.allclose(fine["length"], torch.full_like(fine["length"], cfg.dx / 2))


def test_face_pressure_closures(case):
    cfg, state, mass = case
    seg = wall_segments(cfg)
    fields = wb_fields(cfg, mass)
    adami = wall_face_pressure(seg, fields, mode="adami")
    exact = wall_face_pressure(seg, fields, mode="hydrostatic")
    zero = wall_face_pressure(seg, fields, mode="zero")
    assert float(zero["p_w"].abs().max()) == 0.0
    # the data-driven extrapolation must land on the hydrostatic scale
    scale = float((cfg.rho0 * cfg.gravity * cfg.depth))
    assert 0.5 * scale < float(adami["p_w"].max()) <= 1.5 * scale
    rel = float(((adami["p_w"] - exact["p_w"]).abs().max()) / scale)
    assert rel < 0.35, rel          # measured: the closure is good, not exact
    # face density never below rho0 (the Tait floor of the wall closure)
    assert float(adami["rho_w"].min()) >= cfg.rho0 * (1.0 - 1e-12)


def test_surface_form_reproduces_the_volume_quadrature(case):
    """The implementation check: the wall integral vs the wall-particle sum."""
    cfg, _, mass = case
    check = wall_integral_check(cfg, mass, wall_mode="adami")
    assert 0.0 < check["relative_error"] < 0.10, check  # one-order closure, measured
    assert check["volume_scale"] > 1.0


# --------------------------------------------------------------------------- #
# the operator and its controls
# --------------------------------------------------------------------------- #
def test_controls_of_the_boundary_integral(case):
    cfg, state, mass = case
    h = cfg.h()
    kw = variant_spec("bi_surface_adami")
    nominal = band_stats(cfg, state["pos"], bi_accel(cfg, mass, **kw), h)
    gravity_off = band_stats(cfg, state["pos"],
                             bi_accel(cfg, mass, gravity_on=False, **kw), h)
    scaled = band_stats(cfg, state["pos"],
                        bi_accel(cfg, mass, pressure_scale=1.01, **kw), h)
    no_wall = band_stats(cfg, state["pos"],
                         bi_accel(cfg, mass, wall_term=False), h)
    # gravity off leaves the pressure field: the residual must be the pressure term
    assert gravity_off["max_a_over_g"] > 1.0
    # a 1 % wrong pressure field must move the residual (sensitivity, not invariance)
    assert scaled["max_a_over_g"] != pytest.approx(nominal["max_a_over_g"], rel=1e-3)
    # the wall term is what is being measured: without it the residual explodes
    assert no_wall["max_a_over_g"] > 5.0 * nominal["max_a_over_g"]


def test_naive_surface_form_is_dimensionally_short(case):
    """Kept as a measurement: ``int p gradW dS`` overshoots by ~1/dx."""
    cfg, state, mass = case
    kw = variant_spec("bi_naive_dS_adami")
    naive = band_stats(cfg, state["pos"], bi_accel(cfg, mass, **kw), cfg.h())
    proper = band_stats(cfg, state["pos"],
                        bi_accel(cfg, mass, **variant_spec("bi_surface_adami")), cfg.h())
    assert naive["max_a_over_g"] > 10.0 * proper["max_a_over_g"]


def test_shipped_baseline_is_reproduced(case):
    """The comparison baseline must be the shipped operator, bit-for-bit."""
    cfg, state, mass = case
    acc = wb_accel(wb_fields(cfg, mass), own_pressure="eos", form="symmetrised",
                   corrected_gradient=False)
    stats = band_stats(cfg, state["pos"], acc, cfg.h())
    assert stats["max_ax_over_g"] == pytest.approx(0.1092, abs=2e-3)
    assert stats["max_ay_over_g"] < 1e-10      # vertical balance of the mass solve


def test_mass_resolve_closes_vertically_for_the_shipped_operator_only(case):
    """Contrast: the shipped operator's mass solve closes, the wall integral's does not."""
    cfg, state, _ = case
    shipped = solve_mass(cfg, wall_term=True, form="dS", wall_mode="zero",
                         fluid_term=True)  # no wall traction: not the shipped sum
    # the shipped closure itself (reconstruct) is the reference
    assert float(reconstruct(state, cfg)["residual_history"][-1]) < 1e-10
    # the boundary-integral operator cannot hold the vertical balance
    bi_solved = solve_mass(cfg, max_iter=8, **variant_spec("bi_surface_adami"))
    assert not bi_solved["converged"]
    assert bi_solved["stats"]["max_ay_over_g"] > 1e-2
    assert bi_solved["mass_modulation"] > 1.2
    assert shipped["stats"]["max_a_over_g"] > 0.0   # a solve happened, not skipped


# --------------------------------------------------------------------------- #
# the published numbers (recomputed, never trusted from the document)
# --------------------------------------------------------------------------- #
def test_study_artefact_matches_the_library(case, study):
    cfg, state, mass = case
    row = min(study["resolutions"], key=lambda r: abs(r["dx"] - cfg.dx))
    assert abs(row["dx"] - cfg.dx) < 1e-12
    base = band_stats(cfg, state["pos"],
                      wb_accel(wb_fields(cfg, mass), own_pressure="eos",
                               form="symmetrised", corrected_gradient=False), cfg.h())
    assert base["max_a_over_g"] == pytest.approx(row["baseline"]["max_a_over_g"], rel=1e-6)
    for name in study["variants"]:
        acc = bi_accel(cfg, mass, **variant_spec(name))
        stats = band_stats(cfg, state["pos"], acc, cfg.h())
        assert stats["max_a_over_g"] == pytest.approx(
            row["variants"][name]["max_a_over_g"], rel=1e-6), name


def test_the_verdict_is_pinned(study):
    gates = study["gates"]
    assert gates["anchor_tolerance"] == ANCHOR_TOLERANCE
    # the honest FAIL: nothing reaches the machine-precision tolerance
    assert gates["some_variant_meets_anchor_tolerance"] is False
    assert gates["best_boundary_integral"] > 1e5 * ANCHOR_TOLERANCE
    # and the boundary integral is not even an improvement over the shipped wall
    assert gates["boundary_integral_beats_wall_particles"] is False
    # the wall term does matter (so the comparison is not degenerate)
    assert gates["wall_term_helps"] is True
    assert gates["resolved_boundary_integral_converges_vertically"] is False
    assert "does NOT close" in gates["verdict"]


def test_boundary_integral_degrades_under_refinement(study):
    """The measured trend: unlike a convergent discretisation, it grows."""
    rows = sorted(study["resolutions"], key=lambda r: -r["dx"])
    coarse = rows[0]["variants"]["bi_surface_adami"]["max_a_over_g"]
    finest = rows[-1]["variants"]["bi_surface_adami"]["max_a_over_g"]
    assert finest > coarse
    # while the shipped baseline stays put
    base = [r["baseline"]["max_a_over_g"] for r in rows]
    assert max(base) / min(base) < 1.5
    assert math.isfinite(finest)