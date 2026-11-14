"""Regression guards for the X4 closed-body harness (fsi2 insert body, linear field).

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The published X4 table (``docs/xval_fsi2/GAP-SPH-FSI-BENCHMARK_xval.md``) used a
fluid lattice whose lower bound was ``cy - bh/2 - pad`` while the insert *body*
reaches down to ``cy - r``.  With ``pad = 4 * 1.3 * dx`` shrinking under
refinement, the fraction of the wetted cylinder left dry grew with the ladder,
which is precisely the observed "error grows under refinement" signature
(+20 % / +32 % / +36 % for the panel arm).  ``cases.fsi2_insert_linear`` now
bounds the lattice by the body, and ``wetted=False`` reproduces the defective
box so that the erratum stays checkable.

These tests pin: (a) the corrected case wets every surface particle, (b) the
legacy case is measurably dry, (c) the exact surface quadrature -- which does not
involve any transfer operator -- reproduces ``-c A`` on the corrected case, so a
future operator regression cannot hide behind harness noise, and (d) the panel
operator passes the owner gate on the corrected geometry.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

import cases as C  # noqa: E402
from tensorsph import fsi_force_from_fluid as ff  # noqa: E402


def _cfg(case):
    return ff.ForceFromFluidConfig(dim=2, h=case.h, kernel="wendland")


def _dry_count(case):
    i_s, _ = ff.cross_pairs(case.pos_solid, case.pos_fluid, _cfg(case).support)
    cnt = torch.zeros(case.pos_solid.shape[0], dtype=torch.long)
    cnt.index_add_(0, i_s, torch.ones_like(i_s))
    return int((cnt == 0).sum())


def _quad(case):
    """Exact surface quadrature ``-sum p(x_i) n_i A_i`` on the solver's particles."""
    p = case.pressure_at(case.pos_solid)
    face = case.vol_solid.pow(0.5)
    return -(p[:, None] * case.normal_solid * face[:, None]).sum(dim=0)


def _panel(case):
    return ff.panel_pressure_force(
        case.pos_solid, case.vol_solid, case.normal_solid, case.pos_fluid,
        case.vol_fluid, case.pressure_fluid, _cfg(case)).sum(dim=0)


# --------------------------------------------------------------------------- #
# Wetting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dx", [0.1, 0.05, 0.025])
def test_wetted_body_has_no_dry_surface_particles(dx):
    """Every particle that receives a face load must see fluid within support."""
    case = C.fsi2_insert_linear(dx)
    assert _dry_count(case) == 0, dx


def test_legacy_bounding_box_leaves_the_lower_cylinder_dry():
    """Guard the erratum: the pre-m20600 box must stay reproducible and defective."""
    fracs = []
    for dx in (0.05, 0.025):
        case = C.fsi2_insert_linear(dx, wetted=False, include_buried=True)
        fracs.append(_dry_count(case) / case.pos_solid.shape[0])
    # the dry band is y < cy - bh/2 - pad with pad = 5.2 dx, so it *grows* as the
    # ladder is refined -- that is the signature the published table recorded as
    # "error grows under refinement"
    assert fracs[0] > 0.03, fracs
    assert fracs[1] > fracs[0], fracs


def test_buried_cylinder_arc_is_removed():
    """The arc swallowed by the beam sits inside the solid: no particle there."""
    case = C.fsi2_insert_linear(0.025)
    cx, cy = C.FSI2["insert_circle_center"]
    r, bh, bl = C.FSI2["insert_circle_radius"], C.FSI2["bh"], C.FSI2["bl"]
    pos = case.pos_solid
    on_cyl = (torch.linalg.vector_norm(
        pos - torch.tensor([cx, cy], dtype=torch.float64), dim=1) - r).abs() < 1.0e-9
    inside_beam = ((pos[:, 1] - cy).abs() < 0.5 * bh - 1.0e-12) & (pos[:, 0] > cx)
    assert on_cyl.any()
    assert not bool((on_cyl & inside_beam).any())
    # ... and the legacy surface does contain them, which is why it is not default
    legacy = C.fsi2_insert_linear(0.025, wetted=False, include_buried=True)
    pos_l = legacy.pos_solid
    on_cyl_l = (torch.linalg.vector_norm(
        pos_l - torch.tensor([cx, cy], dtype=torch.float64), dim=1) - r).abs() < 1.0e-9
    assert bool((on_cyl_l & ((pos_l[:, 1] - cy).abs() < 0.5 * bh - 1.0e-12)
                 & (pos_l[:, 0] > cx)).any())
    del bl


def test_analytic_pressure_law_matches_the_lattice_field():
    case = C.fsi2_insert_linear(0.1)
    assert case.pressure_at is not None
    assert torch.allclose(case.pressure_at(case.pos_fluid), case.pressure_fluid)


# --------------------------------------------------------------------------- #
# Loads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dx", [0.1, 0.05])
def test_surface_quadrature_reproduces_the_closed_body_reference(dx):
    """No operator involved: the discretised wetted surface integrates to -c A."""
    case = C.fsi2_insert_linear(dx)
    r, bh, bl = C.FSI2["insert_circle_radius"], C.FSI2["bh"], C.FSI2["bl"]
    area = math.pi * r * r + bh * bl
    assert case.reference_force[1] == pytest.approx(area, rel=1e-12)
    got = _quad(case)
    # Only the gradient (-y) component is graded: ``-c A`` says nothing about the
    # streamwise component, and the wetted surface has an open root face at
    # x = cx + r (height bh) whose closing face is interior to the solid, so a
    # residual of order p_root * bh = bh is expected and observed there.
    assert got[1] == pytest.approx(area, rel=0.02), got
    assert abs(got[0]) < 0.15 * area, got


@pytest.mark.parametrize("dx", [0.1, 0.05])
def test_panel_arm_passes_the_gate_on_the_wetted_body(dx):
    """The WLS panel arm is linear-exact: on a fully wetted closed body it must
    reach the 3 % target, not merely the 15 % gate."""
    case = C.fsi2_insert_linear(dx)
    ref = float(case.reference_force[1])
    err = (float(_panel(case)[1]) - ref) / ref
    assert abs(err) < 0.03, (dx, err)


def test_panel_error_shrinks_under_refinement():
    """The pre-fix harness had this quantity *growing*; pin the sign of the trend."""
    errs = []
    for dx in (0.1, 0.05, 0.025):
        case = C.fsi2_insert_linear(dx)
        ref = float(case.reference_force[1])
        errs.append(abs((float(_panel(case)[1]) - ref) / ref))
    assert errs[-1] <= errs[0] + 1.0e-3, errs
    assert max(errs) < 0.03, errs