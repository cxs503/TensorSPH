"""Tests for the ported SPHinXsys ``ForceFromFluid`` load transfer.

The assertions are the analytic references of ``benchmarks/xval_fsi2``:
a linear pressure field must be transferred exactly by the panel operator, the
one-sided OSS operator must reproduce its documented systematic over-load
(so a future "fix" that silently changes it is caught), signs must point the
right way, and the viscous branch must drag the wall in the fluid direction.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

import cases as C  # noqa: E402
from tensorsph.fsi_force_from_fluid import (ForceFromFluidConfig,  # noqa: E402
                                            OSS_PROVENANCE, cross_pairs,
                                            one_sided_viscous_truncation_factor,
                                            panel_pressure_force,
                                            pressure_force_from_fluid,
                                            viscous_force_from_fluid,
                                            wall_shear_from_gradient)


def _cfg(case, riemann=False):
    return ForceFromFluidConfig(dim=2, h=case.h, kernel="wendland", riemann=riemann)


def _pressure(case, riemann=False):
    return pressure_force_from_fluid(
        case.pos_solid, case.vol_solid, case.normal_solid,
        torch.zeros_like(case.pos_solid), case.vel_solid,
        case.pos_fluid, case.vol_fluid, case.pressure_fluid, case.rho_fluid,
        case.vel_fluid, case.acc_ext_fluid, _cfg(case, riemann))


def _along(force, ref):
    d = ref / ref.norm()
    force = force.sum(dim=0) if force.dim() == 2 else force
    return float(force @ d), float(ref.norm())


def test_provenance_is_pinned():
    """The blueprint must name the OSS repo, ref and the files actually read."""
    assert OSS_PROVENANCE["repo"] == "Xiangyu-Hu/SPHinXsys"
    assert OSS_PROVENANCE["ref"] == "master"
    assert any("force_on_structure" in f for f in OSS_PROVENANCE["files"])


def test_cross_pairs_are_solid_fluid_and_within_support():
    case = C.horizontal_gate(dx=0.05)
    i_s, j_f = cross_pairs(case.pos_solid, case.pos_fluid, case.support)
    assert i_s.numel() > 0
    assert i_s.numel() == j_f.numel()
    assert int(i_s.max()) < case.pos_solid.shape[0]
    assert int(j_f.max()) < case.pos_fluid.shape[0]
    r = torch.linalg.vector_norm(case.pos_solid[i_s] - case.pos_fluid[j_f], dim=1)
    assert float(r.max()) < case.support


def test_panel_is_exact_on_a_linear_pressure_field():
    """A plane fit reproduces p = a + b.x exactly, so the wall load is analytic."""
    for dx in (0.05, 0.025):
        case = C.vertical_wall(dx=dx)
        f = panel_pressure_force(case.pos_solid, case.vol_solid, case.normal_solid,
                                 case.pos_fluid, case.vol_fluid, case.pressure_fluid,
                                 _cfg(case))
        got, ref = _along(f, case.reference_force)
        assert abs(got - ref) / ref < 0.01, (dx, got, ref)


def test_panel_load_on_a_floor_points_down():
    """The fluid column sits on the floor, so the force it applies is downward."""
    case = C.horizontal_gate(dx=0.05)
    f = panel_pressure_force(case.pos_solid, case.vol_solid, case.normal_solid,
                             case.pos_fluid, case.vol_fluid, case.pressure_fluid,
                             _cfg(case)).sum(dim=0)
    assert float(f[1]) < 0.0
    assert abs(float(f[0])) < 1.0e-6 * abs(float(f[1]))
    got, ref = _along(f, case.reference_force)
    assert abs(got - ref) / abs(ref) < 0.05


def test_oss_operator_signs_and_documented_overload():
    """The OSS pressure branch pushes the solid away from the fluid, and its
    systematic over-load on a single surface layer is pinned to the measured
    +24..+29 % band (dx independent) so that any change is visible."""
    for builder in (C.vertical_wall, C.horizontal_gate):
        case = builder(dx=0.05)
        got, ref = _along(_pressure(case), case.reference_force)
        assert got > 0.0, "pressure force must push the solid away from the fluid"
        err = (got - ref) / abs(ref)
        assert 0.20 < err < 0.32, (builder.__name__, err)


def test_oss_overload_is_insensitive_to_dx():
    errs = []
    for dx in (0.05, 0.025):
        case = C.vertical_wall(dx=dx)
        got, ref = _along(_pressure(case), case.reference_force)
        errs.append((got - ref) / abs(ref))
    assert abs(errs[1] - errs[0]) < 0.01


def test_riemann_term_is_inactive_on_a_static_field():
    case = C.vertical_wall(dx=0.05)
    a = _along(_pressure(case, riemann=False), case.reference_force)[0]
    b = _along(_pressure(case, riemann=True), case.reference_force)[0]
    assert abs(a - b) / abs(a) < 1.0e-12


def test_viscous_branch_drags_the_wall_towards_the_fluid_velocity():
    case = C.couette_shear(dx=0.025)
    f = viscous_force_from_fluid(case.pos_solid, case.vol_solid, case.vel_solid,
                                 case.pos_fluid, case.vol_fluid, case.vel_fluid,
                                 case.mu_fluid, _cfg(case)).sum(dim=0)
    got, ref = _along(f, case.reference_force)
    assert got > 0.0
    # the OSS one-sided form recovers ~62 % of mu U / h and does not converge;
    # pin the measured band instead of pretending it is exact
    assert 0.55 < got / ref < 0.70, got / ref


def test_closed_body_reference_is_archimedes():
    case = C.fsi2_insert_linear(dx=0.1)
    r, bh, bl = C.FSI2["insert_circle_radius"], C.FSI2["bh"], C.FSI2["bl"]
    area = math.pi * r * r + bh * bl
    assert case.reference_force[0] == pytest.approx(0.0)
    assert case.reference_force[1] == pytest.approx(area, rel=1e-12)
    got, ref = _along(_pressure(case), case.reference_force)
    assert got > 0.0, "the closed body must be pushed along -c"


def test_no_neighbours_is_a_zero_load():
    case = C.horizontal_gate(dx=0.05)
    far = case.pos_solid + torch.tensor([100.0, 100.0], dtype=torch.float64)
    f = panel_pressure_force(far, case.vol_solid, case.normal_solid,
                             case.pos_fluid, case.vol_fluid, case.pressure_fluid,
                             _cfg(case))
    assert float(f.abs().sum()) == 0.0


# --------------------------------------------------------------------------- #
# Viscous branch: the OSS truncation deficit and its kernel-corrected repair
# --------------------------------------------------------------------------- #
def _viscous_oss(case):
    return viscous_force_from_fluid(case.pos_solid, case.vol_solid, case.vel_solid,
                                    case.pos_fluid, case.vol_fluid, case.vel_fluid,
                                    case.mu_fluid, _cfg(case))


def _viscous_corrected(case):
    return wall_shear_from_gradient(case.pos_solid, case.vol_solid,
                                    case.normal_solid, case.pos_fluid,
                                    case.vol_fluid, case.vel_fluid, case.mu_fluid,
                                    case.vel_solid, _cfg(case))


def test_continuum_truncation_factor_is_the_wendland_closed_form():
    """One-sided half-space sum on a linear shear returns 7/(3 pi) of mu*gamma.

    ``-2 int_0^R (dW/dr) r dr = -2 alpha_2 int f'(q) q dq = 7/(3 pi)`` for
    Wendland C2, independent of h and of dx -- which is exactly why the measured
    OSS deficit does not shrink under refinement.
    """
    for h in (0.0325, 1.0):
        got = one_sided_viscous_truncation_factor(h, 2, "wendland", spacing=h)
        assert got == pytest.approx(7.0 / (3.0 * math.pi), rel=1e-9)
    # with the lattice spacing s = dx and h = 1.3 dx the prediction is 7/(3 pi 1.3)
    got = one_sided_viscous_truncation_factor(1.3, 2, "wendland", spacing=1.0)
    assert got == pytest.approx(7.0 / (3.0 * math.pi * 1.3), rel=1e-9)


def test_oss_viscous_deficit_is_dx_independent():
    """Pinned: the OSS branch keeps ~62 % of mu U/h at every resolution."""
    ratios = []
    for dx in (0.025, 0.0125):
        case = C.couette_shear(dx=dx)
        got, ref = _along(_viscous_oss(case), case.reference_force)
        ratios.append(got / ref)
    assert abs(ratios[1] - ratios[0]) < 0.006, ratios
    # lattice ratio sits below the continuum 7/(3 pi) because the near-wall
    # shell r < dx/2 carries no particle; the band pins that offset
    case = C.couette_shear(dx=0.025)
    continuum = one_sided_viscous_truncation_factor(case.h, 2, "wendland",
                                                    spacing=case.dx)
    assert 1.00 < ratios[0] / continuum < 1.15, (ratios[0], continuum)


def test_gradient_corrected_arm_is_exact_on_plane_couette():
    """The kernel-corrected reconstruction transfers a linear shear to machine
    precision at every resolution (the 15 % gate is passed by ~12 orders)."""
    for dx in (0.025, 0.0125, 0.00625):
        case = C.couette_shear(dx=dx)
        got, ref = _along(_viscous_corrected(case), case.reference_force)
        assert abs(got - ref) / ref < 1.0e-9, (dx, got, ref)


def test_gradient_corrected_arm_is_zero_for_a_rigid_body_flow():
    case = C.couette_shear(dx=0.025)
    vel = torch.full_like(case.pos_fluid, 0.4)
    f = wall_shear_from_gradient(case.pos_solid, case.vol_solid, case.normal_solid,
                                 case.pos_fluid, case.vol_fluid, vel,
                                 case.mu_fluid, torch.full_like(case.pos_solid, 0.4),
                                 _cfg(case))
    assert float(f.abs().max()) < 1.0e-12


def test_gradient_corrected_arm_converges_on_a_curved_wall():
    """Closed disk in a quadratic shear field: exact load = mu beta A.

    The arm is first-order on a curved one-sided support; require monotone
    convergence and the gate at the resolutions the bundle reports.
    """
    errs = []
    for dx in (0.1, 0.05, 0.025):
        case = C.curved_disk_shear(dx=dx)
        got, ref = _along(_viscous_corrected(case), case.reference_force)
        errs.append(abs(got - ref) / ref)
    assert errs[0] < 0.15 and errs[1] < 0.10 and errs[2] < 0.05, errs
    assert errs[0] > errs[1] > errs[2], errs


def test_gradient_corrected_arm_needs_no_slip_consistent_solid_velocity():
    """Documented pitfall, pinned as a guard: the reconstruction reads
    ``u_fluid - u_solid``, so a solid velocity that is *not* the no-slip value of
    the same field makes the arm return nonsense (the OSS sum degrades too, but
    far less).  This is invalid input, not a variant case."""
    case = C.curved_disk_shear(dx=0.05)
    bad = wall_shear_from_gradient(
        case.pos_solid, case.vol_solid, case.normal_solid, case.pos_fluid,
        case.vol_fluid, case.vel_fluid, case.mu_fluid,
        torch.zeros_like(case.pos_solid), _cfg(case))
    got, ref = _along(bad, case.reference_force)
    assert abs(got - ref) / ref > 1.0, got / ref
