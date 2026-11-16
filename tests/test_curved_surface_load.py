"""Guards for the curved-body near-field load channel (X6/X7).

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The tests pin *properties*, not just execution:

1. degree exactness -- the new quadratic surface reconstruction reproduces a
   quadratic field at station level to round-off **when the neighbour set can
   carry six parameters**, on both a flat wall and a curved (closed-disk) wall;
2. the information limit -- at the coupled FSI2 support ratio (``h/dx = 1.3``,
   ~9 neighbours per station) the same arm is rank deficient / extrapolating and
   is off by >5 %, so it must not be presented as admissible there;
3. quadrature closure -- the discrete surface measure closes exactly, which is
   what lets the net force of a *closed* body be blind to quadratic content
   (the reason the ruling is written in station-level terms);
4. the reference chain -- the analytic 4096-station pressure quadrature agrees
   with Kutta-Joukowski, so the X7 comparison does not rest on one reference;
5. the ruling -- no near-field arm is inside the 15 % gate at the coupled
   spacing, all of them converge under refinement, and the ladder entry points
   quoted in ``docs/xval_fsi2/CURVED_SURFACE_LOAD_ADMISSIBILITY.md`` are the
   coarsest spacings that actually pass.
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

import curved_surface_load_audit as D  # noqa: E402
from tensorsph import fsi2_turek_hron as f2  # noqa: E402
from tensorsph.farfield_force import potential_flow_cylinder  # noqa: E402

DT = torch.float64
COEFFS = (1.0, 0.8, -0.6, 3.0, -2.0, 4.0)


def _stations(kind, h, dx, radius=0.2):
    if kind == "flat_wall":
        wall, nrm, trib = D.flat_wall_stations(-0.5, 0.5, 0.0, dx)
        fluid = D.FFA.lattice(-0.5, 0.5, 0.6 * dx, 6.0 * h, dx)
        return wall, nrm, trib, fluid
    centre = torch.tensor([2.0, 2.0], dtype=DT)
    wall, nrm, trib = D.disk_stations(centre, radius, int(round(2 * math.pi * radius / dx)))
    fluid = D.FFA.annulus_lattice(centre, radius + 0.5 * dx, radius + 5.0 * h, dx)
    return wall, nrm, trib, fluid


@pytest.mark.parametrize("kind", ["flat_wall", "closed_disk"])
def test_quadratic_arm_is_exact_when_well_sampled(kind):
    """Six parameters, ~56 neighbours: round-off exactness on a quadratic field."""
    h, ratio = 0.03, 3.0
    dx = h / ratio
    wall, nrm, trib, fluid = _stations(kind, h, dx)
    p_exact = D.quadratic_field(wall, COEFFS)
    arms = D.run_arms(fluid, wall, nrm, trib, h, D.quadratic_field(fluid, COEFFS))
    scale = float(p_exact.abs().max())

    quad_err = float((arms["quadratic"]["p_wall"] - p_exact).abs().max()) / scale
    plane_err = float((arms["plane"]["p_wall"] - p_exact).abs().max()) / scale
    kernel_err = float((arms["kernel"]["p_wall"] - p_exact).abs().max()) / scale

    assert quad_err < 1.0e-7, f"quadratic arm not exact: {quad_err:.3e}"
    # the lower-degree arms must be *worse*, otherwise the test proves nothing
    assert plane_err > 10.0 * quad_err
    assert kernel_err > 1.0e-3


def test_quadratic_arm_is_information_limited_at_coupled_ratio():
    """At h/dx = 1.3 (the coupled fsi2 support) the arm must not be trusted."""
    h, ratio = 0.03, 1.3
    dx = h / ratio
    wall, nrm, trib, fluid = _stations("flat_wall", h, dx)
    p_exact = D.quadratic_field(wall, COEFFS)
    arms = D.run_arms(fluid, wall, nrm, trib, h, D.quadratic_field(fluid, COEFFS))
    scale = float(p_exact.abs().max())
    err = float((arms["quadratic"]["p_wall"] - p_exact).abs().max()) / scale
    diag = D._design_condition(fluid, wall, h, int(wall.shape[0]))

    assert diag["n_neighbours_median"] <= 12
    assert err > 0.05, "starved quadratic fit unexpectedly exact -- update the doc"
    if diag["cond_median"] > 1.0e12:
        # rank deficiency: the fit is only kept finite by the Tikhonov term
        assert err > 0.05


def test_surface_measure_closes_on_a_closed_body():
    """A purely quadratic field must produce no net force through the quadrature.

    This is the invariant behind ``net force cannot see a degree defect``: the
    quadratic term integrates to zero over the body surface (divergence
    theorem), so the discrete closed-surface quadrature must reproduce that.
    """
    centre = torch.tensor([2.0, 2.0], dtype=DT)
    radius = 0.25
    wall, nrm, trib = D.disk_stations(centre, radius, 4096)
    # the quadratic form must be centred on the body, otherwise its expansion
    # carries a linear term and the net force is *not* zero
    xi = wall - centre[None, :]
    quad_only = D.quadratic_field(xi, (0.0, 0.0, 0.0, 3.0, -2.0, 4.0))
    f = f2.boundary_force_from_pressure(quad_only, nrm, float(trib)).sum(dim=0)
    scale = float(quad_only.abs().max()) * float(trib) * wall.shape[0]
    assert float(quad_only.abs().max()) > 0.01          # the field is not trivial
    assert float(f.norm()) / scale < 1.0e-12


def test_potential_flow_reference_chain_matches_kutta_joukowski():
    """The analytic pressure integrated on a fine circle equals -rho U Gamma."""
    a, u_inf, rho, circ_f = 0.05, 0.2, 1000.0, 0.8
    circulation = 2.0 * math.pi * a * u_inf * circ_f
    centre = torch.zeros(2, dtype=DT)
    fine, nrm, trib = D.disk_stations(centre, a, 4096)
    _, p = potential_flow_cylinder(fine, a, u_inf, circulation, rho, centre=centre)
    f = f2.boundary_force_from_pressure(p, nrm, float(trib)).sum(dim=0)
    kj = -rho * u_inf * circulation
    assert abs(float(f[1]) - kj) / abs(kj) < 1.0e-9       # lift
    assert abs(float(f[0])) / abs(kj) < 1.0e-9            # d'Alembert drag


@pytest.mark.parametrize("shell", [3.0, 5.0])
def test_ruling_no_nearfield_arm_admissible_at_coupled_spacing(shell):
    """Coupled spacing fails the 15 % gate for every arm; refinement helps."""
    dx_list = (0.02, 0.005)
    rows = [r for r in D.gate_x7_ladder(dx_list, shell_list=(shell,))]
    coupled = max(r["dx"] for r in rows)
    fine = min(r["dx"] for r in rows)
    rul = D.ruling(rows)

    assert rul["coupled_dx"] == coupled
    for arm in ("kernel", "plane", "quadratic"):
        worst = max(abs(r[f"{arm}_lift_rel_err"]) for r in rows if r["dx"] == coupled)
        assert worst > 0.15, f"{arm} passes 15 % at the coupled spacing -- re-rule"
        # monotone improvement under refinement (this is what the ladder claims)
        coarse_err = next(abs(r[f"{arm}_lift_rel_err"]) for r in rows if r["dx"] == coupled)
        fine_err = next(abs(r[f"{arm}_lift_rel_err"]) for r in rows if r["dx"] == fine)
        assert fine_err < coarse_err

    # the coarsest passing spacing, not the first one found
    plane_adh = [r for r in rows if abs(r["plane_lift_rel_err"]) <= D.GATE_15]
    worst_pass = max(r["dx"] for r in plane_adh) if plane_adh else None
    if worst_pass is not None:
        assert rul["arms"]["plane"]["gate_entry"][f"{shell:g}h"]["first_dx_gate_15"] == worst_pass


def test_starved_station_stays_finite():
    """One neighbour cannot carry six parameters: fall back, do not emit NaN."""
    h = 0.03
    wall = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=DT)
    nrm = torch.tensor([[0.0, 1.0]] * 3, dtype=DT)
    trib = 1.0
    # station 0: 3 neighbours (too few for six parameters); station 1: plenty;
    # station 2: none at all
    fluid = torch.tensor([[0.0, 0.02], [0.03, 0.02], [-0.03, 0.03],
                          [1.0, 0.02], [1.03, 0.02], [0.97, 0.03], [1.0, 0.05],
                          [1.04, 0.05], [0.96, 0.055]], dtype=DT)
    p_all = torch.tensor([7.0] * 3 + [1.0] * 6, dtype=DT)
    arms = D.run_arms(fluid, wall, nrm, trib, h, p_all)
    p = arms["quadratic"]["p_wall"]
    assert torch.isfinite(p).all()
    # too few samples -> kernel-weighted mean (all three samples are 7.0)
    assert float(p[0]) == pytest.approx(7.0)
    # no support at all -> zero face pressure (documented convention)
    assert float(p[2]) == pytest.approx(0.0)