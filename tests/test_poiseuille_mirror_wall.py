"""Mirror (ghost) wall boundary condition and the Poiseuille benchmark on it.

Two kinds of test:

* exact, simulation-free checks of :mod:`tensorsph.wall_mirror` -- the mirror
  geometry, the velocity reflection, and the property that makes the boundary
  condition work: a fluid/ghost pair straddling the plane has zero *mean*
  velocity, so the discrete viscous operator sees ``u = 0`` on the plane;
* one coarse but real channel simulation, which pins the accuracy claim in
  ``docs/verification_poiseuille_mirror_wall.md`` (centre-line and profile
  within 5 % of the closed-form parabola) so a change to the viscous term or to
  the ghost layer cannot pass unnoticed.
"""

import math

import pytest
import torch

from benchmarks.poiseuille.run import make_config
from benchmarks.poiseuille_mirror.run import (
    check_analytic_symbolically,
    evaluate,
    gates,
    run_case_mirror,
)
from tensorsph.wall_mirror import mirror_plane, plane_walls

# Same pin as the benchmark drivers: these tensors are small enough that the
# default thread pool costs more than it saves (13x on this platform).
torch.set_num_threads(1)

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
RHO0 = 1000.0


def _cloud(points, vel=None):
    pos = torch.tensor(points, dtype=torch.float64)
    vel = (torch.zeros_like(pos) if vel is None
           else torch.tensor(vel, dtype=torch.float64))
    mass = torch.full((pos.shape[0],), RHO0 * 0.01 ** 2, dtype=torch.float64)
    return pos, vel, mass


# --------------------------------------------------------------------------- #
# Geometry and velocity reflection
# --------------------------------------------------------------------------- #

def test_ghosts_are_the_mirror_image_and_only_from_the_support_band():
    pos, vel, mass = _cloud([[0.0, 0.005], [0.0, 0.02], [0.0, 0.05]],
                            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    gp, gv, gm = mirror_plane(pos, vel, mass, 0.0, support=0.03, side=+1)
    # Only the two rows within 0.03 of the plane are mirrored.
    assert gp.shape[0] == 2
    assert torch.allclose(gp[:, 1], torch.tensor([-0.005, -0.02], dtype=torch.float64))
    assert torch.allclose(gp[:, 0], pos[:2, 0])


def test_no_slip_ghost_velocity_is_negated():
    pos, vel, mass = _cloud([[0.0, 0.01]], [[2.0, 0.5]])
    _, gv, _ = mirror_plane(pos, vel, mass, 0.0, support=0.05, side=+1,
                            slip="no_slip")
    assert torch.allclose(gv, torch.tensor([[-2.0, -0.5]], dtype=torch.float64))


def test_free_slip_ghost_velocity_reverses_only_the_normal_component():
    pos, vel, mass = _cloud([[0.0, 0.01]], [[2.0, 0.5]])
    _, gv, _ = mirror_plane(pos, vel, mass, 0.0, support=0.05, side=+1,
                            slip="free_slip")
    assert torch.allclose(gv, torch.tensor([[2.0, -0.5]], dtype=torch.float64))


def test_a_straddling_pair_has_zero_mean_tangential_velocity_on_the_plane():
    """The defining property of the no-slip mirror: (u_fluid + u_ghost)/2 = 0."""
    y = 0.0123
    pos, vel, mass = _cloud([[0.0, y]], [[1.7, 0.0]])
    gp, gv, _ = mirror_plane(pos, vel, mass, 0.0, support=0.05, side=+1,
                             slip="no_slip")
    mid = 0.5 * (pos[0] + gp[0])
    assert abs(float(mid[1])) < 1e-15          # pair midpoint sits on the plane
    assert abs(float(0.5 * (vel[0, 0] + gv[0, 0]))) < 1e-15


def test_the_top_wall_mirrors_downwards():
    pos, vel, mass = _cloud([[0.0, HEIGHT - 0.01]], [[1.0, 0.2]])
    gp, gv, _ = mirror_plane(pos, vel, mass, HEIGHT, support=0.05, side=-1)
    assert gp.shape[0] == 1
    assert math.isclose(float(gp[0, 1]), HEIGHT + 0.01, rel_tol=0, abs_tol=1e-12)


def test_plane_walls_stack_both_walls_and_drop_particles_outside():
    pos, vel, mass = _cloud([[0.0, 0.01], [0.0, HEIGHT - 0.01], [0.0, 0.05]])
    gp, gv, gm = plane_walls(pos, vel, mass, [(0.0, +1), (HEIGHT, -1)],
                             support=0.03)
    # Middle row is out of both support bands.
    assert gp.shape[0] == 2
    assert math.isclose(float(gp[0, 1]), -0.01, abs_tol=1e-12)
    assert math.isclose(float(gp[1, 1]), HEIGHT + 0.01, abs_tol=1e-12)


def test_an_empty_band_returns_empty_tensors_of_the_right_shape():
    pos, vel, mass = _cloud([[0.0, 0.05]])
    gp, gv, gm = mirror_plane(pos, vel, mass, 0.0, support=0.01, side=+1)
    assert gp.shape == (0, 2) and gv.shape == (0, 2) and gm.shape == (0,)


def test_unknown_slip_and_bad_side_are_rejected():
    pos, vel, mass = _cloud([[0.0, 0.01]])
    with pytest.raises(ValueError):
        mirror_plane(pos, vel, mass, 0.0, 0.05, slip="slip-ish")
    with pytest.raises(ValueError):
        mirror_plane(pos, vel, mass, 0.0, 0.05, side=0)


# --------------------------------------------------------------------------- #
# Reference solution provenance
# --------------------------------------------------------------------------- #

def test_the_closed_form_satisfies_the_navier_stokes_reduction():
    """u = g (H-y) y / (2 nu) must zero nu u'' + g and both boundary conditions."""
    prov = check_analytic_symbolically()
    if not prov.get("available"):
        pytest.skip(f"sympy unavailable: {prov.get('reason')}")
    assert prov["residual_is_zero"]
    assert prov["u_max_matches_gH2_over_8nu"]
    assert prov["tau_w_matches_gH_over_2"]
    assert prov["bc_at_y0"] == "0" and prov["bc_at_yH"] == "0"


# --------------------------------------------------------------------------- #
# Real channel simulation
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.timeout(300)
def test_mirror_wall_channel_matches_the_parabola_within_5_percent():
    device = torch.device("cpu")
    resolution = 8
    dx = HEIGHT / resolution
    cfg = make_config(dx, VISCOSITY, 1.0, device)
    length = 12 * dx
    case = run_case_mirror(cfg, HEIGHT, length, BODY_FORCE, t_end=1.0,
                           t_average=0.2, device=device)
    res = evaluate(case, HEIGHT, BODY_FORCE, VISCOSITY, RHO0, resolution)
    g = gates(res)

    assert g["A_centreline_5pct"], f"u_max error {res['u_max_error_pct']:.2f}%"
    assert g["B_profile_rms_5pct"], \
        f"profile RMS {res['profile_rms_interior_pct']:.2f}%"
    assert g["C_wall_shear_5pct"], (
        f"wall shear {res['wall_shear_bottom_error_pct']:.2f}% / "
        f"{res['wall_shear_top_error_pct']:.2f}%")
    # The profile is a parabola, not just close at the peak.
    assert res["ode_residual_max_over_g"] < 0.05
