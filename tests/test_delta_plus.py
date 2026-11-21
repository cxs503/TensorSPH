from pathlib import Path
import runpy

import pytest
import torch

from tensorsph.delta_plus import delta_plus_step
from tensorsph.neighbor_search import brute_force_pairs

DX = .01
PARAMS = dict(dx=DX, sound_speed=20., reference_speed=.5, dt=.0001, gravity=0.)
DTYPE = torch.float64


def lattice(n=15):
    axis = (torch.arange(n, dtype=DTYPE) + .5)*DX
    pos = torch.cartesian_prod(axis, axis)
    rho = torch.full_like(pos[:, 0], 1000.)
    return pos, torch.zeros_like(pos), rho, rho*DX**2


def test_continuity_keeps_evolved_density_in_free_flight():
    # An isolated translating parcel has no interacting pairs or density summation.
    pos, vel, rho, mass = lattice(2)
    pos, vel, rho, mass = pos[:1], vel[:1]+.25, rho[:1]+3., mass[:1]
    for _ in range(3):
        result = delta_plus_step(pos, vel, rho, mass, **PARAMS)
        torch.testing.assert_close(result.position, pos+vel*PARAMS["dt"], atol=1e-15, rtol=0)
        torch.testing.assert_close(result.density, rho)
        assert torch.count_nonzero(result.shift) == 0
        pos, vel, rho = result.position, result.velocity, result.density


def test_sun_bulk_shift_and_surface_projection_use_distinct_directions():
    pos, vel, rho, mass = lattice()
    middle = 7*15+7
    pos[middle, 0] += .23*DX
    # Asymmetric spacing one row below a straight, y-normal free surface.
    top = 7*15+13
    pos[top, 0] -= .2*DX
    step = delta_plus_step(pos, vel, rho, mass, **PARAMS)
    # Independent direct sum of eq. (5) for the interior particle (h = 2 dx).
    x = pos[middle]-pos
    h = 2*DX
    q = x.norm(dim=1)/h
    t = (1-q/2).clamp_min(0)
    sigma = 7/(4*torch.pi*h*h)
    weight = sigma*t**4*(2*q+1)
    w_dx = sigma*(1-.25)**4*2
    gradient = (-5*sigma/h**2*t**3)[:, None]*x
    raw = ((1+.2*(weight/w_dx)**4)*mass/(rho+rho[middle]))[:, None]*gradient
    expected = -4*h*PARAMS["dt"]*PARAMS["reference_speed"]*raw.sum(0)
    torch.testing.assert_close(step.shift[middle], expected, rtol=2e-12, atol=1e-15)
    assert step.shift[middle, 0] < 0
    assert step.shift[top, 0] > 0
    assert step.shift[top, 1].abs() < .10*step.shift[top, 0]
    assert step.shift[0].norm() == 0
    assert torch.count_nonzero(step.velocity) == 0


def test_clipped_dt_scales_shift_and_skin_pairs_do_not_change_interface():
    pos, vel, rho, mass = lattice()
    pos[99] += DX*.2
    a = delta_plus_step(pos, vel, rho, mass, **PARAMS)
    opts = {**PARAMS, "dt": PARAMS["dt"]/2}
    pairs = brute_force_pairs(pos, 5.1*DX)
    b = delta_plus_step(pos, vel, rho, mass, pairs=pairs, **opts)
    torch.testing.assert_close(b.shift*2, a.shift, atol=1e-14, rtol=1e-12)


def test_density_diffusion_preserves_affine_profile_and_damps_grid_noise():
    pos, vel, rho, _ = lattice()
    options = {**PARAMS, "reference_speed": 0.}
    for affine in (True, False):
        perturbation = 3*(pos[:, 1]-.1) if affine else .15*(-1.)**torch.arange(len(pos))
        rho_i = (rho+perturbation).to(DTYPE)
        mass = rho_i*DX**2
        a = delta_plus_step(pos, vel, rho_i, mass, diffusion=.1, **options)
        b = delta_plus_step(pos, vel, rho_i, mass, diffusion=0., **options)
        if affine:
            torch.testing.assert_close(a.density, b.density, atol=1e-12, rtol=0)
        else:
            assert a.density.std() < b.density.std()
            diff = a.density-b.density
            assert float(diff.sum().abs()) < 1e-10


def test_moving_wall_compresses_fluid_and_receives_equal_opposite_force():
    pos, vel, rho, mass = lattice(8)
    rho += 1.5
    wall = pos[:8] - pos.new_tensor([DX, 0.])
    wall_velocity = torch.zeros_like(wall)
    wall_velocity[:, 0] = .4
    normal = torch.zeros_like(wall)
    normal[:, 0] = 1
    moving = delta_plus_step(pos, vel, rho, mass, wall=wall,
                             wall_velocity=wall_velocity, wall_normal=normal, **PARAMS)
    fixed = delta_plus_step(pos, vel, rho, mass, wall=wall, wall_normal=normal, **PARAMS)
    assert torch.all(moving.density[:8] > fixed.density[:8])
    slide = delta_plus_step(pos, vel, rho, mass, wall=wall, wall_normal=normal,
                            wall_velocity=wall_velocity.flip(1), **PARAMS)
    torch.testing.assert_close(slide.density, fixed.density)
    assert moving.wall_reaction.sum(0)[0] < 0
    residual = (mass[:, None]*moving.acceleration).sum(0) + moving.wall_reaction.sum(0)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)


def test_unsupported_timestep_is_rejected():
    pos, vel, rho, mass = lattice(3)
    with pytest.raises(ValueError, match="time step"):
        delta_plus_step(pos, vel, rho, mass, **{**PARAMS, "dt": .01})


def test_slip_wall_pst_stays_tangential_and_contact_line_is_pinned():
    from tensorsph.delta_plus import _wall_tangent

    raw = torch.tensor([[3., 1.], [0., 2.], [1., 1.], [-1., 3.]], dtype=DTYPE)
    normal = torch.tensor([[0., 1.], [1., 0.], [0., 1.], [1., 0.]], dtype=DTYPE)
    band = torch.tensor([False, True, False, False])
    # Tilted wall, intersecting free surface, two tied corner normals, open bulk.
    f = torch.tensor([0, 1, 2, 2])
    wall_n = torch.tensor([[1., 1.], [1., 1.], [0., 1.], [1., 0.]], dtype=DTYPE)
    wall_n /= wall_n.norm(dim=1, keepdim=True)
    applied = _wall_tangent(raw, normal, band, f, torch.ones(4, dtype=DTYPE), wall_n)
    expected = raw.new_tensor([[1., -1.], [0., 0.], [0., 0.], [-1., 3.]])
    torch.testing.assert_close(applied, expected)


def test_elliptical_drop_against_incompressible_free_surface_solution():
    path = Path(__file__).resolve().parents[1] / "benchmarks/delta_plus/elliptical_drop.py"
    run = runpy.run_path(str(path))["run"]
    result = run(8)
    assert max(map(abs, result["moment_relative_errors"])) < .01
    assert result["peak_relative_density_deviation"] < .01 and result["peak_mach"] < .1
    assert result["final_momentum_norm"] < 1e-9


def test_shifting_reduces_bulk_spacing_disorder_without_imparting_velocity():
    from tensorsph.neighbor_search import PairListCache

    pos, vel, rho, mass = lattice()
    bulk = ((pos > 4*DX) & (pos < 11*DX)).all(dim=1)
    jitter = torch.rand(pos.shape, generator=torch.Generator().manual_seed(19), dtype=pos.dtype)
    pos += .44*DX*(jitter-.5)

    def spacing_error(points):
        distances = torch.cdist(points, points)
        distances.fill_diagonal_(torch.inf)
        return ((distances[bulk].amin(1)/DX-1).square().mean()).sqrt().item()

    initial = spacing_error(pos)
    cache = PairListCache(4*DX, .4*DX)
    for _ in range(80):
        pairs, _ = cache.get(pos)
        step = delta_plus_step(pos, vel, rho, mass, pairs=pairs, **PARAMS)
        pos, vel, rho = step.position, step.velocity, step.density
    assert spacing_error(pos) < .8*initial
    assert torch.count_nonzero(vel) == 0
    assert (rho == 1000).all()
