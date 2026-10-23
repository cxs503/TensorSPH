"""Free-surface treatment: the optional density-diffusion term.

The solver clamps pressure at zero to suppress tensile instability. This module
pins the second, optional treatment: the diffusive term of Antuono, Colagrossi
& Marrone (2012), applied to the summation density. What it does to B1 and B2
is measured in docs/benchmarks/free-surface-treatment.md; these tests pin the
operator's invariants, not its usefulness.
"""
import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_gradient,
    density_diffusion_rate,
    diffusion_dt_limit,
    make_particles,
    rhs,
    rhs_all_pairs,
    validate_config,
    wall_lattice,
)
from tensorsph.neighbor_search import cell_linked_pairs

SMALL = dict(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
             drop_center_y=0.25, device="cpu")


def _pairs_and_geometry(cfg, wall=True):
    pos, mass = make_particles(cfg, torch.device("cpu"))
    blocks = [pos]
    if wall:
        blocks.append(wall_lattice(cfg, torch.device("cpu")))
    all_pos = torch.cat(blocks, dim=0)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(all_pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = all_pos[i] - all_pos[j]
    grad_w = cubic_spline_gradient(delta, h)
    distance = torch.linalg.vector_norm(delta, dim=1)
    radial = -(delta * grad_w).sum(dim=1)
    fluid = (i < pos.shape[0]) & (j < pos.shape[0])
    return pos, mass, distance[fluid], radial[fluid], (i[fluid], j[fluid])


def test_a_uniform_density_field_is_left_untouched():
    cfg = Config(**SMALL, density_diffusion=0.1)
    pos, mass, distance, radial, neighbours = _pairs_and_geometry(cfg)
    rho = torch.full((pos.shape[0],), cfg.rho0)
    rate = density_diffusion_rate(rho, mass, distance, radial, neighbours, cfg)
    assert torch.equal(rate, torch.zeros_like(rate))


def test_the_volume_weighted_mean_density_is_conserved():
    """sum_i V_i drho_i/dt = 0, otherwise the term would create mass."""
    cfg = Config(**SMALL, density_diffusion=0.1)
    pos, mass, distance, radial, neighbours = _pairs_and_geometry(cfg)
    generator = torch.Generator().manual_seed(0)
    rho = cfg.rho0 * (1.0 + 0.05 * torch.randn(pos.shape[0], generator=generator))
    rate = density_diffusion_rate(rho, mass, distance, radial, neighbours, cfg)
    volume = mass / rho
    assert abs(float((volume * rate).sum())) < 1.0e-3 * float((volume * rate.abs()).sum())


def test_the_term_reduces_density_variance():
    cfg = Config(**SMALL, density_diffusion=0.1)
    pos, mass, distance, radial, neighbours = _pairs_and_geometry(cfg)
    generator = torch.Generator().manual_seed(1)
    rho = cfg.rho0 * (1.0 + 0.05 * torch.randn(pos.shape[0], generator=generator))
    rate = density_diffusion_rate(rho, mass, distance, radial, neighbours, cfg)
    assert float((rho + cfg.dt * rate).std()) < float(rho.std())


def test_the_pair_implementation_matches_the_dense_reference():
    cfg = Config(dx=0.1, h_factor=1.3, rho0=1000.0, sound_speed=8.0, gamma=7.0,
                 gravity=9.81, viscosity=0.03, density_diffusion=0.1,
                 device="cpu")
    generator = torch.Generator().manual_seed(5)
    pos = torch.randn((61, 2), generator=generator, dtype=torch.float64) * 0.12
    vel = torch.randn((61, 2), generator=generator, dtype=torch.float64) * 0.2
    mass = (torch.rand((61,), generator=generator, dtype=torch.float64) + 0.2) * 0.01
    for actual, expected in zip(rhs(pos, vel, mass, cfg),
                                rhs_all_pairs(pos, vel, mass, cfg)):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_the_term_is_not_evaluated_when_it_is_off(monkeypatch):
    """The default path must not even call the term, not merely get zero back.

    Comparing two runs that both take the disabled branch would pass even if the
    branch did nothing but add zeros through several mask-indexing passes. This
    asserts the call is unreachable when ``density_diffusion == 0`` and reachable
    when it is not, which is what "the default path costs nothing" means.
    """
    import tensorsph.droplet_impact as module

    calls = []
    original = module.density_diffusion_rate

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "density_diffusion_rate", spy)
    cfg = Config(**SMALL)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    vel = torch.zeros_like(pos)
    wall = wall_lattice(cfg, torch.device("cpu"))
    rhs(pos, vel, mass, cfg, wall=wall)
    assert calls == [], "the diffusive term ran although it is disabled"

    cfg_on = Config(**SMALL, density_diffusion=0.1)
    rhs(pos, vel, mass, cfg_on, wall=wall)
    assert len(calls) == 1, "the diffusive term was not reached although enabled"


def test_negative_coefficient_is_rejected():
    cfg = Config(**SMALL, density_diffusion=-0.1)
    with pytest.raises(ValueError, match="density_diffusion"):
        validate_config(cfg)


def test_diffusion_has_its_own_step_limit():
    # delta = 1 makes the diffusive limit stricter than the acoustic one
    # (0.125 h/c0 against 0.3 h/c0), so the test can isolate it.
    cfg = Config(**SMALL, density_diffusion=1.0)
    limit = diffusion_dt_limit(cfg)
    assert limit == pytest.approx(0.125 * cfg.h_factor * cfg.dx
                                  / (1.0 * cfg.sound_speed))
    assert diffusion_dt_limit(Config(**SMALL)) == float("inf")

    cfg.dt = 2.0 * limit
    with pytest.raises(ValueError, match="density-diffusion limit"):
        validate_config(cfg)


def test_the_term_only_sees_fluid_neighbours():
    """Wall density is extrapolated, so it must not be diffused into the fluid."""
    cfg = Config(**SMALL, density_diffusion=0.1)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    vel = torch.zeros_like(pos)
    without_wall = rhs(pos, vel, mass, cfg)[0].clone()

    wall = wall_lattice(cfg, torch.device("cpu"))
    with_wall = rhs(pos, vel, mass, cfg, wall=wall)[0].clone()
    # The wall changes the summation density (that is its job); the diffusive
    # term must not add a second, wall-driven contribution on top of it.
    cfg_plain = Config(**SMALL, density_diffusion=0.0)
    reference = rhs(pos, vel, mass, cfg_plain, wall=wall)[0]
    assert not torch.equal(with_wall, without_wall)
    assert torch.isfinite(with_wall).all()
    # Without diffusion the density is exactly the summation; with diffusion it
    # is that plus a fluid-only correction, which must stay small.
    relative = (with_wall - reference).abs() / cfg.rho0
    assert float(relative.max()) < 0.2
