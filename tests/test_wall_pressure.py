"""The wall pressure extrapolation is Adami et al. (2012) eq. (11), complete.

The equation has two terms:

    p_w = ( sum_f p_f W_wf + sum_f rho_f (g - a_w).(r_w - r_f) W_wf ) / sum_f W_wf

An earlier version implemented only the first one, which makes the wall pressure
the *average* of the neighbouring fluid pressure instead of its extrapolation to
the wall face. At the bottom of a pool that is low by roughly `rho g dx / 2`, so
the fluid near the floor is under-supported, and a resting pool rings.

These tests pin both terms against a closed form on a two-particle system: one
fluid particle one spacing above one wall particle, with nothing else in range.
With a single neighbour the kernel weights cancel and the wall pressure is

    p_w = p_f + rho_f g dy,      dy = y_w - y_f = -dx

so the fluid acceleration is known analytically.
"""
import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_gradient,
    cubic_spline_kernel,
    rhs,
    tait_pressure,
)

SMALL = dict(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
             drop_center_y=0.25, device="cpu")


def two_particle_state(cfg, mass_scale):
    dx = cfg.dx
    pos = torch.tensor([[0.5 * dx, 0.5 * dx]])
    wall = torch.tensor([[0.5 * dx, -0.5 * dx]])
    mass = torch.tensor([mass_scale * cfg.rho0 * dx * dx])
    return pos, wall, mass


def closed_form(cfg, mass_scale):
    """Wall pressure and fluid acceleration for the two-particle state."""
    dx, h = cfg.dx, cfg.h_factor * cfg.dx
    pos, wall, mass = two_particle_state(cfg, mass_scale)
    separation = pos[0] - wall[0]
    distance = torch.linalg.vector_norm(separation)
    wall_mass = cfg.rho0 * dx * dx
    rho = (mass[0] * cubic_spline_kernel(torch.zeros(()), h)
           + wall_mass * cubic_spline_kernel(distance, h))
    pressure = tait_pressure(rho.reshape(1), cfg)[0]
    # (g - a_w).(r_w - r_f) with g = (0, -g), a_w = 0 and r_w - r_f = -separation:
    # (0, -g).(-separation) = +g * separation_y, and separation_y = y_f - y_w = dx
    # for a wall particle below the fluid. So the term is +rho g dx.
    wall_pressure = (pressure + cfg.gravity * separation[1] * rho).clamp_min(0.0)
    bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
    wall_density = (cfg.rho0 * (wall_pressure / bulk + 1.0).pow(1.0 / cfg.gamma)
                    ).clamp_min(cfg.rho0)
    force = (-wall_mass * (pressure / rho.square() + wall_pressure / wall_density.square())
             * cubic_spline_gradient(separation, h))
    acceleration = torch.stack((torch.zeros_like(force[0]),
                                torch.full_like(force[0], -cfg.gravity))) + force
    return wall_pressure, acceleration


@pytest.mark.parametrize("mass_scale", [0.8, 3.0])
def test_wall_pressure_matches_the_published_extrapolation(mass_scale):
    """Both terms of eq. (11): the kernel average and the body-force term.

    ``mass_scale = 0.8`` keeps the fluid particle below ``rho0`` so ``p_f = 0``
    and only the body-force term survives; ``3.0`` puts it above ``rho0`` so the
    two terms are exercised together.
    """
    cfg = Config(**SMALL)
    pos, wall, mass = two_particle_state(cfg, mass_scale)
    _, pressure, acceleration = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)

    wall_pressure, expected = closed_form(cfg, mass_scale)
    assert wall_pressure > 0.0
    # The wall below the fluid is at a higher pressure than the fluid above it.
    assert float(wall_pressure) > float(pressure[0])
    assert torch.allclose(acceleration[0], expected, rtol=1e-5, atol=1e-6)


def test_the_body_force_term_vanishes_without_gravity():
    """No gravity, no body force, and the kernel average of zero is zero.

    ``mass_scale = 0.8`` leaves the fluid particle below ``rho0``, so ``p_f = 0``
    and the kernel-average term contributes nothing: every newton of wall force
    seen with gravity comes from the body-force term alone.
    """
    cfg = Config(**SMALL)
    weightless = Config(**SMALL, gravity=0.0)

    pos, wall, mass = two_particle_state(cfg, 0.8)
    _, _, accel = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
    _, _, accel_free = rhs(pos, torch.zeros_like(pos), mass, weightless, wall=wall)
    assert float(accel_free.abs().max()) == 0.0
    # The wall below supports part of the weight: |a| is smaller than g.
    assert float(accel[0, 1]) > -cfg.gravity
    assert float(accel[0, 0]) == 0.0
