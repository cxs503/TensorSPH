"""The 3-D diffusive density term: invariants, defaults and the hydrostatic property."""
from __future__ import annotations

import pytest
import torch

from tensorsph.density_diffusion import molteni_colagrossi_rate, rate_from_positions
from tensorsph.threed import accelerations, cubic_lattice, summation_density
from tensorsph.threed.neighbors3 import box_pairs
from tensorsph.threed.operators3 import (
    density_diffusion_rate_3d,
    pair_geometry,
)

RHO0 = 1000.0
DX = 0.02
H = 1.3 * DX


def _lattice_block(nz: int = 6, nxy: int = 5):
    """Fluid block on a lattice plus three wall rows below it (no periodic wrap)."""
    fluid = cubic_lattice((nxy, nxy, nz), DX, origin=(0.0, 0.0, 0.0))
    wall = []
    for k in range(1, 4):
        layer = cubic_lattice((nxy, nxy, 1), DX, origin=(0.0, 0.0, -k * DX))
        wall.append(layer)
    pos = torch.cat([fluid] + wall)
    n_fluid = fluid.shape[0]
    mass = torch.full((pos.shape[0],), RHO0 * DX ** 3, dtype=torch.float64)
    pairs = box_pairs(pos, 2.0 * H)
    return pos, mass, pairs, n_fluid


def test_zero_delta_is_exactly_zero():
    pos, mass, pairs, n_fluid = _lattice_block()
    rho = torch.full((pos.shape[0],), RHO0, dtype=torch.float64)
    rate = density_diffusion_rate_3d(pos, rho, mass, H, pairs,
                                     sound_speed=20.0, delta=0.0,
                                     n_fluid=n_fluid)
    assert torch.equal(rate, torch.zeros_like(rate))


def test_rate_conserves_volume_weighted_mean_density():
    """The pairwise split makes ``sum_i V_i drho_i/dt`` vanish to round-off."""
    pos, mass, pairs, n_fluid = _lattice_block()
    z = pos[:, 2]
    rho = RHO0 + 0.5 * z + 1e-3 * torch.sin(z / DX * 7.0)
    rate = density_diffusion_rate_3d(pos, rho, mass, H, pairs,
                                     sound_speed=20.0, delta=0.1,
                                     n_fluid=n_fluid)
    volume = mass / rho
    total = (volume * rate).sum()
    scale = (volume * rate.abs()).sum()
    assert torch.isfinite(rate).all()
    assert abs(float(total)) <= 1e-9 * float(scale)


def test_wall_rows_are_not_diffused():
    pos, mass, pairs, n_fluid = _lattice_block()
    z = pos[:, 2]
    rho = RHO0 + 0.5 * (z - z.min())
    rate = density_diffusion_rate_3d(pos, rho, mass, H, pairs,
                                     sound_speed=20.0, delta=0.1,
                                     n_fluid=n_fluid)
    assert torch.equal(rate[n_fluid:], torch.zeros_like(rate[n_fluid:]))


def test_hydrostatic_form_is_small_on_a_truncated_linear_profile():
    """Antuono/Fourtakas ``psi`` form ~ 0 for a linear density field, Molteni not.

    This is the property that motivates the default form: a resting hydrostatic
    column (linear density, free surface, wall) must carry no diffusive flux.
    """
    pos, mass, pairs, n_fluid = _lattice_block()
    z = pos[:, 2]
    rho = RHO0 + 2.0 * (z - z.min())          # exactly linear in z
    pair_delta, _, grad, _ = pair_geometry(pos, pairs, H)
    i, j = pairs[:, 0], pairs[:, 1]
    mask = (i < n_fluid) & (j < n_fluid)
    i, j = i[mask], j[mask]
    pair_delta, grad = pair_delta[mask], grad[mask]
    distance_sq = pair_delta.square().sum(dim=1)
    radial = -(pair_delta * grad).sum(dim=1)
    scale = 2.0 * 0.1 * 20.0 * H
    molteni = molteni_colagrossi_rate(rho, mass, radial, distance_sq, (i, j),
                                      scale)
    antuono = rate_from_positions(pos, rho, mass, pair_delta, grad, radial,
                                  distance_sq, (i, j), scale, form="antuono")
    m = float(molteni[:n_fluid].abs().max())
    a = float(antuono[:n_fluid].abs().max())
    assert m > 0.0, "the control form must produce a non-zero flux here"
    assert a < 0.1 * m, f"antuono flux {a:g} is not << molteni {m:g}"


def test_accelerations_default_is_unchanged():
    """``density_diffusion = 0`` must reproduce the pre-change right-hand side."""
    pos, mass, pairs, n_fluid = _lattice_block()
    z = pos[:, 2]
    rho = RHO0 + 0.5 * z
    vel = torch.zeros_like(pos)
    kwargs = dict(rho0=RHO0, sound_speed=20.0, gamma=7.0, viscosity=0.0,
                  gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
                  n_fluid=n_fluid, wall_model="adami")
    acc_ref, rho_ref, p_ref = accelerations(pos, vel, mass, H, pairs, **kwargs)
    acc_new, rho_new, p_new = accelerations(pos, vel, mass, H, pairs,
                                            density_diffusion=0.0, dt=1e-4,
                                            **kwargs)
    assert torch.equal(acc_ref, acc_new)
    assert torch.equal(rho_ref, rho_new)
    assert torch.equal(p_ref, p_new)


def test_delta_without_dt_is_rejected():
    pos, mass, pairs, n_fluid = _lattice_block()
    vel = torch.zeros_like(pos)
    with pytest.raises(ValueError):
        accelerations(pos, vel, mass, H, pairs, rho0=RHO0, sound_speed=20.0,
                      gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
                      n_fluid=n_fluid, density_diffusion=0.1)
