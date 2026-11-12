"""Invariant tests for :mod:`tensorsph.particle_shifting` (Lind et al. 2012).

Every assertion here is an invariant that can be derived from the scheme, not a
record of what the implementation happens to produce:

* a perfect lattice is a *fixed point* (``grad C = 0`` by symmetry);
* the shift moves positions only, so mass and velocity are untouched;
* the shift stays under the ``limit * h`` cap;
* at a free surface the shift has no normal component;
* Fickian diffusion is *monotone*: an irregular packing relaxes towards
  uniformity under repeated passes (``RMS |grad C| h`` must fall);
* the operator is deterministic and free of RNG.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.particle_shifting import (concentration,
                                         concentration_gradient,
                                         estimate_reference_concentration,
                                         shift_positions)


def square_lattice(n: int = 21, dx: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(n, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    volume = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    return pos, volume


def hex_lattice(n: int = 21, dx: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for j in range(n):
        y = j * dx * math.sqrt(3.0) / 2.0
        x0 = 0.5 * dx * (j % 2)
        rows.append(torch.stack((x0 + torch.arange(n, dtype=torch.float64) * dx,
                                 torch.full((n,), y, dtype=torch.float64)), dim=-1))
    pos = torch.cat(rows, dim=0)
    volume = torch.full((pos.shape[0],), dx * dx * math.sqrt(3.0) / 2.0,
                        dtype=torch.float64)
    return pos, volume


def interior_mask(pos: torch.Tensor, margin: float) -> torch.Tensor:
    lo = pos.min(dim=0).values + margin
    hi = pos.max(dim=0).values - margin
    return ((pos > lo) & (pos < hi)).all(dim=1)


def disorder(pos: torch.Tensor, volume: torch.Tensor, h: float) -> float:
    """Dimensionless lattice disorder ``RMS(|grad C|) * h``."""
    c = concentration(pos, volume, h)
    g = concentration_gradient(pos, volume, c, h)
    return float(torch.sqrt(torch.mean(g.square())) * h)


H_FACTOR = 1.3


@pytest.mark.parametrize("builder", [square_lattice, hex_lattice],
                         ids=["square", "hex"])
def test_perfect_lattice_is_a_fixed_point(builder):
    """``grad C = 0`` on a full lattice, so no particle moves (interior)."""
    pos, volume = builder()
    h = H_FACTOR * 0.01
    c = concentration(pos, volume, h)
    c_ref = estimate_reference_concentration(c)
    assert 0.9 < c_ref < 1.1, f"full-support concentration {c_ref} is not ~1"
    out, state = shift_positions(pos, volume, h, dt=1.0e-4, v_ref=1.0,
                                 return_state=True)
    inner = interior_mask(pos, 3.0 * h)
    assert int(inner.sum()) > 40
    moved = torch.linalg.vector_norm((out - pos)[inner], dim=-1).max()
    assert float(moved) < 1.0e-9, f"interior moved by {float(moved):.3e} m"
    assert float(torch.linalg.vector_norm(state.shift[inner], dim=-1).max()) < 1.0e-9


def test_shift_moves_positions_only():
    """Mass/velocity are inputs; the shift returns positions out of place."""
    pos, volume = square_lattice(11)
    jitter = torch.randn(pos.shape, generator=torch.Generator().manual_seed(7),
                         dtype=torch.float64) * 1.0e-3
    pos = pos + jitter
    pos_before = pos.clone()
    volume_before = volume.clone()
    out = shift_positions(pos, volume, H_FACTOR * 0.01, dt=1.0e-4, v_ref=1.0)
    assert torch.equal(pos, pos_before), "input positions were mutated"
    assert torch.equal(volume, volume_before), "volumes were mutated"
    assert out.shape == pos.shape
    assert float(volume.sum()) == float(volume_before.sum())
    assert not torch.equal(out, pos), "a jittered lattice must move"


def test_shift_is_capped_in_units_of_h():
    pos, volume = square_lattice(11)
    # A violent vacancy makes grad C large; the cap must hold regardless.
    keep = torch.ones(pos.shape[0], dtype=torch.bool)
    keep[60] = False
    pos, volume = pos[keep], volume[keep]
    h = H_FACTOR * 0.01
    for limit in (0.05, 0.25):
        out, state = shift_positions(pos, volume, h, dt=1.0e-4, v_ref=50.0,
                                     strength=10.0, limit=limit,
                                     return_state=True)
        mag = torch.linalg.vector_norm(out - pos, dim=-1).max()
        assert float(mag) <= limit * h * (1.0 + 1.0e-12)
        assert state.clipped > 0.0, "the cap should have bound on this input"


def test_free_surface_shift_has_no_normal_component():
    pos, volume = square_lattice(21)
    h = H_FACTOR * 0.01
    out, state = shift_positions(pos, volume, h, dt=1.0e-4, v_ref=20.0,
                                 strength=5.0, return_state=True)
    fs = state.free_surface
    assert int(fs.sum()) > 0, "the detector must fire on a lattice with edges"
    normal_component = (state.shift * state.normal).sum(dim=-1).abs()[fs]
    applied = torch.linalg.vector_norm(state.shift[fs], dim=-1)
    assert float(normal_component.max()) <= 1.0e-12 * max(1.0, float(applied.max()))


def test_fickian_diffusion_is_monotone():
    """Repeated passes reduce the disorder of an irregular packing."""
    pos, volume = square_lattice(15)
    h = H_FACTOR * 0.01
    rng = torch.Generator().manual_seed(20261112)
    pos = pos + torch.randn(pos.shape, generator=rng, dtype=torch.float64) * 0.25 * 0.01
    inner = interior_mask(pos, 3.0 * h)
    history = [disorder(pos[inner], volume[inner], h)]
    for _ in range(20):
        pos = shift_positions(pos, volume, h, dt=2.0e-2, v_ref=1.0, strength=1.0,
                              limit=0.25, free_surface=0.0)
        history.append(disorder(pos[inner], volume[inner], h))
    assert history[-1] < 0.4 * history[0], f"disorder did not fall 2.5x: {history}"
    # monotone up to the relaxation floor
    assert all(b <= a * (1.0 + 1.0e-3) for a, b in zip(history, history[1:])), history


def test_vacancy_pulls_neighbours_in():
    """A missing particle draws its neighbours towards the hole (void repair)."""
    pos, volume = square_lattice(15)
    h = H_FACTOR * 0.01
    keep = torch.ones(pos.shape[0], dtype=torch.bool)
    hole = 15 * 7 + 7
    keep[hole] = False
    hole_center = pos[hole].clone()
    pos, volume = pos[keep], volume[keep]
    out, state = shift_positions(pos, volume, h, dt=1.0e-3, v_ref=1.0, strength=1.0,
                                 free_surface=0.0, return_state=True)
    d0 = torch.linalg.vector_norm(pos - hole_center, dim=-1)
    near = d0 < 1.5 * 0.01
    assert int(near.sum()) >= 4
    d1 = torch.linalg.vector_norm(out - hole_center, dim=-1)[near]
    assert float(d1.mean()) < float(d0[near].mean()), \
        "neighbours of a vacancy moved away from the hole"


def test_deterministic():
    pos, volume = square_lattice(13)
    h = H_FACTOR * 0.01
    rng = torch.Generator().manual_seed(11)
    pos = pos + torch.randn(pos.shape, generator=rng, dtype=torch.float64) * 0.2 * 0.01
    a = shift_positions(pos, volume, h, dt=2.0e-4, v_ref=7.0)
    b = shift_positions(pos, volume, h, dt=2.0e-4, v_ref=7.0)
    assert torch.equal(a, b)


def test_strength_zero_is_a_no_op_and_inputs_are_validated():
    pos, volume = square_lattice(9)
    h = H_FACTOR * 0.01
    out = shift_positions(pos, volume, h, dt=1.0e-4, v_ref=1.0, strength=0.0)
    assert torch.equal(out, pos)
    with pytest.raises(ValueError):
        shift_positions(pos, volume, h, dt=0.0, v_ref=1.0)
    with pytest.raises(ValueError):
        shift_positions(pos, volume, -1.0, dt=1.0e-4, v_ref=1.0)
    with pytest.raises(ValueError):
        shift_positions(pos, volume[:5], h, dt=1.0e-4, v_ref=1.0)