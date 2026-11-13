"""Invariant tests for :mod:`tensorsph.transport_velocity` (Zhang, Hu & Adams 2017).

Every assertion here is an invariant that can be derived from the scheme:

* a perfect lattice is a *fixed point* (``∇C = 0`` by symmetry → ``δv = 0``);
* the correction is zero when ``strength = 0``;
* at a free surface the correction has no normal component;
* the operator is deterministic and free of RNG;
* the correction scales linearly with ``strength``, ``v_ref``, and ``dt``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.transport_velocity import (
    TransportVelocityState,
    concentration,
    concentration_gradient,
    estimate_reference_concentration,
    transport_velocity_correction,
)


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


H_FACTOR = 1.3


@pytest.mark.parametrize("builder", [square_lattice, hex_lattice],
                         ids=["square", "hex"])
def test_perfect_lattice_is_a_fixed_point(builder):
    """``∇C = 0`` on a full lattice, so ``δv = 0`` (interior)."""
    pos, volume = builder()
    h = H_FACTOR * 0.01
    c = concentration(pos, volume, h)
    c_ref = estimate_reference_concentration(c)
    assert 0.9 < c_ref < 1.1, f"full-support concentration {c_ref} is not ~1"
    delta_v, state = transport_velocity_correction(
        pos, volume, h, dt=1.0e-4, v_ref=1.0, return_state=True)
    inner = interior_mask(pos, 3.0 * h)
    assert int(inner.sum()) > 40
    correction_mag = torch.linalg.vector_norm(delta_v[inner], dim=-1).max()
    assert float(correction_mag) < 1.0e-9, \
        f"interior δv = {float(correction_mag):.3e} m/s on perfect lattice"


def test_correction_moves_fluid_only():
    """The correction returns a velocity field; it does not mutate inputs."""
    pos, volume = square_lattice(11)
    jitter = torch.randn(pos.shape, generator=torch.Generator().manual_seed(7),
                         dtype=torch.float64) * 1.0e-3
    pos = pos + jitter
    pos_before = pos.clone()
    volume_before = volume.clone()
    delta_v = transport_velocity_correction(pos, volume, H_FACTOR * 0.01,
                                            dt=1.0e-4, v_ref=1.0)
    assert torch.equal(pos, pos_before), "input positions were mutated"
    assert torch.equal(volume, volume_before), "volumes were mutated"
    assert delta_v.shape == pos.shape
    # A jittered lattice must have non-zero correction
    assert float(torch.linalg.vector_norm(delta_v, dim=-1).max()) > 1.0e-15


def test_strength_zero_is_a_no_op():
    pos, volume = square_lattice(11)
    h = H_FACTOR * 0.01
    jitter = torch.randn(pos.shape, generator=torch.Generator().manual_seed(42),
                         dtype=torch.float64) * 1.0e-3
    pos = pos + jitter
    delta_v = transport_velocity_correction(pos, volume, h, dt=1.0e-4,
                                            v_ref=1.0, strength=0.0)
    assert float(torch.linalg.vector_norm(delta_v, dim=-1).max()) < 1.0e-30


def test_correction_scales_linearly():
    """δv ∝ strength · v_ref · dt."""
    pos, volume = square_lattice(11)
    h = H_FACTOR * 0.01
    jitter = torch.randn(pos.shape, generator=torch.Generator().manual_seed(99),
                         dtype=torch.float64) * 1.0e-3
    pos = pos + jitter
    base = transport_velocity_correction(pos, volume, h, dt=1.0e-4, v_ref=1.0,
                                         strength=1.0)
    doubled_s = transport_velocity_correction(pos, volume, h, dt=1.0e-4, v_ref=1.0,
                                              strength=2.0)
    doubled_dt = transport_velocity_correction(pos, volume, h, dt=2.0e-4, v_ref=1.0,
                                               strength=1.0)
    doubled_v = transport_velocity_correction(pos, volume, h, dt=1.0e-4, v_ref=2.0,
                                              strength=1.0)
    assert torch.allclose(doubled_s, 2.0 * base, rtol=1.0e-10)
    assert torch.allclose(doubled_dt, 2.0 * base, rtol=1.0e-10)
    assert torch.allclose(doubled_v, 2.0 * base, rtol=1.0e-10)


def test_free_surface_correction_has_no_normal_component():
    pos, volume = square_lattice(21)
    h = H_FACTOR * 0.01
    delta_v, state = transport_velocity_correction(pos, volume, h, dt=1.0e-4,
                                                   v_ref=20.0, strength=5.0,
                                                   return_state=True)
    fs = state.free_surface
    assert int(fs.sum()) > 0, "the detector must fire on a lattice with edges"
    normal_component = (delta_v * state.normal).sum(dim=-1).abs()[fs]
    applied = torch.linalg.vector_norm(delta_v[fs], dim=-1)
    assert float(normal_component.max()) <= 1.0e-12 * max(1.0, float(applied.max()))


def test_deterministic():
    pos, volume = square_lattice(13)
    h = H_FACTOR * 0.01
    rng = torch.Generator().manual_seed(11)
    pos = pos + torch.randn(pos.shape, generator=rng, dtype=torch.float64) * 0.2 * 0.01
    a = transport_velocity_correction(pos, volume, h, dt=2.0e-4, v_ref=7.0)
    b = transport_velocity_correction(pos, volume, h, dt=2.0e-4, v_ref=7.0)
    assert torch.equal(a, b)


def test_inputs_validated():
    pos, volume = square_lattice(9)
    h = H_FACTOR * 0.01
    with pytest.raises(ValueError):
        transport_velocity_correction(pos, volume, h, dt=0.0, v_ref=1.0)
    with pytest.raises(ValueError):
        transport_velocity_correction(pos, volume, -1.0, dt=1.0e-4, v_ref=1.0)
    with pytest.raises(ValueError):
        transport_velocity_correction(pos, volume[:5], h, dt=1.0e-4, v_ref=1.0)


def test_correction_drives_towards_uniformity():
    """Repeated correction passes reduce the concentration gradient magnitude."""
    pos, volume = square_lattice(15)
    h = H_FACTOR * 0.01
    rng = torch.Generator().manual_seed(20261112)
    pos = pos + torch.randn(pos.shape, generator=rng, dtype=torch.float64) * 0.25 * 0.01
    inner = interior_mask(pos, 3.0 * h)

    def disorder(p, v):
        c = concentration(p, v, h)
        g = concentration_gradient(p, v, c, h)
        return float(torch.sqrt(torch.mean(g.square())) * h)

    vol_inner = volume[inner].clone()
    history = [disorder(pos[inner], vol_inner)]
    for _ in range(20):
        delta_v = transport_velocity_correction(pos, volume, h, dt=2.0e-2,
                                                v_ref=1.0, strength=1.0,
                                                free_surface_threshold=0.0)
        pos = pos + delta_v * 1.0  # apply as a pseudo-position shift
        inner = interior_mask(pos, 3.0 * h)
        vol_inner = volume[inner].clone()
        history.append(disorder(pos[inner], vol_inner))
    assert history[-1] < 0.4 * history[0], \
        f"disorder did not fall 2.5x: {history}"
