"""Regression gates for the Langevin LJ-fluid lattice and temperature observable.

Guards two defects that made the whole transport-coefficient verification of
``tensorsph.fluctuating_hydro`` unrunnable:

1. ``initialize_lattice`` padded non-perfect-cube ``N`` with uniform random
   positions, creating ``r << 0.5 sigma`` pairs and a Lennard-Jones blow-up.
   The production verification fixture used ``N = 128`` and was therefore
   silently skipped by a ``pytest.skip("numerical blow-up")`` guard.
2. The instantaneous temperature was computed as ``2 <KE> / (d N)`` although
   ``<KE>`` is already a per-particle mean, shrinking the reading by ``N``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.fluctuating_hydro import (
    LJFluidConfig,
    initialize_lattice,
    run_lj_fluid,
)

OVERLAP_LIMIT = 0.9  # sigma; below this the LJ force is already ~1e2


def _min_image_distance(positions: torch.Tensor, box: float) -> float:
    n = positions.shape[0]
    ij = torch.triu_indices(n, n, offset=1)
    d = positions[ij[0]] - positions[ij[1]]
    d = d - box * torch.round(d / box)
    return float((d ** 2).sum(dim=1).sqrt().min())


@pytest.mark.parametrize("n_particles", [64, 128, 216, 256, 300, 500, 999])
def test_lattice_has_no_overlapping_pairs(n_particles):
    """Every supported N must start without LJ-singular pairs."""
    pos, vel, box = initialize_lattice(n_particles, 3, 0.8, 1.0, seed=42)
    assert pos.shape == (n_particles, 3)
    assert _min_image_distance(pos, box) > OVERLAP_LIMIT


@pytest.mark.parametrize("n_particles", [64, 128, 256, 999])
def test_lattice_density_is_exact(n_particles):
    _, _, box = initialize_lattice(n_particles, 3, 0.8, 1.0, seed=42)
    assert abs(n_particles / box ** 3 - 0.8) < 1e-12


def test_lattice_is_seed_deterministic_but_not_a_slab():
    """The occupied subset must be reproducible and spatially extended."""
    a, _, box = initialize_lattice(300, 3, 0.8, 1.0, seed=7)
    b, _, _ = initialize_lattice(300, 3, 0.8, 1.0, seed=7)
    assert torch.equal(a, b)
    # A "first N sites" fill would leave one coordinate span at zero.
    spans = a.max(dim=0).values - a.min(dim=0).values
    assert float(spans.min()) > 0.5 * box


@pytest.mark.parametrize("n_particles", [128, 256])
def test_previously_padded_size_does_not_blow_up(n_particles):
    """N=128/256 used to reach KE ~ 1e35; they must now stay physical."""
    cfg = LJFluidConfig(
        n_particles=n_particles, dimension=3, number_density=0.8,
        temperature=1.0, n_steps=800, sample_every=10,
    )
    result = run_lj_fluid(cfg)
    assert math.isfinite(result.equipartition_ratio)
    assert result.equipartition_ratio < 1.5, result.equipartition_ratio
    assert abs(result.mean_temperature - 1.0) < 0.20, result.mean_temperature


def test_temperature_observable_matches_kinetic_energy():
    """T = 2 <KE> / d must reproduce the velocity variance directly."""
    pos, vel, box = initialize_lattice(256, 3, 0.8, 1.75, seed=3)
    mean_ke = 0.5 * (vel ** 2).sum(dim=-1).mean().item()
    assert abs(2.0 * mean_ke / 3.0 - 1.75) / 1.75 < 0.05
