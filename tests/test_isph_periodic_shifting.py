"""Periodic particle regularisation: what ``shift_positions`` guarantees.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``, round 2 (mission m21075).
The Taylor-Green projection run fails for a *cloud* reason, not an operator
reason: ``outputs/isph_taylor_green/error_budget.json`` shows the local per-step
error is 0.0185 % while the smallest pair distance collapses from 0.88 dx to
0.36 dx and then to 0.003 dx, at which point the projection stops annihilating
divergence (``div_after/div_before`` > 1).  These tests pin the behaviour of the
regularisation primitive that the driver gained, *including* its measured
limitation: it keeps a cloud regular but does not by itself fix the run (see
``docs/xval_isph_projection_taylor_green_r2.md`` §4).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.isph_periodic import (concentration, concentration_gradient,
                                     periodic_pairs, shift_positions)

L = 1.0
DX = 0.05
H = 1.3 * DX


@pytest.fixture(scope="module", autouse=True)
def _dtype():
    torch.set_default_dtype(torch.float64)


def _lattice(n_side: int):
    line = (torch.arange(n_side, dtype=torch.float64) + 0.5) * (L / n_side)
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    vol = torch.full((pos.shape[0],), (L * L) / pos.shape[0], dtype=torch.float64)
    return pos, vol


def _min_pair_distance(pos, h=H, box=L):
    pairs = periodic_pairs(pos, 2.0 * h, box)
    i, j = pairs.unbind(dim=1)
    d = pos[i] - pos[j]
    d = d - box * torch.round(d / box)
    return float(torch.linalg.vector_norm(d, dim=-1).min())


def test_concentration_is_one_on_a_complete_lattice():
    """The kernel sum must recover ``int W dV = 1`` on a regular cloud."""
    pos, vol = _lattice(20)
    pairs = periodic_pairs(pos, 2.0 * H, L)
    c = concentration(pos, vol, pairs, H, L)
    assert float(c.mean()) == pytest.approx(1.0, abs=0.05)


def test_concentration_gradient_vanishes_on_a_lattice():
    """Symmetry: a perfect lattice must not be shifted (Lind et al. 2012, §2)."""
    pos, vol = _lattice(20)
    pairs = periodic_pairs(pos, 2.0 * H, L)
    grad = concentration_gradient(pos, vol, pairs, H, L)
    assert float(torch.linalg.vector_norm(grad, dim=-1).max()) < 1e-8


def test_a_lattice_is_left_alone():
    pos, vol = _lattice(20)
    pairs = periodic_pairs(pos, 2.0 * H, L)
    moved = shift_positions(pos, vol, pairs, H, L, amplitude=0.5, dx=DX)
    # positions live in [-L/2, L/2) after the wrap, so compare minimum images
    d = moved - pos
    d = d - L * torch.round(d / L)
    assert float(d.abs().max()) < 1e-8


def test_shift_is_clipped_and_stays_inside_the_box():
    pos, vol = _lattice(20)
    pos = pos.clone()
    pos[0] += torch.tensor([0.4 * DX, 0.0])   # compress one particle: large |grad C|
    pairs = periodic_pairs(pos, 2.0 * H, L)
    moved = shift_positions(pos, vol, pairs, H, L, amplitude=0.5, dx=DX,
                            max_shift_fraction=0.2)
    raw = moved - pos
    raw = raw - L * torch.round(raw / L)
    disp = torch.linalg.vector_norm(raw, dim=-1)
    assert float(disp.max()) <= 0.2 * DX + 1e-12
    assert bool(((moved >= -0.5 * L - 1e-12) & (moved <= 0.5 * L + 1e-12)).all())


def test_repeated_shifting_un_clumps_a_jittered_cloud():
    """The mechanism must do what it is for: raise the smallest pair distance."""
    pos, vol = _lattice(20)
    gen = torch.Generator().manual_seed(7)
    pos = pos + 0.12 * DX * (torch.rand(pos.shape, generator=gen, dtype=torch.float64) - 0.5)
    pos = pos - L * torch.round(pos / L)
    before = _min_pair_distance(pos)
    for _ in range(12):
        pairs = periodic_pairs(pos, 2.0 * H, L)
        pos = shift_positions(pos, vol, pairs, H, L, amplitude=0.5, dx=DX)
    after = _min_pair_distance(pos)
    assert after > before, f"shifting must un-clump ({before:.4f} -> {after:.4f})"


def test_measured_limitation_shifting_does_not_rescue_the_taylor_green_run():
    """Honest negative result, pinned so it is not "fixed" by wishful edits.

    ``outputs/isph_taylor_green/results_shifting_cd0.5.json`` and
    ``..._shifting_geometric005.json``: both regularised runs diverge
    (error > 1e3 %).  A future improvement must replace these artefacts with
    *new measurements*, not delete the assertion.
    """
    import json
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "outputs" / "isph_taylor_green"
    for name in ("results_shifting_cd0.5.json", "results_shifting_geometric005.json"):
        doc = json.loads((base / name).read_text())
        row = doc["rows"][0]
        assert row["shifting_amplitude"] > 0.0
        assert row["velocity_rel_l2_pct"] > 1e3, f"{name}: recorded run is not the divergent one"
        assert doc["pass"] is False
