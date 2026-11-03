"""Analytic gates for the CSF delta-function closure (``delta_s_scheme``).

The m2588 handoff left ``int delta_s dn = 0.74-0.76`` (25 % low) as the next
debt: the colour ramp's outer half lives in the void, so the one-sided gradient
of the outermost layer cannot see it.  ``delta_s_scheme="ghost"`` restores that
half by mirroring the layer's neighbours across the local tangent plane with the
known void value ``c = 0``.

Numbers below are the measured values of
``benchmarks/csf_delta_s/validation_report.json`` with a small margin, so a
regression in the closure turns a test red instead of silently shrinking the
CSF force again.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import (
    csf_fields,
    surface_tension_acceleration,
    void_mirror_gradient,
)
from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

RADIUS = 0.02
SIGMA = 7.2
RHO0 = 1000.0
BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "csf_delta_s"


def _droplet(dx=0.002):
    """Hexagonal lattice droplet from the shared benchmark generator."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "laplace_law_run", BENCH.parent / "laplace_law" / "run.py")
    law = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(law)
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature="fit_quadratic", csf_delta_s="ghost")
    h = cfg.h_factor * cfg.dx
    pos, mass = law.hexagonal_droplet(RADIUS, dx, RHO0)
    return cfg, pos, mass, h


def _integral(pos, mass, delta_s):
    r = torch.linalg.vector_norm(pos, dim=1)
    m = (r > 0.6 * RADIUS) & (r < 1.05 * RADIUS)
    dv = mass / RHO0
    return float((delta_s[m] * dv[m] / (2.0 * math.pi * r[m].clamp_min(1e-30))).sum())


def _cut(pos, mass, acc):
    right = pos[:, 0] > 0
    return float((mass[right, None] * acc[right]).sum(dim=0)[0])


def test_ghost_closure_restores_the_delta_function_normalisation():
    """``int delta_s dn = 1`` to within the 15 % gate (was 0.75)."""
    cfg, pos, mass, h = _droplet(0.002)
    wide = cell_linked_pairs(pos, 4.0 * h)
    base = csf_fields(pos, mass, wide, h, RHO0, delta_s_scheme="gradient")
    ghost = csf_fields(pos, mass, wide, h, RHO0, delta_s_scheme="ghost")

    one_sided = _integral(pos, mass, base.delta_s)
    two_sided = _integral(pos, mass, ghost.delta_s)
    # measured: 0.751 (one-sided) -> 1.027 (ghost)
    assert 0.70 < one_sided < 0.85, one_sided
    assert abs(two_sided - 1.0) < 0.15, two_sided
    assert two_sided > one_sided * 1.25


def test_void_mirror_is_exactly_zero_away_from_the_surface():
    """Bulk physics must stay bit-for-bit unchanged."""
    cfg, pos, mass, h = _droplet(0.002)
    wide = cell_linked_pairs(pos, 4.0 * h)
    state = csf_fields(pos, mass, wide, h, RHO0, delta_s_scheme="gradient")
    mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    extra = void_mirror_gradient(pos, mass, wide, h, RHO0, state.color,
                                 state.normal, mask)
    assert torch.equal(extra, torch.zeros_like(extra))

    deep = torch.linalg.vector_norm(pos, dim=1) < 0.6 * RADIUS
    full = csf_fields(pos, mass, wide, h, RHO0, delta_s_scheme="ghost")
    assert torch.equal(full.delta_s[deep], state.delta_s[deep])


def test_cut_force_with_rest_volume_enters_the_15_percent_gate():
    """``int f_x dV = 2 sigma`` with the volume convention delta_s is built on."""
    cfg, pos, mass, h = _droplet(0.002)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho_ref = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    acc, _ = surface_tension_acceleration(
        pos, rho_ref, mass, wide, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic", delta_s_scheme="ghost")
    ratio = _cut(pos, mass, acc) / (-2.0 * SIGMA)
    # measured +0.936 at dx = 2e-3
    assert abs(ratio - 1.0) < 0.15, ratio


def test_summation_density_overshoots_and_is_recorded_as_debt():
    """Honest accounting: the shipped path divides by the deficit density.

    With ``rho_i`` in the denominator the same force reads +22 % instead of
    -6 %: the free-surface density deficit now stands alone once kappa and
    delta_s stop cancelling.  This test pins the *documented* state -- if the
    density convention is unified later, this test must be updated with the new
    evidence, not deleted.
    """
    cfg, pos, mass, h = _droplet(0.002)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0 * h))
    acc, _ = surface_tension_acceleration(
        pos, rho, mass, wide, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic", delta_s_scheme="ghost")
    ratio = _cut(pos, mass, acc) / (-2.0 * SIGMA)
    # measured +1.221: the outer shell sits at rho/rho0 = 0.734, i.e. 1/0.734
    assert 1.10 < ratio < 1.40, ratio


def test_ghost_only_touches_the_support_deficit_layer():
    """Ghost must add support (never remove it) and only where c < 0.75."""
    cfg, pos, mass, h = _droplet(0.002)
    wide = cell_linked_pairs(pos, 4.0 * h)
    base = csf_fields(pos, mass, wide, h, RHO0, delta_s_scheme="gradient")
    ghost = csf_fields(pos, mass, wide, h, RHO0, delta_s_scheme="ghost")
    changed = (ghost.delta_s - base.delta_s).abs() > 1e-12
    assert bool(changed.any())
    # every changed particle is in the layer that lost kernel support
    assert float(base.color[changed].max()) < 0.75
    # a masked particle can be untouched when every mirror image lands outside
    # the 2h support, so the converse is only checked in bulk
    assert float(base.color[~changed].median()) > 0.95
    # the interface-weighted gain is the missing void half: 1.25x .. 1.5x
    gain = float(ghost.delta_s.sum() / base.delta_s.sum())
    assert 1.15 < gain < 1.60, gain


def test_unknown_delta_s_scheme_is_rejected():
    cfg, pos, mass, h = _droplet(0.002)
    with pytest.raises(ValueError):
        csf_fields(pos, mass, torch.zeros((0, 2), dtype=torch.long), h, RHO0,
                   delta_s_scheme="shepard")


def test_committed_validation_report_matches_the_gates():
    """The published JSON must agree with the gates asserted above."""
    import json
    payload = json.loads((BENCH / "validation_report.json").read_text())
    ghost = [r for r in payload["instant"]
             if r["delta_s_scheme"] == "ghost" and r["curvature"] == "fit_quadratic"]
    assert ghost, "ghost rows missing from the report"
    for row in ghost:
        assert abs(row["delta_s"]["integral_dn"] - 1.0) < 0.15, row
        assert row["cut_force_rest_volume"]["within_15pct"], row
    one_sided = [r for r in payload["instant"]
                 if r["delta_s_scheme"] == "gradient"]
    for row in one_sided:
        assert row["delta_s"]["integral_dn"] < 0.85, row