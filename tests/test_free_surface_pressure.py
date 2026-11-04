"""Unit tests for the free-surface pressure closure (tensorsph.free_surface_pressure).

These are the *analysis* checks: they use synthetic fields whose exact
extrapolation is known in closed form, so the closure can be verified without
running a drop.  The end-to-end acceptance numbers live in
``benchmarks/laplace_pressure_floor`` and ``tests/test_laplace_pressure_floor.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    tait_pressure,
    validate_config,
)
from tensorsph.free_surface_pressure import (
    extrapolated_pressure,
    flagged_fraction,
)
from tensorsph.neighbor_search import cell_linked_pairs

RHO0 = 1000.0


def _hex_patch(nx: int = 24, ny: int = 24, dx: float = 0.002):
    """Rectangular hex patch centred on the origin."""
    ys = []
    pts = []
    for j in range(ny):
        y = (j - 0.5 * (ny - 1)) * dx * math.sqrt(3.0) / 2.0
        offset = 0.5 * dx if j % 2 else 0.0
        for i in range(nx):
            pts.append((i * dx + offset - 0.5 * nx * dx, y))
    del ys
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos


def _config(**kw):
    base = dict(dx=0.002, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                gamma=7.0, gravity=0.0, surface_tension_coefficient=0.0,
                dt=1.0e-5)
    base.update(kw)
    return Config(**base)


def _state(pos, rho_value=None, cfg=None):
    cfg = cfg or _config()
    h = cfg.h_factor * cfg.dx
    mass0 = RHO0 * cfg.dx * (cfg.dx * math.sqrt(3.0) / 2.0) * torch.ones(
        pos.shape[0], dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = (torch.full((pos.shape[0],), rho_value, dtype=pos.dtype)
           if rho_value is not None
           else density_from_neighbor_pairs(pos, mass0, h, pairs))
    return mass0, pairs, rho


def test_off_mode_is_the_identity():
    pos = _hex_patch()
    cfg = _config(free_surface_pressure="off")
    mass, pairs, rho = _state(pos)
    p = tait_pressure(rho, cfg)
    p_eff, rho_eff = extrapolated_pressure(pos, rho, p, mass, pairs,
                                           pos.shape[0], cfg)
    assert torch.equal(p_eff, p)
    assert torch.equal(rho_eff, rho)


def test_uniform_pressure_is_reproduced_across_a_deficient_shell():
    """A shell particle whose neighbours all carry p0 must end up at p0."""
    pos = _hex_patch()
    cfg = _config(free_surface_pressure="shell", sound_speed=1.0)
    rho = torch.full((pos.shape[0],), 0.90 * RHO0, dtype=pos.dtype)  # deficient:
    # every particle is flagged, so the closure acts everywhere
    p = torch.full_like(rho, 137.0)
    mass, pairs, _ = _state(pos)
    p_eff, _ = extrapolated_pressure(pos, rho, p, mass, pairs, pos.shape[0], cfg)
    interior = torch.linalg.vector_norm(pos, dim=1) < 0.45 * float(
        torch.linalg.vector_norm(pos, dim=1).max())
    assert torch.allclose(p_eff[interior], p[interior], rtol=1e-12)


def test_linear_pressure_field_is_extrapolated_exactly():
    """Hydrostatic p = rho0 G (y_s - y) survives the body-force term exactly.

    For any stencil -- complete, truncated, or lopsided -- the sum
    ``sum_j (p_j + rho_j g . (r_i - r_j)) W_ij / sum_j W_ij`` returns ``p_i``
    for a linear field.  A kernel-weighted average *without* the body-force term
    is biased at the boundary by the truncation, which is exactly the failure
    mode the wall treatment already fixed.
    """
    pos = _hex_patch()
    G = 100.0
    cfg = _config(free_surface_pressure="shell", gravity=G, sound_speed=1.0)
    y = pos[:, 1]
    y_surface = float(y.max())
    rho_value = 0.95 * RHO0
    rho = torch.full((pos.shape[0],), rho_value, dtype=pos.dtype)
    # The field must be in equilibrium with the density actually stored, so that
    # the body-force term has no residual: p(y) = rho * G * (y_s - y).
    p = rho_value * G * (y_surface - y)
    mass, pairs, _ = _state(pos)
    p_eff, _ = extrapolated_pressure(pos, rho, p, mass, pairs, pos.shape[0], cfg)

    # Boundary particles (truncated stencil) must be exact, not merely close.
    edge = torch.linalg.vector_norm(pos, dim=1) > 0.6 * float(
        torch.linalg.vector_norm(pos, dim=1).max())
    err = (p_eff[edge] - p[edge]).abs() / p[edge].clamp_min(1.0)
    assert float(err.max()) < 1e-10

    # ...and the plain average (no body force) is *not* exact there, which is
    # what makes the test meaningful.
    h = cfg.h_factor * cfg.dx
    from tensorsph.droplet_impact import pair_kernels
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    wij, _ = pair_kernels(delta, h)
    num = torch.zeros(pos.shape[0], dtype=pos.dtype).index_add(0, i, p[j] * wij)
    num = num.index_add(0, j, p[i] * wij)
    den = torch.zeros(pos.shape[0], dtype=pos.dtype).index_add(0, i, wij)
    den = den.index_add(0, j, wij)
    plain_err = ((num / den)[edge] - p[edge]).abs() / p[edge].clamp_min(1.0)
    assert float(plain_err.max()) > 1e-4


def test_isolated_particle_stays_at_zero_pressure():
    pos = torch.tensor([[0.0, 0.0]])
    cfg = _config(free_surface_pressure="shell")
    rho = torch.tensor([0.5 * RHO0])
    p = torch.zeros(1, dtype=torch.float64)
    pairs = torch.zeros((0, 2), dtype=torch.long)
    p_eff, rho_eff = extrapolated_pressure(pos, rho, p, torch.tensor([1.0]),
                                           pairs, 1, cfg)
    assert float(p_eff[0]) == 0.0
    assert float(rho_eff[0]) == 0.5 * RHO0


def test_shell_density_mode_is_tait_consistent():
    pos = _hex_patch()
    cfg = _config(free_surface_pressure="shell_density", sound_speed=5.0)
    mass, pairs, rho = _state(pos)
    p = torch.zeros(pos.shape[0], dtype=pos.dtype)
    p[: pos.shape[0] // 2] = 500.0
    p_eff, rho_eff = extrapolated_pressure(pos, rho, p, mass, pairs,
                                           pos.shape[0], cfg)
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    recovered = bulk * ((rho_eff / cfg.rho0) ** cfg.gamma - 1.0)
    free = rho < cfg.free_surface_density_threshold * cfg.rho0
    assert bool(free.any())
    assert torch.allclose(recovered[free], p_eff[free], rtol=1e-10, atol=1e-8)
    # The density of *unflagged* particles is untouched.
    assert torch.equal(rho_eff[~free], rho[~free])


def test_threshold_selects_the_deficient_shell_only():
    pos = _hex_patch()
    cfg = _config(free_surface_pressure="shell")
    rho = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    assert flagged_fraction(rho, cfg) == 0.0
    rho[:5] *= 0.95
    assert flagged_fraction(rho, cfg) == pytest.approx(5.0 / pos.shape[0])


def test_rhs_keeps_the_continuity_density_while_closing_the_surface():
    """The closure must not leak into the density the solver returns."""
    pos = _hex_patch(12, 12)
    h = 1.3 * 0.002
    mass0 = RHO0 * 0.002 * (0.002 * math.sqrt(3.0) / 2.0) * torch.ones(
        pos.shape[0], dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    vel = torch.zeros_like(pos)
    out = {}
    for mode in ("off", "shell", "shell_density"):
        cfg = _config(free_surface_pressure=mode, dt=1.0e-6)
        out[mode] = rhs(pos, vel, mass0, cfg, pairs=pairs, dt=1.0e-6)
    # Same sum-density in every mode: the closure is a momentum-side transform.
    assert torch.equal(out["off"][0], out["shell"][0])
    assert torch.equal(out["off"][0], out["shell_density"][0])
    # And the acceleration actually changes, otherwise the test proves nothing.
    assert not torch.allclose(out["off"][2], out["shell"][2])


def test_invalid_mode_is_rejected():
    cfg = _config(free_surface_pressure="bogus")
    with pytest.raises(ValueError, match="free_surface_pressure"):
        validate_config(cfg)


def test_threshold_must_be_a_fraction():
    cfg = _config(free_surface_density_threshold=1.5)
    with pytest.raises(ValueError, match="free_surface_density_threshold"):
        validate_config(cfg)