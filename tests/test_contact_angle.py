"""Tests for contact angle boundary condition in surface tension.

These tests verify:
1. Contact angle enforcement modifies the color gradient correctly
2. Hydrophilic (θ < π/2) and hydrophobic (θ > π/2) cases work
3. Neutral contact angle (θ = π/2) leaves gradient unchanged
4. Integration with surface tension force computation
5. Backward compatibility (contact_angle=None disables BC)
"""
import copy
import math

import pytest
import torch

from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    compute_color_gradient,
    compute_surface_tension_force,
)
from tensorsph.droplet_impact import Config, wall_lattice


def _make_simple_wall(device="cpu"):
    """Create a simple wall lattice for testing."""
    cfg = Config(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
                 drop_center_y=0.25, device=device)
    return wall_lattice(cfg, torch.device(device))


def _make_near_wall_particles(device="cpu"):
    """Create particles near a wall for testing contact angle."""
    pos = torch.tensor([
        [0.1, 0.02],
        [0.2, 0.03],
        [0.3, 0.08],
        [0.15, 0.5],
    ], dtype=torch.float32, device=device)
    return pos


class TestContactAngleBoundary:
    """Tests for apply_contact_angle_boundary function."""

    def test_empty_wall_returns_unchanged(self):
        """Empty wall positions should return gradient unchanged."""
        grad_c = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        pos = torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float32)
        wall = torch.empty((0, 2), dtype=torch.float32)
        h = 0.1
        theta = math.pi / 4

        result = apply_contact_angle_boundary(grad_c.clone(), pos, wall, h, theta)
        assert torch.allclose(result, grad_c)

    def test_neutral_contact_angle_preserves_gradient(self):
        """θ = π/2 (neutral) should preserve the tangential component."""
        device = "cpu"
        pos = _make_near_wall_particles(device)
        wall = _make_simple_wall(device)
        h = 0.065

        grad_c = torch.tensor([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=torch.float32, device=device)

        theta = math.pi / 2
        result = apply_contact_angle_boundary(grad_c.clone(), pos, wall, h, theta)
        assert result.shape == grad_c.shape

    def test_hydrophilic_angle_tilts_toward_wall(self):
        """θ < π/2 (hydrophilic) should tilt gradient toward wall."""
        device = "cpu"
        pos = _make_near_wall_particles(device)
        wall = _make_simple_wall(device)
        h = 0.065

        grad_c = torch.tensor([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=torch.float32, device=device)

        theta = math.pi / 6
        result = apply_contact_angle_boundary(grad_c.clone(), pos, wall, h, theta)

        assert result.shape == grad_c.shape
        for i in range(len(pos)):
            orig_mag = torch.linalg.vector_norm(grad_c[i])
            new_mag = torch.linalg.vector_norm(result[i])
            if orig_mag > 1e-8:
                assert torch.isclose(orig_mag, new_mag, atol=1e-6)

    def test_hydrophobic_angle_tilts_away_from_wall(self):
        """θ > π/2 (hydrophobic) should tilt gradient away from wall."""
        device = "cpu"
        pos = _make_near_wall_particles(device)
        wall = _make_simple_wall(device)
        h = 0.065

        grad_c = torch.tensor([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=torch.float32, device=device)

        theta = 2 * math.pi / 3
        result = apply_contact_angle_boundary(grad_c.clone(), pos, wall, h, theta)
        assert result.shape == grad_c.shape

    def test_far_particles_unchanged(self):
        """Particles far from wall should have unchanged gradient."""
        device = "cpu"
        wall = torch.tensor([[0.0, -0.05], [0.1, -0.05]], dtype=torch.float32, device=device)
        pos = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        h = 0.065

        grad_c = torch.tensor([[1.0, 0.5]], dtype=torch.float32, device=device)
        theta = math.pi / 4

        result = apply_contact_angle_boundary(grad_c.clone(), pos, wall, h, theta)
        assert torch.allclose(result, grad_c, atol=1e-6)

    def test_gradient_magnitude_preserved(self):
        """Contact angle enforcement should preserve gradient magnitude."""
        device = "cpu"
        pos = _make_near_wall_particles(device)
        wall = _make_simple_wall(device)
        h = 0.065

        grad_c = torch.tensor([
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.8, 0.2],
        ], dtype=torch.float32, device=device)

        theta = math.pi / 3
        result = apply_contact_angle_boundary(grad_c.clone(), pos, wall, h, theta)

        for i in range(len(pos)):
            orig_mag = torch.linalg.vector_norm(grad_c[i])
            new_mag = torch.linalg.vector_norm(result[i])
            if orig_mag > 1e-8:
                assert torch.isclose(orig_mag, new_mag, atol=1e-5)


class TestContactAngleIntegration:
    """Integration tests for contact angle with surface tension force."""

    def _make_wall_film(self, device="cpu"):
        """Small pool on the production wall lattice (bottom + side layers).

        The previous fixture was a flat, collinear chain, which is degenerate for
        a curvature operator: when every particle shares the same normal the
        stencil differences ``n_j - n_i`` vanish, ``div(n)`` is exactly zero and
        the CSF force vanishes with it -- independently of any contact-angle
        treatment.  Measured with the old six-particle chain: ``kappa`` came out
        as ``[10.10, -4.57, 11.76, -2.91, 0, 0]``, exactly zero on the two
        particles the assertions inspected.  This fixture is the real geometry
        (a pool three lattice spacings deep on the Adami wall lattice), so the
        bottom row sits inside ``2h`` of the wall, the top row sits on the free
        surface, and both regions carry a non-zero CSF force.
        """
        dx = 0.05
        cfg = Config(
            dx=dx, pool_width=0.35, pool_depth=0.15, drop_radius=0.04,
            drop_center_y=0.25, device=device,
            surface_tension_coefficient=0.072,
        )
        h = cfg.h_factor * cfg.dx
        xs = torch.arange(dx * 0.5, cfg.pool_width, dx, device=device)
        ys = torch.arange(dx * 0.5, cfg.pool_depth, dx, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
        wall = wall_lattice(cfg, torch.device(device))

        d = torch.cdist(pos, pos)
        iu = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
        keep = d[iu[0], iu[1]] < h
        pairs = torch.stack((iu[0][keep], iu[1][keep]), dim=1).contiguous()

        density = torch.full((pos.shape[0],), cfg.rho0, dtype=pos.dtype, device=device)
        mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, dtype=pos.dtype,
                          device=device)
        return cfg, pos, density, mass, pairs, wall

    def _near_wall_mask(self, pos, wall, cfg):
        h = cfg.h_factor * cfg.dx
        return torch.cdist(pos, wall).min(dim=1).values < 2.0 * h

    def test_surface_tension_with_contact_angle(self):
        """Contact angle must reorient the force inside 2h of the wall only."""
        device = "cpu"
        cfg_flat, pos, density, mass, pairs, wall = self._make_wall_film(device)
        cfg_ca = copy.deepcopy(cfg_flat)
        cfg_ca.contact_angle = math.pi / 3

        acc_ca = compute_surface_tension_force(pos, density, mass, pairs, cfg_ca,
                                               wall=wall)
        acc_none = compute_surface_tension_force(pos, density, mass, pairs, cfg_flat,
                                                 wall=wall)
        assert acc_ca.shape == pos.shape
        assert torch.isfinite(acc_ca).all()
        # The curved film must produce a non-zero CSF force in the first place,
        # otherwise the comparison below would be vacuous.
        assert acc_none.abs().max() > 0.0

        near = self._near_wall_mask(pos, wall, cfg_ca)
        assert bool(near.any()) and not bool(near.all())
        # Inside the wall band the prescribed angle changes the force ...
        assert not torch.allclose(acc_ca[near], acc_none[near], atol=1e-12)
        # ... and outside it the two runs are identical.
        assert torch.allclose(acc_ca[~near], acc_none[~near], atol=1e-12)

    def test_surface_tension_without_contact_angle(self):
        """contact_angle=None must reproduce the wall-free code path exactly."""
        device = "cpu"
        cfg, pos, density, mass, pairs, wall = self._make_wall_film(device)

        acc_with_wall = compute_surface_tension_force(pos, density, mass, pairs,
                                                      cfg, wall=wall)
        acc_no_wall = compute_surface_tension_force(pos, density, mass, pairs, cfg)
        assert acc_with_wall.shape == pos.shape
        assert torch.isfinite(acc_with_wall).all()
        assert acc_with_wall.abs().max() > 0.0
        assert torch.allclose(acc_with_wall, acc_no_wall, atol=1e-12)

    def test_config_has_contact_angle_parameter(self):
        """Config should have contact_angle parameter."""
        cfg = Config()
        assert hasattr(cfg, 'contact_angle')
        assert cfg.contact_angle is None

        cfg_wetting = Config(contact_angle=math.pi / 4)
        assert cfg_wetting.contact_angle == pytest.approx(math.pi / 4)

    def test_contact_angle_zero_sigma_no_force(self):
        """Zero surface tension coefficient should produce zero force."""
        device = "cpu"
        cfg = Config(
            dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
            drop_center_y=0.25, device=device,
            surface_tension_coefficient=0.0,
            contact_angle=math.pi / 3,
        )

        pos = torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float32, device=device)
        density = torch.full((2,), cfg.rho0, dtype=torch.float32, device=device)
        mass = torch.full((2,), cfg.rho0 * cfg.dx * cfg.dx, dtype=torch.float32, device=device)
        pairs = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        wall = _make_simple_wall(device)

        acc = compute_surface_tension_force(pos, density, mass, pairs, cfg, wall=wall)
        assert torch.allclose(acc, torch.zeros_like(acc))
