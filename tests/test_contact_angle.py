"""Tests for contact angle boundary condition in surface tension.

These tests verify:
1. Contact angle enforcement modifies the color gradient correctly
2. Hydrophilic (θ < π/2) and hydrophobic (θ > π/2) cases work
3. Neutral contact angle (θ = π/2) leaves gradient unchanged
4. Integration with surface tension force computation
5. Backward compatibility (contact_angle=None disables BC)
"""
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


def _cap_droplet(radius=0.06, dx=0.01, rho0=1000.0):
    """Rank-sufficient geometry: hex half-disc standing on the wall.

    The original fixture of these two integration tests was a six-particle
    cluster whose neighbours all share one coordinate, so ``div(normal)`` -- and
    hence the CSF curvature -- is *identically* zero there for any contact
    angle.  The assertion "surface particles must have non-zero acceleration"
    was therefore unsatisfiable, independently of the boundary condition.  A
    half disc supports a curvature estimate, so the test now checks the wiring
    claim it makes in its name.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    pts = []
    j = 0
    y = 0.0
    while y <= radius:
        x = -radius + (0.5 * dx if j % 2 else 0.0)
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y + 0.5 * dx))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), rho0 * dx * dy, dtype=torch.float64)


def _cap_force(theta, wall, sigma=0.072, dx=0.01):
    from tensorsph.droplet_impact import density_from_neighbor_pairs
    from tensorsph.neighbor_search import cell_linked_pairs
    from tensorsph.surface_tension import compute_surface_tension_force

    pos, mass = _cap_droplet(dx=dx)
    cfg = Config(dx=dx, rho0=1000.0, sound_speed=20.0, gravity=0.0,
                 viscosity=0.01, surface_tension_coefficient=sigma,
                 contact_angle=theta)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    return pos, h, compute_surface_tension_force(pos, rho, mass, pairs, cfg,
                                                 wall=wall)


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

    def _make_droplet_particles(self, device="cpu"):
        """Create a droplet-like particle configuration with surface particles."""
        cfg = Config(dx=0.02, device=device)
        h = cfg.h_factor * cfg.dx  # ~0.026

        # Particles within 2h of each other
        pos = torch.tensor([
            [0.2, 0.1],    # 0: interior
            [0.2, 0.12],   # 1: interior
            [0.22, 0.1],   # 2: interior
            [0.18, 0.1],   # 3: interior
            [0.2, 0.08],   # 4: surface (fewer neighbors)
            [0.24, 0.1],   # 5: surface (fewest neighbors)
        ], dtype=torch.float32, device=device)

        # Pairs for particles within interaction range
        pairs = torch.tensor([
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [1, 3], [1, 4],
            [2, 3], [2, 5],
            [3, 5],
        ], dtype=torch.long, device=device)

        density = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0, 800.0, 700.0],
                               dtype=torch.float32, device=device)
        mass = torch.full((6,), cfg.rho0 * cfg.dx * cfg.dx, dtype=torch.float32, device=device)

        return cfg, pos, density, mass, pairs

    def test_surface_tension_with_contact_angle(self):
        """Surface tension force should use contact angle when wall is provided."""
        device = "cpu"
        cfg, pos, density, mass, pairs = self._make_droplet_particles(device)
        cfg = Config(
            dx=0.02, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
            drop_center_y=0.25, device=device,
            surface_tension_coefficient=0.072,
            contact_angle=math.pi / 3,
        )

        wall = torch.tensor([[(k - 30) * 0.01, 0.0] for k in range(60)],
                            dtype=torch.float64)
        cap_pos, h, with_bc = _cap_force(math.pi / 3, wall)
        _, _, without = _cap_force(None, wall)
        delta = (with_bc - without).norm(dim=1)
        dist = torch.cdist(cap_pos, wall).min(dim=1).values
        # the boundary condition must bite on the wall band ...
        assert float(delta[dist <= 2.0 * h].max()) > 0.0
        # ... and leave everything beyond one kernel radius untouched
        far = dist > 4.0 * h
        assert int(far.sum()) > 0
        assert torch.equal(delta[far], torch.zeros_like(delta[far]))

    def test_surface_tension_without_contact_angle(self):
        """Surface tension force should work without contact angle (backward compat)."""
        device = "cpu"
        cfg = Config(
            dx=0.02, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
            drop_center_y=0.25, device=device,
            surface_tension_coefficient=0.072,
            contact_angle=None,
        )

        from tensorsph.droplet_impact import density_from_neighbor_pairs
        from tensorsph.neighbor_search import cell_linked_pairs

        pos, mass = _cap_droplet()
        wall = torch.tensor([[(k - 30) * 0.01, 0.0] for k in range(60)],
                            dtype=torch.float64)
        h = cfg.h_factor * cfg.dx
        pairs = cell_linked_pairs(pos, 2.0 * h)
        density = density_from_neighbor_pairs(pos, mass, h, pairs)
        with_wall = compute_surface_tension_force(pos, density, mass, pairs, cfg,
                                                  wall=wall)
        no_wall = compute_surface_tension_force(pos, density, mass, pairs, cfg)
        # backward compatibility is bit-exact: an unset contact angle means the
        # wall is not consulted at all
        assert with_wall.shape == no_wall.shape
        assert torch.equal(with_wall, no_wall)

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
