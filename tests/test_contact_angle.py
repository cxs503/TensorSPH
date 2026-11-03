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

from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    compute_color_gradient,
    compute_curvature,
    compute_normal,
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


class TestContactAngleEnforcement:
    """Quantitative properties of the enforced angle.

    The originals of these assertions were shape checks; the angle itself was
    never measured, which is how the magnitude-collapse bug below could pass.
    """

    H = 0.026
    POS = [[0.1, 0.02], [0.2, 0.01], [0.3, 0.015], [0.4, 0.9]]
    WALL = [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]

    def _setup(self):
        pos = torch.tensor(self.POS, dtype=torch.float64)
        wall = torch.tensor(self.WALL, dtype=torch.float64)
        grad = torch.tensor([[0.3, 0.9], [-0.7, 0.2], [0.0, 1.0], [0.5, -0.5]],
                            dtype=torch.float64)
        normal = pos - wall
        normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        return pos, wall, grad, normal

    @pytest.mark.parametrize("theta", [math.pi / 6, math.pi / 3, math.pi / 2,
                                       2 * math.pi / 3])
    def test_enforced_angle_equals_contact_angle(self, theta):
        """n . n_wall must equal cos(theta) after the rotation."""
        pos, wall, grad, normal = self._setup()
        out = apply_contact_angle_boundary(grad.clone(), pos, wall, self.H, theta)
        unit = out[:2] / torch.linalg.vector_norm(out[:2], dim=1, keepdim=True)
        dot = (unit * normal[:2]).sum(dim=1)
        assert torch.allclose(dot, torch.full_like(dot, math.cos(theta)), atol=1e-6)

    @pytest.mark.parametrize("theta", [math.pi / 6, math.pi / 3, 2 * math.pi / 3])
    def test_magnitude_preserved_exactly(self, theta):
        pos, wall, grad, normal = self._setup()
        out = apply_contact_angle_boundary(grad.clone(), pos, wall, self.H, theta)
        before = torch.linalg.vector_norm(grad, dim=1)
        after = torch.linalg.vector_norm(out, dim=1)
        assert torch.allclose(before, after, rtol=1e-12)

    def test_gradient_parallel_to_wall_normal_is_left_alone(self):
        """No tangential direction exists, so no tilt is invented.

        The first revision multiplied by ``cos(theta) * n_wall`` here, shrinking
        the gradient to 0.5 of its magnitude (deleting the interface), and the
        magnitude test still passed because it only compared magnitudes of
        particles that had been modified in a unit-preserving way.
        """
        pos, wall, grad, normal = self._setup()
        out = apply_contact_angle_boundary(grad.clone(), pos, wall, self.H,
                                           math.pi / 3)
        assert torch.allclose(out[2], grad[2], rtol=0, atol=1e-12)

    def test_far_particle_is_bit_identical(self):
        pos, wall, grad, normal = self._setup()
        out = apply_contact_angle_boundary(grad.clone(), pos, wall, self.H,
                                           math.pi / 4)
        assert torch.equal(out[3], grad[3])


class TestContactAngleIntegration:
    """Integration of the contact angle with the CSF force.

    The configuration is a 7x7 fluid lattice sitting half a smoothing length
    above a single row of wall particles. It replaces a 6-particle toy cloud
    whose four "interior" particles all carry the same unit normal: there the
    discrete curvature of every particle is exactly zero (measured: kappa =
    [10.1, -4.6, 11.8, -2.9, 0.0, 0.0] 1/m), so the force on the two particles
    the original test asserted to be non-zero was identically zero. The
    assertion could not have held for any contact-angle treatment, because it
    is the curvature operator, not the boundary condition, that returns zero.
    """

    DX = 0.02

    def _setup(self):
        cfg = Config(dx=self.DX, pool_width=0.4, pool_depth=0.15,
                     drop_radius=0.05, drop_center_y=0.25, device="cpu",
                     surface_tension_coefficient=0.072)
        h = cfg.h_factor * cfg.dx
        n = 7
        idx = torch.arange(n, dtype=torch.float64)
        gx, gy = torch.meshgrid(idx, idx, indexing="xy")
        pos = torch.stack(((gx.reshape(-1) + 0.5) * cfg.dx + 0.06,
                           (gy.reshape(-1) + 0.5) * cfg.dx + 0.5 * h), dim=-1)
        wx = torch.arange(-2.0 * cfg.dx, n * cfg.dx, cfg.dx, dtype=torch.float64)
        wall = torch.stack((wx, torch.full_like(wx, pos[:, 1].min() - 0.5 * cfg.dx)),
                           dim=-1)
        density = torch.full((pos.shape[0],), cfg.rho0, dtype=torch.float64)
        mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                          dtype=torch.float64)
        pairs = cell_linked_pairs(pos, 2.0 * h)
        return cfg, pos, density, mass, pairs, wall

    def test_force_is_non_zero_on_surface_particles(self):
        cfg, pos, density, mass, pairs, wall = self._setup()
        acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
        moved = (acc.abs().sum(dim=1) > 1e-12)
        assert int(moved.sum()) > 0
        assert acc.shape == (pos.shape[0], 2)

    def test_contact_angle_changes_only_the_near_wall_force(self):
        cfg, pos, density, mass, pairs, wall = self._setup()
        wet = Config(dx=self.DX, device="cpu", surface_tension_coefficient=0.072,
                     contact_angle=math.pi / 3)
        base = compute_surface_tension_force(pos, density, mass, pairs, cfg)
        tilted = compute_surface_tension_force(pos, density, mass, pairs, wet,
                                               wall=wall)
        changed = (base - tilted).abs().sum(dim=1) > 1e-12
        assert int(changed.sum()) > 0
        h = cfg.h_factor * cfg.dx
        distance = (pos[:, None, :] - wall[None, :, :]).pow(2).sum(dim=2).sqrt()
        # The rotation acts within 2h of the wall, and the curvature operator
        # couples neighbours, so the influence reaches one more smoothing length.
        near = distance.min(dim=1).values < 4.0 * h
        assert bool((changed <= near).all()), "a far-from-wall force changed"

    def test_no_contact_angle_is_backward_compatible(self):
        """contact_angle=None plus a wall cloud must equal the wall-free result."""
        cfg, pos, density, mass, pairs, wall = self._setup()
        assert cfg.contact_angle is None
        plain = compute_surface_tension_force(pos, density, mass, pairs, cfg)
        with_wall = compute_surface_tension_force(pos, density, mass, pairs, cfg,
                                                  wall=wall)
        assert torch.equal(plain, with_wall)

    def test_config_has_contact_angle_parameter(self):
        cfg = Config()
        assert hasattr(cfg, "contact_angle")
        assert cfg.contact_angle is None
        wetting = Config(contact_angle=math.pi / 4)
        assert wetting.contact_angle == pytest.approx(math.pi / 4)

    def test_contact_angle_zero_sigma_no_force(self):
        cfg = Config(dx=0.05, device="cpu", surface_tension_coefficient=0.0,
                     contact_angle=math.pi / 3)
        _, pos, density, mass, pairs, wall = self._setup()
        cfg = Config(dx=self.DX, device="cpu", surface_tension_coefficient=0.0,
                     contact_angle=math.pi / 3)
        acc = compute_surface_tension_force(pos, density, mass, pairs, cfg,
                                           wall=wall)
        assert torch.allclose(acc, torch.zeros_like(acc))

    def test_uniform_normal_cloud_has_zero_curvature(self):
        """Recorded degeneracy: the old 6-particle cloud cannot carry a force."""
        cfg = Config(dx=0.02, device="cpu", surface_tension_coefficient=0.072)
        pos = torch.tensor([[0.2, 0.1], [0.2, 0.12], [0.22, 0.1], [0.18, 0.1],
                            [0.2, 0.08], [0.24, 0.1]], dtype=torch.float64)
        pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
                              [1, 4], [2, 3], [2, 5], [3, 5]], dtype=torch.long)
        density = torch.tensor([1000.0] * 4 + [800.0, 700.0], dtype=torch.float64)
        mass = torch.full((6,), cfg.rho0 * cfg.dx * cfg.dx, dtype=torch.float64)
        h = cfg.h_factor * cfg.dx
        count = torch.zeros(6, dtype=torch.float64)
        i, j = pairs.unbind(dim=1)
        count = count.index_add(0, i, torch.ones_like(i, dtype=torch.float64))
        count = count.index_add(0, j, torch.ones_like(j, dtype=torch.float64))
        surface = count < 0.7 * count.mean()
        color = torch.ones(6, dtype=torch.float64)
        color[surface] = count[surface] / count.mean()
        grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
        normal = compute_normal(-grad_c)
        kappa = compute_curvature(pos, normal, mass, density, pairs, h)
        assert torch.allclose(kappa[surface], torch.zeros(2, dtype=torch.float64),
                              atol=1e-10)
