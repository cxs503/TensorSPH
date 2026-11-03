"""Analytic verification of the wall contact-angle boundary condition.

The boundary condition has a closed-form target, so it can be scored exactly
rather than statistically: after :func:`apply_contact_angle_boundary` the
interface normal of a particle sitting on the wall must satisfy

    |cos angle(n, n_wall)| = cos(theta)

to machine precision, its magnitude must be unchanged (the boundary condition
steers direction only, so it cannot rescale ``delta_s``), and the correction
must switch off completely beyond one support radius.

These are reference-value tests, not smoke tests: the earlier suite only
asserted ``result.shape`` for the three angle cases, which a stub returning the
input unchanged satisfies. The integration class at the bottom is the part that
was previously vacuous -- its six-particle configuration was collinear, so its
curvature was identically zero and the "surface particles must accelerate"
assertion could never hold.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config
from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    compute_surface_tension_force,
)

DX = 0.02
H = 1.3 * DX          # 0.026
REACH = 2.0 * H       # 0.052


def bottom_wall(y: float = 0.0, half_width: float = 0.20, dx: float = DX):
    """Horizontal wall patch: a particle line at ``y`` spanning +/-half_width."""
    xs = torch.arange(-half_width, half_width + 0.5 * dx, dx)
    return torch.stack([xs, torch.full_like(xs, y)], dim=1)


def wall_unit_normal(points: torch.Tensor, y: float = 0.0) -> torch.Tensor:
    """Exact wall normal of the analytic line wall, oriented towards the fluid."""
    n = torch.zeros((points.shape[0], 2), dtype=points.dtype)
    n[:, 1] = torch.where(points[:, 1] >= y, 1.0, -1.0)
    return n


def angle_to(result: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Unsigned angle between rows of two unit-ish vectors, in radians."""
    r = result / result.norm(dim=1, keepdim=True).clamp_min(1e-30)
    q = reference / reference.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return torch.acos((r * q).sum(dim=1).clamp(-1.0, 1.0))


def abs_cosine(result: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """|cos| of the angle between rows, the quantity the contact angle fixes.

    The sign of the normal is not physical -- grad c may point either way -- so
    the contact angle constrains only ``|n . n_wall|``; comparing unsigned angles
    would conflate a flipped normal with a wrong angle.
    """
    r = result / result.norm(dim=1, keepdim=True).clamp_min(1e-30)
    q = reference / reference.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return (r * q).sum(dim=1).abs()


class TestWorktreeIsUnderTest:
    def test_package_comes_from_this_worktree(self):
        """`import tensorsph` must resolve to this worktree, not a sibling one."""
        import tensorsph

        root = Path(__file__).resolve().parents[1]
        assert Path(tensorsph.__file__).resolve().is_relative_to(root), (
            f"tensorsph imported from {tensorsph.__file__}; the suite would be "
            "scoring another agent's worktree"
        )

    def test_contact_angle_bc_is_exported(self):
        import tensorsph

        assert "apply_contact_angle_boundary" in tensorsph.__all__
        assert hasattr(tensorsph, "apply_contact_angle_boundary")


class TestAnalyticAngle:
    """On-wall particles get ``w = 1``, so the target angle is met exactly."""

    @pytest.mark.parametrize(
        "theta_deg", [30.0, 60.0, 90.0, 120.0, 150.0]
    )
    def test_normal_makes_the_requested_angle_with_the_wall(self, theta_deg):
        theta = math.radians(theta_deg)
        wall = bottom_wall()
        pos = torch.tensor([[0.01, 0.0], [-0.05, 0.0]], dtype=torch.float64)
        grad = torch.tensor([[0.3, 1.0], [-0.7, 0.2]], dtype=torch.float64)

        out = apply_contact_angle_boundary(grad, pos, wall, H, theta)
        measured = abs_cosine(out, wall_unit_normal(pos))
        expected = torch.full_like(measured, abs(math.cos(theta)))

        assert torch.allclose(measured, expected, atol=1e-12), (
            f"theta={theta_deg}deg: |cos| measured {measured.tolist()} vs "
            f"expected {expected.tolist()}"
        )

    def test_neutral_angle_makes_the_interface_wall_normal(self):
        """theta = pi/2 means the interface is perpendicular to the wall."""
        wall = bottom_wall()
        pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        grad = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

        out = apply_contact_angle_boundary(grad, pos, wall, H, math.pi / 2)
        n_w = wall_unit_normal(pos)
        assert abs(float((out[0] / out[0].norm()) @ n_w[0])) < 1e-9

    def test_wetting_tilts_the_normal_onto_the_wall_normal(self):
        """theta -> 0 aligns the interface normal with the wall normal."""
        wall = bottom_wall()
        pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        grad = torch.tensor([[1.0, 0.0]], dtype=torch.float64)   # tangential input

        out = apply_contact_angle_boundary(grad, pos, wall, H, 1e-6)
        out = out[0] / out[0].norm()
        assert abs(float(out[1])) > 1.0 - 1e-6       # aligned with +y
        assert abs(float(out[0])) < 1e-3

    def test_angle_error_uses_the_correct_sense_of_rotation(self):
        """theta < pi/2 must increase |n . n_wall|, theta > pi/2 must decrease it."""
        wall = bottom_wall()
        pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        grad = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        n_w = wall_unit_normal(pos)

        before = abs(float((grad[0] / grad[0].norm()) @ n_w[0]))
        wet = apply_contact_angle_boundary(grad, pos, wall, H, math.radians(30))
        dry = apply_contact_angle_boundary(grad, pos, wall, H, math.radians(120))
        wet = abs(float((wet[0] / wet[0].norm()) @ n_w[0]))
        dry = abs(float((dry[0] / dry[0].norm()) @ n_w[0]))

        assert wet > before > dry


class TestInvariants:
    def test_magnitude_is_preserved_everywhere(self):
        wall = bottom_wall()
        torch.manual_seed(0)
        pos = torch.rand(40, 2, dtype=torch.float64) * 0.05
        pos[:, 1] = pos[:, 1].abs()
        grad = torch.randn(40, 2, dtype=torch.float64)

        out = apply_contact_angle_boundary(grad, pos, wall, H, math.radians(45))
        assert torch.allclose(out.norm(dim=1), grad.norm(dim=1), atol=1e-12)

    def test_far_particles_are_untouched(self):
        wall = bottom_wall()
        pos = torch.tensor([[0.0, REACH * 1.5], [0.0, 1.0]], dtype=torch.float64)
        grad = torch.tensor([[1.0, 0.5], [-0.2, 0.9]], dtype=torch.float64)

        out = apply_contact_angle_boundary(grad, pos, wall, H, math.radians(30))
        assert torch.equal(out, grad)

    def test_empty_wall_is_the_identity(self):
        grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        pos = torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float64)
        empty = torch.empty((0, 2), dtype=torch.float64)

        out = apply_contact_angle_boundary(grad, pos, empty, H, math.radians(45))
        assert torch.equal(out, grad)

    def test_zero_gradient_is_left_zero(self):
        wall = bottom_wall()
        pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        grad = torch.zeros((1, 2), dtype=torch.float64)

        out = apply_contact_angle_boundary(grad, pos, wall, H, math.radians(60))
        assert torch.equal(out, grad)

    def test_correction_fades_with_distance(self):
        """The angular deviation must shrink monotonically away from the wall."""
        wall = bottom_wall()
        theta = math.radians(30)
        grad = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        deviations = []
        for frac in (0.0, 0.25, 0.5, 0.75, 0.99):
            pos = torch.tensor([[0.0, frac * REACH]], dtype=torch.float64)
            out = apply_contact_angle_boundary(grad, pos, wall, H, theta)
            dev = angle_to(out, grad).item()
            deviations.append(dev)

        assert deviations[0] > 0.1
        assert all(a >= b - 1e-9 for a, b in zip(deviations, deviations[1:])), deviations
        assert deviations[-1] < deviations[0]

    def test_input_tensor_is_not_modified_in_place(self):
        wall = bottom_wall()
        pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        grad = torch.tensor([[1.0, 0.3]], dtype=torch.float64)
        keep = grad.clone()

        apply_contact_angle_boundary(grad, pos, wall, H, math.radians(75))
        assert torch.equal(grad, keep)


class TestIntegrationWithTheForce:
    """A real lattice patch: the configuration must support a non-zero curvature."""

    @staticmethod
    def _patch(radius=0.10, y0=0.0, dx=DX):
        """Hexagonal half-disc resting on the wall at ``y0``."""
        dy = dx * math.sqrt(3.0) / 2.0
        pts = []
        row = 0
        y = 0.0
        while y <= radius:
            offset = 0.5 * dx if row % 2 else 0.0
            x = -radius + offset
            while x <= radius:
                if x * x + y * y <= radius * radius:
                    pts.append((x, y0 + y))
                x += dx
            y += dy
            row += 1
        pos = torch.tensor(pts, dtype=torch.float64)
        mass = torch.full((pos.shape[0],), 1000.0 * dx * dy, dtype=torch.float64)
        return pos, mass

    def _fields(self):
        from tensorsph.neighbor_search import cell_linked_pairs

        pos, mass = self._patch()
        pairs = cell_linked_pairs(pos, 2.0 * H)
        rho = torch.full((pos.shape[0],), 1000.0, dtype=torch.float64)
        wall = bottom_wall()
        return pos, mass, rho, pairs, wall

    def test_patch_carries_a_nonzero_capillary_force(self):
        pos, mass, rho, pairs, wall = self._fields()
        cfg = Config(dx=DX, surface_tension_coefficient=0.072, gravity=0.0,
                     contact_angle=None)
        acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg, wall=wall)
        assert acc.abs().max() > 0.0

    def test_contact_angle_changes_the_near_wall_force_only(self):
        pos, mass, rho, pairs, wall = self._fields()
        base = Config(dx=DX, surface_tension_coefficient=0.072, gravity=0.0,
                      contact_angle=None)
        wet = Config(dx=DX, surface_tension_coefficient=0.072, gravity=0.0,
                     contact_angle=math.radians(40))

        a0 = compute_surface_tension_force(pos, rho, mass, pairs, base, wall=wall)
        a1 = compute_surface_tension_force(pos, rho, mass, pairs, wet, wall=wall)

        height = pos[:, 1] - wall[:, 1].max()
        near = height <= REACH
        upper = height > REACH
        assert near.any() and upper.any()

        # The BC acts on the wall band.
        assert not torch.allclose(a0[near], a1[near], atol=1e-12)

        # It reaches one support radius further through kappa, because the
        # curvature at a particle uses the normals of its neighbours; it must
        # nevertheless decay, not leak into the interior.
        delta = (a1 - a0).abs().max(dim=1).values
        assert delta[upper].max() < delta[near].max(), (
            delta[near].max().item(), delta[upper].max().item()
        )

    def test_contact_angle_without_a_wall_is_a_no_op(self):
        pos, mass, rho, pairs, wall = self._fields()
        base = Config(dx=DX, surface_tension_coefficient=0.072, gravity=0.0,
                      contact_angle=None)
        wet = Config(dx=DX, surface_tension_coefficient=0.072, gravity=0.0,
                     contact_angle=math.radians(40))

        a0 = compute_surface_tension_force(pos, rho, mass, pairs, base)
        a1 = compute_surface_tension_force(pos, rho, mass, pairs, wet)
        assert torch.equal(a0, a1)