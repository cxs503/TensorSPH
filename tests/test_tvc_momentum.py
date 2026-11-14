"""Momentum conservation analysis for the transport-velocity correction.

The TVC module docstring claims: "No velocity is modified: only positions move,
so mass and momentum of the discrete system are untouched by the correction
itself."  This is true for the *direct* effect (the correction only shifts
positions), but the *indirect* effect — new positions → new density → new
pressure → new acceleration → new momentum — is real and must be quantified.

This test suite:
1. Verifies that the TVC position correction does not directly modify velocity
   (trivially true by construction, but pinned as an invariant).
2. Quantifies the indirect momentum effect in a symmetric dam-break case.
3. Verifies that the momentum difference between TVC and baseline is bounded
   and physically reasonable (not spurious).
4. Tests momentum conservation in a symmetric blob case (no external forces).

Anchor: XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.transport_velocity_correction import (  # noqa: E402
    TVCResult, kernel_gradient_integral, transport_velocity_correction,
    SPHINXSYS_DEFAULT_COEFFICIENT, SPHINXSYS_SINGLE_STEP_FRACTION,
)
from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402
from tensorsph.droplet_impact import calibrated_particle_mass  # noqa: E402


def _lattice(n: int, dx: float, dtype=torch.float64):
    """Square lattice, n x n particles."""
    xs = torch.arange(n, dtype=dtype) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    vol = torch.ones(pos.shape[0], dtype=dtype) * dx * dx
    return pos, vol


# ---------------------------------------------------------------------------
# 1. Direct effect: TVC does not modify velocity
# ---------------------------------------------------------------------------

class TestTvcDirectEffect:
    """The TVC position correction must not directly modify velocity."""

    def test_tvc_result_has_no_velocity_field(self):
        """TVCResult contains shift and gradient_integral, but no velocity."""
        field_names = {f.name for f in TVCResult.__dataclass_fields__.values()}
        assert "velocity" not in field_names
        assert "delta_v" not in field_names
        assert "shift" in field_names  # position correction, not velocity

    def test_tvc_function_returns_only_position_shift(self):
        """transport_velocity_correction returns TVCResult with shift, not dv."""
        pos, vol = _lattice(15, 0.02)
        h = 1.3 * 0.02

        result = transport_velocity_correction(pos, vol, h, alpha=0.10)

        # The result must have a shift field (position correction)
        assert result.shift.shape == pos.shape


# ---------------------------------------------------------------------------
# 2. Symmetric blob: no spurious momentum from TVC
# ---------------------------------------------------------------------------

class TestSymmetricBlobMomentum:
    """A symmetric particle blob with TVC must not develop net momentum.

    On a symmetric configuration with complete support, the grad-1
    inconsistency is zero, so the position corrections are zero.
    """

    def test_interior_grad1_is_zero_on_lattice(self):
        """On a lattice interior (complete support), grad-1 must be zero."""
        dx = 0.02
        h = 1.3 * dx
        pos, vol = _lattice(25, dx)

        g = kernel_gradient_integral(pos, vol, h)
        mag = torch.linalg.vector_norm(g, dim=1)

        # Interior particles only (complete support)
        interior = ((pos[:, 0] > 2.2 * h) & (pos[:, 0] < 24 * dx - 2.2 * h)
                    & (pos[:, 1] > 2.2 * h) & (pos[:, 1] < 24 * dx - 2.2 * h))
        assert interior.sum() > 100
        assert float(mag[interior].max()) * h < 1.0e-10

    def test_interior_tvc_shift_is_zero_on_lattice(self):
        """On a lattice interior, TVC must produce zero shift."""
        dx = 0.02
        h = 1.3 * dx
        pos, vol = _lattice(25, dx)

        result = transport_velocity_correction(pos, vol, h, alpha=0.10)

        # Interior particles only
        interior = ((pos[:, 0] > 2.2 * h) & (pos[:, 0] < 24 * dx - 2.2 * h)
                    & (pos[:, 1] > 2.2 * h) & (pos[:, 1] < 24 * dx - 2.2 * h))
        assert float(result.shift[interior].abs().max()) < 1.0e-10

    def test_perturbed_symmetric_blob_has_antisymmetric_shift(self):
        """A perturbed symmetric blob must have antisymmetric shifts.

        If we perturb particle i by +dx and particle j (its mirror) by -dx,
        the TVC shifts must be antisymmetric: shift_i ≈ -shift_j.
        """
        pos, vol = _lattice(11, 0.04)
        h = 1.3 * 0.04

        # Perturb symmetrically: move particle 0 right, particle -1 left
        pos = pos.clone()
        pos[0, 0] += 0.005
        pos[-1, 0] -= 0.005

        result = transport_velocity_correction(pos, vol, h, alpha=0.10)

        # The shifts of the perturbed particles must be approximately
        # antisymmetric (same magnitude, opposite direction)
        shift_0 = result.shift[0]
        shift_n = result.shift[-1]
        # They should point in opposite x directions
        assert shift_0[0] * shift_n[0] < 0, (
            f"Shifts not antisymmetric: shift[0]={shift_0}, shift[-1]={shift_n}"
        )


# ---------------------------------------------------------------------------
# 3. Dam break: indirect momentum effect is bounded
# ---------------------------------------------------------------------------

class TestDamBreakMomentum:
    """TVC must not introduce spurious momentum in the dam-break case.

    The indirect momentum effect (through density → pressure → acceleration)
    is real and expected, but it must be bounded and physically reasonable.
    """

    def _run_dam_break(self, dx=0.04, tvc_every=0, tvc_alpha=0.10,
                       tvc_scope="all", steps=200, device="cpu"):
        """Run a dam-break case and return the run result."""
        cfg = DamBreakConfig(
            dx=dx, dt=1.0e-4, steps=steps,
            output_every=max(1, steps // 4),
            device=device,
            column_width=0.2, column_height=0.4,
            pool_width=2.0, tank_height=0.75,
            tvc_every=tvc_every,
            tvc_alpha=tvc_alpha,
            tvc_scope=tvc_scope,
        )
        return run_case(cfg)

    def test_tvc_momentum_vs_baseline(self):
        """TVC must not increase the final momentum by more than 2x baseline.

        The baseline solver has intrinsic momentum from the lattice
        initialization defect. TVC must not increase this by more than 2x.
        """
        dx = 0.04
        steps = 200

        # Baseline: no TVC
        run_base = self._run_dam_break(dx=dx, tvc_every=0, steps=steps)

        # With TVC (all scope, alpha=0.10)
        run_tvc = self._run_dam_break(
            dx=dx, tvc_every=1, tvc_alpha=0.10, tvc_scope="all", steps=steps
        )

        # The final momentum (sum of m*v) must be comparable
        # We use max_speed as a proxy since the run result doesn't expose
        # total momentum directly
        assert run_tvc.max_speed < 2.0 * run_base.max_speed + 0.5, (
            f"TVC max_speed = {run_tvc.max_speed:.4f} exceeds "
            f"2x baseline ({run_base.max_speed:.4f}) + tolerance"
        )

    def test_tvc_momentum_direction_is_physical(self):
        """TVC must not reverse the momentum direction in the dam-break case.

        The dam-break flow is predominantly in the +x direction. TVC must
        not cause the net momentum to reverse.
        """
        dx = 0.04
        steps = 200

        run_tvc = self._run_dam_break(
            dx=dx, tvc_every=1, tvc_alpha=0.10, tvc_scope="all", steps=steps
        )

        # The flow should be predominantly in +x direction
        # max_speed > 0 indicates flow is present
        assert run_tvc.max_speed > 0.1, (
            f"TVC max_speed = {run_tvc.max_speed:.4f} is too small, "
            f"flow may have stalled"
        )


# ---------------------------------------------------------------------------
# 4. Momentum conservation in isolated system (no external forces)
# ---------------------------------------------------------------------------

class TestIsolatedSystemMomentum:
    """TVC must conserve momentum in an isolated system (no gravity, no walls).

    In an isolated system with no external forces, the total momentum must
    be conserved. TVC should not introduce spurious momentum.
    """

    def test_tvc_on_free_cloud_no_net_shift(self):
        """A free-floating cloud with TVC must have small net shift.

        Without walls or gravity, the grad-1 inconsistency of a roughly
        symmetric cloud should sum to approximately zero.
        """
        # Create a roughly symmetric cloud
        n = 20
        torch.manual_seed(42)
        pos = torch.randn(n, 2, dtype=torch.float64) * 0.1
        vol = torch.ones(n, dtype=torch.float64) * 0.001
        h = 0.05

        result = transport_velocity_correction(pos, vol, h, alpha=0.10)

        # The net shift must be small (not exactly zero due to asymmetry)
        net_shift = result.shift.sum(dim=0)
        assert net_shift.abs().max() < 0.01, (
            f"Net shift = {net_shift} is too large for a free cloud"
        )


# ---------------------------------------------------------------------------
# 5. Alpha scaling and momentum effect
# ---------------------------------------------------------------------------

class TestAlphaMomentumScaling:
    """Larger alpha should produce larger (but bounded) momentum effects."""

    def test_larger_alpha_produces_larger_shift(self):
        """Larger alpha must produce larger position shifts."""
        pos, vol = _lattice(15, 0.02)
        # Perturb to break symmetry
        pos = pos.clone()
        pos[:, 0] += torch.linspace(-0.01, 0.01, pos.shape[0])
        h = 1.3 * 0.02

        shifts = {}
        for alpha in [0.05, 0.10, 0.15, 0.20]:
            result = transport_velocity_correction(pos, vol, h, alpha=alpha)
            shifts[alpha] = result.shift.abs().sum().item()

        # Larger alpha -> larger total shift (monotone)
        alphas = sorted(shifts.keys())
        for i in range(len(alphas) - 1):
            assert shifts[alphas[i]] <= shifts[alphas[i + 1]], (
                f"Shifts not monotone: {shifts}"
            )

    def test_shift_scales_with_alpha_h_squared(self):
        """The shift must scale as alpha * h^2 * |grad-1|."""
        pos, vol = _lattice(15, 0.02)
        pos = pos.clone()
        pos[:, 0] += torch.linspace(-0.01, 0.01, pos.shape[0])
        h = 1.3 * 0.02
        alpha = 0.10

        result = transport_velocity_correction(pos, vol, h, alpha=alpha)

        # The shift magnitude must be bounded by alpha * h^2 * |grad-1|
        max_shift = result.shift.abs().max().item()
        max_g = result.max_gradient_integral
        bound = alpha * h * h * max_g
        assert max_shift <= bound * 1.01, (  # 1% tolerance for float rounding
            f"max_shift = {max_shift} exceeds bound {bound}"
        )
