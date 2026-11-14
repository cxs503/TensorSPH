"""Round-3 invariants for ``CASE-SPH-OSC-DROP-FREQ``.

Three layers, each of which a future change can break independently:

1. **projection invariant** -- an *analytically exact* mode-l surface must score
   ``omega` error < 1.5 %``.  Round 2's projection does not: it keeps the
   equilibrium curvature ``2 sigma/R`` while projecting on the deformed measure,
   which adds an ``O(a)`` term of the same size as the physics, so an error-free
   geometry scores ``+41.6 %`` (l=2) / ``+18.3 %`` (l=3).  The old convention is
   kept in the module behind two flags and pinned here as a negative control, so
   nobody "fixes" the flags back.
2. **field invariant** -- :func:`tensorsph.interface_level_set.color_at_points`
   must reproduce the shipped CSF colour ``csf_state(...).color`` *at particle
   centres* to machine precision: the reconstruction is allowed to interpolate
   the repository's field, not to invent a different one.
3. **measured numbers** -- the transfer law ``1 - 1.9/(R/dx)`` and the gate claim
   (``< 6 %`` at ``R/dx >= 24`` for both modes and both amplitudes) are pinned to
   the values in ``benchmarks/oscillating_drop_freq/round3.json``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "oscillating_drop_freq"))

from tensorsph.interface_level_set import (  # noqa: E402
    COLOR_LEVEL,
    color_at_points,
    level_set_radius,
    measure_revolution,
    modal_capillary_stress,
    reconstructed_surface,
    rms_amplitude,
)
from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_drop_geometry,
    exact_modal_stress,
)
from tensorsph.surface_tension_dispersion import csf_state  # noqa: E402

R = 0.02
SIGMA = 7.2
RHO = 1000.0
GATE_PCT = 6.0


def _exact_surface(mode: int, eps: float, radius: float = R) -> dict:
    coef = torch.zeros(7, dtype=torch.float64)
    coef[0] = radius
    coef[mode] = eps * radius
    return reconstructed_surface(coef, 6, radius=radius)


# --------------------------------------------------------------------------- #
# 1. projection invariant
# --------------------------------------------------------------------------- #
class TestProjection:
    @pytest.mark.parametrize("mode", [2, 3, 4])
    @pytest.mark.parametrize("eps", [0.01, 0.02])
    def test_exact_surface_must_score_its_own_geometry(self, mode, eps):
        """The operator's own geometry, no measurement: the answer is 0 % error."""
        st = modal_capillary_stress(_exact_surface(mode, eps), sigma=SIGMA,
                                    radius=R, mode=mode)
        dp_exact = exact_modal_stress(SIGMA, R, mode, eps)
        err = (st["dp"] / dp_exact - 1.0) * 100.0
        assert abs(err) < 1.5, f"l={mode} eps={eps}: exact surface scored {err:+.2f} %"

    def test_sphere_curvature_and_area_are_exact(self):
        surf = reconstructed_surface(torch.tensor([R, 0, 0, 0, 0, 0, 0.0],
                                                  dtype=torch.float64), 6, radius=R)
        assert float((surf["kappa"] * surf["darea"]).sum() / surf["darea"].sum()) * R == \
            pytest.approx(2.0, abs=1e-5)
        assert float(surf["darea"].sum()) == pytest.approx(4.0 * math.pi * R ** 2, rel=1e-6)

    def test_the_old_convention_overcounts_stiffness(self):
        """Negative control: round 2's convention doubles l=2 and 1.4x's l=3."""
        for mode, expected in ((2, 2.0), (3, 1.4)):
            surf = _exact_surface(mode, 0.02)
            old = modal_capillary_stress(surf, sigma=SIGMA, radius=R, mode=mode,
                                         subtract_base_curvature=False,
                                         reference_measure=False)
            new = modal_capillary_stress(surf, sigma=SIGMA, radius=R, mode=mode)
            assert old["dp"] / new["dp"] == pytest.approx(expected, abs=0.1)

    def test_sign_is_restoring(self):
        """A bulge must give an inward (negative) modal traction."""
        st = modal_capillary_stress(_exact_surface(2, 0.02), sigma=SIGMA, radius=R,
                                    mode=2)
        assert st["dp"] < 0.0


# --------------------------------------------------------------------------- #
# 2. field invariant
# --------------------------------------------------------------------------- #
class TestColourField:
    def test_interpolated_colour_equals_shipped_field_at_particles(self):
        geom = axisym_drop_geometry(R, R / 12.0, mode=2, eps=0.02, rho0=RHO)
        state, h, _ = csf_state(geom, SIGMA, periodic_x=False)
        c = color_at_points(geom["pos"], geom["mass"], geom["pos"], h, RHO)
        assert torch.allclose(c, state.color, atol=1e-10)

    def test_level_set_of_an_undeformed_drop_invents_no_odd_mode(self):
        geom = axisym_drop_geometry(R, R / 12.0, mode=2, eps=0.0, rho0=RHO)
        m = measure_revolution(geom, mode=3, n_rays=160, n_radial=61)
        # P_3 is odd in cos(theta) and the lattice is mirror symmetric: exactly 0
        assert abs(m["amplitude"]) < 1e-6
        # ... while the even l=2 null input is small but not zero (orientation noise)
        m2 = measure_revolution(geom, mode=2, n_rays=160, n_radial=61)
        assert abs(m2["amplitude"]) < 3.0e-3

    def test_level_set_lies_within_a_cell_of_the_true_radius(self):
        for rdx in (12.0, 20.0):
            geom = axisym_drop_geometry(R, R / rdx, mode=2, eps=0.0, rho0=RHO)
            ls = level_set_radius(geom, n_rays=120, n_radial=61)
            assert ls["crossed_fraction"] == pytest.approx(1.0)
            offset_cells = abs(float(torch.median(ls["r"])) - R) / geom["dx"]
            assert offset_cells < 1.0, f"R/dx={rdx}: offset {offset_cells:.2f} dx"

    def test_rms_estimator_brackets_the_modal_projection(self):
        """Non-modal cross-check: ``sqrt(<(r-<r>)^2>) = eps R/sqrt(2 l + 1) + noise``.

        The measured surface is the mode *plus* sub-particle jitter, and the two
        add in quadrature, so the roughness is an **upper** bound on the modal
        amplitude -- measured at ``R/dx = 20`` the jitter is ``0.008 R``
        (``0.17 dx``), i.e. +28 % on the roughness of an ``eps = 0.02`` mode.
        A sign that the reconstruction were throwing away structure rather than
        adding jitter would break the lower bound.
        """
        geom = axisym_drop_geometry(R, R / 20.0, mode=2, eps=0.02, rho0=RHO)
        ls = level_set_radius(geom, n_rays=240, n_radial=81)
        rms = rms_amplitude(ls["theta"], ls["r"])
        exact = 0.02 * R / math.sqrt(5.0)
        assert rms > 0.8 * exact
        assert rms < 1.5 * exact


# --------------------------------------------------------------------------- #
# 3. measured transfer law and the case gate
# --------------------------------------------------------------------------- #
class TestTransferLaw:
    @pytest.mark.parametrize("rdx", [16.0, 32.0])
    def test_amplitude_recovery_follows_one_minus_two_over_rinv(self, rdx):
        geom = axisym_drop_geometry(R, R / rdx, mode=3, eps=0.02, rho0=RHO)
        m = measure_revolution(geom, mode=3, n_rays=240, n_radial=81)
        predicted = 1.0 - 1.913 / rdx
        assert m["amplitude_recovery"] == pytest.approx(predicted, abs=0.03)

    @pytest.mark.parametrize("mode", [2, 3])
    def test_gate_is_met_at_r_over_dx_24(self, mode):
        """The anchor's acceptance: |omega error| < 6 % against Rayleigh 1879."""
        for eps in (0.02, 0.05):
            geom = axisym_drop_geometry(R, R / 24.0, mode=mode, eps=eps, rho0=RHO)
            m = measure_revolution(geom, mode=mode, sigma=SIGMA, n_rays=240,
                                   n_radial=81)
            dp_exact = exact_modal_stress(SIGMA, R, mode, eps)
            ratio = m["dp"] / dp_exact
            err = (math.copysign(math.sqrt(abs(ratio)), ratio) - 1.0) * 100.0
            assert abs(err) < GATE_PCT, f"l={mode} eps={eps}: {err:+.2f} %"

    def test_gate_is_not_met_at_r_over_dx_12(self):
        """Honest boundary: coarser than the transfer-law crossing must fail."""
        geom = axisym_drop_geometry(R, R / 12.0, mode=3, eps=0.02, rho0=RHO)
        m = measure_revolution(geom, mode=3, sigma=SIGMA, n_rays=240, n_radial=81)
        dp_exact = exact_modal_stress(SIGMA, R, 3, 0.02)
        err = (math.sqrt(m["dp"] / dp_exact) - 1.0) * 100.0
        assert abs(err) > GATE_PCT

    def test_color_level_constant_is_the_field_half_value(self):
        assert COLOR_LEVEL == 0.5