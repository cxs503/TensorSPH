"""B3 surface-tension verification: Young-Laplace pressure jump on a static droplet.

These tests verify the **consistent** CSF scheme (kernel-sum colour + MLS
operators, merged in m2188) against the 2-D Laplace law ``dp = sigma / R``.

The legacy scheme (neighbour-count colour, uncorrected SPH) is known to fail
the Laplace law; its field analysis is still reported by the benchmark for
historical comparison but is no longer the subject of pass/fail gates.

Measured state of the consistent scheme (this file, 2026-11-03):
- ``dp_ratio`` at R/dx = 10: 0.986 (within 15 %)
- ``dp_ratio`` at R/dx = 20: 1.109 (within 15 %)
- ``consistent_kappa_median_times_R``: ~1.18 (18 % high, outside 15 %)
- ``consistent_normal_alignment``: 0.99 (excellent)
- ``consistent_N_delta``: ~0.34 (under-normalised; see csf_consistent.py docstring)
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config
from tensorsph.surface_tension import compute_surface_tension_force

_RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_droplet" / "run.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("laplace_droplet_run", _RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_harness()


@pytest.fixture(scope="module")
def coarse():
    """Consistent scheme, R/dx = 10."""
    return H.run_case(radius=0.1, dx=0.01, sigma=0.072, lattice="hex")


@pytest.fixture(scope="module")
def medium():
    """Consistent scheme, R/dx = 20."""
    return H.run_case(radius=0.1, dx=0.005, sigma=0.072, lattice="hex")


@pytest.fixture(scope="module")
def coarse_legacy():
    """Legacy scheme, R/dx = 10 (for comparison)."""
    return H.run_case(radius=0.1, dx=0.01, sigma=0.072, lattice="hex", scheme="legacy")


# --------------------------------------------------------------------------- #
# Geometry and basic sanity
# --------------------------------------------------------------------------- #

def test_geometry_density_and_mass(coarse):
    """The disc must carry rho0 * pi R^2 of mass and reach rho0 inside."""
    assert coarse["n_particles"] == 363
    assert abs(coarse["mass_error_vs_rho0_area"]) < 0.05
    assert 0.9 < coarse["density_max_ratio"] <= 1.05


def test_force_is_linear_in_sigma():
    """CSF force is linear in sigma, so the implied jump must scale with it."""
    a = H.run_case(radius=0.1, dx=0.01, sigma=0.072, lattice="hex")
    b = H.run_case(radius=0.1, dx=0.01, sigma=0.72, lattice="hex")
    assert b["dp_model_Pa"] / a["dp_model_Pa"] == pytest.approx(10.0, rel=1e-9)
    assert b["dp_ratio"] == pytest.approx(a["dp_ratio"], rel=1e-9)


def test_zero_sigma_produces_no_force():
    cfg = Config(dx=0.01, surface_tension_coefficient=0.0, gravity=0.0)
    pos, mass, _ = H.droplet_lattice(0.1, 0.01, "hex")
    from tensorsph.droplet_impact import density_from_neighbor_pairs
    from tensorsph.neighbor_search import cell_linked_pairs
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    assert torch.equal(acc, torch.zeros_like(acc))


# --------------------------------------------------------------------------- #
# Instrument validation (the measurement itself is correct)
# --------------------------------------------------------------------------- #

def test_instrument_recovers_ndelta_with_exact_normal(coarse, medium):
    """Instrument B: exact kappa and exact normal must return N_delta exactly."""
    for case in (coarse, medium):
        assert case["instrument_B_exact_kappa_exact_normal_ratio"] == pytest.approx(
            case["consistent_N_delta"], rel=1e-9)


def test_instrument_satisfies_laplace_with_correct_operators(coarse, medium):
    """Instrument C: exact operators + unit delta_s must return the Laplace law."""
    for case in (coarse, medium):
        assert case["instrument_C_exact_operators_ratio"] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Consistent scheme: Laplace law verification
# --------------------------------------------------------------------------- #

def test_consistent_scheme_passes_laplace_at_coarse(coarse):
    """The consistent scheme reproduces dp = sigma/R within 15 % at R/dx = 10."""
    assert coarse["within_15pct"] is True
    assert coarse["dp_ratio"] == pytest.approx(0.986, abs=0.02)


def test_consistent_scheme_passes_laplace_at_medium(medium):
    """The consistent scheme reproduces dp = sigma/R within 15 % at R/dx = 20."""
    assert medium["within_15pct"] is True
    assert medium["dp_ratio"] == pytest.approx(1.109, abs=0.03)


def test_consistent_curvature_is_positive(coarse, medium):
    """A convex droplet needs kappa > 0; the consistent scheme delivers."""
    for case in (coarse, medium):
        assert case["consistent_kappa_median"] > 0.0
        # kappa*R should be close to 1.0 (measured ~1.18, i.e. 18 % high)
        assert 0.8 < case["consistent_kappa_median_times_R"] < 1.5


def test_consistent_normal_is_coherent(coarse, medium):
    """The consistent scheme's normals point outward coherently."""
    for case in (coarse, medium):
        assert case["consistent_normal_alignment"] > 0.95


def test_consistent_color_function_spans_interface(coarse):
    """Kernel-sum colour goes from ~0.6 at the surface to 1.0 in the bulk."""
    assert coarse["consistent_color_min"] < 0.7
    assert coarse["consistent_color_max"] == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Legacy scheme: known defects (for regression / comparison)
# --------------------------------------------------------------------------- #

def test_legacy_scheme_fails_laplace(coarse_legacy):
    """The legacy scheme is far outside the 15 % tolerance."""
    assert coarse_legacy["within_15pct"] is False
    # Legacy dp_ratio is ~0.62 (38 % low) -- still well outside 15 %
    assert coarse_legacy["dp_ratio"] < 0.80


def test_legacy_delta_is_under_normalised(coarse_legacy):
    """Legacy N_delta is ~0.34 instead of 1.0."""
    assert 0.2 < coarse_legacy["legacy_N_delta"] < 0.5


def test_legacy_curvature_differs_from_consistent(coarse, coarse_legacy):
    """Legacy curvature differs from the consistent scheme's curvature."""
    # Legacy kappa*R ~1.56, consistent ~1.19 -- both positive but different
    assert abs(coarse_legacy["legacy_kappa_median_times_R"] -
               coarse["consistent_kappa_median_times_R"]) > 0.2


# --------------------------------------------------------------------------- #
# Consistent vs legacy comparison
# --------------------------------------------------------------------------- #

def test_consistent_improves_over_legacy(coarse, coarse_legacy):
    """The consistent scheme's dp_ratio is much closer to 1.0 than legacy."""
    cons_err = abs(coarse["dp_ratio"] - 1.0)
    leg_err = abs(coarse_legacy["dp_ratio"] - 1.0)
    assert cons_err < leg_err * 0.3  # consistent error is < 30 % of legacy error


def test_consistent_curvature_closer_to_exact(coarse, coarse_legacy):
    """Consistent |kappa*R - 1| is smaller than legacy |kappa*R - 1|."""
    cons_err = abs(coarse["consistent_kappa_median_times_R"] - 1.0)
    leg_err = abs(coarse_legacy["legacy_kappa_median_times_R"] - 1.0)
    assert cons_err < leg_err


# --------------------------------------------------------------------------- #
# Resolvability audit
# --------------------------------------------------------------------------- #

def test_resolvability_audit_rejects_pressure_based_measurement():
    """A density-based Laplace measurement is below SPH noise for water."""
    audit = H.laplace_resolvability(sigma=0.072, radius=0.1, c0=20.0)
    assert audit["density_ratio_signal"] < 1e-5
    assert audit["verdict"] == "not resolvable"
    assert math.isfinite(audit["signal_to_noise"])
