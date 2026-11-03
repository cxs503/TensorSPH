"""B3 surface-tension verification: Young-Laplace pressure jump on a static droplet.

These tests (1) hold the *verification instrument* to its own standard and
(2) pin the currently measured behaviour of the merged CSF model, so that a
future fix of the curvature / colour-field treatment shows up as a change here.

Measured state of the merged model (see docs/benchmarks/laplace-droplet.md):
it does **not** reproduce the Laplace jump; the tests below encode that fact
rather than a passing criterion.
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
    return H.run_case(radius=0.1, dx=0.01, sigma=0.072, lattice="hex")


@pytest.fixture(scope="module")
def medium():
    return H.run_case(radius=0.1, dx=0.005, sigma=0.072, lattice="hex")


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


def test_instrument_recovers_ndelta_with_exact_normal(coarse, medium):
    """Instrument B: exact kappa and exact normal must return N_delta exactly."""
    for case in (coarse, medium):
        assert case["instrument_B_exact_kappa_exact_normal_ratio"] == pytest.approx(
            case["N_delta"], rel=1e-9)


def test_instrument_satisfies_laplace_with_correct_operators(coarse, medium):
    """Instrument C: exact operators + unit delta_s must return the Laplace law."""
    for case in (coarse, medium):
        assert case["instrument_C_exact_operators_ratio"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.xfail(strict=False, reason="pins the pre-m2592 CSF deficit; "
                                        "superseded by the curvature fix "
                                        "(docs/verification_csf_curvature_bias.md)")
def test_current_model_fails_laplace_criterion(coarse, medium):
    """Pinned regression: the merged model is far outside the 15 % tolerance."""
    assert coarse["within_15pct"] is False
    assert medium["within_15pct"] is False
    assert coarse["dp_ratio"] == pytest.approx(0.43, abs=0.05)
    assert medium["dp_ratio"] == pytest.approx(1.39, abs=0.10)


@pytest.mark.xfail(strict=False, reason="pins the pre-m2592 CSF deficit; "
                                        "superseded by the curvature fix "
                                        "(docs/verification_csf_curvature_bias.md)")
def test_curvature_error_grows_under_refinement(coarse, medium):
    """div(normal) curvature is not convergent: |kappa R| grows with resolution."""
    assert abs(medium["kappa_delta_weighted_times_R"]) > \
        1.5 * abs(coarse["kappa_delta_weighted_times_R"])


@pytest.mark.xfail(strict=False, reason="pins the pre-m2592 CSF deficit; "
                                        "superseded by the curvature fix "
                                        "(docs/verification_csf_curvature_bias.md)")
def test_curvature_has_the_wrong_sign_on_the_interface(coarse, medium):
    """A convex droplet needs kappa > 0; the model returns a negative median."""
    for case in (coarse, medium):
        assert case["kappa_model_median"] < 0.0
        assert 0.3 < case["kappa_model_positive_fraction"] < 0.7


def test_surface_delta_is_under_normalised(coarse, medium):
    """CSF needs integral |grad c| dr = 1; the model's colour field carries ~1/3."""
    for case in (coarse, medium):
        assert 0.2 < case["N_delta"] < 0.5


@pytest.mark.xfail(strict=False, reason="pins the pre-m2592 CSF deficit; "
                                        "superseded by the curvature fix "
                                        "(docs/verification_csf_curvature_bias.md)")
def test_normal_field_is_not_radially_coherent(coarse, medium):
    """The model's normals do not point outwards coherently (this is the defect)."""
    for case in (coarse, medium):
        assert abs(case["normal_alignment_delta_weighted"]) < 0.1


def test_resolvability_audit_rejects_pressure_based_measurement():
    """A density-based Laplace measurement is below SPH noise for water."""
    audit = H.laplace_resolvability(sigma=0.072, radius=0.1, c0=20.0)
    assert audit["density_ratio_signal"] < 1e-5
    assert audit["verdict"] == "not resolvable"
    assert math.isfinite(audit["signal_to_noise"])
