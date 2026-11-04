"""B3 surface-tension verification: Young-Laplace pressure jump on a static droplet.

These tests (1) hold the *verification instrument* to its own standard and
(2) pin the currently measured behaviour of the merged CSF model, so that a
future regression shows up as a change here.

History (kept as an honest record, per backlog_update_m2971.md P2):
before the m2592 CSF refactor the model's curvature had the wrong sign, the
normals were incoherent, the curvature error grew under refinement, and the
implied jump was far outside tolerance (dp_ratio ~0.43 / ~1.39). Four tests
used to pin those broken facts under ``xfail``. After the consistent-scheme
refactor (docs/verification_csf_curvature_bias.md,
docs/verification_surface_tension_curvature.md) those facts no longer hold,
so the four tests were rewritten to record the *current* state with new
criteria instead of being deleted. The remaining gap - the implied jump
sits at ~0.79-0.84 of sigma/R, i.e. 16-21 % low - is the free-surface
pressure closure (P0 in backlog_update_m2971.md, measured in
benchmarks/laplace_pressure_floor/), not the curvature operator.
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


def test_laplace_jump_current_state(coarse, medium):
    """Post-refactor state: the implied jump is ~0.79-0.84 of sigma/R.

    Pre-m2592 this test pinned the broken fact ``within_15pct is False`` with
    dp_ratio ~0.43 (coarse) / ~1.39 (medium). The consistent-scheme CSF now
    lands both resolutions on the same side of the analytic jump, within
    16-21 % of it. The residual is the free-surface pressure closure (P0 in
    backlog_update_m2971.md), so ``within_15pct`` is still False and is
    asserted as such rather than left to drift.
    """
    for case in (coarse, medium):
        assert 0.70 < case["dp_ratio"] < 0.95
        assert case["within_15pct"] is False
    # The two resolutions must agree with each other to within 10 %: the
    # pre-refactor spread (0.43 vs 1.39) is the regression this guards.
    assert coarse["dp_ratio"] == pytest.approx(medium["dp_ratio"], rel=0.10)


def test_curvature_error_bounded_under_refinement(coarse, medium):
    """Post-refactor state: |kappa| R no longer diverges under refinement.

    Pre-m2592 the delta-weighted |kappa| R grew by more than 1.5x from
    dx=0.01 to dx=0.005 (divergent operator). The current fit_quadratic
    operator gives 1.46 -> 1.72 (ratio ~1.18): the residual is a bias, not a
    divergence. The bound below is the negation of the old pinned fact.
    """
    ratio = (abs(medium["kappa_delta_weighted_times_R"])
             / abs(coarse["kappa_delta_weighted_times_R"]))
    assert ratio < 1.5
    assert 1.0 < ratio < 1.5


def test_curvature_sign_is_positive_on_the_interface(coarse, medium):
    """A convex droplet needs kappa > 0; the model now returns a positive median.

    Pre-m2592 the median was negative with a ~50/50 positive fraction. The
    current operator returns a positive median at both resolutions with the
    positive fraction well above 0.8.
    """
    for case in (coarse, medium):
        assert case["kappa_model_median"] > 0.0
        assert case["kappa_model_positive_fraction"] > 0.8


def test_surface_delta_is_under_normalised(coarse, medium):
    """CSF needs integral |grad c| dr = 1; the model's colour field carries ~1/3."""
    for case in (coarse, medium):
        assert 0.2 < case["N_delta"] < 0.5


def test_normal_field_is_radially_coherent(coarse, medium):
    """Post-refactor state: the model's normals point outwards coherently.

    Pre-m2592 the delta-weighted radial alignment was < 0.1 (incoherent).
    The current colour field gives > 0.98 at both resolutions.
    """
    for case in (coarse, medium):
        assert case["normal_alignment_delta_weighted"] > 0.9


def test_resolvability_audit_rejects_pressure_based_measurement():
    """A density-based Laplace measurement is below SPH noise for water."""
    audit = H.laplace_resolvability(sigma=0.072, radius=0.1, c0=20.0)
    assert audit["density_ratio_signal"] < 1e-5
    assert audit["verdict"] == "not resolvable"
    assert math.isfinite(audit["signal_to_noise"])
