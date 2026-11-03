"""B3 surface-tension verification: Young-Laplace pressure jump on a static droplet.

These tests (1) hold the *verification instrument* to its own standard and
(2) pin the currently measured behaviour of the merged CSF model, so that any
future change to the curvature / colour-field treatment shows up here.

History (why the pins changed at m2730)
---------------------------------------
The harness used to rebuild the colour field locally with the m277
neighbour-count heuristic and score *that*:

    N_delta = 0.34   (|grad c| under-normalised 3x)
    kappa R = 1.46 .. 1.72 (and sign partly wrong)
    normal alignment = 0.99 but force 5x too weak

while ``compute_surface_tension_force`` had already switched to
``csf_scheme="consistent"`` (the ``Config`` default, m2188).  The tests below
therefore pinned the failure of a pipeline that no longer produced the force,
and they disagreed with the pressure jump measured from the force itself
(``dp/dp_theory = 0.99``).  Since m2730 the harness scores the shipped fields
(``tensorsph.csf_consistent.csf_fields``) and keeps the legacy numbers under
explicit ``legacy_*`` keys as the A/B reference, so the pins below now describe
the model that actually runs.
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
    assert coarse["n_particles"] == 363
    assert abs(coarse["mass_error_vs_rho0_area"]) < 0.05
    assert 0.9 < coarse["density_max_ratio"] <= 1.05


def test_force_is_linear_in_sigma():
    a = H.run_case(radius=0.1, dx=0.01, sigma=0.072, lattice="hex")
    b = H.run_case(radius=0.1, dx=0.01, sigma=0.72, lattice="hex")
    assert b["dp_model_Pa"] / a["dp_model_Pa"] == pytest.approx(10.0, rel=1e-9)
    assert b["dp_ratio"] == pytest.approx(a["dp_ratio"], rel=1e-9)


def test_zero_sigma_produces_no_force():
    from tensorsph.neighbor_search import cell_linked_pairs

    dx = 0.01
    art = torch.arange(4, dtype=torch.float64)
    pos = torch.stack([art * dx, torch.zeros_like(art)], dim=1)
    mass = torch.full((4,), 1000.0 * dx * dx, dtype=torch.float64)
    rho = torch.full((4,), 1000.0, dtype=torch.float64)
    cfg = Config(dx=dx, gravity=0.0, surface_tension_coefficient=0.0)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    assert torch.equal(acc, torch.zeros_like(acc))


def test_instrument_recovers_ndelta_with_exact_normal(coarse, medium):
    """The radial integration itself must be exact: with kappa = 1/R and the
    exact normal rhat the same CSF form has to return N_delta."""
    for case in (coarse, medium):
        assert case["instrument_B_exact_kappa_exact_normal_ratio"] == pytest.approx(
            case["N_delta"], rel=1e-3)


def test_instrument_satisfies_laplace_with_correct_operators(coarse, medium):
    for case in (coarse, medium):
        assert case["instrument_C_exact_operators_ratio"] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# the shipped model against the three analytic references (m2730 re-measurement)
# --------------------------------------------------------------------------- #
def test_current_model_reproduces_laplace_criterion(coarse, medium):
    """The consistent CSF lands inside the 15 % platform tolerance at both
    resolutions (measured dp/dp_theory = 0.986 and 1.109)."""
    assert coarse["within_15pct"] is True
    assert medium["within_15pct"] is True
    assert coarse["dp_ratio"] == pytest.approx(0.99, abs=0.03)
    assert medium["dp_ratio"] == pytest.approx(1.11, abs=0.05)


def test_curvature_converges_toward_one_over_R(coarse, medium):
    """kappa * R = 1.15 (dx=0.01) -> 1.12 (dx=0.005): inside 25 % and moving in
    the right direction (the legacy estimator was 1.46 -> 1.72, i.e. diverging)."""
    for case in (coarse, medium):
        assert 1.05 < case["kappa_delta_weighted_times_R"] < 1.25
    err_coarse = abs(coarse["kappa_delta_weighted_times_R"] - 1.0)
    err_medium = abs(medium["kappa_delta_weighted_times_R"] - 1.0)
    assert err_medium < err_coarse
    # the legacy estimator must keep diverging -- this is the A/B control that
    # stopped the stale-pin mistake from going unnoticed
    assert abs(medium["legacy_kappa_delta_weighted_times_R"] - 1.0) > abs(
        coarse["legacy_kappa_delta_weighted_times_R"] - 1.0)


def test_curvature_has_the_right_sign_on_the_interface(coarse, medium):
    """A convex droplet needs kappa > 0 everywhere on the interface."""
    for case in (coarse, medium):
        assert case["kappa_model_median"] > 0.0
        assert case["kappa_model_positive_fraction"] > 0.95


def test_surface_delta_is_nearly_normalised(coarse, medium):
    """int |grad c| dn = 1 across the interface; the consistent colour field
    reaches 0.79-0.83 (legacy: 0.34-0.38, i.e. 3x too small)."""
    for case in (coarse, medium):
        assert 0.70 < case["N_delta"] < 0.95
        assert 0.30 < case["legacy_N_delta"] < 0.45
        assert case["N_delta"] > 1.8 * case["legacy_N_delta"]
        assert case["fields_under_test"].startswith("tensorsph.csf_consistent")


def test_normal_field_is_radially_coherent(coarse, medium):
    for case in (coarse, medium):
        assert case["normal_alignment_delta_weighted"] > 0.99
        assert case["normal_coherent_fraction"] > 0.99


def test_resolvability_audit_rejects_pressure_based_measurement():
    """Why the jump is measured by integrating the force and not by reading the
    Tait pressure: the density signal of dp = sigma/R is below the summation
    noise of a lattice."""
    audit = H.laplace_resolvability()
    assert audit["density_ratio_signal"] < 1e-5
    assert audit["verdict"] == "not resolvable"
    assert math.isfinite(audit["signal_to_noise"])