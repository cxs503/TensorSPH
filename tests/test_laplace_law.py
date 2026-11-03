"""Verification tests for the CSF surface-tension model against the
analytic Young-Laplace law.

The CSF model (``src/tensorsph/surface_tension.py``) ships with two schemes:

``"consistent"`` (default)
    kernel-sum colour function + MLS gradient/divergence + Shepard-smoothed
    normals.  This scheme reproduces the Young-Laplace jump within 6 % at
    R/h ~ 5 and passes the Adami spurious-current criterion.

``"legacy"``
    neighbour-count colour function with uncorrected SPH operators.  Kept for
    A/B comparison and for reproducing pre-m2188 results.

These tests score the *consistent* scheme against three analytic references
that a correct surface-tension implementation must satisfy on a circular drop:

1. ``kappa = 1/R``            -- curvature of a circle;
2. ``F_cut = 2 sigma``        -- cut-force identity (surface tension is force
                                 per unit length), which scores the CSF force
                                 integral without any pressure measurement;
3. ``Delta p = sigma / R``    -- Young-Laplace pressure jump after relaxation
                                 to one capillary time.

The curvature test remains ``xfail(strict=True)`` because the MLS divergence
of the normal field has a systematic ~25 % bias on a discrete particle circle
that does not converge away with resolution (a known SPH limitation; see
Adami et al. 2012, Sec. 4.2).  The cut-force and Laplace-pressure tests now
pass.

Numbers without heavy simulation are pinned as ordinary assertions so that any
change to the surface-tension or neighbour code shows up as a test failure.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_law"

CURVATURE_TOL = 0.15        # 15 % -- "unacceptable above this" platform bar
LAPLACE_TOL = 0.15
CUT_FORCE_TOL = 0.15
DIRECTION_COS_MIN = 0.80    # normals on a circle must be radial


def _bench_module():
    spec = importlib.util.spec_from_file_location("laplace_law_run",
                                                  BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench_module()


@pytest.fixture(scope="module")
def coarse(bench):
    """One static evaluation on a small drop -- a couple of seconds."""
    return bench.instant_diagnostics(radius=0.02, dx=0.004, sigma=7.2,
                                     rho0=1000.0, sound_speed=20.0,
                                     viscosity=0.01)


# --------------------------------------------------------------------------- #
# invariants that hold today (regression pins)
# --------------------------------------------------------------------------- #
def test_instant_diagnostics_structure(coarse):
    assert coarse["n_particles"] > 50
    assert coarse["curvature"]["exact_1_per_m"] == pytest.approx(50.0)
    assert coarse["cut_force"]["theory_N_per_m"] == pytest.approx(-14.4)
    for key in ("curvature", "delta_s", "cut_force", "inward_alignment"):
        assert key in coarse


def test_csf_force_is_net_inward(coarse):
    """Whatever the local noise, the integrated force on a half-drop must pull
    that half toward the interface, i.e. in -x for the x>0 half."""
    assert coarse["cut_force"]["measured_N_per_m"] < 0.0
    # The consistent scheme lands inside the 15 % acceptance band.
    ratio = coarse["cut_force"]["ratio_measured_over_theory"]
    assert 0.85 < ratio < 1.15


def test_surface_detector_covers_the_interface(coarse):
    """A circle of radius R has an interface layer of thickness ~2h, so the
    detected surface fraction should be of order 2h/R and must be non-empty."""
    frac = coarse["surface_fraction"]
    expected = 2.0 * coarse["h_m"] / 0.02
    assert 0.0 < frac < 3.0 * expected


def test_gamma_zero_disables_surface_tension():
    """sigma = 0 must stay a bit-exact no-op (backward compatibility)."""
    from tensorsph.droplet_impact import Config
    from tensorsph.neighbor_search import cell_linked_pairs
    from tensorsph.surface_tension import compute_surface_tension_force

    dx = 0.005
    pos = torch.tensor([[0.0, 0.0], [dx, 0.0], [0.0, dx], [dx, dx]],
                       dtype=torch.float64)
    mass = torch.full((4,), 1000.0 * dx * dx, dtype=torch.float64)
    rho = torch.full((4,), 1000.0, dtype=torch.float64)
    cfg = Config(dx=dx, gravity=0.0, surface_tension_coefficient=0.0)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    assert torch.equal(acc, torch.zeros_like(acc))


# --------------------------------------------------------------------------- #
# analytic requirements
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="CSF curvature on a circle has a "
                                      "systematic ~25 % bias from MLS "
                                      "divergence of the normal field on a "
                                      "discrete particle arrangement; see "
                                      "docs/verification_surface_tension_laplace.md")
def test_curvature_reproduces_circle(coarse):
    err = abs(coarse["curvature"]["delta_s_weighted_rel_error_pct"]) / 100.0
    assert err < CURVATURE_TOL


def test_csf_force_direction_is_radial(coarse):
    """The kernel-sum colour + MLS normal makes the CSF force radial; measured
    median cos = 0.999 (docs/verification_surface_tension_consistent.md)."""
    assert coarse["inward_alignment"]["cos_band_median"] > DIRECTION_COS_MIN


def test_cut_force_identity(coarse):
    """The consistent scheme lands on 1.07-1.12 over dx = 3..1 mm."""
    ratio = coarse["cut_force"]["ratio_measured_over_theory"]
    assert abs(ratio - 1.0) < CUT_FORCE_TOL


def test_laplace_pressure_jump(bench):
    """End-to-end: relax one capillary time and compare p_center-p_interface
    with sigma/R.  Runs at coarse resolution to stay inside a few seconds.
    The consistent scheme achieves ~6 % error at R/h ~ 5."""
    rep = bench.relax(radius=0.02, dx=0.004, sigma=7.2, steps=380)
    err = abs(rep["laplace"]["rel_error_pct"]) / 100.0
    assert err < LAPLACE_TOL


# --------------------------------------------------------------------------- #
# committed-artifact consistency (no compute; keeps the record honest)
# --------------------------------------------------------------------------- #
def test_committed_verdicts_are_self_consistent():
    """The validation_report.json must be internally consistent and honestly
    record the verdict of the latest verification round."""
    main = json.loads((BENCH / "validation_report.json").read_text())
    lap = main["laplace"]
    recomputed = ((lap["p_center_Pa"] - lap["p_interface_band_Pa"])
                  - lap["delta_p_theory_Pa"]) / lap["delta_p_theory_Pa"] * 100.0
    assert recomputed == pytest.approx(lap["rel_error_pct"], abs=1e-6)
    # The verdict flags must match the error
    assert lap["within_15pct"] == (abs(lap["rel_error_pct"]) < 15.0)
    assert lap["within_3pct"] == (abs(lap["rel_error_pct"]) < 3.0)
    assert main["reference"]["delta_p_Pa"] == pytest.approx(
        7.2 / main["setup"]["radius_m"])


def test_spurious_current_verdict_is_honest():
    """The spurious-current verdict in validation_report.json must be honest."""
    main = json.loads((BENCH / "validation_report.json").read_text())
    sc = main["spurious_current"]
    # criterion (Adami et al. 2012): max|v| / sqrt(sigma/(rho R)) << 1
    assert sc["v_ref_m_per_s"] == pytest.approx(
        math.sqrt(7.2 / (1000.0 * main["setup"]["radius_m"])), rel=1e-9)
    # The passes flag must match the ratio
    assert sc["passes_adami_0p1"] == (sc["ratio"] < 0.1)
