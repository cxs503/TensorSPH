"""Gates for round 5 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (curvature variation).

Rounds 1-4 verified ``dp = sigma/R`` on a **circular** droplet: an interface of
*constant* curvature, where the mean and the local form of the law coincide up to
the estimator's constant bias. Round 5 prescribes an **ellipse** of the same area
(``pi a b = pi R_eq^2``), whose curvature varies by ``(a/b)^3`` along the
interface, and reads the *differential* law ``dp(t) = sigma kappa(t)`` station by
station with two independent instruments (traction line integral; pressure
readout from the rho-blind force-density Poisson of round 4).

The gates below are the ones declared in
``scripts/xval_projection_laplace/curvature_variation.py`` (``GATES``, copied
verbatim into ``results/curvature_variation.json``) *before* the final sweep.
Two of them are broken by the measurement and are registered as strict
``xfail``s rather than relaxed:

* the station-scale (pointwise) scatter of the jump is
  ``~30-40 %`` and resolution-independent -- it is the ``MLS div n`` curvature
  noise on the ``delta_s`` shell, isolated by the ``kappa_exact`` ablation
  (which drops it to a few percent);
* the *pressure readout* reaches ``~16 %`` at ``a:b = 3:1`` where the traction
  integral on the same equilibrium is inside 1 %.

The day either is fixed these tests start failing loudly.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "xval_projection_laplace"))

RESULTS = ROOT / "scripts" / "xval_projection_laplace" / "results" / "curvature_variation.json"


@pytest.fixture(scope="module")
def data():
    if not RESULTS.exists():  # pragma: no cover - artifact is committed
        pytest.skip("curvature_variation.json not generated")
    return json.loads(RESULTS.read_text())


def _cfg(d, aspect, rdx):
    return next(c for c in d["configs"]
                if c["aspect_ratio"] == aspect and c["r_eq_over_dx"] == rdx)


# --------------------------------------------------------------------------- #
# the declared gates, replayed from the artifact
# --------------------------------------------------------------------------- #
def test_declared_gates_present_and_replayed(data):
    verdicts = data["gate_verdicts"]
    declared = data["gates_declared"]
    assert verdicts, "no gate verdicts in the artifact"
    # every declared gate has a verdict (the window-ladder bookkeeping keys excepted)
    for key in declared:
        assert any(key in k for k in verdicts), f"gate {key} has no verdict"
    # the passing set must not have shrunk silently
    passed = {k for k, v in verdicts.items() if v["passed"]}
    assert "local_bias_pct_aspect_le_1p5_resolved" in passed
    assert "whitney_graustein_err_pct_resolved" in passed
    assert "both_exact_bias_pct_aspect_le_2_resolved" in passed


def test_local_law_bias_inside_3pct_at_every_aspect(data):
    """The differential law, in the mean, on curved interfaces up to 3:1."""
    for c in data["configs"]:
        if c["r_eq_over_dx"] != 40.0:
            continue
        assert abs(c["flux_err"]["mean"]) < 3.0, c["aspect_ratio"]
    # and on the circle at every resolution (rounds 1-4 regression)
    for c in data["configs"]:
        if c["aspect_ratio"] == 1.0 and c["r_eq_over_dx"] >= 20.0:
            assert abs(c["flux_err"]["mean"]) < 3.0


def test_curvature_integral_is_two_pi(data):
    """Whitney-Graustein: oint kappa dl = 2 pi for any closed convex curve."""
    for c in data["configs"]:
        if not c["resolved"]:
            continue
        assert abs(c["whitney_graustein_err_pct"]) < 1.0, c["aspect_ratio"]


def test_closed_interface_carries_no_net_traction(data):
    for c in data["configs"]:
        assert c["net_traction_residual"] < 1e-2, c["aspect_ratio"]
        # the Poisson source of a closed interface is charge neutral on a lattice
        assert abs(c["poisson_source_charge_rel"]) < 1e-6


def test_curvature_ratio_of_the_case_is_what_the_ellipse_says(data):
    """Geometry check: kappa_max/kappa_min = (a/b)**3 exactly."""
    for c in data["configs"]:
        expected = c["aspect_ratio"] ** 3
        assert c["kappa_ratio_max_over_min"] == pytest.approx(expected, rel=1e-9)


def test_zero_sigma_has_no_traction(data):
    assert data["controls"]["zero_sigma_max_abs_dp_flux_Pa"] == 0.0


def test_circle_control_matches_the_shipped_module(data):
    """a = b must reproduce the radius-based case of rounds 1-4."""
    d = data["module_circle"]
    assert abs(d["dp_error_pct"]) < 3.0
    circle = _cfg(data, 1.0, 20.0)
    assert abs(circle["mean_dp_flux_Pa"] / d["dp_measured_Pa"] - 1.0) * 100.0 < 2.0


def test_mean_law_closes_with_analytic_curvature_and_delta(data):
    """Mechanism: with exact kappa *and* exact delta_s normalisation, <1 %."""
    for c in data["configs"]:
        if not c["resolved"] or c["aspect_ratio"] > 2.0:
            continue
        assert abs(c["ablation_both_exact"]["flux_err"]["mean"]) < 1.5, c["aspect_ratio"]


def test_window_averaging_beats_the_kernel_scale_scatter(data):
    for aspect in (1.0, 1.5):
        sel = sorted([w for w in data["window_ladder"] if w["aspect_ratio"] == aspect],
                     key=lambda w: w["half_window_deg"])
        assert sel[-1]["flux_err"]["rms"] <= 0.5 * sel[0]["flux_err"]["rms"]


# --------------------------------------------------------------------------- #
# honest FAILs, registered (strict xfail: they start failing when fixed)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "LIVE DEFECT: the pointwise (station-scale) jump carries 30-40 % scatter "
    "from the MLS div-n curvature field; it does not decay with refinement at "
    "fixed h/dx. Isolated by the kappa_exact ablation (2-5 %)."))
def test_pointwise_jump_within_15pct(data):
    for c in data["configs"]:
        if not c["resolved"] or c["aspect_ratio"] > 2.0:
            continue
        assert c["flux_err"]["rms"] < 15.0, c["aspect_ratio"]


@pytest.mark.xfail(strict=True, reason=(
    "LIVE DEFECT: the pressure readout (what a projection solver reports) reaches "
    "+16 % at a:b = 3:1, R_eq/dx = 40, while the traction integral of the same "
    "equilibrium is inside 1 %. Readout/geometry, not the force model."))
def test_pressure_readout_within_15pct_at_high_curvature(data):
    for c in data["configs"]:
        if c["r_eq_over_dx"] != 40.0:
            continue
        assert abs(c["pressure_err"]["mean"]) < 15.0, c["aspect_ratio"]


# --------------------------------------------------------------------------- #
# live recomputation: the code path, not just the artifact
# --------------------------------------------------------------------------- #
def test_live_ellipse_case_is_reproducible_and_isotropic():
    """Recompute a cheap equilibrium and check the two instruments agree there."""
    from curvature_variation import SIGMA, R_EQ, ellipse_case

    r = ellipse_case(aspect=1.5, dx=R_EQ / 10.0)
    assert r["n_particles"] > 500
    # the differential law in the mean (owner gate 15 % at the coarse level)
    assert abs(r["flux_err"]["mean"]) < 15.0
    # the curvature integral is still 2 pi at a coarse, unresolved-shape level
    assert abs(r["whitney_graustein_err_pct"]) < 5.0
    # delta_s is still normalised, i.e. the band selection is the true distance
    assert 0.95 < r["delta_s_norm_mean"] < 1.05
    # every station got both probe bands
    for s in r["stations"]:
        if s is None:
            continue
        assert s["n_inner"] > 0 and s["n_outer"] > 0
    # mean traction ~ sigma * mean kappa to the neighbour-window accuracy
    ratio = r["mean_dp_flux_Pa"] / (SIGMA * r["mean_kappa_exact_per_m"])
    assert abs(ratio - 1.0) < 0.15
    assert torch.isfinite(torch.tensor(r["net_traction_residual"]))