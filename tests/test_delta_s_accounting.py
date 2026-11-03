"""Acceptance bookkeeping for the CSF surface weight (delta_s).

Background: the verification record carried two criteria that are analytically
the same identity -- ``int delta_s dn = 1`` and ``F_cut = 2 sigma`` -- with
values 0.739 and 0.893-1.110.  ``benchmarks/csf_accounting/run.py`` shows why:
the first is measured with the colour function's nominal volumes
(``V = m/rho0``) while the force integrates with the actual volumes
(``V = m/rho``), and the free-surface density deficit sits exactly where
``delta_s`` peaks.  It also shows that the colour function *is* the summation
density over ``rho0`` (clipped), so ``delta_s`` is a density-gradient operator,
not a geometric surface measure.

These tests do two things:

* pin the committed artefact (``validation_report.json``) so the record cannot
  drift away from the numbers quoted in ``docs/verification_delta_s_accounting.md``;
* recompute the quadrature relationships live on one small drop, so a change in
  the operator or in the neighbour/density code shows up here instead of
  silently invalidating the accounting.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorsph.csf_consistent import (  # noqa: E402
    csf_fields, mls_gradient, surface_tension_acceleration)
from tensorsph.droplet_impact import Config, density_from_neighbor_pairs  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

BENCH = ROOT / "benchmarks" / "csf_accounting"
REPORT = BENCH / "validation_report.json"
DOC = ROOT / "docs" / "verification_delta_s_accounting.md"

RADIUS = 0.02
SIGMA = 7.2
RHO0 = 1000.0
GATE = 0.15


def _hex_droplet():
    spec = importlib.util.spec_from_file_location(
        "_ll_law_acct", ROOT / "benchmarks" / "laplace_law" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.hexagonal_droplet


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def live():
    """One static evaluation on the production entry point (dx = 2 mm)."""
    dx = 0.002
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature="fit_quadratic")
    h = cfg.h_factor * cfg.dx
    pos, mass = _hex_droplet()(RADIUS, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc, state = surface_tension_acceleration(
        pos, rho, mass, pairs, h, SIGMA, rho0=RHO0,
        curvature_scheme="fit_quadratic")
    r = torch.linalg.vector_norm(pos, dim=1)
    v_nom, v_act = mass / RHO0, mass / rho
    two_pi_r = 2.0 * math.pi * r.clamp_min(1e-30)
    right = pos[:, 0] > 0.0
    f_cut = float((mass[right, None] * acc[right]).sum(dim=0)[0]) / (-2.0 * SIGMA)
    g_rho = mls_gradient(pos, rho / RHO0, v_nom, pairs, h)
    return {
        "pos": pos, "mass": mass, "rho": rho, "pairs": pairs, "h": h,
        "state": state, "r": r,
        "i_nominal": float((state.delta_s * v_nom / two_pi_r).sum()),
        "i_force": float((state.delta_s * v_act / two_pi_r).sum()),
        "f_cut_ratio": f_cut,
        "identity": float((state.color - (rho / RHO0).clamp(0.0, 1.0)).abs().max()),
        "grad_identity": float(
            (state.grad_c - g_rho).norm(dim=1).max())
        / float(torch.linalg.vector_norm(state.grad_c, dim=1).max()),
        "kappa_over_exact": float(
            (state.kappa.abs() * state.delta_s).sum() / state.delta_s.sum())
        * RADIUS,
    }


# --------------------------------------------------------------------------- #
# live recomputation of the accounting relations
# --------------------------------------------------------------------------- #
def test_nominal_and_force_measures_differ_as_reported(live):
    """The two quadratures of the *same* field must bracket the reported pair:
    nominal ~0.75, force ~0.96.  If a future change makes them agree, the
    accounting in the doc is obsolete and this test must be revisited."""
    assert live["i_nominal"] == pytest.approx(0.751, abs=0.03)
    assert live["i_force"] == pytest.approx(0.957, abs=0.03)
    assert live["i_force"] > live["i_nominal"] + 0.15


def test_colour_function_is_the_clipped_summation_density(live):
    """``c = clip(rho_sum / rho0, 0, 1)``: the colour field is not independent of
    the density field, which is why delta_s cannot be normalised independently
    of the free-surface density deficit."""
    assert live["identity"] < 1e-2


def test_colour_gradient_is_the_density_gradient(live):
    """``|grad c| = |grad rho| / rho0`` to within a fraction of a percent."""
    assert live["grad_identity"] < 2e-2


def test_cut_force_pass_relies_on_the_density_deficit(live):
    """The 15 % pass of F_cut is a cancellation: the nominal-measure prediction
    (which is what the force would give if the density deficit were fixed) is
    >5 % below the measured ratio."""
    pred_nominal = live["kappa_over_exact"] * live["i_nominal"]
    assert live["f_cut_ratio"] - pred_nominal > 0.05


# --------------------------------------------------------------------------- #
# committed artefact: internal consistency and agreement with the doc
# --------------------------------------------------------------------------- #
def test_report_rows_recompute_their_own_predictions(report):
    for row in report["rows"]:
        cf = row["cut_force"]
        kappa = row["kappa_over_exact"]
        assert cf["prediction_nominal_measure"] == pytest.approx(
            kappa * row["delta_s"]["integral_nominal_measure"], abs=1e-9)
        assert cf["prediction_force_measure"] == pytest.approx(
            kappa * row["delta_s"]["integral_force_measure"], abs=1e-9)
        assert cf["residual_nominal_measure"] == pytest.approx(
            cf["ratio_measured_over_theory"]
            - cf["prediction_nominal_measure"], abs=1e-9)
        assert cf["residual_force_measure"] == pytest.approx(
            cf["ratio_measured_over_theory"]
            - cf["prediction_force_measure"], abs=1e-9)


def test_report_verdicts_match_the_rows(report):
    rows = report["rows"]
    v = report["verdict"]
    gaps = [abs(r["cut_force"]["residual_nominal_measure"]) for r in rows]
    act = [abs(r["cut_force"]["residual_force_measure"]) for r in rows]
    assert v["mean_abs_residual_nominal_measure"] == pytest.approx(
        sum(gaps) / len(gaps), abs=1e-12)
    assert v["mean_abs_residual_force_measure"] == pytest.approx(
        sum(act) / len(act), abs=1e-12)
    assert v["force_measure_reduces_residual_on_average"] == (
        sum(act) / len(act) < sum(gaps) / len(gaps))
    assert v["color_function_is_clipped_density"] is True
    assert v["field_normalisation_deficit_pct_systematic"] is True


def test_report_keeps_the_fragile_pass_visible(report):
    """The honest verdict must not be able to disappear from the record: the
    measured cut-force ratio passes the gate while the density-corrected
    prediction does not."""
    rows = report["rows"]
    assert report["verdict"]["cut_force_gate_relies_on_density_deficit_cancellation"] is True
    # The forward operator (fit_quadratic, the default of the production
    # pipeline) is the one whose pass is propped up: with the density deficit
    # removed its cut-force ratio leaves the gate.  The legacy mls operator
    # happens to compensate with its +25 % curvature, which is the residual
    # sign flip reported in the doc -- not a second line of defence.
    fit_rows = sorted((r for r in rows if r["curvature"] == "fit_quadratic"),
                      key=lambda r: r["dx_m"])
    assert fit_rows
    for row in fit_rows:
        assert abs(row["cut_force"]["ratio_measured_over_theory"] - 1.0) < 0.26
    # At the finest resolution (the only one where the curvature operator is
    # converged, |kappa R - 1| < 5 %) removing the density deficit would push
    # the cut-force ratio out of the 15 % gate.
    finest = fit_rows[0]   # sorted by dx ascending -> finest first
    assert abs(finest["kappa_over_exact"] - 1.0) < 0.05
    assert abs(finest["cut_force"]["ratio_if_density_deficit_were_fixed"] - 1.0) > GATE


def test_report_covers_both_curvature_operators_and_three_resolutions(report):
    assert {r["curvature"] for r in report["rows"]} == {"mls", "fit_quadratic"}
    assert sorted(r["dx_m"] for r in report["rows"] if r["curvature"] == "mls") == \
        sorted(report["setup"]["dx_list"])


def test_force_measure_closes_the_gap_at_converged_curvature(report):
    """At the one configuration whose curvature is inside 5 % of 1/R, matching
    the volume measure must cut the cut-force residual by more than 2x."""
    conv = [r for r in report["rows"] if abs(r["kappa_over_exact"] - 1.0) < 0.05]
    assert len(conv) >= 1
    for row in conv:
        cf = row["cut_force"]
        assert abs(cf["residual_force_measure"]) < 0.5 * abs(
            cf["residual_nominal_measure"])


def test_document_quotes_no_placeholder_numbers():
    text = DOC.read_text()
    for bad in ("nan", "{:", "{get", "float('nan')"):
        assert bad not in text, f"doc contains {bad!r}"
    assert "honest" in text
    assert "delta_s_accounting.png" in text