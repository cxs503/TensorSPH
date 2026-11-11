"""Gates for CASE-LAPLACE-SPH (``Delta p = sigma / R`` on a projection-SPH drop).

The committed ``benchmarks/laplace_sph/results.json`` is the artifact; these
tests assert the *invariants* behind it rather than re-running the whole ladder:

* the Laplace gate really is inside 5 % at every level, and the finest level is
  inside 5 % by a margin that is not a rounding accident;
* the instrument self-test (exact kappa + exact delta) is inside 3 %, so the
  measured error is attributable to the model and not to the harness;
* the independent flux route agrees with the Poisson route;
* the negative control (sigma = 0) returns no jump;
* the spurious-current gate is still FAILING (strict): the moment the time
  integration starts working, this test fails loudly and forces the docs and the
  capability claim to be revisited instead of leaving a stale claim behind;
* the rendered page equals a fresh rendering of the JSON (anti-drift).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.laplace_isph import laplace_isph_case

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "benchmarks" / "laplace_sph" / "results.json"
DOC = HERE / "docs" / "verification_laplace_sph_isph.md"

TOL = 0.05


@pytest.fixture(scope="module")
def report():
    return json.loads(RES.read_text())


def test_results_artifact_exists(report):
    assert report["case"] == "CASE-LAPLACE-SPH"
    assert report["ladder_mls"], "no ladder recorded"


def test_laplace_gate_every_level(report):
    for row in report["ladder_mls"]:
        err = abs(row["dp_over_theory"] - 1.0)
        assert err <= TOL, (row["R_over_dx"], err)
        assert row["dp_theory_Pa"] == pytest.approx(0.072 / 0.02, rel=1e-12)


def test_finest_level_margin(report):
    # The gate is 5 %; the committed result must clear it by a real margin so a
    # future regression of the operators is visible before it crosses the gate.
    finest = report["ladder_mls"][-1]
    assert abs(finest["dp_over_theory"] - 1.0) < 0.02


def test_gauge_self_test_bounds_instrument(report):
    gauge = [v for v in report["variants"] if v["variant"] == "gauge"][0]["rows"][-1]
    assert abs(gauge["dp_over_theory"] - 1.0) <= 0.03


def test_cross_check_routes_agree(report):
    for row in report["ladder_mls"]:
        rel = row["dp_flux_Pa"] / row["dp_measured_Pa"] - 1.0
        assert abs(rel) <= 0.03, (row["R_over_dx"], rel)


def test_bulk_momentum_balance(report):
    # Away from the interface shell the projection must cancel the traction.
    for row in report["ladder_mls"]:
        assert row["residual_bulk_median_over_mean_csf_acc"] <= 0.05


def test_negative_control_has_no_jump(report):
    assert abs(report["negative_control"]["dp_measured_Pa"]) <= 1e-10


def test_strict_xfail_spurious_currents(report):
    """The time integration must stay known-broken until it is actually fixed."""
    gates = report["gates"]["spurious_currents_adami_0p1"]
    assert gates["pass"] is False, (
        "the projection time integration now passes the Adami limit - update "
        "docs/verification_laplace_sph_isph.md and the capability claim"
    )
    dyn = report["dynamics"]
    assert dyn["diverged"] is True
    assert dyn["steps_to_divergence"] >= 1


def test_documented_page_matches_rendering():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "laplace_sph_make_doc",
        HERE / "benchmarks" / "laplace_sph" / "make_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rep = json.loads(RES.read_text())
    assert DOC.read_text() == mod.render(rep)


def test_case_is_reproducible_at_coarse_level():
    """A live re-run of one level must reproduce the committed number."""
    rep = json.loads(RES.read_text())
    coarse = rep["ladder_mls"][0]
    live = laplace_isph_case(radius=0.02, dx=coarse["dx_m"], sigma=0.072,
                             rho0=1000.0, outer=0.012, h_factor=1.3,
                             curvature_method="mls")
    assert live["n_particles"] == coarse["n_particles"]
    assert live["dp_over_theory"] == pytest.approx(coarse["dp_over_theory"], abs=1e-9)
    assert math.isfinite(live["dp_measured_Pa"])


def test_operator_laplacian_solves_a_known_problem():
    """The Poisson operator on a uniform lattice must reproduce a linear field.

    A linear pressure field is annihilated by the Laplacian, so CG started from
    zero must stay at (near) zero: this pins the sign and the pair symmetry of
    the operator, which is what makes the CG loop stable.
    """
    from tensorsph.laplace_isph import hex_block, pressure_poisson

    pos, mass, liquid, half = hex_block(0.02, 0.002, 0.012, 1000.0)
    vol = mass / 1000.0
    from tensorsph.neighbor_search import cell_linked_pairs

    pairs = cell_linked_pairs(pos, 2.6e-3)
    zero = torch.zeros(pos.shape[0], dtype=pos.dtype)
    fixed = torch.zeros(pos.shape[0], dtype=torch.bool)
    p, _ = pressure_poisson(pos, zero, pairs, vol, 2.6e-3, fixed,
                            rel_tol=1e-12, max_iter=200)
    assert float(p.abs().max()) == pytest.approx(0.0, abs=1e-14)
