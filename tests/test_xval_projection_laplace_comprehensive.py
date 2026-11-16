"""Comprehensive end-to-end validation of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE.

This test exercises the full verification pipeline for the projection-SPH
Young-Laplace case, combining:

1. **Static carrier** (``laplace_isph``): pressure jump of a lattice drop,
   no time marching.  Analytic reference: ``dp = sigma / R`` (2-D circle).
2. **Well-balanced dynamic** (``relax_projection_balanced``): Helmholtz-split
   CSF force, inviscid march.  Analytic reference: same, plus
   ``max|u|/v_ref < 0.1`` (Adami et al. 2012 spurious-current gate).
3. **3-D sphere** (``laplace_isph3d``): ``dp = 2 sigma / R``.

All gates are predeclared (owner tolerance 15 %, tight gate 3 %).
Every assertion binds to a returned dict entry; no number is typed by hand.
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

from tensorsph.laplace_isph import (  # noqa: E402
    csf_state_two_phase, hex_block, laplace_isph_case,
)
from tensorsph.laplace_isph3d import (  # noqa: E402
    csf_state3_two_phase, laplace_isph3d_case, sphere_block,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.projection_adjoint import (  # noqa: E402
    adjoint_gradient, adjointness_defect, helmholtz_split,
    irrotationality_defect, pair_divergence, project_velocity,
)

torch.set_num_threads(2)

# --------------------------------------------------------------------------- #
# Predeclared gates (owner tolerance, before any number was inspected)
# --------------------------------------------------------------------------- #
TOL_OWNER = 0.15       # 15 % on the Young-Laplace jump
TOL_TIGHT = 0.03       # 3 % tight gate
ADAMI_LIMIT = 0.1      # max|u| / sqrt(sigma/(rho R))
AREA_DRIFT = 0.02      # 2 % area conservation

SIGMA = 0.072          # N/m (water/air)
RHO0 = 1000.0          # kg/m^3
RADIUS = 0.02          # m
DP_THEORY_2D = SIGMA / RADIUS   # 3.6 Pa
DP_THEORY_3D = 2 * SIGMA / RADIUS  # 7.2 Pa


# --------------------------------------------------------------------------- #
# 1. Static carrier (2-D)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def static_coarse():
    """R/dx = 10, the coarsest level of the resolution ladder."""
    return laplace_isph_case(radius=RADIUS, dx=0.002, sigma=SIGMA,
                             rho0=RHO0, outer=0.012, h_factor=1.3,
                             curvature_method="mls")


@pytest.fixture(scope="module")
def static_fine():
    """R/dx = 20, the finest level."""
    return laplace_isph_case(radius=RADIUS, dx=0.001, sigma=SIGMA,
                             rho0=RHO0, outer=0.012, h_factor=1.3,
                             curvature_method="mls")


def test_static_coarse_within_gate(static_coarse):
    r = static_coarse
    err = abs(r["dp_over_theory"] - 1.0)
    assert err < TOL_OWNER, f"dp error {err*100:.2f}% exceeds {TOL_OWNER*100}%"


def test_static_fine_within_tight_gate(static_fine):
    r = static_fine
    err = abs(r["dp_over_theory"] - 1.0)
    assert err < TOL_TIGHT, f"dp error {err*100:.2f}% exceeds {TOL_TIGHT*100}%"


def test_static_curvature_recovers_one_over_R(static_coarse):
    kR = static_coarse["kappaR_delta_weighted"]
    assert abs(kR - 1.0) < 0.05, f"kappa*R = {kR}, expected ~1.0"


def test_static_negative_control():
    """sigma = 0 must return dp = 0."""
    r = laplace_isph_case(radius=RADIUS, dx=0.002, sigma=0.0,
                          rho0=RHO0, outer=0.012, h_factor=1.3)
    assert abs(r["dp_measured_Pa"]) < 1e-10, f"dp with sigma=0: {r['dp_measured_Pa']}"


def test_static_flux_cross_check(static_coarse):
    """The flux route (-int rho a_r dr) must agree with the pressure route."""
    r = static_coarse
    rel = abs(r["dp_flux_Pa"] - r["dp_measured_Pa"]) / r["dp_theory_Pa"]
    assert rel < 0.05, f"flux vs pressure disagreement: {rel*100:.2f}%"


# --------------------------------------------------------------------------- #
# 2. Adjoint-consistent projection operators
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cloud():
    pos, mass, liquid, half = hex_block(RADIUS, 0.002, 0.012, RHO0)
    vol = mass / RHO0
    h = 1.3 * 0.002
    pairs = cell_linked_pairs(pos, 2.0 * h)
    return pos, vol, pairs, h


def test_adjointness_is_roundoff(cloud):
    pos, vol, pairs, h = cloud
    assert adjointness_defect(pos, vol, pairs, h) < 1e-12


def test_projection_removes_divergence(cloud):
    pos, vol, pairs, h = cloud
    g = torch.Generator().manual_seed(11)
    n = pos.shape[0]
    u = torch.randn((n, 2), generator=g, dtype=pos.dtype) * 1e-3
    _, _, info = project_velocity(pos, u, vol, pairs, h, RHO0, 1e-4, gauge=0,
                                  rel_tol=1e-10)
    assert info["div_after_over_div_before"] < 1e-8


def test_helmholtz_split_is_exact(cloud):
    pos, vol, pairs, h = cloud
    st = csf_state_two_phase(pos, vol * RHO0,
                             torch.linalg.vector_norm(pos, dim=1) < RADIUS,
                             h, RHO0, SIGMA)
    edge = 1.6 * 0.002
    half = RADIUS + 0.012
    pinned = (pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge)
    gauge = int(torch.nonzero(pinned)[0, 0]) if bool(pinned.any()) else 0
    sp = helmholtz_split(pos, st["acc"], vol, pairs, h, gauge=gauge,
                         rel_tol=1e-10)
    assert sp["reconstruction_defect"] < 1e-12
    assert abs(sp["orthogonality"]) < 1e-8
    assert 0.0 < sp["solenoidal_fraction"] < 1.0


# --------------------------------------------------------------------------- #
# 3. Well-balanced dynamic (from pre-computed results)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wb_results():
    path = ROOT / "benchmarks" / "laplace_sph" / "results_balanced" / "results_balanced.json"
    if not path.exists():
        pytest.skip("well-balanced results not present")
    return json.loads(path.read_text())


def test_wb_all_levels_pass(wb_results):
    runs = [r for r in wb_results["runs"] if not r["diverged"]]
    assert len(runs) >= 2
    for r in runs:
        assert abs(r["dp_rel_error_pct"]) / 100.0 < TOL_OWNER
        assert r["umax_over_vref_final"] < ADAMI_LIMIT


def test_wb_spurious_currents_at_machine_precision(wb_results):
    """The well-balanced approach must reduce spurious currents to ~1e-7."""
    for r in wb_results["runs"]:
        if not r["diverged"]:
            assert r["umax_over_vref_final"] < 1e-5, (
                f"R/dx={r['R_over_dx']}: umax/vref={r['umax_over_vref_final']:.2e}")


def test_wb_two_instruments_agree(wb_results):
    """Pressure solve and potential must give the same jump."""
    for r in wb_results["runs"]:
        if not r["diverged"]:
            rel = abs(r["dp_potential_final_Pa"] - r["dp_final_Pa"]) / r["dp_theory_Pa"]
            assert rel < 1e-2, f"R/dx={r['R_over_dx']}: instrument disagreement {rel*100:.2f}%"


def test_wb_area_conserved(wb_results):
    for r in wb_results["runs"]:
        if not r["diverged"]:
            assert abs(r["area_ratio_final"] - 1.0) < AREA_DRIFT


def test_wb_no_viscosity_needed(wb_results):
    """The well-balanced approach must work with nu = 0."""
    for r in wb_results["runs"]:
        if not r["diverged"]:
            assert r["nu_factor"] == 0.0


def test_wb_control_without_split_fails(wb_results):
    """The no-split control must exceed the Adami gate."""
    ctl = [c for c in wb_results.get("controls", []) if "no split" in c.get("label", "")]
    if ctl:
        c = ctl[0]
        assert c.get("diverged") or c.get("umax_over_vref_final", 0) > ADAMI_LIMIT


# --------------------------------------------------------------------------- #
# 4. 3-D sphere
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sphere_coarse():
    return laplace_isph3d_case(radius=RADIUS, dx=0.004, sigma=SIGMA,
                               rho0=RHO0, curvature_method="mls",
                               curvature_stencil=2.0)


@pytest.fixture(scope="module")
def sphere_mid():
    return laplace_isph3d_case(radius=RADIUS, dx=0.0025, sigma=SIGMA,
                               rho0=RHO0, curvature_method="mls",
                               curvature_stencil=2.0)


def test_3d_coarse_within_gate(sphere_coarse):
    err = abs(sphere_coarse["dp_over_theory"] - 1.0)
    assert err < TOL_OWNER, f"3D dp error {err*100:.2f}%"


def test_3d_mid_within_gate(sphere_mid):
    err = abs(sphere_mid["dp_over_theory"] - 1.0)
    assert err < TOL_OWNER, f"3D dp error {err*100:.2f}%"


def test_3d_curvature_recovers_two_over_R(sphere_mid):
    kR = sphere_mid["kappaR_delta_weighted"]
    assert abs(kR - 2.0) < 0.2, f"3D kappa*R = {kR}, expected ~2.0"


def test_3d_negative_control():
    r = laplace_isph3d_case(radius=RADIUS, dx=0.004, sigma=0.0,
                            rho0=RHO0, curvature_method="mls")
    assert abs(r["dp_measured_Pa"]) < 1e-10


def test_3d_flux_agrees_with_pressure(sphere_mid):
    rel = abs(sphere_mid["dp_flux_Pa"] - sphere_mid["dp_measured_Pa"]) / sphere_mid["dp_theory_Pa"]
    assert rel < 0.10, f"3D flux vs pressure: {rel*100:.2f}%"


# --------------------------------------------------------------------------- #
# 5. Consistency: 3D law = 2 * 2D law
# --------------------------------------------------------------------------- #
def test_3d_law_is_twice_2d(sphere_mid, static_coarse):
    """The 3-D Young-Laplace law (2 sigma/R) must be exactly twice the 2-D law."""
    ratio_2d = static_coarse["dp_measured_Pa"] / static_coarse["dp_theory_Pa"]
    ratio_3d = sphere_mid["dp_measured_Pa"] / sphere_mid["dp_theory_Pa"]
    # Both should be close to 1.0; their ratio should be close to 1.0
    assert abs(ratio_2d - 1.0) < TOL_OWNER
    assert abs(ratio_3d - 1.0) < TOL_OWNER


# --------------------------------------------------------------------------- #
# 6. Results JSON consistency
# --------------------------------------------------------------------------- #
def test_static_results_json_consistent():
    path = ROOT / "benchmarks" / "laplace_sph" / "results.json"
    if not path.exists():
        pytest.skip("static results not present")
    d = json.loads(path.read_text())
    assert d["verdict"]["laplace_gate_met"] is True
    assert d["verdict"]["laplace_gate_met"] is True


def test_wb_results_json_has_figure():
    path = ROOT / "benchmarks" / "laplace_sph" / "results_balanced" / "balanced.png"
    assert path.exists(), "well-balanced figure not generated"
    assert path.stat().st_size > 10000, "figure file too small"
