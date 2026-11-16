"""Resolution ladder for CASE-SPH-LAKE-AT-REST (periodic tank).

Tests the convergence of the interior-band profile error at multiple
resolutions.  This fills the gap documented in
``docs/verification_lake_at_rest.md`` §5 ("Not done"): the interior-band
profile error had no measured convergence order.

The ladder runs at dx = 0.04 and 0.02 (periodic tank) and verifies:
- equilibrium residual is machine precision at all resolutions
- interior-band density error decreases with dx
- interior-band pressure error decreases with dx
- convergence order is approximately first-order (expected for SPH with
  the cubic spline kernel on a regular lattice)
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    reconstruct, residual_report)
from tensorsph import eos_hydrostatics as eos


def _interior_band_mask(state, cfg, margin_factor=2.0):
    h = cfg.h()
    y = state["pos"][:, 1]
    return (y > margin_factor * h) & (y < cfg.depth - margin_factor * h)


def _profile_error(state, cfg, mass):
    mask = _interior_band_mask(state, cfg)
    if mask.sum() == 0:
        return {"density_max_rel": None, "pressure_max_rel": None}
    y = state["pos"][mask, 1]
    H = cfg.depth
    rho_exact = eos.eos_hydrostatic_density(
        y, H, rho0=cfg.rho0, sound_speed=cfg.sound_speed,
        gamma=cfg.gamma, gravity=cfg.gravity)
    p_exact = eos.eos_hydrostatic_pressure(
        y, H, rho0=cfg.rho0, sound_speed=cfg.sound_speed,
        gamma=cfg.gamma, gravity=cfg.gravity)
    report = residual_report(state, cfg, mass)
    rho_recon = report["density"][mask]
    p_recon = report["pressure"][mask]
    p_scale = cfg.rho0 * cfg.gravity * H
    return {
        "density_max_rel": float(((rho_recon - rho_exact) / rho_exact).abs().max()),
        "pressure_max_rel": float(((p_recon - p_exact) / p_scale).abs().max()),
    }


def _convergence_order(err_coarse, err_fine, dx_coarse, dx_fine):
    if err_coarse is None or err_fine is None or err_fine == 0:
        return None
    ratio = err_coarse / err_fine
    if ratio <= 1.0:
        return None
    return math.log(ratio) / math.log(dx_coarse / dx_fine)


@pytest.fixture(scope="module")
def ladder():
    """Run the resolution ladder at dx = 0.04 and 0.02."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        results = []
        for dx in [0.04, 0.02]:
            cfg = LakeConfig(dx=dx, depth=0.30, width=0.40, t_end=0.50,
                             periodic_x=True, solver_max_iter=40)
            state = build_state(cfg)
            prior = analytic_profile(state, cfg)
            sol = reconstruct(state, cfg, prior=prior["mass"])
            prof_err = _profile_error(state, cfg, sol["mass"])
            results.append({
                "dx": dx,
                "cfg": cfg,
                "state": state,
                "solution": sol,
                "profile_error": prof_err,
            })
        yield results
    finally:
        torch.set_default_dtype(previous)


def test_equilibrium_is_machine_precision_at_all_resolutions(ladder):
    """G1/G2: both acceleration components are at roundoff for every dx."""
    for row in ladder:
        sol = row["solution"]
        assert sol["converged"] is True, f"dx={row['dx']}: Newton did not converge"
        assert sol["a_max_over_g"] < 1e-9, f"dx={row['dx']}: a_y/g = {sol['a_max_over_g']:.3e}"
        assert sol["a_max_over_g_horizontal"] < 1e-9, (
            f"dx={row['dx']}: a_x/g = {sol['a_max_over_g_horizontal']:.3e}")


def test_density_error_decreases_with_dx(ladder):
    """The interior-band density error must decrease as dx is refined."""
    coarse_err = ladder[0]["profile_error"]["density_max_rel"]
    fine_err = ladder[1]["profile_error"]["density_max_rel"]
    assert coarse_err is not None and fine_err is not None
    assert fine_err < coarse_err, (
        f"density error did not decrease: {coarse_err:.3e} (dx=0.04) -> "
        f"{fine_err:.3e} (dx=0.02)")


def test_pressure_error_decreases_with_dx(ladder):
    """The interior-band pressure error must decrease as dx is refined."""
    coarse_err = ladder[0]["profile_error"]["pressure_max_rel"]
    fine_err = ladder[1]["profile_error"]["pressure_max_rel"]
    assert coarse_err is not None and fine_err is not None
    assert fine_err < coarse_err, (
        f"pressure error did not decrease: {coarse_err:.3e} (dx=0.04) -> "
        f"{fine_err:.3e} (dx=0.02)")


def test_convergence_order_is_positive(ladder):
    """The observed convergence order must be positive (errors decrease with dx)."""
    coarse = ladder[0]
    fine = ladder[1]
    dx_ratio = coarse["dx"] / fine["dx"]
    density_order = _convergence_order(
        coarse["profile_error"]["density_max_rel"],
        fine["profile_error"]["density_max_rel"],
        coarse["dx"], fine["dx"])
    pressure_order = _convergence_order(
        coarse["profile_error"]["pressure_max_rel"],
        fine["profile_error"]["pressure_max_rel"],
        coarse["dx"], fine["dx"])
    assert density_order is not None and density_order > 0, (
        f"density convergence order: {density_order}")
    assert pressure_order is not None and pressure_order > 0, (
        f"pressure convergence order: {pressure_order}")
    # SPH with cubic spline on a regular lattice typically shows ~1st order
    # for profile errors (the kernel approximation error dominates)
    assert 0.5 < density_order < 3.0, (
        f"density order {density_order:.2f} is outside the expected range [0.5, 3.0]")
    assert 0.5 < pressure_order < 3.0, (
        f"pressure order {pressure_order:.2f} is outside the expected range [0.5, 3.0]")


def test_profile_errors_are_small_at_published_resolution(ladder):
    """At dx = 0.02, the profile errors are within the documented bounds."""
    fine = ladder[1]
    assert fine["dx"] == pytest.approx(0.02)
    # Density error < 0.1% (documented: 2.53e-04)
    assert fine["profile_error"]["density_max_rel"] < 1e-3
    # Pressure error < 10% of p_scale (documented: 4.54e-02)
    assert fine["profile_error"]["pressure_max_rel"] < 0.10


def test_interior_band_grows_with_resolution(ladder):
    """The number of interior-band particles must increase with finer dx."""
    coarse_mask = _interior_band_mask(ladder[0]["state"], ladder[0]["cfg"])
    fine_mask = _interior_band_mask(ladder[1]["state"], ladder[1]["cfg"])
    assert int(fine_mask.sum()) > int(coarse_mask.sum()), (
        f"interior band did not grow: {coarse_mask.sum()} (dx=0.04) -> "
        f"{fine_mask.sum()} (dx=0.02)")
