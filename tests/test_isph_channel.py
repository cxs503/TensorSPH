"""Tests for the ISPH (projection) channel solver.

The tests pin the two things that make the solver's benchmark claims
meaningful: (a) the four discrete operators match their analytic values on
smooth fields, so a good profile is not an accident of the discretisation, and
(b) the projection step actually removes divergence rather than only being
documented as doing so.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.isph_channel import (
    ChannelConfig,
    build_channel,
    divergence,
    fit_profile,
    gates,
    operator_signatures,
    pair_geometry,
    pressure_gradient,
    project,
    run_case,
    summation_density,
    viscous_laplacian,
)

DEVICE = torch.device("cpu")


def test_lattice_summation_density_is_uniform_in_the_interior():
    cfg = ChannelConfig(resolution=12)
    cloud = build_channel(cfg, DEVICE)
    n_fluid = cloud["n_fluid"]
    geo = pair_geometry(cloud["pos"], cfg.h, cfg.length)
    rho = summation_density(geo, cloud["mass"], cfg.h)
    y = cloud["pos"][:n_fluid, 1]
    interior = (y > 2.0 * cfg.h) & (y < cfg.height - 2.0 * cfg.h)
    assert int(interior.sum()) > 0
    err = (rho[:n_fluid][interior] / cfg.rho0 - 1.0).abs().max()
    assert float(err) < 1e-6


def test_operators_match_analytic_values():
    sig = operator_signatures(ChannelConfig(resolution=12), DEVICE)
    # Each probe is judged relative to its analytic magnitude (6, 0.5, 2, 1):
    # the grid is only 12 interior rows wide, so an absolute tolerance would be
    # resolution-dependent.
    assert sig["laplacian_of_quadratic_y"]["rms"] / 6.0 < 1.0e-2
    assert sig["divergence_linear_y"]["rms"] / 0.5 < 1.0e-2
    assert sig["gradient_linear_y"]["rms"] / 2.0 < 1.0e-2
    assert sig["viscous_laplacian_quadratic_y"]["rms"] / 1.0 < 1.0e-2
    # The Poisson operator used by the solver is +grad^2, measured not assumed.
    assert sig["laplacian_sign"].startswith("plus")


def test_laplacian_of_constant_field_vanishes():
    cfg = ChannelConfig(resolution=12)
    cloud = build_channel(cfg, DEVICE)
    n_fluid = cloud["n_fluid"]
    geo = pair_geometry(cloud["pos"], cfg.h, cfg.length)
    rho = summation_density(geo, cloud["mass"], cfg.h)
    volume = cloud["mass"] / rho
    const = torch.ones_like(cloud["pos"])
    const[:, 1] = 0.0
    lap = viscous_laplacian(geo, const, volume, cfg.laplacian_eta * cfg.h)
    assert float(lap.abs().max()) < 1e-12
    grad = pressure_gradient(geo, torch.full((cloud["pos"].shape[0],), 5.0), volume,
                             n_fluid)
    assert float(grad.abs().max()) < 1e-12


def test_projection_reduces_divergence():
    cfg = ChannelConfig(resolution=12)
    cloud = build_channel(cfg, DEVICE)
    n_fluid = cloud["n_fluid"]
    k = 2.0 * math.pi / cfg.length
    pos = cloud["pos"]
    vel = torch.zeros_like(pos)
    vel[:, 0] = 0.3 * torch.sin(k * pos[:, 0])     # genuinely divergent field
    cloud["vel"] = vel
    geo = pair_geometry(pos, cfg.h, cfg.length)
    rho = summation_density(geo, cloud["mass"], cfg.h)
    out = project(cloud, geo, cfg, 1e-3, rho, n_fluid)
    assert out["div_before"] > 0.0
    assert out["div_after"] < 0.5 * out["div_before"]


def test_short_run_stays_symmetric_and_finite():
    cfg = ChannelConfig(resolution=8, t_end=0.1, t_average=0.05)
    out = run_case(cfg, DEVICE)
    assert out["steps"] > 0
    assert math.isfinite(out["u_max_measured"])
    profile = out["profile"]
    assert len(profile) >= 6
    first, last = profile[0]["u_mean"], profile[-1]["u_mean"]
    u_max = max(p["u_mean"] for p in profile)
    assert u_max > 0.0
    assert abs(first - last) / u_max < 0.05


def test_fit_recovers_the_analytic_parabola_exactly():
    cfg = ChannelConfig()
    profile = []
    for k in range(cfg.resolution):
        y = (k + 0.5) * cfg.dx
        profile.append({
            "row": k, "y": y, "y_over_H": y / cfg.height,
            "u_mean": cfg.body_force * y * (cfg.height - y) / (2.0 * cfg.viscosity),
        })
    fit = fit_profile(profile, cfg.height, cfg.body_force, cfg.viscosity)
    assert fit["nu_eff_over_nu"] == pytest.approx(1.0, rel=1e-9)
    assert fit["du_dy_wall"] == pytest.approx(cfg.body_force * cfg.height
                                              / (2.0 * cfg.viscosity), rel=1e-9)


def test_gate_report_has_all_three_criteria():
    cfg = ChannelConfig(resolution=8, t_end=0.2, t_average=0.1)
    out = run_case(cfg, DEVICE)
    fit = fit_profile(out["profile"], cfg.height, cfg.body_force, cfg.viscosity,
                      lo=0.05, hi=0.95)
    g = gates(out, fit, cfg)
    for key in ("density_fluctuation_max", "centre_velocity", "wall_shear_budget"):
        assert key in g
        assert "pass" in g[key]
    assert set(("measured", "reference", "rel_error_pct")) <= set(g["wall_shear_budget"])
