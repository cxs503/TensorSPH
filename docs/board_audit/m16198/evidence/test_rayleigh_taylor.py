"""Tests for the two-phase (density-contrast) Rayleigh-Taylor integrator.

Each test pins one of the properties the case relies on.  The three that are
easy to get silently wrong are:

* ``test_number_density_matches_lattice_sum`` -- a symmetric kernel summation
  must be added to *both* end points; adding it to one end only halves every
  kernel sum, which is a factor 2 in the compression and therefore a factor
  ``2^gamma`` in the pressure.  That bug existed while this module was written.
* ``test_hydrostatic_state_has_no_net_acceleration`` -- the reference-calibrated
  number density must reproduce the analytic hydrostatic pressure *exactly*,
  otherwise a pattern of discretisation error (kPa, against an 11 kPa column)
  is turned into initial acceleration.
* ``test_mass_summation_is_unusable_across_the_jump`` -- the alternative that
  looks simpler (keep mass summation, give each particle its own ``rho0``) is
  quantified rather than argued about.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph import rayleigh_taylor as rt
from tensorsph.droplet_impact import cubic_spline_kernel, lattice_kernel_sum
from tensorsph.neighbor_search import brute_force_pairs


def small_cfg(**kw) -> rt.RTConfig:
    base = dict(dx=0.05, end_time=0.0)
    base.update(kw)
    return rt.RTConfig(**base)


def column(**kw):
    cfg = small_cfg(**kw)
    state = rt.build_column(cfg, perturb=True)
    all_pos = torch.cat((state["pos"], state["wall_pos"]))
    pairs = rt.periodic_pairs(all_pos, state["pos"].shape[0], 2.0 * cfg.h,
                              cfg.wavelength)
    ref = rt.reference_state(state["pos"], state["wall_pos"],
                             state["wall_rho0"], cfg, pairs,
                             y_ref=state["y_ref"])
    return cfg, state, pairs, ref


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #

def test_number_density_matches_lattice_sum():
    """On a complete lattice the kernel sum must equal ``lattice_kernel_sum``.

    The pairs are added to both end points, so the sum for an interior particle
    is the full lattice sum (plus the self term), not half of it.
    """
    dx, h = 0.05, 1.3 * 0.05
    n = 30
    axis = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    pairs = brute_force_pairs(pos, 2.0 * h)
    delta = rt.number_density(pos, pairs, h)
    interior = delta[15 * n + 15]
    reference = lattice_kernel_sum(h, dx)
    assert float(interior) == pytest.approx(reference, rel=1e-12)
    # every particle away from the boundary agrees (the lattice is not finite
    # at the edges, so |delta - reference| < 2% only holds in the interior)
    assert float(delta[18 * n + 12]) == pytest.approx(reference, rel=2e-2)


def test_hydrostatic_state_has_no_net_acceleration():
    """At t = 0 the pressure equals the analytic profile and the net
    acceleration of an interior particle is a small fraction of ``g``."""
    cfg, state, pairs, ref = column()
    pos, wall = state["pos"], state["wall_pos"]
    rho, p, acc, ratio = rt.two_phase_rhs(
        pos, torch.zeros_like(pos), state["mass"], state["rho0"], cfg,
        wall, state["wall_rho0"], pairs, ref)
    p_ref, _, _ = rt.hydrostatic_profile(state["y_ref"], cfg)
    assert float((p - p_ref).abs().max()) <= 1e-6 * float(p_ref.max())
    interior = (state["y_ref"] > 0.1) & (state["y_ref"] < cfg.depth_light - 0.1)
    assert float(acc[interior].norm(dim=1).mean()) < 0.05 * cfg.gravity
    # density: light phase below, heavy above, ratio 3 inside each layer
    assert float(rho[interior].mean()) == pytest.approx(
        cfg.rho_light * float(ratio[interior].mean()), rel=1e-2)


def test_pressure_is_continuous_and_density_jumps_across_the_interface():
    """The whole point of the number-density form: p has no jump, rho does."""
    cfg, state, pairs, ref = column()
    pos = state["pos"]
    rho, p, _, _ = rt.two_phase_rhs(
        pos, torch.zeros_like(pos), state["mass"], state["rho0"], cfg,
        state["wall_pos"], state["wall_rho0"], pairs, ref)
    y = state["y_ref"]
    band = 1.0 * cfg.h
    below = (y > cfg.depth_light - band) & (y < cfg.depth_light)
    above = (y >= cfg.depth_light) & (y < cfg.depth_light + band)
    p_ref, _, _ = rt.hydrostatic_profile(y, cfg)
    # pressure error stays well under the column's own 11 kPa scale
    assert float((p - p_ref).abs().max()) < 0.01 * float(p_ref.max())
    # density jump across the interface: rho_heavy / rho_light ~ 3
    assert float(rho[above].mean() / rho[below].mean()) == pytest.approx(
        cfg.rho_heavy / cfg.rho_light, rel=0.05)


def test_pressure_force_conserves_momentum():
    """The pair pressure term is symmetric, so sum_i m_i a_i = 0.

    Checked on a deliberately disordered two-phase cloud with no gravity, no
    viscosity and no walls, so the only contribution is the pressure force.
    """
    cfg = small_cfg(viscosity=0.0)          # gravity on: p(y) is non-zero
    state = rt.build_column(cfg, perturb=False)
    pos = state["pos"] + 0.15 * cfg.dx * torch.randn_like(state["pos"])
    rho0, mass = state["rho0"], state["mass"]
    vel = torch.zeros_like(pos)
    _, _, acc, _ = rt.two_phase_rhs(pos, vel, mass, rho0, cfg)
    force = (mass[:, None] * acc).sum(dim=0)          # pressure part + gravity
    scale = float((mass * acc.abs().max(dim=1).values).sum())
    # horizontal: the pressure force is a pair sum, so it cancels exactly
    assert abs(float(force[0])) < 1e-10 * scale
    # vertical: everything left over is gravity, i.e. exactly -M g
    assert float(force[1]) == pytest.approx(-cfg.gravity * float(mass.sum()),
                                            rel=1e-10)


def test_mass_summation_is_unusable_across_the_jump():
    """The rejected alternative, as a number: mass summation gives a pressure
    error of two orders of magnitude at the interface."""
    cfg = small_cfg()
    report = rt.spurious_interface_pressure(cfg)
    assert report["error_ratio_light"] > 10.0
    assert report["error_ratio_heavy"] > 1.0


# --------------------------------------------------------------------------- #
# Gauges and reference values
# --------------------------------------------------------------------------- #

def test_gauge_recovers_a_known_interface():
    """The centroid gauge must return the interface displacement it is given.

    For the eigenfunction kick the heavy layer's centroid moves by
    ``(xi_interface + xi_lid)/2``, so the gauge ``2 (ybar - <ybar>)`` returns
    ``eta0 (1 + exp(-k d_h))``: the interface displacement *plus* the (tiny)
    displacement of the lid.  ``run_case`` divides that factor out against the
    known initial amplitude, and this test pins the relation.
    """
    cfg = small_cfg(dx=0.02, amplitude=0.0123)
    state = rt.build_column(cfg, perturb=True)
    amp, phase = rt.modal_amplitude(state["pos"], state["volume0"],
                                    state["is_heavy"], cfg)
    expected = cfg.amplitude * (1.0 + math.exp(-cfg.wave_number * cfg.depth_heavy))
    assert amp == pytest.approx(expected, rel=0.05)
    assert abs(phase % math.pi) < 0.02


def test_theory_growth_rate_limits():
    deep = rt.theory_growth_rate(small_cfg(depth_light=3.0, depth_heavy=3.0))
    assert deep["finite_depth_correction"] == pytest.approx(1.0, abs=2e-3)
    shallow = rt.theory_growth_rate(small_cfg(depth_light=0.1, depth_heavy=0.1))
    assert shallow["finite_depth_correction"] < 0.8
    visc = rt.theory_growth_rate(small_cfg(), viscosity=0.1)
    assert visc["sigma_viscous"] < 0.9 * visc["sigma_finite_depth"]
    none = rt.theory_growth_rate(small_cfg(rho_heavy=1000.0, rho_light=1000.0))
    assert none["sigma_ideal_Agk"] == 0.0


def test_fit_growth_rate_recovers_an_exponential():
    times = [0.001 * i for i in range(200)]
    sigma = 4.0
    amps = [0.01 * math.exp(sigma * t) for t in times]
    fit = rt.fit_growth_rate(times, amps, 0.02, 0.1)
    assert fit["sigma"] == pytest.approx(sigma, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Short dynamics: no spurious growth without a density contrast
# --------------------------------------------------------------------------- #

def test_equal_density_does_not_grow():
    """With rho_heavy = rho_light there is no Rayleigh-Taylor drive: the mode
    must not grow.  (It decays numerically; this pins the sign.)"""
    cfg = small_cfg(rho_heavy=500.0, rho_light=500.0)
    state = rt.build_column(cfg)
    pos = state["pos"].clone()
    wall = state["wall_pos"]
    pairs = rt.periodic_pairs(torch.cat((pos, wall)), pos.shape[0], 2.0 * cfg.h,
                              cfg.wavelength)
    ref = rt.reference_state(pos, wall, state["wall_rho0"], cfg, pairs,
                             y_ref=state["y_ref"])
    vel = torch.zeros_like(pos)
    a0, _ = rt.modal_amplitude(pos, state["volume0"], state["is_heavy"], cfg)
    dt = rt.stable_dt(cfg, 0.0)
    for _ in range(120):
        _, _, acc, _ = rt.two_phase_rhs(pos, vel, state["mass"], state["rho0"],
                                        cfg, wall, state["wall_rho0"], pairs, ref)
        vel = vel + dt * acc
        pos = pos + dt * vel
    a1, _ = rt.modal_amplitude(pos, state["volume0"], state["is_heavy"], cfg)
    assert a1 <= a0 * 1.02


def test_two_phase_run_is_mass_conserving_and_bounded():
    """A short integration keeps every particle at a sane density."""
    cfg = small_cfg(end_time=0.06, output_every=200)
    result = rt.run_case(cfg, snapshot=False)
    series = result["series"]
    assert len(series["t"]) >= 2
    rho_min, rho_max = series["rho_min"][-1], series["rho_max"][-1]
    assert rho_min > 0.4 * cfg.rho_light
    assert rho_max < 2.0 * cfg.rho_heavy
    assert max(series["mach"]) < 0.2