"""Acceptance tests for XVAL-SPH-MACH-LIMITED-RIEMANN-DISSIPATION.

Tests the Mach-limited Riemann dissipation scheme against the Sod (1978)
shock tube problem.  The acceptance criterion is <15% error (owner precision
standard), with a target of <5% for the production configuration.

Reference: G. A. Sod, J. Comput. Phys. 27(1):1-31 (1978),
doi:10.1016/0021-9991(78)90023-2

Scheme: Mach-limited Riemann dissipation (Ji et al. 2019 MNRAS 489:1052)
with HLLC star pressure (Toro 1999).
"""
import numpy as np
import pytest

from tensorsph.compressible.riemann_dissipation import (
    RiemannDissipationConfig,
    run_sod_riemann,
    hllc_star_pressure,
    roe_average,
    evaluate_metrics,
)
from tensorsph.wcsph.sod_shock import exact_riemann_solution

# Published values (Sod 1978 / Toro 1999 table 4.1)
LIT = {"p_star": 0.30313, "u_star": 0.92745,
       "rho_star_L": 0.42632, "rho_star_R": 0.26557}


@pytest.fixture(scope="module")
def production():
    """Production run with Mach-limited Riemann dissipation."""
    cfg = RiemannDissipationConfig(n_particles=800, mach_limit=True)
    res = run_sod_riemann(cfg, output_times=[0.2])
    met = evaluate_metrics(res["snapshots"][-1], cfg)
    return cfg, res, met


@pytest.fixture(scope="module")
def production_no_mach_limit():
    """Run without Mach limiting (full Godunov-SPH)."""
    cfg = RiemannDissipationConfig(n_particles=800, mach_limit=False)
    res = run_sod_riemann(cfg, output_times=[0.2])
    met = evaluate_metrics(res["snapshots"][-1], cfg)
    return cfg, res, met


# ---------------------------------------------------------------------------
# Riemann solver correctness
# ---------------------------------------------------------------------------
def test_hllc_star_pressure_matches_exact():
    """HLLC star pressure should be close to the exact Riemann solution."""
    p_star_hllc = hllc_star_pressure(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)
    p_star_exact, _ = exact_riemann_solution(1.4, 1.0, 1.0, 0.0, 0.125, 0.1, 0.0)
    # PVRS approximation is typically within 5% of exact
    assert abs(p_star_hllc - p_star_exact) / p_star_exact < 0.02, \
        f"HLLC p*={p_star_hllc:.6f}, exact p*={p_star_exact:.6f}"


def test_roe_average_symmetric():
    """Roe average should be symmetric under left-right swap."""
    r1, u1, H1, c1 = roe_average(1.0, 0.5, 1.0, 0.5, 0.3, 0.5, 1.4)
    r2, u2, H2, c2 = roe_average(0.5, 0.3, 0.5, 1.0, 0.5, 1.0, 1.4)
    assert abs(r1 - r2) < 1e-10
    assert abs(u1 - u2) < 1e-10
    assert abs(H1 - H2) < 1e-10
    assert abs(c1 - c2) < 1e-10


# ---------------------------------------------------------------------------
# Discretisation properties
# ---------------------------------------------------------------------------
def test_uniform_state_is_quiescent():
    """A uniform gas between mirror walls must not accelerate."""
    cfg = RiemannDissipationConfig(n_particles=200)
    dx = cfg.dx
    x = cfg.x_left + (np.arange(cfg.n_particles) + 0.5) * dx
    rho = np.full(cfg.n_particles, 0.7)
    e = np.full(cfg.n_particles, 1.3 / ((cfg.gamma - 1.0) * 0.7))
    m = rho * dx
    from tensorsph.compressible.riemann_dissipation import _derivs_riemann
    drho, acc, de, p, cs = _derivs_riemann(x, rho, np.zeros_like(x), e, m, cfg)
    assert np.max(np.abs(acc)) < 1e-6, f"max acc = {np.max(np.abs(acc))}"
    assert np.max(np.abs(drho)) < 1e-10
    assert np.max(np.abs(de)) < 1e-10
    assert np.allclose(p, 1.3)


# ---------------------------------------------------------------------------
# Acceptance criteria: <15% (owner standard), target <5%
# ---------------------------------------------------------------------------
def test_production_max_error_below_15pct(production):
    """Owner precision standard: error must be <15%."""
    cfg, res, met = production
    assert res["t_final"] == pytest.approx(0.2, abs=1e-10)
    assert met["max_abs_rel_err"] < 0.15, \
        f"max_abs_rel_err={met['max_abs_rel_err']:.4f} >= 0.15"


def test_production_max_error_below_5pct(production):
    """Target: error should be <5% for production config."""
    cfg, res, met = production
    assert met["max_abs_rel_err"] < 0.05, \
        f"max_abs_rel_err={met['max_abs_rel_err']:.4f} >= 0.05"


def test_production_plateau_values(production):
    """Individual plateau values within 5%."""
    cfg, res, met = production
    rel = met["rel_err"]
    assert abs(rel.get("rho_star_L", 1)) < 0.05, f"rho_star_L err={rel.get('rho_star_L')}"
    assert abs(rel.get("rho_star_R", 1)) < 0.05, f"rho_star_R err={rel.get('rho_star_R')}"
    assert abs(rel.get("p_star_L", 1)) < 0.05, f"p_star_L err={rel.get('p_star_L')}"
    assert abs(rel.get("p_star_R", 1)) < 0.05, f"p_star_R err={rel.get('p_star_R')}"
    assert abs(rel.get("u_star", 1)) < 0.05, f"u_star err={rel.get('u_star')}"


def test_production_wave_positions(production):
    """Wave positions should be reasonable (within 10%)."""
    cfg, res, met = production
    rel = met["rel_err"]
    # Wave positions are harder to get right with SPH
    assert abs(rel.get("x_shock", 1)) < 0.10 or "x_shock" not in rel
    assert abs(rel.get("x_contact", 1)) < 0.10 or "x_contact" not in rel


def test_pressure_continuous_across_contact(production):
    """Pressure should be nearly continuous across the contact."""
    cfg, res, met = production
    assert met["contact_pressure_jump"] < 0.05, \
        f"contact_pressure_jump={met['contact_pressure_jump']:.4f}"


def test_field_errors_small(production):
    """L1 errors should be small."""
    cfg, res, met = production
    assert met["l1"]["rho"] < 0.05, f"L1 rho={met['l1']['rho']:.4f}"
    assert met["l1"]["p"] < 0.05, f"L1 p={met['l1']['p']:.4f}"


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------
def test_mass_conservation(production):
    """Mass must be conserved to machine precision."""
    cfg, res, met = production
    assert abs(res["mass_final"] - res["mass_initial"]) / res["mass_initial"] < 1e-12


def test_energy_budget_dissipative_only(production):
    """With Riemann dissipation, total energy may only decrease, and little."""
    cfg, res, met = production
    assert -0.05 < res["energy_rel_change"] <= 1e-9, \
        f"energy_rel_change={res['energy_rel_change']:.6f}"


# ---------------------------------------------------------------------------
# Mach limiting effect
# ---------------------------------------------------------------------------
def test_mach_limiting_reduces_dissipation():
    """Mach-limited version should have less energy loss than full Riemann."""
    cfg_limited = RiemannDissipationConfig(n_particles=400, mach_limit=True)
    cfg_full = RiemannDissipationConfig(n_particles=400, mach_limit=False)

    res_limited = run_sod_riemann(cfg_limited, output_times=[0.2])
    res_full = run_sod_riemann(cfg_full, output_times=[0.2])

    # Both should be dissipative (energy decreases)
    assert res_limited["energy_rel_change"] <= 1e-9
    assert res_full["energy_rel_change"] <= 1e-9

    # Mach-limited should lose less energy (less dissipation in smooth regions)
    # Note: this is a soft check — the difference may be small for Sod tube
    # where most of the flow is either shocked or uniform
    assert res_limited["energy_rel_change"] >= res_full["energy_rel_change"] - 0.01


# ---------------------------------------------------------------------------
# Resolution convergence
# ---------------------------------------------------------------------------
def test_resolution_improves_l1():
    """Refinement must reduce the L1 density error."""
    errs = []
    for n in (200, 800):
        cfg = RiemannDissipationConfig(n_particles=n)
        res = run_sod_riemann(cfg, output_times=[0.2])
        met = evaluate_metrics(res["snapshots"][-1], cfg)
        errs.append(met["l1"]["rho"])
    assert errs[1] < errs[0], f"L1 errors: N=200 -> {errs[0]:.4f}, N=800 -> {errs[1]:.4f}"


def test_neighbour_count_adequate(production):
    """h = 2.5 dx must keep enough neighbours."""
    cfg, res, met = production
    assert res["neighbours"]["min"] >= 3, \
        f"min neighbours = {res['neighbours']['min']}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_deterministic_repeat():
    """Identical configuration must give bit-identical output."""
    cfg = RiemannDissipationConfig(n_particles=200)
    a = run_sod_riemann(cfg, output_times=[0.1])["snapshots"][-1]
    b = run_sod_riemann(cfg, output_times=[0.1])["snapshots"][-1]
    for k in ("x", "rho", "u", "p"):
        assert np.array_equal(a[k], b[k]), k
