"""Acceptance tests for CASE-SPH-SOD-SHOCK-COMPRESSIBLE (compressible SPH Sod tube).

Reference: G. A. Sod, J. Comput. Phys. 27(1):1-31 (1978),
doi:10.1016/0021-9991(78)90023-2 ; scheme: Inutsuka (2002) JCP 179:238
(non-Riemannian limit of Godunov-SPH pressure) + Monaghan artificial viscosity.

Every assertion here is a *measured* criterion of the case contract:
plateau values / wave positions within 5% (and 3%) of the exact solution at
t = 0.2, plus the numerical properties the scheme must have (exact contact
equilibrium, quiescent uniform state, conservation, determinism).
"""
import numpy as np
import pytest

from tensorsph.compressible import sod as S
from tensorsph.wcsph.sod_shock import _cubic_w_1d

# published values (Sod 1978 / Toro 1999 table 4.1) — cross-check only
LIT = {"p_star": 0.30313, "u_star": 0.92745,
       "rho_star_L": 0.42632, "rho_star_R": 0.26557}


@pytest.fixture(scope="module")
def production():
    cfg = S.SodSPHConfig(n_particles=800)
    res = S.run_sod_sph(cfg, output_times=[0.2])
    met = S.evaluate_metrics(res["snapshots"][-1], cfg)
    return cfg, res, met


# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------
def test_exact_star_state_matches_literature():
    """Iterative Riemann solver reproduces published Sod star values to 0.05%."""
    star = S.exact_star_state()
    for key, ref in LIT.items():
        assert abs(star[key] - ref) / ref < 5e-4, (key, star[key], ref)


def test_exact_wave_positions_at_t02():
    """Rarefaction head/tail, contact and shock positions at t=0.2."""
    w = S.exact_wave_positions(0.2)
    assert w["x_rarefaction_head"] == pytest.approx(0.2634, abs=2e-3)
    assert w["x_rarefaction_tail"] == pytest.approx(0.4860, abs=2e-3)
    assert w["x_contact"] == pytest.approx(0.6855, abs=2e-3)
    assert w["x_shock"] == pytest.approx(0.8504, abs=2e-3)
    # ordering required for any admissible solution
    assert (w["x_rarefaction_head"] < w["x_rarefaction_tail"]
            < w["x_contact"] < w["x_shock"])


# ---------------------------------------------------------------------------
# discretisation properties
# ---------------------------------------------------------------------------
def test_contact_discontinuity_equilibrium():
    """Static 1:8 density jump at uniform pressure must feel *zero* force.

    This is the defect that rules out the p_i/rho_i^2 + p_j/rho_j^2 form and
    justifies the arithmetic average used in the solver.
    """
    acc = S.contact_force_check()
    assert acc["arithmetic"] < 1e-6 * 1.0        # ~1e-11 measured
    assert acc["p_over_rho2"] > 1.0e2            # ~5.0e3 measured


def test_uniform_state_is_quiescent():
    """A uniform gas between mirror walls must not accelerate or change density."""
    cfg = S.SodSPHConfig(n_particles=200)
    dx = cfg.dx
    x = cfg.x_left + (np.arange(cfg.n_particles) + 0.5) * dx
    rho = np.full(cfg.n_particles, 0.7)
    e = np.full(cfg.n_particles, 1.3 / ((cfg.gamma - 1.0) * 0.7))
    m = rho * dx
    drho, acc, de, p, cs = S._derivs(x, rho, np.zeros_like(x), e, m, cfg)
    assert np.max(np.abs(acc)) < 1e-8
    assert np.max(np.abs(drho)) < 1e-12
    assert np.max(np.abs(de)) < 1e-12
    assert np.allclose(p, 1.3)


def test_kernel_normalised_in_1d():
    """Cubic B-spline (re-used from tensorsph.wcsph) integrates to one in 1-D."""
    grid = np.linspace(-2.0, 2.0, 40001)
    integral = np.sum(_cubic_w_1d(grid, 1.0)) * (grid[1] - grid[0])
    assert integral == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# the acceptance contract: <5% (and <3%) at t = 0.2
# ---------------------------------------------------------------------------
def test_production_max_error_below_5pct(production):
    cfg, res, met = production
    assert res["t_final"] == pytest.approx(0.2, abs=1e-12)
    assert met["max_abs_rel_err"] < 0.05, met["rel_err"]
    assert met["pass_5pct"] is True


def test_production_max_error_below_3pct(production):
    cfg, res, met = production
    assert met["max_abs_rel_err"] < 0.03, met["rel_err"]


def test_production_plateau_values(production):
    """Individual plateaus: star densities, star pressure and velocity."""
    cfg, res, met = production
    rel = met["rel_err"]
    assert abs(rel["rho_star_L"]) < 0.02
    assert abs(rel["rho_star_R"]) < 0.02
    assert abs(rel["p_star_L"]) < 0.02
    assert abs(rel["p_star_R"]) < 0.02
    assert abs(rel["u_star"]) < 0.02
    # undisturbed plateaus must stay put
    assert abs(rel["rho_L"]) < 1e-5 and abs(rel["rho_R"]) < 1e-5
    assert abs(rel["p_L"]) < 1e-5 and abs(rel["p_R"]) < 1e-5


def test_production_wave_positions(production):
    cfg, res, met = production
    rel = met["rel_err"]
    assert abs(rel["x_shock"]) < 0.02
    assert abs(rel["x_contact"]) < 0.02


def test_pressure_continuous_across_contact(production):
    """Exact solution has p constant across the contact: |dp|/p* must be small."""
    cfg, res, met = production
    assert met["contact_pressure_jump"] < 0.02


def test_field_errors_small(production):
    cfg, res, met = production
    assert met["l1"]["rho"] < 0.02
    assert met["l1"]["p"] < 0.02


# ---------------------------------------------------------------------------
# conservation and reproducibility
# ---------------------------------------------------------------------------
def test_mass_conservation(production):
    cfg, res, met = production
    assert abs(res["mass_final"] - res["mass_initial"]) / res["mass_initial"] < 1e-12


def test_energy_budget_dissipative_only(production):
    """With artificial viscosity the total energy may only decrease, and little."""
    cfg, res, met = production
    assert -0.01 < res["energy_rel_change"] <= 1e-9


def test_deterministic_repeat():
    """Identical configuration (no RNG) must give bit-identical output."""
    cfg = S.SodSPHConfig(n_particles=200)
    a = S.run_sod_sph(cfg, output_times=[0.1])["snapshots"][-1]
    b = S.run_sod_sph(cfg, output_times=[0.1])["snapshots"][-1]
    for k in ("x", "rho", "u", "p"):
        assert np.array_equal(a[k], b[k]), k


def test_resolution_improves_l1():
    """Refinement must reduce the L1 density error (convergence, not stagnation)."""
    errs = []
    for n in (200, 800):
        cfg = S.SodSPHConfig(n_particles=n)
        res = S.run_sod_sph(cfg, output_times=[0.2])
        errs.append(S.evaluate_metrics(res["snapshots"][-1], cfg)["l1"]["rho"])
    assert errs[1] < errs[0], errs


def test_neighbour_count_adequate(production):
    """h = 2.5 dx must keep enough neighbours in the *expanded* star-left plateau."""
    cfg, res, met = production
    assert res["neighbours"]["min"] >= 3


# ---------------------------------------------------------------------------
# the documented failure mode: h too small starves the expanded region
# ---------------------------------------------------------------------------
def test_small_h_is_neighbour_starved_and_wrong():
    """h = 1.3 dx leaves 1-2 neighbours in the expanded gas -> >5% error (documented)."""
    cfg = S.SodSPHConfig(n_particles=400, h_factor=1.3)
    res = S.run_sod_sph(cfg, output_times=[0.2])
    met = S.evaluate_metrics(res["snapshots"][-1], cfg)
    assert met["max_abs_rel_err"] > 0.05
    assert res["neighbours"]["min"] <= 2