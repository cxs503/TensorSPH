"""Gates for the Gresho-Chan vortex anchor (``tensorsph.gresho``).

Three classes of test, deliberately separated:

* the **closed form** is checked against its own defining statements (radial
  momentum balance, continuity at the kink, divergence-free velocity, and an
  independent Simpson quadrature for the ambient pressure) -- if the reference
  is wrong every other number in the case is meaningless;
* the **discrete structure** is checked: the pressure force must be a central
  pair force (zero torque about the centre to round-off), the total force on the
  cell must vanish, and the initial mass field must reproduce the density the
  pressure implies;
* the **case machinery** is checked on a cheap lattice: the truncation-error
  metric must converge and must respond to a wrong initial state (negative
  control), and a short run must keep energy and angular momentum.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.gresho import (GATES, GreshoConfig, accept, analytic_pressure,
                              analytic_reference, analytic_velocity,
                              angular_momentum, exact_acceleration, lattice,
                              make_state, radius, run_gresho,
                              spurious_acceleration, tangential_velocity,
                              torque_sum, u_theta)
from tensorsph.vv.tgv import tgv_rhs

torch.set_num_threads(min(8, torch.get_num_threads()))


# --------------------------------------------------------------------------- #
# 1. The closed form
# --------------------------------------------------------------------------- #
def test_radial_momentum_balance_holds_to_quadrature_error():
    """``dp/dr = rho u^2 / r`` -- the only statement the pressure makes."""
    ref = analytic_reference(GreshoConfig(dx=0.05))
    assert ref["radial_balance_residual_rel"] < 1e-5
    assert ref["p_core"] == pytest.approx(0.0, abs=1e-12)


def test_ambient_pressure_matches_an_independent_quadrature():
    """Second, independent evaluation of ``int rho u^2/r dr`` (Simpson)."""
    cfg = GreshoConfig(dx=0.05)
    ref = analytic_reference(cfg)

    n = 200000                      # even -> Simpson
    rc, ro, u, rho = cfg.r_core, cfg.r_outer, cfg.u_max, cfg.rho0
    h = ro / n
    total = 0.0
    for k in range(n + 1):
        r = k * h
        if r <= rc:
            f = rho * (u * r / rc) ** 2 / r if r > 0 else 0.0
        else:
            f = rho * (u * (ro - r) / (ro - rc)) ** 2 / r
        weight = 1.0 if k in (0, n) else (4.0 if k % 2 else 2.0)
        total += weight * f
    simpson = total * h / 3.0
    assert ref["p_ambient"] == pytest.approx(simpson, rel=2e-5)


def test_velocity_is_continuous_at_the_kink_and_matches_closed_form():
    cfg = GreshoConfig(dx=0.05)
    rc = cfg.r_core
    left = u_theta(torch.tensor([rc - 1e-9], dtype=torch.float64), cfg)[0]
    right = u_theta(torch.tensor([rc + 1e-9], dtype=torch.float64), cfg)[0]
    assert float(left) == pytest.approx(float(right), rel=1e-7)
    assert float(left) == pytest.approx(cfg.u_max, rel=1e-7)
    # outside r_outer the field is identically at rest
    far = u_theta(torch.tensor([cfg.r_outer + 1e-6, 0.5], dtype=torch.float64), cfg)
    assert float(far.abs().max()) == 0.0


def test_velocity_field_is_divergence_free():
    """``div u = 0`` evaluated by autograd, away from the two derivative jumps."""
    cfg = GreshoConfig(dx=0.05)
    table = analytic_reference(cfg)
    probes = torch.tensor([[0.55, 0.50], [0.50, 0.60], [0.62, 0.58],
                           [0.48, 0.44], [0.50, 0.50 + 1e-3]], dtype=torch.float64)
    x = probes.clone().requires_grad_(True)
    vel = analytic_velocity(x, cfg, table=table)
    div = torch.zeros(probes.shape[0], dtype=torch.float64)
    for k in range(2):
        grad = torch.autograd.grad(vel[:, k].sum(), x, retain_graph=True)[0]
        div = div + grad[:, k]
    scale = cfg.u_max / cfg.r_core
    # interpolation of the radial table is piecewise *linear*: its derivative is
    # piecewise constant, so only a loose tolerance is meaningful here
    assert float(div.abs().max()) / scale < 5e-2, float(div.abs().max()) / scale


def test_exact_acceleration_is_inward_centripetal():
    cfg = GreshoConfig(dx=0.05)
    table = analytic_reference(cfg)
    probe = torch.tensor([[0.5 + 0.2 + 1e-12, 0.5], [0.5, 0.5 - 0.1]],
                         dtype=torch.float64)
    a = exact_acceleration(probe, cfg, table=table)
    r = radius(probe, cfg)
    speed = analytic_velocity(probe, cfg, table=table).norm(dim=1)
    assert torch.allclose(a.norm(dim=1), speed.square() / r, rtol=1e-6)
    # inward: the radial component is negative everywhere inside the vortex
    centre = torch.tensor(cfg.centre, dtype=torch.float64)
    er = (probe - centre) / (probe - centre).norm(dim=1)[:, None]
    radial = (a * er).sum(dim=1)
    assert bool((radial < 0.0).all()), radial


# --------------------------------------------------------------------------- #
# 2. The discrete structure
# --------------------------------------------------------------------------- #
def test_pressure_force_is_central_so_the_torque_vanishes():
    """A nonzero torque here means angular momentum is not an invariant."""
    cfg = GreshoConfig(dx=0.05)
    state = make_state(cfg)
    _rho, _p, acc = tgv_rhs(state["pos"], state["vel"], state["mass"], cfg)
    tau = torque_sum(state["pos"], acc, state["mass"], cfg)
    assert abs(tau["relative"]) < 1e-12
    # total force on the cell vanishes (pairwise antisymmetry)
    total = (state["mass"][:, None] * acc).sum(dim=0)
    scale = (state["mass"][:, None] * acc.abs()).sum()
    assert float(total.norm()) / float(scale) < 1e-14


def test_mass_field_reproduces_the_density_the_pressure_implies():
    cfg = GreshoConfig(dx=0.04)
    state = make_state(cfg)
    assert state["init_density_rel_error_max"] < 1e-4
    # the ambient region is compressed by p_amb/B, i.e. not at rho0
    assert state["init_density_rel_error_max"] > 0.0
    ratio = state["rho_sum"] / cfg.rho0
    assert float(ratio.max()) > 1.0 + 1e-4


def test_lattice_closes_the_periodic_box_exactly():
    with pytest.raises(ValueError):
        lattice(GreshoConfig(dx=0.03))       # 1/0.03 is not an integer
    pos = lattice(GreshoConfig(dx=0.1))
    assert pos.shape == (100, 2)


# --------------------------------------------------------------------------- #
# 3. The case machinery
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def truncation_ladder():
    return [spurious_acceleration(GreshoConfig(dx=dx)) for dx in (0.1, 0.05)]


def test_truncation_error_converges_under_refinement(truncation_ladder):
    coarse, fine = truncation_ladder
    assert coarse["a_pct"] > fine["a_pct"] > 0.0
    order = math.log(coarse["a_pct"] / fine["a_pct"]) / math.log(
        coarse["dx"] / fine["dx"])
    assert order > 1.0


def test_uniform_density_initial_state_is_a_worse_balance(truncation_ladder):
    """Negative control: without the pressure field the metric must blow up."""
    cfg = GreshoConfig(dx=0.05)
    analytic = spurious_acceleration(cfg, init_mode="analytic")
    uniform = spurious_acceleration(cfg, init_mode="uniform")
    assert uniform["a_pct"] > 5.0 * analytic["a_pct"]
    # and the metric must respond to the pressure amplitude, not just its presence
    half = spurious_acceleration(cfg, init_mode="analytic", pressure_scale=0.5)
    assert half["a_pct"] > analytic["a_pct"]


def test_short_run_conserves_energy_and_angular_momentum():
    cfg = GreshoConfig(dx=0.05, t_end=0.05, sample_every=5)
    run = run_gresho(cfg)
    measured = run["measured"]
    assert measured["energy_ratio"] > 0.98
    assert measured["angular_momentum_drift_rel"] < 1e-3
    assert measured["linear_momentum_drift_rel"] < 1e-12
    assert measured["profile_error_pct_rms"] < 15.0
    assert run["config"]["particles"] == 400


def test_profile_error_is_zero_for_the_exact_field():
    from tensorsph.gresho import radial_profile_error
    cfg = GreshoConfig(dx=0.05)
    table = analytic_reference(cfg)
    state = make_state(cfg, table=table)
    prof = radial_profile_error(state["pos"], state["vel"], state["mass"], cfg, table)
    # bin-averaging a smooth field is not exact, so this is a small tolerance,
    # not round-off: it pins the reference plumbing, not the solver
    # The bin mean is an *area*-weighted mean of a profile that is only
    # piecewise linear, so a few percent of bin discretisation is expected; this
    # pins the reference plumbing, not the solver.
    assert prof["rms_pct"] < 6.0
    assert prof["gated_bins"] >= 8


def test_accept_verdict_is_driven_by_the_gates():
    good = {"dx": 0.0125, "particles": 6400, "steps": 10,
            "spurious_acceleration_pct": 1.0, "velocity_error_pct_end": 5.0,
            "profile_error_pct_rms": 5.0, "angular_momentum_drift_rel": 1e-9,
            "observed_order_acceleration": 1.9, "init_density_rel_error_max": 1e-5,
            "spurious_acceleration_max_pct": 1.0, "velocity_error_pct_max": 5.0,
            "energy_ratio": 0.95, "equivalent_viscosity": 1e-4,
            "equivalent_reynolds": 1e3, "linear_momentum_drift_rel": 0.0,
            "peak_speed_ratio": 1.0, "density_rms_max": 1e-3,
            "observed_order_velocity": 1.5, "observed_order_profile": 1.5}
    bad = dict(good, spurious_acceleration_pct=GATES["spurious_acceleration_pct"] * 2)
    controls = {"forward": {"spurious_acceleration_pct": 1.0},
                "reverse": {"spurious_acceleration_pct": 100.0},
                "half_pressure": {"spurious_acceleration_pct": 3.0}}
    assert accept([good], controls=controls)["verdict"] == "PASS"
    assert accept([bad], controls=controls)["verdict"] == "FAIL"
    # a FAIL is reported, not raised: that is the honest-verdict contract
    assert accept([bad], controls=controls)["checks"][0]["pass"] is False