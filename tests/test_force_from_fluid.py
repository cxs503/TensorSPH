"""Force-from-fluid audit: the momentum budget and the three load routes.

The assertions here are invariants, not smoke tests:

* Newton's third law between the fluid's acceleration and the reaction the walls
  report (a bookkeeping identity, so the tolerance is near round-off),
* closure of the control-volume momentum budget when nothing but the pair force
  changes the velocity, and the *order-of-magnitude increase* of the residual
  once the asymmetric viscous Laplacian is switched on (a leak the monitor is
  supposed to see),
* the pair and panel routes agree with each other on the gate face, which is
  what distinguishes a `wall-pressure estimate` deficit from a `force formula`
  deficit,
* the resolution trend of that deficit (it must shrink as ``dx`` shrinks).
"""
import pytest
import torch

from tensorsph import droplet_impact as di
from tensorsph.dam_break import make_column
from tensorsph.force_from_fluid import (
    GateCaseConfig,
    MomentumBudget,
    hydrostatic_reference,
    run_gate_case,
)


def _case(**kw) -> GateCaseConfig:
    base = dict(dx=0.025, steps=0, pool_width=1.0, tank_height=0.5)
    base.update(kw)
    return GateCaseConfig(**base)


def _static(cfg):
    out = run_gate_case(cfg)
    ref = hydrostatic_reference(cfg)
    reaction = {k: v[-1] for k, v in out["reaction_forces"].items()}
    panel = {k: v[-1] for k, v in out["panel_forces"].items()}
    return out, ref, reaction, panel


def test_split_is_opt_in_and_does_not_move_the_solution():
    """The 3-value return contract is unchanged, and split=True changes nothing."""
    cfg = _case()
    dev = di.resolve_device(cfg.device)
    pos, mass = make_column(cfg, dev)
    wall = di.wall_lattice(cfg, dev, top=cfg.tank_top())
    vel = torch.zeros_like(pos)
    plain = di.rhs(pos, vel, mass, cfg, wall=wall)
    assert len(plain) == 3
    split = di.rhs(pos, vel, mass, cfg, wall=wall, split=True)
    assert len(split) == 4
    assert set(split[3]) == {"reaction", "face_pressure"}
    assert torch.equal(plain[0], split[0])
    assert torch.equal(plain[1], split[1])
    assert torch.equal(plain[2], split[2])


def test_reaction_obeys_newtons_third_law_against_the_momentum_change():
    """sum_i m_i a_i - M g must be exactly minus the summed wall reaction."""
    cfg = _case()
    dev = di.resolve_device(cfg.device)
    pos, mass = make_column(cfg, dev)
    wall = di.wall_lattice(cfg, dev, top=cfg.tank_top())
    vel = torch.zeros_like(pos)
    _, _, acc, loads = di.rhs(pos, vel, mass, cfg, wall=wall, split=True)
    weight = torch.tensor([0.0, -cfg.gravity * float(mass.sum())])
    implied_on_fluid = (mass[:, None] * acc).sum(dim=0) - weight
    reported = -loads["reaction"].sum(dim=0)
    assert torch.allclose(implied_on_fluid, reported, rtol=1e-4,
                          atol=1e-3), (implied_on_fluid, reported)


def test_budget_closes_for_the_conservative_solver():
    cfg = _case(steps=120, viscosity=0.0)
    summary = run_gate_case(cfg, record_every=120)["summary"]
    assert summary["residual_relative"] < 1.0e-3, summary["residual_relative"]


def test_viscous_laplacian_leak_is_visible_in_the_budget():
    """The asymmetric Laplacian weight is not momentum conserving; the monitor sees it."""
    inviscid = run_gate_case(_case(steps=120, viscosity=0.0),
                             record_every=120)["summary"]
    viscous = run_gate_case(_case(steps=120, viscosity=0.01),
                            record_every=120)["summary"]
    assert viscous["residual_relative"] > 20.0 * inviscid["residual_relative"]
    # ... and it is a *small* force: convert the momentum residual to an
    # equivalent force over the run and compare with the hydrostatic thrust.
    run_time = viscous["steps"] * viscous["dt"]
    equivalent_force = viscous["residual_norm"] / run_time
    thrust = hydrostatic_reference(_case())["vertical_wall_thrust"]
    assert equivalent_force / thrust < 0.01


def test_pair_and_panel_routes_agree_on_the_gate_face():
    """Two discretizations of the same load must agree well below 15%."""
    _, ref, reaction, panel = _static(_case())
    ratio = reaction["gate"][0] / panel["gate"][0]
    assert abs(ratio - 1.0) < 0.02, ratio
    # ... and both fall short of the analytic thrust, i.e. the shortfall is
    # shared by the routes and cannot be a property of the summation formula.
    pair = reaction["gate"][0] / ref["vertical_wall_thrust"]
    assert 0.6 < pair < 0.95, pair


def test_wall_support_deficit_shrinks_with_resolution():
    """A boundary-support effect must converge under refinement."""
    coarse = _static(_case(dx=0.025))
    fine = _static(_case(dx=0.0125))
    coarse_pair = coarse[2]["gate"][0] / coarse[1]["vertical_wall_thrust"]
    fine_pair = fine[2]["gate"][0] / fine[1]["vertical_wall_thrust"]
    assert fine_pair > coarse_pair, (coarse_pair, fine_pair)
    assert 1.0 - fine_pair < 1.0 - coarse_pair


def test_gateless_case_runs_and_reports_no_gate_load():
    cfg = _case(steps=60)
    out = run_gate_case(cfg, gate=False, record_every=60)
    assert out["summary"]["gate_count"] == 0
    assert out["summary"]["residual_relative"] < 0.05
    assert all(v[0] == 0.0 for v in out["reaction_forces"]["gate"])


def test_budget_accumulator_units():
    """Left-endpoint impulse of a constant force equals force times time."""
    vel = torch.zeros(4, 2)
    mass = torch.full((4,), 2.0)
    b = MomentumBudget(9.81, float(mass.sum()), vel)
    reaction = torch.tensor([[1.0, 0.0]] * 4)
    for _ in range(10):
        b.step(reaction, 0.5)
    assert torch.allclose(b.impulse_reaction, torch.tensor([-20.0, 0.0]))
    assert torch.allclose(b.impulse_gravity,
                          torch.tensor([0.0, -9.81 * 8.0 * 5.0]))
    assert torch.allclose(b.residual, b.momentum_change - (b.impulse_gravity
                                                         + b.impulse_reaction))