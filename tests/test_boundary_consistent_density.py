"""Boundary-consistent summation density: invariants and the repaired closure.

``docs/force_from_fluid/README.md`` recorded an honest FAIL: the static
hydrostatic column supported 73.4 % of its own weight on the floor and 74.0 % of
``0.5 rho g H^2`` on the gate, with the *pressure gradient of the two
wall-adjacent rows inverted*. The cause is that the wall particles entered the
fluid's summation density with the nominal ``rho0 dx^2`` while the fluid beside
them carries the EOS density of its own hydrostatic pressure; at ``c0 = 20 m/s``,
``gamma = 7`` the Tait law amplifies the 0.3 % density step into a 30 % pressure
step.

The assertions below are invariants, not smoke tests:

* the switch is opt-in and bitwise invisible when off (every published number),
* it removes the *attribution* of the deficit: with the correction on, the two
  wall-adjacent rows stop pushing downwards (their band contribution goes from
  about -0.11 to about +0.04 of the column weight) and the floor reaction moves
  from 0.74 to >= 0.92 of the closed form,
* the remaining error is inside the owner's 15 % acceptance band at every
  resolution of the ladder, and the loads on the gate follow,
* the exact-EOS oracle wall density (diagnostic, not a scheme) closes the
  deficit completely, which is what makes the wall mass — not the force formula,
  not the free-surface truncation — the identified cause,
* ``density_renormalization`` alone does *not* move the metric (it removes
  support truncation, not the density step), so the two switches are not
  redundant,
* the conservative transient keeps mass and the momentum budget untouched.
"""
import torch
import pytest

from tensorsph import droplet_impact as di
from tensorsph.dam_break import make_column
from tensorsph.droplet_impact import (hydrostatic_density, tait_pressure,
                                      wall_lattice)
from tensorsph.force_from_fluid import (GateCaseConfig, gate_lattice,
                                        group_force, hydrostatic_reference,
                                        run_gate_case, wall_groups)


def _setup(dx=0.02, **kw):
    cfg = GateCaseConfig(dx=dx, steps=0, dt=0.25 * dx / 20.0, **kw)
    dev = di.resolve_device(cfg.device)
    pos, mass = make_column(cfg, dev)
    base = wall_lattice(cfg, dev, top=cfg.tank_top())
    wall = torch.cat((base, gate_lattice(cfg, dev)), dim=0)
    groups = wall_groups(wall, cfg, base.shape[0])
    return cfg, pos, mass, wall, groups


def _loads(cfg, pos, mass, wall, groups):
    rho, _, acc, loads = di.rhs(pos, torch.zeros_like(pos), mass, cfg,
                                wall=wall, dt=cfg.dt, split=True)
    ref = hydrostatic_reference(cfg)
    reaction = group_force(loads["reaction"], groups)
    panel = float(loads["face_pressure"][groups["gate_face"]].sum()) * cfg.dx
    return {
        "rho": rho,
        "acc": acc,
        "floor": -float(reaction["floor"][1]) / ref["fluid_weight"],
        "gate_pair": float(reaction["gate"][0]) / ref["vertical_wall_thrust"],
        "gate_panel": panel / ref["vertical_wall_thrust"],
        "p": tait_pressure(rho, cfg),
        "mass": mass,
        "pos": pos,
        "ref": ref,
    }


def test_switch_is_off_by_default_and_bitwise_invisible():
    base = _setup()
    off = _loads(*base)
    explicit = _loads(*_setup(boundary_consistent_density=False))
    on = _loads(*_setup(boundary_consistent_density=True))
    assert torch.equal(off["rho"], explicit["rho"])
    assert torch.equal(off["acc"], explicit["acc"])
    assert off["floor"] == explicit["floor"] == pytest.approx(off["floor"])
    # ... and the switch must actually change the state.
    assert not torch.equal(off["rho"], on["rho"])


def test_switch_cannot_be_combined_with_compression_regularization():
    cfg, pos, mass, wall, _ = _setup(density_regularization="compression",
                                     boundary_consistent_density=True)
    with pytest.raises(ValueError, match="boundary_consistent_density"):
        di.rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall, dt=cfg.dt)


def test_wall_adjacent_rows_stop_pushing_downwards():
    """The deficit sits in the two wall rows; the switch must remove it there."""
    for kw, floor_lo, band_lo in ((dict(), 0.70, -0.20),
                                  (dict(boundary_consistent_density=True),
                                   0.92, 0.0)):
        cfg, pos, mass, wall, groups = _setup(**kw)
        out = _loads(cfg, pos, mass, wall, groups)
        weight = cfg.gravity * float(mass.sum())
        band = 0.0
        for k in (0, 1):
            sel = (pos[:, 1] >= k * cfg.dx) & (pos[:, 1] < (k + 1) * cfg.dx)
            band += float((mass[sel] * (out["acc"][sel, 1]
                                        + cfg.gravity)).sum()) / weight
        if kw:
            assert out["floor"] > floor_lo, out["floor"]
            assert band > band_lo, band
            assert out["gate_pair"] > 0.90, out["gate_pair"]
            assert out["gate_panel"] > 0.90, out["gate_panel"]
        else:
            assert out["floor"] < 0.80, out["floor"]
            assert band < -0.05, band


def test_repaired_static_loads_land_inside_the_15_percent_acceptance():
    for dx in (0.02, 0.01):
        cfg, pos, mass, wall, groups = _setup(dx=dx,
                                             boundary_consistent_density=True)
        out = _loads(cfg, pos, mass, wall, groups)
        assert abs(out["floor"] - 1.0) < 0.15, (dx, out["floor"])
        assert abs(out["gate_pair"] - 1.0) < 0.15, (dx, out["gate_pair"])
        assert abs(out["gate_panel"] - 1.0) < 0.15, (dx, out["gate_panel"])
        # pair and panel remain two discretizations of one load
        assert abs(out["gate_pair"] / out["gate_panel"] - 1.0) < 0.03


def test_exact_eos_wall_density_closes_the_deficit_completely():
    """Diagnostic ceiling: the wall mass alone explains the shortfall."""
    cfg, pos, mass, wall, groups = _setup()
    rho_w = hydrostatic_density(wall[:, 1], cfg, surface=cfg.column_height)
    override = (rho_w * cfg.dx * cfg.dx).to(wall.dtype)
    original = di.wall_mass
    di.wall_mass = lambda c, nw, device: override
    try:
        out = _loads(cfg, pos, mass, wall, groups)
    finally:
        di.wall_mass = original
    assert abs(out["floor"] - 1.0) < 0.03, out["floor"]
    assert abs(out["gate_pair"] - 1.0) < 0.05, out["gate_pair"]
    # the bottom rows are no longer inverted and carry their own weight
    p0 = float(out["p"][pos[:, 1] < cfg.dx].mean())
    p1 = float(out["p"][(pos[:, 1] >= cfg.dx) & (pos[:, 1] < 2 * cfg.dx)].mean())
    assert p0 > p1


def test_density_renormalization_is_not_the_same_mechanism():
    """Shepard normalization removes support truncation, not the density step."""
    cfg, pos, mass, wall, groups = _setup()
    plain = _loads(cfg, pos, mass, wall, groups)
    renorm = _loads(*_setup(density_renormalization=True))
    assert abs(renorm["floor"] - plain["floor"]) < 0.02, (plain["floor"],
                                                          renorm["floor"])
    fixed = _loads(*_setup(boundary_consistent_density=True))
    assert fixed["floor"] - plain["floor"] > 0.15


def test_conservative_transient_keeps_mass_and_budget_with_the_switch_on():
    base = dict(dx=0.025, steps=120, pool_width=1.0, tank_height=0.5,
                viscosity=0.0)
    summary = run_gate_case(
        GateCaseConfig(boundary_consistent_density=True, **base),
        record_every=120)["summary"]
    assert abs(summary["mass_drift_relative"]) < 1.0e-12
    assert summary["residual_relative"] < 1.0e-3, summary["residual_relative"]
    assert summary["boundary_clamp_events"] == 0