"""Does the Gresho anchor's ``L_z`` gate measure the scheme or the box?

The anchor used to gate ``angular_momentum_drift_rel`` at round-off on the
argument that "a pairwise central-force scheme must conserve ``L_z``".  That
argument is true for a closed domain and **false for a periodic one**: the
pressure term is central with respect to the *minimum-image* separation, so a
pair that wraps the boundary contributes ``-n L x F`` to the torque about a
fixed point and cannot cancel.  There is no continuous rotational symmetry of a
periodic lattice, hence no rotational Noether charge.

These tests pin the corrected reading (mission m21457):

* the in-box part of the torque is round-off (the real central-force property);
* all of the drift is carried by wrapping pairs;
* the drift is *reproduced* by the discrete torque integral (ledger closure);
* the ``L_z`` measurement itself is clean (zero-force control);
* ``accept`` gates the ledger and the in-box torque, not the drift size.
"""
from __future__ import annotations

import torch

from tensorsph.gresho import (GATES, GreshoConfig, accept, analytic_reference,
                             angular_momentum, make_state, pair_torque_split,
                             run_gresho)
from tensorsph.vv.tgv import periodic_pairs


def _two_particle_state(wrap: bool):
    cfg = GreshoConfig(dx=0.02)
    x0, y0 = 0.49, 0.49
    pos = torch.tensor([[x0, y0], [x0 + 0.02, y0 + 0.02]], dtype=torch.float64)
    if wrap:                       # put the pair across the periodic boundary
        pos[0, 0] -= 0.5
        pos[1, 0] -= 0.48          # second particle wraps through x = 1 -> 0
        pos = torch.remainder(pos, cfg.L)
    mass = torch.tensor([2.5, 2.5], dtype=torch.float64)
    rho = torch.tensor([1000.0, 1000.0], dtype=torch.float64)
    pressure = torch.tensor([500.0, 500.0], dtype=torch.float64)
    pairs = periodic_pairs(pos, 2.0 * cfg.h(), cfg.L)
    return cfg, pos, mass, rho, pressure, pairs


def test_two_particle_pair_in_the_box_has_no_torque():
    """Control: a pair that does not wrap contributes exactly zero torque."""
    cfg, pos, mass, rho, pressure, pairs = _two_particle_state(wrap=False)
    out = pair_torque_split(pos, pos.clone(), mass, cfg, rho, pressure, pairs=pairs)
    assert out["pairs"] == 1 and out["wrap_pairs"] == 0
    assert abs(out["tau_total"]) < 1e-12
    assert abs(out["tau_in_box"]) < 1e-12


def test_two_particle_pair_across_the_boundary_carries_a_lattice_torque():
    """The same pair, wrapped: the lever arm gains one lattice vector ``L e_x``.

    ``tau = (lever_true - lever_minimum_image) x F = L e_x x F``, so the torque
    is geometry -- a nonzero value here is *expected*, not a scheme defect.
    """
    cfg, pos, mass, rho, pressure, pairs = _two_particle_state(wrap=True)
    out = pair_torque_split(pos, pos.clone(), mass, cfg, rho, pressure, pairs=pairs)
    assert out["pairs"] == 1 and out["wrap_pairs"] == 1
    assert abs(out["tau_in_box"]) < 1e-12          # nothing in-box: still exact
    assert abs(out["tau_wrap"]) > 1e-6             # the lattice term is real
    # sign/magnitude: L * F_y with the lever difference exactly +L in x
    assert abs(abs(out["tau_wrap"]) / (cfg.L * abs(out["tau_total"]) + 1e-30) - 1.0) < 1e-9


def test_ballistic_measurement_conserves_lz():
    """Zero-force control: the unwrapped-coordinate ``L_z`` must not drift.

    ``a = 0`` is a straight line, so any drift here would be a bookkeeping bug in
    the measurement (this is what makes the drift below evidence about forces).
    """
    cfg = GreshoConfig(dx=0.05, t_end=0.05)
    table = analytic_reference(cfg)
    state = make_state(cfg, table=table)
    pos, vel, mass = state["pos"], state["vel"], state["mass"]
    pos_abs = pos.clone()
    dt = cfg.dt()
    lz0 = angular_momentum(pos_abs, vel, mass, cfg)
    for _ in range(200):
        pos_abs = pos_abs + dt * vel
        pos = torch.remainder(pos + dt * vel, cfg.L)
    drift = abs(angular_momentum(pos_abs, vel, mass, cfg) - lz0) / abs(lz0)
    assert drift < 1e-13, drift


def test_in_box_torque_is_round_off_and_the_ledger_closes():
    cfg = GreshoConfig(dx=0.05, t_end=0.08, sample_every=10)
    run = run_gresho(cfg)
    m = run["measured"]
    assert m["torque_pairs"] > 0 and m["torque_wrap_pairs_max"] > 0
    # (a) the force is central: the in-box torque is round-off ...
    assert m["torque_in_box_rel_max"] < 1e-12, m["torque_in_box_rel_max"]
    # (b) ... all of the drift is carried by the wrapping pairs ...
    assert m["torque_wrap_rel_max"] > 1e3 * m["torque_in_box_rel_max"]
    assert m["angular_momentum_drift_rel"] > 0.0
    # (c) ... and the drift equals the time integral of that torque
    assert m["lz_ledger_closure_pct"] <= GATES["lz_ledger_closure_pct"], m


def test_accept_gates_the_ledger_not_the_drift_size():
    """A large periodic drift with a closed ledger must not be a FAIL by itself."""
    row = {"dx": 0.02, "particles": 2500, "steps": 1616,
           "spurious_acceleration_pct": 3.67, "velocity_error_pct_end": 5.0,
           "angular_momentum_drift_rel": 0.0415,          # 4 % drift, explained
           "torque_in_box_rel_max": 1e-15,
           "lz_ledger_closure_pct": 0.01,
           "observed_order_acceleration": 1.86,
           "init_density_rel_error_max": 3e-7,
           "spurious_acceleration_max_pct": 19.0, "velocity_error_pct_max": 5.0,
           "energy_ratio": 0.92, "equivalent_viscosity": 7e-4,
           "equivalent_reynolds": 552.0, "linear_momentum_drift_rel": 6e-17,
           "peak_speed_ratio": 1.0, "density_rms_max": 6e-3,
           "observed_order_velocity": 0.43, "observed_order_profile": 0.19}
    controls = {"forward": {"spurious_acceleration_pct": 3.67},
                "reverse": {"spurious_acceleration_pct": 99.0},
                "half_pressure": {"spurious_acceleration_pct": 53.0}}
    verdict = accept([row], controls=controls)
    torque_checks = [c for c in verdict["checks"] if "torque" in c["name"]
                     or "ledger" in c["name"]]
    assert torque_checks and all(c["pass"] for c in torque_checks)
    assert all(c["gate"] for c in torque_checks)
    diagnostic = [c for c in verdict["checks"] if "diagnostic" in c["name"]]
    assert diagnostic and diagnostic[0]["gate"] is False
    assert verdict["verdict"] == "PASS"

    # an open ledger (drift the torque integral cannot explain) is a FAIL: that
    # is what a bookkeeping bug looks like
    bad = dict(row, lz_ledger_closure_pct=GATES["lz_ledger_closure_pct"] * 4.0)
    assert accept([bad], controls=controls)["verdict"] == "FAIL"

    # a non-central force is a FAIL too
    non_central = dict(row, torque_in_box_rel_max=1e-3)
    assert accept([non_central], controls=controls)["verdict"] == "FAIL"