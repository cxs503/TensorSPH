"""M3 -- mass-conservation gate on a free-surface simulation.

SPH carries a *constant* particle mass, so a naive "total mass at the end equals
total mass at the start" check is satisfied by construction and measures nothing.
What can actually fail is the *bookkeeping at the boundary*: a particle that
leaves the domain must either be returned (closed tank) or removed **and its
mass added to an outflow accumulator**.  A solver that deletes it and forgets it
loses mass silently, and that is what this gate is built to catch.

Two runs are taken on the same trajectory:

``accounted``
    ``M(t) = sum_i m_i(inside) + M_outflow(t)``.  Must stay within 0.1 % of
    ``M(0)`` for the whole run (the campaign's 1000-step horizon is reported
    separately as well).
``unaccounted``
    the bookkeeping a leaky implementation would do: ``M(t) = sum_i m_i(inside)``
    only.  This is the negative control, and it must show a loss well above 1 %
    once water has left through the open boundary -- if it does not, the test is
    not reaching the boundary and says nothing.

The open boundary is a pure bookkeeping device at ``x = open_x``; no wall
particle is placed there, so the flow is unaffected by it beyond the deletion of
the particles that cross it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dam_break import DamBreakConfig, make_column, validate_case
from ..droplet_impact import (PairListCache, resolve_device, rhs, wall_lattice)


@dataclass
class ConservationCase:
    """Dam break in a tank whose downstream end may be open."""

    dx: float = 0.02
    column_width: float = 0.2
    column_height: float = 0.4
    tank_width: float = 1.0
    tank_height: float = 0.6
    sound_speed: float = 25.0
    viscosity: float = 1.0e-6
    seconds: float = 0.3
    open_x: float | None = 0.40      # None keeps the tank closed
    horizon_steps: int = 1000
    sample_every: int = 20

    def solver_config(self) -> DamBreakConfig:
        cfg = DamBreakConfig(dx=self.dx, column_width=self.column_width,
                             column_height=self.column_height,
                             pool_width=self.tank_width,
                             tank_height=self.tank_height,
                             pool_depth=self.tank_height,
                             sound_speed=self.sound_speed,
                             viscosity=self.viscosity)
        cfg.dt = 0.3 * cfg.h_factor * self.dx / self.sound_speed
        cfg.steps = int(round(self.seconds / cfg.dt))
        cfg.output_every = max(1, int(round(self.sample_every))
                               if self.sample_every >= 1 else 1)
        return cfg


def run(case: ConservationCase) -> dict:
    cfg = case.solver_config()
    validate_case(cfg)
    device = resolve_device("cpu")
    top = cfg.tank_top()
    pos, mass = make_column(cfg, device)
    wall = wall_lattice(cfg, device, top=top)
    vel = torch.zeros_like(pos)
    h = cfg.h_factor * cfg.dx
    n_fluid = pos.shape[0]
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=n_fluid)
    all_pos = torch.cat((pos, wall), dim=0)
    dt = cfg.dt

    mass_total = float(mass.sum())
    outflow = 0.0
    removed = 0
    history = []
    mass_at_horizon = None
    for step in range(cfg.steps + 1):
        if case.open_x is not None:
            escape = pos[:, 0] > case.open_x
            if bool(escape.any()):
                outflow += float(mass[escape].sum())
                removed += int(escape.sum())
                keep = ~escape
                pos, vel, mass = pos[keep], vel[keep], mass[keep]
                n_fluid = pos.shape[0]
                all_pos = torch.cat((pos, wall), dim=0)
                cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=n_fluid)
        if n_fluid == 0:
            break
        all_pos[:n_fluid] = pos
        pairs, _ = cache.get(all_pos)
        _rho, _p, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        inside = float(mass.sum())
        if step % case.sample_every == 0 or step == cfg.steps:
            history.append({"step": step, "time": step * dt,
                            "mass_inside": inside, "mass_outflow": outflow,
                            "mass_accounted": inside + outflow,
                            "particles_inside": int(n_fluid),
                            "particles_removed": removed})
        if step == case.horizon_steps:
            mass_at_horizon = inside + outflow
        vel = vel + dt * acc
        pos = pos + dt * vel
        if case.open_x is None:
            # closed tank: the walls put the particles back
            from ..droplet_impact import apply_boundary
            apply_boundary(pos, vel, cfg, top=top)
        step += 1

    final = history[-1]
    drift_accounted = abs(final["mass_accounted"] - mass_total) / mass_total
    drift_unaccounted = abs(final["mass_inside"] - mass_total) / mass_total
    horizon_drift = (abs(mass_at_horizon - mass_total) / mass_total
                     if mass_at_horizon is not None else None)
    return {
        "case": "mass_conservation_dam_break",
        "config": {
            "dx": case.dx, "dt": dt, "steps": cfg.steps,
            "seconds": case.seconds, "open_x": case.open_x,
            "tank_width": case.tank_width, "horizon_steps": case.horizon_steps,
        },
        "measured": {
            "mass_initial": mass_total,
            "mass_final_inside": final["mass_inside"],
            "mass_outflow": outflow,
            "mass_final_accounted": final["mass_accounted"],
            "particles_removed": removed,
            "particles_inside_end": final["particles_inside"],
            "relative_loss_accounted": drift_accounted,
            "relative_loss_unaccounted": drift_unaccounted,
            "relative_loss_accounted_at_horizon": horizon_drift,
            "steps_run": cfg.steps,
        },
        "history": history,
    }


def gate(closed: dict, limit: float = 1.0e-3, min_steps: int = 1000) -> dict:
    """J3 first arm: the closed tank must hold its mass for at least 1000 steps."""
    m = closed["measured"]
    history_loss = max((abs(r["mass_accounted"] - m["mass_initial"]) / m["mass_initial"]
                        for r in closed["history"]), default=float("nan"))
    enough_steps = m["steps_run"] >= min_steps
    passed = bool(history_loss <= limit and enough_steps)
    return {"quantity": "max relative mass loss over the run (accounted)",
            "measured": history_loss,
            "limit": limit,
            "steps_run": m["steps_run"],
            "min_steps": min_steps,
            "steps_ok": enough_steps,
            "pass": passed}


def gate_overflow(open_case: dict, min_loss: float = 0.01,
                  limit: float = 1.0e-3) -> dict:
    """J3 second arm: bookkeeping keeps the open case exact, and forgetting it loses >1 %."""
    m = open_case["measured"]
    accounted = m["relative_loss_accounted"]
    unaccounted = m["relative_loss_unaccounted"]
    passed = bool(accounted <= limit and unaccounted >= min_loss
                  and m["particles_removed"] > 0)
    return {"quantity": "accounted loss <= limit AND unaccounted loss >= 1 %",
            "accounted_loss": accounted,
            "unaccounted_loss": unaccounted,
            "particles_removed": m["particles_removed"],
            "limit": limit,
            "min_loss": min_loss,
            "pass": passed}
