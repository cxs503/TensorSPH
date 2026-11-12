"""Rung 1 of the verification ladder: hydrostatic pressure balance.

A rectangular tank of still water, released from rest under gravity, must settle
onto the hydrostatic profile

.. math::  p(y) = \\rho_0 g (H - y)

where ``H`` is the free-surface height and ``y`` the height above the floor.
The rung compares the *time-averaged* pressure of the interior bands with that
profile. Time averaging is not cosmetic: a lattice released at rest is not in a
discrete equilibrium, so the pool sloshes acoustically and the instantaneous
profile oscillates by tens of percent (the library's own B1 benchmark reports
per-band fluctuations of 6-15 %). The time mean removes the oscillation; the
fluctuation is reported next to it as a diagnostic, not as a gate.

What the rung gates
-------------------
``max_band_error_pct``: the largest relative error of any *interior* band mean.
The topmost band is excluded because its density is truncated by the free
surface, where SPH summation density is not comparable with a hydrostatic
reference (that band is reported separately as ``surface_band``).

Negative control: with ``gravity = 0`` the analytic profile collapses to zero
pressure, so a correct implementation must show ~100 % error against the
``rho g h`` reference -- i.e. the rung really measures the gravity/pressure
balance and is not satisfied by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..droplet_impact import (Config, PairListCache, apply_boundary,
                              cell_centres, hydrostatic_density,
                              lattice_mass_unit, resolve_device, rhs,
                              wall_lattice)


@dataclass
class HydrostaticConfig:
    """Square tank of still water (independent of any benchmark case)."""

    dx: float = 0.02
    tank_width: float = 0.4
    depth: float = 0.4
    seconds: float = 2.0
    sample_every: int = 20
    bands: int = 8
    average_from: float = 0.5      # fraction of the run used for the time mean
    hydrostatic_init: bool = True
    gravity: float = 9.81
    gravity_reference: float = 9.81   # keeps the rho g h reference fixed when
                                      # gravity is switched off in the solver
    sound_speed: float = 20.0
    device: str = "cpu"
    symmetry_floor: float = 1.0e-12

    def solver_config(self) -> Config:
        return Config(
            dx=self.dx,
            pool_width=self.tank_width,
            pool_depth=self.depth,
            gravity=self.gravity,
            sound_speed=self.sound_speed,
            boundary="wall",
            device=self.device,
            hydrostatic_init=self.hydrostatic_init,
            dt=self.dt(),
            steps=int(math.ceil(self.seconds / self.dt())),
            output_every=self.sample_every,
        )

    def dt(self) -> float:
        return 0.3 * 1.3 * self.dx / self.sound_speed


def make_pool(cfg: HydrostaticConfig):
    """Lattice of the pool with either uniform or hydrostatic masses."""
    device = resolve_device(cfg.device)
    xs = cell_centres(cfg.tank_width, cfg.dx, device)
    ys = cell_centres(cfg.depth, cfg.dx, device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    solver = cfg.solver_config()
    unit = lattice_mass_unit(solver)
    mass = torch.full((pos.shape[0],), unit, device=device)
    if cfg.hydrostatic_init:
        mass = hydrostatic_density(pos[:, 1], solver) * (unit / solver.rho0)
    return pos, mass, solver


def sample_pressure(pos, rho, cfg: HydrostaticConfig, solver: Config) -> torch.Tensor:
    from ..droplet_impact import tait_pressure
    return tait_pressure(rho, solver)


def band_table(pos, pressure, weight, cfg: HydrostaticConfig) -> list[dict]:
    """Banded pressure means against ``rho g (H - y)`` at the band centre."""
    edges = [cfg.depth * k / cfg.bands for k in range(cfg.bands + 1)]
    # Kernel-support band: within 2h of the free surface the summation density
    # is truncated by the surface, so the SPH pressure is not a hydrostatic
    # pressure there. Those bands are reported but excluded from the gate.
    support = 2.0 * 1.3 * cfg.dx
    rows = []
    for k, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (pos[:, 1] >= low) & (pos[:, 1] < high)
        count = int(mask.sum())
        if count == 0:
            continue
        y = float((pos[mask, 1] * weight[mask]).sum() / weight[mask].sum())
        p = float((pressure[mask] * weight[mask]).sum() / weight[mask].sum())
        p_ref = 1000.0 * cfg.gravity_reference * (cfg.depth - y)
        rows.append({
            "band": k,
            "y": y,
            "y_low": low,
            "y_high": high,
            "particles": count,
            "pressure": p,
            "pressure_analytic": p_ref,
            "rel_error_pct": ((p - p_ref) / p_ref * 100.0) if p_ref > 0 else float("nan"),
            "free_surface": (k == cfg.bands - 1) or (y > cfg.depth - support),
        })
    return rows


def run_hydrostatic(cfg: HydrostaticConfig, gravity_override: float | None = None,
                    seconds_override: float | None = None,
                    return_fields: bool = False) -> dict:
    if gravity_override is not None:
        # The analytic reference keeps the nominal g: switching gravity off in
        # the solver must *fail* the rung, not move its target.
        cfg = HydrostaticConfig(**{**cfg.__dict__, "gravity": gravity_override,
                                   "gravity_reference": cfg.gravity})
    if seconds_override is not None:
        cfg = HydrostaticConfig(**{**cfg.__dict__, "seconds": seconds_override})
    pos, mass, solver = make_pool(cfg)
    device = pos.device
    wall = wall_lattice(solver, device, top=cfg.depth)
    vel = torch.zeros_like(pos)
    h = solver.h_factor * cfg.dx
    fluid_count = pos.shape[0]
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=fluid_count)
    all_pos = torch.cat((pos, wall), dim=0)

    dt = cfg.dt()
    steps = int(math.ceil(cfg.seconds / dt))
    dt = cfg.seconds / steps
    accum_p = torch.zeros(fluid_count, dtype=pos.dtype, device=device)
    accum_p2 = torch.zeros_like(accum_p)
    accum_x = torch.zeros((fluid_count, pos.shape[1]), dtype=pos.dtype, device=device)
    samples = 0
    times: list[float] = []
    band_history: list[list[dict]] = []
    t = 0.0
    for step in range(steps):
        all_pos[:fluid_count] = pos
        pairs, _ = cache.get(all_pos)
        rho, pressure, acc = rhs(pos, vel, mass, solver, pairs=pairs, wall=wall, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        if t >= cfg.average_from * cfg.seconds and step % cfg.sample_every == 0:
            accum_p += pressure
            accum_p2 += pressure.square()
            accum_x += pos
            samples += 1
            times.append(t)
            band_history.append(band_table(pos, pressure, mass, cfg))
        vel = vel + dt * acc
        pos = pos + dt * vel
        apply_boundary(pos, vel, solver, top=cfg.depth)
        t += dt

    if samples == 0:
        raise RuntimeError("no pressure samples were collected")
    mean_p = accum_p / samples
    var_p = (accum_p2 / samples - mean_p.square()).clamp_min(0.0)
    std_p = var_p.sqrt()
    rows = band_table(pos, mean_p, mass, cfg)
    for row in rows:
        mask = (pos[:, 1] >= row["y_low"]) & (pos[:, 1] < row["y_high"])
        fluct = float((std_p[mask] / mean_p[mask].clamp_min(1.0)).max() * 100.0)
        row["fluctuation_pct"] = fluct

    interior = [r for r in rows if not r["free_surface"] and r["pressure_analytic"] > 0]
    # Instantaneous extreme over the whole averaging window, per interior band.
    instantaneous = [
        max((abs(band["rel_error_pct"]) for snapshot in band_history
             for band in snapshot
             if band["band"] == r["band"] and not band["free_surface"]),
            default=float("nan"))
        for r in interior]
    errors = [abs(r["rel_error_pct"]) for r in interior]
    max_err = max(errors) if errors else float("nan")
    mean_err = sum(errors) / len(errors) if errors else float("nan")
    surface = [r for r in rows if r["free_surface"]]

    result = {
        "case": "hydrostatic_square_tank",
        "config": {
            "dx": cfg.dx, "tank_width": cfg.tank_width, "depth": cfg.depth,
            "seconds": cfg.seconds, "dt": dt, "steps": steps,
            "particles": int(fluid_count), "bands": cfg.bands,
            "hydrostatic_init": cfg.hydrostatic_init, "gravity": cfg.gravity,
            "gravity_reference": cfg.gravity_reference,
            "sound_speed": cfg.sound_speed, "samples": samples,
            "average_from": cfg.average_from,
            "free_surface_height": cfg.depth,
        },
        "reference": {
            "kind": "analytic",
            "solution": "p(y) = rho0 g (H - y)",
            "rho0": 1000.0, "g": cfg.gravity_reference, "H": cfg.depth,
        },
        "measured": {
            "bands": rows,
            "max_band_error_pct": max_err,
            "mean_band_error_pct": mean_err,
            "surface_band_error_pct": (surface[0]["rel_error_pct"] if surface
                                       else float("nan")),
            "max_band_error_pct_instantaneous": max(instantaneous, default=float("nan")),
            "gated_bands": [r["band"] for r in interior],
            "support_band_width": 2.0 * 1.3 * cfg.dx,
        },
        "error_pct": max_err,
        "series": band_history,
        "gate": {"quantity": "max interior band pressure error vs rho g h",
                 "limit_pct": 1.0, "pass": bool(max_err < 1.0)},
    }
    if return_fields:
        # Time-averaged *fields* of the sampling window.  A band mean of the
        # pressure paired with the instantaneous (settling) positions is two
        # different times, so the consistent reduction needs the mean position
        # as well; the tensors stay out of the JSON-able part of the record
        # unless asked for.
        result["fields"] = {
            "mean_pos": accum_x / samples,
            "mean_pressure": mean_p,
            "mean_rms": std_p,
            "mass": mass,
            "samples": samples,
        }
    return result
