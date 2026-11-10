"""Ritter dry-bed dam break: WCSPH versus the exact shallow-water solution.

Why this case
-------------
The Ritter (1892) solution is the only *exact, closed-form, unsteady* free-surface
solution that a dam break can be compared against, and it is the standard first
quantitative test of a free-surface SPH solver. It is also the case with the
sharpest published answer to "does this implementation lose the front?", because
the front speed ``2 sqrt(g h0)`` is exact and independent of resolution.

The anchor
----------
For a reservoir of still water of depth ``h0`` occupying ``x <= x_dam`` released
onto a dry, frictionless bed at ``t = 0``, with ``c0 = sqrt(g h0)``:

    x_front(t) = x_dam + 2 c0 t
    h(x, t)    = h0                        x <= x_tail = x_dam - c0 t
               = (2 c0 - xi)^2 / (9 g)     x_tail < x < x_front
    u(x, t)    = 0                         x <= x_tail
               = (2/3)(c0 + xi)            x_tail < x < x_front
    xi         = (x - x_dam) / t

(``tensorsph.ritter``, which is itself verified against the three exact
properties of the fan: ``h(x_dam) = 4 h0/9``, released volume ``h0 c0 t`` and the
Riemann invariant ``u + 2 sqrt(g h) = 2 c0``.)

Scope and honesty
-----------------
The solution is a **shallow-water** solution, and it only exists while the
rarefaction has not reached the back wall of a *finite* reservoir, i.e.
``t < a / c0`` for a reservoir of width ``a`` (:func:`tensorsph.ritter.validity_window`).
This module therefore runs a *wide and shallow* reservoir (``h0 / a = 0.2``, as
opposed to the aspect ratio ~2 of the classic Martin & Moyce column used by
``tensorsph.dam_break``) and stops well inside the validity window. Both
restrictions are reported with every measurement rather than assumed:

* the time is always quoted as ``t / (a / c0)``;
* the shallow-water parameter ``h0 / a`` is quoted, and the depth/velocity
  comparison is restricted to the rarefaction fan, which is where the exact
  solution carries information (the reservoir region is still at rest and would
  only dilute the norms).

What is measured
----------------
``front_error``      ``|x_front - (x_dam + 2 c0 t)|`` relative to the *travel*
                     of the front, ``2 c0 t`` (the only scale that makes the
                     number resolution-independent);
``depth_l2``         relative L2 of the measured depth against the exact depth
                     over the fan;
``velocity_l2``      the same for the horizontal velocity;
``released_volume``  measured water volume downstream of the dam against the
                     exact ``h0 c0 t`` -- an independent mass/volume statement
                     (the solver conserves particle mass exactly by construction,
                     so that quantity on its own would be a tautology).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .. import dam_break, droplet_impact, ritter
from . import profiles

__all__ = ["RitterCase", "RitterResult", "run_ritter", "compare_snapshot"]

GRAVITY = 9.81


@dataclass
class RitterCase:
    """Geometry and numerics of one semi-infinite dry-bed run.

    Defaults are the "base" resolution of the convergence ladder: a reservoir
    ``a = 5 h0`` wide, a tank 20 ``h0`` long so the surge never reaches the far
    wall inside the validity window, and 10 particles through the depth.
    """

    depth0: float = 0.10
    column_width: float = 0.50
    tank_length: float = 2.00
    tank_height: float = 0.30
    cells_per_depth: float = 10.0
    sound_speed_ratio: float = 20.0     # c_s / sqrt(g h0)
    viscosity: float = 5.0e-3           # numerical (Monaghan) viscosity, m^2/s
    density_diffusion: float = 0.0
    wall_friction: float = 0.0
    time_fraction: float = 0.60         # of the validity window a / c0
    sample_fractions: tuple = (0.35, 0.50, 0.60)   # of the validity window
    device: str = "cpu"
    gravity: float = GRAVITY
    label: str = ""

    # -- derived -----------------------------------------------------------
    @property
    def dx(self) -> float:
        return float(self.depth0 / self.cells_per_depth)

    @property
    def wave_speed(self) -> float:
        return ritter.wave_speed(self.depth0, self.gravity)

    @property
    def sound_speed(self) -> float:
        return float(self.sound_speed_ratio * self.wave_speed)

    @property
    def validity_window(self) -> float:
        """``a / c0``: last time the Ritter solution is defined."""
        return float(self.column_width / self.wave_speed)

    @property
    def shallow_ratio(self) -> float:
        """``h0 / a``: how far the case is from the shallow-water limit."""
        return float(self.depth0 / self.column_width)

    @property
    def end_time(self) -> float:
        return float(self.time_fraction * self.validity_window)

    @property
    def snapshot_times(self) -> list[float]:
        return [float(f * self.validity_window) for f in self.sample_fractions]

    def dam_break_config(self) -> dam_break.DamBreakConfig:
        """The dam-break configuration equivalent to this case."""
        dx = self.dx
        cfg = dam_break.DamBreakConfig(
            dx=dx,
            tank_height=float(self.tank_height),
            column_width=float(self.column_width),
            column_height=float(self.depth0),
            pool_width=float(self.tank_length),
            pool_depth=float(self.tank_height),
            sound_speed=float(self.sound_speed),
            viscosity=float(self.viscosity),
            density_diffusion=float(self.density_diffusion),
            wall_friction=float(self.wall_friction),
            gravity=float(self.gravity),
            device=self.device,
            hydrostatic_init=True,
            adaptive_dt=False,
            output_every=max(1, int(round(self.end_time / (20.0 * self.dt)))),
        )
        cfg.dt = self.dt
        cfg.steps = int(math.ceil(self.end_time / cfg.dt))
        return cfg

    @property
    def dt(self) -> float:
        """Acoustic-Courant limited step at 0.25 of ``h / c_s``."""
        h = 1.3 * self.dx            # droplet_impact default h_factor
        return float(0.25 * h / self.sound_speed)


@dataclass
class RitterResult:
    label: str
    case: RitterCase
    times: list = field(default_factory=list)
    front: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    gravity: float = GRAVITY
    mass_drift: float = 0.0
    initial_mass: float = 0.0
    escaped: int = 0
    steps_taken: int = 0
    simulated_time: float = 0.0
    max_density_compression: float = 0.0
    max_acoustic_courant: float = 0.0
    wall_clock: float = 0.0
    particle_count: int = 0

    def front_series_errors(self, start: float = 0.2) -> list[dict]:
        """Front error against time, restricted to the developed flow.

        The first ``start`` fraction of the validity window is skipped: before
        the rarefaction has developed the "front" is the initial column face and
        the relative error is meaningless.
        """
        window = self.case.validity_window
        rows = []
        for t, x in zip(self.times, self.front):
            if t < start * window:
                continue
            exact = analytic_front_at_depth(t, self.case.dx, self.case)
            travel = 2.0 * self.case.wave_speed * t
            rows.append({
                "time": float(t),
                "time_fraction": float(t / window),
                "front_sim": float(x),
                "front_exact": float(exact),
                "front_error": float(abs(x - exact) / travel),
            })
        return rows

    def summary(self) -> dict:
        """Everything a gate or a report table needs, as plain numbers."""
        rows = self.rows
        front_rows = self.front_series_errors()
        return {
            "label": self.label,
            "depth0": self.case.depth0,
            "column_width": self.case.column_width,
            "shallow_ratio": self.case.shallow_ratio,
            "validity_window": self.case.validity_window,
            "dx": self.case.dx,
            "cells_per_depth": self.case.cells_per_depth,
            "sound_speed": self.case.sound_speed,
            "dt": self.case.dt,
            "steps": self.steps_taken,
            "particles": self.particle_count,
            "wall_clock": self.wall_clock,
            "max_density_compression": self.max_density_compression,
            "max_acoustic_courant": self.max_acoustic_courant,
            "escaped": self.escaped,
            "mass_drift": self.mass_drift,
            "mass_drift_relative": (abs(self.mass_drift) / self.initial_mass
                                    if self.initial_mass > 0 else float("nan")),
            "particles_escaped": self.escaped,
            "front_error_max": max((r["front_error"] for r in front_rows), default=float("nan")),
            "front_error_mean": (sum(r["front_error"] for r in front_rows) / len(front_rows)
                                 if front_rows else float("nan")),
            "front_rows": front_rows,
            "snapshot_rows": rows,
            "depth_l2": max((r["depth_l2"] for r in rows), default=float("nan")),
            "depth_l2_resolved": max((r["depth_l2_resolved"] for r in rows),
                                     default=float("nan")),
            "front_error_zero_depth": max((r.get("front_error_zero_depth",
                                                  float("nan")) for r in rows),
                                          default=float("nan")),
            "velocity_l2": max((r["velocity_l2"] for r in rows), default=float("nan")),
            "released_volume_error": max((r["released_volume_error"] for r in rows),
                                         default=float("nan")),
        }


def analytic_front_at_depth(time: float, depth_threshold: float,
                           case: "RitterCase") -> float:
    """Where the exact depth profile falls to ``depth_threshold``.

    The zero-depth front ``x_dam + 2 c0 t`` is not a fair reference for a
    particle method: the exact depth *inside* the last few percent of the front
    travel is far below one particle spacing (at ``h0/dx = 10`` the outer ~48 %
    of the travel carries ``h < dx``), so no lattice can represent it. The
    resolution-matched reference is the position where the exact depth equals
    the smallest depth the discretisation can hold, ``dx``:

        h = (2 c0 - xi)^2 / (9 g) = dx  =>  xi = 2 c0 - 3 sqrt(g dx).

    Both references are reported; the criterion uses this one.
    """
    c0 = case.wave_speed
    xi = 2.0 * c0 - 3.0 * math.sqrt(case.gravity * float(depth_threshold))
    if xi <= 0.0:
        raise ValueError("threshold too large for the resolution")
    return float(case.column_width + time * xi)


def compare_snapshot(case: RitterCase, time: float, positions: torch.Tensor,
                     velocity: torch.Tensor, mass: torch.Tensor,
                     rho0: float = 1000.0) -> dict:
    """Measure a particle snapshot against the exact Ritter solution.

    Returns the scalar errors plus the sampled profiles, so a figure and a table
    are built from exactly the same arrays.
    """
    dx = case.dx
    slab = 2.0 * dx
    dam = case.column_width
    c0 = case.wave_speed
    x_front = dam + 2.0 * c0 * time
    x_tail = dam - c0 * time
    centres, depth = profiles.binned_depth(
        positions, mass, rho0, slab, x_min=0.0, x_max=case.tank_length)
    fan = (centres >= x_tail) & (centres <= x_front + slab)
    x_exact = centres[fan]
    depth_exact_all = ritter.depth(centres, time, case.depth0, dam_x=dam,
                                  gravity=case.gravity)
    depth_exact = depth_exact_all[fan]
    depth_error = profiles.relative_l2(depth[fan], depth_exact, scale=case.depth0)
    # Second norm, restricted to the part of the fan the lattice can hold
    # (exact depth >= two particle spacings). The full-fan norm is dominated by
    # the outer film -- the exact depth there is below one spacing over the last
    # ~48 % of the front travel at h0/dx = 10 -- which no particle method can
    # represent, so the two norms answer different questions and both are
    # reported: "how wrong is the answer where it can be measured" versus "how
    # wrong is the whole field including the part that is under-resolved".
    resolvable = fan & (depth_exact_all >= 2.0 * dx)
    depth_error_resolved = (profiles.relative_l2(depth[resolvable],
                                                depth_exact_all[resolvable],
                                                scale=case.depth0)
                            if int(resolvable.sum()) >= 4 else float("nan"))

    # velocity: one-sided (mass weighted) average per strip
    _, u_sim = profiles.binned_velocity(
        positions, velocity, mass, slab, x_min=0.0, x_max=case.tank_length)
    u_sim = u_sim[fan]
    xi = (x_exact - dam) / time
    u_exact = (2.0 / 3.0) * (c0 + xi)
    valid = torch.isfinite(u_sim)
    if bool(valid.any()):
        velocity_error = profiles.relative_l2(
            u_sim[valid], u_exact[valid], scale=float(torch.abs(u_exact[valid]).max()))
    else:
        velocity_error = float("nan")

    # Water downstream of the dam. Integrating the exact fan gives
    #   V(t) = int_0^{2 c0} (2 c0 - xi)^2 / (9 g) t dxi = 8 h0 c0 t / 27,
    # which is the volume that has actually crossed the dam section. (The
    # h0 c0 t quoted in ``ritter``'s docstring is the volume of the *whole fan*,
    # which straddles the dam; mixing the two up would report a 70 % error for a
    # perfect solution.)
    downstream = positions[:, 0] > dam
    released = float(mass[downstream].sum() / rho0)
    # ritter.released_volume is the volume of the *whole* fan (tail to front);
    # integrating only the downstream part, x in [dam, front], leaves 8/27 of it.
    released_exact = (8.0 / 27.0) * ritter.released_volume(time, case.depth0,
                                                            case.gravity)
    released_error = abs(released - released_exact) / released_exact

    # Resolution-matched front: the last strip still holding ~one particle
    # spacing of water, against the exact position of that same depth.
    front = profiles.front_from_profile(centres, depth, dx, slab)
    front_exact_resolved = analytic_front_at_depth(time, dx, case)
    return {
        "time": float(time),
        "time_fraction": float(time / case.validity_window),
        "front_sim": float(front),
        "front_exact": float(front_exact_resolved),
        "front_error": float(abs(front - front_exact_resolved) / (2.0 * c0 * time)),
        "front_zero_depth_exact": float(x_front),
        "front_error_zero_depth": float(abs(front - x_front) / (2.0 * c0 * time)),
        "front_error_max_particle": float(
            abs(float(positions[:, 0].max()) - x_front) / (2.0 * c0 * time)),
        "front_threshold": float(dx),
        "depth_l2": float(depth_error),
        "depth_l2_resolved": float(depth_error_resolved),
        "velocity_l2": float(velocity_error),
        "released_volume": released,
        "released_volume_exact": float(released_exact),
        "released_volume_error": float(released_error),
        "centres": centres.tolist(),
        "depth": depth.tolist(),
        "depth_exact": ritter.depth(centres, time, case.depth0, dam_x=dam,
                                    gravity=case.gravity).tolist(),
        "u_sim": u_sim.tolist(),
        "u_exact": [(2.0 / 3.0) * (c0 + float((x - dam) / time))
                    for x in centres.tolist()],
    }


def run_ritter(case: RitterCase, snapshot_times=None) -> RitterResult:
    """Integrate the case and return the measured trajectory and profiles."""
    import time as _time

    cfg = case.dam_break_config()
    times_requested = (case.snapshot_times if snapshot_times is None
                       else list(snapshot_times))
    started = _time.perf_counter()
    run = dam_break.run_case(cfg, snapshot_times=times_requested)
    elapsed = _time.perf_counter() - started

    result = RitterResult(
        label=case.label or f"dx={case.dx:g}",
        case=case,
        times=list(run.times),
        front=list(run.front),
        snapshots=list(run.snapshots),
        gravity=case.gravity,
        mass_drift=float(run.final_mass - run.initial_mass),
        initial_mass=float(run.initial_mass),
        escaped=int(run.escaped),
        steps_taken=int(run.steps_taken),
        simulated_time=float(run.simulated_time),
        max_density_compression=float(run.max_density_compression),
        max_acoustic_courant=float(run.max_acoustic_courant),
        wall_clock=float(elapsed),
        particle_count=int(run.snapshots[0][1].shape[0]) if run.snapshots else 0,
    )
    # The solver keeps the initial mass array for the whole run (SPH mass is a
    # constant), so rebuild it exactly instead of assuming a uniform lattice:
    # the hydrostatic initialisation compresses the lower rows by ~0.2 %.
    device = droplet_impact.resolve_device(cfg.device)
    _, mass = dam_break.make_column(cfg, device)
    for t, pos, vel in run.snapshots:
        if mass.shape[0] != pos.shape[0]:
            raise RuntimeError("initial and snapshot particle counts differ")
        result.rows.append(compare_snapshot(case, t, pos, vel, mass,
                                            rho0=float(cfg.rho0)))
    return result
