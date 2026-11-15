"""3-D Kleefsman dam-break-with-obstacle case (the geometry the experiment used).

Why this module exists
----------------------
``tensorsph.kleefsman_obstacle`` solves the case in **two** dimensions, where the
0.161 x 0.161 m box necessarily spans the whole tank width: 100 % blockage.
The experiment (Kleefsman et al. 2005, Fig. 17/19) has a 1.00 m wide tank and a
0.403 m wide box, so only 40 % of the width is blocked and 60 % of the water
bypasses the box along its side.

That difference is not cosmetic.  In the 2-D runs the surge piles up against a
full-width wall and forms a run-up jet of 0.42-0.54 m in front of the box, while
the experimental H2 gauge never sees more than 0.370 m.  Every committed 2-D
verdict of this case therefore failed the peak metric by 15-52 % (missions
m21324, m21586), and the residual was *attributed* to the blockage without ever
being measured.  This module measures it: it is the same solver, the same gauge
protocol and the same reference, with the third dimension of the tank restored
and the box given its real 0.403 m width.

What is measured
----------------
``slab@x2.426`` and ``height@x2.426`` are computed exactly as in the 2-D case
(frozen protocol: 0.08 m aperture, station 0.05 m upstream of the box face,
0.02 m arrival threshold), but in 3-D the slab is a *volume* slab and the water
height it reports is ``V / (aperture_x * blocked_width)`` -- the same
"metres of water over the blocked width" the 2-D areal operator reports.
Bypass is measured explicitly (``n_bypass``: fluid particles downstream of the
box that are outside the box's spanwise extent), so the geometry difference is
a reported number and not an assumption.

Scope / limitations, stated up front
-----------------------------------
* The lateral position of the box inside the 1.00 m tank is not stated in words
  in the paper and the setup figure (Fig. 2/17) is a raster.  This module
  therefore makes it a parameter (``obstacle_z``) and defaults to the box being
  flush with the front side wall (``z = 0``), which is the placement used by
  most SPH reproductions.  The bypass gap is then 0.597 m on one side; the
  alternative (centred box, two 0.2985 m gaps) is one keyword away.
* The tank walls are free-slip (``wall_friction = 0``, the solver default).
* Only the *peak* and *arrival* of the in-front-of-box water height are
  claimed.  Pressure-impulse comparisons on the box face are not attempted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path

import torch

from .dam_break_3d import (
    DamBreak3DConfig,
    apply_boundary_3d,
    make_column_3d,
    rhs_3d,
    wall_lattice_3d,
)
from .droplet_impact import next_dt
from .neighbor_search import PairListCache

# ---- experimental geometry (Kleefsman et al. 2005; see reference.json) ------
TANK_LENGTH_M = 3.22
TANK_HEIGHT_M = 1.00
TANK_WIDTH_M = 1.00
COLUMN_LENGTH_M = 1.228
COLUMN_HEIGHT_M = 0.550
OBSTACLE_X_M = 2.476          # upstream face of the box (1.228 + 1.248)
OBSTACLE_LENGTH_M = 0.161     # streamwise
OBSTACLE_HEIGHT_M = 0.161
OBSTACLE_WIDTH_M = 0.403      # spanwise; 3-D only

# ---- frozen acceptance protocol (identical to the 2-D case) ---------------
FROZEN_APERTURE_M = 0.08
FROZEN_STATION_OFFSET_M = 0.05
FROZEN_ARRIVAL_THRESHOLD_M = 0.02

# ---- digitised experimental H2 (Kleefsman 2005 Fig. 19 right panel) -------
REFERENCE_PEAK_HEIGHT_M = 0.370
REFERENCE_PEAK_TIME_S = 1.7027
REFERENCE_ARRIVAL_TIME_S = 0.3591
TOLERANCE_REL = 0.10


@dataclass
class Kleefsman3DConfig(DamBreak3DConfig):
    """Geometry and gauge set-up of the 3-D obstacle case."""

    # tank: x in [0, tank_length], y in [0, tank_height], z in [0, tank_width]
    dx: float = 0.04
    pool_width: float = TANK_LENGTH_M          # tank x-extent
    tank_depth: float = TANK_WIDTH_M           # tank z-extent
    tank_height: float = TANK_HEIGHT_M
    # water column: x in [0, column_width], z spans the full tank width
    column_width: float = COLUMN_LENGTH_M
    column_height: float = COLUMN_HEIGHT_M
    column_depth: float = TANK_WIDTH_M
    # obstacle
    obstacle_x: float = OBSTACLE_X_M
    obstacle_length: float = OBSTACLE_LENGTH_M
    obstacle_height: float = OBSTACLE_HEIGHT_M
    obstacle_width: float = OBSTACLE_WIDTH_M
    obstacle_z: float = 0.0                    # spanwise origin of the box
    # gauge
    station_offset: float = FROZEN_STATION_OFFSET_M
    gauge_aperture: float = FROZEN_APERTURE_M
    arrival_threshold: float = FROZEN_ARRIVAL_THRESHOLD_M
    # run control
    dt: float = 2.6e-4
    t_end: float = 2.6
    output_every: int = 25
    max_steps: int = 0
    wall_layers: int = 2
    pair_skin_factor: float = 0.25

    # ---- derived ----------------------------------------------------------
    def obstacle_x1(self) -> float:
        return self.obstacle_x + self.obstacle_length

    def obstacle_z1(self) -> float:
        return self.obstacle_z + self.obstacle_width

    def station_x(self) -> float:
        return self.obstacle_x - self.station_offset

    def blocked_fraction(self) -> float:
        return self.obstacle_width / self.tank_depth

    def steps_for_t_end(self) -> int:
        return int(math.ceil(self.t_end / self.dt)) + 2


def validate_case(cfg: Kleefsman3DConfig) -> None:
    """Reject geometry that does not reproduce the published apparatus."""
    tol = 1.0e-9
    if abs(cfg.pool_width - TANK_LENGTH_M) > tol:
        raise ValueError("tank length must be 3.22 m (Kleefsman et al. 2005)")
    if abs(cfg.tank_height - TANK_HEIGHT_M) > tol:
        raise ValueError("tank height must be 1.00 m")
    if abs(cfg.tank_depth - TANK_WIDTH_M) > tol:
        raise ValueError("tank width must be 1.00 m")
    if abs(cfg.column_width - COLUMN_LENGTH_M) > tol:
        raise ValueError("column length must be 1.228 m")
    if abs(cfg.column_height - COLUMN_HEIGHT_M) > tol:
        raise ValueError("column height must be 0.550 m")
    if abs(cfg.column_depth - cfg.tank_depth) > tol:
        raise ValueError("the water column must span the full tank width")
    if abs(cfg.obstacle_x - OBSTACLE_X_M) > tol:
        raise ValueError("obstacle upstream face must be at x = 2.476 m")
    if abs(cfg.obstacle_height - OBSTACLE_HEIGHT_M) > tol:
        raise ValueError("obstacle height must be 0.161 m")
    if abs(cfg.obstacle_width - OBSTACLE_WIDTH_M) > tol:
        raise ValueError("obstacle width must be 0.403 m (3-D case only)")
    if cfg.obstacle_z < -tol or cfg.obstacle_z1() > cfg.tank_depth + tol:
        raise ValueError("obstacle must lie inside the tank width")
    if cfg.obstacle_x1() > cfg.pool_width:
        raise ValueError("obstacle must lie inside the tank length")
    if cfg.obstacle_height > cfg.tank_height:
        raise ValueError("obstacle must be shorter than the tank")
    if cfg.dx <= 0.0:
        raise ValueError("dx must be positive")
    # the box has to be resolved at least 3 cells across in every direction
    for name, extent in (("length", cfg.obstacle_length),
                         ("height", cfg.obstacle_height),
                         ("width", cfg.obstacle_width)):
        if extent < 3.0 * cfg.dx:
            raise ValueError(
                f"obstacle {name} {extent:g} m is under-resolved at dx={cfg.dx:g}")
    if cfg.gauge_aperture <= 0.0 or cfg.gauge_aperture >= cfg.obstacle_x:
        raise ValueError("gauge aperture must be positive and upstream of the box")
    if cfg.station_x() <= cfg.column_width:
        raise ValueError("gauge station falls inside the reservoir")
    courant = cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx)
    if courant > cfg.max_courant * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} gives acoustic Courant {courant:.3f} > "
            f"max_courant {cfg.max_courant:g}; use dt <= {cfg.dt_from_courant():.3e}")


# --------------------------------------------------------------------------- #
# Obstacle as solid boundary particles + a hard non-penetration clamp
# --------------------------------------------------------------------------- #

def _lattice_extent(extent: float, dx: float) -> float:
    """Length of ``extent`` rounded to the nearest whole number of cells."""
    return max(1, int(round(extent / dx))) * dx


def obstacle_lattice_3d(cfg: Kleefsman3DConfig, device: torch.device) -> torch.Tensor:
    """Solid particles filling the box volume on the fluid lattice.

    A filled block (rather than a surface shell) keeps the pressure mirroring in
    :func:`tensorsph.dam_break_3d.rhs_3d` well defined on every face, which a
    three-layer shell does not at the box corners at these resolutions.
    """
    dx = cfg.dx

    def span(lo: float, hi: float) -> torch.Tensor:
        n = max(1, int(round((hi - lo) / dx)))
        return lo + (torch.arange(n, device=device, dtype=torch.float64) + 0.5) * dx

    xs = span(cfg.obstacle_x, cfg.obstacle_x1())
    ys = span(0.0, cfg.obstacle_height)
    zs = span(cfg.obstacle_z, cfg.obstacle_z1())
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)),
                       dim=-1)


def clamp_out_of_obstacle(pos: torch.Tensor, vel: torch.Tensor,
                          cfg: Kleefsman3DConfig) -> int:
    """Project fluid particles out of the box volume and kill normal velocity.

    Wall particles alone cannot be trusted to stop a 9 m/s run-up jet on a
    coarse grid, and a particle that tunnels into the box would corrupt the
    water-height gauge (it would read a particle that is *inside* the
    obstacle).  The clamp makes non-penetration an invariant rather than an
    expectation, and returns how many particles had to be moved so that a run
    can report it.
    """
    if pos.numel() == 0:
        return 0
    x0, x1 = cfg.obstacle_x, cfg.obstacle_x1()
    y0, y1 = 0.0, cfg.obstacle_height
    z0, z1 = cfg.obstacle_z, cfg.obstacle_z1()
    inside = ((pos[:, 0] > x0) & (pos[:, 0] < x1)
              & (pos[:, 1] > y0) & (pos[:, 1] < y1)
              & (pos[:, 2] > z0) & (pos[:, 2] < z1))
    count = int(inside.sum().item())
    if count == 0:
        return 0

    idx = inside.nonzero(as_tuple=False).reshape(-1)
    px = pos[idx]
    # distance to each of the six faces, exit through the nearest one
    candidates = torch.stack((px[:, 0] - x0, x1 - px[:, 0],
                              px[:, 1] - y0, y1 - px[:, 1],
                              px[:, 2] - z0, z1 - px[:, 2]), dim=1)
    which = candidates.argmin(dim=1)
    new = px.clone()
    eps = 0.01 * cfg.dx
    # face 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
    out_x_lo = which == 0
    out_x_hi = which == 1
    out_y_lo = which == 2
    out_y_hi = which == 3
    out_z_lo = which == 4
    out_z_hi = which == 5
    new[:, 0] = torch.where(out_x_lo, torch.full_like(new[:, 0], x0 - eps),
                            torch.where(out_x_hi, torch.full_like(new[:, 0], x1 + eps),
                                        new[:, 0]))
    new[:, 1] = torch.where(out_y_lo, torch.full_like(new[:, 1], y0 - eps),
                            torch.where(out_y_hi, torch.full_like(new[:, 1], y1 + eps),
                                        new[:, 1]))
    new[:, 2] = torch.where(out_z_lo, torch.full_like(new[:, 2], z0 - eps),
                            torch.where(out_z_hi, torch.full_like(new[:, 2], z1 + eps),
                                        new[:, 2]))
    pos[idx] = new

    nv = vel[idx]
    nv[:, 0] = torch.where(out_x_lo | out_x_hi, torch.zeros_like(nv[:, 0]), nv[:, 0])
    nv[:, 1] = torch.where(out_y_lo | out_y_hi, torch.zeros_like(nv[:, 1]), nv[:, 1])
    nv[:, 2] = torch.where(out_z_lo | out_z_hi, torch.zeros_like(nv[:, 2]), nv[:, 2])
    vel[idx] = nv
    return count


# --------------------------------------------------------------------------- #
# Gauges: "metres of water in front of the box", the same two operators as 2-D
# --------------------------------------------------------------------------- #

def _station_mask(pos: torch.Tensor, cfg: Kleefsman3DConfig) -> torch.Tensor:
    half = 0.5 * cfg.gauge_aperture
    return ((pos[:, 0] > cfg.station_x() - half)
            & (pos[:, 0] < cfg.station_x() + half)
            & (pos[:, 2] > cfg.obstacle_z)
            & (pos[:, 2] < cfg.obstacle_z1()))


def slab_height(pos: torch.Tensor, cfg: Kleefsman3DConfig) -> float:
    """Volume-equivalent water height over the blocked width inside the slab.

    ``h = sum(m_i / rho0) / (aperture_x * blocked_width)``.  In 2-D this reduces
    exactly to the areal operator ``A / aperture`` used by the frozen protocol,
    so the two runs are compared with the same definition of "water height".
    """
    if pos.numel() == 0:
        return 0.0
    mask = _station_mask(pos, cfg)
    n = int(mask.sum().item())
    if n == 0:
        return 0.0
    area = cfg.gauge_aperture * cfg.obstacle_width
    return n * cfg.dx ** 3 / area


def peak_height(pos: torch.Tensor, cfg: Kleefsman3DConfig) -> float:
    """Highest wetted point inside the slab (the 2-D ``height`` operator)."""
    if pos.numel() == 0:
        return 0.0
    mask = _station_mask(pos, cfg)
    if not bool(mask.any()):
        return 0.0
    return float(pos[mask, 1].max())


def bypass_count(pos: torch.Tensor, cfg: Kleefsman3DConfig) -> int:
    """Fluid particles past the box that are *outside* its spanwise extent.

    This is the direct measurement of the geometry difference between the 2-D
    model (this count is identically zero) and the experiment (60 % of the
    width is open).
    """
    if pos.numel() == 0:
        return 0
    past = pos[:, 0] > cfg.obstacle_x1()
    outside_z = (pos[:, 2] < cfg.obstacle_z) | (pos[:, 2] > cfg.obstacle_z1())
    return int((past & outside_z).sum().item())


# --------------------------------------------------------------------------- #
# Time series helpers
# --------------------------------------------------------------------------- #

def arrival_time(times, values, threshold: float) -> float | None:
    """First time a series crosses ``threshold``, linearly interpolated."""
    for k in range(1, len(values)):
        if values[k - 1] < threshold <= values[k]:
            t0, t1 = times[k - 1], times[k]
            v0, v1 = values[k - 1], values[k]
            if v1 == v0:
                return float(t1)
            return float(t0 + (threshold - v0) * (t1 - t0) / (v1 - v0))
    return None


@dataclass
class Kleefsman3DRun:
    cfg: Kleefsman3DConfig
    times: list[float] = field(default_factory=list)
    slab: list[float] = field(default_factory=list)
    height: list[float] = field(default_factory=list)
    front: list[float] = field(default_factory=list)
    bypass: list[float] = field(default_factory=list)
    max_speed: float = 0.0
    max_density_compression: float = 0.0
    boundary_escapes: int = 0
    obstacle_clamps: int = 0
    mass_error_rel: float = 0.0
    fluid_particles: int = 0
    wall_particles: int = 0
    steps: int = 0
    simulated_time: float = 0.0
    wall_clock_s: float = 0.0
    resumed: bool = False

    # ---- derived ----------------------------------------------------------
    def peak(self):
        if not self.slab:
            return 0.0, 0.0
        k = max(range(len(self.slab)), key=lambda i: self.slab[i])
        return float(self.slab[k]), float(self.times[k])

    def arrival(self) -> float | None:
        return arrival_time(self.times, self.slab, self.cfg.arrival_threshold)

    def to_dict(self) -> dict:
        peak_h, peak_t = self.peak()
        return {
            "dx": self.cfg.dx,
            "t_end": self.cfg.t_end,
            "dt": self.cfg.dt,
            "sound_speed": self.cfg.sound_speed,
            "fluid_particles": self.fluid_particles,
            "wall_particles": self.wall_particles,
            "blocked_fraction": self.cfg.blocked_fraction(),
            "obstacle_z": self.cfg.obstacle_z,
            "obstacle_width_nominal_m": self.cfg.obstacle_width,
            "obstacle_width_on_lattice_m": _lattice_extent(
                self.cfg.obstacle_width, self.cfg.dx),
            "stations_x_m": [self.cfg.station_x()],
            "gauge_aperture_m": self.cfg.gauge_aperture,
            "peak_height_m": peak_h,
            "peak_time_s": peak_t,
            "peak_height_operator_m": max(self.height) if self.height else 0.0,
            "arrival_time_s": self.arrival(),
            "arrival_threshold_m": self.cfg.arrival_threshold,
            "max_speed": self.max_speed,
            "max_density_compression": self.max_density_compression,
            "boundary_escapes": self.boundary_escapes,
            "obstacle_clamps": self.obstacle_clamps,
            "mass_error_rel": self.mass_error_rel,
            "steps": self.steps,
            "simulated_time_s": self.simulated_time,
            "wall_clock_s": self.wall_clock_s,
            "max_bypass_particles": max(self.bypass) if self.bypass else 0.0,
            # Every field the ladder compares runs on.  Written here (and not
            # reconstructed downstream) because a missing key makes
            # ladder.check_protocol() skip that key, which silently turns a
            # comparability guard into a vacuous one.
            "fingerprint": {
                "t_end": self.cfg.t_end,
                "sound_speed": self.cfg.sound_speed,
                "wall_layers": self.cfg.wall_layers,
                "obstacle_z": self.cfg.obstacle_z,
                "station_offset": self.cfg.station_offset,
                "gauge_aperture": self.cfg.gauge_aperture,
                "arrival_threshold": self.cfg.arrival_threshold,
            },
        }


# --------------------------------------------------------------------------- #
# Crash-safe checkpointing
# --------------------------------------------------------------------------- #
#
# The production rungs of this case die intermittently of a glibc heap abort
# (``free(): corrupted unsorted chunks``) after 1-2 s of simulated time; the
# abort takes the whole process down, so no Python-level ``except`` can save the
# run and every restart began again at ``t = 0``.  The solver state needed to
# continue is small (positions, velocities, elapsed time, the gauge series and
# the running diagnostics), so it is checkpointed here with an atomic replace:
# a checkpoint written by a process that is killed *while writing* is either the
# previous one or the new one, never a truncated file.
#
# ``t_end`` is deliberately **not** part of the fingerprint: it is a stopping
# criterion, not physics, and a run has to be resumable with a longer horizon
# precisely because the crash usually happens before the requested end.

CHECKPOINT_VERSION = 1

#: config fields that make a checkpoint of another run useless
CHECKPOINT_FINGERPRINT_FIELDS = (
    "dx", "sound_speed", "gamma", "rho0", "h_factor", "wall_layers",
    "obstacle_z", "obstacle_width", "station_offset", "gauge_aperture",
    "arrival_threshold", "pair_skin_factor", "density_diffusion",
    "surface_pressure_closure", "adaptive_dt", "max_courant", "viscosity",
    "pool_width", "tank_depth", "tank_height", "column_width",
    "column_height", "column_depth", "obstacle_x", "obstacle_length",
    "obstacle_height",
)


def checkpoint_fingerprint(cfg) -> dict:
    """The subset of ``cfg`` that decides whether a checkpoint is reusable."""
    return {name: getattr(cfg, name) for name in CHECKPOINT_FINGERPRINT_FIELDS}


def save_checkpoint(path, cfg, *, step, elapsed, dt, positions, velocity,
                    run, speeds, compression, escapes, clamps) -> None:
    """Write the integration state atomically (``tmp`` + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": checkpoint_fingerprint(cfg),
        "step": int(step),
        "elapsed": float(elapsed),
        "dt": float(dt),
        "positions": positions.detach().to("cpu").clone(),
        "velocity": velocity.detach().to("cpu").clone(),
        "times": list(run.times),
        "height": list(run.height),
        "front": list(run.front),
        "bypass": list(run.bypass),
        "speeds": float(speeds),
        "compression": float(compression),
        "escapes": int(escapes),
        "clamps": int(clamps),
    }
    if hasattr(run, "slab"):
        payload["slab"] = list(run.slab)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, cfg) -> dict:
    """Load a checkpoint, refusing one written for different physics."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint version {payload.get('version')} != {CHECKPOINT_VERSION}")
    got = payload.get("fingerprint", {})
    want = checkpoint_fingerprint(cfg)
    diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    if diff:
        raise ValueError(
            "checkpoint was written for a different configuration: "
            + ", ".join(f"{k}: {g!r} -> {w!r}" for k, (g, w) in diff.items()))
    return payload


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_case(cfg: Kleefsman3DConfig, device: str | torch.device = "cpu",
             progress=None, on_sample=None,
             checkpoint_path=None, checkpoint_every: int = 0,
             resume: bool = True) -> Kleefsman3DRun:
    """Integrate the 3-D case to ``cfg.t_end`` and record the gauge series.

    ``on_sample(i, sample)`` is called once per stored output sample with a dict
    of that sample.  It exists so a driver can persist the series *as it is
    produced*: these 3-D runs are half-hour cluster jobs and the two production
    rungs submitted on 11-15 died of a glibc abort (``free(): corrupted unsorted
    chunks``) at t = 2.04 s and t = 0.58 s, throwing away every sample because
    the series was only written after the loop returned.
    """
    import time as _time

    validate_case(cfg)
    dev = torch.device(device)
    t_start = _time.time()

    positions, mass = make_column_3d(cfg, dev)
    fluid_count = positions.shape[0]
    wall = wall_lattice_3d(cfg, dev)
    obstacle = obstacle_lattice_3d(cfg, dev).to(wall.dtype)
    wall = torch.cat((wall, obstacle), dim=0)
    n_wall = wall.shape[0]

    # the box is solid: make sure no initial fluid particle sits inside it
    clamp_out_of_obstacle(positions, torch.zeros_like(positions), cfg)

    velocity = torch.zeros_like(positions)
    all_positions = torch.cat((positions, wall), dim=0)
    cache = PairListCache(cutoff=2.0 * cfg.h_factor * cfg.dx,
                          skin=cfg.pair_skin_factor * cfg.h_factor * cfg.dx,
                          fluid_count=fluid_count)

    run = Kleefsman3DRun(cfg=cfg, fluid_particles=fluid_count,
                         wall_particles=n_wall)
    initial_mass = float(mass.sum())

    dt = cfg.dt
    elapsed = 0.0
    max_steps = cfg.max_steps or cfg.steps_for_t_end()
    speeds = 0.0
    compression = 0.0
    escapes = 0
    clamps = 0
    start_step = 0
    resumed = False

    if checkpoint_path is not None and resume and Path(checkpoint_path).exists():
        state = load_checkpoint(checkpoint_path, cfg)
        if state["positions"].shape != positions.shape:
            raise ValueError(
                f"checkpoint holds {tuple(state['positions'].shape)} particles, "
                f"this configuration builds {tuple(positions.shape)}")
        positions = state["positions"].to(dev).clone()
        velocity = state["velocity"].to(dev).clone()
        dt = state["dt"]
        elapsed = state["elapsed"]
        start_step = state["step"] + 1
        run.times = list(state["times"])
        run.height = list(state["height"])
        run.front = list(state["front"])
        run.bypass = list(state["bypass"])
        if hasattr(run, "slab"):
            run.slab = list(state.get("slab", []))
        speeds = state["speeds"]
        compression = state["compression"]
        escapes = state["escapes"]
        clamps = state["clamps"]
        resumed = True
        if progress is not None:
            progress(elapsed, run.slab[-1] if getattr(run, "slab", None) else 0.0)

    def _checkpoint(step: int) -> None:
        save_checkpoint(checkpoint_path, cfg, step=step, elapsed=elapsed, dt=dt,
                        positions=positions, velocity=velocity, run=run,
                        speeds=speeds, compression=compression,
                        escapes=escapes, clamps=clamps)

    for step in range(start_step, max_steps + 1):
        all_positions[:fluid_count] = positions
        pairs, _ = cache.get(all_positions)
        density, _, acc = rhs_3d(positions, velocity, mass, cfg, pairs=pairs,
                                 wall=wall, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        compression = max(compression, float(density.max()) / cfg.rho0 - 1.0)
        speed_now = float(torch.linalg.vector_norm(velocity, dim=-1).max())
        speeds = max(speeds, speed_now)

        final = elapsed >= cfg.t_end - 1.0e-12
        if step % cfg.output_every == 0 or final:
            run.times.append(elapsed)
            run.slab.append(slab_height(positions, cfg))
            run.height.append(peak_height(positions, cfg))
            run.front.append(float(positions[:, 0].max()))
            run.bypass.append(float(bypass_count(positions, cfg)))
            if on_sample is not None:
                on_sample(len(run.times) - 1, {
                    "t_s": elapsed,
                    "slab_height_m": run.slab[-1],
                    "height_operator_m": run.height[-1],
                    "front_x_m": run.front[-1],
                    "n_bypass": run.bypass[-1],
                })
            if progress is not None and step % (cfg.output_every * 20) == 0:
                progress(elapsed, run.slab[-1])

        if final:
            break
        velocity = velocity + dt * acc
        positions = positions + dt * velocity
        escapes += apply_boundary_3d(positions, velocity, cfg,
                                     top=cfg.tank_top() if hasattr(cfg, "tank_top") else None)
        clamps += clamp_out_of_obstacle(positions, velocity, cfg)
        elapsed += dt
        if cfg.adaptive_dt:
            dt = next_dt(cfg, dt, speed_now)
        if checkpoint_path is not None and checkpoint_every > 0 and \
                step % checkpoint_every == 0:
            _checkpoint(step)

    run.max_speed = speeds
    run.max_density_compression = compression
    run.boundary_escapes = escapes
    run.obstacle_clamps = clamps
    run.mass_error_rel = abs(float(mass.sum()) - initial_mass) / max(initial_mass, 1e-30)
    run.steps = step
    run.simulated_time = elapsed
    run.wall_clock_s = _time.time() - t_start
    run.resumed = resumed
    if checkpoint_path is not None:
        # a completed run must not be resumed by the next attempt of the
        # wrapper loop: drop the state file once t_end has been reached
        try:
            os.unlink(checkpoint_path)
        except FileNotFoundError:
            pass
    return run


# --------------------------------------------------------------------------- #
# Scoring against the digitised experiment
# --------------------------------------------------------------------------- #

def score(run: Kleefsman3DRun, tolerance_rel: float = TOLERANCE_REL) -> dict:
    """Peak and arrival of the in-front-of-box height against Kleefsman Fig. 19."""
    peak_h, peak_t = run.peak()
    arrival = run.arrival()
    rows = []

    def add(metric, ref, got):
        rel = abs(got - ref) / ref if got is not None and ref else float("inf")
        rows.append({"metric": metric, "ref": ref, "got": got,
                     "rel_error": rel, "pass": bool(rel <= tolerance_rel)})

    add("peak_height", REFERENCE_PEAK_HEIGHT_M, peak_h)
    if arrival is None:
        rows.append({"metric": "arrival_time", "ref": REFERENCE_ARRIVAL_TIME_S,
                     "got": None, "rel_error": float("inf"), "pass": False})
    else:
        add("arrival_time", REFERENCE_ARRIVAL_TIME_S, arrival)

    # a maximum sitting on the last samples of a record that stops before the
    # experimental peak instant is a *censored* maximum: the run was cut short
    # before the event the reference measures had finished.  Such a run cannot
    # pass even if its truncated maximum happens to match the reference.
    censored = False
    if run.times and run.slab:
        k_peak = max(range(len(run.slab)), key=lambda i: run.slab[i])
        censored = bool(k_peak >= len(run.times) - 2
                        and run.times[-1] < REFERENCE_PEAK_TIME_S)
    passed = all(r["pass"] for r in rows) and not censored

    return {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "geometry": "3-D (0.403 m box in 1.00 m tank, 40 % blockage)",
        "reference": {
            "peak_height_m": REFERENCE_PEAK_HEIGHT_M,
            "peak_time_s": REFERENCE_PEAK_TIME_S,
            "arrival_time_s": REFERENCE_ARRIVAL_TIME_S,
            "source": "Kleefsman et al. (2005) J. Comput. Phys. 206:363-393, "
                      "doi:10.1016/j.jcp.2004.12.007, Fig. 19 right panel",
        },
        "tolerance_rel": tolerance_rel,
        "rows": rows,
        "window_covers_reference_peak": bool(run.times and run.times[-1] >= REFERENCE_PEAK_TIME_S),
        "censored": censored,
        "verdict": "PASS" if passed else "FAIL",
        "run": run.to_dict(),
    }


__all__ = [
    "Kleefsman3DConfig", "Kleefsman3DRun", "arrival_time", "bypass_count",
    "clamp_out_of_obstacle", "obstacle_lattice_3d", "peak_height", "run_case",
    "score", "slab_height", "validate_case",
    "REFERENCE_PEAK_HEIGHT_M", "REFERENCE_PEAK_TIME_S",
    "REFERENCE_ARRIVAL_TIME_S", "TOLERANCE_REL",
]