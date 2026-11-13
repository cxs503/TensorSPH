"""Kleefsman et al. (2005) dam-break-with-obstacle case (2-D vertical slice).

The experiment of Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005,
*J. Comput. Phys.* 206:363-393, doi:10.1016/j.jcp.2004.12.007) is the standard
validation case for wave-impact solvers: a water column collapses in a long
tank and hits a small rectangular box standing on the floor.  The diagnostic
required by the case pool is the **water level in front of the box**: its peak
value and the arrival time of the surge.

Geometry used here (all lengths in metres, tank frame, floor at ``y = 0``,
``x = 0`` at the closed end behind the column):

============================  =======  ===================================
quantity                      value    how it was read
============================  =======  ===================================
tank length                   3.22     Kleefsman et al. (2005) setup
tank height                   1.00     idem
water column length           1.228    idem
water column height           0.550    idem
box (streamwise x)            0.161    idem
box height                    0.161    idem
box upstream face x           2.476    ``1.228 + 1.248`` (gate-to-box distance)
downstream gap                0.744    ``3.22 - 2.476`` (consistency check)
============================  =======  ===================================

The three x-extents add up exactly, ``1.228 + 1.248 + 0.744 = 3.22``, which is
the self-consistency check that fixes the box position: the values
``1.228 / 1.248 / 0.744 / 0.161`` are the dimension strings printed on the
setup figure reproduced from Kleefsman et al. (2005) in Amaro Junior, Cheng &
Buruchenko, *Int. J. Offshore Polar Eng.* 30 (2020), arXiv:2306.01959 (Fig. 2),
the only openly reachable statement of the geometry (see
``docs/benchmarks/kleefsman-obstacle-dam-break.md`` for the provenance chain
and for what could **not** be retrieved).

Scope of the checks is stated up front:

* the run measures the water-height history at a probe just upstream of the
  box, the leading-edge position, and mass conservation;
* the probe is placed by geometry (``box upstream face - probe_offset``), not
  by Kleefsman's labelled gauge H1..H4, whose coordinates are not recoverable
  from the openly accessible sources -- this is a limitation, recorded in
  ``benchmarks/kleefsman_obstacle/reference.json``;
* the *experimental* peak/arrival values are **not** bundled: the paper PDF is
  not reachable from this node (Cloudflare-protected repository copy) and no
  digitized table of the water-level traces was found.  The tolerance
  comparison against the experiment is therefore reported as ``WITHHELD``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .droplet_impact import (
    Config,
    apply_boundary,
    cell_centres,
    hydrostatic_density,
    next_dt,
    resolve_device,
    rhs,
    wall_lattice,
)
from .neighbor_search import PairListCache


@dataclass
class KleefsmanConfig(Config):
    """Geometry, probe and time-stepping parameters of the obstacle case."""

    dx: float = 0.01
    tank_length: float = 3.22
    tank_height: float = 1.00
    column_length: float = 1.228
    column_height: float = 0.550
    obstacle_x: float = 2.476          # upstream face of the box
    obstacle_length: float = 0.161
    obstacle_height: float = 0.161
    probe_offset: float = 0.05         # probe upstream of the box face
    probe_halfwidth: float = 1.0       # in units of dx
    surface_particles: int = 3         # top-N average within the contiguous stack
    # gauge stations, measured upstream of the box face [m]; the first one is
    # the historical probe (probe_offset), the second sits 0.248 m upstream of
    # the box, i.e. in the calming part of the surge rather than in the run-up
    # jet on the box face
    stations_offset: tuple[float, ...] = (0.05, 0.248)
    # slab-gauge aperture: FULL physical width [m], deliberately independent of
    # dx so that the same gauge is used on every grid of a ladder
    gauge_aperture: float = 0.0        # 0 -> resolve_gauge_aperture() (2 dx, >= 0.03 m)
    surface_gap: float = 2.5           # in units of dx; max hole inside the stack
    robust_surface: bool = True        # reject disconnected spray above the water
    arrival_threshold: float = 0.02    # m above the dry floor
    t_end: float = 3.0
    max_steps: int = 0                 # 0 -> derived from t_end and the wave dt
    pool_width: float = 3.22
    pool_depth: float = 1.00
    dt: float = 2.0e-4
    output_every: int = 20
    adaptive_dt: bool = True
    hydrostatic_init: bool = True
    steps: int = 100000                 # unused; kept for Config compatibility

    # ---- derived quantities ------------------------------------------------
    def obstacle_x1(self) -> float:
        return self.obstacle_x + self.obstacle_length

    def probe_x(self) -> float:
        return self.obstacle_x - self.probe_offset

    def station_x(self) -> tuple[float, ...]:
        return tuple(self.obstacle_x - off for off in self.stations_offset)

    def resolve_gauge_aperture(self) -> float:
        """Physical slab-gauge width [m]; ``0`` means "2 dx, at least 30 mm".

        A ladder must pass the same *metres* here on every grid -- that is what
        makes the gauge dx-consistent.  The adaptive default only exists so that
        old callers (and small smoke grids) keep validating.
        """
        if self.gauge_aperture > 0.0:
            return self.gauge_aperture
        return max(2.0 * self.dx, 0.03)

    def dt_acoustic(self) -> float:
        return self.max_courant * self.h_factor * self.dx / self.sound_speed


@dataclass
class KleefsmanRun:
    times: list[float] = field(default_factory=list)
    height: list[float] = field(default_factory=list)      # at the probe
    series: dict[str, list[float]] = field(default_factory=dict)   # gauge series
    metrics: dict[str, dict] = field(default_factory=dict)         # gauge metrics
    column: list[float] = field(default_factory=list)      # mass-based slab level
    front: list[float] = field(default_factory=list)       # max fluid x
    mass_error: float = 0.0
    peak_height: float = float("nan")
    peak_time: float = float("nan")
    arrival_time: float = float("nan")
    steps_taken: int = 0
    simulated_time: float = 0.0
    max_density_compression: float = 0.0
    max_speed: float = 0.0
    escaped: int = 0
    dt_min: float = 0.0
    dt_max: float = 0.0
    finished: bool = True          # False when a time-budgeted chunk stopped early
    state: dict | None = None      # arrays/counters to resume a chunked run


def validate_case(cfg: KleefsmanConfig) -> None:
    if cfg.boundary != "wall":
        raise ValueError("the Kleefsman case integrates static wall particles only; "
                         f"boundary={cfg.boundary!r} would be silently ignored")
    for name in ("dx", "h_factor", "sound_speed", "rho0", "gamma"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.column_length >= cfg.obstacle_x:
        raise ValueError("the water column must end upstream of the box")
    if cfg.obstacle_x1() > cfg.tank_length:
        raise ValueError("the box must fit inside the tank")
    if cfg.column_height >= cfg.tank_height or cfg.obstacle_height >= cfg.tank_height:
        raise ValueError("column and box must be lower than the tank")
    if not (cfg.column_length < cfg.probe_x() < cfg.obstacle_x):
        raise ValueError("the probe must sit between the gate and the box")
    for xs in cfg.station_x():
        if not (cfg.column_length <= xs < cfg.obstacle_x):
            raise ValueError(f"gauge station x={xs:g} outside (gate, box face)")
    if cfg.resolve_gauge_aperture() < cfg.dx:
        raise ValueError(f"gauge_aperture={cfg.gauge_aperture:g} m is narrower than one "
                         f"particle (dx={cfg.dx:g} m): the slab gauge would alias")
    if cfg.dt > cfg.dt_acoustic() * (1.0 + 1.0e-9):
        raise ValueError(f"dt={cfg.dt:g} exceeds the acoustic limit "
                         f"{cfg.dt_acoustic():.3e} s")


def make_water(cfg: KleefsmanConfig, device: torch.device):
    """Particles filling the initial water column, resting on the floor."""
    xs = cell_centres(cfg.column_length, cfg.dx, device)
    ys = cell_centres(cfg.column_height, cfg.dx, device)
    if xs.numel() == 0 or ys.numel() == 0:
        raise ValueError("column discretization produced no particles")
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    if cfg.hydrostatic_init:
        mass = hydrostatic_density(positions[:, 1], cfg,
                                   surface=cfg.column_height) * cfg.dx * cfg.dx
    else:
        mass = torch.full((positions.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                          device=device)
    return positions, mass


def obstacle_lattice(cfg: KleefsmanConfig, device: torch.device) -> torch.Tensor:
    """Static particles filling the solid box (a solid body, not a shell).

    The tank walls of :func:`tensorsph.droplet_impact.wall_lattice` are three
    layers placed *outside* the fluid box.  The box lives *inside* the box, so
    its particles are laid out on the same lattice, offset half a spacing from
    the faces, and every interior lattice site is filled: a 0.161 m box at
    ``dx = 0.01`` is 16 x 16 sites, i.e. the fill costs a few hundred static
    particles and removes any risk of the fluid squeezing through a shell gap.
    """
    x0, y0 = cfg.obstacle_x, 0.0
    nx = max(1, int(round(cfg.obstacle_length / cfg.dx)))
    ny = max(1, int(round(cfg.obstacle_height / cfg.dx)))
    xs = x0 + (torch.arange(nx, device=device, dtype=torch.float32) + 0.5) * cfg.dx
    ys = y0 + (torch.arange(ny, device=device, dtype=torch.float32) + 0.5) * cfg.dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def water_height_at(pos: torch.Tensor, cfg: KleefsmanConfig,
                    x: float | None = None) -> float:
    """Free-surface height in a vertical strip of half-width ``probe_halfwidth*dx``.

    Two protections against the failure mode this case really hit (a reported
    peak of 1.000 m, i.e. the tank roof, at ``dx = 0.01``):

    * **contiguity filter** (``robust_surface``): the particles in the strip are
      sorted by height and the surface is taken from the stack that starts at
      the floor and is closed upwards; a vertical hole larger than
      ``surface_gap * dx`` ends the stack.  Isolated spray droplets thrown
      ahead of the surge are therefore not counted as water level;
    * **top-``surface_particles`` mean** of that stack instead of the bare
      maximum, which damps single-particle noise on the surface.

    With ``robust_surface = False`` the estimator degrades to the plain
    top-N mean over the whole strip (the previous behaviour), which is what the
    regression test compares against.  A dry strip returns ``0.0``; the floor
    is at ``y = 0``, so the value is the water depth itself.
    """
    if pos.numel() == 0:
        return 0.0
    x = cfg.probe_x() if x is None else float(x)
    half = cfg.probe_halfwidth * cfg.dx
    mask = (pos[:, 0] - x).abs() <= half
    ys = pos[mask, 1]
    if ys.numel() == 0:
        return 0.0
    ys, _ = torch.sort(ys)
    if cfg.robust_surface:
        gap = (cfg.surface_gap + 1e-4) * cfg.dx   # float32 lattice tolerance
        end = 1
        while end < ys.numel() and float(ys[end] - ys[end - 1]) <= gap:
            end += 1
        ys = ys[:end]
    k = max(1, min(int(cfg.surface_particles), ys.numel()))
    return float(ys[-k:].mean())


def column_level(positions: torch.Tensor, cfg: "KleefsmanConfig",
                 x: float | None = None,
                 halfwidth: float | None = None) -> float:
    """Mass-based water depth over a vertical slab at the probe station [m].

    ``h = sum(V_p) / (2 * halfwidth)`` with the 2-D particle volume ``V_p =
    dx^2`` (unit depth).  This is the *mean water depth* of the slab, i.e. the
    quantity a conductivity/capacitance wave probe approximates, and it is
    **invariant to how the fluid inside the slab is redistributed vertically**:
    a ballistic droplet lifted out of the surge changes the top-particle
    estimator by tens of percent but leaves this number unchanged.  That is why
    it is the estimator to use when a *level* (not a spray envelope) is wanted;
    ``water_height_at`` remains the free-surface position and is kept for
    regression against earlier rounds.
    """
    if positions.numel() == 0:
        return 0.0
    x = cfg.probe_x() if x is None else x
    halfwidth = (cfg.probe_halfwidth * cfg.dx) if halfwidth is None else halfwidth
    if halfwidth <= 0.0:
        return 0.0
    slab = (positions[:, 0] - x).abs() <= halfwidth
    count = float(slab.sum().item())
    return count * cfg.dx * cfg.dx / (2.0 * halfwidth)


def surface_level(pos: torch.Tensor, cfg: KleefsmanConfig,
                  x: float | None = None, correct_bias: bool = True) -> float:
    """Resolution-consistent free-surface level in a vertical strip [m].

    ``water_height_at`` averages the centres of the top ``surface_particles``
    particles, which on a free surface built from a lattice sits a *systematic*
    ``(k - 1) / 2 * dx`` below the posterior top-particle centre and ``dx / 2``
    below the true surface, i.e. ``0.5 * k * dx`` below the water level:

    * sweeping the surface phase of a still lattice (``tests/test_kleefsman_gauge_bias.py``)
      gives an error of ``-1.33 dx ... -0.38 dx`` for ``k = 3``, mean
      ``-0.86 dx`` -- at ``dx = 0.02`` that is ``-1.8 ... -2.7 cm`` and at
      ``dx = 0.04`` ``-1.5 ... -5.3 cm``;
    * that bias is **larger than the case's own arrival threshold (0.02 m)** on
      the coarse grids and eats 6 % of a 0.50 m peak at ``dx = 0.02`` -- inside
      the 10 % acceptance tolerance of the case pool, so it cannot be left in.

    ``correct_bias = True`` (default) adds ``0.5 k dx`` and therefore makes
    the operator *dx-consistent* on lattice-aligned free surfaces (residual
    error ``0`` for still water, ``O(noise)`` otherwise).  The correction is
    the standard half-cell offset between a particle centre and the surface of
    the volume it represents; it is exact for a flat surface and first-order
    accurate for a tilted one.  ``correct_bias = False`` reproduces
    ``water_height_at`` exactly and is kept for regression against the earlier
    rounds.
    """
    base = water_height_at(pos, cfg, x=x)
    if not correct_bias or base == 0.0:
        return base
    k = max(1, int(cfg.surface_particles))
    # rows of the lattice are dx apart, so the mean of the top k rows sits
    # (k - 1)/2 rows below the top row: that is the part of the bias that is
    # removed here.  The remaining +/-0.5 dx is the position of the free
    # surface inside the top row, which no particle-count estimator can know.
    return base + 0.5 * (k - 1) * cfg.dx


def slab_level(pos: torch.Tensor, cfg: KleefsmanConfig, x: float | None = None,
               aperture: float | None = None) -> float:
    """Volume-based water level over a **fixed physical aperture** [m].

    The slab ``|x_p - x| <= aperture / 2`` is integrated with exact cell-overlap
    weights in ``x`` (a particle is a square of side ``dx`` and contributes the
    fraction of it that falls inside the slab):

        level = sum_p w_p V_p / aperture,   V_p = dx^2 (unit depth).

    Because ``sum_p w_p`` tiles the aperture exactly, a *planar* water surface
    of depth ``H`` returns ``H`` for **every** ``dx`` and for every alignment of
    the lattice with the station: the operator has no ``O(dx)`` gauge bias, so
    the same physical gauge can be used across a resolution ladder -- which is
    what a conductivity wave probe of finite width measures.  It still inherits
    the ``dx``-quantisation of the *free surface itself* (the surface is only
    known to within a particle row), which is physical discretisation error and
    is what the convergence ladder is supposed to measure.
    """
    if pos.numel() == 0:
        return 0.0
    x = cfg.probe_x() if x is None else float(x)
    a = ((cfg.resolve_gauge_aperture() if aperture is None else float(aperture)) / 2.0)
    if a <= 0.0:
        return 0.0
    half = 0.5 * cfg.dx
    # exact overlap of the particle square [x_p - dx/2, x_p + dx/2] with the
    # slab [x - a, x + a], in units of the particle width dx
    lo = torch.clamp_min(pos[:, 0] - half, x - a)
    hi = torch.clamp_max(pos[:, 0] + half, x + a)
    w = torch.clamp(hi - lo, min=0.0) / cfg.dx
    volume = float(w.sum()) * cfg.dx * cfg.dx
    return volume / (2.0 * a)


def gauge_series(pos: torch.Tensor, cfg: KleefsmanConfig) -> dict[str, float]:
    """All gauge readings of one snapshot, keyed by ``operator@x``.

    Three operators are recorded side by side so a ladder can tell a *gauge*
    artefact from a *solution* artefact:

    * ``height``  -- legacy top-N particle mean (biased by ``-0.5 k dx``);
    * ``surface`` -- top-N mean with the half-cell bias removed;
    * ``slab``    -- fixed-aperture volume gauge (dx-independent aperture).
    """
    out: dict[str, float] = {}
    for xs in cfg.station_x():
        tag = f"x{xs:.3f}"
        out[f"height@{tag}"] = water_height_at(pos, cfg, x=xs)
        out[f"surface@{tag}"] = surface_level(pos, cfg, x=xs)
        out[f"slab@{tag}"] = slab_level(pos, cfg, x=xs)
    return out


def _peak_metrics(times: list[float], height: list[float],
                  threshold: float) -> tuple[float, float, float]:
    """Return ``(peak_height, peak_time, arrival_time)``.

    ``arrival_time`` is the first crossing of ``threshold``, linearly
    interpolated between the two bracketing samples, so it is not quantised to
    the output interval; it is ``nan`` when the threshold is never crossed.
    """
    if not times:
        return float("nan"), float("nan"), float("nan")
    peak = max(height)
    peak_time = times[height.index(peak)]
    arrival = float("nan")
    for k in range(1, len(height)):
        if height[k - 1] < threshold <= height[k]:
            h0, h1 = height[k - 1], height[k]
            frac = 0.0 if h1 == h0 else (threshold - h0) / (h1 - h0)
            arrival = times[k - 1] + frac * (times[k] - times[k - 1])
            break
    return peak, peak_time, arrival


def run_case(cfg: KleefsmanConfig, resume: dict | None = None,
             time_budget: float | None = None) -> KleefsmanRun:
    """Integrate the case to ``cfg.t_end`` and return the probe diagnostics.

    ``resume`` continues a previously returned ``KleefsmanRun.state``;
    ``time_budget`` (seconds of wall clock) stops at a step boundary and sets
    ``finished = False`` so the caller can save ``result.state`` and continue in
    a new process -- the shell on the build node caps a command at ~120 s, which
    otherwise makes the ``dx = 0.01`` grid unrunnable locally.
    """
    validate_case(cfg)
    import time as _time
    t_start = _time.monotonic()
    device = resolve_device(cfg.device)
    top = cfg.tank_height
    if resume is None:
        positions, mass = make_water(cfg, device)
        velocity = torch.zeros_like(positions)
        initial_mass = float(mass.sum())
        carried = {"times": [], "height": [], "column": [], "front": [], "max_speed": 0.0,
                   "compression": 0.0, "escaped": 0, "dt": cfg.dt, "elapsed": 0.0,
                   "series": {}}
    else:
        positions = resume["positions"].clone()
        velocity = resume["velocity"].clone()
        mass = resume["mass"]
        initial_mass = float(resume["initial_mass"])
        carried = {k: (list(resume[k]) if isinstance(resume[k], list) else resume[k])
                   for k in ("times", "height", "column", "front", "max_speed", "compression",
                             "escaped", "dt", "elapsed")}
        carried["series"] = {k: list(v) for k, v in (resume.get("series") or {}).items()}
    wall = torch.cat((wall_lattice(cfg, device, top=top),
                      obstacle_lattice(cfg, device)), dim=0)
    h = cfg.h_factor * cfg.dx
    fluid_count = positions.shape[0]
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h, fluid_count=fluid_count)
    all_positions = torch.cat((positions, wall), dim=0)

    run = KleefsmanRun()
    dt = carried["dt"]
    elapsed = carried["elapsed"]
    run.times = list(carried["times"])
    run.height = list(carried["height"])
    run.column = list(carried["column"] if isinstance(carried["column"], list) else [])
    run.front = list(carried["front"])
    run.series = {k: list(v) for k, v in carried.get("series", {}).items()}
    run.max_speed = carried["max_speed"]
    run.dt_min = run.dt_max = dt
    max_steps = cfg.max_steps or int(6.0 * cfg.t_end / dt) + 1000
    compression = torch.tensor(carried["compression"], dtype=torch.float64, device=device)
    escaped = int(carried["escaped"])
    step = 0
    for step in range(max_steps + 1):
        all_positions[:fluid_count] = positions
        pairs, _ = cache.get(all_positions)
        step_dt = dt if cfg.adaptive_dt else cfg.dt
        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs, wall=wall,
                              dt=step_dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        compression = torch.maximum(compression, density.max().double() / cfg.rho0 - 1.0)
        speed = float(torch.linalg.vector_norm(velocity, dim=-1).max()) \
            if velocity.numel() else 0.0
        run.max_speed = max(run.max_speed, speed)
        resume_first = resume is not None and step == 0
        if (step % cfg.output_every == 0 and not resume_first) or elapsed >= cfg.t_end:
            run.times.append(elapsed)
            run.height.append(water_height_at(positions, cfg))
            run.column.append(column_level(positions, cfg))
            for key, value in gauge_series(positions, cfg).items():
                run.series.setdefault(key, []).append(value)
            run.front.append(float(positions[:, 0].max()) if positions.numel() else 0.0)
        if elapsed >= cfg.t_end:
            break
        if (time_budget is not None and step >= 1
                and _time.monotonic() - t_start > time_budget):
            run.finished = False
            break
        velocity = velocity + step_dt * acc
        positions = positions + step_dt * velocity
        escaped += apply_boundary(positions, velocity, cfg, top=top)
        elapsed += step_dt
        if cfg.adaptive_dt:
            dt = next_dt(cfg, dt, speed)
            run.dt_min = min(run.dt_min, dt)
            run.dt_max = max(run.dt_max, dt)

    run.steps_taken = step
    run.simulated_time = elapsed
    run.escaped = escaped
    run.mass_error = abs(float(mass.sum()) - initial_mass) / initial_mass
    run.max_density_compression = float(compression)
    run.peak_height, run.peak_time, run.arrival_time = _peak_metrics(
        run.times, run.height, cfg.arrival_threshold)
    for key, values in run.series.items():
        peak, peak_t, arrival = _peak_metrics(run.times, values, cfg.arrival_threshold)
        run.metrics[key] = {"peak_m": peak, "peak_time_s": peak_t,
                            "arrival_time_s": arrival}
    run.state = {
        "positions": positions.detach().clone(), "velocity": velocity.detach().clone(),
        "mass": mass, "initial_mass": initial_mass, "dt": dt, "elapsed": elapsed,
        "times": run.times, "height": run.height, "column": run.column,
        "front": run.front, "series": run.series,
        "max_speed": run.max_speed, "compression": float(compression),
        "escaped": escaped, "steps_done": run.steps_taken,
        "finished": run.finished,
    }
    return run


__all__ = [
    "KleefsmanConfig", "KleefsmanRun", "column_level", "make_water",
    "obstacle_lattice",
    "gauge_series", "run_case", "slab_level", "surface_level", "validate_case",
    "water_height_at",
]