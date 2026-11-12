"""Bow ice-breaking transient free surface: a minimal, verified SPH capability face.

Physics background
------------------
When a ship's stem breaks a level ice sheet, the water trapped between the bow
and the floe is expelled as a short, violent free-surface event: a bore that
climbs the ice/bow ramp, a jet that leaves the crest, and — where the deck edge
sits above the waterline — an aerated splash whose throw and apex decide whether
the forecastle deck gets wetted ("甲板上浪") and how far broken ice is tossed
("碎冰抛掷"). The event lasts O(0.1-0.5 s) and is dominated by inertia and
gravity, not by viscosity or surface tension, which is why a weakly
compressible SPH (WCSPH) model with a rigid bed is the right minimal tool.

Geometry (2-D, per unit width, consistent with the rest of TensorSPH)
--------------------------------------------------------------------
A tank floor ``[0, pool_width] x y=0``. A water column ``[0, column_width] x
[0, column_height]`` rests on the floor and is released at ``t = 0`` (dam-break
release, the classic way of producing a bore of known height). From
``x = sheet_start`` the bed becomes the rigid ice sheet: a ramp of inclination
``sheet_angle_deg`` rising to ``deck_height`` and then a flat deck that runs to
the far wall. The deck plateau is the abstraction of the ice-sheet top /
forecastle deck. Setting ``sheet_start >= pool_width`` removes the ice bed and
recovers the flat-bed dam break, which is what the Ritter benchmark anchor
below is run on.

What this module *verifies* (the minimal capability face)
---------------------------------------------------------
1. ``ritter_front`` / Martin & Moyce front timing of the first wave on the flat
   bed (standard dam-break benchmark; see :func:`front_timing_error`).
2. Total mechanical energy drift of the event window (before any domain-boundary
   contact), :meth:`IceSplashRun.energy_drift_peak`.
3. Bit-for-bit reproducibility from ``cfg.seed``.
4. Ballistic self-consistency of the splash jet: for every water particle that
   leaves the deck level upward, the apex it actually reaches is compared with
   ``y0 + v_y^2 / (2 g)`` (:meth:`IceSplashRun.ballistic_error`).
5. Deck-inundation and splash gauges (volume per unit width, apex, throw), which
   are the quantitative hooks for the deck-wetting spectrum.
6. A particle-initialisation exchange format for the DEM channel scenario
   (:func:`build_dem_exchange` / :func:`validate_dem_exchange`).

What this module explicitly does *not* do
-----------------------------------------
The floe launch estimate is a **reduced model**: floes are not resolved as
rigid bodies inside the SPH run. The water-side momentum flux onto a virtual
force plate at the floe location is measured on the resolved field
(:func:`floe_force_plate`), integrated in time into an impulse, and converted to
a launch velocity by ``v = J / (rho_ice pi r^2)``. That is an order-of-magnitude
estimate (rigid-body coupling, drag and rotation are absent); :func:`floe_launch_estimate`
also returns a rigorous kinetic-energy upper bound so the estimate comes with a
stated interval. Turning the qualitative spectral event into engineering
quantities is deliberately left to a later wave.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .dam_break import DamBreakConfig, make_column, ritter_front
from .droplet_impact import (
    apply_boundary,
    boundary_box,
    next_dt,
    rhs,
)
from .neighbor_search import PairListCache

__all__ = [
    "DEM_SCHEMA",
    "IceSplashConfig",
    "IceSplashRun",
    "ballistic_apex",
    "bed_height",
    "ballistic_range",
    "build_dem_exchange",
    "deck_start_x",
    "eos_internal_energy",
    "floe_force_plate",
    "floe_launch_estimate",
    "front_coefficient",
    "front_timing_error",
    "ice_bed_lattice",
    "ice_surface",
    "read_dem_exchange",
    "run_case",
    "validate_config",
    "validate_dem_exchange",
    "write_dem_exchange",
]

DEM_SCHEMA = "tensorsph.dem-exchange/1.0"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class IceSplashConfig(DamBreakConfig):
    """Bow ice-splash case parameters.

    Inherits the WCSPH parameters (``dx``, ``h_factor``, ``rho0``,
    ``sound_speed``, ``viscosity``, ``dt``, ...) from
    :class:`tensorsph.droplet_impact.Config` and the dam-break water column
    (``column_width``, ``column_height``, ``pool_width``, ``tank_height``) from
    :class:`tensorsph.dam_break.DamBreakConfig`.
    """

    # Ice bed: ramp of inclination ``sheet_angle_deg`` from ``sheet_start`` up
    # to ``deck_height``, then a flat deck to the far wall.
    sheet_start: float = 0.20
    sheet_angle_deg: float = 25.0
    deck_height: float = 0.28
    # Floe layout used by the impulse probe (reduced model, not resolved).
    n_floes: int = 5
    floe_radius: float = 0.030
    floe_rho: float = 917.0
    # Added-mass coefficient of the floating section used by the reduced launch
    # model. A body accelerated in water must accelerate the displaced water
    # too, so the impulse is shared with ``C_m rho_w pi r^2`` of "entrained"
    # inertia. For a 2-D circular section in an unbounded inviscid fluid the
    # added mass equals the displaced mass, ``C_m = 1`` (Lamb, *Hydrodynamics*,
    # §71). ``0`` recovers the old dry-mass (upper-bound) estimate.
    floe_added_mass_coefficient: float = 1.0
    floe_start: float | None = None
    floe_spacing: float | None = None
    jitter: float = 0.25           # floe-placement jitter, in units of dx
    seed: int = 0
    # Compression cap used when integrating the elastic energy of the Tait
    # liquid. The physical compression of a WCSPH run at Mach <= 0.15 is a few
    # per cent; a single particle wedged in a wall corner can reach tens of per
    # cent and would otherwise dominate the whole budget through ``e ~ B d^2``.
    # The cap is a quadrature guard, not a physics parameter, and the unclipped
    # value is reported next to it in ``IceSplashRun.internal_energy_unclipped``.
    compression_clip: float = 0.10


def ice_surface(x: float, cfg: IceSplashConfig) -> float:
    """Height of the rigid bed at ``x``: floor, then ramp, then deck plateau."""
    if x < cfg.sheet_start:
        return 0.0
    slope = math.tan(math.radians(cfg.sheet_angle_deg))
    return min(cfg.deck_height, (x - cfg.sheet_start) * slope)


def bed_height(x: torch.Tensor, cfg: IceSplashConfig) -> torch.Tensor:
    """Vectorised ``ice_surface`` for the gauge pass."""
    slope = math.tan(math.radians(cfg.sheet_angle_deg))
    rise = (x - cfg.sheet_start) * slope
    return torch.where(x >= cfg.sheet_start, rise.clamp(max=cfg.deck_height),
                       torch.zeros_like(x))


def deck_start_x(cfg: IceSplashConfig) -> float:
    """``x`` at which the ramp reaches ``deck_height`` (start of the plateau)."""
    return cfg.sheet_start + cfg.deck_height / math.tan(math.radians(cfg.sheet_angle_deg))


def validate_config(cfg: IceSplashConfig) -> None:
    """Reject configurations whose geometry or discretisation is inconsistent."""
    if cfg.dx <= 0 or cfg.h_factor <= 0 or cfg.dt <= 0:
        raise ValueError("dx, h_factor and dt must be positive")
    if cfg.column_width <= 0 or cfg.column_height <= 0:
        raise ValueError("water column dimensions must be positive")
    if cfg.pool_width <= cfg.column_width:
        raise ValueError("pool_width must exceed column_width")
    if not 0.0 < cfg.sheet_angle_deg <= 89.0:
        raise ValueError("sheet_angle_deg must lie in (0, 89] degrees")
    if cfg.deck_height <= 0:
        raise ValueError("deck_height must be positive")
    if cfg.deck_height + cfg.dx >= cfg.tank_height:
        raise ValueError(
            "tank_height must clear deck_height by at least one spacing so the "
            "splash is not clipped by the domain top")
    if cfg.sheet_start >= cfg.pool_width:
        # Degenerate case: no ice bed at all (flat-bed dam-break anchor).
        pass
    elif cfg.sheet_start < cfg.column_width - 1.0e-12:
        raise ValueError("sheet_start must not overlap the water column")
    if cfg.n_floes < 0:
        raise ValueError("n_floes must be non-negative")
    if cfg.n_floes > 0:
        if cfg.floe_radius <= 0 or cfg.floe_rho <= 0:
            raise ValueError("floe_radius and floe_rho must be positive")
        if cfg.floe_radius >= 0.5 * cfg.deck_height:
            raise ValueError("floe_radius is too large for the deck height")
        if not 0.0 <= cfg.floe_added_mass_coefficient <= 2.0:
            raise ValueError("floe_added_mass_coefficient must lie in [0, 2]")
        if cfg.floe_start is not None:
            extent = (cfg.floe_start + (cfg.n_floes - 1) * (cfg.floe_spacing or 0.0)
                      + cfg.floe_radius + cfg.jitter * cfg.dx)
            if extent > cfg.pool_width - cfg.dx:
                raise ValueError("the floe layout does not fit on the deck")
    if cfg.seed < 0:
        raise ValueError("seed must be non-negative")
    courant = cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx)
    if courant > cfg.max_courant * (1.0 + 1.0e-9):
        raise ValueError(
            f"dt={cfg.dt:g} gives acoustic Courant {courant:.3f} > max_courant "
            f"{cfg.max_courant:g}; use dt <= {cfg.dt_from_courant():.3e} s")
    if cfg.boundary != "wall":
        raise ValueError("the ice-splash case integrates static wall particles only")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _surface_points(cfg: IceSplashConfig, device: torch.device):
    """Bed surface samples as ``(x, y, nx, ny)`` with the normal into the fluid.

    Particle placement follows ``wall_lattice``: ``(k + 1/2) dx`` behind the
    surface along ``-n``, so the outermost layer sits half a spacing outside the
    fluid domain and the water lattice keeps its full kernel support.
    """
    dx = cfg.dx
    pts: list[tuple[float, float, float, float]] = []
    sheet = cfg.sheet_start if cfg.sheet_start < cfg.pool_width else cfg.pool_width
    # Floor of the tank, across the full width: it closes the wedge under the
    # ramp so no water can tunnel into the solid through the toe corner.
    n_floor = int(math.floor((cfg.pool_width - 0.5 * dx) / dx)) + 1
    for k in range(max(n_floor, 0)):
        pts.append((0.5 * dx + k * dx, 0.0, 0.0, 1.0))
    # Vertical end walls of the tank: the dam-break wave also runs backwards and
    # the splash eventually reaches the far end. Without them the clamp in
    # ``apply_boundary`` would silently absorb momentum and energy.
    n_wall = int(math.floor((cfg.tank_height - dx) / dx)) + 1
    for k in range(max(n_wall, 0)):
        y = 0.5 * dx + k * dx
        pts.append((0.0, y, 1.0, 0.0))
        pts.append((cfg.pool_width, y, -1.0, 0.0))
    if sheet >= cfg.pool_width:
        return pts
    # Ice ramp, sampled along its arc length, up to the deck plateau.
    theta = math.radians(cfg.sheet_angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    length = cfg.deck_height / sin_t
    n_ramp = int(math.floor((length - 0.5 * dx) / dx)) + 1
    for k in range(max(n_ramp, 0)):
        s = min(0.5 * dx + k * dx, length)
        pts.append((sheet + s * cos_t, s * sin_t, -sin_t, cos_t))
    # Deck plateau.
    start = deck_start_x(cfg)
    n_deck = int(math.floor((cfg.pool_width - start - 0.5 * dx) / dx)) + 1
    for k in range(max(n_deck, 0)):
        pts.append((start + 0.5 * dx + k * dx, cfg.deck_height, 0.0, 1.0))
    return pts


def ice_bed_lattice(cfg: IceSplashConfig, device: torch.device) -> torch.Tensor:
    """Static wall particles of the tank floor and the rigid ice bed.

    Particles are placed ``(k + 1/2) dx`` *inside* the bed along its outward
    normal, so the water sees a smooth surface one half spacing away from the
    sampled line (the same convention as ``wall_lattice``).
    """
    pts = _surface_points(cfg, device)
    if not pts:
        raise ValueError("ice bed produced no particles")
    columns = []
    layers = max(int(cfg.wall_layers), 1)
    for x, y, nx, ny in pts:
        for k in range(layers):
            off = -(k + 0.5) * cfg.dx
            columns.append((x + off * nx, y + off * ny))
    out = torch.tensor(columns, dtype=torch.float64, device=device)
    # The ramp toe pads can dip below the tank floor; drop those (the floor
    # particles already cover that solid) so no mass is double counted.
    keep = out[:, 1] >= -0.5 * cfg.dx
    return out[keep]


# --------------------------------------------------------------------------- #
# Reduced floe model
# --------------------------------------------------------------------------- #

def floe_layout(cfg: IceSplashConfig, device: torch.device) -> torch.Tensor:
    """Floe centres on the deck, placed from ``cfg.seed``.

    The layout is drawn from a private :class:`torch.Generator` so that it does
    not depend on how much of the global RNG stream the rest of the run has
    consumed: same seed -> same layout, bit for bit.
    """
    if cfg.n_floes <= 0:
        return torch.zeros((0, 2), dtype=torch.float64, device=device)
    radius = cfg.floe_radius
    x0 = (deck_start_x(cfg) + 2.0 * cfg.dx + radius
          if cfg.floe_start is None else cfg.floe_start)
    jitter = cfg.jitter * cfg.dx
    available = cfg.pool_width - cfg.dx - x0 - radius - jitter
    if cfg.floe_spacing is not None:
        spacing = cfg.floe_spacing
    elif cfg.n_floes > 1:
        spacing = min(2.5 * radius, max(available, 0.0) / (cfg.n_floes - 1))
    else:
        spacing = 0.0
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    jx = (torch.rand(cfg.n_floes, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * jitter
    jy = (torch.rand(cfg.n_floes, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * jitter
    xs = x0 + spacing * torch.arange(cfg.n_floes, dtype=torch.float64) + jx
    ys = cfg.deck_height + radius + jy
    xs = xs.clamp(min=cfg.dx, max=cfg.pool_width - cfg.dx)
    return torch.stack((xs, ys), dim=-1).to(device)


def floe_force_plate(positions: torch.Tensor, velocity: torch.Tensor,
                     mass: torch.Tensor, cfg: IceSplashConfig,
                     floes: torch.Tensor) -> torch.Tensor:
    """Momentum flux ``rho u_x^2 h_wet`` onto a thin plate just upstream of each floe.

    The probe is a slab ``1.5 dx`` thick in front of the floe. ``mass`` is a
    2-D mass per unit depth (kg/m), so ``sum m_i u_x^2 / T`` over the slab and
    the floe height ``h_wet`` equals ``rho u_x^2 h_wet`` -- the force per unit
    depth a rigid floe face would feel -- provided ``T`` is the *effective*
    thickness of the sampled slab. On a lattice that slab holds
    ``round(1.5) = 2`` particle columns, so ``T = 2 dx``; using ``dx`` double
    counts the flux and using ``1.5 dx`` over-counts it by 4/3 (both were
    measured by ``test_force_plate_reproduces_the_momentum_flux_of_a_uniform_jet``
    before this note existed). Only the upstream half of the floe height
    (``|y - y_c| < r``) is counted, and only positive downstream momentum.
    """
    if floes.numel() == 0 or positions.numel() == 0:
        return torch.zeros((floes.shape[0],), dtype=positions.dtype, device=positions.device)
    cy = floes[:, 1:2]
    cx = floes[:, 0:1]
    r = cfg.floe_radius
    thickness = 1.5 * cfg.dx
    n_columns = max(1, int(round(thickness / cfg.dx)))
    slab = (positions[None, :, 0] >= cx - thickness) & (positions[None, :, 0] <= cx)
    band = (positions[None, :, 1] - cy).abs() < r
    ux = velocity[:, 0].clamp(min=0.0)
    flux = mass * ux.square() / (n_columns * cfg.dx)
    return (flux[None, :] * (slab & band).to(positions.dtype)).sum(dim=1)


def _floe_mass_per_depth(cfg: IceSplashConfig) -> float:
    return cfg.floe_rho * math.pi * cfg.floe_radius ** 2


def _floe_induced_mass_per_depth(cfg: IceSplashConfig) -> float:
    """Water inertia dragged along by the floe, ``C_m rho_w pi r^2``."""
    return (cfg.floe_added_mass_coefficient * cfg.rho0
            * math.pi * cfg.floe_radius ** 2)


def ballistic_apex(v_up: float, gravity: float) -> float:
    """Rise of a projectile launched vertically with speed ``v_up``."""
    return 0.5 * v_up * v_up / gravity


def ballistic_range(speed: float, angle_deg: float, gravity: float) -> float:
    """Horizontal range of a projectile over a level surface."""
    return speed * speed * math.sin(2.0 * math.radians(angle_deg)) / gravity


def eos_internal_energy(rho: torch.Tensor, cfg: IceSplashConfig) -> torch.Tensor:
    """Specific internal (elastic) energy of the Tait liquid, ``int p drho / rho^2``.

    For ``p = B[(rho/rho0)^gamma - 1]`` the integral from ``rho0`` to ``rho`` is

        e = B/rho0 * [ ((rho/rho0)^(gamma-1) - 1)/(gamma-1) + rho0/rho - 1 ],

    which is ``O(delta^2)`` in the compression ``delta = rho/rho0 - 1`` and
    vanishes at rest. The naive ``p/((gamma-1) rho)`` is ``O(delta)`` and about
    an order of magnitude too large here (``delta < 4%``), which is why the
    elastic reservoir has to be integrated properly before the energy budget of
    a weakly compressible run can be closed.

    Units: J/kg. Multiply by the (per unit depth) mass and sum for J/m.
    """
    b_const = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    ratio = rho / cfg.rho0
    energy = (b_const / cfg.rho0) * (
        (ratio ** (cfg.gamma - 1.0) - 1.0) / (cfg.gamma - 1.0) + 1.0 / ratio - 1.0
    )
    # The scheme's pressure law is clamped at zero (``tait_pressure``), i.e. the
    # liquid is not allowed to pull. The energy of that *clamped* equation of
    # state therefore vanishes on the tension side: the free-surface density
    # deficit (``rho < rho0``) is a truncation artefact, not stored energy, and
    # integrating the unclamped parabola there injects a spurious reservoir that
    # is an order of magnitude larger than the real one.
    return torch.where(rho > cfg.rho0, energy, torch.zeros_like(energy))


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

@dataclass
class IceSplashRun:
    """Gauges of one bow ice-splash event."""

    times: list[float]
    front: list[float]
    deck_volume: list[float]          # m^3 per metre width on/above the deck
    apex: list[float]                 # highest water particle, m
    throw: list[float]                # furthest water particle past the deck edge, m
    kinetic_energy: list[float]
    potential_energy: list[float]
    internal_energy: list[float]        # elastic (WCSPH) energy, compression capped
    internal_energy_unclipped: list[float]  # same without the quadrature guard
    initial_mass: float
    final_mass: float
    escaped: int
    first_boundary_contact: float | None
    fragmentation_time: float | None   # first airborne particle with <=1 fluid neighbour
    deck_arrival_time: float | None
    crest_arrival_time: float | None
    peak_deck_volume: float
    peak_apex: float
    peak_throw: float
    simulated_time: float
    steps_taken: int
    dt_min: float
    dt_max: float
    floe_impulse: list[float]
    floe_peak_force: list[float]
    peak_splash_kinetic_energy: float
    peak_compression: float
    apex_predicted: list[float]       # ballistic prediction, per tracked particle
    apex_measured: list[float]
    final_positions: torch.Tensor
    final_velocity: torch.Tensor
    benchmark: dict
    snapshots: list = field(default_factory=list)

    # -- energy ---------------------------------------------------------- #
    def energy(self, include_internal: bool = False) -> list[float]:
        """Total energy per unit width.

        ``KE + PE`` by default. With ``include_internal`` the elastic energy of
        the Tait liquid, ``sum m e(rho)`` with ``e`` from
        :func:`eos_internal_energy`, is added: that is where the compressible
        part of the pressure work goes, so the sum is the full budget of a
        weakly compressible run and not just its mechanical half.
        """
        if include_internal:
            return [k + p + u for k, p, u in
                    zip(self.kinetic_energy, self.potential_energy, self.internal_energy)]
        return [k + p for k, p in zip(self.kinetic_energy, self.potential_energy)]

    def energy_drift(self, t_end: float | None = None,
                     include_internal: bool = False) -> list[float]:
        """Relative drift of the total energy against its initial value."""
        energy = self.energy(include_internal=include_internal)
        e0 = energy[0]
        if e0 == 0.0:
            return [0.0 for _ in energy]
        rows = [(t, (e - e0) / abs(e0))
                for t, e in zip(self.times, energy)
                if t_end is None or t <= t_end + 1.0e-12]
        return [row[1] for row in rows]

    def energy_drift_peak(self, t_end: float | None = None,
                          include_internal: bool = False) -> float:
        drift = self.energy_drift(t_end, include_internal=include_internal)
        return max((abs(d) for d in drift), default=0.0)

    def energy_window_end(self) -> float:
        """End of the event window used for the energy criterion.

        The window runs from the release to the first arrival of water at the
        deck: bore generation and run-up, in which the flow is driven by
        gravity and pressure against the rigid bed only. Past the deck arrival
        the splash breaks into fragments whose support is truncated, and the
        mechanical budget stops being a sharp instrument; the ballistic apex
        check (``ballistic_error``) takes over there. Disclosed, not hidden:
        ``energy_drift_full`` also reports the whole-run number.
        """
        if self.deck_arrival_time is not None:
            return min(self.deck_arrival_time, self.simulated_time)
        return self.simulated_time

    # -- splash ----------------------------------------------------------- #
    def ballistic_error(self, min_speed: float = 0.05) -> dict:
        """Median/percentile relative error of the measured splash apex.

        Only particles that left the deck level upward with
        ``v_y > min_speed`` are graded: below that the ballistic prediction is
        noise-dominated (apex rise < 1 mm at v_y = 0.05 m/s).
        """
        if not self.apex_predicted:
            return {"count": 0, "median_rel_error": None, "p90_rel_error": None}
        rows = []
        for pred, meas in zip(self.apex_predicted, self.apex_measured):
            if pred is None or meas is None:
                continue
            rise = pred - self._launch_level
            if rise <= 1.0e-6 or pred <= 0.0:
                continue
            rows.append(abs(meas - pred) / pred)
        if not rows:
            return {"count": 0, "median_rel_error": None, "p90_rel_error": None}
        rows.sort()
        k90 = min(len(rows) - 1, int(math.ceil(0.9 * len(rows))) - 1)
        return {"count": len(rows), "median_rel_error": rows[len(rows) // 2],
                "p90_rel_error": rows[k90], "min_speed": min_speed}

    _launch_level: float = 0.0

    def summary(self) -> dict:
        window = self.energy_window_end()
        return {
            "simulated_time_s": self.simulated_time,
            "steps": self.steps_taken,
            "dt_min_s": self.dt_min,
            "dt_max_s": self.dt_max,
            "deck_arrival_time_s": self.deck_arrival_time,
            "crest_arrival_time_s": self.crest_arrival_time,
            "first_boundary_contact_s": self.first_boundary_contact,
            "fragmentation_time_s": self.fragmentation_time,
            "energy_window_end_s": window,
            "energy_drift_window": self.energy_drift_peak(window),
            "energy_drift_window_with_internal": self.energy_drift_peak(
                window, include_internal=True),
            "energy_drift_full": self.energy_drift_peak(),
            "energy_drift_full_with_internal": self.energy_drift_peak(
                include_internal=True),
            "peak_deck_volume_m3_per_m": self.peak_deck_volume,
            "peak_apex_m": self.peak_apex,
            "peak_throw_m": self.peak_throw,
            "peak_splash_ke_J_per_m": self.peak_splash_kinetic_energy,
            "peak_compression": self.peak_compression,
            "initial_mass_kg_per_m": self.initial_mass,
            "final_mass_kg_per_m": self.final_mass,
            "escaped": self.escaped,
            "ballistic": self.ballistic_error(),
            "benchmark": dict(self.benchmark),
        }


def front_timing_error(run: IceSplashRun, cfg: IceSplashConfig,
                       t_end: float | None = None) -> dict:
    """Relative error of the first-wave front timing against the two anchors.

    * ``ritter``: analytic dry-bed front ``x = x_dam + 2 sqrt(g h0) t``.
    * ``front_speed``: least-squares slope of the front trajectory, compared
      with Ritter's ``2 sqrt(g h0)``.

    Only samples up to ``t_end`` (default: the event window end) are used, so a
    later wall impact cannot pollute the first-wave comparison.
    """
    limit = run.energy_window_end() if t_end is None else t_end
    times, front = [], []
    for t, f in zip(run.times, run.front):
        if t <= limit + 1.0e-12:
            times.append(t)
            front.append(f)
    if len(times) < 3:
        return {"points": len(times), "ritter_rel_error": None, "front_speed": None,
                "ritter_speed": None, "speed_ratio": None}
    dam = cfg.column_width
    c_ritter = 2.0 * math.sqrt(cfg.gravity * cfg.column_height)
    num = den = 0.0
    for t, f in zip(times, front):
        ref = dam + ritter_front(t, cfg)
        num += (f - ref) ** 2
        den += ref ** 2
    l2 = math.sqrt(num / den) if den > 0 else None
    # Front speed by least squares on (t, x - x_dam), skipping the first sample
    # (the rupture transient) and any samples pinned at the far wall.
    pts = [(t, f - dam) for t, f in list(zip(times, front))[1:]
           if f < cfg.pool_width - 0.5 * cfg.dx]
    speed = None
    if len(pts) >= 2:
        st = sum(t for t, _ in pts)
        sx = sum(x for _, x in pts)
        stt = sum(t * t for t, _ in pts)
        stx = sum(t * x for t, x in pts)
        denom = len(pts) * stt - st * st
        speed = (len(pts) * stx - st * sx) / denom if denom != 0 else None
    return {
        "points": len(times),
        "t_end": limit,
        "ritter_rel_error": l2,
        "ritter_speed": c_ritter,
        "front_speed": speed,
        "speed_ratio": (speed / c_ritter) if speed is not None and c_ritter else None,
    }


def front_coefficient(run: IceSplashRun, cfg: IceSplashConfig,
                      tau_window: tuple[float, float] = (0.5, 4.0)) -> dict:
    """Reference-free first-wave front coefficient ``c = d(Z/a)/d tau``.

    This is the metric the platform's dam-break audit settled on after the
    bundled Martin & Moyce table was found unusable
    (``docs/benchmarks/dam-break-reference-audit.md``, section 7): the front is
    fitted over ``tau`` in ``[0.5, 4.0]`` to skip the rupture transient, with
    ``tau = t sqrt(2 g / a)`` and ``Z`` measured from the dam face. Ritter's
    dry-bed solution gives ``c = 2``; the verified solver value for the flat
    bed at ``dx = 0.02`` with free slip and ``viscosity = 0.01`` is
    ``c = 1.2468``.

    Samples pinned at the far wall are dropped: once the surge piles up there
    the leading edge no longer measures the wave.
    """
    a = cfg.column_width
    scale = math.sqrt(2.0 * cfg.gravity / a)
    rows = []
    for t, x in zip(run.times, run.front):
        tau = t * scale
        if not (tau_window[0] <= tau <= tau_window[1]):
            continue
        if x >= cfg.pool_width - 1.5 * cfg.dx:
            continue
        rows.append((tau, x / a - 1.0))
    if len(rows) < 3:
        return {"c": None, "points": len(rows), "tau_window": list(tau_window)}
    n = len(rows)
    st = sum(r[0] for r in rows)
    sz = sum(r[1] for r in rows)
    stt = sum(r[0] * r[0] for r in rows)
    stz = sum(r[0] * r[1] for r in rows)
    denom = n * stt - st * st
    c = (n * stz - st * sz) / denom if denom else None
    return {"c": c, "points": n, "tau_window": list(tau_window),
            "tau_fit": [rows[0][0], rows[-1][0]], "ritter_c": 2.0}


def floe_launch_estimate(run: IceSplashRun, cfg: IceSplashConfig,
                         floes: torch.Tensor) -> dict:
    """Order-of-magnitude launch estimate for the floes (reduced model).

    Two impulse-transfer readings are reported, because the divisor is a
    modelling choice rather than a detail:

    * ``launch_speed_m_s = J / (rho_ice pi r^2)`` -- **dry** mass. This is an
      *upper bound*: it assumes the water the floe pushes aside costs nothing.
    * ``launch_speed_with_added_mass_m_s = J / (rho_ice pi r^2 + C_m rho_w pi r^2)``
      -- the added inertia of the entrained water, ``C_m =
      ``cfg.floe_added_mass_coefficient`` (default 1, the exact value for a 2-D
      circular section in an unbounded fluid, Lamb §71).

    Upper bound on the speed: all the splash kinetic energy above the deck into
    one floe, ``v <= sqrt(2 KE_deck / m_floe)``. ``floe_added_mass.py`` in the
    benchmark bundle quantifies the dry-mass bias; the spread is reported, not
    hidden.
    """
    m_floe = _floe_mass_per_depth(cfg)
    m_added = _floe_induced_mass_per_depth(cfg)
    m_wet = m_floe + m_added
    ratio = m_floe / m_wet if m_wet > 0 else 0.0
    rows = []
    v_upper = (math.sqrt(2.0 * run.peak_splash_kinetic_energy / m_floe)
               if run.peak_splash_kinetic_energy > 0.0 else 0.0)
    for k, impulse in enumerate(run.floe_impulse):
        v = impulse / m_floe
        v_wet = impulse / m_wet if m_wet > 0 else 0.0
        centre = floes[k].tolist() if floes.numel() else None
        rows.append({
            "floe": k,
            "centre_m": centre,
            "impulse_Ns_per_m": impulse,
            "peak_force_N_per_m": run.floe_peak_force[k],
            "launch_speed_m_s": v,
            "ballistic_apex_m": ballistic_apex(v, cfg.gravity),
            "ballistic_range_m": ballistic_range(v, 45.0, cfg.gravity),
            "mass_kg_per_m": m_floe,
            "added_mass_kg_per_m": m_added,
            "speed_ratio_with_added_mass": ratio,
            "launch_speed_with_added_mass_m_s": v_wet,
            "ballistic_apex_with_added_mass_m": ballistic_apex(v_wet, cfg.gravity),
            "ballistic_range_with_added_mass_m": ballistic_range(v_wet, 45.0, cfg.gravity),
        })
    return {
        "model": "impulse transfer on a virtual force plate (reduced model)",
        "floes": rows,
        "upper_bound_speed_m_s": v_upper,
        "upper_bound_apex_m": ballistic_apex(v_upper, cfg.gravity),
        "added_mass_coefficient": cfg.floe_added_mass_coefficient,
        "dry_mass_speed_ratio": ratio,
        "level": ("order-of-magnitude; rigid-body coupling not resolved; "
                  "launch_speed_m_s is a dry-mass upper bound, "
                  "launch_speed_with_added_mass_m_s is the recommended reading"),
    }


def _floe_launch_speed(impulse: float, cfg: IceSplashConfig) -> float:
    return impulse / _floe_mass_per_depth(cfg)


def run_case(cfg: IceSplashConfig, device: torch.device | None = None,
             snapshot_times=None) -> IceSplashRun:
    """Integrate the bow ice-splash event and return its gauges.

    ``snapshot_times`` (seconds, increasing) captures ``(t, position, velocity,
    rho, pressure)`` for the field figures; it does not perturb the trajectory.
    """
    validate_config(cfg)
    if device is None:
        device = torch.device(cfg.device)
    torch.manual_seed(int(cfg.seed))

    positions, mass = make_column(cfg, device)
    bed = ice_bed_lattice(cfg, device).to(device=device, dtype=positions.dtype)
    velocity = torch.zeros_like(positions)
    initial_mass = float(mass.sum())

    h = cfg.h_factor * cfg.dx
    fluid_count = int(positions.shape[0])
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h, fluid_count=fluid_count)
    all_positions = torch.cat((positions, bed), dim=0)

    floes = floe_layout(cfg, device)
    n_floes = int(floes.shape[0])
    impulse = torch.zeros(n_floes, dtype=torch.float64, device=device)
    peak_force = torch.zeros(n_floes, dtype=torch.float64, device=device)

    deck_y = cfg.deck_height
    deck_x = deck_start_x(cfg)
    deck_tol = 0.5 * cfg.dx
    inverse_two_g = 1.0 / (2.0 * cfg.gravity)

    apex_pred = torch.full((fluid_count,), float("nan"), dtype=torch.float64, device=device)
    apex_meas = torch.full((fluid_count,), float("nan"), dtype=torch.float64, device=device)
    prev_above = positions[:, 1] > deck_y

    times: list[float] = []
    front: list[float] = []
    deck_volume: list[float] = []
    apex: list[float] = []
    throw: list[float] = []
    kinetic: list[float] = []
    potential: list[float] = []
    internal: list[float] = []
    internal_unc: list[float] = []
    peak_deck_volume = 0.0
    peak_apex = 0.0
    peak_throw = 0.0
    peak_splash_ke = 0.0
    peak_compression = 0.0
    deck_arrival: float | None = None
    crest_arrival: float | None = None
    first_contact: float | None = None
    fragment_time: float | None = None

    snap_targets = sorted(float(t) for t in (snapshot_times or []))
    snapshots: list = []
    snap_next = 0

    dt = cfg.dt
    escaped = 0
    elapsed = 0.0
    dt_min = dt_max = dt
    target_time = cfg.steps * cfg.dt
    tolerance = 1.0e-12 * max(1.0, target_time)
    max_steps = cfg.steps if not cfg.adaptive_dt else 4 * cfg.steps + 1000

    for step in range(max_steps + 1):
        all_positions[:fluid_count] = positions
        pairs, _ = cache.get(all_positions)
        step_dt = dt if not cfg.adaptive_dt else min(dt, target_time - elapsed)
        rho, pressure, acc = rhs(positions, velocity, mass, cfg, pairs=pairs,
                                 wall=bed, dt=step_dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")

        # -- floe force plate (reduced model) ------------------------------ #
        if n_floes:
            force = floe_force_plate(positions, velocity, mass, cfg, floes)
            impulse = impulse + force.double() * step_dt
            peak_force = torch.maximum(peak_force, force.double())

        # -- onset of discrete-fragment dynamics ---------------------------- #
        # Up to this point the water is a continuum with full support; past it
        # airborne particles with a handful of neighbours make the mechanical
        # budget a much blunter instrument. Recorded, not used to truncate a
        # criterion silently.
        if fragment_time is None and pairs.numel():
            # The pair list is unordered (i < j), so a particle's degree has to
            # be counted from both columns; counting only the first one would
            # give the highest-indexed particle degree 0 at every step.
            low = pairs[:, 0] < fluid_count
            high = pairs[:, 1] < fluid_count
            degree = torch.bincount(pairs[low][:, 0], minlength=fluid_count)
            if bool(high.any()):
                degree = degree + torch.bincount(pairs[high][:, 1], minlength=fluid_count)
            airborne = positions[:, 1] > bed_height(positions[:, 0], cfg) + 1.5 * cfg.dx
            if bool((airborne & (degree <= 1)).any()):
                fragment_time = elapsed

        # -- gauges --------------------------------------------------------- #
        on_deck = (positions[:, 0] > deck_x) & (positions[:, 1] > deck_y - deck_tol)
        above = positions[:, 1] > deck_y + deck_tol
        volume = float(on_deck.sum()) * cfg.dx * cfg.dx
        highest = float(positions[:, 1].max())
        farthest = float(positions[:, 0].max()) - deck_x
        if deck_arrival is None and bool(on_deck.any()):
            deck_arrival = elapsed
        if crest_arrival is None and bool((positions[:, 0] > deck_x).any()):
            crest_arrival = elapsed
        if above.any():
            mass_above = mass[above]
            vel_above = velocity[above]
            ke_above = float(0.5 * (mass_above * vel_above.square().sum(dim=1)).sum())
            peak_splash_ke = max(peak_splash_ke, ke_above)
        # Ballistic bookkeeping: a particle that crosses the deck level upward.
        crossing = above & (~prev_above) & (velocity[:, 1] > 0.0)
        if bool(crossing.any()):
            pred = positions[:, 1] + velocity[:, 1].square() * inverse_two_g
            apex_pred = torch.where(crossing, pred, apex_pred)
            apex_meas = torch.where(crossing, positions[:, 1], apex_meas)
        tracked = torch.isfinite(apex_pred)
        if bool(tracked.any()):
            apex_meas = torch.where(tracked, torch.maximum(apex_meas, positions[:, 1]), apex_meas)
        prev_above = above

        final = (step == cfg.steps if not cfg.adaptive_dt
                 else elapsed >= target_time - tolerance)
        if step % cfg.output_every == 0 or final:
            times.append(elapsed)
            front.append(float(positions[:, 0].max()))
            deck_volume.append(volume)
            apex.append(highest)
            throw.append(farthest)
            kinetic.append(float(0.5 * (mass[:, None] * velocity.square()).sum()))
            potential.append(float((mass * cfg.gravity * positions[:, 1]).sum()))
            internal.append(float((mass * eos_internal_energy(
                rho.clamp(max=(1.0 + cfg.compression_clip) * cfg.rho0), cfg)).sum()))
            internal_unc.append(float((mass * eos_internal_energy(rho, cfg)).sum()))
            peak_compression = max(peak_compression,
                                   float(rho.max().double() / cfg.rho0 - 1.0))
            while (snap_next < len(snap_targets)
                   and elapsed >= snap_targets[snap_next] - 1.0e-12):
                snapshots.append((elapsed, positions.detach().clone(),
                                  velocity.detach().clone(), rho.detach().clone(),
                                  pressure.detach().clone()))
                snap_next += 1
            peak_deck_volume = max(peak_deck_volume, volume)
            peak_apex = max(peak_apex, highest)
            peak_throw = max(peak_throw, farthest)

        if final:
            break
        velocity = velocity + step_dt * acc
        positions = positions + step_dt * velocity
        escaped_step = apply_boundary(positions, velocity, cfg, top=cfg.tank_top())
        escaped += escaped_step
        # Contact condition. ``apply_boundary`` zeroes the normal velocity when a
        # particle *overshoots* the box; a particle pushed exactly onto the face
        # then sits inside the tolerance and keeps accumulating normal momentum
        # against the clamp (measured: an unphysical 15 m/s after 0.2 s at the
        # floor corner). Zeroing the inward component on the face itself is the
        # boundary condition the clamp was meant to impose.
        x_min, x_max, y_min, y_max = boundary_box(cfg, cfg.tank_top())
        eps = 1.0e-9 * max(1.0, cfg.pool_width)
        on_floor = positions[:, 1] <= y_min + eps
        on_roof = positions[:, 1] >= y_max - eps
        on_left = positions[:, 0] <= x_min + eps
        on_right = positions[:, 0] >= x_max - eps
        velocity[on_floor, 1] = velocity[on_floor, 1].clamp(min=0.0)
        velocity[on_roof, 1] = velocity[on_roof, 1].clamp(max=0.0)
        velocity[on_left, 0] = velocity[on_left, 0].clamp(min=0.0)
        velocity[on_right, 0] = velocity[on_right, 0].clamp(max=0.0)
        if escaped_step and first_contact is None:
            first_contact = elapsed
        elapsed += step_dt
        if cfg.adaptive_dt:
            speed = float(torch.linalg.vector_norm(velocity, dim=-1).max())
            dt = next_dt(cfg, dt, speed)
            dt_min = min(dt_min, dt)
            dt_max = max(dt_max, dt)

    result = IceSplashRun(
        times=times, front=front, deck_volume=deck_volume, apex=apex, throw=throw,
        kinetic_energy=kinetic, potential_energy=potential, internal_energy=internal,
        internal_energy_unclipped=internal_unc,
        initial_mass=initial_mass, final_mass=float(mass.sum()),
        escaped=int(escaped), first_boundary_contact=first_contact,
        fragmentation_time=fragment_time,
        deck_arrival_time=deck_arrival, crest_arrival_time=crest_arrival,
        peak_deck_volume=peak_deck_volume, peak_apex=peak_apex, peak_throw=peak_throw,
        simulated_time=elapsed, steps_taken=step, dt_min=dt_min, dt_max=dt_max,
        floe_impulse=[float(v) for v in impulse],
        floe_peak_force=[float(v) for v in peak_force],
        peak_splash_kinetic_energy=peak_splash_ke, peak_compression=peak_compression,
        apex_predicted=[None if not math.isfinite(float(p)) else float(p) for p in apex_pred],
        apex_measured=[None if not math.isfinite(float(p)) else float(p) for p in apex_meas],
        final_positions=positions.detach().clone(),
        final_velocity=velocity.detach().clone(),
        benchmark={}, snapshots=snapshots,
    )
    result._launch_level = deck_y
    result.benchmark = {
        "ritter": front_timing_error(result, cfg),
        "front_coefficient": front_coefficient(result, cfg),
    }
    return result


# --------------------------------------------------------------------------- #
# DEM exchange format
# --------------------------------------------------------------------------- #

def build_dem_exchange(run: IceSplashRun, cfg: IceSplashConfig,
                       floes: torch.Tensor | None = None) -> dict:
    """Particle-initialisation payload for the DEM channel scenario.

    The DEM side receives the SPH end state (water particles with positions,
    velocities, masses and radii) plus the reduced-model floe launch estimate,
    so that a channel-scale DEM run can be seeded from a resolved bow event.
    """
    pos = run.final_positions.detach().cpu()
    vel = run.final_velocity.detach().cpu()
    dx = cfg.dx
    radius = 0.5 * dx
    rows = []
    materials = []
    for k in range(pos.shape[0]):
        rows.append([float(pos[k, 0]), float(pos[k, 1])])
        materials.append("water")
    velocities = [[float(vel[k, 0]), float(vel[k, 1])] for k in range(vel.shape[0])]
    # Latticed mass unit of the run, not the raw rho0 dx^2: a DEM channel seeded
    # with a 0.3 % heavier water particle shifts the coupled dynamics it is
    # meant to represent (docs/verification_case_lattice_mass_calibration.md).
    from .droplet_impact import lattice_mass_unit
    mass_unit = lattice_mass_unit(cfg)
    masses = [float(mass_unit) for _ in range(pos.shape[0])]
    radii = [radius for _ in range(pos.shape[0])]
    if floes is None:
        floes = floe_layout(cfg, torch.device("cpu"))
    for k in range(floes.shape[0]):
        centre = floes[k].tolist()
        rows.append([float(centre[0]), float(centre[1])])
        speed = _floe_launch_speed(run.floe_impulse[k], cfg)
        # Hand the DEM solver the added-mass-corrected speed: the dry-mass value
        # over-predicts the launch by 1 + C_m rho_w/rho_ice (2.09x at C_m = 1).
        if cfg.floe_added_mass_coefficient > 0.0:
            speed = (speed * _floe_mass_per_depth(cfg)
                     / (_floe_mass_per_depth(cfg) + _floe_induced_mass_per_depth(cfg)))
        velocities.append([0.0, float(speed)])
        masses.append(_floe_mass_per_depth(cfg))
        radii.append(float(cfg.floe_radius))
        materials.append("ice")
    payload = {
        "schema": DEM_SCHEMA,
        "source": {
            "solver": "tensorsph",
            "case": "ice_splash",
            "dx_m": dx,
            "spacing_m": dx,
            "rho0_kg_m3": cfg.rho0,
            "rho_ice_kg_m3": cfg.floe_rho,
            "floe_added_mass_coefficient": cfg.floe_added_mass_coefficient,
            "gravity_m_s2": cfg.gravity,
            "seed": int(cfg.seed),
        },
        "domain": {"x_min": 0.0, "x_max": cfg.pool_width,
                   "y_min": 0.0, "y_max": cfg.tank_height},
        "particles": {
            "position_m": rows,
            "velocity_m_s": velocities,
            "mass_kg_per_m": masses,
            "radius_m": radii,
            "material": materials,
        },
        "counts": {"water": int(pos.shape[0]), "ice": int(floes.shape[0]),
                   "total": len(rows)},
    }
    return payload


def validate_dem_exchange(payload: dict) -> None:
    """Validate a DEM initialisation payload; raise ``ValueError`` when unusable."""
    if not isinstance(payload, dict):
        raise ValueError("DEM payload must be a dict")
    if payload.get("schema") != DEM_SCHEMA:
        raise ValueError(f"unsupported schema {payload.get('schema')!r}; expected {DEM_SCHEMA}")
    source = payload.get("source") or {}
    spacing = source.get("spacing_m")
    if spacing is None:
        raise ValueError("payload.source.spacing_m is required")
    if not math.isfinite(float(spacing)):
        raise ValueError("particle spacing must be finite")
    if float(spacing) <= 0.0:
        raise ValueError(f"particle spacing must be positive, got {spacing!r}")
    particles = payload.get("particles")
    if not isinstance(particles, dict):
        raise ValueError("payload.particles is required")
    position = particles.get("position_m")
    velocity = particles.get("velocity_m_s")
    mass = particles.get("mass_kg_per_m")
    radius = particles.get("radius_m")
    material = particles.get("material")
    for name, value in (("position_m", position), ("velocity_m_s", velocity),
                        ("mass_kg_per_m", mass), ("radius_m", radius),
                        ("material", material)):
        if not isinstance(value, list):
            raise ValueError(f"payload.particles.{name} must be a list")
    count = len(position)
    if count == 0:
        raise ValueError("payload contains no particles")
    for name, value in (("velocity_m_s", velocity), ("mass_kg_per_m", mass),
                        ("radius_m", radius), ("material", material)):
        if len(value) != count:
            raise ValueError(f"payload.particles.{name} has {len(value)} entries, "
                             f"expected {count}")
    allowed = {"water", "ice"}
    for k in range(count):
        x = position[k]
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError(f"particle {k}: position must be a 2-vector")
        if not all(math.isfinite(float(c)) for c in x):
            raise ValueError(f"particle {k}: non-finite position")
        if not all(math.isfinite(float(c)) for c in velocity[k]):
            raise ValueError(f"particle {k}: non-finite velocity")
        m = float(mass[k])
        if not math.isfinite(m):
            raise ValueError(f"particle {k}: non-finite mass")
        if m <= 0.0:
            raise ValueError(f"particle {k}: mass must be positive, got {m!r}")
        r = float(radius[k])
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError(f"particle {k}: radius must be positive, got {radius[k]!r}")
        if material[k] not in allowed:
            raise ValueError(f"particle {k}: unknown material {material[k]!r}")
    counts = payload.get("counts") or {}
    n_water = sum(1 for m in material if m == "water")
    n_ice = count - n_water
    if counts:
        if int(counts.get("water", -1)) != n_water or int(counts.get("ice", -1)) != n_ice:
            raise ValueError("payload.counts disagrees with the particle materials")
        if int(counts.get("total", -1)) != count:
            raise ValueError("payload.counts.total disagrees with the particle list")


def write_dem_exchange(payload: dict, path: str | Path) -> Path:
    validate_dem_exchange(payload)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return out


def read_dem_exchange(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    validate_dem_exchange(payload)
    return payload
