"""M1 -- 2-D dam-break free-surface verification against measured data.

The case is the classical Martin & Moyce (1952) collapse: a rectangular column
of water, width ``a`` and height ``H = 2a``, released from rest on a dry floor
inside a long tank.  Three references are used, with deliberately different
epistemic status:

``experimental``
    ``Z/a`` against ``tau = t sqrt(2 g / a)`` from
    ``benchmarks/dam_break/martin_moyce_1952.json``.  The repository's own audit
    (``benchmarks/dam_break/reference_audit.json``) grades this table
    **WITHHELD**: its abscissae sit on an exact arithmetic lattice, which no
    manual figure read-off produces.  It is therefore used here as a *gated
    target only because the campaign specifies it*, and the grade travels with
    every number derived from it.

``ritter``
    The exact shallow-water front ``Z = 2 sqrt(g H) t``.  Exact for a
    semi-infinite shallow reservoir, which this column is not (``H/a = 2``), so
    it is a *bounded* reference used here only to report how far the collapse is
    from the shallow limit -- never as a 10 % gate.

``height anchors``
    No digitised experimental *height* curve exists in this repository or in any
    licence-free source reachable from it, so the height gate is anchored on two
    **exact** statements instead of on data:

    * ``residual column``: upstream of the Ritter rarefaction tail the reservoir
      is undisturbed, so the free-surface height at a gauge ``x_g`` is ``H``
      until the tail arrives at ``t = (a - x_g) / c0``, ``c0 = sqrt(g H)``;
    * ``ballistic corner``: the top corner of the column is on the free surface
      with zero pressure on both faces, so it falls as ``H - g t^2 / 2`` until
      the acoustic signal from the bulk reaches it (``t = a / c0``).

    Both are consequences of the initial/boundary data, not fits, and both are
    falsifiable by a wrong gravity coupling or a free surface that carries
    spurious pressure.  The experimental height comparison stays ``WITHHELD``;
    the report says so instead of quoting a number that cannot be traced.

Measurement definitions (they disagree, and the disagreement is diagnostic):

``front_raw``
    largest ``x`` of any fluid particle;
``front_coherent``
    largest ``x`` among particles that still have at least
    ``min_neighbours`` neighbours inside ``2h`` -- suppresses spray;
``h_back``
    highest fluid particle at the back-wall gauge ``x <= a/2``;
``h_dam``
    highest fluid particle at the dam gauge ``|x - a| <= 2 dx``.

Negative control (``gravity = 0``): the column must keep its initial shape.  The
control is run with *uniform* mass and calibrated density, so the initial
pressure is zero to machine precision -- otherwise the control would be testing
the initialisation rather than the gravity coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from ..dam_break import DamBreakConfig, run_case

EXPERIMENTAL_TABLE = (Path(__file__).resolve().parents[3] / "benchmarks"
                      / "dam_break" / "martin_moyce_1952.json")
EXPERIMENTAL_GRADE = "WITHHELD"


@dataclass
class DamBreak2DConfig:
    """Parameters of the verification case (not of the solver)."""

    dx: float = 0.02
    column_width: float = 0.2
    aspect: float = 2.0            # H / a, the Martin & Moyce geometry
    tank_width: float = 2.0
    tank_height: float = 0.75
    sound_speed: float = 25.0
    viscosity: float = 1.0e-6      # water at 20 C; Config's default is 1e-2
    gravity: float = 9.81
    seconds: float = 0.45          # covers the whole experimental window (tau<=4.45)
    sample_interval: float = 0.01  # s between surface measurements
    max_courant: float = 0.3
    h_factor: float = 1.3
    min_neighbours: int = 3
    surface_pressure_closure: bool = False
    density_renormalization: bool = False
    density_diffusion: float = 0.0
    hydrostatic_init: bool = True
    wall_friction: float = 0.0
    device: str = "cpu"

    @property
    def column_height(self) -> float:
        return self.aspect * self.column_width

    @property
    def c0_analytic(self) -> float:
        """Shallow-water wave speed ``sqrt(g H)`` of the initial column."""
        return math.sqrt(self.gravity * self.column_height)

    def dt(self) -> float:
        return self.max_courant * self.h_factor * self.dx / self.sound_speed

    def solver_config(self) -> DamBreakConfig:
        cfg = DamBreakConfig(
            dx=self.dx,
            column_width=self.column_width,
            column_height=self.column_height,
            pool_width=self.tank_width,
            tank_height=self.tank_height,
            pool_depth=self.tank_height,
            sound_speed=self.sound_speed,
            viscosity=self.viscosity,
            gravity=self.gravity,
            wall_friction=self.wall_friction,
            density_diffusion=self.density_diffusion,
            density_renormalization=self.density_renormalization,
            surface_pressure_closure=self.surface_pressure_closure,
            hydrostatic_init=self.hydrostatic_init,
            h_factor=self.h_factor,
            max_courant=self.max_courant,
            device=self.device,
        )
        cfg.dt = self.dt()
        cfg.steps = int(round(self.seconds / cfg.dt))
        cfg.output_every = max(1, int(round(self.sample_interval / cfg.dt)))
        return cfg

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["column_height"] = self.column_height
        d["c0_analytic"] = self.c0_analytic
        return d


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #

def load_experimental(path: Path | None = None) -> dict:
    """The campaign's experimental table, with its own audit verdict attached."""
    path = EXPERIMENTAL_TABLE if path is None else Path(path)
    with open(path) as handle:
        table = json.load(handle)
    table["_path"] = str(path)
    grade = (table.get("integrity_audit") or {}).get("grade", EXPERIMENTAL_GRADE)
    table["_grade"] = grade
    return table


def experimental_front(table: dict, tau: float) -> float | None:
    """``Z/a`` of the experiment at ``tau`` (linear interpolation, no extrapolation)."""
    rows = table["data"]
    if tau < rows[0]["tau"] or tau > rows[-1]["tau"]:
        return None
    for i in range(1, len(rows)):
        if rows[i]["tau"] >= tau:
            t0, t1 = rows[i - 1]["tau"], rows[i]["tau"]
            v0, v1 = rows[i - 1]["Z_over_a"], rows[i]["Z_over_a"]
            if t1 == t0:
                return v0
            return v0 + (tau - t0) / (t1 - t0) * (v1 - v0)
    return rows[-1]["Z_over_a"]


def ritter_front(cfg: DamBreak2DConfig, time: float) -> float:
    """``Z/a = 2 sqrt(g H) t / a`` -- exact for shallow semi-infinite water."""
    return 2.0 * cfg.c0_analytic * time / cfg.column_width


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def surface_profile(pos: torch.Tensor, h: float, x_max: float) -> list[float]:
    """Free-surface height per ``2h`` column, ``nan`` where the bin is dry."""
    nbins = max(1, int(math.ceil(x_max / (2.0 * h))))
    prof = torch.full((nbins,), float("nan"), dtype=torch.float64)
    idx = (pos[:, 0] / (2.0 * h)).long().clamp(0, nbins - 1)
    top = torch.full((nbins,), -1.0, dtype=torch.float64)
    top.scatter_reduce_(0, idx, pos[:, 1].double(), reduce="amax")
    wet = top > -0.5
    prof[wet] = 0.0
    prof[wet] = top[wet]
    return [None if math.isnan(v) else float(v) for v in prof]


def measure(pos: torch.Tensor, cfg: DamBreak2DConfig) -> dict:
    """Free-surface observables of one snapshot."""
    h = cfg.h_factor * cfg.dx
    a = cfg.column_width
    n = pos.shape[0]
    if n == 0:
        raise ValueError("no fluid particles to measure")
    delta = pos[:, None, :] - pos[None, :, :]
    counts = (delta.square().sum(-1) < (2.0 * h) ** 2).sum(-1) - 1
    front_raw = float(pos[:, 0].max())
    core = pos[counts >= cfg.min_neighbours]
    front_coherent = float(core[:, 0].max()) if core.shape[0] else front_raw
    back = pos[pos[:, 0] <= 0.5 * a]
    dam = pos[(pos[:, 0] >= a - 2.0 * cfg.dx) & (pos[:, 0] <= a + 2.0 * cfg.dx)]
    x_max = float(pos[:, 0].max())
    prof = surface_profile(pos, h, x_max)
    bin_w = 2.0 * h
    # Leading edge of the *water layer*: the last bin whose surface height is at
    # least one particle spacing.  A bin thinner than that is a spray particle,
    # not the front of the collapsing column -- ``front_raw`` counts it, this
    # does not, and the two are reported side by side.
    layer = [i for i, v in enumerate(prof)
             if v is not None and v >= cfg.dx]
    front_layer = (layer[-1] + 0.5) * bin_w if layer else float("nan")
    return {
        "front_raw": front_raw,
        "front_layer": front_layer,
        "front_coherent": front_coherent,
        "h_back": float(back[:, 1].max()) if back.shape[0] else float("nan"),
        "h_dam": float(dam[:, 1].max()) if dam.shape[0] else float("nan"),
        "h_max": float(pos[:, 1].max()),
        "profile": prof,
        "profile_bin": bin_w,
        "detached": int((counts < cfg.min_neighbours).sum()),
        "particles": int(n),
        "water_area": float(n) * cfg.dx * cfg.dx,
        "x_max": x_max,
    }


def run(cfg: DamBreak2DConfig, label: str = "") -> dict:
    """Integrate the case and return the raw observable time series."""
    solver = cfg.solver_config()
    # Snapshots are only captured on output steps (``dam_break.run_case``), so
    # the requested times are placed exactly on them.
    every = solver.output_every
    targets = [round(k * every * solver.dt, 12)
               for k in range(1, solver.steps // every + 1)]
    result = run_case(solver, snapshot_times=targets)
    rows = []
    for (t, pos, vel) in result.snapshots:
        row = measure(pos, cfg)
        row["t"] = float(t)
        row["tau"] = float(t * math.sqrt(2.0 * cfg.gravity / cfg.column_width))
        row["speed_max"] = float(torch.linalg.vector_norm(vel, dim=-1).max())
        rows.append(row)
    return {
        "label": label or f"dam_break_2d_dx{cfg.dx:g}",
        "config": cfg.as_dict(),
        "rows": rows,
        "mass_initial": float(result.initial_mass),
        "mass_final": float(result.final_mass),
        "mass_drift": float(result.mass_drift),
        "escaped": int(result.escaped),
        "steps_taken": int(result.steps_taken),
        "simulated_time": float(result.simulated_time),
        "max_density_compression": float(result.max_density_compression),
        "max_mach": float(result.max_speed / cfg.sound_speed),
        "dt": float(solver.dt),
    }


# --------------------------------------------------------------------------- #
# comparisons and gates
# --------------------------------------------------------------------------- #

def front_errors(run_result: dict, table: dict, key: str = "front_coherent",
                 min_tau: float = 0.0) -> dict:
    """Relative / absolute front deviation at every experimental point.

    ``Z/a`` is measured from the initial column face, exactly as in the
    experiment: ``Z/a = (x_front - a) / a``.
    """
    a = run_result["config"]["column_width"]
    rows = sorted(run_result["rows"], key=lambda r: r["tau"])
    points = []
    for exp in table["data"]:
        tau_exp = exp["tau"]
        if tau_exp < min_tau:
            continue
        za_exp = exp["Z_over_a"]
        za_sim = _interp_front(rows, tau_exp, key, a)
        if za_sim is None:
            continue
        if za_exp > 0:
            rel = abs(za_sim - za_exp) / za_exp
        else:
            rel = abs(za_sim - za_exp)          # absolute at tau = 0
        points.append({"tau": tau_exp, "Z_over_a_experiment": za_exp,
                       "Z_over_a_simulation": za_sim,
                       "abs_error": abs(za_sim - za_exp),
                       "rel_error": rel,
                       "experimental_uncertainty": exp.get("uncertainty")})
    if not points:
        raise ValueError("no overlapping experimental points")
    rels = [p["rel_error"] for p in points if p["tau"] > 0]
    return {
        "points": points,
        "max_rel_error": max(rels) if rels else float("nan"),
        "rms_rel_error": (math.sqrt(sum(r * r for r in rels) / len(rels))
                          if rels else float("nan")),
        "max_abs_error": max(p["abs_error"] for p in points),
        "n_points": len(points),
    }


def _interp_front(rows: list[dict], tau: float, key: str, a: float) -> float | None:
    """Monotone (running maximum) interpolation of ``Z/a`` at ``tau``."""
    best = None
    for row in rows:
        za = (row[key] - a) / a
        if row["tau"] <= tau + 1e-12:
            best = za if best is None else max(best, za)
        else:
            if best is None:
                return za
            prev = [r for r in rows if r["tau"] < row["tau"]][-1]
            span = row["tau"] - prev["tau"]
            w = 0.0 if span <= 0 else (tau - prev["tau"]) / span
            return best + w * (za - best) if za > best else best
    return best


def height_anchor_errors(run_result: dict) -> dict:
    """Deviation of the two exact height anchors (see module docstring)."""
    cfg = run_result["config"]
    a = cfg["column_width"]
    H = cfg["column_height"]
    c0 = cfg["c0_analytic"]
    g = cfg["gravity"]
    t_back = 0.5 * a / c0     # rarefaction tail reaches the gauge x = a/2
    t_ball = a / c0           # acoustic crossing of the column width
    resid = [r for r in run_result["rows"] if r["t"] <= t_back + 1e-12]
    ball = [r for r in run_result["rows"] if r["t"] <= t_ball + 1e-12]
    back_err = ([abs(r["h_back"] - H) / H for r in resid] or [float("nan")])
    ball_err = ([abs(r["h_dam"] - (H - 0.5 * g * r["t"] ** 2)) / H
                 for r in ball if math.isfinite(r["h_dam"])] or [float("nan")])
    return {
        "t_back": t_back,
        "t_ballistic": t_ball,
        "residual_points": len(resid),
        "residual_max_rel_error": max(back_err),
        "ballistic_points": len(ball),
        "ballistic_max_rel_error": max(ball_err),
        "h_back_at_end_of_residual_window": resid[-1]["h_back"] if resid else None,
        "h_dam_at_end_of_ballistic_window": ball[-1]["h_dam"] if ball else None,
        "ballistic_drop_fraction": 0.5 * g * t_ball ** 2 / H,
    }


def ritter_errors(run_result: dict) -> dict:
    """How far the measured front is from the shallow-water front (reported)."""
    cfg = run_result["config"]
    a = cfg["column_width"]
    c0 = cfg["c0_analytic"]
    t_valid = a / c0          # rarefaction returns from the closed back wall
    pts = []
    for r in run_result["rows"]:
        if r["t"] <= 0 or r["t"] > t_valid:
            continue
        za_ref = 2.0 * c0 * r["t"] / a
        za_sim = (r["front_coherent"] - a) / a
        pts.append({"tau": r["tau"], "t": r["t"], "Z_over_a_simulation": za_sim,
                    "Z_over_a_ritter": za_ref,
                    "ratio": za_sim / za_ref if za_ref > 0 else float("nan")})
    ratios = [p["ratio"] for p in pts if p["tau"] >= 0.5]
    return {"points": pts, "t_valid": t_valid,
            "ratio_min": min(ratios) if ratios else float("nan"),
            "ratio_max": max(ratios) if ratios else float("nan"),
            "ratio_mean": sum(ratios) / len(ratios) if ratios else float("nan")}


def negative_control_gravity_off(cfg: DamBreak2DConfig, seconds: float = 0.2) -> dict:
    """Negative control for M1: without gravity the column must not collapse."""
    ctrl = DamBreak2DConfig(**{**cfg.__dict__, "gravity": 0.0, "seconds": seconds,
                               "hydrostatic_init": False,
                               "sample_interval": 0.02})
    solver = ctrl.solver_config()
    result = run_case(solver)
    # initial lattice, so the control has a reference shape that does not come
    # from the solver's own trajectory
    from ..dam_break import make_column
    from ..droplet_impact import resolve_device
    pos0, _ = make_column(solver, resolve_device("cpu"))
    x0 = float(pos0[:, 0].max())
    y0 = float(pos0[:, 1].max())
    x1 = float(result.final_positions[:, 0].max())
    y1 = float(result.final_positions[:, 1].max())
    matched = _match(pos0, result.final_positions)
    drift = torch.linalg.vector_norm(result.final_positions - matched, dim=-1).max()
    return {
        "case": "dam_break_2d_gravity_off",
        "seconds": seconds,
        "dx": ctrl.dx,
        "front_drift_over_dx": abs(x1 - x0) / ctrl.dx,
        "height_drift_rel": abs(y1 - y0) / y0,
        "max_particle_displacement_over_dx": float(drift) / ctrl.dx,
        "particles": int(pos0.shape[0]),
        "steps": int(result.steps_taken),
        "gate": {"front_drift_over_dx": 0.5, "height_drift_rel": 0.01,
                 "max_particle_displacement_over_dx": 2.0},
    }


def _match(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Nearest neighbour of every particle of ``a`` among ``b`` (same count)."""
    if a.shape == b.shape and a.shape[0] <= 2000:
        d = torch.cdist(a.double(), b.double())
        return b[d.argmin(dim=1)]
    return b


def boundary_control_particles(cfg: DamBreak2DConfig) -> int:
    """Particles needed for the case at ``cfg.dx`` (for the evidence record)."""
    nx = int(math.floor(cfg.column_width / cfg.dx))
    ny = int(math.floor(cfg.column_height / cfg.dx))
    return nx * ny
