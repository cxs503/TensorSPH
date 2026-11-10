"""Rung 3 of the verification ladder: dam-break free surface.

A column of water, width ``a`` and height ``H = 2a``, collapses onto a dry floor
inside a tank. Two references are used, with different epistemic status:

* **Primary gate (analytic).** The Ritter (1892) shallow-water front
  ``x_front = a + 2 c0 t``, ``c0 = sqrt(g H)``, evaluated only inside its
  validity window ``t < a / c0`` (before the rarefaction returns from the closed
  back wall). This is exact for a semi-infinite reservoir of *shallow* water;
  the ``H = 2a`` column is not shallow, so the front is expected to run *below*
  it. The rung therefore reports both the front error and the ratio, and the
  ratio's deviation is bounded, not assumed away.
* **Cross-check (experimental, reported, not gated).** Martin & Moyce (1952)
  ``Z/a`` versus ``tau = t sqrt(2 g / a)``. The bundled table is a
  transcription from published figures (see ``references/README.md``) and the
  repository's own audit (``docs/benchmarks/dam-break-reference-audit.md``)
  grades it WITHHELD as a digitisation: its abscissae are exact multiples of a
  round step. It is quoted here because the campaign asks for it, and the
  measured disagreement is reported unmodified.

Free-surface position is measured two ways, because the two disagree at the
dry front and the disagreement is itself diagnostic:

* ``front_raw``: the largest ``x`` of any fluid particle;
* ``front_filtered``: the largest ``x`` among particles with at least
  ``min_neighbours`` neighbours inside ``2h`` (the repo's ``filtered_front``
  criterion), which suppresses spray particles that are individually ahead of
  the coherent front.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ..dam_break import DamBreakConfig, run_case

REFERENCE_DIR = Path(__file__).resolve().parent / "references"
MARTIN_MOYCE_FILE = REFERENCE_DIR / "martin_moyce_1952.json"


def load_martin_moyce(path: Path | None = None) -> dict:
    path = MARTIN_MOYCE_FILE if path is None else Path(path)
    with open(path) as handle:
        return json.load(handle)


def _interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] != xs[i - 1] else 0.0
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def filtered_front(pos: torch.Tensor, h: float, min_neighbours: int = 3) -> float:
    """Leading-edge position of the *coherent* front (neighbour-count filter)."""
    n = pos.shape[0]
    if n == 0:
        return float("nan")
    counts = torch.zeros(n, dtype=torch.long, device=pos.device)
    delta = pos[:, None, :] - pos[None, :, :]
    r2 = delta.square().sum(-1)
    counts = (r2 < (2.0 * h) ** 2).sum(-1) - 1
    core = pos[counts >= min_neighbours]
    source = core if core.shape[0] > 0 else pos
    return float(source[:, 0].max())


def dam_break_case(dx: float, seconds: float, tank_width: float = 2.0,
                   column_width: float = 0.2, column_height: float = 0.4,
                   sound_speed: float = 25.0, surface_tension: float = 0.0,
                   wall_friction: float = 0.0) -> DamBreakConfig:
    cfg = DamBreakConfig(dx=dx, column_width=column_width,
                         column_height=column_height,
                         pool_width=tank_width, tank_height=0.75,
                         sound_speed=sound_speed)
    cfg.dt = 0.3 * cfg.h_factor * dx / sound_speed
    cfg.steps = int(math.ceil(seconds / cfg.dt))
    cfg.dt = seconds / cfg.steps
    cfg.output_every = max(1, cfg.steps // 120)
    cfg.wall_friction = wall_friction
    cfg.surface_tension_coefficient = surface_tension
    return cfg


def jet_dispersion(pos: torch.Tensor, h: float, speed: torch.Tensor | None = None,
                   min_neighbours: int = 3, fraction: float = 0.05) -> dict:
    """Fragmentation metrics of the leading edge (used by the negative control).

    * ``detached_particles``: particles with fewer than ``min_neighbours``
      neighbours inside ``2h`` -- the spray/fragments;
    * ``leader_spread``: standard deviation of the transverse (``y``)
      coordinate of the fastest ``fraction`` of particles -- how far the jet has
      fanned out;
    * ``front_raw``: extreme particle position, comparable with the trajectory;
    * ``front``: coherent front position, so the three numbers are comparable
      between the two runs of the control.
    """
    n = pos.shape[0]
    delta = pos[:, None, :] - pos[None, :, :]
    r2 = delta.square().sum(-1)
    counts = (r2 < (2.0 * h) ** 2).sum(-1) - 1
    detached = int((counts < min_neighbours).sum())
    if speed is None:
        order = pos[:, 0].argsort(descending=True)
        take = max(1, int(fraction * n))
        leaders = pos[order[:take]]
    else:
        magnitude = torch.linalg.vector_norm(speed, dim=-1)
        take = max(1, int(fraction * n))
        leaders = pos[torch.topk(magnitude, take).indices]
    return {
        "front_raw": float(pos[:, 0].max()),
        "detached_particles": detached,
        "detached_fraction": detached / max(n, 1),
        "leader_spread": float(leaders[:, 1].std()) if leaders.shape[0] > 1 else 0.0,
        "leader_mean_y": float(leaders[:, 1].mean()),
        "front_filtered": filtered_front(pos, h, min_neighbours),
    }


def _front_speed_ratio(rows, c0: float, column_width: float) -> float:
    """Least-squares ``dZ/dt / (2 c0)`` over the established front.

    ``2 c0`` is the exact Ritter front speed, so the ratio measures how much of
    the shallow-water front speed the collapse actually delivers. Unlike the
    relative position error it does not diverge at ``t = 0``, because it is a
    slope, not a value.
    """
    points = [(r["t"], r["Z_over_a_sim"]) for r in rows]
    if len(points) < 3:
        return float("nan")
    n = float(len(points))
    mean_t = sum(t for t, _ in points) / n
    mean_z = sum(z for _, z in points) / n
    num = sum((t - mean_t) * (z - mean_z) for t, z in points)
    den = sum((t - mean_t) ** 2 for t, _ in points)
    if den <= 0.0:
        return float("nan")
    slope = num / den                       # d(Z/a)/dt
    analytic = 2.0 * c0 / column_width      # d(Z/a)/dt for Ritter
    return slope / analytic


def run_dambreak(dx: float = 0.02, seconds: float = 0.4,
                 sound_speed: float = 25.0, surface_tension: float = 0.0,
                 wall_friction: float = 0.0, exp: dict | None = None,
                 coarse_factor: float = 0.0) -> dict:
    """Integrate the dam break and compare the front with both references.

    ``coarse_factor > 0`` runs the identical case at ``dx * coarse_factor`` as
    well and reports the observed order between the two rungs. Refinement is the
    only axis that can turn a systematic front offset into evidence, so the rung
    carries it instead of quoting a single resolution.
    """
    exp = load_martin_moyce() if exp is None else exp
    cfg = dam_break_case(dx=dx, seconds=seconds, sound_speed=sound_speed,
                         surface_tension=surface_tension,
                         wall_friction=wall_friction)
    snap_time = 0.9 * seconds
    # The snapshot is always requested: the surface-tension control compares the
    # leading-edge geometry of the two runs, and a missing snapshot would make
    # the comparison silently degenerate to NaN instead of failing loudly.
    run = run_case(cfg, snapshot_times=[snap_time])
    g = cfg.gravity
    a = cfg.column_width
    h = cfg.h_factor * cfg.dx
    c0 = math.sqrt(g * cfg.column_height)
    t_valid = a / c0
    tau = [t * math.sqrt(2.0 * g / a) for t in run.times]
    za = [(f - a) / a for f in run.front]

    # ---- experimental cross-check ----------------------------------------
    table = exp["data"]
    taus = [d["tau"] for d in table]
    zas = [d["Z_over_a"] for d in table]
    pairs = [(ta, za_tab, _interp(tau, za, ta)) for ta, za_tab in zip(taus, zas)
             if tau[0] <= ta <= tau[-1]]
    if pairs:
        abs_err = [abs(z - z_ref) for _ta, z_ref, z in pairs]
        mae = sum(abs_err) / len(abs_err)
        rmse = math.sqrt(sum(e * e for e in abs_err) / len(abs_err))
        worst_tau, worst_ref, worst_sim = max(pairs, key=lambda p: abs(p[2] - p[1]))
    else:  # pragma: no cover - the default case always overlaps
        mae = rmse = float("nan")
        worst_tau = worst_ref = worst_sim = float("nan")
    # Relative error is quoted against the experimental front *displacement*
    # Z (not Z/a), so it is a dimensionless percentage of the column width.
    rel_exp_pct = mae * 100.0

    # ---- analytic front (validity window only) ---------------------------
    ritter_rows = []
    for ta, za_sim in zip(tau, za):
        t = ta / math.sqrt(2.0 * g / a)
        if t <= 0.0:
            continue
        z_ref = 2.0 * c0 * t / a          # (x_front - a)/a = 2 c0 t / a
        ritter_rows.append({"tau": ta, "t": t, "Z_over_a_sim": za_sim,
                            "Z_over_a_ritter": z_ref,
                            "ratio": za_sim / z_ref if z_ref > 0 else float("nan"),
                            "inside_validity_window": t <= t_valid})
    inside = [r for r in ritter_rows if r["inside_validity_window"]]
    # The relative error is singular at t -> 0, where the analytic front starts
    # from zero while the discrete front has already moved by one spacing. Two
    # metrics are therefore reported: the absolute displacement error in units
    # of the column width (finite at t = 0) and the relative error restricted to
    # the *established* front (tau >= 0.5), which is the gated one.
    abs_err_of_a = ([abs(r["Z_over_a_sim"] - r["Z_over_a_ritter"]) * 100.0
                     for r in inside] or [float("nan")])
    established = [r for r in inside if r["tau"] >= 0.5]
    ritter_err = ([abs(r["Z_over_a_sim"] - r["Z_over_a_ritter"]) /
                   r["Z_over_a_ritter"] * 100.0 for r in established] or [float("nan")])
    ritter_max = max(ritter_err)
    ritter_mean = sum(ritter_err) / len(ritter_err)

    t_end = run.times[-1]
    dispersion = None
    if True:
        if run.snapshots:
            t_snap, pos_snap, vel_snap = run.snapshots[0]
            dispersion = jet_dispersion(pos_snap, h, speed=vel_snap,
                                       fraction=0.2)
            dispersion["time"] = t_snap
        else:  # pragma: no cover
            dispersion = jet_dispersion(run.positions, h)

    convergence = None
    if coarse_factor and coarse_factor > 1.0:
        coarse = run_dambreak(dx=dx * coarse_factor, seconds=seconds,
                              sound_speed=sound_speed, exp=exp,
                              wall_friction=wall_friction, coarse_factor=0.0)
        c_mae = coarse["measured"]["experimental_mae_Z_over_a"]
        f_mae = mae
        c_spd = coarse["measured"]["ritter_front_speed_ratio"]
        f_spd = _front_speed_ratio(established, c0, a)
        order_mae = (math.log(c_mae / f_mae) / math.log(coarse_factor)
                     if f_mae > 0 and c_mae > 0 else float("nan"))
        order_deficit = (math.log((1.0 - c_spd) / (1.0 - f_spd))
                         / math.log(coarse_factor)
                         if 0.0 < f_spd < 1.0 and 0.0 < c_spd < 1.0 else float("nan"))
        convergence = {
            "coarse_dx": dx * coarse_factor,
            "fine_dx": dx,
            "experimental_mae_coarse": c_mae,
            "experimental_mae_fine": f_mae,
            "observed_order_experimental_mae": order_mae,
            "front_speed_ratio_coarse": c_spd,
            "front_speed_ratio_fine": f_spd,
            "observed_order_front_speed_deficit": order_deficit,
            "ritter_error_coarse_pct": coarse["error_pct"],
        }

    return {
        "case": "dam_break_2d_free_surface",
        "config": {
            "dx": dx, "seconds": seconds, "dt": cfg.dt, "steps": cfg.steps,
            "fluid_particles": run.final_positions.shape[0],
            "column_width": a, "column_height": cfg.column_height,
            "aspect_H_over_a": cfg.column_height / a,
            "tank_width": cfg.pool_width, "sound_speed": sound_speed,
            "sound_speed_analytic": c0, "ritter_validity_window": t_valid,
            "wall_friction": wall_friction,
            "surface_tension": surface_tension,
            "weber_number": (1000.0 * 4.0 * g * cfg.column_height * a / surface_tension
                             if surface_tension > 0 else float("inf")),
            "t_end": t_end,
        },
        "reference": {
            "kind": "analytic+experimental",
            "analytic": "Ritter (1892) shallow-water front, x_f = a + 2 sqrt(gH) t",
            "experimental": exp.get("reference"),
            "experimental_status": "transcription, audited WITHHELD as a digitisation",
        },
        "measured": {
            "experimental_mae_Z_over_a": mae,
            "experimental_rmse_Z_over_a": rmse,
            "experimental_worst_point": {"tau": worst_tau,
                                         "Z_over_a_experiment": worst_ref,
                                         "Z_over_a_simulation": worst_sim},
            "experimental_relative_pct_of_a": rel_exp_pct,
            "ritter_max_error_pct": ritter_max,
            "ritter_mean_error_pct": ritter_mean,
            "ritter_window_points": len(established),
            "ritter_max_abs_error_pct_of_a": max(abs_err_of_a),
            "ritter_ratio_range": [min((r["ratio"] for r in established), default=float("nan")),
                                   max((r["ratio"] for r in established), default=float("nan"))],
            "ritter_front_speed_ratio": _front_speed_ratio(established, c0, a),
            "convergence": convergence,
            "mass_drift": run.mass_drift,
            "max_mach": run.max_speed / sound_speed,
            "max_density_compression": run.max_density_compression,
            "escaped_particles": run.escaped,
            "steps_taken": run.steps_taken,
            "dispersion": dispersion,
        },
        # The gated quantity is the free-surface position error against the
        # analytic front inside its validity window; the experimental number is
        # carried alongside and never overwrites it.
        "error_pct": ritter_max,
        "series": {"tau": tau, "Z_over_a": za, "ritter": ritter_rows},
        "gate": {"quantity": "front position error vs Ritter, validity window, "
                             "established front (tau >= 0.5)",
                 "limit_pct": 5.0, "pass": bool(ritter_max < 5.0)},
    }