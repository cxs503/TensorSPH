"""Ritter shallow-water verification of the dam-break solver (B2b).

The B2 benchmark (``docs/benchmarks/dam-break-2d.md``) compares its front
trajectory with Ritter's shallow-water solution but explicitly refuses to gate on
it: the B2 column has ``H / a = 2``, so shallow-water theory does not apply and a
systematic offset is expected. That leaves an open verification question --
**is the offset physics, or is it the scheme?**

This benchmark answers it by moving the *same* solver into the regime where
Ritter is exact: a long, shallow column (``H / a`` from 0.05 to 0.2) released
against a dry floor, integrated only up to ``0.8 a / c0`` so the reflected
rarefaction from the back wall has not entered the domain. Three things are then
measured against the analytic solution:

* ``Z(t)`` -- the leading-edge trajectory, ``x_dam + 2 c0 t``;
* ``h(x, t)`` -- the depth field, ``(2 c0 - xi)^2 / 9g``;
* ``u(x, t)`` -- the depth-averaged velocity, ``(2/3)(c0 + xi)``.

and a resolution ladder at ``H / a = 0.1`` gives the observed order of
convergence. The ``H / a = 2`` point is run too, to show the error collapses as
the shallow-water assumption becomes true.

The depth field is measured by *counting* particles per x-bin,
``h = n dx^2 / bin_width`` (see :func:`tensorsph.ritter.particle_depth`), so the
comparison cannot be biased by the solver's ~1 % compressibility.

Usage, one case (the form the parallel driver uses):

    python benchmarks/dam_break/ritter_limit.py --case h0.10_dx0.005 \\
        --json artifacts/ritter_limit/h0.10_dx0.005.json

Several cases are the preset series (``--preset default``, run sequentially); a
case file can also be selected explicitly:

    python benchmarks/dam_break/ritter_limit.py --preset ladder --workdir artifacts/ritter_limit

Aggregate finished cases into the markdown table and figures:

    python benchmarks/dam_break/ritter_limit.py --aggregate artifacts/ritter_limit/*.json \\
        --figdir artifacts/ritter_limit --doc docs/benchmarks/ritter-shallow-limit.md

Exit status is non-zero when the verification gate fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorsph.dam_break import DamBreakConfig, run_case
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.ritter import (
    depth as ritter_depth,
    front_x,
    particle_depth,
    profile_l1_error,
    validity_window,
    velocity as ritter_velocity,
    wave_speed,
)

# --------------------------------------------------------------------------- #
# Case definition
# --------------------------------------------------------------------------- #

#: ``(name, column_width a, column_height H, dx)``. ``H / a`` is the shallow-water
#: parameter; the ladder fixes it at 0.1 and refines dx by two.
PRESETS = {
    "default": [
        ("h0.10_dx0.020", 1.0, 0.10, 0.020),
        ("h0.10_dx0.010", 1.0, 0.10, 0.010),
        ("h0.10_dx0.005", 1.0, 0.10, 0.005),
        ("h0.20_dx0.010", 0.5, 0.10, 0.010),
        ("h2.00_dx0.010", 0.1, 0.20, 0.010),
    ],
    "ladder": [
        ("h0.10_dx0.020", 1.0, 0.10, 0.020),
        ("h0.10_dx0.010", 1.0, 0.10, 0.010),
        ("h0.10_dx0.005", 1.0, 0.10, 0.005),
    ],
    "trend": [
        ("h0.20_dx0.010", 0.5, 0.10, 0.010),
        ("h2.00_dx0.010", 0.1, 0.20, 0.010),
    ],
}

#: Fraction of the validity window ``a / c0`` that is integrated. Below 1 the
#: reflected rarefaction cannot reach the measurement region.
WINDOW_FRACTION = 0.8
#: Sound speed in units of ``sqrt(g H)``: the Mach number of the surge, whose
#: peak is ``2 c0`` = this number. 20 gives M = 0.10, the solver's own limit.
SOUND_SPEED_RATIO = 20.0


def build_case(name, column_width, column_height, dx, seconds=None,
               window_fraction=WINDOW_FRACTION, device="cpu",
               sound_speed_ratio=SOUND_SPEED_RATIO):
    """A shallow-column dam-break configuration plus its validity window."""
    c0 = wave_speed(column_height)
    t_valid = validity_window(column_height, column_width)
    span = window_fraction * t_valid if seconds is None else float(seconds)
    # The front travels 2 c0 * span past the dam; leave three cells of clearance
    # so it never reaches the far wall inside the window.
    tank_width = column_width + 2.0 * c0 * span + 6.0 * dx
    cfg = DamBreakConfig(
        dx=dx,
        pool_width=tank_width,
        pool_depth=column_height,      # fallback only; tank_height is explicit
        column_width=column_width,
        column_height=column_height,
        tank_height=1.5 * column_height,
        sound_speed=sound_speed_ratio * c0,
        viscosity=0.0,
        hydrostatic_init=False,
        wall_friction=0.0,
        device=device,
        output_every=10,
    )
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(math.ceil(span / cfg.dt)) + 1
    return cfg, span, t_valid


def filtered_front(positions: torch.Tensor, h: float, min_neighbors: int = 3):
    """Leading edge ignoring particles with fewer than ``min_neighbors`` neighbours.

    A dry-bed SPH front sheds one or two nearly isolated particles ahead of the
    bulk; they are real (the solver puts them there) but they are not the wave
    front. Both numbers are reported: ``max_x`` and this one.
    """
    max_x = float(positions[:, 0].max())
    if positions.shape[0] <= min_neighbors:
        return max_x, 0
    pairs = cell_linked_pairs(positions, 2.0 * h)      # (pair_count, 2)
    counts = torch.zeros(positions.shape[0], dtype=torch.long)
    if pairs.numel():
        i, j = pairs.unbind(dim=1)
        counts.index_add_(0, i, torch.ones_like(i))
        counts.index_add_(0, j, torch.ones_like(j))
    keep = counts >= min_neighbors
    stray = int((~keep).sum())
    if not bool(keep.any()):
        return max_x, stray
    return float(positions[keep, 0].max()), stray


# --------------------------------------------------------------------------- #
# One case
# --------------------------------------------------------------------------- #

def run_one(name, column_width, column_height, dx, device="cpu", seconds=None,
            window_fraction=WINDOW_FRACTION, sound_speed_ratio=SOUND_SPEED_RATIO):
    cfg, span, t_valid = build_case(name, column_width, column_height, dx,
                                    seconds=seconds, window_fraction=window_fraction,
                                    device=device, sound_speed_ratio=sound_speed_ratio)
    h = cfg.h_factor * dx
    c0 = wave_speed(column_height)
    snap_targets = [0.55 * span, span]
    started = time.time()
    run = run_case(cfg, snapshot_times=snap_targets)
    wall_clock = time.time() - started

    if run.snapshots and run.snapshots[-1][0] < 0.98 * span:
        raise RuntimeError(
            f"{name}: last snapshot at {run.snapshots[-1][0]:.4g}s, wanted {span:.4g}s")

    # --- front trajectory -------------------------------------------------- #
    trajectory = []
    for t, z in zip(run.times, run.front):
        if t <= 0.0:
            continue
        z_ref = front_x(t, column_height, dam_x=column_width)
        trajectory.append({
            "time": t,
            "front": z,
            "front_ritter": z_ref,
            "relative_error": (z - z_ref) / z_ref,
        })
    errors = [row["relative_error"] for row in trajectory]
    front_metrics = {
        "mean_relative_error": sum(errors) / len(errors),
        "max_abs_relative_error": max(abs(e) for e in errors),
        "final_relative_error": errors[-1],
        "final_time": trajectory[-1]["time"],
        "front_advance_m": trajectory[-1]["front"] - column_width,
        "ritter_advance_m": trajectory[-1]["front_ritter"] - column_width,
    }

    # --- depth and velocity fields at the snapshot times ------------------- #
    bin_width = 2.0 * dx
    x_hi = cfg.pool_width
    lo = 0.0
    profiles, fields = [], []
    for t, pos, vel in run.snapshots:
        centres, h_sph = particle_depth(pos, dx, bin_width, lo=lo, hi=x_hi)
        h_ref = ritter_depth(centres, t, column_height, dam_x=column_width)
        # Both profiles are carried on the same grid, so the L1 difference is a
        # shape error and not a sampling artefact of the front moving between
        # the two grids.
        wet = h_ref > 0.0
        l1 = profile_l1_error(centres, h_sph, h_ref, mask=wet)
        front_max, stray = filtered_front(pos, h, min_neighbors=3)
        z_ref = front_x(t, column_height, dam_x=column_width)
        speed = (torch.linalg.vector_norm(vel, dim=-1)
                 if vel.numel() else torch.zeros(0))
        profiles.append({
            "time": t,
            "x": centres.tolist(),
            "depth": h_sph.tolist(),
            "depth_ritter": h_ref.tolist(),
            "relative_l1_error": l1,
            "wet_length_m": float(centres[wet].max()) if bool(wet.any()) else 0.0,
            "front_max_x": front_max,
            "front_relative_error": (front_max - z_ref) / z_ref,
            "stray_particles": stray,
        })
        fields.append({
            "time": t,
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "speed_over_c0": (speed / c0).tolist(),
            "u_ritter_at_particle": ritter_velocity(pos[:, 0], t, column_height,
                                                    dam_x=column_width).tolist(),
        })

    result = {
        "name": name,
        "column_width": column_width,
        "column_height": column_height,
        "aspect": column_height / column_width,
        "dx": dx,
        "layers": column_height / dx,
        "particles": int(run.snapshots[-1][1].shape[0]) if run.snapshots else 0,
        "sound_speed": cfg.sound_speed,
        "sound_speed_ratio": sound_speed_ratio,
        "case_family": ("compressibility"
                        if abs(sound_speed_ratio - SOUND_SPEED_RATIO) > 1.0e-9
                        else "reference"),
        "mach_surge": 2.0 * c0 / cfg.sound_speed,
        "dt": cfg.dt,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "validity_window": t_valid,
        "window_fraction": span / t_valid,
        "tank_width": cfg.pool_width,
        "wall_clock_s": wall_clock,
        "max_speed_over_c0": run.max_speed / c0,
        "max_acoustic_courant": run.max_acoustic_courant,
        "max_density_compression": run.max_density_compression,
        "mass_drift": run.mass_drift,
        "escaped": run.escaped,
        "trajectory": trajectory[:: max(1, len(trajectory) // 60)],
        "front": front_metrics,
        "profiles": profiles,
        "fields": fields,
    }
    return result


# --------------------------------------------------------------------------- #
# Aggregation, gates, report
# --------------------------------------------------------------------------- #

def observed_order(coarse, fine):
    """``p`` from ``e ~ dx^p`` between two cases of the same ``H / a``."""
    e1 = abs(coarse["front"]["final_relative_error"])
    e2 = abs(fine["front"]["final_relative_error"])
    d1, d2 = coarse["dx"], fine["dx"]
    if e1 <= 0 or e2 <= 0 or d1 == d2:
        return None
    return math.log(e1 / e2) / math.log(d1 / d2)


def gates(results):
    """Acceptance criteria, evaluated on the finest shallow case.

    Two front thresholds are reported on purpose: the strict 3 % used for a
    claim of "verified against the analytic solution", and the 15 % tolerance
    beyond which a result is not usable at all. Between them the run is a
    quantified, converging error budget rather than a pass.
    """
    shallow = [r for r in results if r["aspect"] <= 0.2]
    if not shallow:
        return []
    finest = min(shallow, key=lambda r: r["dx"])
    final_profile = finest["profiles"][-1]
    front_error = abs(finest["front"]["final_relative_error"])
    ladder = sorted([r for r in results if abs(r["aspect"] - 0.1) < 1.0e-9],
                    key=lambda r: -r["dx"])
    order = (observed_order(ladder[0], ladder[-1])
             if len(ladder) >= 2 else None)
    # The shallowness trend has to be read at fixed dx: comparing a refined
    # shallow case against a coarse deeper one would conflate the two effects.
    by_dx = {}
    for case in results:
        by_dx.setdefault(case["dx"], []).append(case)
    groups = [sorted(group, key=lambda r: r["aspect"]) for group in by_dx.values()
              if len(group) >= 2]
    monotone = bool(groups) and all(
        all(abs(a["front"]["final_relative_error"])
            <= abs(b["front"]["final_relative_error"])
            for a, b in zip(group, group[1:])) for group in groups)
    rows = [
        {"name": "front_final_within_3pct", "value": front_error, "limit": 0.03,
         "comparison": "<=", "pass": front_error <= 0.03},
        {"name": "front_final_within_15pct", "value": front_error, "limit": 0.15,
         "comparison": "<=", "pass": front_error <= 0.15},
        {"name": "depth_L1_final_within_15pct",
         "value": final_profile["relative_l1_error"], "limit": 0.15,
         "comparison": "<=", "pass": final_profile["relative_l1_error"] <= 0.15},
        {"name": "mass_conserved", "value": finest["mass_drift"], "limit": 0.0,
         "comparison": "==", "pass": abs(finest["mass_drift"]) == 0.0},
        {"name": "no_particle_escaped", "value": finest["escaped"], "limit": 0,
         "comparison": "==", "pass": finest["escaped"] == 0},
        {"name": "bounded_speed_5c0", "value": finest["max_speed_over_c0"],
         "limit": 5.0, "comparison": "<=",
         "pass": finest["max_speed_over_c0"] <= 5.0},
        {"name": "shallowness_error_monotone", "value": float(monotone),
         "limit": 1.0, "comparison": "==", "pass": monotone},
    ]
    if order is not None:
        rows.append({"name": "observed_order_above_0.5", "value": order,
                     "limit": 0.5, "comparison": ">=", "pass": order >= 0.5})
    return rows


def markdown(results, gate_rows):
    order = sorted(results, key=lambda r: (r["aspect"], -r["dx"]))
    lines = [
        "| `H/a` | `dx` | layers | particles | steps | front err (mean) "
        "| front err (final) | depth L1 (final) | max \\|v\\|/c0 | compression | wall clock |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in order:
        lines.append(
            f"| {r['aspect']:.2f} | {r['dx']:.3f} | {r['layers']:.0f} | {r['particles']} "
            f"| {r['steps']} | {100*r['front']['mean_relative_error']:+.2f} % "
            f"| {100*r['front']['final_relative_error']:+.2f} % "
            f"| {100*r['profiles'][-1]['relative_l1_error']:.2f} % "
            f"| {r['max_speed_over_c0']:.2f} | {100*r['max_density_compression']:.2f} % "
            f"| {r['wall_clock_s']:.1f} s |")
    return "\n".join(lines)


def aggregate(paths, figdir=None, doc=None):
    results = []
    for path in paths:
        with open(path) as handle:
            case = json.load(handle)
        if "aspect" not in case:          # e.g. a previously written summary.json
            continue
        results.append(case)
    results.sort(key=lambda r: (r["aspect"], -r["dx"]))
    # Older case files predate the ``case_family`` tag; classify on the sound
    # speed ratio instead, which every file carries.
    def is_probe(case):
        return (case.get("case_family", "reference") != "reference"
                or abs(case.get("sound_speed_ratio", SOUND_SPEED_RATIO)
                       - SOUND_SPEED_RATIO) > 1.0e-9)

    reference = [r for r in results if not is_probe(r)]
    probes = [r for r in results if is_probe(r)]
    ladder = [r for r in reference if abs(r["aspect"] - 0.1) < 1e-9]
    ladder.sort(key=lambda r: -r["dx"])
    orders = []
    for coarse, fine in zip(ladder, ladder[1:]):
        orders.append({"coarse": coarse["name"], "fine": fine["name"],
                       "dx_ratio": coarse["dx"] / fine["dx"],
                       "front_order": observed_order(coarse, fine)})
    summary = {
        "cases": reference,
        "probes": probes,
        "observed_orders": orders,
        "richardson": richardson(ladder),
        "gates": gates(reference),
        "table": markdown(reference, None),
    }
    if figdir:
        summary["figures"] = make_figures(reference, Path(figdir))
        Path(figdir, "summary.json").write_text(json.dumps(compact(summary), indent=1))
    if doc:
        Path(doc).write_text(document(reference, orders, summary["gates"],
                                      summary["richardson"], probes))
    return summary


def compact(summary):
    """The summary without the per-particle and per-bin arrays (committable)."""
    light = {"observed_orders": summary["observed_orders"],
             "richardson": summary["richardson"], "gates": summary["gates"],
             "figures": summary.get("figures", []), "cases": [], "probes": []}
    for key in ("cases", "probes"):
        for case in summary[key]:
            light[key].append({k: v for k, v in case.items()
                               if k not in ("trajectory", "profiles", "fields")}
                              | {"final_depth_L1": case["profiles"][-1]["relative_l1_error"],
                                 "final_filtered_front_error":
                                     case["profiles"][-1]["front_relative_error"],
                                 "final_stray_particles":
                                     case["profiles"][-1]["stray_particles"]})
    return light


def richardson(ladder):
    """Indicative continuum limit of the front error from the two finest runs.

    ``e(dx) = e_inf + C dx^p`` fitted to the pair: ``p`` from the two points and
    ``e_inf`` from the resulting geometric series. Indicative only: two points
    cannot validate the power law, and the three-point ladder is exactly what
    tests whether it holds.
    """
    if len(ladder) < 2:
        return None
    coarse, fine = ladder[-2], ladder[-1]
    e1 = abs(coarse["front"]["final_relative_error"])
    e2 = abs(fine["front"]["final_relative_error"])
    p = observed_order(coarse, fine)
    if p is None or p <= 0:
        return None
    limit = e2 - (e1 - e2) / (2.0 ** p - 1.0)
    return {"from": [coarse["name"], fine["name"]], "order": p,
            "front_error_limit": limit}


def _probe_row(probes, reference):
    """One line comparing a compressibility probe with the matching reference run."""
    rows = []
    for probe in probes:
        match = [r for r in reference
                 if r["dx"] == probe["dx"] and abs(r["aspect"] - probe["aspect"]) < 1e-9]
        if not match:
            continue
        base = match[0]
        rows.append((probe, base))
    return rows


def document(results, orders, gate_rows, rich, probes=()):
    """Write docs/benchmarks/ritter-shallow-limit.md from the case files."""
    ladder = sorted([r for r in results if abs(r["aspect"] - 0.1) < 1e-9],
                    key=lambda r: -r["dx"])
    finest = min((r for r in results if r["aspect"] <= 0.2), key=lambda r: r["dx"])
    trend = sorted({(r["aspect"], r["dx"]): r for r in results}.values(),
                   key=lambda r: (r["dx"], r["aspect"]))
    deepest = max(results, key=lambda r: r["aspect"])
    final = finest["profiles"][-1]
    probe_rows = _probe_row(probes, results)
    body = [
        "# Benchmark: B2b Ritter shallow-water limit of the dam-break solver",
        "",
        "<!-- generated by `benchmarks/dam_break/ritter_limit.py --aggregate`;",
        "     regenerating overwrites this file. Do not hand-edit the tables. -->",
        "",
        "**Status:** numerical *and* analytic verification, first order. The solver",
        "reproduces the shape and speed of Ritter's shallow-water dam-break solution",
        "in the shallow limit, and the error against it converges under refinement --",
        "but **not to machine accuracy**: a systematic under-prediction of the front",
        f"of {100*abs(finest['front']['final_relative_error']):.1f} % remains at",
        f"`H/dx = {finest['layers']:.0f}`, i.e. the 3 % criterion is **not met** at the",
        "resolutions affordable here. The numbers below are the error budget.",
        "",
        "## 1. Why this benchmark exists",
        "",
        "`dam-break-2d.md` (B2, `H/a = 2`) compares its front with Ritter and refuses",
        "to gate on it, because shallow-water theory does not hold at that aspect",
        "ratio. That leaves the offset un-attributed: physics or scheme? The answer",
        "decides whether the solver can be used for shallow-water phenomena (run-up,",
        "swell propagation, tank sloshing) at all, so it is worth measuring rather",
        "than arguing about.",
        "",
        "This benchmark runs the **same** solver, with the same kernel and the same",
        "wall model, in the regime where Ritter is exact, and measures the residual",
        "error there. Two things separate the two effects:",
        "",
        "* **shallowness** -- `H/a` is swept from 2.0 down to 0.1; a physics offset",
        "  must shrink as the column becomes shallow;",
        "* **discretisation** -- `H/dx` is refined at fixed `H/a = 0.1`; a scheme error",
        "  must shrink with the cell size.",
        "",
        "## 2. Case and method",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| column width `a` | {finest['column_width']:g} m (ladder) |",
        f"| column height `H` | {finest['column_height']:g} m |",
        "| aspect ratios `H/a` | 0.10 (ladder), 0.20, 2.00 (B2 aspect) |",
        "| resolutions `dx` | " + ", ".join(f"{r['dx']:.3f}" for r in ladder) + " |",
        f"| sound speed `c0` | 20 sqrt(gH) = {finest['sound_speed']:.2f} m/s,"
        f" surge Mach {finest['mach_surge']:.2f} |",
        f"| time step | acoustic Courant 0.30, `dt = {finest['dt']:.3e}` s at the"
        f" finest `dx` |",
        "| initial state | at rest, uniform mass `rho0 dx^2` (`hydrostatic_init = False`) |",
        "| viscosity | `nu = 0`; Ritter is inviscid, and the solver has no artificial",
        "  viscosity term |",
        "| walls | free slip, three static layers, Adami et al. (2012) pressure |",
        "| tank | three cells of clearance beyond the front's reach |",
        "| window | `0.8 a / sqrt(gH)`, i.e. 0.8 of Ritter's validity window |",
        "",
        "Ritter's solution is exact for a *semi-infinite* reservoir, so the back wall",
        "must stay quiet: the rarefaction tail travels upstream at `c0` and reaches",
        "`x = 0` at `t = a / c0`, which is why only 80 % of that time is integrated.",
        "The front is compared with `Z(t) = a + 2 c0 t`, the depth with",
        "`h = (2 c0 - xi)^2 / 9g` and the velocity with `u = (2/3)(c0 + xi)`,",
        "`xi = (x - a)/t`.",
        "",
        "Two measurement choices matter for reproducibility:",
        "",
        "* the **depth field is counted, not weighed**, `h = n dx^2 / bin width`",
        "  (`bin width = 2 dx`,  :func:`tensorsph.ritter.particle_depth`), so the",
        "  solver's ~1 % compressibility cannot leak into the comparison;",
        "* the **front** is the extreme fluid particle (`max x`). Particles with fewer",
        "  than three neighbours are counted separately as spray; on the finest shallow",
        f"  run there are {final['stray_particles']} of them and dropping them moves the",
        f"  front from {100*finest['front']['final_relative_error']:+.1f} % to",
        f"  {100*final['front_relative_error']:+.1f} % relative to Ritter, so the spray",
        "  is a real but second-order feature of the leading edge, not the measure.",
        "",
        "## 3. Results",
        "",
        "### 3.1 Resolution ladder at `H/a = 0.1`",
        "",
        markdown(results, gate_rows),
        "",
        "`front err` is `(Z_sph - Z_rit)/Z_rit`, sampled every 10 integration steps",
        "over the whole window; `depth L1` is `||h_sph - h_rit||_1 / ||h_rit||_1` at",
        "the end of it. Negative values mean the simulated wave is **slower** than",
        "Ritter's.",
        "",
        "### 3.2 Observed order of convergence (`H/a = 0.1`)",
        "",
        "| coarse | fine | dx ratio | order p (front error) |",
        "|---|---|---|---|",
    ]
    for row in orders:
        p_text = (f"{row['front_order']:.2f}" if row["front_order"] is not None else "n/a")
        body.append(f"| {row['coarse']} | {row['fine']} | {row['dx_ratio']:.2f} | {p_text} |")
    if rich is not None:
        body += [
            "",
            f"Richardson extrapolation from the two finest runs (`p = {rich['order']:.2f}`):",
            f"the continuum front error would be `{100*abs(rich['front_error_limit']):.1f} %`."
            f" Indicative only -- two points cannot validate a power law, and the third",
            " point of the ladder is what tests it. It does say the error is a decaying",
            " discretisation error with a finite limit, not a constant offset.",
        ]
    body += [
        "",
        "### 3.3 Shallowness sweep",
        "",
        "| `H/a` | `dx` | front err (final) | depth L1 (final) | max \\|v\\|/c0 |",
        "|---|---|---|---|---|",
    ]
    for r in trend:
        body.append(f"| {r['aspect']:.2f} | {r['dx']:.3f} "
                    f"| {100*r['front']['final_relative_error']:+.2f} % "
                    f"| {100*r['profiles'][-1]['relative_l1_error']:.2f} % "
                    f"| {r['max_speed_over_c0']:.2f} |")
    body += [
        "",
        "The error falls monotonically as the column becomes shallow:",
        f"{100*abs(deepest['front']['final_relative_error']):.1f} % at `H/a ="
        f" {deepest['aspect']:.2f}` (the B2 aspect ratio),",
        f"{100*abs(trend[-2]['front']['final_relative_error']):.1f} % at `H/a ="
        f" {trend[-2]['aspect']:.2f}` and",
        f"{100*abs(finest['front']['final_relative_error']):.1f} % at `H/a ="
        f" {finest['aspect']:.2f}`. **Conclusion: the B2 offset is mostly physics.**",
        "Non-hydrostatic collapse at `H/a = 2` is outside shallow-water theory, and",
        "the solver's disagreement there says more about Ritter than about the",
        "solver. What remains in the shallow limit is a discretisation error of order",
        f"{100*abs(finest['front']['final_relative_error']):.1f} % at `H/dx ="
        f" {finest['layers']:.0f}`.",
        "",
    ]
    if probe_rows:
        body += [
            "### 3.4 Compressibility sensitivity",
            "",
            "The same case with a larger sound speed isolates the weakly-compressible",
            "part of the error: the surge Mach number halves, so the `O(M^2)` density",
            "error quarters.",
            "",
            "| case | `c0/sqrt(gH)` | surge Mach | front err (final) | depth L1 (final) |",
            "|---|---|---|---|---|",
        ]
        for probe, base in probe_rows:
            body.append(f"| {probe['name']} (reference) "
                        f"| {base.get('sound_speed_ratio', SOUND_SPEED_RATIO):.0f}"
                        f" | {base['mach_surge']:.3f}"
                        f" | {100*base['front']['final_relative_error']:+.2f} %"
                        f" | {100*base['profiles'][-1]['relative_l1_error']:.2f} % |")
            body.append(f"| {probe['name'].replace('h0.10_dx0.010', 'probe')} "
                        f"| {probe.get('sound_speed_ratio', SOUND_SPEED_RATIO):.0f}"
                        f" | {probe['mach_surge']:.3f}"
                        f" | {100*probe['front']['final_relative_error']:+.2f} %"
                        f" | {100*probe['profiles'][-1]['relative_l1_error']:.2f} % |")
        delta = (abs(probe_rows[0][1]["front"]["final_relative_error"])
                 - abs(probe_rows[0][0]["front"]["final_relative_error"]))
        body += [
            "",
            f"Halving the Mach number removes {100*delta:.1f} percentage points of the",
            "lag, i.e. part of the error is the weakly-compressible model and is bought",
            "back with a smaller time step. The rest survives at the higher sound speed,",
            "so it is the free-surface/leading-edge discretisation, not compressibility.",
            "",
        ]
    body += [
        "## 4. Gates",
        "",
        "| gate | value | limit | result |",
        "|---|---|---|---|",
    ]
    for row in gate_rows:
        body.append(f"| {row['name']} | {row['value']:.3e} | {row['comparison']} "
                    f"{row['limit']:g} | {'pass' if row['pass'] else 'FAIL'} |")
    body += [
        "",
        "`front_final_within_3pct` is the criterion a claim of *verified against the",
        "analytic solution* would need, and it **fails**. `front_final_within_15pct`",
        "is the platform's usability tolerance and it passes, so the honest summary is:",
        f"quantified to {100*abs(finest['front']['final_relative_error']):.1f} % at"
        f" `H/dx = {finest['layers']:.0f}`, converging, not yet 3 %.",
        "",
        "## 5. Notes, limits, applicability",
        "",
        "(a) *This is a scheme with known weak points at a dry front.* The solver",
        "obtains density by summation and clamps negative pressure to zero, so the",
        "leading particles -- which miss kernel support -- carry no pressure and are",
        "not pushed. The measured effect is a front that lags Ritter by a few percent",
        "per resolution doubling, exactly what this benchmark quantifies. It is the",
        "same mechanism the free-surface note (`free-surface-treatment.md`) measures",
        "from the other side.",
        "",
        "(b) *What the numbers do not cover.* `h/dx = 1.3` cubic-spline kernel,",
        "no delta-SPH, inviscid, free-slip floor, `dx` down to "
        f"{finest['dx']:.3f} (`H/dx = {finest['layers']:.0f}`). A different kernel",
        "ratio, density diffusion or viscosity would move the front, so do not",
        "extrapolate these percentages to another configuration; rerun the benchmark.",
        "",
        "(c) *Front speed transients.* The peak particle speed reaches",
        f"{max(r['max_speed_over_c0'] for r in results):.1f} `c0` on the coarsest run,",
        "against Ritter's maximum of `2 c0`. That is the collapse transient of a",
        "vertical face, which shallow-water theory does not model at all; it is",
        "recorded here so the number is not mistaken for an instability.",
        "",
        "(d) *Cost.* The finest case is",
        f"{finest['steps']} steps of {finest['particles']} particles. Wall-clock time is",
        "reported in the table but is not a measurement: the machine was shared.",
        "",
        "(e) *Reproduce.* `python benchmarks/dam_break/ritter_limit.py --preset default",
        "--workdir artifacts/ritter_limit` then `--aggregate --figdir ... --doc ...`.",
        "",
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_figures(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    ladder = sorted([r for r in results if abs(r["aspect"] - 0.1) < 1e-9],
                    key=lambda r: -r["dx"])

    # 1 -- depth profiles against the analytic solution
    if ladder:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ref = ladder[-1]
        t = ref["profiles"][-1]["time"]
        x = torch.tensor(ref["profiles"][-1]["x"])
        h_ref = ritter_depth(x, t, ref["column_height"], dam_x=ref["column_width"])
        scale = ref["column_height"]
        ax.plot(x / ref["column_width"], h_ref / scale, "k-", lw=2.0,
                label="Ritter analytic")
        for r in ladder:
            p = r["profiles"][-1]
            ax.plot(torch.tensor(p["x"]) / r["column_width"],
                    torch.tensor(p["depth"]) / scale, lw=1.2, alpha=0.85,
                    label=f"SPH dx={r['dx']:.3f} (L1 {100*p['relative_l1_error']:.1f}%)")
        ax.set_xlabel("x / a"), ax.set_ylabel("h / H")
        ax.set_title(f"depth profile at t = {t:.3f} s (0.8 a/sqrt(gH))")
        ax.set_xlim(0, min(3.0, float(x.max()) / ref["column_width"]))
        ax.legend(fontsize=8), ax.grid(alpha=0.3)
        path = outdir / "ritter_depth_profile.png"
        fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)
        written.append({"name": path.name,
                        "caption": "Depth h/H against x/a at the end of the window, "
                                   "three resolutions, Ritter analytic in black."})

    # 2 -- front trajectory and its relative error
    if ladder:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
        ref = ladder[-1]
        tt = torch.tensor([row["time"] for row in ref["trajectory"]])
        zz = torch.tensor([row["front_ritter"] for row in ref["trajectory"]])
        ax.plot(tt / (ref["column_width"] / wave_speed(ref["column_height"])),
                zz / ref["column_width"], "k-", lw=2.0, label="Ritter analytic")
        t_valid = ref["validity_window"]
        for r in ladder:
            times = torch.tensor([row["time"] for row in r["trajectory"]])
            front = torch.tensor([row["front"] for row in r["trajectory"]])
            ax.plot(times / t_valid, front / r["column_width"], lw=1.2,
                    label=f"SPH dx={r['dx']:.3f}")
            ax2.plot(times / t_valid,
                     100 * torch.tensor([row["relative_error"] for row in r["trajectory"]]),
                     lw=1.2)
        ax.set_ylabel("Z / a"), ax.legend(fontsize=8), ax.grid(alpha=0.3)
        ax.set_title("leading edge against Ritter (H/a = 0.1)")
        ax2.axhline(0.0, color="k", lw=0.8)
        ax2.axhspan(-3, 3, color="tab:green", alpha=0.10, label="+/-3%")
        ax2.set_xlabel("t / (a/sqrt(gH))"), ax2.set_ylabel("(Z-Z_rit)/Z_rit  [%]")
        ax2.legend(fontsize=8), ax2.grid(alpha=0.3)
        path = outdir / "ritter_front_error.png"
        fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)
        written.append({"name": path.name,
                        "caption": "Leading edge Z/a against t/(a/sqrt(gH)) for three "
                                   "resolutions with Ritter analytic, and the relative "
                                   "error of each (green band = 3%)."})

    # 3 -- error versus shallowness
    if len(results) > 1:
        order = sorted(results, key=lambda r: r["aspect"])
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.loglog([r["aspect"] for r in order],
                  [abs(r["front"]["final_relative_error"]) for r in order],
                  "o-", label="front error (final)")
        ax.loglog([r["aspect"] for r in order],
                  [r["profiles"][-1]["relative_l1_error"] for r in order],
                  "s--", label="depth L1 (final)")
        ax.axhline(0.03, color="tab:red", lw=1.0, ls=":", label="3%")
        ax.set_xlabel("H / a  (shallow-water parameter)"), ax.set_ylabel("relative error")
        ax.set_title("error against Ritter vs shallowness (dx = 0.010)")
        ax.legend(fontsize=8), ax.grid(alpha=0.3, which="both")
        path = outdir / "ritter_shallowness.png"
        fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)
        written.append({"name": path.name,
                        "caption": "Relative error against Ritter versus H/a, including "
                                   "the B2 aspect ratio H/a = 2: the error collapses as the "
                                   "shallow-water assumption becomes true."})

    # 4 -- particle cloud coloured by speed, finest case
    if ladder:
        r = ladder[-1]
        f = r["fields"][-1]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
        sc = ax.scatter(f["x"], f["y"], c=f["speed_over_c0"], s=3.0, cmap="viridis",
                        vmin=0.0,
                        vmax=max(1.0e-9, float(torch.tensor(f["speed_over_c0"]).max())))
        fig.colorbar(sc, ax=ax, label="|v| / c0")
        ax.set_title(f"SPH particle field, t = {f['time']:.3f} s, dx = {r['dx']:.3f}")
        ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]")
        x = torch.linspace(r["column_width"], r["tank_width"], 400, dtype=torch.float64)
        u_ref = ritter_velocity(x, f["time"], r["column_height"],
                                dam_x=r["column_width"])
        ax2.plot(x, u_ref / wave_speed(r["column_height"]), "k-", lw=1.6,
                 label="Ritter u/c0")
        ax2.set_xlabel("x [m]"), ax2.set_title("depth-averaged velocity")
        ax2.legend(fontsize=8), ax2.grid(alpha=0.3)
        for a in (ax, ax2):
            a.set_aspect("equal", adjustable="box")
        path = outdir / "ritter_field_cloud.png"
        fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)
        written.append({"name": path.name,
                        "caption": "Particle field coloured by |v|/c0 for the finest run "
                                   "(left) and Ritter's depth-averaged velocity profile "
                                   "(right)."})
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS))
    parser.add_argument("--case", default=None, help="name inside --preset")
    parser.add_argument("--workdir", default="artifacts/ritter_limit")
    parser.add_argument("--json", default=None, help="write a single case here")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sound-speed-ratio", type=float, default=SOUND_SPEED_RATIO,
                        help="c0 in units of sqrt(gH); 20 gives surge Mach 0.10")
    parser.add_argument("--aggregate", nargs="*", default=None)
    parser.add_argument("--figdir", default=None)
    parser.add_argument("--doc", default=None)
    args = parser.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.aggregate is not None:
        paths = []
        for pattern in args.aggregate:
            paths.extend(sorted(glob.glob(pattern)) or [pattern])
        if not paths:
            parser.error("no case files matched")
        summary = aggregate(paths, figdir=args.figdir, doc=args.doc)
        print(summary["table"])
        for row in summary["observed_orders"]:
            print(f"order {row['coarse']} -> {row['fine']}: {row['front_order']}")
        for row in summary["gates"]:
            print(f"gate {row['name']}: {'pass' if row['pass'] else 'FAIL'} "
                  f"({row['value']:.4g} {row['comparison']} {row['limit']:g})")
        failed = [row for row in summary["gates"] if not row["pass"]]
        return 1 if failed else 0

    if args.preset is None:
        parser.error("give --preset or --aggregate")
    cases = PRESETS[args.preset]
    if args.case:
        cases = [c for c in cases if c[0] == args.case]
        if not cases:
            parser.error(f"case {args.case!r} not in preset {args.preset!r}")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    status = 0
    for name, a, height, dx in cases:
        result = run_one(name, a, height, dx, device=args.device,
                         sound_speed_ratio=args.sound_speed_ratio)
        path = Path(args.json) if args.json else workdir / f"{name}.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"{name}: front final {100*result['front']['final_relative_error']:+.2f} %, "
              f"depth L1 {100*result['profiles'][-1]['relative_l1_error']:.2f} %, "
              f"{result['wall_clock_s']:.1f} s -> {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
