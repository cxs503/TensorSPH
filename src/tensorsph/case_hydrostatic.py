"""CASE-HYDROSTATIC -- still-water pressure profile ``p(y) = rho0 g (H - y)``.

This module is the library-side implementation of the case *anchor*
``CASE-HYDROSTATIC`` of the benchmark-case pool: it runs a resting tank with the
production WCSPH solver, reduces the particle field to a banded pressure
profile, compares it with the analytic reference and returns one JSON-able
record that carries the reference, the measurement and the gate verdict.

Reference (analytic, no table look-up)
-------------------------------------
Hydrostatic equilibrium of an incompressible fluid at rest integrates
``dp/dy = -rho0 g`` between the free surface ``y = H`` (``p = 0``) and ``y``::

    p(y) = rho0 g (H - y)

The same closed form is the target of the classic SPH hydrostatic-tank test
(Adami, Hu & Adams, *J. Comput. Phys.* 231 (2012) 7057-7075, hydrostatic
wall-boundary test) and is the reference used by this repository's own B1
benchmark (``benchmarks/hydrostatic/run.py``).  Reference values are therefore
*evaluated* from the closed form at the measured band centres -- nothing is
transcribed from a table, so there is no hand-copying step to get wrong.

What the gate measures, and what it must not
--------------------------------------------
* **Interior bands only.**  SPH summation density is truncated within ``2h`` of
  the free surface, so the top band(s) are not comparable with a hydrostatic
  pressure.  A band is excluded as soon as its *upper edge* enters the kernel
  support of the surface (overlap rule), which is stricter than the rung's
  "band centre within 2h" rule and keeps the shelf band out of the gate.
* **Time mean.**  A lattice released at rest is not in discrete equilibrium and
  the pool sloshes acoustically; the instantaneous profile oscillates by tens
  of per cent.  The measurement is the time mean over the second half of the
  run; the fluctuation is reported next to it but never gated.
* **Negative control.**  Switching gravity off in the solver while the
  reference keeps ``rho0 g`` collapses the measured pressure to ~0 and must read
  ~-100 %: a metric that passes there measures nothing.

Usage
-----
    python -m tensorsph.case_hydrostatic --json-out runs/case_hydrostatic.json
    python -m tensorsph.case_hydrostatic --quick          # CI-sized smoke run
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

from .droplet_impact import (PairListCache, apply_boundary, cell_centres,
                             hydrostatic_density, lattice_mass_unit,
                             resolve_device, rhs, wall_lattice)
from .vv.hydrostatic import HydrostaticConfig, band_table

CASE_ID = "CASE-HYDROSTATIC"
TOLERANCE_PCT = 1.0
RHO0 = 1000.0
G = 9.81
DEPTH = 0.4
H_FACTOR = 1.3

REFERENCE = {
    "kind": "analytic",
    "formula": "p(y) = rho0 * g * (H - y)",
    "derivation": "hydrostatic equilibrium dp/dy = -rho0 g integrated from p(H)=0",
    "provenance": (
        "closed form; the same hydrostatic-tank reference is used by Adami, Hu & "
        "Adams, J. Comput. Phys. 231 (2012) 7057-7075 and by this repository's "
        "benchmarks/hydrostatic/run.py. Reference values are evaluated from the "
        "closed form at the measured band centres (no table transcription)."
    ),
    "rho0": RHO0,
    "g": G,
    "H": DEPTH,
}


def analytic_pressure(y, depth: float = DEPTH, rho0: float = RHO0,
                      g: float = G):
    """Hydrostatic pressure at height ``y`` (tensor in, tensor out)."""
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=torch.float64)
    return (rho0 * g * (depth - y)).clamp_min(0.0)


def support_width(dx: float, h_factor: float = H_FACTOR) -> float:
    """Kernel support radius ``2h`` -- the free-surface exclusion band."""
    return 2.0 * h_factor * dx


GEOMETRY_RTOL = 1e-9


class HydrostaticGeometryError(ValueError):
    """The declared tank cannot be represented by the fluid lattice it builds.

    ``cell_centres`` fits ``floor(extent / dx)`` cells *inside*
    ``[0, extent]``, so the top of the column is ``floor(depth/dx) * dx`` and not
    ``depth``.  Whenever those differ, the analytic reference is evaluated at a
    height the lattice does not reach and every band reads systematically low --
    a *geometry* failure that looks like a solver failure.  The same mismatch
    moves the wall lattice (built at ``top=depth``) off the fluid surface, so the
    free-surface truncation is measured on a gap that only exists because the
    config was not representable.
    """


def _lattice_cells(extent: float, dx: float) -> int:
    """Cell count of the lattice ``cell_centres`` builds inside ``[0, extent]``."""
    return int(math.floor(extent / dx + 1e-12))


def _is_aligned(extent: float, dx: float, rtol: float = GEOMETRY_RTOL) -> bool:
    n = round(extent / dx)
    return n >= 1 and abs(extent - n * dx) <= rtol * max(abs(extent), dx)


def check_geometry(dx: float, tank_width: float | None = None,
                   depth: float = DEPTH, bands: int = 8,
                   h_factor: float = H_FACTOR, rtol: float = GEOMETRY_RTOL,
                   allow_misaligned: bool = False) -> dict:
    """Precondition gate: geometry before numbers.

    Verifies that the declared tank is representable by the lattice (both
    extents an integer number of cells), that a band is at least one cell thick,
    and that at least one band survives the free-surface overlap rule.  Raises
    :class:`HydrostaticGeometryError` listing every violated condition unless
    ``allow_misaligned`` is set, in which case the report is returned with
    ``aligned=False`` and ``problems`` filled in, so the resulting (spurious)
    band errors stay attributable to the geometry instead of to the solver.

    Returns a JSON-able report; ``reference_bias_pct`` is the uniform bias the
    analytic reference inherits from taking ``H = depth`` on a lattice whose top
    is only ``lattice_depth``: ``(lattice_depth / depth - 1) * 100``.
    """
    if not (dx > 0 and depth > 0) or (tank_width is not None and not tank_width > 0):
        raise HydrostaticGeometryError(
            f"dx={dx!r}, tank_width={tank_width!r}, depth={depth!r} must all be > 0")
    columns = _lattice_cells(tank_width, dx) if tank_width is not None else None
    rows = _lattice_cells(depth, dx)
    lattice_width = columns * dx if columns is not None else None
    lattice_depth = rows * dx
    support = support_width(dx, h_factor)
    band_thickness = depth / bands if bands > 0 else float("nan")
    gated = [b for b in range(bands)
             if (b + 1) * band_thickness <= depth - support + 1e-12] if bands > 0 else []

    problems: list[str] = []
    if not _is_aligned(depth, dx, rtol):
        problems.append(
            f"depth={depth:g} m is not an integer number of cells at dx={dx:g} m "
            f"(depth/dx = {depth / dx:.4f}): the lattice tops out at "
            f"{lattice_depth:g} m while the reference keeps H={depth:g} m, which "
            f"reads every band low by {abs(1.0 - lattice_depth / depth) * 100:.2f} %; "
            f"use depth={lattice_depth:g} m (or the next cell top "
            f"{(rows + 1) * dx:g} m)")
    if tank_width is not None and not _is_aligned(tank_width, dx, rtol):
        problems.append(
            f"tank_width={tank_width:g} m is not an integer number of cells at "
            f"dx={dx:g} m (tank_width/dx = {tank_width / dx:.4f}): the lattice is "
            f"{lattice_width:g} m wide, so the fluid/wall gap differs from the "
            f"declared {tank_width:g} m by {tank_width - lattice_width:+.3g} m")
    if rows < 3:
        problems.append(
            f"depth={depth:g} m gives {rows} lattice row(s) at dx={dx:g} m; a "
            "banded pressure profile needs at least 3")
    if columns is not None and columns < 3:
        problems.append(
            f"tank_width={tank_width:g} m gives {columns} lattice column(s) at "
            f"dx={dx:g} m; at least 3 are needed to stay clear of both walls")
    if bands < 2:
        problems.append(f"bands={bands}: a profile needs at least 2 bands")
    elif band_thickness < dx * (1.0 - rtol):
        problems.append(
            f"bands={bands} over depth={depth:g} m gives {band_thickness:g} m thick "
            f"bands at dx={dx:g} m ({band_thickness / dx:.3f} cells); a band thinner "
            "than one cell cannot hold a complete row")
    if not gated:
        problems.append(
            f"no band survives the free-surface overlap rule: depth={depth:g} m is "
            f"{depth / support:.2f} kernel supports deep (2h = {support:g} m)")

    report = {
        "dx": dx, "tank_width": tank_width, "depth": depth, "bands": bands,
        "columns": columns, "rows": rows,
        "lattice_width": lattice_width, "lattice_depth": lattice_depth,
        "support": support, "band_thickness": band_thickness,
        "band_thickness_cells": band_thickness / dx if dx > 0 else float("nan"),
        "gated_bands": gated,
        "reference_bias_pct": (lattice_depth / depth - 1.0) * 100.0,
        "aligned": not problems,
        "allow_misaligned": bool(allow_misaligned),
        "problems": problems,
    }
    if problems and not allow_misaligned:
        raise HydrostaticGeometryError(
            "declared tank is not representable by the lattice: " + "; ".join(problems))
    return report


def pressure_profile(pos, pressure, mass, depth: float = DEPTH,
                     dx: float = 0.02, bands: int = 8,
                     allow_misaligned: bool = False) -> list[dict]:
    """Mass-weighted band means of the pressure field against the closed form.

    ``pos`` / ``pressure`` / ``mass`` must be the *same* state (the time-averaged
    position with the time-averaged pressure); mixing the end-of-run position
    with the window mean pairs two different times.
    """
    # This function bands by depth only; the width is validated by the runners
    # that build the lattice (``run_case`` / ``run_initial_state``).
    check_geometry(dx=dx, tank_width=None, depth=depth, bands=bands,
                   allow_misaligned=allow_misaligned)
    cfg = HydrostaticConfig(dx=dx, depth=depth, bands=bands, gravity_reference=G)
    rows = band_table(pos, pressure, mass, cfg)
    support = support_width(dx)
    for row in rows:
        # Overlap rule: a band is contaminated as soon as its upper edge enters
        # the kernel support of the free surface, not only when its centre does.
        row["free_surface"] = bool(row["free_surface"] or row["y_high"] > depth - support)
        row["gated"] = bool(not row["free_surface"] and row["pressure_analytic"] > 0)
        row["abs_error_pct"] = abs(row["rel_error_pct"]) if row["gated"] else None
    return rows


def evaluate(rows: list[dict], tol_pct: float = TOLERANCE_PCT) -> dict:
    """Gate on the largest interior-band error; free-surface bands never enter."""
    for row in rows:
        row.setdefault("free_surface", False)
        reference = row["pressure_analytic"]
        row["rel_error_pct"] = ((row["pressure"] - reference) / reference * 100.0
                                if reference > 0 else float("nan"))
        row["gated"] = bool(not row["free_surface"] and reference > 0)
    gated = [r for r in rows if r["gated"]]
    if not gated:
        raise ValueError("no gated band: the tank is shallower than the kernel support")
    errors = [abs(r["rel_error_pct"]) for r in gated]
    worst = max(gated, key=lambda r: abs(r["rel_error_pct"]))
    return {
        "quantity": "max interior-band |p_SPH - p_ref| / p_ref",
        "limit_pct": tol_pct,
        "max_band_error_pct": max(errors),
        "mean_band_error_pct": sum(errors) / len(errors),
        "worst_band": worst["band"],
        "worst_band_y": worst["y"],
        "gated_bands": [r["band"] for r in gated],
        "free_surface_bands": [r["band"] for r in rows if r["free_surface"]],
        "pass": bool(max(errors) <= tol_pct),
    }


def benchmark_rows(rows: list[dict]) -> list[dict]:
    """ref-vs-got table (one row per gated band) for the case record."""
    out = []
    for r in rows:
        if not r["gated"]:
            continue
        out.append({
            "band": r["band"],
            "y_m": r["y"],
            "ref_Pa": r["pressure_analytic"],
            "got_Pa": r["pressure"],
            "rel_error_pct": r["rel_error_pct"],
            "within_tolerance": bool(abs(r["rel_error_pct"]) <= TOLERANCE_PCT),
        })
    return out


def _pool(cfg: HydrostaticConfig):
    device = resolve_device(cfg.device)
    xs = cell_centres(cfg.tank_width, cfg.dx, device)
    ys = cell_centres(cfg.depth, cfg.dx, device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    solver = cfg.solver_config()
    unit = lattice_mass_unit(solver)
    mass = hydrostatic_density(pos[:, 1], solver) * (unit / solver.rho0)
    return pos, mass, solver


def run_case(dx: float = 0.02, tank_width: float = 1.0, depth: float = DEPTH,
             seconds: float = 8.0, sound_speed: float = 20.0, bands: int = 8,
             average_from: float = 0.5, sample_every: int = 5,
             gravity_override: float | None = None, device: str = "cpu",
             min_particles: int = 1000, allow_misaligned: bool = False) -> dict:
    """Run the case with the production solver and return the case record.

    The solver path is :func:`tensorsph.droplet_impact.rhs` driven by the same
    integration loop as :func:`tensorsph.vv.hydrostatic.run_hydrostatic`; the
    difference is that the loop is *not* hidden inside the rung: this function
    owns it so that the case can report the time-averaged field, the profile and
    the gate together in one record.
    """
    geometry = check_geometry(dx=dx, tank_width=tank_width, depth=depth,
                              bands=bands,
                              allow_misaligned=allow_misaligned)
    cfg = HydrostaticConfig(dx=dx, tank_width=tank_width, depth=depth,
                            seconds=seconds, sound_speed=sound_speed,
                            bands=bands, average_from=average_from,
                            sample_every=sample_every, device=device,
                            hydrostatic_init=True)
    gravity = G if gravity_override is None else gravity_override
    pos, mass, solver = _pool(cfg)
    # The solver's gravity is overridden per case; the reference keeps rho0*g
    # (a control that also moves the reference proves nothing).
    solver.gravity = gravity
    fluid = pos.shape[0]
    wall = wall_lattice(solver, pos.device, top=depth)
    vel = torch.zeros_like(pos)
    h = solver.h_factor * dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=fluid)
    dt = cfg.dt()
    steps = int(math.ceil(seconds / dt))
    dt = seconds / steps

    accum_p = torch.zeros(fluid, dtype=pos.dtype, device=pos.device)
    accum_p2 = torch.zeros_like(accum_p)
    accum_x = torch.zeros_like(pos)
    accum_w = torch.zeros(fluid, dtype=pos.dtype, device=pos.device)
    samples = 0
    fluct_max = 0.0
    started = time.time()
    for step in range(steps):
        t = step * dt
        pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
        rho, pressure, acc = rhs(pos, vel, mass, solver, pairs=pairs, wall=wall, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        if t >= average_from * seconds and step % sample_every == 0:
            accum_p += pressure
            accum_p2 += pressure.square()
            accum_x += pos
            accum_w += mass
            samples += 1
            fluct_max = max(fluct_max, float(pressure.std() / max(float(pressure.mean()), 1.0)))
        vel = vel + dt * acc
        pos = pos + dt * vel
        apply_boundary(pos, vel, solver, top=depth)
    if samples == 0:
        raise RuntimeError("no pressure samples were collected")

    mean_pos = accum_x / samples
    mean_mass = accum_w / samples
    mean_p = accum_p / samples
    std_p = (accum_p2 / samples - mean_p.square()).clamp_min(0.0).sqrt()
    rows = pressure_profile(mean_pos, mean_p, mean_mass, depth, dx, bands,
                            allow_misaligned=allow_misaligned)
    for row in rows:
        mask = (mean_pos[:, 1] >= row["y_low"]) & (mean_pos[:, 1] < row["y_high"])
        row["fluctuation_pct"] = float(
            (std_p[mask] / mean_p[mask].clamp_min(1.0)).max() * 100.0) if bool(mask.any()) else None

    gate = evaluate(rows)
    record = {
        "case_id": CASE_ID,
        "kind": "dynamic_time_mean",
        "config": {
            "dx": dx, "tank_width": tank_width, "depth": depth,
            "fluid_particles": int(fluid), "min_particles": int(min_particles),
            "particle_requirement_met": bool(fluid >= min_particles),
            "columns": int(round(tank_width / dx)),
            "sound_speed": sound_speed, "gravity_solver": gravity,
            "gravity_reference": G, "seconds": seconds, "dt": dt, "steps": steps,
            "samples": samples, "sample_every": sample_every,
            "average_from": average_from, "bands": bands,
            "h": h, "support": support_width(dx), "device": device,
        },
        "reference": dict(REFERENCE, H=depth,
                          H_lattice=geometry["lattice_depth"]),
        "geometry": geometry,
        "nondimensional": {
            # Mach at the hydrostatic wave speed that the column actually sees.
            "mach_acoustic": float(math.sqrt(2.0 * G * depth) / sound_speed),
            "gravity_wave_courant": float(sound_speed * dt / dx),
            "H_over_dx": float(depth / dx),
            "W_over_H": float(tank_width / depth),
            "acoustic_period_fraction": float(
                (2.0 * depth / sound_speed) / seconds),
        },
        "gate": gate,
        "benchmark": benchmark_rows(rows),
        "profile": rows,
        "max_fluctuation_pct": float(
            max(r["fluctuation_pct"] for r in rows if r["fluctuation_pct"] is not None)),
        "wall_seconds": round(time.time() - started, 3),
    }
    return record


def run_initial_state(dx: float = 0.02, tank_width: float = 1.0,
                      depth: float = DEPTH, sound_speed: float = 20.0,
                      bands: int = 8, gravity_override: float | None = None,
                      device: str = "cpu",
                      allow_misaligned: bool = False) -> dict:
    """Deterministic ``t = 0`` record: one pressure evaluation of the lattice.

    Used as the mechanism diagnostic.  It carries no acoustic oscillation, so
    whatever error it shows is owned by the free-surface/wall truncation of the
    summation density -- the resolution ladder is read on this record, and the
    dynamic record is read for the gate.
    """
    geometry = check_geometry(dx=dx, tank_width=tank_width, depth=depth,
                              bands=bands,
                              allow_misaligned=allow_misaligned)
    cfg = HydrostaticConfig(dx=dx, tank_width=tank_width, depth=depth,
                            seconds=0.0, sound_speed=sound_speed, bands=bands,
                            device=device, hydrostatic_init=True)
    gravity = G if gravity_override is None else gravity_override
    pos, mass, solver = _pool(cfg)
    solver.gravity = gravity
    wall = wall_lattice(solver, pos.device, top=depth)
    h = solver.h_factor * dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=pos.shape[0])
    pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
    vel = torch.zeros_like(pos)
    dt = 0.3 * h / sound_speed
    _, pressure, _ = rhs(pos, vel, mass, solver, pairs=pairs, wall=wall, dt=dt)
    rows = pressure_profile(pos, pressure, mass, depth, dx, bands,
                            allow_misaligned=allow_misaligned)
    gate = evaluate(rows)
    return {
        "case_id": CASE_ID,
        "kind": "static_initial_state",
        "config": {
            "dx": dx, "tank_width": tank_width, "depth": depth,
            "fluid_particles": int(pos.shape[0]), "columns": int(round(tank_width / dx)),
            "sound_speed": sound_speed, "gravity_solver": gravity,
            "gravity_reference": G, "h": h, "support": support_width(dx),
            "bands": bands, "device": device,
        },
        "reference": dict(REFERENCE, H=depth,
                          H_lattice=geometry["lattice_depth"]),
        "geometry": geometry,
        "gate": gate,
        "benchmark": benchmark_rows(rows),
        "profile": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tensorsph.case_hydrostatic",
                                     description=__doc__)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--tank-width", type=float, default=1.0)
    parser.add_argument("--depth", type=float, default=DEPTH)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--sound-speed", type=float, default=20.0)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--average-from", type=float, default=0.5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--gravity-override", type=float, default=None)
    parser.add_argument("--tolerance-pct", type=float, default=TOLERANCE_PCT)
    parser.add_argument("--quick", action="store_true",
                        help="CI-sized smoke configuration (~seconds, not the gate)")
    parser.add_argument("--json-out", default=None, help="write the case record here")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-misaligned", action="store_true",
                        help="run a tank whose depth/width is not an integer "
                             "number of cells (default: refuse; the reference "
                             "height would not be reached by the lattice)")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)

    if args.quick:
        args.dx, args.tank_width, args.seconds = 0.05, 0.5, 1.0
        args.sample_every, args.bands = 10, 5
    record = run_case(dx=args.dx, tank_width=args.tank_width, depth=args.depth,
                      seconds=args.seconds, sound_speed=args.sound_speed,
                      bands=args.bands, average_from=args.average_from,
                      sample_every=args.sample_every,
                      gravity_override=args.gravity_override, device=args.device,
                      allow_misaligned=args.allow_misaligned)
    record["gate"] = evaluate(record["profile"], args.tolerance_pct)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1))
    gate = record["gate"]
    print(f"{CASE_ID}: {record['config']['fluid_particles']} particles, "
          f"{record['config']['steps']} steps, {record['wall_seconds']} s")
    print(f"  max interior-band error {gate['max_band_error_pct']:.4f} % "
          f"(limit {gate['limit_pct']:.2f} %, band {gate['worst_band']}) "
          f"-> {'PASS' if gate['pass'] else 'FAIL'}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    sys.exit(main())
