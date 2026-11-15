"""Rigid-body CFD precursor of the FSI2 case: fluid only, structure frozen.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this exists
---------------
Round-3 of this anchor recorded ``Cd_mean = 0.052`` against the Turek-Hron value
3.2 at ``dx = 0.02`` (``docs/xval_fsi2/VIV_PERIODIC_VERDICT.md``, artifact
``viv_dx002_aitken.json``) and left the cause unattributed: "the load channel or
the volume-force scale".  The coupled driver cannot answer that question,
because at ``rho_s/rho_f = 1`` it needs interface sub-iterations, and any defect
of the interface (or of the rod) contaminates the drag history.

This driver removes the structure from the loop: the cylinder *and* the beam
boundary are held rigid (zero wall velocity, nodes never advanced), the fluid
operators are exactly the production ones (``case.fluid_config`` ``rhs``,
``body_force_from_fluid`` for the load, ``_apply_open_boundaries`` for the
through-flow), so what is measured is the fluid half of the benchmark against
the published Turek-Hron / Schaefer-Turek rigid-cylinder values.

Consequence for attribution: if the rigid body also reads ``Cd ~ 0.05`` then the
deficit is in the fluid solve, not in the FSI load channel.  If it reads
``Cd = O(1)`` the fluid is innocent and the coupled path is the defect.

Usage::

    PYTHONPATH=src python benchmarks/xval_fsi2/rigid_cfd_precursor.py \
        --dx 0.02 --steps 4000 --out artifacts/xval_fsi2/rigid_cfd_dx002.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import dam_break as di  # noqa: E402
from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_turek_hron import (  # noqa: E402
    _apply_open_boundaries,
    beam_boundary_geometry,
    body_force_from_fluid,
    initial_state,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402


def rigid_step(state, case, dt: float) -> dict:
    """One fluid step with the whole structure frozen."""
    cfg = case.fluid_config()
    cfg.dt = dt
    counters = {"injected": 0, "removed": 0}
    if case.open_boundaries:
        counters = _apply_open_boundaries(state, case, dt)

    beam_boundary, beam_normals, _beam_interp, tributary = beam_boundary_geometry(
        state.beam_nodes, case)
    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_boundary),
                     dim=0)
    m_unit = di.lattice_mass_unit(cfg)
    wall_mass = torch.cat((
        state.wall_mass_static, state.cylinder_mass,
        torch.full((beam_boundary.shape[0],), m_unit,
                   dtype=beam_boundary.dtype, device=beam_boundary.device)))
    # rigid structure: no wall velocity anywhere
    wall_vel = torch.zeros_like(wall)

    all_pos = torch.cat((state.fluid_pos, wall), dim=0)
    pairs = cell_linked_pairs(all_pos, 2.0 * case.h())
    rho, pressure, acc = di.rhs(state.fluid_pos, state.fluid_vel, state.fluid_mass,
                                cfg, pairs=pairs, wall=wall, dt=dt,
                                wall_vel=wall_vel)
    body = body_force_from_fluid(
        state.fluid_pos, rho, pressure, wall, pairs, case,
        cyl_start=state.wall_static.shape[0],
        cyl_normals=state.cylinder_normals,
        cyl_tributary=state.cylinder_tributary,
        vel=state.fluid_vel, wall_vel=wall_vel, fluid_mass=state.fluid_mass,
        beam_boundary=beam_boundary, beam_normals=beam_normals,
        beam_tributary=tributary, beam_boundary_vel=torch.zeros_like(beam_boundary))

    f_cyl = body["f_cyl"]
    f_cyl_fit = body["f_cyl_fit"]
    f_visc_cyl = body["f_visc_cyl"]
    f_visc_beam = body["f_visc_beam"]

    # pressure traction on the (rigid) beam faces, same operator as the driver
    from tensorsph.fsi2_turek_hron import (  # local import: avoid cycles
        boundary_force_from_pressure,
        boundary_pressure_from_fluid,
    )
    wall_start = state.wall_static.shape[0] + state.cylinder_particles.shape[0]
    p_wall, _w = boundary_pressure_from_fluid(
        state.fluid_pos, pressure, wall, wall_start, pairs, case.h(),
        normals=beam_normals if case.wall_impedance else None,
        vel=state.fluid_vel if case.wall_impedance else None,
        wall_vel=wall_vel, rho0=case.rho0, sound_speed=case.sound_speed,
        impedance_sign=case.wall_impedance_sign)
    f_beam = boundary_force_from_pressure(p_wall, beam_normals, tributary)

    state.fluid_vel = state.fluid_vel + dt * acc
    state.fluid_pos = state.fluid_pos + dt * state.fluid_vel
    state.fluid_pos[:, 1] = state.fluid_pos[:, 1].clamp(0.0, case.channel_height)
    state.time += dt

    # --- conservative-flow bookkeeping (an attribution quantity, not a gate)
    pos, vel = state.fluid_pos, state.fluid_vel
    x, y = pos[:, 0], pos[:, 1]
    dx = case.dx
    speed = vel.norm(dim=1)
    near_wall = (y < 2.0 * dx) | (y > case.channel_height - 2.0 * dx)
    bulk = (x > case.inlet_zone * dx + 4.0 * dx) & (~near_wall)
    slab = (x > 1.0) & (x <= 1.0 + dx)
    flux_in = case.inlet_u_mean() * case.channel_height
    flux_x = float((vel[slab, 0] * dx).sum()) if bool(slab.any()) else float("nan")
    # upstream blockage probe: is the flow decelerating in front of the cylinder?
    ahead = (x > case.cylinder_x - 0.15) & (x < case.cylinder_x - 0.02) & (~near_wall)
    behind = (x > case.cylinder_x + 0.12) & (x < case.cylinder_x + 0.25) & (~near_wall)

    f_press_body = f_cyl.sum(dim=0) + f_beam.sum(dim=0)
    f_visc_body = (f_visc_cyl.sum(dim=0) if f_visc_cyl.numel()
                   else torch.zeros(2)) + (f_visc_beam.sum(dim=0)
                                           if f_visc_beam.numel()
                                           else torch.zeros(2))
    scale = 0.5 * case.rho0 * case.inlet_u_mean() ** 2 * case.cylinder_d
    return {
        "t": float(state.time),
        "drag_press": float(f_press_body[0]),
        "lift_press": float(f_press_body[1]),
        "drag_visc": float(f_visc_body[0]),
        "lift_visc": float(f_visc_body[1]),
        "drag_total": float(f_press_body[0] + f_visc_body[0]),
        "lift_total": float(f_press_body[1] + f_visc_body[1]),
        "cd_press": float(f_press_body[0] / scale),
        "cd_total": float((f_press_body[0] + f_visc_body[0]) / scale),
        "cl_total": float((f_press_body[1] + f_visc_body[1]) / scale),
        "cd_fit": float(f_cyl_fit.sum(dim=0)[0] / scale),
        "cyl_coverage": float(body["cyl_coverage"]),
        "max_speed": float(speed.max()),
        "mass_total": float(state.fluid_mass.sum()),
        "n_fluid": int(pos.shape[0]),
        "injected": int(counters["injected"]),
        "removed": int(counters["removed"]),
        "bulk_speed_mean": float(speed[bulk].mean()) if bool(bulk.any()) else float("nan"),
        "bulk_ux_mean": float(vel[bulk, 0].mean()) if bool(bulk.any()) else float("nan"),
        "flux_ratio": float(flux_x / flux_in),
        "ux_ahead": float(vel[ahead, 0].mean()) if bool(ahead.any()) else float("nan"),
        "ux_behind": float(vel[behind, 0].mean()) if bool(behind.any()) else float("nan"),
        "u_mean_ref": float(case.inlet_u_mean()),
        "u_max_ref": float(case.inlet_u_max),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--sound-speed", type=float, default=5.0)
    ap.add_argument("--u-mean", type=float, default=0.2)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="artifacts/xval_fsi2/rigid_cfd.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    case = similarity_corrected_case(args.u_mean, dx=args.dx,
                                     sound_speed=args.sound_speed)
    dt = case.dt()
    state = initial_state(case)
    keys = ("t", "drag_press", "lift_press", "drag_visc", "lift_visc",
            "drag_total", "lift_total", "cd_press", "cd_total", "cl_total",
            "cd_fit", "cyl_coverage", "max_speed", "mass_total", "n_fluid",
            "injected", "removed", "bulk_speed_mean", "bulk_ux_mean",
            "flux_ratio", "ux_ahead", "ux_behind")
    hist = {k: [] for k in keys}
    injected_total = 0
    removed_total = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        s = rigid_step(state, case, dt)
        injected_total += s["injected"]
        removed_total += s["removed"]
        s["injected_total"], s["removed_total"] = injected_total, removed_total
        if step % args.sample_every == 0 or step == 1:
            for k in keys:
                hist[k].append(s[k])
            if step % (args.sample_every * 4) == 0 or step == 1:
                print(f"[{step:6d}] t={s['t']:.4f} cd={s['cd_total']:+.4f} "
                      f"cd_p={s['cd_press']:+.4f} "
                      f"cd_v={s['drag_visc'] / (0.5 * case.rho0 * case.inlet_u_mean() ** 2 * case.cylinder_d):+.4f} "
                      f"cl={s['cl_total']:+.4f} max|u|={s['max_speed']:.4f} "
                      f"bulk={s['bulk_speed_mean']:.4f} flux={s['flux_ratio']:.3f} "
                      f"n={s['n_fluid']} mass={s['mass_total']:.4f} "
                      f"inj/rem={s['injected_total']}/{s['removed_total']} "
                      f"wall={time.time() - t0:.0f}s", flush=True)

    out = {
        "case": {"dx": args.dx, "sound_speed": args.sound_speed,
                 "u_mean": case.inlet_u_mean(), "re": case.reynolds(),
                 "dt": dt, "steps": args.steps, "h": case.h(),
                 "drag_scale_0.5_rho_U2_D": 0.5 * case.rho0
                 * case.inlet_u_mean() ** 2 * case.cylinder_d,
                 "structure": "rigid (cylinder + beam boundary frozen)"},
        "history": hist,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())