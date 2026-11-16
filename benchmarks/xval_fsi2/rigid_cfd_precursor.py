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
import os
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
from tensorsph.farfield_force import wake_plane_drag  # noqa: E402


def _apply_open_boundaries_global(state, case, dt: float) -> dict:
    """Pre-fix open boundary: **one global** flux clock (``U_mean``) feeding a
    whole column at a time.

    Kept verbatim (behaviourally) as the *control arm* of the clock comparison:
    it delivers the right total volume but over-feeds the slow near-wall rows by
    ``U_mean / u(y_j)`` (7x at ``y = dx/2``), so their particles stack up at the
    inlet corner.  Never used in production.
    """
    dx = case.dx
    injected = removed = 0
    keep = state.fluid_pos[:, 0] < case.channel_length - 0.25 * dx
    if not bool(keep.all()):
        removed = int((~keep).sum())
        state.fluid_pos = state.fluid_pos[keep]
        state.fluid_vel = state.fluid_vel[keep]
        state.fluid_mass = state.fluid_mass[keep]
    if state.fluid_pos.shape[0] == 0:
        return {"injected": 0, "removed": removed}
    dtype, device = state.fluid_pos.dtype, state.fluid_pos.device
    state.inflow_deficit += case.inlet_u_mean() * dt
    if state.inflow_deficit >= dx:
        state.inflow_deficit -= dx
        ys = torch.arange(0.5 * dx, case.channel_height, dx, dtype=dtype, device=device)
        col = torch.stack((torch.full_like(ys, 0.5 * dx), ys), dim=-1)
        cx, cy = case.cylinder_x, case.cylinder_y
        r = 0.5 * case.cylinder_d + 0.5 * dx
        col = col[((col[:, 0] - cx) ** 2 + (col[:, 1] - cy) ** 2) > r ** 2]
        if col.shape[0]:
            from tensorsph.fsi2_turek_hron import _inlet_ramp, set_inlet_velocity
            vel = set_inlet_velocity(case, col) * _inlet_ramp(case, state.time)
            mass = torch.full((col.shape[0],), di.lattice_mass_unit(case.fluid_config()),
                              dtype=dtype, device=device)
            state.fluid_pos = torch.cat((state.fluid_pos, col), dim=0)
            state.fluid_vel = torch.cat((state.fluid_vel, vel), dim=0)
            state.fluid_mass = torch.cat((state.fluid_mass, mass), dim=0)
            injected = int(col.shape[0])
    from tensorsph.fsi2_turek_hron import _inlet_ramp, set_inlet_velocity
    on_inlet = state.fluid_pos[:, 0] < case.inlet_zone * dx
    if bool(on_inlet.any()):
        state.fluid_vel[on_inlet] = (set_inlet_velocity(case, state.fluid_pos[on_inlet])
                                     * _inlet_ramp(case, state.time))
    return {"injected": injected, "removed": removed}


# --------------------------------------------------------------------------- #
# Far-field (control-volume) drag on the same field: a second estimator whose
# error mechanism is independent of the wall-station load channel
# --------------------------------------------------------------------------- #
#: Planes of the momentum budget: ``0.10 m`` sits upstream of the cylinder
#: (surface at 0.15 m), ``0.75 m`` downstream of the beam tip (0.60 m).  Over
#: that 0.65 m the no-slip walls add a laminar friction term of order
#: ``6 mu U_mean / H * 0.65 = 4e-4 N/m`` against a load of ~6 N/m (0.007 %), so
#: the plane difference is the body load to that accuracy.
FARFIELD_UP_X = 0.10
FARFIELD_DOWN_X = 0.75


def momentum_plane_flux(pos, vel, pressure, mass, case, x_plane: float) -> dict:
    """``int (rho u_x^2 + p) dy`` across the vertical plane ``x = x_plane``.

    ``wake_plane_drag`` with ``u_inf = 0`` and ``p_ref = 0`` evaluates
    ``int (-rho u_x^2 - p) dy`` (its ``(p_ref - p)`` term is the pressure
    integral), so the momentum-plus-pressure flux is minus that value.  The
    solver's pressure datum is ``tait_pressure`` (``p = 0`` at ``rho = rho0``),
    hence ``p_ref = 0`` is the correct reference and no offset is estimated.

    The surface values are reconstructed with the module's ``fit`` arm, which is
    exact for linear fields and removes the ``sum ds_b delta_b (grad p.n_b)``
    residue of a plain bin average (see ``farfield_force`` docstring).
    """
    half = 0.5 * case.channel_height
    # plane_surface_bins bins symmetrically about y = 0; the channel is 0..H
    pos_shifted = torch.stack((pos[:, 0], pos[:, 1] - half), dim=-1)
    res = wake_plane_drag(pos_shifted, vel, pressure, rho=case.rho0, u_inf=0.0,
                          p_ref=0.0, x_plane=float(x_plane), y_range=half,
                          dx=case.dx, volumes=mass / case.rho0,
                          reconstruct="fit")
    return {"flux": -float(res["drag"]), "coverage": float(res["coverage"]),
            "n_bins": int(res["n_bins"])}


def farfield_drag(pos, vel, pressure, mass, case) -> dict:
    """Steady momentum-budget drag between the two planes (per unit thickness).

    The unsteady term ``-d/dt int rho u dV`` is *not* included (the module does
    not ship it); it cancels over a shedding period, so only the time-averaged
    far-field number is interpretable, never an instantaneous one.
    """
    up = momentum_plane_flux(pos, vel, pressure, mass, case, FARFIELD_UP_X)
    down = momentum_plane_flux(pos, vel, pressure, mass, case, FARFIELD_DOWN_X)
    return {"ff_flux_up": up["flux"], "ff_flux_down": down["flux"],
            "ff_drag": up["flux"] - down["flux"],
            "ff_coverage_up": up["coverage"], "ff_coverage_down": down["coverage"]}


#: Injection clock actually used by :func:`rigid_step`.  ``row`` is the
#: production per-streamline clock; ``global`` is the control arm above.
CLOCK = os.environ.get("FSI2_INFLOW_CLOCK", "row").strip().lower()


def rigid_step(state, case, dt: float, farfield: bool = False) -> dict:
    """One fluid step with the whole structure frozen.

    ``farfield=True`` additionally evaluates the control-volume drag
    (two plane reconstructions); it is only worth its cost at the
    sampling cadence, so the caller decides.
    """
    cfg = case.fluid_config()
    cfg.dt = dt
    counters = {"injected": 0, "removed": 0}
    if case.open_boundaries:
        if CLOCK == "global":
            counters = _apply_open_boundaries_global(state, case, dt)
        else:
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
    # inlet-corner stacking census: max number of particles sharing one inlet
    # row inside the first lattice spacing.  A clock that over-feeds the slow
    # near-wall rows piles them up here (>= 3 means stacking).
    strip = x < dx
    if bool(strip.any()):
        rid = torch.round(y[strip] / dx - 0.5).long()
        counts = torch.bincount(rid - rid.min(),
                                minlength=int(rid.max() - rid.min() + 1))
        strip_max_per_row, strip_n = float(counts.max()), int(strip.sum())
    else:
        strip_max_per_row, strip_n = 0.0, 0
    scale = 0.5 * case.rho0 * case.inlet_u_mean() ** 2 * case.cylinder_d
    ff_keys = ("ff_drag", "ff_flux_up", "ff_flux_down", "ff_coverage_up",
               "ff_coverage_down", "cd_farfield")
    if farfield:
        ff = farfield_drag(pos, vel, pressure, state.fluid_mass, case)
        ff["cd_farfield"] = float(ff["ff_drag"] / scale)
    else:
        ff = {k: float("nan") for k in ff_keys}
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
        "inlet_strip_max_per_row": strip_max_per_row,
        "inlet_strip_n": strip_n,
        "ux_ahead": float(vel[ahead, 0].mean()) if bool(ahead.any()) else float("nan"),
        "ux_behind": float(vel[behind, 0].mean()) if bool(behind.any()) else float("nan"),
        "u_mean_ref": float(case.inlet_u_mean()),
        "u_max_ref": float(case.inlet_u_max),
        **ff,
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
    ap.add_argument("--farfield", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="control-volume drag alongside the surface load")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop early and still write the artifact (0 = never)")
    ap.add_argument("--cluster", default="",
                    help="free-form label recorded in the artifact")
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
            "flux_ratio", "ux_ahead", "ux_behind",
            "inlet_strip_max_per_row", "inlet_strip_n",
            "ff_drag", "ff_flux_up", "ff_flux_down", "ff_coverage_up",
            "ff_coverage_down", "cd_farfield")
    hist = {k: [] for k in keys}
    injected_total = 0
    removed_total = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sample_now = (step % args.sample_every == 0) or step == 1
        s = rigid_step(state, case, dt,
                       farfield=args.farfield and sample_now)
        injected_total += s["injected"]
        removed_total += s["removed"]
        s["injected_total"], s["removed_total"] = injected_total, removed_total
        if args.max_seconds > 0 and (time.time() - t0) > args.max_seconds:
            print(f"stopping at step {step} (--max-seconds {args.max_seconds})",
                  flush=True)
            break
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
                 "structure": "rigid (cylinder + beam boundary frozen)",
                 "inflow_clock": CLOCK,
                 "farfield_planes": ([FARFIELD_UP_X, FARFIELD_DOWN_X]
                                     if args.farfield else None),
                 "farfield_reconstruct": "fit" if args.farfield else None,
                 "label": args.cluster,
                 "max_seconds": args.max_seconds},
        "history": hist,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())