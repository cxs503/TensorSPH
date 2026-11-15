"""Attribute the FSI2 ``Cd = 0.052 vs 3.2`` deficit to a *stage* of the pipeline.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Round-3 of this anchor left the body force coefficient 60x low at ``dx = 0.02``
(``Cd_mean = 0.052`` vs Turek-Hron 3.2) and attributed the deficit, without
evidence, to "the load channel or the volume-force scale" (``VERDICT_ADMISSIBILITY.md``
section "unfinished").  Two mutually exclusive explanations are consistent with
that artifact:

``flow_dead``
    the fluid field never develops: the bulk velocity collapses away from the
    inlet buffer, so the body sees no dynamic pressure and no shear.  Then the
    load operator is innocent and the deficit is a *fluid* problem.
``load_scale``
    the flow is a real flow (bulk speed ~``U_mean``, flux conserved, fluid
    accelerating around the blockage) but the traction integral is small, i.e.
    the operator/scale up to ``F = -p n trib`` is wrong.  Then the flow is
    innocent and the operator is the defect.

The discriminator is predeclared so the outcome cannot be read off a posteriori:

* ``flux_ratio`` = downstream flux / inlet flux.  A developed through-flow has
  ``flux_ratio = 1 +- 0.05``; a dead field has ``flux_ratio << 1``.
* ``bulk_speed_ratio`` = mean bulk ``|u|`` (outside the inlet buffer and away
  from walls) / ``U_mean``.  Developed: 0.8..1.2; dead: < 0.5.
* ``accel_ratio`` = max ``|u|`` in the domain / ``U_max``.  Flow around a
  blockage of 0.24 in a channel must accelerate: > 1.05.  A field that never
  gets past the inlet profile stays at 1.00.

Nothing here changes the solver: it drives the production loose driver
(``step_fsi2``) with the recorded artifact's own parameters set
(``dx = 0.02``, ``c0 = 5``, ``U_mean = 0.2``) and samples the state.

Usage::

    PYTHONPATH=src python benchmarks/xval_fsi2/diagnose_load_channel.py \
        --dx 0.02 --steps 1200 --sample-every 100 \
        --out artifacts/xval_fsi2/diagnose_load_channel.json
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

from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2  # noqa: E402


def sample(state, case, dt: float) -> dict:
    """One diagnostic sample of the fluid state (all quantities in SI)."""
    pos, vel = state.fluid_pos, state.fluid_vel
    x, y = pos[:, 0], pos[:, 1]
    dx = case.dx
    speed = vel.norm(dim=1)

    inlet_zone = case.inlet_zone * dx
    near_wall = (y < 2.0 * dx) | (y > case.channel_height - 2.0 * dx)
    bulk = (x > inlet_zone + 4.0 * dx) & (~near_wall)
    # Downstream flux through a slab one cell wide, per unit depth: sum of
    # u_x * dx^2 / dx = u_x * dx.  Compared with the inlet flux U_mean * H.
    slab = (x > 1.0) & (x <= 1.0 + dx)
    flux_in = case.inlet_u_mean() * case.channel_height
    flux_x = float((vel[slab, 0] * dx).sum()) if bool(slab.any()) else float("nan")

    # Streamwise development: bulk x-velocity in three windows.
    windows = {}
    for name, lo, hi in (("x_0.3", 0.3, 0.5), ("x_0.7", 0.7, 0.9),
                         ("x_1.5", 1.5, 1.7)):
        sel = (x > lo) & (x < hi) & (~near_wall)
        windows[name] = float(vel[sel, 0].mean()) if bool(sel.any()) else float("nan")

    return {
        "t": float(state.time),
        "dt": dt,
        "n_fluid": int(pos.shape[0]),
        "max_speed": float(speed.max()),
        "bulk_speed_mean": float(speed[bulk].mean()) if bool(bulk.any()) else float("nan"),
        "bulk_ux_mean": float(vel[bulk, 0].mean()) if bool(bulk.any()) else float("nan"),
        "flux_x_slab": flux_x,
        "flux_in_ref": float(flux_in),
        "flux_ratio": float(flux_x / flux_in) if flux_in else float("nan"),
        "u_mean_ref": float(case.inlet_u_mean()),
        "u_max_ref": float(case.inlet_u_max),
        "windows_ux": windows,
        "tip_x": float(state.beam_nodes[-1, 0]),
        "tip_y": float(state.beam_nodes[-1, 1]),
        "history_tail": {k: (float(v[-1]) if v else None)
                         for k, v in state.history.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--sound-speed", type=float, default=5.0)
    ap.add_argument("--u-mean", type=float, default=0.2)
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="artifacts/xval_fsi2/diagnose_load_channel.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    case = similarity_corrected_case(args.u_mean, dx=args.dx,
                                     sound_speed=args.sound_speed)
    dt = case.dt()
    state = initial_state(case)
    samples = [dict(sample(state, case, dt), step=0)]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        stats = step_fsi2(state, case, dt)
        if step % args.sample_every == 0 or step == args.steps:
            s = sample(state, case, dt)
            s["step"] = step
            s["wall_s"] = time.time() - t0
            s["step_stats"] = {k: float(v) for k, v in stats.items()
                               if isinstance(v, (int, float))}
            samples.append(s)
            print(f"[{step:6d}] t={s['t']:.4f} max|u|={s['max_speed']:.4f} "
                  f"bulk={s['bulk_speed_mean']:.4f} flux_ratio={s['flux_ratio']:.3f} "
                  f"drag={stats.get('drag', float('nan')):.4g} "
                  f"cyl_drag={stats.get('cyl_drag', float('nan')):.4g}",
                  flush=True)

    out = {
        "case": {"dx": args.dx, "sound_speed": args.sound_speed,
                 "u_mean": args.u_mean, "dt": dt, "steps": args.steps,
                 "h": case.h(), "n_beam_nodes": case.n_beam_nodes,
                 "re": case.reynolds(),
                 "drag_scale_0.5_rho_U2_D": 0.5 * case.rho0
                 * case.inlet_u_mean() ** 2 * case.cylinder_d},
        "samples": samples,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())