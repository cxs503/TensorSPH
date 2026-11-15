"""Coupled transient of the FSI2 case with the body force coefficients on.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Integrates the strong-coupling driver (``rho_s/rho_f = 1`` needs interface
sub-iterations) and records, per step, the load on the *whole* body: rigid
cylinder (two arms: OSS kernel average and linear-exact plane fit) plus the
elastic beam, and the friction arm.  Output is checkpointed so a job that hits
its wall-clock limit still leaves a usable series.

Usage::

    python benchmarks/xval_fsi2/run_body_force_transient.py --dx 0.02 \
        --steps 4000 --out artifacts/xval_fsi2/body_force_transient.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_reference import summarise_body_forces  # noqa: E402
from tensorsph.fsi2_strong_coupling import step_fsi2_strong  # noqa: E402
from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--sound-speed", type=float, default=5.0)
    ap.add_argument("--u-mean", type=float, default=0.2)
    ap.add_argument("--max-iter", type=int, default=12)
    # The interface iteration cannot reduce the residual below ~1e-2 on this
    # configuration (non-smooth load transfer + acoustic stiffness, see
    # docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md §5.3).  Asking for
    # less makes every step burn max_iter chasing an unreachable floor and
    # return an unconverged last-iterate load, so the default sits at the
    # measured floor.  Override with --tol to reproduce the old behaviour.
    ap.add_argument("--tol", type=float, default=1.0e-2)
    #: interface accelerator.  ``iqn-ils`` is the library default; ``aitken``
    #: is the arm that ``docs/xval_fsi2/COUPLING_CONVERGENCE_BUDGET.md``
    #: measured to reach a 1e-10 reduction floor in <= 3 iterations (the
    #: IQN-ILS secant history cannot get below ~1e-2 on this interface).
    ap.add_argument("--interface-solver", default="iqn-ils",
                    choices=("iqn-ils", "aitken"))
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--out", default="artifacts/xval_fsi2/body_force_transient.json")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    case = similarity_corrected_case(U_mean=args.u_mean)
    case.dx = args.dx
    case.sound_speed = args.sound_speed
    # Beam discretisation follows the fluid grid, but the ``max(15, ...)`` floor
    # used to bite silently at every dx above ~0.023 m: at dx = 0.05 the beam
    # carried 15 nodes over 0.35 m (spacing 0.023 m, *finer* than the fluid),
    # while at dx = 0.02 it carried 17 (spacing 0.021 m).  The two runs of a
    # dx ladder therefore refined the fluid 2.5x and the beam 1.1x, so they were
    # not a refinement study of the coupled problem.  The clamp is kept (it
    # protects the rod solver from a starved beam) but it is now recorded in the
    # artefact so a ladder cannot be read as converged on a clamped beam.
    _raw_nodes = int(case.beam_length / case.dx)
    case.n_beam_nodes = max(15, _raw_nodes)
    beam_nodes_clamped = bool(case.n_beam_nodes != _raw_nodes)
    case.max_rod_substeps = 2000
    case.rod_stability_target = 0.2
    state = initial_state(case)
    dt = case.dt()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = ("t", "tip_x", "tip_y", "body_drag", "body_lift", "cyl_drag",
            "cyl_lift", "cyl_drag_fit", "cyl_lift_fit", "drag_viscous",
            "lift_viscous", "max_speed", "drag", "lift", "body_drag_fit",
            "body_lift_fit", "cyl_coverage", "coupling_iterations",
            "coupling_converged")  # coupling_residual appended separately
    hist = {k: [] for k in keys}
    iters, converged = [], []
    t0 = time.time()
    aborted = False
    for k in range(args.steps):
        row, stats = step_fsi2_strong(state, case, dt, max_iter=args.max_iter,
                                      tol=args.tol,
                                      interface_solver=args.interface_solver)
        for kk in keys:
            hist[kk].append(float(row.get(kk, float("nan"))))
        iters.append(int(getattr(stats, "iterations", 0)))
        converged.append(bool(getattr(stats, "converged", True)))
        # residual is not a row key: it lives on the stats object
        hist.setdefault("coupling_residual", []).append(
            float(getattr(stats, "residual_rel", float("nan"))))
        if not np.isfinite(hist["max_speed"][-1]) or hist["max_speed"][-1] > 50.0:
            aborted = True
            break
        if (k + 1) % args.checkpoint_every == 0:
            print(f"[{k+1}/{args.steps}] t={state.time:.4f}s "
                  f"Cd_fit={hist['body_drag_fit'][-1]:+.3f} "
                  f"iters={iters[-1]} wall={time.time()-t0:.0f}s", flush=True)
            payload = dict(history=hist, iterations=iters, converged=converged,
                           dt=dt, dx=args.dx, case=_case_summary(case),
                           interface_solver=args.interface_solver,
                           n_steps=len(hist["t"]), aborted=aborted,
                           wall_clock_s=time.time() - t0)
            out.write_text(json.dumps(payload))

    n = len(hist["t"])
    warm = int(args.warmup_frac * n)
    summary = {
        "avg_arm": summarise_body_forces(hist["body_drag"], hist["body_lift"],
                                         case.rho0, case.inlet_u_mean(),
                                         case.cylinder_d, warmup=warm),
        "fit_arm": summarise_body_forces(hist["body_drag_fit"],
                                         hist["body_lift_fit"], case.rho0,
                                         case.inlet_u_mean(), case.cylinder_d,
                                         warmup=warm),
        "viscous_mean": float(np.mean(hist["drag_viscous"][warm:])) if n > warm else 0.0,
        "tip_x_amplitude": float(0.5 * (max(hist["tip_x"][warm:]) - min(hist["tip_x"][warm:]))),
        "tip_y_amplitude": float(0.5 * (max(hist["tip_y"][warm:]) - min(hist["tip_y"][warm:]))),
        "mean_coupling_iterations": float(np.mean(iters)) if iters else 0.0,
        "coupling_converged_fraction": float(np.mean(converged)) if converged else 0.0,
        "wall_clock_s": time.time() - t0,
        "n_steps": n, "aborted": aborted,
    }
    payload = dict(history=hist, iterations=iters, converged=converged,
                   summary=summary, dt=dt, dx=args.dx,
                   case=_case_summary(case), n_steps=n, aborted=aborted,
                   interface_solver=args.interface_solver,
                   beam_nodes_clamped=beam_nodes_clamped,
                   wall_clock_s=time.time() - t0)
    out.write_text(json.dumps(payload))
    print(json.dumps(summary, indent=2))
    return 0


def _case_summary(case) -> dict:
    return {"dx": case.dx, "sound_speed": case.sound_speed,
            "viscosity": case.viscosity, "rho0": case.rho0,
            "beam_E": case.beam_E, "beam_rho": case.beam_rho,
            "inlet_u_max": case.inlet_u_max, "reynolds": case.reynolds(),
            "mach": case.mach(), "dt": case.dt(),
            "n_beam_nodes": case.n_beam_nodes,
            # Round-tripped so a reader of the artefact can run the
            # admissibility screen (beam containment) without guessing the
            # geometry, and can see the beam/fluid resolution ratio.
            "beam_length": case.beam_length,
            "beam_node_spacing": (case.beam_length / case.n_beam_nodes
                                  if case.n_beam_nodes else None),
            "beam_spacing_over_dx": (case.beam_length / case.n_beam_nodes
                                     / case.dx
                                     if case.n_beam_nodes and case.dx else None)}


if __name__ == "__main__":
    raise SystemExit(main())
