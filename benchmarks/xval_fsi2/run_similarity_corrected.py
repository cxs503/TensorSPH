"""Similarity-corrected FSI2 coupled transient: the run that was never done.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this driver exists
----------------------
``docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md`` closes with an explicit
open item: the Turek-Hron amplitudes are only reproduced *by construction* at
the similarity-corrected parameter point, and that case has **never been run**
("group only, not yet run").  ``docs/xval_fsi2/COUPLED_BENCHMARK_REPORT.md``
§5 lists the same hole: a 40-step probe is far too short for a case whose beam
first bending period is ~2.5 s.  This driver is the missing measurement.

What is predeclared (before any result is inspected)
----------------------------------------------------
Decision
    Is the beam-tip displacement amplitude of the similarity-corrected FSI2
    case, integrated long enough to contain several beam periods, within the
    owner accuracy standard of Turek-Hron FSI2?
Alternatives
    ``strong-iqn-ils`` (IQN-ILS, Degroote 2009) -- default;
    ``strong-aitken`` (Irons-Tuck secant relaxation) -- cross-check.
Metric
    Half-range amplitudes ``(max - min)/2`` of the beam-tip displacement,
    measured only over samples with ``t >= --t-measure-start`` so the inlet
    ramp (``ramp_time``) is excluded.  Physical units [m].
Thresholds
    ``|rel_error| <= 15 %`` -> ``pass_15`` (owner acceptance);
    ``|rel_error| <= 3 %``  -> ``pass_3`` (reporting target).
    Reference = ``fsi2_reference.BENCHMARK_TIP`` (Turek-Hron FSI2).
Stopping rule
    Integrate to ``--t-end`` (default 10 s = ~4 beam periods).  If the strong
    coupling aborts or a non-finite value appears the run stops there and is
    reported as an honest FAIL with the abort step -- no threshold, window or
    resolution is retuned after looking at the result.

Environment
    torch CPU (float64), ``dx`` in metres, ``c0`` in m/s, single process.
    Cluster submission: ``run_compute`` with ``stage_dir`` = the work tree.

Outputs (nothing hand-typed into the report)
    ``artifacts/xval_fsi2/similarity_corrected_<tag>.json``  summary
    ``artifacts/xval_fsi2/similarity_corrected_<tag>.csv``   tip/drag history
    ``artifacts/xval_fsi2/similarity_corrected_<tag>.npz``   full history
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.fsi2_reference import BENCHMARK_TIP, compare_to_benchmark  # noqa: E402
from tensorsph.fsi2_similarity import (  # noqa: E402
    BENCHMARK_FSI2, similarity_corrected_case, similarity_report,
)
from tensorsph.fsi2_strong_coupling import run_fsi2_strong  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state  # noqa: E402


def build_case(dx: float, u_mean: float, c0: float, ratio: float):
    """Similarity-corrected case at the requested discretisation.

    ``ratio`` is left at the benchmark value (1.0) by default; the OSS
    regression point (10.0) is available only for reproduction runs and is
    flagged in the payload as a different physics problem.
    """
    case = similarity_corrected_case(U_mean=u_mean)
    case.dx = float(dx)
    case.sound_speed = float(c0)
    case.beam_rho = 1000.0 * float(ratio)
    case.n_beam_nodes = max(15, int(round(case.beam_length / float(dx))))
    case.max_rod_substeps = 4000
    case.rod_stability_target = 0.2
    return case


def beam_first_frequency(case) -> float:
    """Clamped-free beam first bending frequency ``1.8751^2/(2 pi) sqrt(EI/rhoA L^4)``.

    Only a scale for choosing the measurement window -- a geometry/material
    fact, not a fitted number.
    """
    t = case.beam_thickness
    inertia = t ** 3 / 12.0
    ei = case.beam_E * inertia
    rho_a = case.beam_rho * t
    return (1.8751 ** 2 / (2.0 * math.pi)) * math.sqrt(ei / (rho_a * case.beam_length ** 4))


def window_stats(series: dict, t_start: float) -> dict:
    """Half-range amplitudes and means over ``t >= t_start``."""
    t = np.asarray(series["t"], dtype=float)
    mask = t >= t_start
    n = int(mask.sum())
    if n < 2:
        return {"n_samples": n, "t_start": t_start,
                "tip_x_amplitude": 0.0, "tip_y_amplitude": 0.0,
                "tip_x_mean": float("nan"), "tip_y_mean": float("nan")}
    x = np.asarray(series["tip_x"], dtype=float)[mask]
    y = np.asarray(series["tip_y"], dtype=float)[mask]
    return {
        "n_samples": n,
        "t_start": t_start,
        "t_end": float(t[-1]),
        "tip_x_amplitude": float(0.5 * (x.max() - x.min())),
        "tip_y_amplitude": float(0.5 * (y.max() - y.min())),
        "tip_x_mean": float(x.mean()),
        "tip_y_mean": float(y.mean()),
        "tip_x_min": float(x.min()), "tip_x_max": float(x.max()),
        "tip_y_min": float(y.min()), "tip_y_max": float(y.max()),
    }


def dominant_frequency(t: np.ndarray, y: np.ndarray) -> float:
    """Largest non-DC FFT peak of a detrended series (Hz); NaN if degenerate."""
    if t.size < 8:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0.0:
        return float("nan")
    sig = y - y.mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    freqs = np.fft.rfftfreq(sig.size, d=dt)
    if spec.size < 3:
        return float("nan")
    k = int(np.argmax(spec[1:]) + 1)
    return float(freqs[k])


STATE_FIELDS = ("beam_nodes", "beam_vel", "fluid_pos", "fluid_vel",
                "fluid_mass", "load")


def save_checkpoint(path: Path, state, hist: dict, meta: dict) -> None:
    """Persist everything a restart needs (geometry is rebuilt from the case)."""
    blob = {k: getattr(state, k) for k in STATE_FIELDS if getattr(state, k) is not None}
    torch.save({
        "tensors": {k: v.detach().cpu() for k, v in blob.items()},
        "time": float(state.time),
        "history": {k: list(v) for k, v in hist.items()},
        "meta": meta,
    }, path)


def load_checkpoint(path: Path, state):
    """Restore a checkpoint into ``state``; returns (history, meta)."""
    blob = torch.load(path, weights_only=False)
    for k, v in blob["tensors"].items():
        cur = getattr(state, k)
        if cur is None:
            setattr(state, k, torch.as_tensor(v))
        else:
            setattr(state, k, torch.as_tensor(v, dtype=cur.dtype, device=cur.device))
    state.time = float(blob["time"])
    return blob["history"], blob.get("meta", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--t-measure-start", type=float, default=None,
                    help="default: 2 beam periods + ramp_time")
    ap.add_argument("--u-mean", type=float, default=0.2)
    ap.add_argument("--c0", type=float, default=5.0)
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="rho_s/rho_f; 1.0 = benchmark, 10.0 = OSS regression point")
    ap.add_argument("--solver", default="iqn-ils")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "xval_fsi2"))
    ap.add_argument("--ckpt", default="",
                    help="checkpoint file; written every --ckpt-every steps")
    ap.add_argument("--resume", action="store_true",
                    help="continue from --ckpt (same dx / c0 / ratio / solver required)")
    ap.add_argument("--ckpt-every", type=int, default=200)
    args = ap.parse_args()

    case = build_case(args.dx, args.u_mean, args.c0, args.ratio)
    sim = similarity_report(case)
    AE_VALUE = next(g["value"] for g in sim["groups"] if g["name"] == "Ae")
    f1 = beam_first_frequency(case)
    period = 1.0 / f1
    ramp = float(case.inlet_ramp_time)
    t_start = args.t_measure_start
    if t_start is None:
        t_start = ramp + 2.0 * period
    dt = case.dt()
    n_steps = int(round(args.t_end / dt))
    tag = args.tag or f"dx{args.dx:g}_{args.solver}_ratio{args.ratio:g}"

    print(f"[setup] dx={args.dx} dt={dt:.6e} steps={n_steps} "
          f"n_beam_nodes={case.n_beam_nodes} f1={f1:.4f} Hz (T={period:.3f} s) "
          f"measure from t={t_start:.3f} s", flush=True)
    print(f"[similarity] passed={sim['passed']} groups={json.dumps(sim['groups'])}",
          flush=True)

    state = initial_state(case)
    hist: dict = {k: [] for k in ("t", "tip_x", "tip_y", "drag", "lift",
                                  "max_speed", "n_fluid", "coupling_iterations",
                                  "coupling_residual", "coupling_converged")}
    ckpt = Path(args.ckpt) if args.ckpt else None
    meta = {"dx": args.dx, "c0": args.c0, "ratio": args.ratio,
            "solver": args.solver, "u_mean": args.u_mean}
    if args.resume:
        if ckpt is None or not ckpt.exists():
            raise SystemExit("--resume needs an existing --ckpt")
        hist, prev = load_checkpoint(ckpt, state)
        bad = [k for k in meta if prev.get(k, meta[k]) != meta[k]]
        if bad:
            raise SystemExit(f"checkpoint/case mismatch on {bad}")
        print(f"[resume] t={state.time:.4f} s, {len(hist['t'])} samples", flush=True)

    t0 = time.time()
    done = 0
    aborted = False
    while done < n_steps:
        n = min(args.ckpt_every, n_steps - done)
        state, chunk = run_fsi2_strong(
            case, n, interface_solver=args.solver,
            max_iter=args.max_iter, tol=args.tol, state=state, stop_on_abort=True)
        for k in chunk["history"]:
            hist.setdefault(k, []).extend(chunk["history"][k])
        done += int(chunk["n_steps"])
        aborted = bool(chunk["aborted"])
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt, state, hist, meta)
        print(f"  [chunk] done={done}/{n_steps} t={state.time:.4f}s "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
        if aborted:
            break
    wall = time.time() - t0
    result = {"aborted": aborted}

    for k in ("t", "tip_x", "tip_y", "drag", "lift", "max_speed", "n_fluid",
              "coupling_iterations", "coupling_residual", "coupling_converged"):
        hist.setdefault(k, [])
    t = np.asarray(hist["t"], dtype=float)
    finite = bool(np.all(np.isfinite(np.asarray(hist["tip_x"], dtype=float))) and
                  np.all(np.isfinite(np.asarray(hist["tip_y"], dtype=float))))
    stats = window_stats(hist, t_start)
    gate = compare_to_benchmark(stats) if stats["n_samples"] > 1 and finite else {
        "rows": [], "passed_15": False, "passed_3": False,
        "worst_rel_error": float("inf")}

    # body force coefficients over the same window
    mask = t >= t_start
    drag = np.asarray(hist["drag"], dtype=float)[mask] if mask.any() else np.array([])
    lift = np.asarray(hist["lift"], dtype=float)[mask] if mask.any() else np.array([])
    forces = {
        "n_samples": int(drag.size),
        "drag_mean": float(drag.mean()) if drag.size else float("nan"),
        "drag_amplitude": float(0.5 * (drag.max() - drag.min())) if drag.size else float("nan"),
        "lift_mean": float(lift.mean()) if lift.size else float("nan"),
        "lift_amplitude": float(0.5 * (lift.max() - lift.min())) if lift.size else float("nan"),
    }
    f_tip = dominant_frequency(t[mask], np.asarray(hist["tip_y"], dtype=float)[mask]) \
        if mask.sum() > 8 else float("nan")
    iters = np.asarray(hist["coupling_iterations"], dtype=float)
    payload = {
        "anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
        "tag": tag,
        "decision": ("Beam-tip amplitude of the similarity-corrected FSI2 case vs "
                     "Turek-Hron FSI2 within the owner 15 % standard"),
        "reference": dict(BENCHMARK_TIP),
        "benchmark_groups": dict(BENCHMARK_FSI2),
        "similarity": {k: v for k, v in sim.items() if k != "case"},
        "case": {
            "dx": args.dx, "dt": dt, "c0": args.c0, "u_mean": args.u_mean,
            "u_max_inlet": case.inlet_u_max, "viscosity": case.viscosity,
            "Re": case.reynolds(), "Ae": float(AE_VALUE),
            "beam_E": case.beam_E, "beam_rho": case.beam_rho,
            "beam_length": case.beam_length, "beam_thickness": case.beam_thickness,
            "rho_s_over_rho_f": args.ratio,
            "n_beam_nodes": case.n_beam_nodes, "n_fluid": int(state.fluid_pos.shape[0]),
            "ramp_time": ramp, "beam_f1_hz": f1, "beam_period_s": period,
        },
        "run": {
            "solver": args.solver, "max_iter": args.max_iter, "tol": args.tol,
            "requested_steps": n_steps, "completed_steps": int(t.size),
            "t_end_requested": args.t_end,
            "t_end_actual": float(t[-1]) if t.size else 0.0,
            "aborted": bool(result["aborted"]), "finite": finite,
            "wall_clock_s": round(wall, 2),
            "s_per_step": round(wall / max(1, t.size), 4),
            "mean_coupling_iterations": float(iters.mean()) if iters.size else float("nan"),
            "max_coupling_iterations": float(iters.max()) if iters.size else float("nan"),
        },
        "window": stats,
        "gate": gate,
        "forces": forces,
        "tip_y_dominant_frequency_hz": f_tip,
        "tip_y_dominant_strouhal": (f_tip * case.cylinder_d / (args.u_mean * 1.0)
                                    if math.isfinite(f_tip) else float("nan")),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    base = Path(args.out_dir) / f"similarity_corrected_{tag}"
    with open(base.with_suffix(".json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(base.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "tip_x", "tip_y", "drag", "lift", "max_speed",
                    "coupling_iterations", "coupling_residual"])
        for i in range(t.size):
            w.writerow([hist["t"][i], hist["tip_x"][i], hist["tip_y"][i],
                        hist["drag"][i], hist["lift"][i], hist["max_speed"][i],
                        hist["coupling_iterations"][i], hist["coupling_residual"][i]])
    np.savez_compressed(
        base.with_suffix(".npz"),
        **{k: np.asarray(v, dtype=float) for k, v in hist.items()})

    print(json.dumps({k: payload[k] for k in
                      ("run", "window", "forces", "tip_y_dominant_frequency_hz")},
                     indent=2), flush=True)
    for row in gate["rows"]:
        print(f"  {row['quantity']:18s} ref={row['reference']:.6g} "
              f"got={row['computed']:.6g} rel={row['rel_error']*100:+.2f}% "
              f"pass15={row['pass_15']} pass3={row['pass_3']}", flush=True)
    print(f"wrote {base}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())