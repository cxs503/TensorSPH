"""Resolution ladder for the Young-Laplace pressure jump, at a fixed time.

Why this exists
---------------
``benchmarks/laplace_law/convergence/`` held three relaxation runs at
``dx = 2.5 / 2.0 / 1.5 mm`` produced with a *fixed step count* (380). Because
``dt ~ dx/c0`` the physical windows came out at 0.33 / 0.44 / 0.56 capillary
times: the ladder compared three different instants of a capillary oscillation
and the shortest window even fell below the repository's own minimum
(``tests/test_laplace_law.py::test_committed_verdicts_are_self_consistent``
requires ``t > 0.5 t_sigma``) -- that test was red on HEAD.

This runner regenerates the ladder at a *fixed duration* in capillary times, so
every rung is the same physical instant, and it records both pressure read-outs
(see ``relax`` in ``run.py``): the historical ``bandA`` difference, which
includes the free-surface density deficit, and the ``absolute`` core pressure,
which is the physical jump under the Tait normalisation ``p(rho0) = 0``.

Usage::

    PYTHONPATH=src python benchmarks/laplace_law/run_ladder.py            # 1.0 t_sigma
    PYTHONPATH=src python benchmarks/laplace_law/run_ladder.py --duration 2.0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERGENCE = HERE / "convergence"
SUMMARY = HERE / "resolution_ladder.json"


def load_run():
    spec = importlib.util.spec_from_file_location("laplace_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1.0,
                    help="window length in capillary times (identical for every rung)")
    ap.add_argument("--dxs", type=float, nargs="+",
                    default=[0.0025, 0.002, 0.0015])
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--sound-speed", type=float, default=20.0)
    ap.add_argument("--viscosity", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path, default=CONVERGENCE)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    args = ap.parse_args()

    run = load_run()
    CONVERGENCE.mkdir(exist_ok=True)
    rungs = []
    for dx in args.dxs:
        rep = run.relax(radius=args.radius, dx=dx, sigma=args.sigma,
                        sound_speed=args.sound_speed,
                        viscosity=args.viscosity,
                        duration_t_sigma=args.duration,
                        field_snapshot=True)
        out = args.out_dir / f"relax_dx{dx:.4f}.json"
        out.write_text(json.dumps(rep, indent=2))
        lap = rep["laplace"]
        set_ = rep["setup"]
        rung = {
            "dx_m": dx,
            "n_particles": set_["n_particles"],
            "dt_s": set_["dt_s"],
            "steps": set_["steps"],
            "t_over_t_sigma": set_["t_over_t_sigma"],
            "artifact": str(out.relative_to(HERE)),
            "bandA": {
                "delta_p_Pa": lap["delta_p_measured_Pa"],
                "rel_error_pct": lap["rel_error_pct"],
                "within_15pct": bool(lap["within_15pct"]),
            },
            "absolute": {
                "delta_p_Pa": lap["p_center_absolute_Pa"],
                "rel_error_pct": lap["rel_error_absolute_pct"],
                "rel_error_tailavg_pct": lap["rel_error_absolute_tailavg_pct"],
                "within_3pct": bool(lap["within_3pct_absolute"]),
                "within_15pct": bool(lap["within_15pct_absolute"]),
            },
            "compression_ratio_measured_over_required":
                lap["compression_ratio_measured_over_required"],
            "density_readout": {
                "rho_core_rel_std": lap["rho_core_rel_std"],
                "p_from_mean_core_density_Pa": lap["p_from_mean_core_density_Pa"],
                "rel_error_from_mean_core_density_pct":
                    lap["rel_error_from_mean_core_density_pct"],
            },
            "spurious_current_ratio": rep["spurious_current"]["ratio"],
            "curvature_band_mean_over_exact": (
                rep["curvature_final"]["band_mean_1_per_m"]
                / rep["curvature_final"]["exact_1_per_m"]),
        }
        rungs.append(rung)
        print(f"dx={dx:g} N={set_['n_particles']:4d} steps={set_['steps']:5d} "
              f"t/ts={set_['t_over_t_sigma']:.3f} "
              f"bandA {lap['rel_error_pct']:+7.1f}%  "
              f"abs {lap['rel_error_absolute_pct']:+7.1f}%  "
              f"abs_tail {lap['rel_error_absolute_tailavg_pct']:+7.1f}%  "
              f"comp_ratio {lap['compression_ratio_measured_over_required']:.2f}  "
              f"abs_of_mean {lap['rel_error_from_mean_core_density_pct']:+7.1f}%  "
              f"rho_std {lap['rho_core_rel_std']:.2e}  "
              f"v/vref {rep['spurious_current']['ratio']:.3f}", flush=True)

    ladder = {
        "benchmark": "laplace_law_resolution_ladder",
        "reference": {
            "delta_p_Pa": args.sigma / args.radius,
            "radius_m": args.radius,
            "sigma_N_per_m": args.sigma,
            "rho0": 1000.0,
            "sound_speed": args.sound_speed,
            "viscosity": args.viscosity,
        },
        "protocol": {
            "duration_t_over_t_sigma": args.duration,
            "start": "hexagonal lattice, at rest, uniform mass",
            "read_out": "absolute = p(r<0.5R) - 0; bandA = p(r<0.5R) - p(0.80-0.95R)",
            "tail_window_fraction": run.TAIL_FRACTION,
        },
        "rungs": rungs,
    }
    args.summary.write_text(json.dumps(ladder, indent=2))
    print("wrote", args.summary)


if __name__ == "__main__":
    main()
