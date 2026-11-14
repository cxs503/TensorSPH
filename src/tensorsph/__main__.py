"""Command-line entry point: ``python -m tensorsph``.

Equivalent to ``python examples/droplet_impact_2d/run.py``. Kept as a module
``__main__`` rather than a block inside ``droplet_impact`` so that importing the
package does not shadow the module that is being executed.
"""
import argparse
import sys

from .droplet_impact import Config, SimulationUnstableError, __version__, run

parser = argparse.ArgumentParser(prog="python -m tensorsph", description=__doc__)
parser.add_argument("--output", default="artifacts/droplet_impact_2d")
parser.add_argument("--steps", type=int, default=None,
                    help="integration steps (default: Config().steps)")
parser.add_argument("--dt", type=float, default=None,
                    help="fixed time step in seconds (default: Config().dt)")
parser.add_argument("--boundary", default=None, choices=("wall", "clamp"))
parser.add_argument("--device", default="auto")
parser.add_argument("--no-plot", action="store_true",
                    help="skip the snapshot figure (saves ~1.5 s)")
parser.add_argument("--case", default="droplet",
                    choices=("droplet", "hydrostatic", "taylor_green", "compressible_sod",
                             "stokes_second", "poiseuille_morris"),
                    help="case to run: the droplet impact example (default), the "
                         "CASE-HYDROSTATIC still-water pressure-profile case, the "
                         "CASE-TAYLOR-GREEN-SPH vortex-decay benchmark, or the "
                         "CASE-SPH-SOD-SHOCK-COMPRESSIBLE 1-D shock tube.")
parser.add_argument("--json-out", default=None,
                    help="hydrostatic case: write the case record here")
parser.add_argument("--quick", action="store_true",
                    help="hydrostatic case: CI-sized smoke configuration")
parser.add_argument("--version", action="version", version=f"tensorsph {__version__}")
args = parser.parse_args()

if args.case == "compressible_sod":
    from .compressible.sod import SodSPHConfig, run_sod_sph, evaluate_metrics
    import json
    import os

    n = 200 if args.quick else 800
    cfg = SodSPHConfig(n_particles=n)
    res = run_sod_sph(cfg, output_times=[0.2])
    met = evaluate_metrics(res["snapshots"][-1], cfg)
    print(f"CASE-SPH-SOD-SHOCK-COMPRESSIBLE (N={n}): "
          f"max|rel err|={met['max_abs_rel_err']*100:.2f}%  "
          f"<5%: {met['pass_5pct']}  <3%: {met['pass_3pct']}  "
          f"steps={res['steps']}")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump({"config": res["config"], "steps": res["steps"],
                       "metrics": {k: v for k, v in met.items() if k != "windows"}},
                      fh, indent=1, default=float)
        print(f"wrote {args.json_out}")
    raise SystemExit(0 if met["pass_5pct"] else 1)

if args.case == "taylor_green":
    from tensorsph.taylor_green import main as taylor_green_main

    argv = ["--out", args.output]
    if args.no_plot:
        argv.append("--no-figures")
    raise SystemExit(taylor_green_main(argv))

if args.case == "poiseuille_morris":
    from .case_poiseuille_morris import main as poiseuille_morris_main

    argv = ["--out", args.output, "--no-sensitivity"]
    if args.quick:
        argv.append("--quick")
    raise SystemExit(poiseuille_morris_main(argv))

if args.case == "hydrostatic":
    from .case_hydrostatic import main as hydrostatic_main

    argv = ["--quick"] if args.quick else []
    if args.json_out:
        argv += ["--json-out", args.json_out]
    raise SystemExit(hydrostatic_main(argv))

if args.case == "stokes_second":
    from benchmarks.stokes_second.run import run_one
    import torch
    torch.set_num_threads(1)
    dev = "cpu" if args.device == "auto" else args.device
    res = run_one(12, dev, periods=3.0, measure_periods=1.0,
                  sound_speed=4.0, label="cli")
    m = res["metrics"]
    print(f"CASE-SPH-STOKES-2ND-PROBLEM (delta/dx=12): "
          f"L2={m['amplitude_l2_half_space_pct']:.3f}%  "
          f"decay_err={m['decay_rate_error_pct']:.3f}%  "
          f"delta_ratio={m['delta_ratio']:.5f}  "
          f"rows_in_layer={m['n_rows_stokes_layer']}")
    raise SystemExit(0 if m["amplitude_l2_half_space_pct"] < 5.0 else 1)

cfg = Config(device=args.device)
if args.steps is not None:
    cfg.steps = args.steps
if args.dt is not None:
    cfg.dt = args.dt
if args.boundary is not None:
    cfg.boundary = args.boundary

try:
    rows = run(cfg, args.output, plot=not args.no_plot)
except SimulationUnstableError as exc:
    print(f"FAILED numerical acceptance: {exc}", file=sys.stderr)
    raise SystemExit(1)
except ValueError as exc:
    # A configuration the solver refuses is a user error, not a crash.
    print(f"FAILED configuration check: {exc}", file=sys.stderr)
    raise SystemExit(2)

last = rows[-1]
print(f"Completed {cfg.steps} steps (t={last['time']:.4f} s) -> {args.output}")
print(f"  particles={last['particle_count']} max_speed={last['max_speed']:.3f} m/s "
      f"mach={last['mach_number']:.3f} courant={last['courant_number']:.3f}")
print(f"  rho=[{last['rho_min']:.1f}, {last['rho_max']:.1f}] KE={last['kinetic_energy']:.2f} J")
