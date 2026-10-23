"""Run the first TensorSPH droplet-impact prototype.

From a repository checkout:
    pip install -e .
    python examples/droplet_impact_2d/run.py --steps 750
"""
import argparse
import sys

from tensorsph.droplet_impact import Config, SimulationUnstableError, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=None,
                    help="integration steps (default: Config().steps)")
parser.add_argument("--dt", type=float, default=None,
                    help="fixed time step in seconds (default: Config().dt)")
parser.add_argument("--boundary", default=None, choices=("wall", "clamp"),
                    help="boundary treatment (default: Config().boundary)")
parser.add_argument("--device", default="auto")
parser.add_argument("--no-plot", action="store_true",
                    help="skip the snapshot figure (saves ~1.5 s)")
parser.add_argument("--output", default="artifacts/droplet_impact_2d")
args = parser.parse_args()

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
print(f"Completed {len(rows) - 1} output intervals "
      f"({cfg.steps} steps, t={last['time']:.4f} s)")
print(f"  particles={last['particle_count']} max_speed={last['max_speed']:.3f} m/s "
      f"mach={last['mach_number']:.3f} courant={last['courant_number']:.3f}")
print(f"  rho=[{last['rho_min']:.1f}, {last['rho_max']:.1f}] "
      f"KE={last['kinetic_energy']:.2f} J")
