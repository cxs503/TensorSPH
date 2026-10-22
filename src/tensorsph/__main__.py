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
parser.add_argument("--version", action="version", version=f"tensorsph {__version__}")
args = parser.parse_args()

cfg = Config(device=args.device)
if args.steps is not None:
    cfg.steps = args.steps
if args.dt is not None:
    cfg.dt = args.dt
if args.boundary is not None:
    cfg.boundary = args.boundary

try:
    rows = run(cfg, args.output)
except SimulationUnstableError as exc:
    print(f"FAILED numerical acceptance: {exc}", file=sys.stderr)
    raise SystemExit(1)

last = rows[-1]
print(f"Completed {cfg.steps} steps (t={last['time']:.4f} s) -> {args.output}")
print(f"  particles={last['particle_count']} max_speed={last['max_speed']:.3f} m/s "
      f"mach={last['mach_number']:.3f} courant={last['courant_number']:.3f}")
print(f"  rho=[{last['rho_min']:.1f}, {last['rho_max']:.1f}] KE={last['kinetic_energy']:.2f} J")
