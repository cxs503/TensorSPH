"""Run the free-surface verification surface and assemble its report.

Examples
--------
One part per cluster job (the parts are independent and each writes its own
``part_*.json``)::

    python scripts/run_freesurface_verification.py --part ritter --out docs/benchmarks/free-surface
    python scripts/run_freesurface_verification.py --part ladder --cells 6 10 20 --out ...
    python scripts/run_freesurface_verification.py --part sloshing --out ...
    python scripts/run_freesurface_verification.py --part solitary --out ...

and once every part is on disk::

    python scripts/run_freesurface_verification.py --part assemble --out ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorsph.freesurface import runner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", required=True,
                        choices=["ritter", "ladder", "sloshing", "solitary", "assemble"],
                        help="part to run (repeatable)")
    parser.add_argument("--out", default="docs/benchmarks/free-surface",
                        help="directory for part files, figures and report.json")
    parser.add_argument("--cells", nargs="+", type=float, default=[6.0, 10.0, 20.0],
                        help="cells per depth for the resolution ladder")
    parser.add_argument("--cells-per-depth", type=float, default=None,
                        help="override the single-run resolution")
    parser.add_argument("--viscosity", type=float, default=None)
    parser.add_argument("--periods", type=float, default=3.0,
                        help="sloshing simulated time in first-mode periods")
    parser.add_argument("--xsph", type=float, default=0.2)
    parser.add_argument("--density-diffusion", type=float, default=None,
                        help="overrides the delta-SPH coefficient (solitary run)")
    parser.add_argument("--depth", type=float, default=None,
                        help="solitary-wave still-water depth")
    parser.add_argument("--height", type=float, default=None,
                        help="solitary-wave crest height")
    parser.add_argument("--flume-length", type=float, default=None)
    parser.add_argument("--gauges", nargs="+", type=float, default=None,
                        help="solitary-wave gauge positions")
    parser.add_argument("--duration", type=float, default=None,
                        help="simulated time [s] for the solitary-wave run")
    parser.add_argument("--label", default=None, help="part label (default: base)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    leftovers = []
    for part in args.part:
        if part == "ritter":
            payload = runner.ritter_part(
                out,
                cells_per_depth=(args.cells_per_depth or 10.0),
                viscosity=(1.0e-4 if args.viscosity is None else args.viscosity),
                label=args.label or "base", device=args.device)
            print(json.dumps({k: v for k, v in payload["summary"].items()
                              if not isinstance(v, list)}, indent=2))
        elif part == "ladder":
            payload = runner.ladder_part(
                out, cells=tuple(args.cells),
                viscosity=(1.0e-4 if args.viscosity is None else args.viscosity),
                device=args.device)
            print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2))
        elif part == "sloshing":
            payload = runner.sloshing_part(
                out,
                cells_per_depth=(args.cells_per_depth or 20.0),
                viscosity=(1.0e-3 if args.viscosity is None else args.viscosity),
                periods=args.periods, device=args.device,
                label=args.label or "base")
            print(json.dumps({k: v for k, v in payload["summary"].items()
                              if not isinstance(v, list)}, indent=2))
        elif part == "solitary":
            payload = runner.solitary_part(
                out,
                cells_per_depth=(args.cells_per_depth or 8.0),
                depth=args.depth, height=args.height,
                flume_length=args.flume_length, gauges=args.gauges,
                duration=args.duration,
                density_diffusion=args.density_diffusion,
                viscosity=(1.0e-4 if args.viscosity is None else args.viscosity),
                xsph=args.xsph, device=args.device, label=args.label or "base")
            print(json.dumps({k: v for k, v in payload["summary"].items()
                              if not isinstance(v, (list, dict))}, indent=2))
        elif part == "assemble":
            report = runner.assemble(out, figures=not args.no_figures)
            leftovers.append(report)
            print(json.dumps({"measured": report["measured"],
                              "maturity": report["maturity"],
                              "figures": report.get("figures", [])}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
