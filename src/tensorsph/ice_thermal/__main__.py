"""Command-line entry point of the ice multi-physics face: ``python -m tensorsph.ice_thermal``.

Two modes, both writing a machine-readable JSON that the campaign node
IN13W2-N07 audits:

``python -m tensorsph.ice_thermal``            full anchor set (delegates to the
                                               benchmark, ~8 min, writes figures)
``python -m tensorsph.ice_thermal --quick``    compact anchor set that runs
                                               straight out of the package
                                               (~40 s, no figures) -- used when
                                               the repository benchmark script is
                                               not on disk (installed wheel) and
                                               as the smoke entry in CI

The full mode is a thin :mod:`runpy` shim over
``benchmarks/ice_thermal_multiphysics/run.py`` so there is exactly one
implementation of the anchor set, not two.
"""
from __future__ import annotations

import argparse
import json
import platform
import runpy
import sys
import time
from pathlib import Path

from . import properties as props
from .darcy import RA_CRITICAL_EXACT, critical_rayleigh
from .enthalpy import neumann_lambda
from .sph_thermal import StefanConfig, lattice_diffusion_factor, run_stefan_sph
from .thermal_stress import biaxial_modulus, max_tensile_stress, restrained_stress


def benchmark_script() -> Path:
    """Path of ``benchmarks/ice_thermal_multiphysics/run.py`` in the source tree."""
    return Path(__file__).resolve().parents[3] / "benchmarks" / "ice_thermal_multiphysics" / "run.py"


def quick_anchors(dx: float = 0.02) -> dict:
    """Compact anchor set: one Stefan resolution, lattice check, Darcy onset, stress."""
    import torch

    t0 = time.time()
    factor, scatter = lattice_diffusion_factor()
    n = int(round(0.8 / dx))
    cfg = StefanConfig(n_particles=n, dx=dx, t_end=4.0e4)
    stefan = run_stefan_sph(cfg)
    front_rel = abs(stefan.front[-1] - stefan.front_analytic[-1]) / stefan.front_analytic[-1]

    ra, a_c, _ = critical_rayleigh(n=40, n_wavenumbers=17)
    ra_rel = abs(ra - RA_CRITICAL_EXACT) / RA_CRITICAL_EXACT

    z = torch.linspace(-0.25, 0.25, 41, dtype=torch.float64)
    t_ref = props.T_MELT - 2.0
    linear = t_ref + 10.0 * z / 0.5
    sigma = max_tensile_stress(torch.abs(restrained_stress(linear, t_ref=t_ref)))
    timoshenko = biaxial_modulus() * props.ALPHA_TH * 10.0 / 2.0
    stress_rel = abs(sigma - timoshenko) / timoshenko

    rows = [
        {"anchor": "lattice diffusion factor", "got": factor, "ref": 1.0,
         "rel_error": abs(factor - 1.0), "tolerance": 0.05, "pass": abs(factor - 1.0) < 0.05},
        {"anchor": "Stefan front vs Neumann", "got": stefan.front[-1],
         "ref": stefan.front_analytic[-1], "rel_error": front_rel,
         "tolerance": 0.03, "pass": front_rel < 0.03},
        {"anchor": "Darcy-Benard Ra_c vs 4 pi^2", "got": ra, "ref": RA_CRITICAL_EXACT,
         "rel_error": ra_rel, "tolerance": 0.01, "pass": ra_rel < 0.01},
        {"anchor": "restrained plate stress vs Timoshenko", "got": sigma, "ref": timoshenko,
         "rel_error": stress_rel, "tolerance": 0.01, "pass": stress_rel < 0.01},
    ]
    return {
        "node": "IN13W2-N07",
        "mode": "quick",
        "module": "tensorsph.ice_thermal",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "anchors": rows,
        "all_pass": all(row["pass"] for row in rows),
        "stefan_number": float(stefan.stefan_number),
        "neumann_lambda": float(stefan.lam),
        "wall_time_s": round(time.time() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tensorsph.ice_thermal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="run the compact in-package anchor set instead of the full benchmark")
    ap.add_argument("--out", default=None,
                    help="output directory (quick mode default: runs/IN13W2-N07-quick; "
                         "full mode: passed through to the benchmark)")
    ap.add_argument("--no-figures", action="store_true", help="full mode: skip figures")
    args, rest = ap.parse_known_args(argv)

    if not args.quick:
        script = benchmark_script()
        if script.is_file():
            sys.argv = [str(script)] + rest
            if args.out is not None:
                sys.argv += ["--out", str(args.out)]
            runpy.run_path(str(script), run_name="__main__")
            return 0
        print(f"[ice_thermal] benchmark script not found at {script}; "
              f"falling back to --quick", file=sys.stderr)

    out_dir = Path(args.out) if args.out else Path.cwd() / "runs" / "IN13W2-N07-quick"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = quick_anchors()
    (out_dir / "anchors_quick.json").write_text(json.dumps(res, indent=2, default=float))
    for row in res["anchors"]:
        flag = "PASS" if row["pass"] else "FAIL"
        print(f"[{flag}] {row['anchor']:<42} rel_error={row['rel_error']:.3e} "
              f"(tol {row['tolerance']:.0e})")
    print(f"all_pass={res['all_pass']} wall_time={res['wall_time_s']}s -> {out_dir / 'anchors_quick.json'}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())