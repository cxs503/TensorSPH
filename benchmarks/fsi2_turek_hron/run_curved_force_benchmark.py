"""Force-from-fluid on the *curved* FSI2 cylinder (the benchmark body).

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.
TensorBench contract: this program prints **one** JSON document on stdout
(``extract.format = "json_stdout"``); every number the platform judges is read
from it by dot-pointer, and the platform recomputes ref/got/error itself.

What is judged, and against what
--------------------------------
The existing ``fsi2_turek_hron`` and ``fsi2_gate_load`` workloads validate the
force-from-fluid on *flat* walls (parabolic inlet, hydrostatic gate face).  The
``curved_disk_shear`` case in ``benchmarks/xval_fsi2`` exercises only the
*viscous* branch on the curved body.  This workload closes the gap on the
**pressure** branch of the "force from fluid" on the curved benchmark body
(``fsi2.h:14-35``, cylinder ``r = 0.5`` at ``(2, 2)``):

``curved_uniform_net_force``
    A closed disk in a *uniform* pressure field ``p = 1`` must feel **zero**
    net force -- ``oint p n dl = p oint n dl = 0`` over the closed surface,
    independent of how the normals are discretised.  This is the strongest
    curvature invariant: a spurious net force here would be a curved-surface
    integration error, not a physics one.
    Reference: ``oint n dl = 0`` (closed-surface identity, exact).  Tolerance:
    machine precision (1e-9).

``curved_hydrostatic_buoyancy``
    The same closed disk in a *linear* (hydrostatic) field
    ``p = 1 - (y - cy)`` (``c = (0, -1)``) must feel the Archimedes buoyancy
    ``-c A = +A`` (``A = pi R^2`` per unit width) -- the "force from fluid"
    that actually drives the FSI2 beam.
    Reference: ``F = rho g A`` (Archimedes, per unit width).  Tolerance: 3 %
    (the owner's precision standard) at the coarsest resolution, 1 % at the
    finer ones.

The **panel** operator (``panel_pressure_force``, the production transfer path)
is the arm under test.  The OSS one-sided branch is *not* a metric here: it
carries the documented +24..+29 % systematic over-load on a single surface
layer (pinned in ``test_oss_operator_signs_and_documented_overload``), so it is
not a clean closed-surface integrator and would fail the uniform-field
invariant by design.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

import cases as C  # noqa: E402
from tensorsph.fsi_force_from_fluid import (ForceFromFluidConfig,  # noqa: E402
                                            panel_pressure_force)

BENCHMARK_ID = "SPH-FSI2-CURVED-FORCE-FROM-FLUID"
BENCHMARK_VERSION = "1.0.0"
WORKLOAD_ID = "fsi2-curved-force-from-fluid"
WORKLOAD_VERSION = "1.0.0"


def _revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _log(msg: str) -> None:
    print(f"[fsi2-cff] {msg}", file=sys.stderr, flush=True)


def _cfg(case):
    return ForceFromFluidConfig(dim=2, h=case.h, kernel="wendland")


def _uniform_net(case) -> float:
    """Panel pressure force on the closed disk in a uniform field p = 1."""
    p = torch.ones_like(case.pressure_fluid)
    f = panel_pressure_force(case.pos_solid, case.vol_solid, case.normal_solid,
                             case.pos_fluid, case.vol_fluid, p, _cfg(case))
    return float(f.sum(dim=0).norm())


def _hydrostatic_buoyancy(case) -> tuple[float, float]:
    """Panel pressure force on the closed disk in p = 1 - (y - cy)."""
    cy = float(case.pos_solid[:, 1].mean())
    p = 1.0 - (case.pos_fluid[:, 1] - cy)
    f = panel_pressure_force(case.pos_solid, case.vol_solid, case.normal_solid,
                             case.pos_fluid, case.vol_fluid, p, _cfg(case))
    net = f.sum(dim=0)
    return float(net[0]), float(net[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional JSON copy path")
    ap.add_argument("--dx", type=float, default=0.05,
                    help="particle spacing [m] (default 50 mm)")
    args = ap.parse_args()

    started = time.time()
    case = C.curved_disk_shear(dx=args.dx)
    R = 0.5
    A = math.pi * R * R

    # V-uniform: zero net force on the closed curved body
    t0 = time.time()
    uniform_net = _uniform_net(case)
    _log(f"V-uniform: |net|={uniform_net:.3e} (ref 0, closed-surface identity)")

    # V-hydro: Archimedes buoyancy on the closed curved body
    t0 = time.time()
    fx, fy = _hydrostatic_buoyancy(case)
    rel = abs(fy - A) / A
    _log(f"V-hydro: F_y={fy:.6f} A={A:.6f} rel={rel*100:.3f}%")

    doc = {
        "schema_version": "1.0.0",
        "run_id": f"fsi2-cff-{int(started)}-{_revision()[:8]}",
        "benchmark": {"benchmark_id": BENCHMARK_ID, "version": BENCHMARK_VERSION},
        "implementation": {"project": "TensorSPH", "revision": _revision()},
        "workload": {"workload_id": WORKLOAD_ID, "version": WORKLOAD_VERSION,
                     "host": platform.node(), "device": "cpu",
                     "argv": sys.argv},
        "execution": {"status": "ok", "wall_seconds": None},
        "metrics": {
            "curved_uniform_net_force": {
                "got": uniform_net, "ref": 0.0,
                "unit": "N/m", "rel_error": None,
                "note": "closed-surface identity oint n dl = 0; machine precision"},
            "curved_hydrostatic_buoyancy": {
                "got": fy, "ref": A,
                "unit": "N/m", "rel_error": rel,
                "note": "Archimedes F = rho g A, A = pi R^2 per unit width"},
        },
        "validation": {
            "refs": {
                "uniform": "oint p n dl = p oint n dl = 0 (closed-surface identity)",
                "hydrostatic": "F = rho g A (Archimedes, per unit width)",
            },
            "not_judged": {
                "oss_one_sided_branch": (
                    "carries the documented +24..+29 % systematic over-load on a "
                    "single surface layer; not a clean closed-surface integrator, "
                    "so excluded from the uniform-field invariant by design."),
            },
        },
        "details": {
            "geometry": {"radius": R, "area_per_width": A,
                         "n_solid": int(case.pos_solid.shape[0]),
                         "source": "fsi2.h:14-35 (cylinder r=0.5 at (2,2))"},
            "curved_uniform": {"net_force": uniform_net},
            "curved_hydrostatic": {"F_x": fx, "F_y": fy, "rel_error": rel},
        },
    }
    doc["execution"]["wall_seconds"] = time.time() - started

    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
