#!/usr/bin/env python3
"""One-at-a-time sensitivity runs for the mirror-wall Poiseuille case.

Every run is the same case as ``run.py`` at ``H/dx = 12`` with exactly one
parameter changed, so the effect of each parameter is separable:

    g_x x2        body force doubled      (peak velocity doubles, Re doubles)
    nu  x2        viscosity doubled       (peak halves, Re halves)
    L   x2        domain length doubled   (x-periodicity independence check)

The decision this supports: the 5 % agreement in the main ladder is a property
of the scheme, not of one lucky parameter set.

    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/run_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from benchmarks.poiseuille.run import make_config
from benchmarks.poiseuille_mirror.run import evaluate, gates, run_case_mirror
from tensorsph.droplet_impact import resolve_device

REPO = Path(__file__).resolve().parents[2]
HEIGHT = 0.1
NU = 0.01
G = 9.81
RHO0 = 1000.0

VARIATIONS = [
    ("g_x x2", {"body_force": 2 * G}),
    ("nu x2", {"viscosity": 2 * NU}),
    ("L x2", {"columns": 24}),
]


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    out = Path(args[0]) if args else REPO / "benchmarks/poiseuille_mirror/out"
    torch.set_num_threads(1)
    device = resolve_device("cpu")
    resolution = 12
    dx = HEIGHT / resolution

    rows = []
    for label, override in VARIATIONS:
        body_force = override.get("body_force", G)
        viscosity = override.get("viscosity", NU)
        columns = override.get("columns", 12)
        length = columns * dx
        cfg = make_config(dx, viscosity, 1.0, device)
        print(f"[sensitivity] {label} ...", flush=True)
        case = run_case_mirror(cfg, HEIGHT, length, body_force, t_end=1.0,
                               t_average=0.2, device=device)
        res = evaluate(case, HEIGHT, body_force, viscosity, RHO0, resolution)
        res["label"] = label
        res["overrides"] = override
        res["length"] = length
        res["gates"] = gates(res)
        print(f"[sensitivity]   u_max {res['u_max_error_pct']:+.2f}%  "
              f"rms {res['profile_rms_interior_pct']:.2f}%  tau "
              f"{res['wall_shear_bottom_error_pct']:+.2f}%  "
              f"Re {res['reynolds_number']:.1f}  "
              f"({res['wall_clock_s']:.1f}s)", flush=True)
        rows.append(res)

    payload = {
        "case": "plane_poiseuille_channel_mirror_wall",
        "baseline": {"body_force": G, "viscosity": NU, "columns": 12,
                     "resolution_H_over_dx": resolution},
        "variations": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(payload, indent=2))
    print(f"[sensitivity] wrote {out / 'sensitivity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
