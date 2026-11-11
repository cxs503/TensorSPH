#!/usr/bin/env python3
"""Why a cap at the set angle does not stay there: the colour field near a wall.

Probe only (no time stepping).  ``csf_consistent.kernel_sum_color`` is an
*unnormalised* kernel sum ``c_i = sum_j V_j W_ij`` clipped to ``[0, 1]``: it
drops below one wherever the support is incomplete, which is exactly how a free
surface is detected.  A **wall** truncates the support of the fluid particles
next to it in the same way, but nothing there is an interface -- the liquid
simply continues into the wall.  The probe measures how far ``c`` and
``|grad c|`` fall below their bulk values as a function of the distance to the
wall, on the *bulk* part of the drop (particles far from the free surface), and
counts how many of those bulk particles are misclassified as interface by
``delta_s``.

Usage:
    PYTHONPATH=src python benchmarks/contact_angle_equilibrium/probe_wall_color_deficit.py \
        [--dx 0.0015] [--theta 90] [--out benchmarks/contact_angle_equilibrium]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields, nearest_wall
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EQ = _load("contact_angle_equilibrium_run", HERE / "run.py")
CA = EQ.CA


def probe(theta_deg: float, dx: float) -> dict:
    theta = math.radians(theta_deg)
    cfg, h, pos, mass, wall, x0 = EQ.make_cap_on_wall(theta, dx, torch.device("cpu"))
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st = csf_fields(pos, mass, pairs, h, CA.RHO0, wall=wall,
                    contact_angle=theta, curvature_scheme="fit_quadratic")

    d, n_wall = nearest_wall(pos, wall)
    # bulk = far from the free surface (c ~ 1) and away from the drop axis edge
    bulk = st.color > 0.99
    # distance to the free surface for a cap: |r - a| with r from the cap centre
    a = EQ.cap_radius(theta)
    centre = torch.tensor([x0, -a * math.cos(theta)])
    r = (pos - centre).norm(dim=1)
    far_from_surface = (a - r) > 2.0 * h
    use = bulk & far_from_surface

    bins = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    rows = []
    dmax = float(st.delta_s.max())
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = use & (d / dx >= lo) & (d / dx < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append({
            "d_over_dx": [lo, hi],
            "n": n,
            "color_mean": float(st.color[sel].mean()),
            "grad_c_mean": float(st.grad_c[sel].norm(dim=1).mean()),
            "delta_s_over_max_mean": float((st.delta_s[sel] / dmax).mean()),
            "kappa_abs_mean": float(st.kappa[sel].abs().mean()),
        })

    # how many particles credited as "interface" are wall-contaminated bulk
    interface = st.delta_s > 0.05 * dmax
    contaminated = interface & use
    return {
        "theta_set_deg": theta_deg,
        "dx_m": dx,
        "h_m": h,
        "R_over_dx": CA.RADIUS / dx,
        "n_particles": int(pos.shape[0]),
        "n_bulk": int(use.sum()),
        "n_interface": int(interface.sum()),
        "n_interface_but_bulk": int(contaminated.sum()),
        "interface_bulk_fraction": float(contaminated.sum()) / max(1, int(interface.sum())),
        "bulk_color_min": float(st.color[use].min()),
        "bulk_color_fraction_below_0p99": float((st.color[use] < 0.99).sum())
        / max(1, int(use.sum())),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.0015)
    ap.add_argument("--theta", type=float, default=90.0)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = probe(args.theta, args.dx)
    print(json.dumps(res, indent=1))
    (out / "wall_color_deficit.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()