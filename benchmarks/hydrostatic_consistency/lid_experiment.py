#!/usr/bin/env python3
"""Does the rigid lid above the free surface change the hydrostatic profile?

``wall_shell`` closes all six faces of the fluid block, so the shipped
``run_hydrostatic`` configuration carries two layers of static wall particles
*above* the free surface (``z = H`` and ``z = H + dx``) even though its own
docstring says the walls are on "the bottom and on all four sides".  With
``h = 1.3 dx`` the kernel reaches ``2 h = 2.6 dx``, so both lid layers are
inside the support of the topmost fluid row: the layer whose pressure is
supposed to vanish is the one that gets a boundary condition it never asked for.

This script re-runs a shipped configuration with the lid removed
(``run_hydrostatic(..., lid=False)``) and writes the same profile block as
``benchmarks/threed_hydrostatic/report.json`` so the two can be diffed row by
row.  It is deliberately *not* part of the shipped benchmark: the shipped
artifact keeps its original meaning (``lid=True`` is still the default).

Cost: the shipped cases are 3692 steps of a 2880-particle float64 simulation
(~0.5 s/step on one core).  Run them on the cluster.

Usage::

    PYTHONPATH=src python benchmarks/hydrostatic_consistency/lid_experiment.py \
        --case adami-c0_20-nz16 --out benchmarks/hydrostatic_consistency/open_top.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.threed import acoustic_dt, run_hydrostatic  # noqa: E402

DX = 0.01
NX = NY = 8
T_END = 0.60

#: same table as ``benchmarks/threed_hydrostatic/run.py``
CONFIGS = {
    "adami-c0_20-nz16": (20.0, "adami", 16),
    "adami-c0_10-nz16": (10.0, "adami", 16),
    "adami-c0_5-nz16": (5.0, "adami", 16),
}


def run_case(name: str, *, lid: bool, t_end: float = T_END,
             viscosity: float = 0.02) -> dict:
    sound_speed, wall_model, nz = CONFIGS[name]
    h = 1.3 * DX
    dt = acoustic_dt(DX, h, sound_speed)
    steps = int(round(t_end / dt))
    result = run_hydrostatic(nx=NX, ny=NY, nz=nz, dx=DX, sound_speed=sound_speed,
                             viscosity=viscosity, wall_model=wall_model,
                             steps=steps, dt=dt, sample_every=max(1, steps // 40),
                             settle_fraction=0.5, lid=lid)
    profile = result.profile
    bottom = profile["rows"][0]
    half = len(result.bottom_pressure) // 2
    first = sum(result.bottom_pressure[:half]) / max(half, 1)
    second = sum(result.bottom_pressure[half:]) / max(
        len(result.bottom_pressure) - half, 1)
    return {
        "case": name,
        "lid": lid,
        "sound_speed": sound_speed,
        "wall_model": wall_model,
        "nz": nz,
        "dx": DX,
        "h": h,
        "height": result.height,
        "dt": dt,
        "steps": steps,
        "t_end": steps * dt,
        "fluid_particles": result.fluid_particles,
        "wall_particles": result.wall_particles,
        "pressure": {
            "bottom_measured": bottom["p_measured"],
            "bottom_analytic": bottom["p_analytic"],
            "bottom_rel_error": bottom["p_rel_error"],
            "core_layers": profile["core_layers"],
            "max_rel_error_core": profile["max_rel_error_core"],
            "mean_rel_error_core": profile["mean_rel_error_core"],
            "max_abs_error_over_rho0_g_H": profile["max_abs_error_over_scale"],
            "layers_within_15pct": profile["layers_within_15pct"],
            "integral_rel_error": profile["integral_rel_error"],
            "odd_even_amplitude_over_scale": profile[
                "odd_even_amplitude_over_scale"],
        },
        "profile": profile["rows"],
        "settling": {
            "bottom_pressure_first_half": first,
            "bottom_pressure_second_half": second,
            "bottom_pressure_window_drift": (
                (second - first) / first if first else float("nan")),
            "centre_of_mass_drift_over_height": (
                result.centre_of_mass_z[-1] - result.centre_of_mass_z[0])
            / (2.0 * DX + result.height),
            "wall_displacement": profile.get("wall_displacement"),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="adami-c0_20-nz16", choices=sorted(CONFIGS))
    ap.add_argument("--lid", action="store_true",
                    help="keep the closed-top wall shell (shipped behaviour)")
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--out", default=str(REPO / "benchmarks" /
                                        "hydrostatic_consistency" / "open_top.json"))
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    record = run_case(args.case, lid=args.lid, t_end=args.t_end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "hydrostatic_consistency/lid_experiment",
        "question": "does the wall_shell lid above z = H change p(z)?",
        "case": record,
        "generated_by": "benchmarks/hydrostatic_consistency/lid_experiment.py",
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(record["pressure"], indent=1))
    print("settling:", json.dumps(record["settling"], indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())