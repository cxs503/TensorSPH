"""Which knob moves the plane-Couette wall-shear reading: the grid or the clock?

Why this exists
---------------
The method-selection registry banks ``x_couette_wall_shear`` = 32.6 % as
``wcsph_wall_no_slip``'s worst evidence.  That value comes from
``benchmarks/couette/out/results.json``, which ``run.py --quick`` produced at
``t_end = 0.3 H^2/nu`` -- a quarter of the ``DESIGN.md`` protocol
(``t_end = 1.2 H^2/nu``).  ``benchmarks/couette/viscous_time_convergence.json``
already shows that at the *same* grid (``H/dx = 8``) the bottom-wall reading
moves 32.6 % -> 22.6 % -> 4.34 % as the terminal time is raised
0.3 -> 0.6 -> 1.2 viscous times, i.e. the over-tolerance value is a startup
reading, not a discretisation symptom.  The adjudication (m22787/m22957) still
grades the record on the *resolution* axis (``resolution_pair_only`` because the
committed design ladder has only two rungs), so the two knobs were never
separated in one artefact with one estimator, one lattice and one solver.

This driver separates them: a 2 x 3 matrix of ``protocol`` x ``H/dx`` where
``protocol`` is the terminal time in units of ``H^2/nu``.  Same ``run_one`` the
shipped benchmark uses, same estimator, same gates; nothing is calibrated
against the run.

Predeclared questions (fixed before the numbers were read):

Q1  At the design protocol, does the bottom-wall reading stay inside the owner
    15 % standard when the ladder is extended to ``H/dx = 32`` (the rung m22957
    named as missing)?  -> resolution axis answerable on 3 rungs.
Q2  At the quick protocol, does refining the grid bring the 32.6 % reading
    inside the standard?  -> if yes, the record is a resolution symptom; if no,
    the record's cause is the terminal time.
Q3  At a fixed grid, does raising the terminal time bring the reading inside
    the standard?  -> the fixed-grid column of the same matrix.

Each cell records the run's own steadiness diagnostics (``steady_drift``), so a
cell can be called *admissible* only when the flow is actually steady, exactly
as ``viscous_time_convergence.py`` defines it.  A pass that rests on a
non-steady cell is reported as such; it is not silently promoted.

Run::

    PYTHONPATH=src python benchmarks/couette/protocol_matrix.py
    PYTHONPATH=src python benchmarks/couette/protocol_matrix.py --cached   # re-render only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import run as couette  # noqa: E402  (benchmarks/couette/run.py)

OUT_DIR = HERE / "out_protocol_matrix"
RESULT_PATH = OUT_DIR / "matrix.json"

#: Terminal-time protocols swept, in units of ``H^2/nu``.  ``quick`` is the
#: ``run.py --quick`` override that produced the registry record; ``design`` is
#: the ``DESIGN.md`` protocol.
PROTOCOLS = (("quick", 0.3), ("design", 1.2))

#: Ladder rungs.  32 is the rung m22957 named as the missing evidence.
RESOLUTIONS = (8, 16, 32)

#: Gate the steady-state diagnostic must clear for a cell to count as steady.
STEADY_GATE = couette.GATES["steady_drift"]

#: Metrics kept per cell (the rest of ``run_one``'s payload is bulk trace data).
KEPT_METRICS = (
    "wall_shear_measured", "wall_shear_analytic", "wall_shear_ratio",
    "wall_shear_ratio_top", "interior_l2", "flux_ratio",
    "slip_length_bottom_over_dx", "line_fit_slope_ratio",
)


def run_cell(protocol: str, viscous_times: float, resolution: int,
             device) -> dict:
    """One matrix cell, through the shipped benchmark's own ``run_one``."""
    height = couette.HEIGHT
    te = viscous_times * height ** 2 / couette.VISCOSITY
    label = f"{protocol}-r{resolution}"
    started = time.time()
    row = couette.run_one(resolution, couette.VISCOSITY, couette.WALL_SPEED,
                          couette.FRICTION, height, te,
                          couette.AVERAGE_FRACTION * te, device, label=label)
    metrics = row["metrics"]
    analytic = float(metrics["wall_shear_analytic"])
    measured = float(metrics["wall_shear_measured"])
    kept = {name: metrics[name] for name in KEPT_METRICS if name in metrics}
    drift = float(row["steady_drift"])
    return {
        "label": label,
        "protocol": protocol,
        "viscous_times": viscous_times,
        "resolution_H_over_dx": resolution,
        "dx": row["dx"],
        "t_end": row["t_end"],
        "n_fluid": row["n_fluid"],
        "n_wall": row["n_wall"],
        "steps": row["steps"],
        "n_averaged_samples": row["n_averaged_samples"],
        "steady_drift": drift,
        "steady_gate": STEADY_GATE,
        "admissible": drift <= STEADY_GATE,
        "wall_shear_error_pct_bottom": abs(measured - analytic) / abs(analytic) * 100.0,
        "wall_clock_s": row["wall_clock_s"],
        "metrics": kept,
    }


def build(device_override: str | None = None) -> dict:
    device = couette.resolve_device(device_override or "auto")
    cells = []
    for protocol, viscous_times in PROTOCOLS:
        for resolution in RESOLUTIONS:
            print(f"[protocol-matrix] {protocol} H/dx={resolution} "
                  f"(t_end={viscous_times * couette.HEIGHT ** 2 / couette.VISCOSITY:.3f} s)",
                  flush=True)
            cell = run_cell(protocol, viscous_times, resolution, device)
            print(f"[protocol-matrix]   tau_err={cell['wall_shear_error_pct_bottom']:.3f}% "
                  f"drift={cell['steady_drift']:.2e} "
                  f"admissible={cell['admissible']} "
                  f"({cell['wall_clock_s']:.0f}s, {cell['steps']} steps)", flush=True)
            cells.append(cell)
    return {
        "case": "plane_couette_channel",
        "quantity": "bottom-wall shear |1 - tau_SPH/tau_exact| [%]",
        "reference": couette.REFERENCE_NOTE,
        "height": couette.HEIGHT,
        "viscosity": couette.VISCOSITY,
        "wall_speed": couette.WALL_SPEED,
        "wall_friction": couette.FRICTION,
        "sound_speed": couette.SOUND_SPEED,
        "rho0": couette.RHO0,
        "h_factor": couette.H_FACTOR,
        "wall_layers": couette.WALL_LAYERS,
        "columns": couette.COLUMNS,
        "average_fraction": couette.AVERAGE_FRACTION,
        "gates": couette.GATES,
        "protocols": {name: vt for name, vt in PROTOCOLS},
        "resolutions": list(RESOLUTIONS),
        "generated_by": "benchmarks/couette/protocol_matrix.py",
        "torch": __import__("torch").__version__,
        "cells": cells,
        "wall_clock_total_s": sum(c["wall_clock_s"] for c in cells),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true",
                        help="re-render from an existing matrix.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cached:
        payload = json.loads(RESULT_PATH.read_text())
        print(f"[protocol-matrix] cached: {len(payload['cells'])} cells")
        return 0
    payload = build(args.device)
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[protocol-matrix] wrote {RESULT_PATH} "
          f"({payload['wall_clock_total_s']:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())