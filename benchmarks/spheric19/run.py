"""Run the SPHERIC Test 19 coupled fixture from two installed solver packages."""
import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from tensorsph.spheric19 import GateMotion, simulate

HERE = Path(__file__).resolve().parent


def first_rise(times, values, target):
    crossings = np.flatnonzero((values[:-1] < target) & (values[1:] >= target))
    if len(crossings) == 0:
        return None
    i = crossings[0]
    return float(np.interp(target, values[i:i+2], times[i:i+2]))


def compare(history, reference):
    ts, *markers = np.asarray(history).T
    scores = {}
    for idx, curve in enumerate(markers, start=1):
        data = reference[reference["marker"] == idx]
        if len(data) == 0 or ts[0] > min(data["t_s"]) or ts[-1] < max(data["t_s"]):
            raise ValueError("simulation must span all three measured marker records")
        residual = np.interp(data["t_s"], ts, curve) - data["dx_m"]
        rmse = float(np.sqrt(np.mean(residual**2)))
        peak = float(max(abs(data["dx_m"])))
        level = peak / 2
        scores[str(idx)] = {
            "points": len(data), "rmse_m": rmse, "rmse_over_measured_peak": rmse / peak,
            "experiment_sampled_peak_m": peak,
            "experiment_half_rise_s": first_rise(data["t_s"], data["dx_m"], level),
            "simulation_half_rise_s": first_rise(ts, curve, level),
        }
    return scores


def main():
    from tensorfem.dynamic_strip import DynamicStrip2D

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", type=Path, default=HERE / "gate_fig2.dat")
    p.add_argument("--dx", type=float, default=.01)
    p.add_argument("--elements", type=int, default=8)
    p.add_argument("--courant", type=float, default=.22)
    p.add_argument("--tank-contact", action="store_true",
                   help="add inelastic normal contact on fixed walls and record its impulse")
    p.add_argument("--sound-speed", type=float, default=60.)
    p.add_argument("--end", type=float, default=.46)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    strip = DynamicStrip2D(length=.09, thickness=.004, young=3.5e6,
                           density=1161.54, elements=args.elements)
    history, report = simulate(GateMotion.read(args.gate), strip, dx=args.dx,
                               end=args.end, courant=args.courant, sound_speed=args.sound_speed,
                               contact=args.tank_contact)
    ref = np.genfromtxt(HERE / "markers_h030_fig3.csv", delimiter=",", names=True)
    scores = compare(history, ref)
    report.update(elements=args.elements, markers=scores,
                  end_s=args.end, gate_file=args.gate.name,
                  gate_sha256=hashlib.sha256(args.gate.read_bytes()).hexdigest(),
                  assessment="experimental_comparison_only",
                  reference="SPHERIC Test 19, Figure 3, digitized experimental symbols, H=0.30 m")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "markers.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "marker1_dx_m", "marker2_dx_m", "marker3_dx_m"])
        writer.writerows(history)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
