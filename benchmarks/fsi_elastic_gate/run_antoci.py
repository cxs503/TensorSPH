"""Reproduction harness: elastic-gate dam-break (Antoci et al. 2007 case).

Runs, in order:

* V1  hydrostatic load on the gate through the SPH transfer path vs
      ``0.5 rho g h_s^2``;
* V2  rod tip deflection under a scaled hydrostatic load vs the
      Euler-Bernoulli cantilever Green's function;
* FSI coarse / fine resolutions -> gate-tip horizontal displacement history,
      peak value and peak time, self-convergence between the two lattices.

Usage::

    python benchmarks/fsi_elastic_gate/run_antoci.py [--out artifacts/fsi_elastic_gate]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import fsi_elastic_gate as fg  # noqa: E402


def fsi_run(dx: float, t_end: float, tag: str, load_scale: float = 1.0) -> dict:
    case = fg.ElasticGateCase(dx=dx, t_end=t_end, load_scale=load_scale)
    t0 = time.time()
    state, metrics = fg.run(case)
    wall = time.time() - t0
    metrics["wall_seconds"] = wall
    metrics["tag"] = tag
    metrics["load_scale"] = load_scale
    # kinematic sanity: no particle may end up outside the tank box
    pos = state.fluid_pos
    metrics["fluid_inside_box"] = bool(
        ((pos[:, 0] >= 0) & (pos[:, 0] <= case.tank_length)
         & (pos[:, 1] >= 0) & (pos[:, 1] <= case.tank_height)).all())
    metrics["tip_history"] = {
        "t": state.history["t"], "tip_x": state.history["tip_x"],
        "tip_y": state.history["tip_y"], "gate_force_x": state.history["gate_force_x"],
        "max_speed": state.history["max_speed"], "fluid_ke": state.history["fluid_ke"]}
    print(f"[{tag}] dx={dx:g} m, {metrics['n_fluid']} fluid particles, "
          f"{metrics['n_steps']} steps, {wall:.1f} s wall, "
          f"peak tip_x = {metrics['peak_tip_x']*1000:.2f} mm "
          f"at t = {metrics['peak_time']*1000:.0f} ms")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/fsi_elastic_gate")
    ap.add_argument("--t-end", type=float, default=0.4)
    ap.add_argument("--dx", nargs="*", type=float, default=[0.0125, 0.01])
    ap.add_argument("--skip-verification", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {"case": {"reference": {
        "authors": "Antoci, Gallati & Sibilla",
        "title": "Numerical simulation of fluid-structure interaction by SPH",
        "journal": "Computers & Structures 85 (2007) 879-890",
        "doi": "10.1016/j.compstruc.2007.01.002",
        "access": "closed (Unpaywall is_oa=false, S2 CLOSED, OpenAlex no OA copy)",
        "reference_values": "WITHHELD - experimental gate-tip curve not read"},
        "parameter_provenance": fg.PARAMETER_PROVENANCE}}

    case0 = fg.ElasticGateCase()
    result["case"]["parameters"] = {
        "dx": case0.dx, "h_factor": case0.h_factor, "rho0": case0.rho0,
        "sound_speed": case0.sound_speed, "gamma": case0.gamma,
        "gravity": case0.gravity, "viscosity": case0.viscosity,
        "tank_length": case0.tank_length, "tank_height": case0.tank_height,
        "water_width": case0.water_width, "water_height": case0.water_height,
        "gate_height": case0.gate_height, "gate_thickness": case0.gate_thickness,
        "gate_E": case0.gate_E, "gate_nu": case0.gate_nu, "gate_rho": case0.gate_rho,
        "n_gate_nodes": case0.n_gate_nodes, "t_end": args.t_end,
        "dt": case0.dt(), "mach_at_c0": case0.sound_speed / math.sqrt(
            case0.gamma * 1.0e5 / 1000.0)}

    if not args.skip_verification:
        print("V1 hydrostatic load on the gate (SPH transfer path) ...")
        v1 = fg.verify_hydrostatic_load(case0)
        v1["verdict"] = "PASS" if abs(v1["rel_error"]) < 0.10 else "FAIL"
        print(f"    load = {v1['load_sph']:.1f} N/m vs exact "
              f"{v1['load_exact']:.1f} N/m, rel err {v1['rel_error']*100:+.2f} % -> "
              f"{v1['verdict']}")
        print("V2 rod tip deflection vs Euler-Bernoulli cantilever ...")
        v2 = fg.verify_cantilever(case0)
        v2["verdict"] = "PASS" if abs(v2["rel_error"]) < 0.10 else "FAIL"
        print(f"    tip = {v2['tip_rod']*1000:.4f} mm vs exact "
              f"{v2['tip_exact']*1000:.4f} mm (load scale {v2['load_scale']:g}), "
              f"rel err {v2['rel_error']*100:+.2f} % -> {v2['verdict']}")
        result["verification"] = {"V1_hydrostatic_load": v1, "V2_cantilever": v2}

    # load calibration: the Adami pressure term of the SPH transfer path is a
    # discretisation of the wall load and is biased low by the missing interior
    # half of the kernel support (measured by V1).  The bias is a property of
    # the lattice/kernel, not of the case, so it is divided out and both the raw
    # and the calibrated numbers are reported.
    load_scale = 1.0
    if not args.skip_verification:
        ratio = result["verification"]["V1_hydrostatic_load"]["rel_error"]
        load_scale = 1.0 / (1.0 + ratio)
        result["verification"]["V1_hydrostatic_load"]["load_calibration_factor"] = load_scale

    runs = []
    for dx in args.dx:
        runs.append(fsi_run(dx, args.t_end, f"dx={dx:g}", load_scale=load_scale))
    result["runs"] = runs

    if len(runs) >= 2:
        coarse, fine = runs[0], runs[-1]
        conv = {
            "peak_rel_diff": abs(fine["peak_tip_x"] - coarse["peak_tip_x"])
            / max(abs(fine["peak_tip_x"]), 1e-12),
            "time_rel_diff": abs(fine["peak_time"] - coarse["peak_time"])
            / max(fine["peak_time"], 1e-12),
            "coarse_dx": coarse["dx"], "fine_dx": fine["dx"]}
        conv["verdict"] = ("CONVERGED" if conv["peak_rel_diff"] < 0.10
                           and conv["time_rel_diff"] < 0.10 else "NOT CONVERGED")
        result["self_convergence"] = conv
        print(f"self-convergence {coarse['dx']:g}->{fine['dx']:g}: "
              f"peak {conv['peak_rel_diff']*100:+.1f} %, phase "
              f"{conv['time_rel_diff']*100:+.1f} % -> {conv['verdict']}")

    with (out / "antoci_elastic_gate.json").open("w") as fh:
        json.dump(result, fh, indent=2)

    try:
        plot(result, out / "antoci_elastic_gate.png")
        print("figure:", out / "antoci_elastic_gate.png")
    except Exception as exc:                       # pragma: no cover - optional
        print("plot skipped:", exc)
    print("json:", out / "antoci_elastic_gate.json")
    return 0


def plot(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for run in result["runs"]:
        h = run["tip_history"]
        ax.plot([t * 1000 for t in h["t"]], [v * 1000 for v in h["tip_x"]],
                label=f"TensorSPH WCSPH+rod, dx={run['dx']*1000:.1f} mm")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("gate tip horizontal displacement [mm]")
    ax.set_title("Elastic gate, E = 1 MPa (Antoci 2007 configuration)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.text(0.02, 0.02, "experimental reference: WITHHELD (paper closed access)",
            transform=ax.transAxes, fontsize=7, color="crimson")

    ax = axes[1]
    for run in result["runs"]:
        h = run["tip_history"]
        ax.plot([t * 1000 for t in h["t"]], [v * 1000 for v in h["tip_y"]],
                label=f"vertical, dx={run['dx']*1000:.1f} mm")
        ax.plot([t * 1000 for t in h["t"]], [max(v, 0) for v in h["gate_force_x"]],
                "--", label=f"fluid load on gate, dx={run['dx']*1000:.1f} mm")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("tip vertical displacement [mm] / gate load [N/m]")
    ax.set_title("Rod response and coupled load")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())