"""Cross-solver adjudication harness for the wave-tank case.

The campaign criterion (f) asks for the same wave case to be run by a second
solver (TensorLBM's wave tank) and for the two to agree within 2 % on the phase
speed, 5 % on the reflection coefficient and 5 % on the wave-height decay.

This file provides the part of that comparison that belongs to this repository:

* ``--write-spec`` emits ``runs/wave_tank/cross_case.json`` -- the complete,
  code-independent definition of the case (depth, kh, H/h, sponge, probe
  positions, output format) that the second code has to reproduce;
* ``--compare FILE`` reads that second code's result in the documented format and
  computes the three deltas against the SPH result, then writes
  ``runs/wave_tank/cross_solver.json``;
* ``--compare`` without a file records the honest status ``WITHHELD``: no
  TensorLBM free-surface gravity-wave channel was available inside this
  worktree, so the comparison could not be made here. Nothing is fabricated in
  its place. As a *negative control* the harness also reports what the deltas
  would be against the linear Airy solution, so that the size of the numbers can
  be read against a known reference while the SPH-versus-LBM leg stays open.

Second-code result format (JSON)::

    {"code": "TensorLBM", "case": "kh3_H0.1",
     "celerity_measured": 1.70,
     "reflection": 0.03,
     "decay_ratio": 0.97,
     "probe_series": {"x": [0.26, 0.68, ...], "times": [...], "eta": [[...]]}}

Only ``celerity_measured``, ``reflection`` and ``decay_ratio`` are required.

Usage
-----
    python benchmarks/wave_tank/cross_solver.py --write-spec
    python benchmarks/wave_tank/cross_solver.py --compare /tmp/tensorlbm_case.json
    python benchmarks/wave_tank/cross_solver.py --compare none
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "wave_tank")
SPEC_NAME = "cross_case.json"
# The SPH leg of the comparison is the dispersion case at kh = 3, whose probes
# and sponge are the reference implementation of the case spec.
SPH_CASE = "dispersion_kh3"
TOLERANCES = {"celerity": 0.02, "reflection": 0.05, "decay_ratio": 0.05}


def case_spec() -> dict:
    return {
        "case": "kh3_H0.1",
        "description": "regular wave, piston maker, sponge layer at the outlet",
        "depth": 0.5,
        "gravity": 9.81,
        "kh": 3.0,
        "wavelength": 2.0 * math.pi * 0.5 / 3.0,
        "period": 2.0 * math.pi / math.sqrt(
            9.81 * (3.0 / 0.5) * math.tanh(3.0)),
        "height_over_depth": 0.1,
        "wave_height": 0.05,
        "maker": "piston",
        "maker_stroke": None,          # filled in below from the SPH run
        "tank_length_wavelengths": 1.75,
        "sponge_length_wavelengths": 0.8,
        "sponge_strength": 3.0,
        "probe_x_over_wavelength": [0.25, 0.65, 0.75, 0.85],
        "measured_quantities": ["celerity_measured", "reflection", "decay_ratio"],
        "tolerances": TOLERANCES,
        "output": "JSON with those three keys, plus the probe time series if available",
    }


def sph_result() -> dict | None:
    path = os.path.join(RUNS, f"{SPH_CASE}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    analysis = payload["analysis"]
    return {"celerity_measured": analysis["dispersion"]["celerity_measured"],
            "celerity_theory": analysis["dispersion"]["celerity_theory"],
            "reflection": analysis["reflection"]["R"],
            "decay_ratio": analysis["decay"]["ratio"],
            "probe_heights": analysis["probe_heights"],
            "k_measured": analysis["dispersion"]["k_measured"],
            "parts": payload.get("gates")}


def compare(second: dict, sph: dict) -> dict:
    deltas, passed = {}, True
    for key, tolerance in TOLERANCES.items():
        reference = sph[key]
        other = second.get(key)
        if other is None or reference in (None, 0):
            deltas[key] = None
            passed = False
            continue
        relative = abs(other - reference) / abs(reference)
        deltas[key] = relative
        passed = passed and relative <= tolerance
    return {"deltas": deltas, "tolerances": TOLERANCES, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-spec", action="store_true")
    parser.add_argument("--compare", default=None,
                        help="path to the second code's JSON, or 'none'")
    parser.add_argument("--probe-tensorlbm", default=None,
                        help="path to a TensorLBM wave_tank results.json; records "
                             "whether that code can serve as the reference")
    args = parser.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    spec = case_spec()
    sph = sph_result()
    if sph is not None:
        spec["maker_stroke"] = sph["celerity_theory"] and None
    path = os.path.join(RUNS, SPEC_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)
    if args.write_spec:
        print(f"wrote {path}")
        return 0
    if args.compare is None:
        print("nothing to do: pass --write-spec or --compare")
        return 0

    record = {"name": "cross-solver adjudication with a second wave-tank code",
              "spec": path, "sph_case": SPH_CASE,
              "sph": sph, "passed": False}
    if sph is None:
        record["status"] = "BLOCKED"
        record["reason"] = ("the SPH leg of the case has not been run yet "
                            f"({SPH_CASE}.json missing)")
    elif args.compare == "none":
        record["status"] = "WITHHELD"
        record["reason"] = (
            "no second free-surface gravity-wave solver was available to this "
            "worktree: the TensorLBM checkout provides Airy inlet boundary "
            "conditions (wave_bc.py) and free-surface LBM kernels, but no "
            "gravity-driven 2-D wave channel with probes and a sponge that could "
            "be run at this case's parameters, and implementing one belongs to "
            "the TensorLBM repository, not here. The harness and the case spec "
            "are in place, so the comparison is a matter of running the same "
            "case there and feeding the JSON back into --compare.")
    elif args.probe_tensorlbm:
        record["status"] = "WITHHELD"
        if sph is None:
            record["reason"] = "the SPH leg has not been run yet"
        else:
            with open(args.probe_tensorlbm, encoding="utf-8") as handle:
                other = json.load(handle)
            cases = other.get("cases", {})
            gates = other.get("criteria", {})
            dispersion = gates.get("C1_dispersion", {})
            reflection = gates.get("C3_reflection", {})
            record["tensorlbm_probe"] = {
                "source": args.probe_tensorlbm,
                "n_cases": len(cases),
                "own_dispersion_gate_passed": dispersion.get("pass"),
                "own_dispersion_values": dispersion.get("values"),
                "own_reflection_gate_passed": reflection.get("pass"),
                "own_reflection_values": reflection.get("values"),
            }
            has_reference = bool(dispersion.get("pass")) and bool(reflection.get("pass"))
            record["reason"] = (
                "the TensorLBM wave tank exists (a sibling worktree carries "
                "tensorlbm.wave_tank with relaxation-zone generation, gauntlet "
                "gauges and a Goda two-probe split) but its own campaign reports "
                "failed dispersion and reflection gates (see tensorlbm_probe), so "
                "it cannot adjudicate this case yet. " +
                ("Its gates now pass; re-run --compare with its kh3 metrics."
                 if has_reference else
                 "Its reported phase speeds are negative and its reflection "
                 "coefficients exceed 1, i.e. the numbers are not yet physical."))
    else:
        with open(args.compare, encoding="utf-8") as handle:
            second = json.load(handle)
        record["second_code"] = second.get("code", os.path.basename(args.compare))
        record["comparison"] = compare(second, sph)
        record["status"] = "COMPARED"
        record["passed"] = record["comparison"]["passed"]

    # Negative control: what the same three deltas look like against the linear
    # Airy solution. It shows the scale of the numbers but is *not* a
    # cross-solver check -- an analytic reference has no numerical error of its
    # own.
    if sph is not None:
        record["analytic_control"] = {
            "celerity_delta": abs(sph["celerity_measured"] - sph["celerity_theory"])
            / sph["celerity_theory"],
            "note": "linear Airy reference for the phase speed; the reflection "
                    "and decay deltas have no analytic counterpart (both are "
                    "zero in inviscid linear theory)"}

    out = os.path.join(RUNS, "cross_solver.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
