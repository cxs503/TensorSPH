#!/usr/bin/env python3
"""Stage-wise runner for R3 (combined Couette-Poiseuille) + assembler.

The cluster job (``cj3815``) sat in a deep queue, so each rung is also runnable
as an independently staged short job; ``--stage assemble`` glues whatever stages
exist into the same payload ``combined.py`` writes, so the artifact schema and
the document generator do not care which path produced the numbers.

    python3 benchmarks/case_poiseuille/run_combined_stage.py --stage r12
    python3 benchmarks/case_poiseuille/run_combined_stage.py --stage assemble
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import combined as cb  # noqa: E402

STAGE = HERE / "out" / "stage"
LEG_RESOLUTION = 8


def stage_rung(r: int) -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    case = cb.run_channel(r, torch.device("cpu"))
    graded = cb.grade_run(case, f"combined_Hdx{r}")
    graded["_y"] = [float(x) for x in case["y"]]
    graded["_u"] = [float(x) for x in case["u_profile"]]
    (STAGE / f"combined_r{r}.json").write_text(json.dumps(graded, indent=2))
    print(f"[stage] r{r}: C1 {graded['profile_rms_pct']:.3f}%  "
          f"C2 {graded['zero_crossing_pct_of_H']:.3f}%H  "
          f"tau_b {graded['tau_bottom_pct']:+.2f}%  tau_t {graded['tau_top_pct']:+.2f}%")
    return graded


def stage_legs(r: int) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for label, kw in (("poiseuille", dict(wall_speed=0.0, plane_vel=False)),
                      ("couette", dict(g=0.0))):
        case = cb.run_channel(r, torch.device("cpu"), **kw)
        graded = cb.grade_run(case, f"{label}_only_Hdx{r}")
        graded["_y"] = [float(x) for x in case["y"]]
        graded["_u"] = [float(x) for x in case["u_profile"]]
        (STAGE / f"{label}_r{r}.json").write_text(json.dumps(graded, indent=2))
        print(f"[stage] {label} r{r}: C1 {graded['profile_rms_pct']:.3f}%")


def assemble(resolutions) -> dict:
    wall_clock = time.time()
    runs, raw = [], {}
    for r in resolutions:
        f = STAGE / f"combined_r{r}.json"
        if not f.exists():
            raise SystemExit(f"missing stage {f}")
        d = json.loads(f.read_text())
        raw[r] = (np.array(d.pop("_y")), np.array(d.pop("_u")))
        runs.append(d)
    runs.sort(key=lambda x: x["resolution_H_over_dx"])

    r = LEG_RESOLUTION
    legs = {}
    for label in ("poiseuille", "couette"):
        f = STAGE / f"{label}_r{r}.json"
        if not f.exists():
            raise SystemExit(f"missing leg stage {f}")
        d = json.loads(f.read_text())
        d.pop("_y"), d.pop("_u")
        d["_y"] = [float(x) for x in json.loads(f.read_text())["_y"]]
        d["_u"] = [float(x) for x in json.loads(f.read_text())["_u"]]
        legs[label] = d

    # the staged leg metrics were graded against the combined analytic; recompute
    # them here from the stashed profiles with each leg's own closed form
    for label, g, uw in (("poiseuille", cb.G_ADVERSE, 0.0), ("couette", 0.0, cb.WALL_SPEED)):
        d = legs[label]
        yl, ul = np.array(d["_y"]), np.array(d["_u"])
        ua = cb.analytic_profile(yl, g=g, wall_speed=uw)
        d["profile_rms_pct"] = 100.0 * cb._rms((ul - ua) / float(np.max(np.abs(ua))))
        d["analytic_own"] = {"g": g, "wall_speed": uw}

    y, u_comb = raw[r]
    u_pois = np.interp(y, np.array(legs["poiseuille"]["_y"]),
                       np.array(legs["poiseuille"]["_u"]))
    u_cout = np.interp(y, np.array(legs["couette"]["_y"]),
                       np.array(legs["couette"]["_u"]))
    residual = u_comb - (u_pois + u_cout)
    for d in legs.values():
        d.pop("_y"), d.pop("_u")
    superposition = {
        "resolution_H_over_dx": r,
        "rows": [{"y": float(yy), "y_over_H": float(yy / cb.HEIGHT),
                  "u_combined": float(uc), "u_poiseuille_leg": float(up),
                  "u_couette_leg": float(uw), "residual": float(rr)}
                 for yy, uc, up, uw, rr in zip(y, u_comb, u_pois, u_cout, residual)],
        "rms_pct": 100.0 * cb._rms(residual / abs(cb.WALL_SPEED)),
        "worst_row_pct": 100.0 * float(np.max(np.abs(residual)) / abs(cb.WALL_SPEED)),
        "worst_row_y_over_H": float(y[int(np.argmax(np.abs(residual)))] / cb.HEIGHT),
        "legs": {"poiseuille_profile_rms_pct": legs["poiseuille"]["profile_rms_pct"],
                 "couette_profile_rms_pct": legs["couette"]["profile_rms_pct"],
                 "combined_profile_rms_pct": next(
                     x for x in runs if x["resolution_H_over_dx"] == r)["profile_rms_pct"],
                 # wall-shear percentages are deliberately not carried over for
                 # the legs: they were graded against the combined closed form
                 # and a leg is a different problem (see analytic_own)
                 }, 
        "note": ("u(combined) vs u(poiseuille) + u(couette), all three runs at "
                 "the same resolution and from the same at-rest initial state; "
                 "the PDE is linear, the discrete WCSPH system is not"),
    }

    reference = cb.reference_check()
    static_control = cb.static_equivalence(torch.device("cpu"))
    gates = {
        "C1_profile_rms_5pct": all(x["profile_rms_pct"] <= cb.GATES["C1_profile_rms_pct"] for x in runs),
        "C2_zero_crossing_5pct_H": all(x["zero_crossing_pct_of_H"] <= cb.GATES["C2_zero_crossing_pct_of_H"] for x in runs),
        "C3_superposition_5pct": superposition["rms_pct"] <= cb.GATES["C3_superposition_rms_pct"],
        "C4a_tau_bottom_5pct": all(abs(x["tau_bottom_pct"]) <= cb.GATES["C4a_tau_bottom_pct"] for x in runs),
        "C4b_tau_top_5pct": all(abs(x["tau_top_pct"]) <= cb.GATES["C4b_tau_top_pct"] for x in runs),
        "C5_momentum_budget_2pct": all(abs(x["momentum_budget"]["pct"]) <= cb.GATES["C5_momentum_budget_pct"] for x in runs),
        "C6_backflow_peak_5pct": all(x["backflow"]["sign_ok"] and abs(x["backflow"]["peak_pct"]) <= cb.GATES["C6_backflow_peak_pct"] for x in runs),
        "C7_reference_fd_1pct": reference["profile_rms_pct"] <= cb.GATES["C7_reference_fd_vs_analytic_pct"],
        "C8_static_wall_equivalence": static_control["bit_identical"],
    }
    payload = {
        "case": cb.CASE_KEY,
        "anchor": "CASE-POISEUILLE-SPH",
        "note": "R3 combined Couette-Poiseuille: superposition, reversal, moving wall",
        "parameters": {
            "height": cb.HEIGHT, "viscosity": cb.VISCOSITY, "rho0": cb.RHO0,
            "sound_speed": cb.SOUND_SPEED, "g": cb.G_ADVERSE,
            "wall_speed": cb.WALL_SPEED, "columns": cb.caserun.COLUMNS,
            "t_end": cb.T_END,
            "steady_window": [cb.STEADY_TIMES[0], cb.STEADY_TIMES[-1]],
            "reynolds": cb.WALL_SPEED * cb.HEIGHT / cb.VISCOSITY,
            "max_mach_analytic": float(np.max(np.abs(cb.analytic_profile(
                np.linspace(0.0, cb.HEIGHT, 2001))))) / cb.SOUND_SPEED,
        },
        "analytic": {
            "profile": "g y (H-y) / (2 nu) + U_w y / H",
            "wall_shear": cb.analytic_wall_shear(),
            "zero_crossing_y": cb.analytic_zero_crossing(),
            "zero_crossing_over_H": (cb.analytic_zero_crossing() or 0.0) / cb.HEIGHT,
            "extrema": cb.analytic_extrema(),
            "references": [
                "White, Viscous Fluid Flow 3rd ed. sec. 3-2.2 (plane Poiseuille)",
                "White, Viscous Fluid Flow 3rd ed. sec. 3-2.1 (Couette)",
                "Batchelor 1967 sec. 4.2 (superposition of the two)",
            ],
        },
        "thresholds": cb.GATES,
        "runs": runs,
        "superposition": superposition,
        "legs": legs,
        "reference_check": reference,
        "static_wall_control": static_control,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "wall_clock_s": time.time() - wall_clock,
        "generated_by": "benchmarks/case_poiseuille/run_combined_stage.py --stage assemble",
    }
    out = HERE / "out" / "combined.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[stage] wrote {out}")
    for k, v in gates.items():
        print(f"  {k:32s} {'PASS' if v else 'FAIL'}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["r8", "r12", "r16", "legs", "assemble"])
    ap.add_argument("--leg-resolution", type=int, default=LEG_RESOLUTION)
    args = ap.parse_args()
    if args.stage == "assemble":
        assemble(cb.RESOLUTIONS)
    elif args.stage == "legs":
        stage_legs(args.leg_resolution)
    else:
        stage_rung(int(args.stage[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())