"""Resolution ladder of the *dynamic* projection-SPH Young-Laplace gate.

Why this exists
---------------
``run_dynamic.py`` (mission 19433) established that the adjoint-consistent
projection pair of ``tensorsph.projection_adjoint`` keeps a static drop at rest
for 60 steps at **one** resolution, ``R/dx = 8`` (780 particles) -- by far the
coarsest set-up of the whole ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` program,
whose static arms ladder up to ``R/dx = 80``.  It reported
``dp = +2.02 %`` (inside the owner 15 % gate) but also one red gate:
``max|u|/v_ref = 0.179`` against Adami et al. (2012)'s ``< 0.1`` admissibility
line.  A single resolution cannot say whether that residual is

* a **lattice-scale artefact** (the solenoidal part of the discrete CSF
  traction, a field built out of ``grad W`` differences, so ``O(dx)``-ish),
  in which case it must fall as ``R/dx`` grows; or
* a **formulation defect** (the projection leaves a spurious current that does
  not vanish under refinement, i.e. no convergence to rest in the limit).

This script answers that with a ladder at fixed physics (``R = 20 mm``,
``sigma = 0.072 N/m``, ``rho0 = 1000``, ``h/dx = 1.3``, ``dt = 0.1 t_sigma``,
carrier padding 12 mm, outer layer velocity-pinned), varying only ``dx``.

Two secondary debts are also paid here because they are cheap once the ladder
exists:

* the ship doc asserts *"the equilibrium does not depend on nu"* from a
  3-value sweep at a single resolution; the ladder runs three
  ``nu_factor`` values at one level and gates the spread;
* ``step 60`` is only ``6 t_sigma``; a *static* solution must be invariant in
  time, so a 5x longer march is run and gated for drift.

Gates (declared **before** any new number was read; copied verbatim into
``results_dynamic_ladder/summary.json`` under ``gates_declared``)
---------------------------------------------------------------------------
``laplace_all_levels_15pct``        |dp_err| <= 15 at every (level, nu)   hard (owner)
``area_drift_all_levels_2pct``      |area_ratio-1| <= 0.02 everywhere     hard
``projection_exact``                div_after/div_before <= 1e-5        hard
``spurious_gate_finest``            umax/v_ref(finest) <= 0.10 (Adami)   hard
``spurious_decay_order_min``        fitted order in R/dx >= 0.50         hard (H1)
``horizon_drift_hold``              long march: |dp_err| <= 15 and
                                    |area-1| <= 0.02 and no divergence   hard (H2)
``viscosity_independence_pct``      spread of dp over nu_factor <= 3      hard
``no_divergence_viscous``           every viscously damped run completes  hard

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --list
    PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --job dx16_nu01
    PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --shard 3/6
    PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --summarize

Each ``--job`` writes ``results_dynamic_ladder/<job>.json`` (one march's record,
including the full sampled history); ``--summarize`` merges whatever is present,
computes the ladder statistics and the gate verdicts.  Jobs may be run in any
order and on any host: the record is per-job, so a shard that dies leaves the
others usable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from tensorsph.projection_adjoint import relax_projection_consistent

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "results_dynamic_ladder"

R = 0.02
SIGMA, RHO = 0.072, 1000.0
OUTER = 0.012
HF = 1.3

# ladder: R/dx -> dx.  R/dx = 8 is the level of results_dynamic.json (re-run here
# with the same step count so the two records can be cross-checked).
LEVELS = {8: R / 8.0, 12: R / 12.0, 16: R / 16.0, 24: R / 24.0}
NU_PRIMARY = 0.01
STEPS_PRIMARY = 120
SAMPLE = 10
HORIZON_STEPS = 600          # 5x the primary march, at a mid level
HORIZON_LEVEL = 12
NU_SWEEP_LEVEL = 16
NU_SWEEP = (0.003, 0.01, 0.03)

ADAMI_GATE = 0.10
LAPLACE_GATE_PCT = 15.0
AREA_GATE = 0.02
DECAY_ORDER_MIN = 0.50
VISC_SPREAD_PCT = 3.0

GATES_DECLARED = {
    "laplace_all_levels_15pct": {"op": "<=", "value": LAPLACE_GATE_PCT,
                                 "unit": "percent", "scope": "every (level, nu)"},
    "area_drift_all_levels_2pct": {"op": "<=", "value": AREA_GATE,
                                   "unit": "fraction", "scope": "every run"},
    "projection_exact": {"op": "<=", "value": 1e-5, "unit": "ratio",
                         "scope": "every sampled step"},
    "spurious_gate_finest": {"op": "<=", "value": ADAMI_GATE, "unit": "v_ref",
                             "scope": "finest level, primary nu"},
    "spurious_decay_order_min": {"op": ">=", "value": DECAY_ORDER_MIN,
                                 "unit": "order in R/dx",
                                 "scope": "primary nu, R/dx=8..24 (H1)"},
    "horizon_drift_hold": {"op": "<=", "value": LAPLACE_GATE_PCT, "unit": "percent",
                           "scope": f"{HORIZON_STEPS} steps at R/dx={HORIZON_LEVEL} (H2)"},
    "viscosity_independence_pct": {"op": "<=", "value": VISC_SPREAD_PCT,
                                   "unit": "percent spread",
                                   "scope": f"nu_factor {NU_SWEEP} at R/dx={NU_SWEEP_LEVEL}"},
    "no_divergence_viscous": {"op": "==", "value": True,
                              "scope": "every nu_factor > 0 run completes"},
}


def job_specs():
    """The job list.  Deterministic order -- the shard split depends on it."""
    specs = []
    for lvl in (8, 12, 16, 24):
        specs.append({"name": f"dx{lvl}_nu01", "level": lvl,
                      "nu_factor": NU_PRIMARY, "steps": STEPS_PRIMARY,
                      "role": "ladder"})
    for nu in NU_SWEEP:
        if nu == NU_PRIMARY:
            continue
        specs.append({"name": f"dx{NU_SWEEP_LEVEL}_nu{int(round(nu * 1000)):02d}",
                      "level": NU_SWEEP_LEVEL, "nu_factor": nu,
                      "steps": STEPS_PRIMARY, "role": "nu_sweep"})
    specs.append({"name": f"dx{HORIZON_LEVEL}_horizon", "level": HORIZON_LEVEL,
                  "nu_factor": NU_PRIMARY, "steps": HORIZON_STEPS,
                  "role": "horizon"})
    return specs


def run_job(spec):
    dx = LEVELS[spec["level"]]
    t0 = time.time()
    rec = relax_projection_consistent(
        radius=R, dx=dx, sigma=SIGMA, rho0=RHO, outer=OUTER, h_factor=HF,
        steps=spec["steps"], sample_every=SAMPLE, nu_factor=spec["nu_factor"],
        pin_outer=True, rel_tol=1e-6, cg_max_iter=4000)
    rec["wall_s"] = time.time() - t0
    rec["job"] = spec["name"]
    rec["role"] = spec["role"]
    rec["threads"] = int(torch.get_num_threads())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / (spec["name"] + ".json")).write_text(json.dumps(rec, indent=1))
    print("[%s] R/dx=%.1f n=%d steps=%d dp_err=%.3f%% umax=%.4g area=%+.4f "
          "wall=%.0fs" % (spec["name"], rec["R_over_dx"], rec["n_particles"],
                          spec["steps"], rec["dp_rel_error_pct"],
                          rec["umax_over_vref_final"],
                          rec["area_ratio_final"] - 1.0, rec["wall_s"]),
          flush=True)
    return rec


# --------------------------------------------------------------------------- #
# summary + gates
# --------------------------------------------------------------------------- #
def _fit_order(xs, ys):
    """Least-squares slope of log10(y) vs log10(x); y must be positive."""
    if any(y <= 0 for y in ys) or len(xs) < 2:
        return float("nan")
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


def summarize():
    # The directory also holds records that are *not* marches of this ladder
    # (``diagnose_solenoid.json`` from the one-shot solenoidal diagnostic).  The
    # job list is the authority on what a ladder row is: a file is consumed only
    # if its stem is a declared job, so a side diagnostic can never crash the
    # summariser (it did: ``KeyError: 'role'``, before this filter).
    wanted = {s["name"] for s in job_specs()}
    runs = {}
    if OUTDIR.exists():
        for p in sorted(OUTDIR.glob("*.json")):
            if p.stem not in wanted:
                continue
            blob = json.loads(p.read_text())
            if "role" not in blob:  # truncated / foreign record: never half-read it
                continue
            runs[blob.get("job", p.stem)] = blob

    ladder = sorted([r for r in runs.values() if r["role"] == "ladder"],
                    key=lambda r: r["R_over_dx"])
    nu_sweep = sorted([r for r in runs.values() if r["role"] == "nu_sweep"],
                      key=lambda r: r["nu_factor"])
    nu_at_level = [r for r in ladder if r["R_over_dx"] == float(NU_SWEEP_LEVEL)]
    horizon = [r for r in runs.values() if r["role"] == "horizon"]

    rows = []
    for r in sorted(runs.values(), key=lambda r: (r["role"], r["R_over_dx"],
                                                  r["nu_factor"])):
        rows.append({
            "job": r["job"], "role": r["role"], "R_over_dx": r["R_over_dx"],
            "R_over_h": r["R_over_h"], "n_particles": r["n_particles"],
            "steps_requested": len([h for h in r["history"]]) and
                                r["history"][-1]["step"],
            "nu_factor": r["nu_factor"], "diverged": r["diverged"],
            "dp_rel_error_pct": r["dp_rel_error_pct"],
            "umax_over_vref_final": r["umax_over_vref_final"],
            "umean_over_vref_final": r["umean_over_vref_final"],
            "area_ratio_final": r["area_ratio_final"],
            "max_div_ratio": max([h.get("div_after_over_div_before", 0.0)
                                  for h in r["history"]], default=float("nan")),
            "cg_iterations_final": r["history"][-1].get("cg_iterations")
                                   if r["history"] else None,
            "wall_s": r["wall_s"],
        })

    spurious_x = [r["R_over_dx"] for r in ladder]
    spurious_y = [r["umax_over_vref_final"] for r in ladder]
    dp_x = [r["R_over_dx"] for r in ladder]
    dp_abs = [abs(r["dp_rel_error_pct"]) for r in ladder]
    order_spurious = _fit_order(spurious_x, spurious_y)
    order_dp = _fit_order(dp_x, [max(v, 1e-9) for v in dp_abs])

    nu_values = sorted([r for r in nu_at_level + nu_sweep],
                       key=lambda r: r["nu_factor"])
    dp_nu = [r["dp_rel_error_pct"] for r in nu_values]
    visc_spread = (max(dp_nu) - min(dp_nu)) if dp_nu else float("nan")

    horizons = []
    for h in horizon:
        hist = [x for x in h["history"] if not x.get("diverged")]
        late = [x for x in hist if x["step"] > 0.5 * h["history"][-1]["step"]]
        horizons.append({
            "job": h["job"], "steps": h["history"][-1]["step"],
            "diverged": h["diverged"],
            "dp_rel_error_pct_final": h["dp_rel_error_pct"],
            "area_ratio_final": h["area_ratio_final"],
            "umax_over_vref_late_max": max([x["umax_over_vref"] for x in late],
                                           default=float("nan")),
            "dp_pa_late_min": min([x["dp_Pa"] for x in late], default=float("nan")),
            "dp_pa_late_max": max([x["dp_Pa"] for x in late], default=float("nan")),
            "dp_pa_theory": h["dp_theory_Pa"],
        })

    all_runs = list(runs.values())
    finest = max(ladder, key=lambda r: r["R_over_dx"]) if ladder else None
    gates = {}
    gates["laplace_all_levels_15pct"] = {
        "measured": max([abs(r["dp_rel_error_pct"]) for r in all_runs],
                        default=float("nan")),
        "declared": LAPLACE_GATE_PCT, "op": "<=",
        "pass": (bool(all_runs) and
                 max(abs(r["dp_rel_error_pct"]) for r in all_runs) <= LAPLACE_GATE_PCT),
    }
    gates["area_drift_all_levels_2pct"] = {
        "measured": max([abs(r["area_ratio_final"] - 1.0) for r in all_runs],
                        default=float("nan")),
        "declared": AREA_GATE, "op": "<=",
        "pass": (bool(all_runs) and
                 max(abs(r["area_ratio_final"] - 1.0) for r in all_runs) <= AREA_GATE),
    }
    div_ratios = [h.get("div_after_over_div_before", 0.0)
                  for r in all_runs for h in r["history"]]
    gates["projection_exact"] = {
        "measured": max(div_ratios, default=float("nan")),
        "declared": 1e-5, "op": "<=",
        "pass": bool(div_ratios) and max(div_ratios) <= 1e-5,
    }
    gates["spurious_gate_finest"] = {
        "measured": finest["umax_over_vref_final"] if finest else float("nan"),
        "declared": ADAMI_GATE, "op": "<=",
        "pass": bool(finest) and finest["umax_over_vref_final"] <= ADAMI_GATE,
    }
    gates["spurious_decay_order_min"] = {
        "measured": order_spurious, "declared": DECAY_ORDER_MIN, "op": ">=",
        "pass": bool(order_spurious == order_spurious) and order_spurious >= DECAY_ORDER_MIN,
    }
    hz_ok = bool(horizons) and all(
        (not h["diverged"]) and abs(h["dp_rel_error_pct_final"]) <= LAPLACE_GATE_PCT
        and abs(h["area_ratio_final"] - 1.0) <= AREA_GATE for h in horizons)
    gates["horizon_drift_hold"] = {
        "measured": max([abs(h["dp_rel_error_pct_final"]) for h in horizons],
                        default=float("nan")),
        "declared": LAPLACE_GATE_PCT, "op": "<=", "pass": hz_ok,
    }
    gates["viscosity_independence_pct"] = {
        "measured": visc_spread, "declared": VISC_SPREAD_PCT, "op": "<=",
        "pass": bool(visc_spread == visc_spread) and visc_spread <= VISC_SPREAD_PCT,
        "n_values": len(dp_nu),
    }
    viscous = [r for r in all_runs if r["nu_factor"] > 0.0]
    gates["no_divergence_viscous"] = {
        "measured": sum(1 for r in viscous if r["diverged"]),
        "declared": 0, "op": "==",
        "pass": bool(viscous) and all(not r["diverged"] for r in viscous),
        "n_runs": len(viscous),
    }

    summary = {
        "case": "CASE-LAPLACE-SPH-DYNAMIC-LADDER",
        "anchor": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE",
        "title": "Dynamic projection-SPH Young-Laplace: does the residual scale with dx?",
        "setup": {
            "radius_m": R, "sigma_N_per_m": SIGMA, "rho0": RHO,
            "outer_carrier_m": OUTER, "h_over_dx": HF,
            "dt_over_t_sigma": 0.1, "sample_every": SAMPLE,
            "levels_R_over_dx": sorted(LEVELS), "nu_primary": NU_PRIMARY,
            "steps_primary": STEPS_PRIMARY, "horizon_steps": HORIZON_STEPS,
            "nu_sweep": list(NU_SWEEP), "nu_sweep_level": NU_SWEEP_LEVEL,
            "cg_rel_tol": 1e-6, "cg_max_iter": 4000,
            "v_ref_m_per_s": math.sqrt(SIGMA / (RHO * R)),
            # t_sigma = sqrt(rho h^3/sigma) is *level dependent*; the record used to
            # carry one value computed with h of R/dx = 8 under the bare name
            # ``t_sigma_s``, which reads as "the capillary time of this run".
            "t_sigma_s_by_level": {
                "R_over_dx_%d" % lvl: math.sqrt(RHO * (HF * LEVELS[lvl]) ** 3 / SIGMA)
                for lvl in sorted(LEVELS)},
            "t_sigma_s_ref_R_over_dx_8": math.sqrt(RHO * (HF * LEVELS[8]) ** 3 / SIGMA),
        },
        "gates_declared": GATES_DECLARED,
        "rows": rows,
        "ladder_primary": [
            {"R_over_dx": r["R_over_dx"], "dp_rel_error_pct": r["dp_rel_error_pct"],
             "umax_over_vref": r["umax_over_vref_final"],
             "umean_over_vref": r["umean_over_vref_final"],
             "area_ratio": r["area_ratio_final"],
             "n_particles": r["n_particles"]}
            for r in ladder],
        "orders": {"spurious_in_R_over_dx": order_spurious,
                   "abs_dp_error_in_R_over_dx": order_dp},
        "viscosity_sweep": [
            {"nu_factor": r["nu_factor"], "nu_m2_per_s": r["nu_m2_per_s"],
             "R_over_dx": r["R_over_dx"],
             "dp_rel_error_pct": r["dp_rel_error_pct"],
             "umax_over_vref": r["umax_over_vref_final"],
             "area_ratio": r["area_ratio_final"]} for r in nu_values],
        "horizon": horizons,
        "gate_verdicts": gates,
        "n_jobs_present": len(runs),
        "n_jobs_expected": len(job_specs()),
        "wall_s_total": sum(r["wall_s"] for r in runs.values()),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", OUTDIR / "summary.json",
          "jobs %d/%d" % (len(runs), len(job_specs())))
    for k, v in gates.items():
        print("  %-32s measured=%-12.4g declared=%-8.4g pass=%s"
              % (k, v["measured"] if isinstance(v["measured"], float) else float(v["measured"]),
                 v["declared"] if isinstance(v["declared"], float) else float(v["declared"]),
                 v["pass"]))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--shard", default=None, help="i/n round-robin over jobs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    specs = job_specs()
    if args.list:
        for s in specs:
            print(s)
        return
    if args.summarize:
        summarize()
        return
    if args.shard:
        i, n = (int(t) for t in args.shard.split("/"))
        todo = [s for k, s in enumerate(specs) if k % n == i]
    elif args.job:
        todo = [s for s in specs if s["name"] == args.job]
        if not todo:
            raise SystemExit("unknown job %r; --list to see them" % args.job)
    else:
        todo = specs
    for s in todo:
        path = OUTDIR / (s["name"] + ".json")
        if path.exists() and not os.environ.get("FORCE"):
            print("[%s] already present, skipped" % s["name"], flush=True)
            continue
        run_job(s)


if __name__ == "__main__":
    main()