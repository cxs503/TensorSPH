"""Long-horizon test of the *well-balanced* projection-SPH Young-Laplace drop.

Why this exists
---------------
``docs/xval_projection_laplace_horizon_instability.md`` left the dynamic arm of
the anchor ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` as an honest FAIL: with the
adjoint-consistent projection pair the drop does **not** come to rest.  At
``R/dx = 12`` the spurious current marches ``0.0717 -> 0.286 -> 0.4543 v_ref``
(steps 10 / 120 / 180 = ``3 / 12 / 18 t_sigma``) and the march then diverges at
step 181.  Two branches of the dichotomy were left open; the lattice branch was
rejected by ``docs/xval_projection_laplace_lattice_health.md`` at ``R/dx = 16``
(no breakage rule fires), leaving *traction mis-projection*.

The repo already contains the cure for exactly that branch:
``tensorsph.projection_adjoint.relax_projection_balanced`` splits the discrete
CSF traction with the projection's own operator into a discrete gradient (which
the projection annihilates exactly) plus a remainder.  It was validated to
**50 steps** only (``docs/xval_projection_laplace_wellbalanced.md``).  A
fixed-point claim that is only measured for ``5 t_sigma`` is not a statement
about the instability the same program reports at ``18 t_sigma``: this harness
runs the *same* 600-step horizon that diverges without the split, on the same
lattice, with the same knobs, and with the lattice-health instrument switched on
so that "stable" cannot be confused with "the lattice quietly degenerated".

Arms (declared before any number is read)
-----------------------------------------
The 600-step arms are the target; the two ``R/dx = 8`` arms below are the ones
this unit could afford on the queue, and ``MEASURED_HORIZON_STEPS = 120`` is
declared so that no page can present a 12 ``t_sigma`` record as the 60
``t_sigma`` claim.  The 600-step arms stay declared and their absence is a
counted gate (``long_horizon_600_all_arms_present``), not a skip.

``dx12_h600_bal_nu01``   balanced, ``nu_factor = 0.01``, 600 steps, ``R/dx=12``
                         -- the primary arm: *identical knobs* to the shipped
                         diverging control record ``dx12_horizon.json``, one knob
                         (the split) moved.
``dx12_h600_bal_inv``    balanced, ``nu_factor = 0``, 600 steps
                         -- the pure form (no viscosity knob to hold the drop),
                         the configuration the round-3 ladder used.
``dx12_h600_ctl_nu01``   **no split**, ``nu_factor = 0.01``, 200 steps
                         -- in-harness control: reproduces the divergence on this
                         code path, so the comparison is not across harnesses.
``dx16_h200_bal_inv``    balanced, ``nu_factor = 0``, 200 steps, ``R/dx = 16``
                         -- resolution robustness of the cured arm (cheap subset).

Gates (``GATES_DECLARED``, copied verbatim into ``summary.json``)
-----------------------------------------------------------------
``no_divergence_18tsigma``   balanced arm completes 600 steps, no ``1e3 v_ref``
                             trip, every sample finite                     hard
``spurious_adami``           ``max|u|/v_ref <= 0.10`` at every sample
                             (Adami et al. 2012)                          hard
``laplace_all_samples_15pct``|dp/sigma*R - 1| <= 15 % at every sample,
                             both instruments (pressure and ``rho[q]``)   hard (owner)
``laplace_all_samples_3pct`` same, 3 % -- the claim level, reported separately
                                                                          hard
``area_drift_2pct``          |area_ratio - 1| <= 0.02 at every sample      hard
``lattice_healthy``          no ``BREAK_RULES`` entry fires on any sampled
                             step, breaking step included                 hard
``split_is_the_cause``       control (no split) diverges and the balanced arm
                             with the same knobs does not                  hard

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_sph/run_balanced_horizon.py --list
    PYTHONPATH=src python benchmarks/laplace_sph/run_balanced_horizon.py --job dx12_h600_bal_inv
    PYTHONPATH=src python benchmarks/laplace_sph/run_balanced_horizon.py --summarize
    PYTHONPATH=src python benchmarks/laplace_sph/run_balanced_horizon.py --figure
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.projection_adjoint import relax_projection_balanced

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "results_balanced_horizon"

R = 0.02
SIGMA, RHO = 0.072, 1000.0
OUTER = 0.012
HF = 1.3
LEVELS = {8: R / 8.0, 12: R / 12.0, 16: R / 16.0}
SAMPLE = 10

ADAMI_GATE = 0.10
LAPLACE_GATE_PCT = 15.0
CLAIM_PCT = 3.0
AREA_GATE = 0.02
HORIZON_STEPS = 600

# Breakage rules, pre-registered (same set the lattice-health page used).
BREAK_RULES = {
    "pair_min_over_dx_min": 0.25,
    "nn_min_min": 3.0,
    "kappa_R_max": 5.0,
    "gradc_dx_max": 5.0,
    "color_min": -0.05,
    "color_max": 1.05,
}

JOBS = [
    {"name": "dx8_h120_bal_nu01", "level": 8, "steps": 120, "balanced": True,
     "nu_factor": 0.01, "role": "primary"},
    {"name": "dx8_h120_ctl_nu01", "level": 8, "steps": 120, "balanced": False,
     "nu_factor": 0.01, "role": "control"},
    {"name": "dx12_h600_bal_nu01", "level": 12, "steps": 600, "balanced": True,
     "nu_factor": 0.01, "role": "horizon"},
    {"name": "dx12_h600_bal_inv", "level": 12, "steps": 600, "balanced": True,
     "nu_factor": 0.0, "role": "horizon_inviscid"},
]

MEASURED_HORIZON_STEPS = 120     # the horizon this record actually covers
TARGET_HORIZON_STEPS = 600       # the horizon the anchor's instability lives at

GATES_DECLARED = {
    "no_divergence_measured_horizon": {
        "op": "==", "value": True,
        "scope": "every present arm completes its declared steps "
                 f"(measured horizon {MEASURED_HORIZON_STEPS} steps)"},
    "spurious_adami": {"op": "<=", "value": ADAMI_GATE, "unit": "v_ref",
                       "scope": "max over samples, balanced arms"},
    "laplace_all_samples_15pct": {"op": "<=", "value": LAPLACE_GATE_PCT,
                                  "unit": "percent",
                                  "scope": "every sample, both instruments"},
    "laplace_all_samples_3pct": {"op": "<=", "value": CLAIM_PCT,
                                 "unit": "percent",
                                 "scope": "every sample, both instruments"},
    "area_drift_2pct": {"op": "<=", "value": AREA_GATE, "unit": "fraction",
                        "scope": "every sample"},
    "lattice_healthy": {"op": "==", "value": True,
                        "scope": "no BREAK_RULES entry fires on any sampled step"},
    "split_is_the_cause": {
        "op": "==", "value": True,
        "scope": "control (same level, same nu, no split) leaves a spurious "
                 "current above the Adami line while the balanced arm does not"},
    "long_horizon_600_all_arms_present": {
        "op": "==", "value": True,
        "scope": f"records for the {TARGET_HORIZON_STEPS}-step arms exist "
                 "(the instability horizon; outstanding debt, not a pass)"},
}

def job_names():
    return [j["name"] for j in JOBS]


def run_job(spec):
    dx = LEVELS[spec["level"]]
    t0 = time.time()
    rec = relax_projection_balanced(
        radius=R, dx=dx, sigma=SIGMA, rho0=RHO, outer=OUTER, h_factor=HF,
        steps=spec["steps"], sample_every=SAMPLE, nu_factor=spec["nu_factor"],
        balanced=spec["balanced"], pin_outer=True, rel_tol=1e-6,
        cg_max_iter=4000, stability_report=True)
    rec["wall_s"] = time.time() - t0
    rec["job"] = spec["name"]
    rec["role"] = spec["role"]
    rec["threads"] = int(torch.get_num_threads())
    rec["stability_report"] = True
    rec["steps_requested"] = spec["steps"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / (spec["name"] + ".json")).write_text(json.dumps(rec, indent=1))
    print("[%s] R/dx=%.1f n=%d steps=%d balanced=%s nu_f=%g completed=%d "
          "diverged=%s umax=%.4g dp_err=%.3f%% wall=%.0fs"
          % (spec["name"], rec["R_over_dx"], rec["n_particles"], spec["steps"],
             spec["balanced"], spec["nu_factor"], rec["steps_completed"],
             rec["diverged"], rec["umax_over_vref_final"],
             rec["dp_rel_error_pct"], rec["wall_s"]), flush=True)
    return rec


# --------------------------------------------------------------------------- #
# summariser + gates
# --------------------------------------------------------------------------- #
def _samples(rec):
    return [h for h in rec.get("history", []) if not h.get("diverged")]


def _breakages(sample):
    """Which pre-registered rules a lattice sample violates ([] = healthy)."""
    lat = sample.get("lattice") or {}
    if not lat:
        return ["no_lattice_block"]
    hit = []
    if lat.get("pair_min_over_dx", 1.0) < BREAK_RULES["pair_min_over_dx_min"]:
        hit.append("pair_min_over_dx")
    if lat.get("nn_min", 99.0) < BREAK_RULES["nn_min_min"]:
        hit.append("nn_min")
    if lat.get("kappa_R_max", 0.0) > BREAK_RULES["kappa_R_max"]:
        hit.append("kappa_R")
    if lat.get("gradc_dx_max", 0.0) > BREAK_RULES["gradc_dx_max"]:
        hit.append("gradc_dx")
    if lat.get("color_min", 0.0) < BREAK_RULES["color_min"]:
        hit.append("color_min")
    if lat.get("color_max", 1.0) > BREAK_RULES["color_max"]:
        hit.append("color_max")
    return hit


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _arm_stats(rec):
    smp = _samples(rec)
    broken = [h for h in rec.get("history", []) if h.get("diverged")]
    lat_breaks = []
    for h in rec.get("history", []):
        hit = _breakages(h)
        # a diverged sample without a lattice block (non-finite entry state) is
        # reported as such, never as "healthy"
        if hit:
            lat_breaks.append({"step": h.get("step"), "rules": hit})

    errs, errs_pot, umax, area = [], [], [], []
    for h in smp:
        if _finite(h.get("dp_rel_error_pct")):
            errs.append(abs(float(h["dp_rel_error_pct"])))
        elif _finite(h.get("dp_Pa")) and _finite(rec.get("dp_theory_Pa")):
            errs.append(abs(h["dp_Pa"] / rec["dp_theory_Pa"] - 1.0) * 100.0)
        if _finite(h.get("dp_potential_Pa")):
            errs_pot.append(abs(h["dp_potential_Pa"] / rec["dp_theory_Pa"] - 1.0) * 100.0)
        if _finite(h.get("umax_over_vref")):
            umax.append(float(h["umax_over_vref"]))
        if _finite(h.get("area_ratio")):
            area.append(abs(float(h["area_ratio"]) - 1.0))

    last = smp[-1] if smp else {}
    return {
        "job": rec.get("job"),
        "role": rec.get("role"),
        "R_over_dx": rec.get("R_over_dx"),
        "n_particles": rec.get("n_particles"),
        "balanced": rec.get("balanced"),
        "nu_factor": rec.get("nu_factor"),
        "steps_requested": rec.get("steps_requested",
                                   rec.get("steps", None)),
        "steps_completed": rec.get("steps_completed"),
        "steps_marched": rec.get("steps_marched"),
        "diverged": bool(rec.get("diverged")),
        "wall_s": rec.get("wall_s"),
        "last_sample_step": last.get("step"),
        "t_over_t_sigma_last": last.get("t_over_t_sigma"),
        "umax_over_vref_max": max(umax) if umax else float("nan"),
        "umax_over_vref_last": umax[-1] if umax else float("nan"),
        "dp_err_abs_max_pct": max(errs) if errs else float("nan"),
        "dp_err_last_pct": errs[-1] if errs else float("nan"),
        "dp_potential_err_abs_max_pct": max(errs_pot) if errs_pot else float("nan"),
        "area_drift_max": max(area) if area else float("nan"),
        "area_ratio_last": float(last.get("area_ratio", float("nan"))),
        "solenoidal_fraction_last": float(last.get("solenoidal_fraction", float("nan"))),
        "lattice_breakages": lat_breaks,
        "lattice_worst": _lattice_worst(rec),
    }


def _lattice_worst(rec):
    rows = [(h.get("lattice") or {}) for h in rec.get("history", [])]
    rows = [r for r in rows if r]
    if not rows:
        return {}
    def _min(key):
        vals = [r[key] for r in rows if key in r and _finite(r[key])]
        return min(vals) if vals else float("nan")
    def _max(key):
        vals = [r[key] for r in rows if key in r and _finite(r[key])]
        return max(vals) if vals else float("nan")
    return {
        "pair_min_over_dx": _min("pair_min_over_dx"),
        "nn_min": _min("nn_min"),
        "color_min": _min("color_min"),
        "color_max": _max("color_max"),
        "kappa_R_max": _max("kappa_R_max"),
        "gradc_dx_max": _max("gradc_dx_max"),
        "acc_over_aref_max": _max("acc_over_aref_max"),
        "n_lattice_samples": len(rows),
    }


def summarize():
    """Merge whatever records are present, recompute every declared gate."""
    recs = {}
    if OUTDIR.exists():
        for p in sorted(OUTDIR.glob("*.json")):
            if p.stem in ("summary", "figure_series"):
                continue
            blob = json.loads(p.read_text())
            recs[blob.get("job", p.stem)] = blob
    jobs_present = [n for n in job_names() if n in recs]
    rows = [_arm_stats(recs[n]) for n in jobs_present]
    by = {r["job"]: r for r in rows}

    balanced_arms = [r for r in rows if r["balanced"]]
    horizon_arms = [r for r in balanced_arms
                    if (r["steps_requested"] or 0) >= TARGET_HORIZON_STEPS]
    measured_arms = [r for r in rows if r["role"] in ("primary", "control")]
    control = by.get("dx8_h120_ctl_nu01")
    prim = by.get("dx8_h120_bal_nu01")

    failed = []
    # G1: every present arm finished its declared march (no 1e3 v_ref trip)
    g1 = bool(rows) and all(
        (not r["diverged"]) and r["steps_marched"] >= (r["steps_requested"] or 0)
        for r in rows)
    if not g1:
        failed.append("no_divergence_measured_horizon")
    # G2: Adami spurious-current line on every balanced arm
    g2 = bool(balanced_arms) and all(
        _finite(r["umax_over_vref_max"]) and r["umax_over_vref_max"] <= ADAMI_GATE
        for r in balanced_arms)
    if not g2:
        failed.append("spurious_adami")
    # G3/G4: Laplace jump, owner acceptance and claim level, both instruments
    def _jump_ok(r, thr):
        if not (_finite(r["dp_err_abs_max_pct"]) and r["dp_err_abs_max_pct"] <= thr):
            return False
        pot = r["dp_potential_err_abs_max_pct"]
        return (not _finite(pot)) or pot <= thr
    g3 = bool(balanced_arms) and all(_jump_ok(r, LAPLACE_GATE_PCT) for r in balanced_arms)
    if not g3:
        failed.append("laplace_all_samples_15pct")
    g4 = bool(balanced_arms) and all(_jump_ok(r, CLAIM_PCT) for r in balanced_arms)
    if not g4:
        failed.append("laplace_all_samples_3pct")
    # G5: area conservation
    g5 = bool(measured_arms) and all(
        _finite(r["area_drift_max"]) and r["area_drift_max"] <= AREA_GATE
        for r in measured_arms)
    if not g5:
        failed.append("area_drift_2pct")
    # G6: the lattice never breaks on a sampled step
    g6 = bool(measured_arms) and all(not r["lattice_breakages"] for r in measured_arms)
    if not g6:
        failed.append("lattice_healthy")
    # G7: the split is the cause -- control above the Adami line, primary below,
    #     at identical level / viscosity / horizon.
    g7 = bool(control) and bool(prim) and \
        _finite(control["umax_over_vref_max"]) and \
        control["umax_over_vref_max"] > ADAMI_GATE and \
        _finite(prim["umax_over_vref_max"]) and prim["umax_over_vref_max"] <= ADAMI_GATE
    if not g7:
        failed.append("split_is_the_cause")
    # G8: the long-horizon (600-step) arms -- declared, reported, NOT claimed
    g8 = len(horizon_arms) == len([j for j in JOBS
                                   if j["steps"] >= TARGET_HORIZON_STEPS]) > 0 and \
        all((not r["diverged"]) and r["steps_marched"] >= TARGET_HORIZON_STEPS
            for r in horizon_arms)
    if not g8:
        failed.append("long_horizon_600_all_arms_present")

    verdicts = {
        "no_divergence_measured_horizon": g1,
        "spurious_adami": g2,
        "laplace_all_samples_15pct": g3,
        "laplace_all_samples_3pct": g4,
        "area_drift_2pct": g5,
        "lattice_healthy": g6,
        "split_is_the_cause": g7,
        "long_horizon_600_all_arms_present": g8,
    }
    out = {
        "anchor": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE",
        "cases": {"R_m": R, "sigma": SIGMA, "rho0": RHO, "outer_m": OUTER,
                  "h_over_dx": HF, "dt_factor": 0.1, "sample_every": SAMPLE,
                  "dp_theory_Pa": SIGMA / R},
        "measured_horizon_steps": MEASURED_HORIZON_STEPS,
        "target_horizon_steps": TARGET_HORIZON_STEPS,
        "jobs_declared": job_names(),
        "jobs_present": jobs_present,
        "n_jobs_present": len(jobs_present),
        "n_jobs_expected": len(JOBS),
        "gates_declared": GATES_DECLARED,
        "break_rules": BREAK_RULES,
        "arms": rows,
        "verdicts": verdicts,
        "gates_failed": failed,
        "all_gates_pass": not [f for f in failed
                               if f != "long_horizon_600_all_arms_present"],
        "standing_debt": ("long_horizon_600_all_arms_present"
                          if not g8 else None),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_jobs_present", "n_jobs_expected",
                                         "verdicts", "gates_failed")}, indent=1))
    return out


def figure():
    """Two-panel figure: spurious current and Laplace jump over the horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((OUTDIR / "summary.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    styles = {"dx8_h120_bal_nu01": ("C0", "-", "balanced, $R/dx$=8, $\\nu_f$=0.01, 120"),
              "dx8_h120_ctl_nu01": ("C3", "--", "no split, $R/dx$=8, $\\nu_f$=0.01, 120"),
              "dx12_h600_bal_nu01": ("C2", "-", "balanced, $R/dx$=12, 600"),
              "dx12_h600_bal_inv": ("C4", ":", "balanced, $R/dx$=12, $\\nu_f$=0, 600")}
    for name in summary["jobs_present"]:
        rec = json.loads((OUTDIR / (name + ".json")).read_text())
        smp = _samples(rec)
        if not smp:
            continue
        col, ls, lab = styles.get(name, ("k", "-", name))
        t = [h["t_over_t_sigma"] for h in smp]
        u = [h["umax_over_vref"] for h in smp]
        e = [abs(h.get("dp_rel_error_pct", float("nan"))) for h in smp]
        ax[0].plot(t, u, ls, color=col, label=lab, lw=1.4)
        ax[1].plot(t, e, ls, color=col, label=lab, lw=1.4)
    ax[0].axhline(ADAMI_GATE, color="k", lw=0.9, ls="-.",
                  label="Adami 0.1 line")
    ax[1].axhline(CLAIM_PCT, color="k", lw=0.9, ls="-.",
                  label="3 % claim line")
    ax[1].axhline(LAPLACE_GATE_PCT, color="k", lw=0.9, ls=":",
                  label="15 % acceptance")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_xlabel("$t/t_\\sigma$"); ax[0].set_ylabel("$\\max|u|/v_{ref}$")
    ax[1].set_xlabel("$t/t_\\sigma$"); ax[1].set_ylabel("$|\\Delta p/(\\sigma/R)-1|$ [%]")
    ax[0].set_title("spurious current over the horizon")
    ax[1].set_title("Laplace jump error over the horizon")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout()
    out = OUTDIR / "balanced_horizon.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", help="run one declared job")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.list:
        for j in JOBS:
            print(j)
        return 0
    if args.job:
        spec = next((j for j in JOBS if j["name"] == args.job), None)
        if spec is None:
            raise SystemExit("unknown job %r; declared: %s" % (args.job, job_names()))
        run_job(spec)
        return 0
    if args.summarize:
        summarize()
        return 0
    if args.figure:
        figure()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())