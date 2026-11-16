"""Lattice-health diagnosis of the *diverging* marched projection-SPH arm.

Why this exists
---------------
``docs/xval_projection_laplace_horizon_instability.md`` ⑤ states the open
question of the anchor ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` verbatim:

    The blow-up is **located, not diagnosed**: ``lattice_health`` was added in
    this unit but no marched run with ``stability_report=True`` has been made,
    so "lattice broke" vs "traction mis-projected" is still open.  This is the
    next unit's first experiment.

This harness is that experiment.  It marches the same arm
(:func:`tensorsph.projection_adjoint.relax_projection_consistent`) with
``stability_report=True`` and ``sample_every=1`` -- every step sampled, because a
run that dies on step 3 cannot be diagnosed from samples every 10 steps -- and
writes one record per level.

Two levels matter and they fail differently:

* ``R/dx = 16`` (``dx16_nu01``) dies on its **third** step with a NaN, so the
  question is whether the entry lattice of that step was already broken;
* ``R/dx = 12`` (``dx12_horizon``) marches 180 steps and then trips the
  ``1e3 v_ref`` divergence guard, so the question there is whether the lattice
  degrades *during* the march or stays healthy while the traction grows.

Nothing here re-measures ``Delta p``: the anchor's Laplace-jump number already
passes at every finite sample (``docs/xval_projection_laplace_solenoid_scaling_m22880.md``
④).  The object under test is the *instrument reading*, not the acceptance number.

Declaration discipline
----------------------
The classification thresholds are pre-registered below (``BREAK_RULES``) before
any run, and :func:`classify` is a pure function of a sample dict so the verdict
can be re-derived from a stored record without re-running anything.
"""
from __future__ import annotations

import argparse
import json
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
NU_PRIMARY = 0.01

# --------------------------------------------------------------------------- #
# pre-registered breakage rules
# --------------------------------------------------------------------------- #
# Each rule is a property a *healthy* lattice must have.  The thresholds are
# not tuned: they are the values at which the quantity has lost its meaning.
#
#   pair_min_over_dx   a cubic-spline pair closer than 0.25 dx contributes a
#                      gradient that is 1/0.25 = 4x the lattice-scale value;
#                      0.25 dx is where the kernel normalisation is meaningless.
#   nn_min_over_median the 2h-support neighbour count of the worst particle
#                      against the run's own median.  Losing 80% of the support
#                      is what a "colour function stops being a partition of
#                      unity" event looks like on this lattice.
#   color_min/max      [0,1] is exact for a two-phase indicator; +-5% is the
#                      overshoot a *healthy* Shepard-smoothed shell produces.
#   kappa_R_max        5/R is 5x the curvature of the circle the case is made
#                      of -- a single-pair spike, not a discretisation bias.
#   gradc_dx_max       |grad c| dx is O(1) for a resolved interface (the 2-D
#                      cubic spline normalisation); 5 is a spike.
#   acc_over_aref_max  the CSF acceleration in units of sigma/(rho R^2) = 180
#                      m/s^2; 50x the reference is a force that no pressure
#                      solve can balance.
BREAK_RULES = {
    "pair_min_over_dx": {"op": "<", "value": 0.25},
    "nn_min_over_median": {"op": "<", "value": 0.20},
    "color_min": {"op": "<", "value": -0.05},
    "color_max": {"op": ">", "value": 1.05},
    "kappa_R_max": {"op": ">", "value": 5.0},
    "gradc_dx_max": {"op": ">", "value": 5.0},
    "acc_over_aref_max": {"op": ">", "value": 50.0},
}


def classify(sample: dict):
    """Pure verdict on one lattice block: ``(broken, fired_rules)``."""
    lat = sample.get("lattice") or {}
    if not lat:
        # fail-closed: a sample that carries no lattice block is not evidence of
        # health, it is *no verdict*, and the caller must see it as such.
        return True, ["no_lattice_block"]
    nn_med = lat.get("nn_median")
    fired = []
    for name, spec in BREAK_RULES.items():
        if name == "nn_min_over_median":
            if not nn_med:
                continue
            val = lat.get("nn_min", float("nan")) / nn_med
        else:
            val = lat.get(name, float("nan"))
        if val != val:                     # NaN: a broken lattice that cannot
            fired.append(name)             # even be measured counts as broken
            continue
        if (spec["op"] == "<" and val < spec["value"]) or \
           (spec["op"] == ">" and val > spec["value"]):
            fired.append(name)
    return bool(fired), fired


def first_break(rec: dict):
    """The first sampled step whose lattice block fires a breakage rule."""
    for s in rec["history"]:
        if "lattice" not in s:
            continue
        broken, fired = classify(s)
        if broken:
            return s["step"], fired
    return None, []


def health_trend(rec: dict):
    """Per-rule max/min over the sampled steps, as ``{rule: (value, step)}``."""
    trend = {}
    for s in rec["history"]:
        lat = s.get("lattice")
        if not lat:
            continue
        nn_med = lat.get("nn_median") or float("nan")
        vals = dict(lat)
        vals["nn_min_over_median"] = lat.get("nn_min", float("nan")) / nn_med
        for name, spec in BREAK_RULES.items():
            v = vals.get(name, float("nan"))
            if v != v:
                continue
            cur = trend.get(name)
            worse = (cur is None
                     or (spec["op"] == ">" and v > cur[0])
                     or (spec["op"] == "<" and v < cur[0]))
            if worse:
                trend[name] = (v, s["step"])
    return trend


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_level(level: int, steps: int, nu_factor: float = NU_PRIMARY,
              sample_every: int = 1, radius: float = R):
    dx = radius / float(level)
    t0 = time.time()
    rec = relax_projection_consistent(
        radius=radius, dx=dx, sigma=SIGMA, rho0=RHO, outer=OUTER, h_factor=HF,
        steps=steps, sample_every=sample_every, nu_factor=nu_factor,
        pin_outer=True, rel_tol=1e-6, cg_max_iter=4000,
        stability_report=True)
    rec["wall_s"] = time.time() - t0
    rec["job"] = "health_dx%d" % level
    rec["role"] = "lattice_health"
    rec["threads"] = int(torch.get_num_threads())
    broke, fired = first_break(rec)
    rec["first_break_step"] = broke
    rec["first_break_rules"] = fired
    rec["health_trend"] = health_trend(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, action="append", default=None,
                    help="R/dx; repeatable")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--nu-factor", type=float, default=NU_PRIMARY)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    levels = args.level or [16, 12]
    outdir = Path(args.outdir) if args.outdir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    for lvl in levels:
        steps = args.steps if args.steps else (8 if lvl >= 16 else 200)
        rec = run_level(lvl, steps, args.nu_factor, args.sample_every)
        p = outdir / ("lattice_health_dx%d.json" % lvl)
        p.write_text(json.dumps(rec, indent=1))
        print("[dx%d] n=%d steps=%d diverged=%s first_break=%s rules=%s "
              "umax=%.4g dp_err=%.3f%% wall=%.1fs"
              % (lvl, rec["n_particles"], rec["steps_completed"],
                 rec["diverged"], rec["first_break_step"],
                 rec["first_break_rules"], rec["umax_over_vref_final"],
                 rec["dp_rel_error_pct"], rec["wall_s"]), flush=True)


if __name__ == "__main__":
    main()