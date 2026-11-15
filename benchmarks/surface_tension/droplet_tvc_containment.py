"""Does transport-velocity correction buy back a *usable* dynamic surface-tension reading?

Anchor of this round: ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` (TensorSPH method
selection).  The method-selection registry carries four ``csf_surface_tension``
records and **all four come from one benchmark family** (the static Young-Laplace
jump, ``CASE-LAPLACE-SPH``): two of them pass the 15 % owner gate (5.599 %,
5.987 %), two do not (23.80 %, 45.69 %).  A single-family bound is one
measurement, so the capillary regime recommendation rests on one physics family
only.  The natural second family -- the *dynamic* Rayleigh-Lamb drop oscillation
-- was attempted before (``benchmarks/surface_tension/droplet_oscillation.py``,
``docs/xval_droplet_oscillation/README.md``) and ended in an honest FAIL: the
drop loses containment at t = 1.729 s (39 particles beyond 1.6 R, largest at
4.26 R), so the -9.76 % period reading is an escape artefact rather than a
measurement.  That document names the required fix:

    "the free surface needs a treatment that keeps the interface particles
     attached (transport-velocity / particle shifting ...), and the containment
     audit above must be a gate of every dynamic surface-tension benchmark"

This driver performs exactly that test, using the transport-velocity path that
already exists in ``tensorsph.capillary_oscillation.run_point`` (``tvc_alpha`` /
``tvc_every`` / ``tvc_max_shift`` / ``tvc_limiter``), so no solver change is
made: the question is whether an existing, already-merged knob turns a
non-measurement into a measurement.

Gates applied to every arm (all three must hold before a period reading is
allowed to count as evidence):

``contained``      no particle beyond 1.6 R, and max r/R <= 1.05
``multi_period``   at least 3 zero crossings of the mode-2 order parameter, so
                   the period is not read from a single half-oscillation
``signal``         q2 half-range >= 0.5 * eps (the mode did not relax away)

Exact reference (no adjustable parameter): Rayleigh 1879 / Lamb, *Hydrodynamics*
6th ed. arts. 313-315, ``omega_m^2 = sigma m (m^2 - 1) / (rho R^3)``, ``m = 2``.

Usage::

    PYTHONPATH=src python benchmarks/surface_tension/droplet_tvc_containment.py --arms 0,1,2
    PYTHONPATH=src python benchmarks/surface_tension/droplet_tvc_containment.py --merge
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

from tensorsph.transport_velocity_correction import truncated_linear
from tensorsph.capillary_oscillation import (
    MODE_DEFAULT,
    RADIUS_DEFAULT,
    RHO_WATER,
    SIGMA_WATER_AIR,
    rayleigh_2d,
    run_point,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(ROOT, "runs", "droplet_tvc_containment")
GATE_PCT = 15.0
CONTAINMENT_LIMIT_R = 1.05
CONTAINMENT_ESCAPE_R = 1.6

#: (group, name, R/dx, kwargs for run_point, periods to integrate)
#: ``group`` only selects which slice of the sweep a job runs; every arm is
#: stored under its own name, so the artefact is order independent.
ARMS: list[tuple[str, str, int, dict, float]] = [
    # --- A reproduction: the committed baseline reading, unchanged -------------
    ("repro", "repro_baseline_no_tvc", 10, {}, 1.0),
    # --- B containment screen: is any TVC strength usable at all? -------------
    # alpha = 0.05 is upstream's default for a *single acoustic* step
    # (``transport_velocity_correction.py``: 0.2 / 4); 0.2 is its dual-criteria
    # default; 0.25 and 0.5 are above both and test where containment dies.
    ("screen", "tvc_a002_e1", 10, dict(tvc_alpha=0.02, tvc_every=1), 1.0),
    ("screen", "tvc_a005_e1", 10, dict(tvc_alpha=0.05, tvc_every=1), 1.0),
    ("screen", "tvc_a010_e1", 10, dict(tvc_alpha=0.10, tvc_every=1), 1.0),
    ("screen", "tvc_a200_e1", 10, dict(tvc_alpha=0.20, tvc_every=1), 1.0),
    ("screen", "tvc_a025_e1", 10, dict(tvc_alpha=0.25, tvc_every=1), 1.0),
    ("screen", "tvc_a050_e1", 10, dict(tvc_alpha=0.50, tvc_every=1), 1.0),
    # clamps and limiter: the two safety devices the operator ships, applied to
    # the strongest strength that still needs help
    ("screen", "tvc_a010_e1_cap005", 10,
     dict(tvc_alpha=0.10, tvc_every=1, tvc_max_shift=0.05), 1.0),
    ("screen", "tvc_a025_e1_cap010", 10,
     dict(tvc_alpha=0.25, tvc_every=1, tvc_max_shift=0.10), 1.0),
    ("screen", "tvc_a010_e1_lin", 10,
     dict(tvc_alpha=0.10, tvc_every=1, tvc_limiter=truncated_linear(100.0)), 1.0),
    # --- C two-period runs on the configs that survived the screen ------------
    ("two_period", "tvc2p_a002_e1", 10, dict(tvc_alpha=0.02, tvc_every=1), 2.0),
    ("two_period", "tvc2p_a005_e1", 10, dict(tvc_alpha=0.05, tvc_every=1), 2.0),
    ("two_period", "tvc2p_a010_e1", 10, dict(tvc_alpha=0.10, tvc_every=1), 2.0),
    ("two_period", "tvc2p_a005_e1_cap005", 10,
     dict(tvc_alpha=0.05, tvc_every=1, tvc_max_shift=0.05), 2.0),
    # --- D0 baseline (no TVC) resolution ladder: is the committed n = 10
    #        containment failure a resolution artefact?  The committed matrix in
    #        ``droplet_oscillation.py`` lists n = 16 and n = 20 arms but the
    #        committed ``oscillation.json`` only carries the two n = 10 rows.
    ("base_ladder", "base_no_tvc_n16", 16, {}, 2.5),
    ("base_ladder", "base_no_tvc_n20", 20, {}, 2.5),
    # --- D resolution ladder on the accepted config --------------------------
    ("ladder", "tvc2p_a005_e1_n16", 16, dict(tvc_alpha=0.05, tvc_every=1), 2.0),
    ("ladder", "tvc2p_a005_e1_n20", 20, dict(tvc_alpha=0.05, tvc_every=1), 2.0),
    ("ladder", "tvc2p_a005_e1_n24", 24, dict(tvc_alpha=0.05, tvc_every=1), 2.0),
    # --- E falsification control: sigma = 0, q2 must freeze -------------------
    ("control", "control_sigma0_a005_e1", 10,
     dict(tvc_alpha=0.05, tvc_every=1, sigma=0.0), 2.0),
]


def exact_period(sigma: float = SIGMA_WATER_AIR, rho: float = RHO_WATER,
                 radius: float = RADIUS_DEFAULT, mode: int = MODE_DEFAULT) -> float:
    return 2.0 * math.pi / rayleigh_2d(sigma, rho, radius, mode)


def run_arm(idx: int, threads: int, sample_every: int) -> dict:
    _group, name, n, kw_in, periods = ARMS[idx]
    kw = dict(kw_in)
    sigma = float(kw.pop("sigma", SIGMA_WATER_AIR))
    t_exact = exact_period(sigma if sigma > 0 else SIGMA_WATER_AIR)
    r = run_point(n, sigma, periods if sigma > 0 else 0.0,
                  sample_every=sample_every,
                  t_end_s=None if sigma > 0 else periods * t_exact, **kw)
    r["arm"] = name
    r["arm_index"] = idx
    r["requested_periods"] = periods
    r["sigma_case"] = sigma
    frozen = (sigma == 0.0)
    contained = bool(r["escaped_particles"] == 0
                     and r["containment_max_r_over_R"] <= CONTAINMENT_LIMIT_R)
    finite = r["period_err_pct"] == r["period_err_pct"]
    r["gates"] = {
        "contained": contained,
        "multi_period": bool(r["zero_crossings"] >= 3),
        "signal": bool(r["q_half_range"] >= 0.5 * 0.05),
    }
    r["usable_measurement"] = bool(contained and r["gates"]["multi_period"]
                                   and r["gates"]["signal"] and finite)
    r["period_pass"] = bool(finite and abs(r["period_err_pct"]) < GATE_PCT)
    r["dp_pass"] = bool(abs(r["dp_err_pct"]) < GATE_PCT)
    r["frozen_control_ok"] = bool(frozen and r["q_half_range"] < 1.0e-3)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="comma separated arm indices (default: all)")
    ap.add_argument("--group", default=None,
                    help="run one group of the sweep (repro/screen/two_period/ladder/control)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for i, (g, name, n, kw, p) in enumerate(ARMS):
            print(i, g, name, f"R/dx={n}", kw, f"periods={p}")
        return 0

    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    os.makedirs(args.out, exist_ok=True)

    if args.merge:
        return merge(args.out)

    if args.arms:
        idxs = [int(x) for x in args.arms.split(",")]
    elif args.group:
        idxs = [i for i, arm in enumerate(ARMS) if arm[0] == args.group]
    else:
        idxs = list(range(len(ARMS)))
    if not idxs:
        print("no arms selected", file=sys.stderr)
        return 2
    for idx in idxs:
        r = run_arm(idx, args.threads, args.sample_every)
        path = os.path.join(args.out, f"arm_{idx:02d}_{r['arm']}.json")
        with open(path, "w") as fh:
            json.dump(r, fh, indent=2)
        g = r["gates"]
        print(f"[{idx:02d}] {r['arm']:24s} R/dx={r['radius_over_dx']:3d} "
              f"N={r['particles']:5d} steps={r['steps']:5d} "
              f"T={r['period_measured_s']:.4f}s err={r['period_err_pct']:+7.3f}% "
              f"dp_err={r['dp_err_pct']:+8.3f}% zc={r['zero_crossings']:2d} "
              f"max_r/R={r['containment_max_r_over_R']:.3f} "
              f"esc={r['escaped_particles']:3d} "
              f"contained={g['contained']} usable={r['usable_measurement']} "
              f"tvc_steps={r['tvc_steps']:5d} shift/h={r['tvc_shift_peak_over_h']:.3f} "
              f"wall={r['wall_s']}s", flush=True)
    return 0


def merge(out_dir: str) -> int:
    """Concatenate the per-arm files into one artefact (numbers only, no edits)."""
    files = sorted(glob.glob(os.path.join(out_dir, "arm_*.json")))
    if not files:
        print("no arm files in", out_dir, file=sys.stderr)
        return 1
    arms = []
    for f in files:
        with open(f) as fh:
            arms.append(json.load(fh))
    arms.sort(key=lambda r: r["arm_index"])
    doc = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "question": "can transport-velocity correction give a usable dynamic "
                    "surface-tension reading (2-D Rayleigh-Lamb drop oscillation)?",
        "reference": {
            "law": "omega_m^2 = sigma m (m^2-1) / (rho R^3)  (Rayleigh 1879; Lamb arts. 313-315)",
            "sigma_n_m": SIGMA_WATER_AIR,
            "rho0_kg_m3": RHO_WATER,
            "radius_m": RADIUS_DEFAULT,
            "mode": MODE_DEFAULT,
            "amp_eps": 0.05,
            "period_exact_s": exact_period(),
            "laplace_exact_pa": SIGMA_WATER_AIR / RADIUS_DEFAULT,
            "gate_pct": GATE_PCT,
        },
        "gates": {
            "contained": f"escaped == 0 and max r/R <= {CONTAINMENT_LIMIT_R}",
            "multi_period": "zero crossings of q2 >= 3",
            "signal": "q2 half range >= 0.5 * eps",
        },
        "arms": arms,
    }
    path = os.path.join(out_dir, "tvc_containment.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)
    print("wrote", path, f"({len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())