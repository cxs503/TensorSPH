"""Does *particle shifting* buy back the dynamic surface-tension reading that TVC lost?

Anchor: ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` (TensorSPH method selection).

The chain of evidence this driver continues
-------------------------------------------
1. ``csf_surface_tension`` carries four registry records and **all four come from
   one physics family** (the static Young-Laplace jump, ``CASE-LAPLACE-SPH``), so
   the capillary recommendation rests on a single family
   (``benchmarks/method_selection/build_registry.py``).
2. The dynamic second family -- 2-D Rayleigh-Lamb drop oscillation, exact
   ``omega_m^2 = sigma m (m^2 - 1) / (rho R^3)`` (Rayleigh 1879; Lamb,
   *Hydrodynamics* 6th ed. arts. 313-315) -- was attempted and ended in an honest
   FAIL: containment is lost at t = 1.729 s (39 particles beyond 1.6 R, largest at
   4.26 R), so the -9.76 % period reading is an escape artefact
   (``docs/xval_droplet_oscillation/README.md``).
3. ``docs/xval_method_selection_csf_dynamic_m22526.md`` tested the first recorded
   remedy, transport-velocity correction, over 9 arms and **falsified** it: TVC
   displaces particles along ``-grad(1)``, which at a free surface points
   *outward*, so every TVC arm lost containment harder than the baseline
   (max r/R 3.08-52.6).  That document names the one route left: particle
   shifting, whose free-surface branch projects the displacement onto the tangent
   plane (``src/tensorsph/particle_shifting.py``, ``fs_mask``).

This driver performs that test.  It changes no physics: ``run_point`` gained a
``shift_*`` pass-through to the already-merged Lind et al. (2012) operator
(``shift_strength`` / ``shift_every`` / ``shift_limit`` / ``shift_free_surface``,
all defaulting to off), and the sweep is a one-dimensional scan of the operator's
own dimensionless coefficient ``C_d``.

Gates (all three must hold before a period reading counts as evidence)
---------------------------------------------------------------------
``contained``     no particle beyond 1.6 R **and** max r/R <= 1.05
``multi_period``  at least 3 zero crossings of the mode-2 order parameter
``signal``        q2 half range >= 0.5 * eps (the mode did not relax away)

Plus one falsification control: ``sigma = 0`` with shifting on must freeze the
mode, otherwise a passing arm could be an artefact of the shifting operator
manufacturing motion rather than transporting surface energy.

Usage::

    PYTHONPATH=src python benchmarks/surface_tension/droplet_shifting_containment.py --list
    PYTHONPATH=src python benchmarks/surface_tension/droplet_shifting_containment.py --arms 3
    PYTHONPATH=src python benchmarks/surface_tension/droplet_shifting_containment.py --merge
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

from tensorsph.capillary_oscillation import (
    MODE_DEFAULT,
    RADIUS_DEFAULT,
    RHO_WATER,
    SIGMA_WATER_AIR,
    rayleigh_2d,
    run_point,
)

DEFAULT_OUT = os.path.join("runs", "droplet_shifting_containment")
CONTAINMENT_LIMIT_R = 1.05
GATE_PCT = 15.0
EPS = 0.05
#: C_d of Lind et al. (2012) is 1; the sweep brackets it on both sides.
ARMS: list[tuple[str, str, int, dict, float, str]] = [
    # group, name, R/dx, run_point kwargs, periods, intent
    # --- A repro gate: the committed escape must be reproduced --------------
    ("repro", "repro_baseline_no_shift", 10, {}, 1.0,
     "reproduce the committed -9.76 % / 39 escaped particles (gate for the series)"),
    # --- B duration control: the 2-period window at R/dx = 10 --------------
    #      Measured before this control existed: the drop blows up over 2
    #      periods *without* any shifting (max r/R 2.4e4, 301 escaped), so a
    #      blow-up in a shifting arm at 2 periods cannot be attributed to
    #      shifting until this baseline is on the record.
    ("duration", "baseline_no_shift_2p", 10, {}, 2.0,
     "equal-duration control: what does the drop do over 2 periods with no shifting?"),
    # --- C the C_d scan at R/dx = 10 (same window as the control) ----------
    ("screen", "shift_cd0p05", 10, dict(shift_strength=0.05), 2.0, "weak shifting"),
    ("screen", "shift_cd0p20", 10, dict(shift_strength=0.20), 2.0,
     "shifting at 1/5 of Lind's C_d"),
    ("screen", "shift_cd0p50", 10, dict(shift_strength=0.50), 2.0,
     "shifting at 1/2 of Lind's C_d"),
    ("screen", "shift_cd1", 10, dict(shift_strength=1.0), 2.0,
     "Lind et al. (2012) coefficient C_d = 1"),
    ("screen", "shift_cd2", 10, dict(shift_strength=2.0), 2.0,
     "twice Lind's C_d (over-shifting check)"),
    ("screen", "shift_cd1_cap0p1", 10, dict(shift_strength=1.0, shift_limit=0.1), 2.0,
     "same C_d, tighter 0.1 h displacement cap"),
    # --- D resolution ladder: R/dx = 16 (the drop is 315 particles at 10) --
    ("ladder", "baseline_no_shift_n16", 16, {}, 2.0,
     "no-shift baseline at the same resolution"),
    ("ladder", "shift_cd0p20_n16", 16, dict(shift_strength=0.20), 2.0,
     "resolution ladder, weak shifting"),
    ("ladder", "shift_cd1_n16", 16, dict(shift_strength=1.0), 2.0,
     "resolution ladder, Lind's C_d"),
    ("ladder", "shift_cd2_n16", 16, dict(shift_strength=2.0), 2.0,
     "resolution ladder, strong shifting"),
    # --- E resolution ladder: R/dx = 20 ------------------------------------
    ("ladder", "baseline_no_shift_n20", 20, {}, 2.0,
     "no-shift baseline at R/dx = 20"),
    ("ladder", "shift_cd1_n20", 20, dict(shift_strength=1.0), 2.0,
     "R/dx = 20 with Lind's C_d"),
    # --- F falsification control: sigma = 0, shifting on -------------------
    ("control", "control_sigma0_shift_cd1", 10,
     dict(shift_strength=1.0, sigma=0.0), 2.0,
     "sigma = 0: shifting alone must not make the mode oscillate"),
    # --- G top of the ladder ----------------------------------------------
    ("ladder", "shift_cd1_n24", 24, dict(shift_strength=1.0), 2.0,
     "R/dx = 24 with Lind's C_d (cost ceiling of the ladder)"),
    # --- H discriminator: is the blow-up clumping/pressure noise (delta-SPH's
    #      target) or a surface-tension force imbalance (which it cannot fix)?
    #      ``density_diffusion`` is the Antuono, Colagrossi & Marrone (2012)
    #      delta term already wired into ``run_point`` and default 0.
    ("diagnose", "dd0p1_n16", 16, dict(density_diffusion=0.1), 2.0,
     "delta-SPH density diffusion 0.1 at R/dx = 16, no shifting: does the drop "
     "survive when the pressure field is filtered instead?"),
    ("diagnose", "dd0p1_n20", 20, dict(density_diffusion=0.1), 2.0,
     "same at R/dx = 20"),
    ("diagnose", "dd0p1_shift_cd1_n16", 16,
     dict(density_diffusion=0.1, shift_strength=1.0), 2.0,
     "both remedies together at R/dx = 16"),
]


def exact_period(sigma: float = SIGMA_WATER_AIR, rho: float = RHO_WATER,
                 radius: float = RADIUS_DEFAULT, mode: int = MODE_DEFAULT) -> float:
    return 2.0 * math.pi / rayleigh_2d(sigma, rho, radius, mode)


def run_arm(idx: int, threads: int, sample_every: int) -> dict:
    _group, name, n, kw_in, periods, intent = ARMS[idx]
    kw = dict(kw_in)
    sigma = float(kw.pop("sigma", SIGMA_WATER_AIR))
    t_exact = exact_period(sigma if sigma > 0 else SIGMA_WATER_AIR)
    r = run_point(n, sigma, periods if sigma > 0 else 0.0,
                  sample_every=sample_every, eps=EPS,
                  t_end_s=None if sigma > 0 else periods * t_exact, **kw)
    r["arm"] = name
    r["arm_index"] = idx
    r["group"] = _group
    r["intent"] = intent
    r["requested_periods"] = periods
    r["sigma_case"] = sigma
    frozen = (sigma == 0.0)
    contained = bool(r["escaped_particles"] == 0
                     and r["containment_max_r_over_R"] <= CONTAINMENT_LIMIT_R)
    finite = r["period_err_pct"] == r["period_err_pct"]
    r["gates"] = {
        "contained": contained,
        "multi_period": bool(r["zero_crossings"] >= 3),
        "signal": bool(r["q_half_range"] >= 0.5 * EPS),
    }
    r["usable_measurement"] = bool(contained and r["gates"]["multi_period"]
                                   and r["gates"]["signal"] and finite)
    r["period_pass"] = bool(finite and abs(r["period_err_pct"]) < GATE_PCT)
    r["dp_pass"] = bool(abs(r["dp_err_pct"]) < GATE_PCT)
    r["frozen_control_ok"] = bool(frozen and r["q_half_range"] < 1.0e-3)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None, help="comma separated arm indices")
    ap.add_argument("--group", default=None,
                    help="one group of the sweep (repro/screen/ladder/control)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for i, (g, name, n, kw, p, intent) in enumerate(ARMS):
            print(i, g, name, f"R/dx={n}", kw, f"periods={p}", "|", intent)
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
        print(f"[{idx:02d}] {r['arm']:26s} R/dx={r['radius_over_dx']:3d} "
              f"N={r['particles']:5d} steps={r['steps']:5d} "
              f"T={r['period_measured_s']:.4f}s err={r['period_err_pct']:+8.3f}% "
              f"dp_err={r['dp_err_pct']:+8.3f}% zc={r['zero_crossings']:2d} "
              f"q_half={r['q_half_range']:.5f} "
              f"max_r/R={r['containment_max_r_over_R']:.3f} "
              f"esc={r['escaped_particles']:3d} "
              f"contained={g['contained']} usable={r['usable_measurement']} "
              f"shift_steps={r['shift_steps']:5d} "
              f"peak/h={r['shift_peak_over_h']:.2e} "
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
        "question": "can Lind et al. (2012) particle shifting give a usable "
                    "dynamic surface-tension reading (2-D Rayleigh-Lamb drop "
                    "oscillation) where transport-velocity correction failed?",
        "predecessor": "docs/xval_method_selection_csf_dynamic_m22526.md",
        "reference": {
            "law": "omega_m^2 = sigma m (m^2-1) / (rho R^3)  "
                   "(Rayleigh 1879; Lamb arts. 313-315)",
            "sigma_n_m": SIGMA_WATER_AIR,
            "rho0_kg_m3": RHO_WATER,
            "radius_m": RADIUS_DEFAULT,
            "mode": MODE_DEFAULT,
            "amp_eps": EPS,
            "period_exact_s": exact_period(),
            "laplace_exact_pa": SIGMA_WATER_AIR / RADIUS_DEFAULT,
            "gate_pct": GATE_PCT,
        },
        "gates": {
            "contained": f"escaped == 0 and max r/R <= {CONTAINMENT_LIMIT_R}",
            "multi_period": "zero crossings of q2 >= 3",
            "signal": f"q2 half range >= 0.5 * eps = {0.5 * EPS}",
        },
        "arms": arms,
    }
    path = os.path.join(out_dir, "shifting_containment.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)
    print("wrote", path, f"({len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())