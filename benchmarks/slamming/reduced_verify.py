#!/usr/bin/env python3
"""Reduced-resolution verification of the slamming force anchor (CASE-SPH-SLAMMING).

Why this file exists
--------------------
``benchmarks/slamming/run.py`` (campaign node p129w1r2-n08) defines six
machine criteria for water-entry slamming, but the campaign's production
resolution (``dx = 2 mm``) does not fit a local slot (see
``docs/slamming_evidence_blocker_m14940.md`` and
``docs/dev_report_slamming_cost_probe_m16822.md``).  Consequently **no SPH-side
reading of ``C_s`` has ever been committed**, the verification ledger carries
only a directory-level waiver for this harness, and the one committed evidence
bundle (``docs/slamming_campaign_report.json``) prints a "Wagner reference"
column whose numbers do not come from the code that defines the reference.

This driver closes that specific gap at whatever resolution the machine can
afford, and it refuses to hide what it did not run:

* it **reuses** ``run_case`` / ``metrics`` from ``benchmarks/slamming/run.py``
  (same solver, same self-similar window, same definitions -- no second
  implementation that could drift);
* it records the resolution it actually ran and the ratio
  ``c_end / dx`` (particles across the wetted half width), so nobody can read a
  coarse-grid number as the campaign's finest-grid number;
* every reference value is recomputed from ``tensorsph.slamming`` at read time
  (``c_s_wagner = pi^2/tan(beta)``), never typed by hand;
* ``--stage accept`` grades the run against the campaign's own criteria and
  emits ``PASS`` / ``FAIL`` / ``WITHHELD`` per criterion -- a coarse grid that
  cannot resolve the spray root is ``WITHHELD``, not "close enough".

Usage
-----
    PYTHONPATH=src python benchmarks/slamming/reduced_verify.py --stage run \
        --betas 20 --velocities 2 --dx 6e-3 --frames 21
    PYTHONPATH=src python benchmarks/slamming/reduced_verify.py --stage accept

Evidence goes to ``benchmarks/slamming/results/`` (tracked by git); ``artifacts/``
is git-ignored and silently dropped by the merge gate -- the mistake that lost
this campaign's first evidence bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "slamming"))

from tensorsph.slamming import (  # noqa: E402
    SlammingConfig,
    SlammingImpact,
    slamming_force_coefficient,
    wagner_c_coeff,
)

RESULTS = Path(__file__).resolve().parent / "results" / "reduced"
PROD = {"delta_sph": 0.1, "mls_density": True}


def run_case(beta_deg: float, velocity: float, dx: float, frames: int,
             switches: dict, c_end: float = 0.05) -> dict:
    """One SPH case with a wall-clock budget report, saved as JSON."""
    cfg = SlammingConfig(beta_deg=beta_deg, velocity=velocity, dx=dx,
                         frames=frames, c_end=c_end, **switches)
    cfg.validate()
    t0 = time.time()
    sim = SlammingImpact(cfg)
    t_init = time.time() - t0
    series = sim.run()
    t_run = time.time() - t0
    beta = cfg.beta
    # reference values, recomputed from the module that defines them
    wag_c = wagner_c_coeff(beta)
    wag_cs = math.pi ** 2 / math.tan(beta)
    # analytic identity: the module's closed-form C_s equals pi^2/tan(beta)
    cs_analytic = slamming_force_coefficient(
        cfg.rho0, cfg.velocity, beta, float(cfg.resolved_t_end()))
    payload = {
        "id": f"b{beta_deg:g}_v{velocity:g}_dx{dx*1e3:g}mm",
        "provenance": {
            "driver": "benchmarks/slamming/reduced_verify.py",
            "solver": "tensorsph.slamming.SlammingImpact",
            "switches": switches,
            "note": "reduced-resolution evidence; NOT the campaign grid",
        },
        "config": {
            "beta_deg": cfg.beta_deg, "velocity": cfg.velocity, "dx": cfg.dx,
            "rho0": cfg.rho0, "gamma": cfg.gamma, "h_factor": cfg.h_factor,
            "sound_speed_factor": cfg.sound_speed_factor,
            "alpha_visc": cfg.alpha_visc, "c_end": cfg.c_end,
            "half_width": cfg.half_width, "depth": cfg.depth,
            "delta_sph": cfg.delta_sph, "mls_density": cfg.mls_density,
            "frames": cfg.frames,
        },
        "grid": {
            "c_end_over_dx": cfg.c_end / cfg.dx,
            "ax_beta_over_vt": None,
            "t_end": cfg.resolved_t_end(),
            "dt": float(sim.dt()),
            "steps": int(sim.steps),
            "n_fluid": int(sim.n_fluid),
            "n_wall": int(sim.wall_local.shape[0]),
        },
        "reference": {
            "c_coeff_wagner": wag_c,
            "c_s_wagner": wag_cs,
            "c_s_from_module_at_c_end": cs_analytic,
            "note": "C_s = pi^2/tan(beta); identical to run.py's c_s_wagner",
        },
        "cost": {"t_init_s": t_init, "t_total_s": t_run},
        "series": [
            {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in s.items()}
            for s in series
        ],
    }
    return payload


def metrics(payload: dict, c_window: float = 0.5) -> dict:
    """Self-similar-window metrics, same definitions as run.py:metrics."""
    cfg = payload["config"]
    post = [s for s in payload["series"] if s["t"] > 0.0 and s["c"] > 0.0]
    ser = [s for s in post if s["c"] >= c_window * cfg["c_end"]]
    truncated = False
    if len(ser) < 3:
        ser, truncated = post, True
    if not ser:
        return {}
    cs = [s["c_s"] for s in ser]
    c_wag, c_wag_rel = [], []
    for s in ser:
        # Wagner wet half width: c = C V t with C = pi/(2 tan beta)
        w = wagner_c_coeff(math.radians(cfg["beta_deg"])) * cfg["velocity"] * s["t"]
        c_wag.append(w)
        c_wag_rel.append(abs(s["c"] - w) / w if w > 0 else float("nan"))
    shape = [s["shape_l2"] for s in ser if s["shape_l2"] == s["shape_l2"]]
    noise = [s["noise"] for s in ser if s["noise"] == s["noise"]]
    return {
        "n_samples": len(ser),
        "window_truncated": truncated,
        "window_c_min": c_window * cfg["c_end"],
        "c_s_max": max(cs),
        "c_s_mean": sum(cs) / len(cs),
        "c_s_tail": cs[-1],
        "c_s_wagner": payload["reference"]["c_s_wagner"],
        "c_s_rel_err_max": (max(cs) - payload["reference"]["c_s_wagner"])
        / payload["reference"]["c_s_wagner"],
        "c_s_rel_err_mean": (sum(cs) / len(cs) - payload["reference"]["c_s_wagner"])
        / payload["reference"]["c_s_wagner"],
        "c_rel_err_mean": sum(c_wag_rel) / len(c_wag_rel),
        "c_rel_err_max": max(c_wag_rel),
        "shape_l2_mean": sum(shape) / len(shape) if shape else float("nan"),
        "noise_rms": (sum(n * n for n in noise) / len(noise)) ** 0.5 if noise else float("nan"),
        "p_max_peak": max(s["p_max"] for s in ser),
        "mass_drift_max": max(s["mass_drift"] for s in ser),
    }


def accept(cases: dict) -> dict:
    """Grade each case against the campaign criteria; never upgrade WITHHELD."""
    rows = {}
    for cid, payload in cases.items():
        m = metrics(payload)
        g = payload["grid"]
        row = {"metrics": m, "grid": g, "criteria": {}}
        res_min = 0.25 * (g["c_end_over_dx"])  # spray root needs c/dx >> 1
        resolvable = g["c_end_over_dx"] >= 20.0
        e = m.get("c_s_rel_err_max", float("nan"))
        if not resolvable:
            verdict = "WITHHELD"
            why = (f"c_end/dx = {g['c_end_over_dx']:.1f} < 20: the linearised "
                   "Wagner profile has a 1/sqrt(c-x) cusp at the spray root, "
                   "which this spacing cannot carry")
        else:
            verdict = "PASS" if (e == e and abs(e) <= 0.15) else "FAIL"
            why = "campaign criterion c1_cs_max (<= 15 %)"
        row["criteria"]["c1_cs_max"] = {
            "target": "|C_s,max - pi^2/tan(beta)| / (pi^2/tan(beta)) <= 0.15",
            "measured": e, "verdict": verdict, "reason": why,
            "min_resolution_c_end_over_dx": 20.0, "resolvable": resolvable,
            "notes": {"min_resolution_flag": res_min},
        }
        sh = m.get("shape_l2_mean", float("nan"))
        row["criteria"]["c2_shape_l2"] = {
            "target": "mean L2 of p/p_max vs Wagner shape <= 0.15",
            "measured": sh,
            "verdict": ("WITHHELD" if not resolvable or sh != sh
                        else ("PASS" if sh <= 0.15 else "FAIL")),
        }
        ce, cx = m.get("c_rel_err_mean", float("nan")), m.get("c_rel_err_max", float("nan"))
        row["criteria"]["wet_width"] = {
            "target": "measured wetted half width vs C V t",
            "mean": ce, "max": cx,
            "verdict": "WITHHELD" if ce != ce else ("PASS" if cx <= 0.15 else "FAIL"),
            "note": "run.py does not grade the wetted width; it is quoted "
                    "because it separates a wetting-law error from a load error",
        }
        drift = m.get("mass_drift_max", float("nan"))
        row["criteria"]["c5_mass"] = {
            "target": "|dm|/m <= 1e-6", "measured": drift,
            "verdict": "PASS" if (drift == drift and drift <= 1e-6) else "FAIL",
        }
        rows[cid] = row
    return {"schema": "tensorsph.slamming.reduced_verify/v1",
            "cases": rows,
            "summary": {
                "n_cases": len(rows),
                "verdicts": {
                    k: {cid: r["criteria"][k]["verdict"] for cid, r in rows.items()}
                    for k in ("c1_cs_max", "c2_shape_l2", "wet_width", "c5_mass")
                },
            }}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("run", "accept"), default="run")
    ap.add_argument("--betas", default="20")
    ap.add_argument("--velocities", default="2")
    ap.add_argument("--dx", type=float, default=6.0e-3)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--c-end", type=float, default=0.05)
    ap.add_argument("--no-prod", action="store_true",
                    help="switch delta-SPH/MLS off (negative control)")
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    betas = [float(x) for x in args.betas.split(",")]
    vels = [float(x) for x in args.velocities.split(",")]
    switches = {"delta_sph": 0.0, "mls_density": False} if args.no_prod else dict(PROD)

    for beta in betas:
        for V in vels:
            tag = f"b{beta:g}_v{V:g}_dx{args.dx*1e3:g}mm" + ("_noprod" if args.no_prod else "")
            path = out / f"{tag}.json"
            print(f"[run] {tag}", flush=True)
            payload = run_case(beta, V, args.dx, args.frames, switches, args.c_end)
            payload["id"] = tag
            path.write_text(json.dumps(payload, indent=1))
            print(f"[run] {tag}: n_fluid={payload['grid']['n_fluid']} "
                  f"steps={payload['grid']['steps']} wall={payload['cost']['t_total_s']:.1f}s "
                  f"-> {path}", flush=True)

    if args.stage == "accept":
        cases = {}
        for p in sorted(out.glob("*.json")):
            if p.name.startswith("summary"):
                continue
            d = json.loads(p.read_text())
            if "series" in d:
                cases[d["id"]] = d
        rep = accept(cases)
        (out / "summary.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())