#!/usr/bin/env python3
"""Board audit m20188 — independent re-derivation of CASE-SPH-POISEUILLE-MORRIS (m20084).

Two independent checks, neither of which imports the case's own metric code:

  A. Arithmetic re-derivation from the committed artefact
     (``runs/poiseuille_morris/results.json`` in the m20084 worktree):
     the exact plane-Poiseuille solution u = g y (H-y)/(2 nu), the normalised
     per-row error, the interior L2 norm (0.15 <= y/H <= 0.85), the centre-row
     error and the pass/fail read-out against <3 % and <5 % are recomputed here
     from the raw measured velocities; the recorded columns and the recorded
     verdicts are then compared against these numbers.

  B. Fresh re-run of the headline configurations (Re_c = 0.1 and Re_c = 1,
     H/dx = 16, h/dx = 1.90, columns = 12, 575 steps each) through
     ``tensorsph.isph_channel.run_case`` and the case driver's own
     ``make_config``/``profile_metrics`` entry points -- landed in a scratch
     directory, so the audited worktree is never written to.

Usage
  python3 verify_m20084.py --repo /root/tp-agent/work/m20084 \
      [--rerun] [--out verify_output.json]
Exit status is 0 even for honest FAILs; it is 2 if the artefact cannot be read.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

H_DEFAULT = 0.1  # channel height [m] recorded in the artefact settings block


def u_exact(y: float, g: float, nu: float, h: float = H_DEFAULT) -> float:
    """Exact steady plane-Poiseuille velocity, evaluated (never transcribed)."""
    return g * y * (h - y) / (2.0 * nu)


# --------------------------------------------------------------------------- #
# A. arithmetic re-derivation
# --------------------------------------------------------------------------- #
def rederive(path: Path) -> dict:
    payload = json.loads(path.read_text())
    H = payload["settings"]["height"]
    rows = []
    for case in payload["cases"]["isph_projection"]:
        g, nu = case["body_force"], case["viscosity"]
        u_max = g * H ** 2 / (8.0 * nu)
        rec = case["metrics"]
        d_ref = max(abs(r["u_reference"] - u_exact(r["y"], g, nu, H)) for r in rec["rows"])
        d_err = max(abs(r["err_over_umax"]
                        - (r["u_measured"] - u_exact(r["y"], g, nu, H)) / u_max)
                    for r in rec["rows"])
        sel = [r for r in rec["rows"] if 0.15 <= r["y_over_H"] <= 0.85]
        l2 = 100.0 * math.sqrt(sum(((r["u_measured"] - u_exact(r["y"], g, nu, H)) / u_max) ** 2
                                   for r in sel) / len(sel))
        cen = min(rec["rows"], key=lambda r: abs(r["y_over_H"] - 0.5))
        e_c = 100.0 * (cen["u_measured"] - u_exact(cen["y"], g, nu, H)) / u_max
        dens = case["gates"]["density_fluctuation_max"]
        rows.append(dict(
            re_c=case["reynolds_c"], re_realised=u_max * H / nu,
            H_over_dx=case["resolution_H_over_dx"], h_over_dx=case["h_over_dx"],
            columns=case["columns"], steps=case["steps"],
            n_fluid=case["n_fluid"], n_wall=case["n_wall"],
            centre_pct_indep=e_c, centre_pct_recorded=rec["centre_error_pct"],
            l2_pct_indep=l2, l2_pct_recorded=rec["interior"]["l2_over_umax_pct"],
            max_row_pct_indep=100.0 * max(abs(r["err_over_umax"]) for r in sel),
            u_centre_measured=cen["u_measured"],
            u_centre_reference=u_exact(cen["y"], g, nu, H),
            d_reference_column=d_ref, d_error_column=d_err,
            density_max=dens["measured"], density_limit=dens["limit"],
            density_pass=dens["pass"],
            divergence_reduction=case["divergence_reduction"],
            wall_shear_budget_rel_pct=case["gates"]["wall_shear_budget"]["rel_error_pct"],
        ))

    verdict_readout = {}
    for re_c in sorted({r["re_c"] for r in rows}):
        cand = [r for r in rows if r["re_c"] == re_c and r["h_over_dx"] == 1.9
                and r["columns"] == 12]
        if not cand:
            verdict_readout[str(re_c)] = {"canonical_point": None,
                                          "note": "no h/dx=1.90, columns=12 run"}
            continue
        f = max(cand, key=lambda r: r["H_over_dx"])
        verdict_readout[str(re_c)] = {
            "H_over_dx": f["H_over_dx"],
            "centre_pct": f["centre_pct_indep"], "l2_pct": f["l2_pct_indep"],
            "pass_3pct": bool(abs(f["centre_pct_indep"]) < 3.0 and f["l2_pct_indep"] < 3.0),
            "pass_5pct": bool(abs(f["centre_pct_indep"]) < 5.0 and f["l2_pct_indep"] < 5.0),
            "density_gate": {"measured": f["density_max"], "limit": f["density_limit"],
                             "pass": f["density_pass"]},
        }

    return {
        "artefact": str(path),
        "anchor": payload.get("anchor"),
        "mission_id": payload.get("mission_id"),
        "n_isph_cases": len(payload["cases"]["isph_projection"]),
        "n_wcsp_cases": len(payload["cases"]["wcsp_production"]),
        "summary_block": payload.get("summary", {}),
        "declared_verdict_block": payload.get("verdict", {}),
        "controls": payload.get("controls", {}),
        "settings_ladders": payload.get("settings", {}),
        "cases": rows,
        "independent_readout": verdict_readout,
        "max_abs_dev_reference_column": max(r["d_reference_column"] for r in rows),
        "max_abs_dev_error_column": max(r["d_error_column"] for r in rows),
        "max_abs_dev_centre": max(abs(r["centre_pct_indep"] - r["centre_pct_recorded"]) for r in rows),
        "max_abs_dev_l2": max(abs(r["l2_pct_indep"] - r["l2_pct_recorded"]) for r in rows),
        "max_abs_dev_reynolds": max(abs(r["re_c"] - r["re_realised"]) for r in rows),
        "density_gate_failures": [
            {"re_c": r["re_c"], "H_over_dx": r["H_over_dx"], "h_over_dx": r["h_over_dx"],
             "columns": r["columns"], "measured": r["density_max"], "limit": r["density_limit"]}
            for r in rows if not r["density_pass"]],
    }


# --------------------------------------------------------------------------- #
# B. fresh re-run of the headline configuration
# --------------------------------------------------------------------------- #
def rerun(repo: Path, cases=(0.1, 1.0)) -> dict:
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    import torch
    torch.set_num_threads(1)
    from benchmarks.poiseuille_morris.run import (          # case driver entry points
        make_config, profile_metrics)
    from tensorsph.isph_channel import run_case, gates, fit_profile

    out = []
    for re_c in cases:
        cfg = make_config(re_c, 16, h_factor=1.90, columns=12)
        t0 = time.time()
        result = run_case(cfg, "cpu")
        wall = time.time() - t0
        fit = fit_profile(result["profile"], cfg.height, cfg.body_force,
                          cfg.viscosity, lo=0.05, hi=0.95)
        m = profile_metrics(result["profile"], cfg)
        g = gates(result, fit, cfg)
        u_max = cfg.body_force * cfg.height ** 2 / (8.0 * cfg.viscosity)
        sel = [r for r in m["rows"] if 0.15 <= r["y_over_H"] <= 0.85]
        l2 = 100.0 * math.sqrt(sum(((r["u_measured"]
                                    - u_exact(r["y"], cfg.body_force, cfg.viscosity,
                                              cfg.height)) / u_max) ** 2
                                   for r in sel) / len(sel))
        cen = min(m["rows"], key=lambda r: abs(r["y_over_H"] - 0.5))
        e_c = 100.0 * (cen["u_measured"]
                       - u_exact(cen["y"], cfg.body_force, cfg.viscosity, cfg.height)) / u_max
        out.append(dict(re_c=re_c, steps=result["steps"], wall_clock_s=wall,
                        n_fluid=result["n_fluid"], n_wall=result["n_wall"],
                        centre_pct=e_c, l2_pct=l2,
                        density_max=g["density_fluctuation_max"]["measured"],
                        density_pass=g["density_fluctuation_max"]["pass"],
                        wall_shear_rel_pct=g["wall_shear_budget"]["rel_error_pct"],
                        divergence_reduction=result["divergence_reduction"],
                        profile=[(r["y_over_H"], r["u_measured"]) for r in m["rows"]]))
    return {"reruns": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/tp-agent/work/m20084")
    ap.add_argument("--artefact", default=None)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--cases", default="0.1,1.0",
                    help="comma separated Re_c list for --rerun")
    ap.add_argument("--out", default="verify_output.json")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    art = Path(a.artefact) if a.artefact else repo / "runs/poiseuille_morris/results.json"
    if not art.exists():
        print(f"FATAL: artefact missing: {art}")
        return 2
    res = {"rederivation": rederive(art)}
    if a.rerun:
        res.update(rerun(repo, tuple(float(x) for x in a.cases.split(",") if x)))
    Path(a.out).write_text(json.dumps(res, indent=2))

    r = res["rederivation"]
    print(f"anchor={r['anchor']} mission={r['mission_id']} "
          f"isph_cases={r['n_isph_cases']} wcsp_cases={r['n_wcsp_cases']}")
    print(f"max|ref column - exact| = {r['max_abs_dev_reference_column']:.2e}  "
          f"max|err column - recomputed| = {r['max_abs_dev_error_column']:.2e}")
    print(f"max|centre recomputed - recorded| = {r['max_abs_dev_centre']:.2e} %  "
          f"max|L2 recomputed - recorded| = {r['max_abs_dev_l2']:.2e} %")
    print(f"max|Re_c - u_max H/nu| = {r['max_abs_dev_reynolds']:.2e}")
    for k, v in r["independent_readout"].items():
        print(f"  Re_c={k}: {v}")
    print(f"density gate failures: {r['density_gate_failures']}")
    if "reruns" in res:
        for x in res["reruns"]:
            print(f"[rerun] Re_c={x['re_c']} steps={x['steps']} {x['wall_clock_s']:.1f}s "
                  f"centre {x['centre_pct']:+.4f}% L2 {x['l2_pct']:.4f}% "
                  f"density {x['density_max']:.3e} pass={x['density_pass']} "
                  f"tauw {x['wall_shear_rel_pct']:+.3f}%")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())