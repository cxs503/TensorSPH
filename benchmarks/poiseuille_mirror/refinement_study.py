#!/usr/bin/env python3
"""Independent refinement ladder past ``H/dx = 16`` for the Poiseuille anchor.

Why this exists
---------------
``docs/verification_case_poiseuille_sph_anchor.md`` closes the anchor
``CASE-POISEUILLE-SPH`` as ``PASS_WITH_CAVEAT`` and states, verbatim, two
limits of that verdict:

* "Everything here is a re-reading of one committed trajectory; nothing is
  re-solved.  Convergence under refinement is therefore inferred from the
  three rungs in the artifact, not from an independently refined run."
* the strictest local wall-shear gauge (G1, bias-corrected) only "enters the
  band at ``H/dx = 16`` (+4.61 %)", so the honest resolution requirement for
  the wall layer is *extrapolated* off the top rung of a ladder that stops at
  ``H/dx = 16``.

Both statements are testable.  This study

1. **re-solves** the case from scratch at ``H/dx = 16, 24, 32, 48`` (the
   ``16`` rung is deliberately repeated: it is the same-session control that
   binds this ladder to the committed one at
   ``benchmarks/poiseuille_mirror/out/results.json``), and
2. measures the **observed convergence order** of every published reading,
   including the two strict local secants G1/G2 and the fitted effective
   viscosity ``nu_eff / nu``, instead of inferring it from three rungs.

The question this answers is not "does the case pass at ``H/dx = 16``" (that is
already published) but "**does the error keep decaying above it, and at what
resolution does the 5 % / 3 % line get met**" -- i.e. whether the published
resolution clause is a floor or a plateau.

Method: nothing in the solver changes.  ``run_case_mirror`` from
``benchmarks/poiseuille_mirror/run.py`` is driven with the same configuration
(``h = 1.3 dx``, ``12`` columns, ``t_end = 1.0``, ``t_average = 0.2``, Tait EOS
at ``c0 = 20``), so the refinement axis is the row spacing alone.  The gauges
are the published ones from ``gauge_audit.py`` (G1 wall-plane secant, G2
row-pair secant, G3 three-point one-sided, G4 global parabola fit), each
reported raw and stencil-bias-corrected, so the numbers stay comparable with
the anchor verdict.

Usage (per-rung, then assemble):

    PYTHONPATH=src python benchmarks/poiseuille_mirror/refinement_study.py \
        --only 24
    PYTHONPATH=src python benchmarks/poiseuille_mirror/refinement_study.py \
        --assemble

``--assemble`` is pure post-processing of the per-rung JSON files and needs no
solver state; it writes ``out/refinement.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from benchmarks.poiseuille.run import make_config
from benchmarks.poiseuille_mirror.gauge_audit import audit_run
from benchmarks.poiseuille_mirror.run import evaluate, run_case_mirror
from tensorsph.droplet_impact import resolve_device

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/poiseuille_mirror/out"
COMMITTED = OUT / "results.json"
TOL_PCT = 5.0          # the anchor's own line
TOL_STRICT_PCT = 3.0   # owner standard ("<3 %" asked for explicitly)

# Reading -> where it lives in the audit payload.  Each entry is
# (label, getter, direction) with direction "abs" (error in %) or "value".
READINGS = {
    "u_max_error_pct": ("centreline |u_max/u_ref - 1|", "abs"),
    "rms_interior_rel_pct": ("profile row-rms, interior 0.1<=y/H<=0.9", "abs"),
    "rms_all_rel_pct": ("profile row-rms, all rows", "abs"),
    "rms_umax_pct": ("profile row-rms normalised by u_max", "abs"),
    "max_row_rel_pct": ("worst single row", "abs"),
    "G1_raw_pct": ("wall shear, wall-plane secant (raw)", "abs"),
    "G1_corrected_pct": ("wall shear, wall-plane secant (stencil-corrected)", "abs"),
    "G2_raw_pct": ("wall shear, row-pair secant (raw)", "abs"),
    "G2_corrected_pct": ("wall shear, row-pair secant (stencil-corrected)", "abs"),
    "G3_corrected_pct": ("wall shear, 3-point one-sided (stencil-corrected)", "abs"),
    "G4_pct": ("wall shear, global parabola fit (published)", "abs"),
    "nu_eff_rel_pct": ("effective viscosity |nu_eff/nu - 1|", "abs"),
    "slip_over_umax": ("slip at the wall plane / u_max", "value"),
}


# --------------------------------------------------------------------------- #
# one rung
# --------------------------------------------------------------------------- #

def run_rung(resolution: int, *, viscosity: float, body_force: float,
             height: float, columns: int, t_end: float, t_average: float,
             device, threads: int) -> dict:
    if device in ("auto", "cpu"):
        torch.set_num_threads(threads)
    dev = resolve_device(device)
    dx = height / resolution
    length = columns * dx
    cfg = make_config(dx, viscosity, 1.0, dev)
    t0 = time.time()
    case = run_case_mirror(cfg, height, length, body_force, t_end, t_average,
                           dev, slip="no_slip")
    res = evaluate(case, height, body_force, viscosity, 1000.0, resolution)
    audit = audit_run(res, 1000.0, height)
    gauges = audit["wall"]["gauges"]
    fit = audit["wall"]["fit"]
    nu_eff = fit.get("nu_eff_over_nu")
    payload = {
        "resolution_H_over_dx": resolution,
        "dx": dx,
        "h": 1.3 * dx,
        "length": length,
        "n_fluid": res["n_fluid"],
        "n_ghost": res["n_ghost"],
        "steps": res["steps"],
        "dt": res["dt"],
        "t_end": t_end,
        "t_average": t_average,
        "max_mach": res["max_mach"],
        "steady_drift": res["steady_drift"],
        "wall_clock_s": time.time() - t0,
        "profile": {
            "u_max_measured": audit["profile"]["u_max_measured"],
            "u_max_analytic": audit["profile"]["u_max_analytic"],
            "u_max_error_pct": audit["profile"]["u_max_error_pct"],
            "rms_interior_rel_pct": audit["profile"]["rms_interior_rel_pct"],
            "rms_all_rel_pct": audit["profile"]["rms_all_rel_pct"],
            "rms_umax_pct": audit["profile"]["rms_umax_pct"],
            "max_row_rel_pct": audit["profile"]["max_row_rel_pct"],
            "n_rows": audit["profile"]["n_rows"],
        },
        "wall": {
            "tau_ref_pa": audit["wall"]["tau_ref_pa"],
            "slope_ref": audit["wall"]["slope_ref"],
            "fit_c0_c1_c2": [fit["c0"], fit["c1"], fit["c2"]],
            "nu_eff_over_nu": nu_eff,
            "nu_eff_rel_pct": (100.0 * abs(nu_eff - 1.0) if nu_eff is not None else None),
            "fit_residual_rms_over_umax": (fit["fit_residual_rms"]
                                           / audit["profile"]["u_max_measured"]),
            "gauges": {k: {"ratio": v["ratio"],
                           "error_pct": v["error_pct"],
                           "stencil_bias": v["stencil_bias"],
                           "error_pct_bias_corrected": v["error_pct_bias_corrected"]}
                       for k, v in gauges.items()},
        },
        "slip_over_umax": res["wall_slip_intercept_over_umax"],
        # raw rows kept so any later re-reading needs no solver run
        "rows": [{"y": r["y"], "y_over_H": r["y_over_H"],
                  "u_mean": r["u_mean"], "u_std_x": r["u_std_x"]}
                 for r in res["rows"]],
        "trace": res["trace"],
    }
    return payload


def flatten(rung: dict) -> dict:
    """One rung -> the scalar readings that get order-fitted."""
    p = rung["profile"]
    g = rung["wall"]["gauges"]
    return {
        "u_max_error_pct": p["u_max_error_pct"],
        "rms_interior_rel_pct": p["rms_interior_rel_pct"],
        "rms_all_rel_pct": p["rms_all_rel_pct"],
        "rms_umax_pct": p["rms_umax_pct"],
        "max_row_rel_pct": p["max_row_rel_pct"],
        "G1_raw_pct": g["G1_secant_wall_plane"]["error_pct"],
        "G1_corrected_pct": g["G1_secant_wall_plane"]["error_pct_bias_corrected"],
        "G2_raw_pct": g["G2_secant_row_pair"]["error_pct"],
        "G2_corrected_pct": g["G2_secant_row_pair"]["error_pct_bias_corrected"],
        "G3_corrected_pct": g["G3_three_point_quad"]["error_pct_bias_corrected"],
        "G4_pct": g["G4_global_fit_slope"]["error_pct"],
        "nu_eff_rel_pct": rung["wall"]["nu_eff_rel_pct"],
        "slip_over_umax": rung["slip_over_umax"],
    }


# --------------------------------------------------------------------------- #
# convergence analysis (pure, testable)
# --------------------------------------------------------------------------- #

def observed_orders(rs, es) -> list:
    """Pairwise observed order between consecutive rungs: e ~ C r^-p."""
    out = []
    for i in range(1, len(rs)):
        e1, e2 = abs(es[i - 1]), abs(es[i])
        if e1 <= 0.0 or e2 <= 0.0 or rs[i] == rs[i - 1]:
            out.append(None)
            continue
        out.append(math.log(e1 / e2) / math.log(rs[i] / rs[i - 1]))
    return out


def fit_power_law(rs, es):
    """Least-squares fit of log|e| = log C - p log r.  Returns (p, C, rms)."""
    pairs = [(r, abs(e)) for r, e in zip(rs, es) if e is not None and abs(e) > 0.0]
    if len(pairs) < 2:
        return None
    xs = [math.log(r) for r, _ in pairs]
    ys = [math.log(e) for _, e in pairs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return {"p": -slope, "C": math.exp(intercept), "log_rms": resid}


def resolution_for(error_pct_model: dict, tol_pct: float):
    """Resolution at which the fitted model crosses ``tol_pct``; None if it can't."""
    if not error_pct_model or error_pct_model["p"] <= 0.0:
        return None
    return (error_pct_model["C"] / tol_pct) ** (1.0 / error_pct_model["p"])


def analyse(rungs: list, tol_pct: float = TOL_PCT,
            tol_strict_pct: float = TOL_STRICT_PCT) -> dict:
    rungs = sorted(rungs, key=lambda r: r["resolution_H_over_dx"])
    rs = [r["resolution_H_over_dx"] for r in rungs]
    flats = [flatten(r) for r in rungs]
    table = {}
    for key, (label, kind) in READINGS.items():
        es = [f[key] for f in flats]
        finite = [e for e in es if e is not None]
        model = fit_power_law(rs, es)
        worst = max(abs(e) for e in finite) if finite else None
        table[key] = {
            "label": label,
            "kind": kind,
            "values": es,
            "worst_abs": worst,
            "pass_5pct_all_rungs": (all(abs(e) <= tol_pct for e in finite)
                                    if finite else None),
            "pass_3pct_all_rungs": (all(abs(e) <= tol_strict_pct for e in finite)
                                    if finite else None),
            "remaining_error_at_coarsest_pct": (abs(es[0]) if es[0] is not None else None),
            "observed_order_pairwise": observed_orders(rs, es),
            "power_law_fit": model,
            "resolution_for_5pct": resolution_for(model, tol_pct),
            "resolution_for_3pct": resolution_for(model, tol_strict_pct),
        }
    return {
        "resolutions": rs,
        "tolerance_pct": tol_pct,
        "tolerance_strict_pct": tol_strict_pct,
        "readings": table,
    }


def verdict(analysis: dict) -> dict:
    """Answer the two questions the anchor verdict left open."""
    t = analysis["readings"]
    g1 = t["G1_corrected_pct"]
    g1_raw = t["G1_raw_pct"]
    rms_int = t["rms_interior_rel_pct"]
    u_max = t["u_max_error_pct"]

    def decays(entry):
        m = entry["power_law_fit"]
        return bool(m and m["p"] > 0.2)

    questions = {
        "Q1_wall_layer_keeps_decaying_above_16": {
            "reading": g1["label"],
            "values_pct": g1["values"],
            "observed_order": g1["power_law_fit"]["p"] if g1["power_law_fit"] else None,
            "decays": decays(g1),
        },
        "Q2_resolution_for_5pct_wall_layer": {
            "reading": g1["label"],
            "H_over_dx": g1["resolution_for_5pct"],
            "basis": "power-law fit over this ladder (extrapolation, not a run)",
        },
        "Q3_resolution_for_3pct_wall_layer": {
            "reading": g1["label"],
            "H_over_dx": g1["resolution_for_3pct"],
        },
        "Q4_profile_and_centreline_under_3pct": {
            "interior_rms_worst_pct": rms_int["worst_abs"],
            "interior_rms_pass_3pct": rms_int["pass_3pct_all_rungs"],
            "centreline_worst_pct": u_max["worst_abs"],
            "centreline_pass_3pct": u_max["pass_3pct_all_rungs"],
        },
        "Q5_raw_local_secant_without_stencil_correction": {
            "reading": g1_raw["label"],
            "values_pct": g1_raw["values"],
            "observed_order": g1_raw["power_law_fit"]["p"] if g1_raw["power_law_fit"] else None,
            "note": ("a raw one-spacing secant carries an O(dx) stencil bias; "
                     "if its error plateaus it is measuring the stencil, not the wall"),
        },
    }
    return {"questions": questions}


# --------------------------------------------------------------------------- #
# reproducibility against the committed ladder
# --------------------------------------------------------------------------- #

def reproducibility(rungs: list, committed_path: Path = COMMITTED) -> dict:
    if not committed_path.exists():
        return {"available": False, "reason": f"missing {committed_path}"}
    committed = json.loads(committed_path.read_text())
    by_r = {r["resolution_H_over_dx"]: r for r in committed["results"]}
    checks = []
    for rung in rungs:
        r = rung["resolution_H_over_dx"]
        if r not in by_r:
            continue
        old = by_r[r]
        new = flatten(rung)
        row = {"resolution_H_over_dx": r}
        for key, old_key in (("u_max_error_pct", "u_max_error_pct"),
                             ("rms_interior_rel_pct", "profile_rms_interior_pct"),
                             ("rms_all_rel_pct", "profile_rms_all_rows_pct")):
            a, b = old[old_key], new[key]
            row[key] = {
                "committed": a,
                "resolved": b,
                "abs_diff_pct_points": abs(b - a),
                "agree_within_0p5pp": abs(b - a) <= 0.5,
            }
        checks.append(row)
    return {
        "available": True,
        "committed_artifact": str(committed_path.relative_to(REPO)),
        "same_session_control_agrees": (all(all(v["agree_within_0p5pp"]
                                               for k, v in row.items()
                                               if isinstance(v, dict))
                                           for row in checks) if checks else None),
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_figures(rungs: list, analysis: dict, figdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    g = rungs[0]["body_force"] if "body_force" in rungs[0] else 9.81
    nu = rungs[0]["viscosity"] if "viscosity" in rungs[0] else 0.01
    H = None
    out = []

    # (a) profiles vs analytic
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for rung in rungs:
        rows = rung["rows"]
        ys = [r["y_over_H"] for r in rows]
        us = [r["u_mean"] for r in rows]
        ax[0].plot(us, ys, "o-", ms=3, lw=1,
                   label=f"H/dx={rung['resolution_H_over_dx']}")
    # analytic parabola in normalised coordinates: u/u_ref = 4*(1-y/H)*(y/H)
    yy = [i / 200.0 for i in range(201)]
    ax[0].plot([4.0 * (1 - y) * y for y in yy], yy, "k--", lw=1.5,
               label="analytic $g(H-y)y/2\\nu$")
    ax[0].set_xlabel("$u_x / u_{max}^{ref}$")
    ax[0].set_ylabel("$y/H$")
    ax[0].set_title("steady profiles, refinement ladder")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # (b) log-log error vs resolution with fitted orders
    keys = ["rms_interior_rel_pct", "G1_corrected_pct", "G1_raw_pct",
            "u_max_error_pct"]
    styles = {"rms_interior_rel_pct": ("o-", "profile rms (interior)"),
              "G1_corrected_pct": ("s-", "wall shear G1 (bias-corrected)"),
              "G1_raw_pct": ("^-", "wall shear G1 (raw)"),
              "u_max_error_pct": ("d-", "centreline $u_{max}$")}
    rs = analysis["resolutions"]
    for key in keys:
        entry = analysis["readings"][key]
        es = [abs(e) if e is not None else float("nan") for e in entry["values"]]
        mk, lab = styles[key]
        p = entry["power_law_fit"]["p"] if entry["power_law_fit"] else float("nan")
        ax[1].loglog(rs, es, mk, ms=5, label=f"{lab} (p={p:.2f})")
    ax[1].axhline(5.0, color="r", ls="--", lw=1, label="5 % band")
    ax[1].axhline(3.0, color="orange", ls=":", lw=1, label="3 % band")
    ax[1].set_xlabel("$H/\\Delta x$")
    ax[1].set_ylabel("|error|  [%]")
    ax[1].set_title("observed order above H/dx = 16")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    name = "poiseuille_refinement_ladder.png"
    fig.savefig(figdir / name, dpi=150)
    plt.close(fig)
    out.append({"name": name,
                "caption": ("(a) steady velocity profiles at H/dx = "
                            + "/".join(str(r) for r in rs)
                            + " against the analytic parabola; (b) log-log error "
                              "ladder with least-squares observed orders, against "
                              "the 5 % anchor band and the 3 % owner band.")})

    # (c) wall-layer detail: the two wall rows' relative error
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for rung in rungs:
        rows = sorted(rung["rows"], key=lambda r: r["y"])
        u_ref = rung["profile"]["u_max_analytic"]
        ys = [r["y_over_H"] for r in rows[:4]]
        es = [100.0 * (r["u_mean"] - u_ref * 4.0 * (1 - r["y_over_H"]) * r["y_over_H"])
              / (u_ref * 4.0 * (1 - r["y_over_H"]) * r["y_over_H"]) for r in rows[:4]]
        ax.plot(range(len(ys)), es, "o-", ms=4,
                label=f"H/dx={rung['resolution_H_over_dx']}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"row {i+1}" for i in range(4)])
    ax.set_ylabel("relative error [%]")
    ax.set_title("wall-layer rows: does the deficit shrink with resolution?")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    name2 = "poiseuille_refinement_wall_rows.png"
    fig.savefig(figdir / name2, dpi=150)
    plt.close(fig)
    out.append({"name": name2,
                "caption": ("relative error of the first four off-wall rows at each "
                            "resolution; a shrinking wall-layer deficit is the "
                            "signature of a boundary condition applied on the plane.")})
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", type=int, default=None,
                    help="run a single resolution and write out/refinement/rN.json")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--viscosity", type=float, default=0.01)
    ap.add_argument("--body-force", type=float, default=9.81)
    ap.add_argument("--height", type=float, default=0.1)
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--t-average", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    per_rung_dir = args.out / "refinement"
    per_rung_dir.mkdir(parents=True, exist_ok=True)

    if args.only is not None:
        rung = run_rung(args.only, viscosity=args.viscosity,
                        body_force=args.body_force, height=args.height,
                        columns=args.columns, t_end=args.t_end,
                        t_average=args.t_average, device=args.device,
                        threads=args.threads)
        rung["body_force"] = args.body_force
        rung["viscosity"] = args.viscosity
        rung["height"] = args.height
        rung["columns"] = args.columns
        path = per_rung_dir / f"r{args.only}.json"
        path.write_text(json.dumps(rung, indent=2))
        f = flatten(rung)
        print(f"[refinement] H/dx={args.only}  u_max {f['u_max_error_pct']:+.2f}%  "
              f"rms_int {f['rms_interior_rel_pct']:.3f}%  "
              f"G1corr {f['G1_corrected_pct']:+.2f}%  "
              f"({rung['wall_clock_s']:.0f}s) -> {path}", flush=True)
        return 0

    if args.assemble:
        files = sorted(per_rung_dir.glob("r*.json"),
                       key=lambda p: int(p.stem[1:]))
        rungs = [json.loads(p.read_text()) for p in files]
        if len(rungs) < 2:
            raise SystemExit("need >= 2 per-rung files to fit an order")
        an = analyse(rungs)
        payload = {
            "case": "plane_poiseuille_channel_mirror_wall",
            "anchor": "CASE-POISEUILLE-SPH",
            "study": "independent refinement past H/dx = 16",
            "question": ("does the published error ladder keep decaying above its top "
                         "rung, and at what resolution is the 5 % / 3 % line met?"),
            "config": {k: rungs[0][k] for k in ("viscosity", "body_force", "height",
                                                "columns", "t_end", "t_average")
                       if k in rungs[0]},
            "analysis": an,
            "verdict": verdict(an),
            "reproducibility_vs_committed_ladder": reproducibility(rungs),
            "rungs": [{k: v for k, v in r.items() if k not in ("rows", "trace")}
                      for r in rungs],
            "torch": torch.__version__,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if not args.no_figures:
            try:
                payload["figures"] = make_figures(rungs, an, args.out / "figures")
            except Exception as exc:  # pragma: no cover - matplotlib optional
                payload["figures"] = []
                payload["figures_error"] = str(exc)
        path = args.out / "refinement.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"[refinement] assembled {path}")
        for key in ("rms_interior_rel_pct", "G1_corrected_pct", "G1_raw_pct"):
            e = an["readings"][key]
            p = e["power_law_fit"]["p"] if e["power_law_fit"] else None
            print(f"  {key:24s} p={p:.2f}  values={[round(v,3) for v in e['values']]}")
        return 0

    ap.error("pass --only N or --assemble")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())