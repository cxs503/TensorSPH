#!/usr/bin/env python3
"""Independent audit of m16239 (CASE-SPH-STOKES-2ND-PROBLEM, TensorSPH).

Read-only verifier: it never rewrites the audited worktree.  It re-derives every
headline number of the m16239 deliverable from the raw per-row amplitudes that
the delivered SPH run stored, instead of trusting the metric fields the author
computed.

Checks
  C1  the shipped ladder artifact is internally consistent (delta, plate BC,
      analytic reference, per-row rel_err recomputed from |u|)
  C2  the criterion metric of tensor_case_pool id=126 ("velocity amplitude decay
      exp(-y/delta) over the Stokes layer, >=5 layering rows, <5%") is met, and
      the value is recomputed from scratch rather than read from the artifact
  C3  provenance integrity: the machine provenance block must describe the run
      that actually executed (c0/Ma, periods, window)
  C4  ladder completeness: how many resolutions the artifact contains vs the
      LADDER constant of run.py vs the probes that are still on disk; whether
      the convergence verdict is vacuous
  C5  deliverables that the mission claims but that do not exist on disk
      (docs page, report body, generator syntax)

Usage:  python3 scripts/audit_m16239_stokes_second.py [--src /root/tp-agent/work/m16239]
Writes: runs/audit_m16239/audit_facts.json (+ prints a summary table)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT_DIR = HERE / "runs" / "audit_m16239"


def rel_err_profile(rows, delta, u0):
    """Recompute the amplitude relative error row by row from |u| only.

    The artifact stores both |u| (``amplitude``) and what the author's own
    analytic evaluator produced (``amplitude_half_space``); the audit uses the
    closed form straight from Landau & Lifshitz sec.24, A(y)=U0 exp(-y/delta),
    so a bug in the evaluator cannot hide.
    """
    out = []
    for r in rows:
        y_over_delta = r["y"] / delta
        ref = u0 * math.exp(-y_over_delta)
        out.append({
            "row": r["row"],
            "y_over_delta": y_over_delta,
            "amplitude": r["amplitude"],
            "ref": ref,
            "rel_err": (r["amplitude"] - ref) / ref,
            "author_rel_err": r["rel_err_half_space"],
        })
    return out


def l2(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def audit_run(run, tol_pct, want_rows=5):
    delta = run["delta"]
    u0 = run["u0"]
    prof = rel_err_profile(run["rows"], delta, u0)
    layer = [p for p in prof if p["y_over_delta"] <= 1.0 + 1e-12]
    errs = [p["rel_err"] for p in layer]
    audit_l2 = 100.0 * l2(errs)
    audit_worst = 100.0 * max(abs(e) for e in errs)
    # log-linear decay rate of the amplitude, and the delta it implies
    n = len(layer)
    xs = [p["y_over_delta"] for p in layer]
    ys = [math.log(p["amplitude"]) for p in layer]
    mx = sum(xs) / n
    my = sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return {
        "resolution_delta_over_dx": run["resolution_delta_over_dx"],
        "delta": delta,
        "delta_from_sqrt_2nu_over_omega": math.sqrt(2.0 * run["viscosity"] / run["omega"]),
        "u0": u0,
        "sound_speed_executed": run["sound_speed"],
        "mach_executed": run["mach_number"],
        "t_end_periods": run["t_end"],
        "measure_from_period": run["measure_from"],
        "n_fluid": run["n_fluid"],
        "steps": run["steps"],
        "dt": run["dt"],
        "rows_total": len(run["rows"]),
        "rows_in_stokes_layer": len(layer),
        "crit_row_count_ok": len(layer) >= want_rows,
        "audit_l2_pct": audit_l2,
        "author_l2_pct": run["metrics"]["amplitude_l2_half_space_pct"],
        "l2_agree": abs(audit_l2 - run["metrics"]["amplitude_l2_half_space_pct"]) < 1e-9,
        "audit_worst_row_pct": audit_worst,
        "author_worst_row_pct": run["metrics"]["amplitude_worst_row_half_space_pct"],
        "crit_tolerance_pct": tol_pct,
        "crit_pass": audit_l2 < tol_pct and len(layer) >= want_rows,
        "fit_decay_slope_per_delta": slope,
        "fit_delta_over_analytic_delta": 1.0 / (-slope),
        "first_row_amplitude_over_u0": run["rows"][0]["amplitude_over_U0"],
        "first_row_analytic_over_u0": math.exp(-run["rows"][0]["y"] / delta),
        "phase_l2_err_rad": run["metrics"]["phase_l2_err_rad"],
        "rectified_mean_max_over_u0": run["metrics"]["rectified_mean_max_over_U0"],
        "gates": run["gates"],
        "gates_all_pass": all(run["gates"].values()),
        "wall_clock_s": run["wall_clock_s"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/tp-agent/work/m16239")
    ap.add_argument("--tol-pct", type=float, default=5.0,
                    help="criterion of tensor_case_pool case SPH-STOKES-2ND-PROBLEM (<5%)")
    args = ap.parse_args(argv)
    src = Path(args.src)
    bench = src / "benchmarks" / "stokes_second"
    facts = {"audited_worktree": str(src), "criterion_tolerance_pct": args.tol_pct,
             "findings": []}

    ladder_path = bench / "out" / "results_ladder.json"
    ladder = json.loads(ladder_path.read_text())
    facts["artifact"] = {"path": str(ladder_path),
                         "sha256": __import__("hashlib").sha256(ladder_path.read_bytes()).hexdigest(),
                         "mode": ladder["mode"],
                         "n_runs": len(ladder["runs"]),
                         "resolutions": [r["resolution_delta_over_dx"] for r in ladder["runs"]],
                         "assembled_from": ladder.get("assembled_from"),
                         "missing_resolutions": ladder.get("missing_resolutions"),
                         "convergence_block": ladder.get("convergence"),
                         "figures": ladder.get("figures"),
                         "finished": ladder.get("finished")}

    # ---- C2/C1: re-derive every headline number of the shipped run ----------
    shipped = audit_run(ladder["runs"][0], args.tol_pct)
    facts["shipped_run"] = shipped

    # ---- C3: provenance integrity ------------------------------------------
    prov_c0 = ladder["provenance"]["case"]["c0"]
    prov = {
        "provenance_case_c0": prov_c0,
        "provenance_case_ma": prov_c0 and 0.01 / prov_c0,
        "executed_c0": shipped["sound_speed_executed"],
        "executed_ma": shipped["mach_executed"],
        "matches": abs(prov_c0 - shipped["sound_speed_executed"]) < 1e-12,
        "provenance_carries_periods": "periods" in json.dumps(ladder["provenance"]),
    }
    facts["provenance"] = prov
    if not prov["matches"]:
        facts["findings"].append({
            "id": "F4", "severity": "medium", "class": "provenance-integrity",
            "claim": "provenance.case.c0 describes the executed run",
            "fact": f"provenance says c0={prov_c0} m/s (Ma {prov['provenance_case_ma']:.1e}) "
                    f"but the run entry records sound_speed={shipped['sound_speed_executed']} "
                    f"(Ma {shipped['mach_executed']:.1e}); DESIGN.md sec.6 documents c0=4, so the "
                    f"machine provenance block is filled from module defaults (SOUND_SPEED=10), "
                    f"not from the args actually used"})

    # ---- C4: ladder completeness ------------------------------------------
    ladder_const = None
    run_py = (bench / "run.py").read_text()
    for line in run_py.splitlines():
        if line.startswith("LADDER ="):
            ladder_const = line.split("=", 1)[1].strip()
    probes = {}
    for d in ("/tmp/p8", "/tmp/p12", "/tmp/p18"):
        f = Path(d) / "results_probe.json"
        if f.exists():
            p = json.loads(f.read_text())["runs"][0]
            probes[d] = {"resolution_delta_over_dx": p["resolution_delta_over_dx"],
                         "c0": p["sound_speed"], "periods": p["t_end"],
                         "l2_pct": p["metrics"]["amplitude_l2_half_space_pct"],
                         "rows_in_layer": p["metrics"]["n_rows_stokes_layer"],
                         "wall_clock_s": p["wall_clock_s"],
                         "gates_all_pass": all(p["gates"].values())}
    conv = ladder.get("convergence") or {}
    vacuous = len(conv.get("resolutions", [])) < 2
    facts["ladder"] = {"LADDER_constant": ladder_const,
                       "shipped_resolutions": [r["resolution_delta_over_dx"] for r in ladder["runs"]],
                       "probes_on_disk": probes,
                       "shipped_convergence_is_vacuous": vacuous,
                       "shipped_convergence_claim": conv}
    if vacuous:
        facts["findings"].append({
            "id": "F1", "severity": "high", "class": "over-claim",
            "claim": "convergence.block G6_not_growing=true / all_gates=true evidences a "
                     "resolution ladder",
            "fact": f"the artifact holds {len(conv.get('resolutions', []))} resolution(s) "
                    f"{conv.get('resolutions')} while run.py LADDER={ladder_const}; "
                    f"G6_not_growing is computed as l2[-1] <= l2[0]+0.5, trivially true for n=1; "
                    f"missing_resolutions={ladder.get('missing_resolutions')} was recorded and "
                    f"ignored by the assembler"})

    # ---- a real ladder built from the probes that are still on disk --------
    have = {k: v for k, v in probes.items() if v["resolution_delta_over_dx"] in (8, 12, 18)}
    if len(have) >= 2:
        ks = sorted(have, key=lambda k: have[k]["resolution_delta_over_dx"])
        e = [have[k]["l2_pct"] for k in ks]
        r = [have[k]["resolution_delta_over_dx"] for k in ks]
        resid = e[0] / e[1] if e[1] > 0 else float("inf")
        p = math.log(resid) / math.log(r[1] / r[0])
        facts["reconstructed_ladder"] = {
            "sources": ks, "resolutions": r, "l2_pct": e,
            "error_ratio": resid, "observed_order": p,
            "monotone_decreasing": e[-1] < e[0],
            "note": "built from the probes left in /tmp (durable? no: /tmp is scratch; "
                    "the numbers are only reproducible by re-running --probe 8 / --probe 12)",
        }

    # ---- C5: deliverables that should exist -------------------------------
    make_doc = bench / "make_doc.py"
    try:
        subprocess.run(["python3", "-m", "py_compile", str(make_doc)],
                       check=True, capture_output=True, text=True)
        make_doc_ok, make_doc_err = True, None
    except subprocess.CalledProcessError as exc:
        make_doc_ok, make_doc_err = False, exc.stderr.strip().splitlines()[-1]
    facts["deliverables"] = {
        "design_md": (bench / "DESIGN.md").exists(),
        "run_py": (bench / "run.py").exists(),
        "tests": (src / "tests" / "test_stokes_second.py").exists(),
        "make_doc_compiles": make_doc_ok,
        "make_doc_error": make_doc_err,
        "docs_page_on_disk": any((src / "docs").glob("*stokes_second*")),
        "out_artifacts": sorted(p.name for p in (bench / "out").glob("*")),
    }
    if not make_doc_ok:
        facts["findings"].append({
            "id": "F5", "severity": "high", "class": "missing-deliverable",
            "claim": "docs page + platform report generated by make_doc.py",
            "fact": f"make_doc.py does not compile ({make_doc_err}); no docs/*stokes_second* "
                    f"page exists and tensor_reports has no row for mission 16239"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit_facts.json").write_text(json.dumps(facts, indent=1))

    print("shipped run  : res=%(resolution_delta_over_dx)s  L2=%(audit_l2_pct).4f%%  "
          "worst=%(audit_worst_row_pct).4f%%  rows_in_layer=%(rows_in_stokes_layer)s  "
          "c0=%(sound_speed_executed)s  periods=%(t_end_periods)s" % shipped)
    print("criterion    : <%.1f%% and >=5 rows -> %s (audit-recomputed, author L2=%.4f%%, agree=%s)"
          % (args.tol_pct, shipped["crit_pass"], shipped["author_l2_pct"], shipped["l2_agree"]))
    print("delta_fit/an : %.5f   first row A/U0: %.6f vs %.6f"
          % (shipped["fit_delta_over_analytic_delta"],
             shipped["first_row_amplitude_over_u0"], shipped["first_row_analytic_over_u0"]))
    print("provenance   : %s" % json.dumps(prov))
    print("ladder       : shipped %s (vacuous=%s), LADDER const %s, probes %s"
          % (facts["ladder"]["shipped_resolutions"], vacuous,
             ladder_const, {k: v["l2_pct"] for k, v in probes.items()}))
    if "reconstructed_ladder" in facts:
        rl = facts["reconstructed_ladder"]
        print("rebuilt      : resolutions %s  L2 %s  ratio %.3f  order %.2f"
              % (rl["resolutions"], ["%.4f" % v for v in rl["l2_pct"]],
                 rl["error_ratio"], rl["observed_order"]))
    print("make_doc     : %s %s" % (make_doc_ok, make_doc_err or ""))
    print("facts -> %s" % (OUT_DIR / "audit_facts.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())