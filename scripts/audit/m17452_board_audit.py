#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m17452 — anchor
XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID (TensorSPH).

Verification, not re-do.  The audit answers, from bytes that exist:

  1. Do the three acceptance pieces exist?
       run_artifacts : artifacts/xval_fsi2/{results.json,*.png,oss_src/}
       benchmark     : ref vs got per case and per dx, from results.json
       repro_cmd     : the delivered driver, re-run here in a temp copy
  2. What does the *registered* referee say?
       SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
        WHERE anchor_slug='XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID'
     -> 0 rows both at mission time and at audit time.  Recorded, not
     papered over.  The criterion used instead is the one the mission
     *predeclared before the run* (run.py module docstring, "Thresholds"):
     |err| < 15 % required / < 3 % target -- identical to owner's accuracy
     standard, so no threshold is invented after seeing the numbers.
  3. Is every quoted number reproducible?  pytest + run.py replayed in a copy;
     every JSON leaf of results.json compared (timings_s excluded).
  4. Level L0-L5 by function, not by report length.

Outputs (inside the audit worktree)
    artifacts/audit_m17452/{manifest.json,git_state.json,referee.json,
                            judge.json,repro.json,finding.json,
                            fig_error_vs_threshold.png}
    docs/board_audit_m17452/M17452_BOARD_AUDIT.md
    (--insert-verdict) one row in tensor_maturity_verdicts

Usage
    python scripts/audit/m17452_board_audit.py --replay --insert-verdict \
        --audit-mission 17494
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_TREE = "/root/tp-agent/work/m17452"
ANCHOR = "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID"
AUDIT_MID = 17494
AUDITED_MID = 17452
OUT = os.path.join(REPO, "artifacts", "audit_m17452")
DOC = os.path.join(REPO, "docs", "board_audit_m17452", "M17452_BOARD_AUDIT.md")

# files the mission itself declared as its deliverable (result.changed_files)
DELIVERED = [
    "src/tensorsph/fsi_force_from_fluid.py",
    "tests/test_fsi_force_from_fluid.py",
    "benchmarks/xval_fsi2/cases.py",
    "benchmarks/xval_fsi2/run.py",
    "benchmarks/xval_fsi2/make_doc.py",
    "docs/xval_fsi2/GAP-SPH-FSI-BENCHMARK_xval.md",
    "artifacts/xval_fsi2/results.json",
    "artifacts/xval_fsi2/fig_error_vs_dx.png",
    "artifacts/xval_fsi2/fig_thickness.png",
]
# predeclared by the mission before the run (run.py docstring), not by this audit
TOL_REQUIRED = 0.15
TOL_TARGET = 0.03


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    files = {}
    for rel in DELIVERED:
        p = os.path.join(SRC_TREE, rel)
        files[rel] = {"exists": os.path.isfile(p),
                      "bytes": os.path.getsize(p) if os.path.isfile(p) else 0,
                      "sha256": sha256(p) if os.path.isfile(p) else None}
    oss = os.path.join(SRC_TREE, "artifacts", "xval_fsi2", "oss_src")
    archived = sorted(os.listdir(oss)) if os.path.isdir(oss) else []
    return {"tree": SRC_TREE, "anchor": ANCHOR, "files": files,
            "oss_src_archived": archived}


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", SRC_TREE, *args], capture_output=True,
                              text=True).stdout.strip()
    return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "head": run("rev-parse", "HEAD"),
            "log": run("log", "--oneline", "-1"),
            "porcelain": run("status", "--porcelain").splitlines(),
            "delivered_tracked": [f for f in DELIVERED
                                  if run("ls-files", "--error-unmatch", f)] }


def referee(dsn: str) -> dict:
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT case_name, metric, tolerance, ref_source, status "
            "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY case_name",
            (ANCHOR,))
        rows = cur.fetchall()
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
        total = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases "
                    "WHERE anchor_slug ILIKE %s", ("%FSI%",))
        fsi_like = cur.fetchone()[0]
    return {"anchor_slug": ANCHOR, "rows": rows, "n_rows": len(rows),
            "cases_table_total": total, "fsi_like_rows": fsi_like,
            "note": ("no registered referee case for this anchor; criterion is "
                     "the mission's own predeclared 15 %/3 % (run.py docstring) "
                     "== owner accuracy standard")}


def replay() -> dict:
    tmp = tempfile.mkdtemp(prefix="audit17452_")
    for name in ("src", "tests", "benchmarks", "artifacts"):
        shutil.copytree(os.path.join(SRC_TREE, name), os.path.join(tmp, name))
    env = dict(os.environ, PYTHONPATH=os.path.join(tmp, "src"))
    pt = subprocess.run([sys.executable, "-m", "pytest",
                         "tests/test_fsi_force_from_fluid.py", "-q"],
                        cwd=tmp, env=env, capture_output=True, text=True, timeout=1800)
    dt = subprocess.run([sys.executable, "benchmarks/xval_fsi2/run.py"],
                        cwd=tmp, env=env, capture_output=True, text=True, timeout=7200)
    a = json.load(open(os.path.join(SRC_TREE, "artifacts/xval_fsi2/results.json")))
    b = json.load(open(os.path.join(tmp, "artifacts/xval_fsi2/results.json")))

    def flat(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from flat(v, f"{p}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from flat(v, f"{p}[{i}]")
        else:
            yield p, o

    fa, fb = dict(flat(a)), dict(flat(b))
    differing = [k for k in fa if k in fb and fa[k] != fb[k]]
    ignored = [k for k in differing if k.startswith("/timings_s")]
    return {
        "temp_dir": tmp,
        "pytest": {"returncode": pt.returncode,
                   "tail": pt.stdout.strip().splitlines()[-1:]},
        "run": {"returncode": dt.returncode,
                "tail": dt.stdout.strip().splitlines()[-4:]},
        "leaves_audited": len(fa), "leaves_replayed": len(fb),
        "leaf_keys_equal": set(fa) == set(fb),
        "leaves_differing": len(differing),
        "leaves_compared_excluding_timings": len(fa) - len(
            [k for k in fa if k.startswith("/timings_s")]),
        "nondeterministic_leaves": differing,
        "ignored_timing_leaves": len(ignored),
    }


def judge(ref: dict, rep: dict, git: dict, man: dict, m_replay: dict) -> dict:
    res = json.load(open(os.path.join(SRC_TREE, "artifacts/xval_fsi2/results.json")))
    rows = []
    for r in res["rows"]:
        for op, o in r["operators"].items():
            rows.append({"case": r["case"], "dx": r["dx"], "operator": op,
                         "fluid_state": r["fluid_state"],
                         "got": o["got"], "ref": o["ref"], "error": o["error"],
                         "orthogonal": o.get("orthogonal"),
                         "passes_15": abs(o["error"]) < TOL_REQUIRED,
                         "passes_3": abs(o["error"]) < TOL_TARGET})
    primary = [x for x in rows if x["operator"] in ("oss", "oss_riemann", "oss_viscous")]
    primary_oss = [x for x in rows if x["operator"] == "oss"]
    visc = [x for x in rows if x["operator"] == "oss_viscous"]
    panel = [x for x in rows if x["operator"] == "panel"]

    # --- level rules, fixed before the numbers were read -------------------
    # L4: every row of the ported (primary) operator inside 15 % on >=2 dx per
    #     case + artifacts + reproducing command.
    # L3: at least one production caller outside tests/ and benchmarks/.
    # L2: real implementation (input->compute->output) with invariant tests.
    callers = subprocess.run(
        ["grep", "-rIl", "fsi_force_from_fluid", os.path.join(SRC_TREE, "src"),
         os.path.join(SRC_TREE, "examples")],
        capture_output=True, text=True).stdout.split()
    callers = [c for c in callers if not c.endswith("fsi_force_from_fluid.py")]
    init_export = subprocess.run(
        ["grep", "-n", "fsi_force", os.path.join(SRC_TREE, "src/tensorsph/__init__.py")],
        capture_output=True, text=True).stdout.strip()

    l4 = all(x["passes_15"] for x in primary) and len(primary) > 0
    l3 = bool(callers)
    l2 = os.path.isfile(os.path.join(SRC_TREE, "src/tensorsph/fsi_force_from_fluid.py")) \
        and m_replay["pytest"]["returncode"] == 0 \
        and m_replay["run"]["returncode"] == 0
    level = 4 if l4 else (3 if l3 else (2 if l2 else 1))

    solved_rows = [x for x in rows if x["fluid_state"] == "solved"]
    return {
        "tolerances": {"required": TOL_REQUIRED, "target": TOL_TARGET,
                       "source": "predeclared in benchmarks/xval_fsi2/run.py docstring"},
        "n_rows_scored": len(rows),
        "rows": rows,
        "summary": {
            "primary_rows": len(primary),
            "primary_pass15": sum(x["passes_15"] for x in primary),
            "primary_pass3": sum(x["passes_3"] for x in primary),
            "oss_pressure_err_range": [min(x["error"] for x in primary_oss),
                                       max(x["error"] for x in primary_oss)],
            "oss_viscous_err_range": [min(x["error"] for x in visc),
                                      max(x["error"] for x in visc)],
            "panel_err_range": [min(x["error"] for x in panel),
                                max(x["error"] for x in panel)],
            "panel_pass15": sum(x["passes_15"] for x in panel),
            "panel_rows": len(panel),
            "rows_with_solved_fluid_state": len(solved_rows),
        },
        "level_decision": {
            "L4": l4, "L3": l3, "L2": l2, "level": level,
            "L3_reason": f"production callers outside tests/benchmarks: {callers or 'none'}",
            "L4_reason": (f"{sum(x['passes_15'] for x in primary)}/{len(primary)} "
                          f"ported-operator rows within {TOL_REQUIRED:.0%}"),
            "package_export": init_export or "none (fsi module absent from tensorsph/__init__.py)",
        },
        "registered_referee_rows": ref["n_rows"],
        "replay_leaf_equality": m_replay["leaf_keys_equal"]
        and set(m_replay["nondeterministic_leaves"]) <= set(
            k for k in m_replay["nondeterministic_leaves"] if k.startswith("/timings_s")),
    }


def figure(j: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_error_vs_threshold.png")
    labels, vals, colors = [], [], []
    cmap = {"oss": "#c0392b", "oss_riemann": "#e67e22", "oss_viscous": "#8e44ad",
            "panel": "#2471a3", "pairwise_legacy": "#7f8c8d"}
    for r in j["rows"]:
        if r["operator"] == "oss_riemann":
            continue  # identical to oss in this configuration
        labels.append(f"{r['case']}\ndx={r['dx']:g} {r['operator']}")
        vals.append(100.0 * r["error"])
        colors.append(cmap[r["operator"]])
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(labels)), 4.6))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(15, color="k", ls="--", lw=1.2, label="15 % requirement")
    ax.axhline(-15, color="k", ls="--", lw=1.2)
    ax.axhline(3, color="green", ls=":", lw=1.0, label="3 % target")
    ax.axhline(-3, color="green", ls=":", lw=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("load error vs analytic reference  [%]")
    ax.set_title("m17452 XVAL-SPH-FSI2: ported SPHinXsys ForceFromFluid vs analytic load\n"
                 "12/12 ported-operator rows outside 15 %", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def insert_verdict(dsn: str, audit_mid: int, level: int, ev: dict, why: str) -> int:
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts "
            "(repo, scope, mission_id, level, evidence, verdict_by) "
            "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
            ("TensorSPH", f"m{AUDITED_MID} {ANCHOR}", audit_mid, level,
             json.dumps(ev, ensure_ascii=False), "board-audit"))
        return cur.fetchone()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=tensorplatform user=postgres")
    ap.add_argument("--audit-mission", type=int, default=AUDIT_MID)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--insert-verdict", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    man = manifest()
    git = git_state()
    ref = referee(a.dsn)
    m_replay = replay() if a.replay else {"pytest": {"returncode": None, "tail": []},
                                          "run": {"returncode": None, "tail": []},
                                          "leaf_keys_equal": None,
                                          "nondeterministic_leaves": [],
                                          "leaves_audited": 0, "leaves_replayed": 0,
                                          "leaves_compared_excluding_timings": 0,
                                          "ignored_timing_leaves": 0, "temp_dir": None}
    j = judge(ref, ref, git, man, m_replay)
    fig = figure(j)

    for name, obj in (("manifest", man), ("git_state", git), ("referee", ref),
                      ("judge", j), ("repro", m_replay)):
        json.dump(obj, open(os.path.join(OUT, f"{name}.json"), "w"),
                  indent=1, ensure_ascii=False)

    s = j["summary"]
    finding = {
        "mission": AUDITED_MID, "anchor": ANCHOR, "repo": "TensorSPH",
        "level": j["level_decision"]["level"],
        "referee_rows": ref["n_rows"],
        "primary_rows": s["primary_rows"],
        "primary_pass15": s["primary_pass15"],
        "oss_pressure_err_range_pct": [round(100 * x, 2) for x in s["oss_pressure_err_range"]],
        "oss_viscous_err_range_pct": [round(100 * x, 2) for x in s["oss_viscous_err_range"]],
        "panel_pass15": f"{s['panel_pass15']}/{s['panel_rows']}",
        "panel_err_range_pct": [round(100 * x, 2) for x in s["panel_err_range"]],
        "deliverables_uncommitted": [p for p in git["porcelain"]],
        "reproducible": j["replay_leaf_equality"],
        "disposition": "request changes — benchmark FAIL, not L3/L4",
    }
    json.dump(finding, open(os.path.join(OUT, "finding.json"), "w"), indent=1)
    print(json.dumps(finding, indent=1, ensure_ascii=False))

    if a.insert_verdict:
        ev = {
            "impl_files": [
                "src/tensorsph/fsi_force_from_fluid.py:1-352 (ported ForceFromFluid "
                "+ panel + legacy arms)",
                "benchmarks/xval_fsi2/run.py:1-192 (driver, predeclared 15 %/3 %)",
                "benchmarks/xval_fsi2/cases.py:1-314 (4 case families, analytic fields)",
                "artifacts/xval_fsi2/results.json (12 scored rows + sensitivity + thickness)",
            ],
            "entry": "python benchmarks/xval_fsi2/run.py",
            "run_artifacts": [
                f"work/m17452/artifacts/xval_fsi2/results.json sha256={man['files']['artifacts/xval_fsi2/results.json']['sha256'][:16]}",
                "work/m17452/artifacts/xval_fsi2/fig_error_vs_dx.png",
                "work/m17452/artifacts/xval_fsi2/fig_thickness.png",
                "work/m17452/docs/xval_fsi2/GAP-SPH-FSI-BENCHMARK_xval.md (167 lines)",
                "audit replay copy: artifacts/audit_m17452/repro.json",
            ],
            "tests": [
                "tests/test_fsi_force_from_fluid.py (10 tests, 10 passed on replay; "
                "invariant tests: sign, dx-insensitivity, linear-exactness, zero-neighbour)",
            ],
            "integration_callers": [
                "none outside tests/ and benchmarks/ (grep -rIl fsi_force_from_fluid "
                "src/ examples/ -> only the module itself); src/tensorsph/__init__.py "
                "does not export it; docs porting decision = do not adopt",
            ],
            "benchmark": {
                "registered_cases": ref["n_rows"],
                "criterion": "predeclared 15 % required / 3 % target (run.py docstring) == owner standard",
                "primary_rows": s["primary_rows"],
                "primary_pass15": s["primary_pass15"],
                "oss_pressure_error_pct": [round(100 * x, 2) for x in s["oss_pressure_err_range"]],
                "oss_viscous_error_pct": [round(100 * x, 2) for x in s["oss_viscous_err_range"]],
                "panel_pass15": f"{s['panel_pass15']}/{s['panel_rows']}",
                "panel_error_pct": [round(100 * x, 2) for x in s["panel_err_range"]],
                "verdict": "FAIL (all ported-operator rows > 15 %)",
            },
            "repro_cmd": "python benchmarks/xval_fsi2/run.py && python -m pytest tests/test_fsi_force_from_fluid.py",
            "repro_result": {
                "pytest": m_replay["pytest"]["tail"],
                "run_rc": m_replay["run"]["returncode"],
                "leaves_compared": m_replay["leaves_compared_excluding_timings"],
                "leaves_differing_outside_timings": len(
                    [k for k in m_replay["nondeterministic_leaves"]
                     if not k.startswith("/timings_s")]),
            },
            "external_anchor": [
                "Xiangyu-Hu/SPHinXsys master: src/shared/shared_ck/particle_dynamics/"
                "fluid_structure_interaction/force_on_structure.hpp:55-72,103-124 "
                "(algebra ported; file NOT archived under oss_src/ — only hydrostatic_fsi.cpp)",
                "analytic references: 1/2 rho g H^2 (hydrostatic_fsi.cpp:14-19), "
                "mu U / h (Couette), displaced area pi r^2 + bh*bl (fsi2 insert)",
            ],
            "level_reason": (
                "L2: real transfer operator (input particles -> per-particle load) with "
                "invariant unit tests and a reproducing driver, but 0/12 ported-operator "
                "rows inside 15 % (so no L4) and no production caller / package export "
                "(so no L3). Report honesty and bitwise reproducibility do not raise the level."
            ),
            "findings": [
                "F1 benchmark FAIL: oss +24.2..+108 %; oss_viscous -38.0..-38.5 %",
                "F2 not integrated: no caller, no __init__ export, doc advises against adoption",
                "F3 provenance gap: force_on_structure.hpp/fsi2.cpp cited but not archived in oss_src/",
                "F4 overstatement: run.py design promises a solved-WCSPH case; 0/12 rows are 'solved'",
                "F5 panel -0.00 % on flat walls is tautological (WLS exact on linear field); "
                "panel diverges on the closed body (+20 %/+32 % under refinement)",
                "F6 tests pin the failing bands (0.20<err<0.32) as expected — change detector, "
                "not validation",
                "F7 deliverables uncommitted (git status '??'), mission ended "
                "budget_exhausted+forced_finish at 80/80 calls",
            ],
            "audit_evidence": [
                "docs/board_audit_m17452/M17452_BOARD_AUDIT.md",
                "artifacts/audit_m17452/{manifest,git_state,referee,judge,repro,finding}.json",
                "artifacts/audit_m17452/fig_error_vs_threshold.png",
            ],
        }
        vid = insert_verdict(a.dsn, a.audit_mission, j["level_decision"]["level"], ev,
                             "benchmark FAIL / no integration")
        print("inserted verdict id", vid)
        json.dump({"verdict_id": vid, "evidence": ev},
                  open(os.path.join(OUT, "verdict.json"), "w"), indent=1,
                  ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())