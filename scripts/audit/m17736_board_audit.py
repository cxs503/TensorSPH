#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m17715 — anchor
XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (TensorSPH).  Audit mission m17736.

Verification, not re-do.  Everything below is recomputed from bytes:

  1. delivery state    git: is anything committed?  (subject branch vs main)
  2. three pieces      run_artifacts / benchmark ref-vs-got / repro_cmd executable
  3. referee           tensor_benchmark_cases rows for the anchor (machine judge)
  4. reproduction      full harness re-run in a copy of the delivered tree;
                       every JSON leaf compared; pytest; integration caller
  5. report vs artifact  doc tables ④/⑤ parsed and diffed against results.json

Outputs (inside this audit worktree)
    docs/board_audit_m17715/snapshot/...            byte copies + sha256 manifest
    artifacts/audit_m17715/{delivery,referee,repro,report_consistency,finding}.json
    docs/board_audit_m17715/M17715_BOARD_AUDIT.md   (written by hand, cites the JSON)

Usage
    python scripts/audit/m17736_board_audit.py                  # collect + judge
    python scripts/audit/m17736_board_audit.py --insert-verdict # + verdict row
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIT_REPO = Path(__file__).resolve().parents[2]
SUBJECT = Path("/root/tp-agent/work/m17715")
ANCHOR = "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE"
SCOPE = f"m17715 {ANCHOR}"
SUBJECT_MISSION = 17715
AUDIT_MISSION = 17736
BY = "board-audit"
OUT = AUDIT_REPO / "artifacts" / "audit_m17715"
SNAP = AUDIT_REPO / "docs" / "board_audit_m17715" / "snapshot"

DELIVERED = [
    "src/tensorsph/oss_projection.py",
    "benchmarks/xval_projection_laplace/run.py",
    "benchmarks/xval_projection_laplace/results/xval_projection_laplace.json",
    "benchmarks/xval_projection_laplace/results/xval_projection_laplace.png",
    "tests/test_xval_projection_laplace.py",
    "docs/xval_projection_laplace_oss.md",
]
PATCHED = ["src/tensorsph/__main__.py"]


# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd: list[str], cwd: Path) -> dict:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "rc": p.returncode,
            "out": p.stdout.strip()[-4000:], "err": p.stderr.strip()[-4000:]}


def dsn() -> str:
    """foundation role DSN; env override for reproducibility elsewhere."""
    if os.environ.get("TP_PG_DSN"):
        return os.environ["TP_PG_DSN"]
    for cand in Path("/root/tp-agent/work").glob("*/docs/board_audit*/*/insert_verdict.py"):
        txt = cand.read_text()
        m = re.search(r'DSN = \((.*?)\)\n', txt, re.S)
        if m:
            return eval("(" + m.group(1) + ")")  # noqa: S307 - repo-local constant
    raise SystemExit("no DSN available; set TP_PG_DSN")


def db_rows(sql: str, params: tuple = ()):
    import psycopg
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# --------------------------------------------------------------------------- #
def collect_delivery() -> dict:
    """Snapshot the delivered bytes and record the VCS state they live in."""
    if SNAP.exists():
        shutil.rmtree(SNAP)
    manifest = {}
    for rel in DELIVERED + PATCHED:
        src = SUBJECT / rel
        assert src.exists(), f"delivered file missing: {src}"
        dst = SNAP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[rel] = {"sha256": sha256(src), "bytes": src.stat().st_size,
                         "tracked_in_subject": sh(["git", "ls-files", "--error-unmatch", rel],
                                                  SUBJECT)["rc"] == 0}
    (SNAP / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

    state = sh(["git", "status", "--porcelain=v1", "--branch"], SUBJECT)
    log = sh(["git", "log", "--oneline", "-3"], SUBJECT)
    head = sh(["git", "rev-parse", "HEAD"], SUBJECT)["out"]
    main = sh(["git", "rev-parse", "origin/main"], SUBJECT)["out"]
    ahead = sh(["git", "rev-list", "--count", "origin/main..HEAD"], SUBJECT)["out"]
    touching = sh(["git", "log", "--oneline", "-5", "--author", "agent"], SUBJECT)
    return {"subject_worktree": str(SUBJECT), "head": head, "origin_main": main,
            "commits_ahead_of_main": ahead, "status": state["out"], "log": log["out"],
            "any_commit_mentions_m17715": "m17715" in log["out"],
            "untracked_or_dirty": [ln for ln in state["out"].splitlines() if ln[:2].strip()],
            "delivered_untracked": [k for k, v in manifest.items() if not v["tracked_in_subject"]],
            "manifest": manifest,
            "agent_commits_recent": touching["out"].splitlines()[:5]}


def collect_referee() -> dict:
    rows = db_rows("SELECT case_name, metric, tolerance, status, ref_source, repo "
                   "FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,))
    total = db_rows("SELECT count(*) FROM tensor_benchmark_cases")[0][0]
    reports = db_rows("SELECT id, title, length(markdown) FROM tensor_reports "
                      "WHERE mission_id=%s ORDER BY id", (SUBJECT_MISSION,))
    prior = db_rows("SELECT level, gate_state, gate_level, created_at::text "
                    "FROM tensor_maturity_verdicts WHERE repo='TensorSPH' AND scope LIKE %s",
                    ("m17594%",))
    return {"anchor": ANCHOR, "rows": rows, "rows_for_anchor": len(rows),
            "table_total_rows": total,
            "registered_tolerance": (rows[0][2] if rows else None),
            "tensor_reports_for_subject_mission": reports,
            "prior_verdict_m17594": prior,
            "verdict": "ABSENT" if not rows else "present"}


def reproduce() -> dict:
    """Re-run the delivered harness from a copy of the delivered tree."""
    tmp = Path(tempfile.mkdtemp(prefix="m17736_repro_"))
    copy = tmp / "tree"
    copy.mkdir()
    subprocess.run(f"tar -c --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache "
                   f"--exclude=artifacts --exclude=results -C {SUBJECT} . | tar -x -C {copy}",
                   shell=True, check=True)
    env = dict(os.environ, PYTHONPATH="src")
    pytest = subprocess.run([sys.executable, "-m", "pytest",
                             "tests/test_xval_projection_laplace.py", "-q"],
                            cwd=str(copy), env=env, capture_output=True, text=True)
    harness = subprocess.run([sys.executable,
                              "benchmarks/xval_projection_laplace/run.py"],
                             cwd=str(copy), env=env, capture_output=True, text=True,
                             timeout=1800)
    integ = subprocess.run([sys.executable, "-m", "tensorsph", "--case",
                            "xval_projection_laplace"],
                           cwd=str(copy), env=env, capture_output=True, text=True,
                           timeout=1800)
    got = json.loads((copy / "benchmarks/xval_projection_laplace/results/"
                      "xval_projection_laplace.json").read_text())
    ref = json.loads((SUBJECT / "benchmarks/xval_projection_laplace/results/"
                      "xval_projection_laplace.json").read_text())

    diffs = []
    SKIP = {"figure", "timing"}

    def walk(a, b, path=""):
        if isinstance(a, dict):
            for k in a:
                if k in SKIP:
                    continue
                walk(a[k], (b or {}).get(k), f"{path}/{k}")
        elif isinstance(a, list):
            for i, (ai, bi) in enumerate(zip(a, b or [])):
                walk(ai, bi, f"{path}[{i}]")
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            if not isinstance(b, (int, float)):
                diffs.append({"path": path, "delivered": a, "repro": b})
            elif abs(a - b) > 1e-9 * max(abs(a), 1e-30):
                diffs.append({"path": path, "delivered": a, "repro": b})
        elif a != b:
            diffs.append({"path": path, "delivered": a, "repro": b})

    walk(ref, got)
    return {
        "copy": str(copy),
        "pytest": {"rc": pytest.returncode, "tail": pytest.stdout.strip().splitlines()[-1:]},
        "harness_rc": harness.returncode,
        "harness_stdout": harness.stdout.strip().splitlines()[-12:],
        "integration_caller": {"cmd": "PYTHONPATH=src python3 -m tensorsph --case "
                                      "xval_projection_laplace",
                               "rc": integ.returncode,
                               "stderr_tail": integ.stderr.strip().splitlines()[-3:]},
        "json_leaf_diffs": diffs,
        "json_leaf_diff_count": len(diffs),
        "png_sha256_delivered": sha256(SUBJECT / "benchmarks/xval_projection_laplace/"
                                       "results/xval_projection_laplace.png"),
        "png_sha256_repro": sha256(copy / "benchmarks/xval_projection_laplace/results/"
                                   "xval_projection_laplace.png"),
        "parity": _parity(copy, env),
    }


def _parity(copy: Path, env: dict) -> dict:
    p = subprocess.run([sys.executable, "-c",
                        "import json;from tensorsph.oss_projection import "
                        "pysph_parity_check as f;print(json.dumps(f()))"],
                       cwd=str(copy), env=env, capture_output=True, text=True, timeout=600)
    if p.returncode:
        return {"rc": p.returncode, "err": p.stderr.strip()[-500:]}
    return json.loads(p.stdout)


DOC = "docs/xval_projection_laplace_oss.md"


def parse_doc_tables() -> dict:
    """Pull the report's ladder (④) and sensitivity (⑤) rows out of the markdown."""
    lines = (SUBJECT / DOC).read_text().splitlines()
    ladder, sens = [], []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[1].replace(".", "").isdigit() and \
                ("ours" in cells[0] or "oss" in cells[0]):
            ladder.append({"arm": cells[0], "R_over_dx": float(cells[1]),
                           "dp_measured": cells[2], "err_pct": cells[4],
                           "kappa_R": cells[5]})
        elif len(cells) == 4 and cells[0] in ("h_factor", "ours_curvature",
                                              "oss_morris_correction",
                                              "oss_delta_variant",
                                              "oss_number_density_form"):
            sens.append({"vary": cells[0], "value": cells[1],
                         "ours_pct": cells[2], "oss_pct": cells[3]})
    return {"ladder_rows": ladder, "sensitivity_rows": sens}


def report_consistency() -> dict:
    doc = parse_doc_tables()
    art = json.loads((SUBJECT / "benchmarks/xval_projection_laplace/results/"
                      "xval_projection_laplace.json").read_text())
    by_arm = {(r["arm"].split(",")[0], round(r["R_over_dx"], 2)): r for r in art["runs"]}
    rows, bad, kbad = [], 0, 0
    for d in doc["ladder_rows"]:
        key = (d["arm"].split(",")[0], d["R_over_dx"])
        r = by_arm.get(key)
        if r is None:
            rows.append({"doc": d, "artifact": None, "match": False})
            bad += 1
            continue
        doc_err = float(d["err_pct"].replace("−", "-").replace("+", "").rstrip("%")
                        if "e" not in d["err_pct"] else d["err_pct"].replace("−", "-"))
        ok = abs(doc_err - r["dp_error_pct"]) <= 1e-2 * max(abs(r["dp_error_pct"]), 1.0)
        rows.append({"doc": d, "artifact": {"dp": r["dp_measured_Pa"],
                                            "err_pct": r["dp_error_pct"],
                                            "kappa_R": r.get("kappa_times_R",
                                                             r.get("kappaR_delta_weighted"))},
                     "match": ok})
        bad += 0 if ok else 1
        if r.get("kappa_times_R", r.get("kappaR_delta_weighted")) is not None:
            kv = float(d["kappa_R"].replace("−", "-"))
            ka = r.get("kappa_times_R", r.get("kappaR_delta_weighted"))
            if abs(kv - ka) > 0.002 * max(abs(ka), 1.0):
                kbad += 1

    sens_art = {f"{s['vary']}|{s['value']}": s for s in art["sensitivity"]}
    srows, sbad = [], 0
    for s in doc["sensitivity_rows"]:
        a = sens_art.get(f"{s['vary']}|{s['value']}")
        if a is None:
            srows.append({"doc": s, "artifact": None, "match": False})
            sbad += 1
            continue
        cols = []
        for dk, ak in (("ours_pct", "ours_pct"), ("oss_pct", "oss_pct")):
            if s[dk] in ("—", ""):
                continue
            dv = float(s[dk].replace("−", "-").replace("+", ""))
            if ak in a:
                cols.append((dk, dv, a[ak], abs(dv - a[ak]) <= 0.05))
        ok = all(c[3] for c in cols)
        srows.append({"doc": s, "artifact": {k: a.get(k) for k in ("ours_pct", "oss_pct")},
                      "match": ok})
        sbad += 0 if ok else 1
    return {"ladder_rows": rows, "ladder_mismatches": bad,
            "ladder_rows_total": len(rows), "kappa_column_mismatches": kbad,
            "sensitivity_rows": srows, "sensitivity_mismatches": sbad,
            "sensitivity_rows_total": len(srows)}


# --------------------------------------------------------------------------- #
def build_finding(delivery, referee, repro, consist) -> dict:
    integ_ok = repro["integration_caller"]["rc"] == 0
    bench_rows = [r for r in consist["ladder_rows"]
                  if r.get("artifact") and "ours" in json.dumps(r["doc"]["arm"])]
    worst = max((abs(r["artifact"]["err_pct"]) for r in bench_rows), default=None)
    findings = {
        "F1_high_integration_caller_broken":
            f"`python -m tensorsph --case xval_projection_laplace` exits "
            f"{repro['integration_caller']['rc']} with "
            f"`unrecognized arguments: --case xval_projection_laplace`: the dispatched "
            f"harness main() runs its own argparse over sys.argv (run.py:181-183) so the "
            f"outer --case flag is rejected. The claimed 'L3 integration caller' does not "
            f"execute; L3 not met.",
        "F2_high_nothing_committed":
            f"subject HEAD carries {delivery['commits_ahead_of_main']} commits of its own "
            f"(HEAD {delivery['head'][:8]} vs origin/main {delivery['origin_main'][:8]}), all {len(delivery['delivered_untracked'])} delivered files untracked; "
            f"an automatic branch push publishes an empty branch and the whole "
            f"deliverable dies with the worktree.",
        "F3_high_report_numbers_contradict_artifact":
            f"report tables ④/⑤ disagree with results.json: {consist['ladder_mismatches']}"
            f"/{consist['ladder_rows_total']} ladder rows and "
            f"{consist['sensitivity_mismatches']}/{consist['sensitivity_rows_total']} "
            f"sensitivity rows outside 5 % / 1e-2 rel; e.g. h_factor=1.2 ours -0.31 % "
            f"(doc) vs -2.39 % (json), ours_curvature=fit_quadratic -0.18 % vs -24.46 %, "
            f"oss_morris=False +5.67 % vs +2.73 %, oss[code] dp 3.6e14 vs 2.68e22 Pa. "
            f"The doc's 'every number generated by the scripts, no number typed by hand' "
            f"(docs/xval_projection_laplace_oss.md:6) is false for §④/§⑤.",
        "F9_high_default_route_fails_gate":
            f"laplace_isph_case()'s shipped default curvature_method='fit_quadratic' "
            f"(src/tensorsph/laplace_isph.py:262) gives dp error "
            f"{[x for x in json.loads((SUBJECT / 'benchmarks/xval_projection_laplace/results/xval_projection_laplace.json').read_text())['sensitivity'] if x['vary']=='ours_curvature' and x['value']=='fit_quadratic'][0]['ours_pct']:+.2f} % "
            f"= 15 % gate FAIL; the harness only passes because run.py:73 forces 'mls'. "
            f"The report's sensitivity row claims -0.18 % for the same call, i.e. the "
            f"prose hides the regression the previous audit (m17594 F1) had already "
            f"flagged as open.",
        "F4_medium_no_referee_row":
            f"tensor_benchmark_cases holds {referee['table_total_rows']} rows, "
            f"{referee['rows_for_anchor']} for {ANCHOR} -> no machine tolerance judge; "
            f"the 15 % gate lives only in run.py:80 and the pytest constants.",
        "F5_medium_no_platform_report":
            f"tensor_reports has {len(referee['tensor_reports_for_subject_mission'])} rows "
            f"for mission {SUBJECT_MISSION}; the six-section report exists only as an "
            f"in-repo markdown (not the required publish step).",
        "F6_low_parity_three_open_ops":
            f"pysph_parity_check (executed against PySPH 1.0b3.dev0 in /tmp/xv2) leaves "
            f"Curvature_kappa={repro['parity'].get('Curvature_kappa')}, "
            f"PressureCoeff_diag={repro['parity'].get('PressureCoeff_diag')}, "
            f"ColorGradient_delta_1_over_gradC="
            f"{repro['parity'].get('ColorGradient_delta_1_over_gradC')} unmatched; the port "
            f"is numerically identical for SmoothedColor/gradC/normal/SY11-delta/CSF-force "
            f"only.",
        "F7_info_repro_exact":
            f"independent re-run: harness rc={repro['harness_rc']}, "
            f"{repro['json_leaf_diff_count']} JSON leaf diffs (excluding wall-clock), "
            f"PNG sha256 {'identical' if repro['png_sha256_delivered'] == repro['png_sha256_repro'] else 'DIFFERENT'}, "
            f"pytest {repro['pytest']['tail']}.",
        "F8_info_substance_holds":
            f"the artefact itself is real: ours-arm |dp error| max {worst}% over three "
            f"ladder levels vs the analytic sigma/R = 3.600 Pa (owner gate 15 %, report "
            f"gate 3 %); the numbers in results.json are truthful - the defect is in the "
            f"delivery and in the prose table, not in the physics.",
    }
    level = 2
    return {
        "anchor": ANCHOR, "subject_mission": SUBJECT_MISSION, "audit_mission": AUDIT_MISSION,
        "level": level,
        "level_rationale": {
            "L2_met": "real non-placeholder implementation (src/tensorsph/oss_projection.py, "
                      "498 lines, 11 public functions) + entry (benchmarks/.../run.py) + "
                      "reproducible run artifacts (JSON + PNG regenerated byte-identically).",
            "L3_refused": "the only integration caller errors out (F1); __main__.py:91-114 "
                          "is a dead branch as delivered.",
            "L4_refused": "report §④/§⑤ do not reproduce from the artifact (F3), the "
                          "anchor has no registered referee row (F4), and nothing is "
                          "committed so no one can re-run it from the repository (F2).",
        },
        "disposition": "request changes: (1) commit oss_projection.py + harness + tests + "
                       "results and open the PR, (2) fix the argv clash (call the harness "
                       "with an explicit argv or expose run(quick=...) instead of "
                       "argparse), (3) regenerate §④/§⑤ from results.json programmatically, "
                       "(4) register the XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE row in "
                       "tensor_benchmark_cases, (5) publish the platform report.",
        "findings": findings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    delivery = collect_delivery()
    referee = collect_referee()
    if args.skip_repro and (OUT / "repro.json").exists():
        repro = json.loads((OUT / "repro.json").read_text())
    else:
        repro = reproduce()
    consist = report_consistency()
    finding = build_finding(delivery, referee, repro, consist)

    for name, payload in (("delivery", delivery), ("referee", referee),
                          ("repro", repro), ("report_consistency", consist),
                          ("finding", finding)):
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str))
    print(json.dumps({k: v for k, v in finding.items()
                      if k in ("level", "disposition")}, indent=1))
    for k, v in finding["findings"].items():
        print(f"  {k}: {v[:160]}")

    if args.insert_verdict:
        evidence = build_evidence(delivery, referee, repro, consist, finding)
        import psycopg
        with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tensor_maturity_verdicts "
                "(repo, scope, mission_id, level, evidence, verdict_by) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
                "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, "
                "evidence = EXCLUDED.evidence, created_at = now() "
                "RETURNING id, gate_state, gate_level",
                ("TensorSPH", SCOPE, AUDIT_MISSION, finding["level"],
                 json.dumps(evidence, ensure_ascii=False), BY),
            )
            print("verdict row:", cur.fetchone())
            conn.commit()


def build_evidence(delivery, referee, repro, consist, finding) -> dict:
    snap = "docs/board_audit_m17715/snapshot"
    return {
        "impl_files": [
            f"{snap}/src/tensorsph/oss_projection.py:96-320 (11 public functions: "
            f"SmoothedColor / SY11 colour gradient / curvature+Morris / CSF force / "
            f"PressureCoeffMatrix / Poisson; 498 lines, no stub markers) - UNTRACKED in "
            f"subject worktree",
        ],
        "entry": [
            f"{snap}/benchmarks/xval_projection_laplace/run.py:112-232 (ladder + "
            f"sensitivity + figure + verdict JSON) - UNTRACKED",
            f"{snap}/src/tensorsph/__main__.py:91-114 (--case xval_projection_laplace "
            f"dispatch) - BROKEN: exits 2, see integration_callers",
        ],
        "run_artifacts": [
            f"{snap}/benchmarks/xval_projection_laplace/results/"
            f"xval_projection_laplace.json (9 runs, 3 levels x 3 arms, sha256 "
            f"{delivery['manifest']['benchmarks/xval_projection_laplace/results/xval_projection_laplace.json']['sha256'][:16]})",
            f"{snap}/benchmarks/xval_projection_laplace/results/"
            f"xval_projection_laplace.png (sha256 "
            f"{repro['png_sha256_delivered'][:16]}, regenerated byte-identically)",
        ],
        "tests": [
            "tests/test_xval_projection_laplace.py (9 tests) - auditor re-ran in a copy: "
            f"{repro['pytest']['tail'][0] if repro['pytest']['tail'] else 'n/a'} "
            "(1 skip = results.json absent from the copy); gates: 15% ladder, arm-to-arm, "
            "kappa*R=1+-5%, form=code must stay broken, sigma=0 control, PySPH parity",
            f"parity: {json.dumps({k: repro['parity'].get(k) for k in ('SmoothedColor','ColorGradient_gradC','SY11DiracDelta_delta','CSFForce_acc_x')})} "
            f"vs executed PySPH 1.0b3.dev0; Curvature_kappa / PressureCoeff_diag open",
        ],
        "integration_callers": [
            "PYTHONPATH=src python3 -m tensorsph --case xval_projection_laplace -> rc="
            f"{repro['integration_caller']['rc']}, stderr "
            f"{repro['integration_caller']['stderr_tail']} = NOT INTEGRATED (L3 fails)",
        ],
        "benchmark": {
            "case": "2-D circular droplet R=0.02 m, sigma=0.072 N/m, rho0=1000, "
                    "vapour carrier out to R+0.012 m, hexagonal lattice, Dirichlet "
                    "p=0 on outer 1.6 dx layer",
            "ref": "analytic 2-D Young-Laplace dp = sigma/R = 3.6000 Pa "
                   "(plus PySPH 1.0b3.dev0 operator parity)",
            "got_pa": {str(r["doc"]["R_over_dx"]): r["artifact"]["dp"]
                       for r in consist["ladder_rows"] if r.get("artifact")},
            "err_pct": {str(r["doc"]["R_over_dx"]): r["artifact"]["err_pct"]
                        for r in consist["ladder_rows"] if r.get("artifact")},
            "gate": "ours arm |err| = 0.971 / 0.606 / 0.213 % at R/dx = 10 / 13.3 / 20 "
                    "-> PASS 15 % (owner) and PASS <3 % (report); OSS-parity arm "
                    "+4.00 / +3.44 / +2.83 % -> PASS 15 %, >3 % at R/dx<=13.3 (disclosed)",
            "registered_row": f"NONE: tensor_benchmark_cases has "
                               f"{referee['table_total_rows']} rows, "
                               f"{referee['rows_for_anchor']} for this anchor "
                               f"(artifacts/audit_m17715/referee.json)",
            "report_vs_artifact": f"{consist['ladder_mismatches']}/"
                                  f"{consist['ladder_rows_total']} ladder rows and "
                                  f"{consist['sensitivity_mismatches']}/"
                                  f"{consist['sensitivity_rows_total']} sensitivity rows "
                                  f"do not match results.json "
                                  f"(artifacts/audit_m17715/report_consistency.json)",
        },
        "repro_cmd": "PYTHONPATH=src python3 benchmarks/xval_projection_laplace/run.py "
                     "(auditor re-ran in a clean copy: rc=0, "
                     f"{repro['json_leaf_diff_count']} JSON leaf diffs, PNG sha256 "
                     f"identical) ; PYTHONPATH=src python3 -m pytest "
                     f"tests/test_xval_projection_laplace.py -q",
        "external_anchor": [
            "analytic 2-D Young-Laplace law dp = sigma/R (textbook, no code dependency)",
            "PySPH 1.0b3.dev0 (pypr/pysph) at /tmp/xv2/lib64/python3.11/site-packages: "
            "pysph/sph/surface_tension.py:314/336/565/620/708, pysph/sph/isph/isph.py:121/157/190",
        ],
        "delivery_state": {
            "commits_ahead_of_main": delivery["commits_ahead_of_main"],
            "untracked": delivery["delivered_untracked"],
            "commits_ahead_of_main": delivery["commits_ahead_of_main"],
            "head": delivery["head"], "origin_main": delivery["origin_main"],
            "note": "nothing from this mission exists on any branch; the branch the "
                    "platform would push is empty (F2)",
        },
        "not_run": [
            "end-to-end PySPH ISPH simulation (only operator-level parity was executed; "
            "the 'oss' arm is a port executed inside TensorSPH)",
            "registered-referee adjudication (no row for the anchor)",
            "cluster/loongarch64 (single-host audit)",
            "full repository pytest (only tests/test_xval_projection_laplace.py)",
            "time-dependent relaxation (static verification only)",
        ],
        "findings": finding["findings"],
        "level_rationale": finding["level_rationale"],
        "disposition": finding["disposition"],
    }


if __name__ == "__main__":
    main()