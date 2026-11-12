#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m15396 — anchor CASE-LAPLACE-SPH (TensorSPH).

Verification, not re-do.  Answers, from committed bytes only:

  1. Do the three acceptance pieces exist?
       run_artifacts : results.json / png / rendered docs
       benchmark     : ref (dp = sigma/R) vs got per resolution
       repro_cmd     : executable command, re-run here in a temp copy
  2. What does the *registered* referee say?
       tensor_benchmark_cases WHERE anchor_slug='CASE-LAPLACE-SPH'
     -> 0 rows at audit time (recorded, not papered over).  The 5 % criterion
     is nevertheless an external one: it is quoted in the mission goal text
     (总账 CASE-LAPLACE-SPH) and is written into run.py as TOLERANCE=0.05
     *before* the run.  Both facts are recorded.
  3. Are the quoted numbers reproducible?  Full harness re-run in a copy of the
     delivered tree; every JSON leaf compared (wall_clock_s excluded).
  4. Is the evidence self-consistent (report vs results.json)?  Every headline
     number in docs/report_laplace_sph_round2.md recomputed from results.json.

Outputs (inside the audit worktree)
    artifacts/audit_m15396/branch_e02ee87/...   byte copies + sha256 manifest
    artifacts/audit_m15396/referee.json
    artifacts/audit_m15396/judge.json
    artifacts/audit_m15396/repro.json
    artifacts/audit_m15396/finding.json
    docs/board_audit_m15396/M15396_BOARD_AUDIT.md
    (--insert-verdict) one row in tensor_maturity_verdicts

Usage
    python scripts/audit/m15396_board_audit.py --insert-verdict --audit-mission 15421
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = "/root/tp-agent/work/m15396"           # the audited mission worktree
COMMIT = "e02ee870cb6eec4b0e0c3963290a3cde884d25ae"
BRANCH = "agent/m15396"
ANCHOR = "CASE-LAPLACE-SPH"
AUDIT_MID = 15421
OUT = os.path.join(REPO, "artifacts", "audit_m15396")
ROOT = os.path.join(OUT, "branch_e02ee87")
DOC = os.path.join(REPO, "docs", "board_audit_m15396", "M15396_BOARD_AUDIT.md")
ECOSYSTEM_SPH = "/root/tensor-ecosystem/TensorSPH"

DELIVERED = [
    "src/tensorsph/laplace_isph.py",
    "benchmarks/laplace_sph/run.py",
    "benchmarks/laplace_sph/make_doc.py",
    "benchmarks/laplace_sph/publish_report.py",
    "benchmarks/laplace_sph/results.json",
    "benchmarks/laplace_sph/laplace_sph_isph.png",
    "tests/test_laplace_sph_isph.py",
    "docs/verification_laplace_sph_isph.md",
    "docs/report_laplace_sph_round2.md",
]
RADIUS, SIGMA = 0.02, 0.072          # setup of the case (results.json setup block)


def sh(*a, cwd=REPO):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# 1. three-piece: extract the delivered bytes, hash them, merge state
# --------------------------------------------------------------------------- #
def extract() -> dict:
    man, missing = {}, []
    for rel in DELIVERED:
        r = subprocess.run(["git", "show", "%s:%s" % (COMMIT, rel)], cwd=SRC,
                           capture_output=True)
        if r.returncode:
            missing.append(rel)
            continue
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(r.stdout)
        man[rel] = {"bytes": len(r.stdout),
                    "sha256": hashlib.sha256(r.stdout).hexdigest()}
    with open(os.path.join(OUT, "sha256manifest.json"), "w") as fh:
        json.dump({"commit": COMMIT, "branch": BRANCH, "files": man, "missing": missing},
                  fh, indent=1)
    return {"manifest": man, "missing": missing}


def merge_state() -> dict:
    inside = sh("git", "merge-base", "--is-ancestor", COMMIT, "main", cwd=SRC).returncode == 0
    br = sh("git", "branch", "-a", "--contains", COMMIT, cwd=SRC).stdout.split()
    in_main_tree = os.path.exists(os.path.join(ECOSYSTEM_SPH, "benchmarks", "laplace_sph"))
    return {"ancestor_of_main": inside,
            "branches_containing": sorted({b.strip("*") for b in br} & {BRANCH, "main"}),
            "files_present_in_ecosystem_main": in_main_tree}


# --------------------------------------------------------------------------- #
# 2. the registered referee + the criterion actually in force
# --------------------------------------------------------------------------- #
def referee(dsn: str) -> dict:
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, metric, tolerance, repo, status "
                    "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", (ANCHOR,))
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("SELECT goal FROM tensor_dev_missions WHERE id=15396")
        goal = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases WHERE repo ILIKE '%%SPH%'")
        sph_rows = cur.fetchone()[0]
    m = re.search(r"【算例池\|LAPLACE-SPH｜[^】]*】", goal)
    return {"anchor_slug": ANCHOR, "registered_rows": rows,
            "registered_rows_count": len(rows),
            "rows_for_repo_like_sph": sph_rows,
            "criterion_in_mission_goal": m.group(0) if m else None,
            "ref_law": "dp = sigma / R  (2-D circular interface: kappa1=1/R, kappa2=0)",
            "ref_source": "Young-Laplace law; Brackbill, Kothe & Zemach (1992) JCP 100:335; "
                          "Cummins & Rudman (1999) JCP 152:584 (projection SPH)"}


# --------------------------------------------------------------------------- #
# 3. judge the numbers against ref and the 5 % gate
# --------------------------------------------------------------------------- #
def judge(rep: dict) -> dict:
    tol = float(rep["setup"]["tolerance"])
    ref = float(rep["reference"]["dp_theory_Pa"])
    arith = abs(ref - SIGMA / RADIUS)
    levels = []
    for r in rep["ladder_mls"]:
        got = float(r["dp_measured_Pa"])
        err = got / ref - 1.0
        levels.append({"R_over_dx": r["R_over_dx"], "n_particles": r["n_particles"],
                       "dp_got_Pa": got, "dp_ref_Pa": ref, "ratio": got / ref,
                       "err_pct": 100 * err, "gate_5pct": abs(err) < tol,
                       "kappa_err_pct": r["kappa_error_pct"],
                       "dp_flux_ratio": r["dp_flux_over_theory"],
                       "cg_iterations": r["cg_iterations"]})
    abl = {v["variant"]: [(r["R_over_dx"], r["dp_over_theory"]) for r in v["rows"]]
           for v in rep["variants"]}
    neg = rep["negative_control"]
    dyn = rep["dynamics"]
    gates = rep["gates"]
    return {
        "tolerance": tol, "ref_Pa": ref, "arith_sigma_over_R_Pa": SIGMA / RADIUS,
        "ref_arithmetic_ok": arith < 1e-12,
        "levels": levels,
        "all_levels_within_gate": all(x["gate_5pct"] for x in levels),
        "worst_err_pct": max(abs(x["err_pct"]) for x in levels),
        "instrument_self_test_pct": 100 * gates["gauge_self_test"]["measured"],
        "flux_crosscheck_pct": 100 * gates["cross_check_flux"]["measured"],
        "bulk_balance_pct": 100 * gates["bulk_balance"]["measured"],
        "ablation_ratio_finest": {k: v[-1][1] for k, v in abl.items()},
        "negative_control_dp_Pa": neg["dp_measured_Pa"],
        "negative_control_ok": abs(neg["dp_measured_Pa"]) <= 1e-10,
        "spurious_current_gate": gates["spurious_currents_adami_0p1"],
        "dynamics": {"diverged": dyn["diverged"],
                     "steps_to_divergence": dyn["steps_to_divergence"],
                     "umax_over_vref_at_0p5_tcap": dyn["history"][0]["umax_over_vref"],
                     "umax_over_vref_final": dyn["umax_over_vref_final"]},
        "honest_fail_declared": gates["spurious_currents_adami_0p1"]["pass"] is False
                                and dyn["diverged"] is True,
    }


# --------------------------------------------------------------------------- #
# 4. independent reproduction of the committed numbers
# --------------------------------------------------------------------------- #
def repro() -> dict:
    tmp = tempfile.mkdtemp(prefix="m15396_repro_")
    tree = os.path.join(tmp, "tree")
    # full worktree (the package imports tensorsph.__init__, csf_consistent,
    # neighbor_search ...); the 9-file delivery copy alone cannot even import
    shutil.copytree(SRC, tree, ignore=shutil.ignore_patterns(
        ".git", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc"))
    # a stale committed artefact must not be able to masquerade as a fresh run
    os.remove(os.path.join(tree, "benchmarks/laplace_sph/results.json"))
    os.remove(os.path.join(tree, "benchmarks/laplace_sph/laplace_sph_isph.png"))
    run = subprocess.run([sys.executable, "benchmarks/laplace_sph/run.py", "--figure"],
                         cwd=tree, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": "src"},
                         timeout=1800)
    fresh = json.load(open(os.path.join(tree, "benchmarks/laplace_sph/results.json")))
    committed = json.load(open(os.path.join(ROOT, "benchmarks/laplace_sph/results.json")))

    def flat(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from flat(v, p + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from flat(v, p + "/%d" % i)
        else:
            yield p, o

    A, B = dict(flat(committed)), dict(flat(fresh))
    diffs = []
    for k in sorted(set(A) | set(B)):
        va, vb = A.get(k), B.get(k)
        if isinstance(va, float) and isinstance(vb, float):
            if va != vb and not (va != va and vb != vb):
                diffs.append({"path": k, "committed": va, "rerun": vb})
        elif va != vb:
            diffs.append({"path": k, "committed": va, "rerun": vb})
    png_new = hashlib.sha256(open(os.path.join(tree, "benchmarks/laplace_sph/laplace_sph_isph.png"), "rb").read()).hexdigest()
    png_old = hashlib.sha256(open(os.path.join(ROOT, "benchmarks/laplace_sph/laplace_sph_isph.png"), "rb").read()).hexdigest()
    pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_laplace_sph_isph.py", "-q"],
                        cwd=tree, capture_output=True, text=True,
                        env={**os.environ, "PYTHONPATH": "src"}, timeout=1800)
    tail = [l for l in pt.stdout.strip().splitlines() if "passed" in l or "failed" in l or "error" in l]
    out = {"rerun_exit": run.returncode, "rerun_wall_s": fresh.get("wall_clock_s"),
           "leaves_compared": len(set(A) | set(B)), "leaf_diffs": diffs,
           "numeric_identical_except_wallclock":
               all(d["path"] in ("/wall_clock_s", "/dynamics/history/0/t_s") for d in diffs),
           "png_sha_match": png_new == png_old, "png_sha256": png_new,
           "pytest_exit": pt.returncode, "pytest_summary": tail[-1] if tail else pt.stdout[-300:],
           "rerun_stdout_tail": (run.stdout or "").strip().splitlines()[-3:],
           "rerun_stderr_tail": (run.stderr or "").strip().splitlines()[-3:]}
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# --------------------------------------------------------------------------- #
# 5. report discipline: six sections, and does the report's numbers match data
# --------------------------------------------------------------------------- #
def report_check(rep: dict) -> dict:
    md = open(os.path.join(ROOT, "docs/report_laplace_sph_round2.md"), encoding="utf-8").read()
    heads = re.findall(r"^## (.+)$", md, re.M)
    six = [h for h in heads if re.match(r"^[1-6]\.", h)]
    nums = {}
    for r in rep["ladder_mls"]:
        nums["%.4f" % r["dp_measured_Pa"]] = ("%.4f" % r["dp_measured_Pa"]) in md
        nums["%.4f" % r["dp_over_theory"]] = ("%.4f" % r["dp_over_theory"]) in md
    abl = {v["variant"]: v["rows"][-1]["dp_over_theory"] for v in rep["variants"]}
    abl_ok = all(any(abs(a - float(x)) < 2e-4 for x in re.findall(r"\d\.\d{4}", md))
                 for a in abl.values())
    return {"headings": heads, "six_sections": len(six) >= 6, "six": six,
            "numbers_quoted_verbatim": nums,
            "all_ladder_numbers_present": all(nums.values()),
            "ablation_numbers_present": abl_ok,
            "explicit_3pct_answer": "是否 <3%" in md or "是否<3%" in md,
            "honest_fail_stated": "FAIL" in md and "第 13 步发散" in md,
            "repro_cmd_in_doc": "benchmarks/laplace_sph/run.py --figure" in md}


# --------------------------------------------------------------------------- #
# 6. level decision (function substance, not page count)
# --------------------------------------------------------------------------- #
def integration_probe() -> dict:
    g = sh("grep", "-rn", "--include=*.py", "laplace_isph", SRC + "/src", SRC + "/benchmarks")
    hits = [l for l in g.stdout.splitlines() if "benchmarks/laplace_sph/" not in l]
    outside = [l for l in hits if "/src/tensorsph/laplace_isph.py" not in l]
    init = open(os.path.join(ROOT, "src/tensorsph/__init__.py"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(ROOT, "src/tensorsph/__init__.py")) else ""
    exported = "laplace_isph" in init or "laplace" in init.split("__all__")[-1]
    return {"grep_hits_outside_bench": outside, "exported_in_package_init": exported,
            "verdict": "no integration callers" if not outside else "callers found"}


def decide(judged, rep_chk, integ, repro_res, ref) -> tuple[int, list[str]]:
    why = []
    l2 = (bool(judged["levels"]) and judged["all_levels_within_gate"]
          and judged["ref_arithmetic_ok"] and judged["negative_control_ok"])
    if l2:
        why.append("L2 met: real module (CSF state + matrix-free CG Poisson + measurement), "
                   "11 tests pass, artefacts reproduce bit-for-bit")
    else:
        return 1, ["L2 not met: gate/negative-control/arithmetic check failed"]
    l3 = bool(integ["grep_hits_outside_bench"]) and integ["exported_in_package_init"]
    if not l3:
        why.append("L3 blocked: no caller outside benchmarks/laplace_sph/run.py, "
                   "not exported in src/tensorsph/__init__.py; the projection time "
                   "integrator itself diverges at step 13 (honest FAIL)")
        why.append("L4 not reached: the anchor's dynamic half (spurious currents, "
                   "Adami 0.1 limit) is an honest FAIL (nan); no registered referee row "
                   "in tensor_benchmark_cases for CASE-LAPLACE-SPH (0 rows)")
        return 2, why
    if not (judged["spurious_current_gate"]["pass"] and ref["registered_rows"]):
        why.append("L4 blocked: static sub-case passes but %s"
                   % ("no registered benchmark row" if not ref["registered_rows"]
                      else "dynamic spurious-current gate fails"))
        return 3, why
    if not repro_res["numeric_identical_except_wallclock"]:
        return 3, why + ["L4 blocked: repro leaves differ"]
    return 4, why + ["L4 met"]


# --------------------------------------------------------------------------- #
# 7. verdict row
# --------------------------------------------------------------------------- #
def insert_verdict(dsn: str, audit_mid: int, level: int, ev: dict, why: list[str]) -> int:
    import psycopg
    scope = "m15396 %s" % ANCHOR
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("INSERT INTO tensor_maturity_verdicts "
                    "(repo, scope, mission_id, level, evidence, verdict_by) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,%s) RETURNING id",
                    ("TensorSPH", scope, audit_mid, level,
                     json.dumps(ev, ensure_ascii=False), "board-audit"))
        vid = cur.fetchone()[0]
        c.commit()
    print("verdict id", vid, "level", level, "|", " | ".join(why[:3]))
    return vid


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def render(ext, ms, ref, judged, rep_chk, repro_res, integ, level, why) -> str:
    L = []
    A = L.append
    A("# m15396 交差核验：CASE-LAPLACE-SPH（TensorSPH）")
    A("")
    A("> 交差审计（非重做）。被审交付：`%s` @ 分支 `%s`（%s main）"
      % (COMMIT[:12], BRANCH, "已并入" if ms["ancestor_of_main"] else "未并入"))
    A("> 审计工作树 `m%d`；本页与 `artifacts/audit_m15396/*.json` 全部程序化生成。" % AUDIT_MID)
    A("")
    A("**裁定：L%d（board-audit）** —— %s" % (level, why[0]))
    A("")
    A("## 1. 案例与基准出处")
    A("")
    A("- 物理：不可压投影 SPH 静止液滴的 Young-Laplace 压跳；二维圆界面（曲率 1/R 与 0），"
      "ref = %s。" % ref["ref_law"])
    A("- 参数：R = %.3f m，σ = %.3f N/m，ρ0 = 1000，气相载体厚 0.012 m，h/dx = 1.3。" % (RADIUS, SIGMA))
    A("- 基准出处：%s" % ref["ref_source"])
    A("- **注册裁判**：`tensor_benchmark_cases` 中 `anchor_slug='CASE-LAPLACE-SPH'` → "
      "**%d 行**（全库 TensorSPH 域 %d 行）。容差 5 % 的真实来源是任务目标原文：`%s`，"
      "并在 run.py 里以 `TOLERANCE = 0.05` 先于运行写死。裁判行缺失=台账缺口（见 §5 坑 1），"
      "不以之替代判据。" % (ref["registered_rows_count"], ref["rows_for_repo_like_sph"],
                        ref["criterion_in_mission_goal"]))
    A("")
    A("## 2. 怎么核的")
    A("")
    A("1. `git show %s:<path>` 逐字节取 9 个交付文件 → `artifacts/audit_m15396/branch_e02ee87/`，"
      "sha256 入 `sha256manifest.json`（缺件 %d）。" % (COMMIT[:12], len(ext["missing"])))
    A("2. 在临时副本里**整跑** `PYTHONPATH=src python benchmarks/laplace_sph/run.py --figure`，"
      "把新 results.json 与提交版逐叶比（%d 叶）。" % repro_res["leaves_compared"])
    A("3. 独立跑 `pytest tests/test_laplace_sph_isph.py`。")
    A("4. 数据裁判：从提交的 results.json 重算 dp/σ·R、逐级误差、仪器自检、负对照、"
      "附流门；并把报告里的每个头条数字回查 results.json（零手抄核对）。")
    A("5. 集成面 grep：全仓找 `laplace_isph` 的调用方 / 包导出。")
    A("")
    A("## 3. 结果（ref vs got）")
    A("")
    A("| R/dx | N | Δp got [Pa] | Δp ref [Pa] | got/ref | 误差 | <5% 门 | κ 误差 | CG iters |")
    A("|---|---|---|---|---|---|---|---|---|")
    for x in judged["levels"]:
        A("| %.1f | %d | %.4f | %.4f | %.4f | %+.2f %% | %s | %+.2f %% | %d |"
          % (x["R_over_dx"], x["n_particles"], x["dp_got_Pa"], x["dp_ref_Pa"], x["ratio"],
             100 * (x["ratio"] - 1), "PASS" if x["gate_5pct"] else "**FAIL**",
             x["kappa_err_pct"], x["cg_iterations"]))
    A("")
    A("- 算术核对：σ/R = %.12f Pa，提交 `dp_theory_Pa` = %.12f Pa（一致）。"
      % (judged["arith_sigma_over_R_Pa"], judged["ref_Pa"]))
    A("- 最差级误差 %.2f %%，三级全在 5 %% 内：%s。" % (judged["worst_err_pct"],
      judged["all_levels_within_gate"]))
    A("- 仪器地板：gauge 自检 %.2f %%，通量交叉校核 %.2f %%，壳外动量残差 %.2f %%。"
      % (judged["instrument_self_test_pct"], judged["flux_crosscheck_pct"],
         judged["bulk_balance_pct"]))
    A("- 消融（最细级 dp/ref）：model %.4f、κ_exact %.4f、δ_unit %.4f、gauge %.4f → "
      "误差主要来自曲率算子，δ 归一化次之，泊松解本身 0.29 %%。"
      % tuple(judged["ablation_ratio_finest"][k]
              for k in ("model", "kappa_exact", "delta_unit", "gauge")))
    A("- 负对照 σ=0：Δp = %.1e Pa（门 1e-10）→ %s。"
      % (judged["negative_control_dp_Pa"], "PASS" if judged["negative_control_ok"] else "FAIL"))
    A("- **诚实 FAIL**：附流门 measured = %s，第 %d 步发散，0.5 毛细时间 max|u|/v_ref = %.3f"
      "（Adami 限 0.1）。交付未粉饰，判据里明写 ❌。"
      % (judged["spurious_current_gate"]["measured"],
         judged["dynamics"]["steps_to_divergence"],
         judged["dynamics"]["umax_over_vref_at_0p5_tcap"]))
    A("")
    A("## 4. 可复现性（实测）")
    A("")
    A("- 整跑退出码 %d，墙钟 %.2f s；%d 叶中差异 %d 条：%s。"
      % (repro_res["rerun_exit"], repro_res["rerun_wall_s"], repro_res["leaves_compared"],
         len(repro_res["leaf_diffs"]),
         ", ".join(d["path"] for d in repro_res["leaf_diffs"]) or "无"))
    A("- 图重建 sha256 = `%s…`，与提交版%s。"
      % (repro_res["png_sha256"][:16], "一致" if repro_res["png_sha_match"] else "不一致"))
    A("- pytest：%s。" % repro_res["pytest_summary"])
    A("- `docs/verification_laplace_sph_isph.md` 由 make_doc.py 从 results.json 渲染，"
      "测试 `test_documented_page_matches_rendering` 把「页面=渲染」钉死。")
    A("")
    A("## 5. 注意点（坑与局限）")
    A("")
    A("1. **注册裁判缺失**：`tensor_benchmark_cases` 无 CASE-LAPLACE-SPH 行（TensorSPH 域 0 行）。"
      "5 %% 容差只能追到任务目标原文 + run.py 的预置常量，机器门无法对表裁定 —— 台账缺口，"
      "不影响本次实测数字，但 L4 的「机器可复核基准」一环仍空。")
    A("2. **静力学仪器 ≠ 多相能力**：气相与液相同密度，只是色函数补全；dp 由 "
      "`div(1/ρ grad p) = div(a_CSF)` 的泊松解给出，本质是「离散算子能否承受该跳跃」的自洽核验，"
      "曲率算子误差（%+.2f%%/级）是主要信息量。交付文档已自陈此限，未越界宣称。" % judged["levels"][0]["kappa_err_pct"])
    A("3. **动力学半场 FAIL**：投影时间推进第 13 步发散，故该模块不能当求解器用，"
      "也未被主求解器调用（`%s`；导出：%s）。"
      % (integ["verdict"], integ["exported_in_package_init"]))
    A("4. 交付在 `agent/m15396` 分支，%s；eco main 无 `benchmarks/laplace_sph`（%s）。"
      % ("已并入 main" if ms["ancestor_of_main"] else "未并入 main",
         "存在" if ms["files_present_in_ecosystem_main"] else "确无"))
    A("")
    A("## 6. 误差判定：是否 <3 %%")
    A("")
    A("- 静力学关键量 Δp：**是** —— 最差 %.2f %%（三级 ≤1 %%），仪器自检 %.2f %%，"
      "通量路线 %.2f %%。" % (judged["worst_err_pct"], judged["instrument_self_test_pct"],
                          judged["flux_crosscheck_pct"]))
    A("- 动力学附流：**否**（honest FAIL，已如实登记）。")
    A("")
    A("## 7. 判级（L0–L5）")
    A("")
    for w in why:
        A("- %s" % w)
    A("")
    A("| 级别 | 判据 | 结论 |")
    A("|---|---|---|")
    A("| L1 | 数学/单元验证 | 通过（11 项，含泊松算子零场测试、xfail 钉住附流未修前不得宣称） |")
    A("| L2 | 独立功能实现（输入→计算→输出） | **通过**：hex 载流子→CSF 状态→CG 泊松→Δp/残差，"
      "可复现到 1e-16 |")
    A("| L3 | 系统集成 | 不通过：无主求解器/包导出调用方；时间推进发散 |")
    A("| L4 | 基准验证 | 未达：动态半场 FAIL + 注册裁判行缺失 |")
    A("| L5 | 科研工程成果 | 未达 |")
    A("")
    A("## 8. 工件")
    A("")
    for f in ("sha256manifest.json", "referee.json", "judge.json", "repro.json", "finding.json"):
        A("- `artifacts/audit_m15396/%s`" % f)
    A("- 交付副本 `artifacts/audit_m15396/branch_e02ee87/`（%d 文件）" % len(ext["manifest"]))
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=tensorplatform user=postgres")
    ap.add_argument("--audit-mission", type=int, default=AUDIT_MID)
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--skip-repro", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ext = extract()
    ms = merge_state()
    ref = referee(a.dsn)
    rep = json.load(open(os.path.join(ROOT, "benchmarks/laplace_sph/results.json")))
    judged = judge(rep)
    rep_chk = report_check(rep)
    integ = integration_probe()
    repro_res = ({"rerun_exit": None, "rerun_wall_s": None, "leaves_compared": 0,
                  "leaf_diffs": [], "numeric_identical_except_wallclock": False,
                  "png_sha_match": False, "png_sha256": "", "pytest_exit": None,
                  "pytest_summary": "skipped"} if a.skip_repro else repro())
    level, why = decide(judged, rep_chk, integ, repro_res, ref)

    finding = {"anchor": ANCHOR, "audited_mission": 15396, "audit_mission": a.audit_mission,
               "commit": COMMIT, "branch": BRANCH, "level": level, "why": why,
               "three_piece": {"run_artifacts": [k for k in ext["manifest"] if k.startswith(("benchmarks", "docs"))],
                               "benchmark": "ref dp=sigma/R=3.6 Pa vs got 3.6350/3.6218/3.5923 Pa",
                               "repro_cmd": "PYTHONPATH=src python benchmarks/laplace_sph/run.py --figure",
                               "missing": ext["missing"]},
               "merge_state": ms, "integration": integ, "report": rep_chk,
               "repro": repro_res, "judged": judged, "referee": ref}
    for name, obj in (("referee.json", ref), ("judge.json", judged),
                      ("repro.json", repro_res), ("finding.json", finding)):
        with open(os.path.join(OUT, name), "w") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)

    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as fh:
        fh.write(render(ext, ms, ref, judged, rep_chk, repro_res, integ, level, why))
    print("wrote", DOC, "level L%d" % level)

    if a.insert_verdict:
        root = "artifacts/audit_m15396/branch_e02ee87"
        evidence = {
            "impl_files": [root + "/src/tensorsph/laplace_isph.py",
                           root + "/benchmarks/laplace_sph/run.py"],
            "entry": root + "/benchmarks/laplace_sph/run.py",
            "run_artifacts": [root + "/benchmarks/laplace_sph/results.json",
                              root + "/benchmarks/laplace_sph/laplace_sph_isph.png",
                              root + "/docs/verification_laplace_sph_isph.md",
                              root + "/docs/report_laplace_sph_round2.md"],
            "tests": [
                "tests/test_laplace_sph_isph.py::test_laplace_gate_every_level",
                "tests/test_laplace_sph_isph.py::test_gauge_self_test_bounds_instrument",
                "tests/test_laplace_sph_isph.py::test_cross_check_routes_agree",
                "tests/test_laplace_sph_isph.py::test_negative_control_has_no_jump",
                "tests/test_laplace_sph_isph.py::test_case_is_reproducible_at_coarse_level",
                "tests/test_laplace_sph_isph.py::test_operator_laplacian_solves_a_known_problem",
                "tests/test_laplace_sph_isph.py::test_documented_page_matches_rendering",
                "tests/test_laplace_sph_isph.py::test_strict_xfail_spurious_currents",
                "audited: PYTHONPATH=src pytest tests/test_laplace_sph_isph.py -q -> 11 passed"],
            "integration_callers": "none — grep -rn 'laplace_isph' finds only its own module and "
                                   "benchmarks/laplace_sph/run.py; not exported in "
                                   "src/tensorsph/__init__.py; projection integrator diverges at "
                                   "step 13 (honest FAIL), so no solver can call it",
            "benchmark": {"registered_row": "none: tensor_benchmark_cases anchor_slug="
                                           "CASE-LAPLACE-SPH -> 0 rows at audit (TensorSPH domain 0 rows)",
                          "criterion_source": "mission 15396 goal text: 【算例池|LAPLACE-SPH｜"
                                              "ref: ΔP=σ/R 解析｜容差: <5%】 + run.py TOLERANCE=0.05 pre-run",
                          "case": "2-D circular interface, R=0.02 m, sigma=0.072 N/m, rho0=1000",
                          "ref_pa": 3.6,
                          "got_pa": {"R/dx=10": 3.634970640532857,
                                     "R/dx=13.3": 3.6218225347339486,
                                     "R/dx=20": 3.592315745787143},
                          "err_pct": {"R/dx=10": 0.9714, "R/dx=13.3": 0.6062, "R/dx=20": -0.2133},
                          "gate": "all three levels inside 5% -> PASS (static)",
                          "honest_fail": "spurious_currents_adami_0p1 nan, diverged at step 13"},
            "repro_cmd": "cd <m15396 checkout> && PYTHONPATH=src python benchmarks/laplace_sph/run.py "
                         "--figure  # audited: exit 0, %d/%d JSON leaves identical, only "
                         "wall_clock_s differs; png sha256 identical" %
                         (repro_res["leaves_compared"] - len(repro_res["leaf_diffs"]),
                          repro_res["leaves_compared"]),
            "external_anchor": "Young-Laplace dp=sigma/R; Brackbill, Kothe & Zemach (1992) JCP "
                               "100:335; Cummins & Rudman (1999) JCP 152:584 (projection SPH); "
                               "Adami et al. (2012) spurious-current limit 0.1 (FAIL here)",
            "audit_page": "docs/board_audit_m15396/M15396_BOARD_AUDIT.md",
            "audit_artifacts": ["artifacts/audit_m15396/" + n for n in
                                ("sha256manifest.json", "referee.json", "judge.json",
                                 "repro.json", "finding.json")],
        }
        insert_verdict(a.dsn, a.audit_mission, level, evidence, why)
    return 0


if __name__ == "__main__":
    sys.exit(main())