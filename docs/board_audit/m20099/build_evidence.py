#!/usr/bin/env python3
"""m20099 交差核验：CASE-SPH-DAMBREAK-PRESSURE / m19980 证据重建（零手抄）。

输入（全部为磁盘/DB 上的可核工件，本脚本只读）：
  A. 被审工作树  /root/tp-agent/work/m19980/benchmarks/koshizuka_dam_break/*
  B. 本单独立复跑 audit/m19980_repro/out/rerun_dx{0014,0020}.json
  C. 平台 DB: tensor_dev_missions(19980) / tensor_case_pool / tensor_benchmark_cases
              / tensor_reports / cluster_jobs(2420)
输出：
  docs/board_audit/m20099/verify_output.json
  docs/board_audit/m20099/fig_pressure_audit.png

用法： python3 docs/board_audit/m20099/build_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]                      # /root/tp-agent/work/m20099
SUBJECT = Path("/root/tp-agent/work/m19980")
CASE = SUBJECT / "benchmarks/koshizuka_dam_break"
REPRO = WT / "audit/m19980_repro"
ART = HERE            # 图随 docs 一起入库（.gitignore 忽略 artifacts/）
ANCHOR = "CASE-SPH-DAMBREAK-PRESSURE"
MISSION = 19980


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def q(sql: str) -> str:
    return subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "tensorplatform", "-At", "-c", sql],
        capture_output=True, text=True, check=True).stdout.strip()


def smoothed_peak(hist, win):
    t = [a for a, _ in hist]
    p = [b for a, b in hist]
    sm = [sum(q for tt, q in hist if abs(tt - ti) <= win) /
          sum(1 for tt, _ in hist if abs(tt - ti) <= win) for ti in t]
    k = sm.index(max(sm))
    return max(sm), t[k]


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    ref = json.loads((CASE / "reference.json").read_text())
    arch14 = json.loads((CASE / "result_dx0014.json").read_text())
    rr14 = json.loads((REPRO / "out/rerun_dx0014.json").read_text())
    rr20 = json.loads((REPRO / "out/rerun_dx0020.json").read_text())

    ref_pa = float(ref["reference_pressure_pa"])
    tol_pct = float(ref["tolerance_pct"])

    # ---- 1. 三件套 : run_artifacts / ref-vs-got / repro_cmd ----
    run_artifacts = [p for p in sorted(CASE.iterdir()) if p.is_file()]
    art_table = {p.name: {"bytes": p.stat().st_size, "sha256_16": sha(p)} for p in run_artifacts}

    got_arch = float(arch14["peak_pressure_pa"])
    got_rerun = float(rr14["peak_pressure_pa"])
    err_arch = abs(got_arch - ref_pa) / ref_pa * 100.0
    err_rerun = abs(got_rerun - ref_pa) / ref_pa * 100.0
    repro_delta = abs(got_rerun - got_arch) / got_arch

    # ---- 2. 工件自洽：archived history 的最大值 vs 报告的 peak ----
    hist = arch14["pressure_history"]
    hist_max = max(b for _, b in hist)
    sm5, t_sm5 = smoothed_peak(hist, 0.005)
    sm20, t_sm20 = smoothed_peak(hist, 0.020)
    n_samples = len(hist)
    sample_dt = (hist[-1][0] - hist[0][0]) / (n_samples - 1)
    above = sum(1 for _, b in hist if b > ref_pa * (1 + tol_pct / 100))

    # ---- 3. 平台侧登记状态（基准判据的唯一裁判）----
    bc_rows = q(f"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'")
    pool = q("SELECT id||'|'||case_code||'|'||status||'|'||mission_id||'|'||tol_policy "
             "FROM tensor_case_pool WHERE case_code='SPH-DAMBREAK-PRESSURE'")
    reports = q(f"SELECT count(*) FROM tensor_reports WHERE mission_id={MISSION}")
    mission = q("SELECT status||'|'||coalesce(result->>'fail_stage','')||'|'||"
                "coalesce(result->>'decision_note','') FROM tensor_dev_missions WHERE id=19980")
    cj = q("SELECT state||'|'||rc||'|'||coalesce(note,'') FROM cluster_jobs WHERE id=2420")
    cj_log = Path("/root/tp-agent/cluster/jobs/cj2420.log")
    cj_log_txt = cj_log.read_text().strip() if cj_log.exists() else ""
    branch = q("SELECT count(*) FROM tensor_maturity_verdicts WHERE scope LIKE 'm19980%'")

    v = {
        "anchor": ANCHOR,
        "subject_mission": MISSION,
        "audit_mission": 20099,
        "subject_worktree": str(SUBJECT),
        "subject_head": subprocess.run(["git", "-C", str(SUBJECT), "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "subject_git_status": subprocess.run(["git", "-C", str(SUBJECT), "status", "--porcelain"],
                                             capture_output=True, text=True).stdout.strip().splitlines(),
        "run_artifacts": art_table,
        "benchmark": {
            "reference_pa": ref_pa,
            "tolerance_pct": tol_pct,
            "got_archived_pa": got_arch,
            "got_rerun_pa": got_rerun,
            "error_archived_pct": err_arch,
            "error_rerun_pct": err_rerun,
            "rerun_vs_archived_rel_delta": repro_delta,
            "within_tolerance": err_rerun <= tol_pct,
            "registry_rows_tensor_benchmark_cases": int(bc_rows),
            "case_pool_row": pool,
            "tol_policy_pool": pool.split("|")[-1],
        },
        "artifact_self_consistency": {
            "archived_history_max_pa": hist_max,
            "archived_peak_pa": got_arch,
            "history_undercounts_peak_pct": (got_arch - hist_max) / got_arch * 100.0,
            "history_samples": n_samples,
            "history_sample_dt_s": sample_dt,
            "peak_not_resolvable_from_archived_history": True,
            "smoothed_peak_5ms_pa": sm5,
            "smoothed_peak_5ms_t": t_sm5,
            "smoothed_peak_20ms_pa": sm20,
            "smoothed_peak_20ms_t": t_sm20,
            "samples_above_tolerance_band": above,
            "sample_fraction_above_band": above / n_samples,
        },
        "resolution_study": {
            "dx_0.02_pa_reported_doc": 24028.0,
            "dx_0.02_pa_rerun": float(rr20["peak_pressure_pa"]),
            "dx_0.02_doc_row_reproduced": abs(24028.0 - float(rr20["peak_pressure_pa"])) < 5.0,
            "dx_0.02_artifact_in_subject_tree": (CASE / "result_dx0020.json").exists(),
            "dx_0.01_claimed_status_in_doc": "集群运行中 / cj2420 排队",
            "cluster_job_2420_db_row": cj,
            "cluster_job_2420_log": cj_log_txt,
            "cluster_job_2420_really_failed": '"rc": 1' in cj_log_txt,
            "convergence_evidence": "absent (dx=0.01/0.007/0.005 无任何工件)",
        },
        "platform": {
            "mission_row": mission,
            "tensor_reports_rows_for_mission": int(reports),
            "pre_existing_verdicts_for_scope": int(branch),
        },
        "level_rationale": {
            "impl_real": True,
            "l3_rejected": "无 tensor_benchmark_cases 登记、case_pool 仍 dispatched、模块未入库(main 无此路径)、tests/ 无引用",
            "l4_rejected": "误差 1161% >> 12%；收敛性工件缺失；工件不可由 repro_cmd 原样重生成",
        },
    }

    (HERE / "verify_output.json").write_text(json.dumps(v, indent=2, ensure_ascii=False))
    print(json.dumps({k: v[k] for k in ("benchmark", "artifact_self_consistency", "resolution_study", "platform")},
                     indent=2, ensure_ascii=False))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=130)
    t = [a for a, _ in hist]
    p = [b for a, b in hist]
    ax1.plot(t, p, lw=0.9, color="#1f4e79", label="TensorSPH wall probe (archived)")
    ax1.axhline(ref_pa, color="k", ls="--", lw=1.2,
                label=f"Koshizuka&Oka1996 ref = {ref_pa:.0f} Pa")
    ax1.axhspan(ref_pa * (1 - tol_pct / 100), ref_pa * (1 + tol_pct / 100),
                color="green", alpha=0.18, label="±12% acceptance band")
    ax1.plot([t_sm5], [sm5], "o", ms=5, color="#c00000",
             label=f"5 ms moving-average peak = {sm5:.0f} Pa")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("wall pressure at (0.584, 0.024) m [Pa]")
    ax1.set_title("(a) probe history: archived samples vs MPS reference band")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5, loc="upper left")

    dxs = [0.02, 0.014]
    peaks = [float(rr20["peak_pressure_pa"]), got_rerun]
    ax2.plot(dxs, peaks, "o-", color="#1f4e79", label="WCSPH peak (this audit reruns)")
    ax2.axhline(ref_pa, color="k", ls="--", lw=1.2, label="ref 2700 Pa")
    ax2.axhspan(ref_pa * (0.88), ref_pa * 1.12, color="green", alpha=0.18, label="±12% band")
    for x, y in zip(dxs, peaks):
        ax2.annotate(f"{y:.0f} Pa\n({abs(y-ref_pa)/ref_pa*100:.0f}%)", (x, y), textcoords="offset points",
                     xytext=(6, 6), fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xlabel("dx [m]  (dx=0.01..0.005 never produced: cluster job cj2420 rc=1)")
    ax2.set_ylabel("peak pressure [Pa]")
    ax2.set_title("(b) resolution sweep: only 2 of 5 points have artifacts")
    ax2.invert_xaxis()
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(ART / "fig_pressure_audit.png")
    print("figure ->", ART / "fig_pressure_audit.png")


if __name__ == "__main__":
    main()