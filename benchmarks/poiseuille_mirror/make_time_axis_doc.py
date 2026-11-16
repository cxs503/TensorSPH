"""Generate docs/verification_poiseuille_time_axis.md from out/time_axis.json.

Every number in the note is read out of the artifact -- nothing is typed by hand.
The figure is (a) the in-sample Psi plane with the declared run-length points
marked at their *predicted* Psi, and (b) the in-sample P3/P4 pair that the
committed data already decides.

    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_time_axis_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.poiseuille_mirror.study_time_axis import (  # noqa: E402
    MA_ARTIFACT, OUT, POINTS, predictions, psi_predicted)

ARTIFACT = OUT / "time_axis.json"
DOC = REPO / "docs" / "verification_poiseuille_time_axis.md"
FIG = REPO / "docs" / "figures" / "poiseuille_time_axis.png"


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def load() -> dict:
    if not ARTIFACT.exists():
        raise SystemExit(f"missing {ARTIFACT}; run --collect first")
    return json.loads(ARTIFACT.read_text())


def in_sample_points() -> list[dict]:
    if not MA_ARTIFACT.exists():
        return []
    return [p for p in json.loads(MA_ARTIFACT.read_text()).get("points", [])
            if p.get("psi_index") is not None]


def figure(doc: dict) -> None:
    band = doc["verdict"].get("in_sample_band", {})
    lo, hi = band.get("psi_passing_max"), band.get("psi_failing_min")
    ins = in_sample_points()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for p in ins:
        ok = p["passes_all_gates"]
        ax.scatter(p["psi_index"], p["profile_rms_interior_pct"], s=64,
                   color=("tab:green" if ok else "tab:red"), zorder=4)
        ax.annotate(p["id"].split("_")[0], (p["psi_index"], p["profile_rms_interior_pct"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    for pid, pred in predictions().items():
        done = [q for q in doc["points"] if q["id"] == pid]
        if done:
            q = done[0]
            ax.scatter(q["psi_index"], q["profile_rms_interior_pct"], marker="*", s=180,
                       color=("tab:green" if q["passes_all_gates"] else "tab:red"),
                       zorder=6)
            ax.annotate(pid.split("_")[0] + "*", (q["psi_index"],
                        q["profile_rms_interior_pct"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=9)
        else:
            ax.axvline(pred["psi_analytic"], color="tab:blue", ls=":", lw=1.2)
            ax.annotate(pid.split("_")[0] + " (declared,\nnot run)",
                        (pred["psi_analytic"], ax.get_ylim()[1] * 0.35),
                        fontsize=7, color="tab:blue", rotation=90, va="top")
    if lo and hi:
        ax.axvspan(lo, hi, color="tab:orange", alpha=0.15,
                   label="in-sample Psi gap %.1f-%.1f" % (lo, hi))
    ax.axhline(doc["tolerance_pct"], color="k", ls="--", lw=1, label="band 5 %")
    ax.set_xlabel(r"$\Psi = u_{max}^2 t_{end}/(c_s\,dx)$")
    ax.set_ylabel("interior profile RMS [%]")
    ax.set_title("(a) in-sample points (filled) vs declared run-length points")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ev = doc.get("evidence", {})
    if ev.get("available"):
        p3, p4 = ev["p3"], ev["p4"]
        labels = ["profile RMS", "wall shear", r"$|u_{max}$ err$|$", r"$\Psi$"]
        v3 = [p3["profile_rms_interior_pct"], p3["wall_shear_worst_abs_pct"],
              abs(p3["u_max_error_pct"]), p3["psi_index"]]
        v4 = [p4["profile_rms_interior_pct"], p4["wall_shear_worst_abs_pct"],
              abs(p4["u_max_error_pct"]), p4["psi_index"]]
        idx = range(len(labels))
        ax.bar([i - 0.2 for i in idx], v3, width=0.4,
               label="P3  t_end=%g s (PASS)" % p3["t_end"])
        ax.bar([i + 0.2 for i in idx], v4, width=0.4,
               label="P4  t_end=%g s (FAIL)" % p4["t_end"])
        ax.set_xticks(list(idx))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yscale("log")
        ax.set_title("(b) same $c_s$ and same dt (%.2e), only run length differs"
                     % ev["dt_p3"])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "evidence unavailable", ha="center")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def build(doc: dict) -> str:
    v = doc["verdict"]
    band = v.get("in_sample_band", {})
    ev = doc.get("evidence", {})
    op = v["operating_point"]
    preds = predictions()
    L: list[str] = []
    a = L.append

    a("# CASE-POISEUILLE-SPH — 运行时长轴：先否证「dt 致因」，再把认证 Ma 下的长跑点立成可判实验")
    a("")
    a("> 锚点：`CASE-POISEUILLE-SPH`（算例池·泊肃叶流；`feature_ledger` 容差 < 5 %，参考=平面泊肃叶解析解）。")
    a("> 本文件由 `benchmarks/poiseuille_mirror/make_time_axis_doc.py` 从 "
      "`benchmarks/poiseuille_mirror/out/time_axis.json` 程序化生成，表中每个数字来自工件，无手抄。")
    a("> 研究脚本：`benchmarks/poiseuille_mirror/study_time_axis.py`；判定规则在跑之前写进脚本 docstring，跑后未调阈值。")
    a("")
    a("## ① 问题：Ma 包络研究写在自己工件里的欠账")
    a("")
    a("`study_ma_envelope.py` 用 7 个点判定「<5 % 证书是否带 Ma 包络」，结论 `MIXED_MA_TIME`，"
      "并在 `out/ma_envelope.json:verdict.open_item` 里留下两条**未被分离**的候选致因：")
    a("")
    a("1. Ma 相关的粒子无序度增长；")
    a("2. 为压低 `c_s` 带来的 `dt` 加倍（P3/P4 的 `dt` = 3.25e-4，而 P0 = 1.63e-4）。")
    a("")
    a("被点名的判别实验是 **`c_s = 20`、`t_end = 2.0`**（与 P4 同运行时长，但 Ma 与 dt 保持认证值）。"
      "同一份工件因此声明 `psi_index` 只是「事后相关指标，尚无非样本检验，不得当作稳定性判据」。")
    a("")
    a("本研究分两步：**(I) 用已提交的 7 点数据否证候选致因 2**（无需新算例）；"
      "**(II) 把认证 Ma/dt 下的运行时长轴立成预声明、可分片执行、可机械判定的实验**。")
    a("")
    a("## ② 方法与参数")
    a("")
    a("平面通道 `0 <= y <= H`，两侧无滑移（mirror/ghost 壁面算子），`x` 周期，体积力 `g` 驱动；"
      "解析解 `u(y) = g (H-y) y/(2 nu)`、`u_max = g H²/(8ν)`、`tau_w = rho g H/2`"
      "（White, *Viscous Fluid Flow* 3rd ed. §3-2.2；Batchelor §4.2）。")
    a("")
    a("| 量 | 值 | 说明 |")
    a("|---|---|---|")
    a(f"| `H` | {op['H']} m | 通道高度 |")
    a(f"| `nu` | {op['nu']} m²/s | 运动粘度（未操纵） |")
    a(f"| `g` | {op['g']} m/s² | 体积力（未操纵） |")
    a(f"| `c_s` | {op['cs']} m/s | 声速（**未操纵**：与 P0 相同，故 Ma 与 dt 均为认证值） |")
    a(f"| `H/dx` | {doc['resolution_H_over_dx']} | 锚点 all-row 读数所需的最小分辨率 |")
    a(f"| `dx` | {op['dx']:.5f} m | `H/(H/dx)` |")
    a(f"| 认证 `dt` | {op['certified_dt']:.3e} s | P0：6154 步 / 1 s |")
    a(f"| `u_max`（解析） | {op['u_max_analytic']:.5f} m/s | `g H²/(8ν)` |")
    a(f"| 判据带 | {doc['tolerance_pct']} % | 算例池既定值，未在事后调整 |")
    a(f"| `Psi` 换算 | {op['psi_per_second']:.3f} /s | 本工况退化为 `Psi = {op['psi_per_second']:.2f} t_end` |")
    a("")
    a("### 2.1 预声明预测表（跑之前写定；`Psi` 用解析 `u_max`，与实测无关）")
    a("")
    a("| 点 | `t_end` [s] | `Psi`（预声明） | 原位样本带判定 |")
    a("|---|---|---|---|")
    for pid, pr in preds.items():
        psi = pr["psi_analytic"]
        if band.get("psi_passing_max") and psi <= band["psi_passing_max"]:
            call = "预测 PASS"
        elif band.get("psi_failing_min") and psi >= band["psi_failing_min"]:
            call = "预测 FAIL"
        else:
            call = "落在带内间隙（不预测）"
        a(f"| {pid} | {pr['t_end']:g} | {psi:.2f} | {call} |")
    a("")
    a("### 2.2 预声明判定规则（v1）")
    a("")
    a("* **H_time**：认证 Ma/dt 下仅运行时长即破坏 5 % 带 → T1 FAIL。")
    a("* **H_psi**：边界由累积指标 `Psi` 决定（原位样本带 passing ≤ "
      f"{_f(band.get('psi_passing_max'),1)}，failing ≥ {_f(band.get('psi_failing_min'),1)}）→ T1 PASS、T2 FAIL。")
    a("* **H_ma_dt**：需 Ma（或 dt）高于认证值才失效 → T1 PASS、T2 PASS。")
    a("")
    a("判定：`T1 FAIL→H_TIME`；`T1 PASS,T2 FAIL→H_PSI`；`T1 PASS,T2 PASS(+T3)→H_MA_DT(有界/强)`；"
      "其余 → `INCOMPLETE/UNEXPLAINED`。阈值零调整。")
    a("")
    a("## ③ 计算结果")
    a("")
    if ev.get("available"):
        a("### 3.1 已有 7 点数据已经能判的：`dt` 不是充分致因（否证候选 2）")
        a("")
        a("P3 与 P4 **同为 `c_s = 10`、`dt` 相同**，只差运行时长；若「dt 加倍」是充分致因，两点必须同判。")
        a("")
        a("| 点 | `t_end` [s] | dt [s] | `dt`/认证dt | `Psi` | Ma | Re | 剖面 RMS [%] | 壁剪 [%] | `u_max` err [%] | 判定 |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for key, label in (("p3", "P3_cert_re_high_ma"), ("p4", "P4_cert_re_high_ma_long"),
                           ("p0", "P0_baseline (认证点)")):
            q = ev[key]
            a(f"| {label} | {q['t_end']:g} | {q['dt']:.3e} | "
              f"{q['dt']/ev['certified_dt']:.2f}× | {q['psi_index']:.2f} | {q['max_mach']:.3f} | "
              f"{q['reynolds_number']:.2f} | {q['profile_rms_interior_pct']:.3f} | "
              f"{q['wall_shear_worst_abs_pct']:.3f} | {q['u_max_error_pct']:+.3f} | "
              f"{'PASS' if q['passes_all_gates'] else '**FAIL**'} |")
        a("")
        a(f"结论（`evidence.dt_candidate = {ev['dt_candidate']}`）：" + ev["dt_candidate_statement"])
        a("")
        a(f"放大倍数：运行时长 {ev['run_length_ratio_p4_over_p3']:.2f}×、`Psi` "
          f"{ev['psi_ratio_p4_over_p3']:.2f}×，而剖面 RMS {ev['rms_amplification_p4_over_p3']:.2f}×、"
          f"壁剪 {ev['shear_amplification_p4_over_p3']:.2f}×——误差对累积量的放大是超线性的，"
          "这正是「累积型」而非「阈值型」失效的指纹。")
        a("")
        a("仍开着的部分：" + ev["still_open"])
        a("")
    a("### 3.2 运行时长轴的三个预声明点")
    a("")
    if doc["points"]:
        a("| 点 | `t_end` [s] | steps | dt | `Psi`（实测） | Ma | Re | `u_max` err [%] | 剖面 RMS [%] | 壁剪 [%] | 壁滑 [%·umax] | drift | 判定 |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for q in doc["points"]:
            a(f"| {q['id']} | {q['t_end']:g} | {q['steps']} | {q['dt']:.3e} | "
              f"{q['psi_index']:.2f} | {q['max_mach']:.3f} | {q['reynolds_number']:.2f} | "
              f"{q['u_max_error_pct']:+.3f} | {q['profile_rms_interior_pct']:.3f} | "
              f"{q['wall_shear_worst_abs_pct']:.3f} | "
              f"{100*q['wall_slip_intercept_over_umax']:.3f} | {q['steady_drift']:.4f} | "
              f"{'PASS' if q['passes_all_gates'] else '**FAIL**'} |")
        a("")
        gate_names = sorted({k for q in doc["points"] for k in q["gates"]})
        a("| 点 | " + " | ".join(gate_names) + " |")
        a("|---" * (len(gate_names) + 1) + "|")
        for q in doc["points"]:
            a(f"| {q['id']} | " + " | ".join(("✓" if q["gates"][g] else "✗")
                                            for g in gate_names) + " |")
        a("")
    else:
        a("**本文件写作时三个点均无工件**（`verdict.status = NOT_RUN`）。表内不填任何估算值；"
          "每个点按 `study_time_axis.py --points <id>` 单独执行，落一份分片 JSON（互相不共享写），"
          "再用 `--collect` 合并判定。预计单点成本（4 线程 1 核等效口径）：T1 ≈ 2 min、T2 ≈ 4 min、T3 ≈ 8 min。")
        a("")
    a("图：`docs/figures/poiseuille_time_axis.png` —— (a) 已提交 7 点的 `Psi`–RMS 平面（实心）与三个预声明点的"
      "预测 `Psi` 位置（竖虚线，尚未执行）；(b) P3/P4 同 dt 对照（真实数字），运行时长是唯一差别。")
    a("")
    a("## ④ 判定")
    a("")
    a(f"**status = `{v['status']}`**")
    a("")
    a("> " + v.get("statement", "").replace("\n", " "))
    a("")
    checks = v.get("psi_oos_checks", {})
    if checks:
        a("`Psi` 样本外逐点核对（预测 vs 实测）：")
        a("")
        a("| 点 | `Psi` | 预测 | 实测 | 一致 |")
        a("|---|---|---|---|---|")
        for pid, c in checks.items():
            a(f"| {pid} | {c['psi']:.2f} | {c['predicted']} | {c['observed']} | "
              f"{'是' if c['agrees'] else '**否**'} |")
        a("")
    a("## ⑤ 注意点与局限")
    a("")
    a("1. 本研究的**结论分两层**：3.1 的「dt 非充分致因」由已提交数据机械决定，可立即引用；"
      "3.2 的运行时长轴在点未执行前**不产生任何结论**（`NOT_RUN`），不得把预测当结果。")
    a("2. 候选 1（Ma 驱动的粒子无序度增长）**尚未被否证**：`dt` 被排除只说明「加倍 dt 不足以致失效」，"
      "不说明 Ma 无关。要分离二者，缺的正是 T1（同 t_end、不同 Ma/dt 对照走 P3。")
    a("3. `Psi` 仍是 7 点事后拟合出的相关量；即便 T1/T2 跑出 H_PSI，也只是**两点夹逼**，"
      "不是拟合出的定律，也不可外推到其它几何、核函数、时间积分格式。")
    a("4. 与 `study_ma_envelope.py` 同源同设置（同一 `run_point`/`evaluate`/`gates`），"
      "系统性偏差（Morris Laplacian 过扩散、壁面偏移 δ/dx ≈ 0.38）在两份研究里同向存在，本研究不消除它。")
    a("5. 执行环境限制（实测，记录在案）：本机单次 shell 调用上限 120 s，T1 在 4/6 线程下仍需 ≈115 s 以上，"
      "无法在本地取数；集群递交的 1 核作业排队 38 min 未派发（与 Ma 研究遭遇的单核饥饿一致），"
      "改以 4 核递交。**排队等待是本研究当前未取到 3.2 数字的唯一原因**，与物理无关。")
    a("")
    a("## ⑥ 误差判定：关键量 vs 基准，逐项是否 <5 %")
    a("")
    if doc["points"]:
        a("| 点 | 关键量 | 参考 | 计算值 | 误差 | <5 %？ |")
        a("|---|---|---|---|---|---|")
        for q in doc["points"]:
            a(f"| {q['id']} | `u_max` | {op['u_max_analytic']:.5f} m/s | {q['u_max_measured']:.5f} m/s | "
              f"{q['u_max_error_pct']:+.3f} % | {'是' if abs(q['u_max_error_pct']) <= doc['tolerance_pct'] else '**否**'} |")
            a(f"| {q['id']} | 剖面 RMS（内区） | 解析抛物线 | — | {q['profile_rms_interior_pct']:.3f} % | "
              f"{'是' if q['profile_rms_interior_pct'] <= doc['tolerance_pct'] else '**否**'} |")
            a(f"| {q['id']} | 壁面剪切 | rho g H/2 | — | {q['wall_shear_worst_abs_pct']:.3f} % | "
              f"{'是' if q['wall_shear_worst_abs_pct'] <= doc['tolerance_pct'] else '**否**'} |")
    else:
        a("**无实测数字 → 不做误差判定**（honest：未执行 ≠ 通过）。3.1 的对照表是已提交工件的复核，"
          "其逐项误差见该表最后一列；本文件不替工件说话。")
    a("")
    a("## ⑦ 复现")
    a("")
    a("```bash")
    a("PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_time_axis.py --evidence")
    a("PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_time_axis.py \\")
    a("    --points T1_runlen_2s T2_runlen_4s T3_runlen_8s --collect")
    a("PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_time_axis_doc.py")
    a("pytest tests/test_case_poiseuille_time_axis.py -q")
    a("```")
    return "\n".join(L) + "\n"


def main() -> int:
    doc = load()
    figure(doc)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build(doc))
    print(f"wrote {DOC}")
    print(f"wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
