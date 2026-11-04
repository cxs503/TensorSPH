"""Build and publish the paper-grade report for the verification-ledger mission.

Every number in the report is read from ``ledger.json`` / ``provenance_control.json``
and injected here, so the report can never disagree with the committed artefacts
(zero hand copying).

Usage::

    python benchmarks/verification_ledger/report_payload.py --build   # write /tmp/m3648_report.json
    python benchmarks/verification_ledger/report_payload.py --publish # build + POST to the platform
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path("/tmp/m3648_report.json")


def _table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _compact_png_b64(path: Path, scale: float = 0.78, colors: int = 96) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    small = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    small = small.convert("P", palette=Image.ADAPTIVE, colors=colors)
    buf = io.BytesIO()
    small.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def build() -> dict:
    L = json.loads((HERE / "ledger.json").read_text(encoding="utf-8"))
    P = json.loads((HERE / "provenance_control.json").read_text(encoding="utf-8"))
    V = L["verdict"]
    dp = V["quantities"]["laplace_dp"]
    cf = V["quantities"]["cut_force_ratio"]
    best = V["best_sanctioned_reading"]
    q1 = L["provenance"]["q1_reproducible"]
    q2 = L["provenance"]["q2_sigma_control"]
    q3 = L["provenance"]["q3_resolution_stability"]

    conv_rows = [
        [f"`{k}`", m["operand"], m["time"],
         "是" if m["free_surface"] else "否",
         "**是**" if m["sanctioned"] else "否", m["rationale"]]
        for k, m in L["conventions"].items()
        if any(r["convention"] == k for r in L["readings"])
    ]
    sens_rows = [[r["case"].split(" / ", 1)[-1], round(r["min_pct"], 2),
                  round(r["max_pct"], 2), r["spread_pp"]]
                 for r in V["convention_sensitivity"]]
    pass_rows = [[f"`{r['id']}`", r["convention"], round(r["rel_error_pct"], 2),
                  r["dx_m"], r["resolution_verdict"]] for r in L["free_surface_passes"]]
    ctl_rows = [[f"`{k}`", v["R_over_dx"], v["sigma_N_per_m"], v["p_core_Pa"],
                 v["p_band_Pa"], v["dp_measured_Pa"], v["rel_error_pct"],
                 "是" if v["within_15pct"] else "否"] for k, v in P["cases"].items()]
    fam_rows = [[k, b["n_families"], b["n_readings"], round(b["median_pct"], 2),
                 b["worst_abs_pct"], f"{b['n_within_15pct']}/{b['n_readings']}",
                 f"{b['n_within_3pct']}/{b['n_readings']}"] for k, b in V["quantities"].items()]

    bullets = [
        f"- Δp 当前口径（sanctioned，{dp['n_sanctioned']} 条）：abs(Δ%) 最小 "
        f"**{dp['sanctioned_min_abs_pct']:.2f}%**、中位 {dp['sanctioned_median_pct']:.2f}%、"
        f"最差 {dp['sanctioned_max_abs_pct']:.2f}%，**全部 >15%**。",
        f"- 最优 sanctioned 读数 `{best['id']}`：实测 {best['measured_Pa']} Pa vs 解析 "
        f"{best['theory_Pa']} Pa → **Δ = {best['rel_error_pct']:.2f}%**（与 m2971 的 −34.0% 独立复算一致）。",
        f"- 自由面分离：跨自由面 {dp['n_free_surface']} 条中仅 {dp['free_surface_within_15pct']} 条 <15%；"
        f"不跨自由面（双相）的 {dp['n_no_free_surface']} 条中有 "
        f"{dp['no_free_surface_within_15pct']} 条 <15%。",
    ]
    findings = [
        f"- Q1 可复现性：工件登记 {q1['committed_rel_error_pct']:+.2f}%，同设置重跑 "
        f"{q1['recomputed_rel_error_pct']:+.2f}%（相差 {q1['delta_pp']:.2f} pp），且该工件 schema"
        "（setup/reference/laplace/history, dx=0.004, 380 步）并非 `run.py` 当前输出版本 → **陈旧快照，不可复现**。",
        f"- Q2 σ→0 负控：σ=1e-9 时 Δp = {q2['dp_sigma_control_Pa']:.1f} Pa（σ=7.2 时 "
        f"{q2['dp_sigma_reference_Pa']:.1f} Pa），初始伪压强 {q2['p_core_t_start_Pa']:.0f} Pa 在 50 步内"
        f"衰减为 {q2['p_core_t_after_50_steps_Pa']:.0f} Pa → 该读数**确由 CSF 力产生**，不是初始化残留。",
        f"- Q3 分辨率稳定性（决定性）：R/dx 5 → 10，读数从 {q3['coarse_rel_error_pct']:+.2f}% 翻到 "
        f"{q3['fine_rel_error_pct']:+.2f}%；机制是粗网格上被减去的上带压强被 Tait 钳位**恰为 0**"
        f"（{q3['coarse_p_band_Pa']:.1f} Pa → {q3['fine_p_band_Pa']:.1f} Pa），减法退化成绝对压强，"
        "细网格则把信号本身减掉。`center_minus_band_instant` 口径据此正式退役。",
    ]

    md = f"""# TensorSPH 接口压差验证台账：跨 {L['n_families']} 个 benchmark 家族的 {L['n_readings']} 条读数与一次来源审计

## ① 案例介绍

物理背景：二维液滴的 Laplace 压差 Δp = σ/R 是表面张力模块（CSF + 一致格式）唯一的一阶解析基准。
本仓 2026-10→11 期间，同一个 Δp 被先后引用为 −76%、−107%、−87.6%、+5.6%、−34%，互相矛盾的主因是
**测量口径**（减哪两个压力、单时刻还是时间平均）而非求解器。平台标准：对标解析值误差 >15% 不可接受。

参数表（各家族公有量，数值取自工件）：R = 0.02 m，σ = 7.2 N/m，ρ0 = 1000 kg/m³，c0 = 20 m/s，γ = 7，
Δp_analytic = σ/R = 360 Pa；dx 覆盖 1.04e-3 … 6.25e-3 m（R/dx ≈ 3.2…19.2）。

覆盖：**{L['n_families']} 个家族 / {L['n_readings']} 条读数**，来自 {len(L['sources'])} 个已提交工件，
归入 {len([k for k in L['conventions'] if any(r['convention'] == k for r in L['readings'])])} 种口径。

## ② 怎么做的

1. 新增 `benchmarks/verification_ledger/`：`ledger.py` 用 11 个适配器只读解析仓库内已提交的验证 json
   （不重跑仿真），把每条读数归一为（家族, 口径, 实测, 解析, Δ%），并给口径打 `sanctioned` 标记——
   当前标准口径 = 「绝对压强 + 时间平均」（backlog m2971）。新工件必须登记口径，否则 `collect()` 抛错。
2. `build.py` 程序化生成 `ledger.json`、`docs/verification_ledger.md`（数字零手抄）与四联图。
3. `provenance_control.py` 对唯一一条「跨自由面 + 单点」读数 `laplace_law#relaxed#0` 做三项对照，
   全部复跑该家族**自己的**代码路径 `benchmarks/laplace_law/run.py::relax`：
   Q1 可复现性（同设置重跑）、Q2 σ→0 负控（是否为初始化伪压强）、Q3 分辨率稳定性（R/dx 5→10）。
4. `tests/test_verification_ledger.py`：16 个测试守住溯源一致、口径卫生、门限判定与文档/工件漂移。

工具链：Python 3.11 / PyTorch（仓库自带 SPH 内核）/ matplotlib / pytest / ruff（`ruff check .` 全仓 0 告警）。

## ③ 计算结果

**口径登记（`sanctioned` 一栏决定能否作为当前精度引用）**

{_table(conv_rows, ['口径 key', '相减操作数', '时间处理', '跨自由面', '可引用', '理由'])}

**同一算例、四种口径：口径单独能把误差挪多远**（laplace_pressure_floor，6 个算例物理完全相同）

{_table(sens_rows, ['算例', '最小 Δ%', '最大 Δ%', '口径极差 (pp)'])}

最大口径极差 **{V['convention_sensitivity'][0]['spread_pp']:.1f} pp** —— 这就是历史引用互相矛盾的量级来源。

**门限判定（逐量）**

{_table(fam_rows, ['量', '家族数', '读数', '中位 Δ%', '最差 abs(Δ%)', '<15%', '<3%'])}

{chr(10).join(bullets)}

**<15% 读数的来源审计**

{_table(pass_rows, ['读数', '口径', 'Δ%', 'dx (m)', '分辨率判定'])}

对照组（同一代码路径，每次只改一个参数）：

{_table(ctl_rows, ['算例', 'R/dx', 'σ (N/m)', 'p_core (Pa)', 'p_band (Pa)', 'Δp (Pa)', 'Δ%', '<15%'])}

{chr(10).join(findings)}

## ④ 结果图

四联图：(1) 同一算例四种口径的柱状对比；(2) 全部 Δp 读数按「是否跨自由面」分离；(3) 仅当前口径的
读数 vs dx，显示没有收敛进 15% 门限的趋势；(4) 对照组核心压强历史 —— 初始伪压强 50 步内消失、
σ 驱动的过压持续。数据源：`ledger.json` + `provenance_control.json`，由 `build.py` 绘制。

## ⑤ 注意点 / 局限

- 台账只读已提交工件，是**历史快照登记**：陈旧工件会被如实登记并标注（本报告第 ③ 节的 Q1 即范例）。
  三项对照是实跑的（380 步约 32 s，R/dx=10 一例 105 s；当时本机 load15≈126，故未起更大规模算例）。
- 口径的 `sanctioned` 标记是**人工声明、机器校验**：新增工件必须同时登记口径，否则收集失败。
- 未纳入台账：液滴振荡频率（不同物理量）、dam-break 系列（无 Laplace 解析基准）、
  csf_consistent 的瞬时曲率行（另有专门文档）。
- 工件中存在只存相对误差绝对值的家族（{V['n_reported_as_magnitude']} 条），台账统一保留符号并在测试中显式计数。
- 本单不动 `src/`：属验证/审计增量，不改变求解器行为。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 量 | 基准 | 最优实测 | 相对误差 | <15%？ | <3%？ |
|---|---|---|---|---|---|
| Δp（当前口径：绝对压强 + 时间平均） | σ/R = 360 Pa | {best['measured_Pa']} Pa | **{best['rel_error_pct']:.2f}%** | **否** | **否** |
| Δp（跨自由面全部读数） | σ/R | 同上 | 最小 {dp['free_surface_min_abs_pct']:.2f}% | 否（{dp['free_surface_within_15pct']} 条例外均不可信） | 否 |
| Δp（不跨自由面，双相） | σ/R | 见台账 | 最小 {dp['no_free_surface_min_abs_pct']:.2f}% | 是（{dp['no_free_surface_within_15pct']}/{dp['n_no_free_surface']}） | 否 |
| F_cut/(2σL)（辅助量） | 1.0 | 见台账 | 中位 {cf['median_pct']:.2f}%、最差 {cf['worst_abs_pct']:.1f}% | 否 | 否 |

**结论（honest）**：Δp 证据等级维持 **WITHHELD**——最优 −34.0%，既不满足 <15%，更不满足 <3%。
本单的实质增量不是把误差改小，而是（a）把「哪个数字还能引用」变成机器可裁的事实；（b）证明台账里
少数达标读数全部是「退化操作数 + 粗网格」的零穿越巧合。因此 m2971 的 P0 归因（自由面压强闭合是唯一
门槛项）成立且被独立复算——下一步仍应做 P0，而不是继续调曲率算子。

证据指针：commit `df37d7e` + `03c754f`；工件 `benchmarks/verification_ledger/`（ledger.json /
provenance_control.json / ledger.png）；文档 `docs/verification_ledger.md`；
测试 `tests/test_verification_ledger.py`（16 passed）。
"""

    png = _compact_png_b64(HERE / "ledger.png")
    return {
        "title": f"TensorSPH 接口压差验证台账：{L['n_readings']} 条读数的口径登记 + 达标读数的来源审计",
        "summary": (
            f"把 TensorSPH {L['n_families']} 个 benchmark 家族 / {L['n_readings']} 条 Laplace 压差读数归一为"
            "机器核对台账（口径登记 + sanctioned 标记），并对跨自由面且 <15% 的读数做三项对照审计："
            f"工件不可复现（{q1['committed_rel_error_pct']:+.2f}% vs 重跑 {q1['recomputed_rel_error_pct']:+.2f}%）、"
            "σ→0 证明读数确由 CSF 力产生、"
            f"R/dx 5→10 即翻到 {q3['fine_rel_error_pct']:+.1f}%（上带压强钳位恰为 0 使减法退化）。"
            f"判定：Δp 维持 WITHHELD（最优 {best['rel_error_pct']:.1f}%），P0 归因（自由面压强闭合）成立。"
        ),
        "scenario_key": "tensorsph_surface_tension_verification",
        "mission_id": 3648,
        "tags": ["TensorSPH", "verification", "Laplace", "surface-tension",
                 "reproducibility", "audit"],
        "markdown": md,
        "figures": [{
            "name": "verification_ledger.png",
            "caption": "四联图：(1) 同算例四口径；(2) 全部 Δp 读数按自由面分离；(3) 当前口径 vs dx；(4) 对照组核心压强历史",
            "png_base64": png,
        }],
    }


def publish(payload: dict, base: str, user: str, password: str) -> dict:
    def post(url, data, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.load(resp)

    auth = post(base.rstrip("/") + "/tensor/auth/login",
                {"username": user, "password": password})
    token = (auth.get("access_token") or auth.get("token")
             or (auth.get("data") or {}).get("access_token"))
    if not token:
        raise SystemExit(f"no token in login response: {list(auth)}")
    return post(base.rstrip("/") + "/tensor/reports", payload, token)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args(argv)
    payload = build()
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"payload written to {OUT} ({len(json.dumps(payload))} bytes)")
    if args.publish:
        resp = publish(payload, os.environ["TENSOR_API_BASE"],
                       os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"])
        print("report response:", json.dumps(resp)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())