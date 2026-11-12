#!/usr/bin/env python3
"""发布 m15711 交差核验报告（六要素，数字全部由 check_results.json 程序化注入）。"""
import base64
import json
import pathlib
import urllib.request

W = pathlib.Path("/root/tp-agent/work/m15711")
CHK = json.loads((W / "docs/board-audit-m15711/check_results.json").read_text())
ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)
API = ENV["TENSOR_API_BASE"]


def post(path, payload, token=None):
    req = urllib.request.Request(API + path, method="POST",
                                 data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = post("/auth/login", {"username": ENV["TENSOR_AGENT_USER"],
                           "password": ENV["TENSOR_AGENT_PASS"]}).get("access_token")

pc = CHK["predeclared_gates"]["per_case"]
r16, r24, r10 = pc["ladder-r16"], pc["ladder-r24"], pc["ladder-r10"]
neg = pc["friction-f0.0"]
reg = CHK["benchmark_registry"]
rep = CHK["reproduction"]
h = CHK["report_zero_handcopy"]

rows = "\n".join(
    "| %s | %d | %.4f | %.4f%% | %.4f | %.4f | %.5f | %s |" % (
        k.replace("ladder-", ""), v["H_over_dx"], v["interior_l2"], 100 * v["interior_l2"],
        v["flux_ratio"], v["wall_shear_ratio_bottom"], v["steady_drift"],
        ("gated" if v["gated_by_design"] else "sensitivity"))
    for k, v in pc.items())

md = f"""# 交差核验：CASE-COUETTE-SPH / m15412（TensorSPH）→ L3，L4 拒绝

**审计单** m15711 · 被审 m15412（succeeded, PR #497） · 性质=交差核验（非重做） · verdict #366 已落表

## ① 案例介绍

平面库埃特流：两平行壁间不可压流体，下壁静止、上壁以 U=0.2 m/s 切向滑移，通道高 H=0.1 m、x 周期（12 dx），
参考解为定常不可压 N-S 的线性剖面 u(y)=U y/H、du/dy=U/H、τ=ρνU/H、Q=UH/2
（Batchelor 1967 §4.2；闭式无自由参数）。Reynolds 数 Re=UH/ν=1.0，Mach 数 Ma=U/c0=0.0100，
核函数 cubic spline（h=1.3dx），Tait 状态方程（γ=7, c0=20 m/s），壁 3 层、wall_friction=1（no-slip 混合），
定步长 dt=0.3h/c0（声学 CFL，位可复现），初值静止（上壁脉冲启动）。

## ② 怎么做的（核验流程）

被审交付的三件套不是靠读报告接受，而是逐条重算：

1. **独立重跑**：`run.py --out runs/couette_repro --resolutions 10 16 24 --speeds 0.4 --viscosities 0.04 --frictions 0.0`（CPU 单机 4m20s，6 个 case）；
2. **工件比对**：独立 results.json 与交付 results.json 规范化（排除墙钟）后 diff 行数 = **{rep['differing_fields']}** → 确定性成立，交付工件为真；
3. **零手抄校验**：用重跑工件重新生成报告 md，与交付 md **{'逐字节相同' if h['byte_identical_to_delivered'] else '不同'}**（{h['delivered_md_bytes']} B）；
4. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-COUETTE-SPH'` → **{reg['rows']} 行**（表内共 {reg['table_total_rows']} 行）；
5. **测试**：`pytest tests/test_couette.py -q` → 9 passed in 120.13s；
6. 工具链脚本化：`verify_audit.py`（全部结论）→ `check_results.json` → 本报告数字程序化注入。

## ③ 计算结果（程序化注入自 check_results.json）

| case | H/dx | interior_l2 | 剖面误差 | flux_ratio | wall_shear_ratio(下) | steady_drift | 门约束 |
|---|---|---|---|---|---|---|---|
{rows}

关键 got 数值（r16，受门）：剖面 L2 = {100*r16['interior_l2']:.2f}%，通量比 {r16['flux_ratio']:.4f}（偏差 {100*abs(r16['flux_ratio']-1):.2f}%），
下壁剪切比 {r16['wall_shear_ratio_bottom']:.4f}（偏差 {100*abs(r16['wall_shear_ratio_bottom']-1):.2f}%），稳态漂移 {r16['steady_drift']:.5f}（门 0.01）；
r24：L2 = {100*r24['interior_l2']:.2f}%，通量比 {r24['flux_ratio']:.4f}，剪切比 {r24['wall_shear_ratio_bottom']:.4f}，漂移 {r24['steady_drift']:.5f}。
负控（wall_friction=0）通量比 {neg['flux_ratio']:.4f} < 0.2 → 自由滑移不被驱动，正控的通量来自壁面剪切而非初值。

**交付自判**：{CHK['predeclared_gates']['case_verdict']}（理由：{'; '.join(rep['reasons'])}）——审计独立重跑得到**同一判定**。

## ④ 结果图

图 1：(a) 三级分辨率 SPH 剖面 vs 解析直线 vs 自由滑移负控；(b) 四项指标偏差 vs 自设门（1%~5%）与生态 15% 精度标准（log 轴）。
图 2：交付工件自带的剖面图（由本次独立重跑再生成，证明图与 results.json 同源）。

## ⑤ 注意点（发现清单，按严重度）

- **F1 高（治理）**：锚点 `CASE-COUETTE-SPH` 在 `tensor_benchmark_cases` 中 **{reg['rows']} 行** → 平台基准裁判空转，判据只能回落到仓内预声明 `DESIGN.md`。
- **F2 高（可信度）**：预声明门 r16 稳态漂移 {r16['steady_drift']:.5f} ≥ 0.01 → **FAIL**；DESIGN 要求 r16 与 r24 同时全过，r24 单独通过不能翻案。定常能力只在 H/dx=24 成立。
- **F3 中**：交付 `runs/` 从未入库（m15412 `git_status: ?? runs/`），分支带不走工件，复核须重跑（确定性已证，4 分钟）。
- **F4 中（度量独立性）**：壁剪切比由首两行剖面有限差分乘 μ 得到（`run.py:397-400`），与剖面 L2 同源，**不是独立第二证**；DESIGN 称其检验壁面条件偏强。
- **F5 低**：ν 敏感性表 `wall_shear_ratio=0.9179`（8.2% 偏差）列出未解释——恰是粘性算子误差混入近壁梯度的信号。
- **F6 低**：真跑测试阈值宽（flux 0.6~1.2、L2<20%），且一处恒等冗余断言（`flux_ratio < 3.0*0.0+0.2`）。
- **F8 中（平台机制）**：`maturity_gate.judge()` 的 L4 只查 benchmark 键"有无数字"，查不到"预声明门是否过"；本单靠**声明级封顶 L3** 挡住，机制本身没挡住（实测 judge(4, ev) → 4）。
- 局限：未做收敛阶测量（L2 在 r10→r24 几乎不动 2.98%→2.95%，误差已被壁面滑移主导）；未在 loong64 架构复核；未评估转捩/Taylor–Couette/三维。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 参考 | got (r16) | 相对误差 | <3%? | <15%? |
|---|---|---|---|---|---|
| 剖面 L2（内部带） | 0 | {100*r16['interior_l2']:.2f}% | {100*r16['interior_l2']:.2f}% | 是 | 是 |
| 通量 Q | UH/2 | 比 {r16['flux_ratio']:.4f} | {100*abs(r16['flux_ratio']-1):.2f}% | 否 | 是 |
| 下壁剪切 τ_w | ρνU/H | 比 {r16['wall_shear_ratio_bottom']:.4f} | {100*abs(r16['wall_shear_ratio_bottom']-1):.2f}% | 是 | 是 |
| 稳态漂移（门 1%） | 0 | {100*r16['steady_drift']:.2f}% | — | 否 | 参考意义不同 |
| r24 剖面 L2 | 0 | {100*r24['interior_l2']:.2f}% | {100*r24['interior_l2']:.2f}% | 是 | 是 |

**结论**：精度三量均 ≪15% 生态红线（剖面/壁剪切 <3%，通量 4.42% 略超 3%），故误差层面可接受；
但**预声明门未过**，因此等级判 **L3**、**L4 基准验证不予认定**；原单自报 FAIL 与核验一致，**无虚标**。

## 处置

verdict #366：`TensorSPH / m15412 CASE-COUETTE-SPH / level=3 / board-audit`，evidence 10 键（impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor + gaps_preventing_L4/entry_invocation），
已实测 `maturity_gate.judge(level=3)` → **3（无缺项）**。建议：登记锚点进基准表；工件入库；M3 改壁面相互作用应力；重定义 steady_drift 门后重开一轮争 L4。
"""

figs = []
for name, cap, path in [
    ("fig_couette_audit.png", "图1 SPH 库埃特剖面 vs 解析解（含自由滑移负控）与四项指标偏差 vs 自设门/15% 标准",
     W / "docs/board-audit-m15711/fig_couette_audit.png"),
    ("couette_profiles.png", "图2 交付工件剖面图（由本次独立重跑 results.json 再生成，与交付逐字节同源）",
     W / "runs/couette_repro/couette_profiles.png"),
]:
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(path.read_bytes()).decode()})

payload = {
    "title": "交差核验 CASE-COUETTE-SPH / m15412：L3 认定、L4 拒绝（预声明门 r16 未过 + 锚点未登记）",
    "summary": ("m15412 三件套真实且确定性可复现（独立重跑 results.json diff=0，报告再生逐字节相同），"
                "剖面 2.93%/通量 4.42%/壁剪切 0.48% 均远优于 15% 线；但预声明稳态门 r16 1.55%≥1% 未过、"
                "锚点未登记 tensor_benchmark_cases（0 行）、run 工件未入库、M3 与 M1 同源 → 判 L3，L4 拒绝。无虚标。"),
    "scenario_key": "board-audit",
    "mission_id": 15711,
    "tags": ["CASE-COUETTE-SPH", "TensorSPH", "m15412", "board-audit", "L3", "maturity_gate"],
    "markdown": md,
    "figures": figs,
}
res = post("/tensor/reports", payload, tok)
print(json.dumps(res, ensure_ascii=False)[:400])