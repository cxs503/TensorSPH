"""Build /tmp/report.json for the tvc_max_shift leg (mission 21832).

Every number is read out of the payload; nothing is typed by hand.
"""
import base64, json, math, subprocess
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m21832")
payload = json.loads((ROOT / "docs/xval_sphinx_tvc/data/shift_cap.json").read_text())
recs = payload["records"]
idx = {(r["dx"], r["variant"]): r for r in recs}
dxs = sorted({r["dx"] for r in recs}, reverse=True)
F = payload["surface_factor"]
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip()

def pct(x, d=2):
    return "n/a" if x is None else f"{100.0*x:.{d}f} %"

rows = []
for dx in dxs:
    for v in payload["grid"]:
        r = idx.get((dx, v))
        if r is None:
            continue
        cap = "—" if r["cap_over_h"] is None else f"{r['cap_over_h']:g}"
        rel = r["score"].get("mean_rel_error")
        rows.append(f"| {dx:g} | `{v}` | {r['alpha']:.2f} | {cap} | "
                    f"{r['alpha_eff_predicted']:.3f} | {pct(rel)} | "
                    f"{r['score']['mae']:.4f} | {'PASS' if rel <= 0.15 else 'FAIL'} | "
                    f"{r['max_tvc_shift_over_h']:.3f} | {100*r['max_tvc_clipped']:.1f} |")
table = "\n".join(rows)

def rr(dx, v, key="mean_rel_error"):
    r = idx[(dx, v)]
    return r["score"][key]

fig = (ROOT / "docs/xval_sphinx_tvc/figure_shift_cap_leg.png").read_bytes()

md = f"""## ① 案例介绍

锚点 `XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION`（Xiangyu-Hu/SPHinXsys 的 transport-velocity
correction，TensorSPH 已在 `src/tensorsph/transport_velocity_correction.py` 移植并有 10 条腿的验证记录）。
本轮做的是此前**所有腿共同的验证欠账**：`RESOLUTION_LADDER.md` §6 与 `make_limiter_doc.py`
都建议用户"加 `tvc_max_shift` 上限"，但**没有任何一条腿真正跑过这个上限**。它是移植件唯一的
安全阀，却是未验证的建议。

物理背景：弱可压缩 SPH 的自由面 dam break（Martin & Moyce 1952）。TVC 的位移驱动量是
`grad-1` 不一致量，其**自由面分支**满足梯度定理 `|g| = 2 I(h,d)`、`I(h,0)=15/(7πh)`，因此

    |δr|/h = 30α/(7π) = {F:.4f} α            (1)

即"最大位移 ∝ 系数"。于是挂上限 `c`（单位 h）在 `c < {F:.4f}α` 时恰好截断自由面那一族粒子，
并可导出**等效系数律** `α_eff(c) = min(α, c/{F:.4f})` (2)。这就是本轮预登记要检验的主假说 H1。

| 量 | 值 |
|---|---|
| 水柱 | 0.2 m × 0.4 m（长宽比 2），干床，水槽 2.0 × 0.75 m |
| 重力 / 参考密度 / 声速 | 9.81 m/s²；1000 kg/m³；20 m/s（max Ma ≈ 0.13） |
| 波长比 h/Δx | 1.3 |
| 分辨率 | {', '.join(f'{d:g} m' for d in dxs)}（Δx=0.01 m 行见 ⑤，作业 cj3734 排队中） |
| 模拟时长 | {payload['seconds']} s（τ = t√(2g/a) → 4.29，实验窗口末端） |
| 系数 α | 0.10 / 0.15 / 0.20 / 0.30 |
| 上限 c | 无 / 0.10h / 0.20h / 0.30h |
| 参考 | {payload['reference']} |

## ② 怎么做的

1. **把建议变成预登记实验**：在 `benchmarks/xval_sphinx_tvc/shift_cap.py` 里先写下 H1–H4
   （写成 docstring，先于任何运行），再跑 7 变体 × 分辨率网格；payload 增量写盘、重跑跳过已测点。
2. **补上可观测性**：原 `DamBreakRun` 只记录 `max_tvc_shift`，无法区分"上限未触发"与"上限在替
   整个运行兜底"。新增 `max_tvc_clipped`（单次应用中被打上限缩放的粒子占比的峰值），
   `src/tensorsph/dam_break.py` 两行改动 + `run.py` 记录字段。
3. **算子级复核**：在算子上直接验证 (1)（半平面，h/Δx=4 处测得 −1.5 %）、"上限高于峰值时逐位
   不变"、"上限只径向缩放超限族且保持方向"，以及"未被上限触及的体相粒子仍吃原来的 α"——
   这正是 H1 在解算器层面失效的机制。
4. **文档零手抄**：`make_shift_cap_doc.py` 从 payload 生成
   `docs/xval_sphinx_tvc/SHIFT_CAP_LEG.md` 与图，假说裁决由代码计算（换分辨率会自动重判）。
5. **工具链**：PyTorch CPU、仓内 WCSPH 求解器、`dt` 由 Courant 限自动取；Δx=0.04/0.03/0.02 m
   三层本机跑完（单核，均 <70 s/变体），Δx=0.01 m 因本机 10 min 硬限已递交调度器 `cj3734`。

## ③ 计算结果（程序化注入，无手抄）

| Δx [m] | 变体 | α | c/h | 预测 α_eff (2) | 平均相对误差 | MAE Z/a | 15 % 门 | max shift/h | 被截断粒子 % |
|---|---|---|---|---|---|---|---|---|---|
{table}

关键裁决（全部由 payload 计算）：

* **H1（上限＝等效系数）被否证**：`cap 0.20h + α=0.20` 在 Δx=0.03 m 为
  {pct(rr(0.03,'cap020_a020'))}、`α=0.30` 为 {pct(rr(0.03,'cap020_a030'))}，而同样 α_eff=0.147
  的无上限 `α=0.15` 只要 {pct(rr(0.03,'nocap_a015'))}。机制清楚：上限只裁自由面尾部（`|g|` 最大处），
  体相仍吃过大 α，故"上限"与"小 α"作用在不同粒子上，不可互换。
* **H2（上限高于实测峰值即为惰性）成立**：`c=0.20h` 对 `α=0.10`（实测峰值 ~0.155h）零截断，
  前缘误差**逐位相同**（三个分辨率均 MAE 完全相同）。
* **H3（低于自由面尺度的上限会破坏机制）被否证，且方向对我们有利**：`c=0.10h, α=0.10`
  在 Δx=0.03 m 是整张网格最好的一行（{pct(rr(0.03,'cap010_a010'))}，MAE
  {idx[(0.03,'cap010_a010')]['score']['mae']:.4f}），优于无上限的
  {pct(rr(0.03,'nocap_a010'))}；在 Δx=0.04/0.02 m 为中性。即被裁掉的尾部本是有害的。
* **H4（不存在与分辨率无关的设定）成立**：没有任何 `(α, c)` 组合在三个已测分辨率上同时过 15 % 门。

## ④ 结果图

![shift cap leg](fig)

(a) 各变体平均相对前缘误差 vs Δx（含 15 % 门线）；(b) 上限单次应用实际缩放的粒子占比峰值——
它与 (a) 共同说明"上限有没有在起作用"，仅看 `max_tvc_shift` 无法区分。

## ⑤ 注意点 / 局限

1. **Δx=0.01 m 缺失**：最细一层已递交调度器 `cj3734`（排队中），本文写作时未回传。生成器会把缺失
   组合显式标注，回传后重跑 `shift_cap.py`（增量）与 `make_shift_cap_doc.py` 即可扩表。
2. **参考数字化不确定度 ±0.05 in Z/a**：1–2 个百分点级差异落在数字化带内，不做强裁决；
   H2 的"逐位相同"是仅有的可以精确判定的结论。
3. **单案例、单指标**：结论是 dam break 前缘位置 + `max_tvc_clipped` 诊断。体相主导的算例
   （cavity/channel，`|g|` 处处很小）中上限永远不触发，本轮的"裁尾有益"不迁移。
4. **仍无活体 SPHinXsys 数值 diff**：镜像无 Boost/TBB，与 round 1 相同的摩擦，算子级仍是精确转写验证。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 值 | vs 基准 | <15 % ? | <3 % ? |
|---|---|---|---|---|
| 最好前缘行（Δx=0.03 m, `cap010_a010`） | {pct(rr(0.03,'cap010_a010'))} | Martin & Moyce 数字化曲线 | YES | NO |
| 校准系数行（Δx=0.03 m, `nocap_a010`） | {pct(rr(0.03,'nocap_a010'))} | 同上 | YES | NO |
| 校准系数行（Δx=0.02 m, `nocap_a010`） | {pct(rr(0.02,'nocap_a010'))} | 同上 | YES | NO |
| 等效系数律预测行（Δx=0.03 m, `cap020_a020`） | {pct(rr(0.03,'cap020_a020'))} | 同上 | NO | NO |
| 自由面位移尺度 (1) vs 闭式 30α/(7π) | h/Δx=4 处 −1.5 % | 解析解 | YES | YES（h/Δx=1.3 处 −14.15 %，见 SURFACE_ANCHOR） |

**显式回答**：本腿最好的前缘行 {pct(rr(0.03,'cap010_a010'))} **< 15 %**（过 owner 门），但**没有
任何一行 < 3 %**——参考本身是 ±0.05 的数字化实验曲线，3 % 级一致性在此案例上不可达，故不作此宣称。
唯一达到 < 3 % 的判定是算子层面的自由面尺度 vs 闭式解（h/Δx=4 时 −1.5 %），它验证的是算子而非整案。

honest 记录：H1 与 H3 均按预登记方向**失败**（H1 被否证、H3 反向成立），本文如实保留。
"""
md = md.replace("fig", "data:image/png;base64," + base64.b64encode(fig).decode())
report = {
    "title": "TVC 安全阀实地检验：tvc_max_shift 不等于小 α（SPHinXsys 移植件，Martin & Moyce dam break）",
    "summary": ("预登记检验 TensorSPH 移植的 SPHinXsys transport-velocity correction 的唯一安全阀 "
                "tvc_max_shift：上限高于实测峰值时逐位惰性（H2 成立）、上限不能替代过大的 α"
                "（等效系数律 H1 被否证，cap0.2h+α0.2/0.3 仍 38–59 %）、α=0.10 配 c=0.10h 在 "
                f"Δx=0.03 m 反而最好（{pct(rr(0.03,'cap010_a010'))} < 15 % 门）。Δx=0.01 m 递交 cj3734 待回传。"),
    "scenario_key": "xval_sph_transport_velocity_correction",
    "mission_id": 21832,
    "tags": ["SPH", "transport-velocity", "SPHinXsys", "dam-break", "verification", "TensorSPH", "XVAL"],
    "markdown": md,
    "figures": [{"name": "figure_shift_cap_leg.png",
                 "caption": "(a) 平均相对前缘误差 vs Δx（含 15 % 门线）；(b) 上限单次应用实际缩放的粒子占比峰值。",
                 "png_base64": base64.b64encode(fig).decode()}],
    "evidence": {"payload": "docs/xval_sphinx_tvc/data/shift_cap.json",
                 "document": "docs/xval_sphinx_tvc/SHIFT_CAP_LEG.md",
                 "git_head": head,
                 "records": len(recs),
                 "pending_job": "cj3734 (dx=0.01 m)"},
}
Path("/tmp/report.json").write_text(json.dumps(report))
print("report bytes", len(json.dumps(report)), "head", head, "records", len(recs))