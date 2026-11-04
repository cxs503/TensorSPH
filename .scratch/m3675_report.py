import base64, json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def req(method, path, body=None, token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data, headers=h, method=method)
    with urllib.request.urlopen(r) as resp:
        return json.load(resp)

tok = req("POST", "/auth/login", {"username": user, "password": pw})["access_token"]

lad = json.load(open("benchmarks/dam_break/resolution_ladder.json"))
aud = json.load(open("benchmarks/dam_break/reference_audit.json"))
g = lad["gates"]
t = aud["table_audit"]
res = lad["results"]
pct = lambda x: f"{100*x:.1f}%"

ladder_rows = "\n".join(
    f"| {r['dx']} | {r['particles']} | {pct(r['ritter_mean_relative_error'])} "
    f"| {pct(r['experimental_error']['mean_absolute_error'])} "
    f"| {pct(r['mach_number'])} | {pct(r['max_density_compression'])} |"
    for r in res)

md = f"""# TensorSPH dam-break (B2) 验证状态总览：结清 m239「dam break 未复核」欠账

## ① 案例介绍
重力驱动有限液柱在干刚性底面溃决（2-D，B2 基准）。几何：a=0.2 m，H=0.4 m（H/a=2），
池宽 2.0 m，c0=25 m/s，wall_friction=0，adaptive_dt=False。
参考基准：Ritter 浅水解析解（诊断用，H/a=2 时有限水库系统偏差属物理）；
Martin & Moyce (1952) 实验表（20 点，Z/a vs tau）。

## ② 怎么做的
本单为**验证欠账整理**（不跑新仿真，load15≈127 容量纪律）：
- 新增 `benchmarks/dam_break/make_status_doc.py`：从 `resolution_ladder.json` +
  `reference_audit.json` 程序化注入全部数字（零手抄）生成
  `docs/verification_dam_break.md` 与状态图；
- 新增 `tests/test_dam_break_status.py`（7 测试，0.13 s，无仿真）钉死：
  M&M 表 WITHHELD 判定、实验门 FAIL 但参考不可用、稳定性门全 PASS、
  Ritter 无门（诊断）、细化单调、文档↔JSON 同步。

## ③ 计算结果（自 resolution_ladder.json 注入）
| dx (m) | N | Ritter 相对误差 | M&M MAE | Mach | 密度压缩 |
|---|---|---|---|---|---|
{ladder_rows}

门（最细 dx）：实验 MAE {pct(g['experimental_comparison']['measured'])} vs 限 {pct(g['experimental_comparison']['limit'])} → **FAIL**；
Ritter {pct(g['ritter_comparison']['measured'])}（无门，诊断）；
质量漂移 {g['mass_conservation']['measured']}、逃逸 {g['bounded_flow']['measured']}、
Mach {pct(g['mach_number']['measured'])}/0.25、压缩 {pct(g['density_compression']['measured'])}/0.1 → 全 **PASS**。

## ④ 结果图
`dam_break_status.png`：误差-分辨率曲线（Ritter 与 M&M MAE 随 dx 单调下降，
15% 门线）。无新场量云图：本单不跑仿真，图由既有 JSON 工件重绘（理由在此写明）。

## ⑤ 注意点
1. **M&M 表来源不可溯（阻断项）**：reference_audit 判定「synthetic-looking」——
   tau 为精确等差、纵坐标全为 0.05 的整数倍（二阶差 RMS {t['second_difference_rms']:.5f}
   vs 声称 ±{t['claimed_uncertainty']}），`usable_as_validation_target=false`，grade **{t['grade']}**。
2. 因此 §③ 的实验 FAIL 是**参考问题不是求解器问题**；重开实验门需可溯源数字化
   （超单干范围，建议 owner 决策派单）。
3. Ritter H/a=2 的 {pct(g['ritter_comparison']['measured'])} 是物理偏移，不得当门用；
   精确区（H/a=0.05–0.2）由 `ritter_limit.py` 覆盖。

## ⑥ 误差判定（显式）
- Ritter（H/a=2）：{pct(g['ritter_comparison']['measured'])}，**<3%？否；<15%？否**——
  但属预期物理偏移，非方案缺陷（精确区验证在 ritter_limit.py）。
- 实验（M&M）：MAE {pct(g['experimental_comparison']['measured'])}，**<15%？否（FAIL）**，
  但参考 WITHHELD，**不构成有效精度结论**；在可溯源数字化替换前，
  本线精度主张维持 WITHHELD，不粉饰。
- 稳定性门（质量/逃逸/Mach/压缩）：全 PASS（REAL）。

**结论**：dam-break 求解器稳定且质量守恒（REAL）；精度对实验 **WITHHELD**
（参考不可溯）。m239 欠账从「未复核」变为「已复核+机器钉死」。
"""

png = open("benchmarks/dam_break/dam_break_status.png", "rb").read()
body = {
    "title": "TensorSPH dam-break B2 验证状态总览：m239「未复核」欠账结清（M&M 表 WITHHELD 钉死 + 稳定性门全 PASS）",
    "summary": ("Consolidates the dam-break B2 verification status into a machine-checked doc "
                "(docs/verification_dam_break.md, numbers injected from JSON) + 7 pinning tests. "
                "M&M (1952) table provenance pinned WITHHELD (exact tau lattice + 0.05 quantum); "
                "experimental gate FAIL is a reference problem, not a solver problem; "
                "stability gates all PASS. 7/7 tests green, no simulation run (load15~127)."),
    "scenario_key": "tensorsph-dam-break-verification-status",
    "mission_id": 3675,
    "tags": ["verification", "dam-break", "TensorSPH", "status-report", "verification-debt"],
    "markdown": md,
    "figures": [{"name": "dam_break_status.png",
                 "caption": "Dam-break B2: Ritter / M&M error vs dx (H/a=2), 15% gate line",
                 "png_base64": base64.b64encode(png).decode()}],
}
out = req("POST", "/tensor/reports", body, tok)
print("REPORT_ID:", out.get("id"), "status:", out.get("status"))
