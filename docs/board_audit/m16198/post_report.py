"""m16198 论文级进展报告：发到 tensor_reports（六要素，图用 base64 内嵌，数字从 audit.json 注入）。"""
import base64
import json
import os
import urllib.request
from pathlib import Path

WT = Path("/root/tp-agent/work/m16198")
ART = WT / "artifacts" / "m16198"
A = json.loads((ART / "audit.json").read_text())
ref, runs, model = A["ref"], A["runs"], A["damping_model"]
base = sorted([r for r in runs.values()
               if "sigma_stored" in r and r["nu"] == 1e-3 and r["amp0"] == 0.01], key=lambda r: r["dx"])
rep = A["repro"]["dx0.04_nu0.001_a0.01"]
API = os.environ["TENSOR_API_BASE"].rstrip("/")
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]


def post(url, payload, token=None):
    data = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=hdr, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


tok = post(API + "/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
assert token, tok

rows = "\n".join(
    f"| {r['dx']:g} | {r['particles']} | {r['end_time_recorded']} | {r['sigma_stored']:.3f} | "
    f"{100 * (r['sigma_stored'] / ref['sigma_ideal_Agk'] - 1):+.1f} % | {r['sigma_early']['sigma']:.3f} | "
    f"{r['sigma_late']['sigma']:.3f} | {r['sigma_late']['sigma'] / r['sigma_early']['sigma']:.2f}× |"
    for r in base)
ptab = "\n".join(
    f"| dx={m['dx']:g} | {m['predicted']:.3f} | {m['measured']:.3f} | {m['resid_pct']:+.1f} % |"
    for m in model["predictions"])
gauges = "\n".join(
    f"| {t} | {r['a_end']:.4f} | {r['amplitude_geometric_final']:.4f} | {100 * (r['amplitude_geometric_final'] / r['a_end'] - 1):+.1f} % | "
    f"{r['sigma_vel_refit']['sigma']:.3f} | {r['sigma_vel_refit']['sigma'] / r['sigma_refit']['sigma']:.2f}× |"
    for t, r in runs.items() if "amplitude_geometric_final" in r)

md = f"""# m16198 交差核验：CASE-SPH-RAYLEIGH-TAYLOR-MP（被审单 m16141）→ L2，L4 明确不达

**裁定：L2（独立功能实现）**｜`tensor_maturity_verdicts` id=440（gate_state=pending）｜被审提交 agent/m16141 @ 9736ad7｜被审报告 #15292

## ① 案例介绍
两层密度反差流体单模界面扰动（重相在上），A=(ρ_h−ρ_l)/(ρ_h+ρ_l)={ref['atwood']:.3f}，ρ_h/ρ_l=1500/500 kg/m³，λ=1 m（k={ref['k']:.4f} m⁻¹），上下层深各 0.75 m，η₀=0.01λ，c₀=40 m/s，ν=1e-3 m²/s，g=9.81 m/s²。判据=早期界面扰动指数增长率 vs Taylor (1950) 线性增长率 σ=√(Agk)={ref['sigma_ideal_Agk']:.4f} s⁻¹（有限深度 {ref['sigma_finite_depth']:.4f}、黏性修正 {ref['sigma_viscous']:.4f}）。审计独立复算三档基准，与被审 reference.json 相对差 0.0e+00。

## ② 怎么做的
被审单新建 784 行两相（密度反差）WCSPH 时间推进器 `src/tensorsph/rayleigh_taylor.py`（逐粒子 ρ₀+Tait、数密度连续式、哈密顿对称压力力、静水解析格点初始化、线性本征函数激励、x 周期邻域、双振幅量计、ln 斜率拟合）。
审计路径（非采信）：①三件套取证 + 13 个工件 sha256 快照；②只用 `series` 原始时序**独立重拟合**（不采信存盘 fit）；③**第三量计**——从已存密度场快照按 ρ=(ρ_h+ρ_l)/2 等值线读界面 η(x)；④三点观测阶 + 6 组 Richardson 外推 + 只用 A=0 两点拟合的加性阻尼模型盲预测；⑤在干净拷贝原样跑 repro_cmd（dx=0.04，35 s）、damping.py（30 s）、pytest（61.7 s）；⑥集成面（`__init__.py`/verification/全仓 grep）。

## ③ 计算结果
重拟合与提交值**逐位一致**（工件自洽、无篡改）；但斜率随窗口上抬：

| dx [m] | 粒子 | end_time | σ 提交 [s⁻¹] | vs √(Agk) | 早半窗 | 晚半窗 | 晚/早 |
|---|---|---|---|---|---|---|---|
{rows}

盲预测（λ₀={model['lambda0_per_m']:.1f} 1/m 只用 A=0 两点拟合）：

| 分辨率 | 预测 σ | 实测 σ | 残差 |
|---|---|---|---|
{ptab}

三量计（同一批工件，末步界面振幅）：几何量计与质心量计 0.3–3.8 % 一致（→亏损是**真测量**）；速度带量计偏离 0.96–2.30×：

| 算例 | 质心 a_end | 几何 a_end | 相对差 | 速度带 σ | σ_vel/σ_centroid |
|---|---|---|---|---|---|
{gauges}

可复现性：repro_cmd 可执行但**不复现表中值**（{rep['committed_sigma']:.4f} → {rep['rerun_sigma']:.4f}，{rep['rel_change_pct']:+.2f} %，序列到 t≤0.40 逐位相同 ⇒ 差异纯来自未写进 repro 的 end_time 0.42/0.35）；damping.py 逐位复现（−4.4297/−2.0372）。

## ④ 结果图
图 1：σ/√(Agk) 四档收敛 + 判据线 + A=0 加性阻尼模型；右panel 早/晚半窗漂移。
图 2：三量计对照（几何 vs 质心 vs 速度带）。
图 3：从密度场快照读出的界面 η(x) 与质心振幅 cos 模态——单模形态成立。

## ⑤ 注意点（按严重度）
- **F2 high**：指标未收敛——预注册窗上界 0.1λ 从未达到（最大 a_end=0.048 m），实窗被 end_time 截断，早/晚半窗差 4.1/8.8/18.9/9.3 %；报告的 −24.5 % 是「窗口选择值」不是渐近增长率（FAIL 方向稳健）。
- **F6 high**：文档「二阶外推 dx≈0.01 处 −7 %」是硬编码字符串（make_doc.py:159）：观测阶为 **负**（−0.46、−2.93），6 组 Richardson 外推落 −33.0…+34.7 %，一阶阻尼模型给 −16.4 %——**声明不受工件支持**。
- **F1 high/流程**：`tensor_benchmark_cases` 本锚点 0 行（全表 25 行）⇒ 基准裁判缺席，10 % 门为自定。
- **F3 medium**：repro 不复现表中值（+3.72 %，因 end_time 未写入文档）。
- **F4 medium**：速度带量计与质心量计差 0.96–2.30×，档内可差 25×，工件里 `sigma_velocity_over_ideal`=0.951（dx=0.02）读起来像过 10 % 门，报告完全未提。
- **F5 正面**：第三（几何）量计与质心量计 0.3–3.8 % 一致 ⇒ 亏损方向不是量计伪影。
- **F7–F11 low**：图无生成器（figures/*.png 全仓无 savefig）；`_field_snapshot` 除 Σw 而非 Σ(w·vol) ⇒ 存盘 rho 场为 ρ·dx²（0.203/0.603 而非 500/1500）；`mass_drift` 恒 0（*0.0）；报告③ νk²/σ 写 0.0128 复算 {ref['nu_k2_over_sigma']:.5f}；dx=0.05 NaN 记录无注。

## ⑥ 误差判定
关键量 σ vs √(Agk)：dx=0.0125 **−24.5 %**、0.02 −28.5 %、0.025 −44.2 %、0.04 −61.7 %。
**是否 <3 % —— 否**（四档全部不达）；<10 % —— 否；<15 %（平台不可接受线）—— 否。
被审单自报 honest-FAIL 与工件一致，**无粉饰**（审计确认）；问题不在「有没有失败」，而在「失败数字的量与不确定度」（F2/F4/F6）。
不变量面全部通过：t=0 静水压强逐点=解析（<1e-6 相对）、Σm a_x=0 至 1e-10、Σm a_y=−Mg、界面压连续而密度跳 3×、质量求和替代方案界面压误差 140×/19×、A=0 无伪增长、M<0.2、10 项 pytest 全绿（独立复跑 61.7 s）。

## 判级与处置
L1 达（10 项不变量测试）｜**L2 达**（784 行真推进器 + 8 份可复算算例工件 + 驱动链）｜L3 不达（`__init__.py` 未导出、verification 无引用、caller 仅 benchmarks+tests、无平台入口）｜L4 不达（裁判缺席 + 误差 >15 % + 指标未收敛 + repro 不复现 + 外推声明无支持）｜L5 无。
处置：L2 三件套齐 → 建议门判 confirmed 消账；若消账线要求 L4 则应 reopened，返修 R1（可达振幅窗 + 声明 end_time，证明 dσ/d窗→0）、R2（量计一致性/不确定度）、R3（repro 与工件对齐）、R4（删或程序化重算外推）。

审计工件：`artifacts/m16198/{{audit,refit}}.json`、`refit.py`、`analyse.py`、fig1–3.png；被审工件 sha256 快照 `docs/board_audit/m16198/evidence/`（MANIFEST.json，13 文件）。
"""

figs = []
for name, cap in (("fig1_growth.png", "图1 σ/√(Agk) 四档收敛 + 10%/15% 判据 + A=0 加性阻尼模型；(b) 早/晚半窗拟合漂移"),
                  ("fig2_gauges.png", "图2 三量计对照：几何（密度场等值线）vs 质心 vs 速度带 —— 质心量计被第三量计佐证，速度带量计偏离"),
                  ("fig3_field.png", "图3 从已存密度场快照读出的界面 η(x)（dx=0.02 末步）与质心振幅 cos 模态")):
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode((ART / name).read_bytes()).decode()})

payload = {
    "title": "m16198 交差核验：CASE-SPH-RAYLEIGH-TAYLOR-MP（m16141）→ L2，L4 明确不达",
    "summary": (f"board-audit m16198 核验 m16141：三件套真存在（8 算例工件 sha256 快照、10 pytest 独立复跑全绿、repro 可执行）；"
                f"独立重拟合逐位复现提交 σ；第三几何量计与质心量计 0.3–3.8% 一致（亏损为真测量）；但指标未收敛（早/晚窗 4–19%）、"
                f"repro 不复现表中值（+3.72%）、双量计差 0.96–2.30× 未披露、外推 −7% 声明无工件支持；"
                f"σ/√(Agk) 最细 −24.5% > 15% 平台线 ⇒ L2 成立、L4 不达；落表 tensor_maturity_verdicts id=440 level=2。"),
    "scenario_key": "CASE-SPH-RAYLEIGH-TAYLOR-MP",
    "mission_id": 16198,
    "tags": ["board-audit", "cross-check", "TensorSPH", "Rayleigh-Taylor", "multiphase", "maturity-L2"],
    "markdown": md,
    "figures": figs,
}
res = post(API + "/tensor/reports", payload, token)
print(json.dumps(res, ensure_ascii=False)[:400])
rid = res.get("id") or res.get("data", {}).get("id")
(WT / "docs" / "board_audit" / "m16198" / "report_response.json").write_text(json.dumps(res, ensure_ascii=False)[:2000])
print("report id", rid)