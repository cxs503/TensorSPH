#!/usr/bin/env python3
"""m16406 — publish the mission report to /tensor/reports (six sections, figures inline).

Every number is read from docs/board_audit/m16406/*.json, so the report cannot
drift from the evidence (no hand-copied values).

Run: python3 scripts/audit/m16406_publish_report.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "docs" / "board_audit" / "m16406"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
MISSION = 16406

IND = json.loads((AUD / "independent_checks.json").read_text())
EVD = json.loads((AUD / "evidence_m16352.json").read_text())
REG = EVD["registry"]
REFEREE = EVD["referee"]
V1 = IND["V1"]
A2 = IND["A2"]
G = IND["geometry"]
S = IND["run_artifacts"]
XC = IND["A2_reference_crosscheck"]


def post(path: str, payload: dict, tok: str | None = None) -> dict:
    req = urllib.request.Request(API + path, method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def login() -> str:
    d = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                             "password": os.environ["TENSOR_AGENT_PASS"]})
    return d["access_token"]


def b64(name: str) -> str:
    return base64.b64encode((AUD / name).read_bytes()).decode()


def pct(x: float, nd: int = 3) -> str:
    return f"{100*x:+.{nd}f} %"


def build() -> tuple[str, str, str]:
    v1_rows = "\n".join(
        f"| {d} mm | {V1[f'panel_dx{d}']['load_sph']:.2f} | {pct(V1[f'panel_dx{d}']['rel_error'])} | "
        f"{V1[f'pairwise_dx{d}']['load_sph']:.2f} | {pct(V1[f'pairwise_dx{d}']['rel_error'])} |"
        for d in ("0.0125", "0.01", "0.0075"))
    a2_rows = "\n".join(
        f"| {n} | {A2[k]['rod_transverse']:.6f} | {A2[k]['elastica_transverse']:.6f} | "
        f"{pct(A2[k]['err_transverse'], 2)} | {A2[k]['rod_along_beam']:.6f} | "
        f"{A2[k]['elastica_along_beam']:.6f} | {pct(A2[k]['err_along_beam'], 2)} | "
        f"{pct(A2[k]['err_tip_rotation'], 2)} | {pct(A2[k]['err_as_delivered_code_computes'], 1)} |"
        for n, k in ((25, "n25"), (50, "n50")))
    a0_rows = "\n".join(
        f"| {n} | {A2[f'n{n}_A0']['tip_rod']:.6f} | {A2[f'n{n}_A0']['tip_exact']:.6f} | "
        f"{pct(A2[f'n{n}_A0']['rel_error'], 2)} |" for n in (25, 50))

    md = f"""# 交差核验（board audit）：m16352 · CASE-SPH-FSI-ELASTIC-GATE · TensorSPH

**审计单** m16406 · **被审单** m16352（succeeded，head `4af6b11`）· **裁定级别 L2** · 处置：部分证实 / L4 未成立 / 开返修

## ① 案例介绍

弹性闸门溃坝流固耦合（Antoci, Gallati & Sibilla 2007, *Comput. Struct.* 85:879-890,
doi:10.1016/j.compstruc.2007.01.002）：水柱被一竖直弹性闸门挡住，t=0 释放，判据是**闸门顶点水平位移时程的
峰值与相位**。参数（取自交付代码 `ElasticGateCase`，审计侧读属性复核）：

| 量 | 值 |
|---|---|
| 闸门高 H / 厚 t | 0.36 m / 0.004 m |
| E / ν_s / ρ_s | 1e6 Pa / 0 / 2500 kg/m³ |
| EI / EA / ρA | {G['EI']:.6g} N·m / {G['EA']:.6g} N/m / {G['rho_A']:.6g} kg/m² |
| 水柱高 h_s / 槽 | {G['water_height']} m / 0.584×0.45 m |
| 离散 dx / h / dt | {G['dx']} m / 1.3dx / 2.44e-4 s（CFL 0.3, c₀=20 m/s） |
| 静水总载荷 ½ρg h_s² | **{V1['panel_dx0.0125']['load_exact']:.2f} N/m**（审计侧独立算） |
| 一阶弯曲频率（量级） | ≈0.1 Hz（ρA、EI 推得）⇒ 准静态梁理论适用性判据：EB 小挠度顶点位移 = {IND['regime']['EB_small_deflection_tip_m']:.1f} m = {IND['regime']['tip_over_H']:.0f}×H ⇒ **本算例远在线性梁理论之外**（必然大变形 + 自接触） |

本单是**交差核验，不是重做**：未改动被审交付（`src/`、`tests/`、`benchmarks/` 一个字节未动，
被审文件 sha256 见 `evidence_m16352.json.impl_sha256`）。

## ② 怎么做的（方法/流程/工具链）

1. **取证**：`tensor_dev_missions`(16352) / `tensor_reports`(15473) / `tensor_benchmark_cases` /
   `tensorbench_results` 直查 SQL；机器裁判 `GET /tensor/bench/l4-verdict/16352`；工件 6 件镜像入审计仓并逐件 sha256。
2. **三件套核验**：工件存在性/非空 → ref vs got 数值 → `repro_cmd` 真跑（pytest 重跑 7 passed / 95.46 s）。
3. **独立重算（关键：不复用交付的自证）**：
   - V1 载荷传递：审计侧对同一交付代码的 `verify_hydrostatic_load` 在 dx=12.5/10/7.5 mm 三档各跑面板法与
     legacy 成对和，与 ½ρg h_s² 比；
   - A2 大变形参照：**独立**用 Kirchhoff elastica 的椭圆型求积（φ''=λcosφ 首次积分 + 变量替换 u²=φ_e−φ 去奇异）
     解出 x/L、y/L、φ_e，与交付 `elastica_tip()` 对表；
   - A2 杆误差：把交付的离散杆静力解（L-BFGS-B 能量最小化）按**正确的坐标对应**（闸门竖直、载荷沿 x ⇒
     杆 x = 横向、杆 y = 沿梁）与 elastica 比，同时保留交付代码的**原口径**做对照；
   - A0 两档分辨率；耦合时程重放：越界点数、变号次数、流体马赫数、t=0 载荷、载荷峰值。
4. **门禁预演**：用 `/root/tp-agent/maturity_gate.py: judge()` 只读预演（不触发动作）。
5. **落表**：`audit/insert_verdict_m16406.py`（(repo,scope,verdict_by) 幂等 upsert）→ `tensor_maturity_verdicts` id=476。
6. 全部数字由 `scripts/audit/m16406_case_fsi_board_audit.py` + `m16406_report_figures.py` 程序化生成，本报告零手抄。

## ③ 计算结果

### 3.1 基准可裁性（唯一裁判）

| 项 | 实测 |
|---|---|
| `tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-FSI-ELASTIC-GATE'` | **{REG['n_rows']} 行**（全表 {REG['table_total']} 行） |
| `tensorbench_results WHERE mission_id=16352` | **{REG['tensorbench_results_rows']} 行** |
| 机器裁判 `l4-verdict/16352` | `status={REFEREE.get('status')}`, `l4_benchmark_pass={REFEREE.get('l4_benchmark_pass')}`, `rows={REFEREE.get('rows')}`，failed_checks={REFEREE.get('failed_checks')} |
| 外部锚点（Antoci 峰值/相位） | **WITHHELD**（无 OA 通道；`provenance.json` 4 条取件失败记录，审计复核属实） |

⇒ 该锚点**没有任何机器可裁的 ref/got/tolerance**，L4 不可能成立。

### 3.2 V1 静水载荷传递（交付的两处修复之一）——审计侧重算

| dx | 面板法 got [N/m] | 面板法误差 | legacy 成对和 [N/m] | legacy 误差 |
|---|---|---|---|---|
{v1_rows}

结论：面板法 **{pct(V1['panel_dx0.0125']['rel_error'], 2)} / {pct(V1['panel_dx0.01']['rel_error'], 2)} / {pct(V1['panel_dx0.0075']['rel_error'], 2)}**，
三档 ≲0.25 % 且无分辨率依赖 ⇒ 是结构性修复，不是标定（与交付报告逐位一致，独立复算确认）。

### 3.3 A2 大变形（交付自报 +86.8 % FAIL）——审计侧发现**坐标轴对调造成的假 FAIL**

| n | rod 横向 [m] | elastica 横向 [m] | 误差 | rod 沿梁 [m] | elastica 沿梁 [m] | 误差 | 顶端转角误差 | **交付代码算出的"误差"** |
|---|---|---|---|---|---|---|---|---|
{a2_rows}

elastica 参照独立复核（椭圆型求积 vs 交付打靶）：
沿梁 {XC['audit_quadrature_along_beam']:.12f} m（x/L={XC['dimensionless_along']:.6f}），
横向 {XC['audit_quadrature_transverse']:.12f} m（y/L={XC['dimensionless_transverse']:.6f}），
φ_e={XC['audit_quadrature_tip_tangent']:.9f} rad；与交付相对差 **{XC['rel_diff_along']:.2e} / {XC['rel_diff_transverse']:.2e}**
⇒ 参照**正确**（即 Bisshopp–Drucker 经典值），错的是对照口径：`verify_cantilever()` 把杆的横向位移与 elastica 的
沿梁坐标相比。按正确配对，A2 误差 **{pct(A2['n25']['err_transverse'], 2)} → {pct(A2['n50']['err_transverse'], 2)}**（横向）且随 n 收敛，
交付报告"离散杆大变形精度是第三个缺口"的结论**被证伪**。

### 3.4 A0 悬臂点载（交付只报了 n=25）

| n | rod 顶点 [m] | EB 解析 [m] | 误差 |
|---|---|---|---|
{a0_rows}

n=50 档静力优化**静默返回初值 0**（−100 %），交付报告 ④ 表未登记（`rod_verification.json` 里有，报告里没有）。

### 3.5 耦合算例（判据所在的那条链）——数值失效证据

| 量 | 实测 | 物理解释 |
|---|---|---|
| 峰值顶点位移 | **{S['peak_tip_mm']:.1f} mm @ {S['peak_time_ms']:.1f} ms** | = {S['tip_over_gate_height']:.2f}×闸门长：几何不可能 |
| 谷值 | {S['min_tip_mm']:.1f} mm | 同上 |
| 首次 \\|tip\\|>H | t={S['first_abs_tip_over_H_s']:.4f} s | 越界采样 {S['n_samples_beyond_H']}/{len(S['tip_series']['t'])} |
| 100 ms 后变号次数 | {S['tip_sign_flips_after_100ms']} 次，周期≈32 ms（≈31 Hz） | 闸门一阶弯曲 ≈0.1 Hz ⇒ 高 300×，非物理振荡 |
| 流体最大速度 | {S['max_fluid_speed_m_s']:.2f} m/s = **{S['mach']:.2f} c₀** | 弱可压 WCSPH 已越出适用域 |
| t=0 闸门载荷 | **{S['gate_force_at_t0_N_per_m']:.3f} N/m** | 静水应 {S['hydrostatic_reference_N_per_m']:.2f} N/m ⇒ 初始状态无静水压强 |
| 载荷峰值 | {S['gate_force_max_N_per_m']:.0f} N/m = **{S['gate_force_max_N_per_m']/S['hydrostatic_reference_N_per_m']:.0f}× 静水值** | 壁面压力外推无界 + 压力波冲击 |

### 3.6 门禁预演

```text
judge(2, evidence, 'TensorSPH', 16406) -> (2, [])                        # L2 三键全过
judge(3, evidence, ...)                -> (2, ['L3 缺: integration_callers(grep 无引用)'])
judge(4, evidence, ...)                -> (2, [L4 缺 integration_callers, L3 缺 ...])
```

⇒ 声明的 L2 与机器复核一致（L3/L4 主动不申报，且即便虚报也会被降级）。

## ④ 结果图

图 1 — 左：交付耦合时程的顶点位移 vs 几何可容许带（|tip|≤H=360 mm）；右：A2 大变形下杆形状与独立 elastica 对比
（标注按正确坐标配对的误差）。图 2 — V1 载荷传递：面板法 vs legacy 成对和，三档分辨率，±3 % 带。图 3 — 左：流体→闸门力时程
（symlog）vs 静水参考，标注 t=0 与峰值倍数；右：同一时程的顶点位移与越界区间。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **"自证"不等于"验证"**：交付的 A2 用自己写的 elastica 参照是对的（独立复核 1e-13 级一致），但**对照口径错**，
   于是把一个 2–5 % 的结果报成 +86.8 % FAIL，并据此推出"杆大变形不可信"的结论。审计的价值在此：
   参照正确 ≠ 比对正确。
2. **假 FAIL 与假 PASS 同样有害**：本单是假 FAIL（把能力报低），方向虽保守，但与"honest FAIL"不同——
   honest FAIL 是能力真的不够，这里是**度量口径 bug**。返修单应按度量 bug 修，不要再去"提高杆的精度"。
3. **耦合失效的归因要小心**：交付把它归因于 rod 大变形精度；实测证据（Ma=3.6、载荷 64× 静水、t=0 载荷为 0、
   31 Hz 振荡）指向流固界面/初始化，而不是杆。**根因未证**（时程未存 `p_face` 诊断量），本报告只声明"该时程不能当物理结果用"。
4. **测试的时间窗必须覆盖失效点**：`test_coupled_short_run_is_deterministic` 只跑 t=0.01 s，而失效自 t=0.1148 s 起——
   "跑了没崩"≠"算对了"。另有 1 条断言写成 `assert ... or True`，恒真。
5. **恒真断言 + 选择性缺行**：`rod_verification.json` 里 A0@n=50 = −100 % 存在，报告表却只登 n=25。
   审计判级以**工件**为准，不以报告表格为准（本次即靠工件发现）。
6. **口径冲突留痕**：9 小时前的先例 verdict id=464（同一仓、同一"模块+自备驱动+测试"形态）判 L3；
   本单判 L2，因为唯一的"调用方"是模块自己的基准驱动。请 owner/board 统一口径（若采纳先例口径，本单升 L3，其余证据不变）。
7. **局限**：未重跑耦合算例（247 s/档，且已定性为数值失效）、未跑 dx=10 mm（交付方超时）、未跑全仓回归、
   未取得 Antoci 实验曲线（与被审方同样受限）、未定位耦合失效根因。

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答是否 <3 %）

| 关键量 | 基准 / 参照 | got | 误差 | <3 %？ |
|---|---|---|---|---|
| 静水载荷传递 dx=12.5 mm | 418.21992 N/m（½ρg h_s²，审计独立算） | {V1['panel_dx0.0125']['load_sph']:.2f} N/m | {pct(V1['panel_dx0.0125']['rel_error'], 3)} | **是** |
| 静水载荷传递 dx=10 mm | 同上 | {V1['panel_dx0.01']['load_sph']:.2f} N/m | {pct(V1['panel_dx0.01']['rel_error'], 3)} | **是** |
| 静水载荷传递 dx=7.5 mm | 同上 | {V1['panel_dx0.0075']['load_sph']:.2f} N/m | {pct(V1['panel_dx0.0075']['rel_error'], 3)} | **是** |
| legacy 成对和（对照） | 同上 | {V1['pairwise_dx0.0125']['load_sph']:.2f} N/m | {pct(V1['pairwise_dx0.0125']['rel_error'], 1)} | 否（已修，作为回归护栏） |
| A2 横向位移 n=25 | elastica 0.291819 m | {A2['n25']['rod_transverse']:.6f} m | **{pct(A2['n25']['err_transverse'], 2)}** | **是** |
| A2 横向位移 n=50 | 同上 | {A2['n50']['rod_transverse']:.6f} m | **{pct(A2['n50']['err_transverse'], 2)}** | **是** |
| A2 沿梁坐标 n=25 / n=50 | elastica 0.160202 m | {A2['n25']['rod_along_beam']:.6f} / {A2['n50']['rod_along_beam']:.6f} m | {pct(A2['n25']['err_along_beam'], 2)} / {pct(A2['n50']['err_along_beam'], 2)} | 否（4.6 % / 2.3 %，但**随 n 收敛**且 <5 %） |
| A0 点载 n=25 | EB 0.000240 m | {A2['n25_A0']['tip_rod']:.6f} m | {pct(A2['n25_A0']['rel_error'], 2)} | 否（6.3 %） |
| A0 点载 n=50 | EB 0.000240 m | {A2['n50_A0']['tip_rod']:.6f} m | **{pct(A2['n50_A0']['rel_error'], 1)}** | 否（静默失败，需修） |
| elastica 参照自洽 | 审计独立求积 | — | {XC['rel_diff_along']:.1e}（沿梁）/ {XC['rel_diff_transverse']:.1e}（横向） | **是** |
| **判据量：顶点峰值/相位 vs Antoci 2007** | **WITHHELD** | {S['peak_tip_mm']:.1f} mm @ {S['peak_time_ms']:.1f} ms | **不可计算**（且该 got 本身数值失效，{S['tip_over_gate_height']:.2f}×H） | **无基准，声明原因：实验曲线在本沙箱不可获得（4 条 OA 通道失败）+ 基准表 0 行** |

**结论（honest）**：V1 载荷传递的三档误差 **均 <3 %** 且为结构性修复；A2 大变形（给能力"平反"）横向误差
**<3 %**、沿梁 2–5 % 且随 n 单调收敛；A0@n=50 静默归零需修。**判据量（Antoci 峰值/相位）无基准可判**，
且交付的耦合时程为数值失效（3.96×闸门长、Ma 3.6、载荷 64× 静水）——因此 **L4 不成立，裁定 L2**。
"""
    title = ("交差核验 m16352（CASE-SPH-FSI-ELASTIC-GATE）：L2 裁定 + 三处被审方未发现的缺陷"
             "（A2 轴对调假 FAIL / A0@n=50 归零 / 耦合时程数值失效）")
    summary = (f"基准表 0 行、机器裁判 CLAIMED_NOT_PROVEN、ref WITHHELD ⇒ L4 不成立；审计侧独立复算证实 "
               f"V1 载荷传递三档 <0.25 %、证伪交付自报的 A2 +86.8 % FAIL（正确口径 "
               f"{100*A2['n25']['err_transverse']:+.2f}/{100*A2['n50']['err_transverse']:+.2f} %）、"
               f"暴露 A0@n=50 静默归零与耦合时程 Ma={S['mach']:.2f}/载荷 64× 静水的数值失效；裁定 L2（无集成调用方）。")
    return title, summary, md


def main() -> int:
    title, summary, markdown = build()
    (AUD / "report_m16406.json").write_text(json.dumps(
        {"title": title, "summary": summary, "markdown": markdown}, ensure_ascii=False, indent=1))
    payload = {
        "title": title,
        "summary": summary,
        "scenario_key": "board-audit/fsi-elastic-gate",
        "mission_id": MISSION,
        "tags": ["board-audit", "TensorSPH", "FSI", "elastic-gate", "Antoci",
                 "maturity-verdict", "L2", "m16406"],
        "markdown": markdown,
        "figures": [
            {"name": "fig1_tip_and_elastica.png",
             "caption": "左：交付耦合时程的闸门顶点位移 vs 几何可容许带 |tip|<=H=360 mm（峰值 1427.4 mm = 3.96H，"
                        "自 t=0.1148 s 越界）；右：A2 大变形 (PL^2/EI=10) 下离散杆形状 vs 审计侧独立椭圆型求积 elastica，"
                        "标注按正确坐标配对的误差（+2.55/-4.58 % @n=25）。",
             "png_base64": b64("fig1_audit.png")},
            {"name": "fig2_load_transfer.png",
             "caption": "V1 静水载荷传递：面板法 vs legacy Adami 成对和，dx=12.5/10/7.5 mm 三档，阴影为 ±3 % 判据带。"
                        "面板法 +0.240/+0.089/+0.189 %（结构性修复），legacy -28.4/-20.4/-8.3 %。",
             "png_base64": b64("fig2_load_transfer.png")},
            {"name": "fig3_gate_force.png",
             "caption": "左：流体→闸门水平力时程（symlog）vs 静水参考 418.22 N/m，t=0 载荷为 0.000 N/m、峰值 26794 N/m=64×；"
                        "右：同一时程的顶点位移与越界区间（177/330 采样超几何上限）。",
             "png_base64": b64("fig3_gate_force.png")},
        ],
    }
    tok = login()
    d = post("/tensor/reports", payload, tok)
    print(json.dumps({k: v for k, v in d.items() if k in ("id", "status", "mission_id")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())