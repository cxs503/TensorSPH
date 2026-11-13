"""Publish the m18382 progress report to the platform (documentation-only helper).

Reads the committed artefacts and posts a six-section report with one embedded
figure.  Credentials come from the environment (``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``); nothing is written to disk except
the payload the platform receives.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias" / "bias_report.json"
FIG = ROOT / "docs" / "figures" / "wedge_operator_bias.png"


def post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    tok = post(f"{base}/auth/login",
               {"username": os.environ["TENSOR_AGENT_USER"],
                "password": os.environ["TENSOR_AGENT_PASS"]})
    token = tok.get("access_token") or tok.get("token")
    rep = json.loads(ART.read_text())
    v, s = rep["verdict"], rep["hypothesis"]["signs"]
    r2, r4 = 100.0 * v["dx2mm"]["residual"], 100.0 * v["dx4mm"]["residual"]
    op2 = 100.0 * (rep["grids"]["dx2mm"]["by_frames"]["61"]["synth_smoothed"]["peak_median"]
                   / rep["wagner_cp_max_point"] - 1.0)
    op4 = 100.0 * (rep["grids"]["dx4mm"]["by_frames"]["61"]["synth_smoothed"]["peak_median"]
                   / rep["wagner_cp_max_point"] - 1.0)
    obs2 = 100.0 * s["dx2mm"]["obs_err"]
    obs4 = 100.0 * s["dx4mm"]["obs_err"]

    md = f"""# CASE-SPH-WEDGE-ENTRY-2D 峰值砰击压力：同算子复核（m18382）

## ① 案例介绍
二维刚性楔形体 β = 30°、V = 1 m/s 恒速入静水、忽略重力（自相似，唯一长度尺度 V·t）。参照：Wagner 闭式 `Cp(τ) = 2Cs − s² + 1`，`C = π/(2 tanβ) = 2.7207`，点值峰值 `C² + 1 = 8.4022`（τ_peak = 0.930），顶点 `Cp(0) = 2C = 5.4414`。任务指定参考 Zhao & Faltinsen (1993) JFM 246:593，其 β = 30° 的具体数值仍 **WITHHELD**（闭源、仓内无副本、不以记忆值或二手值代替）。算例：WCSPH + Tait EOS，c₀/V = 20，α = 0.03，h/dx = 1.3，dx ∈ {{4, 2}} mm，61 帧自相似窗（t_end = 18.38 ms）。

## ② 怎么做的
不改求解器、不改算例、不放宽阈值、不重跑 SPH（输入只是已入仓的聚合工件）。把 Wagner 场推过**同一测量算子**：壁面节点采样（`wedge_lattice` 几何 `|x_k| = dx[(k+½)cosβ − ½sinβ]`）→ Adami 核外推（求解器同一 2-D cubic spline、h = 1.3 dx、3 行流体邻域）→ 与 `ensemble_peak.py` 逐行等价的分箱 + 4 样本门 + 取最大（测试对齐到 1e-12）。前向模型先用两次数据对表校准后使用：逐帧湿节点数与工件差 1–2 个节点；τ → 0 处算子无偏到 0.04 %（该位置 `dCp/dτ = 0`，核外推与采样在数学上无偏）。

## ③ 计算结果
| 量（61 帧池化） | dx = 2 mm | dx = 4 mm |
|---|---|---|
| 实测峰值 Cp | 5.850 | 10.360 |
| 实测 vs 点值 8.4022（历轮口径） | {obs2:+.1f} % | {obs4:+.1f} % |
| 算子真值（Wagner 经同一算子） | 7.494 | 6.870 |
| **同算子残差** | **{r2:+.1f} %** | **{r4:+.1f} %** |

算子偏置分解（同一解析场，只换算子环节）：点采样 8.063 / 8.030（对点值仅 −4 %）→ 加核外推 7.494 / 6.870，即**核外推固定吃掉 {op2:+.1f} % / {op4:+.1f} %**；原因是 β = 30° 的峰值只离喷溅根 0.07c ≈ 1.8 dx（2 mm）/ 0.9 dx（4 mm），正在核支撑边缘。跨网格残差差 {100.0 * v['cross_grid_residual_gap']:.1f} 个百分点且**变号**。

## ④ 结果图
左：同一估计量用在两侧（SPH 池化峰值 vs Wagner 经同一算子），黑线 = 点值、灰带 = ±15 %；右：同算子残差 vs dx 与 ±15 % 带。

## ⑤ 注意点
1. 算子偏置是**测量链常数**，不是解的误差；把测量值直接比点值等于把它记在求解器头上。
2. 本单第一版把核 stencil 锚在顶点（而非逐节点），τ > 0.2 全 NaN、被估计量跳过，算出漂亮的 −35 %；是“顶点处算子必然无偏”的校准锚 + 专门测试把它抓出。教训：任何“解释掉失败”的分解必须先有无偏锚点与守恒式自检。
3. **更正 m17899**：逐帧 `n_wet` 单调增（dx = 2 mm：开头 0–2 个 → 末帧 60 个），故“湿面节点随入水加深衰减 / 接触判据在深穿透下失效”被其自身入仓工件否定；jackknife 43 % 的真实原因只是早窗湿节点少。更正经测试钉住。
4. 残差为负的方向与“线性理论在 β = 30° 高估峰值、非线性相似解应更低”一致，但 Z&F 数值不可得，故只作**假说**，不作修正因子。
5. 适用范围：β = 30°、V = 1 m/s、c₀/V = 20、dx ∈ {{4, 2}} mm 单点，不外推。

## ⑥ 误差判定
同算子峰值残差：dx = 2 mm **{r2:+.1f} %（< 15 %？否）**、dx = 4 mm **{r4:+.1f} %（否）**；跨网格同号？**否 → 峰值 Cp 仍不可裁决，验收结论不变（不可裁决/FAIL）**，H1（估计量伪像）被证伪，变号属数据属性。可裁决量：顶点 Cp(0) 4.713 / 4.999 vs 5.4414 → −13.4 % / −8.1 %（**< 15 % 是**）。峰值 vs Zhao & Faltinsen (1993)：**WITHHELD**（基准数值未获得，不以替代值判定）。

工件：`docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json`；全文：`docs/case_sph_wedge_entry_2d_operator_bias_m18382.md`；commit 8869f5c；测试 `tests/test_wedge_estimator_bias.py`（9 passed，含与 `ensemble_peak.py` 的等价性、常量场精确性、顶点无偏性、工件 `n_wet` 单调性回归）。
"""
    payload = {
        "title": ("CASE-SPH-WEDGE-ENTRY-2D：峰值砰击压力同算子复核——测量链固定吃掉 "
                  f"{op2:.1f} %/{op4:.1f} %，峰值 Cp 仍不可裁决（m18382）"),
        "summary": ("Wagner 场经与 ensemble_peak 同一测量算子：核外推吸收 "
                    f"{op2:.1f} %(2 mm)/{op4:.1f} %(4 mm)，同算子残差 {r2:+.1f} %/{r4:+.1f} % "
                    "仍变号 → 峰值不可裁决、Z&F 仍 WITHHELD；并更正 m17899 的湿节点误读。"),
        "scenario_key": "case_sph_wedge_entry_2d",
        "mission_id": 18382,
        "tags": ["CASE-SPH-WEDGE-ENTRY-2D", "TensorSPH", "slamming",
                 "measurement-operator", "verification", "honest-FAIL"],
        "markdown": md,
        "figures": [{"name": "wedge_operator_bias.png",
                     "caption": ("左：SPH 池化峰值 vs Wagner 经同一测量算子（黑线=点值，"
                                 "灰带=±15 %）；右：同算子残差 vs dx。"),
                     "png_base64": base64.b64encode(FIG.read_bytes()).decode()}],
    }
    out = post(f"{base}/tensor/reports", payload, token)
    print(json.dumps({k: out[k] for k in out if k != "markdown"}, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())