#!/usr/bin/env python3
"""Assemble the mission-23237 progress report payload from the committed artefacts.

Writes ``/tmp/report.json`` (title / summary / scenario_key / mission_id / tags /
markdown / figures) so the report can be posted with

    curl -s -X POST $TENSOR_API_BASE/tensor/reports \
         -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
         -d @/tmp/report.json

Nothing in the body is typed twice: the note is read from disk and every number
in the appended tables is read from ``out/wo_envelope_wall_layer.json``.
"""

from __future__ import annotations

import base64
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
ART = REPO / "benchmarks" / "case_poiseuille" / "out" / "wo_envelope_wall_layer.json"
DOC = REPO / "docs" / "case_poiseuille_wo_envelope_wall_layer.md"
FIG = REPO / "docs" / "figures" / "case_poiseuille_wo_envelope_wall_layer.png"
DEST = pathlib.Path("/tmp/report.json")


def main() -> int:
    art = json.loads(ART.read_text())
    m1 = art["models"]["A1_power_law"]
    m2 = art["models"]["A2_saturating"]
    h1 = art["holdout"]["A1_power_law"]
    v = art["verdict"]
    dom = art["anchors"]["rule_domain"]
    best = art["anchors"]["best_measured_wall_point"]
    unb = art["anchors"]["unresolved_branch"]
    req5 = next(r for r in art["requirements"]
                if r["model"] == "A1_power_law" and r["target_wall_err_pct"] == 5.0)
    req3 = next(r for r in art["requirements"]
                if r["model"] == "A1_power_law" and r["target_wall_err_pct"] == 3.0)

    tail = ["", "## 附：本单裁定（由工件程序化判定，非手抄）", "",
            "| 项 | 值 |", "|---|---|",
            f"| 采用模型 | `{v['adopted']}` |",
            f"| W1 幂律可预测 | {v['W1_power_law_is_predictive']} |",
            f"| W2 平台项可辨识 | {v['W2_plateau_is_real']}（ΔAIC = "
            f"{art['models']['aic']['delta_aic_favouring_A2']:.3f} < "
            f"{art['thresholds']['delta_aic_min']}） |",
            f"| W3 行级 H8 | {v['W3_H8_at_row_level']}"
            f"（最大不一致 {v['max_matched_pair_disagreement_pp']:.3f} pp） |",
            f"| 5 % 壁面要求 | delta/dx ≥ {req5['delta_over_dx_required']:.2f}"
            f"（域内: {req5['inside_calibrated_domain']}） |",
            f"| 3 % 壁面要求 | delta/dx ≥ {req3['delta_over_dx_required']:.2f}"
            f"（域内: {req3['inside_calibrated_domain']}） |",
            f"| 标定域 | delta/dx ∈ [{dom['delta_over_dx_min']:.3f}, "
            f"{dom['delta_over_dx_max']:.3f}] |",
            f"| 域内留出 MAPE | {h1['interpolation_mape_pct']:.2f} % |",
            f"| 域外留出（禁外推） | {h1['extrapolation_abs_rel_error_pct']:.2f} % |",
            f"| 未解析支实测平台 | {unb['err_wall_pct_range']} %（均值 "
            f"{unb['err_wall_pct_mean']:.2f} %） |",
            f"| 实测最优壁面点 | `{best['id']}` @ delta/dx="
            f"{best['delta_over_dx']:.3f}: {best['err_wall_pct']:.2f} % |", "", "---", "",
            "完整交付：`benchmarks/case_poiseuille/wall_layer_envelope.py`（分析）、"
            "`benchmarks/case_poiseuille/make_wall_layer_envelope_doc.py`（渲染）、"
            "`tests/test_case_poiseuille_wall_layer_envelope.py`（22 项绑定测试）。",
            "父交付（已并入 main，PR #739）：`docs/case_poiseuille_wo_envelope.md`；"
            "本单不重复其内容，只补它留下的空缺：壁面层所需的网格量。", ""]

    payload = {
        "title": ("CASE-POISEUILLE-SPH · 近壁（Stokes 层）分辨率包线："
                  "判据看不见的那一层要多细"),
        "summary": (
            f"父研究（PR #739 已并入 main）把 delta/dx>=3 的范围声明在它自己附着的内区判据上证伪了，"
            f"但没给出壁面行所需的网格。本单用同一份 14 点工件做二次分析（零次求解器重跑）："
            f"域内 err_wall = {m1['K']:.2f}·(delta/dx)^-{m1['m']:.3f}，R²={m1['r2']:.4f}，"
            f"域内留出 MAPE {h1['interpolation_mape_pct']:.2f} %，行级 H8 最大不一致 "
            f"{v['max_matched_pair_disagreement_pp']:.2f} pp；"
            f"反解 5 % 壁面要求 delta/dx ≥ {req5['delta_over_dx_required']:.2f}"
            f"（外推，最细实测点 {best['err_wall_pct']:.2f} % 仍未进 5 %）。"
            f"平台项假设 W2 在域内被 AIC 证伪（ΔAIC={art['models']['aic']['delta_aic_favouring_A2']:.2f}），"
            f"但域外 3 点实测 {unb['err_wall_pct_range']} % 说明平台真实存在且幂律在那里平均偏 "
            f"{h1['extrapolation_abs_rel_error_pct']:.0f} % ⇒ 包线只能向上用，域边界是硬边界。"
            f"内区判据到 delta/dx=0.87 仍 <1.3 %：两个度量所需分辨率差约一个量级。"),
        "scenario_key": "CASE-POISEUILLE-SPH",
        "mission_id": 23237,
        "tags": ["TensorSPH", "CASE-POISEUILLE-SPH", "Womersley", "verification",
                 "wo-envelope", "wall-layer", "resolution-envelope",
                 "falsified-hypothesis", "out-of-sample"],
        "markdown": DOC.read_text() + "\n".join(tail),
        "figures": [{
            "name": FIG.name,
            "caption": ("图1 (a) 近壁分辨率包线：实心圆=拟合集（r16 频率扫描），"
                        "空心方块/叉=域内/域外留出点，阴影=标定域，绿点线=反解的 "
                        "delta/dx 要求，红/紫虚线=5 %/3 % 门；(b) 逐行振幅误差随 "
                        "y/delta 的塌缩（颜色=delta/dx），误差集中在 y/delta<2，"
                        "正是内区判据平均掉的那几行"),
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        }],
    }
    DEST.write_text(json.dumps(payload))
    print(f"wrote {DEST} ({DEST.stat().st_size} bytes), "
          f"markdown {len(payload['markdown'])} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())