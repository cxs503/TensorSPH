"""Build /tmp/report.json for the modal-transferability deliverable (no hand-typed numbers)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
doc = (ROOT / "docs" / "xval_projection_laplace_modal_transferability.md").read_text()
res = json.loads((ROOT / "benchmarks/laplace_sph/results_modal/results_modal.json").read_text())
png = base64.b64encode(
    (ROOT / "benchmarks/laplace_sph/results_modal/modal_transferability.png").read_bytes()).decode()

g = {x["gate"]: x for x in res["gates"]}
deformed = [r for r in res["runs"] if r["mode_amplitude_eps"] > 1e-9]
robust = [r for r in deformed if r["aspect_a_over_b"] >= 1.05]
worst_robust = max(min(abs(r["omega_full_vs_wall_pct"]), abs(r["omega_full_vs_unconf_pct"]))
                   for r in robust)

md = doc.split("## ④ Figures")[0]
md += (
    "\n\n## ④ 图\n\nmode-2 模态转移率三联图见 figures。\n\n"
    "## ⑤ 注意点/局限\n"
    "等密度载流子；方形载体按 k=1.6 同心刚性壁近似（真值落在壁端与无界端之间）；"
    "a/b=1.02 由格点噪声主导，保留在表内不剔除；R/dx=16 的收敛梯级已递交集群 cj4588，"
    "本页不以未跑出的数为结论；本单只覆盖等密度、二维、无粘。\n\n"
    "## ⑥ 误差判定\n"
    f"- 出货 CSF 臂 mode-2 频率 vs 解析带（a/b≥1.05 最差）：{worst_robust:.1f}%（另：全样本最差 "
    f"{res['worst_frequency_err_vs_band_pct']:.1f}% 含 a/b=1.02 噪声样本）→ 不 <3%，也不 <15% → honest FAIL\n"
    f"- well-balanced 臂动力学保留率 R2 最小：{res['min_dynamics_retention']:.2e}（门 0.95）→ FAIL，"
    "机理是 u*=dt·Gq 的代数恒等式（p=(ρ/dt)q ⇒ u=0），非分辨率/粘性效应\n"
    f"- 静态控制（圆，均衡臂）：max|u|/v_ref = {g['static_control']['measured']:.2e}（门 1e-5）→ PASS，"
    "复现 round-3 数值，但含义是「不能动」而非「被擎住」\n"
    f"- Helmholtz 分裂重构/正交性：{g['split_reconstruction']['measured']:.2e} / "
    f"{g['split_orthogonality']['measured']:.2e} → PASS\n"
    "- 是否<3%：否。本单交付的是动态支路首个解析基准与当前缺口，不宣称收敛。\n")

report = {
    "title": "TensorSPH projection-SPH Young-Laplace：well-balanced 分裂是否保留形状动力学（模态解析校验）",
    "summary": (
        "锚点 XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE：用二维液滴 mode-2 解析频率 "
        "ω²=σn(n²−1)(k^2n−1)/(2ρR³k^2n) 对投影 SPH 的两条力臂做单步线性响应判定。"
        f"结论一：round-3 的 well-balanced 臂把 u*=dt·Gq 精确投影为零（R2 最小 {res['min_dynamics_retention']:.1e}），"
        "任何形状（含大幅变形）都是冻结不动点，静态 3e-7 不是「擎住」而是「不能动」。"
        f"结论二：出货 CSF 臂的 mode-2 频率偏低 {worst_robust:.1f}%（a/b≥1.05 最差），8→10 分辨率无收敛趋势，"
        "动态支路维持 open / honest FAIL 并已定位机理。"),
    "scenario_key": "xval_projection_sph_young_laplace",
    "mission_id": 23080,
    "tags": ["TensorSPH", "surface-tension", "Young-Laplace", "projection-SPH", "verification",
             "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE", "honest-FAIL"],
    "markdown": md,
    "figures": [{"name": "modal_transferability.png",
                 "caption": "mode-2 频率 vs 解析带 / well-balanced 臂保留的动力学 / 丢弃的力与丢弃的动力学",
                 "png_base64": png}],
}
Path("/tmp/report.json").write_text(json.dumps(report))
print("report json bytes:", len(json.dumps(report)), "| worst_robust_pct:", round(worst_robust, 1))