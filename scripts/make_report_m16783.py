"""Build /tmp/report.json for mission m16783 (Kleefsman dt-defect family + grid ladder).

Numbers are read from the committed artefacts (no hand-typing); the markdown body is
the verification document itself, and the figure is embedded as base64.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures" / "kleefsman_grid_ladder_c080.png"
DOC = ROOT / "docs" / "verification_kleefsman_dt_sound_speed_family.md"
LADDER = ROOT / "benchmarks" / "kleefsman_obstacle" / "results" / "c0_scan" / "grid_ladder_c080.json"

res = json.loads(LADDER.read_text())
pk = res["peak_ladder"]
peaks = ", ".join(f"{v:.3f}" for v in pk["peak_height_m"])
comp = 100.0 * res["compression"]["max_density_compression"]

report = {
    "title": ("Kleefsman 障碍物溃坝（TensorSPH）：声速字面量 dt 缺陷族清零 + 第三级网格阶梯"
              "（c0=80 m/s，峰值振荡未收敛）"),
    "summary": (
        "修复 chunked.py/convergence.py 中与 run.py 同族的「dt 推导写死 c0=20 m/s」缺陷"
        "（c0>78 m/s 时静默越过声学 Courant 限），34 项回归测试绿（含 AST 守卫）；"
        f"新增第三级网格 dx=0.01 m @ c0=80 m/s：峰值 {peaks} m，判定 {pk['verdict']}"
        "（符号翻转），拒绝给观测阶/GCI；压缩 "
        f"{comp:.2f}% 落在 5% 可控线内。欠账「峰值随分辨率非单调」未被高 c0 消除——"
        "分辨率一致性仍为未收敛；实验值仍 WITHHELD。"),
    "scenario_key": "sph-kleefsman-obstacle-grid-ladder",
    "mission_id": 16783,
    "tags": ["TensorSPH", "Kleefsman", "dam-break", "grid-convergence", "WCSPH",
             "defect-family", "honest-FAIL"],
    "markdown": DOC.read_text(),
    "figures": [{
        "name": "kleefsman_grid_ladder_c080.png",
        "caption": ("左：dx∈{0.04,0.02,0.01} 的探针峰值液位阶梯（c0=80 m/s，符号翻转→振荡，不给阶）；"
                    f"右：同级密度压缩 vs 5% 可控线（最细级 {comp:.2f}% 贴线通过）"),
        "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
    }],
}
out = Path("/tmp/report.json")
out.write_text(json.dumps(report))
print(f"wrote {out} markdown_chars={len(report['markdown'])} "
      f"fig_kb={len(report['figures'][0]['png_base64']) // 1024}")