"""Build the mission report payload (markdown + base64 figures) for the platform.

The numbers come from ``runs/wave_tank/verdict.json`` and the per-case JSON files,
so the report repeats the evidence instead of retyping it. The generated figures
are re-saved at a small size (the report endpoint silently rejects payloads above
~250 KB, and an unreduced matplotlib PNG is 150-400 KB on its own).

    python benchmarks/wave_tank/make_report.py --out /tmp/report.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
RUNS = os.path.join(ROOT, "runs", "wave_tank")
FIGURES = os.path.join(ROOT, "docs", "figures")
MISSION = 5511

FIGURE_SPECS = (
    ("wave_tank_dispersion.png",
     "图 1 四个 kh 下的相位速度/线性理论（灰带 ±3%）与两点相位差给出的波数。"),
    ("wave_tank_ladder.png",
     "图 2 分辨率阶梯：色散误差 vs dp/h（log–log，附拟合阶数）与误差—成本。"),
    ("wave_tank_decay.png",
     "图 3 波高沿程衰减：0.5λ 到 10λ，虚线为 H/H0 = 0.95 门槛。"),
    ("wave_tank_field.png",
     "图 4 三个时刻的粒子云（按速度着色）与自由面高程剖面。"),
    ("wave_tank_solitary.png",
     "图 5 孤立波剖面（解析 vs 实测）与 crest 到达时间给出的相位速度。"),
)


def shrink(path: str, max_bytes: int = 42000) -> tuple[str, int] | None:
    """Return ``(base64, bytes)`` of a PNG, re-rendered smaller if needed."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) <= max_bytes:
        return base64.b64encode(raw).decode("ascii"), len(raw)
    try:
        from PIL import Image
    except ImportError:
        return base64.b64encode(raw).decode("ascii"), len(raw)
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = 0.7
        while True:
            buffer = io.BytesIO()
            resized = image.resize((max(200, int(image.width * scale)),
                                   max(150, int(image.height * scale))))
            resized.save(buffer, format="PNG", optimize=True)
            if buffer.tell() <= max_bytes or scale < 0.25:
                data = buffer.getvalue()
                break
            scale *= 0.75
    return base64.b64encode(data).decode("ascii"), len(data)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/report.json")
    parser.add_argument("--markdown", default=os.path.join(ROOT, "docs", "benchmarks",
                                                           "wave-tank-2d.md"))
    args = parser.parse_args()

    summary = {}
    verdict_path = os.path.join(RUNS, "verdict.json")
    if os.path.exists(verdict_path):
        summary = json.load(open(verdict_path, encoding="utf-8"))
    criteria = summary.get("criteria", {})
    passed = [key for key, item in sorted(criteria.items()) if item.get("passed")]

    figures = []
    for name, caption in FIGURE_SPECS:
        encoded = shrink(os.path.join(FIGURES, name))
        if encoded is None:
            continue
        figures.append({"name": name, "caption": caption, "png_base64": encoded[0]})
        print(f"{name}: {encoded[1]} bytes")

    body = read(args.markdown) if os.path.exists(args.markdown) else ""
    header = (
        "# TensorSPH 弱可压波浪水槽：造波—色散—消波三锚 + 分辨率/人工声速定标\n\n"
        f"**判据通过**：{', '.join(passed) if passed else '见正文'}；"
        f"**未通过/WITHHELD**："
        f"{', '.join(k for k in sorted(criteria) if not criteria[k].get('passed')) or '无'}\n\n"
        "机器可读证据：`runs/wave_tank/<case>.json`（配置+探针统计+判据+成本）、"
        "`runs/wave_tank/verdict.json`（判据矩阵）。全部数字由 "
        "`benchmarks/wave_tank/summarise.py` 与 `make_doc.py` 程序化注入。\n\n"
        "---\n\n")
    payload = {
        "title": "P76-N11 TensorSPH 弱可压波浪水槽与颗粒离散误差定标",
        "summary": ("在 TensorSPH 中建立第一个含波的验证面：2-D WCSPH 数值波浪水槽"
                    "（松弛区/活塞造波 + 海绵层消波 + 柱体积分探针 + 单频锁相分析），"
                    "把造波传递函数、线性色散、传播衰减、反射系数、孤立波相位速度/波形、"
                    "分辨率阶梯与人工声速扫描逐项对标解析解，并如实标注未通过项与跨求解器"
                    "互裁 WITHHELD 的理由。"),
        "scenario_key": "ship-performance/tensor-sph-wave-tank",
        "mission_id": MISSION,
        "tags": ["TensorSPH", "WCSPH", "numerical-wave-tank", "dispersion",
                 "sponge-layer", "solitary-wave", "convergence", "honest-FAIL"],
        "markdown": header + body,
        "figures": figures,
    }
    text = json.dumps(payload)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"{args.out}: {len(text)} bytes, {len(figures)} figures")
    if len(text) > 250000:
        print("WARNING: payload above the ~250 KB endpoint limit", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
