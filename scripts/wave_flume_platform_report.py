"""Assemble the platform progress report for ``CASE-WAVE-FLUME`` (JSON payload).

The body is the generated write-up (``docs/reports/wave_flume_case.md``), which
is itself rendered from the graded records by
``benchmarks/wave_flume/make_report.py``; the figures are the PNGs written by
``benchmarks/wave_flume/analyse.py``.  Nothing is typed in by hand here either:
the summary line is read out of ``docs/reports/wave_flume_numbers.json``.

    python scripts/wave_flume_platform_report.py --out /tmp/report.json \\
        --mission-id 15861 --scenario-key CASE-WAVE-FLUME
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read(path: Path, default=""):
    return path.read_text() if path.exists() else default


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numbers", default="docs/reports/wave_flume_numbers.json")
    ap.add_argument("--body", default="docs/reports/wave_flume_case.md")
    ap.add_argument("--figures", default="docs/figures/wave_flume")
    ap.add_argument("--out", default="/tmp/report.json")
    ap.add_argument("--mission-id", type=int, default=15861)
    ap.add_argument("--scenario-key", default="CASE-WAVE-FLUME")
    args = ap.parse_args(argv)

    numbers = json.loads((REPO / args.numbers).read_text())
    markdown = read(REPO / args.body)
    fig_dir = REPO / args.figures
    figures = []
    captions = {
        "series_wf_D20": "波高计时序：实测自由面 vs 目标波高带 ±H_target/2，及沿程 H(x) 与文献衰减 e^{-μx}",
        "series_wf_D28": "第二分辨率（H/dx 更大）的同一对照",
        "series_wf_D32": "最高分辨率工况的同一对照",
        "series_wf_D20_T08": "第二参数点：周期 0.8 s 的波高对照",
        "series_wf_D20_Hh010": "第二参数点：H/h = 0.10 的波高对照",
        "ladder": "分辨率阶梯：逐探头波高误差 vs H/dx（红线 = 10% 门）",
        "profile": "自由面剖面 η(x) 快照（试验段内波形）",
    }
    for png in sorted(fig_dir.glob("*.png")):
        figures.append({
            "name": png.stem,
            "caption": captions.get(png.stem, png.name),
            "png_base64": base64.b64encode(png.read_bytes()).decode(),
        })

    worst = numbers["worst_error_percent"]
    summary = (
        f"CASE-WAVE-FLUME 实跑判级：{numbers['cases_graded']} 个工况，"
        f"波高 vs 活塞传递函数目标最差误差 {worst:.2f}%（门 10%），"
        f"passed={numbers['passed']}"
    )
    payload = {
        "title": f"CASE-WAVE-FLUME 波槽波高保真与衰减归因（TensorSPH WCSPH，最差误差 {worst:.2f}%）",
        "summary": summary,
        "scenario_key": args.scenario_key,
        "mission_id": args.mission_id,
        "tags": ["TensorSPH", "CASE-WAVE-FLUME", "SPH", "wave-flume", "validation"],
        "markdown": markdown,
        "figures": figures,
    }
    Path(args.out).write_text(json.dumps(payload))
    print(f"wrote {args.out}: {len(markdown)} chars, {len(figures)} figures, "
          f"worst={worst:.2f}% passed={numbers['passed']}")
    assert math.isfinite(worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())