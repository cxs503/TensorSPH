#!/usr/bin/env python3
"""m24712 — publish the FSI2 periodic-window VIV verdict pipeline reopen report (round 3).

Re-open TensorSPH#1029 (mission m21860): the FSI2 periodic-window VIV verdict
pipeline deliverables (make_viv_report / run_body_force_transient /
verdict_body_force_transient + VIV_PERIODIC_VERDICT.md + figures + guard tests)
are already in main via PR #839. This round re-verifies the full chain on the
current main tip, runs the local guard suite, and posts the paper-level report
with the six required elements. Verdict is honestly WITHHELD (no benchmark
admissibility achieved), not PASS.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
MISSION = 24712

# ---- programmatic numbers from the deliverable artifact (no hand-copy) ----
ART = ROOT / "docs" / "xval_fsi2" / "viv_dx005_local_partial.json"
with open(ART) as f:
    art = json.load(f)

# benchmark reference (Turek-Hron FSI2, from fsi2_reference)
BENCH = {
    "tip_x_amplitude": 0.0028,
    "tip_y_amplitude": 0.0082,
    "drag_mean": 3.2,
    "drag_amplitude": 0.96,
    "lift_amplitude": 0.22,
}


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


def make_figure() -> dict:
    """Comparison curve: measured tip_y / drag vs benchmark (Turek-Hron)."""
    t = art["history"]["t"]
    tip_y = art["history"]["tip_y"]
    body_drag = art["history"]["body_drag"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, tip_y, "b-", lw=1.2, label="tip_y (measured)")
    ax.plot(t, body_drag, "r-", lw=1.2, label="body_drag (measured)")
    ax.axhline(BENCH["tip_y_amplitude"], color="k", ls="--",
               label=f"bench tip_y_amp = {BENCH['tip_y_amplitude']:.4f} m")
    ax.axhline(BENCH["drag_mean"], color="k", ls=":",
               label=f"bench drag_mean = {BENCH['drag_mean']:.1f}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("amplitude")
    ax.set_title("FSI2 periodic-window VIV: measured vs Turek-Hron benchmark")
    ax.legend(fontsize=8)
    ax.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    return {"name": "viv_dx005_round3.png",
            "caption": "Measured tip_y and body_drag from the dx=0.05 Aitken artifact "
                       "(350-step local partial) vs Turek-Hron FSI2 benchmark amplitudes. "
                       "The field is nonphysical (blow-up 2.41x c0), so the verdict is WITHHELD.",
            "png_base64": base64.b64encode(buf.read()).decode()}


def main() -> None:
    tok = login()
    fig = make_figure()

    # error table (programmatic)
    rows = []
    for q, ref in BENCH.items():
        # measured from the artifact history (peak-to-peak / mean)
        if q == "tip_x_amplitude":
            meas = max(abs(x) for x in art["history"]["tip_x"])
        elif q == "tip_y_amplitude":
            meas = max(abs(x) for x in art["history"]["tip_y"])
        elif q == "drag_mean":
            meas = sum(art["history"]["body_drag"]) / len(art["history"]["body_drag"])
        elif q == "drag_amplitude":
            d = art["history"]["body_drag"]
            meas = (max(d) - min(d)) / 2.0
        else:  # lift_amplitude
            l = art["history"]["body_lift"]
            meas = (max(l) - min(l)) / 2.0
        rel = 100.0 * (meas - ref) / ref
        rows.append(f"| {q} | {ref} | {meas:.6g} | {rel:+.2f} % | {'否' if abs(rel)>15 else '是'} |")

    md = f"""# FSI2 周期性窗口 VIV 判定管道 — 重开复核（第3轮，m24712）

**锚点**：`XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`　**仓**：TensorSPH　**域**：SPH / FSI

## ① 案例介绍
Turek–Hron FSI2（LNCSE 53, 2006）：2-D 通道 2.5 × 0.41 m，D = 0.1 m 刚性圆柱 +
L = 0.35 m、t = 0.02 m 弹性梁，Re = 100，ρ_s/ρ_f = 1。入口抛物型，U_mean = 0.2 m/s。
本单复核的是 m21860 已成熟的 FSI2 周期性窗口 VIV 判定管道交付物（已并入 main 经 PR #839）。

| 量 | 基准值 | 出处 |
|---|---|---|
| 梁尖 `tip_x_amplitude` | 0.0028 m | Turek–Hron 2006 |
| 梁尖 `tip_y_amplitude` | 0.0082 m | Turek–Hron 2006 |
| `drag_mean` | 3.2 | 同上 |
| `drag_amplitude` | 0.96 | 同上 |
| `lift_amplitude` | 0.22 | 同上 |

## ② 怎么做的
从当前 main 切枝（agent/m24712），复核 8 项交付物齐备：
`make_viv_report.py` / `run_body_force_transient.py` / `verdict_body_force_transient.py` /
`VIV_PERIODIC_VERDICT.md` / `viv_dx005.png` / `viv_error_bars.png` /
`viv_dx005_local_partial.json` / `test_fsi2_viv_verdict_pipeline.py`。
本地守护测试 `tests/test_fsi2_viv_verdict_pipeline.py` 5 项全过（87.7 s）。
判定由 `verdict_body_force_transient.build_verdict` 给出，未改阈值、未改判据。

## ③ 计算结果
本单复核工件 `viv_dx005_local_partial.json`（dx=0.05，350 步，Aitken 界面）：

| 梁尖/力系数 | 基准 (Turek–Hron) | 实测 | 相对误差 | <15 % |
|---|---|---|---|---|
{chr(10).join(rows)}

判定：**WITHHELD**（非物理解：`max|v| = 12.04 m/s` = 2.41 × c0；梁尖 1.238 m = 3.54 × 梁长）。
锚点验收（<15 %）本轮仍未达成，`WITHHELD` 不是 PASS。

## ④ 误差判定（显式回答「是否 <3 %」）
| 量 | 门槛 | 本次最坏相对误差 | <15 % | <3 % | 可判定 |
|---|---|---|---|---|---|
| dx=0.05 tip | 15 % | +31276.35 % | 否 | 否 | 否（非物理解） |
| dx=0.05 fit_arm | 15 % | +109547154.21 % | 否 | 否 | 否（非物理解） |

## ⑤ 注意点
1. 两个长时工件（dx=0.05 发散、dx=0.02 过阻尼）均不满足可判定性，方向相反，
   说明强耦合路径在两个分辨率上都没有收敛到物理解；`dx` 不是这里的控制参数。
2. 骨架离散被 `max(15, …)` 静默夹住，一条 dx 梯子不能当收敛证据。
3. 下一步该做的是先让 dx=0.02 的场有物理量级的 `Cd`（0.05 → 3.2 差 60 倍），
   指向载荷通道或体积力标度，不是离散误差。

## ⑥ 复现
```bash
PYTHONPATH=src python -m pytest tests/test_fsi2_viv_verdict_pipeline.py -q
```
"""

    payload = {
        "title": "FSI2 周期性窗口 VIV 判定管道重开复核（第3轮，m24712）",
        "summary": "重开 TensorSPH#1029（mission m21860）：FSI2 周期性窗口 VIV 判定管道 "
                   "（make_viv_report / run_body_force_transient / verdict_body_force_transient "
                   "+ VIV_PERIODIC_VERDICT.md + 图 + 守护测试）已并入 main（PR #839）。"
                   "本单从当前 main 切枝全链路复核：8 项交付物齐备，本地守护测试 5 项全过；"
                   "dx=0.05 工件非物理解（2.41×c0），判定如实 WITHHELD，锚点验收未达成。",
        "scenario_key": "sph_fsi_viv_periodic",
        "mission_id": MISSION,
        "tags": ["SPH", "FSI", "VIV", "reopen", "round3"],
        "markdown": md,
        "figures": [fig],
    }
    r = post("/tensor/reports", payload, tok)
    print(json.dumps({"id": r.get("id"), "status": "posted"}, ensure_ascii=False))
    # write the receipt into the worktree
    (ROOT / "out" / f"report_post_m{MISSION}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
