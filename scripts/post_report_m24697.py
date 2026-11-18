"""Post the m24697 (TensorSPH#1029 reopen, mission m21860) progress report.

The report documents the FSI2 periodic-window VIV verdict pipeline: the
deliverables (make_viv_report / run_body_force_transient /
verdict_body_force_transient + VIV_PERIODIC_VERDICT.md + assets + guard test)
are already merged into main via PR #839 (m21860); this reopen re-verifies the
full pipeline and re-states the honest WITHHELD verdict.
"""
import json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        print("HTTP", exc.code, body[:1500])
        raise


tok = post(f"{BASE}/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token")
if not token:
    raise SystemExit("no token: " + json.dumps(tok)[:300])

ROOT = pathlib.Path(__file__).resolve().parents[1]

def b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode()

import base64

fig_dx005 = b64(ROOT / "docs/xval_fsi2/assets/viv_dx005.png")
fig_bars = b64(ROOT / "docs/xval_fsi2/assets/viv_error_bars.png")

markdown = """# FSI2 周期性窗口判定：强耦合长时瞬态 vs Turek–Hron 基准（重开·TensorSPH#1029·第2轮）

**锚点**：`XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`　**仓**：TensorSPH　**域**：SPH / FSI
**mission**：m21860（重开 m24697）　**状态**：交付物已并入 main（PR #839），本轮重开做全链路复核。

## ① 案例介绍

Turek–Hron FSI2（LNCSE 53, 2006）：2-D 通道 2.5 × 0.41 m，D = 0.1 m 刚性圆柱（心 (0.2, 0.2)）
+ L = 0.35 m、t = 0.02 m 弹性梁，Re = 100，ρ_s/ρ_f = 1。入口抛物型，U_mean = 0.2 m/s。

| 量 | 基准值 | 出处 |
|---|---|---|
| 梁尖 `tip_x_amplitude` | 0.0028 m | Turek–Hron 2006（`fsi2_reference.py`）|
| 梁尖 `tip_y_amplitude` | 0.0082 m | 同上 |
| `drag_mean` | 3.2 | 同上 |
| `drag_amplitude` | 0.96 | 同上 |
| `lift_amplitude` | 0.22 | 同上 |

## ② 怎么做的

驱动 `benchmarks/xval_fsi2/run_body_force_transient.py`（强耦合子迭代，ρ_s/ρ_f=1 松耦合一步发散），
加速器 **`aitken`**（每步重启 secant 记账，依据 `COUPLING_CONVERGENCE_BUDGET.md`）。
判定由 `verdict_body_force_transient.build_verdict` 给出（与上一轮同一道门，未改阈值/判据）；
报告由 `make_viv_report.py` 从工件程序化生成（所有数字与图直接取自 JSON，零手抄）。

| 运行 | dx [m] | 步数 | 物理时长 [s] | dt [s] | 界面 | 平均子迭代 | 收敛步占比 | 中止 |
|---|---|---|---|---|---|---|---|---|
| `viv_dx005_aitken.json` | 0.05 | 10000 | 26.000 | 2.600e-03 | aitken | 2.13 | 0.991 | 否 |
| `viv_dx002_aitken.json` | 0.02 | 6000 | 6.240 | 1.040e-03 | aitken | 2.08 | 1.000 | 否 |

## ③ 计算结果

### dx = 0.05 m（窗口 t = 7.803 … 26.000 s）
* 主导周期 **2.1372 s**，窗口覆盖 **8.51 周期** → `regime_ok = True`
* **可判定性** `admissible = False`：⛔ `blowup` max|v| = 12.04 m/s = **2.41 × c0**；
  ⛔ `body_containment` |tip| = 1.238 m = **3.54 × 梁长** → 判定降级 `WITHHELD`
* 最坏相对误差：梁尖 +31276.35 %，力系数 +109547154.21 %（**非物理解，不作精度结论**）

### dx = 0.02 m（窗口 t = 1.873 … 6.240 s）
* 主导周期 **0.2018 s**，出 VIV 带 [1.67, 3.85] s → `regime_ok = False`（ringing，非 shedding）
* **可判定性** `admissible = False`：⛔ `period_band` → 判定降级 `WITHHELD`
* 最坏相对误差：梁尖 +94.87 %，力系数 +2491.64 %

**两腿都不成立**：dx=0.05 发散（炸）、dx=0.02 过阻尼（死），方向相反 → 强耦合路径在两个
分辨率上都没收敛到物理解；`dx` 不是这里的控制参数。锚点验收（<15 %）**本轮仍未达成**，
`WITHHELD` 不是 PASS。

## ④ 结果图

![viv_dx005.png](assets/viv_dx005.png)

![viv_error_bars.png](assets/viv_error_bars.png)

## ⑤ 注意点 / 本轮真正修掉的缺陷

1. **判定门第二个假阳性**：收敛包络 ≠ 物理解。上一轮修掉「启动瞬态被判 FAIL」（包络平稳性
   [0.8, 1.25]），本轮暴露反向漏洞——**发散的场同样有平稳包络**（它饱和了）。修法：在统计
   判据之外增加**可判定性（admissibility）筛检**（finite / blowup / body_containment /
   period_band，全部读自工件自带量或基准几何，无自由参数），不满足一律降级 `WITHHELD`。
2. **`period_band` 依赖分析窗口**：dx=0.05 在 warmup=0.5 识别 0.320 s 出带、warmup=0.3 识别
   2.137 s 落带内——该带时灵时不灵，**不能**单独作否决依据；挡下它的是与窗口无关的
   `blowup`（2.41×c0）和 `body_containment`（3.54×梁长）。
3. **骨架离散被 `max(15,…)` 静默夹住**：dx 0.05→0.02 流体细 2.5×、梁只细 1.1×，这条梯子
   不是收敛研究。夹紧保留但**写进工件**（`beam_nodes_clamped`、`beam_spacing_over_dx`）。
4. **下一步**：不是加分辨率，而是先让 dx=0.02 的场有物理量级 `Cd`（0.05→3.2 差 60 倍，
   指向载荷通道/体积力标度），再谈周期窗口。

## ⑥ 误差判定（显式回答「是否 <3 %」）

| 量 | 门槛 | 本次最坏相对误差 | <15 % | <3 % | 可判定 |
|---|---|---|---|---|---|
| dx=0.05 tip | 15 % | +31276.35 % | 否 | 否 | 否（非物理解）|
| dx=0.05 fit_arm | 15 % | +109547154.21 % | 否 | 否 | 否（非物理解）|
| dx=0.02 tip | 15 % | +94.87 % | 否 | 否 | 否（非物理解）|
| dx=0.02 fit_arm | 15 % | +2491.64 % | 否 | 否 | 否（非物理解）|

**显式回答「是否 <3 %」：否，且不可判定。** 两个长时工件都不是 Turek–Hron 解（一个炸、一个死），
相对误差仅作取证保留（`comparable: false`），不作精度结论。这是 honest-FAIL：锚点验收未达成，
`WITHHELD` 如实记录，不粉饰。

## 复核

`python -m pytest tests/test_fsi2_viv_verdict_pipeline.py tests/test_fsi2_verdict_admissibility.py -q`
→ **14 passed**（含 5 个本管道守护：driver 转发 `--interface-solver`、startup-only 窗口必须
`WITHHELD`、报告 markdown 携带工件数字、增长振荡被包络平稳性降级、Strouhal 带报告）。
"""

body = {
    "title": "FSI2 周期性窗口 VIV 判定：强耦合长时瞬态 vs Turek–Hron（重开复核，WITHHELD 如实）",
    "summary": (
        "重开 TensorSPH#1029（mission m21860）：FSI2 周期性窗口 VIV 判定管道（make_viv_report / "
        "run_body_force_transient / verdict_body_force_transient + VIV_PERIODIC_VERDICT.md + 图 + "
        "守护测试）已并入 main（PR #839）。本轮全链路复核：14 测试全过；两个长时工件（dx=0.05 "
        "发散 2.41×c0、dx=0.02 过阻尼出带）均不满足可判定性，判定如实 WITHHELD，锚点验收未达成。"
    ),
    "scenario_key": "sph_fsi_viv_periodic",
    "mission_id": 24697,
    "tags": ["SPH", "FSI", "VIV", "Turek-Hron", "strong-coupling", "aitken", "WITHHELD", "verification"],
    "markdown": markdown,
    "content": markdown,
    "figures": [
        {"name": "viv_dx005.png", "caption": "FSI2 dx=0.05 强耦合 aitken 长时瞬态：梁尖 y 位移与 Cd/Cl 力系数 vs Turek–Hron 基准（窗口起点虚线）", "png_base64": fig_dx005},
        {"name": "viv_error_bars.png", "caption": "FSI2 报告量相对误差条形图 vs Turek–Hron 基准（15% 验收门 / 3% 报告目标）", "png_base64": fig_bars},
    ],
}
res = post(f"{BASE}/tensor/reports", body, token)
pathlib.Path("out").mkdir(exist_ok=True)
pathlib.Path("out/report_post_m24697.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res)[:500])
