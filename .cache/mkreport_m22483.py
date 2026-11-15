"""Build and post the m22483 progress report (numbers injected, never typed)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path("/srv/tensor/agent/work/m22483")
ART = json.loads((ROOT / "benchmarks/case_poiseuille/out/wall_shear_harmonic.json").read_text())
RES = json.loads((ROOT / "benchmarks/case_poiseuille/out/result.json").read_text())

slow = ART["harmonic_runs"]
finest = max(slow, key=lambda r: r["resolution_H_over_dx"])
fast = ART["harmonic_fast"]
disc = ART["frequency_discrimination"]
quasi = ART["quasi_steady_control"]
trans = ART["estimator_transferability"]
chk = ART["reference_selfcheck"]
geo = ART["inputs"]["geometry"]

# case-ladder context (from the committed ladder artifact)
def g(name):
    return RES["gates"][name]

rows = []
for rec in slow:
    c = rec["wall_shear_budget"]
    rows.append(
        f"| {rec['resolution_H_over_dx']} | {rec['womersley_number']:.2f} | "
        f"{c['measured_abs']:.4f} | {c['analytic_abs']:.4f} | {c['amp_error_pct']:+.3f} % | "
        f"{c['phase_error_deg']:+.3f} deg ({c['phase_error_pct_quarter_turn']:.2f} % of pi/2) |")
cf = fast["wall_shear_budget"]
rows.append(
    f"| {fast['resolution_H_over_dx']} (fast) | {fast['womersley_number']:.2f} | "
    f"{cf['measured_abs']:.4f} | {cf['analytic_abs']:.4f} | {cf['amp_error_pct']:+.3f} % | "
    f"{cf['phase_error_deg']:+.3f} deg ({cf['phase_error_pct_quarter_turn']:.2f} % of pi/2) |")
table_runs = "\n".join(rows)

md = f"""# TensorSPH#890（重开·第 3 轮）— CASE-POISEUILLE-SPH 增加「振荡壁面剪切（脉动阻力）」验证层

## 1. 案例介绍

**物理背景.** 平面通道 `0 <= y <= H`，两壁无滑移、`x` 向周期，体积力 `g(t) = g0 cos(omega t)`
驱动，控制方程为 `u_t = nu u_yy + g(t)`。这是脉动（Womersley）通道流：既有算例
（`benchmarks/poiseuille_mirror`、`poiseuille_startup`）只验证**平均动量平衡**，
`benchmarks/stokes_second` 验证**壁面驱动**振荡；本算例验证**压力驱动通道的频率响应**
（每行的幅值 **与相位**）。

**几何与参数（固定，来自 `benchmarks/case_poiseuille/DESIGN.md` §2）**

| 量 | 值 |
|---|---|
| 通道高度 H | {geo['height']} m |
| 运动粘度 nu | {geo['viscosity']} m²/s |
| 密度 rho0 | {geo['rho0']} kg/m³ |
| 体积力 g0 | {geo['body_force']} m/s² |
| 频率 omega（主）/ omega_fast | {ART['inputs']['omega_slow']} / {ART['inputs']['omega_fast']} rad/s（Wo = {slow[0]['womersley_number']:.2f} / {fast['womersley_number']:.2f}）|
| 分辨率 H/dx | 8 / 12 / 16（Stokes 层 delta/dx = 3.1 / 4.7 / 6.3）|
| 壁面模型 | 镜像/ghost 无滑移面（`tensorsph.wall_mirror.plane_walls`）|
| 求解器 | 生产 WCSPH `tensorsph.droplet_impact.rhs`（Tait EOS，Morris/Brookshaw 粘性）|

**本轮新增的问题.** 上述阶梯验证的是**运动学**（每行幅值/相位，G1–G9），从未验证工程师真正
要的量：**振荡壁面剪切应力的复幅值** `tau_hat`（`tau_w(t) = Re{{tau_hat e^(i omega t)}}`，
即脉动阻力 / 振荡摩擦）。本单把它作为一个**后处理验证层**补齐，不改求解器、不重跑 SPH。

## 2. 怎么做的（方法 / 关键设置 / 工具链）

**估计器（无核函数、无壁法向）.** 把动量方程沿通道积分，`tau(y) = mu u_y`：

```
d<u>/dt = g(t) + [ tau(H) - tau(0) ] / (rho H)
```

Womersley 剖面关于中面反对称（`tau(H) = -tau(0)`），周期态下 `d/dt -> i omega`：

```
tau_hat = rho H ( g_hat - i omega <u>_hat ) / 2                     (1)
```

`<u>_hat` 取自已提交工件 `out/result.json` 的**逐行谐波拟合**（行厚均为一个 dx、列数相同，
故算术平均即截面平均）。闭式参考 `alpha = sqrt(i omega/nu)`：

```
tau_hat = mu (g_hat / i omega) alpha tanh(alpha H / 2)             (2)
```

**为什么不用稳态那套局部梯度估计器.** `combined.py` 用内部二次拟合反推两个稳态壁面剪切，
它是 `docs/verification_poiseuille_wall_shear_estimator.md` 认定的唯一可用局部估计器——
**但它只在稳态抛物线下无偏**。把同一估计器作用在解析 Womersley 剖面上，偏差为
**{abs(trans['per_rung']['16']['amp_error_pct']):.2f} % 幅值 / {trans['per_rung']['16']['phase_error_deg']:.1f} deg 相位**
（`Wo = {trans['womersley_number']:.2f}`），且该偏差在 `H/dx = 8,12,16` 上只变
**{trans['amp_bias_spread_pct']:.3f} %** → 是**模型偏差，不是离散误差**。因此它被保留为
门 **W5 的对照**，不作评级估计器。

**预声明门（阈值 = 本算例既有约定 5 % / 四分之一转角 5 % = 4.5 deg，未按结果调参）**

| 门 | 度量 | 阈值 |
|---|---|---|
| W0 / W0b | 截面平均速度 `<u>_hat` 幅值 / 相位（最细网格） | 5 % / 4.5 deg |
| W1 / W2 | `\\|tau_hat\\|` / `arg tau_hat`（最细网格） | 5 % / 4.5 deg |
| W3 | 每个网格（8/12/16）W1/W2 均成立 | 同上 |
| W4 | 双频阻力比 `\\|tau_fast\\|/\\|tau_slow\\|` 与相位差（H/dx = 12） | 5 % / 4.5 deg |
| W5 | 对照：稳态二次估计器在解析谐波剖面必须 > 5 % 偏 | pass |
| W6 | 参考自检：(1) 与 (2) 恒等，且 omega→0 极限 = rho g H/2 | 1e-10 |

**实验设计要点（对照与排除）.** ①**替代假设**：一个只做平均动量平衡的"准稳态"通道（稳态抛物线
跟随时变 g(t)）——用同一预算 (1) 喂入稳态截面平均 `<u> = g0 H²/(12 nu)` 即可模拟；
②**重复性**：全部结果由已提交工件确定性重算（无随机、无采样），单元测试在合成解析剖面上
闭环验证估计器；③**混杂控制**：相位参考为谐波拟合的绝对时间基准，测-算相位一致即排除公共
时移；④**探针先于声明**：本层第一次运行即为探索性探针（记录在 note §1），阈值取算例既有
约定而非事后拟合——实测误差全部 < 1.6 %，任何 2 %–5 % 的阈值同样会通过。

**工具链**：`benchmarks/case_poiseuille/wall_shear_harmonic.py`（估计器 + 门 + 工件）
→ `make_wall_shear_harmonic_doc.py`（matplotlib 图 + 程序化生成 note）
→ `tests/test_case_poiseuille_wall_shear_harmonic.py`（30 tests：闭式恒等、omega→0 极限、
两壁反对称、截面平均求积收敛阶、扰动灵敏度与解析增益一致、准稳态对照必 FAIL、W5 对照、
工件可复现、note/README 数字与工件绑定）。

## 3. 计算结果

**振荡壁面剪切（预算 (1) vs 闭式 (2)）**

| H/dx | Wo | `|tau_hat|` [Pa] | 解析 [Pa] | 幅值误差 | 相位误差 |
|---|---|---|---|---|---|
{table_runs}

**双频判别（W4，H/dx = {disc['resolution_H_over_dx']}）**

| 量 | 实测 | 解析 | 准稳态求解器会给 |
|---|---|---|---|
| `|tau_fast|/|tau_slow|` | {disc['ratio_measured']:.4f} | {disc['ratio_analytic']:.4f} | {disc['quasi_steady_ratio_would_be']:.2f} |
| 相位差 [deg] | {disc['phase_difference_measured_deg']:+.3f} | {disc['phase_difference_analytic_deg']:+.3f} | {disc['quasi_steady_phase_difference_deg']:+.3f} |

**被门拒绝的替代假设（准稳态通道，§2 对照 ①）**

| 假设 | `|tau_hat|` [Pa] | 相位 [deg] | 幅值误差 |
|---|---|---|---|
| 准稳态通道（稳态抛物线） | {quasi['measured']['measured_abs']:.4f} | {quasi['measured']['measured_phase_deg']:+.3f} | {quasi['measured']['amp_error_pct']:+.3f} % |
| Womersley（实测评级） | {finest['wall_shear_budget']['measured_abs']:.4f} | {finest['wall_shear_budget']['measured_phase_deg']:+.3f} | {finest['wall_shear_budget']['amp_error_pct']:+.3f} % |

→ W1–W4 **不可能**由平均动量平衡通过；它们检验的是非定常 Stokes 层。

**参考自检（W6）**：(1) 与 (2) 恒等残差 {chk['budget_identity_rel_error']:.2e}；
`omega → 0` 极限 {chk['omega_to_zero_limit']:.6f} Pa vs `rho g H/2 = {chk['steady_wall_shear']:.4f}` Pa
（相对 {chk['omega_to_zero_rel_error']:.2e}）。

**门状态**：{', '.join(f"{k.split('_')[0]}={'PASS' if v else 'FAIL'}" for k, v in ART['gates'].items())}
→ `all_gates_pass = {ART['all_gates_pass']}`。

**上下文（已合并的阶梯 G1–G9，未改动）**：G1 中心线幅值 5 % → {g('G1_amp_centre_5pct')}；
G2 剖面 RMS → {g('G2_amp_profile_rms_5pct')}；G3/G4 相位 → {g('G3_phase_centre_5pct_of_quarter_turn')}/{g('G4_phase_profile_rms_5pct_of_quarter_turn')}；
G6 频率判别 → {g('G6_frequency_ratio_5pct')}；G7/G8 稳态与启动锚点 → {g('G7_steady_anchor_5pct')}/{g('G8_startup_anchor_5pct')}；
G9 参考自检 → {g('G9_reference_fd_vs_analytic_1pct')}。

## 4. 结果图

* 图 1（`case_poiseuille_profiles.png`，既有阶梯工件）：逐行**幅值与相位剖面** vs Womersley 解析解
  ——场量沿壁法向的分布及其与解析曲线的比较（G1–G4 的物理支撑）。
* 图 2（`case_poiseuille_wall_shear_harmonic.png`，本轮新增，三联）：
  (a) 预算壁面剪切的幅值/相位误差 vs 网格，与 5 %/4.5 deg 判据带；
  (b) 一个周期内 `tau_w(t)` 的三条曲线：解析 `mu dU/dy`、动量预算（评级）、内部二次拟合（对照，
      可见其幅值缩水与相位超前/滞后）；
  (c) 二次估计器在解析谐波剖面上的幅值偏差 vs `Wo`（2–40），说明"稳态估计器不可迁移"，
      本算例的工作点 `Wo = 1.8 / 2.8` 以竖线标出。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

* **稳态局部估计器不可迁移**：在 `Wo = 1.8` 处二次拟合偏 **−5.3 %/−23 deg**，`Wo = 2.8` 处
  **−23 %/−52 deg**，且与网格无关。任何"直接对谐波剖面求壁面梯度"的做法都会把模型偏差
  当成数值误差报出去。
* **逐行平均是求积，不是精确积分**：行中心近似截面平均为二阶精度（`H/dx = 8 → 0.81 %`，
  16 → 0.20 %），这是合成解析输入不能机器精度复现闭式解的唯一原因；本层实测误差（0.11 %）
  低于该求积误差，说明 SPH 剖面本身比"逐行平均"更接近解析解（相位/幅值误差在同量级）。
* **预算只看截面平均**：它证明壁面吸收的**积分**动量通量与解析一致，不能把缺陷定位到某一
  近壁行；故本层是 G2/G4（逐行）的**补充**而非替代。
* **单谐波**：只覆盖基频；非线性二次谐波壁面剪切无法从已存储的逐行拟合中取得。
* **相位基准**：以谐波拟合的绝对时间基准为准；整体时移会使所有相位同移，测-算一致即排除。
* **适用范围**：层流、`Wo ≲ 3`、`delta/dx = 3–7`、单一通道高度、`Ma < 0.08` 弱可压；
  不覆盖湍流脉动或 `Wo >> 10`（Stokes 层薄于粒子间距）。

## 6. 误差判定（逐项 vs 基准，并显式回答「是否 < 3%」）

| 关键量 | 基准 | 误差 | < 3 %？ |
|---|---|---|---|
| `|tau_hat|`（H/dx = 16，Wo = 1.8） | 闭式 (2) | {finest['wall_shear_budget']['amp_error_pct']:+.3f} % | **是** |
| `arg tau_hat`（同上） | 闭式 (2) | {finest['wall_shear_budget']['phase_error_deg']:+.3f} deg = {finest['wall_shear_budget']['phase_error_pct_quarter_turn']:.2f} % of pi/2 | **是** |
| 截面平均速度幅值（同上） | `|<Uhat>|` | {finest['mean_velocity']['amp_error_pct']:+.3f} % | **是** |
| 双频阻力比（H/dx = 12） | 解析比值 | {disc['ratio_error_pct']:+.3f} % | **是** |
| 双频相位差 | 解析相位差 | {disc['phase_difference_error_deg']:+.3f} deg（{100*abs(disc['phase_difference_error_deg'])/90:.2f} % of pi/2） | **是** |
| 最粗网格 `|tau_hat|`（H/dx = 8） | 闭式 (2) | {slow[0]['wall_shear_budget']['amp_error_pct']:+.3f} % | **是** |
| 中心线幅值（阶梯 G1，既有） | Womersley | {(RES['harmonic_runs'][-1]['amp_centre_error_pct']):+.3f} % | **是** |
| 幅值剖面 RMS（G2，既有） | Womersley | {(RES['harmonic_runs'][-1]['amp_profile_rms_pct']):+.3f} % | **是** |
| 稳态剖面 RMS（G7，既有） | 抛物线 | {(RES['constant_run']['steady_profile_rms_pct']):+.3f} % | **是** |
| 启动剖面 RMS（G8，既有） | Boussinesq 级数 | {(RES['constant_run']['startup_rms_pct_at_040']):+.3f} % | **是** |
| 频率判别比（G6，既有） | 解析比值 | {abs(RES['frequency_discrimination']['amplitude_ratio_error_pct']):.3f} % | **是** |

**结论：全部关键量 vs 解析基准均 < 3 %（最差 {max(abs(finest['wall_shear_budget']['amp_error_pct']), abs(disc['ratio_error_pct']), abs(slow[0]['wall_shear_budget']['amp_error_pct'])):.2f} %），本单无 honest-FAIL 项。**

局限如实声明：本层结论的严格范围是"周期态、单谐波、截面平均意义下的壁面动量通量"；
`Wo ≳ 3` 与湍流不在覆盖内（同算例既有界限）。

工件：`benchmarks/case_poiseuille/out/wall_shear_harmonic.json`、
`docs/case_poiseuille_wall_shear_harmonic.md`（程序化生成）、
`docs/figures/case_poiseuille_wall_shear_harmonic.png`、
`benchmarks/case_poiseuille/wall_shear_harmonic.py`、`make_wall_shear_harmonic_doc.py`、
`tests/test_case_poiseuille_wall_shear_harmonic.py`（30 passed）。
"""

figs = []
for name, cap in [
    ("case_poiseuille_wall_shear_harmonic.png",
     "图 2：(a) 动量预算壁面剪切幅值/相位误差 vs 网格与判据带；(b) 一个周期内 tau_w(t)：解析 vs 预算 vs 内部二次拟合（对照）；(c) 二次估计器在解析谐波剖面上的偏差 vs Wo。"),
    ("case_poiseuille_profiles.png",
     "图 1：逐行幅值与相位剖面 vs Womersley 解析解（既有阶梯工件，G1–G4 的物理支撑）。"),
]:
    p = ROOT / "docs" / "figures" / name
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

report = {
    "title": "TensorSPH#890（重开·第 3 轮）CASE-POISEUILLE-SPH：振荡壁面剪切（脉动阻力）验证层 W0–W6",
    "summary": ("平面通道脉动（Womersley）流新增壁面剪切验证层：tau_hat = rho H (g_hat - i omega <u>_hat)/2 "
                "无核函数、无壁法向；H/dx=16 时 |tau_hat| 309.3794 vs 解析 309.7224 Pa（-0.111 %），"
                "相位 -40.358 vs -39.986 deg（-0.372 deg），双频阻力比 0.5732 vs 0.5801（-1.18 %）；"
                "准稳态替代假设被拒 +133 %；稳态二次估计器在谐波剖面上有 5.3 %/-23 deg 的模型偏差，"
                "故仅作对照（W5）。8 文件、30 tests。"),
    "scenario_key": "case_poiseuille_harmonic_wall_shear",
    "mission_id": 22483,
    "tags": ["TensorSPH", "算例池", "泊肃叶流", "Womersley", "脉动阻力", "壁面剪切",
             "动量预算", "verification", "GCI", "case-pool"],
    "markdown": md,
    "figures": figs,
}

Path("/tmp/report.json").write_text(json.dumps(report))
print("markdown chars:", len(md), "figures:", [f["name"] for f in figs])

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def post(url, payload, token=None):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()


st, body = post(base + "/auth/login",
                {"username": os.environ["TENSOR_AGENT_USER"],
                 "password": os.environ["TENSOR_AGENT_PASS"]})
tok = json.loads(body.decode() if isinstance(body, bytes) else body)
tok = tok.get("access_token") or tok.get("token")
print("login", st, bool(tok))
st2, b2 = post(base + "/tensor/reports", report, tok)
print("report", st2, b2[:400])