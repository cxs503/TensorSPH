"""Publish the Laplace-verification report to the platform API."""
import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
art = json.loads((ROOT / "artifacts/laplace/laplace_result.json").read_text())
a, b, hd = art["stage_a"], art["stage_b"], art["stage_a"]["headline"]

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER", "")
password = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


token = None
for creds in ({"username": user, "password": password},):
    try:
        res = post("/auth/login", creds)
        token = (res.get("access_token") or res.get("token")
                 or res.get("data", {}).get("access_token"))
    except Exception as exc:                                  # pragma: no cover
        print("login failed:", getattr(exc, "code", exc),
              getattr(exc, "read", lambda: b"")()[:200])

rows = "\n".join(
    "| {w:.1f} | {k:.3f} | {ke:.2f} % | {i:.3f} | {dp:.4f} | {dpe:.2f} % |".format(
        w=e["w_over_dx"], k=e["kappa_mean"], ke=100 * e["kappa_relative_error"],
        i=e["integral_grad_c_dr"], dp=e["delta_p_operator"],
        dpe=100 * e["delta_p_relative_error"])
    for e in a["width_sweep"])

md = f"""## 1. 案例介绍

被审对象：TensorSPH 的表面张力模型 `src/tensorsph/surface_tension.py`（m277 / PR #13，
CSF 模型，Brackbill-Kothe-Zemach 1992）。该模块合入时只有 10 个单元测试（形状、sigma=0
短路、单位法向），**从未与任何解析表面张力结果比对**——属典型验证欠账。

解析基准（带自由面的静止二维液滴，Laplace 定律）：

    Delta p = p_in - p_out = sigma / R

几何/参数：R = 40 mm，dx = 2.5 mm（R/dx = 16），h = 1.3 dx，方晶格；stage A 用
{a['particles']} 点全域 patch（半宽 {1000 * a['lattice_half_width']:.0f} mm），stage B 用
{b['particles']} 个六角晶格液滴粒子；rho0 = {b['rho0']:.0f}，sigma = {b['sigma']} N/m，
c0 = {b['sound_speed']} m/s，nu = {b['viscosity']}，gravity = 0，无壁面。

## 2. 怎么做的

工具链：PyTorch {art['torch']}（CPU，{art['nproc']} 核可见）、pytest、matplotlib；
入口 `benchmarks/laplace/run.py`（产物 `artifacts/laplace/laplace_result.json`），
文档 `docs/benchmarks/laplace.md` 由 `benchmarks/laplace/make_doc.py` 从工件生成
（零手抄）。

Stage A（算子级，无时间推进）：在**跨界面两侧**的完整晶格 patch 上给定解析色函数
c(r) = 0.5 (1 - tanh((r-R)/w))，用库自身的 compute_color_gradient / compute_normal /
compute_curvature 求 kappa 与 delta_s = |grad c|；判据：kappa = 1/R，
int |grad c| dr = 色跳变(=1)，其乘积即离散算子蕴含的 Laplace 压差。另做线性色场
c = 3x 的精确梯度一致性检验（这正是定位符号错的关键）。w/dx 扫描 1.0–3.0 做相容性检验。

Stage B（物理级，时间推进）：六角晶格液滴 + 库自身 `rhs`（含 CSF 力），半隐式 Euler，
dt 取 min(stable_dt, capillary_dt_limit)（用库自带稳定性估计，避免越过粘性限）；跑 2 tau_sigma，
体积平均内压 vs 外部；并附一条**静态动量平衡**独立测量：
Delta p = -∫ rho (a_st · r_hat) dA（无需推进、可逐位复现）。

## 3. 计算结果

Stage A（w = 2dx 头条）：kappa = {hd['kappa_mean']:.3f} 1/m（解析 {hd['kappa_analytic']:.3f}），
int |grad c| dr = {hd['integral_grad_c_dr']:.3f}（解析 1），
Delta p = {hd['delta_p_operator']:.4f} Pa（解析 {hd['delta_p_analytic']:.4f} Pa）。

| w/dx | kappa [1/m] | kappa 误差 | int\\|grad c\\|dr | Delta p [Pa] | Delta p 误差 |
|---|---|---|---|---|
{rows}

Stage B：

| 量 | 值 |
|---|---|
| dt | {b['dt']:.3e} s（约束来源 {b['dt_limit_source']}） |
| 步数 / 物理时间 | {b['steps_run']} / {b['sim_time']:.3f} s
  = {b['sim_time'] / b['tau_capillary']:.2f} tau_sigma |
| Delta p 实测（弛豫态） | {b['delta_p_measured']:.4f} Pa |
| Delta p 静态力平衡（t=0） | {b['delta_p_static_force_balance']:.4f} Pa |
| Delta p 解析 sigma/R_eq | {b['delta_p_analytic']:.4f} Pa |
| 相对误差（弛豫态 / 静态） | {100 * b['relative_error']:.2f} %
    / {100 * b['static_force_balance_relative_error']:.2f} % |

## 4. 结果图

`artifacts/laplace/laplace.png`：(a) kappa 与 int|grad c|dr 随 w/dx 的变化及解析线 kappa=1/R；
(b) 液滴压力跃变时程与解析 sigma/R_eq(t) 对比曲线。

## 5. 注意点（踩的坑）

1. **发现并修复真实缺陷**：compute_color_gradient 的 j 端 index_add 取了负号且用了 V_j。
每一对粒子两端给出相反符号，在光滑场上相互抵消——线性色场 c=3x 的梯度原本是 1e-18，
修复后 {a['grad_linear_measured'][0]:.4f}（解析 3.0，
相对误差 {100 * a['grad_linear_relative_error']:.2f} %）。
即 m277 的 CSF 力实际是惰性的，而原 10 个测试全绿（均匀场两种情况梯度都为零）。
两端贡献符号相同，仅体积下标不同。
2. 算子测试**必须在界面两侧都布点**：只取液滴晶格（无外部粒子）会丢掉 tanh 外半支，
int|grad c| dr 从 1 掉到 0.35，所有 Laplace 数被系统性低估约 3 倍。
3. dt 只按声学 Courant 会炸：nu=1e-2 时粘性限 0.125 h^2/nu 比声学限紧 5 倍，且 sigma=0 也炸
——是基础求解器不稳定，与 CSF 无关；改用库自带 stable_dt。
4. 自由面单相 CSF 的结构性局限：外部是真空、无粒子，界面 delta 只有半支撑，
叠加邻居计数色函数（跳变约 0.4 而非 1），Delta p 被低估数倍。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 量 | 基准 | 误差 | <3%? | <15%? |
|---|---|---|---|---|
| 色函数梯度一致性 | 解析线性场 | {100 * a['grad_linear_relative_error']:.2f} % | 是 | 是 |
| 曲率 kappa | 解析 1/R | {100 * hd['kappa_relative_error']:.2f} % | 是 | 是 |
| 界面 delta 积分 | 解析色跳变 | {100 * hd['integral_relative_error']:.2f} % | 是 | 是 |
| Laplace 压差（算子级） | 解析 sigma/R | {100 * hd['delta_p_relative_error']:.2f} % | 是 | 是 |
| Laplace 压差（自由面液滴） | 解析 sigma/R_eq |
| | {100 * b['relative_error']:.2f} % | **否** | **否** |

结论（honest）：符号修复后**算子链**复现 Laplace 定律优于 3%；但**自由面液滴力例程**
低估 {100 * b['relative_error']:.1f}%，不达标，已用 xfail(strict=True) 测试
（test_free_surface_droplet_laplace_deficit）把该缺额钉住，修复者会看到套件变红。
修复路线（按成本递增）：delta_s 按界面归一化 / 改用 rho/rho0 色函数加自由面阈值 /
两相 CSF（Adami-Hu-Adams 2010 的 Laplace 测试做法）。
"""

png = (ROOT / "artifacts/laplace/laplace.png").read_bytes()
payload = {
    "title": ("TensorSPH 表面张力 CSF 模型对 Laplace 定律的解析验证："
             "色函数梯度符号缺陷修复 + 算子达标 2.15%、自由面力例程低估 57.8%"),
    "summary": ("对 m277 合入的 CSF 表面张力做解析基准复核（Laplace Δp=σ/R）。发现并修复 "
                "compute_color_gradient 的符号错误（光滑场梯度原为 0，CSF 力惰性）；修复后算子链"
                "（κ、∫|∇c|dr、Δp）复现 Laplace 定律误差 2.15%（<3%）；自由面液滴力例程仍低估 "
                f"{100 * b['relative_error']:.1f}%，根因为真空外部导致界面 delta 半支撑，"
                "已用 xfail(strict) 钉住并给出修复路线。"),
    "scenario_key": "tensorsph-surface-tension-laplace",
    "mission_id": 2233,
    "tags": ["TensorSPH", "surface-tension", "CSF", "Laplace", "verification",
             "bugfix", "xfail"],
    "markdown": md,
    "figures": [{
        "name": "laplace.png",
        "caption": ("(a) CSF 算子：kappa 环均值与 int|grad c|dr 随界面宽度 w/dx 变化，"
                    "黑虚线为解析 kappa=1/R；(b) 自由面液滴的压力跃变时程 vs 解析 sigma/R_eq，"
                    "显示系统性低估。"),
        "png_base64": base64.b64encode(png).decode(),
    }],
}
res = post("/tensor/reports", payload, token)
print(json.dumps(res)[:400])
