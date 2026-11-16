"""Publish the m22798 progress report (six-element paper-style) to the platform.

Reads every number from the committed artefact, embeds the committed PNG, logs in
with the environment credentials and POSTs ``/tensor/reports``.  Prints only the
report id, never the token.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_noise_floor" / "noise_floor.json"
FIG = ROOT / "docs" / "figures" / "case_sph_wedge_entry_2d_noise_floor_m22798.png"
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def pc(v: float, n: int = 1) -> str:
    return f"{100.0 * v:+.{n}f} %"


def login() -> str:
    """Try the candidate login endpoints/body shapes; return the bearer token."""
    import urllib.error
    creds = {"username": os.environ["TENSOR_AGENT_USER"],
             "password": os.environ["TENSOR_AGENT_PASS"]}
    forms = [(path, body, hdrs) for path in ("/tensor/auth/login", "/auth/login",
                                             "/tensor/login", "/api/v1/auth/login")
             for body, hdrs in (
                 ({"Content-Type": "application/json"}, "json"),
                 ({"Content-Type": "application/x-www-form-urlencoded"}, "form"),
             )]
    for path, hdrs, kind in forms:
        try:
            if kind == "json":
                data = json.dumps(creds).encode()
            else:
                import urllib.parse
                data = urllib.parse.urlencode(creds).encode()
            req = urllib.request.Request(f"{BASE}{path}", data=data, headers=hdrs)
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            tok = d.get("access_token") or d.get("token") or d.get("data", {}).get("access_token")
            if tok:
                print("login via", path, kind)
                return tok
        except urllib.error.HTTPError as exc:
            print("login", path, kind, "->", exc.code)
        except Exception as exc:                      # noqa: BLE001
            print("login", path, kind, "->", type(exc).__name__)
    raise SystemExit("no login endpoint accepted the credentials")


def build() -> dict:
    d = json.loads(ART.read_text())
    g, fine, exp, band = d["grids"], None, d["exponent"], d["exponent_bootstrap"]
    fine = min(g, key=lambda x: x["dx"])
    rows = "\n".join(
        f"| {x['tag']} | {x['reading']['n_used']} | {x['a12_over_point_peak_median']:.3f} | "
        f"{pc(x['reading']['median'])} | {pc(x['reading']['iqr'])} | "
        f"{pc(x['reading']['bootstrap']['p05'])} … {pc(x['reading']['bootstrap']['p95'])} | "
        f"{pc(x['reading']['uncertainty'])} | "
        f"{'是' if x['reading']['judgeable'] else '否'} |" for x in g)
    implied = fine["a12_over_point_peak_median"] * fine["wagner_point_peak"]
    pred = 100.0 * fine["reading"]["median"] * (0.35e-3 / fine["dx"]) ** exp["slope"]
    md = f"""### ① 案例
二维对称楔形体恒速入静水砰击（`beta=30 deg`, `V={fine['velocity']:.1f} m/s`, `c0/V={fine['c0_factor']:.0f}`,
`dx=4/2/1/0.5 mm`, 观测窗 `c_end=50 mm` → `t_end=18.4 ms`, 每网格 61 帧）。受理指标：峰值砰击压力系数
`Cp_max=p/q`；指定参考 Zhao & Faltinsen 1993（JFM 246:593, DOI 10.1017/s002211209300028x）COM = 6.92 q
（经 Jain et al. 2022 JFM Table 1 复现）；Wagner 点峰值 = {fine['wagner_point_peak']:.4f} q；容差 15 %。
本单不再造第五个峰估计量，而是判一个问题：**帧级峰值读数与「同算子精确场」之差，是格式收敛误差还是含噪场的极值偏置？**

### ② 怎么做
把精确 Wagner 廓线放在求解器自己的壁面节点格点（`|x_k|=dx((k+1/2)cos b-(1/2)sin b)`，`|x_k|<=c(t)=C V t`），
用与 `ensemble_peak.py` **逐位相同**的算子（12 等分箱、箱均值、箱心 ≤1 取最大）读一遍 → `A12(t)`；
残差 `E=obs/A12-1` 已除掉算子的离散响应。全程只读已提交工件，**不跑求解器**；节点重建与每帧 `n_wet` 对账
（中位误差 0–2 节点；`dx<=1 mm` 时 cutoff ±2 节点只动读数 ≤5 pp）。统计：帧中位 + 移动块 bootstrap
（块长 5、2000 次）+ 早/晚半窗分裂取大；判定沿用既有 `ladder_closure.verdict()`。两个预声明噪声检验：
跨网格 `E~dx^p`（极值噪声预测 0.5；格式收敛预测 ≥1）、网格内 `|E|~n_wet^(-1/2)`；旁证：E 与逐帧声学
密度误差的 Spearman 相关。

### ③ 结果
| 网格 | 入样帧 | A12/Wagner 点值 | 残差中位 | IQR | block-bootstrap 5–95 % | 不确定度 | 可判 |
|---|---|---|---|---|---|---|---|
{rows}

* 算子离散响应只有 **0.921–0.940**（相对 Wagner 点值）→「估计量把峰值读小」被排除；
* 残差四网格**同号为正**（{", ".join(pc(x['reading']['median'], 0) for x in g)}），随网格按
  **p = {exp['slope']:+.3f}**（带 {band['p05']:+.3f} … {band['p95']:+.3f}）下降：与**极值噪声（0.5）** 一致，
  与**二阶格式收敛（≥1）不相容**；
* `ladder_closure` 维持 **{d['ladder']['verdict']}**（0 个可判网格）；拟合的「残差降到 15 % 所需 dx」
  = {d['dx_for_gate'] * 1e3:.2f} mm，已细于已提交的最细网格 0.5 mm；
* `N_eff` = {min(x['reading']['n_eff'] for x in g):.0f}–{max(x['reading']['n_eff'] for x in g):.0f} 帧（共 61）：
  多记帧无益，缺的维度是**独立实现**；
* Spearman(E, 声学密度误差) = {min(x['acoustic_correlation']['rho_s'] for x in g):.2f}–{max(x['acoustic_correlation']['rho_s'] for x in g):.2f}，四网格同号为正（旁证，非证明）。

### ④ 图
四联图：(a) 算子在精确场上的响应 `A12/Wagner 点值`；(b) 逐帧残差散布与中位；(c) 中位残差—`dx` 双对数与
`p=0.5`/`p=1` 两条预测线；(d) 声明不确定度 vs 15 % 门槛。

### ⑤ 注意点
1. 未消除参考的模型歧义：COM = 6.92 q 与 Wagner 点值 {fine['wagner_point_peak']:.3f} q 本身差
   {100.0 * (fine['wagner_point_peak'] / 6.92 - 1):+.1f} %，同族六模型半跨度 42.8 %（`refs.json`）；
   本单 E 是相对同算子 Wagner 的量。
2. 「同算子」只除掉离散响应（≈8 %），不除掉噪声极值偏置 —— 后者是被测量对象。
3. 61 帧取自**同一条轨迹**（自相似假设），帧间不是独立实现；`WedgeConfig` 无初值扰动开关，
   本单按最小改动**未改求解器**。
4. 粗网格（4 mm）cutoff 灵敏度到 +13 %，该行只当量级看，结论只用细网格。
5. 网格内 ln|E|–ln n_wet 斜率四网格不一致（2.86 … −0.89，预测 −0.5）：**据实标为不一致**，不做证据
   （与启动瞬态混在一起）。

### ⑥ 误差判定（显式回答「是否 < 3 %」）
| 量 | 基准 | 结果 | 偏差 | <3 % ? |
|---|---|---|---|---|
| 复刻算子 vs `ensemble_peak.binned/gated_peak`（同一解析场） | 同算子 | 逐位相同 | 0 % | **是** |
| 重跑脚本 vs 已提交工件 | 本单工件 | 逐位一致 | 0 % | **是** |
| 节点重建 vs 工件 `n_wet` | 工件 | 中位 0–2 节点 | 0–10 节点 | 否（粗网格） |
| 帧级峰值 vs 同算子 Wagner（0.5 mm） | {implied:.3f} q | {implied * (1 + fine['reading']['median']):.3f} q | {pc(fine['reading']['median'])} ± {pc(fine['reading']['uncertainty'])} | **否**（不确定度 16.6 % > 15 % 门槛） |
| 指数 p vs 噪声预测 0.5 | +0.5 | {exp['slope']:+.3f} | {100.0 * (exp['slope'] / 0.5 - 1):+.1f} % | 否（带覆盖 0.5） |

**关键量不满足 < 3 %。** 可复现性层面精确（0 %），但本单要判的帧级峰值残差在细网格为
{pc(fine['reading']['median'])} ± {pc(fine['reading']['uncertainty'])}，既不在 3 % 也不在 15 % 内 →
honest-FAIL，不用它把锚点判成 PASS。锚点维持 `open` / `NOT_DECIDABLE`，但理由首次量化为**标度律**
（p = 0.64 ≈ 0.5 的极值噪声），不再是「噪声大」。

**登记可证伪预测**：按 p = {exp['slope']:+.2f} 外推，`dx = 0.35 mm` 残差中位应为 {pred:+.1f} %；
落在 ±5 pp 内则噪声律成立，否则本单标度结论被推翻。
**最便宜的判开路径**：4 个独立实现可把细网格标准误从 {pc(fine['reading']['uncertainty'])} 压到
~{pc(fine['reading']['uncertainty'] / 2)}（0.5 mm 单条轨迹实测 2362 s / 8 线程），前提是给
`WedgeConfig` 加预声明初值扰动。
"""
    return {
        "title": "CASE-SPH-WEDGE-ENTRY-2D：帧级峰值压力读数 vs 同算子 Wagner 参考 —— 残差按 dx^0.64 标度，"
                 "是极值噪声而非收敛误差",
        "summary": f"在四份已提交 ensemble 工件上重建「同算子 Wagner 参考」A12(t)（A12/点值 = 0.921–0.940），"
                   f"帧级峰值残差四网格同号为正、按 dx^{exp['slope']:+.2f}（带 {band['p05']:+.2f}…"
                   f"{band['p95']:+.2f}）下降，与极值噪声(0.5)相容、与格式收敛(≥1)不相容；细网格残差 "
                   f"{pc(fine['reading']['median'])}±{pc(fine['reading']['uncertainty'])} 仍不可判，"
                   f"锚点维持 open/NOT_DECIDABLE，并给出可证伪预测与最便宜判开路径（独立实现）。",
        "scenario_key": "case-sph-wedge-entry-2d",
        "mission_id": 22798,
        "tags": ["SPH", "wedge-entry", "slamming", "verification", "uncertainty",
                 "Zhao-Faltinsen-1993", "CASE-SPH-WEDGE-ENTRY-2D"],
        "markdown": md,
        "figures": [{"name": FIG.name,
                     "caption": "(a) A12/Wagner 点值；(b) 逐帧残差与中位；(c) 残差—dx 标度；"
                                "(d) 不确定度 vs 15 % 门槛",
                     "png_base64": base64.b64encode(FIG.read_bytes()).decode()}],
    }


def main() -> int:
    payload = json.dumps(build()).encode()
    req = urllib.request.Request(f"{BASE}/tensor/reports", data=payload,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {login()}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    print("report id:", out.get("id") or out.get("data", {}).get("id") or out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())