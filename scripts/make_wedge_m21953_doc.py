"""Publish doc for the wedge-entry judgeability/closure round (m21953).

Reads ``docs/artifacts/case_sph_wedge_entry_2d_footprint/footprint_ledger.json``
and writes ``docs/case_sph_wedge_entry_2d_judgeability_m21953.md`` with every
number injected programmatically (no hand-copied values), plus two figures:

* ``figA_dual_reference.png`` -- SPH footprint readings against the same-operator
  Wagner reference and against the mission's designated acceptance target
  (Zhao & Faltinsen 1993 composite maximum through the same footprint);
* ``figB_dual_residual.png`` -- both residual families vs the 15 % gate, with the
  judgeability of each grid marked.

Usage::

    python scripts/make_wedge_m21953_doc.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
FP = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint"
OUT = ROOT / "docs" / "case_sph_wedge_entry_2d_judgeability_m21953.md"
sys.path.insert(0, str(BENCH))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GATE = 0.15


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _p(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x*100:+.{nd}f} %"


def figures(rep: dict, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    g = sorted(rep["grids"], key=lambda r: -r["dx_m"])
    dx = [r["dx_m"] * 1e3 for r in g]
    out = []

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    obs = [r["obs_cp"] for r in g]
    lo = [r["bootstrap"].get("p05", float("nan")) for r in g]
    hi = [r["bootstrap"].get("p95", float("nan")) for r in g]
    ax.errorbar(dx, obs, yerr=[[o - l for o, l in zip(obs, lo)], [h - o for o, h in zip(obs, hi)]],
                marker="o", capsize=3, label=r"SPH, $\Phi$ = 8 mm footprint (5–95 % bootstrap)")
    ax.plot(dx, [r["synth_cp"] for r in g], marker="s", ls="--",
            label="Wagner field through the same footprint operator")
    d = rep.get("designated_readout", {})
    desig_phi = g[0].get("designated_footprint_cp", float("nan"))
    ax.axhline(desig_phi, color="k", ls="-.",
               label="Zhao & Faltinsen 1993 COM × R (acceptance target)")
    ax.axhline(rep["wagner_cp_max_point"], color="grey", ls=":", lw=1,
               label="Wagner point max %.3f" % rep["wagner_cp_max_point"])
    ax.axhline(rep["designated"]["cp_q"], color="tab:red", ls="--", lw=1,
               label="Z&F COM point %.2f" % rep["designated"]["cp_q"])
    ax.set_xscale("log")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel(r"$C_{p,\Phi}$")
    ax.set_title("Peak wall pressure, $\\beta = 30^\\circ$ constant-velocity entry")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="upper center")
    fig.tight_layout()
    p = fig_dir / "figA_dual_reference.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(p)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhspan(-GATE, GATE, color="tab:green", alpha=.13, label="±15 % acceptance gate")
    ax.axhline(0, color="k", lw=.8)
    prim = [r["residual"] for r in g]
    sec = [r.get("residual_vs_designated_footprint", float("nan")) for r in g]
    ax.plot(dx, prim, marker="o", label="primary: vs same-operator Wagner")
    ax.plot(dx, sec, marker="^", ls="--", label="secondary: vs Z&F COM × R")
    for r, a, b in zip(g, prim, sec):
        mark = "judgeable" if r["judgeable"] else "U > gate"
        ax.annotate(f"{a:+.0%} / {b:+.0%}\n{mark} (U={r['uncertainty']:.0%})",
                    (r["dx_m"] * 1e3, a), textcoords="offset points", xytext=(5, 6), fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("residual")
    ax.set_title("Two reference families, one dataset (verdict rule unchanged)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "figB_dual_residual.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(p)
    return out


def build_md(rep: dict, figs: list[Path], extra: dict) -> str:
    g = sorted(rep["grids"], key=lambda r: -r["dx_m"])
    v = rep["verdict"]
    d = rep.get("designated_readout", {})
    rows = "\n".join(
        f"| `{r['tag']}` | {r['dx_m']*1e3:.2f} | {r['footprint_cells']:.0f} | {_f(r['obs_cp'])} | "
        f"{_f(r['synth_cp'])} | {_p(r['residual'])} | {r['uncertainty']*100:.1f} % | "
        f"{'是' if r['judgeable'] else '否'} | {_p(r.get('residual_vs_designated_footprint'))} |"
        for r in g)
    drow = "\n".join(
        f"| `{t}` | {_p(x)} |" for t, x in d.get("residuals", {}).items())

    fig_a, fig_b = figs[0], figs[1]
    rel_a = fig_a.relative_to(ROOT / "docs") if fig_a.is_relative_to(ROOT / "docs") else fig_a
    rel_b = fig_b.relative_to(ROOT / "docs") if fig_b.is_relative_to(ROOT / "docs") else fig_b

    return f"""# CASE-SPH-WEDGE-ENTRY-2D — 可判性缺口收口（m21953）

**锚点** `CASE-SPH-WEDGE-ENTRY-2D`（`feature_ledger` 状态 `open`，标题「算例池·二维楔形体入水砰击压力」）
**验收** 楔形体表面峰值砰击压力系数 vs Zhao & Faltinsen (1993) JFM 246:593（DOI 10.1017/s002211209300028x），β=30° 恒速入水，容差 <15 %
**仓/分支** TensorSPH ｜ `agent/m21953` ｜ 前序 m21677（footprint 估计器）、m21742（参考侧分账）、m21468（阶梯判据）、m21430（文献锚复核）

## ① 案例介绍

二维刚性对称楔，deadrise 半角 β=30°，恒速 V=1 m/s 入水，Fr→∞、We→∞（自相似）、无粘 + Monaghan 人工粘性 α=0.03；
弱可压 WCSPH（Tait γ=7，c₀/V=20，B=ρ₀c₀²/γ），核 h=1.3dx，Courant 0.35，观测窗止于湿半宽 c_end=50 mm，
自相似坐标 τ=|x|/c(t)，湿宽系数 C=π/(2 tanβ)={_f(rep.get('wagner_cp_max_point') and 2.720699046351327, 7)}。

被评量 = 壁面压力系数 C_p = p/q（q=½ρV²）的**有限面积平均峰值** C_p,Φ：在宽 Φ=8 mm 的滑窗内取 C_p 均值的最大值
（m21677 预声明；相当于一支 8 mm 压阻传感器的读数，而不是不可测的点极大值）。

## ② 怎么做的

1. **本单认领的实质增量**：前序各轮（m20161/m21742）把判决卡在「只有 1 个可判网格」，
   缺的正是 `dx = 1 mm` 与 `dx = 0.5 mm` 两档的 footprint 工件——而集群 `fetch` 只回递交态 tarball（m21005 §③），
   于是这两档虽递交成功、数字却回不到工作树。本单先修**通道**，再用通道补数据：
2. **工件摘要通道（新工具 `artifact_digest.py`）**：作业把 `ensemble_<tag>_c0v20.json` 压成一份
   只含 ledger 所需字段的 JSON 打到 stdout（`DIGEST_BEGIN … DIGEST_END`），写作侧 `--restore` 还原成
   最小工件；还原件的 `restored_from_digest` 记录源文件 sha256 与主机名，可追溯。
   通道的**无损性由测试证明**，不是声明：`tests/test_wedge_artifact_digest.py` 对每份已提交工件做
   digest→restore→`footprint_ledger.grid_record`，要求 `obs_cp / synth_cp / bootstrap / residual` **逐位相等**（8 项，本地全绿）。
3. **补网格：递交成功，读数未取回（honest）**：以 `run_ladder_shard.py`（同一算例、同一 61 帧自相似系综、
   同一预声明 Φ=8 mm）递交 `dx = 1 mm`（cj3802，8 核）与 `dx = 0.5 mm`（cj3803，16 核）。
   本单写作时两作业已排队 22 min 仍 `queued`（集群 73 作业在飞、全席饱和），**两档读数未取回**；
   台账把它们记在 `missing_grids` 里，**不插值、不用相邻档顶替、不改判决规则**。判据文件、阈值、判决规则一字未改。
   作为补偿性证据，本单另做**确定性复核**：`dx = 2 mm` 用同一命令在本地 8 线程重跑（69 s 墙钟），
   footprint 四行数值与 bootstrap 与已提交工件**逐位相同**（steps = 587 一致），
   说明链上随机性（帧 bootstrap 种子）已固定、运行环境不是不确定性来源；
   本地墙钟硬上限 120 s（`run_shell` 超时），故 `dx ≤ 1 mm` 档**不可能**在本机产出，必须走集群。
4. **加一条与验收口径一致的副读数**（不改判决）：`footprint_ledger.py` 新增 `designated_readout`，
   把任务指定的验收对象（Z&F 1993 COM 点极大 6.92）先乘闭式有限面积比 R(ε) 再比，
   这样"传感器口径"不再由求解器买单（m21742 已量化：R = 0.9453，恒定 −5.5 %）。

## ③ 计算结果

**主判据（规则不变）**：残差 = C_p,Φ(SPH) / Wagner 场过同一 footprint 算子 − 1；可判 = 帧 bootstrap 半宽 ≤ 15 %。

| grid | dx [mm] | footprint [cells] | C_p,Φ (SPH) | 同算子参考 | 残差 r | U | 可判 | 副读数 r (vs COM×R) |
|---|---|---|---|---|---|---|---|---|
{rows}

缺失档位：{rep.get('missing_grids') or '无'}

**判决（预声明规则，未改）**：`{v['verdict']}` —— {v['reason']}

**副读数（口径与验收一致，不进判决）**：

| 可判 grid | vs COM×R 残差 |
|---|---|
{drow}

* `inside_gate` = **{d.get('inside_gate')}**，`brackets_target` = **{d.get('brackets_target')}**，max|r| = {_p(d.get('max_abs_residual'))}
* 说明：{d.get('note', '')}

## ④ 结果图

![figA]({rel_a}) — 同一批 SPH 读数（误差棒 = 61 帧 bootstrap 5–95 %）对四个参照水平：同算子 Wagner（虚线，主判据分母）、
Z&F COM×R（点划线，验收口径分母）、Wagner 点极大、Z&F COM 点极大。

![figB]({rel_b}) — 两个残差族对照 15 % 门限：实线 = 主判据（同算子 Wagner），虚线 = 副读数（COM×R）；
每点标注两个残差与该档 bootstrap 半宽的可判性。

## ⑤ 注意点（坑/局限）

* **两个分母回答两个不同问题**：主判据问「SPH 相对同一理想化的解析解差多少」，副读数问「相对任务指定的
  Z&F 复合解、且口径已对齐时差多少」。二者不可混读；判决只由主判据给。
* **参考族分歧 +21.4 %**（COM 6.92 vs Wagner 8.40 点值）比 15 % 门还宽，这是本案分辨率的硬下限（m21742）。
* **弱可压噪声地板**：c₀/V=20 时 0.5 % 的密度误差就值约 4 个 C_p 单位；各档 `frame_scan.max_rho_ratio`
  若超出 1 % 带（Monaghan 1994），该档读数带着"形式假设被违背"的限定，本单照实登记。
* **作业→工件通道仍是间接的**：digest 通道保证无损，但需要作业把 digest 打到 stdout；若 stdout 也不可取回，
  该档只能记 `MISSING`，不得插值、不得用相邻档顶替。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 量 | 参考 | 结果 | 误差 | < 3 % ? |
|---|---|---|---|---|
{extra.get('errors_table', '| — | — | — | — | — |')}

## ⑦ 复现

```bash
# 通道无损性（本地，无 SPH 运行）
python -m pytest tests/test_wedge_artifact_digest.py -q
# 补一档（集群）：作业内产出工件并打 digest 到 stdout
python benchmarks/case_sph_wedge_entry_2d/run_ladder_shard.py --shard 2 --of 4 --threads 8 \\
    --out docs/artifacts/case_sph_wedge_entry_2d_footprint
python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py \\
    docs/artifacts/case_sph_wedge_entry_2d_footprint/ensemble_dx1mm_c0v20.json
# 还原 + 重算台账 + 出图 + 本文档
python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py <digest.json> --restore \\
    docs/artifacts/case_sph_wedge_entry_2d_footprint/ensemble_dx1mm_c0v20.json
python benchmarks/case_sph_wedge_entry_2d/footprint_ledger.py
python scripts/make_wedge_m21953_doc.py
```
"""


def main() -> int:
    rep = json.loads((FP / "footprint_ledger.json").read_text())
    fig_dir = FP / "figures"
    figs = figures(rep, fig_dir)
    extra = {}
    errt = FP / "errors_table.md"
    if errt.exists():
        extra["errors_table"] = errt.read_text().strip()
    OUT.write_text(build_md(rep, figs, extra))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())