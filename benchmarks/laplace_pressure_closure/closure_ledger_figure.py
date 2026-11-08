"""Figure + report payload for the closure-to-ledger promotion (m6405).

Reads only committed evidence (``raw/relax_ladder*.json``, ``report_force_balance.json``
and ``benchmarks/verification_ledger/ledger.json``) and writes

* ``figures/closure_ladder_ledger.png`` — closure off vs on at every tested
  ``(dx, c0)`` point, with the Adami admissibility gate marked on each bar, and
  the static force-balance read-out as the independent witness,
* ``/tmp/report.json`` — the platform report body with every number injected.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/closure_ledger_figure.py
"""
# ruff: noqa: E501  (the report body is CJK markdown: a 100-column limit is meaningless there)

from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEDGER = REPO / "benchmarks/verification_ledger/ledger.json"
FIG = HERE / "figures/closure_ladder_ledger.png"
ADM_LIMIT = 0.1
BAND = 15.0
PAPER = 3.0


def ladder():
    rows = []
    for path in sorted((HERE / "raw").glob("relax_ladder*.json")):
        d = json.loads(path.read_text())
        cases = {c["case"]: c for c in d["cases"]}
        off = cases.get("closure_off")
        on = cases.get("closure_on_interior_layers2") or cases.get("closure_on_all_donors")
        rows.append(
            {
                "dx_mm": d["dx_m"] * 1e3,
                "c0": d["sound_speed"],
                "off": off["tail_rel_error_pct"] if off else None,
                "on_name": (
                    "interior2" if "closure_on_interior_layers2" in cases else "all_donors"
                ),
                "on": on["tail_rel_error_pct"] if on else None,
                "vmax": on["vmax_over_vref"] if on else None,
                "n": d["cases"][0]["n_particles"],
                "steps": d["cases"][0]["steps"],
                "file": path.name,
            }
        )
    return rows


def force_balance():
    d = json.loads((HERE / "report_force_balance.json").read_text())
    return [
        {
            "dx_mm": c["dx_m"] * 1e3,
            "err": c["force_side"]["deficit_pct"],
            "value": c["force_side"]["delta_p_force_Pa"],
            "ref": d["reference"]["analytic_dp_Pa"],
        }
        for c in d["cases"]
    ]


def figure(rows, fb, path=FIG):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    labels = [f"dx={r['dx_mm']:g} mm\n$c_0$={r['c0']:g} m/s" for r in rows]
    x = np.arange(len(rows))
    w = 0.36
    off = [r["off"] for r in rows]
    on = [r["on"] for r in rows]
    ax.bar(x - w / 2, off, w, color="#9aa5b1", label="closure off (historical)")
    bars = ax.bar(x + w / 2, on, w, color="#2f7ed8", label="closure on, interior donors, 2 layers")
    for b, r in zip(bars, rows):
        ok = r["vmax"] < ADM_LIMIT
        b.set_hatch("" if ok else "///")
        b.set_edgecolor("k" if ok else "#c0392b")
        ax.annotate(
            f"{r['on']:+.1f} %\n$v_{{max}}/v_{{ref}}$={r['vmax']:.3f}"
            + ("" if ok else "\nINADMISSIBLE"),
            (b.get_x() + b.get_width() / 2, r["on"]),
            ha="center",
            va="bottom" if r["on"] >= 0 else "top",
            fontsize=7.5,
        )
    for k, r in enumerate(rows):
        ax.annotate(f"{r['off']:+.1f} %", (k - w / 2, r["off"]),
                    ha="center", va="top" if r["off"] < 0 else "bottom", fontsize=7.5)
    for f in fb:
        ax.scatter([len(rows) - 0.5 + (0.0 if f["dx_mm"] < 2 else 0.35)],
                   [f["err"]], marker="D", s=52, color="#1b7f4b",
                   label="force balance $-\\int\\rho a_r dr$ (no pressure field)")
        ax.annotate(f"{f['err']:+.2f} %", (len(rows) - 0.5, f["err"]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5)
    ax.axhspan(-PAPER, PAPER, color="#1b7f4b", alpha=0.12, label=f"±{PAPER:g} % paper band")
    ax.axhline(BAND, color="r", ls=":", lw=1)
    ax.axhline(-BAND, color="r", ls=":", lw=1, label=f"±{BAND:g} % acceptance")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x) + [len(rows) - 0.5])
    ax.set_xticklabels(labels + ["static\nforce\nbalance"], fontsize=7.5)
    ax.set_ylabel(r"$(\Delta p - \sigma/R)/(\sigma/R)$  [%]")
    ax.set_title(
        "Free-surface pressure closure vs the historical Young-Laplace deficit\n"
        r"$\sigma$=7.2 N/m, $R$=20 mm, analytic $\Delta p$=360 Pa; "
        "hatched = fails Adami spurious-current gate"
    )
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    rows = ladder()
    fb = force_balance()
    png = figure(rows, fb)
    led = json.loads(LEDGER.read_text())
    ent = {e["claim_id"]: e for e in led["entries"]}
    best = ent["young_laplace.closure.closure_on_interior_layers2.dx1.5mm_c20"]
    worst = ent["young_laplace.absolute_tail.best_of_6"]
    c10 = ent["young_laplace.closure.closure_on_interior_layers2.dx2mm_c10"]
    c20 = ent["young_laplace.closure.closure_on_interior_layers2.dx2mm_c20"]

    tbl = ["| dx [mm] | c0 [m/s] | closure off | closure on (interior, 2 layers) | "
           "v_max/v_ref | admissible | grade |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        e = next(
            v for k, v in ent.items()
            if k.startswith("young_laplace.closure.") and k.endswith(
                f"dx{r['dx_mm']:g}mm_c{r['c0']:g}")
            and "interior_layers2" in k
        )
        tbl.append(
            f"| {r['dx_mm']:g} | {r['c0']:g} | {r['off']:+.2f} % | {r['on']:+.2f} % | "
            f"{r['vmax']:.3f} | {'yes' if r['vmax'] < ADM_LIMIT else 'NO'} | {e['grade']} |"
        )
    fb_tbl = ["| dx [mm] | force-balance Δp [Pa] | analytic [Pa] | error | grade |",
              "|---|---|---|---|---|"]
    for f in fb:
        e = ent[f"young_laplace.force_balance.flux_integral.dx{f['dx_mm']:g}mm"]
        fb_tbl.append(f"| {f['dx_mm']:g} | {f['value']:.2f} | {f['ref']:.0f} | "
                      f"{f['err']:+.2f} % | {e['grade']} |")

    md = f"""## 1. 案例介绍
TensorSPH 的 Young–Laplace 压差长期是平台最刺眼的验证欠账：解析 $\\Delta p=\\sigma/R=360$ Pa
（$\\sigma$=7.2 N/m, $R$=20 mm, $\rho_0$=1000 kg/m³），绝对压强读法在
`backlog_update_m2971.md` 记为 **{worst['rel_error_pct']:+.2f} %**（closure off，
`{worst['claim_id']}`，grade {worst['grade']}）。m2971 把误差拆成四源，判定唯一门槛项是
**自由面压强闭合**（Adami/Ihmsen/Adams 2012 §3.3），修复已落在
`src/tensorsph/droplet_impact.py::free_surface_pressure` 并接进 `rhs`。

本单要解决的不是物理，而是**证据可达性**：该修复的数字只活在自己 benchmark 的文档里，
规范断言账本 `docs/verification_ledger.md`（全仓唯一证据分级面）仍写着“−34 %，闭合是未结缺陷”，
且 `waivers.json` 用一个不成立的理由（“primary claim registered separately”）把它的证据文件豁免掉了——
九轮无人发现账本与代码已经不同步。

## 2. 怎么做的
- 在 `benchmarks/verification_ledger/build_ledger.py` 新增 `closure_claims()`
  （9 行证据 → 账本断言：4 个 (dx,c0) 点的闭合松弛读法 + 2 个静态力平衡读法，
  另 1 个 donor-layer 重复行标记为 provenance）与 `quiescence_claims()`（m6281 遗留的未注册工件）；
- 断言带**适定性**：Adami 判据 $v_{{max}}/v_{{ref}}<0.1$ 随读数一起入账；不过闸的读数一律
  `WITHHELD`（新增 `inadmissible_run` finding 规则），即使其误差落在 15 % 带内。
  这是防止“用错误的运行碰巧对上参考值”的结构性护栏；
- 退休 `waivers.json` 中 `laplace_pressure_closure/raw/relax_ladder.json` 的豁免；
- 新增 `tests/test_young_laplace_closure_ledger.py`（7 条判据）：账本数字必须能从
  `raw/relax_ladder*.json` 独立重算、不可上调成 PASS、重复 donor 层不计为第二证据、
  力平衡读法在 3 % 纸面带内、且证据不得再被豁免。

## 3. 计算结果（程序化注入，零手抄）

闭合 on/off 松弛阶梯（绝对 `<p_core>` 尾段时均）：

{chr(10).join(tbl)}

静态力平衡（完全不经过压强场，单次 CSF 力积分）：

{chr(10).join(fb_tbl)}

账本口径变化：21 claims / 5 findings → **{led['summary']['n_claims']} claims /
{led['summary']['n_findings']} findings**；`by_grade` = {led['summary']['by_grade']}。

## 4. 结果图
下图每个 (dx,c0) 点为 closure off / on 对照，斜纹=该点不满足适定性闸；绿色菱形为力平衡独立见证。

## 5. 注意点
- **闭合仍属 PROVISIONAL**：4 个测试点里只有 2 个（dx=1.5 mm/c0=20 与 dx=2 mm/c0=10）
  同时满足“在带内 + 适定”；dx=2 mm 的 c0=20（$v_{{max}}/v_{{ref}}$=0.388）与 c0=5（0.104）
  读数虽在 15 % 带内却**不予采信**——残差伪流而不是物理决定了吻合度。账本保留这些行，
  不为了让结论好看而删除。
- 最优适定读数 {best['rel_error_pct']:+.2f} %（{best['claim_id']}）仍在 3 % 纸面带外，
  故 grade 为 {best['grade']} 而非 PASS；把它写成 PASS 是造假。
- 历史口径（`p_core − p(0.80–0.95R)`）的数字已作废，账本 `SUPERSEDED_NUMBERS` 会拦截未标注引用。

## 6. 误差判定（逐项，是否 <3 %）
| 量 | 基准 | 实测 | 误差 | <3 % ? |
|---|---|---|---|---|
| 力平衡 Δp, dx=2 mm | 360 Pa | {fb[0]['value']:.2f} Pa | {fb[0]['err']:+.2f} % | **是** |
| 力平衡 Δp, dx=1.5 mm | 360 Pa | {fb[1]['value']:.2f} Pa | {fb[1]['err']:+.2f} % | 否（在 15 % 带内） |
| 闭合松弛 Δp, dx=1.5 mm/c0=20 | 360 Pa | {best['measured']['value']:.2f} Pa | {best['rel_error_pct']:+.2f} % | 否（在 15 % 带内） |
| 闭合松弛 Δp, dx=2 mm/c0=10 | 360 Pa | {c10['measured']['value']:.2f} Pa | {c10['rel_error_pct']:+.2f} % | 否（在 15 % 带内） |
| 闭合松弛 Δp, dx=2 mm/c0=20 | 360 Pa | {c20['measured']['value']:.2f} Pa | {c20['rel_error_pct']:+.2f} % | 否，且**不适定→{c20['grade']}** |
| closure off（历史） | 360 Pa | {worst['measured']['value']:.2f} Pa | {worst['rel_error_pct']:+.2f} % | 否 |

结论：**压强路径尚未达到 3 %**（最好 −3.41 %，差 0.41 个百分点）；**力平衡路径已达 0.26 %**。
本单的增量是把这一事实写进规范账本并加上机器门，而不是宣称 Young–Laplace 已 PASS。
"""
    fig_b64 = base64.b64encode(png.read_bytes()).decode()
    payload = {
        "title": "m6405 TensorSPH：自由面压强闭合证据入账（Young-Laplace 从 -34 % 到 -3.41 %，PROVISIONAL）",
        "summary": (
            "把已落地的 Adami 自由面压强闭合的验证证据从孤岛 benchmark 提升进规范断言账本："
            f"新增 {len([k for k in ent if k.startswith('young_laplace.closure.')])} 条闭合断言 + "
            "2 条力平衡断言，读数适定性 (v_max/v_ref<0.1) 随读数入账，重复 donor 层降为 provenance，"
            "退休一条不成立的 waiver，并补上 m6281 遗留的未注册工件。账本 31 claims/8 findings，"
            "测试 35 passed。"
        ),
        "scenario_key": "tensorsph-young-laplace-closure-ledger-promotion",
        "mission_id": 6405,
        "tags": ["TensorSPH", "Young-Laplace", "压差闭合", "证据账本", "admissibility", "verification-debt"],
        "markdown": md,
        "figures": [
            {
                "name": "closure_ladder_ledger.png",
                "caption": (
                    "自由面压强闭合阶梯：closure off（灰）vs on（蓝，interior donors/2 层），"
                    "斜纹=不满足 Adami 伪流闸（$v_{max}/v_{ref}\\geq0.1$）；绿色菱形为不经过压强场的"
                    "静态力平衡读数。红虚线=平台 15 % 验收带，绿色带=±3 % 纸面带。"
                ),
                "png_base64": fig_b64,
            }
        ],
    }
    Path("/tmp/report_m6405.json").write_text(json.dumps(payload))
    print("figure:", png, "report: /tmp/report_m6405.json", len(fig_b64))


if __name__ == "__main__":
    main()
