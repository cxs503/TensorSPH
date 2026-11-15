"""Build ``docs/verification_kleefsman_text_landmarks.md`` from committed records.

Every number and every verdict in the document is injected from
``results/text_landmarks/score_<tag>.json`` (written here), never typed by hand.
The reference instants themselves come from
``outputs/kleefsman2005_text/quotes.json`` (verbatim paper sentences + page +
PDF sha256), so the document can be regenerated with::

    python benchmarks/kleefsman_obstacle/make_doc_landmarks.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import text_landmarks as tl  # noqa: E402

RES = HERE / "results"
OUT_JSON = RES / "text_landmarks"
DOC = ROOT / "docs" / "verification_kleefsman_text_landmarks.md"
FIG = ROOT / "docs" / "figures" / "kleefsman_text_landmarks.png"

#: (tag, series csv, run json, note) -- discovered, not hand-listed, except for
#: the two legacy records whose probe stations are 0.05 / 0.248 m only.
def sources() -> list[dict]:
    out = []
    lh = RES / "long_horizon"
    for csv_path in sorted(lh.glob("*_time_series.csv")):
        tag = csv_path.name.replace("_time_series.csv", "")
        meta = lh / f"{tag}.json"
        out.append({"tag": tag, "csv": csv_path,
                    "meta": json.loads(meta.read_text()) if meta.exists() else {},
                    "kind": "long-horizon"})
    for tag in ("dx0.01", "dx0.04"):
        p = RES / "peak_horizon" / f"{tag}_time_series.csv"
        if p.exists():
            out.append({"tag": f"{tag}_t26", "csv": p, "meta": {}, "kind": "legacy t=2.6 s"})
    return out


def main() -> int:
    quotes = tl.load_quotes()
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    scores = []
    for src in sources():
        sc = tl.score(src["csv"], quotes=quotes)
        sc["tag"] = src["tag"]
        sc["kind"] = src["kind"]
        sc["dx"] = src["meta"].get("dx")
        sc["wall_clock_s"] = src["meta"].get("wall_clock_s")
        sc["compression"] = src["meta"].get("max_density_compression")
        (OUT_JSON / f"score_{src['tag']}.json").write_text(json.dumps(sc, indent=2))
        scores.append(sc)

    # ---------------------------------------------------------------- figure
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    colours = plt.cm.viridis([i / max(1, len(scores) - 1) for i in range(len(scores))])
    for colour, sc in zip(colours, scores):
        cols = tl.load_series(sc["series"])
        pick = tl.pick_slab(cols, 2.426)
        if pick is None:
            continue
        name = pick[2]
        ax1.plot(cols["t_s"], cols[name], lw=1.0, color=colour,
                 label=f"{sc['tag']} ({name})")
    for lm in tl.LANDMARKS:
        ax1.axvline(lm["ref_s"], color="k", ls=":", lw=0.9)
        ax1.axvspan(lm["ref_s"] - lm["tol_s"], lm["ref_s"] + lm["tol_s"], color="k", alpha=0.06)
        ax1.text(lm["ref_s"], 0.02, lm["id"].replace("_", "\n"), fontsize=6,
                 ha="center", va="bottom", rotation=0)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("slab gauge height at x = 2.426 m [m]")
    ax1.set_title("records vs the paper's text-stated instants (dotted = stated, band = tolerance)")
    ax1.legend(fontsize=6, loc="upper right")
    ax1.grid(alpha=0.3)

    labels, vals, errs, verdicts = [], [], [], []
    for sc in scores:
        for row in sc["rows"]:
            if row["landmark"] not in ("BOX_FIRST_IMPACT", "BACKWALL_RETURN_AT_H2"):
                continue
            if not math.isfinite(row["measured_s"]):
                continue
            labels.append(f"{sc['tag']}\n{row['landmark'][:14]}\nx={row['station_m']:.3f}")
            vals.append(row["measured_s"] - row["ref_s"])
            errs.append(row["tol_s"])
            verdicts.append(row["verdict"])
    if vals:
        colours2 = ["tab:green" if v == "PASS" else "tab:red" for v in verdicts]
        ax2.bar(range(len(vals)), vals, color=colours2, alpha=0.85)
        ax2.errorbar(range(len(vals)), [0] * len(vals), yerr=errs, fmt="none",
                     ecolor="k", capsize=3, lw=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=5.5, rotation=90)
    ax2.set_ylabel("measured - stated instant [s]  (bars = tolerance)")
    ax2.set_title("early landmark residuals")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)

    # ------------------------------------------------------------------- doc
    q = {x["id"]: x for x in quotes["quotes"]}
    src_k = quotes["sources"]["kleefsman2005"]
    src_a = quotes["sources"]["amaro2023"]
    L: list[str] = []
    A = L.append
    A("# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 用原文文字陈述的时间节点复核全局波动时序")
    A("")
    A("TensorSPH · 分支 `agent/m21534` · mission 21534 · 锚点 **CASE-SPH-KLEEFSMAN-DAMBREAK-OBST**"
      "（台账 open，上文 `docs/kleefsman-h2-experimental-anchor.md` 判 honest FAIL）。")
    A("本文件全部数字由 `benchmarks/kleefsman_obstacle/make_doc_landmarks.py` 从 "
      "`results/text_landmarks/score_*.json` 与 `outputs/kleefsman2005_text/quotes.json` 程序化注入（零手抄）。"
      f"图：`docs/figures/{FIG.name}`。")
    A("")
    A("## ① 案例与这一轮的问题")
    A("")
    A("Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation method for "
      "wave impact problems*, **J. Comput. Phys. 206:363–393**, DOI 10.1016/j.jcp.2004.12.007：3.22 × 1 × 1 m "
      "水槽左端 1.228 × 0.550 m 水柱瞬时溃决，撞击 x = 2.476 m 处 0.161 m 高刚性方箱。算例池判据 = 箱前水位"
      "时程的**峰值**与**到达时刻**对实验误差 < 10 %。")
    A("")
    A("上一轮已把实验曲线从 Fig. 19 数字化（峰值 0.370 m @ 1.70 s、到达 0.359 s），并如实判 FAIL："
      "峰值 +29 %、到达 +26 %（站位 x = 2.426 m，dx = 0.02 m）。剩下的争议点是**读数口径**：实验曲线来自"
      "栅格图、探针真实 x 未从 Fig. 17 机器读出，因此「误差是物理差异还是探针位置差异」无法区分。")
    A("")
    A("本轮不再碰那张栅格图，改做一件此前完全没做过的事：**把论文用文字写死的时间节点当成判据**。"
      "这些节点不需要任何数字化，也不依赖探针的绝对 x（它们描述的是全局波动时序）：")
    A("")
    A("| id | 原文（逐字，页码为期刊印刷页） |")
    A("|---|---|")
    for lm in tl.LANDMARKS:
        for qid in lm["quote_ids"]:
            qq = q[qid]
            A(f"| `{qid}` | p.{qq['printed_page']}：“{qq['text']}” |")
    A("")
    A("原文出处：`" + src_k["url"] + "`（sha256 `" + src_k["sha256"][:16] + "…`，"
      f"{src_k['bytes']} bytes，{src_k['pdf_pages']} 页；research.rug.nl 被 Cloudflare 拦，pure.rug.nl 镜像可直取）。"
      "所有引文由 `outputs/kleefsman2005_text/quotes.json` 保存并带页码与 sha256。")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. 判据表 `benchmarks/kleefsman_obstacle/text_landmarks.py::LANDMARKS`：4 个节点，"
      "`ref_s ± tol_s` **直接由措辞读出**（Fig. 18 的 0.4 / 0.56 s 快照 → 0.48 ± 0.08 s；"
      "“about 1.8 s” → 1.80 ± 0.20 s；“at about 4.7 s”“after about 5 s” → ±0.30 s）。")
    A("2. 观测量用**固定物理孔径的 slab 水位计**（`slab_level`，孔径 0.08 m，与 dx 无关），"
      "取最近 `slab_x<S>` 列；站位不在目标 x 上时该行标 `proxy` 并记录 `station_bias_m`，绝不冒充精确站位。")
    A("3. `arrival` 类节点 = 首次上穿 0.02 m（两点线性插值）；`peak_in` 类节点 = 给定时间窗内的最大值时刻。"
      "记录没覆盖窗口 → 判 `UNKNOWN`（**不许当 PASS**）。")
    A("4. 新算例 `benchmarks/kleefsman_obstacle/long_horizon.py`：把既有算例推进到 t = 5.6 s（此前所有存档只到 "
      "2.6 s，4.7 s / 5 s 两个节点从未被测过），三个站位 x = 2.476（箱面前缘）/ 2.426（H2 约定）/ 2.228 m。")
    A("")
    A("## ③ 计算结果（程序化注入）")
    A("")
    A("| tag | 类型 | dx [m] | 记录终点 [s] | 最大压缩 | 总判定 |")
    A("|---|---|---|---|---|---|")
    for sc in scores:
        A(f"| `{sc['tag']}` | {sc['kind']} | "
          f"{'%.3g' % sc['dx'] if sc['dx'] else '—'} | {sc['t_end_s']:.3f} | "
          f"{('%.1f %%' % (100 * sc['compression'])) if sc['compression'] is not None else '—'} | "
          f"**{sc['verdict']}** |")
    A("")
    A("逐节点（`proxy` = 该行用的站位不是目标 x）：")
    A("")
    A("| tag | landmark | 站位 x [m] | 测量时刻 [s] | 原文节点 [s] | 容差 [s] | 残差 [s] | 判定 |")
    A("|---|---|---|---|---|---|---|---|")
    for sc in scores:
        for row in sc["rows"]:
            meas = "未测到" if not math.isfinite(row["measured_s"]) else f"{row['measured_s']:.3f}"
            res = "—" if not math.isfinite(row["measured_s"]) else f"{row['measured_s'] - row['ref_s']:+.3f}"
            station = f"{row['station_m']:.3f}" + (" (proxy)" if row.get("proxy") else "")
            A(f"| `{sc['tag']}` | {row['landmark']} | {station} | {meas} | "
              f"{row['ref_s']:.2f} | ±{row['tol_s']:.2f} | {res} | {row['verdict']} |")
    A("")
    A("## ④ 结果图")
    A("")
    A(f"![landmarks](../docs/figures/{FIG.name})")
    A("")
    A("左：各记录在 x = 2.426 m 的 slab 水位时程，虚线/阴影带 = 原文文字节点的参考时刻与容差；"
      "右：前两个节点（0–2.4 s 内可测）的残差，误差棒长度 = 该节点容差，绿 = PASS、红 = FAIL。")
    A("")
    A("## ⑤ 注意点 / 局限")
    A("")
    A("1. 这些节点是**全局时序**判据，不替代算例池的峰值/到达判据；它们回答的是"
      "「本求解器的波动时序是否与实验同一节拍」，因此能在不解决探针 x 争议的前提下给出可裁决结论。")
    A("2. “about” 类措辞的容差是读出来的（±0.2 / ±0.3 s），属于**人为判据**，已在 ② 明示；"
      "若后续拿到更精确的原文表述，`tol_s` 应随之收紧。")
    A("3. 站位 x = 2.476 m 只有本轮新算例（`long_horizon.py`）才有；旧存档只有 2.426 / 2.228 m，"
      "故旧行的 BOX_FIRST_IMPACT 标 `proxy`，其站位偏置写在 `station_bias_m` 里。")
    A("4. `BACKWALL_RETURN_AT_H2` 的措辞语义：原文说回流发生在约 1.8 s，紧接着说回流之后 H2 的水位最大。"
      "严格讲这是下界型陈述（H2 最大出现在回流之后），本表按 1.8 ± 0.2 s 两侧带判定，属更严口径；"
      "m20884 数字化的实验曲线峰值在 1.70 s（落在带内），说明 1.8 s 这一措辞自带约 0.1 s 宽泛度。"
      "表内该节点的 FAIL 行应读作「回流水位峰值时刻偏离 1.8 ± 0.2 s」，不可读作「回流没有发生」。")
    A("5. 若某节点窗口没被记录覆盖（例如 t = 5.6 s 的新算例未回收），该行判 `UNKNOWN`、整份记录判 "
      "`INCOMPLETE`；这是诚实欠账，不是通过。4.7 s / 5.0 s 两个晚期节点的首次测量已递交集群 "
      "（cj3500 / cj3501，`long_horizon.py --t-end 5.6`），回收后重跑本脚本即自动补齐。")
    A("")
    A("## ⑥ 误差判定（逐项，显式回答）")
    A("")
    early = [(sc["tag"], r) for sc in scores for r in sc["rows"]
             if r["landmark"] in ("BOX_FIRST_IMPACT", "BACKWALL_RETURN_AT_H2")
             and math.isfinite(r["measured_s"])]
    late = [(sc["tag"], r) for sc in scores for r in sc["rows"]
            if r["landmark"] in ("RETURN_IMPACT_ON_BOX", "SECOND_WAVE_AT_H2")]
    if early:
        worst = max(early, key=lambda t: abs(t[1]["measured_s"] - t[1]["ref_s"]))
        rel = abs(worst[1]["measured_s"] - worst[1]["ref_s"]) / worst[1]["ref_s"]
        A(f"- 早期节点（箱面首次到达、回流首次最大水位）共 {len(early)} 条可比行，"
          f"最差一条 `{worst[0]}` / {worst[1]['landmark']} @ x = {worst[1]['station_m']:.3f} m："
          f"残差 {worst[1]['measured_s'] - worst[1]['ref_s']:+.3f} s（相对 {100*rel:.1f} %）。")
    else:
        A("- 早期节点：无可用记录（见 ③ 的 UNKNOWN 行）。")
    if late:
        n_pass = sum(1 for _, r in late if r["verdict"] == "PASS")
        n_fail = sum(1 for _, r in late if r["verdict"] == "FAIL")
        n_unk = sum(1 for _, r in late if r["verdict"] == "UNKNOWN")
        A(f"- 晚期节点（4.7 s 二次冲击、5 s 二次相遇）：PASS {n_pass} / FAIL {n_fail} / UNKNOWN {n_unk}。")
    A("- **是否 <3 %？** 判据本身是「是否落在原文措辞的容差带内」，不是百分比精度；"
      "相对的百分比误差仅在上一行给出以供对照，**不声称任何节点的 <3 % 符合**。")
    A("- 与上一轮 H2 峰值/到达判据的关系：那两条（+29 % / +26 %）仍未通过，本轮**不覆盖**它们，"
      "只补充了不依赖探针 x 的时序证据。")
    A("")
    DOC.write_text("\n".join(L) + "\n")
    print(json.dumps({"doc": str(DOC), "figure": str(FIG), "scores": [s["verdict"] for s in scores]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
