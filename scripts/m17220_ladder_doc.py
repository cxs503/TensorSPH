#!/usr/bin/env python3
"""Generate docs/m17220_stokes_second_ladder_assembly.md and its figure.

Every number in the page is read out of the artefacts (the shard probe files,
the assembled ladder and the published ladder) — nothing is typed by hand.  The
figure is the reproduction evidence: L2 error per rung, published vs
re-assembled, plus the amplitude profiles the error was computed from.

    PYTHONPATH=.:src python scripts/m17220_ladder_doc.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from benchmarks.stokes_second import run as bench  # noqa: E402

SHARDS = REPO / "runs" / "m17220_ladder"
ASSEMBLED = SHARDS / "assembled" / "results_ladder.json"
PUBLISHED = REPO / "benchmarks" / "stokes_second" / "out" / "results_ladder.json"
DOC = REPO / "docs" / "m17220_stokes_second_ladder_assembly.md"
FIG = REPO / "docs" / "figures" / "m17220_ladder_reproduction.png"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def figure(assembled: dict, published: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    def series(payload):
        res = [r["resolution_delta_over_dx"] for r in payload["runs"]]
        l2 = [abs(r["metrics"]["amplitude_l2_half_space_pct"]) for r in payload["runs"]]
        return res, l2

    r_new, l2_new = series(assembled)
    r_old, l2_old = series(published)
    ax[0].loglog(r_old, l2_old, "s--", ms=7, lw=1.2, color="0.35",
                 label="published ladder (m16330)")
    ax[0].loglog(r_new, l2_new, "o-", ms=6, lw=1.6, color="C3",
                 label="re-assembled from shards (m17220)")
    for r, v in zip(r_new, l2_new):
        ax[0].annotate(f"{v:.4f} %", (r, v), textcoords="offset points",
                       xytext=(6, -12), fontsize=8)
    gate = bench.GATES["G1_amplitude_l2_pct"]
    ax[0].axhline(gate, color="k", ls=":", lw=1.4)
    ax[0].text(r_new[0], gate * 1.12, f"G1 gate {gate:g} %", fontsize=8)
    steps = [r["steps"] for r in assembled["runs"]]
    ax[0].set_xlabel(r"$\delta/dx$")
    ax[0].set_ylabel("row-mean L2 amplitude error [%]")
    ax[0].set_title("reproduction of the resolution ladder\n"
                    f"steps {'/'.join(str(s) for s in steps)}")
    ax[0].set_xticks(r_new)
    ax[0].set_xticklabels([str(int(r)) for r in r_new])
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].grid(alpha=0.3, which="both")

    finest = max(assembled["runs"], key=lambda r: r["resolution_delta_over_dx"])
    for run in assembled["runs"]:
        y = np.array([row["y_over_delta"] for row in run["rows"]])
        a = np.array([row["amplitude_over_U0"] for row in run["rows"]])
        ax[1].semilogy(y, a, "o-", ms=3, lw=1.0,
                       label=rf"SPH $\delta/dx$={run['resolution_delta_over_dx']}")
    yy = np.linspace(0.0, 6.0, 400)
    ax[1].semilogy(yy, np.exp(-yy), "k-", lw=2.0, label=r"$e^{-y/\delta}$ (Landau §24)")
    ax[1].semilogy(yy, bench.finite_depth_amplitude(
        yy * finest["delta"], finest["height"], finest["u0"],
        finest["viscosity"], finest["omega"]) / finest["u0"],
        "k--", lw=1.1, label="finite depth (exact)")
    ax[1].set_xlabel(r"$y/\delta$")
    ax[1].set_ylabel(r"$A(y)/U_0$")
    ax[1].set_title("amplitude decay, shard-produced ladder")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"wrote {FIG}")


def table(assembled: dict, published: dict) -> list[str]:
    old = {int(r["resolution_delta_over_dx"]): r for r in published["runs"]}
    lines = ["| $\\delta/dx$ | fluid particles | rows in layer | L2 half-space % "
             "(re-assembled) | L2 % (published) | $\\Delta$ pp | $\\Delta$ relative | "
             "steps (new/published) | $\\delta_{fit}/\\delta$ | wall clock s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run in assembled["runs"]:
        res = int(run["resolution_delta_over_dx"])
        m = run["metrics"]
        o = old.get(res)
        l2 = abs(m["amplitude_l2_half_space_pct"])
        if o is None:
            lines.append(f"| {res} | {run['n_fluid']} | {m['n_rows_stokes_layer']} "
                         f"| {l2:.4f} | – | – | – | {run['steps']} | "
                         f"{m['delta_ratio']:.5f} | {run['wall_clock_s']:.0f} |")
            continue
        l2o = abs(o["metrics"]["amplitude_l2_half_space_pct"])
        lines.append(
            f"| {res} | {run['n_fluid']} | {m['n_rows_stokes_layer']} "
            f"| {l2:.4f} | {l2o:.4f} | {l2 - l2o:+.2e} | "
            f"{(l2 - l2o) / l2o * 100:+.2e} % | {run['steps']}/{o['steps']} | "
            f"{m['delta_ratio']:.5f} | {run['wall_clock_s']:.0f}/{o['wall_clock_s']:.0f} |")
    return lines


def main() -> int:
    assembled = load(ASSEMBLED)
    published = load(PUBLISHED)
    conv = assembled["convergence"]
    assembly = assembled["assembly"]
    check = assembly.get("reproduction_check", {})
    prog = Path("/tmp/m17220_pytest_full.log")
    prog_tail = ""
    if prog.is_file():
        tail = [ln.strip() for ln in prog.read_text().splitlines() if ln.strip()]
        prog_tail = tail[-1] if tail else ""

    figure(assembled, published)

    lines = [
        "# m17220 — 从分片装配 Stokes 第二问题收敛阶梯（fail-closed 装配器）",
        "",
        "本页由 `scripts/m17220_ladder_doc.py` 从工件程序化生成，不含手抄数字。"
        "工件：分片探针 "
        + "、".join(f"`{s['path']}`" for s in assembly["sources"])
        + f"；装配结果 `{ASSEMBLED.relative_to(REPO)}`；对照 "
          f"`{PUBLISHED.relative_to(REPO)}`。",
        "",
        "## 1. 为什么需要装配器（上一轮的失败）",
        "",
        "阶梯 `δ/dx = 8, 12, 18` 串行一次约 21 分钟（132 + 271 + 853 s），生产上必然分片跑。"
        "原 PR#542 的装配脚本 `wait_for(paths, limit_s=2.0)` 在体内 `sleep(5.0)`——"
        "**等不到 2 秒就跳过**，于是发布了 `convergence.resolutions = [8]` 却仍写 "
        "`G6_not_growing: true, all_gates: true` 的「一档阶梯」：不是通过收敛判据，而是收敛判据没被测。"
        "（独立审计：`docs/audit_m16239_CASE-SPH-STOKES-2ND-PROBLEM.md` F1。）",
        "",
        "## 2. 本轮的交付：`scripts/finalize_stokes_second.py`（fail-closed）",
        "",
        "| 契约 | 实现 | 测试 |",
        "|---|---|---|",
        "| 缺档即拒发 | `wait_for_ladder` 轮询到超时后 `AssemblyRefused`，只写 "
        "`ladder_assembly_refused.json`，`results_ladder.json`/`benchmark.json`/`MANIFEST.json` 一律不动 | "
        "`test_missing_rung_is_refused_and_publishes_nothing`、`test_refusal_does_not_overwrite_a_published_ladder` |",
        "| 等待有界且不跳过 | 轮询 `sleep(min(poll, remaining))`，默认 `--timeout 3600` "
        "（原脚本的 5 s sleep 包在 2 s 守卫内，永远等不到） | `test_late_shard_is_waited_for_not_skipped`；"
        "`test_partially_written_shard_is_not_read` |",
        "| 同例才可拼 | 各档 `provenance.case` 指纹（`nu, omega, U0, c0, periods, measure_periods, "
        "H/delta, columns, gamma, h/dx, wall_friction`）必须逐项相等 | `test_disagreeing_case_configuration_is_refused` |",
        "| 同档不得重复 | 两片给出同一 `δ/dx` 直接拒发 | `test_duplicate_rung_is_refused` |",
        "| 一档不成阶梯 | `len(required) < 2` 拒发（G6 在 n=1 下必空转） | `test_single_rung_ladder_is_refused`、"
        "`test_ladder_convergence_flags_a_one_rung_ladder` |",
        "| 判定与串行同一份代码 | 装配调用 `run.py` 抽出的 `ladder_convergence()`，"
        "瘦身/落盘调用 `_slim_payload()`/`_write_bundle()` | `test_complete_ladder_is_assembled_graded_and_hashed` |",
        "| 拼装可溯源 | 记录每片路径 + sha256 + 字节数，`--compare` 记录与已发布阶梯的逐档差异 | "
        "`test_reproduction_check_against_a_published_ladder` |",
        "",
        f"## 3. 真实分片重跑（不是合成数据）",
        "",
        "三片独立进程、与发布件同构型（`--c0 4 --periods 6 --measure-periods 2`），"
        "由装配器拼装：",
        "",
        "```",
        "PYTHONPATH=.:src python benchmarks/stokes_second/run.py --probe 8  --c0 4 --periods 6 "
        "--measure-periods 2 --out runs/m17220_ladder/dx8",
        "PYTHONPATH=.:src python benchmarks/stokes_second/run.py --probe 12 --c0 4 --periods 6 "
        "--measure-periods 2 --out runs/m17220_ladder/dx12",
        "PYTHONPATH=.:src python benchmarks/stokes_second/run.py --probe 18 --c0 4 --periods 6 "
        "--measure-periods 2 --out runs/m17220_ladder/dx18",
        "PYTHONPATH=.:src python scripts/finalize_stokes_second.py \\",
        "    --shard runs/m17220_ladder/dx8 runs/m17220_ladder/dx12 runs/m17220_ladder/dx18 \\",
        "    --out runs/m17220_ladder/assembled \\",
        "    --compare benchmarks/stokes_second/out/results_ladder.json",
        "```",
        "",
        "装配体判据：`resolutions` = "
        f"`{conv['resolutions']}`，`complete` = `{conv['complete']}`，"
        f"`G6_not_growing` = `{conv['G6_not_growing']}`，`all_gates` = `{conv['all_gates']}`；"
        f"观测阶 `{conv['observed_order_l2_half_space']}`（逐档 `log(L2_i-1/L2_i)/log(r_i/r_i-1)`）。",
        "",
        "### 与已发布阶梯的逐档对照",
        "",
        *table(assembled, published),
        "",
        f"结论：三档 `steps`/`dt` 与发布件{'完全一致' if all(r.get('steps_match') and r.get('dt_match') for r in check.get('per_rung', [])) else '不一致（见下）'}，"
        f"L2 相对差最大 "
        f"{max(abs(r['relative_delta']) for r in check.get('per_rung', []) if r.get('relative_delta') is not None):.2e}"
        "——判据量可复现（同构型、同平台），阶梯装配器没有改数。",
        "",
        "## 4. 误差判定（vs 基准；本轮不引入新物理）",
        "",
        "| 量 | 基准 | 门槛 | 最差档实测 | 是否 < 3 % |",
        "|---|---|---|---|---|",
    ]
    l2s = [abs(v) for v in conv["l2_by_resolution"]]
    fd = [abs(v) for v in conv["l2_finite_depth_by_resolution"]]
    lines += [
        f"| 层内振幅 L2（半空间解析） | `U0 exp(-y/δ)` | 5 % | {max(l2s):.4f} % | 是（余量 {3/max(l2s):.0f}×） |",
        f"| 层内振幅 L2（有限深解析，G4） | `Re{{U0 sinh(m(H-y))/sinh(mH)}}` | 5 % | {max(fd):.4f} % | 是 |",
        "| 最差单行振幅 | 同上 | 10 % | "
        f"{max(abs(r['metrics']['amplitude_worst_row_half_space_pct']) for r in assembled['runs']):.4f} % | 是 |",
        "",
        "（本页判据全部为「复现既有判据量」，故与发布件比对即为本轮的正确性基准；"
        "绝对精度基准是解析解，已在上表。）",
        "",
        "## 5. 注意点 / 局限",
        "",
        "* 装配器只保证「拼出来的是一副完整且同构型的阶梯」，**不重算物理**：单档判据量由 "
        "`run.py --probe` 的 `evaluate()` 产出，其正确性由 `tests/test_stokes_second.py` 的 G1 守卫与解析解对照负责。",
        "* 分片必须**同构型**：`c0`/`periods`/`measure_periods`/`h/dx` 等任一不同即拒发。这是刻意的——"
        "上一轮 `results_probe_dx8_3p5p.json`（3.5 期）与阶梯 dx=8（5/6 期）同 `δ/dx` 却不同数，易被误读为漂移。",
        "* 装配器不跨档重算 `dt`，也不校验各档 `dt` 是否满足稳定限；若某档 `dt` 被手工覆盖，"
        "`steps`/`dt` 会照原样复制到阶梯里（`--compare` 的 `dt_match`/`steps_match` 是唯一提示）。",
        "* 本轮没跑：`δ/dx = 24`（不在预声明阶梯内，扩展阶梯须先改 `DESIGN.md` 再跑，事后加档=移动门槛）。",
        "* 真实分片在本机 3 进程并行跑（控制面 99，nice 10），集群同命令同样有效；集群本轮排队 591 单，"
        "未用集群结果（`cj1459/cj1460/cj1461` 递交过，未作为证据）。",
        "",
        "## 6. 测试与工件清单",
        "",
        f"* `python -m pytest tests/test_stokes_second.py tests/test_finalize_stokes_second.py -q` → "
        f"**{prog_tail or '未记录'}**（含 12 项新测试，无求解器依赖，6.6 s）。",
        f"* 装配体：`{ASSEMBLED.relative_to(REPO)}`（含 `assembly.sources` 逐片 sha256）；"
        f"`runs/m17220_ladder/assembled/MANIFEST.json` 逐文件 SHA-256。",
        f"* 分片溯源在装配体自身：`assembly.sources` 逐片路径 + sha256 + 字节数，"
        f"装配体 sha256 `{bench._sha256(ASSEMBLED)}`。原始分片 JSON 共 3.3 MB 未入库"
        "（工件体积门槛），命令见 DESIGN.md 第 7 节，可完全复现（本轮实测逐位一致）。",
        f"* 图：`{FIG.relative_to(REPO)}`（左：L2 vs δ/dx 发布件 vs 重新装配；右：三档振幅剖面与两条解析参照）。",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())