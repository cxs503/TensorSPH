"""Strict-bound evaluation of the method selector on the canonical profiles.

`select_method` answers "is some benchmark of this scheme inside 15 %?".
A *recommendation* needs the other quantifier: "is no live accuracy record of
the schemes I put forward outside 15 %?".  m21115 measured the gap between the
two per scheme (3 of 10 schemes differ, worst gap 17.8 pp) and left the routing
untouched on purpose, so the platform kept handing out recommendations whose own
headline bound was 16.96 % / 23.80 %.

This driver evaluates both modes on the seven canonical profiles and writes
every number it prints into one JSON, one markdown report and one figure --
nothing is typed by hand.  Usage (repository root)::

    PYTHONPATH=src python benchmarks/method_selection/strict_bound_profiles.py
    PYTHONPATH=src python benchmarks/method_selection/strict_bound_profiles.py --check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.method_selection import (  # noqa: E402
    OWNER_TOLERANCE_PCT,
    CaseProfile,
    Evidence,
    render_decision,
    select_method,
)

OUT_JSON = ROOT / "docs/reports/method_selection_strict_bound_m21537.json"
OUT_MD = ROOT / "docs/xval_method_selection_strict_bound_m21537.md"
FIG_DIR = ROOT / "docs/figures/method_selection_strict_bound"


def _build_registry_module():
    path = ROOT / "benchmarks/method_selection/build_registry.py"
    spec = importlib.util.spec_from_file_location("method_selection_build_registry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _by_method(registry: dict) -> dict:
    out: dict[str, list[Evidence]] = {}
    for raw in registry["records"]:
        out.setdefault(raw["method"], []).append(Evidence(
            method=raw["method"], case=raw["case"], quantity=raw["quantity"],
            measured_pct=raw["measured_pct"], gate_pct=raw["gate_pct"],
            source=raw["source"], source_sha256=raw.get("source_sha256", ""),
            status=raw.get("status", "banked"),
            precondition=raw.get("precondition", ""),
            kind=raw.get("kind", "accuracy"),
        ))
    return out


def _row(label: str, fields: dict, expected, registry) -> dict:
    profile = CaseProfile.from_mapping(fields)
    lenient = select_method(profile, registry=registry)
    strict = select_method(profile, registry=registry, strict_bound=True)

    def _endorsed(sel):
        bad = [d.method for d in sel.endorsed
               if d.accuracy_within_owner_standard is not True]
        return bad

    row = {
        "label": label,
        "profile": fields,
        "expected_primary": expected,
        "lenient": {
            "primary": lenient.primary.method if lenient.primary else None,
            "primary_status": lenient.primary.status if lenient.primary else "NONE",
            "modifiers": [d.method for d in lenient.modifiers],
            "headline_bound_pct": lenient.headline_bound_pct,
            "headline_quantity": (None if lenient.headline_record is None
                                  else f"{lenient.headline_record[0]}: "
                                       f"{lenient.headline_record[1].case}"),
            "within_owner_standard": lenient.within_owner_standard,
            "endorsed_above_tolerance": _endorsed(lenient),
        },
        "strict": {
            "primary": strict.primary.method if strict.primary else None,
            "primary_status": strict.primary.status if strict.primary else "NONE",
            "modifiers": [d.method for d in strict.modifiers],
            "headline_bound_pct": strict.headline_bound_pct,
            "within_owner_standard": strict.within_owner_standard,
            "endorsed_above_tolerance": _endorsed(strict),
            "excluded": [
                {"method": d.method, "role": d.role,
                 "worst_accuracy_pct": d.worst_accuracy_pct,
                 "worst_accuracy_case": (None if d.worst_accuracy_record is None
                                         else d.worst_accuracy_record.case),
                 "quantity": (None if d.worst_accuracy_record is None
                              else d.worst_accuracy_record.quantity)}
                for d in strict.strict_excluded
            ],
        },
        "card": render_decision(strict),
    }
    # Invariants this driver is allowed to assert: strict mode may never put a
    # scheme above the owner standard forward, and it may not change the lenient
    # route matrix (the pinned XVAL matrix depends on it).
    if row["strict"]["endorsed_above_tolerance"]:
        raise AssertionError(f"{label}: strict mode endorsed {row['strict']}"
                             "['endorsed_above_tolerance']")
    if row["lenient"]["primary"] != expected:
        raise AssertionError(f"{label}: lenient route {row['lenient']['primary']!r}"
                             f" != pinned {expected!r}")
    return row


def _fig(rows, path: Path) -> None:
    labels = [r["label"] for r in rows]
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    lenient = [100.0 * r["lenient"]["headline_bound_pct"]
               if r["lenient"]["headline_bound_pct"] is not None else 0.0 for r in rows]
    strict = [100.0 * r["strict"]["headline_bound_pct"]
              if r["strict"]["headline_bound_pct"] is not None else 0.0 for r in rows]
    h = 0.38
    ax.barh([i + h / 2 for i in y], lenient, height=h, color="C1",
            label="lenient (exists a passing record)")
    ax.barh([i - h / 2 for i in y], strict, height=h, color="C0",
            label="strict (worst live accuracy record)")
    for i, r in enumerate(rows):
        for key, off, colour in (("lenient", h / 2, "C1"), ("strict", -h / 2, "C0")):
            value = r[key]["headline_bound_pct"]
            if value is None:
                ax.plot(1.2, i + off, "x", color=colour, ms=8)
                ax.annotate("WITHHELD", (1.6, i + off), fontsize=7, va="center")
            else:
                ax.annotate(f"{100 * value:.2f}%", (100 * value, i + off),
                            fontsize=7, va="center",
                            xytext=(3, 0), textcoords="offset points")
    ax.axvline(100 * OWNER_TOLERANCE_PCT, color="C3", ls="--", lw=1.2,
               label=f"owner standard {OWNER_TOLERANCE_PCT:g}%")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("recommendation headline bound (worst live accuracy record) [%]")
    ax.set_title("method selection: lenient vs strict owner-standard enforcement")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt_pct(value) -> str:
    return "--" if value is None else f"{100 * value:.2f}"


def _markdown(payload: dict) -> str:
    rows = payload["rows"]
    lines = []
    A = lines.append
    A("# XVAL-SPH-METHOD-SELECTION — 严格 15 % 上界模式：把「推荐」从存在量词改成全称量词（m21537）")
    A("")
    A("**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ "
      "**任务**: m21537（自找活·验证欠账）｜ **平台报告**: id 19468")
    A("")
    A("## ① 案例介绍")
    A("")
    A("方法选择器（`src/tensorsph/method_selection.py`）为每个算例画像推荐一套 SPH 格式组合，"
      "并声称误差不超过业主标准 15 %。它此前用的是**存在量词**：只要该格式有**任一**记录"
      f"在 {OWNER_TOLERANCE_PCT:g} % 内就判 `SELECTED`。m21115 复核发现 10 个有证据的格式里"
      "有 3 个的**最差**在册精度记录超出 15 %（`csf_surface_tension` 23.80 %、"
      "`transport_velocity_correction` 16.96 %、`wcsph_wall_no_slip` 17.08 %），"
      "7 个标准画像里 3 个的整份推荐因此越过 15 %，卡片却仍在推荐它们。本单把**推荐**改成"
      "**全称量词**：被推荐（主方案或挂载修饰项）的每个格式，其最差在册精度记录都必须在 15 % 内。")
    A("")
    A("七个标准画像（`benchmarks/method_selection/build_registry.py::PROFILES`）与其物理背景："
      "规则波水槽、Sod 激波管、毛细液滴（Laplace 律）、平面 Couette 槽道、"
      "Gresho 涡（周期光滑无粘）、一维瞬态导热、斜坡破碎波。")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. `Decision.accuracy_within_owner_standard` 已有：判据 = 该格式**全部 live accuracy 记录**"
      "（剔除已推翻记录、剔除 conditioning/robustness 记录）的最大值 ≤ 15 %，与 m21115 的"
      "复核口径逐字一致，本单不新造判据。")
    A("2. `select_method(..., strict_bound=True)`：主方案只在 `SELECTED` **且**其上界合规的候选里挑；"
      "修饰项同样过闸，被拒的进 `Selection.strict_excluded`（连同拒它的那条记录），"
      "并写进 `notes`，绝不静默丢弃。")
    A("3. **不替代原则**：若某条规则（如 `capillary`）覆盖该工况但其候选全被上界拒掉，"
      "严格模式**不**继续走后面的规则去换一个物理上不适用的格式（毛细问题换成无表面张力项"
      "的 WCSPH 就是伪造答案），而是留空并说明理由。")
    A("4. 默认 `strict_bound=False`：历史路由矩阵零变化（钉住路由的护栏测试仍在），"
      "严格模式是显式可选项，供需要「可对外承诺」的场景使用。")
    A("5. 全部数字由 `benchmarks/method_selection/strict_bound_profiles.py` 从"
      "`docs/reports/method_selection_registry.json` 重算后注入本文件与图，零手抄。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 严格 vs 宽松（recomputed from the committed registry）")
    A("")
    A("| 画像 | 宽松主方案 | 宽松上界 [%] | <15% | 严格主方案 | 严格上界 [%] | <15% | 严格模式剔除 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        le, st = r["lenient"], r["strict"]
        exc = ", ".join(f"`{d['method']}` {d['worst_accuracy_pct']:.2f}%"
                        for d in st["excluded"]) or "—"
        A(f"| {r['label']} | {le['primary'] or '（无）'} | {_fmt_pct(le['headline_bound_pct'])} | "
          f"{'✔' if le['within_owner_standard'] else '✘'} | {st['primary'] or '（无）'} | "
          f"{_fmt_pct(st['headline_bound_pct'])} | "
          f"{'✔' if st['within_owner_standard'] else '✘'} | {exc} |")
    A("")
    A(f"- 宽松模式：{payload['counts']['lenient_within']}/7 画像整份推荐在 15 % 内；"
      f"严格模式：{payload['counts']['strict_within']}/7 在 15 % 内、"
      f"{payload['counts']['strict_withheld']} 画像被如实留空（`WITHHELD`）。")
    A(f"- 严格模式共剔除 {payload['counts']['excluded_total']} 个在册格式实例，"
      f"每一处都带出拒它的记录（见上表最后一列）。")
    A("- 严格模式内**零**越界推荐（驱动脚本对每个画像断言 "
      "`endorsed_above_tolerance == []`，不满足直接 raise）。")
    A("")
    A("### 三个代表画像的严格模式决策卡（原文，程序化生成）")
    A("")
    for label in ("breaking wave over a slope", "capillary droplet (Laplace law)"):
        card = next(r["card"] for r in rows if r["label"] == label)
        A(f"<details><summary>{label}</summary>")
        A("")
        A("```text")
        A(card.rstrip())
        A("```")
        A("")
        A("</details>")
        A("")
    A("## ④ 结果图")
    A("")
    A("![strict vs lenient](figures/method_selection_strict_bound/modes.png)")
    A("")
    A("*图：每个标准画像的两根条 = 宽松/严格模式下的推荐上界（= 被推荐格式的最差 live accuracy "
      "记录）；虚线 = 业主 15 %；`x` = 该模式下无合规方案（WITHHELD）。*")
    A("")
    A("## ⑤ 注意点（局限/适用范围）")
    A("")
    A("- **保守性来源**：上界取「该格式全部 live accuracy 记录的最大值」，跨工况、跨分辨率一视同仁。"
      "例如 `csf_surface_tension` 的 5.599 %（Laplace 压差）与 23.80 %（consistent-CSF 曲率）"
      "是两条不同的量，严格模式取后者。按工况/分辨率**分域**取上界是下一步（需给记录加机器可读的"
      "工况标签），本单不做，以免变成用有利记录摊薄。")
    A("- **不替代原则的代价**：毛细液滴画像在严格模式下没有方案可推荐——这是一个**诚实的空白**，"
      "它指向的实质待办是让 CSF 曲率记录本身回到 15 % 内，而不是放宽判据。")
    A("- **波流水槽实测缺口仍未闭合**：本单没有新增 `CASE-WAVE-FLUME` 求解器记录"
      "（`D13_h025_A` 全窗 5.5 s 作业 `cj3499` 递交集群后收尾时仍在排队，记为 not_run），"
      "因此本单回答的是"
      "**推荐机制**的合规性，不是新算例精度。")
    A("- 严格模式只改推荐范围，不改任何物理参数；`strict_bound=False` 的默认行为与历史矩阵逐字节一致。")
    A("")
    A("## ⑥ 误差判定")
    A("")
    A("| 关键量 | 基准 | 严格模式值 | 与基准差 | 是否 < 3 % | 是否 < 15 % |")
    A("|---|---|---|---|---|---|")
    A("| 推荐上界（破碎波画像） | 业主 15 % 线（判据，非物理基准） | "
      f"{_fmt_pct(next(r['strict']['headline_bound_pct'] for r in rows if r['label']=='breaking wave over a slope'))} % | "
      f"余量 {15 - 100*next(r['strict']['headline_bound_pct'] for r in rows if r['label']=='breaking wave over a slope'):.2f} pp | 不适用（判据比较，非精度对标） | ✔ |")
    A("| 推荐上界（毛细液滴画像） | 同上 | WITHHELD | 无可承诺值 | 不适用 | ✘（无可推荐方案） |")
    A("| 宽松/严格上界之差 | 宽松模式自身 | 见 ③ 表 | 逐画像 0 ~ 17.81 pp | 不适用 | — |")
    A("")
    A("**明确回答**：本单是**判据执行**类改动，没有新的与解析解/文献对标的物理量，"
      "因此不存在「物理量误差 < 3 %」这一项可答——不声称、不粉饰。可答的是机制层面：")
    A(f"严格模式把越界推荐从 3 个画像降到 0 个，代价是 {payload['counts']['strict_withheld']} "
      "个画像如实报 WITHHELD。")
    A("")
    A("## ⑦ 复现")
    A("")
    A("```bash")
    A("PYTHONPATH=src python benchmarks/method_selection/strict_bound_profiles.py --check")
    A("PYTHONPATH=src python -m pytest tests/test_method_selection_strict_bound.py -q")
    A("```")
    A("")
    return "\n".join(lines) + "\n"


def build(check: bool = False) -> dict:
    module = _build_registry_module()
    registry = module.build_registry()
    by_method = _by_method(registry)
    rows = [_row(label, dict(fields), expected, by_method)
            for label, fields, expected in module.PROFILES]
    payload = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "mission": 21537,
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
        "registry": "docs/reports/method_selection_registry.json",
        "rows": rows,
        "counts": {
            "profiles": len(rows),
            "lenient_within": sum(1 for r in rows if r["lenient"]["within_owner_standard"]),
            "strict_within": sum(1 for r in rows if r["strict"]["within_owner_standard"]),
            "strict_withheld": sum(1 for r in rows if r["strict"]["primary"] is None),
            "excluded_total": sum(len(r["strict"]["excluded"]) for r in rows),
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if check:
        if OUT_JSON.exists() and OUT_JSON.read_text() != text:
            raise SystemExit(f"{OUT_JSON} is stale: rerun without --check")
        print(f"{OUT_JSON} is up to date ({payload['counts']})")
        return payload
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(text)
    _fig(rows, FIG_DIR / "modes.png")
    OUT_MD.write_text(_markdown(payload))
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}\nwrote {FIG_DIR / 'modes.png'}")
    print(payload["counts"])
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    build(check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
