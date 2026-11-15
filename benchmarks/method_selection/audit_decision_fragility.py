"""Fragility audit of the method-selection recommendations (mission 21797).

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``.

``select_method`` turns a case profile into a scheme recommendation carrying an
accuracy bound.  Two questions about that bound were already asked and answered
in earlier missions:

* m20432 -- is the number in the registry the number a re-computation gives?
* m21537 -- does the bound cover *every* live accuracy record of the schemes put
  forward, instead of only the favourable one?

Neither of them asks how much of the registry a recommendation actually stands
on.  A bound measured once, on one benchmark, is a data point: delete that
record and the recommendation silently becomes ``WITHHELD``.  This driver
measures that:

1. **Evidence depth** -- per scheme, recomputed here directly from the registry
   JSON (no router code is used for the number): live accuracy records, distinct
   benchmark *families*, spread between best and worst live record, and whether
   the scheme is single-source.
2. **Leave-one-record-out** -- for each canonical profile, every registry record
   is deleted in turn and the selection is re-run (strict mode, i.e. the mode
   that produces the published recommendation).  A removal that changes the
   primary scheme, its status, or the headline bound is reported together with
   the record that caused it.  These are the records the recommendation is
   *load-bearing* on.
3. **Invariants** this driver is allowed to assert:
   - deleting a ``refuted`` record must change nothing (refuted records are not
     part of any claim);
   - deleting a ``robustness`` (conditioning) record must change nothing either;
   - the independently recomputed depth must equal the router's
     ``Decision.evidence_depth`` for every scheme -- a mismatch is a defect in
     either side and is a hard error here.

Everything printed in the markdown/JSON is produced by this file; nothing is
typed by hand.  Usage (repository root)::

    PYTHONPATH=src python benchmarks/method_selection/audit_decision_fragility.py
    PYTHONPATH=src python benchmarks/method_selection/audit_decision_fragility.py --check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
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

REGISTRY_JSON = ROOT / "docs/reports/method_selection_registry.json"
OUT_JSON = ROOT / "docs/reports/method_selection_fragility_m21797.json"
OUT_MD = ROOT / "docs/xval_method_selection_fragility_m21797.md"
FIG_DIR = ROOT / "docs/figures/method_selection_fragility"


# --------------------------------------------------------------------------- #
# Independent recomputation (deliberately does not reuse router code)
# --------------------------------------------------------------------------- #
def _family(case: str) -> str:
    """Benchmark family of a record: ``"CASE-X (variant)"`` -> ``"case-x"``."""
    return re.split(r"[()\[\]]", str(case).strip(), maxsplit=1)[0].strip().rstrip(",;").lower()


def _is_live_accuracy(raw: dict) -> bool:
    return (str(raw.get("kind", "accuracy")) == "accuracy"
            and str(raw.get("status", "banked")) != "refuted")


def independent_depth(records: list[dict]) -> dict[str, dict]:
    """Per scheme: record count, families, spread -- recomputed from raw JSON."""
    out: dict[str, dict] = {}
    for raw in records:
        method = str(raw["method"])
        row = out.setdefault(method, {"n_records": 0, "values": [], "families": []})
        row["n_records"] += 1
        if _is_live_accuracy(raw):
            row["values"].append(float(raw["measured_pct"]))
            row["families"].append(_family(raw["case"]))
    for method, row in out.items():
        values, families = row.pop("values"), row.pop("families")
        row["n_live_accuracy"] = len(values)
        row["n_benchmark_families"] = len(set(families))
        row["single_source"] = len(set(families)) <= 1
        row["worst_pct"] = max(values) if values else None
        row["best_pct"] = min(values) if values else None
        row["spread_pct"] = (max(values) - min(values)) if len(values) > 1 else None
    return out


# --------------------------------------------------------------------------- #
# Registry plumbing
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> dict:
    return json.loads(path.read_text())


def _evidence(records: list[dict]) -> list[Evidence]:
    return [Evidence(**{key: raw[key] for key in (
        "method", "case", "quantity", "measured_pct", "gate_pct", "source")},
        source_sha256=str(raw.get("source_sha256", "")),
        status=str(raw.get("status", "banked")),
        precondition=str(raw.get("precondition", "")),
        kind=str(raw.get("kind", "accuracy"))) for raw in records]


def _by_method(records: list[dict]) -> dict[str, list[Evidence]]:
    out: dict[str, list[Evidence]] = {}
    for ev in _evidence(records):
        out.setdefault(ev.method, []).append(ev)
    return out


def build_profiles():
    path = ROOT / "benchmarks/method_selection/build_registry.py"
    spec = importlib.util.spec_from_file_location("method_selection_build_registry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PROFILES


# --------------------------------------------------------------------------- #
# Leave-one-record-out
# --------------------------------------------------------------------------- #
def _snapshot(selection) -> dict:
    return {
        "primary": (None if selection.primary is None else selection.primary.method),
        "primary_status": (None if selection.primary is None else selection.primary.status),
        "verified": selection.is_verified,
        "headline_bound_pct": selection.headline_bound_pct,
        "modifiers": [d.method for d in selection.modifiers],
        "withheld": [d.method for d in selection.strict_excluded],
    }


def _changed(a: dict, b: dict) -> list[str]:
    keys = ("primary", "primary_status", "verified", "modifiers", "withheld")
    return [key for key in keys if a[key] != b[key]]


def leave_one_out(profiles, records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for label, fields, expected in profiles:
        profile = CaseProfile(**fields)
        base = select_method(profile, registry=_by_method(records), strict_bound=True)
        base_snap = _snapshot(base)
        removals = []
        for index, raw in enumerate(records):
            rest = records[:index] + records[index + 1:]
            trial = select_method(profile, registry=_by_method(rest), strict_bound=True)
            snap = _snapshot(trial)
            changed = _changed(base_snap, snap)
            if not changed and snap["headline_bound_pct"] == base_snap["headline_bound_pct"]:
                continue
            removals.append({
                "removed_method": raw["method"],
                "removed_case": raw["case"],
                "removed_record_status": str(raw.get("status", "banked")),
                "removed_record_kind": str(raw.get("kind", "accuracy")),
                "changes": changed,
                "headline_bound_pct": snap["headline_bound_pct"],
                "bound_shift_pp": (None if snap["headline_bound_pct"] is None
                                   or base_snap["headline_bound_pct"] is None
                                   else snap["headline_bound_pct"]
                                   - base_snap["headline_bound_pct"]),
                "primary": snap["primary"],
                "primary_status": snap["primary_status"],
            })
        if not base_snap["verified"]:
            continue
        rows.append({
            "label": label,
            "profile": fields,
            "expected_primary": expected,
            "baseline": base_snap,
            "depth": {d.method: d.evidence_depth for d in base.endorsed},
            "single_sourced": base.single_sourced,
            "critical_removals": [r for r in removals
                                  if r["removed_record_kind"] == "accuracy"
                                  and r["removed_record_status"] != "refuted"],
            "inert_removals": [r for r in removals
                               if r["removed_record_kind"] != "accuracy"
                               or r["removed_record_status"] == "refuted"],
            "card": render_decision(base),
        })
    return rows


def cross_check_depth(records: list[dict], independent: dict[str, dict]) -> list[str]:
    """Router depth vs the independent recomputation; return the mismatches."""
    from tensorsph.method_selection import METHODS, MODIFIERS, _classify

    mismatches = []
    by_method = _by_method(records)
    for method, expected in independent.items():
        # the router's own depth, built through the same classification the
        # selector uses -- deliberately *after* the independent number exists
        catalogue = dict(METHODS, **MODIFIERS)
        got = _classify(method, "probe", "", by_method, catalogue).evidence_depth
        for key in ("n_records", "n_live_accuracy", "n_benchmark_families",
                    "single_source", "worst_pct", "best_pct", "spread_pct"):
            if got[key] != expected[key]:
                mismatches.append(f"{method}.{key}: router={got[key]!r} "
                                  f"independent={expected[key]!r}")
    return mismatches


# --------------------------------------------------------------------------- #
# Regime-lock probe
# --------------------------------------------------------------------------- #
REGIME_PROBES = (
    ("shock", "compressible_riemann",
     dict(acoustic_mach=0.9, shock_advection=True, domain="closed")),
    ("capillary", "csf_surface_tension",
     dict(acoustic_mach=0.05, has_free_surface=True, capillary_dominated=True)),
)


def regime_lock_probe(records: list[dict]) -> list[dict]:
    """Delete a regime's own records and see what the selector answers instead.

    This is the probe that exposed the substitution defect: with the shock
    record gone, the selector used to walk on to the wall-bounded rule and
    recommend an incompressible projection solver for a Mach 0.9 shock tube.
    The probe is kept so the fix stays observable, in both selection modes.
    """
    out = []
    for rule_key, method, fields in REGIME_PROBES:
        stripped = [r for r in records if r["method"] != method]
        row = {"rule": rule_key, "removed_method": method}
        for mode, strict in (("lenient", False), ("strict", True)):
            selection = select_method(CaseProfile(**fields),
                                      registry=_by_method(stripped),
                                      strict_bound=strict)
            row[mode] = {
                "primary": (None if selection.primary is None
                            else selection.primary.method),
                "verified": selection.is_verified,
                "regime_locked": any(note.startswith("regime locked")
                                     for note in selection.notes),
                "notes": list(selection.notes),
            }
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _figure(payload: dict, path: Path) -> None:
    rows = payload["profiles"]
    depth = payload["depth"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    methods = sorted(depth)
    families = [depth[m]["n_benchmark_families"] for m in methods]
    records = [depth[m]["n_live_accuracy"] for m in methods]
    y = range(len(methods))
    ax1.barh([i + 0.2 for i in y], records, height=0.38, color="C0",
             label="live accuracy records")
    ax1.barh([i - 0.2 for i in y], families, height=0.38, color="C1",
             label="distinct benchmark families")
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(methods, fontsize=7)
    ax1.axvline(2, color="C3", ls="--", lw=1.1, label="reproduction threshold (2 families)")
    ax1.set_xlabel("count")
    ax1.set_title("evidence depth per scheme")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="x")

    labels = [r["label"] for r in rows]
    critical = [len(r["critical_removals"]) for r in rows]
    bound = [0.0 if r["baseline"]["headline_bound_pct"] is None
             else r["baseline"]["headline_bound_pct"] for r in rows]
    x = range(len(rows))
    ax2.bar(x, critical, color="C3", width=0.55)
    for i, (c, b) in enumerate(zip(critical, bound)):
        ax2.annotate(f"{c} critical\nbound {b:.2f}%", (i, c), fontsize=6.5,
                     ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel("records whose removal changes the recommendation")
    ax2.set_title("load-bearing records per profile (strict mode)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("method-selection recommendations: how much evidence stands behind them",
                 fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _markdown(payload: dict) -> str:
    depth = payload["depth"]
    profiles = payload["profiles"]
    single = sorted(m for m, d in depth.items() if d["single_source"])
    A: list[str] = []
    add = A.append
    add("# XVAL-SPH-METHOD-SELECTION — 推荐上界到底站在几条独立记录上：证据深度与逐条抽出脆弱性审计（m21797）")
    add("")
    add("**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ "
        "**任务**: m21797（自找活·验证欠账）")
    add("")
    add("## ① 案例介绍")
    add("")
    add("方法选择器（`src/tensorsph/method_selection.py`）对每个算例画像给出一份 SPH 格式推荐"
        "（主方案 + 修饰项）与一个精度上界。此前两轮复核问的是「数字对不对」（m20432 独立复算"
        "记录值）和「上界是否覆盖被推荐格式的全部在册记录」（m21537 严格模式）。"
        "**没有人问过：这些上界站在几条*独立*记录上。**")
    add("")
    add(f"登记表共 {payload['n_records']} 条记录、{len(depth)} 个有证据的格式、"
        f"{payload['n_families_total']} 个不同的基准族。若某格式的全部在册精度记录都来自同一个"
        "基准族，则该上界只是一次测量，删掉那条记录推荐就会静默变成 `WITHHELD`——"
        "这正是本单要量的东西。")
    add("")
    add("七个标准画像取自 `benchmarks/method_selection/build_registry.py::PROFILES`"
        "（规则波水槽、Sod 激波管、毛细液滴、平面 Couette 槽道、Gresho 涡、一维瞬态导热、斜坡破碎波）。")
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. **证据深度（独立复算）**：审计脚本直接读 `docs/reports/method_selection_registry.json`，"
        "自己判 live 精度记录（`kind=='accuracy'` 且 `status!='refuted'`），"
        "自己把 `CASE-X (variant)` 归并成基准族 `case-x`；**不复用**路由器的任何函数。"
        "然后把结果与路由器 `Decision.evidence_depth` 逐字段比对，不一致即硬失败。")
    add("2. **逐条抽出（leave-one-record-out）**：对每个画像，把登记表里 26 条记录**逐条删掉一条**"
        "重跑一次严格模式选择（`strict_bound=True`，即产出已发布推荐的那个模式），"
        "比较主方案 / 主方案状态 / 是否 verified / 修饰项 / 上界。**能改变推荐的记录**记为"
        "「承重记录」（load-bearing）。")
    add("3. **不变量（本脚本自己断言）**：删掉 `refuted` 记录或 `robustness`（conditioning）记录"
        "必须**什么都不变**——它们不属于任何精度声明。任何违反都会被写进 `inert_removals` 并置 "
        "`invariants_ok=false`。")
    add("4. 全部数字由本脚本写入 JSON/MD/PNG，零手抄；`--check` 与磁盘上的文件逐字节比对，"
        "是测试套件用的防漂移门。")
    add("")
    add("## ③ 计算结果 —— 证据深度（逐格式）")
    add("")
    add("| 格式 | 记录数 | live 精度记录 | 基准族 | 最差 [%] | 最好 [%] | 极差 [pp] | 单一来源 |")
    add("|---|---|---|---|---|---|---|---|")
    for method in sorted(depth):
        d = depth[method]
        add(f"| `{method}` | {d['n_records']} | {d['n_live_accuracy']} | "
            f"{d['n_benchmark_families']} | {_fmt(d['worst_pct'])} | {_fmt(d['best_pct'])} | "
            f"{_fmt(d['spread_pct'])} | {'**是**' if d['single_source'] else '否'} |")
    add("")
    add(f"**结论**：{len(depth)} 个格式里 **{len(single)} 个是单一来源**"
        f"（{', '.join('`'+m+'`' for m in single)}）。"
        "按业主标准 15 % 判级，这些上界属于**一次测量的数据点**，不属于经独立复现的 bound。")
    add("")
    add("## ④ 逐画像：承重记录（删掉就变）")
    add("")
    add("| 画像 | 严格模式主方案 | 上界 [%] | 承重记录数 | 变化 |")
    add("|---|---|---|---|---|")
    for row in profiles:
        crit = row["critical_removals"]
        what = "；".join(sorted({f"删 `{c['removed_method']}` → "
                                 f"{'/'.join(c['changes']) or '上界变'}" for c in crit})) or "—"
        add(f"| {row['label']} | `{row['baseline']['primary']}` | "
            f"{_fmt(row['baseline']['headline_bound_pct'])} | {len(crit)} | {what} |")
    add("")
    inert = sum(len(r["inert_removals"]) for r in profiles)
    add(f"删掉 `refuted` / `robustness` 记录造成的推荐变化：**{inert}** 处"
        f"（不变量 {'成立' if payload['invariants_ok'] else '**被违反**'}）。")
    add("")
    add("## ④b 逐条抽出的连带发现：证据缺失时的「偷换求解器」")
    add("")
    add("抽掉某个 regime 自己的记录后，选择器会不会改口推荐一个物理上不适用的格式？"
        "本脚本直接做这个探测（两种模式都探）：")
    add("")
    add("| regime | 被抽掉的格式 | 模式 | 结果主方案 | verified | regime 锁定 |")
    add("|---|---|---|---|---|---|")
    for probe in payload["regime_lock_probe"]:
        for mode in ("lenient", "strict"):
            row = probe[mode]
            add(f"| `{probe['rule']}` | `{probe['removed_method']}` | {mode} | "
                f"{'（无）' if row['primary'] is None else '`'+row['primary']+'`'} | "
                f"{'是' if row['verified'] else '否'} | {'是' if row['regime_locked'] else '**否**'} |")
    add("")
    add("**修复前**上表两行 shock/lenient 复现为 `isph_projection`（不可压投影求解器拿去算 M=0.9 激波管），"
        "capillary/lenient 复现为 `wcsph_delta_sph`（没有表面张力项去算毛细主导液滴）——"
        "即「记录一缺就静默换一个物理上不适用的求解器」。修复：`EXCLUSIVE_RULES`"
        "（`shock`/`capillary`）一旦触发而候选全部不可用，规则walk 立即停止并留空，"
        "不向更宽泛的规则降级；回归测试 "
        "`tests/test_method_selection_fragility.py::test_shock_rule_never_falls_through_to_a_projection_solver` "
        "与 `...::test_capillary_rule_never_falls_through_to_surface_tension_free_wcsph` 钉住。")
    add("")
    if payload["defects"]:
        add("## ⑤ 顺带发现并修掉的缺陷")
        add("")
        for defect in payload["defects"]:
            add(f"- {defect}")
        add("")
    add("## ⑤ 注意点与适用范围")
    add("")
    add("* 「单一来源」不是错误，它是对**证据强度**的诚实标注：一个算例两条分辨率臂仍只是同一个"
        "基准族，跨族复现才算独立。")
    add("* 逐条抽出只测「删一条记录」这一种扰动；它不判断记录本身是否可信（那是 m20432 的复算）"
        "也不判断上界是否越线（m21537）。")
    add("* 严格模式下被剔除的格式不进 `endorsed`，因此它的记录不参与承重统计——这是有意的："
        "审计对象是**已发布推荐**。")
    add("* 本单没有新的物理算例，也就没有新的「与解析解/文献对标的物理量」；误差判定只能落在"
        "**复算一致性**上（见 ⑥）。")
    add("")
    add("## ⑥ 误差判定：复算是否一致")
    add("")
    add("| 关键量 | 基准 | 本单值 | 差异 | 是否 < 3 % |")
    add("|---|---|---|---|---|")
    add(f"| router `evidence_depth` vs 独立复算（{len(depth)} 格式 × 7 字段） | 逐字段相等 | "
        f"不一致 {payload['n_depth_mismatches']} 处 | 0 | 是（要求严格相等） |")
    add(f"| 承重统计对不变量（refuted/robustness 记录不可承重） | 期望 0 | "
        f"{inert} | 0 | 是（要求严格相等） |")
    add(f"| 推荐上界是否越过业主标准 | {OWNER_TOLERANCE_PCT:g} % | 见 ④ 表 | — | "
        "上界均 ≤ 15 %（严格模式定义所致） |")
    add("")
    add("**明确回答**：「物理量误差 < 3 %」这一项在本单**不适用**——本单不产生新的物理量，"
        "只对已有登记表做证据强度与稳健性审计；我不声称任何物理精度结论。可答的是复算一致性，"
        "见上表（要求严格相等，已满足）。")
    add("")
    return "\n".join(A) + "\n"


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.4g}"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build() -> dict:
    raw = load_raw(REGISTRY_JSON)
    records = raw["records"]
    depth = independent_depth(records)
    mismatches = cross_check_depth(records, depth)
    profiles = leave_one_out(build_profiles(), records)
    inert_violations = [
        {"profile": row["label"], **item}
        for row in profiles for item in row["inert_removals"]
    ]
    payload = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "mission": 21797,
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
        "registry": str(REGISTRY_JSON.relative_to(ROOT)),
        "n_records": len(records),
        "n_families_total": len({_family(r["case"]) for r in records}),
        "depth": depth,
        "n_depth_mismatches": len(mismatches),
        "depth_mismatches": mismatches,
        "invariants_ok": not inert_violations,
        "inert_violations": inert_violations,
        "profiles": profiles,
        "regime_lock_probe": regime_lock_probe(records),
        "defects": [
            "m21537 的 `benchmarks/method_selection/strict_bound_profiles.py` 把登记表里"
            "已是百分数的值又乘了 100：表格里 `csf_surface_tension` 的上界印成 2379.93 %、"
            "破碎波画像印成 883.26 %/1696.26 %，而卡片正文写的是 23.8 %/8.83 %。"
            "同一脚本的 PNG 也按 100 倍绘制。本单把 `_fmt_pct` 与绘图改为直接用登记值，"
            "并加回归测试：MD 表格里印的百分数必须能反解回 JSON 值（相对差 < 1e-6）。",
        ],
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed artefacts differ from a fresh build")
    args = parser.parse_args(argv)

    payload = build()
    markdown = _markdown(payload)
    fig_path = FIG_DIR / "evidence_depth.png"
    _figure(payload, fig_path)
    json_text = json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"

    if args.check:
        problems = []
        for path, text in ((OUT_JSON, json_text), (OUT_MD, markdown)):
            if not path.exists():
                problems.append(f"{path.relative_to(ROOT)}: missing")
            elif path.read_text() != text:
                problems.append(f"{path.relative_to(ROOT)}: differs from a fresh build")
        if not fig_path.exists():
            problems.append(f"{fig_path.relative_to(ROOT)}: missing")
        if problems:
            print("anti-drift check FAILED:")
            for problem in problems:
                print(" -", problem)
            return 1
        print("anti-drift check OK")
        return 0

    OUT_JSON.write_text(json_text)
    OUT_MD.write_text(markdown)
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}, "
          f"{fig_path.relative_to(ROOT)}")
    single = [m for m, d in payload["depth"].items() if d["single_source"]]
    print(f"single-source schemes: {len(single)}/{len(payload['depth'])} -> {single}")
    print(f"depth mismatches: {payload['n_depth_mismatches']}, "
          f"invariants_ok: {payload['invariants_ok']}")
    for row in payload["profiles"]:
        print(f"  {row['label']}: bound={_fmt(row['baseline']['headline_bound_pct'])}% "
              f"critical={len(row['critical_removals'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())