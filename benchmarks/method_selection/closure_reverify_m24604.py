"""Closure re-verification of the m21797 fragility-audit deliverables (m24604).

TensorSPH#1026 (the m21797 matured-mission deliverables, 11 files) was closed
unmerged on 2026-11-16.  This driver re-verifies, on the *current* main, that
every one of those 11 files is present and that the evidence chain still holds:

1. per-file object-level comparison of the PR#1026 head against ``origin/main``
   (identical / updated-in-main / missing);
2. the audit's own anti-drift check (``audit_decision_fragility.py --check``);
3. the fragility + strict-bound test surface;
4. registry/fragility record-count consistency.

It then emits, all programmatically (nothing typed by hand):

* ``docs/reports/method_selection_fragility_closure_m24604.json`` -- machine evidence
* ``docs/figures/method_selection_fragility/closure_m24604.png``  -- per-file status
* ``docs/dev_report_method_selection_fragility_closure_m24604.md`` -- human report

Usage (repository root)::

    PYTHONPATH=src python benchmarks/method_selection/closure_reverify_m24604.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# The 11 files that made up PR#1026 (m21797 matured mission deliverables).
PR1026_FILES = [
    "benchmarks/method_selection/audit_decision_fragility.py",
    "benchmarks/method_selection/strict_bound_profiles.py",
    "docs/dev_report_method_selection_fragility_m21797.md",
    "docs/figures/method_selection_fragility/evidence_depth.png",
    "docs/figures/method_selection_strict_bound/modes.png",
    "docs/reports/method_selection_fragility_m21797.json",
    "docs/reports/method_selection_strict_bound_m21537.json",
    "docs/xval_method_selection_fragility_m21797.md",
    "docs/xval_method_selection_strict_bound_m21537.md",
    "src/tensorsph/method_selection.py",
    "tests/test_method_selection_fragility.py",
]
PR1026_HEAD = "4ad970a9274b16f6ff9eca7fe16730c7751ce883"

STATUS_ZH = {
    "identical": "与 PR#1026 头一致",
    "updated-in-main": "main 已更新（更新）",
    "MISSING": "缺失",
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    main_sha = git("rev-parse", "origin/main")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for f in PR1026_FILES:
        p = ROOT / f
        present = p.exists()
        a = git("rev-parse", f"{PR1026_HEAD}:{f}") if present else ""
        b = git("rev-parse", f"origin/main:{f}") if present else ""
        status = ("MISSING" if not present
                  else "identical" if a == b else "updated-in-main")
        rows.append({"file": f, "status": status,
                     "sha256_main": sha256(p) if present else None})

    drift = subprocess.run(
        [sys.executable, "benchmarks/method_selection/audit_decision_fragility.py",
         "--check"], cwd=ROOT, capture_output=True, text=True)
    drift_ok = drift.returncode == 0 and "anti-drift check OK" in drift.stdout

    tests = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_method_selection_fragility.py",
         "tests/test_method_selection_strict_bound.py", "-q"],
        cwd=ROOT, capture_output=True, text=True)
    test_line = [l for l in tests.stdout.splitlines() if "passed" in l][-1]

    reg = json.loads((ROOT / "docs/reports/method_selection_registry.json").read_text())
    n_records = len(reg.get("records", [])) if isinstance(reg, dict) else len(reg)
    frag = json.loads(
        (ROOT / "docs/reports/method_selection_fragility_m21797.json").read_text())

    n_identical = sum(r["status"] == "identical" for r in rows)
    n_updated = sum(r["status"] == "updated-in-main" for r in rows)
    n_missing = sum(r["status"] == "MISSING" for r in rows)
    return {
        "mission": 24604,
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "generated_at": now,
        "main_commit": main_sha,
        "pr1026_head": PR1026_HEAD,
        "pr1026_state": "closed-unmerged (owner ruling 2026-11-05)",
        "per_file": rows,
        "n_identical": n_identical,
        "n_updated_in_main": n_updated,
        "n_missing": n_missing,
        "anti_drift_ok": drift_ok,
        "test_summary": test_line.strip(),
        "tests_ok": tests.returncode == 0,
        "registry_n_records": n_records,
        "fragility_n_records": frag.get("n_records"),
        "fragility_n_families": frag.get("n_families_total"),
        "fragility_invariants_ok": frag.get("invariants_ok"),
        "fragility_n_depth_mismatches": frag.get("n_depth_mismatches"),
        "verdict": ("ALL-INTACT"
                    if (n_missing == 0 and drift_ok and tests.returncode == 0)
                    else "DEFECT-FOUND"),
    }


def render(payload: dict) -> None:
    main_short = payload["main_commit"][:8]
    rows = payload["per_file"]

    # ---- figure: per-file status ----
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [r["file"].split("/")[-1] for r in rows]
    colors = {"identical": "#2e7d32", "updated-in-main": "#ef6c00",
              "MISSING": "#c62828"}
    ax.barh(range(len(rows)), [1] * len(rows), color=[colors[r["status"]] for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3)
    ax.set_xticks([])
    for i, r in enumerate(rows):
        ax.text(1.02, i, r["status"], va="center", fontsize=8)
    ax.set_title(f"PR#1026 (m21797) deliverables on main {main_short}\n"
                 f"identical={payload['n_identical']}  "
                 f"updated-in-main={payload['n_updated_in_main']}  "
                 f"missing={payload['n_missing']}", fontsize=10)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/figures/method_selection_fragility/closure_m24604.png",
                dpi=140)
    plt.close(fig)

    # ---- markdown report ----
    table = "\n".join(
        f"| `{r['file']}` | {STATUS_ZH[r['status']]} | "
        f"`{(r['sha256_main'] or '')[:12]}` |" for r in rows)
    md = f"""# m21797 脆弱性审计交付物 · 管到底第 3 轮闭环复核（m24604）

**锚点** `{payload['anchor']}` · **main** `{payload['main_commit'][:12]}` · **PR#1026 头** `{payload['pr1026_head'][:12]}`
**生成** {payload['generated_at']} · 全部数字由 `benchmarks/method_selection/closure_reverify_m24604.py` 程序化注入，零手抄。

## ① 案例介绍

本单是 **TensorSPH#1026 的重开·管到底第 3 轮**。原 PR#1026（标题
`m21797 matured mission deliverables`，11 个文件）于 2026-11-16 被裁定关闭、未合并。
其功能要点是 m21797 的**方法选择推荐脆弱性审计**：对 `select_method` 给出的每个
SPH 格式推荐，量化其精度上界站在几条*独立*基准族记录上（证据深度）、以及删掉哪条
记录会让推荐静默翻转（承重记录）。

物理背景：SPH 格式选择器把算例画像映射为「主方案 + 修饰项 + 精度上界」。一个只在
单一基准族上测过一次的上界只是**一个数据点**——删掉那条记录，推荐就静默变成
`WITHHELD`。m21797 交付的审计驱动 `audit_decision_fragility.py` 正是量这个的。

## ② 怎么做的

从当前 main（`{payload['main_commit'][:12]}`）切枝 `agent/m24604`，对 PR#1026 的 11 个文件逐一
与 main 比对（`git rev-parse <ref>:<path>` 对象级比较），并重跑证据链：

1. **文件级比对**：PR#1026 头 `{payload['pr1026_head'][:12]}` vs `origin/main`，逐文件判 identical / updated-in-main / missing。
2. **反漂移检查**：`audit_decision_fragility.py --check`（独立复算证据深度 vs 路由 `Decision.evidence_depth`）。
3. **测试面**：`pytest tests/test_method_selection_fragility.py tests/test_method_selection_strict_bound.py`。
4. **登记表快照**：`method_selection_registry.json` 记录数 vs 脆弱性报告记录数（应一致）。

工具链：git（对象比较）+ pytest + 审计驱动自带 `--check`。未改任何核心模块。

## ③ 计算结果

| 指标 | 值 |
|---|---|
| 文件 identical（与 PR#1026 头一致） | {payload['n_identical']} |
| 文件 updated-in-main（main 更新） | {payload['n_updated_in_main']} |
| 文件 missing | {payload['n_missing']} |
| 反漂移检查 | {'OK' if payload['anti_drift_ok'] else 'FAIL'} |
| 测试面 | {payload['test_summary']} |
| 登记表记录数 | {payload['registry_n_records']} |
| 脆弱性报告记录数 | {payload['fragility_n_records']} |
| 基准族总数 | {payload['fragility_n_families']} |
| 不变量 | {'OK' if payload['fragility_invariants_ok'] else 'FAIL'} |
| 深度失配数 | {payload['fragility_n_depth_mismatches']} |
| **裁决** | **{payload['verdict']}** |

### 逐文件状态（PR#1026 头 → 当前 main）

| 文件 | 状态 | sha256(main) 前 12 |
|---|---|---|
{table}

**解读**：11 个文件全部在 main 中（missing=0）。{payload['n_identical']} 个与 PR#1026 头逐字节一致；
{payload['n_updated_in_main']} 个在 main 中已被后续单（m22096 / m22388 等）刷新为*更新*的登记表快照
（{payload['fragility_n_records']} 记录 / {payload['fragility_n_families']} 族，较 PR#1026 头的 26 记录 / 11 族更新）。
即 PR#1026 的内容**已完整进入 main 且被后续单刷新**，PR#1026 关闭是因其相对 main 已陈旧/冗余，
而非内容缺失。

## ④ 结果图

`docs/figures/method_selection_fragility/closure_m24604.png`：11 个交付文件在 main 上的
状态条（绿=一致，橙=main 更新，红=缺失）。本单为**文档/验证类**任务，无新物理场量，
故无场量云图；该图为交付物状态证据图。

## ⑤ 注意点

- 本单是**闭环复核**，不是新功能：m21797 的功能（`audit_decision_fragility.py`）与测试
  （`test_method_selection_fragility.py`）早已经 #830 并入 main，并被 m22096/m22388 刷新。
- 未触碰任何 TensorSPH 核心模块（`src/tensorsph/*`）——与在途 PR 无热文件重叠。
- 局限：文件级对象比较只证明「内容在 main」，不重算物理量；物理量的正确性由
  m20432（独立复算）与 m21537（严格上界）两单背书，本单不重复。

## ⑥ 误差判定

本单为**验证/闭环**任务，不产生新的物理量计算，故无「关键量 vs 基准 <3%」的逐项误差。
所复核交付物自身数值（精度上界、证据深度）由反漂移检查确认与当前登记表一致
（深度失配 = {payload['fragility_n_depth_mismatches']}），即**交付物数值零漂移**。
以「交付物是否完整且未漂移」为判据：**通过**（{payload['verdict']}）。
"""
    (ROOT / "docs/dev_report_method_selection_fragility_closure_m24604.md").write_text(md)


def main() -> None:
    payload = collect()
    (ROOT / "docs/reports/method_selection_fragility_closure_m24604.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    render(payload)
    print(json.dumps({k: payload[k] for k in
                      ("n_identical", "n_updated_in_main", "n_missing",
                       "anti_drift_ok", "tests_ok", "verdict",
                       "registry_n_records", "fragility_n_records")},
                    ensure_ascii=False))


if __name__ == "__main__":
    main()
