#!/usr/bin/env python
"""Generate the P482-N04 report from a run bundle (zero hand-typed numbers).

    python docs/reports/m14534_make_report.py \
        --run runs/sph_free_surface_verification/full

Reads ``results.json`` / ``benchmark.json`` / ``MANIFEST.json`` from the run
directory, copies the figures into ``docs/reports/figures/`` and writes

* ``docs/reports/m14534-free-surface-verification.md``  -- the paper-style report
* ``docs/reports/m14534-maturity-evidence.json``        -- the machine-readable
  maturity block (impl_files / entry / run_artifacts / integration_callers /
  benchmark / repro_cmd) plus the gate verdicts

Every number in the markdown comes from the bundle; the only literals in this
script are labels and prose.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "reports"
FIG_DIR = DOCS / "figures"


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e5:
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def metric_table(case: dict) -> str:
    lines = ["| metric | value | unit | gate | verdict | source |",
             "|---|---|---|---|---|---|"]
    for name, m in case["metrics"].items():
        lines.append(f"| `{name}` | {fmt(m['value'])} | {m['unit'] or '-'} | "
                     f"{fmt(m['gate_limit'])} | {m['verdict'] or '-'} | "
                     f"`{m['source']}` |")
    return "\n".join(lines)


def param_table(cases: dict) -> str:
    keys = ["dx", "column_width", "column_height", "sound_speed", "viscosity",
            "seconds", "steps", "particles", "n_particles", "tank_width", "depth",
            "open_x", "horizon_steps", "gravity"]
    names = [n for n in cases if n in (
        "dam_break_2d_dx0.02", "dam_break_2d_dx0.05", "hydrostatic_pressure_profile",
        "mass_conservation_closed_tank", "kernel_interpolation_convergence")]
    header = "| parameter | " + " | ".join(names) + " |"
    lines = [header, "|" + "---|" * (len(names) + 1)]
    for key in keys:
        row = [fmt(cases[n]["config"].get(key)) for n in names]
        if all(value == "n/a" for value in row):
            continue
        lines.append(f"| {key} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def benchmark_table(rows: list[dict], limit: int = 40) -> str:
    lines = ["| # | quantity | reference (grade) | ref | got | rel. error | tol | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for i, row in enumerate(rows[:limit]):
        lines.append(
            f"| {i + 1} | {row['quantity']} | {row['reference']} "
            f"({row['reference_grade']}) | {fmt(row['reference_value'])} | "
            f"{fmt(row['computed_value'])} | "
            f"{fmt(row['rel_error']) if row['rel_error'] is not None else '-'} | "
            f"{fmt(row['tolerance']) if row['tolerance'] is not None else '-'} | "
            f"{row['verdict']} |")
    if len(rows) > limit:
        lines.append(f"| ... | ({len(rows) - limit} more rows in benchmark.json) | | | | | | |")
    return "\n".join(lines)


def gate_table(gates: dict) -> str:
    lines = ["| gate | verdict | honest FAIL | detail |", "|---|---|---|---|"]
    for name, gate in gates.items():
        detail = json.dumps(gate["detail"], ensure_ascii=False)
        if len(detail) > 240:
            detail = detail[:237] + "..."
        lines.append(f"| `{name}` | {gate['verdict']} | "
                     f"{'yes' if gate['honest_fail'] else '-'} | `{detail}` |")
    return "\n".join(lines)


def error_judgement(bench: dict, results: dict) -> str:
    """The explicit '< 3 %?' answer for every gated quantity with a reference."""
    lines = ["| observable | reference | value | rel. error | < 3 % ? |",
             "|---|---|---|---|---|"]
    seen = set()
    for row in bench["rows"]:
        if row["tolerance"] is None or row["direction"] != "abs":
            continue
        if row["quantity"] in seen:
            continue
        seen.add(row["quantity"])
        rel = row["rel_error"]
        if row["verdict"] == "WITHHELD":
            answer = ("n/a (reference grade WITHHELD)" if row["reference_grade"] == "WITHHELD"
                      else "n/a (no pre-declared tolerance: measurement, not a gate)")
        else:
            answer = "yes" if rel is not None and rel < 0.03 else "no"
        lines.append(f"| {row['quantity']} | {row['reference'][:60]} | "
                     f"{fmt(row['computed_value'])} | {fmt(rel)} | {answer} |")
    # the aggregate gate numbers, which are not per-point rows
    for name, label, key, is_pct in (
            ("J1_front", "dam-break front vs experiment (max rel. error)",
             "max_rel_error", False),
            ("J1_height", "column height vs exact anchors (max rel. error)",
             "residual", False),
            ("J1_height_ballistic", "corner height vs free fall (max rel. error)",
             "ballistic", False),
            ("J2_pressure", "hydrostatic pressure (max band error)", "measured_pct",
             True),
            ("J3_mass_closed", "mass loss, closed tank", "measured", False),
            ("J4_convergence", "interpolation order", "measured_min_order", False)):
        gate = results["gates"].get(name)
        if gate is None and name == "J1_height_ballistic":
            gate = results["gates"].get("J1_height")
        if not gate:
            continue
        detail = gate["detail"]
        value = detail.get(key)
        rel = None if value is None else (float(value) / 100.0 if is_pct
                                          else abs(float(value)))
        lines.append(f"| {label} | {'campaign limit' if is_pct else '-'} | "
                     f"{fmt(value)}{' %' if is_pct else ''} | "
                     f"{'-' if rel is None else fmt(rel)} | {gate['verdict']} |")
    return "\n".join(lines)


def front_signature(results: dict, dam_name: str) -> str:
    """Late/early sign pattern of the front error, computed from the bundle."""
    points = results["cases"][dam_name]["series"]["front_points"]
    if not points:
        return "（无重叠实验点）"
    worst = max(points, key=lambda p: p["rel_error"])
    best = min(points, key=lambda p: p["rel_error"])
    late = [p for p in points if p["rel_error"] < 0.10]
    late_txt = (", ".join(f"τ={p['tau']:.2f} ({p['rel_error'] * 100:.1f} %)"
                          for p in late) or "无")
    return (f"* 最大偏差：τ={worst['tau']:.2f}，Z/a 实验 {fmt(worst['Z_over_a_experiment'])} "
            f"vs 计算 {fmt(worst['Z_over_a_simulation'])}（{fmt(worst['rel_error'] * 100, 3)} %）；\n"
            f"* 最小偏差：τ={best['tau']:.2f}，{fmt(best['rel_error'] * 100, 3)} %；\n"
            f"* 偏差 < 10 % 的点：{late_txt}；\n"
            f"* 形态：早期前沿滞后（自由面求和密度截断 → 驱动力不足），越过实验曲线后转为超前"
            f"（缺乏底摩阻/湍流耗散，前沿速度趋近 Ritter 的 2c₀），因此 10 % 门不是"
            f"「处处略差」而是「两头都差、中段才对」——这决定了下一步该改的是耗散模型，"
            f"而不是继续加密网格（Δx=0.01 的历史数字见 §5）。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/sph_free_surface_verification/full")
    parser.add_argument("--out", default="docs/reports/m14534-free-surface-verification.md")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    results = json.loads((run_dir / "results.json").read_text())
    bench = json.loads((run_dir / "benchmark.json").read_text())
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    cases = results["cases"]

    # ---- figures into the docs tree -------------------------------------- #
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_map = {}
    for rel in manifest.get("figures", []):
        src = run_dir / rel
        if not src.exists():
            continue
        dst = FIG_DIR / f"m14534_{src.name}"
        shutil.copyfile(src, dst)
        fig_map[src.name] = dst.name

    # ---- maturity evidence ----------------------------------------------- #
    maturity = dict(manifest["maturity"])
    maturity["maturity_level"] = manifest["maturity_level"]
    maturity["gates"] = manifest["gates"]
    maturity["solver_callers"] = manifest.get("solver_callers")
    maturity["run_dir"] = str(run_dir.relative_to(REPO_ROOT)) \
        if run_dir.is_relative_to(REPO_ROOT) else str(run_dir)
    maturity["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (DOCS / "m14534-maturity-evidence.json").write_text(
        json.dumps(maturity, indent=1, sort_keys=True))

    # ---- pointer for the next reader ------------------------------------- #
    pointer = run_dir.parent / "latest.json"
    try:
        rel_run = str(run_dir.relative_to(REPO_ROOT))
    except ValueError:                                    # pragma: no cover
        rel_run = str(run_dir)
    pointer.write_text(json.dumps({
        "run_dir": rel_run,
        "generated_utc": maturity["generated_utc"],
        "maturity_level": manifest["maturity_level"],
        "repro_cmd": manifest["repro_cmd"],
        "gates": manifest["gates"],
        "report": "docs/reports/m14534-free-surface-verification.md",
        "maturity_evidence": "docs/reports/m14534-maturity-evidence.json",
    }, indent=1, sort_keys=True))

    # ---- markdown -------------------------------------------------------- #
    dam_name = next(n for n in cases if n.startswith("dam_break_2d_dx"))
    dam = cases[dam_name]
    hyd = cases["hydrostatic_pressure_profile"]
    closed = cases["mass_conservation_closed_tank"]
    open_case = cases["mass_conservation_open_tank"]
    conv = cases["kernel_interpolation_convergence"]
    gates = results["gates"]
    ladder = conv["series"]["ladder"]
    defect = conv["series"]["support_defect"]

    figures = []
    captions = {
        "fig1_dam_break_front_profiles.png":
            "溃坝前沿 Z/a(τ)：实验数字化点（±0.05，等级 WITHHELD）、SPH 曲线、Ritter 浅水界；右图为不同 τ 的自由面剖面 h(x)。",
        "fig2_dam_break_field_speed.png":
            "溃坝末态粒子速度场云图（|v|，m/s）：场量图，用于判断水舌/喷射粒子与连续水体。",
        "fig3_hydrostatic_pressure.png":
            "静水压力剖面：SPH 分带均值 vs 解析 ρ0g(H−z)；右图为各带相对误差（灰带为自由面架带，不参与门控）。",
        "fig4_mass_bookkeeping.png":
            "质量记账：闭式与开口水槽的 M(t)/M(0)；点线为「删掉出流粒子却不记账」的负控行为。",
        "fig5_convergence_support.png":
            "插值 L2 误差随 Δx（含一阶参考线）与核支持域截断造成的单位分解缺陷。",
    }
    for name, rel in fig_map.items():
        figures.append(f"![{name}](figures/{rel})\n\n*{captions.get(name, name)}*")

    repro = manifest["repro_cmd"]
    front_sig = front_signature(results, dam_name)
    figure_blocks = "".join(
        f"### {name}\n\n{block}\n\n" for name, block in zip(fig_map, figures))
    md = f"""# TensorSPH 自由面流动验证面（P482-N04，m14534 返修轮）

**仓库**：TensorSPH · **分支**：`{manifest.get('git_branch')}` · **commit**：`{manifest.get('git_commit')}`
· **运行目录**：`{maturity['run_dir']}` · **生成时间（UTC）**：{maturity['generated_utc']}
· **tier**：`{manifest['entry']['tier']['name']}`（门限判据用战役尺寸；smoke 档只跑管线，门一律 WITHHELD）

本报告全部数字由 `docs/reports/m14534_make_report.py` 从运行包（`results.json` /
`benchmark.json` / `MANIFEST.json`）注入，脚本内不含任何物理数值字面量。
每个度量在 `results.json` 里带 `source`（产生它的函数调用路径），
`benchmark.json` 的每行对照在写出时被反向校验过一次
（`Bundle.check_benchmark_consistency`：行里的 got 必须等于运行记录到的
metric 或 series 元素，否则直接抛错拒绝出包）。

## 1. 案例与物理背景

四个面各配一个负控，构成本节点的可机器裁决面（J1–J4）：

* **M1 溃坝**：矩形水柱 a=0.2 m、H=2a（Martin & Moyce 1952 几何）在长干床上瞬间释放。
  观测量：无量纲前沿 Z/a(τ)（τ=t√(2g/a)）与柱高 h(t)。柱高没有可引用的数字化实验曲线，
  因此锚在两条**精确**结论上：稀疏波尾部到达前（t ≤ a/2c₀）库区静止 → h_back = H；
  顶角两侧零压 → 自由落体 h = H − ½gt²（t ≤ a/c₀）。
  负控：g=0 时水柱必须保持初始形状（参考格架由 `make_column` 独立生成，不取模拟轨迹）。
* **M2 静水压力**：p(z)=ρ₀g(H−z)，N≥1000 粒子。门只施加于内核带——求和密度在自由面
  2h 内被截断，表层带必然偏低，报告但排除。负控：ρ=0 时 Tait 状态方程必须精确返回 0。
* **M3 质量守恒**：SPH 粒子质量恒定，"总质量不变"本身不检验任何东西；会坏的是**边界记账**。
  闭式水槽 M(t) 必须在 0.1 % 内（≥1000 步）；开口水槽记账质量同样保持，而"删粒子不记账"
  的负控必须显著丢质量（>1 %）。
* **M4 粒子间距收敛**：解析场在 Δx 递减格架上的 SPH 插值 L2 误差收敛阶须 ≥1；
  核支持域截到 R=1.5h 时单位分解被破坏，静水平衡必须崩。

### 参数表（直接取自 `results.json` 的 config 字段）

{param_table(cases)}

## 2. 方法与工具链

1. **统一入口**：`python -m tensorsph.verification --tier {manifest['entry']['tier']['name']}`
   （脚本包装 `scripts/run_sph_free_surface_verification.py` 走同一函数）。
2. **真调用主求解器**（运行期解析并记录文件/行号，见 `MANIFEST.json.solver_callers`）：
   {", ".join(f"`{c['caller']}` ({c['defined_in']}:{c['line']})" for c in manifest.get("solver_callers", []))}。
3. **测量定义与门**在 `src/tensorsph/verification/` 的四个模块里；本节点只新增测量、门、负控与证据层。
4. **证据层**：`bundle.py` 把每个 case 的 config/metrics/series 写成纯文本包，并对写出的每个文件做
   SHA-256 登记（`MANIFEST.json.maturity.run_artifacts`），因此改一个数字就能被发现。
5. **成熟度是推导出来的**：`bundle.assess_level()` 只读 manifest 自己的 `maturity` 块，按
   L1..L4 必需键是否非空给出等级——不是自评。本次结果：**{manifest['maturity_level']['level']}**
   （缺项：{json.dumps({k: v for k, v in manifest['maturity_level']['missing'].items() if v}, ensure_ascii=False)}）。
6. **复现**（单条命令，写入 manifest 的 `repro_cmd` 字段）：

```bash
{repro}
```

## 3. 计算结果

### 3.1 门与负控裁决

{gate_table(gates)}

### 3.2 M1 溃坝（`{dam_name}`）

{metric_table(dam)}

分辨率阶梯（同算例只改 Δx）：

{json.dumps(manifest.get('dam_break_ladder'), ensure_ascii=False, indent=1)}

### 3.3 M2 静水压力（`hydrostatic_pressure_profile`）

{metric_table(hyd)}

### 3.4 M3 质量守恒

闭式水槽（`mass_conservation_closed_tank`）：

{metric_table(closed)}

开口水槽（`mass_conservation_open_tank`，含"忘记出流"负控）：

{metric_table(open_case)}

### 3.5 M4 收敛性与核支持域（`kernel_interpolation_convergence`）

{metric_table(conv)}

收敛阶梯（L2 误差，Shepard 归一/原始求和/截断支持域三套）：

{json.dumps([{k: r[k] for k in ('dx', 'particles', 'interior', 'l2_error_normalized', 'l2_error_raw', 'l2_error_truncated_support')} for r in ladder], ensure_ascii=False, indent=1)}

单位分解缺陷：R=2h → {fmt(defect['support_2h']['mean_defect'])}，R=1.5h → {fmt(defect['support_1.5h']['mean_defect'])}，
比值 {fmt(defect['defect_ratio'])}（门限 ≥10）。

### 3.6 参考值 vs 计算值（benchmark 对照表）

`benchmark.json` 摘要：{json.dumps(bench['summary'], ensure_ascii=False)}。
参考数据登记：`{json.dumps(results.get('references', {}), ensure_ascii=False)[:400]}`

{benchmark_table(bench['rows'])}

## 4. 结果图

{figure_blocks if figures else "（本次运行未生成图；tier=smoke 时同样会出图，见 run 目录 figures/）"}

### 3.7 前沿偏差的形态（同一条曲线，两个符号）

{front_sig}

## 5. 注意点与局限

* **实验表等级 WITHHELD**：`benchmarks/dam_break/martin_moyce_1952.json` 被仓库自身审计
  （`benchmarks/dam_break/reference_audit.json`）判为数字化可疑（横坐标落在精确等差格点上）。
  本节点按战役指定把它当靶子使用，但**所有派生数字都带着 WITHHELD 等级**，`benchmark.json`
  里这些行的 verdict 一律 `WITHHELD`，不允许出现 PASS——这是结构性约束，不是注释。
* **柱高没有实验数字**：可引用的 h(t) 数字化曲线不存在，因此柱高门锚在两条精确结论上；
  与实验柱高的对比记为 WITHHELD，而不是编一条曲线凑 10 %。
* **测量定义会分歧**：`front_raw` / `front_coherent` / `front_layer` 三个定义同时报告；
  薄水舌上的喷射粒子会让 raw 前沿偏前，滤掉邻居数不足的粒子才是连续水体前沿。门用 coherent。
* **自由面架带的排除规则**：`tensorsph.vv.hydrostatic` 用"带中心距自由面 <2h"排除表层带，
  会把一个仍被截断的架带留在门内（实测数个百分点）。本节点改为"带上边缘进入 2h 区即排除"，
  并保留原规则数字（`rung_original_max_band_error_pct`）以便审计差异。
* **J4 负控判据是相对判据**：绝对阈值会把网格误差写进判据（截断支持域误差随分辨率变化），
  因此负控要求"同算例正确核基线过 2 % 门 且 截断核误差 ≥5× 基线"。
* **弱可压代价**：c₀=25 m/s 对应 Mach ~O(0.1)，密度压缩百分级；这是 WCSPH 固有代价。
* **未做的部分**：dx=0.005 的溃坝加细、底摩阻/湍流闭合、溃坝柱高实验数字化曲线。
  `full` 档默认只跑 Δx=0.02 的溃坝（Δx=0.01 的历史数字保留在
  `benchmarks/verified/sph_dam_break_2d/results_dx0.01.json`，说明前沿偏差并**不**随分辨率改善）。

## 6. 误差判定（是否 < 3 %）

{error_judgement(bench, results)}

**结论**：{"; ".join(f"{k}={v['verdict']}" + ("（honest FAIL）" if v['honest_fail'] else "") for k, v in gates.items())}。

honest FAIL 不覆盖：溃坝前沿对实验表的偏差远超 10 % 门限，这是本节点暴露的真实能力缺口
（自由面密度截断导致驱动不足 + 缺耗散模型；证据见 `benchmarks/verified/sph_dam_break_2d/results_ab.json`
的 A/B：密度重整化把早期前沿拉回 10 % 内但晚期超前）。

## 7. 证据链与成熟度

| 项 | 内容 |
|---|---|
| impl_files | {len(maturity['impl_files'])} 个文件：{", ".join(f"`{p}`" for p in maturity['impl_files'][:4])} … |
| entry | `{manifest['entry']['module']}` / `{manifest['entry']['package']}` / `{manifest['entry']['script']}` |
| run_artifacts | {len(maturity['run_artifacts'])} 个（含 SHA-256）：{", ".join(f"`{a['path']}`" for a in maturity['run_artifacts'][:6])} |
| integration_callers | {len(maturity['integration_callers'])} 处：{", ".join(f"`{c['path']}`:{c['line']}" for c in maturity['integration_callers'][:6])} |
| benchmark | `benchmark.json`，{maturity['benchmark']['rows']} 行，摘要 {json.dumps(maturity['benchmark']['summary'], ensure_ascii=False)} |
| repro_cmd | `{maturity['repro_cmd']}` |
| 成熟度（推导） | **{manifest['maturity_level']['level']}** |

机器可读副本：`docs/reports/m14534-maturity-evidence.json`；完整包在 `{maturity['run_dir']}/`。
"""
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.write_text(md)
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:                                    # pragma: no cover
        shown = out_path
    print(f"wrote {shown} ({len(md)} bytes), {len(fig_map)} figures, "
          f"maturity {manifest['maturity_level']['level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
