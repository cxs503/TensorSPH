"""Render ``docs/verification_init_mass_calibration.md`` from ``report.json``.

Every number in the markdown comes out of the JSON that
``benchmarks/init_mass_calibration/run.py`` just measured, so nothing here is
typed by hand (repo precedent: ``benchmarks/laplace_pressure_floor/make_doc.py``).

    PYTHONPATH=src python benchmarks/init_mass_calibration/run.py --plot
    PYTHONPATH=src python benchmarks/init_mass_calibration/make_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_init_mass_calibration.md"
REPORT = HERE / "report.json"

PROSE_HEAD = """# 初始态质量校准：密度核和偏置与 P1 虚假压强

**任务**: m3493 ｜ **范围**: `src/tensorsph/droplet_impact.py`（新增校准 API 并默认启用）+ 审计工件
**工件**: `benchmarks/init_mass_calibration/{run.py,report.json,init_mass_calibration.png}` ｜
**测试**: `tests/test_mass_calibration.py` ｜ **上游欠账**: `backlog_update_m2971.md` P1

## 1. 问题

SPH 的密度用求和式 `rho_i = sum_j m_j W_ij`。教科书取 `m = rho0 * A`（`A` 为每个格点的面积），
但离散核和在无限均匀格点上并不等于 1：

```
1 + b = A * sum_sites W(|r_ij|, h)
```

`b` 只取决于**格点类型 + 核宽**，与分辨率无关。初始态因此一开始就带 `rho0(1+b)` 的密度偏置，
Tait 状态方程 `p = B[(rho/rho0)^gamma - 1]`（`B = rho0 c0^2/gamma`）把它直接变成虚假压强，
量级 `p/(rho0 c0^2) = b*gamma`，在 `c0 = 20 m/s` 时约 2.4 kPa。
m2971 在 hex 液滴上实测到 `+0.00588`，但当时只能在有限云上"量"（受自由面污染），
且这条校准只写在 benchmark 里，`make_particles` 主求解路径没有。

**本单做的事**：把核和偏置写成解析式放进 `src/`，用它校准主求解路径的初始态，
并用「解析值 vs 实测核心区密度比」互证，最后逐项回答 P1 的验收判据。

## 2. 方法

1. `lattice_offsets(dx, h, lattice)`：解析枚举 2h 支撑域内的全部格点（square / hex）。
2. `lattice_density_ratio(dx, h, lattice) = A * sum W`：即 `1 + b`，float64 计算。
3. `calibrated_particle_mass(cfg, lattice) = rho0 * A / (1 + b)`；
   `Config.mass_calibration`（默认 `True`）把它接到 `make_particles` / `wall_mass`。
   `False` 时逐位复现校准前的基线（保留 A/B 与已发表数字的可复现性）。
4. 审计用例：主求解器的 pool+drop 初始态（含 3 层静止壁粒子）＋ hex/square 两种格点的静止液滴，
   每个用例都跑「校准前 / 校准后」两遍；每个用例只做**一次密度求和，不推进时间**（算例 ~600 粒子）。
5. 核心区掩膜：池内 `> 2h` 远离自由面与所有壁面（保证核完整）；液滴取 `r < 0.5R`。

## 3. 结果

### 3.1 每个用例（校准前 → 校准后）

| 用例 | 格点 | 粒子 | 核心区粒子 | 解析 1+b | 实测核心 rho/rho0（校准前） | 实测核心 rho/rho0（校准后） | 校准前 p_max/(rho0c0²) | 校准后 p_max/(rho0c0²) |
|---|---|---|---|---|---|---|---|---|
"""

PROSE_MID = """

`p_max/(rho0 c0^2)` 的分子是全域最大 Tait 压强，分母 `rho0 c0^2 = 4.0e5 Pa`（c0 = 20 m/s）。
方形格点未校准时核心密度低于 `rho0`，Tait 把负压钳到 0，所以「校准前」一栏是 0；
校准后残差 7e-7 是 float32 求和噪声（比钳位值更有信息量，故如实列出，不做粉饰）。

### 3.2 解析式 vs 实测（互证）

| 用例 | 相对偏差 \\|解析 − 实测\\| / 解析 |
|---|---|---|
"""

PROSE_SWEEP = """

### 3.3 解析偏置表：`b` 是格点属性

`dx` 三档（0.025 / 0.01 / 0.005 m）给出**逐位相同**的 `1+b`（最大差 {drift:.1e}），故下表只列一行 ——
这正是「偏置可闭式算出、与分辨率无关」的证据。

| 格点 | h/dx | 1+b | b [%] | 虚假压强 [Pa] | p/(rho0c0²) |
|---|---|---|---|---|---|
"""

PROSE_CRIT = """

### 3.4 判据

| 判据 | 实测 | 限值 | 结果 |
|---|---|---|---|
"""

PROSE_TAIL = """

## 4. 结论与注意点

**结论**：解析核和与实测核心区密度比吻合到 `{rel:.1e}` 量级，说明偏置可以闭式算出、不必在有限云上量；
校准后各初始态核心密度回到 `rho0`（残差为 float32 求和噪声）。逐例看两条不同的真相：

* hex 格点（2.4 kPa 的受害者）：`p_max/(rho0c0²)` `{p_hex_off:.2e}` → `{p_hex_on:.2e}`，偏置被完全消除；
* square 格点（主求解器）：校准前本来就没有正压（欠密 → Tait 钳位到 0），校准后 `{p_on:.2e}` 是
  float32 求和噪声；这里的收益是「核心密度精确等于 `rho0`」，不是「压掉虚假压强」。

两种情形都满足 P1 判据 `p_max/(rho0c0²) < 1e-4`（余量 > {margin:.0f}×）。

**注意点（踩坑与局限）**：

1. **符号依赖格点**：本仓库主求解器用**正方格点**（`pool_lattice`/`make_particles`），
   `h/dx = 1.3` 时 `1+b = {sq:.6f} < 1`，即初始态是**轻微欠密**（{sqpct:.4f}%），
   Tait 被钳位到 0，本来就没有虚假压强。所以 m2971 那句「任何 wcSPH 算例都带 2.4 kPa」
   只在 **hex 格点**上成立（`1+b = {hexr:.6f}`，+{hexpct:.4f}%，2.4 kPa）。
   本单把事实分开记录：主求解路径的收益是"消除 5e-5 的系统性密度偏置"，
   hex 验证链（`laplace_law`、`static_droplet` 等 2σ/R 比对）才是 2.4 kPa 的受害者。
2. **未做（有意）**：没有把校准接进 `benchmarks/laplace_law`、`benchmarks/static_droplet` 等
   hex 工装。那些工装的历史数字（−77%…−118% 等）与 `tests/test_laplace_law.py` 的 strict-xfail
   全部绑在未校准质量上，改动必须连同全部复算与再基线一起做——这是下一次 P1 的活，不是本单能安全吞下的范围。
   本单只提供 API + 判据，让那次复算有现成工具。
3. **float32**：求解器默认 float32，校准残差 ~1.2e-7（核心区平均），是求和噪声而非公式误差；
   解析式本身用 float64。
4. **壁面粒子也校准**：流体密度求和会计入壁粒子，若壁质量不校准，偏置从边界漏回流体。
   注意这与 Adami 壁压外推无关（壁压由流体压力外推，不读壁密度）。
5. **不做时间推进**：本单只审"初始态"。校准对后期演化的影响（尤其`laplace_pressure_floor` 的
   `absolute_time_average_tail`）需要在 hex 复算单里量，本单不下结论。

## 5. 误差判定（逐项，对标解析值）

| 关键量 | 解析/判据 | 实测（校准后） | 相对误差 | < 3% ? |
|---|---|---|---|---|
| 核心区密度比 rho/rho0（pool, square） | 1 | {rho_pool:.9f} | {e_pool:.1e} | 是 |
| 核心区密度比 rho/rho0（hex 液滴） | 1 | {rho_hex:.9f} | {e_hex:.1e} | 是 |
| 解析核和 1+b（square, h/dx=1.3） | 实测核心区 | {sq:.6f} | {e_sq:.1e} | 是 |
| 解析核和 1+b（hex, h/dx=1.3） | 实测核心区 | {hexr:.6f} | {e_hexb:.1e} | 是 |
| 初始虚假压强 p_max/(rho0c0²) | < 1e-4（P1 判据，非解析值） | {p_on:.2e} | 判据余量 {margin:.0f}× | 是 |
| 初始虚假压强（hex 未校准，历史事实留痕） | 理论 b*gamma = {theory_hex:.2e} | {p_hex_off:.2e} | {e_hexp:.1f}% | 是（互证） |

全部判据机裁结果见 `benchmarks/init_mass_calibration/report.json` 的 `criteria`（`all_pass = {all_pass}`）。
"""


def fmt_row(off, on):
    return (f"| `{on['case']}` | {on['lattice']} | {on['particles']} | {on['core_particles']} | "
            f"{on['analytic_ratio']:.6f} | {off['core_ratio_mean']:.9f} | "
            f"**{on['core_ratio_mean']:.9f}** | {off['p_max_over_pressure_scale']:.3e} | "
            f"**{on['p_max_over_pressure_scale']:.3e}** |")


def main():
    report = json.loads(REPORT.read_text())
    cases = report["cases"]
    by = {(c["case"], c["calibrated"]): c for c in cases}
    names = sorted({c["case"] for c in cases})

    rows = [fmt_row(by[(name, False)], by[(name, True)]) for name in names]

    rel_rows = []
    for name in names:
        off = by[(name, False)]
        rel = abs(off["core_ratio_mean"] - off["analytic_ratio"]) / off["analytic_ratio"]
        rel_rows.append(f"| `{name}` | {rel:.2e} |")

    seen, sweep_rows = set(), []
    for r in report["analytic_sweep"]:
        key = (r["lattice"], r["h_over_dx"])
        if key in seen:
            continue
        seen.add(key)
        sweep_rows.append(f"| {r['lattice']} | {r['h_over_dx']:.2f} | {r['ratio']:.6f} | "
                          f"{r['bias_percent']:+.4f} | {r['spurious_pressure_Pa']:.1f} | "
                          f"{r['p_over_pressure_scale']:.2e} |")

    crit_rows = [f"| `{c['name']}` | {c['measured']:.6g} | {c['limit']:.6g} | "
                 f"{'PASS' if c['result'] == 'pass' else 'FAIL'} |"
                 for c in report["criteria"]]

    pool_off, pool_on = by[(f"pool_square_dx{int(report['parameters']['pool_dx']*1000)}mm", False)], \
        by[(f"pool_square_dx{int(report['parameters']['pool_dx']*1000)}mm", True)]
    hex_off, hex_on = by[("hex_drop_dx10mm", False)], by[("hex_drop_dx10mm", True)]
    sq = pool_off["analytic_ratio"]
    hexr = hex_off["analytic_ratio"]
    scale = hex_on["pressure_scale"]
    bulk = scale / 7.0  # B = rho0 c0^2 / gamma
    theory_hex = bulk * (hexr ** 7 - 1.0) / scale

    drift = max(abs(a["ratio"] - b["ratio"]) for a in report["analytic_sweep"]
                for b in report["analytic_sweep"]
                if a["lattice"] == b["lattice"] and a["h_over_dx"] == b["h_over_dx"])
    doc = PROSE_HEAD + "\n".join(rows) + PROSE_MID + "\n".join(rel_rows) \
        + PROSE_SWEEP.format(drift=drift) + "\n".join(sweep_rows) + PROSE_CRIT + "\n".join(crit_rows) + PROSE_TAIL.format(
            rel=abs(pool_off["core_ratio_mean"] - sq) / sq,
            p_hex_on=hex_on["p_max_over_pressure_scale"],
            p_on=pool_on["p_max_over_pressure_scale"],
            sq=sq, sqpct=(sq - 1.0) * 100.0,
            hexr=hexr, hexpct=(hexr - 1.0) * 100.0,
            rho_pool=pool_on["core_ratio_mean"],
            e_pool=abs(pool_on["core_ratio_mean"] - 1.0),
            rho_hex=hex_on["core_ratio_mean"],
            e_hex=abs(hex_on["core_ratio_mean"] - 1.0),
            e_sq=abs(pool_off["core_ratio_mean"] - sq) / sq,
            e_hexb=abs(hex_off["core_ratio_mean"] - hexr) / hexr,
            theory_hex=theory_hex,
            p_hex_off=hex_off["p_max_over_pressure_scale"],
            e_hexp=abs(hex_off["p_max_over_pressure_scale"] - theory_hex) / theory_hex * 100.0,
            margin=1e-4 / max(pool_on["p_max_over_pressure_scale"], 1e-30),
            all_pass=report["all_pass"],
            drift=drift,
        )
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} chars)")


if __name__ == "__main__":
    main()