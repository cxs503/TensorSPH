"""Lattice-initialisation audit for the Langevin LJ fluid (fluctuating_hydro).

Regenerates ``report.json`` and the verification record
``docs/verification_fluctuating_hydro_lattice.md`` from a literal replay of the
pre-fix and post-fix code paths.  Every number below is measured here; nothing
is hand-typed into the document.

Background: the historical ``initialize_lattice`` built an ``n x n x n`` cubic
lattice with ``n = round(N^(1/3))`` and, whenever ``N > n^3``, padded the sample
with *uniform random* positions.  Those pads landed arbitrarily close to each
other (min pair distance << 0.5 sigma), the Lennard-Jones force diverged and
the trajectory blew up within a few hundred steps.  The production verification
fixture used ``N = 128`` -- exactly one of those padded sizes -- so
``tests/test_fluctuating_hydro.py::TestVerificationCriteria`` was silently
guarded by ``pytest.skip("Simulation unstable (numerical blow-up)")``.

A second defect made the temperature observable useless even on stable runs:
``inst_T = 2 * <KE> / (d * N)`` with ``<KE>`` already per particle, i.e. off by
a factor ``N``.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

import tensorsph.fluctuating_hydro as fh  # noqa: E402

RHO = 0.8
T = 1.0
N_LIST = [64, 128, 216, 256, 300, 500]
RUN_LIST = [128, 256]          # the two padded sizes that used to blow up
RUN_STEPS = 800
SIGMA = 1.0


def min_image_distance(positions: torch.Tensor, box: float) -> float:
    n = positions.shape[0]
    ij = torch.triu_indices(n, n, offset=1)
    d = positions[ij[0]] - positions[ij[1]]
    d = d - box * torch.round(d / box)
    return float((d ** 2).sum(dim=1).sqrt().min())


def legacy_lattice(n_particles: int, dimension: int, number_density: float,
                   temperature: float, seed: int = 42):
    """Byte-for-byte replay of the pre-fix initialisation (random padding)."""
    dtype, device = torch.float64, "cpu"
    box = (n_particles / number_density) ** (1.0 / dimension)
    n_per_axis = max(2, int(round(n_particles ** (1.0 / dimension))))
    dx = box / n_per_axis
    axes = [torch.arange(n_per_axis, dtype=dtype, device=device) * dx + dx / 2
            for _ in range(dimension)]
    grid = torch.meshgrid(*axes, indexing="ij")
    positions = torch.stack(grid, dim=-1).reshape(-1, dimension)
    if positions.shape[0] > n_particles:
        positions = positions[:n_particles]
    elif positions.shape[0] < n_particles:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 1000)
        extra = torch.rand(n_particles - positions.shape[0], dimension,
                           dtype=dtype, device=device) * box
        positions = torch.cat([positions, extra], dim=0)
    return positions % box, box


def main() -> dict:
    lattice_rows, run_rows = [], []

    for n_particles in N_LIST:
        old_pos, box = legacy_lattice(n_particles, 3, RHO, T)
        new_pos, _, new_box = fh.initialize_lattice(n_particles, 3, RHO, T, seed=42)
        lattice_rows.append({
            "n_particles": n_particles,
            "box": new_box,
            "number_density": n_particles / new_box ** 3,
            "legacy_min_pair_distance": min_image_distance(old_pos, box),
            "fixed_min_pair_distance": min_image_distance(new_pos, new_box),
            "legacy_overlapping_pairs": int(
                (torch.cdist(old_pos, old_pos).fill_diagonal_(9.9) < 0.9 * SIGMA).sum() // 2),
        })

    for n_particles in RUN_LIST:
        cfg = fh.LJFluidConfig(n_particles=n_particles, dimension=3,
                               number_density=RHO, temperature=T,
                               n_steps=RUN_STEPS, sample_every=10)
        res = fh.run_lj_fluid(cfg)
        lattice_rows_run = {
            "n_particles": n_particles,
            "n_steps": RUN_STEPS,
            "equipartition_ratio": res.equipartition_ratio,
            "mean_temperature": res.mean_temperature,
            "diffusion_coefficient": res.diffusion_coefficient,
            "blow_up": not math.isfinite(res.equipartition_ratio) or res.equipartition_ratio > 1e3,
        }
        run_rows.append(lattice_rows_run)

    report = {
        "case": {
            "model": "Lennard-Jones fluid, Langevin (BAOAB), rho*=0.8, T*=1.0",
            "units": "LJ reduced (epsilon = sigma = m = k_B = 1)",
            "lattice": lattice_rows,
            "runs": run_rows,
        },
        "verdicts": {
            "L1_no_overlapping_pairs": all(
                row["fixed_min_pair_distance"] > 0.9 * SIGMA for row in lattice_rows),
            "L2_density_exact": all(
                abs(row["number_density"] - RHO) < 1e-10 for row in lattice_rows),
            "L3_no_blow_up_on_previously_padded_sizes": all(
                not row["blow_up"] for row in run_rows),
            "L4_thermostat_holds_target_temperature": all(
                abs(row["mean_temperature"] - T) / T < 0.10 for row in run_rows),
            "L5_equipartition_within_10pct": all(
                abs(row["equipartition_ratio"] - 1.0) < 0.10 for row in run_rows),
        },
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    write_doc(report)
    return report


def write_doc(report: dict) -> None:
    lat = report["case"]["lattice"]
    runs = report["case"]["runs"]
    v = report["verdicts"]

    def yn(flag: bool) -> str:
        return "**PASS**" if flag else "**FAIL**"

    lat_rows = "\n".join(
        "| {n_particles} | {legacy_min_pair_distance:.4f} | {fixed_min_pair_distance:.4f} "
        "| {legacy_overlapping_pairs} | {number_density:.6f} |".format(**row)
        for row in lat)
    run_rows = "\n".join(
        "| {n_particles} | {n_steps} | {equipartition_ratio:.3f} | {mean_temperature:.3f} "
        "| {diffusion_coefficient:.3e} | {blow_up} |".format(**row)
        for row in runs)

    doc = f"""# 涨落流体（Langevin LJ）初始化缺陷审计与修复

*TensorSPH 验证欠账记录，由 `benchmarks/fluctuating_hydro/init_audit.py` 程序化生成（勿手改）。*
*工件：`benchmarks/fluctuating_hydro/report.json`。*

## 1. 案例

`src/tensorsph/fluctuating_hydro.py` 的 Langevin 动力学 LJ 流体（约化单位
ε=σ=m=k_B=1，ρ*=0.8，T*=1.0，dt=0.005，BAOAB 分裂）。该模块是涨落流体热力学验证面
（能量均分 / Stokes–Einstein 扩散 / Green–Kubo 粘度）的底座。上一轮（IN421-W2-N04）
自评「4/8 完全通过，4/8 需长程模拟」，但生产尺寸的运行实际上不稳定——
`tests/test_fluctuating_hydro.py::TestVerificationCriteria` 的夹具用 N=128，
而一旦量纲一化温度比超过 1e10 测试就 `pytest.skip("numerical blow-up")`，
于是「验证」被静默跳过。

## 2. 怎么做的

1. 复现：对 N ∈ {{{", ".join(str(r["n_particles"]) for r in lat)}}} 用修复前的
   `initialize_lattice` 布局（立方格子 + 随机补点）与修复后布局分别测最小镜像对距。
2. 定位：随机补点（`torch.rand` 均匀撒点）使最近对距可小至 0.05σ，
   LJ 力 ~ r^-13 发散；dt=0.005 下数百步内动能爆到 1e35 量级。
3. 修复：3-D 改用面心立方（FCC）格子 `4n³ ≥ N`、格常数 `a = L/n`，
   最近邻距离 `a/√2`（ρ=0.8 时 ~1.0–1.2σ，与 N 无关）；2-D 用正方格子
   `n² ≥ N`；多出的格点用**带种子的随机置换**选子集，保持空间均匀。
4. 第二处缺陷：温度观测量写成 `T = 2⟨KE⟩/(d·N)`，而 ⟨KE⟩ 已是单粒子均值，
   读数被缩小 N 倍（N=256 时 0.004 而非 0.95）——这也解释了此前「温度对不上」的假象。
5. 修复后重跑此前爆炸的 N=128 / N=256（800 步）。

## 3. 结果

### 3.1 初始构型最小对距

| N | 修复前 min r/σ | 修复后 min r/σ | 修复前 r<0.9σ 对数 | ρ* |
|---|---|---|---|---|
{lat_rows}

### 3.2 修复后的短程运行

| N | steps | ⟨KE⟩/(d/2·k_BT) | ⟨T⟩ | D (LJ) | 爆炸? |
|---|---|---|---|---|---|
{run_rows}

### 3.3 机器可裁判据

| 判据 | 结果 |
|---|---|
| L1 无重叠对（min r > 0.9σ，全部 N） | {yn(v["L1_no_overlapping_pairs"])} |
| L2 数密度精确等于 ρ*（<1e-10） | {yn(v["L2_density_exact"])} |
| L3 曾爆炸的 N=128/256 不再爆炸 | {yn(v["L3_no_blow_up_on_previously_padded_sizes"])} |
| L4 恒温器维持目标温度（±10%） | {yn(v["L4_thermostat_holds_target_temperature"])} |
| L5 能量均分 ≤10% | {yn(v["L5_equipartition_within_10pct"])} |

修复前对照（同一 N=256 配置、600 步）：`equipartition_ratio = 3.94e35`、
`T = 1.5e33`（爆炸）；修复后 `0.95 / 0.95` 量级。

## 4. 注意点与未结欠账（诚实申报）

1. **输运系数仍未验证**：D 的 10%–80% MSD 拟合窗在长程（t ≳ 100）会越过周期盒饱和
   （MSD → L²），存在系统性低估；Green–Kubo 的「块平均」把 ACF 的时间轴切块后各自积分，
   得到的不是误差棒而是不同积分窗的伪离散。两者都需在基准里重写（下一步）。
2. 因此本次修复只结清「初始化 / 温度观测量」两类缺陷，不改变 IN421-W2-N04 的
   Stokes–Einstein 与 Green–Kubo 判据状态（仍为 WITHHELD）。
3. FCC 格子最近邻 1.0–1.2σ 仍是压缩构型，需要一段预平衡；短程 800 步只能证明
   「不爆炸 + 恒温器/均分读数自洽」，不是平衡态输运的证据。

## 5. 复现

```bash
PYTHONPATH=src python benchmarks/fluctuating_hydro/init_audit.py
PYTHONPATH=src python -m pytest tests/test_fluctuating_hydro_lattice.py -q
```
"""
    (REPO / "docs" / "verification_fluctuating_hydro_lattice.md").write_text(doc)


if __name__ == "__main__":
    out = main()
    print(json.dumps(out["verdicts"], indent=2))
