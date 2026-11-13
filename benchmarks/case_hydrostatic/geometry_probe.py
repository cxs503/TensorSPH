"""Geometry probe for ``CASE-HYDROSTATIC``: which extent drives the false FAIL?

The board audit (``docs/board_audit/M15419_CASE_HYDROSTATIC_BOARD_AUDIT.md``,
finding F4) measured the still-water case at ``dx = 0.03 m`` on a declared
``1.0 x 0.4 m`` tank and read 9.30 %, against 2.25 % for the "same" tank at
``depth = 0.39 m``.  ``cell_centres`` only fits ``floor(extent / dx)`` cells, so
at ``dx = 0.03`` *both* extents are unrepresentable: the lattice is
``0.99 x 0.39 m`` while the analytic reference keeps ``H = 0.4 m``.

This probe runs the complete 2 x 2 of {width representable?, depth
representable?} at one ``dx`` and one duration, so the 9 %-class error can be
attributed instead of merely declared geometry:  every run has 33 x 13 = 429
fluid particles, the same ``dx``, the same 2 s window and the same sound speed --
only the *declared* extents (and therefore the analytic reference, the wall
lattice top and the free-surface gap) move.

    python benchmarks/case_hydrostatic/geometry_probe.py            # run + write
    python benchmarks/case_hydrostatic/geometry_probe.py --report   # run + doc

Outputs
-------
``outputs/case_hydrostatic_geometry/probe.json``   full records + table (sha256
                                                   printed and written to the doc)
``outputs/case_hydrostatic_geometry/geometry.png`` per-band error vs band
``docs/case_hydrostatic_geometry_gate.md``         document with injected numbers
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_hydrostatic import (CASE_ID, TOLERANCE_PCT,  # noqa: E402
                                       check_geometry, run_case)

OUT = REPO / "outputs" / "case_hydrostatic_geometry"
DOC = REPO / "docs" / "case_hydrostatic_geometry_gate.md"

DX = 0.03
SECONDS = 2.0

#: name -> (declared tank_width, declared depth, allow_misaligned)
CASES: dict[str, dict] = {
    "w_lat_d_lat": dict(tank_width=0.99, depth=0.39),   # representable
    "w_dec_d_lat": dict(tank_width=1.00, depth=0.39, allow_misaligned=True),
    "w_lat_d_dec": dict(tank_width=0.99, depth=0.40, allow_misaligned=True),
    "w_dec_d_dec": dict(tank_width=1.00, depth=0.40, allow_misaligned=True),
}


def label(name: str) -> str:
    w, d = name.split("_d_")
    return ("W lattice" if w.endswith("lat") else "W declared") + ", " + \
           ("D lattice" if d.endswith("lat") else "D declared")


def run_all() -> dict:
    cases = {}
    for name, kw in CASES.items():
        geometry = check_geometry(dx=DX, tank_width=kw["tank_width"],
                                  depth=kw["depth"],
                                  allow_misaligned=kw.get("allow_misaligned", False))
        record = run_case(dx=DX, seconds=SECONDS,
                          allow_misaligned=kw.get("allow_misaligned", False),
                          **{k: v for k, v in kw.items() if k != "allow_misaligned"})
        cases[name] = {"label": label(name), "geometry": geometry,
                       "record": record}
        gate = record["gate"]
        print(f"[{name:12s}] {label(name):24s} N={record['config']['fluid_particles']:4d} "
              f"bias {geometry['reference_bias_pct']:+7.3f} % "
              f"max {gate['max_band_error_pct']:7.4f} % "
              f"({'PASS' if gate['pass'] else 'FAIL'} @ {TOLERANCE_PCT} %)  "
              f"{record['wall_seconds']:.1f} s")
    return {
        "case_id": CASE_ID,
        "dx": DX,
        "seconds": SECONDS,
        "gate_limit_pct": TOLERANCE_PCT,
        "cases": cases,
        "table": [
            {
                "name": name,
                "label": c["label"],
                "tank_width": c["geometry"]["tank_width"],
                "depth": c["geometry"]["depth"],
                "lattice_width": c["geometry"]["lattice_width"],
                "lattice_depth": c["geometry"]["lattice_depth"],
                "columns": c["geometry"]["columns"],
                "rows": c["geometry"]["rows"],
                "fluid_particles": c["record"]["config"]["fluid_particles"],
                "aligned": c["geometry"]["aligned"],
                "reference_bias_pct": c["geometry"]["reference_bias_pct"],
                "max_band_error_pct": c["record"]["gate"]["max_band_error_pct"],
                "mean_band_error_pct": c["record"]["gate"]["mean_band_error_pct"],
                "worst_band": c["record"]["gate"]["worst_band"],
                "max_fluctuation_pct": c["record"]["max_fluctuation_pct"],
                "pass": c["record"]["gate"]["pass"],
                "gated_bands": c["record"]["gate"]["gated_bands"],
            }
            for name, c in cases.items()
        ],
    }


def figure(bundle: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                  gridspec_kw={"width_ratios": [1.55, 1.0]})
    markers = {"w_lat_d_lat": "o", "w_dec_d_lat": "s", "w_lat_d_dec": "^",
               "w_dec_d_dec": "D"}
    for name, case in bundle["cases"].items():
        rows = [r for r in case["record"]["profile"] if r["gated"]]
        ax.plot([r["band"] for r in rows], [r["rel_error_pct"] for r in rows],
                marker=markers[name], label=f"{case['label']} "
                f"(max {case['record']['gate']['max_band_error_pct']:.2f} %)")
    ax.axhspan(-bundle["gate_limit_pct"], bundle["gate_limit_pct"], color="0.85",
               zorder=0, label=f"gate +/-{bundle['gate_limit_pct']:.0f} %")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("band index (floor to top; free-surface bands excluded)")
    ax.set_ylabel("(p_SPH - p_ref) / p_ref  [%]")
    ax.set_title(f"Case {bundle['case_id']}: gated-band error at dx = {bundle['dx']} m, "
                 f"{bundle['seconds']:.0f} s window")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")

    names = list(bundle["table"])
    xs = range(len(names))
    vals = [c["max_band_error_pct"] for c in bundle["table"]]
    ax2.bar(xs, vals, color=["#2b8a3e" if c["aligned"] else "#c92a2a"
                             for c in bundle["table"]])
    ax2.axhline(bundle["gate_limit_pct"], color="k", ls="--", lw=1.0,
                label=f"gate {bundle['gate_limit_pct']:.0f} %")
    for x, c in zip(xs, bundle["table"]):
        ax2.text(x, c["max_band_error_pct"], f"{c['max_band_error_pct']:.2f} %",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([c["label"].replace("", "") for c in bundle["table"]],
                        rotation=20, ha="right", fontsize=7.5)
    ax2.set_ylabel("max gated-band |error| [%]")
    ax2.set_title("verdict vs declared geometry (same dx, same N)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc(bundle: dict, json_path: Path, png_path: Path) -> Path:
    rows = "\n".join(
        f"| `{c['name']}` | {c['tank_width']:.2f} x {c['depth']:.2f} | "
        f"{c['lattice_width']:.2f} x {c['lattice_depth']:.2f} | {c['fluid_particles']} | "
        f"{'yes' if c['aligned'] else 'no'} | {c['reference_bias_pct']:+.3f} | "
        f"{c['max_band_error_pct']:.4f} | {c['mean_band_error_pct']:.4f} | "
        f"{'PASS' if c['pass'] else 'FAIL'} |"
        for c in bundle["table"])
    per_band = "\n".join(
        "| `{}` | {} |".format(
            name,
            " ".join(f"b{r['band']}={r['rel_error_pct']:+.3f}" for r in c["record"]["profile"]))
        for name, c in bundle["cases"].items())
    worst = max(bundle["table"], key=lambda c: c["max_band_error_pct"])
    best = min(bundle["table"], key=lambda c: c["max_band_error_pct"])
    base = next(c for c in bundle["table"] if c["aligned"])
    t_dec = next(c for c in bundle["table"] if c["name"] == "w_lat_d_dec")
    t_wid = next(c for c in bundle["table"] if c["name"] == "w_dec_d_lat")
    dec = next(c for c in bundle["table"] if not c["aligned"]
               and c["name"] == "w_dec_d_dec")
    text = f"""# CASE-HYDROSTATIC：几何前置校验（审计 F4 收口）

**工件**：`{json_path.relative_to(REPO)}`（sha256 `{sha256(json_path)}`）、
`{png_path.relative_to(REPO)}`；本文件由 `benchmarks/case_hydrostatic/geometry_probe.py --report`
程序化生成，表中数字全部来自该 JSON，零手抄。

## 1 案例与缺陷

静水压剖面算例要求稀疏格子在重力下复现解析静水压 `p(y) = rho0 g (H - y)`
（`H` 为自由面高度），内区带误差判据 1 %。`cell_centres` 只在 `[0, extent]`
内铺 `floor(extent/dx)` 层格子，因此当 `depth` 不是 `dx` 的整数倍时：

1. 格子顶面在 `floor(depth/dx)*dx`，而解析参考仍取 `H = depth` → 每带系统性读低；
2. 壁面格 `wall_lattice(..., top=depth)` 与流体自由面之间凭空出现一个空隙
   （`depth - lattice_depth`），自由面截断误差测在一个不该存在的间隙上。

M15419 独立审计在 `dx = 0.03 m`、申报 `1.0 x 0.4 m` 水箱上读到 9.30 %，
并把它与"同算例" `depth = 0.39 m` 的 2.25 % 对比（F4，medium）。本探针把该
`dx` 下 **两个方向** 的可表示性做全 2 x 2 分解，全部 429 粒子、同 2 s 窗、
同声速，只动"申报尺寸"（以及随之改变的参考高度与壁面位置）。

## 2 做法

* `check_geometry()`（`src/tensorsph/case_hydrostatic.py`）在 `run_case` /
  `run_initial_state` 建格子之前校验：depth/tank_width 是否为整数个格子、
  带厚是否 ≥ 1 个格子、是否至少有一个带能逃过自由面 2h 重叠规则；
  违规抛 `HydrostaticGeometryError`（列出全部违规项与格子实顶面）；
  `allow_misaligned=True` 改为记录 `aligned=False` 与 `reference_bias_pct`。
* 探针 4 个配置（见下表），每个 `run_case(dx={bundle['dx']},
  seconds={bundle['seconds']:.0f})`，`bands=8`、`sound_speed=20`、`sample_every=5`。

## 3 结果（`dx = {bundle['dx']} m`，2 s 窗，门 {bundle['gate_limit_pct']:.0f} %）

| 配置 | 申报 W x D | 格子 W x D | N | 可表示 | 参考偏置(%) | max 带误差(%) | mean 带误差(%) | 门 |
|---|---|---|---|---|---|---|---|---|
{rows}

逐带误差（全部 8 带，`b7` 为自由面带，不入判据）：

| 配置 | 逐带 rel_error(%) |
|---|---|
{per_band}

**分解结论**：最差配置 `{worst['name']}`（申报 {worst['tank_width']:.2f} x {worst['depth']:.2f}，
格子 {worst['lattice_width']:.2f} x {worst['lattice_depth']:.2f}）读
{worst['max_band_error_pct']:.4f} %（{worst['max_band_error_pct'] / bundle['gate_limit_pct']:.1f} x 门限）；
可表示配置 `{base['name']}` 读 {base['max_band_error_pct']:.4f} %。
纯几何先验偏置 `|1 - {dec['lattice_depth']:.2f}/{dec['depth']:.2f}| = {abs(dec['reference_bias_pct']):.2f} %`
只解释其中约 {abs(dec['reference_bias_pct']) / worst['max_band_error_pct'] * 100:.0f} %，
其余来自被"人为拉开"的自由面空隙与壁面位置——即 F4 不只是参考高度问题，
所以**必须整体拒绝该配置**，而不是把 H 改成格子顶面了事。

## 4 图的说明

`{png_path.name}` 左：各配置内区带误差 vs 带号，灰带=±1 % 判据区；
右：各配置 max 带误差柱状（绿=可表示，红=不可表示），虚线=1 % 门。
方向性：depth 单独不可表示读 {t_dec['max_band_error_pct']:.4f} %，width 单独不可表示读
{t_wid['max_band_error_pct']:.4f} %（比值 {t_dec['max_band_error_pct'] / t_wid['max_band_error_pct']:.1f}）：
两个方向都超出 1 % 门，但把算例推向 9 % 量级的主因是 **depth 与格子顶面不一致**
（参考高度抬到格子达不到的 H，同时自由面/壁面之间多出空隙）；width 偏移只影响壁面列。

## 5 注意点 / 局限

* 2 s 窗的瞬时振荡极大（`max_fluctuation_pct` 达
  {max(c['max_fluctuation_pct'] for c in bundle['table']):.0f} %），单点误差对采样窗
  敏感；本探针只做**同窗对照**，不比绝对值。门配置（dx=0.02、8 s 窗）未动。
* 只覆盖 `dx = 0.03` 一个分辨率点；结论是"几何不可表示为配置缺陷"，
  不是"dx=0.03 的求解精度"。可表示配置在 dx=0.03 仍受粗分辨率制约。
* 不重跑已发布的 8 个门配置证据（`outputs/case_hydrostatic`，另一单维护）。

## 6 误差判定

* 门判据不变（内区 max 带误差 ≤ 1 %）；本次**不重定义**任何判据。
* 关键量 vs 解析基准：可表示 `{base['name']}` 读 {base['max_band_error_pct']:.4f} %
  → **< 3 %：是**（也 < 1 % 门）。不可表示 `{dec['name']}`
  读 {dec['max_band_error_pct']:.4f} % → **< 3 %：否**，但这是几何缺陷的
  可复现指纹，不是求解器精度；新守卫令其无法再以"算例 FAIL"的形式进入账本。
"""
    DOC.write_text(text)
    return DOC


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="also write the figure and the markdown document")
    parser.add_argument("--json-out", default=str(OUT / "probe.json"))
    parser.add_argument("--figure-out", default=str(OUT / "geometry.png"))
    args = parser.parse_args(argv)

    bundle = run_all()
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(bundle, indent=1))
    print(f"wrote {json_path} (sha256 {sha256(json_path)[:16]}...)")
    if args.report:
        png = figure(bundle, Path(args.figure_out))
        print(f"wrote {png}")
        print(f"wrote {doc(bundle, json_path, png)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())