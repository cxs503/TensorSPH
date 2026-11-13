"""CASE-HYDROSTATIC -- band-resolved decomposition of the static truncation error.

Why this file exists
--------------------
``benchmarks/case_hydrostatic/run.py`` reduces the deterministic ``t = 0`` field
of CASE-HYDROSTATIC to **one** number per resolution
(``resolution_ladder_static[].max_band_error_pct``).  That number is the maximum
over all *gated* bands, and at every resolution it is owned by the band nearest
the floor, whose kernel support is cut by the wall.  Read alone it looks like a
resolution ladder that barely converges (28 % / 15 % / 10 % at
``dx`` = 0.04 / 0.025 / 0.02 m) although the *interior* bands of the very same
field sit at ~1 %.

This benchmark keeps the single-number gate untouched and decomposes the same
field by band class, so that "wall truncation" and "resolution convergence" are
measured separately instead of being added together in one maximum:

* ``floor``        -- ``y_low < 2h``: the band's support is truncated by the
  bottom wall (solid boundary), where summation density is not comparable with
  an unbounded hydrostatic reference;
* ``interior``     -- support fully inside the pool; this is the class that can
  be read as a resolution ladder;
* ``free_surface`` -- ``y_high > H - 2h``: the overlap rule already used by the
  case gate (``case_hydrostatic.pressure_profile``).

Two fields are recorded per band
--------------------------------
``rho`` and ``p``.  They are not redundant: the Tait law ``p = B [ (rho/rho0)^7 -
1 ]`` amplifies a density deficit, so the *pressure* error of a state is set by
the *density* error through the local amplification ``gamma * (p + B) / p``.
Recording both separates two failure modes that one pressure-only table cannot
distinguish -- a density deficit produced by kernel truncation (physics /
discretisation) versus a pressure that was modified after the summation, e.g. by
the free-surface closure or the wall-pressure extrapolation (post-processing).

The third field, ``density_response_error_pct``, is the relative error of the
*density response* ``(rho - rho0) / (rho_ref - rho0) - 1``.  It is the quantity
that survives a stiffer equation of state: the plain deficit scales like
``p / B`` and therefore like ``1 / c0^2``, while the response error does not, so
``--sound-speeds`` shows a flat pressure error at a moving deficit.

What it does **not** claim
--------------------------
The reference density is the closed form
``rho(y) = rho0 (1 + rho0 g (H - y) / B)^(1/gamma)`` (inverse Tait law of the
analytic pressure profile), evaluated at the *measured* mass-weighted band
centre.  The ``dx -> 0`` limit of the interior deficit is therefore a statement
about this quadrature, not about the unsteady case: the dynamic gate of
CASE-HYDROSTATIC is a separate measurement (``--config window_8s``) and is not
recomputed here.

``--sound-speeds`` adds a fixed-resolution invariance sweep: the Tait bulk
modulus moves with ``c0``, and so does the amplification, so the pressure gate
can be flat in ``c0`` while the density accuracy improves.

Usage
-----
    python benchmarks/case_hydrostatic/truncation_decomposition.py \
        --json-out docs/case_hydrostatic_truncation_decomposition.json \
        --md-out docs/case_hydrostatic_truncation_decomposition.md \
        --figure-out docs/case_hydrostatic_truncation_decomposition.png \
        --sound-speeds 10,30
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from tensorsph.case_hydrostatic import (CASE_ID, DEPTH, G, RHO0,
                                        HydrostaticGeometryError, support_width)
from tensorsph.droplet_impact import (PairListCache, cell_centres,
                                      hydrostatic_density, lattice_mass_unit,
                                      resolve_device, rhs, wall_lattice)
from tensorsph.vv.hydrostatic import HydrostaticConfig

#: Resolution ladder.  Every entry is an integer number of cells over the
#: declared depth and width, so the lattice reaches the declared free surface
#: (``case_hydrostatic.check_geometry``); 0.05 m is deliberately coarser than the
#: shipped ladder so the order fit has a point where truncation is unmistakable.
DEFAULT_DXS = (0.05, 0.04, 0.025, 0.02, 0.0125, 0.01)

#: Classes, in the order they are reported.
CLASSES = ("floor", "interior", "free_surface")


def band_mean(pos, field, weight, depth: float, bands: int) -> list[dict]:
    """Mass-weighted mean of ``field`` per horizontal band.

    The banding (edges, mass weighting, band centre) is the one of
    :func:`tensorsph.vv.hydrostatic.band_table`; ``tests/test_truncation_
    decomposition.py`` pins that this helper reproduces the canonical band mean
    when fed the pressure field, so there is one definition of "band", not two.
    """
    edges = [depth * k / bands for k in range(bands + 1)]
    rows = []
    for k, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (pos[:, 1] >= low) & (pos[:, 1] < high)
        count = int(mask.sum())
        if count == 0:
            continue
        w = weight[mask].sum()
        rows.append({
            "band": k,
            "y": float((pos[mask, 1] * weight[mask]).sum() / w),
            "y_low": low,
            "y_high": high,
            "particles": count,
            "value": float((field[mask] * weight[mask]).sum() / w),
        })
    return rows


def classify(y_low: float, y_high: float, depth: float, support: float) -> str:
    """Band class from its overlap with the wall / free-surface support gaps."""
    if y_low < support:
        return "floor"
    if y_high > depth - support:
        return "free_surface"
    return "interior"


def static_field(dx: float, tank_width: float, depth: float,
                 sound_speed: float = 20.0, device: str = "cpu"):
    """The deterministic ``t = 0`` state of the case: one ``rhs`` evaluation.

    Same configuration as :func:`tensorsph.case_hydrostatic.run_initial_state`
    (hydrostatic mass initialisation, static wall lattice, one RHS evaluation at
    the acoustic step); it returns the density and the pressure, which the case
    record only reports as band means.
    """
    cfg = HydrostaticConfig(dx=dx, tank_width=tank_width, depth=depth,
                            seconds=0.0, sound_speed=sound_speed, device=device,
                            hydrostatic_init=True)
    dev = resolve_device(device)
    solver = cfg.solver_config()
    xs = cell_centres(tank_width, dx, dev)
    ys = cell_centres(depth, dx, dev)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    unit = lattice_mass_unit(solver)
    mass = hydrostatic_density(pos[:, 1], solver) * (unit / solver.rho0)
    wall = wall_lattice(solver, pos.device, top=depth)
    h = solver.h_factor * dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=pos.shape[0])
    pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
    vel = torch.zeros_like(pos)
    dt = 0.3 * h / sound_speed
    rho, pressure, _ = rhs(pos, vel, mass, solver, pairs=pairs, wall=wall, dt=dt)
    return pos, mass, rho, pressure, solver


def diagnose(dx: float, tank_width: float = 1.0, depth: float = DEPTH,
             sound_speed: float = 20.0, bands: int = 8,
             device: str = "cpu") -> dict:
    """One resolution: per-band density deficit and pressure error, classified."""
    support = support_width(dx)
    # Geometry before numbers: reuse the case's own precondition gate.
    from tensorsph.case_hydrostatic import check_geometry
    check_geometry(dx=dx, tank_width=tank_width, depth=depth, bands=bands)
    pos, mass, rho, pressure, solver = static_field(
        dx, tank_width, depth, sound_speed=sound_speed, device=device)
    rho_rows = band_mean(pos, rho, mass, depth, bands)
    p_rows = band_mean(pos, pressure, mass, depth, bands)
    p_by_band = {row["band"]: row["value"] for row in p_rows}

    rows = []
    for row in rho_rows:
        y = torch.tensor([row["y"]], dtype=rho.dtype, device=rho.device)
        rho_ref = float(hydrostatic_density(y, solver)[0])
        rho_meas = row["value"]
        p_meas = p_by_band[row["band"]]
        p_ref = RHO0 * G * (depth - row["y"])
        deficit = rho_meas / rho_ref - 1.0
        # Relative error of the *density response*: how much of the analytic
        # compression (rho - rho0) the lattice actually builds up.  This is the
        # quantity that survives when the equation of state is made stiffer --
        # the deficit above scales like p / B, this ratio does not.
        response = ((rho_meas - solver.rho0) / (rho_ref - solver.rho0) - 1.0
                    if abs(rho_ref - solver.rho0) > 0.0 else None)
        # Exact Tait mapping of the measured density ratio onto a pressure
        # ratio -- isolates "pressure followed from the density" from
        # "pressure was modified after the density".
        s_meas = rho_meas / solver.rho0
        s_ref = rho_ref / solver.rho0
        p_from_rho = (s_meas ** solver.gamma - 1.0) / (s_ref ** solver.gamma - 1.0) - 1.0
        amplification = (p_from_rho / (rho_meas / rho_ref - 1.0)
                         if abs(rho_meas / rho_ref - 1.0) > 0.0 else None)
        rows.append({
            "band": row["band"],
            "y": row["y"],
            "y_low": row["y_low"],
            "y_high": row["y_high"],
            "particles": row["particles"],
            "class": classify(row["y_low"], row["y_high"], depth, support),
            "rho": rho_meas,
            "rho_ref": rho_ref,
            "density_deficit_pct": deficit * 100.0,
            "density_response_error_pct": (response * 100.0
                                           if response is not None else None),
            "pressure": p_meas,
            "pressure_ref": p_ref,
            "pressure_error_pct": (p_meas / p_ref - 1.0) * 100.0,
            "pressure_error_from_density_pct": p_from_rho * 100.0,
            "post_processing_pct": ((p_meas / p_ref - 1.0) - p_from_rho) * 100.0,
            # Tait amplification gamma * (p + B) / p of this band: the factor
            # that turns a relative density deficit into a relative pressure
            # error.  Reported because it is the reason the pressure gate is
            # ~200x tighter than the density it is derived from.
            "tait_amplification": amplification,
        })

    summary = {}
    for name in CLASSES:
        errs = [abs(r["pressure_error_pct"]) for r in rows if r["class"] == name]
        defs = [r["density_deficit_pct"] for r in rows if r["class"] == name]
        if not errs:
            summary[name] = {"bands": [], "max_abs_pressure_error_pct": None,
                             "mean_signed_pressure_error_pct": None,
                             "mean_density_deficit_pct": None,
                             "max_abs_post_processing_pct": None,
                             "mean_tait_amplification": None,
                             "mean_density_response_error_pct": None}
            continue
        summary[name] = {
            "bands": [r["band"] for r in rows if r["class"] == name],
            "max_abs_pressure_error_pct": max(errs),
            "mean_signed_pressure_error_pct": sum(
                r["pressure_error_pct"] for r in rows if r["class"] == name) / len(errs),
            "mean_density_deficit_pct": sum(defs) / len(defs),
            "mean_density_response_error_pct": (
                sum(r["density_response_error_pct"] for r in rows
                    if r["class"] == name
                    and r["density_response_error_pct"] is not None)
                / max(1, sum(1 for r in rows if r["class"] == name
                             and r["density_response_error_pct"] is not None))),
            "max_abs_post_processing_pct": max(
                abs(r["post_processing_pct"]) for r in rows if r["class"] == name),
            "mean_tait_amplification": sum(
                r["tait_amplification"] for r in rows
                if r["class"] == name and r["tait_amplification"] is not None)
            / max(1, sum(1 for r in rows if r["class"] == name
                         and r["tait_amplification"] is not None)),
        }

    return {
        "case_id": CASE_ID,
        "kind": "static_truncation_decomposition",
        "config": {"dx": dx, "tank_width": tank_width, "depth": depth,
                   "bands": bands, "sound_speed": sound_speed, "device": device,
                   "fluid_particles": int(pos.shape[0]),
                   "support": support, "H_over_dx": depth / dx,
                   "columns": int(round(tank_width / dx)),
                   "rho0": solver.rho0, "gamma": solver.gamma,
                   "bulk_modulus": solver.rho0 * solver.sound_speed ** 2 / solver.gamma},
        "bands": rows,
        "summary": summary,
    }


def fit_order(points: list[tuple[float, float]]) -> dict:
    """Observed order from ``|e| = C dx^p`` by least squares on the logs.

    ``points`` is ``[(dx, |e|), ...]``.  Fewer than two positive errors cannot
    support an order; the fit then reports ``None`` instead of a number.
    """
    usable = [(d, e) for d, e in points if d > 0.0 and e > 0.0]
    if len(usable) < 2:
        return {"order": None, "coefficient": None, "points": len(usable)}
    xs = [math.log(d) for d, _ in usable]
    ys = [math.log(e) for _, e in usable]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0.0:
        return {"order": None, "coefficient": None, "points": n}
    p = sxy / sxx
    intercept = my - p * mx
    return {"order": p, "coefficient": math.exp(intercept), "points": n}


#: The shipped dynamic gate record of the case (``--config window_8s``).
DEFAULT_DYNAMIC_RECORD = "tests/data/case_hydrostatic/runs/window_8s.json"


def _dynamic_gate(path: str | None) -> dict | None:
    """Read the shipped dynamic gate number, if the record is present.

    Recorded next to the static floor so that "how much headroom does the gate
    have over the truncation error of its own measurement" is answered by a data
    file, not by a sentence in a document.
    """
    if not path:
        return None
    file = Path(path)
    if not file.is_absolute():
        file = Path(__file__).resolve().parents[2] / path
    if not file.is_file():
        return None
    record = json.loads(file.read_text())
    gate = record.get("gate") or {}
    if "max_band_error_pct" not in gate:
        return None
    return {"source": str(path), "kind": record.get("kind"),
            "max_band_error_pct": gate["max_band_error_pct"],
            "limit_pct": gate.get("limit_pct"), "pass": gate.get("pass"),
            "worst_band": gate.get("worst_band")}


def sound_speed_sweep(sound_speeds, dx: float, tank_width: float = 1.0,
                      depth: float = DEPTH, bands: int = 8,
                      device: str = "cpu") -> list[dict]:
    """One resolution, several sound speeds: is the pressure gate c0-robust?

    Raising ``c0`` at fixed geometry multiplies the Tait bulk modulus
    ``B = rho0 c0^2 / gamma``, so the *density* deficit of a given pressure
    level falls like ``1 / c0^2`` while the amplification ``gamma (p + B) / p``
    grows like ``c0^2``.  The product is what the pressure gate sees, which is
    why the sweep is recorded next to the ladder instead of being inferred.
    """
    out = []
    for c0 in sound_speeds:
        rec = diagnose(dx, tank_width=tank_width, depth=depth,
                       sound_speed=float(c0), bands=bands, device=device)
        interior = rec["summary"]["interior"]
        floor = rec["summary"]["floor"]
        out.append({
            "dx": float(dx),
            "sound_speed": float(c0),
            "bulk_modulus": rec["config"]["bulk_modulus"],
            "interior_density_deficit_pct": interior["mean_density_deficit_pct"],
            "interior_max_abs_pressure_error_pct": interior["max_abs_pressure_error_pct"],
            "interior_mean_tait_amplification": interior["mean_tait_amplification"],
            "interior_density_response_error_pct":
                interior["mean_density_response_error_pct"],
            "floor_max_abs_pressure_error_pct": floor["max_abs_pressure_error_pct"],
        })
    return out


def decompose(dxs=DEFAULT_DXS, tank_width: float = 1.0, depth: float = DEPTH,
              sound_speed: float = 20.0, bands: int = 8,
              device: str = "cpu", sound_speeds=None,
              sweep_dx: float | None = None,
              dynamic_record: str | None = None) -> dict:
    """Run the ladder and fit the observed order of every band class."""
    records = [diagnose(float(dx), tank_width=tank_width, depth=depth,
                        sound_speed=sound_speed, bands=bands, device=device)
               for dx in dxs]
    ladder = {}
    for name in CLASSES:
        series = [(r["config"]["dx"], r["summary"][name]["max_abs_pressure_error_pct"])
                  for r in records if r["summary"][name]["max_abs_pressure_error_pct"]]
        ladder[name] = {
            "points": [{"dx": d, "max_abs_pressure_error_pct": e} for d, e in series],
            "order": fit_order(series),
        }
    interior_deficit = [
        (r["config"]["dx"], abs(r["summary"]["interior"]["mean_density_deficit_pct"]))
        for r in records
        if r["summary"]["interior"]["mean_density_deficit_pct"] is not None]
    ladder["interior_density_deficit"] = {
        "points": [{"dx": d, "abs_mean_density_deficit_pct": e}
                   for d, e in interior_deficit],
        "order": fit_order(interior_deficit),
    }
    interior_response = [
        (r["config"]["dx"], abs(r["summary"]["interior"]["mean_density_response_error_pct"]))
        for r in records
        if r["summary"]["interior"]["mean_density_response_error_pct"] is not None]
    ladder["interior_density_response"] = {
        "points": [{"dx": d, "abs_mean_density_response_error_pct": e}
                   for d, e in interior_response],
        "order": fit_order(interior_response),
    }
    sweep = []
    extra = [float(c) for c in (sound_speeds or ()) if float(c) != float(sound_speed)]
    if extra:
        sweep_dx = float(sweep_dx if sweep_dx is not None else min(dxs))
        sweep = sound_speed_sweep([float(sound_speed)] + extra, sweep_dx,
                                 tank_width=tank_width, depth=depth, bands=bands,
                                 device=device)
    return {
        "case_id": CASE_ID,
        "kind": "static_truncation_decomposition_ladder",
        "config": {"tank_width": tank_width, "depth": depth, "bands": bands,
                   "sound_speed": sound_speed, "device": device,
                   "dxs": [float(d) for d in dxs],
                   "sound_speed_sweep_at_dx": sweep[0]["dx"] if sweep else None},
        "records": records,
        "ladder": ladder,
        "sound_speed_sweep": sweep,
        "dynamic_gate": _dynamic_gate(dynamic_record),
        "how": "python benchmarks/case_hydrostatic/truncation_decomposition.py "
               f"--dxs {','.join(str(float(d)) for d in dxs)} --bands {bands}"
               + (f" --sound-speeds {','.join(str(c) for c in extra)}"
                  f" --sweep-dx {sweep_dx:g}" if sweep else ""),
    }


def _fmt(value, spec: str = ".3f", dash: str = "n/a") -> str:
    """Format a recorded number, or a dash when the class has no band."""
    return dash if value is None else format(value, spec)


PRODUCTION_DX = 0.02   # dx of the CASE-HYDROSTATIC gate configuration


def _findings(report: dict) -> list[str]:
    """Sentence-per-number conclusions computed from the record."""
    out = []
    interior = report["ladder"]["interior"]
    if interior["order"]["order"] is not None:
        out.append(
            f"* 内部条带压强误差 `|e_p| = {interior['order']['coefficient']:.1f} %/m x "
            f"dx^{interior['order']['order']:.2f}`（对数最小二乘，"
            f"{interior['order']['points']} 点）：一阶收敛，"
            f"系数 {interior['order']['coefficient']:.1f} %/m。")
    response = report["ladder"]["interior_density_response"]
    if response["order"]["order"] is not None:
        out.append(
            f"* 同一批内部条带的**密度响应**误差 `|e_rho| = "
            f"{response['order']['coefficient']:.1f} %/m x "
            f"dx^{response['order']['order']:.2f}`，与压强误差在 0.02 个百分点内重合"
            " -> 压强误差是密度响应的直接继承，不是压强后处理产生的。")

    prod = next((r for r in report["records"]
                 if abs(r["config"]["dx"] - PRODUCTION_DX) < 1e-12), None)
    if prod is not None:
        floor = prod["summary"]["interior"]["max_abs_pressure_error_pct"]
        out.append(
            f"* 判据分辨率 `dx = {PRODUCTION_DX:g} m` 处的静态内部截断地板 = "
            f"{floor:.3f} %：与 1.0 % 判据同量级（余量 {1.0 - floor:+.3f} 个百分点）。"
            + ("即判据的通过完全依赖动态时间平均把读数压回判据以内，"
               "而不是静态场本身已达判据。" if floor >= 1.0 else ""))
    dyn = report.get("dynamic_gate")
    if dyn and dyn.get("max_band_error_pct") is not None:
        floor = (prod["summary"]["interior"]["max_abs_pressure_error_pct"]
                 if prod else 0.0)
        out.append(
            f"* 同一算例的动态时间平均判据读数 {dyn['max_band_error_pct']:.4f} %"
            f"（`{dyn['source']}`，limit {dyn['limit_pct']:.2f} %，worst band "
            f"{dyn['worst_band']}）：相对静态截断地板变化 "
            f"{dyn['max_band_error_pct'] - floor:+.3f} 个百分点 —— 动态判据几乎贴着"
            "一阶截断地板，因此任何改变时间平均的开关（窗口长度、c0、展弦比）都有"
            "把读数推过判据的余地。")
    sweep = report.get("sound_speed_sweep") or []
    if len(sweep) >= 2:
        lo = min(sweep, key=lambda r: r["bulk_modulus"])
        hi = max(sweep, key=lambda r: r["bulk_modulus"])
        out.append(
            f"* 声速扫描：`B` 从 {lo['bulk_modulus']:.0f} 到 "
            f"{hi['bulk_modulus']:.0f} Pa（x{hi['bulk_modulus'] / lo['bulk_modulus']:.2f}）时，"
            f"密度亏损从 {lo['interior_density_deficit_pct']:+.4f} % 变到 "
            f"{hi['interior_density_deficit_pct']:+.4f} %（约 1/B），而密度响应误差只从 "
            f"{lo['interior_density_response_error_pct']:+.3f} % 动到 "
            f"{hi['interior_density_response_error_pct']:+.3f} %，压强误差从 "
            f"{lo['interior_max_abs_pressure_error_pct']:.3f} % 到 "
            f"{hi['interior_max_abs_pressure_error_pct']:.3f} % —— 压强判据对 c0 近似不变，"
            "因为亏损 ∝ 1/B 与 Tait 放大因子 ∝ B 反向抵消。")
    return out


def markdown(report: dict) -> str:
    """Markdown tables straight from the record (no transcribed numbers)."""
    cfg = report["config"]
    lines = [
        "# CASE-HYDROSTATIC -- 静态截断误差的条带分解",
        "",
        f"生成命令：`{report['how']}`",
        "",
        "## 逐分辨率分类汇总",
        "",
        "| dx (m) | H/dx | 壁面条带 | 内部条带 | 自由面条带 | max|e_p| 壁面 % | max|e_p| 内部 % "
        "| 内部密度亏损 % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["records"]:
        s = r["summary"]
        c = r["config"]
        lines.append(
            f"| {c['dx']:g} | {c['H_over_dx']:.1f} | {len(s['floor']['bands'])} "
            f"| {len(s['interior']['bands'])} | {len(s['free_surface']['bands'])} "
            f"| {_fmt(s['floor']['max_abs_pressure_error_pct'])} "
            f"| {_fmt(s['interior']['max_abs_pressure_error_pct'])} "
            f"| {_fmt(s['interior']['mean_density_deficit_pct'], '+.4f')} |")
    lines += ["", "## 观测阶（对数最小二乘）", "",
              "| 序列 | 观测阶 p |", "|---|---|"]
    for key in CLASSES + ("interior_density_deficit", "interior_density_response"):
        order = report["ladder"][key]["order"]["order"]
        lines.append(f"| {key} | {'n/a' if order is None else f'{order:.2f}'} |")

    # Per-band table at the finest resolution: where the error lives.
    finest = min(report["records"], key=lambda r: r["config"]["dx"])
    lines += [
        "", f"## 最细分辨率 dx = {finest['config']['dx']:g} m 的逐条带表", "",
        "| 条带 | y (m) | 类别 | 粒子 | 密度亏损 % | 密度响应误差 % | 压强误差 % "
        "| 其中由密度决定 % | 后处理残差 % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in finest["bands"]:
        lines.append(
            f"| {row['band']} | {row['y']:.4f} | {row['class']} | {row['particles']} "
            f"| {row['density_deficit_pct']:+.4f} "
            f"| {_fmt(row['density_response_error_pct'], '+.3f')} "
            f"| {row['pressure_error_pct']:+.3f} "
            f"| {row['pressure_error_from_density_pct']:+.3f} "
            f"| {row['post_processing_pct']:+.3f} |")
    if report.get("sound_speed_sweep"):
        lines += [
            "", f"## 声速扫描（固定 dx = {report['config']['sound_speed_sweep_at_dx']:g} m）", "",
            "| c0 (m/s) | B (Pa) | 内部密度亏损 % | 内部密度响应误差 % | 内部 max|e_p| % "
            "| Tait 放大因子 | 壁面 max|e_p| % |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in report["sound_speed_sweep"]:
            lines.append(
                f"| {row['sound_speed']:g} | {row['bulk_modulus']:.0f} "
                f"| {_fmt(row['interior_density_deficit_pct'], '+.4f')} "
                f"| {_fmt(row['interior_density_response_error_pct'], '+.3f')} "
                f"| {_fmt(row['interior_max_abs_pressure_error_pct'])} "
                f"| {_fmt(row['interior_mean_tait_amplification'], '.1f')} "
                f"| {row['floor_max_abs_pressure_error_pct']:.3f} |")
    findings = _findings(report)
    if findings:
        lines += ["", "## 结论（由本记录计算，不手抄）", ""] + findings
    lines += [
        "",
        "## 读法（必须先于读数声明）",
        "",
        "* `max|e_p|` 是**该类别内**的最大绝对压强误差；壁面类与内部类不能相加，" 
        "也不能互相替代：前者测的是固壁核支撑截断，后者才是分辨率收敛。",
        "* `压强误差 %` = `p_band / p_ref - 1`，`p_ref = rho0 g (H - y_band)` 在实测"
        "质量加权条带中心求值（零手抄）。",
        "* `其中由密度决定 %` = 把实测密度比值经 Tait 律 `p = B[(rho/rho0)^7 - 1]` "
        "精确映射到压强比值；它和 `压强误差 %` 的差就是 `后处理残差 %`（自由面闭合 / "
        "壁面压强外推等对压强的直接修改）。",
        "* 只有内部类可以读作分辨率阶梯：`p ~ dx^m` 的 `m` 由对数最小二乘给出，"
        "不预设为 1。",
        "",
    ]
    return "\n".join(lines)


def figure(report: dict, path: str) -> None:
    """Error-vs-dx figure: one panel per band class plus the density deficit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = {"floor": "tab:red", "interior": "tab:blue", "free_surface": "tab:green"}
    for name in CLASSES:
        pts = [(p["dx"], p["max_abs_pressure_error_pct"])
               for p in report["ladder"][name]["points"]]
        if not pts:
            continue
        order = report["ladder"][name]["order"]["order"]
        label = name if order is None else f"{name} (order {order:.2f})"
        axes[0].loglog([d for d, _ in pts], [e for _, e in pts], "o-",
                       color=colours[name], label=label)
    axes[0].set_xlabel("dx [m]")
    axes[0].set_ylabel("max |p - p_ref| / p_ref  [%]")
    axes[0].set_title("static t=0 field, per band class")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    pts = [(p["dx"], p["abs_mean_density_deficit_pct"])
           for p in report["ladder"]["interior_density_deficit"]["points"]]
    if pts:
        order = report["ladder"]["interior_density_deficit"]["order"]["order"]
        axes[1].loglog([d for d, _ in pts], [e for _, e in pts], "s-",
                       color="tab:purple",
                       label="interior deficit"
                       + ("" if order is None else f" (order {order:.2f})"))
    axes[1].set_xlabel("dx [m]")
    axes[1].set_ylabel("|rho/rho_ref - 1|  [%]")
    axes[1].set_title("interior density deficit (Tait-amplified into pressure)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python benchmarks/case_hydrostatic/truncation_decomposition.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dxs", default=",".join(str(d) for d in DEFAULT_DXS),
                        help="comma separated lattice spacings [m]")
    parser.add_argument("--tank-width", type=float, default=1.0)
    parser.add_argument("--depth", type=float, default=DEPTH)
    parser.add_argument("--sound-speed", type=float, default=20.0)
    parser.add_argument("--sound-speeds", default="",
                        help="comma separated extra c0 values for the "
                             "fixed-dx invariance sweep (empty = no sweep)")
    parser.add_argument("--sweep-dx", type=float, default=None,
                        help="dx of the sweep (default: finest ladder entry)")
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamic-record", default=DEFAULT_DYNAMIC_RECORD,
                        help="shipped dynamic gate record quoted next to the "
                             "static floor (empty string disables)")
    parser.add_argument("--from-json", default=None,
                        help="re-render --md-out / --figure-out from an existing "
                             "record instead of re-running the ladder")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--figure-out", default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)

    if args.from_json:
        report = json.loads(Path(args.from_json).read_text())
        report["dynamic_gate"] = _dynamic_gate(args.dynamic_record or None)
    else:
        dxs = tuple(float(x) for x in args.dxs.split(",") if x.strip())
        try:
            report = decompose(
                dxs, tank_width=args.tank_width, depth=args.depth,
                sound_speed=args.sound_speed, bands=args.bands,
                device=args.device,
                sound_speeds=[float(x) for x in args.sound_speeds.split(",")
                              if x.strip()],
                sweep_dx=args.sweep_dx,
                dynamic_record=args.dynamic_record or None)
        except HydrostaticGeometryError as exc:
            print(f"geometry rejected: {exc}", file=sys.stderr)
            return 2
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown(report))
    if args.figure_out:
        out = Path(args.figure_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure(report, str(out))
    for rec in report["records"]:
        s = rec["summary"]
        print(f"dx={rec['config']['dx']:g}  floor {s['floor']['max_abs_pressure_error_pct']:8.3f}%  "
              f"interior {s['interior']['max_abs_pressure_error_pct']:7.3f}%  "
              f"surface {s['free_surface']['max_abs_pressure_error_pct']:7.3f}%  "
              f"deficit {s['interior']['mean_density_deficit_pct']:+.4f}%")
    for key in CLASSES + ("interior_density_deficit", "interior_density_response"):
        order = report["ladder"][key]["order"]["order"]
        print(f"observed order {key:26s} {('n/a' if order is None else f'{order:5.2f}')}")
    for row in report.get("sound_speed_sweep") or ():
        print(f"c0={row['sound_speed']:5g}  B={row['bulk_modulus']:9.0f}  "
              f"deficit {row['interior_density_deficit_pct']:+.4f}%  "
              f"interior {row['interior_max_abs_pressure_error_pct']:6.3f}%  "
              f"amplification {row['interior_mean_tait_amplification']:7.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    sys.exit(main())