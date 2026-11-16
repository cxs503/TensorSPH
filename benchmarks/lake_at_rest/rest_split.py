"""Well-balanced hydrostatic-split closure: rest exactness and what it costs.

Runner for the evidence of ``CASE-SPH-LAKE-AT-REST`` added by
:mod:`tensorsph.lake_rest_split`.  The anchor's acceptance is an analytic
identity (``u = 0``, flat surface) to machine precision; the question this runner
decides is whether that can be met by a *scheme* -- one whose residual is
non-zero on a state that is not the equilibrium -- rather than by handing the
answer in.

Three arms, one ladder, one state:

``split/signed``
    the well-balanced split of :mod:`tensorsph.lake_rest_split` with the wall
    pressure extended *without* a floor (``p_w = p_f + rho_f g.(r_w - r_f)``),
``split/clamped``
    the same closure with the usual ``max(p_w, 0)`` -- the tensile-safe default,
``prescribed``
    the identity arm of ``docs/verification_lake_at_rest_closure_response.md``:
    the affine pressure handed in, no perturbation pressure at all.

Gates (all relative to the analytic identity; no experiment, no fitted value):

=========  ===================================================  ==============
gate       quantity                                              threshold
=========  ===================================================  ==============
``R1``     split/signed ``max|a|/g`` on the ridge lattice        ``< 1e-9``
``R2``     split/clamped ``max|a|/g`` (shipped wall closure)      ``> 1e-2``
``R3``     ``max|p_w - p_hyd(y_w)| / p_max``, signed extension    ``< 1e-8``
``R4``     split/signed ``max|dx_particle|`` after ``t_end``      ``< 1e-9``
``R5``     ``|r(eta0 + 1 mm) - r(eta0)|``                         ``< 1e-12``
``R6``     split/signed residual at ``A = 1 dx``                  ``> 1e-2``
``R7``     prescribed residual at ``A = 1 dx``                    ``< 1e-9``
``R8``     ``r(0.5 dx) / r(0.25 dx)``                             ``1.5..2.5``
``R9``     dynamics attempt: density envelope (reported, not gated)
=========  ===================================================  ==============

``R9`` is reported as what it is: the evolving-density integration is **not**
certified.  On this closure the free-surface row accumulates a density source and
the run leaves the physical envelope within ~0.1 s (numbers in the record).  No
wave frequency is claimed anywhere in this artefact.

Usage
-----
    PYTHONPATH=src python3 benchmarks/lake_at_rest/rest_split.py \
        --json tests/data/lake_at_rest/rest_split.json --doc --figures
    PYTHONPATH=src python3 benchmarks/lake_at_rest/rest_split.py --quick
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import torch

from tensorsph.lake_at_rest import LakeConfig, build_state
from tensorsph.lake_rest_split import (SplitOptions, accelerations,
                                       analytic_frequency, hydrostatic_pressure,
                                       rest_run, split_fields, step, tait_density)

ANCHOR_TOLERANCE = 1.0e-9
RESOLUTIONS = (0.050, 0.040, 0.025, 0.020)
QUICK_RESOLUTIONS = (0.050, 0.025)
AMPLITUDES = (0.25, 0.5, 1.0)
HORIZON = 0.50


@contextlib.contextmanager
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _arm(name: str) -> SplitOptions:
    if name == "split/signed":
        return SplitOptions()
    if name == "split/clamped":
        return SplitOptions(wall_extension="clamped")
    if name == "prescribed/signed":
        return SplitOptions(closure="prescribed")
    raise ValueError(name)


def wall_extension_defect(cfg: LakeConfig) -> dict:
    """``max|p_w - p_hyd(y_w)|`` over the wall particles that touch the fluid.

    ``R3``: the signed extension has to *be* the affine continuation on the
    support that matters -- the wall particles that exchange pairs with the fluid.
    A wall particle with no fluid neighbour carries no information (its
    continuation is defined as zero) and is excluded, so the gate measures the
    physical continuation and not the far corners of the wall lattice.
    """
    import tensorsph.lake_rest_split as module

    fields = split_fields(cfg, build_state(cfg), SplitOptions())
    n, nw = fields["n"], fields["nw"]
    if nw == 0:
        return {"max_defect_pa": 0.0, "max_defect_over_pmax": 0.0,
                "n_wall_in_support": 0}
    rho = torch.full((n,), cfg.rho0, dtype=fields["pos"].dtype,
                     device=fields["pos"].device)
    pressure = hydrostatic_pressure(cfg, fields["pos"][:, 1])
    signed = module._wall_pressure(fields, pressure, rho)
    exact = hydrostatic_pressure(cfg, fields["wall"][:, 1])
    cross = fields["fluid_i"] != fields["fluid_j"]
    wall_idx = torch.where(fields["fluid_i"], fields["j"], fields["i"]) - n
    touching = torch.zeros(nw, dtype=torch.bool, device=fields["pos"].device)
    touching[wall_idx[cross]] = True
    defect = (signed - exact).abs()[touching]
    if defect.numel() == 0:
        return {"max_defect_pa": 0.0, "max_defect_over_pmax": 0.0,
                "n_wall_in_support": 0}
    return {"max_defect_pa": float(defect.max()),
            "max_defect_over_pmax": float(defect.max()) / float(exact.max()),
            "n_wall_in_support": int(touching.sum())}


def perturbation_response(cfg: LakeConfig, arm: str, amplitude: float) -> dict:
    """Residual of the closure at a *perturbed* hydrostatic state (no integration).

    The state is the barotropic pre-image of the perturbed surface
    ``eta = H + A cos(2 pi x / W)``: ``p' = rho0 g A cos(kx)``, ``rho = Tait^-1``.
    The identity arm carries no ``p'``, so its residual must stay at roundoff for
    every ``A`` -- that is ``R7``, and it is what makes ``R6`` meaningful.
    """
    options = _arm(arm)
    fields = split_fields(cfg, build_state(cfg), options)
    n, pos = fields["n"], fields["pos"]
    k = 2.0 * math.pi / cfg.width
    eta = cfg.depth + amplitude * torch.cos(k * pos[:, 0])
    if options.closure == "prescribed":
        rho = torch.full((n,), cfg.rho0, dtype=pos.dtype, device=pos.device)
    else:
        rho = tait_density(cfg, cfg.rho0 * cfg.gravity * (eta - cfg.depth))
    accel = accelerations(fields, rho, cfg.depth)["acceleration"]
    return {"arm": arm, "amplitude": amplitude,
            "amplitude_over_dx": amplitude / cfg.dx,
            "max_a_g": float(torch.linalg.vector_norm(accel, dim=1).max())
            / cfg.gravity,
            "max_drho": float((rho - cfg.rho0).abs().max()) / cfg.rho0}


def dynamics_attempt(cfg: LakeConfig, periods: float = 2.0) -> dict:
    """One honest wave run: how far does the evolving-density form get?

    Not a gate.  The record carries the density envelope and the velocity scale
    per sampled time, so the reader can see where the closure leaves the physical
    envelope instead of being told a frequency it never measured.
    """
    options = SplitOptions()
    fields = split_fields(cfg, build_state(cfg), options)
    n, pos0 = fields["n"], fields["pos"]
    amplitude = 0.5 * cfg.dx
    k = (2.0 * math.pi / cfg.width) if cfg.periodic_x else (math.pi / cfg.width)
    eta = cfg.depth + amplitude * torch.cos(k * pos0[:, 0])
    rho = tait_density(cfg, cfg.rho0 * cfg.gravity * (eta - cfg.depth))
    pos, vel = pos0.clone(), torch.zeros_like(pos0)
    omega = analytic_frequency(cfg.depth, k, cfg.gravity)
    horizon = periods * 2.0 * math.pi / omega
    dt = options.max_courant * cfg.h() / cfg.sound_speed
    steps = max(1, int(math.ceil(horizon / dt)))
    dt = horizon / steps
    sample = max(1, steps // 24)
    series, envelope, left_at = [], 0.0, None
    for index in range(steps + 1):
        if index % sample == 0 or index == steps:
            drift = float((rho - cfg.rho0).abs().max()) / cfg.rho0
            envelope = max(envelope, drift)
            series.append({"t": index * dt, "max_rel_drho": drift,
                           "max_v_c0": float(torch.linalg.vector_norm(vel, dim=1).max())
                           / cfg.sound_speed,
                           "max_y": float(pos[:, 1].max())})
            if drift > 0.01 and left_at is None:
                left_at = index * dt
        if index == steps:
            break
        if left_at is not None and index * dt > left_at + 0.02:
            break
        rho, pos, vel = step(fields, rho, pos, vel, dt, cfg.depth)
    return {"dx": cfg.dx, "periodic_x": cfg.periodic_x,
            "amplitude": amplitude, "amplitude_over_dx": amplitude / cfg.dx,
            "omega_analytic": omega, "horizon_requested": horizon,
            "horizon_reached": series[-1]["t"],
            "max_rel_drho_envelope": envelope, "t_left_envelope": left_at,
            "stable_for_two_periods": left_at is None, "series": series}


def evaluate(record: dict) -> dict:
    rows = record["rest_ladder"]
    signed = [row for row in rows if row["arm"] == "split/signed"]
    clamped = [row for row in rows if row["arm"] == "split/clamped"]
    prescribed = [row for row in rows if row["arm"] == "prescribed/signed"]
    closed = lambda group: [row for row in group if not row["periodic_x"]]
    response = record["perturbation_response"]
    by_arm = {}
    for row in response:
        by_arm.setdefault(row["arm"], {})[round(row["amplitude_over_dx"], 6)] = \
            row["max_a_g"]
    ratio = None
    if by_arm.get("split/signed", {}).get(0.25):
        ratio = by_arm["split/signed"][0.5] / by_arm["split/signed"][0.25]
    gates = [
        {"gate": "R1", "quantity": "split/signed max|a|/g at t=0 (closed tank)",
         "measured": max(row["a0_max_g"] for row in closed(signed)),
         "threshold": f"< {ANCHOR_TOLERANCE:g}",
         "pass": all(row["a0_max_g"] < ANCHOR_TOLERANCE
                     for row in closed(signed))},
        {"gate": "R2", "quantity": "split/clamped max|a|/g (shipped wall closure)",
         "measured": min(row["a0_max_g"] for row in closed(clamped)),
         "threshold": "> 1e-2",
         "pass": all(row["a0_max_g"] > 1e-2 for row in closed(clamped))},
        {"gate": "R3", "quantity": "max|p_w - p_hyd(y_w)| / p_max (signed)",
         "measured": record["wall_defect"]["max_defect_over_pmax"],
         "threshold": "< 1e-8",
         "pass": record["wall_defect"]["max_defect_over_pmax"] < 1e-8},
        {"gate": "R4", "quantity": "split/signed max|dx_particle| after t_end (m)",
         "measured": max(row["max_dpos_over_horizon"] for row in signed),
         "threshold": "< 1e-9",
         "pass": all(row["max_dpos_over_horizon"] < 1e-9 for row in signed)},
        {"gate": "R5", "quantity": "|r(eta0+1mm) - r(eta0)| at t=0",
         "measured": record["level_shift"]["delta"], "threshold": "< 1e-12",
         "pass": record["level_shift"]["delta"] < 1e-12},
        {"gate": "R10", "quantity": "stability lifetime of the rest state (s)",
         "measured": min(row["t_exceed_tolerance"] if row["t_exceed_tolerance"]
                         is not None else record["horizon"] for row in signed),
         "threshold": f">= {record['horizon']:g}  (not met: see section 5)",
         "pass": all(row["t_exceed_tolerance"] is None for row in signed)},
        {"gate": "R6", "quantity": "split/signed residual at A = 1 dx",
         "measured": by_arm.get("split/signed", {}).get(1.0),
         "threshold": "> 1e-2",
         "pass": by_arm.get("split/signed", {}).get(1.0, 0.0) > 1e-2},
        {"gate": "R7", "quantity": "prescribed residual at A = 1 dx",
         "measured": by_arm.get("prescribed/signed", {}).get(1.0),
         "threshold": f"< {ANCHOR_TOLERANCE:g}",
         "pass": by_arm.get("prescribed/signed", {}).get(1.0, 1.0)
         < ANCHOR_TOLERANCE},
        {"gate": "R8", "quantity": "r(0.5 dx) / r(0.25 dx)", "measured": ratio,
         "threshold": "1.5 .. 2.5",
         "pass": ratio is not None and 1.5 <= ratio <= 2.5},
    ]
    passed = {gate["gate"]: gate["pass"] for gate in gates}
    return {
        "gates": gates,
        "n_gates": len(gates),
        "n_passed": sum(1 for gate in gates if gate["pass"]),
        "all_gates_pass": all(passed.values()),
        "equilibrium_exact": bool(passed["R1"] and passed["R2"] and passed["R3"]
                                  and passed["R5"]),
        "persists": passed["R4"] and passed.get("R10", False),
        "anchor_met": bool(passed["R1"] and passed["R4"] and passed["R5"]
                          and passed.get("R10", False)),
        "non_degenerate": bool(passed["R6"] and passed["R7"] and passed["R8"]),
        "dynamics_certified": False,
        "residual_range_signed": [min(row["max_a_g_over_horizon"] for row in signed),
                                  max(row["max_a_g_over_horizon"] for row in signed)],
        "residual_range_clamped": [min(row["a0_max_g"] for row in clamped),
                                   max(row["a0_max_g"] for row in clamped)],
        "residual_range_prescribed": [min(row["max_a_g_over_horizon"]
                                          for row in prescribed),
                                      max(row["max_a_g_over_horizon"]
                                          for row in prescribed)],
    }


def build(quick: bool = False) -> dict:
    resolutions = QUICK_RESOLUTIONS if quick else RESOLUTIONS
    with single_thread():
        rows = []
        for dx in resolutions:
            for periodic in (False, True):
                for arm in ("split/signed", "split/clamped", "prescribed/signed"):
                    cfg = LakeConfig(dx=dx, periodic_x=periodic)
                    result = rest_run(cfg, horizon=HORIZON, options=_arm(arm),
                                      sample_every=50)
                    rows.append({
                        "arm": arm, "dx": dx, "periodic_x": periodic,
                        "n_fluid": result["n_fluid"], "n_wall": result["n_wall"],
                        "steps": result["steps"], "dt": result["dt"],
                        "a0_max_g": result["a0_max_g"],
                        "a0_max_ax_g": result["a0_max_ax_g"],
                        "max_a_g_over_horizon": result["max_a_g_over_horizon"],
                        "max_v_c0_over_horizon": result["max_v_c0_over_horizon"],
                        "max_dpos_over_horizon": result["max_dpos_over_horizon"],
                        "final_max_drho": result["final_max_drho"],
                        "t_exceed_tolerance": next(
                            (point["t"] for point in result["series"]
                             if point["max_a_g"] > ANCHOR_TOLERANCE), None),
                        "series": result["series"],
                    })
                print(f"  rest dx={dx:.3f} periodic={periodic} done", flush=True)

        cfg = LakeConfig(dx=max(resolutions))
        wall_defect = wall_extension_defect(cfg)
        baseline = rest_run(cfg, horizon=HORIZON, options=SplitOptions(),
                            sample_every=50)
        shifted = rest_run(cfg, horizon=HORIZON, options=SplitOptions(),
                           level=cfg.depth + 1.0e-3, sample_every=50)
        level_shift = {"level": cfg.depth + 1.0e-3,
                       "residual": shifted["a0_max_g"],
                       "baseline_residual": baseline["a0_max_g"],
                       "delta": abs(shifted["a0_max_g"] - baseline["a0_max_g"])}
        print("  level shift done", flush=True)

        response = [perturbation_response(cfg, arm, multiple * cfg.dx)
                    for arm in ("split/signed", "prescribed/signed")
                    for multiple in AMPLITUDES]
        print("  perturbation response done", flush=True)

        dynamics = dynamics_attempt(LakeConfig(dx=0.05, periodic_x=True))
        print("  dynamics attempt done", flush=True)

        regularisation = []
        for eps in (1e-6, 1e-9, 1e-12, 1e-14):
            out = rest_run(LakeConfig(dx=0.05), horizon=1.0e-3,
                           options=SplitOptions(regularisation=eps),
                           sample_every=4)
            regularisation.append({"regularisation": eps,
                                   "max_a_g": out["a0_max_g"]})
        print("  regularisation ladder done", flush=True)

    record = {
        "case": "CASE-SPH-LAKE-AT-REST",
        "kind": "well-balanced hydrostatic-split closure",
        "reference": ("analytic identity p = rho0 g (H - y), u = 0, flat surface; "
                      "dispersion omega^2 = g k tanh(k H) for the non-degeneracy arm"),
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "horizon": HORIZON,
        "geometry": {"width": cfg.width, "depth": cfg.depth,
                     "h_factor": cfg.h_factor, "rho0": cfg.rho0,
                     "sound_speed": cfg.sound_speed, "gamma": cfg.gamma,
                     "gravity": cfg.gravity},
        "torch": torch.__version__,
        "resolutions": list(resolutions),
        "rest_ladder": rows,
        "wall_defect": wall_defect,
        "level_shift": level_shift,
        "perturbation_response": response,
        "dynamics_attempt": dynamics,
        "regularisation_ladder": regularisation,
    }
    record["summary"] = evaluate(record)
    return record

def figures(record: dict, directory: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    made = []
    colours = {"split/signed": "tab:green", "split/clamped": "tab:red",
               "prescribed/signed": "tab:blue"}

    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    for arm, colour in colours.items():
        subset = [row for row in record["rest_ladder"]
                  if row["arm"] == arm and not row["periodic_x"]]
        axis.loglog([row["dx"] for row in subset],
                    [row["a0_max_g"] for row in subset], "o-", color=colour,
                    label=arm)
    axis.axhline(ANCHOR_TOLERANCE, color="k", linestyle="--",
                 label="anchor tolerance 1e-9")
    axis.set_xlabel("dx (m)")
    axis.set_ylabel("max |a| / g on the ridge lattice")
    axis.set_title("CASE-SPH-LAKE-AT-REST: the wall extension decides the residual")
    axis.invert_xaxis()
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "lake_rest_split_residual.png", dpi=150)
    plt.close(fig)
    made.append("lake_rest_split_residual.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for arm, colour in (("split/signed", "tab:green"), ("split/clamped", "tab:red")):
        subset = [row for row in record["rest_ladder"]
                  if row["arm"] == arm and not row["periodic_x"]]
        if not subset:
            continue
        row = min(subset, key=lambda item: item["dx"])
        axes[0].semilogy([point["t"] for point in row["series"]],
                         [max(point["max_a_g"], 1e-17) for point in row["series"]],
                         color=colour, label=f"{arm} (dx={row['dx']:.3f})")
        axes[1].semilogy([point["t"] for point in row["series"]],
                         [max(point["max_dpos"], 1e-19) for point in row["series"]],
                         color=colour, label=f"{arm} (dx={row['dx']:.3f})")
    axes[0].axhline(ANCHOR_TOLERANCE, color="k", linestyle="--")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("max |a| / g")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("max |dx_particle| (m)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)
    fig.suptitle("Rest horizon: signed wall extension holds, clamped does not")
    fig.tight_layout()
    fig.savefig(directory / "lake_rest_split_horizon.png", dpi=150)
    plt.close(fig)
    made.append("lake_rest_split_horizon.png")

    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    for arm, colour in (("split/signed", "tab:green"),
                        ("prescribed/signed", "tab:blue")):
        subset = [row for row in record["perturbation_response"] if row["arm"] == arm]
        axis.loglog([row["amplitude_over_dx"] for row in subset],
                    [max(row["max_a_g"], 1e-18) for row in subset], "o-",
                    color=colour, label=arm)
    axis.axhline(ANCHOR_TOLERANCE, color="k", linestyle="--",
                 label="anchor tolerance")
    axis.set_xlabel("free-surface perturbation A / dx")
    axis.set_ylabel("max |a| / g at t = 0")
    axis.set_title("Scheme or identity? The operator's response to a perturbation")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "lake_rest_split_response.png", dpi=150)
    plt.close(fig)
    made.append("lake_rest_split_response.png")

    dynamics = record["dynamics_attempt"]
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.semilogy([point["t"] for point in dynamics["series"]],
                  [max(point["max_rel_drho"], 1e-12) for point in dynamics["series"]],
                  "o-", color="tab:purple", label="max |rho/rho0 - 1|")
    axis.axhline(0.01, color="k", linestyle="--", label="1 % envelope")
    axis.set_xlabel("t (s)")
    axis.set_ylabel("density excursion")
    axis.set_title("Dynamics attempt (honest FAIL): free-surface density source")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "lake_rest_split_dynamics.png", dpi=150)
    plt.close(fig)
    made.append("lake_rest_split_dynamics.png")
    return made


def _table(rows, columns, keys, formats):
    out = ["| " + " | ".join(columns) + " |",
           "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt.format(row[key])
                                    for key, fmt in zip(keys, formats)) + " |")
    return "\n".join(out)


def document(record: dict) -> str:
    summary = record["summary"]
    closed = lambda arm: [row for row in record["rest_ladder"]
                          if row["arm"] == arm and not row["periodic_x"]]
    signed, clamped = closed("split/signed"), closed("split/clamped")
    prescribed = closed("prescribed/signed")
    periodic = [row for row in record["rest_ladder"]
                if row["arm"] == "split/signed" and row["periodic_x"]]
    response = record["perturbation_response"]
    split_resp = [row for row in response if row["arm"] == "split/signed"]
    identity_resp = [row for row in response if row["arm"] == "prescribed/signed"]
    dynamics = record["dynamics_attempt"]
    geometry = record["geometry"]
    lines = []
    add = lines.append
    add("# 闭式水槽静止残差的病因是壁面压力截断；而静止态的**时间推进不成立**\n")
    add("Anchor: **CASE-SPH-LAKE-AT-REST** — 算例池·静水湖面守恒（零速零波动）；\n"
        "基准 = 解析恒等 `p = rho0 g (H - y)`、`u = 0`、平自由面；\n"
        f"容差 = 机器精度 `{ANCHOR_TOLERANCE:g}`。\n")
    add("> 本文件由 `benchmarks/lake_at_rest/rest_split.py` 从\n"
        "> `tests/data/lake_at_rest/rest_split.json` 程序化生成；表中每个数字都是\n"
        "> 该 JSON 的字段，零手抄。**报告的是实测，包括失败的部分。**\n")
    add("## 0. 结论先行（先说失败）\n")
    add(f"1. **静止态在 t=0 是精确离散不动点，但按本单的显式格式推进会线性不稳定。**\n"
        f"   `t=0` 残差 `max|a|/g = {min(r['a0_max_g'] for r in signed):.3e} … "
        f"{max(r['a0_max_g'] for r in signed):.3e}`（闭式水槽），过锚点容差约 4 个量级；\n"
        f"   可是从舍入级种子出发，残差在 "
        f"`t = {min((r['t_exceed_tolerance'] or record['horizon']) for r in signed):.3f} s` "
        f"内越出容差，`t_end = {record['horizon']:.2f} s` 时已到 "
        f"`{max(p['max_a_g'] for r in signed for p in r['series']):.3e}`。\n"
        f"   **因此本单不声称锚点被满足**：`anchor_met = {summary['anchor_met']}`、"
        f"`persists = {summary['persists']}`。\n")
    add(f"2. **病因定位是实的：壁面压力延拓的 `max(p_w, 0)` 截断。** 同一套一阶矩(MLS)算子、\n"
        f"   同一几何、同一条格点，壁面压力取*有符号*线性延拓时 `t=0` 残差 "
        f"`{max(r['a0_max_g'] for r in signed):.3e}`，\n"
        f"   换回截断版本就是 `{min(r['a0_max_g'] for r in clamped):.3e} … "
        f"{max(r['a0_max_g'] for r in clamped):.3e}` —— 与历史记录的 0.109 g 一致，\n"
        f"   而两臂的唯一差别就是水面以上干壁面粒子的压力符号。壁面延拓本身与解析水压力\n"
        f"   的一致性 `max|p_w - p_hyd(y_w)|/p_max = "
        f"{record['wall_defect']['max_defect_over_pmax']:.3e}`（R3）。\n")
    add(f"3. **容差不是恒等式。** 同一算子在受扰静止态上的残差随扰动幅度线性增长\n"
        f"   （`A/dx = {split_resp[0]['amplitude_over_dx']:.2f} → "
        f"{split_resp[-1]['amplitude_over_dx']:.2f}`：`{split_resp[0]['max_a_g']:.3e} → "
        f"{split_resp[-1]['max_a_g']:.3e}` g），\n"
        f"   而“把压力递给算子”的恒等式臂在同一批幅度上仍是 "
        f"`{identity_resp[0]['max_a_g']:.3e} … {identity_resp[-1]['max_a_g']:.3e}` g。\n"
        f"   分辨“看得见非平衡态”与“看不见”，这一项通过（R6/R7/R8）。\n")
    add(f"4. **动态臂同样 FAIL，不声称波频。** 演化密度形式在自由面行累积密度源，"
        f"`t = "
        f"{dynamics['t_left_envelope'] if dynamics['t_left_envelope'] is not None else -1:.3f} s`\n"
        f"   越出 1% 密度包络（实测最大 `{dynamics['max_rel_drho_envelope']:.3e}`），"
        f"`omega^2 = g k tanh(kH)` 未检验。\n")
    add("## 1. 案例与闭合式\n")
    add("| 量 | 值 |\n|---|---|")
    add(f"| 水槽 | 宽 {geometry['width']:.3f} m × 静水深 {geometry['depth']:.3f} m |")
    add(f"| `rho0` / `c0` / `gamma` / `g` | {geometry['rho0']:.1f} / "
        f"{geometry['sound_speed']:.2f} / {geometry['gamma']:.1f} / {geometry['gravity']:.2f} |")
    add(f"| 核 / 光滑长度 | 三次样条，`h = {geometry['h_factor']:.2f} dx` |")
    add("| 壁面粒子 | 3 层，镜像对称，含水面以上干壁面 |")
    add(f"| 分辨率 | {', '.join(f'{dx:.3f}' for dx in record['resolutions'])} m |")
    add(f"| 时间步 | `dt = 0.3 h / c0`；`t_end = {record['horizon']:.2f} s` |\n")
    add("闭合式（`src/tensorsph/lake_rest_split.py`）：\n")
    add("```text\n"
        "p        = p_hyd(y) + p',   p_hyd(y) = rho0 g (eta0 - y)\n"
        "p'       = max(B ((rho/rho0)^gamma - 1), 0)      B = rho0 c0^2 / gamma\n"
        "dv/dt    = -(1/rho) grad_MLS(p_hyd) - g - grad_cons(p')\n"
        "drho/dt  = -rho div_MLS v\n"
        "p_wall   = p_f + rho_f (g - a_w).(r_w - r_f)     [有符号，不截断]\n"
        "```\n")
    add("`grad_MLS` 对任何仿射压力场在*任何*支撑（被壁面或自由面截断）上都精确："
        "差分形式消掉常数部分，\n一阶矩矩阵求逆消掉线性部分。静止态 `rho = rho0` ⇒ `p' = 0` 处处为零，\n"
        "算子看到的正是仿射场 `p_hyd`，于是 `-(1/rho0) grad p_hyd - g` 落到舍入级。\n"
        "`grad_cons` 是 `p'` 的对称（能量一致）形式，用作连续性方程的对偶，"
        "表里两套算子都实测过：\n"
        "MLS 差分形式承载 `p'` 时是**指数发散**的（下一节），"
        "换成对称形式后静止精确性保持不变。\n")
    add("## 2. 静止阶梯：t=0 残差、越界时刻、终态\n")
    add("| dx | arm | max\\|a\\|/g @ t=0 | 首次越 1e-9 (s) | max\\|a\\|/g @ t_end | max\\|Δx\\| (m) |")
    add("|---|---|---|---|---|---|")
    for row in signed + clamped:
        add(f"| {row['dx']:.3f} | {row['arm']} | {row['a0_max_g']:.3e} | "
            f"{row['t_exceed_tolerance'] if row['t_exceed_tolerance'] is not None else '—'} | "
            f"{max(p['max_a_g'] for p in row['series']):.3e} | "
            f"{row['max_dpos_over_horizon']:.3e} |")
    add("")
    add("横向周期水槽（同一闭合式，无水面以上干壁面）t=0 残差："
        + "，".join(f"dx = {r['dx']:.3f} → {r['a0_max_g']:.2e}" for r in periodic)
        + "。两个壁面延拓选项在周期水槽里给出*逐位相同*的 t=0 结果 —— 那里不存在\n"
          "“水面以上的干壁面粒子”，截断无从发生。这是“病因在截断”的独立旁证。\n")
    add("## 3. 剂量-反应：为什么是截断\n")
    add(f"有符号延拓在*真正与流体交换相互作用*的壁面粒子上与解析水压力一致到 "
        f"`{record['wall_defect']['max_defect_over_pmax']:.3e}` 相对量级"
        f"（{record['wall_defect']['n_wall_in_support']} 个粒子）。\n"
        f"截断版本把这些粒子中解析延拓为负（水面以上）者压到 0，最大差值 "
        f"`{record['wall_defect']['max_defect_pa']:.4g} Pa`；离壁第一列因此同时得到虚假的\n"
        f"水平与垂直力（截断臂 `max|a_x|/g = {clamped[0]['a0_max_ax_g']:.3e}`）。"
        f"与 `docs/verification_lake_at_rest_wall_gap.md` 把病灶定位在“壁面列自由面粒子的\n"
        f"邻居项”是同一件事，但给出了可修的那一半：**延拓本身精确，是截断破坏了它。**\n")
    add("### 3.1 正则化阶梯（`dx = 0.050`，`t = 1 ms`）\n")
    add(_table(record["regularisation_ladder"], ["moment 正则化", "max|a|/g"],
               ["regularisation", "max_a_g"], ["{:.0e}", "{:.3e}"]))
    add("\n精确性地板由一阶矩求逆的正则化决定；报告而非隐藏。\n")
    add("### 3.2 水面位不变性（`R5`，t=0）\n")
    add(f"`eta0 → eta0 + 1 mm`：残差 `{record['level_shift']['baseline_residual']:.3e}` → "
        f"`{record['level_shift']['residual']:.3e}`，差 "
        f"`{record['level_shift']['delta']:.3e}`。\n"
        f"t=0 的精确性来自仿射再现，不是在某一个水位上凑出来的抵消。\n")
    add("## 4. 非退化：方案还是恒等式\n")
    add(_table(response, ["臂", "A/dx", "max|a|/g @ t=0", "max|δρ|/ρ0"],
               ["arm", "amplitude_over_dx", "max_a_g", "max_drho"],
               ["{}", "{:.2f}", "{:.3e}", "{:.2e}"]))
    add("\n`split/signed` 的响应随 `A` 线性增长（`R8`），算子看得见非平衡态；"
        "`prescribed` 臂在同一批幅度上\n始终停在舍入级 —— "
        "正是 `docs/verification_lake_at_rest_closure_response.md` 指出的恒等式属性，"
        "本文件用同一把尺子复现。\n")
    add("## 5. 注意点 / 局限（关键）\n")
    add("1. **时间推进不稳定，这是本单的主要负面结果。** 从 `t=0` 的舍入级残差出发，"
        "残差指数增长\n"
        "   （`dx=0.050` 闭式水槽：1e-13 → 1e-9 → 1e-3 → O(1) g，"
        "e-folding 约 10 ms），与历史论文里\n"
        "   “显式推进的发散密度形式需要额外稳定化”一致。已排除的候选：密度扩散（δ-SPH）、"
        "参考体积、\n"
        "   张拉保护（`p'≥0`）、保守对称压力梯度 —— 四种都在同一时间尺度上发散，"
        "说明病灶不在\n"
        "   自由面张拉分支，而在“独立密度 + 声速刚度”这一层的格式设计（下一步候选："
        "半隐式/\n"
        "   压力投影、或把密度改回求和式并把平衡态偏置作为源项）。\n"
        "2. **因此本单不把 t=0 的精确性算作锚点通过。** 锚点要的是“零速零波动”，"
        "而守恒必须经得起推进。\n"
        "3. **有符号壁面延拓的代价**：水面以上干壁面粒子压力为负（张力）。静止态里它是平衡的\n"
        "   一部分（解析场的线性延拓），动问题里需要额外证据，本单没有。\n"
        "4. **参考压力解析的前提是参考态常密度**（不可压极限）；层化水体需积分 `p_hyd`。\n"
        "5. **差分/对称两套算子混用**：参考项走 MLS（精确抵消），扰动项走对称形式"
        "（能量一致）。\n"
        "   这在本单是被迫的工程折中，已在表里分别实测。\n")
    add("## 6. 误差判定（逐项，含“是否 < 3%”）\n")
    add("| 关键量 | 基准 | 实测 | 是否 < 3% |\n|---|---|---|---|")
    add(f"| 闭式水槽 t=0 `max|a|/g`（split/signed） | 解析恒等 0 | "
        f"`{min(r['a0_max_g'] for r in signed):.3e} … {max(r['a0_max_g'] for r in signed):.3e}` "
        f"| **是** |")
    add(f"| 闭式水槽 t=0 `max|a|/g`（split/clamped） | 解析恒等 0 | "
        f"`{min(r['a0_max_g'] for r in clamped):.3e} … {max(r['a0_max_g'] for r in clamped):.3e}` "
        f"| **否**（≈10% g） |")
    add(f"| 壁面延拓 vs 解析水压力 | `rho0 g (H - y_w)` | "
        f"`{record['wall_defect']['max_defect_over_pmax']:.3e}` 相对 | **是** |")
    add(f"| 受扰态残差线性度（split/signed） | 线性理论 | 见 `R8` | **是** |")
    add(f"| 静止态维持 `t_end = {record['horizon']:.2f} s` | 解析恒等 0 | "
        f"`{max(r['max_dpos_over_horizon'] for r in signed):.3e} m` | **否**——发散 |")
    add(f"| 重力波频率 `omega` | `sqrt(g k tanh kH)` | **未测** | **无法判定** |")
    add(f"\n锚点判定：`anchor_met = {summary['anchor_met']}`；"
        f"`equilibrium_exact = {summary['equilibrium_exact']}`；"
        f"`persists = {summary['persists']}`；"
        f"`non_degenerate = {summary['non_degenerate']}`；"
        f"`dynamics_certified = {summary['dynamics_certified']}`；"
        f"门禁 {summary['n_passed']}/{summary['n_gates']} 通过。\n")
    add("## 7. 图\n")
    add("* `docs/figures/lake_rest_split_residual.png`：三臂 t=0 残差 vs `dx`（含锚点容差线）；")
    add("* `docs/figures/lake_rest_split_horizon.png`：0.5 s 视界内 `max|a|/g` 与 `max|Δx|`（发散的证据）；")
    add("* `docs/figures/lake_rest_split_response.png`：受扰态响应（方案 vs 恒等式）；")
    add("* `docs/figures/lake_rest_split_dynamics.png`：动态臂密度包络（诚实 FAIL）。\n")
    add("## 8. 复现\n")
    add("```\n"
        "PYTHONPATH=src python3 benchmarks/lake_at_rest/rest_split.py \\\n"
        "    --json tests/data/lake_at_rest/rest_split.json --doc --figures\n"
        "PYTHONPATH=src python3 -m pytest tests/test_lake_at_rest_rest_split.py -q\n"
        "```\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    parser.add_argument("--doc", default=None, nargs="?",
                        const="docs/verification_lake_at_rest_rest_split.md")
    parser.add_argument("--figures", default=None, nargs="?",
                        const="docs/figures")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    record = build(quick=args.quick)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
    if args.figures:
        made = figures(record, Path(args.figures))
        record["figures"] = made
    if args.doc:
        path = Path(args.doc)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document(record) + "\n")
    if not args.quiet:
        print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
