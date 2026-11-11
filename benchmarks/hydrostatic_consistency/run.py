#!/usr/bin/env python3
"""Static (zero-simulation) consistency audit of the 3-D hydrostatic column.

Why this exists
---------------
``benchmarks/threed_hydrostatic`` runs a wall-bounded 3-D column and compares
``p(z)`` with ``p = rho0 g (H - z)``.  Its shipped artifact fails its own
criteria (mean profile error ``-11.9 %``, worst core layer ``-43.2 %``) and the
verification ledger grades those readings FAIL / MARGINAL, but *nothing in the
repository says which part of the machinery produces the error*.  A 3692-step
float64 run costs ~30 min per case, so probing the hypothesis by re-running is
expensive and confounded: the run mixes the pressure operator, the density
closure, the free-surface clamp and a possibly-unsettled window.

This benchmark isolates the one ingredient that can be measured *exactly* and
*for free*: the static imbalance of the discrete pressure operator when it is
handed the exact answer.  Impose ``p_i = rho0 g (H - z_i)`` and
``rho_i = tait_density(p_i)`` on the production particle set and evaluate the
repository's own operator plus gravity.  If the scheme were hydrostatically
consistent, the result would be zero.  What is left is a lower bound on every
hydrostatic reading the code can produce, and its *sign* decides whether the
measured error can be attributed to the operator at all.

Quantities
----------
``a_z / g`` per fluid layer, for a grid of
``h / dx`` in {1.3, 1.5} x wall thickness in {1, 2, 3} layers x
free surface in {closed lid, open} x wall pressure in {analytic, Adami}.

Three other cheap probes ride along, because they use the same lattices:

* ``kernel_sum``: the per-layer summation density ``rho/rho0`` of the *initial*
  configuration (what the run actually starts from), with and without the lid;
* ``readability_floor``: the depth below the free surface at which the
  free-surface kernel-sum deficit stops dominating the hydrostatic compression
  ``g d / c0^2`` -- i.e. how many surface layers no weakly-compressible column
  can resolve at a given ``c0``;
* ``measured_vs_static``: the shipped artifact's per-layer error next to the
  equilibrium correction implied by the static imbalance.

Usage::

    PYTHONPATH=src python benchmarks/hydrostatic_consistency/run.py --figure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.threed.kernel3 import calibrated_particle_mass  # noqa: E402
from tensorsph.threed.neighbors3 import box_pairs  # noqa: E402
from tensorsph.threed.operators3 import (  # noqa: E402
    pressure_acceleration,
    summation_density,
    tait_density,
    tait_pressure,
    wall_pressure_adami,
)
from tensorsph.threed.solver import cubic_lattice, wall_shell  # noqa: E402

OUT = REPO / "benchmarks" / "hydrostatic_consistency"
SHIPPED = REPO / "benchmarks" / "threed_hydrostatic" / "report.json"

DX = 0.01
NX = NY = 8
NZ = 16
RHO0 = 1000.0
GRAVITY = 9.81
GAMMA = 7.0
SOUND_SPEED = 20.0
WALL_LAYERS_DEFAULT = 2

#: the grid the audit sweeps; the *first* entry is the shipped configuration
H_FACTORS = (1.3, 1.5)
WALL_LAYER_COUNTS = (1, 2, 3)
LIDS = (True, False)
WALL_FIELDS = ("analytic", "adami")


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #
def column_lattice(*, h_factor: float = 1.3, wall_layers: int = WALL_LAYERS_DEFAULT,
                   lid: bool = True, nz: int = NZ, dx: float = DX,
                   dtype=torch.float64):
    """Fluid block + static wall shell exactly as :func:`run_hydrostatic` builds it.

    ``lid=False`` drops every wall particle at or above the free surface
    ``z = H``; the shipped default keeps them, which is why the *closed* column
    has wall particles 1 and 2 cells above its topmost fluid row.
    """
    h = h_factor * dx
    walls = wall_layers * dx
    height = nz * dx
    top = walls + height
    fluid = cubic_lattice((NX, NY, nz), dx, origin=(walls, walls, walls), dtype=dtype)
    wall = wall_shell((NX, NY, nz), wall_layers, dx, dtype=dtype)
    if not lid:
        wall = wall[wall[:, 2] <= top + 1e-12]
    pos = torch.cat((fluid, wall))
    mass = torch.full((pos.shape[0],), calibrated_particle_mass(dx, h, RHO0),
                      dtype=dtype)
    return {
        "pos": pos, "fluid": fluid, "n_fluid": int(fluid.shape[0]),
        "mass": mass, "h": h, "walls": walls, "height": height, "top": top,
        "wall_particles": int(wall.shape[0]),
        "lid_particles": int((wall[:, 2] > top + 1e-12).sum()) if lid else 0,
        "pairs": box_pairs(pos, 2.0 * h),
    }


def layer_index(z: torch.Tensor, walls: float, dx: float = DX) -> torch.Tensor:
    return torch.floor((z - walls) / dx + 1e-9).to(torch.int64).clamp(0, NZ - 1)


def layer_means(values: torch.Tensor, lay: torch.Tensor, nz: int = NZ) -> list[float]:
    out = []
    for k in range(nz):
        sel = lay == k
        out.append(float(values[sel].mean()) if bool(sel.any()) else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# Probe 1: static imbalance of the pressure operator
# --------------------------------------------------------------------------- #
def analytic_fields(cfg: dict, *, delta: float = 0.0, c0: float = SOUND_SPEED):
    """``p = rho0 g (H - z)`` (clamped at 0) and its Tait-consistent density.

    ``delta`` rescales the pressure so that the residual can be differentiated
    with respect to a uniform pressure change: the slope ``d a_z / d delta``
    turns the residual into an *equilibrium correction* ``delta*`` (the uniform
    rescaling that would put each layer back in vertical balance).
    """
    pos = cfg["pos"]
    p = (RHO0 * GRAVITY * (cfg["top"] - pos[:, 2])).clamp_min(0.0) * (1.0 + delta)
    rho = tait_density(p, RHO0, c0, GAMMA)
    return p, rho


def static_residual(cfg: dict, *, wall_field: str = "analytic", delta: float = 0.0,
                    c0: float = SOUND_SPEED) -> torch.Tensor:
    """``a_z / g`` on the fluid rows for the imposed analytic field."""
    pos, mass, h, pairs = cfg["pos"], cfg["mass"], cfg["h"], cfg["pairs"]
    n_fluid = cfg["n_fluid"]
    p, rho = analytic_fields(cfg, delta=delta, c0=c0)
    if wall_field == "adami":
        # The Adami extrapolation is fed the *analytic* fluid field, i.e. the
        # boundary condition the scheme would apply if the fluid side already
        # carried the exact answer.  Feeding it the initial summation density
        # instead would put ``p = 0`` on every wall particle (rho = rho0 on the
        # calibrated lattice) and measure the start-up transient, not the
        # operator.
        g = torch.tensor([0.0, 0.0, -GRAVITY], dtype=pos.dtype)
        p_wall = wall_pressure_adami(pos, mass, rho, p, h, pairs, n_fluid, g)
        p = torch.cat((p[:n_fluid], p_wall))
        rho = torch.cat((rho[:n_fluid],
                         tait_density(p_wall, RHO0, c0, GAMMA)))
    elif wall_field != "analytic":
        raise ValueError(f"unknown wall_field {wall_field!r}")
    acc = pressure_acceleration(pos, mass, rho, p, h, pairs)
    acc[:, 2] -= GRAVITY
    return acc[:n_fluid, 2] / GRAVITY


def imbalance_profile(cfg: dict, *, wall_field: str = "analytic",
                      slope_step: float = 0.05) -> dict:
    """Per-layer residual, its pressure sensitivity, and the implied correction."""
    lay = layer_index(cfg["fluid"][:, 2], cfg["walls"])
    a0 = static_residual(cfg, wall_field=wall_field)
    a1 = static_residual(cfg, wall_field=wall_field, delta=slope_step)
    slope = (a1 - a0) / slope_step
    mean = layer_means(a0, lay)
    peak = []
    for k in range(NZ):
        sel = lay == k
        peak.append(float(a0[sel].abs().max()) if bool(sel.any()) else float("nan"))
    slope_mean = layer_means(slope, lay)
    correction = []
    for a, s in zip(mean, slope_mean):
        correction.append(-100.0 * a / s if abs(s) > 1e-12 else float("nan"))
    return {
        "h_factor": round(cfg["h"] / DX, 3),
        "wall_layers": int(round(cfg["walls"] / DX)),
        "lid": bool(cfg.get("lid_flag", bool(cfg["lid_particles"]))),
        "wall_particles": cfg["wall_particles"],
        "wall_field": wall_field,
        "az_over_g_mean": mean,
        "az_over_g_peak": peak,
        "d_az_d_delta_mean": slope_mean,
        "implied_correction_pct": correction,
    }


# --------------------------------------------------------------------------- #
# Probe 2: what the run actually starts from (kernel-sum deficit)
# --------------------------------------------------------------------------- #
def kernel_sum_profile(cfg: dict, *, c0: float = SOUND_SPEED) -> dict:
    """Per-layer ``rho/rho0`` of the *initial* lattice and the read-off pressure.

    The run does not start from the analytic density: it starts from the
    summation density of a uniform lattice with calibrated mass, so every layer
    where the kernel support is truncated starts *below* ``rho0`` and the Tait
    law clamps its pressure to zero.
    """
    pos, mass, h, pairs = cfg["pos"], cfg["mass"], cfg["h"], cfg["pairs"]
    n_fluid = cfg["n_fluid"]
    rho = summation_density(pos, mass, h, pairs)[:n_fluid]
    lay = layer_index(cfg["fluid"][:, 2], cfg["walls"])
    ratio = layer_means(rho / RHO0, lay)
    chi_int = max(m for m in ratio if math.isfinite(m))
    deficit = [100.0 * (1.0 - m / chi_int) if math.isfinite(m) else float("nan")
               for m in ratio]
    p0 = layer_means(tait_pressure(rho, RHO0, c0, GAMMA), lay)
    return {
        "h_factor": round(cfg["h"] / DX, 3),
        "wall_layers": int(round(cfg["walls"] / DX)),
        "lid": bool(cfg["lid_particles"]),
        "lid_particles": cfg["lid_particles"],
        "rho_over_rho0": ratio,
        "kernel_sum_deficit_pct": deficit,
        "initial_pressure_pa": p0,
    }


def readability_floor(*, h_factor: float = 1.3, c0_values=(5.0, 10.0, 20.0, 40.0)) -> dict:
    """Can a weakly-compressible column resolve the pressure of its surface layer?

    Two density perturbations compete in the topmost fluid row of an *open*
    column: the hydrostatic compression ``g dx / c0^2`` (the signal that the Tait
    law has to read) and the free-surface kernel-sum deficit ``1 - rho/rho0``
    (an offset the law cannot distinguish from tension, so it clamps).  The
    deficit is a *lattice* property: it does not shrink with ``c0``, while the
    signal does as ``1 / c0^2``.  So the surface layer's pressure is unreadable
    for every sound speed at which ``deficit > signal``, and the equivalent
    clamp pressure ``B[(1 - deficit)^gamma - 1]`` says how many pascals of
    signal are thrown away.
    """
    cfg = column_lattice(h_factor=h_factor, lid=False)
    table = kernel_sum_profile(cfg)
    deficit_top = list(reversed(table["kernel_sum_deficit_pct"]))[0]
    stiffness = RHO0 * SOUND_SPEED ** 2 / GAMMA
    rows = []
    for c0 in c0_values:
        signal = 100.0 * GRAVITY * DX / c0 ** 2
        clamp = (RHO0 * c0 ** 2 / GAMMA) * ((1.0 - deficit_top / 100.0) ** GAMMA - 1.0)
        rows.append({
            "sound_speed": c0,
            "deficit_pct": deficit_top,
            "signal_at_first_layer_pct": signal,
            "deficit_over_signal": deficit_top / signal if signal else float("inf"),
            "clamp_pressure_pa": clamp,
            "analytic_pressure_at_first_layer_pa": RHO0 * GRAVITY * DX,
            "clamp_over_signal": abs(clamp) / (RHO0 * GRAVITY * DX),
        })
    return {"h_factor": h_factor, "deficit_top_layer_pct": deficit_top,
            "stiffness_pa": stiffness, "rows": rows}


# Probe 3: the shipped artifact next to the static prediction
# --------------------------------------------------------------------------- #
def measured_vs_static(profile: dict, shipped_path: Path = SHIPPED) -> dict:
    shipped = json.loads(shipped_path.read_text())
    best = shipped.get("best_case")
    run = next((r for r in shipped["runs"] if r["case"] == best), shipped["runs"][0])
    rows = run["profile"]
    measured = [row["p_rel_error"] for row in rows]
    depth = [row["depth_below_surface"] for row in rows]
    # the shipped profile is binned half a cell off the particle layers, so the
    # prediction is interpolated onto the reported depths
    lay = profile["az_over_g_mean"]
    pred = []
    for d in depth:
        k = (profile_height_layers(profile) - d / DX) - 0.5
        k0 = int(max(0, min(len(lay) - 1, math.floor(k))))
        k1 = min(len(lay) - 1, k0 + 1)
        w = k - k0
        pred.append(lay[k0] * (1 - w) + lay[k1] * w)
    corr = [c for c in profile["implied_correction_pct"]]
    corr_at = []
    for d in depth:
        k = int(max(0, min(len(corr) - 1,
                           round(profile_height_layers(profile) - d / DX - 0.5))))
        corr_at.append(corr[k])
    sign_agree = sum(1 for m, c in zip(corr_at, measured)
                     if (m >= 0) == (c >= 0))
    rms = math.sqrt(sum((c - m) ** 2 for c, m in zip(corr_at, measured))
                    / len(measured))
    return {
        "shipped_artifact": str(shipped_path.relative_to(REPO)),
        "case": run["case"],
        "shipped_verdict": shipped.get("verdict"),
        "shipped_failing_criteria": [k for k, v in shipped["criteria"].items()
                                     if v is False],
        "depth_below_surface_m": depth,
        "measured_rel_error_pct": [100.0 * m for m in measured],
        "static_correction_pct": corr_at,
        "static_residual_az_over_g": pred,
        "measured_mean_pct": 100.0 * sum(measured) / len(measured),
        "static_mean_pct": sum(corr_at) / len(corr_at),
        "sign_agreement_layers": sign_agree,
        "layers": len(measured),
        "rms_difference_pct": rms,
        "settling_flags": {k: shipped["criteria"].get(k) for k in
                           ("column_settled_window_drift_below_2pct",
                            "centre_of_mass_drift_below_1pct")},
    }


def profile_height_layers(profile: dict) -> float:
    return NZ


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_report() -> dict:
    grid = []
    for h_factor in H_FACTORS:
        for layers in WALL_LAYER_COUNTS:
            for lid in LIDS:
                cfg = column_lattice(h_factor=h_factor, wall_layers=layers, lid=lid)
                cfg["lid_flag"] = lid
                cfg["lid_particles"] = int(
                    (cfg["pos"][cfg["n_fluid"]:, 2] > cfg["top"] + 1e-12).sum())
                for wall_field in WALL_FIELDS:
                    grid.append(imbalance_profile(cfg, wall_field=wall_field))

    shipped_cfg = column_lattice(h_factor=1.3, wall_layers=WALL_LAYERS_DEFAULT,
                                 lid=True)
    shipped_cfg["lid_flag"] = True
    shipped_cfg["lid_particles"] = int(
        (shipped_cfg["pos"][shipped_cfg["n_fluid"]:, 2] > shipped_cfg["top"]
         + 1e-12).sum())
    shipped_profile = imbalance_profile(shipped_cfg, wall_field="adami")
    shipped_profile["lid"] = True

    kernel_sum = {
        "closed": kernel_sum_profile(
            column_lattice(h_factor=1.3, wall_layers=2, lid=True)),
        "open": kernel_sum_profile(
            column_lattice(h_factor=1.3, wall_layers=2, lid=False)),
    }
    floor = readability_floor()
    measured = measured_vs_static(shipped_profile)

    # ---- criteria (machine-decidable, no judgement) ----
    def find(hf, layers, lid, field):
        for row in grid:
            if (abs(row["h_factor"] - hf) < 1e-9 and row["wall_layers"] == layers
                    and row["lid"] == lid and row["wall_field"] == field):
                return row
        raise KeyError((hf, layers, lid, field))

    base = find(1.3, 2, True, "adami")
    thin = find(1.3, 1, True, "adami")
    open_top = find(1.3, 2, False, "adami")
    criteria = {
        # the shipped wall thickness is not optional
        "one_wall_layer_at_least_20x_worse": abs(thin["az_over_g_mean"][0])
        >= 20.0 * abs(base["az_over_g_mean"][0]),
        # the lid is a spurious boundary: removing it must cut the top-layer
        # residual by at least 5x
        "open_top_cuts_surface_residual_5x": abs(open_top["az_over_g_mean"][-1])
        <= 0.2 * abs(base["az_over_g_mean"][-1]),
        # the static imbalance is a floor, not the explanation
        "static_imbalance_below_measured_mean": abs(base["az_over_g_mean"][0])
        < abs(measured["measured_mean_pct"]) / 100.0,
        "static_correction_has_opposite_sign_to_measurement":
        (base["implied_correction_pct"][0] > 0)
        and (measured["measured_mean_pct"] < 0),
        # the pressure gradient is applied without a reproducing correction, so
        # even a complete-support interior particle carries a lattice bias; the
        # number is pinned (0.5 % <= |a_z|/g <= 2 %) so a silent change in the
        # operator cannot pass
        "interior_lattice_bias_in_documented_band":
        5e-3 <= abs(base["az_over_g_mean"][NZ // 2]) <= 2e-2,
        # the free-surface kernel-sum deficit is a lattice property: it exceeds
        # the hydrostatic signal of the surface layer by >=10x at c0 = 20 m/s
        # (so the Tait clamp, not the operator, owns that reading) and the ratio
        # grows as c0^2
        "surface_clamp_dominates_signal_10x":
        next(r for r in floor["rows"] if r["sound_speed"] == SOUND_SPEED)[
            "clamp_over_signal"] >= 10.0,
        "surface_clamp_ratio_grows_with_c0":
        floor["rows"][-1]["clamp_over_signal"] > floor["rows"][0]["clamp_over_signal"],
    }
    verdict = "fail" if not all(criteria.values()) else "pass"
    return {
        "benchmark": "hydrostatic_consistency",
        "question": ("with the exact hydrostatic field imposed on the production "
                     "lattice, how far from balance is the discrete pressure "
                     "operator, and can that explain the shipped profile error?"),
        "params": {"dx": DX, "nx": NX, "ny": NY, "nz": NZ, "rho0": RHO0,
                   "gravity": GRAVITY, "gamma": GAMMA, "sound_speed": SOUND_SPEED},
        "grid": grid,
        "shipped_configuration": shipped_profile,
        "kernel_sum": kernel_sum,
        "readability_floor": floor,
        "measured_vs_static": measured,
        "criteria": criteria,
        "verdict": verdict,
        "generated_by": "benchmarks/hydrostatic_consistency/run.py",
        "simulation_cost": "zero (lattice sums + one operator evaluation per cell)",
    }


def make_figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = [(k + 0.5) * DX for k in range(NZ)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    ax = axes[0]
    base = report["shipped_configuration"]
    ax.plot([100.0 * v for v in base["az_over_g_mean"]], z, "o-",
            label="closed (shipped)", color="#c0392b")
    open_top = next(r for r in report["grid"]
                    if r["h_factor"] == 1.3 and r["wall_layers"] == 2
                    and not r["lid"] and r["wall_field"] == "adami")
    ax.plot([100.0 * v for v in open_top["az_over_g_mean"]], z, "s--",
            label="open free surface", color="#2471a3")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"static residual $a_z/g$ [%]")
    ax.set_ylabel("height above wall base [m]")
    ax.set_title("exact field imposed: operator imbalance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    meas = report["measured_vs_static"]
    ax.plot(meas["measured_rel_error_pct"], meas["depth_below_surface_m"],
            "o-", label="shipped run (measured)", color="#8e44ad")
    ax.plot(meas["static_correction_pct"], meas["depth_below_surface_m"],
            "s--", label="static prediction", color="#16a085")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("relative pressure error [%]")
    ax.set_ylabel("depth below free surface [m]")
    ax.set_title("operator floor vs shipped reading")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for row in report["readability_floor"]["rows"]:
        ax.plot(row["unresolvable_layers"], row["sound_speed"], "o",
                label=f"c0 = {row['sound_speed']:g} m/s")
    ax.set_xlabel("unresolvable surface layers (deficit > signal)")
    ax.set_ylabel("sound speed $c_0$ [m/s]")
    ax.set_title("weak-compressibility readability floor")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "report.json"))
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args(argv)
    report = build_report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"verdict: {report['verdict']}")
    for k, v in report["criteria"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(json.dumps(report["measured_vs_static"], indent=1)[:1200])
    if args.figure:
        fig = OUT / "figures" / "hydrostatic_consistency.png"
        make_figure(report, fig)
        print("figure:", fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())