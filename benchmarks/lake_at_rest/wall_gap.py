"""The closed-tank rest gap: why refinement cannot reach the anchor tolerance.

The laterally periodic lake meets `CASE-SPH-LAKE-AT-REST` to machine precision
(`docs/verification_lake_at_rest_periodic.md`).  The closed tank does not, and
`docs/verification_lake_at_rest.md` records that as an honest FAIL.  This runner
answers the question the FAIL leaves open:

  1. is the residual a *discretisation* error, i.e. does refinement remove it?
     (resolution ladder of the reconstructed equilibrium)
  2. can any *wall-face pressure* field remove it?
     (dry-clip / mirrored / analytic-hydrostatic wall pressure)
  3. can any shipped operator option remove it?
     (Adami boundary-consistent density, Shepard renormalisation,
     free-surface pressure closure)
  4. where does it live?  (per-column / per-row maps, term split into the
     target's own pressure times the kernel-gradient deficiency and the
     neighbour-pressure sum)

Produce the evidence + figures:

    PYTHONPATH=src python3 benchmarks/lake_at_rest/wall_gap.py \
        --json tests/data/lake_at_rest/wall_gap.json --figures docs/figures
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import torch

from tensorsph.droplet_impact import (cubic_spline_kernel, pair_kernels,
                                      tait_pressure, wall_mass)
from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    reconstruct, residual_report)

RESOLUTIONS = (0.050, 0.040, 0.025, 0.020)
CLOSURES = (
    ("baseline", {}),
    ("boundary_consistent_density", {"boundary_consistent_density": True}),
    ("density_renormalization", {"density_renormalization": True}),
    ("surface_pressure_closure", {"surface_pressure_closure": True}),
)
WALL_PRESSURE = ("adami", "dry_clip", "mirror", "hydrostatic")


# --------------------------------------------------------------------------- #
# operator re-implementation with a replaceable wall-face pressure
# --------------------------------------------------------------------------- #
def build_fields(cfg: LakeConfig, mass: torch.Tensor) -> dict:
    """Everything the horizontal sum needs, from the shipped pair list."""
    state = build_state(cfg)
    pos, wall = state["pos"], state["wall"]
    n, nw = pos.shape[0], wall.shape[0]
    h = cfg.h()
    i, j = state["pairs"][:, 0], state["pairs"][:, 1]
    all_pos = torch.cat((pos, wall), dim=0)
    all_mass = torch.cat((mass, wall_mass(state["config"], nw, pos.device)))
    delta = all_pos[i] - all_pos[j]
    distance = torch.linalg.vector_norm(delta, dim=1)
    wij, grad_w = pair_kernels(delta, h)
    zero = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    rho_all = torch.zeros(n + nw, dtype=pos.dtype, device=pos.device)
    rho_all[:n] = mass * cubic_spline_kernel(zero, h)
    rho_all = rho_all.index_add(0, i, all_mass[j] * wij)
    rho_all = rho_all.index_add(0, j, all_mass[i] * wij)
    rho = rho_all.clamp_min(0.1 * cfg.rho0)
    pressure = tait_pressure(rho, cfg)
    i_fluid = i < n
    crossing = i_fluid != (j < n)
    fluid_idx = torch.where(i_fluid, i, j)
    wall_idx = (torch.where(i_fluid, j, i) - n).clamp_(0, nw - 1)
    dy_wall = torch.where(i_fluid, -delta[:, 1], delta[:, 1])
    weight = torch.where(crossing, wij, torch.zeros_like(wij))
    source = pressure[fluid_idx] + rho[fluid_idx] * (-cfg.gravity * dy_wall)
    num = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
    num = num.index_add(0, wall_idx, source * weight)
    den = torch.zeros(nw, dtype=pos.dtype, device=pos.device)
    den = den.index_add(0, wall_idx, weight)
    p_wall = (num / den.clamp_min(1e-30)).clamp_min(0.0)
    return dict(cfg=cfg, pos=pos, wall=wall, n=n, nw=nw, i=i, j=j,
                all_pos=all_pos, all_mass=all_mass, grad_w=grad_w, rho=rho,
                pressure=pressure, p_wall=p_wall, crossing=crossing,
                i_fluid=i_fluid, fluid_idx=fluid_idx, wall_idx=wall_idx,
                weight=weight, h=h)


def wall_pressure_candidates(f: dict) -> dict:
    cfg, nw, wall, pos = f["cfg"], f["nw"], f["wall"], f["pos"]
    base = f["p_wall"]
    out = {"adami": base}
    if not nw:
        return out
    fluid_level = f["all_pos"][f["fluid_idx"], 1]
    highest = torch.full((nw,), -math.inf, dtype=pos.dtype, device=pos.device)
    highest = highest.index_reduce(0, f["wall_idx"], fluid_level, "amax",
                                   include_self=True)
    out["dry_clip"] = torch.where(highest < wall[:, 1], torch.zeros_like(base), base)

    mirror = base.clone()
    for index in range(nw):
        x_w = float(wall[index, 0])
        x_m = (2.0 * float(cfg.width) - x_w) if x_w > 0.5 * cfg.width else -x_w
        hit = (((pos[:, 0] - x_m).abs() < 0.5 * cfg.dx)
               & ((pos[:, 1] - float(wall[index, 1])).abs() < 0.5 * cfg.dx))
        if bool(hit.any()):
            mirror[index] = f["pressure"][:f["n"]][hit].mean()
    out["mirror"] = mirror.clamp_min(0.0)
    out["hydrostatic"] = (cfg.rho0 * cfg.gravity
                          * (cfg.depth - wall[:, 1]).clamp_min(0.0))
    return out


def accel_terms(f: dict, p_wall: torch.Tensor) -> dict:
    """Horizontal acceleration split into own-pressure and neighbour-pressure."""
    cfg, n, i, j, grad = f["cfg"], f["n"], f["i"], f["j"], f["grad_w"][:, 0]
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    rho_all = torch.cat((f["rho"][:n], (cfg.rho0 * (p_wall / bulk + 1.0)
                                       .pow(1.0 / cfg.gamma)).clamp_min(cfg.rho0)))
    p_all = torch.cat((f["pressure"][:n], p_wall))
    m_i, m_j = f["all_mass"][i], f["all_mass"][j]
    own_coef = p_all[i] / rho_all[i].square()
    nbr_coef = p_all[j] / rho_all[j].square()

    def scatter(target: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(n, dtype=f["pos"].dtype, device=f["pos"].device)
        return total.index_add(0, target[target < n], values[target < n])

    # A pair contributes -m_j grad (coef_i + coef_j) to i and +m_i grad (...) to j.
    own = scatter(i, -m_j * grad * own_coef) + scatter(j, m_i * grad * nbr_coef)
    nbr = scatter(i, -m_j * grad * nbr_coef) + scatter(j, m_i * grad * own_coef)
    return {"own_pressure_term": own, "neighbour_pressure_term": nbr,
            "total": own + nbr,
            "kernel_deficiency_mass": scatter(i, -m_j * grad)
            + scatter(j, m_i * grad)}


def horizontal_map(f: dict, p_wall: torch.Tensor) -> torch.Tensor:
    return accel_terms(f, p_wall)["total"]


def field_stats(f: dict, ax: torch.Tensor) -> dict:
    cfg, pos = f["cfg"], f["pos"]
    hz = ax.abs() / cfg.gravity
    band = 2.0 * f["h"]
    wall_band = (pos[:, 0] < band) | (pos[:, 0] > cfg.width - band)
    worst = int(ax.abs().argmax())
    columns = sorted(set(round(float(v), 6) for v in pos[:, 0]))
    rows = sorted(set(round(float(v), 6) for v in pos[:, 1]))
    return {
        "a_x_max_over_g": float(hz.max()),
        "a_x_wall_band": float(hz[wall_band].max()) if bool(wall_band.any()) else None,
        "a_x_interior": float(hz[~wall_band].max()) if bool((~wall_band).any()) else None,
        "worst_particle": {"x": float(pos[worst, 0]), "y": float(pos[worst, 1])},
        "per_column_max": {f"{c:.4f}": float(hz[(pos[:, 0] - c).abs() < 0.25 * cfg.dx].max())
                           for c in columns},
        "per_row_max": {f"{r:.4f}": float(hz[(pos[:, 1] - r).abs() < 0.25 * cfg.dx].max())
                        for r in rows},
        "antisymmetry": _antisymmetry(cfg, pos, ax),
    }


def _antisymmetry(cfg: LakeConfig, pos: torch.Tensor, ax: torch.Tensor) -> float:
    """max |a_x(x) + a_x(W - x)| / max|a_x| over mirror pairs (0 is perfect)."""
    scale = float(ax.abs().max())
    if scale == 0.0:
        return 0.0
    worst = 0.0
    for index in range(pos.shape[0]):
        x, y = float(pos[index, 0]), float(pos[index, 1])
        hit = ((pos[:, 0] - (float(cfg.width) - x)).abs() < 0.25 * cfg.dx) & \
              ((pos[:, 1] - y).abs() < 0.25 * cfg.dx)
        if bool(hit.any()):
            other = float(ax[hit][0])
            worst = max(worst, abs(float(ax[index]) + other))
    return worst / scale


def closure_matrix(dx: float, depth: float) -> list:
    """Reconstruct under each shipped operator option and re-measure."""
    rows = []
    for name, options in CLOSURES:
        cfg = LakeConfig(dx=dx, depth=depth, solver_max_iter=40)
        shipped = cfg.shipped()
        for key, value in options.items():
            setattr(shipped, key, value)
        patched = dataclasses.replace(cfg)
        patched.shipped = lambda shipped=shipped: shipped   # type: ignore[assignment]
        t0 = time.time()
        state = build_state(patched)
        solved = reconstruct(state, patched)
        rows.append({
            "closure": name, "options": options,
            "a_y_max_over_g": solved["a_max_over_g"],
            "a_x_max_over_g": solved["a_max_over_g_horizontal"],
            "a_x_wall_band": solved["a_max_over_g_horizontal_wall_band"],
            "converged": bool(solved["converged"]),
            "iterations": int(solved["iterations"]),
            "seconds": round(time.time() - t0, 2),
        })
        print(f"  closure {name:>28s}  a_y={rows[-1]['a_y_max_over_g']:.3e}"
              f"  a_x={rows[-1]['a_x_max_over_g']:.3e}"
              f"  it={rows[-1]['iterations']}", flush=True)
    return rows


def resolution_ladder(dx_values, depth: float) -> list:
    rows = []
    for dx in dx_values:
        cfg = LakeConfig(dx=dx, depth=depth, solver_max_iter=40)
        t0 = time.time()
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)
        prior_report = residual_report(state, cfg, prior["mass"])
        solved = reconstruct(state, cfg)
        fields = build_fields(cfg, solved["mass"])
        baseline = field_stats(fields, horizontal_map(fields, fields["p_wall"]))
        variants = {}
        for name, p_wall in wall_pressure_candidates(fields).items():
            variants[name] = field_stats(fields, horizontal_map(fields, p_wall))
        terms = accel_terms(fields, fields["p_wall"])
        worst = int(terms["total"].abs().argmax())
        row = {
            "dx": dx, "h": cfg.h(),
            "n_fluid": int(state["pos"].shape[0]),
            "n_wall": int(state["wall"].shape[0]),
            "n_pairs": int(state["pairs"].shape[0]),
            "reconstruct": {"converged": bool(solved["converged"]),
                            "iterations": int(solved["iterations"]),
                            "a_y_max_over_g": solved["a_max_over_g"]},
            "prior": {"a_x_max_over_g": prior_report["a_max_over_g_horizontal"],
                      "a_y_max_over_g": prior_report["a_max_over_g"]},
            "adami": baseline,
            "wall_pressure_variants": variants,
            "worst_split": {
                "x": float(fields["pos"][worst, 0]),
                "y": float(fields["pos"][worst, 1]),
                "own_pressure_term_over_g": float(terms["own_pressure_term"][worst] / cfg.gravity),
                "neighbour_pressure_term_over_g": float(
                    terms["neighbour_pressure_term"][worst] / cfg.gravity),
                "total_over_g": float(terms["total"][worst] / cfg.gravity),
                "kernel_deficiency_mass": float(
                    terms["kernel_deficiency_mass"].abs().max()),
                "density_ratio_at_worst": float(fields["rho"][worst] / cfg.rho0),
                "pressure_at_worst": float(fields["pressure"][worst]),
            },
            "seconds": round(time.time() - t0, 2),
        }
        rows.append(row)
        print(f"dx={dx:.3f} n={row['n_fluid']} a_y={solved['a_max_over_g']:.2e}"
              f" a_x={baseline['a_x_max_over_g']:.4e}"
              f" worst=({baseline['worst_particle']['x']:.3f},"
              f"{baseline['worst_particle']['y']:.3f})"
              f" variants=" + ",".join(f"{k}:{v['a_x_max_over_g']:.3e}"
                                       for k, v in variants.items())
              + f" ({row['seconds']}s)", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, nargs="*", default=list(RESOLUTIONS))
    ap.add_argument("--depth", type=float, default=0.30)
    ap.add_argument("--json", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--skip-closures", action="store_true")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    payload = {"case": "CASE-SPH-LAKE-AT-REST", "geometry": {
        "width": 0.40, "depth": args.depth, "rho0": 1000.0,
        "sound_speed": 20.0, "gamma": 7.0, "gravity": 9.81,
        "wall_layers": 3, "h_factor": 1.3, "periodic_x": False},
        "anchor_tolerance": 1.0e-9}
    print("resolution ladder")
    payload["ladder"] = resolution_ladder(args.dx, args.depth)
    if not args.skip_closures:
        print("shipped closure matrix (dx = 0.04)")
        payload["closures"] = closure_matrix(0.04, args.depth)
    payload["observations"] = summarise(payload)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print("wrote", args.json)
    if args.figures:
        figures(payload, args.figures)
    print(json.dumps(payload["observations"], indent=2, sort_keys=True))
    return 0


def summarise(payload: dict) -> dict:
    ladder = payload["ladder"]
    xs = [row["adami"]["a_x_max_over_g"] for row in ladder]
    variants = {}
    for name in WALL_PRESSURE:
        values = [row["wall_pressure_variants"].get(name, {}).get("a_x_max_over_g")
                  for row in ladder]
        variants[name] = [v for v in values if v is not None]
    return {
        "a_x_by_dx": [{"dx": row["dx"], "a_x": row["adami"]["a_x_max_over_g"],
                       "a_y": row["reconstruct"]["a_y_max_over_g"]} for row in ladder],
        "a_x_max_spread_ratio": (max(xs) / min(xs)) if xs else None,
        "toleration_shortfall_orders": (math.log10(min(xs) / payload["anchor_tolerance"])
                                       if xs else None),
        "wall_pressure_variants_factor_vs_adami": {
            name: [row["wall_pressure_variants"][name]["a_x_max_over_g"]
                   / row["adami"]["a_x_max_over_g"]
                   for row in ladder if name in row["wall_pressure_variants"]]
            for name in WALL_PRESSURE if name != "adami"},
        "closures": [{"closure": row["closure"],
                      "a_x": row["a_x_max_over_g"],
                      "a_y": row["a_y_max_over_g"],
                      "converged": row["converged"]}
                     for row in payload.get("closures", [])],
    }


def figures(payload: dict, directory: str) -> None:
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)
    ladder = payload["ladder"]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    dxs = [row["dx"] for row in ladder]
    ax.plot(dxs, [row["adami"]["a_x_max_over_g"] for row in ladder], "o-",
            label="closed tank (shipped Adami wall)")
    ax.plot(dxs, [row["wall_pressure_variants"]["mirror"]["a_x_max_over_g"]
                  for row in ladder], "s--", label="mirrored wall pressure")
    ax.plot(dxs, [row["wall_pressure_variants"]["hydrostatic"]["a_x_max_over_g"]
                  for row in ladder], "^:", label="analytic hydrostatic wall pressure")
    ax.axhline(payload["anchor_tolerance"], color="k", ls="--", lw=1,
               label="anchor tolerance 1e-9")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("particle spacing dx [m]")
    ax.set_ylabel("max |a_x| / g in the side-wall band")
    ax.set_title("Closed tank: the residual does not refine away")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "lake_wall_gap_resolution.png"), dpi=160)
    plt.close(fig)

    row = min(ladder, key=lambda r: abs(r["dx"] - 0.04))
    columns = sorted(row["adami"]["per_column_max"], key=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(range(len(columns)),
           [row["adami"]["per_column_max"][c] for c in columns],
           color="#4477aa")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"{float(c):.2f}" for c in columns], rotation=90, fontsize=6)
    ax.set_xlabel("particle column x [m]")
    ax.set_ylabel("max |a_x| / g in the column")
    ax.set_title(f"dx = {row['dx']:.3f} m: the residual lives in the two wall columns")
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "lake_wall_gap_columns.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())