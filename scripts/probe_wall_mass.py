"""Is the closed-tank residual a wall *mass* (volume) inconsistency?

The reconstruction inflates the mass of the fluid particles next to the side wall
to compensate the truncated summation density (published row means: m/m_unit up
to 2.4 at the wall column, 1.6 nominal).  The wall particles keep the nominal
lattice mass, so in the momentum sum

    a_i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) gradW_ij

the fluid side of the wall carries ~1.5x the mass of its wall counterpart.  Three
wall-mass conventions are compared, everything else (pair list, pressures, fluid
masses, wall-face pressure) held fixed:

  nominal    m_w = lattice mass (what ships today)
  eos        m_w = m_unit * rho_w / rho0
  adami      m_w = rho_w * V_w with V_w = 1 / sum_f W_fw   (Adami et al. 2012
             boundary-particle volume, i.e. the volume that makes the wall
             particle's kernel support consistent with its fluid neighbours)

Run: PYTHONPATH=src python3 scripts/probe_wall_mass.py --dx 0.04
"""
from __future__ import annotations

import argparse
import json

import torch

from tensorsph.droplet_impact import wall_mass
from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct, residual_report
from wall_gap import accel_terms, build_fields, field_stats, horizontal_map


def wall_mass_variants(f: dict) -> dict:
    cfg, n, nw = f["cfg"], f["n"], f["nw"]
    nominal = f["all_mass"][n:].clone()
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    rho_wall = (cfg.rho0 * (f["p_wall"] / bulk + 1.0).pow(1.0 / cfg.gamma)
                ).clamp_min(cfg.rho0)
    out = {"nominal": nominal,
           "eos": nominal * rho_wall / cfg.rho0}
    # Adami boundary volume: 1 / sum_f W_fw over the wall particle's fluid support
    kernel_sum = torch.zeros(nw, dtype=f["pos"].dtype, device=f["pos"].device)
    kernel_sum = kernel_sum.index_add(0, f["wall_idx"], f["weight"])
    out["adami"] = rho_wall / kernel_sum.clamp_min(1e-30)
    return out, rho_wall, kernel_sum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--depth", type=float, default=0.30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg = LakeConfig(dx=args.dx, depth=args.depth, solver_max_iter=40)
    state = build_state(cfg)
    solved = reconstruct(state, cfg)
    shipped = residual_report(build_state(cfg), cfg, solved["mass"])
    f = build_fields(cfg, solved["mass"])
    variants, rho_wall, kernel_sum = wall_mass_variants(f)
    rows = []
    for name, mass_wall in variants.items():
        f_var = dict(f)
        f_var["all_mass"] = torch.cat((f["all_mass"][:f["n"]], mass_wall))
        ax = horizontal_map(f_var, f["p_wall"])
        stats = field_stats(f_var, ax)
        terms = accel_terms(f_var, f["p_wall"])
        worst = int(terms["total"].abs().argmax())
        row = {"convention": name, "a_x_max_over_g": stats["a_x_max_over_g"],
               "a_x_wall_band": stats["a_x_wall_band"],
               "a_x_interior": stats["a_x_interior"],
               "worst": stats["worst_particle"],
               "worst_own_over_g": float(terms["own_pressure_term"][worst] / cfg.gravity),
               "worst_nbr_over_g": float(terms["neighbour_pressure_term"][worst] / cfg.gravity),
               "wall_mass_mean_over_munit": float(mass_wall.mean() / state["m_unit"])}
        rows.append(row)
        print(f"{name:>8s}  max|a_x|/g={row['a_x_max_over_g']:.4e}"
              f"  wall={row['a_x_wall_band']:.4e}"
              f"  <m_w/m_unit>={row['wall_mass_mean_over_munit']:.3f}"
              f"  worst=({row['worst']['x']:.3f},{row['worst']['y']:.3f})")
    print(f"\nshipped residual_report: a_x={shipped['a_max_over_g_horizontal']:.4e}")
    print(f"fluid mass spread m/m_unit: {float(solved['mass'].min() / state['m_unit']):.3f} .. "
          f"{float(solved['mass'].max() / state['m_unit']):.3f}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"dx": args.dx, "rows": rows,
                       "shipped_a_x": shipped["a_max_over_g_horizontal"],
                       "fluid_mass_over_munit": [
                           float(solved["mass"].min() / state["m_unit"]),
                           float(solved["mass"].max() / state["m_unit"])]},
                      handle, indent=2, sort_keys=True)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())