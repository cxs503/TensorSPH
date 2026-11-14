"""Impedance power diagnostic: is the acoustic wall term a sink or a pump?

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The coupled arm comparison in :mod:`impedance_sign` is a *trajectory* comparison
and is therefore sensitive to phase (the beam response to the ramping inflow is
oscillatory), so it cannot by itself attribute the energy flux.  This script
measures the quantity the sign question is actually about, step by step on one
common trajectory (the ``wall_impedance = False`` run):

    Wdot = sum_nodes F_impedance,node . v_node        [W/m]

with ``F_impedance`` the difference between the load with and without the
acoustic term, evaluated on the *same* fluid state.  ``Wdot < 0`` means the term
removes beam energy (dissipative, the derived heavy-wall sign); ``Wdot > 0``
means it pumps.  No fitted constant, no trajectory divergence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from tensorsph import droplet_impact as di
from tensorsph import fsi2_turek_hron as fsi2
from tensorsph.fsi2_similarity import similarity_corrected_case
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main(dx=0.02, every=5, nst=120):
    case = similarity_corrected_case(U_mean=0.2)
    case.dx = dx
    case.wall_impedance = False          # common trajectory, no impedance feedback
    state = fsi2.initial_state(case)
    cfg = case.fluid_config()
    cfg.dt = case.dt()
    rows = []
    t0 = time.time()
    for k in range(nst + 1):
        if k % every == 0 and k > 0:
            beam_pos, normals, interp, trib = fsi2.beam_boundary_geometry(
                state.beam_nodes, case)
            wall = torch.cat((state.wall_static, state.cylinder_particles,
                              beam_pos), 0)
            ws = state.wall_static.shape[0] + state.cylinder_particles.shape[0]
            pairs = cell_linked_pairs(torch.cat((state.fluid_pos, wall), 0),
                                      2.0 * case.h())
            wall_vel = torch.cat((torch.zeros_like(state.wall_static),
                                  torch.zeros_like(state.cylinder_particles),
                                  interp @ state.beam_vel), 0)
            _, p, _ = di.rhs(state.fluid_pos, state.fluid_vel, state.fluid_mass,
                             cfg, pairs=pairs, wall=wall, dt=case.dt(),
                             wall_vel=wall_vel)
            row = {"t": float(state.time),
                   "v_beam_max": float(state.beam_vel.norm(dim=1).max())}
            for name, sgn in (("physical", 1.0), ("pre_fix", -1.0)):
                pw, _ = fsi2.boundary_pressure_from_fluid(
                    state.fluid_pos, p, wall, ws, pairs, case.h(),
                    normals=normals, vel=state.fluid_vel, wall_vel=wall_vel,
                    rho0=case.rho0, sound_speed=case.sound_speed,
                    impedance_sign=sgn)
                pw0, _ = fsi2.boundary_pressure_from_fluid(
                    state.fluid_pos, p, wall, ws, pairs, case.h())
                f = fsi2.boundary_force_from_pressure(pw, normals, trib)
                f0 = fsi2.boundary_force_from_pressure(pw0, normals, trib)
                force = (interp.T @ (f - f0)) * case.load_scale
                row[f"wdot_{name}"] = float((force * state.beam_vel).sum())
            rows.append(row)
        if k < nst:
            fsi2.step_fsi2(state, case, case.dt())
    sgn_ok = all(r["wdot_physical"] <= 1e-12 for r in rows if r["v_beam_max"] > 1e-6)
    out = {"date": time.strftime("%Y-%m-%d"),
           "anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
           "dx": dx, "n_steps": nst, "sample_every": every,
           "statement": "Wdot = sum_nodes F_impedance . v_node on the common "
                        "wall_impedance=False trajectory [W/m]",
           "physical_sign_is_dissipative_everywhere": bool(sgn_ok),
           "wall_clock_s": time.time() - t0,
           "rows": rows}
    path = ROOT / "docs/xval_fsi2/assets/impedance_energy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['t']:.5f} v={r['v_beam_max']:.3e} "
              f"Wdot_phys={r['wdot_physical']:+.3e} Wdot_pre={r['wdot_pre_fix']:+.3e}")
    print("dissipative everywhere:", sgn_ok, "->", path)


if __name__ == "__main__":
    main()
