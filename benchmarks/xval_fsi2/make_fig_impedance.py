"""Figure for the impedance-sign round (reads the JSON artifacts, types nothing)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorsph import fsi2_turek_hron as fsi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "docs/xval_fsi2/assets"


def piston_sweep():
    """p_wall vs rho c u_n for a prescribed wall normal velocity (analytic line)."""
    case = fsi2.FSI2Case(dx=0.02, device="cpu")
    state = fsi2.initial_state(case)
    beam_pos, normals, interp, trib = fsi2.beam_boundary_geometry(state.beam_nodes, case)
    from tensorsph.neighbor_search import cell_linked_pairs
    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_pos), 0)
    ws = state.wall_static.shape[0] + state.cylinder_particles.shape[0]
    pairs = cell_linked_pairs(torch.cat((state.fluid_pos, wall), 0), 2.0 * case.h())
    n_b = beam_pos.shape[0] // 2
    p = torch.zeros((state.fluid_pos.shape[0],), dtype=state.fluid_pos.dtype)
    vel = torch.zeros_like(state.fluid_pos)
    speeds = np.linspace(-0.6, 0.6, 13)
    got = []
    for v in speeds:
        wv = torch.zeros_like(wall)
        wv[ws:ws + n_b, 1] = float(v)
        pw, _ = fsi2.boundary_pressure_from_fluid(
            state.fluid_pos, p, wall, ws, pairs, case.h(), normals=normals,
            vel=vel, wall_vel=wv, rho0=case.rho0, sound_speed=case.sound_speed)
        got.append(float(pw[:n_b].mean()))
    return speeds, np.array(got), case.rho0 * case.sound_speed


def main():
    sign = json.loads((ART / "impedance_sign.json").read_text())
    energy = json.loads((ART / "impedance_energy.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))

    speeds, got, zc = piston_sweep()
    ax[0].plot(speeds, zc * speeds, "k-", lw=1.2, label=r"$\rho c\,u_n$ (heavy-wall Riemann)")
    ax[0].plot(speeds, got, "o", ms=5, label="computed wall pressure")
    ax[0].axhline(0.0, color="0.7", lw=0.8)
    ax[0].axvline(0.0, color="0.7", lw=0.8)
    ax[0].set_xlabel(r"wall normal velocity $u_n$ along the outward normal [m/s]")
    ax[0].set_ylabel(r"$p^*$ at the wall [Pa]")
    ax[0].set_title("(a) piston relation, exact to round-off")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    rows = energy["rows"]
    t = [r["t"] * 1e3 for r in rows]
    ax[1].plot(t, [r["wdot_physical"] for r in rows], "-o", ms=3,
               label="fixed sign (+): dissipation")
    ax[1].plot(t, [r["wdot_pre_fix"] for r in rows], "-s", ms=3,
               label="pre-fix sign (-): energy injection")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xlabel("t [ms]")
    ax[1].set_ylabel(r"$\dot W=\sum F_{\rm imp}\cdot v$ [W/m]")
    ax[1].set_title("(b) impedance power on one common trajectory")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for arm, style in (("off", "k:"), ("physical", "C0-"), ("pre_fix", "C3-")):
        r = sign["arms"][arm]
        ax[2].semilogy(np.array(r["times_sample"]) * 1e3,
                       np.maximum(r["speeds_sample"], 1e-14), style, lw=1.4,
                       label=f"{arm} (v_end={r['v_end']:.2e} m/s)")
    ax[2].axhline(sign["thresholds"]["divergent_beam_speed_m_per_s"], color="r",
                  ls="--", lw=0.9, label="divergence gate 0.3 m/s")
    ax[2].set_xlabel("t [ms]")
    ax[2].set_ylabel(r"$\max|v_{\rm beam}|$ [m/s]")
    ax[2].set_title("(c) coupled probe, $\\rho_s/\\rho_f=1$, dx=20 mm")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = ART / "impedance_sign.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
