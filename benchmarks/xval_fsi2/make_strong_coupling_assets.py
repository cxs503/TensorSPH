"""Generate the figures/data for docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md.

Panel (a): free-beam (zero load) energy ratio vs rod sub-steps per fluid step at
``rho_s/rho_f = 1`` -- isolates the rod integrator from the fluid coupling.
Panel (b): coupled beam-speed time history at ``rho_s/rho_f = 1`` for the three
configurations of the round: axial-only sub-cycling (pre-m20761 default), the
bending-aware sub-cycling, and the bending-aware + acoustic wall-impedance
variant.

Everything is written programmatically: ``assets/strong_coupling_data.json`` +
``assets/fig_rod_substeps_and_coupling.png``.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorsph.fsi2_similarity import similarity_corrected_case
from tensorsph.fsi2_turek_hron import (
    beam_advance,
    beam_energy,
    beam_initial_nodes,
    beam_max_frequency,
    initial_state,
    step_fsi2,
)

OUT = "docs/xval_fsi2/assets"


def _case(ratio=1.0, dx=0.02, c0=5.0, impedance=False):
    case = similarity_corrected_case(U_mean=0.2)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = 1000.0 * ratio
    case.wall_impedance = impedance
    return case


def rod_energy_sweep(ratio=1.0, n_steps=200):
    case = _case(ratio)
    dt = case.dt()
    out = []
    for n_sub in (1, 2, 4, 8, case.rod_substeps_effective()):
        nodes = beam_initial_nodes(case, torch.device("cpu")).clone()
        n = nodes.shape[0]
        nodes[:, 1] += 1.0e-5 * torch.sin(
            math.pi * torch.arange(n, dtype=torch.float64) / (n - 1))
        nodes[0, 1] = case.cylinder_y
        vel = torch.zeros_like(nodes)
        load = torch.zeros_like(nodes)
        e0 = float(beam_energy(nodes, case))
        finite = True
        for _ in range(n_steps):
            for _ in range(n_sub):
                nodes, vel = beam_advance(nodes, vel, load, case, dt / n_sub)
            if not math.isfinite(float(beam_energy(nodes, case))):
                finite = False
                break
        e = float(beam_energy(nodes, case)) if finite else float("inf")
        out.append({"n_sub": int(n_sub), "energy_ratio": e / e0, "finite": finite})
    return {"case": {"ratio": ratio, "dx": case.dx, "dt": dt,
                     "omega_max_rad_s": beam_max_frequency(case),
                     "dt_omega_max": dt * beam_max_frequency(case),
                     "n_sub_auto": case.rod_substeps_effective()},
            "sweep": out, "n_steps": n_steps}


def coupled_history(label, n_steps=200, sub_override=None, impedance=False):
    case = _case(1.0, impedance=impedance)
    if sub_override is not None:
        case.rod_substeps = sub_override
    state = initial_state(case)
    dt = case.dt()
    t, v = [], []
    for k in range(n_steps):
        row = step_fsi2(state, case, dt)
        speed = float(state.beam_vel.norm(dim=1).max())
        t.append(row["t"])
        v.append(speed)
        if not math.isfinite(speed) or speed > 1.0e3:
            break
    return {"label": label, "t": t, "v_beam": v,
            "n_sub": case.rod_substeps_effective(), "impedance": impedance,
            "v_end": v[-1], "steps_done": len(v)}


def main():
    os.makedirs(OUT, exist_ok=True)
    rod = rod_energy_sweep()
    coupled = [
        coupled_history("axial-only sub-cycling (pre-m20761 default)",
                        sub_override=1),
        coupled_history("bending-aware sub-cycling (m20761)"),
        coupled_history("bending-aware + acoustic wall impedance",
                        impedance=True),
    ]
    data = {"rod_energy_sweep": rod, "coupled": coupled}
    with open(os.path.join(OUT, "strong_coupling_data.json"), "w") as fh:
        json.dump(data, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    xs = [r["n_sub"] for r in rod["sweep"]]
    ys = [max(r["energy_ratio"], 1e-3) for r in rod["sweep"]]
    ax.loglog(xs, ys, "o-")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axvline(rod["case"]["n_sub_auto"], color="C3", ls=":",
               label=f"auto (m20761) = {rod['case']['n_sub_auto']}")
    ax.set_xlabel("rod sub-steps per fluid step")
    ax.set_ylabel("free-beam energy ratio E/E0 after 200 steps")
    ax.set_title(f"(a) rod integrator, rho_s/rho_f = 1\n"
                 f"dt*omega_max = {rod['case']['dt_omega_max']:.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for row in coupled:
        ax.semilogy([x * 1e3 for x in row["t"]], [max(v, 1e-12) for v in row["v_beam"]],
                    label=f"{row['label']} (n_sub={row['n_sub']})")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("max |v_beam| [m/s]")
    ax.set_title("(b) coupled FSI2, rho_s/rho_f = 1, dx = 20 mm")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rod_substeps_and_coupling.png"), dpi=140)
    print(json.dumps({"rod": rod["case"],
                      "coupled": [{k: r[k] for k in ("label", "n_sub", "v_end",
                                                     "steps_done")} for r in coupled]},
                     indent=2))


if __name__ == "__main__":
    main()