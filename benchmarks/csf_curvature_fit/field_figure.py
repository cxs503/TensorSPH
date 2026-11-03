"""Field plot: curvature carried by the droplet shell, ``div n`` vs interface fit.

Writes ``kappa_field.png`` next to this script: particles coloured by signed
``kappa`` for both operators, on the same lattice droplet.  Used by
``docs/verification_surface_tension_curvature.md``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorsph.csf_consistent import csf_fields  # noqa: E402
from tensorsph.droplet_impact import Config  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)


def main(radius=0.02, dx=0.002):
    spec = importlib.util.spec_from_file_location(
        "laplace_law_run", HERE.parent / "laplace_law" / "run.py")
    law = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(law)
    cfg = Config(dx=dx, h_factor=1.3, rho0=1000.0, sound_speed=20.0,
                 surface_tension_coefficient=7.2, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    pos, mass = law.hexagonal_droplet(radius, dx, 1000.0)
    pairs = cell_linked_pairs(pos, 4.0 * h)
    r = torch.linalg.vector_norm(pos, dim=1)
    shell = (r / radius > 0.85) & (r / radius < 1.02)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    states = [("mls", csf_fields(pos, mass, pairs, h, 1000.0)),
              ("fit_quadratic", csf_fields(pos, mass, pairs, h, 1000.0,
                                           curvature_scheme="fit_quadratic"))]
    lim = (0.0, 100.0)
    for ax, (name, st) in zip(axes[:2], states):
        sc = ax.scatter(pos[shell, 0] * 1e3, pos[shell, 1] * 1e3,
                        c=st.kappa[shell].abs(), s=22, cmap="viridis",
                        vmin=lim[0], vmax=lim[1])
        ax.set_aspect("equal")
        ax.set_title(f"{name}: |kappa| on the shell\n"
                     f"mean {float(st.kappa[shell].abs().mean()):.1f} 1/m "
                     f"(analytic {1.0 / radius:.0f})", fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(sc, ax=ax, label="|kappa| [1/m]")

    ax = axes[2]
    for name, st in states:
        ax.hist(st.kappa[shell].abs(), bins=18, alpha=0.6, label=name)
    ax.axvline(1.0 / radius, color="k", ls="--", label=r"analytic $1/R$")
    ax.set_xlabel("|kappa| [1/m]")
    ax.set_ylabel("particles")
    ax.set_title("(c) curvature distribution on the shell")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = HERE / "kappa_field.png"
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
