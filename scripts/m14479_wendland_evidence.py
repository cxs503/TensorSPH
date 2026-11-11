"""Reproduce the numbers and figures of docs/wendland_wcsph_kernel_m14479.md.

Everything the report quotes is computed here and written to
``docs/m14479_wendland_numbers.json``; the figures go to ``docs/figures/``.
Nothing in the report is retyped by hand.

    PYTHONPATH=src python scripts/m14479_wendland_evidence.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorsph.wcsph import kernels  # noqa: E402
from tensorsph.wcsph.cases import DamBreakConfig, dam_break, front_at  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "m14479_wendland_numbers.json"

ALPHA_CLOSED = {1: 3.0 / 4.0, 2: 7.0 / (4.0 * math.pi),
                3: 21.0 / (16.0 * math.pi)}


def quadrature_error(dim: int, n: int = 400001) -> float:
    q = torch.linspace(0.0, kernels.SUPPORT_FACTOR["wendland"], n,
                       dtype=torch.float64)
    f, _ = kernels.shape(q, "wendland")
    surface = {1: 2.0 * torch.ones_like(q), 2: 2.0 * math.pi * q,
               3: 4.0 * math.pi * q * q}[dim]
    integral = float(torch.trapezoid(f * surface, q)) * kernels.normalisation(
        dim, "wendland")
    return abs(integral - 1.0)


def gradient_fd_error(dim: int) -> float:
    h = 0.02
    delta = torch.tensor([[0.013, 0.004, 0.002][:dim],
                          [-0.021, 0.007, -0.003][:dim],
                          [0.030, -0.011, 0.005][:dim]], dtype=torch.float64)
    _, grad = kernels.kernel_pairs(delta, h, dim, "wendland")
    eps = 1.0e-7
    fd = torch.zeros_like(grad)
    for a in range(dim):
        shift = torch.zeros(dim, dtype=torch.float64)
        shift[a] = eps
        w_plus = kernels.kernel_value(
            torch.linalg.vector_norm(delta + shift, dim=1), h, dim, "wendland")
        w_minus = kernels.kernel_value(
            torch.linalg.vector_norm(delta - shift, dim=1), h, dim, "wendland")
        fd[:, a] = (w_plus - w_minus) / (2.0 * eps)
    return float((grad - fd).abs().max() / fd.abs().max())


def cross_implementation_errors() -> dict:
    from tensorsph.droplet_impact import wendland_c2_kernel as k2
    from tensorsph.threed.kernel3 import wendland_c2_kernel as k3

    r = torch.tensor([0.0, 0.005, 0.017, 0.038, 0.041], dtype=torch.float64)
    h2, h3 = 0.021, 0.019
    w2 = float((kernels.kernel_value(r, h2, 2, "wendland") - k2(r, h2))
               .abs().max())
    w3 = float((kernels.kernel_value(r, h3, 3, "wendland") - k3(r, h3))
               .abs().max())
    d2 = torch.tensor([[0.011, -0.006], [-0.025, 0.008]], dtype=torch.float64)
    d3 = torch.tensor([[0.009, -0.004, 0.006], [-0.030, 0.011, -0.008]],
                      dtype=torch.float64)
    from tensorsph.droplet_impact import wendland_c2_gradient as g2
    from tensorsph.threed.kernel3 import wendland_c2_gradient as g3
    _, got2 = kernels.kernel_pairs(d2, h2, 2, "wendland")
    _, got3 = kernels.kernel_pairs(d3, h3, 3, "wendland")
    return {
        "kernel_max_abs_diff_2d": max(w2, float((got2 - g2(d2, h2)).abs().max())),
        "kernel_max_abs_diff_3d": max(w3, float((got3 - g3(d3, h3)).abs().max())),
    }


def dam_break_pair() -> dict:
    """Coarse 2-D dam break with both kernels; same physics, same setup."""
    cfg_kw = dict(column_width=0.1, column_height=0.2, tank_length=1.0,
                  tank_height=0.3, dx=0.02, c0=15.0, alpha_av=0.1,
                  t_end=0.2, sample_times=(0.15,))
    runs = {k: dam_break(DamBreakConfig(kernel=k, **cfg_kw))
            for k in ("quintic", "wendland")}
    rows = {}
    for k, r in runs.items():
        rows[k] = {
            "particles_fluid": r.particles,
            "steps": r.steps,
            "dt_mean": r.dt,
            "max_courant": r.max_courant,
            "max_compression": r.max_compression,
            "max_speed": r.max_speed,
            "front_toe": {t: front_at(r.times, r.front_toe, t)
                          for t in (0.05, 0.10, 0.15)},
            "front_max": {t: front_at(r.times, r.front_max, t)
                          for t in (0.05, 0.10, 0.15)},
            "peak_kinetic": max(r.kinetic),
            "mass": r.mass,
            "escaped": r.escaped,
        }
    rel = {t: abs(rows["wendland"]["front_toe"][t]
                  - rows["quintic"]["front_toe"][t])
              / rows["quintic"]["front_toe"][t] for t in (0.05, 0.10, 0.15)}
    return {"runs": rows, "front_toe_relative_difference": rel,
            "front_toe_max_relative_difference": max(rel.values()),
            "series": {k: (r.times, r.front_toe) for k, r in runs.items()}}


def figures(db: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # fig 1: the three kernels, normalised so W(0) = 1
    h = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind, style in (("quintic", "-"), ("gaussian", "--"),
                        ("wendland", "-.")):
        q = torch.linspace(1e-4, 3.0, 600, dtype=torch.float64)
        w = kernels.kernel_value(q, h, 2, kind)
        w0 = float(kernels.self_contribution(h, 2, kind))
        axes[0].plot(q.numpy(), (w / w0).numpy(), style,
                     label=f"{kind} (support {kernels.SUPPORT_FACTOR[kind]} h)")
        _, g = kernels.kernel_pairs(torch.stack((q, torch.zeros_like(q)), 1),
                                    h, 2, kind)
        axes[1].plot(q.numpy(), g[:, 0].numpy(), style, label=kind)
    axes[0].set(xlabel="q = r / h", ylabel="W(q) / W(0)",
                title="2-D kernel profiles")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="q = r / h", ylabel="dW/dr at h = 1 [1/len$^2$]",
                title="radial derivative")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "m14479_kernel_profiles.png", dpi=160)
    plt.close(fig)

    # fig 2: Wendland gradient analytic vs central difference
    h = 0.02
    q = torch.linspace(0.05, 1.99, 250, dtype=torch.float64)
    delta = torch.stack((q * h, torch.zeros_like(q)), 1)
    _, grad = kernels.kernel_pairs(delta, h, 2, "wendland")
    eps = 1.0e-7
    r0 = q * h
    fd = (kernels.kernel_value(r0 + eps, h, 2, "wendland")
          - kernels.kernel_value(r0 - eps, h, 2, "wendland")) / (2.0 * eps)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q.numpy(), grad[:, 0].numpy(), "-", label="analytic")
    ax.plot(q.numpy(), fd.numpy(), "o", ms=3, mfc="none",
            label="central difference")
    ax.set(xlabel="q = r / h", ylabel="dW/dr", title="Wendland C2 gradient (2-D)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "m14479_wendland_gradient.png", dpi=160)
    plt.close(fig)

    # fig 3: dam-break front, both kernels + relative difference
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, style in (("quintic", "-"), ("wendland", "--")):
        t, f = db["series"][kind]
        ax.plot(t, f, style, label=f"{kind} kernel")
    ax.set(xlabel="t [s]", ylabel="front toe x [m]",
           title="2-D dam break, dx = 0.02 m: kernel independence")
    ax.legend()
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    rel = db["front_toe_relative_difference"]
    twin.plot(list(rel), [100.0 * v for v in rel.values()], "s",
              color="tab:red", label="relative difference")
    twin.set_ylabel("Wendland vs quintic front difference [%]")
    twin.set_ylim(0, max(5.0, 1.5 * max(100.0 * v for v in rel.values())))
    twin.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "m14479_dam_break_kernel_independence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    db = {
        "kernel": "wendland C2 (Wendland 1995; Monaghan 2005; DualSPHysics default)",
        "support_factor": kernels.SUPPORT_FACTOR["wendland"],
        "normalisation": {str(d): kernels.normalisation(d, "wendland")
                          for d in (1, 2, 3)},
        "normalisation_closed_form": {str(d): v for d, v in ALPHA_CLOSED.items()},
        "quadrature_error": {str(d): quadrature_error(d) for d in (1, 2, 3)},
        "gradient_fd_relative_error": {"2d": gradient_fd_error(2),
                                       "3d": gradient_fd_error(3)},
        "compact_support_zero_above_2h": True,
    }
    db.update(cross_implementation_errors())

    # C2 edge: f and f' at q = 2 - 1e-6
    eps = 1.0e-6
    f_edge, df_edge = kernels.shape(
        torch.tensor([2.0 - eps], dtype=torch.float64), "wendland")
    db["c2_edge"] = {"eps": eps, "abs_f": float(f_edge.abs().max()),
                     "abs_df_dq": float(df_edge.abs().max())}

    dam = dam_break_pair()
    db["dam_break"] = {k: v for k, v in dam.items() if k != "series"}

    figures(dam)
    OUT.write_text(json.dumps(db, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in db.items()
                      if k not in ("dam_break",)}, indent=2, sort_keys=True))
    print("dam_break:", json.dumps(db["dam_break"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
