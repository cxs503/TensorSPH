"""Young-Laplace verification of the two CSF surface-tension schemes.

Companion of ``benchmarks/laplace_law`` (m2174), which recorded the *failure* of
the original CSF discretisation: curvature 3-12x too high, cut-force identity
missed by up to 2x and the Laplace jump off by 77-118 %.  This harness scores
the replacement scheme (``cfd_scheme="consistent"``, see
``tensorsph.csf_consistent``) against the same three analytic references and
against the legacy scheme, on one axis:

1. ``Delta p = sigma / R``            (Young-Laplace, 2-D drop)
2. ``F_cut = 2 sigma``                (surface tension = force per unit length)
3. ``kappa = 1 / R``                  (CSF curvature on a circle)
4. ``int delta_s dn = 1``             (delta-function normalisation)
5. ``max|v| / sqrt(sigma/rhoR) << 1`` (spurious currents, Adami et al. 2012)

The acceptance threshold is the platform standard: 15 % on every key quantity.

Usage::

    PYTHONPATH=src python benchmarks/csf_consistent/run.py            # full
    PYTHONPATH=src python benchmarks/csf_consistent/run.py --quick    # instant only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields
from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet
radial_band = _law.radial_band


# --------------------------------------------------------------------------- #
# analytic geometry helpers
# --------------------------------------------------------------------------- #
def delta_s_integral(pos, mass, rho0, delta_s, radius, lo=0.6, hi=1.05):
    """``int delta_s dn`` over the interface, in units of 1.

    For a 2-D circular interface ``int_{shell} delta_s dA = 2 pi R * int
    delta_s dn``, so the radial integral is recovered by dividing the shell
    integral by the local circumference ``2 pi r``.
    """
    r = torch.linalg.vector_norm(pos, dim=1)
    m = (r > lo * radius) & (r < hi * radius)
    d_v = mass / rho0
    contrib = (delta_s[m] * d_v[m] / (2.0 * math.pi * r[m].clamp_min(1e-30))).sum()
    return float(contrib)


def force_fields(pos, mass, rho, cfg, wall=None):
    """Acceleration from the scheme selected by ``cfg.csf_scheme``."""
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    from tensorsph.surface_tension import compute_surface_tension_force
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg, wall=wall)
    return acc, pairs


def csf_state(pos, mass, rho0, cfg):
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    if cfg.csf_scheme == "consistent":
        return csf_fields(pos, mass, pairs, h, rho0), pairs
    grad_c = None
    st = _law.csf_fields(pos, density_from_neighbor_pairs(pos, mass, h, pairs),
                         mass, pairs, cfg)
    from tensorsph.csf_consistent import CSFState
    color, grad_c, normal, kappa, delta_s, _ = st
    return CSFState(color=color, grad_c=grad_c, normal=normal, kappa=kappa,
                    delta_s=delta_s), pairs


# --------------------------------------------------------------------------- #
# static diagnostics
# --------------------------------------------------------------------------- #
def instant_diagnostics(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0,
                        sound_speed=20.0, viscosity=0.01, scheme="consistent"):
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme=scheme)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))
    st, pairs = csf_state(pos, mass, rho0, cfg)
    acc, _ = force_fields(pos, mass, rho, cfg)

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    mag = torch.linalg.vector_norm(acc, dim=1)
    # Only particles that actually carry force can be scored for direction.
    active = st.delta_s > 0.05 * st.delta_s.max()
    cos_in = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)
    band = active & (r / radius > 0.6) & (r / radius <= 1.02)

    cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
    theory_cut = -2.0 * sigma

    w = st.delta_s
    kappa_w = float((st.kappa.abs() * w).sum() / w.sum())
    kappa_exact = 1.0 / radius

    return {
        "scheme": scheme,
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "kappa": {"exact_1_per_m": kappa_exact,
                  "delta_s_weighted_mean_1_per_m": kappa_w,
                  "rel_error_pct": (kappa_w - kappa_exact) / kappa_exact * 100.0,
                  "within_15pct": abs(kappa_w - kappa_exact) / kappa_exact < 0.15},
        "delta_s": {"integral_dn": delta_s_integral(pos, mass, rho0, st.delta_s,
                                                    radius),
                    "within_35pct": abs(delta_s_integral(pos, mass, rho0,
                                                         st.delta_s, radius) - 1.0) < 0.35},
        "cut_force": {"theory_N_per_m": theory_cut,
                      "measured_N_per_m": cut,
                      "ratio_measured_over_theory": cut / theory_cut,
                      "rel_error_pct": abs(cut / theory_cut - 1.0) * 100.0,
                      "within_15pct": abs(cut / theory_cut - 1.0) < 0.15},
        "direction": {"n_active_band": int(band.sum()),
                      "cos_band_median": float(cos_in[band].median()),
                      "fraction_inward": float((cos_in[band] > 0).float().mean()),
                      "within_15pct": float((cos_in[band] > 0.85).float().mean()) > 0.9},
        "color_band": radial_band(pos, st.color, radius),
        "max_acc_st_m_per_s2": float(mag.max()),
    }


# --------------------------------------------------------------------------- #
# relaxation: the Laplace jump itself
# --------------------------------------------------------------------------- #
def relax(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
          steps=900, viscosity=0.01, scheme="consistent", output_every=50):
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme=scheme)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    v_ref = math.sqrt(sigma / (rho0 * radius))
    dp_theory = sigma / radius
    comp_needed = dp_theory / (cfg.sound_speed ** 2 * rho0 / cfg.gamma)

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))
    pressure = tait_pressure(rho, cfg)
    hist = {"t": [], "vmax_over_vref": [], "p_center": []}

    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % output_every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            hist["t"].append((step + 1) * dt)
            hist["vmax_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)
            hist["p_center"].append(float(pressure[r < 0.3 * radius].mean()))

    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * radius
    shell = (r > 0.80 * radius) & (r < 0.95 * radius)
    p_core = float(pressure[core].mean())
    p_shell = float(pressure[shell].mean())
    dp = p_core - p_shell
    rel = (dp - dp_theory) / dp_theory * 100.0
    return {
        "scheme": scheme,
        "setup": {"radius_m": radius, "dx_m": dx, "h_m": h, "sigma_N_per_m": sigma,
                  "rho0": rho0, "sound_speed": sound_speed, "steps": steps,
                  "dt_s": dt, "t_end_s": steps * dt,
                  "t_over_t_sigma": steps * dt / math.sqrt(rho0 * radius ** 3 / sigma)},
        "reference": {"delta_p_Pa": dp_theory},
        "laplace": {"p_center_Pa": p_core, "p_interface_band_Pa": p_shell,
                    "delta_p_measured_Pa": dp, "rel_error_pct": rel,
                    "within_3pct": abs(rel) < 3.0, "within_15pct": abs(rel) < 15.0},
        "spurious_current": {"v_ref_m_per_s": v_ref,
                             "vmax_over_vref_final": hist["vmax_over_vref"][-1],
                             "passes_adami_0p1": hist["vmax_over_vref"][-1] < 0.1},
        "density": {"rho_core_over_rho0": float(rho[core].mean() / rho0),
                    "compression_required_by_law": comp_needed},
        "history": hist,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(rows_consistent, rows_legacy, relax_report, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    ax = axes[0]
    for rows, label, style in ((rows_legacy, "legacy CSF (m277)", "s--"),
                              (rows_consistent, "consistent CSF (this work)", "o-")):
        ax.plot([r["dx_m"] for r in rows],
                [r["curvature"]["delta_s_weighted_mean_1_per_m"] for r in rows]
                if "curvature" in rows[0] else
                [r["kappa"]["delta_s_weighted_mean_1_per_m"] for r in rows],
                style, label=label)
    ax.axhline(1.0 / rows_consistent[0].get("radius_m", 0.02), color="k", ls=":",
               label=r"analytic $\kappa=1/R$")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$|\kappa|$ [1/m]")
    ax.set_title(r"(a) CSF curvature vs $\kappa=1/R$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for rows, label, style in ((rows_legacy, "legacy CSF (m277)", "s--"),
                              (rows_consistent, "consistent CSF", "o-")):
        key = "curvature" if "curvature" in rows[0] else "kappa"
        ax.plot([r["dx_m"] for r in rows],
                [abs(r[key]["rel_error_pct"]) for r in rows], style, label=label)
    ax.axhline(15.0, color="r", ls="--", label="15 % acceptance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("|error| [%]")
    ax.set_title("(b) curvature error over the resolution scan")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for rows, label, style in ((rows_legacy, "legacy CSF (m277)", "s--"),
                              (rows_consistent, "consistent CSF", "o-")):
        key = "cut_force"
        ax.plot([r["dx_m"] for r in rows],
                [r[key]["ratio_measured_over_theory"] for r in rows], style,
                label=label)
    ax.axhspan(0.85, 1.15, color="g", alpha=0.12, label="15 % band")
    ax.axhline(1.0, color="k", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_title(r"(c) cut-force identity $F_{cut}/(2\sigma)$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    prof = [(0.5, relax_report["laplace"]["p_interface_band_Pa"]),
            (0.0, relax_report["laplace"]["p_center_Pa"])]
    ax.axhline(relax_report["reference"]["delta_p_Pa"], color="k", ls="--",
               label=r"analytic $\Delta p=\sigma/R$")
    ax.axhline(relax_report["laplace"]["delta_p_measured_Pa"], color="r", ls=":",
               label="measured jump")
    ax.plot([p[0] for p in prof], [p[1] for p in prof], "o", color="C0")
    ax.set_xlim(-0.1, 0.6)
    ax.set_xlabel("r / R")
    ax.set_ylabel("pressure [Pa]")
    ax.set_title(f"(d) Young-Laplace: {relax_report['scheme']} scheme")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "csf_consistent.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="instant diagnostics only")
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()

    scan = (0.003, 0.002, 0.0015)
    rows_c = []
    rows_l = []
    for dx in scan:
        for scheme, rows in (("legacy", rows_l), ("consistent", rows_c)):
            row = instant_diagnostics(dx=dx, scheme=scheme)
            rows.append(row)
            print(f"{scheme:10s} dx={dx:<7g} kappa_err={row['kappa']['rel_error_pct']:+7.1f}% "
                  f"Fcut/(2sig)={row['cut_force']['ratio_measured_over_theory']:.3f} "
                  f"int(ds)={row['delta_s']['integral_dn']:.3f} "
                  f"cos_med={row['direction']['cos_band_median']:+.3f}")

    relax_report = None
    if not a.quick:
        for scheme in ("legacy", "consistent"):
            rep = relax(dx=a.dx, steps=a.steps, scheme=scheme)
            print(f"relax {scheme:10s} Delta p={rep['laplace']['delta_p_measured_Pa']:.1f} Pa "
                  f"vs {rep['reference']['delta_p_Pa']:.1f} Pa "
                  f"({rep['laplace']['rel_error_pct']:+.1f}%)  "
                  f"spurious={rep['spurious_current']['vmax_over_vref_final']:.3f}")
            if scheme == "consistent":
                relax_report = rep
            else:
                (HERE / "relax_legacy.json").write_text(json.dumps(rep, indent=2))

    out = {"benchmark": "csf_consistent_laplace_verification",
           "reference": {"delta_p_Pa": 7.2 / 0.02, "kappa_1_per_m": 50.0,
                         "cut_force_N_per_m": -14.4},
           "setup": {"radius_m": 0.02, "sigma_N_per_m": 7.2, "rho0": 1000.0,
                     "sound_speed": 20.0},
           "instant": {"consistent": rows_c, "legacy": rows_l},
           "relax": relax_report}
    (HERE / "validation_report.json").write_text(json.dumps(out, indent=2))
    if relax_report is not None:
        print("figure ->", make_figures(rows_c, rows_l, relax_report))
    print("report ->", HERE / "validation_report.json")


if __name__ == "__main__":
    main()
