"""Cross-validation of mixture-material surface tension against Young-Laplace.

What is validated
-----------------
The anchor ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION`` asks whether the
surface-tension chain of this library reproduces the mixture-material
(multi-phase, unequal rest density) behaviour of the reference open-source SPH
code (Xiangyu-Hu/SPHinXsys: colour function = normalised kernel ratio of the
target material, per-phase volume, CSF force as a volume force). The exact,
experiment-free reference for a curved interface is the 2-D Young-Laplace law

    Delta p = sigma / R .

Case: a circular disc of material A (radius ``R = 0.04 m``) embedded in material
B, particles on **both** sides of the interface (a complete two-phase cloud, so
the colour function can complete its 1 -> 0 transition), square lattice of
spacing ``dx``, ``h = 1.3 dx``, ``sigma_AB = 0.072 N/m``. The lattice is
volume-consistent: ``m_i = rho0_phase(i) * dx^2``. Three rest-density
configurations are run:

    ``control``     rho_A = rho_B = 1000    (single-material limit)
    ``water/air``   rho_A = 1000, rho_B = 1.2
    ``air/water``   rho_A = 1.2,  rho_B = 1000   (density contrast reversed)

Two arms:

``mixture``
    ``tensorsph.csf_mixture``: normalised phase-indicator colour
    ``c^A = sum_{j in A} V_j W / sum_j V_j W``, ``delta_s = |grad c|``, force
    density ``f = -sigma kappa delta_s n``, resultant over ``f_i V_i``.
``legacy``
    the shipped single-material estimator (``kernel_sum_color`` with one scalar
    ``rho0 = 1000``, ``delta_s = 2 |grad c|``) driven through the same
    curvature operator, i.e. the ablation that isolates the *colour function*
    and the *delta normalisation* from the curvature discretisation.

Measured per configuration (all numbers land in ``results.json``):

    sigma_eff/sigma   = |sum_{x>0} f V| / (2 sigma)      (Laplace law)
    jump              = max c - min c  over the interface shell
    delta_integral    = sum_i delta_s_i V_i / (2 pi R)   (must be 1)
    shell_thickness   = shell particle count / (2 pi R / dx)
    momentum          = |sum_i f_i V_i| / sum_i |f_i| V_i

Run::

    PYTHONPATH=src python benchmarks/xval_mixture_laplace/run.py --plot

Writes ``benchmarks/xval_mixture_laplace/results.json`` and ``figure.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from tensorsph.csf_consistent import mls_divergence, mls_gradient, shepard_smooth
from tensorsph.csf_mixture import (
    MixtureMaterial, MixtureMaterialSet, kernel_sum_color_single_rho0,
    mixture_csf_fields, particle_volume,
)
from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs

RADIUS = 0.04
SIGMA = 0.072
H_FACTOR = 1.3
DX_LIST = (0.005, 0.0025, 0.00125)
RHO_CASES = (
    ("control", 1000.0, 1000.0),
    ("water-air", 1000.0, 1.2),
    ("air-water", 1.2, 1000.0),
)
LEGACY_RHO0 = 1000.0

torch.set_default_dtype(torch.float64)
# Cap the intra-op threads: on a shared 128-core host the default pool thrashes
# and this benchmark is small (<= 13 k particles).  Override with TSPH_THREADS.
torch.set_num_threads(int(os.environ.get("TSPH_THREADS", "8")))


def build_two_phase_disc(radius: float, dx: float, rho_in: float, rho_out: float,
                         outer_shell: float = 0.04,
                         offset: tuple[float, float] = (0.0, 0.0)):
    """Square lattice, particles on both sides of the circle.

    Phase 0 = material A (inside, ``r < R``), phase 1 = material B (outside).
    Mass is ``rho0_phase * dx^2`` so the two materials share one lattice volume
    -- the mixture-material requirement.
    """
    extent = 2.0 * (radius + outer_shell) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    keep = r <= radius + outer_shell
    pos = pos[keep] + torch.tensor(offset)
    r = r[keep]
    phase = (r >= radius).to(torch.long)
    rho = torch.where(phase == 0, torch.tensor(rho_in), torch.tensor(rho_out))
    mass = rho * dx * dx
    return pos, mass, phase, rho


def _measure(pos, vol, phase, force_density, delta_s, color, dx, sigma):
    """Common diagnostics: Laplace ratio, delta integral, shell, momentum."""
    half = pos[:, 0] > 0.0
    f_half = (force_density[half] * vol[half, None]).sum(dim=0)
    resultant = float(torch.linalg.vector_norm(f_half))
    net = (force_density * vol[:, None]).sum(dim=0)
    abs_sum = float((force_density.norm(dim=1) * vol).sum())
    interface = delta_s > 0.10 * float(delta_s.max())
    n_shell = int(interface.sum())
    arc_length = 2.0 * math.pi * RADIUS
    shell_thickness = (n_shell / (arc_length / dx)) if dx > 0 else float("nan")
    return {
        "sigma_eff_over_sigma": resultant / (2.0 * sigma),
        "laplace_error_pct": (resultant / (2.0 * sigma) - 1.0) * 100.0,
        "dp_measured_Pa": (resultant / (2.0 * sigma)) * sigma / RADIUS,
        "dp_theory_Pa": sigma / RADIUS,
        "sinner": None,
        "color_jump": float(color.max() - color.min()),
        "delta_integral_over_2piR": float((delta_s * vol).sum() / arc_length),
        "shell_particles": n_shell,
        "shell_thickness_in_dx": shell_thickness,
        "momentum_residual_rel": float(net.norm()) / max(abs_sum, 1e-300),
    }


def legacy_arm(pos, mass, phase, pairs, h, sigma):
    """The shipped single-material colour + ``delta_s = 2|grad c|`` ablation."""
    vol = mass / LEGACY_RHO0
    color = kernel_sum_color_single_rho0(pos, mass, pairs, h, LEGACY_RHO0)
    grad_c = mls_gradient(pos, color, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    tiny = 1.0e-12
    normal = torch.where(mag > tiny, -grad_c / mag.clamp_min(tiny),
                         torch.zeros_like(grad_c))
    smoothed = shepard_smooth(normal, pos, vol, pairs, h, passes=2)
    norm = torch.linalg.vector_norm(smoothed, dim=1, keepdim=True)
    normal = torch.where(mag > tiny, smoothed / norm.clamp_min(tiny), normal)
    delta_s = 2.0 * mag[:, 0]
    kappa = mls_divergence(pos, normal, vol, pairs, h)
    force_density = (-sigma * kappa * delta_s)[:, None] * normal
    return color, delta_s, force_density, normal, kappa


def run_case(dx: float, label: str, rho_in: float, rho_out: float) -> dict:
    h = H_FACTOR * dx
    pos, mass, phase, rho = build_two_phase_disc(RADIUS, dx, rho_in, rho_out)
    pairs = cell_linked_pairs(pos, 2.0 * h)

    materials = MixtureMaterialSet(
        materials=(MixtureMaterial("A", rho_in), MixtureMaterial("B", rho_out)),
        sigma={(0, 1): SIGMA},
    )

    # ---- mixture arm -----------------------------------------------------
    vol_phys = particle_volume(mass, phase, materials)
    state = mixture_csf_fields(pos, vol_phys, phase, pairs, h, SIGMA,
                               inner_phase=0, curvature_scheme="mls")
    mix = _measure(pos, vol_phys, phase, state.force_density, state.delta_s,
                   state.color, dx, SIGMA)
    mix["kappa_mean_shell"] = float(
        state.kappa[state.delta_s > 0.10 * float(state.delta_s.max())].mean())
    mix["kappa_theory"] = 1.0 / RADIUS
    mix["kappa_error_pct"] = (mix["kappa_mean_shell"] / mix["kappa_theory"] - 1.0) * 100.0

    # ---- legacy arm ------------------------------------------------------
    color_l, delta_l, f_l, normal_l, kappa_l = legacy_arm(
        pos, mass, phase, pairs, h, SIGMA)
    leg = _measure(pos, vol_phys, phase, f_l, delta_l, color_l, dx, SIGMA)
    leg["kappa_mean_shell"] = float(
        kappa_l[delta_l > 0.10 * float(delta_l.max())].mean())
    leg["kappa_theory"] = 1.0 / RADIUS
    leg["kappa_error_pct"] = (leg["kappa_mean_shell"] / leg["kappa_theory"] - 1.0) * 100.0

    # radial profiles for the figure (mixture arm)
    r = pos.norm(dim=1)
    band = (r > RADIUS - 4.0 * h) & (r < RADIUS + 4.0 * h)
    prof_r = r[band]
    order = torch.argsort(prof_r)
    profile = {
        "r_over_dx": ((prof_r[order] - RADIUS) / dx).tolist(),
        "c_mixture": state.color[band][order].tolist(),
        "c_legacy": color_l[band][order].tolist(),
        "grad_c_mixture": state.delta_s[band][order].tolist(),
        "grad_c_legacy_half": (0.5 * delta_l[band][order]).tolist(),
    }

    # momentum balance on an off-lattice centre: on an aligned lattice the
    # resultant cancels by symmetry alone, which would not test anything.
    pos_off, mass_off, phase_off, _ = build_two_phase_disc(
        RADIUS, dx, rho_in, rho_out, offset=(0.37 * dx, 0.21 * dx))
    pairs_off = cell_linked_pairs(pos_off, 2.0 * h)
    vol_off = particle_volume(mass_off, phase_off, materials)
    state_off = mixture_csf_fields(pos_off, vol_off, phase_off, pairs_off, h,
                                   SIGMA, inner_phase=0)
    net = (state_off.force_density * vol_off[:, None]).sum(dim=0)
    abs_sum = float((state_off.force_density.norm(dim=1) * vol_off).sum())
    momentum_offlattice = float(net.norm()) / max(abs_sum, 1e-300)

    return {
        "dx": dx, "case": label, "rho_in": rho_in, "rho_out": rho_out,
        "momentum_residual_offlattice": momentum_offlattice,
        "R_over_dx": RADIUS / dx, "R_over_h": RADIUS / h,
        "n_particles": int(pos.shape[0]),
        "n_inner": int((phase == 0).sum()),
        "mixture": mix, "legacy": leg, "profile": profile,
    }


def make_figure(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [c[0] for c in RHO_CASES]
    colors = {"control": "tab:blue", "water-air": "tab:red", "air-water": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for case in cases:
        rows = [r for r in results["rows"] if r["case"] == case]
        dx = [r["R_over_dx"] for r in rows]
        ax.plot(dx, [r["mixture"]["sigma_eff_over_sigma"] for r in rows], "o-",
                color=colors[case], label=f"mixture, {case}")
        ax.plot(dx, [r["legacy"]["sigma_eff_over_sigma"] for r in rows], "s--",
                color=colors[case], alpha=0.55, label=f"legacy, {case}")
    ax.axhline(1.0, color="k", lw=1.0, label=r"exact $\sigma_{eff}/\sigma=1$")
    ax.axhspan(0.85, 1.15, color="grey", alpha=0.15, label="15 % band")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("R / dx")
    ax.set_ylabel(r"$\sigma_{eff}/\sigma$")
    ax.set_title("Young-Laplace, density-ratio ladder")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for case in cases:
        rows = [r for r in results["rows"] if r["case"] == case]
        dx = [r["R_over_dx"] for r in rows]
        ax.plot(dx, [r["mixture"]["delta_integral_over_2piR"] for r in rows], "o-",
                color=colors[case], label=f"mixture, {case}")
        ax.plot(dx, [r["legacy"]["delta_integral_over_2piR"] for r in rows], "s--",
                color=colors[case], alpha=0.55)
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("R / dx")
    ax.set_ylabel(r"$\sum_i \delta_{s,i} V_i \,/\, (2\pi R)$")
    ax.set_title("Brackbill delta-function mass")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    mid = [r for r in results["rows"] if abs(r["dx"] - 0.0025) < 1e-12]
    for case in cases:
        rows = [r for r in mid if r["case"] == case]
        if not rows:
            continue
        p = rows[0]["profile"]
        ax.plot(p["r_over_dx"], p["c_mixture"], "-", color=colors[case],
                label=f"c mixture, {case}")
        ax.plot(p["r_over_dx"], p["c_legacy"], ":", color=colors[case],
                label=f"c legacy, {case}")
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"$(r - R)/dx$")
    ax.set_ylabel("colour function c")
    ax.set_title("Interface profile at R/dx = 16")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.suptitle("TensorSPH mixture-material surface tension vs Young-Laplace "
                 "(sigma = 0.072 N/m, R = 0.04 m)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--dx", default=None,
                    help="comma separated dx list, default 5,2.5,1.25 mm")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dx_list = ([float(x) * 1e-3 for x in a.dx.split(",")] if a.dx else list(DX_LIST))

    rows = []
    print("=" * 96)
    print("Mixture-material surface tension vs Young-Laplace:  dp = sigma / R")
    print(f"  sigma = {SIGMA} N/m, R = {RADIUS} m, dp = {SIGMA / RADIUS:.4f} Pa")
    print("=" * 96)
    print(f"{'case':10s} {'dx[mm]':>7s} {'N':>6s} | "
          f"{'mix s/s':>8s} {'err%':>7s} {'int ds':>7s} {'kerr%':>7s} | "
          f"{'leg s/s':>8s} {'err%':>7s} {'int ds':>7s} {'kerr%':>7s}")
    for label, rho_in, rho_out in RHO_CASES:
        for dx in dx_list:
            row = run_case(dx, label, rho_in, rho_out)
            rows.append(row)
            m, l = row["mixture"], row["legacy"]
            print(f"{label:10s} {dx*1e3:7.3f} {row['n_particles']:6d} | "
                  f"{m['sigma_eff_over_sigma']:8.4f} {m['laplace_error_pct']:7.1f} "
                  f"{m['delta_integral_over_2piR']:7.3f} {m['kappa_error_pct']:7.1f} | "
                  f"{l['sigma_eff_over_sigma']:8.4f} {l['laplace_error_pct']:7.1f} "
                  f"{l['delta_integral_over_2piR']:7.3f} {l['kappa_error_pct']:7.1f}")

    finest = min(dx_list)  # smallest dx = finest resolution
    def pick(arm, case):
        return [r for r in rows if r["case"] == case and abs(r["dx"] - finest) < 1e-12][0][arm]

    summary = {
        "acceptance": "<15% error on sigma_eff/sigma",
        "finest_R_over_dx": RADIUS / finest,
        "mixture_finest": {c: pick("mixture", c)["sigma_eff_over_sigma"] for c, _, _ in RHO_CASES},
        "legacy_finest": {c: pick("legacy", c)["sigma_eff_over_sigma"] for c, _, _ in RHO_CASES},
        "mixture_within_15pct": {c: abs(pick("mixture", c)["sigma_eff_over_sigma"] - 1.0) < 0.15
                                 for c, _, _ in RHO_CASES},
        "legacy_within_15pct": {c: abs(pick("legacy", c)["sigma_eff_over_sigma"] - 1.0) < 0.15
                                for c, _, _ in RHO_CASES},
        "mixture_density_ratio_spread": max(
            pick("mixture", c)["sigma_eff_over_sigma"] for c, _, _ in RHO_CASES)
            - min(pick("mixture", c)["sigma_eff_over_sigma"] for c, _, _ in RHO_CASES),
        "legacy_density_ratio_spread": max(
            pick("legacy", c)["sigma_eff_over_sigma"] for c, _, _ in RHO_CASES)
            - min(pick("legacy", c)["sigma_eff_over_sigma"] for c, _, _ in RHO_CASES),
    }
    report = {
        "setup": {
            "radius_m": RADIUS, "sigma_N_per_m": SIGMA, "h_factor": H_FACTOR,
            "dx_list_m": dx_list,
            "rho_cases": [{"label": c[0], "rho_in": c[1], "rho_out": c[2]}
                          for c in RHO_CASES],
            "legacy_rho0": LEGACY_RHO0,
            "curvature": "div n (mls), normal Shepard x2",
        },
        "rows": rows,
        "summary": summary,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print("\nsummary:", json.dumps(summary, indent=1))
    if a.plot:
        make_figure(report, out / "figure.png")
        print(f"wrote {out / 'figure.png'}")
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()