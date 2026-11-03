"""Quantitative Young-Laplace validation of the CSF surface-tension model.

The older ``run.py`` benchmark only asserted that the surface-tension force was
non-zero and "predominantly inward". Those criteria pass for a noisy force
field whose magnitude is two orders off, so they cannot detect a broken
curvature estimate. This benchmark replaces them with the analytic reference:

    Young-Laplace (2-D circular droplet):  Delta p = sigma / R
    curvature of a circle:                 kappa   = 1 / R  (outward normal)

and reports, per radius, the relative error of the measured curvature and of the
CSF accelerations. Acceptance threshold: 15 % (platform accuracy standard).

Quantities measured on the interface shell (|grad c| > 0.3 max):

    cos_nr        : n . r_hat, should be ~ +1 (outward normal aligned with r)
    kappa_dS      : |grad c|-weighted mean curvature, compare with 1/R
    a_radial_dS   : |grad c|-weighted mean inward radial acceleration,
                    compare with sigma / (rho R) (the CSF acceleration scale)

Run:
    PYTHONPATH=src python3 benchmarks/static_droplet/laplace_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import create_circular_droplet  # noqa: E402

from tensorsph.droplet_impact import Config, cubic_spline_kernel  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.surface_tension import (  # noqa: E402
    compute_color_gradient,
    compute_surface_tension_force,
)

TOL = 0.15  # platform accuracy standard: 15 %


def measure(radius: float, dx: float = 0.005, sigma: float = 0.072, rho0: float = 1000.0):
    """Measure curvature and CSF acceleration for one droplet radius."""
    center = (0.2, 0.2)
    pos, mass, density = create_circular_droplet(
        radius=radius, dx=dx, center=center, domain_size=(0.4, 0.4), rho0=rho0
    )
    cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=sigma)
    h = cfg.h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)

    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)

    # Recover the field quantities the force was built from, for diagnostics.
    i, j = pairs.unbind(dim=1)
    vol_i = mass[i] / density[i]
    vol_j = mass[j] / density[j]
    w_pair = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    color = torch.zeros(pos.shape[0], dtype=pos.dtype).index_add(0, i, vol_j * w_pair)
    color = color.index_add(0, j, vol_i * w_pair)
    bulk = color > 0.9 * color.max()
    color = color / color[bulk].mean()
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    dS = grad_c.norm(dim=1)

    rvec = pos - torch.tensor(center, dtype=pos.dtype)
    r = rvec.norm(dim=1)
    rhat = rvec / r[:, None]

    shell = dS > 0.3 * dS.max()
    vol = (mass / density)

    # --- curvature (|grad c|-weighted mean on the interface shell) ---
    from tensorsph.surface_tension import compute_curvature

    normal = -grad_c / dS.clamp_min(1e-30)[:, None]
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)
    w = dS[shell] * vol[shell]
    w = w / w.sum()
    kappa_dS = float((kappa[shell] * w).sum())
    kappa_theory = 1.0 / radius

    # --- Young-Laplace pressure jump from the CSF force integral ---
    # 2-D: f = sigma*kappa*delta_s*n per unit area, so the total inward force per
    # unit depth is  sigma*kappa*L  and the equivalent pressure jump is
    #     Delta p = |sum_i rho*a_i*V_i . r_hat_i| / L ,  L = 2*pi*R
    # which must equal sigma/R. Using the raw acceleration scale sigma/(rho R)
    # instead would ignore the delta_s factor and be two orders too small.
    f_radial = ((density[:, None] * acc) * rhat).sum(dim=1) * vol
    l_interface = 2.0 * 3.141592653589793 * radius
    dp_measured = float(abs(f_radial.sum()) / l_interface)
    dp_theory = sigma / radius

    cos_nr = (normal[shell] * rhat[shell]).sum(dim=1)
    a_inward_cos = -(acc[shell] * rhat[shell]).sum(dim=1)
    a_inward_cos = a_inward_cos / acc[shell].norm(dim=1).clamp_min(1e-30)

    return {
        "radius_m": radius,
        "n_particles": int(pos.shape[0]),
        "n_shell": int(shell.sum()),
        "dx_m": dx,
        "h_over_dx": cfg.h_factor,
        "kappa_theory_1_per_m": kappa_theory,
        "kappa_measured_dS_weighted": kappa_dS,
        "kappa_rel_error": abs(kappa_dS / kappa_theory - 1.0),
        "dp_theory_Pa": dp_theory,
        "dp_measured_Pa": dp_measured,
        "dp_rel_error": abs(dp_measured / dp_theory - 1.0),
        "normal_cos_median": float(cos_nr.median()),
        "force_inward_cos_median": float(a_inward_cos.median()),
        "force_inward_cos_min": float(a_inward_cos.min()),
        "delta_s_max": float(dS.max()),
        "delta_s_peak_reference_1_over_h": 1.0 / h,
    }


def operator_sanity(radius: float, dx: float = 0.005, rho0: float = 1000.0):
    """Feed the *analytic* normal (r_hat) into the curvature operator.

    This isolates the two error sources: if div(n) reproduces 1/R with the exact
    normal, then the operator is sound and any remaining error in the coupled
    pipeline comes from the colour-function / normal construction.
    """
    from tensorsph.surface_tension import compute_curvature

    center = (0.2, 0.2)
    pos, mass, density = create_circular_droplet(
        radius=radius, dx=dx, center=center, domain_size=(0.4, 0.4), rho0=rho0
    )
    cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=0.072)
    h = cfg.h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rvec = pos - torch.tensor(center, dtype=pos.dtype)
    r = rvec.norm(dim=1)
    rhat = rvec / r[:, None]
    kappa = compute_curvature(pos, rhat, mass, density, pairs, h)
    ring = (r > 0.8 * radius) & (r <= radius)
    kappa_med = float(kappa[ring].median())
    return {
        "radius_m": radius,
        "kappa_exact_normal_median": kappa_med,
        "kappa_exact_normal_rel_error": abs(kappa_med * radius - 1.0),
    }


def plot(rows, out_png: Path, sigma: float, rho0: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = [x["radius_m"] for x in rows]
    inv_r = [1.0 / x for x in r]
    k_meas = [x["kappa_measured_dS_weighted"] for x in rows]
    k_th = [x["kappa_theory_1_per_m"] for x in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(inv_r, k_th, "k--", label=r"theory $\kappa = 1/R$")
    ax.plot(inv_r, k_meas, "o-", color="tab:red", label=r"measured $\kappa$ (|∇c|-weighted)")
    ax.set_xlabel(r"$1/R$  [1/m]")
    ax.set_ylabel(r"$\kappa$  [1/m]")
    ax.set_title("Young–Laplace curvature check")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    err = [100 * x["kappa_rel_error"] for x in rows]
    ax.bar([f"R={x:.3f}" for x in r], err, color="tab:blue")
    ax.axhline(100 * TOL, color="crimson", ls="--", label="15 % acceptance")
    ax.set_ylabel("relative error  [%]")
    ax.set_title(r"$\kappa$ vs analytic $1/R$")
    ax.legend()
    for idx, e in enumerate(err):
        ax.text(idx, e + 1.0, f"{e:.1f}%", ha="center", fontsize=8)

    fig.suptitle(rf"TensorSPH CSF surface tension: $\sigma$={sigma} N/m, $\rho_0$={rho0} kg/m$^3$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    radii = [0.05, 0.075, 0.10, 0.125, 0.15]
    sigma, rho0 = 0.072, 1000.0

    print("=" * 78)
    print("Young-Laplace validation of CSF surface tension")
    print("=" * 78)
    print(f"{'R [m]':>8} {'1/R':>8} {'kappa':>9} {'err%':>7} "
          f"{'dp_th':>9} {'dp_meas':>9} {'err%':>7} {'cosn':>7} {'cosF':>7}")
    rows = []
    for radius in radii:
        row = measure(radius, sigma=sigma, rho0=rho0)
        rows.append(row)
        print(f"{row['radius_m']:>8.3f} {row['kappa_theory_1_per_m']:>8.2f} "
              f"{row['kappa_measured_dS_weighted']:>9.3f} "
              f"{100 * row['kappa_rel_error']:>7.1f} "
              f"{row['dp_theory_Pa']:>9.4f} "
              f"{row['dp_measured_Pa']:>9.4f} "
              f"{100 * row['dp_rel_error']:>7.1f} "
              f"{row['normal_cos_median']:>7.3f} "
              f"{row['force_inward_cos_median']:>7.3f}")

    kappa_ok = all(r["kappa_rel_error"] < TOL for r in rows)
    dp_ok = all(r["dp_rel_error"] < TOL for r in rows)
    normal_ok = all(r["normal_cos_median"] > 0.9 for r in rows)

    out_dir = Path(__file__).resolve().parent
    report = {
        "benchmark": "young_laplace_csf_quantitative",
        "sigma_N_per_m": sigma,
        "rho0_kg_per_m3": rho0,
        "tolerance": TOL,
        "criteria": {
            "curvature_rel_error_below_15pct_all_radii": kappa_ok,
            "laplace_dp_rel_error_below_15pct_all_radii": dp_ok,
            "normal_cos_median_above_0.9_all_radii": normal_ok,
        },
        "operator_sanity_analytic_normal": [operator_sanity(x) for x in radii],
        "rows": rows,
        "status": "PASS" if (kappa_ok and dp_ok and normal_ok) else "FAIL",
    }
    with open(out_dir / "laplace_validation.json", "w") as fh:
        json.dump(report, fh, indent=2)

    png = out_dir / "laplace_validation.png"
    plot(rows, png, sigma=sigma, rho0=rho0)
    print(f"\nArtifacts: {out_dir / 'laplace_validation.json'} , {png}")
    print(f"Curvature <{100 * TOL:.0f}% at all radii: {kappa_ok}")
    print(f"Normal alignment (cos > 0.9):        {normal_ok}")
    print(f"STATUS: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
