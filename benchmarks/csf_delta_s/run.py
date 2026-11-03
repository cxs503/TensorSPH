"""``delta_s`` normalisation A/B: hard-coded 2 vs measured colour jump.

``docs/verification_surface_tension_curvature.md`` §7 left one force-level debt
open: the CSF surface delta function integrated to ``int delta_s dn = 0.74``
instead of 1 (a systematic -26 % that did not move with resolution), and the
cut-force identity came out 10.7 % weak.  The same document attributed the
deficit to "the outermost colour gradient being smeared" and proposed a ghost
colour layer.

This harness tests a cheaper and testable explanation.  The colour field of a
lattice drop is *not* a 1 -> 1/2 step: ``c`` stays at 1.0 far into the drop and
then steps down in a single particle layer, and the outermost layer carries
``c_surf ~ 0.62`` rather than 0.5.  Because ``int_fluid |grad c| dn =
c_bulk - c_surf`` holds to <1 % on the discrete field (measured below), the
historical factor of 2 -- which *assumes* ``c_surf = 1/2`` -- is weak by exactly

    2 (1 - c_surf) = 2 * 0.3805 = 0.761,

i.e. the observed 0.74.  So the fix is not a ghost layer but replacing the
assumed jump by the measured one:

    delta_s = |grad c| / (1 - c_surf),     c_surf = q_10(c over interface)

Scored criteria (all analytic, no time integration -> runs in seconds):

1. ``int delta_s dn = 1``          (delta-function normalisation)
2. ``F_cut = 2 sigma``             (cut-force identity, force amplitude)
3. ``kappa = 1/R``                 (curvature, fit_quadratic)
4. median ``cos(a, -rhat)``        (force direction)
5. ``int_fluid |grad c| dn`` vs ``1 - c_surf``  (the identity the fix rests on)

Usage::

    PYTHONPATH=src python benchmarks/csf_delta_s/run.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import DELTA_S_QUANTILE, csf_fields, surface_tension_acceleration
from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent


def _load_sibling(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
_consistent = _load_sibling("csf_consistent_run", "csf_consistent/run.py")
hexagonal_droplet = _law.hexagonal_droplet
delta_s_integral = _consistent.delta_s_integral


def evaluate(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
             viscosity=0.01, curvature="fit_quadratic", normalization="jump"):
    """Static (t = 0) diagnostics of one ``delta_s`` normalisation."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature=curvature, csf_delta_s=normalization)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0 * h))
    state = csf_fields(pos, mass, wide, h, rho0, curvature_scheme=curvature,
                       delta_s_normalization=normalization)
    acc, _ = surface_tension_acceleration(pos, rho, mass, wide, h, sigma,
                                          rho0=rho0, curvature_scheme=curvature,
                                          delta_s_normalization=normalization)

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    mag = torch.linalg.vector_norm(acc, dim=1)
    active = state.delta_s > 0.05 * state.delta_s.max()
    cos_in = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)
    band = active & (r / radius > 0.6) & (r / radius <= 1.02)

    # Two bookkeepings of the same cut-force identity.  ``delta_s`` is a
    # *geometric* surface density, so the volume element must be the geometric one
    # (m / rho0, the lattice volume the particle owns); that is the primary
    # criterion.  Summing m * a instead uses V_i = m / rho_i, and at a free
    # surface rho_i is under-counted by ~20 %, which inflates the integral by the
    # same factor -- reported as a diagnostic, not as the criterion.
    right = pos[:, 0] > 0
    cut = float((mass[right, None] * acc[right]).sum(dim=0)[0])
    cut_geometric = float((mass[right] * rho[right] / rho0 * acc[right, 0]).sum())
    theory_cut = -2.0 * sigma
    theory_dp = sigma / radius

    # the identity the measured-jump fix rests on
    grad_mag = torch.linalg.vector_norm(state.grad_c, dim=1)
    dv = mass / rho0
    shell = (r > 0.6 * radius) & (r < 1.05 * radius)
    grad_integral = float(((grad_mag * dv * shell)
                           / (2.0 * math.pi * r.clamp_min(1e-30))).sum())
    c_surf = float(torch.quantile(state.color[active], DELTA_S_QUANTILE))
    rho_w = float((state.delta_s * rho).sum() / state.delta_s.sum()) / rho0
    w = state.delta_s
    kappa_w = float((state.kappa.abs() * w).sum() / w.sum())
    kappa_exact = 1.0 / radius
    dsi = delta_s_integral(pos, mass, rho0, state.delta_s, radius)

    def verdict(err):
        return {"within_3pct": abs(err) < 0.03, "within_15pct": abs(err) < 0.15}

    dsi_err = dsi - 1.0
    cut_rel = cut / theory_cut - 1.0
    kappa_rel = (kappa_w - kappa_exact) / kappa_exact
    return {
        "normalization": normalization,
        "curvature": curvature,
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "color": {
            "c_bulk_interior": float(state.color[r < 0.5 * radius].mean()),
            "c_surf_q10": c_surf,
            "assumed_by_fixed_scheme": 0.5,
            "span_measured": 1.0 - c_surf,
            "grad_integral_in_fluid_dn": grad_integral,
            "grad_integral_rel_error_pct": (grad_integral / (1.0 - c_surf) - 1.0) * 100.0,
        },
        "interface_density": {
            "delta_s_weighted_rho_over_rho0": rho_w,
            "deficit_pct": (1.0 - rho_w) * 100.0,
        },
        "delta_s": {
            "integral_dn": dsi,
            "rel_error_pct": dsi_err * 100.0,
            **verdict(dsi_err),
        },
        "cut_force": {
            "theory_N_per_m": theory_cut,
            "measured_geometric_N_per_m": cut_geometric,
            "ratio_geometric": cut_geometric / theory_cut,
            "geometric_rel_error_pct": abs(cut_geometric / theory_cut - 1.0) * 100.0,
            **{f"geometric_{k}": v for k, v in
               verdict(cut_geometric / theory_cut - 1.0).items()},
            # acceleration-space bookkeeping (V_i = m/rho_i): inflated by the
            # free-surface density deficit, kept for continuity with earlier
            # reports.
            "measured_acceleration_space_N_per_m": cut,
            "ratio_acceleration_space": cut / theory_cut,
            "acceleration_space_rel_error_pct": abs(cut_rel) * 100.0,
        },
        "curvature_kappa": {
            "exact_1_per_m": kappa_exact,
            "delta_s_weighted_mean_1_per_m": kappa_w,
            "rel_error_pct": kappa_rel * 100.0,
            **verdict(kappa_rel),
        },
        "direction": {
            "n_active_band": int(band.sum()),
            "cos_band_median": float(cos_in[band].median()),
            "fraction_inward": float((cos_in[band] > 0).float().mean()),
        },
        "reference": {"laplace_dp_Pa": theory_dp,
                      "capillary_velocity_m_per_s": math.sqrt(sigma / (rho0 * radius))},
    }


def sweep(dxs=(0.003, 0.002, 0.0015), **kwargs):
    rows = []
    for dx in dxs:
        for norm in ("fixed", "jump"):
            rows.append(evaluate(dx=dx, normalization=norm, **kwargs))
    return rows


def make_figures(payload, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    dxs = sorted({r["dx_m"] for r in rows})
    colors = {"fixed": "tab:gray", "jump": "tab:blue"}
    labels = {"fixed": r"fixed: $\delta_s=2|\nabla c|$ (assumes $c_{surf}=1/2$)",
              "jump": r"jump: $\delta_s=|\nabla c|/(1-c_{surf})$"}
    panels = [("delta_s", "rel_error_pct", r"$\int \delta_s\,dn$ vs 1  [%]", 0.0, 1.0),
              ("cut_force", "ratio_geometric", r"$F_{cut}/(2\sigma)$ (geometric volume)", 1.0, 0.0),
              ("curvature_kappa", "rel_error_pct", r"$\kappa$ vs $1/R$  [%]", 0.0, 0.0)]
    for ax, (key, field, title, ref, _) in zip(axes, panels):
        for norm in ("fixed", "jump"):
            xs, ys = [], []
            for dx in dxs:
                hit = [r for r in rows if r["dx_m"] == dx and r["normalization"] == norm]
                if hit:
                    xs.append(dx * 1000.0)
                    ys.append(hit[0][key][field])
            ax.plot(xs, ys, "o-", color=colors[norm], label=labels[norm])
        ax.axhline(15.0 if ref == 0.0 and "rel_error" in field else ref,
                   color="tab:red", ls="--", lw=1, label="15 % gate" if "rel_error" in field else "analytic")
        if "rel_error" in field:
            ax.axhline(-15.0, color="tab:red", ls="--", lw=1)
        ax.invert_xaxis()
        ax.set_xlabel("dx [mm]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("CSF surface delta function: fixed factor 2 vs measured colour jump")
    fig.tight_layout()
    out = Path(out_dir) / "delta_s_normalization.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    rows = sweep(radius=args.radius, sigma=args.sigma)
    payload = {
        "setup": {"radius_m": args.radius, "sigma_N_per_m": args.sigma,
                  "rho0": 1000.0, "h_over_dx": 1.3, "lattice": "hexagonal",
                  "quantile": DELTA_S_QUANTILE, "gravity": 0.0,
                  "curvature_operator": "fit_quadratic"},
        "rows": rows,
    }
    (HERE / "results.json").write_text(json.dumps(payload, indent=2))
    for r in rows:
        print(f"dx={r['dx_m']:.4f} {r['normalization']:5s} "
              f"int_ds={r['delta_s']['integral_dn']:.3f} "
              f"Fcut_geo={r['cut_force']['ratio_geometric']:.3f} "
              f"Fcut_acc={r['cut_force']['ratio_acceleration_space']:.3f} "
              f"kappa={r['curvature_kappa']['rel_error_pct']:+.1f}% "
              f"cos={r['direction']['cos_band_median']:.3f} "
              f"c_surf={r['color']['c_surf_q10']:.4f}")
    if args.figure:
        print("figure:", make_figures(payload))
    return payload


if __name__ == "__main__":
    main()