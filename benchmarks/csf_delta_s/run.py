"""delta_s normalisation: gradient operator vs analytic-profile inversion.

The verification record left one CSF debt open: the surface delta function,
scored as ``int delta_s dn``, read 0.74-0.76 instead of 1 on a circular drop and
did not improve with resolution (docs/verification_surface_tension_curvature.md
§7.1).  Three field-side remedies were proposed in that handoff (ghost colour
extension, Shepard normalisation of |grad c|, narrower colour kernel).  This
harness replaces guesswork with a dissection:

1. **field or operator?**  Build the *exact* mollified indicator of the circle by
   polar quadrature (``c`` = kernel mass fraction inside the drop) and run the
   same scoring pipeline on it.  If the exact field scores like the kernel-sum
   field, the colour function is not the problem.
2. **quadrature or gradient?**  Evaluate the *analytic* gradient magnitude of the
   mollified indicator at every particle and integrate it with the same shell
   quadrature.  If that lands on 1, the shell quadrature is fine and the bias is
   entirely in the lattice derivative.
3. **the three proposed remedies**, re-run here so the ledger is settled by
   numbers rather than by opinion.
4. **a fixed estimator**: ``tensorsph.csf_delta_s.profile_delta_s`` inverts the
   analytic indicator profile and cannot lose normalisation by construction.

Scoring uses the same static-droplet protocol as ``benchmarks/csf_curvature_fit``
(hex lattice, ``rho`` from neighbour summation, force from
``surface_tension_acceleration``), so the numbers are comparable to the
committed ``docs/verification_surface_tension_curvature.md`` tables.

Usage::

    PYTHONPATH=src python benchmarks/csf_delta_s/run.py            # JSON + png
    PYTHONPATH=src python benchmarks/csf_delta_s/run.py --no-plot
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import (
    _mls_operators,
    csf_fields,
    kernel_sum_color,
    mls_gradient,
    surface_tension_acceleration,
)
from tensorsph.csf_delta_s import kernel_profile, profile_delta_s
from tensorsph.droplet_impact import Config, cubic_spline_kernel, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R_DROP = 0.02
RHO0 = 1000.0
SIGMA = 7.2
DX_LADDER = (0.003, 0.002, 0.0015)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load("laplace_law_run", "benchmarks/laplace_law/run.py")
_consistent = _load("csf_consistent_run", "benchmarks/csf_consistent/run.py")
hexagonal_droplet = _law.hexagonal_droplet
delta_s_integral = _consistent.delta_s_integral


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #
def exact_indicator(pos: torch.Tensor, radius: float, h: float,
                    n_r: int = 160, n_t: int = 240) -> torch.Tensor:
    """Kernel mass fraction inside the drop, by polar quadrature (exact ``c``)."""
    rr = torch.linspace(1e-9, 2.0 * h, n_r, dtype=pos.dtype)
    th = torch.linspace(0.0, 2.0 * math.pi, n_t + 1, dtype=pos.dtype)[:-1]
    dr = rr[1] - rr[0]
    dth = th[1] - th[0]
    jac = (rr * dr)[:, None] * dth
    w = cubic_spline_kernel(rr, h)[:, None] * jac
    dx = rr[:, None] * torch.cos(th)[None, :]
    dy = rr[:, None] * torch.sin(th)[None, :]
    out = torch.zeros(pos.shape[0], dtype=pos.dtype)
    for s in range(0, pos.shape[0], 64):
        blk = pos[s:s + 64]
        px = blk[:, 0, None, None] + dx[None]
        py = blk[:, 1, None, None] + dy[None]
        inside = (px * px + py * py) <= radius * radius
        out[s:s + 64] = (w[None] * inside.to(pos.dtype)).sum(dim=(1, 2))
    return out.clamp(0.0, 1.0)


def analytic_gradient(pos: torch.Tensor, radius: float, h: float,
                      n: int = 401) -> torch.Tensor:
    """``|grad c|`` of the mollified indicator at each particle (planar limit)."""
    r = torch.linalg.vector_norm(pos, dim=1)
    d = (radius - r).clamp_min(0.0)                      # distance into the fluid
    y = torch.linspace(-2.02 * h, 2.02 * h, n, dtype=pos.dtype)
    dy = y[1] - y[0]
    rad = torch.sqrt(d[:, None] ** 2 + y[None, :] ** 2)
    return (cubic_spline_kernel(rad, h) * dy).sum(dim=1)


# --------------------------------------------------------------------------- #
# scoring (same protocol as benchmarks/csf_curvature_fit)
# --------------------------------------------------------------------------- #
def score(radius: float = R_DROP, dx: float = 0.002, sigma: float = SIGMA,
          rho0: float = RHO0, curvature: str = "fit_quadratic",
          delta_s_scheme: str = "gradient", grad_scale: float = 1.0,
          with_analytic: bool = False) -> dict:
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01, surface_tension_coefficient=sigma,
                 csf_scheme="consistent", csf_curvature=curvature,
                 csf_delta_s=delta_s_scheme)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    near = cell_linked_pairs(pos, 2.0 * h)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, near)

    state = csf_fields(pos, mass, wide, h, rho0, curvature_scheme=curvature,
                       delta_s_scheme=delta_s_scheme)
    acc, state2 = surface_tension_acceleration(
        pos, rho, mass, wide, h, sigma, rho0=rho0, curvature_scheme=curvature,
        delta_s_scheme=delta_s_scheme)
    acc = acc * grad_scale
    delta_s = state2.delta_s
    kappa = state2.kappa

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    active = delta_s > 0.05 * delta_s.max()
    mag = torch.linalg.vector_norm(acc, dim=1)
    cos_in = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)
    band = active & (r / radius > 0.6) & (r / radius <= 1.02)
    cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
    theory_cut = -2.0 * sigma
    kappa_w = float((kappa.abs() * delta_s).sum() / delta_s.sum())
    dsi = delta_s_integral(pos, mass, rho0, delta_s, radius)
    kappa_exact = 1.0 / radius

    out = {
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "r_over_h": radius / h,
        "curvature": curvature,
        "delta_s_scheme": delta_s_scheme,
        "kappa": {
            "exact_1_per_m": kappa_exact,
            "delta_s_weighted_mean_1_per_m": kappa_w,
            "rel_error_pct": (kappa_w - kappa_exact) / kappa_exact * 100.0,
            "within_15pct": abs(kappa_w - kappa_exact) / kappa_exact < 0.15,
        },
        "delta_s": {
            "integral_dn": dsi,
            "rel_error_pct": (dsi - 1.0) * 100.0,
            "within_15pct": abs(dsi - 1.0) < 0.15,
            "within_3pct": abs(dsi - 1.0) < 0.03,
        },
        "cut_force": {
            "theory_N_per_m": theory_cut,
            "measured_N_per_m": cut,
            "ratio_measured_over_theory": cut / theory_cut,
            "rel_error_pct": abs(cut / theory_cut - 1.0) * 100.0,
            "within_15pct": abs(cut / theory_cut - 1.0) < 0.15,
        },
        "direction": {
            "cos_band_median": float(cos_in[band].median()),
            "fraction_inward": float((cos_in[band] > 0).float().mean()),
        },
    }

    if with_analytic:
        # (1) the exact indicator field, scored by the *gradient* estimator
        c_exact = exact_indicator(pos, radius, h)
        vol = mass / rho0
        g_ex = mls_gradient(pos, c_exact, vol, near, h)
        ds_ex_field = 2.0 * torch.linalg.vector_norm(g_ex, dim=1)
        # (2) the analytic gradient magnitude, scored by the *same* quadrature
        g_an = analytic_gradient(pos, radius, h)
        ds_ex_grad = 2.0 * g_an
        # (3) the profile estimator applied to the exact field
        ds_prof_exact = profile_delta_s(c_exact, h)
        out["analytic"] = {
            "color_rms_kernelsum_vs_exact": float(
                torch.sqrt(((state.color - c_exact) ** 2).mean())),
            "integral_gradient_on_exact_field": delta_s_integral(
                pos, mass, rho0, ds_ex_field, radius),
            "integral_analytic_gradient": delta_s_integral(
                pos, mass, rho0, ds_ex_grad, radius),
            "integral_profile_on_exact_field": delta_s_integral(
                pos, mass, rho0, ds_prof_exact, radius),
            "gradient_ratio_mls_over_analytic": float(
                ds_ex_field[active].sum() / ds_ex_grad[active].sum()),
        }
    return out


def tabulation_control(h: float = 1.3 * 0.002) -> dict:
    """The tabulated indicator profile must integrate to the full jump."""
    d, c, dc_dd = kernel_profile(h)
    dr = d[1] - d[0]
    steps = 0.5 * (dc_dd[1:] + dc_dd[:-1]) * dr
    return {
        "c_at_interface": float(c[0]),
        "c_at_support_edge": float(c[-1]),
        "int_dc_dd_dd": float(steps.sum()),
        "integral_delta_s_dn": float(2.0 * steps.sum()),
        "dc_dd_peak_per_m": float(dc_dd[0]),
        "h_m": h,
    }


# --------------------------------------------------------------------------- #
# the three field-side remedies proposed by the previous handoff
# --------------------------------------------------------------------------- #
def remedy_narrow_colour(radius: float, dx: float) -> dict:
    """Colour function and gradient both computed with h_c = beta h."""
    h = 1.3 * dx
    pos, mass = hexagonal_droplet(radius, dx, RHO0)
    vol = mass / RHO0
    out = {}
    for beta in (1.0, 0.85, 0.7, 0.6):
        hc = beta * h
        pairs = cell_linked_pairs(pos, 2.0 * hc)
        c = kernel_sum_color(pos, mass, pairs, hc, RHO0)
        g = mls_gradient(pos, c, vol, pairs, hc)
        ds = 2.0 * torch.linalg.vector_norm(g, dim=1)
        out[f"{beta:g}"] = delta_s_integral(pos, mass, RHO0, ds, radius)
    return out


def remedy_tikhonov(radius: float, dx: float) -> dict:
    """Regularised MLS gradient ((M^T M + lam tr(M^T M) I)^-1 M^T r)."""
    h = 1.3 * dx
    pos, mass = hexagonal_droplet(radius, dx, RHO0)
    vol = mass / RHO0
    pairs = cell_linked_pairs(pos, 2.0 * h)
    c = kernel_sum_color(pos, mass, pairs, h, RHO0)
    i, j = pairs.unbind(dim=1)
    from tensorsph.droplet_impact import cubic_spline_gradient
    gw = cubic_spline_gradient(pos[i] - pos[j], h)
    dc = c[j] - c[i]
    rhs = torch.zeros((pos.shape[0], 2), dtype=pos.dtype)
    rhs = rhs.index_add(0, i, vol[j, None] * dc[:, None] * gw)
    rhs = rhs.index_add(0, j, vol[i, None] * dc[:, None] * gw)
    big_m, _ = _mls_operators(pos, vol, pairs, h)
    mt = big_m.transpose(-1, -2)
    a = mt @ big_m
    tr = a.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(2, dtype=pos.dtype)
    out = {}
    for lam in (1e-3, 1e-2):
        reg = a + (lam * tr.clamp_min(1e-30))[:, None, None] * eye
        g = torch.linalg.solve(reg, (mt @ rhs.unsqueeze(-1))).squeeze(-1)
        ds = 2.0 * torch.linalg.vector_norm(g, dim=1)
        out[f"{lam:g}"] = delta_s_integral(pos, mass, RHO0, ds, radius)
    return out


def remedy_ghost(radius: float, dx: float) -> dict:
    """Mirror the interface shell across the interface (antisymmetric colour)."""
    h = 1.3 * dx
    pos, mass = hexagonal_droplet(radius, dx, RHO0)
    vol = mass / RHO0
    pairs = cell_linked_pairs(pos, 2.0 * h)
    c = kernel_sum_color(pos, mass, pairs, h, RHO0)
    g = mls_gradient(pos, c, vol, pairs, h)
    mag = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-30)
    n = -g / mag
    s = (c - 0.5) / mag.squeeze(-1)
    shell = (c < 0.995) & (s > 0.0) & (s < 2.0 * h)
    src = torch.nonzero(shell).squeeze(-1)
    gpos = pos[src] + 2.0 * s[src, None] * n[src]
    allpos = torch.cat([pos, gpos])
    allc = torch.cat([c, (1.0 - c[src]).clamp_min(0.0)])
    allvol = torch.cat([vol, vol[src]])
    allpairs = cell_linked_pairs(allpos, 2.0 * h)
    gg = mls_gradient(allpos, allc, allvol, allpairs, h)
    ds = torch.linalg.vector_norm(gg[: pos.shape[0]], dim=1)   # full profile
    return {
        "n_ghosts": int(src.numel()),
        "integral_dn": delta_s_integral(pos, mass, RHO0, ds, radius),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ladder = report["resolution_ladder"]
    dxs = [r["dx_m"] * 1000 for r in ladder]
    for scheme, style in (("gradient", dict(marker="o", ls="--", color="C3")),
                          ("profile", dict(marker="s", ls="-", color="C0"))):
        sel = [r for r in ladder if r["delta_s_scheme"] == scheme]
        ax[0].plot([r["dx_m"] * 1000 for r in sel],
                   [r["delta_s"]["integral_dn"] for r in sel],
                   label=f"delta_s = {scheme}", **style)
        ax[1].plot([r["dx_m"] * 1000 for r in sel],
                   [r["cut_force"]["ratio_measured_over_theory"] for r in sel],
                   label=f"delta_s = {scheme}", **style)
    ax[0].axhline(1.0, color="k", lw=0.8)
    ax[0].axhspan(0.85, 1.15, color="g", alpha=0.08)
    ax[0].set_xlabel("dx [mm]")
    ax[0].set_ylabel(r"$\int \delta_s\,dn$  (target 1)")
    ax[0].set_title("delta-function normalisation")
    ax[0].invert_xaxis()
    ax[0].legend(fontsize=8)

    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].axhspan(0.85, 1.15, color="g", alpha=0.08)
    ax[1].set_xlabel("dx [mm]")
    ax[1].set_ylabel(r"$F_{cut} / 2\sigma$")
    ax[1].set_title("cut-force identity")
    ax[1].invert_xaxis()
    ax[1].legend(fontsize=8)

    rem = report["remedies"]
    labels = ["gradient\n(base)"] + [f"narrow\nh_c={k}h" for k in
                                     report["remedies"]["narrow_colour"]] + \
             [f"Tikhonov\nlam={k}" for k in report["remedies"]["tikhonov"]] + \
             ["ghost\ncolour", "profile\n(this)"]
    vals = [report["resolution_ladder"][0]["delta_s"]["integral_dn"]] + \
           list(report["remedies"]["narrow_colour"].values()) + \
           list(report["remedies"]["tikhonov"].values()) + \
           [report["remedies"]["ghost"]["integral_dn"],
            report["remedies"]["profile"]["integral_dn"]]
    colors = ["C3"] + ["C7"] * (len(vals) - 2) + ["C0"]
    ax[2].bar(range(len(vals)), vals, color=colors)
    ax[2].set_xticks(range(len(vals)))
    ax[2].set_xticklabels(labels, fontsize=6, rotation=90)
    ax[2].axhline(1.0, color="k", lw=0.8)
    ax[2].axhspan(0.85, 1.15, color="g", alpha=0.08)
    ax[2].set_ylabel(r"$\int \delta_s\,dn$")
    ax[2].set_title(f"proposed remedies (dx={dxs[0]:.1f} mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_field(report: dict, path: Path) -> dict:
    """Shell field of delta_s for both estimators, with the analytic reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = 0.002
    h = 1.3 * dx
    pos, mass = hexagonal_droplet(R_DROP, dx, RHO0)
    near = cell_linked_pairs(pos, 2.0 * h)
    wide = cell_linked_pairs(pos, 4.0 * h)
    st_g = csf_fields(pos, mass, wide, h, RHO0, curvature_scheme="fit_quadratic",
                      delta_s_scheme="gradient")
    st_p = csf_fields(pos, mass, wide, h, RHO0, curvature_scheme="fit_quadratic",
                      delta_s_scheme="profile")
    g_an = 2.0 * analytic_gradient(pos, R_DROP, h)
    r = torch.linalg.vector_norm(pos, dim=1)
    shell = r > R_DROP - 2.5 * h

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for a, field, title in (
            (ax[0], st_g.delta_s, r"$\delta_s = 2|\nabla_h c|$ (gradient)"),
            (ax[1], st_p.delta_s, r"$\delta_s$ profile inversion"),
            (ax[2], g_an, r"analytic $2|\nabla c|$")):
        sc = a.scatter(pos[shell, 0] * 1e3, pos[shell, 1] * 1e3,
                       c=field[shell], s=26, cmap="viridis")
        a.set_aspect("equal")
        a.set_title(title, fontsize=9)
        a.set_xlabel("x [mm]")
        fig.colorbar(sc, ax=a, fraction=0.046)
    ax[0].set_ylabel("y [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "shell_particles": int(shell.sum()),
        "gradient_mean": float(st_g.delta_s[shell].mean()),
        "profile_mean": float(st_p.delta_s[shell].mean()),
        "analytic_mean": float(g_an[shell].mean()),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    report = {
        "case": "2-D static droplet, hex lattice, Young-Laplace + cut-force identity",
        "setup": {"radius_m": R_DROP, "sigma_N_per_m": SIGMA, "rho0_kg_per_m3": RHO0,
                  "h_factor": 1.3, "curvature_scheme": "fit_quadratic"},
        "reference": {"kappa_1_per_m": 1.0 / R_DROP,
                      "cut_force_N_per_m": -2.0 * SIGMA,
                      "delta_s_integral": 1.0},
        "tabulation": tabulation_control(),
        "resolution_ladder": [],
    }
    for dx in DX_LADDER:
        for scheme in ("gradient", "profile"):
            entry = score(dx=dx, delta_s_scheme=scheme,
                          with_analytic=(scheme == "gradient"))
            report["resolution_ladder"].append(entry)
            print(f"dx={dx:6.4f} {scheme:9s} int ds dn={entry['delta_s']['integral_dn']:.3f} "
                  f"Fcut/2s={entry['cut_force']['ratio_measured_over_theory']:.3f} "
                  f"kappa err={entry['kappa']['rel_error_pct']:+5.1f}% "
                  f"cos={entry['direction']['cos_band_median']:+.3f}")

    dx0 = DX_LADDER[0]
    report["remedies"] = {
        "narrow_colour": remedy_narrow_colour(R_DROP, dx0),
        "tikhonov": remedy_tikhonov(R_DROP, dx0),
        "ghost": remedy_ghost(R_DROP, dx0),
        "profile": {"integral_dn": next(
            r["delta_s"]["integral_dn"] for r in report["resolution_ladder"]
            if r["dx_m"] == dx0 and r["delta_s_scheme"] == "profile")},
    }
    print("\nremedies at dx={:.4f}: narrow={} tikhonov={} ghost={:.3f} profile={:.3f}"
          .format(dx0, {k: round(v, 3) for k, v in report["remedies"]["narrow_colour"].items()},
                  {k: round(v, 3) for k, v in report["remedies"]["tikhonov"].items()},
                  report["remedies"]["ghost"]["integral_dn"],
                  report["remedies"]["profile"]["integral_dn"]))

    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2))

    if not args.no_plot:
        report["field"] = plot_field(report, HERE / "delta_s_field.png")
        plot(report, HERE / "csf_delta_s.png")
        (HERE / "validation_report.json").write_text(json.dumps(report, indent=2))
        print("\nfield means (shell):", {k: round(v, 1) for k, v in
                                         report["field"].items()})
    print("\nwrote", HERE / "validation_report.json")


if __name__ == "__main__":
    main()