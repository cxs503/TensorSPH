"""Scan remedies for the delta_s (surface-delta) normalisation deficit.

Scores each variant against the three analytic references on a static hex drop:
  int delta_s dn = 1,  F_cut = 2 sigma,  kappa = 1/R
"""
import math
import importlib.util
import pathlib

import torch

from tensorsph.csf_consistent import kernel_sum_color, mls_gradient, _mls_operators
from tensorsph.droplet_impact import cubic_spline_kernel, cubic_spline_gradient
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_curvature import fit_curvature

torch.set_num_threads(2)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ll = load("ll", "benchmarks/laplace_law/run.py")
cs = load("cs", "benchmarks/csf_consistent/run.py")

SIGMA = 7.2


def spherically_averaged_grad(pos, c, vol, pairs, h):
    """Plain (uncorrected) symmetric SPH gradient, for A/B."""
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    gw = cubic_spline_gradient(pos[i] - pos[j], h)
    dc = c[j] - c[i]
    g = torch.zeros((n, 2), dtype=pos.dtype)
    g = g.index_add(0, i, vol[j, None] * dc[:, None] * gw)
    g = g.index_add(0, j, vol[i, None] * dc[:, None] * gw)
    return g


def tikh_mls_gradient(pos, c, vol, pairs, h, lam):
    """Tikhonov-regularised MLS gradient: (M^T M + lam tr(M^T M) I)^-1 M^T r."""
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    gw = cubic_spline_gradient(pos[i] - pos[j], h)
    dc = c[j] - c[i]
    rhs = torch.zeros((n, 2), dtype=pos.dtype)
    rhs = rhs.index_add(0, i, vol[j, None] * dc[:, None] * gw)
    rhs = rhs.index_add(0, j, vol[i, None] * dc[:, None] * gw)
    M, _ = _mls_operators(pos, vol, pairs, h)
    Mt = M.transpose(-1, -2)
    A = Mt @ M
    tr = A.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(2, dtype=pos.dtype)
    A = A + (lam * tr.clamp_min(1e-30))[:, None, None] * eye
    return torch.linalg.solve(A, (Mt @ rhs.unsqueeze(-1))).squeeze(-1)


def score(pos, mass, rho0, h, sigma, radius, delta_s, kappa, grad_c, pairs_wide):
    r = torch.linalg.vector_norm(pos, dim=1)
    vol = mass / rho0
    dsi = cs.delta_s_integral(pos, mass, rho0, delta_s, radius)
    # cut force on the x>0 half, with outward normal n = -grad c/|grad c|
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True).clamp_min(1e-30)
    normal = -grad_c / mag
    acc = (-sigma * kappa * delta_s)[:, None] * normal / rho0
    cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
    w = delta_s
    kappa_w = float((kappa.abs() * w).sum() / w.sum()) if float(w.sum()) > 0 else float("nan")
    return {
        "dsi": dsi, "dsi_err_pct": (dsi - 1.0) * 100,
        "cut_ratio": cut / (-2.0 * sigma),
        "kappa": kappa_w, "kappa_err_pct": (kappa_w / (1.0 / radius) - 1.0) * 100,
    }


def run_case(dx, R=0.02):
    rho0 = 1000.0
    h = 1.3 * dx
    pos, mass = ll.hexagonal_droplet(R, dx, rho0)
    vol = mass / rho0
    wide = cell_linked_pairs(pos, 4.0 * h)
    out = {}

    # ---- MLS fallback rate on the outer layer
    _, ok = _mls_operators(pos, vol, cell_linked_pairs(pos, 2.0 * h), h)
    r = torch.linalg.vector_norm(pos, dim=1)
    outer = r > R - dx
    out["mls_fallback_outer_frac"] = float((~ok[outer]).float().mean())

    # ---- baseline: colour with kernel h, MLS grad, kappa = div n
    c = kernel_sum_color(pos, mass, cell_linked_pairs(pos, 2.0 * h), h, rho0)
    g = mls_gradient(pos, c, vol, cell_linked_pairs(pos, 2.0 * h), h)
    n = -g / torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-30)
    # curvature: reuse the module's own consistent field pipeline for kappa
    from tensorsph.csf_consistent import csf_fields
    base = csf_fields(pos, mass, cell_linked_pairs(pos, 2.0 * h), h, rho0)
    st = csf_fields(pos, mass, wide, h, rho0, curvature_scheme="fit_quadratic")
    for tag, ds, kappa, gg in (
        ("baseline(2|grad c|,mls,fit_q)", st.delta_s, st.kappa, st.grad_c),
    ):
        out[tag] = score(pos, mass, rho0, h, SIGMA, R, ds, kappa, gg, wide)

    # ---- variant: colour kernel scaled down by beta, gradient with same beta*h
    for beta in (0.85, 0.7, 0.6):
        hc = beta * h
        pr = cell_linked_pairs(pos, 2.0 * hc)
        c2 = kernel_sum_color(pos, mass, pr, hc, rho0)
        g2 = mls_gradient(pos, c2, vol, pr, hc)
        ds2 = 2.0 * torch.linalg.vector_norm(g2, dim=1)
        k2 = fit_curvature(pos, -g2 / torch.linalg.vector_norm(g2, dim=1,
                                                             keepdim=True).clamp_min(1e-30),
                           ds2, wide, h, fallback=st.kappa, method="fit_quadratic")
        out[f"colour h_c={beta}h"] = score(pos, mass, rho0, h, SIGMA, R, ds2, k2, g2, wide)

    # ---- variant: Tikhonov MLS
    for lam in (1e-3, 1e-2):
        g3 = tikh_mls_gradient(pos, c, vol, cell_linked_pairs(pos, 2.0 * h), h, lam)
        ds3 = 2.0 * torch.linalg.vector_norm(g3, dim=1)
        m3 = torch.linalg.vector_norm(g3, dim=1, keepdim=True).clamp_min(1e-30)
        k3 = fit_curvature(pos, -g3 / m3, ds3, wide, h, fallback=st.kappa,
                           method="fit_quadratic")
        out[f"tikh lam={lam:g}"] = score(pos, mass, rho0, h, SIGMA, R, ds3, k3, g3, wide)

    # ---- variant: plain SPH gradient
    g4 = spherically_averaged_grad(pos, c, vol, cell_linked_pairs(pos, 2.0 * h), h)
    ds4 = 2.0 * torch.linalg.vector_norm(g4, dim=1)
    m4 = torch.linalg.vector_norm(g4, dim=1, keepdim=True).clamp_min(1e-30)
    k4 = fit_curvature(pos, -g4 / m4, ds4, wide, h, fallback=st.kappa,
                       method="fit_quadratic")
    out["plain sph grad"] = score(pos, mass, rho0, h, SIGMA, R, ds4, k4, g4, wide)
    return out, int(pos.shape[0])


for dx in (0.003, 0.002, 0.0015):
    res, n = run_case(dx)
    print(f"\n===== dx={dx}  N={n}  (R/h={0.02/(1.3*dx):.1f}) =====")
    print(f"{'variant':32s} {'int ds dn':>9s} {'err%':>7s} {'Fcut/2s':>8s} "
          f"{'kappa err%':>10s}")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{'mls fallback frac (outer)':32s} {v:9.3f}")
            continue
        print(f"{k:32s} {v['dsi']:9.3f} {v['dsi_err_pct']:7.1f} "
              f"{v['cut_ratio']:8.3f} {v['kappa_err_pct']:10.1f}")