"""Candidate curvature estimators scored against the analytic circle kappa=1/R."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import torch
from tensorsph.csf_consistent import csf_fields
from tensorsph.droplet_impact import Config, cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("law", ROOT / "benchmarks/laplace_law/run.py")
law = importlib.util.module_from_spec(spec)
spec.loader.exec_module(law)

R, SIGMA, RHO0 = 0.02, 7.2, 1000.0


def parabola_fit_kappa(pos, pairs, h, normal, mask):
    """Local quadratic fit z = a s^2 + b s in the (t,n) frame; kappa = -2a/h.

    Conditioning: coordinates are shifted to the particle and divided by h, so
    the normal matrix is O(1) instead of O(R^-4).
    """
    i, j = pairs.unbind(dim=1)
    n = pos.shape[0]
    out = torch.zeros(n, dtype=pos.dtype)
    # build directed list with kernel weight keyed on src
    src = torch.cat([i, j])
    dst = torch.cat([j, i])
    keep = mask[src] & mask[dst]
    src, dst = src[keep], dst[keep]
    d = pos[dst] - pos[src]
    nn = normal[src]
    tt = torch.stack([-nn[:, 1], nn[:, 0]], dim=1)
    s = (d * tt).sum(1) / h
    z = (d * nn).sum(1) / h
    w = cubic_spline_kernel((d * d).sum(1).sqrt(), h) * 0.5
    A = torch.zeros(n, 2, 2, dtype=pos.dtype)
    b = torch.zeros(n, 2, dtype=pos.dtype)
    feats = torch.stack([s * s, s], dim=1)
    A.index_add_(0, src, w[:, None, None] * feats[:, :, None] * feats[:, None, :])
    b.index_add_(0, src, (w * z)[:, None] * feats)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    ok = det.abs() > 1e-18
    a = torch.where(ok, (b[:, 0] * A[:, 1, 1] - A[:, 0, 1] * b[:, 1]) / det.clamp_min(1e-30),
                    torch.zeros_like(det))
    out = torch.where(ok, -2.0 * a / h, out)
    return out


def run(dx):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0, gravity=0.0,
                 viscosity=0.01, surface_tension_coefficient=SIGMA, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    pos, mass = law.hexagonal_droplet(R, dx, RHO0)
    pos, mass = pos.double(), mass.double()
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st = csf_fields(pos, mass, pairs, h, RHO0)
    r = torch.linalg.vector_norm(pos, 1)
    band = st.delta_s > 0.05 * float(st.delta_s.max())
    kf = parabola_fit_kappa(pos, pairs, h, st.normal, band)
    print(f"\n=== dx={dx} N={pos.shape[0]} band={int(band.sum())} exact={1/R}")
    print("  layer        n   divn  rel%  |  par-fit rel%  |  norm err deg")
    edges = [0.80, 0.88, 0.92, 0.95, 0.97, 0.99, 1.01, 1.10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = band & (r / R > lo) & (r / R <= hi)
        if int(m.sum()) < 3:
            continue
        km = float(st.kappa[m].mean())
        kfm = float(kf[m].mean())
        nerr = float(torch.rad2deg(torch.acos(
            torch.clamp(((st.normal[m] * (pos[m] / r[m][:, None])).sum(1)), -1, 1))).mean())
        print(f"  [{lo:.2f},{hi:.2f}) {int(m.sum()):4d} {km:7.1f} {100*(km/(1/R)-1):+6.1f} "
              f"| {kfm:7.1f} {100*(kfm/(1/R)-1):+7.1f} | {nerr:5.2f}")
    w = st.delta_s[band]
    km = float((st.kappa[band] * w).sum() / w.sum())
    kfm = float((kf[band] * w).sum() / w.sum())
    print(f"  delta_s-weighted: div n {100*(km/(1/R)-1):+.1f}%   par-fit {100*(kfm/(1/R)-1):+.1f}%")


for d in (0.003, 0.002, 0.0015, 0.001):
    run(d)
