"""Diagnostic: where does the 25-30 % curvature error come from?

Instant-only (no time stepping).  Scores kappa = div n on a static hexagonal
droplet, binned by radial position and by delta_s, against the analytic
kappa = 1/R, and evaluates an independent least-squares circle-fit estimator.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields
from tensorsph.droplet_impact import Config, cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

spec = importlib.util.spec_from_file_location("law", ROOT / "benchmarks/laplace_law/run.py")
law = importlib.util.module_from_spec(spec)
spec.loader.exec_module(law)

R = 0.02
SIGMA = 7.2
RHO0 = 1000.0


def circle_fit_curvature(pos, pairs, h, band_mask):
    """Least-squares circle through each surface particle's neighbourhood.

    For a 2-D curve the algebraic fit  x^2+y^2 + a x + b y + c = 0  has
    curvature kappa = 2 / sqrt(a^2 + b^2 - 4 c).  Weighted by the SPH kernel so
    that the fit is a local, smooth reconstruction.
    """
    i, j = pairs.unbind(dim=1)
    n = pos.shape[0]
    # accumulate normal equations per particle i
    A = torch.zeros(n, 3, 3, dtype=pos.dtype)
    rhs = torch.zeros(n, 3, dtype=pos.dtype)
    w = cubic_spline_kernel(torch.linalg.vector_norm(pos[i] - pos[j], dim=1), h)
    # weight only neighbours (both directions of the symmetric pair list)
    x = pos[:, 0]
    y = pos[:, 1]
    for src, dst, ww in ((i, j, w), (j, i, torch.ones_like(w))):
        rows = torch.stack([x[dst], y[dst], torch.ones_like(x[dst])], dim=1)
        z = -(x[dst] ** 2 + y[dst] ** 2)
        if src is i:
            wv = ww
        else:
            # reverse direction: weight must equal W(|r_i - r_j|) for the new src
            wv = cubic_spline_kernel(
                torch.linalg.vector_norm(pos[src] - pos[dst], dim=1), h)
        A.index_add_(0, src, wv[:, None, None] * rows[:, :, None] * rows[:, None, :])
        rhs.index_add_(0, src, (wv * z)[:, None] * rows)
    eye = torch.eye(3) * 1e-12
    sol = torch.linalg.solve(A + eye[None], rhs[:, :, None])[:, :, 0]
    quad = sol[:, 0] ** 2 + sol[:, 1] ** 2 - 4.0 * sol[:, 2]
    kappa = torch.where(quad > 0, 2.0 / quad.clamp_min(1e-30).sqrt(),
                        torch.zeros_like(quad))
    return kappa


def run(dx: float):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01, surface_tension_coefficient=SIGMA,
                 csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    pos, mass = law.hexagonal_droplet(R, dx, RHO0)
    pos = pos.double()
    mass = mass.double()
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st = csf_fields(pos, mass, pairs, h, RHO0)
    r = torch.linalg.vector_norm(pos, dim=1)
    band = st.delta_s > 0.05 * float(st.delta_s.max())

    kappa_fit = circle_fit_curvature(pos, pairs, h, band)

    print(f"\n=== dx={dx:.4f}  N={pos.shape[0]}  band={int(band.sum())}  exact kappa={1/R:.1f}")
    print("  layer (r/R)      n   kappa_mean  rel%   |  fit_mean  rel%")
    edges = [0.80, 0.88, 0.92, 0.95, 0.97, 0.99, 1.01, 1.03, 1.10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = band & (r / R > lo) & (r / R <= hi)
        if int(m.sum()) < 3:
            continue
        km = float(st.kappa[m].mean())
        kf = float(kappa_fit[m].mean())
        print(f"  [{lo:.2f},{hi:.2f})  {int(m.sum()):4d}  "
              f"{km:9.2f} {100*(km/(1/R)-1):7.1f}  |  {kf:8.2f} {100*(kf/(1/R)-1):7.1f}")

    for name, mask in (("band delta_s>5%", band),
                       ("peak delta_s>50%", st.delta_s > 0.5 * float(st.delta_s.max())),
                       ("outermost r>0.97R", band & (r > 0.97 * R))):
        km = float(st.kappa[mask].mean())
        kf = float(kappa_fit[mask].mean())
        print(f"  {name:20s} n={int(mask.sum()):4d}  div n: {100*(km/(1/R)-1):+7.1f}%   "
              f"circle-fit: {100*(kf/(1/R)-1):+7.1f}%")

    # delta_s-weighted mean (what the benchmark scores)
    w = st.delta_s[band]
    km = float((st.kappa[band] * w).sum() / w.sum())
    kf = float((kappa_fit[band] * w).sum() / w.sum())
    print(f"  delta_s-weighted (scored)   div n: {100*(km/(1/R)-1):+7.1f}%   "
          f"circle-fit: {100*(kf/(1/R)-1):+7.1f}%")


if __name__ == "__main__":
    for d in (0.003, 0.002, 0.0015):
        run(d)
