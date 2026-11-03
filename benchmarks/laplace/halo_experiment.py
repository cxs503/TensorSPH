"""Root-cause experiment for docs/benchmarks/laplace-verification.md section 5.

The merged CSF colour field has no exterior phase, so |grad c| integrates to
~0.4 instead of 1 across a free surface. This script adds a three-layer vacuum
halo with c = 0 and recomputes the same gradient, normal and curvature with the
production helpers.

Measured: the colour integral reaches 1.04 at R/dx = 20 (from 0.306), which
confirms defect 1; but kappa*R then comes out ~-10.8, because the normal field
is clipped to zero inside the liquid and in the far halo, so div(n) integrates
the edges of the band rather than the curvature (defect 2). Both numbers are
quoted in the document.

Not part of the routine gate: no pass/fail, this is a diagnostic.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
import torch
from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    compute_color_gradient, compute_normal, compute_curvature)

torch.set_default_dtype(torch.float64)
R, SIGMA = 0.05, 0.072


def latt(lo, hi, dx, off=0.37):
    xs = torch.arange(lo, hi, dx) + off * dx
    ys = torch.arange(lo, hi, dx) + off * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


for levels in (10, 20, 40):
    dx = R / levels
    cfg = Config(dx=dx, h_factor=1.3, rho0=1000.0, sound_speed=20.0,
                 surface_tension_coefficient=SIGMA, gravity=0.0)
    h = cfg.h_factor * dx
    pts = latt(-R - 3 * dx, R + 3 * dx, dx)
    r = torch.linalg.vector_norm(pts, dim=-1)
    fluid = pts[r <= R]
    halo = pts[(r > R) & (r <= R + 3 * dx)]
    pos = torch.cat((fluid, halo))
    nf = fluid.shape[0]
    # colour: 1 inside the liquid, 0 in the vacuum halo
    color = torch.cat((torch.ones(nf), torch.zeros(halo.shape[0])))
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx)
    rho = torch.full((pos.shape[0],), cfg.rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    g = compute_color_gradient(pos, color, mass, rho, pairs, h)
    nm = compute_normal(-g)
    kappa = compute_curvature(pos, nm, mass, rho, pairs, h)
    rr = torch.linalg.vector_norm(pos[:nf], dim=-1)
    gmag = torch.linalg.vector_norm(g, dim=-1)[:nf]
    band = (rr > 0.93 * R)
    edges = torch.tensor([0.8, 0.9, 0.95, 1.0]) * R
    prof = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (rr >= e0) & (rr < e1) & (gmag > 0)
        if m.any():
            prof.append(f"kappa*R[@r/R={float(0.5*(e0+e1)/R):.2f}]={float(kappa[:nf][m].mean()*R):+.3f}")
    print(f"halo prototype R/dx={levels}: N_fluid={nf} N_halo={halo.shape[0]} "
          f"int|gradc|dr={float(gmag[band].mean()*dx*3):.2f} "
          f"kappaR(band mean)={float(kappa[:nf][band].mean()*R):+.3f} " + " ".join(prof))
