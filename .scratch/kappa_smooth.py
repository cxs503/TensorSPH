"""Curvature-postprocessing variants, scored on the analytic circle.

kappa is a smooth quantity along the interface; the inner/outer edges of the
2h shell carry poorly-estimated normals (c close to 1 or below 0.5) yet they
still weigh in the delta_s-weighted average.  Variants:

  V0 shipped                : kappa = div n
  V1 delta_s-weighted smooth, 1 pass
  V2 delta_s-weighted smooth, 2 passes
  V3 delta_s^2-weighted smooth, 1 pass
  V4 delta_s-weighted smooth, 1 pass, band 0.25<c<0.75 only

Also reports the two quantities that must NOT degrade: cut-force identity
F_cut / 2 sigma and the inward-force fraction.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from tensorsph.csf_consistent import csf_fields
from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("law", ROOT / "benchmarks/laplace_law/run.py")
law = importlib.util.module_from_spec(spec)
spec.loader.exec_module(law)
R, SIGMA, RHO0 = 0.02, 7.2, 1000.0


def smooth_kernel(field, weight, pos, pairs, h, passes=1, band=None):
    """Weighted kernel smoothing restricted to band particles as donors."""
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel(torch.linalg.vector_norm(pos[i] - pos[j], dim=1), h)
    out = field
    for _ in range(passes):
        num = torch.zeros_like(out)
        den = torch.zeros_like(out)
        if band is not None:
            ww = torch.where(band[j] & band[i], w * weight[j], torch.zeros_like(w))
        else:
            ww = w * weight[j]
        num = num.index_add(0, i, ww * out[j]).index_add(0, j, ww * out[i]) * 0.5
        den = den.index_add(0, i, ww).index_add(0, j, ww) * 0.5
        out = torch.where(den > 0, num / den.clamp_min(1e-30), out)
    return out


for dx in (0.003, 0.002, 0.0015, 0.001):
    h = 1.3 * dx
    pos, mass = law.hexagonal_droplet(R, dx, RHO0)
    pos, mass = pos.double(), mass.double()
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st = csf_fields(pos, mass, pairs, h, RHO0)
    r = torch.linalg.vector_norm(pos, dim=1)
    band = st.delta_s > 0.05 * float(st.delta_s.max())
    w = st.delta_s[band]

    def rep(kappa, tag):
        km = float((kappa[band] * w).sum() / w.sum())
        # cut force identity: sum m a over x>0 half, a = -sigma kappa ds n / rho0
        acc = (-SIGMA * kappa * st.delta_s)[:, None] * st.normal / RHO0
        fx = float((mass * acc[:, 0] * (pos[:, 0] > 0)).sum())
        cosb = float((acc * -st.normal).sum(1)[band].median())
        print(f"    {tag:34s} kappa {km:7.2f} ({100*(km/(1/R)-1):+6.1f}%)  "
              f"Fcut/2sig {fx/-14.4:6.3f}  cos {cosb:+.3f}")

    print(f"\n=== dx={dx} N={pos.shape[0]} exact=50")
    rep(st.kappa, "V0 shipped")
    rep(smooth_kernel(st.kappa, st.delta_s, pos, pairs, h, 1), "V1 ds-wtd smooth x1")
    rep(smooth_kernel(st.kappa, st.delta_s, pos, pairs, h, 2), "V2 ds-wtd smooth x2")
    rep(smooth_kernel(st.kappa, st.delta_s**2, pos, pairs, h, 1), "V3 ds^2-wtd smooth x1")
    core = band & (st.color > 0.25) & (st.color < 0.75)
    rep(smooth_kernel(st.kappa, st.delta_s, pos, pairs, h, 1, band=core),
        "V4 ds-wtd x1 on 0.25<c<0.75")
    rep(smooth_kernel(st.kappa, st.delta_s**2, pos, pairs, h, 2, band=core),
        "V5 ds^2 x2 on 0.25<c<0.75")
