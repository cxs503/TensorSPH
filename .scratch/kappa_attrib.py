"""Attribution: is the +22 % kappa bias in the MLS divergence operator itself
or in the estimated normal field?  Control = feed the exact radial normal."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from tensorsph.csf_consistent import csf_fields, mls_divergence, shepard_smooth
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("law", ROOT / "benchmarks/laplace_law/run.py")
law = importlib.util.module_from_spec(spec)
spec.loader.exec_module(law)
R, SIGMA, RHO0 = 0.02, 7.2, 1000.0

def score(kappa, mask, ds, label):
    w = ds[mask]
    km = float((kappa[mask] * w).sum() / w.sum())
    print(f"    {label:34s} delta_s-wtd {km:8.2f}   rel {100*(km/(1/R)-1):+7.1f}%")

for dx in (0.003, 0.002, 0.0015, 0.001):
    h = 1.3 * dx
    pos, mass = law.hexagonal_droplet(R, dx, RHO0)
    pos, mass = pos.double(), mass.double()
    pairs = cell_linked_pairs(pos, 2.0 * h)
    vol = mass / RHO0
    st = csf_fields(pos, mass, pairs, h, RHO0)
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    band = st.delta_s > 0.05 * float(st.delta_s.max())
    print(f"\n=== dx={dx} N={pos.shape[0]} exact kappa={1/R}")
    score(st.kappa, band, st.delta_s, "estimated normal (as shipped)")
    k_exact = mls_divergence(pos, rhat, vol, pairs, h)
    score(k_exact, band, st.delta_s, "exact radial normal")
    sm = shepard_smooth(rhat, pos, vol, pairs, h, passes=2)
    sm = sm / torch.linalg.vector_norm(sm, dim=1, keepdim=True).clamp_min(1e-30)
    score(mls_divergence(pos, sm, vol, pairs, h), band, st.delta_s, "smoothed radial normal")
    ang = torch.rad2deg(torch.acos(torch.clamp((st.normal[band] * rhat[band]).sum(1), -1, 1)))
    print(f"    normal angle error on band: mean {float(ang.mean()):.2f} deg  "
          f"p90 {float(ang.quantile(0.9)):.2f} deg  max {float(ang.max()):.2f}")
    # how much of the band is interior (where c saturated)?
    print(f"    band particles: {int(band.sum())}  of which c>0.99: "
          f"{int((band & (st.color > 0.99)).sum())}  c<0.5: {int((band & (st.color < 0.5)).sum())}")
    for lo, hi in ((0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)):
        m = band & (st.color >= lo) & (st.color < hi)
        if int(m.sum()) < 3:
            continue
        w = st.delta_s[m]
        km = float((st.kappa[m] * w).sum() / w.sum())
        ke = float((k_exact[m] * w).sum() / w.sum())
        print(f"    c in [{lo},{hi}) n={int(m.sum()):4d}  "
              f"shipped {100*(km/(1/R)-1):+7.1f}%   exact-n {100*(ke/(1/R)-1):+7.1f}%")
