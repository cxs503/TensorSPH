import sys, math
sys.path.insert(0, "/root/tp-agent/work/m15332/benchmarks/capillary_wave")
sys.path.insert(0, "/root/tp-agent/work/m15332/src")
import torch
from tensorsph.csf_consistent import csf_fields
from tensorsph.neighbor_search import PairListCache
import validation as V
torch.set_grad_enabled(False)
for ppw, dx in ((20, 0.001), (40, 0.0005), (80, 0.00025)):
    lam = 0.02
    for adx in (0.0, 0.5, 2.0):
        pos, mass, cfg = V.make_case(lam, lam, dx, adx*dx, 1000.0, 7.2, 0.0, 10.0)
        pos, mass = pos.float(), mass.float()
        h = cfg.h_factor*dx
        cache = PairListCache(2.0*h, cfg.pair_skin_factor*h, fluid_count=pos.shape[0])
        pairs, _ = cache.get(pos)
        out = []
        for scheme, kw in (("mls", {}), ("fit_quadratic", {}),
                           ("mls", {"curvature_color_smoothing": 1}),
                           ("mls", {"normal_smoothing": 4})):
            st = csf_fields(pos, mass, pairs, h, 1000.0, curvature_scheme=scheme, **kw)
            shell = st.delta_s > 0.5*st.delta_s.max()
            v = st.kappa[shell]
            out.append(f"{scheme}{kw}: mean={float(v.mean()):+7.1f} rms={float((v**2).mean().sqrt()):7.1f} n={int(shell.sum())}")
        print(f"ppw={ppw} A/dx={adx} N={pos.shape[0]}")
        for o in out: print("   ", o)
