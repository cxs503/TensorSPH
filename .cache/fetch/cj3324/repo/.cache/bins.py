import sys, math
sys.path.insert(0, "/root/tp-agent/work/m15332/src")
sys.path.insert(0, "/root/tp-agent/work/m15332/benchmarks/capillary_wave")
import torch
from tensorsph.csf_consistent import csf_fields
from tensorsph.neighbor_search import PairListCache
import validation as V
torch.set_grad_enabled(False)
lam=0.02
for ppw,dx in ((20,0.001),(40,0.0005)):
    for adx in (0.0,0.5,2.0):
        pos,mass,cfg = V.make_case(lam,lam,dx,adx*dx,1000.0,7.2,0.0,10.0)
        pos,mass=pos.float(),mass.float()
        h=cfg.h_factor*dx
        cache=PairListCache(2.0*h,cfg.pair_skin_factor*h,fluid_count=pos.shape[0])
        pairs,_=cache.get(pos)
        st=csf_fields(pos,mass,pairs,h,1000.0)
        ds=st.delta_s; kap=st.kappa; col=st.color
        hot = ds > 0.05*float(ds.max())
        print(f"ppw={ppw} A/dx={adx}: n_hot={int(hot.sum())} c range on hot: {float(col[hot].min()):.2f}..{float(col[hot].max()):.2f}")
        # bin by colour value on the hot set (c=1 interior, c=0 void; interface ~0.3-0.7)
        for lo,hi in ((0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)):
            m = hot & (col>=lo) & (col<hi)
            if int(m.sum())==0: continue
            v=kap[m]
            print(f"   c in [{lo},{hi}): n={int(m.sum()):4d} kappa mean={float(v.mean()):+8.1f} rms={float((v**2).mean().sqrt()):8.1f} ds_med={float(ds[m].median()):6.1f}")
        # also by y-depth below surface
        print("   kappa by row index from top:")
        ys = pos[:,1]
        top = float(ys.max())
        for k in range(4):
            m = hot & (ys > top - (k+1)*h) & (ys <= top - k*h)
            if int(m.sum())==0: continue
            v=kap[m]
            print(f"     row{k}: n={int(m.sum()):4d} mean={float(v.mean()):+8.1f} rms={float((v**2).mean().sqrt()):8.1f} c_med={float(col[m].median()):.2f} ds_med={float(ds[m].median()):.0f}")
