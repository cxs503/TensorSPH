import sys
sys.path.insert(0, "/root/tp-agent/work/m15332/src"); sys.path.insert(0, "/root/tp-agent/work/m15332/benchmarks/capillary_wave")
import torch
from tensorsph.csf_consistent import csf_fields
from tensorsph.neighbor_search import PairListCache
import validation as V
torch.set_grad_enabled(False)
lam=0.02
for ppw,dx in ((20,0.001),(40,0.0005),(80,0.00025)):
    for adx in (0.0,0.5):
        pos,mass,cfg=V.make_case(lam,lam,dx,adx*dx,1000.0,7.2,0.0,10.0)
        pos,mass=pos.float(),mass.float()
        h=cfg.h_factor*dx
        cache=PairListCache(2.0*h,cfg.pair_skin_factor*h,fluid_count=pos.shape[0]); pairs,_=cache.get(pos)
        st=csf_fields(pos,mass,pairs,h,1000.0)
        ds=st.delta_s; kap=st.kappa; col=st.color
        hot = ds > 0.5*float(ds.max())
        W=lam/2
        x=pos[:,0]
        inner = hot & (x>2.5*h) & (x<W-2.5*h)
        wallz = hot & ~inner
        sig = (2*3.141592653589793/lam)**2 * adx*dx
        print(f"ppw={ppw} A/dx={adx} signal={sig:6.1f} | hot n={int(hot.sum())}")
        for name,m in (("all",hot),("inner",inner),("walls",wallz)):
            if int(m.sum())==0: continue
            v=kap[m]
            print(f"   {name:6s} n={int(m.sum()):4d} mean={float(v.mean()):+8.1f} rms={float((v**2).mean().sqrt()):8.1f} p95={float(torch.quantile(v.abs(),0.95)):8.1f}")
