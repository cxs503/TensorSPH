import sys, math
sys.path.insert(0,"/root/tp-agent/work/m15332/src"); sys.path.insert(0,"/root/tp-agent/work/m15332/benchmarks/capillary_wave")
import torch
import open_surface_audit as A
from tensorsph.droplet_impact import wall_lattice, wall_mass
torch.set_grad_enabled(False)
lam=0.02; dx=0.001
kw=dict(wavelength=lam,depth=lam,dx=dx,amp_over_dx=0.5,rho0=1000.0,sigma=7.2,viscosity=0.0,sound_speed=10.0)
pos,mass,cfg,rho,p,acc=A.initial_fields(kw)
a=torch.linalg.vector_norm(acc,dim=1)
W=lam/2; h=cfg.h_factor*dx
x=pos[:,0]; y=pos[:,1]
wallz=(x<2.5*h)|(x>W-2.5*h)
top=float(y.max()); freesurf=y>top-3*dy_ if False else y>kwdepth if False else y>lam-3*dx
print("n=",pos.shape[0],"a_max=",float(a.max()),"a_p99=",float(torch.quantile(a,0.99)))
for name,m in (("wall_adjacent",wallz),("free_surface_rows",freesurf),("interior",(~wallz)&(~freesurf))):
    print(f"{name:18s} n={int(m.sum()):4d} a_max={float(a[m].max()):8.1f} a_med={float(a[m].median()):8.1f} share_of_sum={float(a[m].sum()/a.sum()):.2f}")
# top-10 particles
idx=torch.topk(a,10).indices
print("top10 |a| with (x,y):")
for i in idx.tolist():
    print(f"   x={float(x[i]):.4f} y={float(y[i]):.4f} |a|={float(a[i]):8.1f} wall={bool(wallz[i])} near_top={bool(y[i]>lam-3*dx)}")
