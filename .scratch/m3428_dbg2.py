import math, torch, sys
sys.path.insert(0,"tests")
from tensorsph.droplet_impact import Config, pair_kernels
from tensorsph.free_surface_pressure import extrapolated_pressure
from tensorsph.neighbor_search import cell_linked_pairs
from test_free_surface_pressure import _hex_patch, _state, _config, RHO0
pos=_hex_patch(); G=100.0
cfg=_config(free_surface_pressure="shell", gravity=G, sound_speed=1.0)
y=pos[:,1]; ys=float(y.max())
rho=torch.full((pos.shape[0],),0.95*RHO0,dtype=pos.dtype)
p=RHO0*G*(ys-y)
mass,pairs,_=_state(pos)
k=int((p-p).abs().argmax()) if False else 552
h=cfg.h_factor*cfg.dx
# neighbours of k directly
d=torch.linalg.vector_norm(pos-pos[k],dim=1)
nb=torch.nonzero((d<2*h)&(d>0)).flatten()
delta=pos[nb]-pos[k]
wij,_=pair_kernels(delta,h)
val=p[nb]-cfg.gravity*delta[:,1]*rho[nb]   # g.(r_k - r_j) = -G*(yk-yj)
print("manual shell pressure at k:", float((val*wij).sum()/wij.sum()), "p_k", float(p[k]))
pe,_=extrapolated_pressure(pos,rho,p,mass,pairs,pos.shape[0],cfg)
print("module:", float(pe[k]))
# count pairs incident to k
i,j=pairs.unbind(1)
print("pairs incident", int(((i==k)|(j==k)).sum()), "manual nb", nb.numel())
