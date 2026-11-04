import math, torch
from tensorsph.droplet_impact import Config, pair_kernels
from tensorsph.free_surface_pressure import extrapolated_pressure
from tensorsph.neighbor_search import cell_linked_pairs
import sys; sys.path.insert(0,"tests")
from test_free_surface_pressure import _hex_patch, _state, _config, RHO0
pos = _hex_patch(); G=100.0
cfg=_config(free_surface_pressure="shell", gravity=G, sound_speed=1.0)
y = pos[:,1]; ys=float(y.max())
rho = torch.full((pos.shape[0],), 0.95*RHO0, dtype=pos.dtype)
p = RHO0*G*(ys-y)
mass, pairs, _ = _state(pos)
pe,_ = extrapolated_pressure(pos, rho, p, mass, pairs, pos.shape[0], cfg)
err=(pe-p).abs()
print("max abs err", float(err.max()), "max rel", float((err/p.clamp_min(1.0)).max()))
k=int(err.argmax()); print("worst idx",k,"p",float(p[k]),"pe",float(pe[k]),"y",float(y[k]))
# check the plain average for the same particle
h=cfg.h_factor*cfg.dx
i,j=pairs.unbind(1); delta=pos[i]-pos[j]; wij,_=pair_kernels(delta,h)
num=torch.zeros(pos.shape[0],dtype=p.dtype).index_add(0,i,p[j]*wij).index_add(0,j,p[i]*wij)
den=torch.zeros(pos.shape[0],dtype=p.dtype).index_add(0,i,wij).index_add(0,j,wij)
print("plain",float((num/den)[k]))
# moment check
m1=torch.zeros((pos.shape[0],2),dtype=p.dtype).index_add(0,i,(-delta)*wij[:,None]).index_add(0,j,delta*wij[:,None])
print("sum(dx*W)/sumW at worst", (m1[k]/den[k])*1e3, "mm")
