import importlib.util, math, statistics, json
from pathlib import Path
import torch
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("osc", Path("benchmarks/oscillating_droplet/run.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
om = B.analytic_omega(0.01, 7.2, 1000.0); lam = 2*5e-5*4/0.01**2
# sampling like the CI trace (dt from stable_dt, ~1.8 periods, every 2nd sample)
cfg = B.Config(dx=0.001, h_factor=1.3, rho0=1000.0, sound_speed=20.0, gamma=7.0,
               gravity=0.0, viscosity=5e-5, surface_tension_coefficient=7.2)
dt = float(B.stable_dt(cfg)); T = 2*math.pi/om
steps = int(math.ceil(1.8*T/dt)); t = [(i*2+1)*dt for i in range(steps//2+1)]
print("n samples", len(t), "dt", dt, "T", T, "periods", t[-1]/T)
def synth(omega, rect, phase=0.0, noise=0.0, seed=0):
    ts = torch.tensor(t, dtype=torch.float64)
    w = 0.1*torch.exp(-lam*ts)*torch.cos(omega*ts+phase)
    if rect: w = w.abs()
    s = w
    if noise: 
        g = torch.Generator().manual_seed(seed); s = s + noise*0.1*torch.randn(s.shape, generator=g, dtype=torch.float64)
    return [float(v) for v in s]
def err(s, starts=6, tt=None):
    f = B.damped_cosine_fit(t if tt is None else tt, s, om, starts=starts)
    return (f["omega"]-om)/om*100, f
for rect in (False, True):
    es = [err(synth(om, rect, phase=p))[0] for p in (0, math.pi/4, math.pi/2)]
    print("rect", rect, "errors", [round(e,2) for e in es])
# windows on rectified vs signed
def window_errs(rect):
    out=[]
    for frac in (0.5,0.75,1.0):
        idx=[i for i,x in enumerate(t) if x-t[0] <= frac*(t[-1]-t[0])]
        e,_=err([synth(om,rect)[i] for i in idx], tt=[t[i] for i in idx]); out.append(round(e,2))
    return out
print("window rect", window_errs(True), "signed", window_errs(False))
# round-off on rectified
base = synth(om, True)
def pert(starts):
    vals=[]
    for k in range(5):
        g=torch.Generator().manual_seed(50+k); j=1e-12*torch.randn(len(base), generator=g, dtype=torch.float64)
        vals.append(round(err([b+float(x) for b,x in zip(base,j)], starts)[0],3))
    return vals
print("pert single", pert(1)); print("pert multi", pert(6))
# ambiguity: short window
for frac in (0.35, 0.5, 1.0):
    idx=[i for i,x in enumerate(t) if x-t[0] <= frac*(t[-1]-t[0])]
    f = B.damped_cosine_fit([t[i] for i in idx], [synth(om,True)[i] for i in idx], om)
    print("frac",frac,"ambiguity",round(f["ambiguity_rms_gap"],6),"basins",f["starts"],"alts",[round(a["omega"],1) for a in f["omega_alternatives"]])
