import sys, math, json, time
sys.path.insert(0, "/root/tp-agent/work/m15332/benchmarks/capillary_wave")
sys.path.insert(0, "/root/tp-agent/work/m15332/src")
import torch
import open_surface_audit as A
torch.set_grad_enabled(False)
rows = []
for ppw in (20, 40, 80):
    for adx in (0.5, 1.0, 2.0, 4.0):
        lam = 0.02
        dx = lam / ppw
        kw = dict(wavelength=lam, depth=lam, dx=dx, amp_over_dx=adx,
                  rho0=1000.0, sigma=7.2, viscosity=0.0, sound_speed=10.0)
        t0 = time.perf_counter()
        curv = A.curvature_audit(kw)
        best = min(curv["cells"], key=lambda c: c["noise_over_signal"])
        acc = A.initial_acceleration(kw)
        rows.append(dict(ppw=ppw, adx=adx, dx=dx, steepness=2*math.pi*adx/ppw,
                         n_iface=best["n_interface_particles"],
                         noise=round(best["noise_over_signal"],3),
                         scheme=best["curvature_scheme"],
                         lat=best["lattice"], hf=best["h_factor"],
                         aexcess=round(acc["csf_excess_over_physical"],2),
                         t=round(time.perf_counter()-t0,1)))
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(".cache/explore.json","w"), indent=1)
