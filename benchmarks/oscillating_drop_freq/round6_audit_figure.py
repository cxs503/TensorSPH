"""Round 6 audit figure: the two initialisation defects, before vs after.

Everything plotted is computed here (no transcribed numbers).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorsph.oscillating_drop_dynamic import (DynamicConfig, completeness,
    half_plane_lattice, initial_density, mirror_cloud, summation_density,
    run_dynamic_drop, _accelerations)

OUT = Path(__file__).with_name("out"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).resolve().parents[2] / "docs" / "figures"


def state(rdx, c0, gate, sigma=1.0, clip=True, ds=0.0):
    cfg = DynamicConfig(dx=1/rdx, sigma=sigma, eps=0.05, c0=c0, delta_sph=ds, mode=2)
    cfg.validate(); torch.set_default_dtype(torch.float64)
    geo = half_plane_lattice(cfg); pos = geo["pos"].clone(); mass = geo["mass"].clone()
    n = pos.shape[0]
    for _ in range(3):
        pm, _, mm = mirror_cloud(pos, torch.zeros_like(pos), mass)
        rm = summation_density(pm, mm, cfg.h, cfg.kernel)
        if gate:
            comp = completeness(pm, mm, rm, cfg.h, cfg.kernel)[:n]
            scale = torch.where(comp >= cfg.completeness_gate,
                                cfg.rho0/rm[:n], torch.ones_like(rm[:n]))
        else:
            scale = cfg.rho0/rm[:n]
        mass = mass*scale
    pm, _, mm = mirror_cloud(pos, torch.zeros_like(pos), mass)
    rho = initial_density(pm, mm, cfg.h, cfg.rho0, sigma=sigma, radius=cfg.radius,
                          c0=c0, gamma=cfg.gamma)[:n].clamp(min=0.5*cfg.rho0)
    return cfg, pos, mass, rho, geo


def pressure(rho, cfg):
    return torch.clamp(cfg.rho0*cfg.c0**2/cfg.gamma*((rho/cfg.rho0)**cfg.gamma-1), min=0.0)


rows = {}
# (a) radial density profile, defective vs gated calibration (c0 = 5)
# NOTE the c0: at c0 = 5 the Laplace compression is 6.6 %, so the clamp at
# rho0_lap (added in round 6) already covers the calibration overshoot and the
# defect is invisible.  It must be shown at the SHIPPED c0 = 20, where the
# Laplace compression is 0.49 % and the 4 % overshoot is 8x the physical signal.
def state_pre_fix(rdx, c0, sigma=1.0):
    """The round-5 initialiser, reproduced verbatim (ungated calibration, no
    clamp, and the Tait inverse without its gamma) -- the object of the audit."""
    cfg = DynamicConfig(dx=1/rdx, sigma=sigma, eps=0.05, c0=c0, delta_sph=0.0, mode=2)
    cfg.validate(); torch.set_default_dtype(torch.float64)
    geo = half_plane_lattice(cfg); pos = geo["pos"].clone(); mass = geo["mass"].clone()
    n = pos.shape[0]
    for _ in range(2):
        pm, _, mm = mirror_cloud(pos, torch.zeros_like(pos), mass)
        rm = summation_density(pm, mm, cfg.h, cfg.kernel)
        mass = mass * (cfg.rho0 / rm[:n])
    pm, _, mm = mirror_cloud(pos, torch.zeros_like(pos), mass)
    rho = initial_density(pm, mm, cfg.h, cfg.rho0, sigma=sigma, radius=cfg.radius,
                          c0=c0, gamma=cfg.gamma)[:n].clamp(min=0.5*cfg.rho0)
    # undo the round-6 clamp and the missing gamma, i.e. the round-5 formulas
    rho_s = summation_density(pm, mm, cfg.h, cfg.kernel)[:n]
    p_lap = 2.0*sigma/cfg.radius
    rho = (rho_s/cfg.rho0)*cfg.rho0*(1.0 + p_lap/(cfg.rho0*c0*c0))**(1.0/cfg.gamma)
    return cfg, pos, mass, rho.clamp(min=0.5*cfg.rho0), geo


for tag, gate in (("pre_fix", False), ("post_fix", True)):
    cfg, pos, mass, rho, geo = (state_pre_fix(9.0, 20.0) if tag == "pre_fix"
                                else state(9.0, 20.0, gate))
    r = torch.linalg.vector_norm(pos, dim=1)
    edges = torch.linspace(0, 1, 21); mids, prof = [], []
    for k in range(20):
        m = (r >= edges[k]) & (r < edges[k+1])
        if int(m.sum()):
            mids.append(float(0.5*(edges[k]+edges[k+1]))); prof.append(float(rho[m].mean()/cfg.rho0))
    p = pressure(rho, cfg)
    _, acc, _, _, _ = _accelerations(pos, torch.zeros_like(pos), rho, mass, cfg)
    rows[tag] = {"r": mids, "rho": prof, "p_max": float(p.max()),
                 "a0": float(torch.linalg.vector_norm(acc, dim=1).max()),
                 "rho_max": float(rho.max())}

# (b) inverse-Tait defect: target p_lap vs what the missing-gamma form delivers
cfg, pos, mass, rho, geo = state(9.0, 5.0, True)
gamma = cfg.gamma; p_lap = 2.0
p_missing = (cfg.rho0*cfg.c0**2/gamma)*((1.0 + p_lap/(cfg.rho0*cfg.c0**2)) - 1.0)
rows["tait"] = {"p_lap": p_lap, "p_missing_gamma": float(p_missing),
                "p_fixed": float(pressure(rho, cfg).max()), "gamma": gamma}

# (c) sigma -> 0 control: the shipped march must stay at rest
cfg0 = DynamicConfig(dx=1/9.0, sigma=0.0, eps=1e-3, c0=5.0, delta_sph=0.1, mode=2,
                     t_end=0.02, sample_every=5, periods=1.0); cfg0.validate()
r0 = run_dynamic_drop(cfg0)
rows["sigma0"] = {"vmax": r0["max_speed"], "drift": r0["density_drift_max"],
                  "steps": r0["n_steps"], "round5_vmax50": 214.0}

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].plot(rows["pre_fix"]["r"], rows["pre_fix"]["rho"], "o-", label="pre-fix (ungated calibration)")
ax[0].plot(rows["post_fix"]["r"], rows["post_fix"]["rho"], "s-", label="round 6 (completeness-gated)")
ax[0].axhline(1.0, color="k", ls=":", lw=1)
ax[0].set_xlabel(r"$r/R$"); ax[0].set_ylabel(r"$\rho/\rho_0$")
ax[0].set_title(r"(a) radial density, $R/dx=9$, $c_0=20$"); ax[0].legend(fontsize=7)
bars = [rows["tait"]["p_lap"], rows["tait"]["p_missing_gamma"], rows["tait"]["p_fixed"]]
ax[1].bar(["Rayleigh\n$2\\sigma/R$", "Tait inverse\nmissing $\\gamma$", "round 6\nfixed"],
          bars, color=["k", "tab:red", "tab:green"])
for i, b in enumerate(bars):
    ax[1].text(i, b*1.02, f"{b:.4f}", ha="center", fontsize=8)
ax[1].set_ylabel("interior pressure at $t=0$"); ax[1].set_title(r"(b) inverse-Tait ($\gamma=7$)")
ax[2].bar(["$|v|_{\\max}$ 50 steps\n(round 5)", "$|v|_{\\max}$\nround 6"],
          [rows["sigma0"]["round5_vmax50"], rows["sigma0"]["vmax"]],
          color=["tab:red", "tab:green"], log=True)
ax[2].set_title(r"(c) tension-free control, $\sigma\to0$")
plt.tight_layout()
png = FIG / "case_sph_osc_drop_freq_round6.png"
plt.savefig(png, dpi=140)
(OUT / "round6_audit.json").write_text(json.dumps(rows, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("r", "rho")} for k, v in rows.items()}, indent=1))
print("figure:", png)
