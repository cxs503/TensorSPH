import json, math, os, torch
torch.set_num_threads(8); torch.set_default_dtype(torch.float64)
from tensorsph.capillary_oscillation import run_point, rayleigh_2d, SIGMA_WATER_AIR, RHO_WATER, RADIUS_DEFAULT, MODE_DEFAULT
SIG, RHO, R = SIGMA_WATER_AIR, RHO_WATER, RADIUS_DEFAULT
d = "benchmarks/surface_tension/out_droplet_oscillation"; os.makedirs(d, exist_ok=True)
path = os.path.join(d, "oscillation.json")
out = {"reference": {
    "law": "omega_m^2 = sigma m (m^2-1) / (rho R^3)  (2-D drop, Rayleigh/Lamb)",
    "sigma_n_m": SIG, "rho0_kg_m3": RHO, "radius_m": R, "mode": MODE_DEFAULT,
    "amp_eps": 0.05, "gate_pct": 15.0,
    "period_exact_s": 2*math.pi/rayleigh_2d(SIG, RHO, R, MODE_DEFAULT),
    "laplace_exact_pa": SIG/R,
    "model": "production CSF (cfg.csf_scheme='consistent'), free surface, WCSPH Tait EOS c0=0.5 m/s, gravity off, adaptive dt (CFL 0.25), density_calibration on"},
  "containment_definition": "escaped = any particle beyond 1.6 R; max_r_over_R = largest particle radius / R",
  "ladder": [], "control": None,
  "note": "single-resolution (R/dx = 10) diagnostic matrix over one Rayleigh period; resolution ladder submitted as cluster job cj2559, still queued when this matrix was produced"}
runs = [("baseline", dict()), ("viscous_5e-4", dict(viscosity=5.0e-4))]
for name, kw in runs:
    r = run_point(10, SIG, 1.0, sample_every=10, **kw)
    r["config"] = name
    finite = r["period_err_pct"] == r["period_err_pct"]
    r["period_pass"] = bool(finite and abs(r["period_err_pct"]) < 15.0)
    r["laplace_pass"] = bool(abs(r["dp_err_pct"]) < 15.0)
    r["contained"] = bool(r["escaped_particles"] == 0)
    out["ladder"].append(r)
    json.dump(out, open(path, "w"), indent=2)
    print(name, {k: r[k] for k in ("particles","steps","period_measured_s","period_err_pct","p_interior_mean_pa","dp_err_pct","zero_crossings","containment_max_r_over_R","escaped_particles","t_first_escape_s","wall_s")}, flush=True)
print("wrote", path)
