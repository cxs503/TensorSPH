import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("lr", "benchmarks/laplace_law/run.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
here = Path("benchmarks/laplace_law")
for p in m.convergence_ladder(dxs=m.LADDER_DXS, t_over_t_sigma=1.0,
                              calibrate_mass=True,
                              outdir=here / "convergence_calibrated"):
    d = json.loads(p.read_text())
    print(p.name, "bias=%.5f" % d["setup"]["lattice_bias_rho_over_rho0"],
          "tail_err=%.2f%%" % d["laplace"]["tail"]["rel_error_pct"],
          "p_c=%.1f" % d["laplace"]["tail"]["p_center_Pa"], flush=True)
