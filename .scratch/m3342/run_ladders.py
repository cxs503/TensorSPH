import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("lr", "benchmarks/laplace_law/run.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
here = Path("benchmarks/laplace_law")
# row that matches the main validation report's dx, at the fixed physical time
rep = m.relax(dx=0.004, t_over_t_sigma=1.0)
(here / "convergence" / "relax_dx0.0040.json").write_text(json.dumps(rep, indent=2))
print("uncalibrated dx=0.004 done", rep["laplace"]["tail"]["rel_error_pct"], flush=True)
# same ladder with the lattice kernel-sum offset removed
paths = m.convergence_ladder(dxs=m.LADDER_DXS, t_over_t_sigma=1.0,
                             calibrate_mass=True,
                             outdir=here / "convergence_calibrated")
for p in paths:
    d = json.loads(p.read_text())
    print(p.name, round(d["laplace"]["tail"]["rel_error_pct"], 2),
          round(d["laplace"]["tail"]["p_center_Pa"], 1), flush=True)
