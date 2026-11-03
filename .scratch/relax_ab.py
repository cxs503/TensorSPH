import sys, torch, importlib.util
torch.set_num_threads(2)
sys.path.insert(0, "src")
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
law = load("law", "benchmarks/laplace_law/run.py")
for dx, steps in ((0.004, 380), (0.002, 600)):
    for norm in ("fixed", "jump"):
        rep = law.relax(radius=0.02, dx=dx, sigma=7.2, steps=steps,
                        curvature="fit_quadratic", delta_s=norm)
        lap = rep["laplace"]; sp = rep.get("spurious_current", {})
        print(f"dx={dx} steps={steps} {norm:5s} dp={lap['delta_p_measured_Pa']:+.1f} "
              f"err={lap['rel_error_pct']:+.1f}% vmax_ratio={sp.get('ratio', float('nan')):.3f}", flush=True)
