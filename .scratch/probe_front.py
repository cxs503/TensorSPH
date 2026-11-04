"""Quick probe: dam-break front rate vs the analytic Ritter front (not committed)."""
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorsph.dam_break import DamBreakConfig, run_case, ritter_front

a = 0.2
cfg = DamBreakConfig(dx=0.04, pool_width=2.0, column_width=a, sound_speed=25.0,
                     max_courant=0.3, device="cpu", wall_friction=0.0,
                     density_diffusion=0.0, adaptive_dt=False)
cfg.column_height = 2.0 * a
cfg.dt = cfg.dt_from_courant()
cfg.steps = int(round(0.5 / cfg.dt))
t0 = time.perf_counter()
run = run_case(cfg)
print("wall %.1fs N=%d steps=%d" % (time.perf_counter() - t0,
                                    run.final_positions.shape[0], run.steps_taken))
tau = [t * math.sqrt(2.0 * cfg.gravity / a) for t in run.times]
za = [(z - a) / a for z in run.front]
for k in range(0, len(tau), max(1, len(tau) // 20)):
    print("t=%.4f tau=%.3f za=%+.3f  H*sqrt(gH)t/a=%.3f" % (
        run.times[k], tau[k], za[k],
        ritter_front(run.times[k], cfg) / a))
print("final tau=%.3f za=%+.3f  front=%.4f" % (tau[-1], za[-1], run.front[-1]))
print("H = %.3f, sqrt(gH) = %.3f, 2sqrt(gH)/ (a*sqrt(2g/a)) = %.3f"
      % (cfg.column_height, math.sqrt(cfg.gravity * cfg.column_height),
         2 * math.sqrt(cfg.gravity * cfg.column_height) / (a * math.sqrt(2 * cfg.gravity / a))))
# least-squares slope of za vs tau over tau in [0.5, 4]
xs = [(t, z) for t, z in zip(tau, za) if 0.5 <= t <= 4.0]
n = len(xs)
sx = sum(x for x, _ in xs); sy = sum(y for _, y in xs)
sxx = sum(x * x for x, _ in xs); sxy = sum(x * y for x, y in xs)
slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
print("slope d(Z/a)/d tau = %.4f (n=%d)" % (slope, n))
print("front speed / sqrt(gH) = %.4f" % (slope * a * math.sqrt(2 * cfg.gravity / a)
                                         / math.sqrt(cfg.gravity * cfg.column_height)))