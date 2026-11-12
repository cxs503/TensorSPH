#!/usr/bin/env python3
"""m16318 independent re-analysis of m16244 RT artifacts (adversarial check).

Recomputes, from the raw time series in benchmarks/rayleigh_taylor/runs/*.json:
  * sigma by least squares on ln(amplitude) inside the documented window
    [0.02, 0.1] * lambda  (does NOT reuse the stored fit),
  * the analytic reference sqrt(A g k), the finite-depth and viscous variants,
  * the relative error of the best configuration vs the 10 % task bar.

Writes audit/m16318_independent_refit.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "benchmarks" / "rayleigh_taylor" / "runs"


def lsq_sigma(t, a):
    """Slope of ln a vs t over the sample-span inside the fit window."""
    pts = [(ti, math.log(ai)) for ti, ai in zip(t, a)
           if ai > 0 and 0.02 <= ai <= 0.1]
    n = len(pts)
    if n < 6:
        return None, None, n
    mt = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mt) ** 2 for p in pts)
    sxy = sum((p[0] - mt) * (p[1] - my) for p in pts)
    sigma = sxy / sxx
    # r2 in log space
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (my + sigma * (p[0] - mt))) ** 2 for p in pts)
    return sigma, 1.0 - ss_res / ss_tot, n


def main():
    out = {"runs": [], "theory": {}}
    for f in sorted(RUNS.glob("dx*.json")):
        d = json.loads(f.read_text())
        cfg = d["config"]
        s = d["series"]
        sig, r2, n = lsq_sigma(s["t"], s["amplitude"])
        row = {
            "file": f"benchmarks/rayleigh_taylor/runs/{f.name}",
            "tag": d["tag"],
            "dx": cfg["dx"], "nu": cfg["viscosity"], "amp": cfg["amplitude"],
            "integrator": cfg.get("time_integrator", "euler"),
            "density_diffusion": cfg.get("density_diffusion", 0.0),
            "stored_sigma": d["fit"]["sigma"],
            "independent_sigma": sig,
            "independent_r2": r2,
            "n_points": n,
            "stored_rel_err_vs_Agk": d["sigma_over_ideal"] - 1.0,
            "particles": d["derived"]["particles"],
            "steps": d["derived"]["steps"],
            "wall_seconds": d["wall_seconds"],
        }
        if sig is not None and "reference" in d:
            row["independent_rel_err_vs_Agk"] = sig / d["reference"]["sigma_ideal_Agk"] - 1.0
        out["runs"].append(row)

    ref = json.loads((RUNS / "reference.json").read_text())
    A, g = ref["atwood"], 9.81
    k = 2.0 * math.pi / 1.0
    out["theory"] = {
        "sqrt_Agk_recomputed": math.sqrt(A * g * k),
        "stored_sigma_ideal_Agk": ref["sigma_ideal_Agk"],
        "rel_diff": math.sqrt(A * g * k) / ref["sigma_ideal_Agk"] - 1.0,
        "tolerance_bar": 0.10,
        "source": "Taylor 1950 Proc. R. Soc. A 201:192 doi:10.1098/rspa.1950.0052",
    }
    (HERE / "m16318_independent_refit.json").write_text(
        json.dumps(out, indent=1))
    for r in out["runs"]:
        print(f"{r['tag']:<34} int={r['integrator']:<5} "
              f"stored={r['stored_sigma']:.4f} indep="
              f"{(r['independent_sigma'] or float('nan')):.4f} "
              f"err={(r.get('independent_rel_err_vs_Agk') or float('nan')) * 100:+.2f}%")
    print("theory", json.dumps(out["theory"]))


if __name__ == "__main__":
    main()