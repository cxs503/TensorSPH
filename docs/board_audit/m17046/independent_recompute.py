"""Board audit m17046 — independent recomputation for REPRO-SPH-MONAGHAN1994.

Auditor's own metric implementation: nothing here imports
``benchmarks/repro_monaghan_1994/run.py`` (no ``compare``/``interp`` reuse), so a
match against ``results/summary.json`` is evidence that the stored metrics are
rederivable, not that the same code was called twice.

Also audits m17020's error-budget arithmetic (additivity claim) and its Ritter
comparison window against Ritter's validity limit tau < sqrt(2).

Writes ``artifacts/audit_m17046/independent_recompute.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "benchmarks" / "repro_monaghan_1994"
RES = CASE / "results"
OUT = Path(__file__).resolve().parent / "independent_recompute.json"

WINDOW = (0.5, 3.0)      # pre-registered in run.py:29 / summary.json:protocol
N = 161                  # pre-registered grid
TOL = 0.15               # platform tolerance, 15 % of reference amplitude


def lerp(xs, ys, xq):
    """Clamped linear interpolation (auditor's own; deliberately not run.py's)."""
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:                       # bisection, unlike run.py's scan
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    w = 0.0 if span == 0 else (xq - xs[lo]) / span
    return ys[lo] + w * (ys[hi] - ys[lo])


def metrics(sim_tau, sim_za, ref_tau, ref_za, label):
    lo = max(WINDOW[0], sim_tau[0], ref_tau[0])
    hi = min(WINDOW[1], sim_tau[-1], ref_tau[-1])
    grid = [lo + (hi - lo) * i / (N - 1) for i in range(N)]
    sim = [lerp(sim_tau, sim_za, t) for t in grid]
    ref = [lerp(ref_tau, ref_za, t) for t in grid]
    d = [s - e for s, e in zip(sim, ref)]
    absd = [abs(x) for x in d]
    amp = max(ref)
    mae = sum(absd) / len(absd)
    return {
        "reference": label,
        "n_samples": len(grid),
        "window": [grid[0], grid[-1]],
        "amplitude": amp,
        "mae": mae,
        "mae_over_amplitude": mae / amp,
        "bias": sum(d) / len(d),
        "bias_over_amplitude": (sum(d) / len(d)) / amp,
        "max_abs": max(absd),
        "worst_sample": {
            "tau": grid[absd.index(max(absd))],
            "sim": sim[absd.index(max(absd))],
            "reference": ref[absd.index(max(absd))],
        },
    }


def main():
    summary = json.loads((RES / "summary.json").read_text())
    refs = json.loads((CASE / "refs.json").read_text())
    cross = json.loads((ROOT / "benchmarks" / "dam_break" /
                        "martin_moyce_1952_crosscheck.json").read_text())
    bundled = json.loads((ROOT / "benchmarks" / "dam_break" /
                          "martin_moyce_1952.json").read_text())

    # crosscheck stores X/a measured from the wall: Z/a = X/a - 1 (refs.json convention)
    cross_za = [row["X_over_a"] - 1.0 for row in cross["data"]]
    cross_tau = [row["tau"] for row in cross["data"]]
    b_rows = bundled.get("data") or bundled.get("points") or bundled
    if isinstance(b_rows, list) and b_rows and isinstance(b_rows[0], dict):
        b_tau = [r.get("tau") for r in b_rows]
        b_za = [r.get("Z_over_a", r.get("X_over_a")) for r in b_rows]
    else:  # pragma: no cover - shape guard
        raise SystemExit("unexpected bundled table shape")

    report = {"window": list(WINDOW), "n_samples": N, "tolerance": TOL,
              "ladder_rederived": [], "stored_vs_rederived": []}

    for name in ("dx0.04_c025", "dx0.02_c025", "dx0.01_c025"):
        traj = json.loads((RES / f"trajectory_{name}.json").read_text())
        got_cross = metrics(traj["tau"], traj["za"], cross_tau, cross_za, "crosscheck")
        got_bund = metrics(traj["tau"], traj["za"], b_tau, b_za, "bundled")
        stored = summary["comparisons"][name]
        for got in (got_cross, got_bund):
            st = stored[got["reference"]]
            report["stored_vs_rederived"].append({
                "case": name,
                "reference": got["reference"],
                "stored_mae": st["mae"],
                "rederived_mae": got["mae"],
                "abs_diff": abs(st["mae"] - got["mae"]),
                "stored_mae_over_amplitude": st["mae_over_amplitude"],
                "rederived_mae_over_amplitude": got["mae_over_amplitude"],
                "pass_at_15pct": got["mae_over_amplitude"] <= TOL,
            })
        report["ladder_rederived"].append({
            "case": name, "dx": traj["dx"],
            "particles": traj["particles"], "steps": traj["steps_taken"],
            "wall_clock_s": traj["wall_clock_s"],
            "max_density_compression": traj["max_density_compression"],
            "crosscheck": got_cross, "bundled": got_bund,
        })

    # --- reference self-consistency, recomputed -------------------------------
    lo = max(WINDOW[0], cross_tau[0], b_tau[0])
    hi = min(WINDOW[1], cross_tau[-1], b_tau[-1])
    grid = [lo + (hi - lo) * i / (N - 1) for i in range(N)]
    spread = [abs(lerp(cross_tau, cross_za, t) - lerp(b_tau, b_za, t)) for t in grid]
    claimed = cross.get("claimed_uncertainty", 0.05)
    report["reference_self_consistency"] = {
        "max_spread_Za": max(spread),
        "mean_spread_Za": sum(spread) / len(spread),
        "claimed_uncertainty": claimed,
        "ratio": max(spread) / claimed,
        "stored_max_spread": summary["bracket"]["reference_spread_max"],
        "stored_ratio": summary["bracket"]["reference_spread_over_claimed"],
    }

    # --- m17020 error-budget arithmetic audit --------------------------------
    ebf = RES / "error_budget.json"
    if not ebf.exists():
        # m17020's artefact lives only on branch agent/m17020 (c670396); audit it
        # read-only from that checkout instead of copying it into this tree.
        alt = Path("/root/tp-agent/work/m17020/benchmarks/repro_monaghan_1994/results/error_budget.json")
        if alt.exists():
            ebf = alt
    if ebf.exists():
        eb = json.loads(ebf.read_text())
        comp = eb["error_budget"]["components"]
        total = eb["error_budget"]["total_mae_over_amplitude"]
        pieces = {
            "reference_spread_over_amplitude": comp["reference_uncertainty"]["spread_over_amplitude"],
            "ritter_mae_over_amplitude": comp["ritter_deviation"]["mae_over_amplitude"],
            "compressibility_max_rho": comp["compressibility"]["max_rho_over_rho0_minus_1"],
        }
        report["m17020_error_budget_audit"] = {
            "found": True,
            "total_mae_over_amplitude": total,
            "components": pieces,
            "sum_of_components": sum(pieces.values()),
            "sum_over_total": sum(pieces.values()) / total,
            "residual_claimed_in_doc": total - pieces["reference_spread_over_amplitude"],
            "additive": False,
            "source": str(ebf),
        }
    else:
        report["m17020_error_budget_audit"] = {
            "found": False,
            "note": "results/error_budget.json is NOT on this worktree's base "
                    "(m17020 branch agent/m17020 commit c670396, unmerged at audit time)",
        }

    # --- Ritter validity audit (own arithmetic: Z/a = 2 tau) -----------------
    traj = json.loads((RES / "trajectory_dx0.01_c025.json").read_text())
    tau_valid = math.sqrt(2.0)
    rows, rows_valid = [], []
    for t, z in zip(traj["tau"], traj["za"]):
        if t < WINDOW[0] or t > WINDOW[1]:
            continue
        r = 2.0 * t
        rows.append({"tau": t, "za_sph": z, "za_ritter": r, "delta": z - r})
        if t <= tau_valid:
            rows_valid.append({"tau": t, "za_sph": z, "za_ritter": r, "delta": z - r})
    amp = max(cross_za)

    def summarize(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "tau_range": [rs[0]["tau"], rs[-1]["tau"]],
            "mean_abs_delta_Za": sum(abs(r["delta"]) for r in rs) / len(rs),
            "mean_abs_delta_over_amplitude": sum(abs(r["delta"]) for r in rs) / len(rs) / amp,
            "max_ritter_za": max(r["za_ritter"] for r in rs),
            "max_abs_delta_Za": max(abs(r["delta"]) for r in rs),
        }

    report["ritter_validity_audit"] = {
        "validity_limit_tau": tau_valid,
        "full_pre_registered_window": summarize(rows),
        "inside_ritter_validity": summarize(rows_valid),
        "note": "Ritter (shallow water, 2:1 column) gives Z/a = 2*tau; at the top "
                "of the pre-registered window tau=2.97 it predicts Z/a=5.93 against "
                "a measured front that saturates near 2.6, i.e. the deviation there "
                "measures Ritter's breakdown, not the SPH error.",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"written {OUT}")
    for row in report["stored_vs_rederived"]:
        print(f"{row['case']:12s} {row['reference']:10s} stored={row['stored_mae']:.9f} "
              f"rederived={row['rederived_mae']:.9f} |d|={row['abs_diff']:.2e} "
              f"mae/amp={row['rederived_mae_over_amplitude']*100:.3f}% pass15={row['pass_at_15pct']}")
    print("ref spread:", report["reference_self_consistency"])
    print("ritter:", json.dumps(report["ritter_validity_audit"], indent=1)[:700])
    print("budget audit:", report["m17020_error_budget_audit"])


if __name__ == "__main__":
    main()