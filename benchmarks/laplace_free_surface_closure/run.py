"""P0 verification: does a free-surface pressure closure recover the Young-Laplace jump?

Context
-------
``benchmarks/laplace_pressure_floor`` decomposed the Young-Laplace miss of
TensorSPH into four error sources and found exactly one gate item (there called
C3/P0): ``tait_pressure`` clamps ``p >= 0``, and every particle in the
truncated-support shell has ``rho < rho0``, so the shell sits at ``p = 0`` and the
interior can only hold ~0.57 of ``sigma/R``.  That benchmark explicitly did not
touch ``src/``.  This one measures the fix that was written for it:
``Config.free_surface_pressure`` -- the Adami, Ihmsen & Adams (2012, sec. 3.3)
shell extrapolation

    p_i = max(0, sum_j (p_j + rho_j g.(r_i - r_j)) W_ij / sum_j W_ij),

applied to particles with ``rho < threshold * rho0`` before the pressure force
is assembled.

What is measured
----------------
For every (dx, c0, threshold) case the drop relaxes for the same window used by
the floor benchmark, and the jump is read with the *absolute + time-averaged*
definition that the floor benchmark established (``p_core`` against the
``p(rho0) = 0`` vacuum reference).  ``threshold = 0`` is the EOS field and must
reproduce the floor benchmark's numbers; the difference between the two is the
closure's effect.

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_free_surface_closure/run.py --case 0.002 20 0.99
    PYTHONPATH=src python benchmarks/laplace_free_surface_closure/run.py --aggregate

Each ``--case`` run writes ``cases/<key>.json`` so a long sweep can be resumed;
``--aggregate`` folds whatever is on disk into ``report.json`` and the figure.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLOOR = ROOT / "benchmarks" / "laplace_pressure_floor"


def _load_floor_module():
    spec = importlib.util.spec_from_file_location("lpf_for_closure", FLOOR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lpf = _load_floor_module()

# (dx [m], c0 [m/s], threshold on rho/rho0). 0.99 = the shell layer; 1.02 is a
# sensitivity probe (it flags particles that are only marginally deficient).
CASES = [
    (0.002, 20.0, 0.0),
    (0.002, 20.0, 0.99),
    (0.002, 20.0, 1.02),
    (0.002, 5.0, 0.0),
    (0.002, 5.0, 0.99),
    (0.002, 5.0, 1.02),
    (0.0015, 20.0, 0.0),
    (0.0015, 20.0, 0.99),
    # Reduced-resolution ladder, added when the machine was at load15 ~ 200:
    # coarse enough to finish under contention, used to show that the closure's
    # effect does not depend on dx.
    (0.003, 20.0, 0.0),
    (0.003, 20.0, 0.99),
    (0.003, 5.0, 0.0),
    (0.003, 5.0, 0.99),
]
T_END_OVER_T_SIGMA = 1.5
CASES_DIR = HERE / "cases"


def case_key(dx: float, c0: float, threshold: float) -> str:
    return f"dx{dx:g}_c0{c0:g}_thr{threshold:g}".replace(".", "p")


def run_case(dx: float, c0: float, threshold: float) -> dict:
    run = lpf.relax(dx, c0, free_surface_pressure=threshold,
                    t_end_over_t_sigma=T_END_OVER_T_SIGMA)
    run["threshold"] = threshold
    run["case"] = case_key(dx, c0, threshold)
    run["compression_ratio"] = ((run["density"]["rho_core_over_rho0"] - 1.0)
                                / run["density"]["compression_required"])
    CASES_DIR.mkdir(exist_ok=True)
    (CASES_DIR / f"{run['case']}.json").write_text(json.dumps(run, indent=1))
    print(f"{run['case']}: tail {run['absolute_time_average_tail']['rel_error_pct']:+6.1f} %"
          f"  final {run['definition_absolute']['rel_error_pct']:+6.1f} %"
          f"  compression {run['compression_ratio']:.3f}"
          f"  flagged {run['n_free_surface_final']}/{run['n_particles']}"
          f"  steps {run['steps']}", flush=True)
    return run


def load_cases() -> list[dict]:
    runs = []
    for dx, c0, threshold in CASES:
        path = CASES_DIR / f"{case_key(dx, c0, threshold)}.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
    return runs


def aggregate() -> dict:
    runs = load_cases()
    by_key = {(r["dx_m"], r["sound_speed"], r["threshold"]): r for r in runs}

    pairs = []
    for dx, c0, threshold in CASES:
        if threshold == 0.0:
            continue
        base = by_key.get((dx, c0, 0.0))
        run = by_key.get((dx, c0, threshold))
        if base is None or run is None:
            continue
        pairs.append({
            "dx_m": dx, "sound_speed": c0, "threshold": threshold,
            "error_before_pct": base["absolute_time_average_tail"]["rel_error_pct"],
            "error_after_pct": run["absolute_time_average_tail"]["rel_error_pct"],
            "improvement_points": (
                abs(base["absolute_time_average_tail"]["rel_error_pct"])
                - abs(run["absolute_time_average_tail"]["rel_error_pct"])),
            "compression_ratio_before": base["compression_ratio"],
            "compression_ratio_after": run["compression_ratio"],
            "flagged_fraction": run["n_free_surface_final"] / run["n_particles"],
            "within_15pct_after": abs(
                run["absolute_time_average_tail"]["rel_error_pct"]) < 15.0,
        })

    report = {
        "benchmark": "laplace_free_surface_closure",
        "reference": {"delta_p_Pa": lpf.DP_ANALYTIC, "definition":
                      "absolute p_core against the p(rho0)=0 vacuum reference, \n"
                      "averaged over the last 40 % of the window"},
        "physics": {"sigma_N_per_m": lpf.SIGMA, "radius_m": lpf.RADIUS,
                    "rho0": lpf.RHO0, "gamma": lpf.GAMMA,
                    "t_sigma_s": lpf.T_SIGMA,
                    "window_over_t_sigma": T_END_OVER_T_SIGMA},
        "closure": {"form": "Adami, Ihmsen & Adams (2012) sec. 3.3",
                    "threshold_is": "rho < threshold * rho0",
                    "src_field": "src/tensorsph/droplet_impact.py:"
                                 "free_surface_pressure_closure"},
        "runs": runs,
        "pairs": pairs,
        "complete": len(runs) == len(CASES),
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    if pairs:
        report["figure"] = make_figure(report)
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", HERE / "report.json", "pairs", len(pairs), "runs", len(runs))
    return report


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(report, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = report["pairs"]
    runs = report["runs"]
    _dp = report["physics"]  # noqa: F841  (kept for readability of the panels)
    target = _dp["sigma_N_per_m"] / _dp["radius_m"]  # sigma / R

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

    # (a) error before/after, per case
    ax = axes[0][0]
    labels = [f"dx={p['dx_m'] * 1000:g}mm\nc0={p['sound_speed']:g}"
              f"\nthr={p['threshold']:g}" for p in pairs]
    idx = list(range(len(pairs)))
    ax.bar([i - 0.2 for i in idx], [p["error_before_pct"] for p in pairs], 0.4,
           label="EOS field (closure off)")
    ax.bar([i + 0.2 for i in idx], [p["error_after_pct"] for p in pairs], 0.4,
           label="closed field")
    ax.axhline(-15.0, color="r", ls=":", lw=1, label="-15 % gate")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(r"$(\Delta p - \sigma/R)/(\sigma/R)$ [%]")
    ax.set_title("(a) closure moves the jump, does not close the gate")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # (b) radial pressure profile, best case vs EOS field
    ax = axes[0][1]
    best = min(pairs, key=lambda p: abs(p["error_after_pct"]))
    for thr, style, label in ((0.0, "o--", "EOS field"),
                              (best["threshold"], "s-", "closed field")):
        run = next((r for r in runs if r["dx_m"] == best["dx_m"]
                    and r["sound_speed"] == best["sound_speed"]
                    and r["threshold"] == thr), None)
        if run is None:
            continue
        rr = [q["r_over_R_mid"] for q in run["pressure_profile"]]
        ax.plot(rr, [q["mean"] for q in run["pressure_profile"]], style, ms=4,
                label=label)
    ax.axhline(target, color="k", ls="--", lw=1,
               label=rf"$\sigma/R$={target:.0f} Pa")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel("mean pressure [Pa]")
    ax.set_title(f"(b) profile at dx={best['dx_m'] * 1000:g}mm, "
                 f"$c_0$={best['sound_speed']:g} m/s")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) threshold sensitivity at the finest c0
    ax = axes[1][0]
    for c0 in sorted({p["sound_speed"] for p in pairs}):
        sel = [p for p in pairs if p["sound_speed"] == c0 and p["dx_m"] == 0.002]
        sel.sort(key=lambda p: p["threshold"])
        base = next((r for r in runs if r["sound_speed"] == c0
                     and r["threshold"] == 0.0 and r["dx_m"] == 0.002), None)
        xs = [0.0] + [p["threshold"] for p in sel]
        ys = ([base["absolute_time_average_tail"]["rel_error_pct"]] if base else []) \
            + [p["error_after_pct"] for p in sel]
        if len(xs) == len(ys) and ys:
            ax.plot(xs, ys, "o-", label=rf"$c_0$={c0:g} m/s")
    ax.axhline(-15.0, color="r", ls=":", lw=1)
    ax.set_xlabel("closure threshold on $\\rho/\\rho_0$")
    ax.set_ylabel("tail-averaged error [%]")
    ax.set_title("(c) sensitivity to the shell threshold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (d) achieved vs required compression
    ax = axes[1][1]
    for thr, marker, label in ((0.0, "o", "EOS field"),
                               (0.99, "s", "closed (0.99)")):
        sel = [r for r in runs if r["threshold"] == thr and r["dx_m"] == 0.002]
        sel.sort(key=lambda r: -r["sound_speed"])
        if not sel:
            continue
        ax.plot([r["sound_speed"] for r in sel],
                [r["compression_ratio"] for r in sel], marker + "-", label=label)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="required by Young-Laplace")
    ax.set_xlabel(r"$c_0$ [m/s]")
    ax.set_ylabel("achieved / required interior compression")
    ax.set_title("(d) the closure restores interior compression")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("TensorSPH P0: free-surface pressure closure vs the "
                 "Young-Laplace jump")
    fig.tight_layout()
    path = Path(out_dir) / "laplace_free_surface_closure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs=3, type=float, metavar=("DX", "C0", "THRESHOLD"))
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.case:
        run_case(*args.case)
    if args.aggregate:
        aggregate()
    if not args.case and not args.aggregate:
        ap.error("pass --case DX C0 THRESHOLD or --aggregate")


if __name__ == "__main__":
    main()
