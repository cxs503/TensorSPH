"""Acceptance run for the free-surface pressure closure (P0 of m2971's backlog).

Why this file exists
--------------------
``run.py`` measures *where* the Young-Laplace pressure jump is lost and records
the un-closed (``tait``) answer in ``report.json``.  This file re-runs the four
acceptance points of that record with the closure switched on
(``Config.free_surface_pressure``, Adami et al. 2012 eq. (11) applied to fluid
particles whose sum-density is deficient) and writes ``report_shell.json``.

Only the closed runs are computed here: the un-closed numbers are read back from
``report.json``.  The two are comparable run-for-run because ``relax`` derives
``dt`` from the same ``stable_dt`` cap and the same window length, so both use
the same number of steps (recorded in each run).

Acceptance criterion (m2971, machine-checkable)
-----------------------------------------------
``absolute_time_average_tail`` within 15 % at every one of the four points
c0 = 20 / 5 m/s  x  dx = 0.002 / 0.0015 m.

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run_closure.py
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run_closure.py --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

ACCEPTANCE = ((0.002, 20.0), (0.002, 5.0), (0.0015, 20.0), (0.0015, 5.0))
TOLERANCE_PCT = 15.0


def _load_run_module():
    spec = importlib.util.spec_from_file_location("laplace_pressure_floor_run",
                                                  HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lpf = _load_run_module()


def tait_baseline() -> dict:
    """Un-closed runs from the attribution benchmark, keyed by (dx, c0)."""
    path = HERE / "report.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    if report.get("pressure_closure", "tait") != "tait":
        raise RuntimeError("report.json no longer holds the un-closed baseline")
    return {(run["dx_m"], run["sound_speed"]): run for run in report["runs"]}


def make_figure(report: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) radial pressure profile, one representative case, both closures.
    ax = axes[0]
    case = report["runs"][0]
    base = report["baseline"][f"{case['dx_m']:g}/{case['sound_speed']:g}"]
    for rows, label, style in ((base["pressure_profile"], "p >= 0 clamp (tait)", "o--"),
                               (case["pressure_profile"], "shell closure (Adami)", "s-")):
        ax.plot([r["r_over_R_mid"] for r in rows], [r["mean"] for r in rows],
                style, label=label, markersize=4)
    ax.axhline(report["reference"]["delta_p_Pa"], color="k", lw=0.8, ls=":")
    ax.text(1.02, report["reference"]["delta_p_Pa"], r" $\sigma/R$", va="center",
            fontsize=9)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel("pressure [Pa]")
    ax.set_title(f"(a) dx={case['dx_m']:g} m, $c_0$={case['sound_speed']:g} m/s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) acceptance: tail-averaged jump error, both closures, four points.
    ax = axes[1]
    labels, tait_err, shell_err = [], [], []
    for run in report["runs"]:
        key = f"{run['dx_m']:g}/{run['sound_speed']:g}"
        labels.append(f"dx={run['dx_m']:g}\n$c_0$={run['sound_speed']:g}")
        tait_err.append(report["baseline"][key]["absolute_time_average_tail"]["rel_error_pct"])
        shell_err.append(run["absolute_time_average_tail"]["rel_error_pct"])
    idx = range(len(labels))
    ax.bar([i - 0.2 for i in idx], tait_err, width=0.4, label="tait clamp", color="#b0413e")
    ax.bar([i + 0.2 for i in idx], shell_err, width=0.4, label="shell closure", color="#3f7d20")
    ax.axhline(0.0, color="k", lw=0.8)
    for sign in (-TOLERANCE_PCT, TOLERANCE_PCT):
        ax.axhline(sign, color="k", lw=0.8, ls="--")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"tail-averaged $\Delta p$ error [%]")
    ax.set_title("(b) acceptance: |error| < 15 % (dashed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = HERE / "laplace_pressure_closure.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="short window (diagnostics only, not the acceptance run)")
    ap.add_argument("--t-end", type=float, default=1.5)
    args = ap.parse_args()

    baseline = tait_baseline()
    runs = []
    for dx, c0 in ACCEPTANCE:
        run = lpf.relax(dx, c0, t_end_over_t_sigma=args.t_end,
                        closure="shell",
                        max_steps=1200 if args.quick else 4000)
        runs.append(run)
        key = f"{dx:g}/{c0:g}"
        base = baseline.get((dx, c0))
        delta = ""
        if base is not None:
            delta = (f"  (tait {base['absolute_time_average_tail']['rel_error_pct']:+7.1f} %"
                     f" -> {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %)")
        print(f"shell dx={dx:g} c0={c0:5g}: abs {run['definition_absolute']['rel_error_pct']:+7.1f} %"
              f"  tail-avg {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %"
              f"  steps {run['steps']}  finite {run['finite']}{delta}", flush=True)

    comparison = {}
    for run in runs:
        key = f"{run['dx_m']:g}/{run['sound_speed']:g}"
        base = baseline.get((run["dx_m"], run["sound_speed"]))
        tait_tail = None if base is None else base["absolute_time_average_tail"]["rel_error_pct"]
        shell_tail = run["absolute_time_average_tail"]["rel_error_pct"]
        comparison[key] = {
            "tait_tail_rel_error_pct": tait_tail,
            "shell_tail_rel_error_pct": shell_tail,
            # improvement is measured in magnitude: positive = closer to sigma/R
            "improvement_pt": (None if tait_tail is None
                               else abs(tait_tail) - abs(shell_tail)),
            "passes_15pct": abs(shell_tail) < TOLERANCE_PCT,
            "passes_3pct": abs(shell_tail) < 3.0,
        }

    report = {
        "benchmark": "laplace_pressure_floor",
        "pressure_closure": "shell",
        "closure_reference": "Adami, Hu & Adams, J. Comput. Phys. 231 (2012) eq. (11)",
        "reference": {"delta_p_Pa": lpf.DP_ANALYTIC, "sigma_N_per_m": lpf.SIGMA,
                      "radius_m": lpf.RADIUS, "rho0": lpf.RHO0,
                      "gamma": lpf.GAMMA, "t_sigma_s": lpf.T_SIGMA},
        "acceptance": {"tolerance_pct": TOLERANCE_PCT,
                       "points": [[dx, c0] for dx, c0 in ACCEPTANCE]},
        "runs": runs,
        "baseline": {f"{dx:g}/{c0:g}": baseline[(dx, c0)]
                     for dx, c0 in ACCEPTANCE if (dx, c0) in baseline},
        "comparison": comparison,
    }
    (HERE / "report_shell.json").write_text(json.dumps(report, indent=2))
    figure = make_figure(report)
    report["figures"] = [str(figure.relative_to(HERE.parents[1]))]
    (HERE / "report_shell.json").write_text(json.dumps(report, indent=2))

    all_pass = all(v["passes_15pct"] for v in comparison.values())
    print("acceptance |error| < 15 % at all points:", all_pass)
    print("wrote", HERE / "report_shell.json", "and", figure)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())