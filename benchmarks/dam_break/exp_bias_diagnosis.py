"""B2 root cause: why does the dam-break experimental gate stay at MAE ~ 0.58?

Context
-------
``benchmarks/dam_break/resolution_ladder.json`` (mission m2502) measured the
Martin & Moyce (1952) comparison on a three-rung ladder and found the mean
absolute error in ``Z/a`` to be **flat** across ``dx`` = 0.04 / 0.02 / 0.01 m
(0.5988 / 0.5774 / 0.5728) while the RMSE *grows*.  A resolution-independent
residual is not numerical diffusion of the front, so refinement cannot be the
remedy.  This script decomposes the residual into measurable components instead
of guessing:

1. **Startup lag** -- the simulated front does not start moving immediately
   (the column is released from rest); the lag is reported in ``tau`` units and
   in seconds, and its resolution dependence is checked.
2. **Late-time slope mismatch** -- the front speed d(Z/a)/d(tau) is fitted by
   least squares over ``tau >= 2`` and compared with the experimental and the
   Ritter slopes.
3. **Wall treatment** -- the same case is re-run with ``wall_friction = 0``
   (free slip, the shipped default) and ``1.0`` (no slip), because the
   experiment had a no-slip bed.  This is implementation sequence item 6 of
   ``docs/algorithm-gap-analysis.md`` ("decide the default wall_friction from
   that comparison").
4. **Reference-table provenance audit** -- the bundled digitised table is
   checked for internal consistency with the claim that it was read off a
   published figure (uniform abscissae, ordinate quantisation, scatter versus
   the stated uncertainty).

Everything is written to one JSON artifact; the Markdown write-up
(``docs/benchmarks/dam-break-exp-bias-diagnosis.md``) is generated from that
artifact by ``make_bias_doc.py`` so no number is hand-copied.

Usage::

    python benchmarks/dam_break/exp_bias_diagnosis.py --sizes 0.04,0.02 \
        --frictions 0.0,1.0 --seconds 0.5 \
        --json-out outputs/dam_break_exp_bias/bias_diagnosis.json --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tensorsph.dam_break import (  # noqa: E402
    DamBreakConfig,
    run_case,
    ritter_front,
)

HERE = Path(__file__).resolve().parent
REFERENCE_PATH = HERE / "martin_moyce_1952.json"

#: The gate limit used by ``experimental_validation.py`` (platform standard).
EXP_MAE_LIMIT = 0.15
#: Post-startup window used for the slope fit, in ``tau`` units.
SLOPE_FIT_TAU_MIN = 2.0


# --------------------------------------------------------------------------
# reference-table provenance audit
# --------------------------------------------------------------------------
def audit_reference(ref: dict) -> dict:
    """Internal-consistency audit of the bundled digitised reference table.

    A table that was manually digitised off a figure has (a) irregular
    abscissae, because the digitiser clicks wherever the curve is, and (b)
    ordinates that scatter by at least the reading uncertainty.  The metrics
    below make both statements testable rather than rhetorical.
    """
    data = ref["data"]
    taus = [p["tau"] for p in data]
    za = [p["Z_over_a"] for p in data]
    uncs = [p.get("uncertainty", 0.0) for p in data]

    steps = [taus[i] - taus[i - 1] for i in range(1, len(taus))]
    step_mean = sum(steps) / len(steps)
    step_spread = max(abs(s - step_mean) for s in steps)

    slopes = [(za[i] - za[i - 1]) / steps[i - 1] for i in range(1, len(taus))]

    # Ordinate quantisation: largest q such that every Z/a is (near) a multiple.
    quantum = None
    for q in (0.25, 0.2, 0.1, 0.05, 0.025, 0.01):
        if all(abs(z / q - round(z / q)) < 1e-9 for z in za):
            quantum = q
            break

    # Local curvature: second differences.  A digitised curve has second
    # differences of the order of the reading noise; a table produced from a
    # low-order formula has them vanishingly small.
    second = [abs(za[i + 1] - 2 * za[i] + za[i - 1]) for i in range(1, len(za) - 1)]
    second_max = max(second)
    second_mean = sum(second) / len(second)

    stated_unc = max(uncs) if uncs else 0.0
    # Scatter that the stated uncertainty would imply, versus what the table
    # actually shows.  If the table's second differences are far below the
    # stated uncertainty, the uncertainty was not derived from the table.
    scatter_ratio = (second_mean / stated_unc) if stated_unc else None

    return {
        "n_points": len(data),
        "tau_first": taus[0],
        "tau_last": taus[-1],
        "tau_step_mean": step_mean,
        "tau_step_spread": step_spread,
        "tau_grid_is_uniform": bool(step_spread < 1e-9),
        "z_over_a_values": za,
        "ordinate_quantum": quantum,
        "stated_uncertainty_max": stated_unc,
        "second_difference_mean": second_mean,
        "second_difference_max": second_max,
        "scatter_over_stated_uncertainty": scatter_ratio,
        "slope_min": min(slopes),
        "slope_max": max(slopes),
        "slope_first": slopes[0],
        "slope_last": slopes[-1],
        "monotone": all(slopes),
        "provenance_source": ref.get("source", {}).get("paper", ""),
        "digitization_method": ref.get("digitization", {}).get("method", ""),
    }


def audit_verdict(audit: dict) -> dict:
    """Turn the reference-audit metrics into explicit red flags."""
    flags = []
    if audit["tau_grid_is_uniform"]:
        flags.append(
            "abscissae are exactly uniform (spacing %.3f, spread %.1e): manual "
            "figure digitisation does not produce a uniform grid"
            % (audit["tau_step_mean"], audit["tau_step_spread"])
        )
    if audit["ordinate_quantum"] is not None:
        flags.append(
            "every Z/a lies on a %.3f grid, which is also the stated "
            "uncertainty -- the ordinates are quantised read-off steps, not "
            "measured values with scatter" % audit["ordinate_quantum"]
        )
    if audit["scatter_over_stated_uncertainty"] is not None and (
        audit["scatter_over_stated_uncertainty"] < 0.1
    ):
        flags.append(
            "mean |second difference| is %.3f, i.e. %.0f%% of the stated "
            "uncertainty: the table cannot be a digitisation of the plotted "
            "experimental points, which scatter by the stated amount"
            % (audit["second_difference_mean"],
               100 * audit["scatter_over_stated_uncertainty"])
        )
    return {
        "red_flags": flags,
        "n_flags": len(flags),
        "verdict": (
            "provenance not independently verifiable: the bundled table is "
            "internally inconsistent with the documented digitisation method"
            if flags else "no provenance red flag detected"
        ),
        "consequence": (
            "the experimental gate can still report a disagreement, but it "
            "cannot support a physical-validation *claim* until the numbers "
            "are traced back to the published figure"
            if flags else "none"
        ),
    }


# --------------------------------------------------------------------------
# trajectory diagnostics
# --------------------------------------------------------------------------
def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares y = slope*x + intercept."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx


def interp(xs: list[float], ys: list[float], xq: float) -> float:
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= xq:
            w = (xq - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] != xs[i - 1] else 0.0
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def _mae(sim_tau, sim_za, exp_data, shift: float = 0.0) -> dict:
    """MAE of sim against the reference, optionally after shifting sim time.

    ``shift > 0`` advances the simulation clock, i.e. it compares
    ``sim(tau + shift)`` with ``exp(tau)``; a positive best fit therefore
    measures a *startup lag* of the simulation.
    """
    errs = []
    for p in exp_data:
        tau_e = p["tau"]
        if not sim_tau or tau_e < sim_tau[0] or tau_e > sim_tau[-1] - shift:
            continue
        if tau_e + shift < sim_tau[0]:
            continue
        errs.append(abs(interp(sim_tau, sim_za, tau_e + shift) - p["Z_over_a"]))
    if not errs:
        return {"mae": None, "rmse": None, "n": 0}
    return {
        "mae": sum(errs) / len(errs),
        "rmse": math.sqrt(sum(e * e for e in errs) / len(errs)),
        "n": len(errs),
    }


def diagnose_trajectory(times: list[float], front: list[float], cfg,
                        exp_data: list[dict]) -> dict:
    """Decompose the sim-vs-experiment residual of one run."""
    a = cfg.column_width
    scale = math.sqrt(2.0 * cfg.gravity / a)
    sim_tau = [t * scale for t in times]
    sim_za = [(z - a) / a for z in front]

    # 1. initial offset: the leading particle sits half a cell inside the column
    z0 = sim_za[0]

    # 2. startup lag: first time the front has advanced 0.1 in Z/a past z0
    lag_tau = None
    for tau, za in zip(sim_tau, sim_za):
        if za - z0 >= 0.1:
            lag_tau = tau
            break

    # 3. late-time slope (tau >= 2)
    late = [(t, z) for t, z in zip(sim_tau, sim_za) if t >= SLOPE_FIT_TAU_MIN]
    if len(late) >= 3:
        slope, intercept = linfit([t for t, _ in late], [z for _, z in late])
    else:
        slope, intercept = (None, None)

    # 4. experimental comparison, raw and with the best-fit time shift
    raw = _mae(sim_tau, sim_za, exp_data, 0.0)
    best = {"shift": 0.0, **raw}
    for k in range(1, 61):                     # 0 .. 1.2 in steps of 0.02
        shift = 0.02 * k
        cand = _mae(sim_tau, sim_za, exp_data, shift)
        if cand["mae"] is not None and (best["mae"] is None or cand["mae"] < best["mae"]):
            best = {"shift": shift, **cand}

    # 5. correct Ritter anchor: Z/a = (2 sqrt(g H) t - a)/a
    ritter_za = [(ritter_front(t, cfg) - a) / a for t in times]
    rel = [abs(z - r) / abs(r) for z, r in zip(sim_za, ritter_za) if r > 0.2]
    ritter_err = sum(rel) / len(rel) if rel else None
    ritter_slope = None
    rlate = [(t, z) for t, z in zip(sim_tau, ritter_za) if t >= SLOPE_FIT_TAU_MIN]
    if len(rlate) >= 3:
        ritter_slope, _ = linfit([t for t, _ in rlate], [z for _, z in rlate])

    exp_slope, _ = linfit(
        [p["tau"] for p in exp_data if p["tau"] >= SLOPE_FIT_TAU_MIN],
        [p["Z_over_a"] for p in exp_data if p["tau"] >= SLOPE_FIT_TAU_MIN],
    )

    return {
        "initial_za": z0,
        "startup_lag_tau": lag_tau,
        "startup_lag_seconds": (lag_tau / scale) if lag_tau is not None else None,
        "late_slope_dza_dtau": slope,
        "late_intercept": intercept,
        "ritter_late_slope": ritter_slope,
        "exp_late_slope": exp_slope,
        "slope_ratio_sim_over_exp": (slope / exp_slope) if slope and exp_slope else None,
        "ritter_mean_relative_error": ritter_err,
        "exp_mae_raw": raw["mae"],
        "exp_rmse_raw": raw["rmse"],
        "exp_n_compared_raw": raw["n"],
        "exp_mae_best_shift": best["mae"],
        "exp_rmse_best_shift": best["rmse"],
        "exp_best_shift_tau": best["shift"],
        "gate_pass_raw": bool(raw["mae"] is not None and raw["mae"] <= EXP_MAE_LIMIT),
        "gate_pass_shifted": bool(best["mae"] is not None and best["mae"] <= EXP_MAE_LIMIT),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_config(dx: float, friction: float, seconds: float, *, sound_speed=25.0,
               column_width=0.2, tank_width=2.0, max_courant=0.3,
               device="cpu") -> DamBreakConfig:
    cfg = DamBreakConfig(
        dx=dx,
        pool_width=tank_width,
        column_width=column_width,
        sound_speed=sound_speed,
        max_courant=max_courant,
        device=device,
        wall_friction=friction,
    )
    cfg.column_height = 2.0 * column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="0.04,0.02")
    ap.add_argument("--frictions", default="0.0,1.0")
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--sound-speed", type=float, default=25.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    ref = json.loads(REFERENCE_PATH.read_text())
    exp_data = ref["data"]
    audit = audit_reference(ref)
    verdict = audit_verdict(audit)

    results = []
    for dx in [float(s) for s in args.sizes.split(",") if s.strip()]:
        for fr in [float(s) for s in args.frictions.split(",") if s.strip()]:
            cfg = run_config(dx, fr, args.seconds, sound_speed=args.sound_speed,
                             device=args.device)
            t0 = time.time()
            run = run_case(cfg)
            wall = time.time() - t0
            diag = diagnose_trajectory(run.times, run.front, cfg, exp_data)
            results.append({
                "dx": dx,
                "wall_friction": fr,
                "particles": int(run.final_positions.shape[0]),
                "dt": cfg.dt,
                "steps_taken": run.steps_taken,
                "simulated_time": run.simulated_time,
                "wall_clock_s": wall,
                "max_speed": run.max_speed,
                "mach_number": run.max_speed / cfg.sound_speed,
                "max_density_compression": run.max_density_compression,
                "escaped": run.escaped,
                "sim_tau": run.times and [t * math.sqrt(2 * cfg.gravity / cfg.column_width)
                                          for t in run.times],
                "sim_za": [(z - cfg.column_width) / cfg.column_width for z in run.front],
                **diag,
            })
            print(f"dx={dx:g} friction={fr:g}: MAE_raw={diag['exp_mae_raw']:.4f} "
                  f"MAE_shift({diag['exp_best_shift_tau']:.2f})={diag['exp_mae_best_shift']:.4f} "
                  f"lag_tau={diag['startup_lag_tau']} slope={diag['late_slope_dza_dtau']:.3f} "
                  f"({wall:.0f}s)", flush=True)

    payload = {
        "benchmark": "B2_dam_break_exp_bias_diagnosis",
        "purpose": ("decompose the resolution-independent experimental residual "
                    "reported in resolution_ladder.json into startup lag, "
                    "late-time slope and wall treatment"),
        "gate_limit_mae": EXP_MAE_LIMIT,
        "slope_fit_tau_min": SLOPE_FIT_TAU_MIN,
        "reference": {
            "file": str(REFERENCE_PATH.relative_to(ROOT)),
            "audit": audit,
            "verdict": verdict,
        },
        "results": results,
    }
    out = Path(args.json_out) if args.json_out else (
        ROOT / "outputs" / "dam_break_exp_bias" / "bias_diagnosis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")

    if args.plot:
        try:
            make_plot(payload, exp_data, out.parent / "bias_diagnosis.png")
            print(f"wrote {out.parent / 'bias_diagnosis.png'}")
        except Exception as exc:                       # pragma: no cover
            print(f"plot skipped: {exc}")
    return 0


def make_plot(payload: dict, exp_data: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot([p["tau"] for p in exp_data], [p["Z_over_a"] for p in exp_data],
            "ks", ms=4, label="Martin & Moyce (1952), bundled table")
    ax.fill_between([p["tau"] for p in exp_data],
                    [p["Z_over_a"] - 0.05 for p in exp_data],
                    [p["Z_over_a"] + 0.05 for p in exp_data],
                    color="0.8", alpha=0.6, label=r"stated $\pm0.05$ band")
    for r in payload["results"]:
        ax.plot(r["sim_tau"], r["sim_za"], lw=1.4,
                label=f"dx={r['dx']:g} m, friction={r['wall_friction']:g}")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax.set_ylabel(r"$Z/a$")
    ax.set_title("Front trajectory vs the bundled reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, raw, shifted = [], [], []
    for r in payload["results"]:
        labels.append(f"{r['dx']:g}\nfr={r['wall_friction']:g}")
        raw.append(r["exp_mae_raw"])
        shifted.append(r["exp_mae_best_shift"])
    idx = range(len(labels))
    ax.bar([i - 0.2 for i in idx], raw, width=0.4, label="raw MAE")
    ax.bar([i + 0.2 for i in idx], shifted, width=0.4, label="MAE after best time shift")
    ax.axhline(payload["gate_limit_mae"], color="r", ls="--", lw=1,
               label=f"gate limit {payload['gate_limit_mae']}")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"MAE in $Z/a$")
    ax.set_title("Residual before/after removing the startup lag")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())