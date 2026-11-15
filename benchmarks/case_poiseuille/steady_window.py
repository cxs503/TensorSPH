"""Is the R1 "steady" measurement window actually stationary?

Why this script exists
----------------------
``uncertainty.py`` grades the R1 steady parabola with ``U_input`` = the half
range of the in-run samples of the functional.  That number is only an honest
*repeatability* if the window it is taken over is stationary: if the channel is
still filling, the half range measures the transient and the band is a bound on
the wrong thing.  The committed ladder samples ``0.80 .. 1.00 s`` only, and a
0.2 s window cannot distinguish "noise around a plateau" from "slow drift" --
that ambiguity is what made gate U3b fail for several releases.

This script answers the question directly, with one extra piece of physics and
no new numerics: after the same constant forcing from rest, the residual fill
decays as ``exp(-nu pi^2 t / H^2)`` (e-folding time ``H^2/(nu pi^2) = 0.101 s``
for the case parameters).  Sampling the functional at two windows *separated by
six e-folding times* therefore separates the two hypotheses:

* stationary plateau -> the two window means agree to within the within-window
  scatter, and the scatter is a noise floor (a repeatability, i.e. ``U_input``);
* residual fill      -> the later window sits systematically lower, and the
  difference is ~``exp(-6) = 0.25 %`` of the remaining transient.

Predeclared gates (S = stationarity; fixed before the extended run)
-------------------------------------------------------------------
| gate | question | threshold |
|---|---|---|
| **S1** | does the extended run reproduce the committed window? (same protocol, same answer) | mean over `0.80..1.00 s` within 1 % of the committed rung's sample mean |
| **S2** | is the window stationary? | `|mean(B) - mean(A)|` <= the within-window half range of A ∪ B |
| **S3** | can any surviving fill explain the difference? | predicted fill ratio `exp(-nu pi^2 Delta t/H^2)` <= 1 % |
| **S4** | does the between-window repeatability support the 5 % acceptance? | `U_V = hypot(U_num, U_input_between) % <= 5` |

``A = [0.80, 1.00] s`` (the committed window) and ``B = [1.60, 1.80] s``; both
are sampled on the same 0.02 s grid as the ladder, so the functional is computed
by the identical code path.  The run continues to ``t_end = 2.0 s``; the 0.20 s
tail after B is not used.

Usage::

    python benchmarks/case_poiseuille/steady_window.py            # R = 12, 16
    python benchmarks/case_poiseuille/steady_window.py --resolutions 16 --quick

Outputs ``out/steady_window.json``, ``docs/case_poiseuille_steady_window.md``
and ``docs/figures/case_poiseuille_steady_window.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from tensorsph import womersley as wm  # noqa: E402
from tensorsph.verification.uncertainty import repeatability  # noqa: E402

import run as case  # noqa: E402  (the case driver: constants + drive())

OUT_JSON = HERE / "out" / "steady_window.json"
DOC = REPO / "docs" / "case_poiseuille_steady_window.md"
FIG = REPO / "docs" / "figures" / "case_poiseuille_steady_window.png"

WINDOW_A = (0.80, 1.00)     # the committed measurement window
WINDOW_B = (1.60, 1.80)     # six e-folding times later
SAMPLE_DT = 0.02
T_END = 2.00

TOL_PERCENT = 5.0           # feature-ledger acceptance
S1_REPRODUCE_PERCENT = 1.0
S3_FILL_RATIO = 0.01


# --------------------------------------------------------------------------- #
# the functional (identical to run.evaluate_constant)
# --------------------------------------------------------------------------- #
def steady_functional(u_profile, y, height: float) -> float:
    """``100 * RMS( (u - u_parabola)/u_parabola )`` over the interior band.

    Kept byte-for-byte equivalent to ``run.evaluate_constant`` so the extended
    run and the committed ladder are graded by the same definition; the identity
    is pinned by a test.
    """
    u_par = wm.steady_velocity(y, height, case.BODY_FORCE, case.VISCOSITY)
    interior = ((y / height >= case.INTERIOR_LO) & (y / height <= case.INTERIOR_HI))
    rel = (np.asarray(u_profile) - u_par) / u_par
    return 100.0 * float(np.sqrt(np.mean(rel[interior] ** 2)))


def sample_times(t_end: float = T_END) -> list[float]:
    n = int(round((t_end - WINDOW_A[0]) / SAMPLE_DT)) + 1
    return [WINDOW_A[0] + i * SAMPLE_DT for i in range(n)]


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def _window(times, values, span):
    lo, hi = span
    sel = [i for i, t in enumerate(times) if lo - 1e-9 <= t <= hi + 1e-9]
    return [values[i] for i in sel], [times[i] for i in sel]


def trend_test(times, values) -> dict:
    """Ordinary least squares slope of the functional against time.

    Returns the slope per second, its standard error and the t statistic.  A
    stationary window has ``|t|`` below ~2 for 9 degrees of freedom; the point of
    the test is that a *drift* would be visible against the scatter it is buried
    in, which is exactly what a two-window comparison alone cannot say.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    n = t.size
    if n < 3:
        return {"n": int(n), "slope_per_s": None, "t_stat": None,
                "reason": "fewer than three samples"}
    tm, vm = t.mean(), v.mean()
    sxx = float(np.sum((t - tm) ** 2))
    sxy = float(np.sum((t - tm) * (v - vm)))
    slope = sxy / sxx
    resid = v - (vm + slope * (t - tm))
    dof = n - 2
    s2 = float(np.sum(resid ** 2)) / dof
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    # an exactly linear window leaves no residual (se = 0): the slope is then
    # infinitely significant, not undefined -- reporting None there would hide a
    # pure ramp, which is the one case the test must catch.
    if se > 0:
        t_stat = slope / se
    else:
        t_stat = math.copysign(math.inf, slope) if slope != 0 else 0.0
    return {"n": int(n), "slope_per_s": slope, "slope_se": se,
            "t_stat": t_stat,
            "dof": dof,
            "drift_over_window": slope * (t.max() - t.min())}


def fill_prediction(dt_separation: float, height: float, viscosity: float) -> dict:
    """How much of the start-up transient can survive between A and B.

    ``u = u_steady - sum a_n exp(-nu (n pi/H)^2 t) sin(...)``: the slowest mode
    is ``n = 1``, so the surviving fraction after ``dt`` is bounded by
    ``exp(-nu pi^2 dt / H^2)``.  This is a property of the equation, not of the
    solver, and it is what turns "the two windows agree" into "the window is
    stationary, not merely slow".
    """
    lam = viscosity * math.pi ** 2 / height ** 2
    return {
        "decay_rate_1_over_s": lam,
        "e_folding_s": 1.0 / lam,
        "separation_s": dt_separation,
        "predicted_fill_ratio": math.exp(-lam * dt_separation),
        "note": ("bounded by the slowest decay mode of the start-up series; the "
                 "measured difference between windows A and B cannot exceed this "
                 "fraction of whatever transient was left at the start of A"),
    }


def analyse_rung(rung: dict, committed: dict | None, u_num_percent: float | None) -> dict:
    """Grade one extended rung: S1..S4 plus the repeatability estimators."""
    times = rung["sample_times"]
    vals = rung["functional_percent"]
    a_vals, a_times = _window(times, vals, WINDOW_A)
    b_vals, b_times = _window(times, vals, WINDOW_B)
    rep_a = repeatability(a_vals)
    rep_b = repeatability(b_vals)
    union_half = 0.5 * (max(a_vals + b_vals) - min(a_vals + b_vals))
    between = 0.5 * abs(rep_b["mean"] - rep_a["mean"])
    mean_between = 0.5 * (rep_a["mean"] + rep_b["mean"])

    committed_samples = (committed or {}).get("steady_rms_samples_pct") or []
    committed_mean = float(np.mean(committed_samples)) if committed_samples else None
    s1_dev = (100.0 * abs(rep_a["mean"] - committed_mean) / committed_mean
              if committed_mean else None)

    scale_u = between                      # percent of the functional scale
    mean_u = 100.0 * between / mean_between if mean_between else 0.0
    within_half = max(rep_a["half_range"], rep_b["half_range"])

    fill = fill_prediction(WINDOW_B[0] - WINDOW_A[0], case.HEIGHT, case.VISCOSITY)
    u_num = float(u_num_percent) if u_num_percent is not None else 0.0
    u_v = math.hypot(u_num, scale_u)

    gates = {
        "S1_extended_run_reproduces_committed_window": (
            None if s1_dev is None else bool(s1_dev <= S1_REPRODUCE_PERCENT)),
        "S2_window_is_stationary": bool(abs(rep_b["mean"] - rep_a["mean"])
                                        <= union_half),
        "S3_surviving_fill_cannot_explain_the_difference": bool(
            fill["predicted_fill_ratio"] <= S3_FILL_RATIO),
        "S4_between_window_band_supports_5pct": bool(u_v <= TOL_PERCENT),
    }
    return {
        "resolution_H_over_dx": rung["resolution_H_over_dx"],
        "dt": rung["dt"],
        "steps": rung["steps"],
        "wall_clock_s": rung["wall_clock_s"],
        "n_fluid": rung["n_fluid"],
        "max_mach": rung["max_mach"],
        "sample_times": times,
        "functional_percent": vals,
        "window_A": {"span_s": list(WINDOW_A), "n": rep_a["n"],
                     "mean_percent": rep_a["mean"],
                     "half_range_percent_of_scale": rep_a["half_range"],
                     "half_range_percent_of_mean": rep_a["half_range_percent_of_mean"]},
        "window_B": {"span_s": list(WINDOW_B), "n": rep_b["n"],
                     "mean_percent": rep_b["mean"],
                     "half_range_percent_of_scale": rep_b["half_range"],
                     "half_range_percent_of_mean": rep_b["half_range_percent_of_mean"]},
        "committed_window_mean_percent": committed_mean,
        "S1_deviation_percent": s1_dev,
        "between_window": {
            "half_range_percent_of_scale": scale_u,
            "half_range_percent_of_mean": mean_u,
            "mean_percent": mean_between,
            "within_window_half_range_percent_of_scale": within_half,
            "union_half_range_percent_of_scale": union_half,
        },
        "trend_window_A": trend_test(a_times, a_vals),
        "trend_window_B": trend_test(b_times, b_vals),
        "fill": fill,
        "u_num_percent": u_num,
        "u_input_percent_of_scale": scale_u,
        "U_V_percent": u_v,
        "gates": gates,
    }


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #
def integrate(resolution: int, device, t_end: float = T_END) -> dict:
    """Constant forcing from rest, sampled across ``0.80 .. t_end``."""
    dx = case.HEIGHT / resolution
    length = case.COLUMNS * dx
    cfg = case.make_case_config(dx, case.VISCOSITY, device)
    times = sample_times(t_end)
    raw = case.drive(cfg, case.HEIGHT, length, lambda t: case.BODY_FORCE, t_end,
                     times, device)
    y = raw["y_rows"]
    vals = [steady_functional(u, y, case.HEIGHT) for u in raw["u_samples"]]
    return {
        "resolution_H_over_dx": resolution,
        "dt": raw["dt"],
        "steps": raw["steps"],
        "n_fluid": raw["n_fluid"],
        "n_ghost": raw["n_ghost"],
        "max_mach": max(p["max_mach"] for p in raw["trace"]),
        "wall_clock_s": raw["wall_clock_s"],
        "sample_times": [float(t) for t in raw["sample_times"]],
        "functional_percent": [float(v) for v in vals],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[12, 16])
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="smoke: coarsest rung, functional only, no doc")
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    device = case.resolve_device(args.device)

    ladder = json.loads((HERE / "out" / "result.json").read_text())
    committed = {r["resolution_H_over_dx"]: r
                 for r in ladder["constant_ladder"]["runs"]}
    u_num = None
    unc_path = HERE / "out" / "uncertainty.json"
    if unc_path.is_file():
        unc = json.loads(unc_path.read_text())
        u_num = ((unc.get("r1_quantities") or {})
                 .get("r1_steady_profile_rms", {}).get("U_num_percent"))

    rungs = []
    for r in args.resolutions:
        print(f"[steady_window] H/dx={r}: integrating to t={args.t_end} s ...",
              flush=True)
        raw = integrate(r, device, t_end=args.t_end)
        rungs.append(analyse_rung(raw, committed.get(r), u_num))
        g = rungs[-1]
        print(f"[steady_window]   A={g['window_A']['mean_percent']:.4f} %  "
              f"B={g['window_B']['mean_percent']:.4f} %  "
              f"U_input(between)={g['u_input_percent_of_scale']:.4f} %  "
              f"({raw['wall_clock_s']:.0f}s)", flush=True)

    gates = {}
    for g in rungs:
        gates.update({f"{k}@H/dx={g['resolution_H_over_dx']}": v
                      for k, v in g["gates"].items()})
    payload = {
        "case": "CASE-POISEUILLE-SPH",
        "generated_by": "benchmarks/case_poiseuille/steady_window.py",
        "question": ("is the committed 0.80..1.00 s R1 window stationary, i.e. is "
                     "its sample scatter a repeatability rather than residual fill?"),
        "windows": {"A": list(WINDOW_A), "B": list(WINDOW_B),
                    "sample_dt": SAMPLE_DT, "t_end": args.t_end},
        "thresholds": {"S1_reproduce_percent": S1_REPRODUCE_PERCENT,
                       "S3_fill_ratio": S3_FILL_RATIO,
                       "tolerance_percent": TOL_PERCENT},
        "u_num_percent_from_uncertainty_artifact": u_num,
        "rungs": rungs,
        "gates": gates,
        "all_gates_pass": all(v for v in gates.values() if v is not None),
        "honest_limits": [
            "two rungs (H/dx = 12, 16) with one separation (0.8 s = 7.9 e-folding "
            "times of the slowest start-up mode): the test falsifies *fill* as an "
            "explanation of the scatter, it cannot resolve a drift slower than the "
            "window's own 0.2 s",
            "the 30 % window-to-window scatter reduction expected from averaging 11 "
            "samples instead of one is NOT claimed here: the graded functional is "
            "already the window mean, so the between-window half range is the "
            "repeatability of that mean and is smaller than the single-sample half "
            "range by construction -- the single-sample value stays the conservative "
            "choice used by uncertainty.py",
            "no acoustic (compressibility) scale separation is attempted: the "
            "instantaneous scatter is WCSPH particle noise, not resolved physics",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"gates: {gates}")
    print(f"all_gates_pass: {payload['all_gates_pass']}")

    if not args.no_doc:
        try:
            write_doc(payload)
            write_figure(payload)
            print(f"doc: {DOC.relative_to(REPO)}")
        except Exception as exc:  # pragma: no cover - doc is best effort
            print(f"[doc] skipped: {exc}", file=sys.stderr)
    return 0 if payload["all_gates_pass"] else 1


# --------------------------------------------------------------------------- #
# doc + figure
# --------------------------------------------------------------------------- #
def write_doc(payload: dict) -> Path:
    L: list[str] = []
    A = L.append
    A("# CASE-POISEUILLE-SPH — is the steady window stationary?")
    A("")
    A("*generated by `benchmarks/case_poiseuille/steady_window.py` from "
      "`out/steady_window.json`; numbers are read from the artifact, not typed.*")
    A("")
    A("## 1. Why")
    A("")
    A(payload["question"])
    A("")
    A("The R1 steady functional is an error (RMS relative deviation from the "
      "parabola), and its `U_input` is the half range of the in-run samples. If the "
      "channel were still filling during the window, that half range would be a "
      "bound on a transient, not a repeatability, and gate U3b would be entitled to "
      "fail. The residual fill decays with the slowest Boussinesq mode, so moving "
      "the second window six e-folding times later separates the two hypotheses.")
    A("")
    A("## 2. How")
    A("")
    A(f"* windows: A = {payload['windows']['A']} s (the committed one), "
      f"B = {payload['windows']['B']} s; samples every "
      f"{payload['windows']['sample_dt']} s, integration to "
      f"t = {payload['windows']['t_end']} s from rest under constant forcing")
    A("* the functional is `100 * RMS((u - u_parabola)/u_parabola)` on "
      "`0.15 <= y/H <= 0.85`, computed by the same formula as `run.evaluate_constant`")
    A("* per window: mean, half range; across windows: half range of the two means "
      "(the repeatability an R2-style estimator would use)")
    A("* trend test: OLS slope of the functional against time inside each window")
    A("")
    A("## 3. Results")
    A("")
    A("| H/dx | mean A [%] | mean B [%] | half range A [pp] | half range B [pp] "
      "| between-window [pp] | between / mean [%] | U_V [%] |")
    A("|---|---|---|---|---|---|---|---|")
    for g in payload["rungs"]:
        A(f"| {g['resolution_H_over_dx']} "
          f"| {g['window_A']['mean_percent']:.4f} "
          f"| {g['window_B']['mean_percent']:.4f} "
          f"| {g['window_A']['half_range_percent_of_scale']:.4f} "
          f"| {g['window_B']['half_range_percent_of_scale']:.4f} "
          f"| {g['between_window']['half_range_percent_of_scale']:.4f} "
          f"| {g['between_window']['half_range_percent_of_mean']:.2f} "
          f"| {g['U_V_percent']:.3f} |")
    A("")
    A("Trend tests (a drift would show as a large |t|):")
    A("")
    A("| H/dx | window | slope [pp/s] | t | dof |")
    A("|---|---|---|---|---|")
    def _fmt(v, spec=".4f"):
        return format(v, spec) if isinstance(v, (int, float)) else "n/a"

    for g in payload["rungs"]:
        for name in ("trend_window_A", "trend_window_B"):
            tr = g[name]
            A(f"| {g['resolution_H_over_dx']} | {name[-1]} "
              f"| {_fmt(tr.get('slope_per_s'))} | {_fmt(tr.get('t_stat'), '.2f')} "
              f"| {tr.get('dof')} |")
    A("")
    fill = payload["rungs"][0]["fill"]
    A(f"Fill bound: decay rate {fill['decay_rate_1_over_s']:.3f} 1/s "
      f"(e-folding {fill['e_folding_s']:.4f} s), separation "
      f"{fill['separation_s']:.2f} s -> predicted surviving ratio "
      f"{fill['predicted_fill_ratio']:.3e}.")
    A("")
    A("## 4. Gates")
    A("")
    for k, v in payload["gates"].items():
        A(f"* `{k}`: {'PASS' if v else 'FAIL'}")
    A("")
    A("## 5. Error statement")
    A("")
    for g in payload["rungs"]:
        s1 = g.get("S1_deviation_percent")
        head = (f"* H/dx={g['resolution_H_over_dx']}: S1 deviation from the "
                f"committed window "
                + (f"{s1:.3f}" if s1 is not None else "n/a")
                + " % (threshold ")
        A(head
          + f"{payload['thresholds']['S1_reproduce_percent']} %); "
          + f"U_input(between) = {g['u_input_percent_of_scale']:.3f} pp, "
          + f"U_V = {g['U_V_percent']:.3f} % vs 5 % acceptance -> "
          + ("supports" if g['gates']['S4_between_window_band_supports_5pct']
             else "does not support"))
    A("")
    A("## 6. Limits")
    A("")
    for lim in payload["honest_limits"]:
        A(f"* {lim}")
    A("")
    A("## 7. Figure")
    A("")
    A("![steady window](../docs/figures/case_poiseuille_steady_window.png)")
    A("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")
    return DOC


def write_figure(payload: dict) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[figure] skipped: {exc}", file=sys.stderr)
        return None
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for g in payload["rungs"]:
        ax.plot(g["sample_times"], g["functional_percent"], "o-", ms=3.5, lw=1.0,
                label=f"H/dx={g['resolution_H_over_dx']}")
        for span, style in ((WINDOW_A, "--"), (WINDOW_B, ":")):
            ax.axvline(span[0], color="grey", ls=style, lw=0.8)
            ax.axvline(span[1], color="grey", ls=style, lw=0.8)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel(r"steady functional  $100\,\mathrm{RMS}(\Delta u/u)$  [%]")
    ax.set_title("R1 steady window: window A (committed) vs window B (6 e-folding "
                 "times later)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


if __name__ == "__main__":
    raise SystemExit(main())