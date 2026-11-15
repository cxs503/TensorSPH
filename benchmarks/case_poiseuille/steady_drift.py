#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH — is the R1 steady functional stationary, or still decaying?

Motivation
----------
``steady_window.py`` compares the *level* of two windows, A = [0.80, 1.00] s (the
committed one) and B = [1.60, 1.80] s, and calls the window stationary when the
two means differ by less than the 5 % acceptance (gate ``S2``).  That test says
nothing about a *trend inside* a window, and the committed artifact
``out/steady_window.json`` does contain one: an OLS fit to window B has a
negative slope whose naive t statistic is about -4.3 on 9 samples.  Two readings
are possible and they have opposite consequences for the acceptance claim:

* **noise** — a run of low samples inside a stationary scatter; then the level
  comparison is the right test and nothing changes;
* **signal** — the R1 functional is an *error* (RMS relative deviation from the
  parabola) that is still decaying toward zero at t ~ 1.8 s, i.e. the error keeps
  *improving*.  Then the committed value at window A is an upper bound on the
  asymptotic one and the acceptance is conservative, not fragile.

This script separates the two readings with a test that does not assume
independent samples, because at 0.02 s sampling the residuals are visibly
correlated (measured lag-1 correlation is negative).

Method
------
1. **Baseline = samples strictly before window B** (`0.80 <= t < 1.60`, the 37
   committed samples).  Window B is the test window by construction of the
   earlier experiment, so it is not reused to build the null.  No sample is
   selected by looking at its value.
2. **Circular block bootstrap** of the baseline (block length ``BLOCK`` samples,
   i.e. 0.06 s, chosen to exceed the measured decorrelation scale of the
   baseline; blocks preserve whatever autocorrelation the baseline has).
   ``NBOOT`` resampled 9-sample windows give the *null* distribution of the OLS
   slope of a window of window-B length under "stationary about its own mean".
3. One-sided p value = fraction of null slopes at or below the observed window-B
   slope (the direction in which the observed drop goes), plus the two-sided
   value.  This is the dof-free version of the naive t test, so the naive and the
   bootstrap verdicts can be compared rather than one being trusted.
4. **Extended confirmation** (``--extend``): integrate R1 from rest to
   ``--t-end`` (default 3.6 s) and grade a third window C = [3.40, 3.60] s.
   D4 is *not graded* unless that artifact exists.

Predeclared gates (thresholds fixed before the extended run was executed)
------------------------------------------------------------------------
| gate | question | threshold |
|---|---|---|
| D1 | is the late-window slope negative, i.e. does the error decrease with time? | slope_B < 0 |
| D2 | is that decline distinguishable from the pre-window noise? | one-sided block-bootstrap p < 0.05 |
| D3 | does the acceptance survive the drift, whatever its cause? | committed E + |drift across window B| < 5 % |
| D4 | (needs `--extend`) does the error keep decreasing, so that window A is an upper bound? | mean_C <= mean_B |

D1/D2 are about *interpretation*, D3 is the acceptance-robustness statement, D4
is the falsification lever for "the decay stops".  A D2 pass does **not** weaken
the published U3b verdict: the graded ``U_input`` is the in-window single-sample
half range (0.216 pp at H/dx = 16), which is *larger* than the between-window
mean difference, so the band already contains this drift.

Honest-FAIL policy: if D2 or D4 fail the outcome is recorded as such and the
note says so; the committed ``out/steady_window.json`` is not rewritten.

Usage
-----
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/steady_drift.py            # post-process only, ~1 s
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/steady_drift.py --extend   # + window C, ~4 min
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

OUT_JSON = HERE / "out" / "steady_drift.json"
DOC = REPO / "docs" / "case_poiseuille_steady_drift.md"
FIG = REPO / "docs" / "figures" / "case_poiseuille_steady_drift.png"

WINDOW_A = (0.80, 1.00)
WINDOW_B = (1.60, 1.80)
WINDOW_C = (3.40, 3.60)
BASELINE = (0.80, 1.60)      # strictly before window B
SAMPLE_DT = 0.02

TOL_PERCENT = 5.0            # feature-ledger acceptance
D2_ALPHA = 0.05              # one-sided
BLOCK = 3                    # samples per bootstrap block -> 0.06 s
NBOOT = 20000
SEED = 20260211


# --------------------------------------------------------------------------- #
# series helpers
# --------------------------------------------------------------------------- #
def select(times, values, span) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = span
    sel = [i for i, t in enumerate(times) if lo - 1e-9 <= t <= hi + 1e-9]
    return np.asarray(times, float)[sel], np.asarray(values, float)[sel]


def ols(times: np.ndarray, values: np.ndarray) -> dict:
    """Slope, naive standard error and t statistic of value vs time."""
    n = times.size
    tc = times - times.mean()
    sxx = float(np.dot(tc, tc))
    slope = float(np.dot(tc, values - values.mean()) / sxx)
    intercept = float(values.mean())
    resid = values - (intercept + slope * tc)
    dof = n - 2
    se = float("nan")
    if dof > 0:
        s2 = float(np.dot(resid, resid) / dof)
        se = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    t_stat = float("nan")
    if se and se > 0:
        t_stat = slope / se
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "half_range": float((values.max() - values.min()) / 2.0),
        "slope_per_s": slope,
        "slope_se": se,
        "t_stat": t_stat,
        "drift_over_window": slope * float(times.max() - times.min()),
        "residual_sd": float(np.std(resid, ddof=2)) if dof > 0 else float("nan"),
        "residuals": resid.tolist(),
    }


def lag1(values: np.ndarray) -> float:
    x = np.asarray(values, float)
    if x.size < 4:
        return float("nan")
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def effective_t(t_stat: float, rho: float) -> dict:
    """Naive t corrected for AR(1) residual correlation (report only).

    ``se_eff = se * sqrt((1+rho)/(1-rho))``, so a positive rho makes the naive
    test optimistic and a negative rho makes it conservative.  Reported next to
    the bootstrap p value so the reader can see which way the correction goes.
    """
    if not (rho == rho) or abs(rho) >= 1.0 or not (t_stat == t_stat):
        return {"rho": rho, "t_effective": float("nan"),
                "correction": "undefined"}
    factor = math.sqrt((1.0 + rho) / (1.0 - rho))
    return {"rho": rho, "t_effective": t_stat / factor,
            "se_factor": factor,
            "correction": ("naive t is optimistic" if rho > 0 else
                           "naive t is conservative")}


def block_bootstrap_slopes(times: np.ndarray, values: np.ndarray,
                           n_test: int, block: int = BLOCK,
                           nboot: int = NBOOT, seed: int = SEED) -> np.ndarray:
    """Null distribution of an ``n_test``-sample OLS slope from ``values``.

    Circular block bootstrap: the resampled series keeps runs of ``block``
    consecutive residuals, so the null preserves the baseline's short-range
    autocorrelation instead of assuming independence.
    """
    rng = np.random.default_rng(seed)
    res = np.asarray(values, float) - float(np.mean(values))
    n = res.size
    block = max(1, min(int(block), n))
    nblocks = int(math.ceil(n_test / block))
    t_test = times[:n_test] - np.mean(times[:n_test])
    sxx = float(np.dot(t_test, t_test))
    starts = rng.integers(0, n, size=(nboot, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    sample = res[idx.reshape(nboot, -1)[:, :n_test]]      # (nboot, n_test)
    sample = sample - sample.mean(axis=1, keepdims=True)
    return (sample @ t_test) / sxx


def bootstrap_sensitivity(times, values, observed_slope: float,
                          n_test: int) -> dict:
    """How the D2 p value moves with the two declared choices.

    Neither knob was tuned: the block length and the baseline end were fixed
    before the extended run, and this table exists so a reader can see that the
    verdict does not hinge on them.
    """
    out = {"by_block": [], "by_baseline_end": []}
    for block in (1, 2, BLOCK, 5, 8):
        null = block_bootstrap_slopes(times, values, n_test=n_test, block=block)
        out["by_block"].append({"block": block,
                                "p_one_sided": float(np.mean(null <= observed_slope))})
    for end in (1.40, 1.50, 1.60):
        lo, hi = BASELINE[0], end
        sel = [i for i, t in enumerate(times) if lo - 1e-9 <= t < hi - 1e-9]
        if len(sel) < 2 * BLOCK + n_test:
            continue
        null = block_bootstrap_slopes(np.asarray(times)[sel],
                                      np.asarray(values)[sel], n_test=n_test)
        out["by_baseline_end"].append(
            {"baseline_end_s": end, "n": len(sel),
             "p_one_sided": float(np.mean(null <= observed_slope))})
    return out


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyse_series(times, values, extended: dict | None = None) -> dict:
    """Grade D1-D3 on one functional series; D4 only when ``extended`` is given."""
    t, v = np.asarray(times, float), np.asarray(values, float)
    tb, vb = select(t, v, BASELINE)
    ta, va = select(t, v, WINDOW_A)
    tB, vB = select(t, v, WINDOW_B)

    fit_b = ols(tB, vB)
    fit_a = ols(ta, va)
    rho_b = lag1(np.asarray(fit_b["residuals"]))
    rho_base = lag1(vb - vb.mean())

    null = block_bootstrap_slopes(tb, vb, n_test=tB.size)
    p_one = float(np.mean(null <= fit_b["slope_per_s"]))
    p_two = float(min(1.0, 2.0 * min(p_one, float(np.mean(null >= fit_b["slope_per_s"])))))

    gates = {
        "D1_late_window_slope_is_negative": bool(fit_b["slope_per_s"] < 0.0),
        "D2_decline_distinguishable_from_baseline_noise": bool(p_one < D2_ALPHA),
    }
    committed_e = None
    und = HERE / "out" / "uncertainty.json"
    if und.is_file():
        unc = json.loads(und.read_text())
        committed_e = (((unc.get("r1_quantities") or {})
                        .get("r1_steady_profile_rms") or {})
                       .get("validation", {}).get("E_percent"))
    bound = None
    if committed_e is not None:
        bound = abs(float(committed_e)) + abs(fit_b["drift_over_window"])
        gates["D3_acceptance_survives_the_drift"] = bool(bound < TOL_PERCENT)

    out = {
        "window_A": {k: fit_a[k] for k in
                     ("n", "mean", "half_range", "slope_per_s", "slope_se",
                      "t_stat", "drift_over_window")},
        "window_B": {k: fit_b[k] for k in
                     ("n", "mean", "half_range", "slope_per_s", "slope_se",
                      "t_stat", "drift_over_window")},
        "baseline": {"n": int(tb.size), "mean": float(vb.mean()),
                     "sd": float(vb.std(ddof=1)), "lag1": rho_base},
        "window_B_residual_lag1": rho_b,
        "window_B_effective_t": effective_t(fit_b["t_stat"], rho_b),
        "bootstrap": {"n_boot": int(null.size), "block": BLOCK,
                      "n_test": int(tB.size), "seed": SEED,
                      "p_one_sided": p_one, "p_two_sided": p_two,
                      "null_slope_median": float(np.median(null)),
                      "null_slope_p05": float(np.quantile(null, 0.05)),
                      "observed_slope": fit_b["slope_per_s"]},
        "bootstrap_sensitivity": bootstrap_sensitivity(
            tb, vb, fit_b["slope_per_s"], n_test=tB.size),
        "committed_E_percent": committed_e,
        "D3_bound_percent": bound,
        "gates": gates,
    }

    if extended is not None:
        te, ve = np.asarray(extended["sample_times"], float), \
            np.asarray(extended["functional_percent"], float)
        tc_, vc_ = select(te, ve, WINDOW_C)
        fit_c = ols(tc_, vc_) if tc_.size >= 3 else None
        out["extended"] = {
            "t_end": float(te.max()),
            "wall_clock_s": extended.get("wall_clock_s"),
            "steps": extended.get("steps"),
            "max_mach": extended.get("max_mach"),
            "resolution_H_over_dx": extended.get("resolution_H_over_dx"),
            "window_C": ({k: fit_c[k] for k in
                          ("n", "mean", "half_range", "slope_per_s",
                           "t_stat", "drift_over_window")} if fit_c else None),
        }
        if fit_c is not None:
            out["gates"]["D4_error_keeps_decreasing"] = bool(
                fit_c["mean"] <= fit_b["mean"])
            # log-linear decay rate between window means, as a description only
            out["decay"] = {
                name: {"mean_percent": m, "t_center_s": tc}
                for name, m, tc in (
                    ("A", fit_a["mean"], float(ta.mean())),
                    ("B", fit_b["mean"], float(tB.mean())),
                    ("C", fit_c["mean"], float(tc_.mean())),
                )}

    return out


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #
def write_doc(payload: dict) -> Path:
    L: list[str] = []
    A = L.append
    A("# CASE-POISEUILLE-SPH — steady functional: stationary, or still decaying?")
    A("")
    A("*generated by `benchmarks/case_poiseuille/steady_drift.py` from "
      "`out/steady_drift.json`; every number below is read from the artifact, "
      "none is typed.*")
    A("")
    A("## 1. Question")
    A("")
    A("`steady_window.py` grades the R1 steady functional by comparing the "
      "*level* of two windows (gate `S2`). A level comparison cannot see a "
      "trend inside a window, and the committed series carries one: the OLS "
      "slope of window B is negative with a naive t of "
      f"{payload['window_B']['t_stat']:.2f} on {payload['window_B']['n']} "
      "samples. If that is noise nothing changes; if it is signal the "
      "functional is an *error* that is still decaying, which makes the "
      "committed value an upper bound rather than a fragile point value.")
    A("")
    A("## 2. Method")
    A("")
    A(f"* baseline = committed samples with `{BASELINE[0]:.2f} <= t < "
      f"{BASELINE[1]:.2f}` s (n = {payload['baseline']['n']}), i.e. strictly "
      "before the test window, so no sample is picked by looking at its value;")
    A(f"* circular block bootstrap of the baseline, block = {BLOCK} samples "
      f"(0.06 s), {NBOOT} resamples, gave the null distribution of a "
      f"{payload['bootstrap']['n_test']}-sample slope under 'stationary about "
      "its own mean';")
    A("* p values are read off that null, not from a t table, because the "
      "residuals are correlated (see the lag-1 column below);")
    A("* the naive t is corrected for AR(1) residuals only for *reporting*, so "
      "the direction of the correction is visible;")
    A("* `--extend` integrates R1 from rest to t = "
      f"{payload.get('extended', {}).get('t_end', 3.6):.1f} s and grades window "
      "C = [3.40, 3.60] s, the falsification lever for 'the decay stops'.")
    A("")
    A("## 3. Results")
    A("")
    A("| window | n | mean [%] | half range [pp] | slope [pp/s] | naive t | "
      "drift across window [pp] |")
    A("|---|---|---|---|---|---|---|")
    for name in ("window_A", "window_B"):
        w = payload[name]
        A(f"| {name} | {w['n']} | {w['mean']:.4f} | {w['half_range']:.4f} | "
          f"{w['slope_per_s']:+.4f} | {w['t_stat']:.2f} | "
          f"{w['drift_over_window']:+.4f} |")
    ext = (payload.get("extended") or {}).get("window_C")
    if ext:
        A(f"| window_C | {ext['n']} | {ext['mean']:.4f} | {ext['half_range']:.4f} "
          f"| {ext['slope_per_s']:+.4f} | {ext['t_stat']:.2f} | "
          f"{ext['drift_over_window']:+.4f} |")
    A("")
    b = payload["bootstrap"]
    et = payload["window_B_effective_t"]
    A("Null model and correlation:")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| baseline lag-1 correlation | {payload['baseline']['lag1']:+.3f} |")
    A(f"| window-B residual lag-1 correlation | "
      f"{payload['window_B_residual_lag1']:+.3f} |")
    A(f"| naive t (window B) | {payload['window_B']['t_stat']:+.2f} |")
    A(f"| AR(1)-corrected t (report only) | {et['t_effective']:+.2f} "
      f"({et['correction']}) |")
    A(f"| null slope median / p05 | {b['null_slope_median']:+.4f} / "
      f"{b['null_slope_p05']:+.4f} pp/s |")
    A(f"| observed slope (window B) | {b['observed_slope']:+.4f} pp/s |")
    A(f"| block-bootstrap p (one-sided) | {b['p_one_sided']:.4f} |")
    A(f"| block-bootstrap p (two-sided) | {b['p_two_sided']:.4f} |")
    A("")
    sens = payload.get("bootstrap_sensitivity") or {}
    if sens.get("by_block"):
        A("Sensitivity to the two declared choices (the D2 verdict uses block = "
          f"{BLOCK}):")
        A("")
        A("| block [samples] | p (one-sided) |")
        A("|---|---|")
        for row in sens["by_block"]:
            A(f"| {row['block']} | {row['p_one_sided']:.4f} |")
        A("")
        if sens.get("by_baseline_end"):
            A("| baseline end [s] | n | p (one-sided) |")
            A("|---|---|---|")
            for row in sens["by_baseline_end"]:
                A(f"| {row['baseline_end_s']:.2f} | {row['n']} | "
                  f"{row['p_one_sided']:.4f} |")
            A("")
    A("## 4. Gates")
    A("")
    for k, v in payload["gates"].items():
        A(f"* `{k}`: {'PASS' if v else 'FAIL'}")
    A("")
    if payload.get("D3_bound_percent") is None:
        A("D3 is *not graded*: `out/uncertainty.json` carries no committed `E` "
          "for `r1_steady_profile_rms`, so the drift cannot be added to it.")
    else:
        A(f"D3 binds the acceptance: committed `E` (H/dx = 16) = "
          f"{payload['committed_E_percent']:.4f} % plus the drift across "
          f"window B = {abs(payload['window_B']['drift_over_window']):.4f} pp "
          f"gives {payload['D3_bound_percent']:.4f} % against the 5 % "
          "acceptance.")
    A("")
    A("## 5. Reading (honest)")
    A("")
    if payload["gates"]["D2_decline_distinguishable_from_baseline_noise"]:
        A("* The late decline is **not** reproducible by the pre-window noise, "
          "i.e. the functional is still moving at t ~ 1.8 s and the motion is "
          "toward the parabola (the error decreases). D2 alone does not say "
          "whether that motion continues past 1.8 s -- that is D4's question. "
          "What D3 already establishes is the robustness of the acceptance: "
          "committed error plus the whole observed drift stays at "
          f"{payload['D3_bound_percent']:.4f} %, i.e. "
          f"{TOL_PERCENT - payload['D3_bound_percent']:.3f} pp inside the 5 % "
          "tolerance, so neither reading (decay continues / it stops) can lift "
          "the anchor over the acceptance.")
    else:
        A("* The late decline **cannot** be separated from the pre-window noise "
          "at 5 %: the level comparison of `steady_window.py` stands on its "
          "own, and no decay is claimed.")
    if "D4_error_keeps_decreasing" in payload["gates"]:
        A("* D4 " + ("passes: window C is below window B, so the error keeps "
                     "decreasing over the extended integration and the "
                     "committed value is conservative."
                     if payload["gates"]["D4_error_keeps_decreasing"] else
                     "**fails**: window C is not below window B, so the error "
                     "does not keep decreasing; the 'upper bound' reading is "
                     "then not supported and the window-A/steady verdict must "
                     "be read as a late-time point value only."))
    else:
        A("* D4 is *not graded*: no extended artifact was produced in this run "
          "(`--extend` was not used), so nothing is claimed about the decay "
          "beyond t = 1.8 s.")
    A("")
    A("## 6. Limits")
    A("")
    A("* One resolution (H/dx = 12), one channel, laminar; this is a "
      "post-processing test of an already committed series, not a new "
      "resolution study.")
    A("* The bootstrap block length (0.06 s) is declared, not optimised: a "
      "longer block widens the null and makes D2 easier to pass. It was fixed "
      "before the extended run.")
    A("* The functional is an error (truth = 0), so 'decaying' means 'improving'; "
      "a decaying *physical* quantity would carry the opposite sign and D1 "
      "would have to be re-oriented.")
    A("")
    A("## 7. Figure")
    A("")
    A("![steady drift](../docs/figures/case_poiseuille_steady_drift.png)")
    A("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L))
    return DOC


def write_figure(payload: dict) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                        # pragma: no cover
        return None
    src = json.loads((HERE / "out" / "steady_window.json").read_text())
    series = src["rungs"][0]
    t = np.asarray(series["sample_times"], float)
    v = np.asarray(series["functional_percent"], float)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax[0].plot(t, v, "o-", ms=3.5, lw=1.0, color="C0",
               label="R1 steady functional (H/dx=12)")
    for name, span, col in (("A", WINDOW_A, "C2"), ("B", WINDOW_B, "C3")):
        lo, hi = span
        m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        ax[0].plot(t[m], v[m], "o", ms=5, color=col)
        w = payload[f"window_{name}"]
        ax[0].hlines(w["mean"], lo, hi, color=col, lw=2,
                     label=f"mean {name} = {w['mean']:.3f} %")
        ax[0].fill_between([lo, hi], w["mean"] - w["half_range"],
                           w["mean"] + w["half_range"], color=col, alpha=0.15)
    ext = payload.get("extended")
    if ext and ext.get("window_C"):
        te = np.asarray(ext["sample_times"], float)
        ve = np.asarray(ext["functional_percent"], float)
        ax[0].plot(te, ve, "s-", ms=3.0, lw=0.9, color="C4", alpha=0.8,
                   label="extended run")
        lo, hi = WINDOW_C
        m = (te >= lo - 1e-9) & (te <= hi + 1e-9)
        ax[0].plot(te[m], ve[m], "s", ms=5, color="C5")
        ax[0].hlines(ext["window_C"]["mean"], lo, hi, color="C5", lw=2,
                     label=f"mean C = {ext['window_C']['mean']:.3f} %")
    ax[0].axhline(0.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("RMS deviation from parabola [%]")
    ax[0].set_title("R1 steady functional vs time")
    ax[0].legend(fontsize=7.5, loc="upper right")
    ax[0].grid(alpha=0.3)

    b = payload["bootstrap"]
    null_p05 = b["null_slope_p05"]
    ax[1].axvline(b["observed_slope"], color="C3", lw=2,
                  label=f"observed slope {b['observed_slope']:+.3f} pp/s")
    ax[1].axvline(null_p05, color="C7", lw=1.4, ls="--",
                  label=f"null 5 % quantile {null_p05:+.3f} pp/s")
    ax[1].axvline(0.0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel("OLS slope of a 9-sample window [pp/s]")
    ax[1].set_yticks([])
    ax[1].set_title(f"block-bootstrap null (p$_{{1-sided}}$ = "
                    f"{b['p_one_sided']:.4f})")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extend", action="store_true",
                    help="also integrate R1 from rest to --t-end and grade window C")
    ap.add_argument("--t-end", type=float, default=WINDOW_C[1] + 0.02)
    ap.add_argument("--resolution", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args(argv)

    src = json.loads((HERE / "out" / "steady_window.json").read_text())
    series = src["rungs"][0]
    payload = {
        "case": "CASE-POISEUILLE-SPH",
        "generated_by": "benchmarks/case_poiseuille/steady_drift.py",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "question": ("is the R1 steady functional stationary about its own "
                     "mean, or still decaying toward the parabola?"),
        "baseline_window": list(BASELINE),
        "windows": {"A": list(WINDOW_A), "B": list(WINDOW_B),
                    "C": list(WINDOW_C)},
        "thresholds": {"tolerance_percent": TOL_PERCENT, "D2_alpha": D2_ALPHA,
                       "block_samples": BLOCK, "n_boot": NBOOT, "seed": SEED},
        "source_artifact": "benchmarks/case_poiseuille/out/steady_window.json",
    }

    extended = None
    if args.extend:
        import torch
        sys.path.insert(0, str(HERE))
        import steady_window as sw
        sys.path.insert(0, str(HERE))
        import run as case
        torch.set_num_threads(args.threads)
        device = case.resolve_device(args.device)
        print(f"[steady_drift] integrating H/dx={args.resolution} to "
              f"t={args.t_end} s ...", flush=True)
        extended = sw.integrate(args.resolution, device, t_end=args.t_end)
        print(f"[steady_drift] done in {extended['wall_clock_s']:.0f} s "
              f"({extended['steps']} steps)", flush=True)

    res = analyse_series(series["sample_times"], series["functional_percent"],
                        extended=extended)
    payload.update(res)
    if extended is not None:
        payload["extended"]["sample_times"] = extended["sample_times"]
        payload["extended"]["functional_percent"] = extended["functional_percent"]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    print(f"[steady_drift] wrote {OUT_JSON}")
    if not args.no_doc:
        print(f"[steady_drift] wrote {write_doc(payload)}")
        fig = write_figure(payload)
        if fig:
            print(f"[steady_drift] wrote {fig}")
    failed = [k for k, v in payload["gates"].items() if not v]
    if failed:
        print(f"[steady_drift] gates FAILED: {', '.join(failed)}")
        return 1
    print("[steady_drift] all graded gates pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())