"""Amplitude-band estimator for the growth rate of the lake-at-rest departure.

Why this tool exists
--------------------
``benchmarks/lake_at_rest/stability.py`` measures how long the still lake of
``CASE-SPH-LAKE-AT-REST`` stays at rest beyond the certified ``0.50 s``.  Beyond
the crossing time ``t*`` the departure grows exponentially and the *rate*
``lambda`` is the number that decides the interpretation: if ``lambda`` depends
on ``dt`` it is an integrator artefact, if it depends on the seed it is not a
linear instability, if it depends on a rebuilt neighbour list it is a pairing
artefact.

The shipped study estimated ``lambda`` by least squares over a **single absolute
amplitude window** declared once for every arm (``1e-10 <= max|v|/c0 <= 1e-5``).
That window is wrong for any arm that *starts* moving: a seeded arm enters the
window at ``t = dt`` and spends most of its samples in the pre-asymptotic
transient, so the fitted "rate" is the transient's average slope, not the
exponential rate.  Measured on the frozen artefact: the two seed arms return
``lambda = 0.887`` (R^2 = 0.29) and ``lambda = 3.72`` (R^2 = 0.58) against
``lambda = 16.30`` (R^2 = 1.00) for the base arm.  A control that quotes those
numbers as evidence of seed-independence is quoting an unreliable fit -- the
gate in ``tests/test_lake_at_rest_stability.py`` that compared them (declared
window ``0.90 ... 1.10``) therefore failed.  The failure is real: it is the
*estimator*, not the physics, that is broken.

The estimator used here
-----------------------
Fit inside the **exponential band**, declared a priori and identically for every
arm:

    BAND_LOW = 1e-6      seven decades above the measured roundoff floor
                         (``max|v|/c0`` at rest is ``2.6e-14``, the equilibrium
                         residual ``max|a|/g`` is ``1e-12 .. 1e-13``) and above
                         any start-up transient;
    BAND_HIGH = 1e-3     one decade below the measured saturation amplitude of
                         the base arm (``max|v|/c0 = 8.6e-3`` over 2.5 s) and of
                         the coarse lattice (``1.6e-3``).

Within that band the departure is exponential for every arm and the fits are
``R^2 >= 0.999``; in particular the two seeded arms return ``16.48`` and
``16.60`` 1/s, i.e. the seed-independence claim is **supported** once the
transient is excluded.  Nothing is re-run here: the fits are a pure function of
the raw per-sample series that the study already ships, so re-deriving them
costs nothing and cannot drift from the physics.

The tool also reports the *sensitivity* of ``lambda`` to the window
(``SENSITIVITY``) so a good rate is not sold as a window artefact: on the base
arm the four bands give ``16.39 / 16.45 / 16.47 / 16.47`` 1/s (spread < 1 %).
Every sensitivity window keeps its upper edge below the measured saturation
amplitude on purpose: a window that reaches into saturation returns the
saturation-averaged slope (measured ``8.19`` 1/s at ``1e-5 .. 1e-2``), which is
the same estimator error as the transient bias, at the other end.

Usage
-----
    PYTHONPATH=src python3 benchmarks/lake_at_rest/band_fit.py \\
        --artefact tests/data/lake_at_rest/stability.json \\
        --json tests/data/lake_at_rest/stability_band_fit.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]

#: Declared exponential band (see the module docstring for the derivation).
BAND_LOW = 1.0e-6
BAND_HIGH = 1.0e-3

#: Window the shipped study used; kept for the audit trail only.
LEGACY_WINDOW = (1.0e-10, 1.0e-5)

#: Windows used to show that the rate does not depend on the choice of window.
SENSITIVITY = ((1.0e-8, 1.0e-4), (1.0e-7, 1.0e-3), (1.0e-6, 1.0e-3),
               (3.0e-6, 3.0e-3))

#: A quoted rate must be a straight line in ``ln(amplitude)`` -- declared before
#: the run, not chosen after it.
R2_MIN = 0.98

#: Fewest samples a slope is allowed to be fitted to.
MIN_POINTS = 4

#: Controls and the band the study declared for each of them.
DT_RATIO = (0.75, 1.33)
SEED_RATIO = (0.90, 1.10)

FIT_FORM = "ln(v_over_c0) = a + lambda * t   (least squares)"


def samples_of(arm: dict) -> list[dict]:
    """Raw series of one arm, tolerating both shipped schema variants.

    ``stability.py`` moved the series from ``run.samples`` to ``samples`` when it
    started keeping the peak sample; the frozen artefact carries the arm-level
    form.  Both are read here so an old artefact stays decidable.
    """
    series = arm.get("samples")
    if series is None:
        series = arm.get("run", {}).get("samples")
    if not series:
        raise KeyError(f"arm {arm.get('arm')!r} carries no raw series")
    return series


def fit_lambda(series: list[dict], low: float, high: float,
               field: str = "v_over_c0") -> dict:
    """Least-squares exponential slope of ``field`` inside ``[low, high]``."""
    window = [(s["t"], s[field]) for s in series
              if low <= s[field] <= high and s[field] > 0.0]
    out = {"window": [low, high], "points": len(window),
           "t_from": None, "t_to": None, "lambda_per_s": None,
           "r2_exp": None, "power_exponent": None, "r2_power": None,
           "fit_form": FIT_FORM, "valid": False}
    if len(window) < MIN_POINTS:
        return out
    t = [w[0] for w in window]
    ln_v = [math.log(w[1]) for w in window]
    n = len(t)
    mean_t, mean_y = sum(t) / n, sum(ln_v) / n
    sxx = sum((ti - mean_t) ** 2 for ti in t)
    if sxx == 0.0:
        return out
    sxy = sum((ti - mean_t) * (yi - mean_y) for ti, yi in zip(t, ln_v))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_t
    ss_res = sum((yi - (intercept + slope * ti)) ** 2 for ti, yi in zip(t, ln_v))
    ss_tot = sum((yi - mean_y) ** 2 for yi in ln_v)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else None
    # Alternative hypothesis: a power law in t (same data, ln-ln fit).
    ln_t = [math.log(ti) for ti in t]
    mean_lt = sum(ln_t) / n
    sxx_p = sum((li - mean_lt) ** 2 for li in ln_t)
    slope_p = sxy_p = None
    if sxx_p:
        sxy_p = sum((li - mean_lt) * (yi - mean_y) for li, yi in zip(ln_t, ln_v))
        slope_p = sxy_p / sxx_p
    r2_p = None
    if slope_p is not None:
        ic_p = mean_y - slope_p * mean_lt
        ss_res_p = sum((yi - (ic_p + slope_p * li)) ** 2
                       for li, yi in zip(ln_t, ln_v))
        r2_p = 1.0 - ss_res_p / ss_tot if ss_tot else None
    out.update({"points": n, "t_from": min(t), "t_to": max(t),
                "lambda_per_s": slope, "r2_exp": r2,
                "power_exponent": slope_p, "r2_power": r2_p,
                "valid": bool(r2 is not None and r2 >= R2_MIN)})
    return out


def arm_record(arm: dict) -> dict:
    """Band fit, legacy fit and window sensitivity of one arm."""
    series = samples_of(arm)
    band = fit_lambda(series, BAND_LOW, BAND_HIGH)
    legacy = fit_lambda(series, *LEGACY_WINDOW)
    spread = [fit_lambda(series, lo, hi)["lambda_per_s"] for lo, hi in SENSITIVITY]
    finite = [v for v in spread if v is not None]
    run = arm.get("run", {})
    return {
        "arm": arm["arm"],
        "seed_in_dx": run.get("seed_in_dx"),
        "dt": run.get("dt"),
        "dx": arm.get("config", {}).get("dx"),
        "max_v_over_c0": run.get("max_v_over_c0"),
        "samples": len(series),
        "band": band,
        "legacy": legacy,
        "sensitivity": {"windows": [list(w) for w in SENSITIVITY],
                        "lambdas": spread,
                        "spread_rel": (max(finite) - min(finite)) / abs(
                            sorted(finite)[len(finite) // 2]) if finite else None},
        "legacy_is_a_rate": bool(legacy["valid"]),
    }


def _ratio(num, den):
    if num is None or den in (None, 0):
        return None
    return num / den


def build(payload: dict) -> dict:
    """Re-derive every rate from the frozen raw series and grade the controls."""
    arms = [arm_record(arm) for arm in payload["arms"]]
    by_name = {entry["arm"]: entry for entry in arms}

    def lam(name):
        return by_name.get(name, {}).get("band", {}).get("lambda_per_s")

    base = lam("base")
    ratios = {
        "dt_half_over_base": _ratio(lam("courant_half"), base),
        "seed_1e-6_over_seed_1e-8": _ratio(lam("seed_1e-6"), lam("seed_1e-8")),
        "pairs_rebuilt_over_base": _ratio(lam("pairs_rebuilt"), base),
        "resolution_dx025_over_base": _ratio(lam("resolution_dx025"), base),
        "saturation_over_base": _ratio(lam("saturation_8s"), base),
    }
    verdict = {
        "tolerance": payload.get("tolerance"),
        "band_window": [BAND_LOW, BAND_HIGH],
        "lambda_base_per_s": base,
        "r2_base": by_name.get("base", {}).get("band", {}).get("r2_exp"),
        "r2_min_declared": R2_MIN,
        "ratios": ratios,
        "dt_ratio_window": list(DT_RATIO),
        "seed_ratio_window": list(SEED_RATIO),
        "dt_independent": (ratios["dt_half_over_base"] is not None
                           and DT_RATIO[0] <= ratios["dt_half_over_base"]
                           <= DT_RATIO[1]),
        "seed_independent": (ratios["seed_1e-6_over_seed_1e-8"] is not None
                             and SEED_RATIO[0]
                             <= ratios["seed_1e-6_over_seed_1e-8"]
                             <= SEED_RATIO[1]),
        "legacy_window_valid_arms": [e["arm"] for e in arms
                                     if e["legacy_is_a_rate"]],
        "legacy_window_invalid_arms": [e["arm"] for e in arms
                                       if not e["legacy_is_a_rate"]],
        "base_sensitivity_spread_rel": by_name.get("base", {}).get(
            "sensitivity", {}).get("spread_rel"),
    }
    return {
        "case": payload.get("case", "CASE-SPH-LAKE-AT-REST"),
        "kind": "band-fit re-reduction of the rest-horizon study",
        "derived_from": "tests/data/lake_at_rest/stability.json (raw series)",
        "reference": "analytic hydrostatic identity (u = 0, eta = const)",
        "producer": "benchmarks/lake_at_rest/band_fit.py",
        "band": {
            "window": [BAND_LOW, BAND_HIGH],
            "why": "lower bound is seven decades above the measured roundoff "
                   "floor (max|v|/c0 at rest 2.6e-14) and above the start-up "
                   "transient; upper bound is one decade below the measured "
                   "saturation amplitude (max|v|/c0 8.6e-3).  Declared once, "
                   "applied to every arm.",
            "r2_min": R2_MIN,
        },
        "legacy_window": list(LEGACY_WINDOW),
        "fit_form": FIT_FORM,
        "verdict": verdict,
        "arms": arms,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artefact",
                    default=str(HERE / "tests/data/lake_at_rest/stability.json"))
    ap.add_argument("--json",
                    default=str(HERE / "tests/data/lake_at_rest/"
                                       "stability_band_fit.json"))
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.artefact).read_text())
    derived = build(payload)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derived, indent=2, sort_keys=True))

    verdict = derived["verdict"]
    print(f"lambda(base) = {verdict['lambda_base_per_s']:.3f} 1/s "
          f"(R^2 = {verdict['r2_base']:.5f}, band "
          f"{BAND_LOW:g}..{BAND_HIGH:g})")
    for name, ratio in verdict["ratios"].items():
        print(f"  {name:28s} = {ratio:.4f}" if ratio is not None else
              f"  {name:28s} = -")
    print(f"  dt-independent: {verdict['dt_independent']}, "
          f"seed-independent: {verdict['seed_independent']}")
    print(f"  legacy window NOT a rate for: "
          f"{verdict['legacy_window_invalid_arms']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())