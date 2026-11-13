"""Convergence analysis of the Kleefsman (2005) obstacle case ladder.

Why this module exists
----------------------
``convergence.py`` only *tabulates* the probe diagnostics at several ``dx``.  A
table is not a convergence study: without an observed order, a Richardson
extrapolate and a discretisation-uncertainty band, the "peak water height in
front of the obstacle" cannot be compared with Kleefsman's experiment at the
10 % tolerance the case pool demands -- an unresolved quantity has no
error bar, so a 10 % agreement (or disagreement) would be unfalsifiable.

Two defects are addressed here:

1. **Mixed-protocol ladders.**  ``results/resolution_study.json`` used to
   contain ``dx = 0.04`` / ``0.02`` runs advanced to ``t_end = 1.2 s`` next to
   a ``dx = 0.01`` run stopped at ``t_end = 1.0 s`` (its peak literally sits at
   ``t = 0.966 s``, i.e. the run ends inside the rise).  Sorting those by
   ``dx`` and calling the column a ladder makes the residual differences
   meaningless.  ``check_protocol`` compares a *protocol fingerprint*
   (``t_end``, artificial sound speed, probe placement, free-surface estimator,
   arrival threshold, kernel factor) and the ladder is declared
   ``INCOMPARABLE`` as soon as one entry differs.

2. **Splash-dominated peaks.**  The instantaneous maximum of the probe height
   is set by ballistic spray above the surge, not by a water level, so it is
   not a convergent functional (documented peak sequence 0.426 / 0.518 /
   >=0.377 m at ``dx`` = 0.04 / 0.02 / 0.01).  Experimental capacitance-type
   wave probes are low-pass in the same sense, so the module also evaluates a
   *gauge-equivalent* level: the height trace averaged over a symmetric window
   ``W`` about each sample (default 30 ms) before taking the peak.

The functions are pure (numpy-free, no solver import) so they can be unit
tested against synthetic series with a known order of accuracy.

Usage::

    python benchmarks/kleefsman_obstacle/convergence_analysis.py \
        --results benchmarks/kleefsman_obstacle/results --out results
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Fields that must agree across resolutions for a ladder to mean anything.
PROTOCOL_KEYS = (
    "t_end", "sound_speed", "probe_offset", "probe_halfwidth",
    "surface_particles", "surface_gap", "arrival_threshold", "h_factor",
)

# Gauge-equivalent smoothing windows [s] checked by default.
WINDOWS = (0.020, 0.030, 0.050)
DEFAULT_WINDOW = 0.030


# --------------------------------------------------------------------------- #
# pure numerics
# --------------------------------------------------------------------------- #
def _same(a, b, rtol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)


def moving_average(times: list[float], values: list[float],
                   window: float) -> list[float]:
    """Symmetric moving average of ``values`` over ``+-window/2`` in time.

    The window is *time based* rather than sample based so the same physical
    filter is applied to every rung of the ladder, even though the output
    interval changes with ``dx``/``dt``.
    """
    out: list[float] = []
    n = len(values)
    half = 0.5 * window
    for k in range(n):
        t0 = times[k]
        lo = k
        hi = k
        while lo > 0 and t0 - times[lo - 1] <= half:
            lo -= 1
        while hi < n - 1 and times[hi + 1] - t0 <= half:
            hi += 1
        seg = values[lo:hi + 1]
        out.append(sum(seg) / len(seg))
    return out


def peak_and_time(times: list[float], values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    peak = max(values)
    return peak, times[values.index(peak)]


def observed_order(f_coarse: float, f_med: float, f_fine: float,
                   r: float) -> float:
    """Observed order of accuracy from three grids with refinement ratio ``r``.

    ``p = ln(|f_med - f_coarse| / |f_fine - f_med|) / ln(r)``; ``nan`` when the
    differences vanish (converged to machine precision) or change sign
    (non-monotone sequence, i.e. outside the asymptotic range).
    """
    d1 = f_med - f_coarse
    d2 = f_fine - f_med
    if not (math.isfinite(d1) and math.isfinite(d2)):
        return float("nan")
    if d1 == 0.0 or d2 == 0.0:
        return float("nan")
    if (d1 > 0) != (d2 > 0):
        return float("nan")
    ratio = abs(d1 / d2)
    if ratio <= 0.0 or r <= 1.0:
        return float("nan")
    return math.log(ratio) / math.log(r)


def richardson(f_med: float, f_fine: float, p: float, r: float) -> float:
    """Richardson extrapolate to ``dx = 0`` from the two finest grids."""
    if not math.isfinite(p) or r <= 1.0 or (r ** p - 1.0) == 0.0:
        return float("nan")
    return f_fine + (f_fine - f_med) / (r ** p - 1.0)


def gci_fine(f_coarse: float, f_med: float, f_fine: float, p: float,
             r: float, safety: float = 1.25) -> float:
    """Roache grid convergence index of the *fine* solution (fraction)."""
    if not math.isfinite(p) or r <= 1.0 or (r ** p - 1.0) == 0.0 or f_fine == 0.0:
        return float("nan")
    return safety * abs((f_med - f_fine) / f_fine) / (r ** p - 1.0)


# --------------------------------------------------------------------------- #
# protocol guard
# --------------------------------------------------------------------------- #
def protocol(row: dict) -> dict:
    return {k: row.get(k) for k in PROTOCOL_KEYS}


def check_protocol(rows: list[dict]) -> list[dict]:
    """Return one entry per row whose protocol fingerprint differs from the first.

    An empty list means the rows form a legitimate ladder.
    """
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return []
    base = protocol(rows[0])
    out = []
    for r in rows[1:]:
        p = protocol(r)
        diff = {k: [base[k], p[k]] for k in PROTOCOL_KEYS if not _same(base[k], p[k])}
        if diff:
            out.append({"dx": r.get("dx"), "differs": diff})
    return out


# --------------------------------------------------------------------------- #
# ladder analysis
# --------------------------------------------------------------------------- #
def ladder(rows: list[dict]) -> list[dict]:
    """Sort comparable rows coarse -> fine and report refinement ratios."""
    rows = sorted([r for r in rows if r.get("dx")], key=lambda r: -float(r["dx"]))
    for a, b in zip(rows, rows[1:]):
        a["r_next"] = float(a["dx"]) / float(b["dx"])
    return rows


def analyze_metric(rows: list[dict], key: str, safety: float = 1.25) -> dict:
    """Order / extrapolate / GCI for one metric across the ladder.

    ``rows`` must already be sorted coarse -> fine.  The stationarity and GCI
    diagnostics are taken on the finest triple with a uniform refinement ratio.
    """
    vals = [float(r.get(key, float("nan"))) for r in rows]
    res = {"metric": key, "values": vals, "dx": [float(r["dx"]) for r in rows],
           "orders": [], "verdict": "INSUFFICIENT_LEVELS"}
    if len(rows) < 3 or any(math.isnan(v) for v in vals):
        return res
    for k in range(len(vals) - 2):
        d1 = float(rows[k]["dx"]) / float(rows[k + 1]["dx"])
        d2 = float(rows[k + 1]["dx"]) / float(rows[k + 2]["dx"])
        # p = ln(|f_med-f_coarse|/|f_fine-f_med|)/ln(r) needs the *consecutive*
        # refinement ratio, and the triplet only forms a ladder when the two
        # ratios agree (uniform refinement).
        r = d1 if abs(d1 - d2) <= 1e-6 * max(d1, d2) else float("nan")
        p = observed_order(vals[k], vals[k + 1], vals[k + 2], r)
        res["orders"].append({
            "dx_triplet": [float(rows[k]["dx"]), float(rows[k + 1]["dx"]),
                           float(rows[k + 2]["dx"])],
            "r": r, "r_uniform": bool(math.isfinite(r)), "p": p,
            "richardson": richardson(vals[k + 1], vals[k + 2], p, r),
            "gci_fine": gci_fine(vals[k], vals[k + 1], vals[k + 2], p, r, safety),
        })
    last = res["orders"][-1]
    res["richardson"] = last["richardson"]
    res["gci_fine"] = last["gci_fine"]
    res["p_finest"] = last["p"]
    monotone = all((b - a) == 0 or (b - a) * (vals[1] - vals[0]) > 0
                   for a, b in zip(vals, vals[1:]))
    res["monotone"] = bool(monotone)
    if math.isnan(last["p"]) or not monotone:
        res["verdict"] = "NOT_CONVERGED"
    elif math.isnan(res["gci_fine"]) or res["gci_fine"] > 0.10:
        res["verdict"] = "NOT_CONVERGED"
    else:
        res["verdict"] = "CONVERGED"
    return res


def load_series(results_dir: Path, dx: float) -> tuple[list[float], list[float]]:
    """Read ``dx<tag>_time_series.csv`` and return ``(times, heights)``."""
    for cand in (f"dx{dx:g}_time_series.csv", f"dx{dx:.3f}_time_series.csv"):
        p = results_dir / cand
        if p.exists():
            times, heights = [], []
            with p.open() as fh:
                next(fh, None)
                for line in fh:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        times.append(float(parts[0]))
                        heights.append(float(parts[1]))
            return times, heights
    return [], []


def metrics_from_series(times: list[float], heights: list[float],
                        window: float = DEFAULT_WINDOW,
                        threshold: float = 0.02,
                        window_t: float | None = None) -> dict:
    """Probe metrics recomputed from a stored trace.

    ``window_t`` truncates the trace to a common horizon before any metric is
    taken.  That is how rungs advanced to different ``t_end`` are made
    comparable: the solver protocol stays untouched, only the observation
    window is equalised (and the caller must check that no peak sits on the
    truncation edge -- ``censored`` reports that).
    """
    if not times:
        return {"raw_peak": float("nan"), "raw_peak_time": float("nan"),
                "gauge_peak": float("nan"), "arrival_time_s": float("nan"),
                "censored": True, "n_samples": 0}
    if window_t is not None:
        keep = [k for k, t in enumerate(times) if t <= window_t + 1e-12]
        times = times[:len(keep)]
        heights = heights[:len(keep)]
    raw_peak, raw_time = peak_and_time(times, heights)
    gauge_peak, gauge_time = peak_and_time(times, moving_average(times, heights, window))
    arrival = float("nan")
    for k in range(1, len(heights)):
        if heights[k - 1] < threshold <= heights[k]:
            h0, h1 = heights[k - 1], heights[k]
            frac = 0.0 if h1 == h0 else (threshold - h0) / (h1 - h0)
            arrival = times[k - 1] + frac * (times[k] - times[k - 1])
            break
    span = times[-1] - times[0]
    return {"raw_peak": raw_peak, "raw_peak_time": raw_time,
            "gauge_peak": gauge_peak, "gauge_peak_time": gauge_time,
            "arrival_time_s": arrival, "n_samples": len(times),
            "horizon_s": times[-1],
            "censored": bool(raw_time >= times[-1] - 0.05 * span
                             or gauge_time >= times[-1] - 0.05 * span)}


def analyse_rows(rows: list[dict], results_dir: Path,
                 windows=WINDOWS, window_t: float | None = None) -> dict:
    """Full ladder analysis: protocol guard + raw/filtered peak + arrival time.

    With ``window_t`` set, every rung's metrics are recomputed from its stored
    time series inside ``[0, window_t]`` and the *horizon* stops being part of
    the protocol fingerprint (it is equalised by construction); the solver-side
    keys (sound speed, probe, surface estimator, threshold, kernel factor) still
    have to agree.
    """
    if window_t is not None:
        rows = [dict(r, t_end=window_t) for r in rows]
    mismatch = check_protocol(rows)
    report: dict = {
        "protocol": protocol(rows[0]) if rows else {},
        "protocol_mismatch": mismatch,
        "comparable": not mismatch,
        "dx": sorted([float(r["dx"]) for r in rows], reverse=True),
        "metrics": {},
    }
    if mismatch:
        report["verdict"] = "INCOMPARABLE"
        return report

    ordered = ladder(rows)
    report["truncated_to_s"] = window_t
    metrics: dict[str, dict] = {}

    if window_t is not None:
        series = {}
        for r in ordered:
            times, heights = load_series(results_dir, float(r["dx"]))
            series[float(r["dx"])] = metrics_from_series(
                times, heights, DEFAULT_WINDOW,
                float(r.get("arrival_threshold") or 0.02), window_t)
        ordered = [dict(r, **{"arrival_time_s": series[float(r["dx"])]["arrival_time_s"],
                              "peak_height_m": series[float(r["dx"])]["raw_peak"]})
                   for r in ordered]
        report["series_metrics"] = {f"{k:g}": v for k, v in series.items()}
        report["censored"] = any(v["censored"] for v in series.values())

    metrics["arrival_time_s"] = analyze_metric(ordered, "arrival_time_s")
    metrics["peak_height_m_raw"] = analyze_metric(ordered, "peak_height_m")

    # gauge-equivalent peaks from the stored time series
    for w in windows:
        vals = []
        for r in ordered:
            times, heights = load_series(results_dir, float(r["dx"]))
            if not times:
                vals.append(float("nan"))
                continue
            m = metrics_from_series(times, heights, w,
                                    float(r.get("arrival_threshold") or 0.02), window_t)
            vals.append(m["gauge_peak"])
        rows_w = [dict(r, **{"_gauge": v}) for r, v in zip(ordered, vals)]
        key = f"gauge_level_peak_w{w * 1000:03.0f}ms"
        m = analyze_metric(rows_w, "_gauge")
        m["metric"] = key
        m["window_s"] = w
        metrics[key] = m
    report["metrics"] = metrics

    verdicts = [m["verdict"] for m in metrics.values()]
    if all(v == "CONVERGED" for v in verdicts):
        report["verdict"] = "CONVERGED"
    elif any(v == "CONVERGED" for v in verdicts):
        report["verdict"] = "PARTIALLY_CONVERGED"
    else:
        report["verdict"] = "NOT_CONVERGED"
    return report


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--study", default="resolution_study.json")
    ap.add_argument("--out", default="convergence_ladder.json")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW)
    ap.add_argument("--window-t", type=float, default=None,
                    help="truncate every rung to this horizon [s] before scoring "
                         "(equalises runs advanced to different t_end)")
    args = ap.parse_args()
    results = Path(args.results)
    study = json.loads((results / args.study).read_text())
    windows = tuple(sorted({args.window, *WINDOWS}))
    report = analyse_rows(study["runs"], results, windows=windows,
                          window_t=args.window_t)
    report["source_study"] = args.study
    out = results / args.out
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"verdict": report["verdict"],
                      "protocol_mismatch": report["protocol_mismatch"],
                      "metrics": {k: {"values": v["values"], "p": v.get("p_finest"),
                                      "gci": v.get("gci_fine"),
                                      "verdict": v["verdict"]}
                                  for k, v in report["metrics"].items()}}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())