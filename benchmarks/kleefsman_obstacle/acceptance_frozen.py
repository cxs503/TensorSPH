"""Frozen acceptance scorer for ``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`` (m21324).

The case-pool rule is: relative error < 10 % on **both** the water-level peak and
the arrival time in front of the box, against Kleefsman et al. (2005) Fig. 19
(``0.370 m`` at ``t = 1.703 s``, arrival ``0.3591 s``).

``reference.json`` froze the measurement protocol (m20773):

============================  ==========================================
operator                      fixed-aperture slab gauge
aperture                      0.08 m (physical, dx-independent)
station                       0.05 m upstream of the box face (x = 2.426 m)
peak filter                   30 ms symmetric moving average
arrival threshold             0.02 m, first crossing
============================  ==========================================

Until m21324 that protocol existed only as prose in ``reference.json``: the
solver's own scorer (``tensorsph.kleefsman_obstacle._peak_metrics``) takes a
plain ``max`` of the raw series with no filter and no aperture control, and
``run.py`` defaulted to a 0.03 m aperture.  A frozen protocol that no committed
code evaluates is not a measurement, so this module is the evaluation path: it
is pure (no solver import), so the verdict is reproducible from a committed CSV
alone, and it is pinned by ``tests/test_kleefsman_acceptance_frozen.py`` on
synthetic inputs.

Two peak readings are reported, because they answer different questions:

``peak``
    the maximum of the *whole* record -- the literal reading of the case rule,
    only meaningful when the record covers the whole experimental record up to
    its maximum (the ``censored`` flag says whether it does);
``event_peak``
    the maximum in a window around the reference peak time, i.e. "does the
    simulation reproduce the *secondary* impact that produces the experimental
    maximum at 1.703 s, and how high is it".  Reported even when the record max
    is a different event (the primary impact at ~0.7 s).

The reference arrival exists in two readings of the *same* digitised curve
(``reference.json`` -> ``arrival_time_definition``): the first digitised sample
above 0.02 m (0.3591 s) and the interpolated crossing (0.3481 s), which is the
operator the solver uses.  The 3.1 % gap is a third of the 10 % tolerance, so
both are scored and a PASS must hold against both.

Usage::

    python benchmarks/kleefsman_obstacle/acceptance_frozen.py \
        --series results/peak_horizon/dx0.02_time_series.csv --t-end 2.6
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decidability as dc  # noqa: E402

FROZEN = {
    "operator": "slab",
    "column": "slab_x2.426",
    "aperture_m": 0.08,
    "station_offset_m": 0.05,
    "peak_window_s": 0.03,
    "arrival_threshold_m": 0.02,
}
EVENT_HALF_WIDTH_S = 0.30


# --------------------------------------------------------------------------- #
# pure series operators
# --------------------------------------------------------------------------- #
def moving_average(times: list[float], values: list[float],
                   window_s: float) -> list[float]:
    """Symmetric moving average over ``[t - w/2, t + w/2]``.

    The window is truncated (not zero-padded) at the ends, so the returned
    series has the same length as the input and the first/last samples are
    averages over fewer points.  Averaging over *samples inside the window*
    rather than over a fixed index count keeps the result correct for a
    non-uniform time base.
    """
    if window_s <= 0.0:
        return list(values)
    half = 0.5 * window_s
    n = len(values)
    out: list[float] = []
    lo = 0
    hi = 0
    for k in range(n):
        t = times[k]
        while lo < n and times[lo] < t - half:
            lo += 1
        while hi < n and times[hi] <= t + half:
            hi += 1
        seg = values[lo:hi]
        out.append(sum(seg) / len(seg) if seg else values[k])
    return out


def first_crossing(times: list[float], values: list[float],
                   threshold: float) -> float:
    """First upward crossing of ``threshold``, linearly interpolated."""
    for k in range(1, len(values)):
        if values[k - 1] < threshold <= values[k]:
            h0, h1 = values[k - 1], values[k]
            frac = 0.0 if h1 == h0 else (threshold - h0) / (h1 - h0)
            return times[k - 1] + frac * (times[k] - times[k - 1])
    return float("nan")


def series_peak(times: list[float], values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    k = max(range(len(values)), key=lambda i: values[i])
    return values[k], times[k]


def window_peak(times: list[float], values: list[float],
                t0: float, t1: float) -> tuple[float, float]:
    """Maximum inside ``[t0, t1]``; ``(nan, nan)`` when the window is empty."""
    best = (float("nan"), float("nan"))
    for t, v in zip(times, values):
        if t0 <= t <= t1:
            if not math.isfinite(best[0]) or v > best[0]:
                best = (v, t)
    return best


def load_series(path: str | Path, column: str = FROZEN["column"]) -> tuple[list[float], list[float]]:
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        if column not in reader.fieldnames:
            raise KeyError(f"{column!r} not in {path}: columns = {reader.fieldnames}")
        times, values = [], []
        for row in reader:
            times.append(float(row["t_s"]))
            values.append(float(row[column]))
    return times, values


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def rel_error(got: float, want: float) -> float:
    if not (math.isfinite(got) and math.isfinite(want)) or want == 0.0:
        return float("nan")
    return abs(got - want) / abs(want)


def score(times: list[float], values: list[float], ref: dict,
          t_end: float | None = None,
          window_s: float = FROZEN["peak_window_s"],
          threshold: float = FROZEN["arrival_threshold_m"]) -> dict:
    """Score one gauge series against the reference under the frozen protocol.

    ``peak`` is taken on the 30 ms-smoothed series (the filter removes the
    ballistic-spray sample that would otherwise set the peak, see m20773), and
    the arrival on the *unsmoothed* series so that the 20 mm threshold is not
    blurred by the filter.  Both choices are the ones frozen in
    ``reference.json``.
    """
    exp = ref["experimental_values"]
    tol = float(ref["tolerance"]["relative"])
    ref_peak = float(exp["peak_height_m"])
    ref_peak_t = float(exp["peak_time_s"])
    ref_arrival = float(exp["arrival_time_s"])

    smooth = moving_average(times, values, window_s)
    peak, peak_t = series_peak(times, smooth)
    arrival = first_crossing(times, values, threshold)
    ev_peak, ev_peak_t = window_peak(times, smooth,
                                     ref_peak_t - EVENT_HALF_WIDTH_S,
                                     ref_peak_t + EVENT_HALF_WIDTH_S)

    t_end_eff = float(t_end) if t_end is not None else (times[-1] if times else float("nan"))
    censor = dc.check_horizon({"t_end": t_end_eff, "peak_time_s": peak_t})

    ref_arrival_interp = float(exp.get("arrival_time_s_interpolated", ref_arrival))
    rows = []
    for name, got, want in (("peak_height", peak, ref_peak),
                            ("arrival_time", arrival, ref_arrival_interp),
                            ("arrival_time_first_above_sample", arrival, ref_arrival)):
        err = rel_error(got, want)
        rows.append({"metric": name, "ref": want, "got": got, "rel_error": err,
                     "pass": bool(math.isfinite(err) and err < tol)})
    peak_row, arrival_row = rows[0], rows[1]

    if censor["censored"]:
        verdict = "NOT_COMPARABLE"
        why = ("series maximum sits within the censoring margin of t_end: the "
               "stored peak is a lower bound on a truncated record")
    else:
        verdict = "PASS" if all(r["pass"] for r in rows) else "FAIL"
        why = ("both metrics inside tolerance" if verdict == "PASS"
               else "at least one metric outside tolerance")

    return {
        "protocol": dict(FROZEN, peak_window_s=window_s,
                         arrival_threshold_m=threshold),
        "record": {"t_start_s": times[0] if times else float("nan"),
                   "t_end_s": t_end_eff, "samples": len(times)},
        "censoring": censor,
        "peak_height_m": peak, "peak_time_s": peak_t,
        "arrival_time_s": arrival,
        "event_peak_m": ev_peak, "event_peak_time_s": ev_peak_t,
        "event_window_s": [ref_peak_t - EVENT_HALF_WIDTH_S,
                           ref_peak_t + EVENT_HALF_WIDTH_S],
        "event_peak_rel_error": rel_error(ev_peak, ref_peak),
        "reference": {"peak_height_m": ref_peak, "peak_time_s": ref_peak_t,
                      "arrival_time_s": ref_arrival,
                      "arrival_time_s_interpolated": ref_arrival_interp,
                      "arrival_definition_band_rel":
                          rel_error(ref_arrival_interp, ref_arrival),
                      "source": exp.get("source", "")},
        "tolerance_rel": tol,
        "rows": rows,
        "peak_event_mismatch": bool(math.isfinite(peak_t) and math.isfinite(ev_peak_t)
                                    and abs(peak_t - ev_peak_t) > 1.0e-9),
        "verdict": verdict,
        "why": why,
        "peak_row": peak_row,
        "arrival_row": arrival_row,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="time series CSV path")
    ap.add_argument("--column", default=FROZEN["column"])
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--window", type=float, default=FROZEN["peak_window_s"])
    ap.add_argument("--threshold", type=float, default=FROZEN["arrival_threshold_m"])
    ap.add_argument("--reference", default=str(HERE / "reference.json"))
    ap.add_argument("--out", default=None, help="write the score JSON here")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    times, values = load_series(args.series, args.column)
    result = score(times, values, ref, t_end=args.t_end,
                   window_s=args.window, threshold=args.threshold)
    result["series"] = str(args.series)
    result["column"] = args.column
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())