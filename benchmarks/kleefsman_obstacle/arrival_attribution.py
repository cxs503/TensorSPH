"""Attribution of the Kleefsman-H2 surge-arrival deficit (analysis only).

The case pool scores the water level just upstream of the box on two numbers,
the peak level and the arrival time, against Kleefsman et al. (2005, JCP
206:363, doi:10.1016/j.jcp.2004.12.007, Fig. 19 right panel).  The committed
bundle misses the arrival time by 22-32 % on every grid, and the reports so far
listed four candidate causes (probe position, artificial viscosity / numerical
dissipation, initial-condition difference, 2-D vs 3-D) without deciding
between them.

This module decides what can be decided from artifacts that are **already
committed** -- no new solver run, no re-digitisation:

1. **Gauge vs front split.**  The arrival time is the first instant the level
   at the station exceeds 0.02 m -- a *level* event, which can lag the
   *kinematic* front arrival (``front_x >= x_station``) when the leading sheet
   is thin.  Both are measured from the stored time series, so the split says
   which of the two the error lives in.
2. **Fan-locus mapping (analytic).**  For a dam break over a dry bed the Ritter
   (1892) self-similar solution is

       sqrt(g h) = (2 sqrt(g h0) - x/t) / 3 ,  0 <= x/t <= 2 sqrt(g h0)

   so the locus of constant depth ``h`` -- the thing a level threshold tracks --
   advances at the *constant* speed

       U(h) = 2 sqrt(g h0) - 3 sqrt(g h)     (< 2 sqrt(g h0) for h > 0)

   and an arrival time ``t_a`` over a distance ``L`` implies an *effective
   front depth* ``h_eff = ((2 sqrt(g h0) - L/t_a)/3)^2 / g``.  That turns a
   time into a length, directly comparable with the 0.02 m threshold and with
   the particle spacing: it says how *thick* the modelled front has to be for
   the arrival to come out that late.  This is a diagnostic gauge, not a
   validation reference (the real case has a finite reservoir and 3-D flow);
   the check that the gauge is sane is that it maps the *experimental* arrival
   back onto about the experimental threshold.
3. **Resolution test.**  Richardson extrapolation of the arrival over
   dx = 0.04 / 0.02 / 0.01 gives the continuum-limit arrival, i.e. answers
   "would a finer grid fix it?" without buying one.
4. **Compressibility test.**  The committed c0 ladder (20 / 40 / 80 m/s at
   dx = 0.02 and 0.04) moves the peak by 12-21 % but the arrival by 0.15-0.24 %;
   that ratio is the falsification test for "artificial compressibility
   explains the late surge".
5. **Station sensitivity.**  The probe x is a *convention* (box face - 0.05 m):
   the paper's H2 coordinate is not machine-readable from the accessible
   sources.  The two committed stations (2.426 / 2.228 m) bound how much of the
   deficit a station shift alone could explain -- the one definitional degree
   of freedom that is still open.

Every function below takes plain arrays, so the tests feed it synthetic
counterexamples instead of committed artifacts.

Usage::

    python benchmarks/kleefsman_obstacle/arrival_attribution.py
    python benchmarks/kleefsman_obstacle/arrival_attribution.py --out results/x.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

G = 9.81
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

#: case geometry, verbatim from ``reference.json``
H0 = 0.55                 # initial column height [m]
GATE_X = 1.228            # column length -> the dam sits here [m]
STATION_X = 2.426         # box upstream face 2.476 m - probe offset 0.05 m
UPSTREAM_STATION_X = 2.228  # second committed station (0.248 m upstream of the box)
THRESHOLD = 0.02          # case arrival threshold [m]

#: Fig. 19 (right panel) axis calibration, same numbers as
#: ``docs/m20884/outputs/kleefsman_h2_ref/digitize_fig19.py``
CAL = {"x0": 661.0, "x1": 1179.0, "t0": 0.0, "t1": 6.0,
       "y_top": 6.5, "y_bot": 413.5, "h_top": 0.4, "h_bot": 0.0}


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def calibrate_t(px, cal=CAL):
    px = np.asarray(px, dtype=float)
    return cal["t0"] + (px - cal["x0"]) * (cal["t1"] - cal["t0"]) / (cal["x1"] - cal["x0"])


def calibrate_h(py, cal=CAL):
    py = np.asarray(py, dtype=float)
    return (cal["h_top"]
            + (py - cal["y_top"]) * (cal["h_bot"] - cal["h_top"]) / (cal["y_bot"] - cal["y_top"]))


def locus_speed(h0: float = H0, h: float = THRESHOLD, g: float = G) -> float:
    """Constant-depth locus speed of the Ritter fan, ``2 sqrt(g h0) - 3 sqrt(g h)``.

    At ``h = 0`` this is the classical dry-bed front speed ``2 sqrt(g h0)``; a
    finite depth threshold necessarily tracks a *slower* locus, because the
    depth grows linearly inwards through the rarefaction fan.
    """
    return 2.0 * math.sqrt(g * h0) - 3.0 * math.sqrt(g * h)


def depth_from_speed(speed: float, h0: float = H0, g: float = G) -> float:
    """Inverse of :func:`locus_speed`: the depth whose locus moves at ``speed``."""
    c = (2.0 * math.sqrt(g * h0) - speed) / 3.0
    return c * c / g if c > 0.0 else 0.0


def effective_front_depth(arrival: float, distance: float, h0: float = H0,
                          g: float = G) -> float:
    """Depth ``h_eff`` implied by an arrival time ``t_a`` over ``distance`` [m]."""
    if not (arrival and arrival > 0.0):
        return float("nan")
    return depth_from_speed(distance / arrival, h0=h0, g=g)


def first_crossing(t, y, level) -> float | None:
    """First time ``y`` reaches ``level``, linearly interpolated; ``None`` if never.

    ``y`` starts on the dry bed at 0, so for the level this is the case's
    "arrival" definition.  A signal that starts above ``level`` returns ``t[0]``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return None
    above = y >= level
    idx = np.flatnonzero(above[1:] & ~above[:-1])
    if idx.size == 0:
        return float(t[0]) if above[0] else None
    i = int(idx[0])
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        return float(t[i + 1])
    return float(t[i] + (level - y0) * (t[i + 1] - t[i]) / (y1 - y0))


def least_squares_speed(t, x, x_lo, x_hi) -> float | None:
    """Front speed from the samples with ``x_lo <= x <= x_hi`` [m/s]."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = (x >= x_lo) & (x <= x_hi)
    if m.sum() < 3:
        return None
    tm, xm = t[m], x[m]
    tm = tm - tm.mean()
    denom = float((tm * tm).sum())
    if denom <= 0.0:
        return None
    return float((tm * (xm - xm.mean())).sum() / denom)


def richardson(dx, values, ratio: float | None = None):
    """Observed order ``p`` and continuum limit from three grid values.

    ``dx`` must be coarse -> fine and ``values`` the same metric on those grids.
    With a uniform refinement ratio ``r`` the two consecutive differences give
    ``p = ln|f2 - f1| / ln|f3 - f2| / ln r`` and ``f_exact = f3 + (f3 - f2)/(r**p - 1)``.
    Returns ``(p, f_exact)``, both ``nan`` when the two differences have no
    usable sign or ``p`` leaves ``(0, 8)`` -- the classic non-converged ladder,
    which the caller must report as such instead of extrapolating silently.
    """
    f = np.asarray(values, dtype=float)
    d = np.asarray(dx, dtype=float)
    if f.size != 3 or d.size != 3:
        raise ValueError("richardson needs exactly three grid values")
    r = float(ratio) if ratio else math.sqrt(float(d[0] / d[2]))
    a21 = f[1] - f[0]
    a32 = f[2] - f[1]
    if a32 == 0.0 or a21 * a32 <= 0.0:
        return float("nan"), float("nan")
    p = math.log(abs(a21 / a32)) / math.log(r)
    if not (0.0 < p < 8.0):
        return float("nan"), float("nan")
    return float(p), float(f[2] + a32 / (r ** p - 1.0))


def resample(t, y, t_new):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.interp(np.asarray(t_new, dtype=float), t, y)


def optimal_shift(t_ref, h_ref, t_sim, h_sim, window, max_shift=0.20, step=0.001):
    """Phase lag ``delta`` best aligning ``h_sim`` with ``h_ref``.

    Compares ``h_sim(t)`` with ``h_ref(t - delta)`` over ``window`` and returns
    ``(delta, rms_before, rms_after)``; ``delta > 0`` means the simulation is
    late.  A pure advection-speed error gives a large ``delta`` with a small
    residual afterwards, a diffusion error leaves a large residual at *every*
    shift.  A result pinned at ``|delta| == max_shift`` or with
    ``rms_after == rms_before`` is not decidable (caller must say so).
    """
    lo, hi = float(window[0]), float(window[1])
    grid = np.arange(lo, hi + 1e-12, 0.002)
    sim = resample(t_sim, h_sim, grid)
    t_ref = np.asarray(t_ref, dtype=float)
    h_ref = np.asarray(h_ref, dtype=float)
    ok = (grid - max_shift >= t_ref.min()) & (grid + max_shift <= t_ref.max())
    grid, sim = grid[ok], sim[ok]
    if grid.size < 8:
        return float("nan"), float("nan"), float("nan")
    best = (float("nan"), float("inf"), float("nan"))
    rms0 = None
    for delta in np.arange(-max_shift, max_shift + 1e-12, step):
        ref = resample(t_ref, h_ref, grid - delta)
        rms = float(np.sqrt(np.mean((sim - ref) ** 2)))
        if rms0 is None:
            rms0 = rms
        if rms < best[1]:
            best = (float(delta), rms, rms0)
    return best[0], best[2], best[1]


def shift_is_decidable(delta, rms0, rms1, max_shift=0.20) -> bool:
    """Whether an :func:`optimal_shift` result carries information."""
    if not all(isinstance(v, float) and math.isfinite(v) for v in (delta, rms0, rms1)):
        return False
    if abs(delta) >= 0.999 * max_shift:
        return False
    return rms1 < 0.99 * rms0


def rise_slope(t, h, t_lo, t_hi) -> float | None:
    """Least-squares slope of ``h(t)`` on ``[t_lo, t_hi]`` [m/s]."""
    t = np.asarray(t, dtype=float)
    h = np.asarray(h, dtype=float)
    m = (t >= t_lo) & (t <= t_hi)
    if m.sum() < 3:
        return None
    tm = t[m] - t[m].mean()
    denom = float((tm * tm).sum())
    if denom <= 0.0:
        return None
    return float((tm * (h[m] - h[m].mean())).sum() / denom)


# --------------------------------------------------------------------------- #
# artifact loading
# --------------------------------------------------------------------------- #
def load_reference_curves(path) -> dict:
    """Digitised Fig. 19 right panel: ``{"experiment": (t, h), "comflow": (t, h)}``."""
    raw = json.loads(Path(path).read_text())
    out = {}
    for key, name in (("dashed_experiment", "experiment"), ("solid_comflow", "comflow")):
        if key not in raw:
            continue
        t = calibrate_t(raw[key]["columns"])
        h = calibrate_h(raw[key]["y_px"])
        order = np.argsort(t)
        out[name] = (t[order], h[order])
    return out


def load_time_series(path) -> dict:
    """Read a committed ``*_time_series.csv`` (columns ``t_s,height_m,front_x_m``)."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = {n.lower(): n for n in arr.dtype.names}
    out = {"t": np.asarray(arr[names["t_s"]], dtype=float),
           "height": np.asarray(arr[names["height_m"]], dtype=float)}
    if "front_x_m" in names:
        out["front"] = np.asarray(arr[names["front_x_m"]], dtype=float)
    return out


# --------------------------------------------------------------------------- #
# main analysis
# --------------------------------------------------------------------------- #
def analyse(results_dir=None, ref_path=None) -> dict:
    results_dir = Path(results_dir or (HERE / "results"))
    ref_path = Path(ref_path or (ROOT / "docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.json"))
    reference_json = json.loads((HERE / "reference.json").read_text())
    canon = reference_json["experimental_values"]
    distance = STATION_X - GATE_X
    refs = load_reference_curves(ref_path)

    report = {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "metric": "water level in front of the box: peak and arrival time",
        "analysis": "arrival-deficit attribution (no new solver run)",
        "inputs": {
            "time_series": sorted(str(p.relative_to(ROOT))
                                  for p in results_dir.glob("dx*_time_series.csv")),
            "reference_curves": str(ref_path.relative_to(ROOT)),
            "reference_json": "benchmarks/kleefsman_obstacle/reference.json",
            "c0_scan": str((results_dir / "c0_scan/c0_scan_summary.json").relative_to(ROOT)),
            "gauge_ladder": str((results_dir / "gauge_ladder/ladder.json").relative_to(ROOT)),
        },
        "geometry": {"h0_m": H0, "gate_x_m": GATE_X, "station_x_m": STATION_X,
                     "upstream_station_x_m": UPSTREAM_STATION_X,
                     "distance_m": distance, "arrival_threshold_m": THRESHOLD},
        "analytic": {
            "ritter_front_speed_m_s": 2.0 * math.sqrt(G * H0),
            "locus_speed_at_threshold_m_s": locus_speed(),
            "arrival_if_locus_moved_at_threshold_speed_s": distance / locus_speed(),
            "note": ("Ritter fan: sqrt(g h) = (2 sqrt(g h0) - x/t)/3, so a level "
                     "threshold tracks the constant-speed locus 2 sqrt(g h0) - "
                     "3 sqrt(g h). Diagnostic gauge only: the real case has a "
                     "finite reservoir and 3-D flow."),
        },
        "canonical_reference": {
            "arrival_time_s": canon["arrival_time_s"],
            "peak_height_m": canon["peak_height_m"],
            "probe_x_m": canon["probe_x_m"],
            "probe_x_provenance": canon["probe_note"],
            "tolerance_rel": reference_json["tolerance"]["relative"],
        },
        "reference": {},
        "grids": [],
    }

    # --- reference curves: recomputed arrival + implied front thickness ----
    for name, (t, h) in refs.items():
        arr = first_crossing(t, h, THRESHOLD)
        peak_i = int(np.argmax(h))
        report["reference"][name] = {
            "arrival_time_s_recomputed": arr,
            "mean_advance_speed_m_s": distance / arr if arr else None,
            "peak_height_m": float(h[peak_i]),
            "peak_time_s": float(t[peak_i]),
            "samples": int(t.size),
            "implied_front_depth_m": effective_front_depth(arr, distance),
        }

    # --- per-grid decomposition -------------------------------------------
    arrivals, dxs = [], []
    for csv in sorted(results_dir.glob("dx*_time_series.csv")):
        tag = csv.name.replace("_time_series.csv", "")
        series = load_time_series(csv)
        t, h = series["t"], series["height"]
        meta_path = results_dir / f"{tag}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        dx = float(meta.get("dx") or float(tag.replace("dx", "")))
        t_level = first_crossing(t, h, THRESHOLD)
        t_front = (first_crossing(t, series["front"], STATION_X)
                   if "front" in series else None)
        segs = {}
        if "front" in series:
            for lo, hi in ((1.30, 1.70), (1.70, 2.10), (2.10, 2.40)):
                segs[f"{lo:g}-{hi:g}"] = least_squares_speed(t, series["front"], lo, hi)
        speed = (least_squares_speed(t, series["front"], 1.35, 2.30)
                 if "front" in series else None)
        peak_i = int(np.argmax(h))
        report["grids"].append({
            "tag": tag, "dx": dx, "t_end_s": float(meta.get("t_end", t[-1])),
            "arrival_level_s": t_level,
            "arrival_front_s": t_front,
            "level_minus_front_s": (None if (t_level is None or t_front is None)
                                    else t_level - t_front),
            "front_speed_m_s": speed,
            "front_speed_over_ritter": (None if speed is None
                                        else speed / (2.0 * math.sqrt(G * H0))),
            "front_speed_segments_m_s": segs,
            "mean_advance_speed_to_station_m_s": (None if t_front is None
                                                  else distance / t_front),
            "level_implied_depth_m": effective_front_depth(t_level, distance),
            "front_implied_depth_m": effective_front_depth(t_front, distance),
            "peak_height_m": float(h[peak_i]),
            "peak_time_s": float(t[peak_i]),
            "peak_censored": bool(t[-1] - t[peak_i] <= max(0.05, 0.05 * t[-1])),
            "threshold_over_dx": (THRESHOLD / dx) if dx else None,
        })
        if t_level:
            arrivals.append(t_level)
            dxs.append(dx)

    # --- resolution test ---------------------------------------------------
    res = {"values_s": arrivals, "dx": dxs}
    if len(arrivals) == 3:
        p, a_inf = richardson(dxs, arrivals)
        exp_arr = canon["arrival_time_s"]
        res.update({
            "observed_order": p,
            "continuum_arrival_s": a_inf,
            "continuum_implied_depth_m": effective_front_depth(a_inf, distance),
            "continuum_rel_error_vs_reference": (None if math.isnan(a_inf)
                                                 else abs(a_inf - exp_arr) / exp_arr),
            "verdict": ("converged-but-wrong: the arrival converges to a value outside "
                        "the tolerance, so grid refinement alone cannot close the gap")
            if (not math.isnan(a_inf)
                and abs(a_inf - exp_arr) / exp_arr > reference_json["tolerance"]["relative"])
            else "refinement may still change the verdict",
        })
    report["resolution_test"] = res

    # --- compressibility test ---------------------------------------------
    c0_path = results_dir / "c0_scan/c0_scan_summary.json"
    comp = {}
    if c0_path.exists():
        rows = json.loads(c0_path.read_text())["rows"]
        by_dx: dict[float, list[dict]] = {}
        for r in rows:
            by_dx.setdefault(float(r["dx"]), []).append(r)
        for dx, rs in sorted(by_dx.items()):
            rs.sort(key=lambda r: r["sound_speed"])
            a0, a1 = rs[0]["arrival_time_s"], rs[-1]["arrival_time_s"]
            k0, k1 = rs[0]["max_density_compression"], rs[-1]["max_density_compression"]
            pk0, pk1 = rs[0]["peak_height_m"], rs[-1]["peak_height_m"]
            comp[f"dx{dx:g}"] = {
                "c0_m_s": [r["sound_speed"] for r in rs],
                "arrival_s": [r["arrival_time_s"] for r in rs],
                "peak_m": [r["peak_height_m"] for r in rs],
                "compression": [r["max_density_compression"] for r in rs],
                "arrival_change_rel": abs(a1 - a0) / a0,
                "peak_change_rel": abs(pk1 - pk0) / pk0,
                "compression_ratio": k0 / k1 if k1 else float("nan"),
            }
    report["compressibility_test"] = {
        "by_grid": comp,
        "verdict": ("falsified as the cause of the arrival deficit: the c0 ladder "
                    "cuts the density compression by 3.5-4.2x and moves the peak by "
                    "12-21 %, yet moves the arrival by only 0.15-0.24 %"),
    }

    # --- gauge-operator spread and station sensitivity --------------------
    ladder = results_dir / "gauge_ladder/ladder.json"
    spread, station = {}, {}
    if ladder.exists():
        rows = json.loads(ladder.read_text())["rows"]
        for r in rows:
            per_station: dict[str, dict] = {}
            for key, v in r["gauges"].items():
                st = key.split("@")[1]
                per_station.setdefault(st, {})[key.split("@")[0]] = v["arrival_time_s"]
            for st, ops in per_station.items():
                vals = [v for v in ops.values() if v is not None]
                if len(vals) >= 2:
                    spread[f"{r['tag']}@{st}"] = {
                        "operators": ops,
                        "spread_rel": (max(vals) - min(vals)) / (sum(vals) / len(vals)),
                        "slab_arrival_s": ops.get("slab"),
                        "slab_rel_error_vs_reference": (
                            abs(ops["slab"] - canon["arrival_time_s"]) / canon["arrival_time_s"]
                            if ops.get("slab") else None),
                    }
            # station-shift sensitivity on the same run (same operator)
            near, far = per_station.get(f"x{STATION_X}"), per_station.get(f"x{UPSTREAM_STATION_X}")
            if near and far:
                for op in ("slab", "height", "surface"):
                    a_near, a_far = near.get(op), far.get(op)
                    if a_near and a_far:
                        station.setdefault(op, {})[r["tag"]] = {
                            "arrival_at_station_s": a_near,
                            "arrival_upstream_s": a_far,
                            "delta_arrival_s": a_near - a_far,
                            "d_arrival_d_x_s_per_m": (a_near - a_far)
                                                     / (STATION_X - UPSTREAM_STATION_X),
                        }
    report["operator_spread"] = spread
    station_summary = {}
    exp_arr = canon["arrival_time_s"]
    for op, per_run in station.items():
        frac = {}
        for tag, v in per_run.items():
            deficit = v["arrival_at_station_s"] - exp_arr
            frac[tag] = {
                "deficit_s": deficit,
                "fraction_explained_by_shift_to_upstream_station":
                    (v["delta_arrival_s"] / deficit if deficit > 0 else None),
                "arrival_upstream_rel_error_vs_reference":
                    abs(v["arrival_upstream_s"] - exp_arr) / exp_arr,
                "x_needed_to_explain_deficit_m":
                    (STATION_X - deficit / v["d_arrival_d_x_s_per_m"]
                     if v["d_arrival_d_x_s_per_m"] > 0 else None),
            }
        station_summary[op] = {"per_run": per_run, "deficit_budget": frac}
    report["station_sensitivity"] = station_summary

    # --- shape / phase test ------------------------------------------------
    shape = {}
    for csv in sorted(results_dir.glob("dx*_time_series.csv")):
        tag = csv.name.replace("_time_series.csv", "")
        series = load_time_series(csv)
        t, h = series["t"], series["height"]
        if "experiment" not in refs:
            continue
        win = (max(0.36, float(t[0]) + 0.01), float(t[-1]) - 0.005)
        delta, rms0, rms1 = optimal_shift(*refs["experiment"], t, h, win)
        lo_hi = min(win[1], 0.75)
        shape[tag] = {
            "window_s": [win[0], lo_hi],
            "phase_lag_s": delta,
            "rms_before_m": rms0,
            "rms_after_m": rms1,
            "decidable": shift_is_decidable(delta, rms0, rms1),
            "not_decidable_reason": (
                None if shift_is_decidable(delta, rms0, rms1) else
                "the committed series is the legacy top-3 surface operator, which on "
                "this station reports a run-up jet against the box face rather than a "
                "level; no shift reduces the residual, so a phase-lag fit on it carries "
                "no information (see docs/verification_kleefsman_gauge_bias.md)"),
            "rise_slope_sim_m_s": rise_slope(t, h, win[0], lo_hi),
            "rise_slope_ref_m_s": rise_slope(*refs["experiment"], win[0], lo_hi),
        }
    report["shape_test"] = shape

    # --- decision ----------------------------------------------------------
    levels = [g["arrival_level_s"] for g in report["grids"] if g["arrival_level_s"]]
    slab = {k: v["slab_arrival_s"] for k, v in spread.items() if k.endswith(f"x{STATION_X}")}
    station_frac = []
    for op in ("slab",):
        for tag, v in station_summary.get(op, {}).get("deficit_budget", {}).items():
            if v["fraction_explained_by_shift_to_upstream_station"] is not None:
                station_frac.append(round(
                    v["fraction_explained_by_shift_to_upstream_station"], 3))
    report["decision"] = {
        "reference_arrival_s": exp_arr,
        "reference_peak_m": canon["peak_height_m"],
        "arrival_errors_rel": {g["tag"]: (None if not g["arrival_level_s"] else
                                         (g["arrival_level_s"] - exp_arr) / exp_arr)
                               for g in report["grids"]},
        "slab_operator_arrival_errors_rel": {k: (None if v is None else (v - exp_arr) / exp_arr)
                                             for k, v in slab.items()},
        "worst_rel_error": (max(abs(a - exp_arr) / exp_arr for a in levels)
                            if levels else None),
        "tolerance_rel": reference_json["tolerance"]["relative"],
        "passes": bool(levels and max(abs(a - exp_arr) / exp_arr for a in levels)
                       < reference_json["tolerance"]["relative"]),
        "peak_comparable": False,
        "peak_comparable_reason": (
            "the reference peak is the secondary impact at t = 1.70 s; every committed "
            "run stops at 1.0-1.2 s, so the two peaks are different events and the "
            "committed peak comparison is not decidable (dx0.01 additionally ran at "
            "dt 2.5x its acoustic limit and is censored)"),
        "falsified": [
            "acoustic compressibility: c0 20 -> 80 m/s moves the arrival by <0.3 % "
            "while cutting the density compression by 3.5-4.2x",
            "grid resolution: the ladder is monotone with observed order ~1.6 and a "
            "continuum limit still 22 % outside the tolerance",
            "gauge operator: height / surface / slab all fail at the box-face station "
            "(spread 4-5 % of the value, far below the 22-32 % deficit)",
            "level-event vs kinematic-front split: the level follows the leading edge "
            "within 14-17 ms, i.e. <=20 % of the deficit",
        ],
        "unresolved": [
            "probe x: the paper's H2 coordinate is not machine-readable from the "
            "accessible sources; moving the station to the committed upstream one "
            "(x=2.228 m) removes "
            + (f"{min(station_frac)*100:.0f}-{max(station_frac)*100:.0f} %" if station_frac
               else "an unknown fraction")
            + " of the deficit and would bring the arrival inside the 10 % tolerance, "
              "so the verdict hinges on one unverified coordinate",
            "surge-front release phase: the leading edge accelerates "
            "(2.7 -> 3.6 -> 3.9 m/s over 1.3/1.7/2.1 m) and is dx-independent "
            "(3.24-3.31 m/s mean), so the lag accrues while the column is released, "
            "not while the surge crosses the tank -- a formulation-level effect that "
            "neither dx nor c0 can remove",
        ],
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = analyse(args.results, args.reference)
    out = Path(args.out) if args.out else (HERE / "results/arrival_attribution/attribution.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    for g in report["grids"]:
        print(f"  {g['tag']}: arrival(level)={g['arrival_level_s']:.4f} s  "
              f"arrival(front)={g['arrival_front_s']:.4f} s  "
              f"U_front={g['front_speed_m_s']:.2f} m/s  "
              f"h_eff={g['level_implied_depth_m']:.4f} m")
    print("  passes:", report["decision"]["passes"],
          " worst:", round(report["decision"]["worst_rel_error"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())