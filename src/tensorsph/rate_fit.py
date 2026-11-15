"""Band-rule estimation of exponential rates: fit a rate only where it is a rate.

Background
----------
``CASE-SPH-LAKE-AT-REST`` decided its rest horizon on an exponential fit of the
departure ``max|v|/c0``.  The *first* estimator fitted one hand-declared
**absolute** amplitude window for every arm and re-used it across arms.  On an
arm that starts moving that window is spent in the start-up transient, and the
"rate" it returns is the transient's average slope (measured: ``0.887 1/s``,
``R^2 = 0.29`` against ``16.47 1/s``, ``R^2 = 1.0000`` for the base arm).  On a
window whose upper edge reaches the saturated amplitude it returns the
saturation-averaged slope (measured ``8.19 1/s``).  Those are the *same* error at
the two ends of the window: the estimator, not the physics.

The rule that fixed it -- *declare the band once: its lower edge above the
roundoff floor and above any start-up transient, its upper edge below
saturation; then show that the window is not the answer* -- was shipped as a
rationale and a constant pair in ``benchmarks/lake_at_rest/band_fit.py``, for one
case.  A constant pair does not travel: the next case has a different floor and a
different saturation amplitude, and the case after that declares its window as
multiples of the seed amplitude (``low*a0``), which silently moves the physical
band whenever the seed changes -- the exact defect that was measured on the lake.

This module makes the rule **mechanical and case-independent**, so a new case
inherits the discipline instead of rediscovering the failure:

``fit_band(t, a, low, high)``
    Least-squares slope of ``ln a`` inside ``[low, high]``, with ``R^2`` and a
    validity flag.

``ladder(t, a, decades=1.0)``
    Every band of at least one decade that the observed amplitude range contains
    (edges log-spaced at ``EDGE_PER_DECADE``).  A rate must survive all of them.

``plateau(t, a)``
    The verdict.  Admissible bands are those with ``R^2 >= R2_MIN``; a *plateau*
    is a maximal run of adjacent, admissible, overlapping bands whose rates agree
    to ``SPREAD_MAX``.  The reported rate is the median over the plateau, and the
    plateau's two amplitude edges are the band to use.  A plateau shorter than
    ``MIN_PLATEAU_DECADES`` means **there is no exponential regime here**: quote
    no rate, and say which edge is responsible (``limited_low`` = transient or
    roundoff floor, ``limited_high`` = saturation).

``relative_seed_defect(t, a, factors)``
    What a ``[low*a0, high*a0]`` window -- the form used by
    ``tensorsph.threed.liquid_column`` and by any study whose band is quoted
    relative to the initial perturbation -- returns on the same series, and how
    far it sits from the plateau rate.  This is the defect made measurable rather
    than asserted.

Everything here is a pure function of a series: it re-reduces shipped data, it
does not run physics.  The lake case's own gates are the reference behaviour:
``plateau()`` on the shipped base arm reproduces ``lambda = 16.47 1/s`` over
``1e-6 ... 1e-3`` (see ``tests/test_rate_fit_band_rule.py``).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "R2_MIN", "MIN_POINTS", "MIN_PLATEAU_DECADES", "SPREAD_MAX",
    "EDGE_PER_DECADE", "fit_band", "floor_amplitude", "ladder", "plateau",
    "auto_band", "relative_seed_band", "relative_seed_defect", "report",
    "verdict", "sub_band_spread",
]

#: A band whose fit is worse than this is not exponential and cannot be a rate.
R2_MIN = 0.98
#: Fewer samples than this in a band and the slope is a two-point estimate.
MIN_POINTS = 6
#: A rate must be recoverable over at least this many decades of amplitude.
MIN_PLATEAU_DECADES = 1.0
#: Relative agreement required between the bands of one plateau.
SPREAD_MAX = 0.05
#: Band-edge resolution of the ladder, edges per decade.
EDGE_PER_DECADE = 4


def _series(t, a):
    """Finite, positive, time-ordered ``(t, a)`` as float64 arrays."""
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    if t.shape != a.shape:
        raise ValueError(f"t and a must have the same shape, got {t.shape} / {a.shape}")
    keep = np.isfinite(t) & np.isfinite(a) & (a > 0.0)
    t, a = t[keep], a[keep]
    order = np.argsort(t, kind="stable")
    return t[order], a[order]


def fit_band(t, a, low, high, r2_min=R2_MIN, min_points=MIN_POINTS) -> dict:
    """Least-squares slope of ``ln a`` against ``t`` inside ``low <= a <= high``.

    Returns ``sigma`` in 1/s (positive = growth), ``r2``, ``n_points``, the band
    actually populated, and ``valid`` (enough points *and* exponential enough).
    The returned ``t_start``/``t_end`` are what makes a band auditable: a window
    that grabs the transient shows up as a short ``t`` span at low amplitude.
    """
    t, a = _series(t, a)
    lo = -math.inf if low is None else float(low)
    hi = math.inf if high is None else float(high)
    sel = (a >= lo) & (a <= hi)
    out = {"sigma": float("nan"), "r2": float("nan"), "n_points": int(sel.sum()),
           "band": [lo, hi], "t_start": None, "t_end": None,
           "a_start": None, "a_end": None, "valid": False}
    if int(sel.sum()) < min_points:
        return out
    tt, aa = t[sel], a[sel]
    slope, intercept = np.polyfit(tt, np.log(aa), 1)
    resid = np.log(aa) - (slope * tt + intercept)
    ss_tot = float(((np.log(aa) - np.log(aa).mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / max(ss_tot, 1e-300)
    out.update({"sigma": float(slope), "r2": r2,
                "t_start": float(tt[0]), "t_end": float(tt[-1]),
                "a_start": float(aa[0]), "a_end": float(aa[-1]),
                "valid": bool(r2 >= r2_min)})
    return out


def floor_amplitude(a, quantile: float = 0.05) -> float:
    """Measured floor of a series: the low quantile of its positive amplitudes.

    For a departure from rest the floor is the roundoff/equilibrium residual the
    case sits at before it starts moving; a band whose lower edge is below it
    measures noise, not a rate.  A low quantile rather than the minimum, so one
    outlier sample cannot define the floor.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a) & (a > 0.0)]
    if a.size == 0:
        return float("nan")
    return float(np.quantile(a, quantile))


def ladder(t, a, decades: float = MIN_PLATEAU_DECADES,
           per_decade: int = EDGE_PER_DECADE,
           r2_min: float = R2_MIN, min_points: int = MIN_POINTS) -> list[dict]:
    """Every band of ``decades`` decades contained in the observed amplitude range.

    Edges are log-spaced at ``per_decade`` per decade, so adjacent bands overlap
    by ``decades - 1/per_decade`` decades.  Bands holding fewer than
    ``min_points`` samples are returned with ``valid=False`` and ``n_points``
    set, so the ladder itself explains why a range was unusable.
    """
    t, a = _series(t, a)
    if a.size == 0:
        return []
    a_min, a_max = float(a.min()), float(a.max())
    if not (a_max > a_min > 0.0):
        return []
    span = math.log10(a_max / a_min)
    steps = int(math.floor(span * per_decade))
    width = int(round(decades * per_decade))
    if width < 1:
        width = 1
    edges = [a_min * 10.0 ** (i / per_decade) for i in range(steps + 1)]
    bands = []
    for i in range(0, len(edges) - width):
        band = fit_band(t, a, edges[i], edges[i + width],
                        r2_min=r2_min, min_points=min_points)
        band["decades"] = math.log10(edges[i + width] / edges[i])
        bands.append(band)
    return bands


def plateau(t, a, decades: float = MIN_PLATEAU_DECADES,
            per_decade: int = EDGE_PER_DECADE, r2_min: float = R2_MIN,
            spread_max: float = SPREAD_MAX,
            min_decades: float = MIN_PLATEAU_DECADES,
            min_points: int = MIN_POINTS) -> dict:
    """The verdict: is there an amplitude band inside which the rate is *the* rate?

    A plateau is a maximal run of adjacent admissible bands (``R^2 >= r2_min``,
    enough points) whose slopes agree to ``spread_max``.  Among the plateaus the
    longest wins (ties broken by sample count, then by ``R^2``).  The reported
    rate is the median over the plateau, which is what a case should quote; the
    plateau edges are what a case should declare as its band.
    """
    bands = ladder(t, a, decades=decades, per_decade=per_decade,
                   r2_min=r2_min, min_points=min_points)
    out = {"sigma": float("nan"), "r2": float("nan"), "spread_rel": None,
           "plateau_decades": 0.0, "plateau_bands": 0, "band": None,
           "band_median": None, "n_points": 0, "valid": False,
           "limited_low": None, "limited_high": None, "reason": "",
           "reason_kind": None, "points_per_decade": None,
           "n_bands": len(bands), "n_admissible": 0,
           "sigma_ladder": [b["sigma"] for b in bands],
           "r2_ladder": [b["r2"] for b in bands],
           "band_ladder": [b["band"] for b in bands]}
    if not bands:
        out["reason"] = "no band of one decade fits the amplitude range"
        out["reason_kind"] = "range"
        return out

    t_s, a_s = _series(t, a)
    span = math.log10(float(a_s.max()) / float(a_s.min()))
    out["points_per_decade"] = float(a_s.size) / span if span > 0 else None

    admissible = [b["valid"] for b in bands]
    out["n_admissible"] = int(sum(admissible))
    if not out["n_admissible"]:
        # A band with too few samples cannot be "not exponential" -- it is
        # unmeasured.  Keep the two reasons apart: one is a verdict about the
        # physics, the other about the sampling.
        best_n = max(b["n_points"] for b in bands)
        if best_n < min_points:
            out["reason"] = ("series too sparse for a %g-decade ladder: at most "
                             "%d samples in any band (< %d); quote the declared "
                             "band's own point count, or use shorter bands"
                             % (decades, best_n, min_points))
            out["reason_kind"] = "sparse"
        else:
            r2s = [b["r2"] for b in bands
                   if not math.isnan(b["r2"]) and b["n_points"] >= min_points]
            out["reason"] = ("no band is exponential (max R^2 = %.3f < %.3f): "
                             "there is no rate to quote"
                             % (max(r2s) if r2s else float("nan"), r2_min))
            out["reason_kind"] = "not_exponential"
        return out

    # maximal adjacent runs of admissible bands with agreeing slopes
    runs: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(admissible + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    best = None
    for lo_i, hi_i in runs:
        sigmas = [bands[k]["sigma"] for k in range(lo_i, hi_i + 1)
                  if not math.isnan(bands[k]["sigma"])]
        if not sigmas:
            continue
        med = float(np.median(sigmas))
        spread = (max(sigmas) - min(sigmas)) / abs(med) if med else math.inf
        # walk inwards until the slopes agree, so one mismatched band at the
        # edge of an otherwise good run does not throw the run away
        while len(sigmas) > 1 and spread > spread_max:
            lo2, hi2 = lo_i, hi_i
            inner = [bands[k]["sigma"] for k in range(lo_i, hi_i + 1)]
            left_gap = abs(inner[0] - med)
            right_gap = abs(inner[-1] - med)
            if left_gap >= right_gap:
                lo2 += 1
            else:
                hi2 -= 1
            sigmas = [bands[k]["sigma"] for k in range(lo2, hi2 + 1)
                      if not math.isnan(bands[k]["sigma"])]
            if len(sigmas) < 2:
                break
            med = float(np.median(sigmas))
            spread = (max(sigmas) - min(sigmas)) / abs(med) if med else math.inf
            lo_i, hi_i = lo2, hi2
        if len(sigmas) < 1:
            continue
        lo_edge = bands[lo_i]["band"][0]
        hi_edge = bands[hi_i]["band"][1]
        span = math.log10(hi_edge / lo_edge)
        n_pts = max(bands[k]["n_points"] for k in range(lo_i, hi_i + 1))
        r2_worst = min(bands[k]["r2"] for k in range(lo_i, hi_i + 1))
        cand = {"sigma": med, "spread_rel": spread, "plateau_decades": span,
                "plateau_bands": hi_i - lo_i + 1,
                "band": [lo_edge, hi_edge],
                "band_median": [bands[(lo_i + hi_i) // 2]["band"][0],
                                bands[(lo_i + hi_i) // 2]["band"][1]],
                "n_points": n_pts, "r2": r2_worst,
                "lo_index": lo_i, "hi_index": hi_i}
        if best is None or (cand["plateau_decades"], cand["n_points"]) > (
                best["plateau_decades"], best["n_points"]):
            best = cand

    if best is None:
        out["reason"] = "admissible bands are isolated: no two adjacent agree"
        return out

    out.update(best)
    out["valid"] = bool(best["plateau_decades"] >= min_decades
                        and best["spread_rel"] <= spread_max)
    # which end of the ladder stopped the plateau
    out["limited_low"] = best["lo_index"] > 0
    out["limited_high"] = best["hi_index"] < len(bands) - 1
    if not out["valid"]:
        if best["plateau_decades"] < min_decades:
            out["reason"] = ("longest run of agreeing bands spans %.2f decades "
                             "(< %.2f required): no exponential regime to "
                             "measure; no rate to quote"
                             % (best["plateau_decades"], min_decades))
            out["reason_kind"] = "narrow_plateau"
        else:
            out["reason"] = ("band slopes never agree to %.0f %% (best spread "
                             "%.1f %% over %.2f decades): the series is curved "
                             "in log-amplitude, so no single rate describes it"
                             % (100 * spread_max, 100 * best["spread_rel"],
                                best["plateau_decades"]))
            out["reason_kind"] = "curved"
    return out


def sub_band_spread(t, a, low, high, r2_min: float = R2_MIN,
                    min_points: int = MIN_POINTS) -> dict:
    """Sweep nested sub-bands of a declared window: is the window itself stable?

    This is the declared-window analogue of the plateau, and the check the lake
    study did by hand (``SENSITIVITY``): cut ``[low, high]`` into quarter-decade
    nested sub-bands and fit each.  A window that is genuinely exponential gives
    the same slope at every sub-band; a window that straddles a transient or a
    saturation gives a *trend* in the sub-band slopes, which is exactly what
    ``spread_rel`` and ``slope_trend`` expose.
    """
    t, a = _series(t, a)
    lo, hi = float(low), float(high)
    out = {"band": [lo, hi], "n_sub": 0, "sigmas": [], "spread_rel": None,
           "slope_trend": None, "valid": False}
    if not (hi > lo > 0.0):
        return out
    span = math.log10(hi / lo)
    # Sub-bands must stay measurable: at least half a decade wide, and never so
    # narrow that a window of 40 samples cannot fill them.  Nested (half-width
    # step) so the sweep is a set of overlapping windows, not a partition.
    width = max(0.5, span / 3.0)
    step = width / 2.0
    edges, x = [], lo
    while x * 10.0 ** width <= hi * (1.0 + 1e-12):
        edges.append((x, x * 10.0 ** width))
        x *= 10.0 ** step
    sigmas, mids = [], []
    for lo_i, hi_i in edges:
        fit = fit_band(t, a, lo_i, hi_i,
                       r2_min=r2_min, min_points=min(min_points, 4))
        if fit["valid"] and not math.isnan(fit["sigma"]):
            sigmas.append(fit["sigma"])
            mids.append(math.sqrt(lo_i * hi_i))
    out["sigmas"] = sigmas
    out["n_sub"] = len(sigmas)
    if len(sigmas) < 2:
        return out
    med = float(np.median(sigmas))
    out["spread_rel"] = (max(sigmas) - min(sigmas)) / abs(med) if med else None
    # slope of sigma against log10(amplitude): ~0 for a window on a plateau,
    # non-zero (negative into saturation, positive out of a transient) otherwise
    if len(sigmas) >= 3:
        out["slope_trend"] = float(np.polyfit(np.log10(mids), sigmas, 1)[0])
    out["valid"] = bool(out["spread_rel"] is not None
                        and out["spread_rel"] <= SPREAD_MAX)
    return out


def auto_band(t, a, **kw) -> tuple[float, float] | None:
    """The band a case should declare: the plateau's amplitude edges."""
    p = plateau(t, a, **kw)
    if not p["valid"]:
        return None
    return (p["band"][0], p["band"][1])


def relative_seed_band(a, low_factor: float, high_factor: float) -> tuple[float, float]:
    """The ``[low*a0, high*a0]`` band form used when a window is quoted relative
    to the initial perturbation.

    This is the form in ``tensorsph.threed.liquid_column.ColumnResult.growth``
    (``low=3.0, high=25.0`` of ``amplitudes[0]``) and in every study that says
    "fit while the amplitude is a few times its seed".  It looks seed-free and is
    not: ``a0`` is the seed, so the physical band moves with the seed.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a) & (a > 0.0)]
    if a.size == 0:
        return (float("nan"), float("nan"))
    a0 = float(a[0])
    return (low_factor * a0, high_factor * a0)


def relative_seed_defect(t, a, factors=(3.0, 25.0), **kw) -> dict:
    """Measure what a seed-relative window returns, against the plateau rate."""
    t, a = _series(t, a)
    lo, hi = relative_seed_band(a, *factors)
    legacy = fit_band(t, a, lo, hi)
    p = plateau(t, a, **kw)
    bias = None
    if p["valid"] and not math.isnan(legacy["sigma"]) and p["sigma"]:
        bias = legacy["sigma"] / p["sigma"] - 1.0
    inside = None
    if p["valid"]:
        inside = bool(lo >= p["band"][0] and hi <= p["band"][1])
    return {"window_form": "relative to first sample a0",
            "factors": list(factors), "band": [lo, hi],
            "seed": float(a[0]) if a.size else None,
            "legacy": legacy, "plateau_sigma": p["sigma"],
            "bias_rel": bias, "band_inside_plateau": inside,
            "legacy_is_a_rate": bool(legacy["valid"] and inside)}


def verdict(rec: dict) -> dict:
    """Reduce an audit record to its decision.

    The reusable rule is *not* "quote the plateau median".  A case declares its
    band a priori (that is what makes it free of hindsight), and the rule is that
    the declared band must sit **inside** a plateau of agreeing, exponential
    bands -- and must itself be exponential with enough points.  If there is no
    plateau, the case has no rate to quote at all, however good the fit to its
    declared window looks.
    """
    p = rec["plateau"]
    d = rec.get("declared")
    sweep = rec.get("declared_sweep")
    legal = None
    why = None
    if d is not None:
        reasons = []
        if not p["valid"]:
            reasons.append("no plateau in the series")
        if not d["valid"]:
            reasons.append("declared band is not exponential (R^2 = %.3f)" % d["r2"])
        bias = d.get("bias_rel")
        if bias is not None and abs(bias) > SPREAD_MAX:
            reasons.append("declared rate differs from the plateau rate by "
                           "%+.1f %% (> %.0f %%)" % (100 * bias, 100 * SPREAD_MAX))
        if sweep is not None and not sweep["valid"]:
            reasons.append("sub-band sweep of the declared window is unstable "
                           "(spread %s)" % ("n/a" if sweep["spread_rel"] is None
                                            else "%.1f %%" % (100 * sweep["spread_rel"])))
        legal = not reasons
        why = "; ".join(reasons) if reasons else "declared band sits on a plateau"
    return {"plateau_valid": bool(p["valid"]),
            "plateau_sigma": p["sigma"],
            "plateau_band": p["band"],
            "reason": p["reason"], "reason_kind": p["reason_kind"],
            "declared_band_legal": legal,
            "declared_legal_reason": why,
            "declared_sigma": None if d is None else d["sigma"],
            "declared_bias_rel": None if d is None else d.get("bias_rel"),
            "declared_sweep_spread_rel": (None if sweep is None
                                          else sweep["spread_rel"])}


def report(t, a, declared=None, **kw) -> dict:
    """Full audit record of one series: plateau verdict + optional declared band."""
    t, a = _series(t, a)
    p = plateau(t, a, **kw)
    rec = {"n_samples": int(a.size), "a_min": float(a.min()) if a.size else None,
           "a_max": float(a.max()) if a.size else None,
           "floor": floor_amplitude(a), "plateau": p,
           "declared": None, "seed_relative": None}
    rec["declared_sweep"] = None
    if declared is not None:
        rec["declared_sweep"] = sub_band_spread(t, a, *declared)
        d = fit_band(t, a, *declared)
        inside = (p["valid"] and declared[0] >= p["band"][0]
                  and declared[1] <= p["band"][1])
        d["inside_plateau"] = bool(inside)
        d["bias_rel"] = (d["sigma"] / p["sigma"] - 1.0
                         if p["valid"] and not math.isnan(d["sigma"]) else None)
        rec["declared"] = d
    rec["seed_relative"] = relative_seed_defect(t, a, **kw)
    rec["verdict"] = verdict(rec)
    return rec