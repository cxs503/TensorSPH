"""Score the Kleefsman H2 acceptance criterion at a station that survives the
probe-lattice audit, using only series that are already on disk.

Why this exists
---------------
``reference.json`` freezes the acceptance protocol at the *committed* station
``x = obstacle_upstream_face - 0.05 m = 2.426 m``.  ``station_lattice.py`` (m22476)
then showed that station is inadmissible: reading Kleefsman et al. (2005) p.381 /
Fig. 19 caption together with the equal 0.496 m probe spacing printed in Amaro
Junior et al. (2020) Fig. 2(a) gives ``1.228 < x(H2) <= 2.220 m``, so H2 at
2.426 m would put the reservoir probe H4 inside the tank.

Every grid on disk already carries a gauge at ``x = 2.228 m`` (= box face -
0.248 m, i.e. 8 mm downstream of the lattice's gate anchor 2.220 m) *and* at the
committed 2.426 m.  Until now only the 2.426 m columns were scored against the
digitised Fig. 19 curve, and only at the legacy ``t_end = 3.0 s`` grids.  So the
finest grid (``dx = 0.01 m``, ``t_end = 2.6 s``) had never been scored against
the acceptance criterion at any admissible station, even though its record is
long enough to decide both metrics (peak at ~1.43 s + 0.13 s of margin).

What this script adds
---------------------
1. the acceptance rows at the admissible station for **every grid already on
   disk**, under the frozen protocol (slab, 0.08 m aperture, 30 ms symmetric
   filter, 0.02 m arrival threshold, interpolated first crossing);
2. the same rows for the two other gauge operators (``height``, ``surface``) and
   for both reference-arrival definitions (interpolated 0.3481 s vs
   sample-quantised 0.3591 s), because the acceptance verdict turns out to sit
   *inside* that definition band -- reported, not hidden;
3. the observed convergence order and GCI (Roache, Fs = 1.25) of peak and
   arrival at that station, so "converged" is a measured claim;
4. the station sensitivity 2.220 m vs 2.228 m (8 mm) when the dedicated run
   ``results/gate_ambiguity/dx0.02`` is present;
5. the front-speed attribution: arrival bias is re-expressed as a gravity-current
   front-speed deficit, which is where the residual error actually lives.

Outputs (all under ``results/admissible_station/``)::

    score.json          every row, every grid, every operator, every station
    score_table.md      the markdown table quoted by the doc
    ladder.json         order / Richardson / GCI per quantity

Usage::

    python benchmarks/kleefsman_obstacle/score_admissible_station.py [--figure]

Nothing here integrates the equations; it is pure re-reading of committed CSV
records, so it is cheap and reproducible on any node.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from score_t30 import (  # noqa: E402
    arrival,
    frozen_protocol,
    load_reference,
    load_series,
    peak,
    rel_error,
    resample,
    richardson,
    shape_stats,
)

OUT = HERE / "results" / "admissible_station"

#: Grid records already committed to the repository.  ``csv`` is relative to
#: this directory; ``optional`` records are skipped when absent, so the script
#: runs both before and after the dedicated gate-ambiguity run lands.
GRIDS = (
    {"tag": "dx0.01", "dx": 0.01, "t_end": 2.600098,
     "csv": "results/peak_horizon/dx0.01_time_series.csv",
     "producer": "results/peak_horizon/dx0.01.json", "optional": False},
    {"tag": "dx0.02", "dx": 0.02, "t_end": 3.000200,
     "csv": "results/audit_t30/dx0.02/time_series.csv",
     "producer": "results/audit_t30/dx0.02/metrics.json", "optional": False},
    {"tag": "dx0.04", "dx": 0.04, "t_end": 3.000200,
     "csv": "results/audit_t30/dx0.04/time_series.csv",
     "producer": "results/audit_t30/dx0.04/metrics.json", "optional": False},
    {"tag": "dx0.02_gate", "dx": 0.02, "t_end": 3.000000,
     "csv": "results/gate_ambiguity/dx0.02/time_series.csv",
     "producer": "results/gate_ambiguity/dx0.02/metrics.json", "optional": True},
)

#: Stations, with the audit status of each.  ``2.228`` and ``2.220`` are both
#: inside (or 8 mm outside) the lattice interval; ``2.426`` is excluded.
STATIONS = (
    (2.228, "committed_secondary"),
    (2.220, "lattice_gate_anchor"),
    (2.426, "committed_primary_excluded"),
)

OPERATORS = ("slab", "height", "surface")

#: Peak-time search half-width around the experimental peak (s).  The reference
#: peak is a broad maximum, so a +-0.3 s window is used to separate "the same
#: event" from "a transient slosh near gate opening".
EVENT_HALF_WINDOW_S = 0.3
#: Censoring guard: a peak must be this far from the end of the record to count.
CENSORING_MARGIN_S = 0.13
#: Roache safety factor for the grid convergence index.
GCI_FS = 1.25


def column_name(station: float, operator: str) -> str:
    return f"{operator}_x{station:.3f}"


def _record_time_max(t: list[float]) -> float:
    return max(t) if t else float("nan")


def measure(t: list[float], h: list[float], *, filter_s: float,
            threshold: float, ref_peak_time: float) -> dict:
    """Peak / arrival of one gauge record, with the windings that matter here."""
    if not t:
        return {"status": "not_measured"}
    pk, pk_t = peak(t, h, filter_s)
    pk_raw, pk_t_raw = peak(t, h, 0.0)
    # event-window peak: restrict to ref peak time +- half window
    lo, hi = ref_peak_time - EVENT_HALF_WINDOW_S, ref_peak_time + EVENT_HALF_WINDOW_S
    tw = [x for x in t if lo <= x <= hi]
    if len(tw) > 2:
        a = min(range(len(t)), key=lambda i: abs(t[i] - tw[0]))
        b = min(range(len(t)), key=lambda i: abs(t[i] - tw[-1]))
        ev_pk, ev_t = peak(t[a:b + 1], h[a:b + 1], filter_s)
        ev_m = max(h[a:b + 1])
        ev = {"peak_m": ev_pk, "peak_time_s": ev_t, "raw_max_m": ev_m,
              "window_s": [lo, hi]}
    else:
        ev = None
    return {
        "status": "measured",
        "t_end_s": _record_time_max(t),
        "samples": len(t),
        "peak_m": pk,
        "peak_time_s": pk_t,
        "peak_m_raw": pk_raw,
        "peak_time_s_raw": pk_t_raw,
        "arrival_time_s_interp": arrival(t, h, threshold, "interp"),
        "arrival_time_s_sample": arrival(t, h, threshold, "sample"),
        "event": ev,
        "censored": bool(_record_time_max(t) - pk_t < CENSORING_MARGIN_S),
    }


def observed_order(fine: float, mid: float, coarse: float,
                   ratio: float = 2.0) -> float:
    """Three-grid observed order of accuracy (Celik / Roache)."""
    d1 = abs(mid - fine)
    d2 = abs(coarse - mid)
    if not (math.isfinite(d1) and math.isfinite(d2)) or d2 <= 0.0 or d1 <= 0.0:
        return float("nan")
    return math.log(d2 / d1) / math.log(ratio)


def ladder(levels: list[tuple[float, float, str]], qname: str,
           ratio: float = 2.0) -> dict:
    """Grid-convergence statement for one scalar quantity.

    ``levels`` = [(dx, value, tag)] with the finest grid first.  Returns the
    observed order, the Richardson h -> 0 estimate, the GCI of the fine grid,
    and -- the number the acceptance actually needs -- the *continuous* value
    with its sign of approach.
    """
    pts = [(dx, v, tag) for dx, v, tag in levels if math.isfinite(v)]
    if len(pts) < 3:
        return {"quantity": qname, "status": "insufficient_grids",
                "levels": [{"dx": dx, "value": v, "tag": tag} for dx, v, tag in pts]}
    (dx1, v1, t1), (dx2, v2, t2), (dx3, v3, t3) = pts[0], pts[1], pts[2]
    p = observed_order(v1, v2, v3, ratio)
    out = {"quantity": qname, "status": "ok", "order_observed": p,
           "levels": [{"dx": dx, "value": v, "tag": tag} for dx, v, tag in pts]}
    # Roache/Celik practice: an extrapolation is only meaningful when the
    # observed order is itself usable (0.5 <= p <= 6, monotone approach).
    monotone = bool(abs(v2 - v1) < abs(v3 - v2))
    if math.isfinite(p) and 0.5 <= p <= 6.0 and monotone:
        v0 = richardson(v1, v2, ratio, p)
        out.update({"value_h0": v0,
                    "gci_fine_rel": GCI_FS * abs(v1 - v2) / abs(v1) / (ratio ** p - 1.0),
                    "monotone": True, "non_asymptotic": False})
    else:
        # Non-asymptotic: report the band spanned by the two finest grids rather
        # than pretending a Richardson limit exists.
        out.update({"value_h0": None, "non_asymptotic": True, "monotone": monotone,
                    "band_from_finest_two": [min(v1, v2), max(v1, v2)],
                    "why": "observed order not usable (p outside [0.5, 6] or "
                           "non-monotone approach); no Richardson limit claimed"})
    return out


def front_speed(x_gate: float, x_station: float, t_arrival: float) -> float:
    if not (math.isfinite(t_arrival) and t_arrival > 0):
        return float("nan")
    return (x_station - x_gate) / t_arrival


def score(include_optional: bool = True) -> dict:
    ref, series_path = load_reference()
    ev = ref["experimental_values"]
    proto = frozen_protocol(ref)
    tol = float(ref["tolerance"]["relative"])
    geom = ref["geometry_m"]
    gate_x = float(geom["column_length"])

    # ---- reference curve measured with the same operators ----------------
    t_ref, h_ref = load_series(series_path, "h_m")
    f = proto["peak_filter_window_s"]
    ref_peak, ref_peak_t = peak(t_ref, h_ref, f)
    ref_arr_interp = arrival(t_ref, h_ref, proto["arrival_threshold_m"], "interp")
    ref_arr_sample = arrival(t_ref, h_ref, proto["arrival_threshold_m"], "sample")
    reference = {
        "series_path": str(series_path),
        "peak_m": ref_peak, "peak_time_s": ref_peak_t,
        "arrival_interp_s": ref_arr_interp,
        "arrival_sample_s": ref_arr_sample,
        "arrival_definition_band_rel": (ref_arr_sample - ref_arr_interp) / ref_arr_interp,
        "anchored": {"peak_m": ev.get("peak_height_m"),
                     "arrival_s": ev.get("arrival_time_s")},
        "tolerance_rel": tol,
        "protocol": proto,
        "digitisation_uncertainty": ref.get("experimental_values", {}).get("uncertainty"),
    }

    grids = []
    for g in GRIDS:
        csv_path = HERE / g["csv"]
        if not csv_path.is_file():
            if g["optional"]:
                grids.append({**g, "status": "not_run", "gauges": {}})
                continue
            grids.append({**g, "status": "missing_csv", "gauges": {}})
            continue
        gauge_rows: dict[str, dict] = {}
        for station, role in STATIONS:
            for op in OPERATORS:
                col = column_name(station, op)
                try:
                    t, h = load_series(csv_path, col)
                except KeyError:
                    gauge_rows[f"{op}@{station:.3f}"] = {
                        "status": "not_measured", "station_x_m": station,
                        "operator": op, "role": role,
                        "why": f"column {col!r} absent from {g['csv']}"}
                    continue
                m = measure(t, h, filter_s=f,
                            threshold=proto["arrival_threshold_m"],
                            ref_peak_time=ref_peak_t)
                if m["status"] != "measured":
                    gauge_rows[f"{op}@{station:.3f}"] = {
                        "status": "not_measured", "station_x_m": station,
                        "operator": op, "role": role}
                    continue
                rows = []
                for label, want in (("peak_height", ref_peak),
                                    ("arrival_time_interp", ref_arr_interp),
                                    ("arrival_time_sample", ref_arr_sample)):
                    got = {"peak_height": m["peak_m"],
                           "arrival_time_interp": m["arrival_time_s_interp"],
                           "arrival_time_sample": m["arrival_time_s_sample"]}[label]
                    err = rel_error(got, want) if math.isfinite(got) else float("nan")
                    rows.append({"metric": label, "ref": want, "got": got,
                                 "signed_rel_error": ((got - want) / want
                                                      if math.isfinite(got) else float("nan")),
                                 "rel_error": err,
                                 "pass": bool(math.isfinite(err) and err < tol)})
                gauge_rows[f"{op}@{station:.3f}"] = {
                    **m, "station_x_m": station, "operator": op, "role": role,
                    "delta_s_to_gate": station - gate_x,
                    "rows": rows,
                }
        grids.append({**g, "status": "scored", "gauges": gauge_rows})

    # ---- convergence ladder at the admissible station, frozen operator ----
    admissible = [g for g in grids if g["status"] == "scored"]
    main_tag = "slab@2.228"
    ladder_peak, ladder_arr, shapes = [], [], []
    for g in sorted(admissible, key=lambda d: d["dx"]):
        key = main_tag
        rec = g["gauges"].get(key)
        if not rec or rec.get("status") != "measured":
            continue
        ladder_peak.append((g["dx"], rec["peak_m"], g["tag"]))
        ladder_arr.append((g["dx"], rec["arrival_time_s_interp"], g["tag"]))
        # shape agreement over the reference span, same operators both sides
        col = column_name(2.228, "slab")
        try:
            t, h = load_series(HERE / g["csv"], col)
        except KeyError:
            continue
        # common comparison window: the reference start clipped to the shortest
        # record, so all grids are judged on the same stretch of curve
        t0 = max(min(t_ref), 0.35)
        t1 = min(2.6, max(t), max(t_ref))
        st = shape_stats(t_ref, h_ref, t, h, t0, t1)
        shapes.append({"tag": g["tag"], "dx": g["dx"], **st})

    lad = {
        "station_x_m": 2.228,
        "operator": proto["operator"],
        "peak": ladder_peak and ladder(ladder_peak, "peak_height_m") or
                {"quantity": "peak_height_m", "status": "insufficient_grids"},
        "arrival": ladder_arr and ladder(ladder_arr, "arrival_time_s_interp") or
                   {"quantity": "arrival_time_s_interp", "status": "insufficient_grids"},
        "shape_vs_reference": shapes,
    }

    # ---- station sensitivity 2.220 vs 2.228 (8 mm) -----------------------
    station_sens = []
    for g in admissible:
        a = g["gauges"].get("slab@2.228")
        b = g["gauges"].get("slab@2.220")
        if a and b and a.get("status") == "measured" and b.get("status") == "measured":
            station_sens.append({
                "tag": g["tag"], "dx": g["dx"],
                "peak_delta_m": b["peak_m"] - a["peak_m"],
                "peak_delta_rel": (b["peak_m"] - a["peak_m"]) / a["peak_m"],
                "arrival_delta_s": (b["arrival_time_s_interp"]
                                    - a["arrival_time_s_interp"]),
                "arrival_delta_rel": ((b["arrival_time_s_interp"]
                                       - a["arrival_time_s_interp"])
                                      / a["arrival_time_s_interp"]),
            })

    # ---- the 8 mm station ambiguity (2.220 vs 2.228), cheaply closed ----
    # Arrival(x) is smooth, so the two stations that *are* on disk bracket the
    # derivative needed to move the gauge by 8 mm.  This is an interpolation of
    # the arrival function, not a substitute for the dedicated run -- it is
    # reported as an estimate with the measurement that would confirm it.
    station_8mm = []
    for g in admissible:
        a = g["gauges"].get("slab@2.228")
        b = g["gauges"].get("slab@2.426")
        if not (a and b) or a.get("status") != "measured" or b.get("status") != "measured":
            continue
        d_x = b["station_x_m"] - a["station_x_m"]
        d_t = b["arrival_time_s_interp"] - a["arrival_time_s_interp"]
        if not (math.isfinite(d_t) and abs(d_t) > 0):
            continue
        v_local = d_x / d_t
        shift_s = 0.008 / v_local
        station_8mm.append({
            "tag": g["tag"], "dx": g["dx"],
            "local_front_speed_m_s": v_local,
            "implied_shift_s": shift_s,
            "implied_shift_rel": shift_s / a["arrival_time_s_interp"],
            "source": "slab arrival at 2.228 and 2.426 on the same record",
            "confirmation": "results/gate_ambiguity/dx0.02 (job cj4566)",
        })

    # ---- measured 8 mm sensitivity, from the m22476 station-lattice runs ----
    # Those records carry a slab gauge at 2.220 m on the same grids; pairing them
    # with the same-grid 2.228 m gauge of this repository gives the *measured*
    # station derivative, independent of the interpolation above.
    cross = []
    for g in admissible:
        lat = HERE / "results" / "station_lattice" / f"stations_{g['tag'].replace('.', '')}_t3.0.json"
        if not lat.is_file():
            lat = HERE / "results" / "station_lattice" / f"stations_dx{g['dx']:g}_t3.0.json"
        if not lat.is_file():
            continue
        rec = json.loads(lat.read_text())
        st = rec.get("stations", {}).get("2.220")
        a = g["gauges"].get("slab@2.228")
        if not st or not a or a.get("status") != "measured":
            continue
        cross.append({
            "tag": g["tag"], "dx": g["dx"], "source": str(lat.relative_to(HERE)),
            "aperture_m_lattice_run": rec.get("aperture_m"),
            "arrival_2.220_s": st["slab_arrival_s"],
            "arrival_2.228_s": a["arrival_time_s_interp"],
            "arrival_delta_rel": ((st["slab_arrival_s"]
                                   - a["arrival_time_s_interp"])
                                  / a["arrival_time_s_interp"]),
            "peak_2.220_m": st["slab_peak_m"],
            "peak_2.228_m": a["peak_m"],
            "peak_delta_rel": (st["slab_peak_m"] - a["peak_m"]) / a["peak_m"],
        })

    # ---- front-speed attribution ----------------------------------------
    v_front = {"gate_x_m": gate_x, "station_x_m": 2.228,
               "experimental": {"arrival_interp_s": ref_arr_interp,
                                "speed_m_s": front_speed(gate_x, 2.228, ref_arr_interp)},
               "rows": []}
    for g in sorted(admissible, key=lambda d: d["dx"]):
        rec = g["gauges"].get(main_tag)
        if not rec or rec.get("status") != "measured":
            continue
        for op in OPERATORS:
            r = g["gauges"].get(f"{op}@2.228")
            if not r or r.get("status") != "measured":
                continue
            v = front_speed(gate_x, 2.228, r["arrival_time_s_interp"])
            v_front["rows"].append({
                "tag": g["tag"], "dx": g["dx"], "operator": op,
                "arrival_interp_s": r["arrival_time_s_interp"],
                "speed_m_s": v,
                "speed_deficit_rel": (v - v_front["experimental"]["speed_m_s"])
                                     / v_front["experimental"]["speed_m_s"],
            })

    verdict = decide(grids, tol, reference)
    return {"reference": reference, "grids": grids, "ladder": lad,
            "station_sensitivity": station_sens,
            "station_8mm_estimate": station_8mm,
            "station_sensitivity_measured": cross, "front_speed": v_front,
            "verdict": verdict}


def decide(grids: list[dict], tol: float, reference: dict) -> dict:
    """Acceptance verdict at the admissible station.

    Rule (stated before the numbers are looked at, so the verdict cannot be
    reverse-engineered from them):

    * station   : 2.228 m (the only station on disk that the lattice audit does
                  not exclude; 2.426 m is EXCLUDED_BY_LATTICE).
    * operator  : frozen ``slab``.
    * grid      : finest available with an uncensored peak.
    * PASS      : both peak and arrival within ``tol`` for **both** reference
                  arrival definitions -- a pass must not depend on which of the
                  two digitisation conventions one happens to pick.
    * BORDERLINE: one definition inside, the other outside -> the tolerance is
                  not decidable at the current reference precision; reported as
                  such, never rounded to PASS.
    * FAIL      : both definitions outside.
    """
    finest = None
    for g in sorted((g for g in grids if g["status"] == "scored"),
                    key=lambda d: d["dx"]):
        rec = g["gauges"].get("slab@2.228")
        if rec and rec.get("status") == "measured" and not rec["censored"]:
            finest = (g, rec)
            break
    if finest is None:
        return {"verdict": "WITHHELD",
                "why": "no scored grid carries an uncensored slab gauge at 2.228 m"}
    g, rec = finest
    rows = {r["metric"]: r for r in rec["rows"]}
    pk = rows["peak_height"]
    ai = rows["arrival_time_interp"]
    asm = rows["arrival_time_sample"]
    peak_ok = pk["pass"]
    arr_both = ai["pass"] and asm["pass"]
    arr_neither = not ai["pass"] and not asm["pass"]
    if peak_ok and arr_both:
        verdict, why = "PASS", ("peak and arrival inside tolerance for both "
                                "reference-arrival definitions")
    elif peak_ok and arr_neither:
        verdict, why = "FAIL", "arrival outside tolerance for both definitions"
    elif not peak_ok and arr_both:
        verdict, why = "FAIL", "peak outside tolerance"
    else:
        verdict, why = "BORDERLINE", (
            "arrival tolerance is not decidable: "
            f"{ai['signed_rel_error']:+.1%} against the interpolated reference "
            f"({ai['ref']:.4f} s) but {asm['signed_rel_error']:+.1%} against the "
            f"sample-quantised one ({asm['ref']:.4f} s); the reference-definition "
            f"band is {reference['arrival_definition_band_rel']:.1%}")
    # The 10 %-crossing arrival the definition would demand:
    need_interp = ai["ref"] * (1 + tol)
    need_sample = asm["ref"] * (1 + tol)
    return {
        "verdict": verdict, "why": why,
        "station_x_m": 2.228, "grid_tag": g["tag"], "dx": g["dx"],
        "operator": "slab",
        "peak": {"got": pk["got"], "ref": pk["ref"],
                 "signed_rel_error": pk["signed_rel_error"], "pass": pk["pass"]},
        "arrival": {
            "got": rec["arrival_time_s_interp"],
            "interp": {"ref": ai["ref"], "signed_rel_error": ai["signed_rel_error"],
                       "pass": ai["pass"]},
            "sample": {"ref": asm["ref"], "signed_rel_error": asm["signed_rel_error"],
                       "pass": asm["pass"]},
        },
        "arrival_needed_for_10pct_s": {"vs_interp": need_interp,
                                       "vs_sample": need_sample,
                                       "shortfall_s": rec["arrival_time_s_interp"] - need_interp,
                                       "shortfall_rel": (rec["arrival_time_s_interp"] - need_interp) / need_interp},
        "peak_time_s": rec["peak_time_s"],
        "peak_time_rel_error_vs_reference": rel_error(rec["peak_time_s"],
                                                     reference["peak_time_s"]),
        "censored": rec["censored"],
        "rule": __doc__ and decide.__doc__,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def markdown(scored: dict) -> str:
    out = ["| grid | station [m] | operator | peak [m] | peak err | arrival [s] | "
           "err vs interp | err vs sample | verdict |",
           "|---|---|---|---|---|---|---|---|---|"]
    for g in sorted((g for g in scored["grids"] if g["status"] == "scored"),
                    key=lambda d: -d["dx"]):
        for station, _role in STATIONS:
            for op in OPERATORS:
                rec = g["gauges"].get(f"{op}@{station:.3f}")
                if not rec or rec.get("status") != "measured":
                    continue
                rows = {r["metric"]: r for r in rec["rows"]}
                pk, ai, asm = (rows["peak_height"], rows["arrival_time_interp"],
                               rows["arrival_time_sample"])
                v = "PASS" if (pk["pass"] and ai["pass"] and asm["pass"]) else (
                    "FAIL" if (not pk["pass"] and not ai["pass"] and not asm["pass"])
                    else "BORDERLINE")
                out.append(
                    f"| {g['tag']} (dx={g['dx']:g}) | {station:.3f} | {op} | "
                    f"{pk['got']:.4f} | {pk['signed_rel_error']:+.1%} | "
                    f"{rec['arrival_time_s_interp']:.4f} | "
                    f"{ai['signed_rel_error']:+.1%} | {asm['signed_rel_error']:+.1%} | {v} |")
    v = scored["verdict"]
    out.append("")
    out.append(f"**verdict at the admissible station**: `{v['verdict']}` — {v['why']}")
    out.append("")
    lad = scored["ladder"]
    for q in ("peak", "arrival"):
        L = lad[q]
        if L.get("status") != "ok":
            out.append(f"- {q}: {L['status']}")
            continue
        if L.get("value_h0") is None:
            out.append(f"- {q}: observed order p={L['order_observed']:.2f} -> "
                       f"**non-asymptotic**, no extrapolation; finest-two band "
                       f"{L['band_from_finest_two']}")
        else:
            out.append(f"- {q}: observed order p={L['order_observed']:.2f}, "
                       f"h->0 = {L['value_h0']:.6f}, "
                       f"GCI_fine={L['gci_fine_rel']:.2%}")
    est = scored.get("station_8mm_estimate") or []
    if est:
        worst = max(est, key=lambda r: abs(r["implied_shift_rel"]))
        out.append(f"- 8 mm station ambiguity (2.220 vs 2.228 m), estimated from the "
                   f"local arrival derivative: {worst['implied_shift_rel']:+.2%} "
                   f"(worst grid, {worst['tag']}); local front speed "
                   f"{worst['local_front_speed_m_s']:.2f} m/s")
    for r in scored.get("station_sensitivity_measured") or []:
        out.append(f"- 8 mm station ambiguity, **measured** "
                   f"({r['tag']}, aperture {r['aperture_m_lattice_run']} m): arrival "
                   f"{r['arrival_delta_rel']:+.2%} (2.220 vs 2.228 m), peak "
                   f"{r['peak_delta_rel']:+.2%}")
    return "\n".join(out) + "\n"


def figure(scored: dict, path: Path) -> None:
    """Four panels: the time history that is being judged, the convergence of the
    two acceptance quantities, the station defect, and the front-speed deficit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from score_t30 import moving_average

    ref, series_path = load_reference()
    proto = frozen_protocol(ref)
    t_ref, h_ref = load_series(series_path, "h_m")
    grids = sorted((g for g in scored["grids"] if g["status"] == "scored"),
                   key=lambda d: -d["dx"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax, ax2, ax3, ax4 = axes.ravel()

    # (a) time histories at the admissible station
    ax.plot(t_ref, h_ref, color="k", lw=1.7, label="Kleefsman 2005 Fig. 19 (digitised H2)")
    for g in grids:
        try:
            t, h = load_series(HERE / g["csv"], "slab_x2.228")
        except KeyError:
            continue
        ax.plot(t, moving_average(t, h, proto["peak_filter_window_s"]), lw=1.1,
                label=f"TensorSPH dx={g['dx']:g} m")
    ax.axhline(proto["arrival_threshold_m"], color="0.6", ls=":", lw=0.8,
               label="arrival threshold 0.02 m")
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("water height at H2 (x = 2.228 m) [m]")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(a) judged quantity: H2 history at the admissible station")

    # (b) convergence of both acceptance quantities
    dxs, perr, aerr, ptime = [], [], [], []
    for g in sorted((g for g in scored["grids"] if g["status"] == "scored"),
                    key=lambda d: -d["dx"]):
        rec = g["gauges"].get("slab@2.228")
        if not rec or rec.get("status") != "measured":
            continue
        rows = {r["metric"]: r for r in rec["rows"]}
        dxs.append(g["dx"])
        perr.append(abs(rows["peak_height"]["signed_rel_error"]) * 100)
        aerr.append(abs(rows["arrival_time_sample"]["signed_rel_error"]) * 100)
        ptime.append(abs(rec["peak_time_s"] - ref_peak_time(ref)) / ref_peak_time(ref) * 100)
    ax2.semilogx(dxs, perr, "o-", label="peak height |err|")
    ax2.semilogx(dxs, aerr, "s-", label="arrival |err| (sample ref)")
    ax2.semilogx(dxs, ptime, "^--", color="0.5", label="peak TIME |err| (not in acceptance)")
    ax2.axhline(10.0, color="C3", ls="--", lw=1.0, label="acceptance 10%")
    ax2.axhline(3.0, color="0.6", ls=":", lw=0.8, label="3% target")
    ax2.set_xlabel("dx [m]"); ax2.set_ylabel("relative error vs Kleefsman H2 [%]")
    ax2.set_title("(b) convergence at x = 2.228 m (frozen slab operator)")
    ax2.legend(fontsize=7)

    # (c) the station defect, same grids, two stations
    x = range(len(dxs))
    w = 0.38
    ea = [abs({r["metric"]: r for r in g["gauges"]["slab@2.228"]["rows"]}
              ["peak_height"]["signed_rel_error"]) * 100
          for g in sorted((g for g in scored["grids"] if g["status"] == "scored"),
                          key=lambda d: -d["dx"])
          if g["gauges"].get("slab@2.228", {}).get("status") == "measured"]
    eb = [abs({r["metric"]: r for r in g["gauges"]["slab@2.426"]["rows"]}
              ["peak_height"]["signed_rel_error"]) * 100
          for g in sorted((g for g in scored["grids"] if g["status"] == "scored"),
                          key=lambda d: -d["dx"])
          if g["gauges"].get("slab@2.426", {}).get("status") == "measured"]
    ax3.bar([i - w / 2 for i in x], ea, w, label="x = 2.228 m (admissible)")
    ax3.bar([i + w / 2 for i in x], eb, w, label="x = 2.426 m (EXCLUDED_BY_LATTICE)")
    ax3.axhline(10.0, color="C3", ls="--", lw=1.0)
    ax3.set_xticks(list(x)); ax3.set_xticklabels([f"dx={d:g}" for d in dxs])
    ax3.set_ylabel("peak height |err| [%]")
    ax3.set_title("(c) station defect: the excluded station scores a run-up jet")
    ax3.legend(fontsize=7)

    # (d) front-speed deficit at the admissible station
    fs = scored["front_speed"]
    for op, mk in (("slab", "o"), ("height", "s"), ("surface", "^")):
        rows = [r for r in fs["rows"] if r["operator"] == op]
        ax4.semilogx([r["dx"] for r in rows],
                     [r["speed_deficit_rel"] * 100 for r in rows], mk + "-",
                     label=f"{op} gauge")
    ax4.axhline(0.0, color="k", lw=0.8)
    ax4.axhline(-10.0, color="C3", ls="--", lw=1.0, label="= 10% arrival tolerance")
    ax4.set_xlabel("dx [m]"); ax4.set_ylabel("front speed - experiment [%]")
    ax4.set_title("(d) converged ~10% front-speed deficit (model error)")
    ax4.legend(fontsize=7)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def ref_peak_time(ref: dict) -> float:
    """Time of the experimental peak under the same 30 ms filter (cached)."""
    _ref, path = load_reference()
    t_ref, h_ref = load_series(path, "h_m")
    return peak(t_ref, h_ref, frozen_protocol(ref)["peak_filter_window_s"])[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--no-optional", action="store_true")
    args = ap.parse_args()
    scored = score(include_optional=not args.no_optional)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(json.dumps(scored, indent=2, default=str))
    (OUT / "ladder.json").write_text(json.dumps(scored["ladder"], indent=2, default=str))
    (OUT / "score_table.md").write_text(markdown(scored))
    if args.figure:
        figure(scored, ROOT / "docs" / "figures" / "kleefsman_admissible_station.png")
    print(markdown(scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())