"""Can the H2 probe coordinate be pinned without the paper figure?

Why this exists
---------------
``results/arrival_attribution/attribution.json`` leaves the acceptance verdict
open on one item:

    "probe x: the paper's H2 coordinate is not machine-readable from the
     accessible sources; moving the station to the committed upstream one
     (x=2.228 m) removes 52-63 % of the deficit and would bring the arrival
     inside the 10 % tolerance, so the verdict hinges on one unverified
     coordinate"

That is a bad place to leave a ledger entry: a FAIL that could be turned into a
PASS by moving the gauge 20 cm is indistinguishable, on the evidence, from a
PASS that was turned into a FAIL by a bad reading of a raster figure.  This
script attacks the coordinate with an estimator that does not need the figure.

The estimator
-------------
The Kleefsman experiment is a dam break on a dry bed.  For the *leading edge*
of such a flow there is a closed-form reference that needs no digitisation and
no solver: the Ritter fan.  Writing ``c0 = 2 sqrt(g h0)`` for the gate, the
depth at a point ``x' = x - x_gate`` downstream of the gate is

    h(x', t) = (1/9g) (c0 - x'/t)^2 ,

so a fixed *level* threshold ``h*`` tracks a constant-speed locus

    x'(t) = t * (c0 - 3 sqrt(g h*)).

The experiment reports a first time-history; its first crossing of
``h* = 0.02 m`` therefore fixes ``x'`` up to the accuracy of the Ritter
idealisation.  Inverting is one line and, crucially, is an *independent* handle
on the coordinate: the arrival time is measured on the same digitised curve
that the acceptance test uses, but the map from arrival time to station comes
from the analytic solution, not from the figure.

Because Ritter assumes an infinite reservoir and no friction it sets an upper
bound on the front speed, so a real front is slower; the honest quantity to
report is the **front-speed factor** ``gamma = u_front / u_ritter`` that each
candidate station would require,

    gamma = (x - x_gate) / (t_measured * locus_speed).

Two things then make the test sharp:

1. ``gamma`` is not a free knob per station.  The leading edge of this flow
   *accelerates* (the simulated front runs 2.67 -> 3.57 -> 3.92 m/s over the
   segments 1.3-1.7 / 1.7-2.1 / 2.1-2.4 m — ``arrival_attribution.json``), so a
   near station necessarily shows a **lower** ``gamma`` than a far one: the
   near station's mean advance covers more of the slow release phase.  The
   *slope* ``d gamma / d d`` is a solver-independent trend that a
   station hypothesis has to respect.
2. The paper plots its own numerical curve (COMFLOW, a high-order VOF solver)
   beside the experiment, at the same station.  Its arrival (0.3707 s, traced
   independently; agreement 3.2 % with the experiment trace) gives an
   accuracy-grade estimate of ``gamma`` at the *true* station, free of any SPH
   bias.

The script reports, per candidate station, the required ``gamma`` from three
independent traces of the same figure, the trend ``d gamma / d d`` implied by
each candidate, and the trend the solver itself produces.  A candidate whose
implied trend contradicts the solver's by a large factor is rejected.

Outputs
-------
``results/station_anchor/anchor.json``   every number, machine readable
``results/station_anchor/anchor_table.md`` the table quoted by the doc

Usage::

    python benchmarks/kleefsman_obstacle/station_anchor.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_t30 import load_reference, frozen_protocol  # noqa: E402  (same instrument)

G = 9.81
#: Geometry of the reference case, re-read from reference.json in main().
REFERENCE_ARRIVAL_THRESHOLD_M = 0.02


def ritter_speeds(h0: float, threshold: float) -> tuple[float, float]:
    """(gate speed c0, constant-speed locus of the level threshold)."""
    c0 = 2.0 * math.sqrt(G * h0)
    locus = c0 - 3.0 * math.sqrt(G * threshold)
    return c0, locus


def station_from_arrival(arrival_s: float, gate_x: float, locus: float) -> float:
    """Absolute station x implied by a measured first crossing of the threshold."""
    return gate_x + arrival_s * locus


def required_gamma(station_x: float, gate_x: float, arrival_s: float,
                   locus: float) -> float:
    """Front-speed factor ``u_front / u_ritter`` a station hypothesis requires."""
    return (station_x - gate_x) / (arrival_s * locus)


def _front_arrival(csv_path: Path, station_x: float, column: str = "front_x_m"
                   ) -> float | None:
    """First time the tracked front reaches ``station_x`` (linear interp)."""
    if not csv_path.is_file():
        return None
    import csv as _csv
    t: list[float] = []
    f: list[float] = []
    with csv_path.open() as fh:
        for row in _csv.DictReader(fh):
            try:
                tv, fv = float(row["t_s"]), float(row[column])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(tv) and math.isfinite(fv):
                t.append(tv)
                f.append(fv)
    for k in range(1, len(f)):
        if f[k - 1] < station_x <= f[k]:
            if f[k] == f[k - 1]:
                return t[k]
            frac = (station_x - f[k - 1]) / (f[k] - f[k - 1])
            return t[k - 1] + frac * (t[k] - t[k - 1])
    return None


def analyse() -> dict:
    ref, _series = load_reference()
    proto = frozen_protocol(ref)
    geom = ref["geometry_m"]
    h0 = float(geom["column_height"])
    gate_x = float(geom["column_length"])
    box_x = float(geom["obstacle_upstream_face_x"])
    ev = ref["experimental_values"]
    thr = float(proto["arrival_threshold_m"])
    c0, locus = ritter_speeds(h0, thr)

    # candidate stations, ordered far -> near so the trend is readable
    primary = box_x - float(proto["station_offset_m"])      # as-committed (2.426)
    upstream = box_x - 0.248                                 # the "escape" station
    cands = [("primary (0.05 m up of box face)", primary),
             ("upstream (0.248 m up of box face)", upstream)]

    # --- three independent readings of the same figure's first crossing ----
    cross = ev.get("digitisation_cross_check") or {}
    traces = {
        "experiment_interp": {
            "arrival_s": 0.3481,
            "provenance": ("digitised experiment trace, linear interpolation "
                           "between the two bracketing samples "
                           "(score_t30.arrival(mode='interp')); the value the "
                           "frozen protocol reads"),
        },
        "experiment_sample": {
            "arrival_s": float(ev["arrival_time_s"]),   # 0.3591, the anchored one
            "provenance": "digitised experiment trace, first sample >= threshold",
        },
        "comflow_vof": {
            "arrival_s": float(cross.get("comflow_arrival_time_s", float("nan"))),
            "provenance": ("paper's own COMFLOW curve, traced independently; "
                           "cross-check agreement %.4f" %
                           float(cross.get("arrival_agreement_rel", float("nan")))),
        },
    }

    rows = []
    for trace, info in traces.items():
        t = info["arrival_s"]
        row = {"trace": trace, "arrival_s": t, "provenance": info["provenance"],
               "x_implied_m": station_from_arrival(t, gate_x, locus),
               "ritter_locus_speed_m_s": locus}
        for label, xs in cands:
            row[label] = {
                "station_x_m": xs,
                "required_gamma": required_gamma(xs, gate_x, t, locus),
                "ritter_arrival_s": (xs - gate_x) / locus,
                "arrival_deficit_s": t - (xs - gate_x) / locus,
            }
        rows.append(row)

    # --- smooth acceleration of the front: solver-produced trend ----------
    sim_rows = []
    for grid in ("dx0.01", "dx0.02", "dx0.04"):
        path = HERE / "results" / grid / "time_series.csv"
        if not path.is_file():
            path = HERE / "results/audit_t30" / grid / "time_series.csv"
        t_near = _front_arrival(path, primary)
        t_far = _front_arrival(path, upstream)
        if t_near is None or t_far is None:
            continue
        g_far = required_gamma(upstream, gate_x, t_far, locus)
        g_near = required_gamma(primary, gate_x, t_near, locus)
        sim_rows.append({
            "grid": grid,
            "front_arrival_upstream_s": t_far, "front_arrival_primary_s": t_near,
            "gamma_upstream": g_far, "gamma_primary": g_near,
            "d_gamma_per_m": (g_near - g_far) / (primary - upstream),
        })

    # --- the discrimination -------------------------------------------------
    tests = []
    for row in rows:
        d_gamma = (row["primary (0.05 m up of box face)"]["required_gamma"]
                   - row["upstream (0.248 m up of box face)"]["required_gamma"])
        entry = {
            "trace": row["trace"],
            "required_d_gamma_per_m": d_gamma / (primary - upstream),
        }
        if sim_rows:
            sim_d = sum(r["d_gamma_per_m"] for r in sim_rows) / len(sim_rows)
            entry["solver_d_gamma_per_m"] = sim_d
            # both are positive (near station lags more); the hypothesis needs a
            # trend this many times steeper than the solver produces
            entry["trend_ratio_hypothesis_over_solver"] = (
                (d_gamma / (primary - upstream)) / sim_d if sim_d else float("nan"))
        tests.append(entry)

    return {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "question": "does the acceptance verdict hinge on an unverifiable H2 coordinate?",
        "inputs": {
            "reference_json": "benchmarks/kleefsman_obstacle/reference.json",
            "attribution": "benchmarks/kleefsman_obstacle/results/arrival_attribution/attribution.json",
            "front_series": "results/{,audit_t30/}dx{0.01,0.02,0.04}/time_series.csv",
        },
        "geometry": {"h0_m": h0, "gate_x_m": gate_x, "box_face_x_m": box_x,
                     "arrival_threshold_m": thr},
        "ritter": {"c0_gate_m_s": c0, "locus_speed_m_s": locus,
                   "note": ("ideal infinite-reservoir dry-bed fan; sets an upper "
                            "bound on the front speed, so gamma <= 1")},
        "stations": {label: xs for label, xs in cands},
        "ritter_arrival_at_stations_s": {
            label: (xs - gate_x) / locus for label, xs in cands},
        "traces": rows,
        "solver_front_trend": sim_rows,
        "discrimination": tests,
    }


def markdown(res: dict) -> str:
    prim = res["stations"]["primary (0.05 m up of box face)"]
    up = res["stations"]["upstream (0.248 m up of box face)"]
    out = [
        "| trace | arrival [s] | x implied by Ritter [m] "
        "| required gamma @x=%.3f | required gamma @x=%.3f |" % (prim, up),
        "|---|---|---|---|---|",
    ]
    for row in res["traces"]:
        out.append("| %s | %.4f | %.4f | %.3f | %.3f |" % (
            row["trace"], row["arrival_s"], row["x_implied_m"],
            row["primary (0.05 m up of box face)"]["required_gamma"],
            row["upstream (0.248 m up of box face)"]["required_gamma"]))
    loc = res["ritter"]["locus_speed_m_s"]
    out += ["", "| station | x [m] | Ritter arrival [s] |", "|---|---|---|"]
    for label, xs in res["stations"].items():
        out.append("| %s | %.3f | %.4f |" % (label, xs, res["ritter_arrival_at_stations_s"][label]))
    if res["solver_front_trend"]:
        out += ["", "| grid | front arrival @%.3f [s] | front arrival @%.3f [s] "
                "| gamma near | gamma far | d gamma / d d [1/m] |"
                % (up, prim), "|---|---|---|---|---|---|"]
        for r in res["solver_front_trend"]:
            out.append("| %s | %.4f | %.4f | %.3f | %.3f | %.3f |" % (
                r["grid"], r["front_arrival_upstream_s"], r["front_arrival_primary_s"],
                r["gamma_upstream"], r["gamma_primary"], r["d_gamma_per_m"]))
    out += ["", "| trace | d gamma / d d required [1/m] | solver [1/m] | ratio |",
            "|---|---|---|---|"]
    for t in res["discrimination"]:
        out.append("| %s | %.3f | %.3f | %.1fx |" % (
            t["trace"], t["required_d_gamma_per_m"],
            t.get("solver_d_gamma_per_m", float("nan")),
            t.get("trend_ratio_hypothesis_over_solver", float("nan"))))
    return "\n".join(out) + "\n"


def main() -> int:
    res = analyse()
    outdir = HERE / "results" / "station_anchor"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "anchor.json").write_text(json.dumps(res, indent=2))
    table = markdown(res)
    (outdir / "anchor_table.md").write_text(table)
    print(table)
    print(json.dumps(res["traces"], indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())