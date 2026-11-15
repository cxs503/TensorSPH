"""Where is gauge H2?  Probe-station decidability of the Kleefsman case.

The acceptance metric of ``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`` is the water
level history at the experimental gauge *H2*, and every committed grade of this
case was computed at ``obstacle_upstream_face - 0.05 m = 2.426 m`` -- an
*assumed* station (``KleefsmanConfig.probe_offset``).  ``m21173`` left the
station as the highest-priority open cause because it alone decides whether the
arrival criterion passes.

This module adds the one piece of geometry evidence that the provenance chain in
``reference.json`` never used: the setup figure of Amaro Junior, Cheng &
Buruchenko (2020, Fig. 2a) carries **three equal dimension strings of 0.496 m
between the four wave probes** H1..H4, i.e. the probes are equally spaced by
``S = 0.496 m``.  Together with the two statements of the primary source

* Kleefsman et al. (2005) p. 381: "Four vertical height probes have been used;
  one in the reservoir and the other three in the tank"
* Kleefsman et al. (2005) Fig. 19 caption (p. 383): "Vertical water heights in
  the reservoir H4 (left) and the tank H2 (right)"

the spacing turns the unknown station into a one-parameter lattice,
``x(H2) = a + 2 S`` with ``a = x(H4)`` measured from the reservoir end, and the
constraints ``H4`` in the reservoir (``a <= column_length``) and ``H2`` in the
tank upstream of the box bound it:

    0.236 < a <= 1.228  =>  1.228 < x(H2) <= 2.220 m,

which **excludes the committed 2.426 m station**: that would need
``a = 2.426 - 2*0.496 = 1.434 m > 1.228 m``, i.e. H4 inside the tank, against
the primary source.

Two anchors inside the admissible interval are supported by a printed string:

* ``anchor_gate``      ``a = 1.228``  (H4 at the reservoir/tank gate) -> H2 = 2.220 m
* ``anchor_far_wall``  H1 = 3.22 - 1.150 = 2.070 m (the remaining 1.150 m string
  dimensions the far wall to H1) -> ``a = 2.070 - 3 S = 0.582`` -> H2 = 1.574 m

The script (a) writes the lattice audit and (b) can *measure* the frozen slab
metric at those stations in one run, so the verdict bracket is computed from the
solver instead of from the two-point linear extrapolation of ``m21173``.

Usage::

    python benchmarks/kleefsman_obstacle/station_lattice.py --audit-only
    python benchmarks/kleefsman_obstacle/station_lattice.py --dx 0.04 --t-end 1.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "station_lattice"

# --- evidence, machine-readable -------------------------------------------------
SPACING_M = 0.496          # Amaro Junior et al. (2020) Fig. 2a, three equal strings
ARRIVAL_THRESHOLD_M = 0.02  # frozen primary operator (reference.json)
# committed station candidates, upstream of the box face [m]
CANDIDATE_OFFSETS = (0.05, 0.256, 0.902, 1.148)   # 2.426, 2.220, 1.574, 1.328


def geometry() -> dict:
    ref = json.loads((HERE / "reference.json").read_text())
    return ref


def lattice_audit(ref: dict) -> dict:
    """Enumerate the admissible probe lattice; no solver involved."""
    g = ref["geometry_m"]
    tank = g["tank_length"]
    column = g["column_length"]
    box_face = g["obstacle_upstream_face_x"]

    a_hi = min(column, tank - 3.0 * SPACING_M)          # H4 in the reservoir, H1 in the tank
    a_lo = max(0.0, column - 2.0 * SPACING_M)           # H2 downstream of the gate
    h2_lo = a_lo + 2.0 * SPACING_M
    h2_hi = a_hi + 2.0 * SPACING_M

    committed = box_face - ref["metric_definition"]["frozen_primary"]["station_offset_m"]
    need_a = committed - 2.0 * SPACING_M

    anchors = {
        "anchor_gate": {
            "a_m": column,
            "why": "the only string that fixes H4 absolutely is the reservoir length: "
                   "H4 sits at the reservoir/tank gate 1.228 m from the closed end",
        },
        "anchor_far_wall": {
            "a_m": (tank - 1.150) - 3.0 * SPACING_M,
            "why": "the five figure strings must all dimension something; 1.150 m from "
                   "the far (open) wall to H1 puts H1 at 2.070 m and H4 at 0.582 m, "
                   "i.e. inside the reservoir",
        },
    }
    for name, anc in anchors.items():
        anc["H2_m"] = anc["a_m"] + 2.0 * SPACING_M
        anc["H3_m"] = anc["a_m"] + 1.0 * SPACING_M
        anc["H1_m"] = anc["a_m"] + 3.0 * SPACING_M
        anc["admissible"] = bool(a_lo <= anc["a_m"] <= a_hi)

    return {
        "case": ref["case"],
        "question": "x of the experimental gauge H2 that the acceptance metric is compared against",
        "evidence": {
            "probe_spacing_m": SPACING_M,
            "probe_spacing_source": "Amaro Junior, Cheng & Buruchenko (2020) Fig. 2a, three "
                                    "equal 0.496 m strings between the four wave probes",
            "one_probe_in_reservoir": "Kleefsman et al. (2005) p.381 (probes H1..H4); "
                                      "Fig. 19 caption p.383 names H4 as the reservoir probe",
            "h2_location": "Kleefsman et al. (2005) p.382: 'in the tank just in front of the box'",
            "unused_strings": ["1.150", "0.496", "0.496", "0.496"],
            "why_now": "the committed provenance chain (geometry_provenance) reads only "
                       "1.228/1.248/0.744/0.161 and therefore never constrained the probe "
                       "station: probe_offset = 0.05 m is a convention, not a measurement",
        },
        "geometry": {"tank_length": tank, "column_length": column, "box_upstream_face": box_face},
        "constraints": {
            "H4_in_reservoir": f"0 <= a <= {column}",
            "H1_inside_tank": f"a + 3S <= {tank} -> a <= {tank - 3*SPACING_M}",
            "H2_in_tank_downstream_of_gate": f"a + 2S > {column} -> a > {column - 2*SPACING_M}",
            "admissible_a_interval": [a_lo, a_hi],
            "admissible_H2_interval": [h2_lo, h2_hi],
        },
        "committed_station": {
            "x_m": committed,
            "implied_a_m": need_a,
            "admissible": bool(a_lo <= need_a <= a_hi),
            "verdict": "EXCLUDED_BY_LATTICE" if not (a_lo <= need_a <= a_hi) else "admissible",
            "why": f"the committed station needs a = {need_a:.3f} m > column_length "
                   f"{column:.3f} m, i.e. H4 inside the tank, contradicting the primary source",
        },
        "anchors": anchors,
        "consequence": "no committed grade of this case was taken at a station the probe "
                       "lattice admits; the arrival verdict must be re-evaluated at the "
                       "admissible anchors",
    }


def run_lattice(dx: float, t_end: float, sound_speed: float = 20.0,
                stations_offsets: tuple[float, ...] = CANDIDATE_OFFSETS) -> dict:
    """Measure the frozen slab metric at several candidate stations in one run."""
    import time

    from tensorsph.kleefsman_obstacle import KleefsmanConfig, run_case, validate_case

    ref = json.loads((HERE / "reference.json").read_text())
    g = ref["geometry_m"]
    cfg = KleefsmanConfig(
        dx=dx, t_end=t_end, device="cpu",
        tank_length=g["tank_length"], tank_height=g["tank_height"],
        column_length=g["column_length"], column_height=g["column_height"],
        obstacle_x=g["obstacle_upstream_face_x"],
        obstacle_length=g["obstacle_length"], obstacle_height=g["obstacle_height"],
        pool_width=g["tank_length"], pool_depth=g["tank_height"],
        sound_speed=sound_speed, stations_offset=tuple(stations_offsets))
    cfg.dt = min(2.0e-4, cfg.dt_acoustic())
    cfg.gauge_aperture = ref["metric_definition"]["aperture_m"]
    validate_case(cfg)
    t0 = time.monotonic()
    run = run_case(cfg)
    wall = time.monotonic() - t0

    exp = ref["experimental_values"]
    tol = ref["tolerance"]["relative"]
    stations = {}
    for tag in sorted(run.series, key=lambda k: k):
        if not tag.startswith("slab@"):
            continue
        x = float(tag.split("@x")[1])
        series = run.series[tag]
        # 30 ms symmetric moving average: the frozen primary protocol
        peak, peak_t, arrival = _filtered_metrics(run.times, series,
                                                  ref["metric_definition"]["frozen_primary"]
                                                  ["arrival_threshold_m"], window=0.03)
        stations[f"{x:.3f}"] = {
            "x_m": x,
            "slab_peak_m": peak, "slab_peak_time_s": peak_t, "slab_arrival_s": arrival,
            "peak_rel_error": abs(peak - exp["peak_height_m"]) / exp["peak_height_m"],
            "arrival_rel_error": abs(arrival - exp["arrival_time_s"]) / exp["arrival_time_s"],
            "arrival_pass": bool(abs(arrival - exp["arrival_time_s"]) / exp["arrival_time_s"] < tol),
            "peak_pass": bool(abs(peak - exp["peak_height_m"]) / exp["peak_height_m"] < tol),
        }
    return {
        "dx": dx, "t_end": t_end, "sound_speed": sound_speed,
        "wall_time_s": round(wall, 1), "dt": cfg.dt, "aperture_m": cfg.gauge_aperture,
        "reference": {k: exp[k] for k in ("peak_height_m", "arrival_time_s", "arrival_time_s_interpolated")},
        "tolerance": tol,
        "stations": stations,
    }


def _filtered_metrics(times, series, threshold, window=0.03):
    """30 ms symmetric moving average, then peak/arrival (reference.json protocol)."""
    n = len(series)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    half = max(1, int(round(window / max(times[-1] - times[0], 1e-9) * (n - 1) / 2)))
    smooth = []
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        seg = series[lo:hi]
        smooth.append(sum(seg) / len(seg))
    peak = max(smooth)
    peak_t = times[smooth.index(peak)]
    arrival = float("nan")
    for k in range(1, n):
        if smooth[k - 1] < threshold <= smooth[k]:
            h0, h1 = smooth[k - 1], smooth[k]
            frac = 0.0 if h1 == h0 else (threshold - h0) / (h1 - h0)
            arrival = times[k - 1] + frac * (times[k] - times[k - 1])
            break
    return peak, peak_t, arrival


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=1.2)
    ap.add_argument("--sound-speed", type=float, default=20.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads((HERE / "reference.json").read_text())
    audit = lattice_audit(ref)
    (OUT / "lattice.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({k: audit[k] for k in ("constraints", "committed_station", "anchors")},
                     indent=1))
    if args.audit_only or args.dx is None:
        return 0
    res = run_lattice(args.dx, args.t_end, args.sound_speed)
    tag = f"dx{args.dx:.2f}_t{args.t_end:.1f}"
    (OUT / f"stations_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["stations"].items():
        print(f"x={k}  peak={v['slab_peak_m']:.4f} ({v['peak_rel_error']:+.1%})  "
              f"arrival={v['slab_arrival_s']:.4f} ({v['arrival_rel_error']:+.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())