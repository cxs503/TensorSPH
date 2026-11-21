"""3-D grid ladder for the Kleefsman (2005) obstacle case.

Why this module exists
----------------------
The 3-D arm of the case exists because the 2-D model blocks 100 % of the tank
width while the experiment blocks only 40 % (0.403 m box in a 1.00 m tank, so
60 % of the water bypasses the box).  Every committed 2-D verdict of this case
failed the peak metric by 15-52 %, and that residual was *attributed* to the
blockage -- but a single 3-D run cannot confirm or refute an attribution:
one grid gives one number with no error bar, so "the 3-D peak is closer" and
"the 3-D peak is further" are equally unfalsifiable statements.

This module turns the 3-D arm into a *ladder*: it reads the committed
``results/<tag>.json`` + ``results/<tag>_time_series.csv`` records, refuses to
compare runs whose protocol fingerprints differ, and then

1. reports the frozen-protocol error of peak and arrival against Kleefsman
   Fig. 19, per grid;
2. fits ``f(dx) = f_inf + C dx^p`` through the ladder (a generalisation of
   Richardson extrapolation that admits a non-uniform refinement ratio, which
   is what dx = 0.05 / 0.04 / 0.03 is) and reports the extrapolated continuum
   value together with the observed order ``p``;
3. decides whether the 3-D arm *converges toward* the experimental value or
   away from it -- i.e. whether the blockage explanation survives being
   measured instead of asserted.

The numerics that are already shared with the 2-D study (moving average,
observed order, Richardson, GCI) are imported from
``benchmarks/kleefsman_obstacle/convergence_analysis.py`` rather than
re-implemented, so both arms of the case are scored by the same code.

Usage::

    python benchmarks/kleefsman_obstacle_3d/ladder.py \
        --results benchmarks/kleefsman_obstacle_3d/results \
        --out benchmarks/kleefsman_obstacle_3d/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "kleefsman_obstacle"))

from convergence_analysis import (  # noqa: E402
    DEFAULT_WINDOW,
    gci_fine,
    moving_average,
    observed_order,
    peak_and_time,
    richardson,
)

# Fields that must agree for two 3-D runs to belong to the same ladder.
FROZEN_PROTOCOL = {
    "t_end",
    "sound_speed",
    "wall_layers",
    "obstacle_z",
    "obstacle_x",
    "wall_operator",
    "viscosity",
    "station_offset",
    "gauge_aperture",
    "arrival_threshold",
}


# --------------------------------------------------------------------------- #
# rung admissibility: the case's own validator has the last word
# --------------------------------------------------------------------------- #
MAX_ADMISSIBLE_DX = 0.161 / 3.0     # box must be >= 3 cells across


def rung_validity(dx: float) -> dict:
    """Is ``dx`` admissible for this case, judged by ``validate_case`` itself?

    A ladder is only as good as its coarsest rung, and the rung that was
    actually submitted on 11-15 (dx = 0.08) is rejected by the case validator:
    a 0.161 m box cannot be resolved by 3 cells of 0.08 m.  The job died on the
    first call with ``ValueError`` after 0 s of compute, so asking the validator
    -- instead of trusting the assumption "the 2-D ladder's rungs transfer" --
    is the difference between a three-level fit and a two-level one.
    """
    from tensorsph.kleefsman_obstacle_3d import Kleefsman3DConfig, validate_case

    cfg = Kleefsman3DConfig(dx=float(dx))
    cfg.dt = 0.9 * cfg.dt_from_courant()
    try:
        validate_case(cfg)
    except ValueError as exc:
        return {"dx": float(dx), "valid": False, "why": str(exc)}
    return {"dx": float(dx), "valid": True, "why": None}


def valid_rungs(dxs: list[float], min_levels: int = 3) -> dict:
    """Split candidate rungs into the admissible ones and the rejected ones."""
    rows = [rung_validity(d) for d in dxs]
    ok = [r["dx"] for r in rows if r["valid"]]
    return {
        "requested": [float(d) for d in dxs],
        "valid": sorted(ok, reverse=True),
        "rejected": [r for r in rows if not r["valid"]],
        "max_admissible_dx": MAX_ADMISSIBLE_DX,
        "n_levels": len(ok),
        "usable": len(ok) >= min_levels,
    }


# --------------------------------------------------------------------------- #
# pure numerics
# --------------------------------------------------------------------------- #
def check_protocol(entries: list[dict]) -> tuple[bool, dict]:
    """Return ``(comparable, disagreements)`` for a set of run records."""
    if len(entries) < 2:
        return True, {}
    keys = sorted(FROZEN_PROTOCOL)
    bad: dict[str, dict] = {}
    checked: list[str] = []
    for key in keys:
        values = {}
        for e in entries:
            v = e.get("fingerprint", {}).get(key)
            if v is None:
                continue
            values[e["tag"]] = round(float(v), 12) if isinstance(v, (int, float)) else v
        if not values:
            continue                    # nobody recorded it: not comparable-on
        checked.append(key)
        if len(values) < len(entries):
            # recorded by some runs only: a mismatch cannot be excluded
            bad[key] = {**values, "_missing": [e["tag"] for e in entries
                                               if e["tag"] not in values]}
            continue
        if len(set(values.values())) > 1:
            bad[key] = values
    if len(checked) < 2 and not bad:
        # a guard that compared nothing is not a passing guard
        bad["_fingerprint"] = {"_checked": checked,
                              "_why": "fewer than 2 protocol keys are present "
                                      "in the run records; comparability "
                                      "cannot be established"}
    return not bad, bad


def fit_power_law(dxs: list[float], values: list[float]) -> dict:
    """Solve ``values[i] = f_inf + C * dxs[i]**p`` for ``(f_inf, C, p)``.

    ``p`` is found by golden-section search on the residual of the two-unknown
    linear least-squares solve for ``(f_inf, C)`` at fixed ``p``.  For two grids
    the problem is under-determined and the function returns
    ``status = "INSUFFICIENT_LEVELS"`` with a plain bracket instead of a
    fabricated order of accuracy.
    """
    pairs = sorted(zip(dxs, values), key=lambda t: -t[0])  # coarse -> fine
    n = len(pairs)
    out: dict = {"n_levels": n, "dx": [d for d, _ in pairs],
                 "values": [v for _, v in pairs]}
    if n < 2:
        out["status"] = "INSUFFICIENT_LEVELS"
        return out
    if n == 2:
        out["status"] = "UNDERDETERMINED"
        out["bracket"] = [pairs[0][1], pairs[1][1]]
        out["gap"] = abs(pairs[1][1] - pairs[0][1])
        out["finest"] = pairs[-1][1]
        return out

    def residual(p: float) -> float:
        # linear least squares for f_inf, C on the over-determined system
        xs = [d ** p for d, _ in pairs]
        ys = [v for _, v in pairs]
        m = len(xs)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sx = sum(xs)
        sy = sum(ys)
        det = m * sxx - sx * sx
        if abs(det) < 1e-18:
            return float("inf")
        c = (m * sxy - sx * sy) / det
        a = (sy - c * sx) / m
        return sum((y - (a + c * x)) ** 2 for x, y in zip(xs, ys))

    lo, hi = 0.1, 8.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    for _ in range(200):
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)
        if residual(x1) < residual(x2):
            b = x2
        else:
            a = x1
    p = 0.5 * (a + b)
    xs = [d ** p for d, _ in pairs]
    ys = [v for _, v in pairs]
    m = len(xs)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx = sum(xs)
    sy = sum(ys)
    det = m * sxx - sx * sx
    c = (m * sxy - sx * sy) / det if abs(det) > 1e-18 else float("nan")
    f_inf = (sy - c * sx) / m
    out.update({"status": "OK", "p_observed": p, "C": c, "f_inf": f_inf,
                "finest": pairs[-1][1], "residual": residual(p)})
    return out


def converges_toward(ladder_fit: dict, reference: float) -> dict:
    """Does the finest grid / the extrapolate sit closer to ``reference``?"""
    if ladder_fit["status"] != "OK":
        return {"status": "NOT_DECIDABLE",
                "why": "need three grids to separate discretisation from noise"}
    finest = ladder_fit["finest"]
    extrap = ladder_fit["f_inf"]
    dxs = ladder_fit["dx"]
    vals = ladder_fit["values"]
    # per-grid distance to the reference, ordered coarse -> fine
    dist = [abs(v - reference) for v in vals]
    monotone = all(dist[i] > dist[i + 1] for i in range(len(dist) - 1))
    return {
        "status": "OK",
        "reference": reference,
        "distance_to_reference": dist,
        "monotone_closing": bool(monotone),
        "finest_rel_error": abs(finest - reference) / reference,
        "extrapolated_rel_error": abs(extrap - reference) / reference,
        "verdict": (
            "BLOCKAGE_HYPOTHESIS_SUPPORTED"
            if monotone and abs(extrap - reference) < abs(vals[0] - reference)
            else "BLOCKAGE_HYPOTHESIS_NOT_SUPPORTED"
        ),
    }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_run(results_dir: Path, tag: str) -> dict:
    meta_path = results_dir / f"{tag}.json"
    csv_path = results_dir / f"{tag}_time_series.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text())
    times: list[float] = []
    slab: list[float] = []
    height: list[float] = []
    bypass: list[int] = []
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                times.append(float(row["t_s"]))
                slab.append(float(row["slab_height_m"]))
                height.append(float(row["height_operator_m"]))
                bypass.append(int(float(row["n_bypass"])))
    run = meta.get("run", {})
    fp = run.get("fingerprint") or {
        "t_end": run.get("t_end"),
        "sound_speed": run.get("sound_speed"),
        "wall_layers": run.get("wall_layers"),
        "obstacle_z": run.get("obstacle_z"),
        "station_offset": run.get("station_offset"),
        "gauge_aperture": run.get("gauge_aperture"),
        "arrival_threshold": run.get("arrival_threshold"),
    }
    # Completeness is decided from the evidence, not from a promise: a record
    # that says nothing about it is complete only if its series actually
    # reaches the requested t_end.  The 11-15 production rungs died at
    # t = 2.04 s / 0.58 s of a 2.6 s window and would otherwise have entered
    # the ladder as ordinary levels.
    t_end = fp.get("t_end")
    reached = bool(times and t_end is not None
                   and times[-1] >= float(t_end) - max(1e-6, 1e-3 * float(t_end)))
    complete = bool(meta["complete"]) if "complete" in meta else reached
    entry = {
        "tag": tag,
        "dx": float(run.get("dx", meta.get("dx", float("nan")))),
        "verdict": meta.get("verdict"),
        "censored": meta.get("censored"),
        "window_covers_reference_peak": meta.get("window_covers_reference_peak"),
        "peak_m": run.get("peak_height_m"),
        "peak_time_s": run.get("peak_time_s"),
        "arrival_s": run.get("arrival_time_s"),
        "samples": len(times),
        "t_last": times[-1] if times else None,
        "max_bypass": max(bypass) if bypass else None,
        "wall_clock_s": run.get("wall_clock_s"),
        "rows": meta.get("rows", []),
        "complete": complete,
        "reached_t_end": reached,
        "fingerprint": fp,
        "_series": (times, slab, height, bypass),
    }
    return entry


def gauge_peaks(entry: dict, window_s: float = DEFAULT_WINDOW) -> dict:
    """Gauge-equivalent peak of the *slab* trace, after a symmetric window."""
    times, slab, height, _ = entry["_series"]
    out: dict = {"window_s": window_s}
    if len(times) < 3:
        return out
    for name, series in (("slab", slab), ("height", height)):
        raw_val, raw_t = peak_and_time(times, series)
        sm = moving_average(times, series, window_s)
        sm_val, sm_t = peak_and_time(times, sm)
        out[name] = {"raw_peak_m": raw_val, "raw_peak_time_s": raw_t,
                     "filtered_peak_m": sm_val, "filtered_peak_time_s": sm_t}
    return out


def analyse(results_dir: Path, tags: list[str]) -> dict:
    entries = []
    for tag in tags:
        try:
            entries.append(load_run(results_dir, tag))
        except FileNotFoundError:
            continue
    if not entries:
        return {"status": "NO_RUNS", "tags": tags}
    comparable, bad = check_protocol(entries)
    inadmissible = {}
    for e in entries:
        v = rung_validity(e["dx"]) if e["dx"] == e["dx"] else {
            "dx": None, "valid": False, "why": "dx not recorded"}
        if not v["valid"]:
            inadmissible[e["tag"]] = v["why"]
    incomplete = [e["tag"] for e in entries if not e.get("complete", True)]
    rows = []
    for e in entries:
        g = gauge_peaks(e)
        rows.append({
            "tag": e["tag"], "dx": e["dx"], "verdict": e["verdict"],
            "censored": e["censored"],
            "window_covers_reference_peak": e["window_covers_reference_peak"],
            "peak_m": e["peak_m"], "peak_time_s": e["peak_time_s"],
            "arrival_s": e["arrival_s"], "samples": e["samples"],
            "t_last": e["t_last"], "max_bypass": e["max_bypass"],
            "wall_clock_s": e["wall_clock_s"],
            "slab_raw_peak_m": g.get("slab", {}).get("raw_peak_m"),
            "slab_filtered_peak_m": g.get("slab", {}).get("filtered_peak_m"),
            "height_raw_peak_m": g.get("height", {}).get("raw_peak_m"),
            "errors": e["rows"],
            "fingerprint": e["fingerprint"],
        })
    out: dict = {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "arm": "3-D (0.403 m box in 1.00 m tank, 40 % blockage)",
        "protocol_comparable": comparable,
        "protocol_disagreements": bad,
        "inadmissible_rungs": inadmissible,
        "incomplete_runs": incomplete,
        "max_admissible_dx": MAX_ADMISSIBLE_DX,
        "runs": rows,
    }
    if not comparable:
        out["status"] = "INCOMPARABLE"
        return out
    if inadmissible or incomplete:
        # a run that the case validator rejects, or that never reached its
        # t_end, is not a level of the ladder
        out["status"] = "RUNGS_REJECTED"
        out["usable_rungs"] = [r["tag"] for r in rows
                               if r["tag"] not in inadmissible
                               and r["tag"] not in incomplete]
        return out
    have = [r for r in rows if r["peak_m"] is not None and r["dx"] == r["dx"]]
    if len(have) >= 2:
        peak_fit = fit_power_law([r["dx"] for r in have], [r["peak_m"] for r in have])
        arr = [r for r in have if r["arrival_s"] is not None]
        arr_fit = (fit_power_law([r["dx"] for r in arr], [r["arrival_s"] for r in arr])
                   if len(arr) >= 2 else {"status": "INSUFFICIENT_LEVELS"})
        out["peak_ladder"] = peak_fit
        out["arrival_ladder"] = arr_fit
        out["peak_vs_reference"] = converges_toward(peak_fit, 0.370)
        out["arrival_vs_reference"] = converges_toward(arr_fit, 0.3591)
    out["status"] = "OK"
    return out


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--out", default=str(here / "results"))
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    tags = args.tags
    if tags is None:
        tags = sorted(p.stem for p in results_dir.glob("dx*_time_series.csv"))
        tags = [t.rsplit("_time_series", 1)[0] for t in tags]
    report = analyse(results_dir, tags)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ladder.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())