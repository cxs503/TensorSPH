"""Score the Kleefsman (2005) H2 water-height history against the digitised experiment.

Why this file exists
--------------------
The case ledger criterion is: *peak height in front of the obstacle* and *arrival time*
vs the Kleefsman et al. (2005) experiment, with a 10 % tolerance.  Three separate
things can silently move that number:

1. **window**  -- the experimental *global* peak at H2 sits at t = 1.703 s (the flow
   returning from the back wall re-impinges).  A run that stops at 1.0-1.2 s only sees
   a *local* peak of a different physical event, so comparing it with the experimental
   global peak compares two different quantities.
2. **operator** -- three gauge operators are recorded per station (legacy top-N mean,
   half-cell-corrected surface level, and a dx-independent fixed-aperture slab gauge);
   they differ by 0.4-1.4 dx on a smooth surface and much more inside a run-up wedge.
3. **station**  -- the paper only says H2 is "in the tank just in front of the box"
   (Fig. 19 text); the coordinate is not machine-readable from Fig. 17, which is a
   photograph pair.  Every plausible station is therefore scored and the spread is
   reported instead of a single guessed point.

This script reads ``artifacts/<tag>/time_series.csv`` (written by ``run.py``), scores
every (operator, station) pair against the digitised reference, and emits

* ``results/h2_score/<tag>.json``  -- full matrix for that run,
* ``results/h2_score/verdict.json`` -- cross-run summary + window adequacy + verdict,
* ``results/h2_score/h2_timeseries.png`` -- reference vs simulation histories,
* ``results/h2_score/h2_error_envelope.png`` -- relative-error envelope vs station.

Everything is injected programmatically (no hand-copied numbers).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF_JSON = HERE / "reference.json"
REF_SERIES = ROOT / "docs" / "m20884" / "outputs" / "kleefsman_h2_ref" / "h2_fig19.csv"
OUTDIR = HERE / "results" / "h2_score"

# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #


def load_reference(path: Path = REF_JSON) -> dict:
    ref = json.loads(path.read_text())
    exp = ref["experimental_values"]
    for key in ("peak_height_m", "arrival_time_s"):
        if exp.get(key) is None:
            raise ValueError(f"reference has no numeric {key}; the criterion is WITHHELD")
    return ref


def load_reference_series(path: Path = REF_SERIES) -> tuple[list[float], list[float]]:
    """Digitised ``(t, h)`` of the experimental H2 trace (Kleefsman Fig. 19, right panel)."""
    if not path.exists():
        return [], []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    t = [float(r["t_s"]) for r in rows]
    h = [float(r["h_m"]) for r in rows]
    return t, h


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def arrival_time(times: list[float], values: list[float], threshold: float) -> float:
    """First crossing of ``threshold``, linearly interpolated between samples.

    ``nan`` when the series never reaches ``threshold`` (an honest "not observed",
    never a zero).
    """
    for i in range(1, len(values)):
        v0, v1 = values[i - 1], values[i]
        if v1 >= threshold:
            if v0 >= threshold:
                return float(times[i - 1])
            span = v1 - v0
            if span <= 0.0:
                return float(times[i])
            frac = (threshold - v0) / span
            return float(times[i - 1] + frac * (times[i] - times[i - 1]))
    if values and values[0] >= threshold:
        return float(times[0])
    return float("nan")


def peak_metrics(times: list[float], values: list[float],
                 threshold: float) -> dict:
    if not times:
        return {"peak_m": float("nan"), "peak_time_s": float("nan"),
                "arrival_s": float("nan")}
    peak = max(values)
    return {"peak_m": peak, "peak_time_s": float(times[values.index(peak)]),
            "arrival_s": arrival_time(times, values, threshold)}


def rel_error(got: float, ref: float) -> float:
    if got is None or ref is None or ref == 0 or (isinstance(got, float) and math.isnan(got)):
        return float("nan")
    return abs(got - ref) / abs(ref)


def front_arrival(times: list[float], front: list[float], station: float) -> float:
    """Time at which the tracked wave front passes ``station`` [m] (probe-independent)."""
    for i in range(1, len(front)):
        if front[i] >= station:
            if front[i - 1] >= station:
                return float(times[i - 1])
            span = front[i] - front[i - 1]
            if span <= 0:
                return float(times[i])
            return float(times[i - 1] + (station - front[i - 1]) / span
                         * (times[i] - times[i - 1]))
    return float("nan")


# --------------------------------------------------------------------------- #
# run loader
# --------------------------------------------------------------------------- #


def load_run(csv_path: Path) -> dict:
    """Read one ``time_series.csv`` into ``{times, front, gauges{col: values}}``."""
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty time series: {csv_path}")
    cols = rows[0].keys()
    times = [float(r["t_s"]) for r in rows]
    front = [float(r["front_x_m"]) for r in rows] if "front_x_m" in cols else []
    gauges = {}
    for c in cols:
        if "@" in c or "_x" in c and c.split("_")[0] in ("height", "surface", "slab"):
            try:
                gauges[c] = [float(r[c]) for r in rows]
            except (TypeError, ValueError):
                continue
    return {"times": times, "front": front, "gauges": gauges,
            "t_end": times[-1] if times else float("nan")}


def split_gauge(col: str) -> tuple[str, float]:
    """``'slab_x2.426'`` -> ``('slab', 2.426)``."""
    op, _, xs = col.partition("_x")
    return op, float(xs)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def score_run(run: dict, ref: dict) -> dict:
    exp = ref["experimental_values"]
    tol = ref["tolerance"]["relative"]
    thr = exp.get("arrival_threshold_m", 0.02)
    box_face = ref["geometry_m"]["obstacle_upstream_face_x"]
    # a window that stops before the experimental global peak cannot be compared to it
    window_covers_peak = bool(run["t_end"] >= exp["peak_time_s"])

    cells = []
    for col, values in run["gauges"].items():
        op, xs = split_gauge(col)
        m = peak_metrics(run["times"], values, thr)
        cells.append({
            "operator": op, "station_x_m": xs,
            "peak_m": m["peak_m"], "peak_time_s": m["peak_time_s"],
            "arrival_s": m["arrival_s"],
            "peak_err_rel": rel_error(m["peak_m"], exp["peak_height_m"]),
            "arrival_err_rel": rel_error(m["arrival_s"], exp["arrival_time_s"]),
            "peak_pass": bool(rel_error(m["peak_m"], exp["peak_height_m"]) < tol),
            "arrival_pass": bool(rel_error(m["arrival_s"], exp["arrival_time_s"]) < tol),
        })
    t_front = front_arrival(run["times"], run["front"], box_face) if run["front"] else float("nan")
    return {"t_end_s": run["t_end"],
            "window_covers_experimental_peak": window_covers_peak,
            "front_arrival_at_box_face_s": t_front,
            "front_arrival_err_rel": rel_error(t_front, exp["arrival_time_s"]),
            "cells": cells}


def envelope(per_run: dict) -> dict:
    """Spread of the verdict over (operator, station) -- a robustness statement."""
    ops = {}
    for tag, res in per_run.items():
        for c in res["cells"]:
            ops.setdefault(c["operator"], {"peak_err_rel": [], "arrival_err_rel": [],
                                           "peak_pass": [], "arrival_pass": []})
            ops[c["operator"]]["peak_err_rel"].append(c["peak_err_rel"])
            ops[c["operator"]]["arrival_err_rel"].append(c["arrival_err_rel"])
            ops[c["operator"]]["peak_pass"].append(c["peak_pass"])
            ops[c["operator"]]["arrival_pass"].append(c["arrival_pass"])
    out = {}
    for op, d in ops.items():
        out[op] = {
            "peak_err_rel_min": min(d["peak_err_rel"]),
            "peak_err_rel_max": max(d["peak_err_rel"]),
            "arrival_err_rel_min": min(d["arrival_err_rel"]),
            "arrival_err_rel_max": max(d["arrival_err_rel"]),
            "any_peak_pass": bool(any(d["peak_pass"])),
            "any_arrival_pass": bool(any(d["arrival_pass"])),
            "n_cells": len(d["peak_err_rel"]),
        }
    return out


def verdict(per_run: dict, tol: float) -> dict:
    """PASS only if *every* operator x station cell passes with a full window.

    A verdict is not allowed to be rescued by picking the operator or station that
    happens to agree: the criterion is about the case, not about post-processing.
    """
    cells = [c for res in per_run.values() if res["window_covers_experimental_peak"]
             for c in res["cells"]]
    full_window = bool(cells)
    if not full_window:
        return {"verdict": "WINDOW_LIMITED", "n_cells": 0,
                "reason": "no run reaches the experimental global peak at t="
                          f"{per_run and list(per_run.values())[0].get('t_end_s')} s"}
    worst_peak = max(c["peak_err_rel"] for c in cells)
    worst_arrival = max(c["arrival_err_rel"] for c in cells)
    best_peak = min(c["peak_err_rel"] for c in cells)
    best_arrival = min(c["arrival_err_rel"] for c in cells)
    ok = worst_peak < tol and worst_arrival < tol
    return {"verdict": "PASS" if ok else "FAIL", "n_cells": len(cells),
            "worst_peak_err_rel": worst_peak, "worst_arrival_err_rel": worst_arrival,
            "best_peak_err_rel": best_peak, "best_arrival_err_rel": best_arrival,
            "tolerance": tol}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(per_run: dict, ref_t: list[float], ref_h: list[float],
                 ref: dict, outdir: Path = OUTDIR) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    exp = ref["experimental_values"]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    if ref_t:
        ax.plot(ref_t, ref_h, color="k", lw=1.8, label="Kleefsman 2005 experiment (digitised H2)")
    ax.axvline(exp["peak_time_s"], color="k", ls=":", lw=0.8)
    ax.axhline(exp.get("arrival_threshold_m", 0.02), color="0.6", ls=":", lw=0.8)
    colours = ["C0", "C1", "C2", "C3"]
    for (tag, res), colr in zip(per_run.items(), colours):
        run = res["_run"]
        for i, (col, values) in enumerate(sorted(run["gauges"].items())):
            op, xs = split_gauge(col)
            if op != "slab":
                continue
            ax.plot(run["times"], values, lw=1.0, ls="--", color=colr, alpha=0.85,
                    label=f"{tag} slab@x={xs:.3f}")
    ax.set_xlim(0, max([r["t_end_s"] for r in per_run.values()] + [2.0]))
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height at H2 / probe [m]")
    ax.set_title("Kleefsman dam-break-with-obstacle: H2 water height, experiment vs TensorSPH")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    p = outdir / "h2_timeseries.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, metric, ref_v in ((axes[0], "peak_err_rel", exp["peak_height_m"]),
                              (axes[1], "arrival_err_rel", exp["arrival_time_s"])):
        for tag, res in per_run.items():
            cells = sorted(res["cells"], key=lambda c: (c["operator"], -c["station_x_m"]))
            for op, marker in (("height", "o"), ("surface", "s"), ("slab", "^")):
                sel = [c for c in cells if c["operator"] == op]
                if not sel:
                    continue
                ax.plot([c["station_x_m"] for c in sel], [c[metric] for c in sel],
                        marker=marker, ms=4, lw=0.9, label=f"{tag} {op}")
        ax.axhline(ref["tolerance"]["relative"], color="C3", ls="--", lw=1.0,
                   label=f"{ref['tolerance']['relative']:.0%} tolerance")
        ax.set_xlabel("gauge station x [m] (box face = "
                      f"{ref['geometry_m']['obstacle_upstream_face_x']:.3f} m)")
        ax.set_ylabel("relative error")
        ax.set_title(metric.replace("_rel", " error vs station"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    p = outdir / "h2_error_envelope.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", default=None,
                    help="run tags under artifacts/<tag>/time_series.csv")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    ref = load_reference()
    exp = ref["experimental_values"]
    ref_t, ref_h = load_reference_series()

    tags = args.tags or sorted(p.parent.name for p in
                               (HERE / "artifacts").glob("*/time_series.csv"))
    per_run = {}
    for tag in tags:
        # a tag resolves under artifacts/ (run.py output); a path with a separator is
        # taken as-is so the tracked ladder under results/ can be re-scored directly
        if "/" in tag or tag.endswith(".csv"):
            path = Path(tag)
            if path.is_dir():
                path = path / "time_series.csv"
            tag = path.parent.name
        else:
            path = HERE / "artifacts" / tag / "time_series.csv"
        if not path.exists():
            print(f"skip {tag}: no {path}")
            continue
        run = load_run(path)
        res = score_run(run, ref)
        res["_run"] = run
        per_run[tag] = res
    if not per_run:
        print("no runs scored")
        return 1

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, res in per_run.items():
        clean = {k: v for k, v in res.items() if k != "_run"}
        (OUTDIR / f"{tag}.json").write_text(json.dumps(clean, indent=2))

    summary = {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "reference": {"peak_height_m": exp["peak_height_m"],
                      "peak_time_s": exp["peak_time_s"],
                      "arrival_time_s": exp["arrival_time_s"],
                      "probe": exp.get("probe_id"), "source": exp.get("source")},
        "runs": {t: {k: v for k, v in r.items() if k != "_run"} for t, r in per_run.items()},
        "envelope": envelope(per_run),
        "verdict": verdict(per_run, ref["tolerance"]["relative"]),
        "obstacle_blockage": {
            "box_width_m": 0.403, "tank_width_m": 1.0,
            "blocked_fraction_3d_experiment": 0.403 / 1.0,
            "blocked_fraction_2d_model": 1.0,
            "note": "the 2D model makes the 0.403 m box span the whole 1.0 m width, so 100 % of "
                    "the flow must climb the 0.161 m step while the experiment lets 60 % of the "
                    "width bypass it laterally; this bounds how far a 2D solution can agree "
                    "with the H2 trace inside the run-up zone",
        },
    }
    if not args.no_figures:
        summary["figures"] = [str(p) for p in make_figures(per_run, ref_t, ref_h, ref)]
    (OUTDIR / "verdict.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"verdict": summary["verdict"], "envelope": summary["envelope"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())