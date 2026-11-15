"""Audit the time/height calibration of the digitised Kleefsman (2005) Fig. 19 panel.

Why.  The obstacle case is scored against a reference that exists only as a
raster JPEG inside the paper, digitised in m20884.  The two reference curves
(experiment, COMFLOW) agree with each other to 3 %, but that agreement cannot
detect a *shared* calibration error: both are traced in the same pixel frame.
So the last escape hatch for the residual arrival-time bias (~0.08 s) is "the
axis mapping is off".  This script measures the mapping from the raster itself:

* the panel frame is found as the rows/columns that are dark along the whole
  panel (the previous digitisation took x = 661..1179 and y = 6.5..413.5);
* the dashed (experiment) and solid (COMFLOW) traces stored in
  ``h2_fig19.json`` are re-mapped with the measured frame and the old one;
* the shift of the arrival time / peak between the two mappings is reported in
  seconds and in pixels, and compared with the residual the SPH run shows.

Reads only committed files (``docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.json``
plus the source JPEG if it is still on the node), writes
``results/viscosity_sweep/axis_audit.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SERIES = ROOT / "docs" / "m20884" / "outputs" / "kleefsman_h2_ref" / "h2_fig19.json"
RESULT = HERE / "results" / "viscosity_sweep" / "axis_audit.json"
JPEG_CANDIDATES = [Path("/tmp/kleef/fig19.jpg")]

OLD = {"x0": 661.0, "x1": 1179.0, "y_top": 6.5, "y_bot": 413.5,
       "t0": 0.0, "t1": 6.0, "h_top": 0.4, "h_bot": 0.0}


def find_frame(jpeg: Path):
    import cv2
    im = cv2.imread(str(jpeg), cv2.IMREAD_GRAYSCALE)
    g = (im < 140).astype(np.uint8)
    # the panel is the right half of the figure; the frame lines are the only
    # rows/columns dark over the whole panel height/width
    cols = [x for x in range(650, g.shape[1]) if g[5:414, x].sum() > 380]
    rows = [y for y in range(g.shape[0]) if g[y, 661:1178].sum() > 480]
    return {"width": int(g.shape[1]), "height": int(g.shape[0]),
            "frame_cols": cols, "frame_rows": rows}


def trace(series: dict, x0, x1, y_top, y_bot, key="dashed_experiment"):
    d = series[key]
    xs = np.asarray(d["columns"], float)
    ys = np.asarray(d["y_px"], float)
    t = OLD["t0"] + (xs - x0) * (OLD["t1"] - OLD["t0"]) / (x1 - x0)
    h = OLD["h_top"] + (ys - y_top) * (OLD["h_bot"] - OLD["h_top"]) / (y_bot - y_top)
    return t, h


def arrival(t, h, thr=0.02):
    for i in range(1, len(t)):
        if h[i] >= thr:
            if h[i] == h[i - 1]:
                return float(t[i])
            f = (thr - h[i - 1]) / (h[i] - h[i - 1])
            return float(t[i - 1] + f * (t[i] - t[i - 1]))
    return float("nan")


def main() -> int:
    series = json.loads(SERIES.read_text())
    out = {"source_series": str(SERIES.relative_to(ROOT)), "old_calibration": OLD}
    jpeg = next((p for p in JPEG_CANDIDATES if p.exists()), None)
    if jpeg is None:
        out["jpeg"] = None
        out["note"] = "source JPEG not on this node; only the stored pixel columns are audited"
        frame = {"frame_cols": [661], "frame_rows": [5]}
        new = dict(OLD)
    else:
        out["jpeg"] = str(jpeg)
        frame = find_frame(jpeg)
        out["frame"] = frame
        cols = frame["frame_cols"]
        rows = frame["frame_rows"]
        new = {"x0": float(min(cols)), "x1": float(max(cols)),
               "y_top": float(min(rows)), "y_bot": float(max(rows)),
               "t0": 0.0, "t1": 6.0, "h_top": 0.4, "h_bot": 0.0}
    out["measured_calibration"] = new
    out["frame_px_shift"] = {k: new[k] - OLD[k] for k in ("x0", "x1", "y_top", "y_bot")}

    rows = {}
    for key in ("dashed_experiment", "solid_comflow"):
        old_t, old_h = trace(series, OLD["x0"], OLD["x1"], OLD["y_top"], OLD["y_bot"], key)
        new_t, new_h = trace(series, new["x0"], new["x1"], new["y_top"], new["y_bot"], key)
        rows[key] = {
            "arrival_old_s": arrival(old_t, old_h),
            "arrival_new_s": arrival(new_t, new_h),
            "peak_old_m": float(np.nanmax(old_h)),
            "peak_new_m": float(np.nanmax(new_h)),
        }
        rows[key]["arrival_shift_s"] = (rows[key]["arrival_new_s"]
                                        - rows[key]["arrival_old_s"])
        rows[key]["peak_shift_m"] = rows[key]["peak_new_m"] - rows[key]["peak_old_m"]
    out["curves"] = rows

    # how many pixels would the residual SPH gap correspond to, and how many
    # pixels did the calibration actually move?
    sweep = sorted(HERE.glob("results/viscosity_sweep/dx*/summary.json"))
    if sweep:
        runs = [r for p in sweep for r in json.loads(p.read_text())["runs"]]
        best = min(runs, key=lambda r: r["rel_error"]["arrival_time"])
        exp_arr = min(rows[k]["arrival_old_s"] for k in rows)
        dt = best["arrival_time_s"] - exp_arr
        px_per_s = (new["x1"] - new["x0"]) / (OLD["t1"] - OLD["t0"])
        out["residual"] = {
            "best_run": best["tag"],
            "best_arrival_s": best["arrival_time_s"],
            "experiment_arrival_s": exp_arr,
            "residual_s": dt,
            "residual_px": dt * px_per_s,
            "calibration_shift_px": abs(new["x1"] - OLD["x1"]),
            "calibration_shift_s": abs(new["x1"] - OLD["x1"]) / px_per_s,
        }
        out["residual"]["residual_explained_by_calibration"] = bool(
            abs(dt) <= abs(new["x1"] - OLD["x1"]) / px_per_s)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())