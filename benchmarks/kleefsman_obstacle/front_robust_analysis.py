"""Front-metric integrity + frozen acceptance rescoring for the Kleefsman case (m21404).

Two questions, one artefact:

1. **Is the committed ``front_x_m`` column usable?**  It is a bare ``max`` over
   the fluid x, and ``apply_boundary`` clamps a wall-crossing particle *onto*
   the wall, where it stays.  One clamp event therefore pins the front at the
   far wall for the rest of the record.  This script measures that failure on
   the committed long-horizon records and on the re-run records that carry the
   corrected column (``front_x_robust_m``, wall-crossed particles excluded).
2. **What is the case verdict on the completed ladder, at both committed
   stations?**  Frozen protocol from ``reference.json`` (slab gauge, 0.08 m
   aperture, 30 ms symmetric peak filter, 0.02 m arrival threshold), evaluated
   by :mod:`acceptance_frozen` on the committed CSVs.

Outputs (all committed paths)::

    results/front_robust/analysis.json     every number in the report
    docs/front_robust_m21404.png           figure (2 panels)
    docs/benchmarks/kleefsman-front-metric-m21404.md   report with numbers injected

Usage::

    python benchmarks/kleefsman_obstacle/front_robust_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import acceptance_frozen as af  # noqa: E402

RESULTS = HERE / "results"
NEW_DIR = RESULTS / "front_robust"
OUT_JSON = NEW_DIR / "analysis.json"
DOC = ROOT / "docs" / "benchmarks" / "kleefsman-front-metric-m21404.md"
FIG = ROOT / "docs" / "front_robust_m21404.png"

STATIONS = {"x2.426": 2.426, "x2.228": 2.228}
GRIDS = ("dx0.04", "dx0.02", "dx0.01")


def read_series(path: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open()))
    out: dict[str, list[float]] = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        out[key] = vals
    return out


def score(series: dict[str, list[float]], station: str, ref: dict) -> dict:
    t = series["t_s"]
    col = series[f"slab_{station}"]
    smooth = af.moving_average(t, col, af.FROZEN["peak_window_s"])
    peak, peak_t = af.series_peak(t, smooth)
    arrival = af.first_crossing(t, smooth, af.FROZEN["arrival_threshold_m"])
    trunk = ref["experimental_values"]
    pref = trunk["peak_height_m"]
    out = {
        "station_m": STATIONS[station],
        "peak_m": peak, "peak_time_s": peak_t, "arrival_s": arrival,
        "peak_err_rel": (peak - pref) / pref,
        "arrival_err_rel_interp": (arrival - trunk["arrival_time_s_interpolated"])
        / trunk["arrival_time_s_interpolated"],
        "arrival_err_rel_firsts": (arrival - trunk["arrival_time_s"])
        / trunk["arrival_time_s"],
        "t_end_s": t[-1],
    }
    out["pass"] = bool(abs(out["peak_err_rel"]) < 0.10
                       and abs(out["arrival_err_rel_interp"]) < 0.10
                       and abs(out["arrival_err_rel_firsts"]) < 0.10)
    windows = af.event_windows(t, [peak_t], af.EVENT_HALF_WIDTH_S) \
        if hasattr(af, "event_windows") else None
    if windows:
        out["event_window_s"] = windows[0]
    return out


def front_metrics(series: dict[str, list[float]]) -> dict:
    t = series["t_s"]
    out = {}
    for key in ("front_x_m", "front_x_robust_m"):
        if key not in series:
            continue
        f = series[key]
        out[key] = {
            "max_m": max(f),
            "at_t0_m": f[0],
            "cross_box_face_s": af.first_crossing(t, f, 2.426),
            "final_m": f[-1],
        }
    if "excluded" in series:
        out["excluded_cumulative"] = series["excluded"][-1]
        out["excluded_first_sample_above_zero"] = next(
            (t[i] for i, v in enumerate(series["excluded"]) if v > 0), None)
    return out


def main() -> int:
    ref = json.loads((HERE / "reference.json").read_text())
    data: dict[str, dict] = {"mission": "m21404", "grids": {}, "sources": {}}

    for grid in GRIDS:
        entry: dict = {}
        for tag, path in (
            ("peak_horizon", RESULTS / "peak_horizon" / f"{grid}_time_series.csv"),
            ("front_robust", NEW_DIR / f"{grid}_time_series.csv"),
        ):
            if not path.exists():
                entry[tag] = None
                continue
            series = read_series(path)
            entry[tag] = {
                "csv": str(path.relative_to(ROOT)),
                "fronts": front_metrics(series),
                "scores": {st: score(series, st, ref) for st in STATIONS},
            }
        if entry:
            data["grids"][grid] = entry

    # the contaminated record: quantify how wrong the legacy front is
    ph = data["grids"].get("dx0.01", {}).get("peak_horizon")
    if ph and "front_x_m" in ph["fronts"] and "front_x_robust_m" in ph["fronts"]:
        legacy = ph["fronts"]["front_x_m"]
        data["contamination"] = {
            "legacy_cross_box_face_s": legacy["cross_box_face_s"],
            "robust_cross_box_face_s": ph["fronts"]["front_x_robust_m"]["cross_box_face_s"],
            "legacy_final_m": legacy["final_m"],
            "robust_final_m": ph["fronts"]["front_x_robust_m"]["final_m"],
            "excluded_particles": ph["fronts"].get("excluded_cumulative"),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(json.dumps(data.get("contamination", {}), indent=1))
    for grid, entry in data["grids"].items():
        for tag, block in entry.items():
            if block:
                for st, sc in block["scores"].items():
                    print(grid, tag, st, f"peak {sc['peak_m']:.4f} "
                          f"({100*sc['peak_err_rel']:+.1f}%) arrival {sc['arrival_s']:.4f} "
                          f"({100*sc['arrival_err_rel_interp']:+.1f}%) pass={sc['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())