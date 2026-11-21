"""Text-anchored landmarks of the Kleefsman (2005) dam-break-with-obstacle case.

Why text landmarks
------------------
The case-pool acceptance for ``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`` compares the
peak height and the arrival time of the in-front-of-box probe against the
experiment.  That comparison needs the experimental curve, which only exists as
a raster figure (digitised in m20884) and carries a probe-position ambiguity.
The *same paper* also states several **timing landmarks in words**, and those
are checkable with no digitisation at all:

* the surge first reaches the box inside the 0.40-0.56 s bracket (Fig. 18 is
  "compared with experiment at time 0.4 and 0.56 s" and the text says "The time
  instant when the water is first hitting the box is the same");
* "the water has returned from the back wall (after about 1.8 s)";
* "the return wave hits the box again (at about 4.7 s)";
* "this second wave meets the height probe at H2 again (after about 5 s)".

Every quote is stored verbatim (with page number, PDF URL and sha256) in
``docs/m21534/outputs/kleefsman2005_text/quotes.json`` by ``quotes_from_pdf.py``; this module
only *measures* our own records and binds them to those ids.  A landmark is
``PASS`` when the measured instant lies inside the tolerance the wording allows,
``FAIL`` when it does not, and ``UNKNOWN`` when the record does not cover the
window -- an uncovered window is never reported as a pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUOTES_PATH = ROOT / "docs" / "m21534" / "outputs" / "kleefsman2005_text" / "quotes.json"
THRESHOLD_M = 0.02

# ``ref_s``/``tol_s`` are read off the wording itself: the snapshot bracket
# 0.4-0.56 s becomes 0.48 +- 0.08 s; "about 1.8 s" becomes 1.8 +- 0.2 s;
# "at about 4.7 s" / "after about 5 s" become +- 0.3 s (the paper prints these
# late landmarks to one decimal place only).
LANDMARKS = [
    dict(id="BOX_FIRST_IMPACT", quote_ids=["first_hit_same", "fig18_snapshots"],
         kind="arrival", window=(0.20, 1.20), ref_s=0.48, tol_s=0.08,
         stations=[2.476, 2.426],
         statement="first arrival of the surge at the box front, inside the 0.40-0.56 s snapshot bracket"),
    dict(id="BACKWALL_RETURN_AT_H2", quote_ids=["backwall_return_1p8s"],
         kind="peak_in", window=(1.40, 2.40), ref_s=1.80, tol_s=0.20,
         stations=[2.426, 2.228],
         statement="largest water height at H2 after the surge returns from the back wall"),
    dict(id="RETURN_IMPACT_ON_BOX", quote_ids=["return_impact_box_4p7s"],
         kind="peak_in", window=(4.30, 5.10), ref_s=4.70, tol_s=0.30,
         stations=[2.476, 2.426],
         statement="second impact of the return wave on the box"),
    dict(id="SECOND_WAVE_AT_H2", quote_ids=["second_wave_h2_5s"],
         kind="peak_in", window=(4.60, 5.40), ref_s=5.00, tol_s=0.30,
         stations=[2.426, 2.228],
         statement="second meeting of the return wave with probe H2"),
]


def load_quotes(path: Path | str = QUOTES_PATH) -> dict:
    return json.loads(Path(path).read_text())


def quote_index(quotes: dict) -> dict:
    return {q["id"]: q for q in quotes["quotes"]}


def load_series(path: Path | str) -> dict:
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        names = reader.fieldnames or []
    cols: dict[str, list[float]] = {k: [] for k in names}
    for row in rows:
        for k in names:
            cols[k].append(float(row[k]))
    return cols


def pick_slab(cols: dict, x_target: float) -> tuple[float, float, str] | None:
    """Nearest ``slab_x<S>`` gauge to ``x_target`` -> (|dx|, station_x, column)."""
    best = None
    for name in cols:
        m = re.fullmatch(r"slab_x([0-9.]+)", name)
        if not m:
            continue
        x = float(m.group(1))
        dist = abs(x - x_target)
        if best is None or dist < best[0]:
            best = (dist, x, name)
    return best


def crossing(times, values, threshold: float = THRESHOLD_M, t_min: float = 0.0) -> float:
    """First upward crossing of ``threshold`` after ``t_min``, linearly interpolated."""
    for k in range(1, len(values)):
        if times[k - 1] < t_min:
            continue
        h0, h1 = values[k - 1], values[k]
        if h0 < threshold <= h1:
            frac = 0.0 if h1 == h0 else (threshold - h0) / (h1 - h0)
            return times[k - 1] + frac * (times[k] - times[k - 1])
    return float("nan")


def peak_in(times, values, lo: float, hi: float) -> tuple[float, float]:
    """(time, value) of the maximum inside ``[lo, hi]``; (nan, nan) if empty."""
    best_v, best_t, seen = float("-inf"), float("nan"), False
    for t, v in zip(times, values):
        if lo <= t <= hi and v > best_v:
            best_v, best_t, seen = v, t, True
    return (best_t, best_v) if seen else (float("nan"), float("nan"))


def measure(lm: dict, times, values) -> tuple[float, float]:
    if lm["kind"] == "arrival":
        return crossing(times, values, t_min=lm["window"][0]), THRESHOLD_M
    return peak_in(times, values, *lm["window"])


def verdict(lm: dict, t: float) -> str:
    if not math.isfinite(t):
        return "UNKNOWN"
    return "PASS" if abs(t - lm["ref_s"]) <= lm["tol_s"] else "FAIL"


def score(series_path, landmarks=None, quotes=None) -> dict:
    landmarks = LANDMARKS if landmarks is None else landmarks
    quotes = load_quotes() if quotes is None else quotes
    qi = quote_index(quotes)
    cols = load_series(series_path)
    times = cols["t_s"]
    t_end = times[-1] if times else float("nan")
    rows = []
    for lm in landmarks:
        for x in lm["stations"]:
            pick = pick_slab(cols, x)
            if pick is None:
                rows.append(dict(landmark=lm["id"], station_m=x, column=None, verdict="UNKNOWN",
                                 why="no slab gauge in this file", measured_s=float("nan"),
                                 value_m=float("nan"), ref_s=lm["ref_s"], tol_s=lm["tol_s"]))
                continue
            dist, x_got, name = pick
            t, v = measure(lm, times, cols[name])
            vd = verdict(lm, t)
            why = ("measured instant inside the wording's tolerance" if vd == "PASS"
                   else "outside the wording's tolerance" if vd == "FAIL"
                   else ("record ends at %.2f s, before the window opens at %.2f s"
                         % (t_end, lm["window"][0])))
            rows.append(dict(landmark=lm["id"], station_m=x_got, column=name,
                             proxy=bool(dist > 1e-9), station_bias_m=dist,
                             measured_s=t, value_m=v, ref_s=lm["ref_s"],
                             tol_s=lm["tol_s"], window_s=list(lm["window"]),
                             verdict=vd, why=why))
    if any(r["verdict"] == "FAIL" for r in rows):
        overall = "FAIL"
    elif any(r["verdict"] == "UNKNOWN" for r in rows):
        overall = "INCOMPLETE"
    else:
        overall = "PASS"
    return {"series": str(series_path), "t_end_s": t_end, "samples": len(times),
            "threshold_m": THRESHOLD_M, "rows": rows, "verdict": overall,
            "quotes": {qid: qi[qid] for lm in landmarks for qid in lm["quote_ids"] if qid in qi},
            "landmarks": [{k: v for k, v in lm.items() if k != "stations"} | {"stations": lm["stations"]}
                          for lm in landmarks]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = [score(s) for s in args.series]
    text = json.dumps(out, indent=2)
    if args.json:
        Path(args.json).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
