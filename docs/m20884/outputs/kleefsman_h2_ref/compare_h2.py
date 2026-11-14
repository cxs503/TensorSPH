"""Grade a TensorSPH Kleefsman dambreak run against the digitised H2 experimental anchor.

Case-pool metric (feature_ledger CASE-SPH-KLEEFSMAN-DAMBREAK-OBST):
peak water height in front of the obstacle and the arrival time of the surge, tolerance <10 %.
Reference: outputs/kleefsman_h2_ref/reference_h2.json (digitised from Kleefsman et al. 2005 Fig. 19).
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = json.load(open(os.path.join(HERE, "reference_h2.json")))
REF_PEAK = ANCHOR["peak_height_m"]
REF_ARR = ANCHOR["arrival_time_s"]
TOL = ANCHOR["tolerance"]["relative"]


def series(tag):
    base = os.path.join(HERE, "..", "..", "benchmarks", "kleefsman_obstacle", "results", tag)
    t, h = [], []
    with open(base + "_time_series.csv") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_s"]))
            h.append(float(row["height_m"]))
    return t, h


def grade(tag, window=None):
    t, h = series(tag)
    pts = [(ti, hi) for ti, hi in zip(t, h) if window is None or ti <= window]
    tp, hp = max(pts, key=lambda p: p[1])
    thr = ANCHOR["arrival_threshold_m"]
    arr = None
    for i in range(1, len(pts)):
        if pts[i][1] > thr:
            (t0, h0), (t1, h1) = pts[i - 1], pts[i]
            arr = t0 + (thr - h0) * (t1 - t0) / (h1 - h0) if h1 > h0 else t1
            break
    ep = abs(hp - REF_PEAK) / REF_PEAK
    ea = abs(arr - REF_ARR) / REF_ARR if arr is not None else float("nan")
    return dict(tag=tag, window_s=window, t_end=pts[-1][0],
                peak_height_m=hp, peak_time_s=tp, arrival_time_s=arr,
                peak_err_rel=ep, arrival_err_rel=ea,
                peak_pass=bool(ep < TOL), arrival_pass=bool(ea < TOL))


def main(tags):
    rows = []
    for tag in tags:
        if not os.path.exists(os.path.join(HERE, "..", "..", "benchmarks", "kleefsman_obstacle",
                                          "results", tag + "_time_series.csv")):
            print(f"[skip] {tag}: no time series")
            continue
        rows.append(grade(tag))
    print(f"{'tag':16s} {'window':>6s} {'t_end':>6s} {'peak[m]':>8s} {'err':>7s} {'arr[s]':>7s} {'err':>7s}  verdict")
    for r in rows:
        print(f"{r['tag']:16s} {str(r['window_s']):>6s} {r['t_end']:6.2f} {r['peak_height_m']:8.3f} "
              f"{r['peak_err_rel']*100:6.1f}% {r['arrival_time_s']:7.3f} {r['arrival_err_rel']*100:6.1f}%  "
              f"{'PASS' if (r['peak_pass'] and r['arrival_pass']) else 'FAIL'}")
    out = dict(reference=ANCHOR, tolerance=TOL, runs=rows)
    json.dump(out, open(os.path.join(HERE, "comparison.json"), "w"), indent=1)
    print("ref: peak %.3f m, arrival %.3f s (tol %.0f%%)" % (REF_PEAK, REF_ARR, TOL * 100))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
