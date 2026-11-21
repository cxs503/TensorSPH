"""Compare solver CSVs with the official SPHERIC Test 2 experimental workbook."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = "https://www.spheric-sph.org/tests/test-02"
XLS_SHA = "8126f8c8f6effb9e8bfb245a2d098186e9ebee2ef6e75fcb1b3166af17883d8c"
TAPS = [f"P{i}_Pa" for i in range(1, 9)]
PEAK_START, PEAK_END = 0.35, 0.70
THRESHOLD_PA = 500.0


def experiment(path):
    import xlrd  # Only the comparison requires this reader for the original .xls.

    if hashlib.sha256(path.read_bytes()).hexdigest() != XLS_SHA:
        raise ValueError("expected the original test_case_2_exp_data.xls")
    sheet = xlrd.open_workbook(str(path)).sheet_by_name("Experimental_data")
    names = {"t_s": "Time (s)", "H2_local_m": "H2 (m)"}
    names.update({tap: tap.replace("_Pa", " (Pa)") for tap in TAPS})
    header = sheet.row_values(0)
    return {key: np.array(sheet.col_values(header.index(label), 1)) for key, label in names.items()}


def read_series(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as source:
        table = np.genfromtxt(source, names=True, delimiter=",")
    data = {name: table[name] for name in table.dtype.names}
    if (
        len(table) < 2
        or not all(np.isfinite(v).all() for v in data.values())
        or np.any(np.diff(data["t_s"]) <= 0)
    ):
        raise ValueError(f"incomplete or invalid trace: {path}")
    return data


def clip(t, y, start, end):
    if start < t[0] or end > t[-1]:
        raise ValueError("trace does not span the requested event")
    between = (t > start) & (t < end)
    return np.r_[start, t[between], end], np.r_[
        np.interp(start, t, y), y[between], np.interp(end, t, y)
    ]


def crossing(t, y, level):
    idx = np.flatnonzero((y[:-1] < level) & (y[1:] >= level))
    if not len(idx):
        return None
    k = idx[0]
    return float(t[k] + (level - y[k]) * (t[k + 1] - t[k]) / (y[k + 1] - y[k]))


def tap_event(t, y):
    baseline = float(y[t <= 0.20].mean())
    t, y = clip(t, y - baseline, PEAK_START, PEAK_END)
    i = int(np.argmax(y))
    return dict(
        peak_Pa=float(y[i]),
        peak_t_s=float(t[i]),
        impulse_Pa_s=float(np.dot(0.5 * (y[:-1] + y[1:]), np.diff(t))),
        arrival_s=crossing(t, y, THRESHOLD_PA),
        baseline_Pa=baseline,
    )


def rel(value, ref):
    return 100 * (value / ref - 1.0)


def compare(data, exp):
    t, te = data["t_s"], exp["t_s"]
    height, observed = data["H2_local_m"], exp["H2_local_m"]
    end = min(t[-1], te[-1])
    shared = (te >= t[0]) & (te <= end)
    out = {
        "end_s": float(t[-1]),
        "samples": len(t),
        "H2": {
            "observed_arrival_s": crossing(te, observed, 0.02),
            "local_arrival_s": crossing(t, height, 0.02),
            "blocked_width_arrival_s": crossing(t, data["slab_height_m"], 0.02),
            "common_window_RMSE_m": float(
                np.mean((np.interp(te[shared], t, height) - observed[shared]) ** 2) ** 0.5
            ),
        },
        "taps": {},
    }
    if end >= 2.6:
        for prefix, ts, h in (
            ("measured", te, observed),
            ("local", t, height),
            ("blocked_width", t, data["slab_height_m"]),
        ):
            ts, h = clip(ts, h, 0.35, 2.6)
            i = int(np.argmax(h))
            out["H2"][prefix + "_peak_m"] = float(h[i])
            out["H2"][prefix + "_peak_t_s"] = float(ts[i])
    for tap in TAPS:
        ref, got = tap_event(te, exp[tap]), tap_event(t, data[tap])
        row = {"measured": ref, "computed": got}
        if ref["arrival_s"] is not None:
            row["peak_error_pct"] = rel(got["peak_Pa"], ref["peak_Pa"])
            row["impulse_error_pct"] = rel(got["impulse_Pa_s"], ref["impulse_Pa_s"])
            row["arrival_error_s"] = (
                got["arrival_s"] - ref["arrival_s"] if got["arrival_s"] is not None else None
            )
        row["support_at_peak"] = int(
            data[tap.replace("_Pa", "_support")][np.abs(t - got["peak_t_s"]).argmin()]
        )
        out["taps"][tap] = row
    out["body_impulse_Ns"] = {}
    for axis in "xyz":
        parts = {}
        for k in ("sph", "clamp"):
            j = data[f"J_{k}_{axis}_Ns"]
            parts[k] = float(np.interp(PEAK_END, t, j) - np.interp(PEAK_START, t, j))
        out["body_impulse_Ns"][axis] = {**parts, "total": sum(parts.values())}
    return out


def plot(curves, exp, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(9, 7.3), layout="constrained")
    for ax, field in zip(axes.flat, ["H2_local_m", "P1_Pa", "P2_Pa", "P3_Pa", "P4_Pa", "P5_Pa"]):
        height = field == "H2_local_m"
        left, right = (0.29, 2.60) if height else (PEAK_START, PEAK_END)
        for label, record in [("MARIN experiment", exp), *curves]:
            t, y = record["t_s"], record[field]
            mask = (t >= left) & (t <= right)
            values = y if height else (y - y[t <= 0.2].mean()) / 1000.0
            ax.plot(
                t[mask], values[mask], lw=1.2, label=label, color="black" if record is exp else None
            )
        ax.set(
            title="H2, local water height" if height else field.split("_")[0],
            ylabel="Water height (m)" if height else "Gauge pressure (kPa)",
            xlim=(left, right),
            xlabel="Time (s)",
        )
    axes.flat[0].legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xls", type=Path, required=True)
    parser.add_argument("--series", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    exp = experiment(args.xls)
    curves = [(p.name.split("_time_series")[0], read_series(p)) for p in args.series]
    results = dict(
        source=SOURCE,
        workbook_sha256=XLS_SHA,
        experiment_rows=len(exp["t_s"]),
        first_impact_window_s=[PEAK_START, PEAK_END],
        threshold_Pa=THRESHOLD_PA,
        runs={label: compare(data, exp) for label, data in curves},
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    plot(curves, exp, args.out / "pressure_height.png")
