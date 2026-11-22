"""Compile flume records and plot their measured gauge histories."""

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regular_wave_campaign import assess  # noqa: E402
from tensorsph.wave_flume import paddle_stroke_for_height  # noqa: E402
from tensorsph.wave_flume_grade import harmonic_amplitude, solver_record_integrity  # noqa: E402


def measure(paths, out):
    report = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            record = json.load(f)
        key = path.name.split(".")[0]
        cfg, diag = record["config"], record["diagnostics"]
        still = cfg["settle_time"] >= record["requested_duration_s"]
        entry = {
            "paddle_held": still,
            "reference_density_increment": cfg["reference_density_increment"],
            "xsph": cfg["xsph"],
            "physical_time_s": diag["simulated_time"],
            "complete": record["complete"],
            "max_speed_m_s": diag["max_speed"],
            "min_density_ratio": diag["min_density_ratio"],
            "regime": record["regime"],
            "first_exit": diag["first_regime_exit"],
            "escapes": diag["escaped"],
            "record_integrity": solver_record_integrity(record),
        }
        if not still:
            entry["wave"] = assess(record)
            gauges = entry["wave"].get("gauges")
            if gauges and gauges[0]["height_m"] > 0:
                spec = record["spec"]
                start, end = entry["wave"]["window_s"]
                stroke = harmonic_amplitude(
                    record["times"], record["paddle_offset"], spec["omega"], start, end
                )["height"]
                entry["paddle_stroke_m"] = stroke
                entry["target_paddle_stroke_m"] = spec["stroke"]
                entry["linear_height_from_measured_stroke_m"] = (
                    paddle_stroke_for_height(stroke, spec["k"], spec["depth"])
                    if stroke > 0
                    else 0.0
                )
                entry["first_height_over_target"] = gauges[0]["height_m"] / record["spec"]["height"]
                entry["last_height_over_first"] = gauges[-1]["height_m"] / gauges[0]["height_m"]
        report[key] = entry
        cells = int(cfg["depth_cells"])
        if not still and cells in (16, 20):
            row = 0 if cells == 16 else 1
            for panel, gauge in zip(axes[row], (0, -1)):
                t = np.asarray(record["times"])
                eta = np.asarray(record["probe_eta"][gauge]) * 1000
                label = "corrected" if cfg["reference_density_increment"] else "original"
                panel.plot(t, eta, lw=0.9, label=f"{label}, smooth={cfg['xsph']:g}")
                panel.set(
                    title=f"D{cells}, station {1 if gauge == 0 else 4}",
                    xlabel="Time (s)",
                    ylabel="Elevation (mm)",
                    xlim=(0, 5),
                )
                panel.axhline(0, lw=0.3, color="grey")
    for panel in axes.flat:
        if panel.get_legend_handles_labels()[1]:
            panel.legend(fontsize=6)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "wave_gauges.png", dpi=145)
    plt.close(fig)
    (out / "assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    measure(args.records, args.out)
