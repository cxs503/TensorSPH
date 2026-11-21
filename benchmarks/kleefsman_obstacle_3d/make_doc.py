"""Figures + programmatic tables for the 3-D Kleefsman obstacle case.

Reads only committed artefacts:

    benchmarks/kleefsman_obstacle_3d/results/<tag>.json
    benchmarks/kleefsman_obstacle_3d/results/<tag>_time_series.csv
    benchmarks/kleefsman_obstacle/results/peak_horizon/*_time_series.csv   (2-D, for contrast)
    docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.csv                      (digitised experiment)

and writes ``results/report/`` (png + a machine-generated markdown fragment),
so that every number in the case document is injected, never typed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorsph.marin_probes import H2_ARRIVAL_S, H2_PEAK_M, H2_PEAK_TIME_S  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
OUT = RES / "report"
EXPERIMENT = ROOT / "benchmarks" / "marin_box_loads" / "h2_figure19_red.csv"
TWO_D = ROOT / "benchmarks" / "kleefsman_obstacle" / "results" / "peak_horizon"

REF_PEAK_H = H2_PEAK_M
REF_PEAK_T = H2_PEAK_TIME_S
REF_ARRIVAL_T = H2_ARRIVAL_S
TOL = 0.10


def read_series(path: Path, height_col: str, t_col: str = "t_s"):
    ts, hs = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            ts.append(float(row[t_col]))
            hs.append(float(row[height_col]))
    return ts, hs


def load_runs():
    runs = {}
    for path in sorted(RES.glob("dx*.json")):
        data = json.loads(path.read_text())
        if data.get("reference", {}).get("peak_height_m") != REF_PEAK_H:
            continue  # archived verdicts used a different experimental reference
        tag = data.get("tag", path.stem)
        csv_path = RES / f"{tag}_time_series.csv"
        if not csv_path.exists():
            continue
        ts, hs = read_series(csv_path, "slab_height_m")
        runs[tag] = {"verdict": data, "t": ts, "h": hs}
    return runs


def fig_timeseries(runs, exp):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if TWO_D.exists():
        for path in sorted(TWO_D.glob("*_time_series.csv")):
            tag = path.stem.replace("_time_series", "")
            try:
                ts, hs = read_series(path, "slab_x2.426")
            except KeyError:
                continue
            ax.plot(ts, hs, lw=1.0, ls=":", color="0.55",
                    label=f"2-D full-width box ({tag})")
    if exp:
        ax.plot(exp[0], exp[1], "k-", lw=1.6, label="experiment H2 (digitised)")
        ax.plot([REF_PEAK_T], [REF_PEAK_H], "k*", ms=10)
    for tag, run in runs.items():
        ax.plot(run["t"], run["h"], lw=1.4, label=f"3-D 40 % blockage ({tag})")
    ax.axhline(0.02, color="0.4", lw=0.7, ls="--")
    ax.axvline(REF_PEAK_T, color="0.4", lw=0.7, ls="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height in front of the box [m]")
    ax.set_title("Kleefsman obstacle: H2 gauge, 2-D vs 3-D vs experiment")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, max(2.7, max((r["t"][-1] for r in runs.values()), default=2.7)))
    path = OUT / "h2_timeseries_3d.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_errors(runs):
    labels, peaks, arrivals = [], [], []
    for tag, run in runs.items():
        rows = {r["metric"]: r for r in run["verdict"]["rows"]}
        labels.append(tag)
        peaks.append(rows["peak_height"]["rel_error"] * 100.0)
        got = rows["arrival_time"]["got"]
        arrivals.append(rows["arrival_time"]["rel_error"] * 100.0
                        if got is not None else float("nan"))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], peaks, width=0.4, label="peak height")
    ax.bar([i + 0.2 for i in x], arrivals, width=0.4, label="arrival time")
    ax.axhline(TOL * 100.0, color="crimson", lw=1.0, ls="--",
               label="10 % acceptance bound")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("relative error vs experiment [%]")
    ax.set_title("H2 acceptance metrics, 3-D runs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = OUT / "h2_errors_3d.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_bypass(runs):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for tag in runs:
        csv_path = RES / f"{tag}_time_series.csv"
        ts, nb = read_series(csv_path, "n_bypass")
        ax.plot(ts, nb, lw=1.2, label=f"3-D {tag}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("particles past the box, outside its span")
    ax.set_title("Spanwise bypass: the geometry the 2-D model cannot represent")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = OUT / "bypass_3d.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def markdown(runs):
    lines = []
    lines.append("| run | dx [m] | fluid | wall | peak [m] | peak t [s] | peak err | "
                 "arrival [s] | arrival err | max bypass | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, run in runs.items():
        v = run["verdict"]
        r = v["run"]
        rows = {x["metric"]: x for x in v["rows"]}
        arr = rows["arrival_time"]["got"]
        lines.append(
            f"| {tag} | {r['dx']:g} | {r['fluid_particles']} | {r['wall_particles']} | "
            f"{r['peak_height_m']:.4f} | {r['peak_time_s']:.4f} | "
            f"{rows['peak_height']['rel_error']*100:.1f} % | "
            f"{'—' if arr is None else f'{arr:.4f}'} | "
            f"{'—' if arr is None else f'{rows['arrival_time']['rel_error']*100:.1f} %'} | "
            f"{r['max_bypass_particles']:.0f} | {v['verdict']} |")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("no scored runs yet in", RES)
        return 1
    exp = None
    if EXPERIMENT.exists():
        exp = read_series(EXPERIMENT, "h_m")
    paths = [fig_timeseries(runs, exp), fig_errors(runs), fig_bypass(runs)]
    (OUT / "table_runs.md").write_text(markdown(runs) + "\n")
    for p in paths:
        print("wrote", p)
    print((OUT / "table_runs.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())