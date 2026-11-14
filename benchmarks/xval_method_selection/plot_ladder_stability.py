"""Figures for the method-A resolution ladder (XVAL-SPH-METHOD-SELECTION-2026-11-12).

Reads the sweep summaries written by ``ladder_stability_sweep.py`` and draws the
only two things the data support:

* **when** each resolution dies (blow-up onset time) or that it survived the
  0.60 s horizon, against the H/Δx literature threshold that the repository's own
  method research records (``docs/wave_flume_method_research.md``: DualSPHysics'
  empirical ``H/dx ≳ 10`` for wave-height accuracy);
* **how violent** the surviving runs are (max|v|/c₀ against the < 1 gate and peak
  density compression against the < 10 % gate).

No wave-height / accuracy panel is drawn: the horizon stops before the first
wave reaches the gauges, so an amplitude comparison would be pure fabrication.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/plot_ladder_stability.py \
        --summary runs/xval/ladder_stability.json runs/xval/ladder_stability_rest.json \
        --fig-dir docs/figures/xval_method_selection_m21050
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# docs/wave_flume_method_research.md § literature: DualSPHysics' empirical rule of
# thumb for wave-height accuracy.  It is a *cited* threshold, not a fitted value.
LITERATURE_H_OVER_DX = 10.0

COLORS = {"blew_up": "#c0392b", "survived_stable": "#1e8449",
          "survived_unstable": "#d68910", "rejected": "#7f8c8d",
          "no_record": "#000000"}


def load_rows(paths) -> list:
    rows: list = []
    seen = set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        for row in data["rows"]:
            if row["label"] in seen:
                continue
            seen.add(row["label"])
            row["horizon_s"] = data["case"]["horizon_s"][0] if len(
                data["case"]["horizon_s"]) == 1 else row.get("horizon_s")
            rows.append(row)
    return rows


def fig_ladder(rows, path: Path) -> bool:
    """Onset time vs resolution, with the survived points pinned to the horizon."""
    pts = [r for r in rows if r["group"] == "ladder"]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in pts:
        colour = COLORS.get(r["verdict"], "#000000")
        x = r["h_over_dx"]
        if r["verdict"] == "blew_up" and r["onset_time_s"] is not None:
            ax.plot([x], [r["onset_time_s"]], "o", ms=9, color=colour)
            ax.annotate(f"blow-up, step {r['onset_step']}",
                        (x, r["onset_time_s"]), textcoords="offset points",
                        xytext=(8, -12), fontsize=8, color=colour)
        elif r["verdict"] in ("survived_stable", "survived_unstable"):
            ax.plot([x], [r["horizon_s"]], "s", ms=9, color=colour,
                    markerfacecolor="none", mew=2)
            ax.annotate(r["verdict"].replace("survived_", "survived, "),
                        (x, r["horizon_s"]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=colour)
        else:
            ax.plot([x], [r["horizon_s"] * 0.02], "x", ms=9, color=colour)
            ax.annotate(r["verdict"], (x, r["horizon_s"] * 0.02),
                        textcoords="offset points", xytext=(6, 4), fontsize=8,
                        color=colour)
    ax.axhline(pts[0]["horizon_s"], ls=":", color="#555555", lw=1)
    ax.text(min(r["h_over_dx"] for r in pts), pts[0]["horizon_s"] * 1.04,
            "sweep horizon (no wave reaches the gauges)", fontsize=8, color="#555555")
    ax.axvspan(LITERATURE_H_OVER_DX, max(r["h_over_dx"] for r in pts) * 1.35,
               color="#1e8449", alpha=0.08)
    ax.text(LITERATURE_H_OVER_DX * 1.02, pts[0]["horizon_s"] * 0.35,
            "H/Δx ≳ 10\n(DualSPHysics rule of thumb,\ndocs/wave_flume_method_research.md)",
            fontsize=8, color="#1e8449")
    ax.axvline(3.20, ls="--", lw=1, color="#2c3e50")
    ax.text(3.24, pts[0]["horizon_s"] * 0.02,
            "recorded method A\n(cell 0, H/Δx = 3.20)", fontsize=8, color="#2c3e50")
    ax.set_xlabel("H/Δx  (wave height / particle spacing)")
    ax.set_ylabel("time to blow-up  [s]")
    ax.set_title("Method A stability ladder, h = 0.25 m, H/h = 0.16, c₀ = 30 m/s")
    ax.set_ylim(0, pts[0]["horizon_s"] * 1.25)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_violence(rows, path: Path) -> bool:
    """Peak speed ratio and peak compression of the runs that survived."""
    pts = [r for r in rows if r.get("speed_ratio") is not None
           and r.get("max_compression") is not None]
    if not pts:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, key, gate, label in (
            (ax1, "speed_ratio", 1.0, "max |v| / c₀  (gate < 1)"),
            (ax2, "max_compression", 0.10, "max density compression (gate < 10 %)")):
        for r in pts:
            colour = COLORS.get(r["verdict"], "#000000")
            marker = "s" if r["group"] == "ladder" else "^"
            ax.plot([r["h_over_dx"]], [r[key]], marker, ms=8, color=colour)
        ax.axhline(gate, ls="--", color="#c0392b", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("H/Δx")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both")
    ax1.set_title("surviving cells: speed ratio")
    ax2.set_title("surviving cells: compression")
    handles = [plt.Line2D([], [], marker="s", ls="", color=COLORS["survived_stable"],
                          label="ladder point"),
               plt.Line2D([], [], marker="^", ls="", color=COLORS["survived_stable"],
                          label="mechanism probe (d20)")]
    ax2.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--fig-dir", default="docs/figures/xval_method_selection_m21050")
    args = parser.parse_args(argv)

    rows = load_rows(args.summary)
    fig_dir = REPO / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in (("ladder_stability.png", fig_ladder),
                     ("ladder_violence.png", fig_violence)):
        if fn(rows, fig_dir / name):
            made.append(str(fig_dir / name))
    for path in made:
        print(f"wrote {path}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())