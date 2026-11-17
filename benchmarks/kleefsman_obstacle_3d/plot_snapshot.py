"""Field figure for the 3-D Kleefsman obstacle case (report only).

Reads the particle snapshot CSV (x, y, z, speed) and renders a side view
(x horizontal, y vertical) of the fluid particles coloured by speed, with the
obstacle box and the H2 gauge station drawn for context.  Writes a PNG.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
SNAP = HERE / "results" / "snapshot" / "particles_t0.62.csv"
OUT = HERE / "results" / "report" / "field_snapshot_t0.62.png"

# geometry (Kleefsman 2005)
TANK_L, TANK_H, TANK_W = 3.22, 1.00, 1.00
BOX_X0, BOX_L, BOX_H, BOX_W, BOX_Z0 = 2.476, 0.161, 0.161, 0.403, 0.0
GAUGE_X = 2.426
APERTURE = 0.08


def main() -> int:
    xs, ys, zs, sp = [], [], [], []
    with SNAP.open() as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["x_m"]))
            ys.append(float(row["y_m"]))
            zs.append(float(row["z_m"]))
            sp.append(float(row["speed_m_s"]))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sc = ax.scatter(xs, ys, c=sp, cmap="viridis", s=4.0, alpha=0.85,
                    edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("particle speed [m/s]")

    # obstacle box (side view: it spans z in [0, 0.403], drawn at the bottom)
    ax.add_patch(Rectangle((BOX_X0, 0.0), BOX_L, BOX_H,
                           facecolor="0.35", edgecolor="k", lw=1.0,
                           label="obstacle box (0.403 m wide)"))
    # gauge station
    ax.axvline(GAUGE_X, color="crimson", lw=1.0, ls="--")
    ax.plot([GAUGE_X], [0.02], "v", color="crimson", ms=7)
    ax.annotate("H2 gauge\nx=2.426 m", (GAUGE_X, 0.02),
                xytext=(GAUGE_X - 0.55, 0.30),
                arrowprops=dict(arrowstyle="->", color="crimson"),
                color="crimson", fontsize=8)

    ax.set_xlim(0, TANK_L)
    ax.set_ylim(0, TANK_H)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("3-D Kleefsman obstacle, t = 0.62 s (front at the box)\n"
                 "fluid particles coloured by speed; box 0.403 m wide "
                 "(40 % blockage)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
