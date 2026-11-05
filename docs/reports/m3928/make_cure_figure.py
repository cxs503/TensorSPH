"""Figures + tables for the PR #113 merge-cure record (m3928).

Every number here is read programmatically from data that the merged tree
already carries -- ``benchmarks/laplace_pressure_floor/closure_ladder.json``
(the free-surface pressure closure ladder) and the optional live re-probe in
``/tmp/probe_merged.json``.  Nothing is typed by hand.

Usage:  PYTHONPATH=<worktree>/src python docs/reports/m3928/make_cure_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LADDER = ROOT / "benchmarks" / "laplace_pressure_floor" / "closure_ladder.json"
FIG = HERE / "pr113_closure_ladder.png"
PROBE = Path("/tmp/probe_merged.json")


def load_ladder() -> dict:
    return json.loads(LADDER.read_text())


def figure(ladder: dict) -> None:
    rows = ladder["ladder"]
    points = sorted({(r["dx_m"], r["sound_speed"]) for r in rows})
    by_point: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        by_point.setdefault((r["dx_m"], r["sound_speed"]), []).append(r)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- (a) Laplace jump error vs number of closure sweeps ----------------
    ax.axhspan(-15.0, 15.0, color="0.85", zorder=0, label="15 % acceptance band")
    ax.axhspan(-3.0, 3.0, color="0.70", zorder=0, label="3 % target band")
    for dx, c0 in points:
        series = sorted(by_point[(dx, c0)], key=lambda r: r["passes"])
        ax.plot([r["passes"] for r in series],
                [r["rel_error_pct"] for r in series],
                marker="o", label=f"dx={dx*1e3:.2f} mm, c0={c0:g} m/s")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("free-surface pressure closure sweeps (-)")
    ax.set_ylabel("$\\Delta p$ error vs $\\sigma/R$ [%]")
    ax.set_title("(a) Young-Laplace jump: closure ladder\n"
                 f"$\\sigma/R$ = {ladder['reference']['delta_p_Pa']:.0f} Pa reference")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # ---- (b) radial pressure profile, open vs closed -----------------------
    key = (max(points)[0], max(points)[1])
    series = sorted(by_point[key], key=lambda r: r["passes"])
    dp_ref = ladder["reference"]["delta_p_Pa"]
    for r in series:
        xs = [p["r_over_R_mid"] for p in r["pressure_profile"]]
        ys = [p["mean"] for p in r["pressure_profile"]]
        ax2.plot(xs, ys, marker=".", label=f"{r['passes']} sweep(s), "
                                            f"$\\Delta p$={r['delta_p_Pa']:.1f} Pa")
    ax2.axhline(dp_ref, color="k", ls="--", lw=1.0, label=f"analytic $\\sigma/R$ = {dp_ref:.0f} Pa")
    ax2.axhline(0.0, color="k", lw=0.6)
    ax2.set_xlabel("$r/R$ (-)")
    ax2.set_ylabel("mean pressure [Pa]")
    ax2.set_title(f"(b) Radial pressure profile, dx={key[0]*1e3:.2f} mm, "
                  f"c0={key[1]:g} m/s\n(zero pressure outside the drop = vacuum)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    print(f"wrote {FIG.relative_to(ROOT)}  ({FIG.stat().st_size / 1024:.1f} KB)")


def tables(ladder: dict) -> str:
    rows = ladder["ladder"]
    lines = ["| dx [mm] | c0 [m/s] | sweeps | $\\Delta p$ tail [Pa] | err tail [%] | "
             "err snapshot [%] | compression ratio |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["dx_m"], r["sound_speed"], r["passes"])):
        lines.append(
            f"| {r['dx_m']*1e3:.2f} | {r['sound_speed']:g} | {r['passes']} | "
            f"{r['delta_p_Pa']:.2f} | {r['rel_error_pct']:+.2f} | "
            f"{r['rel_error_final_snapshot_pct']:+.2f} | {r['compression_ratio']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    ladder = load_ladder()
    figure(ladder)
    (HERE / "closure_ladder_table.md").write_text(tables(ladder) + "\n")
    if PROBE.exists():
        print("live re-probe:", json.dumps(json.loads(PROBE.read_text())))


if __name__ == "__main__":
    main()