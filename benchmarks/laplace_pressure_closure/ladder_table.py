"""Collapse the relaxation ladder into a (dx, c0) sensitivity table.

``relax_ladder.py`` writes one evidence file per ``(dx, c0)`` point
(``raw/relax_ladder.json`` for the baseline, ``raw/relax_ladder_dx*mm_c*.json``
otherwise).  This script reads all of them and answers the question the single
point could not:

  * does the closure ranking survive a change of resolution, or is the
    +8.8 % of ``donors = interior`` a one-mesh accident?
  * does the residual follow ``c0``?  If it did, it would be an EOS noise
    floor; if it does not, it is a closure/geometry effect.

It also draws ``laplace_closure_ladder.png`` (used by the verification record)
and writes ``raw/ladder_table.json``, so no number in the document is typed by
hand.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/ladder_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = RAW / "ladder_table.json"
FIG = HERE / "laplace_closure_ladder.png"

GATE_PCT = 15.0
LABEL = {
    "closure_off": "closure off (Tait clamp)",
    "closure_on_all_donors": "on, donors = all",
    "closure_on_interior_layers2": "on, donor = interior, 2 layers",
    "closure_on_interior_layers4": "on, donor = interior, 4 layers",
}
ORDER = list(LABEL)


def load_points() -> list[dict]:
    """Every ladder file, as a point {dx_m, sound_speed, cases} with integrity checks."""
    points = []
    for path in sorted(RAW.glob("relax_ladder*.json")):
        rec = json.loads(path.read_text())
        dx = rec.get("dx_m")
        c0 = rec.get("sound_speed")
        if dx is None or c0 is None:
            continue
        # A case's provenance is the file header.  If a header ever disagrees with
        # the cases it carries -- the failure mode that made a 1.5 mm run report
        # 2 mm numbers -- the aggregation refuses the point instead of averaging it in.
        for case in rec.get("cases", []):
            case.setdefault("source_file", path.name)
        points.append({"dx_m": dx, "sound_speed": c0, "source_file": path.name,
                       "max_steps": rec.get("max_steps"),
                       "analytic_dp_Pa": rec.get("analytic_dp_Pa"),
                       "cases": rec.get("cases", [])})
    points.sort(key=lambda p: (p["sound_speed"], -p["dx_m"]))
    return points


def by_case(points: list[dict]) -> dict[str, list[dict]]:
    table: dict[str, list[dict]] = {}
    for point in points:
        for case in point["cases"]:
            table.setdefault(case["case"], []).append(
                {"dx_m": point["dx_m"], "sound_speed": point["sound_speed"],
                 "tail_rel_error_pct": case["tail_rel_error_pct"],
                 "tail_mean_Pa": case["tail_mean_Pa"],
                 "final_abs_rel_error_pct": case["final_abs_rel_error_pct"],
                 "vmax_over_vref": case["vmax_over_vref"],
                 "within_15pct": case.get("tail_within_15pct",
                                          abs(case["tail_rel_error_pct"]) < GATE_PCT),
                 "n_particles": case.get("n_particles")})
    for rows in table.values():
        rows.sort(key=lambda r: (r["sound_speed"], -r["dx_m"]))
    return table


def verdicts(table: dict[str, list[dict]]) -> dict:
    """The statements the table licenses, each as a machine-checkable record."""
    resolution_spread = {}
    for name, rows in table.items():
        at_c0 = [r for r in rows if abs(r["sound_speed"] - 20.0) < 1e-9]
        if len(at_c0) >= 2:
            errs = [r["tail_rel_error_pct"] for r in at_c0]
            resolution_spread[name] = {
                "error_pct_by_dx": {f"{r['dx_m']:g}": r["tail_rel_error_pct"] for r in at_c0},
                "spread_pct_points": max(errs) - min(errs),
            }
    sound_speed_spread = {}
    for name, rows in table.items():
        at_dx = {}
        for r in rows:
            if f"{r['dx_m']:g}" not in at_dx:
                at_dx[f"{r['dx_m']:g}"] = []
            at_dx[f"{r['dx_m']:g}"].append(r["tail_rel_error_pct"])
        for dx, errs in at_dx.items():
            if len(errs) >= 2:
                sound_speed_spread[f"{name}@{dx}"] = {
                    "error_pct_by_c0": {f"{r['sound_speed']:g}": r["tail_rel_error_pct"]
                                        for r in rows if f"{r['dx_m']:g}" == dx},
                    "spread_pct_points": max(errs) - min(errs),
                }
    inside = {name: all(r["within_15pct"] for r in rows) for name, rows in table.items()}
    return {"gate_pct": GATE_PCT,
            "resolution_spread": resolution_spread,
            "sound_speed_spread": sound_speed_spread,
            "variant_inside_gate_everywhere": inside}


def figure(table: dict[str, list[dict]], verdict: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    colours = {"closure_off": "C0", "closure_on_all_donors": "C1",
               "closure_on_interior_layers2": "C2", "closure_on_interior_layers4": "C3"}

    # (a) error vs dx at fixed c0, one line per variant, +/-15 % band
    ax = axes[0]
    for name, rows in table.items():
        at_c0 = [r for r in rows if abs(r["sound_speed"] - 20.0) < 1e-9]
        if not at_c0:
            continue
        ax.plot([r["dx_m"] * 1e3 for r in at_c0],
                [r["tail_rel_error_pct"] for r in at_c0],
                "o-", ms=5, lw=1.4, color=colours.get(name, None),
                label=LABEL.get(name, name))
    ax.axhspan(-GATE_PCT, GATE_PCT, color="g", alpha=0.12,
               label=rf"$\pm{GATE_PCT:.0f}\%$ gate")
    ax.axhline(0, color="k", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\Delta x$ [mm]")
    ax.set_ylabel(r"tail $|\langle p_{\rm core}\rangle-\sigma/R|/\sigma/R$  [%]")
    ax.set_title(r"(a) resolution ladder at $c_0=20$ m/s")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    # (b) error vs c0 at fixed dx -- separates a closure effect from an EOS noise floor
    ax = axes[1]
    for name, rows in table.items():
        at_dx = {}
        for r in rows:
            at_dx.setdefault(r["dx_m"], []).append(r)
        for dx, rs in sorted(at_dx.items(), reverse=True):
            if len(rs) < 2:
                continue
            rs = sorted(rs, key=lambda r: r["sound_speed"])
            ax.plot([r["sound_speed"] for r in rs],
                    [r["tail_rel_error_pct"] for r in rs],
                    "s-" if "interior" in name else "o--", ms=5, lw=1.3,
                    color=colours.get(name, None),
                    label=f"{LABEL.get(name, name)}  ($\\Delta x$={dx*1e3:.1f} mm)")
    ax.axhspan(-GATE_PCT, GATE_PCT, color="g", alpha=0.12)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$c_0$ [m/s]")
    ax.set_ylabel("tail error [%]")
    ax.set_title(r"(b) sound-speed ladder")
    ax.legend(fontsize=6.0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)
    return FIG


def main() -> None:
    points = load_points()
    table = by_case(points)
    verdict = verdicts(table)
    OUT.write_text(json.dumps({"gate_pct": GATE_PCT, "points": points,
                               "by_variant": table, "verdicts": verdict},
                              indent=2))
    for name in ORDER:
        for r in table.get(name, []):
            print(f"{name:30s} dx={r['dx_m']*1e3:.1f}mm c0={r['sound_speed']:4.0f} "
                  f"tail {r['tail_rel_error_pct']:+7.1f} %  "
                  f"{'inside' if r['within_15pct'] else 'OUTSIDE'} gate")
    for key, val in verdict["resolution_spread"].items():
        print(f"resolution spread {key:30s} {val['spread_pct_points']:5.1f} pt")
    path = figure(table, verdict)
    print("wrote", OUT, "and", path.name)


if __name__ == "__main__":
    main()
