"""Complete and grade the long-horizon grid ladder under the frozen protocol.

Why this exists
---------------
``results/audit_t30/score.json`` grades the two long runs that were on disk
(dx = 0.02, 0.04) under the frozen slab protocol.  Three gaps remain, and all
three are the same gap -- *the finest grid was never run past the reference
peak*:

* the committed ``results/dx0.01`` series stops at t = 1.0 s, while the
  reference peak sits at t = 1.714 s, so the peak criterion is undecidable on
  the finest grid that the two-grid Richardson extrapolation actually points
  at;
* that same dx = 0.01 series was integrated at dt = 4.86e-4 s, i.e. 2.5x the
  acoustic limit ``KleefsmanConfig.dt_acoustic()`` = 1.95e-4 s -- it is
  *censored*, and ``validate_case`` now rejects it outright, so it cannot be
  extended, only re-run;
* two grids cannot measure a convergence order, so ``score_t30`` has to assume
  p = 1 or 2 and report a bound instead of a rate.

This script grades the re-run ladder (dx = 0.01 and, if present, 0.005) with
the *same* operators imported from ``score_t30`` -- the frozen protocol is read
from ``reference.json`` and cross-checked against ``score_t30.FROZEN_DEFAULTS``,
so this cannot silently become a more permissive instrument.  A regression test
(``tests/test_kleefsman_long_ladder.py``) asserts that the rows it reproduces
for dx = 0.02/0.04 are bit-for-bit the frozen ones.

What it reports
---------------
1. ``rows``: peak (30 ms filtered), peak time and arrival (interp + anchored)
   for every grid x station x operator available.
2. ``order``: the *measured* observed order from the three finest grids,
   ``p = log((f_c - f_m)/(f_m - f_f)) / log r``, and the Richardson continuum
   limit it implies, instead of an assumed order.
3. ``break_even``: because the H2 station is an *undocumented convention*
   (see ``station_anchor.py``), the arrival tolerance is re-expressed as the
   smallest station offset from the box face at which the arrival would still
   pass, per grid.  That is the amount of coordinate error the verdict can
   absorb -- the number the ledger needs.

Outputs
-------
``results/long_ladder/score.json``      every number
``results/long_ladder/score_table.md``  the tables quoted by the doc
``docs/figures/kleefsman_long_ladder.png`` (with ``--figure``)

Usage::

    python benchmarks/kleefsman_obstacle/score_long_ladder.py [--figure]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from score_t30 import (  # noqa: E402  -- one instrument, imported not copied
    arrival,
    frozen_protocol,
    load_reference,
    load_series,
    peak,
    rel_error,
    richardson,
)

#: Long-run roots, in preference order; the first file that carries a station wins.
ROOTS = (HERE / "results" / "audit_t30_long", HERE / "results" / "audit_t30")
GRIDS = ("dx0.04", "dx0.02", "dx0.01", "dx0.005")
DX_OF = {"dx0.04": 0.04, "dx0.02": 0.02, "dx0.01": 0.01, "dx0.005": 0.005}
OPERATORS = ("slab", "surface", "height")


def _grid_csv(grid: str) -> Path | None:
    best: tuple[int, Path] | None = None
    for root in ROOTS:
        path = root / grid / "time_series.csv"
        if not path.is_file():
            continue
        with path.open() as fh:
            header = fh.readline()
        n = sum(1 for c in header.split(",") if c.split("_x")[0] in OPERATORS)
        if best is None or n > best[0]:
            best = (n, path)
    return best[1] if best else None


def _stations(path: Path, operator: str) -> list[float]:
    with path.open() as fh:
        header = fh.readline().strip().split(",")
    out = []
    for col in header:
        if "_x" not in col:
            continue
        op, xs = col.split("_x")
        if op == operator:
            try:
                out.append(float(xs))
            except ValueError:
                continue
    return sorted(out)


def _aperture(grid: str) -> float | None:
    for root in ROOTS:
        m = root / grid / "metrics.json"
        if m.is_file():
            d = json.loads(m.read_text())
            if "gauge_aperture_m" in d:
                return float(d["gauge_aperture_m"])
    return None


def score() -> dict:
    ref, series_path = load_reference()
    ev = ref["experimental_values"]
    proto = frozen_protocol(ref)
    tol = float(ref["tolerance"]["relative"])
    geom = ref["geometry_m"]
    box_x = float(geom["obstacle_upstream_face_x"])
    station = box_x - proto["station_offset_m"]
    thr = proto["arrival_threshold_m"]
    win = proto["peak_filter_window_s"]

    # reference curve measured with the same operators
    t_ref, h_ref = load_series(series_path, "h_m")
    ref_peak, ref_peak_t = peak(t_ref, h_ref, win)
    ref_arrival_interp = arrival(t_ref, h_ref, thr, "interp")
    ref_arrival_anchor = float(ev["arrival_time_s"])
    anchors = {"peak_m": float(ev["peak_height_m"]), "peak_time_s": float(ev["peak_time_s"]),
               "arrival_time_s": ref_arrival_anchor, "tolerance_rel": tol,
               "arrival_interp_s": ref_arrival_interp, "arrival_anchor_s": ref_arrival_anchor}

    rows = []
    for grid in GRIDS:
        path = _grid_csv(grid)
        if path is None:
            continue
        t_end = None
        for op in OPERATORS:
            for xs in _stations(path, op):
                col = f"{op}_x{xs:.3f}"
                t_got, h_got = load_series(path, col)
                if not t_got:
                    continue
                t_end = max(t_end or 0.0, t_got[-1])
                pk, pk_t = peak(t_got, h_got, win)
                arr = arrival(t_got, h_got, thr, "interp")
                arr_sample = arrival(t_got, h_got, thr, "sample")
                rows.append({
                    "grid": grid, "dx": DX_OF[grid], "operator": op,
                    "station_x_m": xs, "offset_m": box_x - xs,
                    "primary": bool(op == proto["operator"]
                                    and math.isclose(xs, station, abs_tol=1e-6)),
                    "peak_m": pk, "peak_time_s": pk_t,
                    "arrival_time_s": arr, "arrival_time_sample_s": arr_sample,
                    "peak_rel_error": rel_error(pk, anchors["peak_m"]),
                    "peak_time_rel_error": rel_error(pk_t, anchors["peak_time_s"]),
                    "arrival_rel_error": rel_error(arr, ref_arrival_interp),
                    "arrival_rel_error_anchored": rel_error(arr, ref_arrival_anchor),
                    "peak_pass": bool(rel_error(pk, anchors["peak_m"]) < tol),
                    "arrival_pass": bool(rel_error(arr, ref_arrival_interp) < tol),
                    "arrival_pass_anchored": bool(
                        rel_error(arr, ref_arrival_anchor) < tol),
                    "t_end_s": t_got[-1],
                })

    for r in rows:
        r["aperture_m"] = _aperture(r["grid"])

    primary_rows = {r["grid"]: r for r in rows if r["primary"]}
    verdict = "WITHHELD"
    if primary_rows:
        verdict = "PASS" if all(r["peak_pass"] and r["arrival_pass"]
                                and r["arrival_pass_anchored"]
                                for r in primary_rows.values()) else "FAIL"

    # --- measured order and continuum limit at the primary station --------
    def _ladder(key: str, station_x: float, operator: str) -> dict:
        got = {}
        for r in rows:
            if (r["operator"] == operator
                    and math.isclose(r["station_x_m"], station_x, abs_tol=1e-6)
                    and math.isfinite(r[key])):
                got[r["grid"]] = r
        present = [g for g in GRIDS if g in got]
        out: dict = {"stations_x_m": station_x, "operator": operator,
                     "quantity": key, "grids": present,
                     "values": {g: got[g][key] for g in present}}
        if len(present) >= 2:
            f, m = got[present[-1]], got[present[-2]]
            ratio = m["dx"] / f["dx"]
            out["finest_pair"] = [present[-1], present[-2]]
            out["ratio"] = ratio
            out["pair_difference"] = m[key] - f[key]
            for order in (1.0, 2.0):
                out[f"richardson_p{order:g}_assumed"] = richardson(
                    f[key], m[key], ratio, order)
        if len(present) >= 3:
            c, m, f = (got[present[-3]], got[present[-2]], got[present[-1]])
            d1, d2 = m[key] - c[key], f[key] - m[key]
            if d1 * d2 > 0 and abs(d2) > 0:
                p = math.log(abs(d1 / d2)) / math.log(c["dx"] / f["dx"])
                out["measured_order_p"] = p
                out["richardson_measured_p"] = richardson(
                    f[key], m[key], m["dx"] / f["dx"], p)
                out["continuum_rel_error_vs_anchor"] = rel_error(
                    out["richardson_measured_p"],
                    anchors["peak_m"] if key == "peak_m" else ref_arrival_interp)
            else:
                out["measured_order_p"] = float("nan")
                out["order_note"] = ("successive differences change sign: the "
                                     "sequence is not monotone, so no order is "
                                     "measurable and no continuum limit exists")
        return out

    ladders = []
    for op in (proto["operator"],):
        for key in ("peak_m", "arrival_time_s", "peak_time_s"):
            ladders.append(_ladder(key, station, op))
        for xs in sorted({r["station_x_m"] for r in rows
                          if r["operator"] == op and not math.isclose(
                              r["station_x_m"], station, abs_tol=1e-6)}):
            ladders.append(_ladder("arrival_time_s", xs, op))
            ladders.append(_ladder("peak_m", xs, op))

    # --- break-even station offset for the arrival tolerance --------------
    break_even = {}
    for grid in GRIDS:
        cand = sorted((r for r in rows if r["grid"] == grid
                       and r["operator"] == proto["operator"]),
                      key=lambda r: r["offset_m"])
        if not cand:
            continue
        entry: dict = {"stations_checked_m": [r["offset_m"] for r in cand],
                       "arrival_rel_error": {f"{r['offset_m']:.3f}":
                                             r["arrival_rel_error"] for r in cand},
                       "arrival_rel_error_anchored": {
                           f"{r['offset_m']:.3f}": r["arrival_rel_error_anchored"]
                           for r in cand}}
        passing = [r["offset_m"] for r in cand
                   if r["arrival_pass"] and r["arrival_pass_anchored"]]
        entry["smallest_passing_offset_m"] = min(passing) if passing else None
        entry["committed_offset_m"] = proto["station_offset_m"]
        entry["note"] = ("the arrival would pass at or beyond this offset from the "
                         "box face; the paper's raster figure cannot discriminate "
                         "0.05 m from 0.248 m, so this is the coordinate "
                         "uncertainty the verdict absorbs")
        break_even[grid] = entry

    return {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "protocol": proto,
        "tolerance_rel": tol,
        "anchors": anchors,
        "reference_curve_controls": {
            "t_span_s": [t_ref[0], t_ref[-1]], "n_samples": len(t_ref),
            "peak_at_filtered": [ref_peak, ref_peak_t],
            "arrival_interp_s": ref_arrival_interp,
            "arrival_anchor_s": ref_arrival_anchor,
            "note": ("the anchor (0.3591 s) is the sample-quantised crossing; the "
                     "interpolated crossing of the same curve is 0.3481 s, a "
                     "3 % spread that the table reports both ways"),
        },
        "rows": rows,
        "primary_rows": primary_rows,
        "verdict": verdict,
        "ladders": ladders,
        "break_even": break_even,
        "peaks_reference_time_s": anchors["peak_time_s"],
        "window_covers_reference_peak": {
            g: any(r["t_end_s"] >= anchors["peak_time_s"] - 1e-9
                   for r in rows if r["grid"] == g) for g in GRIDS
            if any(r["grid"] == g for r in rows)},
    }


def markdown(res: dict) -> str:
    out = ["| grid | dx [m] | op@x [m] | t_end [s] | peak [m] | peak t [s] "
           "| peak err | peak-t err | arrival [s] | arrival err interp/anchored | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res["rows"], key=lambda r: (-r["dx"], -r["station_x_m"], r["operator"])):
        verdict = "PASS" if (r["peak_pass"] and r["arrival_pass"]
                             and r["arrival_pass_anchored"]) else "FAIL"
        star = "**" if r["primary"] else ""
        out.append("| %s%s%s | %.3f | %s@%.3f | %.2f | %.4f | %.3f | %+.1f%% | %+.1f%% "
                   "| %.4f | %+.1f%% / %+.1f%% | %s |" % (
                       star, r["grid"], star, r["dx"], r["operator"], r["station_x_m"],
                       r["t_end_s"], r["peak_m"], r["peak_time_s"],
                       100 * (r["peak_m"] - res["anchors"]["peak_m"])
                       / res["anchors"]["peak_m"],
                       100 * (r["peak_time_s"] - res["anchors"]["peak_time_s"])
                       / res["anchors"]["peak_time_s"],
                       r["arrival_time_s"], 100 * r["arrival_rel_error"],
                       100 * r["arrival_rel_error_anchored"], verdict))
    out += ["", "| quantity @x [m] | grids | values | measured order p "
            "| continuum (measured p) | continuum err |", "|---|---|---|---|---|---|"]
    for lad in res["ladders"]:
        p = lad.get("measured_order_p", float("nan"))
        cont = lad.get("richardson_measured_p", float("nan"))
        err = lad.get("continuum_rel_error_vs_anchor", float("nan"))
        out.append("| %s@%.3f | %s | %s | %s | %s | %s |" % (
            lad["quantity"], lad["stations_x_m"], ",".join(lad["grids"]),
            "; ".join("%s=%.4f" % (g, v) for g, v in lad["values"].items()),
            "-" if not math.isfinite(p) else "%.2f" % p,
            "-" if not math.isfinite(cont) else "%.4f" % cont,
            "-" if not math.isfinite(err) else "%+.1f%%" % (100 * err)))
    out += ["", "| grid | smallest station offset that would pass the arrival |",
            "|---|---|"]
    for grid, be in res["break_even"].items():
        val = be["smallest_passing_offset_m"]
        out.append("| %s | %s |" % (grid, "none of the checked stations"
                                    if val is None else "%.3f m" % val))
    return "\n".join(out) + "\n"


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for grid, style in zip(GRIDS, ("C3", "C1", "C0", "C2")):
        be = res["break_even"].get(grid)
        if not be:
            continue
        offs = sorted(float(k) for k in be["arrival_rel_error"])
        errs = [100 * be["arrival_rel_error"][f"{o:.3f}"] for o in offs]
        ax.plot(offs, errs, style + "o-", ms=3, label=grid)
    ax.axhline(100 * res["tolerance_rel"], color="k", ls="--", lw=1,
               label="10 % tolerance")
    ax.axvline(res["protocol"]["station_offset_m"], color="0.6", ls=":", lw=1,
               label="committed 0.05 m")
    ax.set_xlabel("gauge station offset upstream of the box face [m]")
    ax.set_ylabel("first-arrival error [%]")
    ax.set_title("arrival error vs station (frozen slab gauge)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for key, style, lab in (("peak_m", "o-", "peak height"),
                            ("arrival_time_s", "s-", "arrival time")):
        pts = [(r["dx"], r[key]) for r in res["rows"] if r["primary"] and key in r]
        pts = sorted({(round(dx, 5), v) for dx, v in pts if math.isfinite(v)})
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, ms=4,
                    label=lab + " (primary station)")
    ax.axhline(res["anchors"]["peak_m"], color="C3", ls=":", lw=1,
               label="experiment peak %.3f m" % res["anchors"]["peak_m"])
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("peak height [m]")
    ax.set_xlim(0, 0.045)
    ax.set_title("primary-station peak vs grid")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print("figure ->", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    res = score()
    outdir = HERE / "results" / "long_ladder"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "score.json").write_text(json.dumps(res, indent=2))
    table = markdown(res)
    (outdir / "score_table.md").write_text(table)
    print(table)
    print("verdict:", res["verdict"])
    if args.figure:
        figure(res, ROOT / "docs" / "figures" / "kleefsman_long_ladder.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())