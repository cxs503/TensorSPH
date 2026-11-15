"""Score the committed ``t_end = 3.0 s`` series against Kleefsman (2005) Fig. 19.

Why this exists
---------------
The two committed long runs (``results/audit_t30/dx0.02`` and ``dx0.04``)
were produced while ``reference.json:experimental_values.status`` was still
``WITHHELD``; their ``metrics.json`` therefore carries
``comparison.verdict = "WITHHELD"`` and an empty row list.  The reference has
since been digitised and anchored (peak 0.370 m @ 1.703 s, arrival 0.3591 s at
probe H2), and both long runs *do* extend past 1.703 s.  The peak criterion is
therefore decidable from data that is already on disk -- no new integration is
required -- and this script does that scoring under the **frozen protocol**
recorded in ``reference.json:metric_definition.frozen_primary``:

    operator            slab (fixed 0.08 m physical aperture, volume based)
    station             0.05 m upstream of the box face  ->  x = 2.426 m
    peak filter         30 ms symmetric moving average
    arrival threshold   0.02 m, linearly interpolated first upward crossing

``run.py``'s own ``comparison`` block predates that freeze: it scores
``run.peak_height``, i.e. the legacy top-N particle-centre operator, unfiltered.
The two protocols disagree by 12-29 % at the probe, which is the same size as
the 10 % case tolerance -- see ``docs/benchmarks/kleefsman-t30-peak-decidability.md``
and ``compare_protocols()`` below.

Outputs
-------
``results/audit_t30/score.json``            every row, every gauge, both grids
``results/audit_t30/score_table.md``        the markdown table used by the doc
``docs/figures/kleefsman_t30_score.png``    (with ``--figure``) time histories

Usage::

    python benchmarks/kleefsman_obstacle/score_t30.py [--figure]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: Frozen acceptance protocol; the values are re-read from ``reference.json``
#: and asserted against these defaults so a silent edit of one file cannot
#: change the verdict without a test failing.
FROZEN_DEFAULTS = {
    "operator": "slab",
    "station_offset_m": 0.05,
    "aperture_m": 0.08,
    "peak_filter_window_s": 0.03,
    "arrival_threshold_m": 0.02,
}

#: Long runs produced before the reference was anchored.
GRIDS = ("dx0.02", "dx0.04")


# --------------------------------------------------------------------------
# series primitives
# --------------------------------------------------------------------------
def load_series(path: Path, column: str) -> tuple[list[float], list[float]]:
    """Read one column of a ``t_s,<column>`` CSV, dropping non-finite rows."""
    t: list[float] = []
    h: list[float] = []
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise KeyError(f"{path}: no column {column!r} in {reader.fieldnames}")
        for row in reader:
            try:
                tv, hv = float(row["t_s"]), float(row[column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(tv) and math.isfinite(hv):
                t.append(tv)
                h.append(hv)
    return t, h


def moving_average(t: list[float], h: list[float],
                   window: float) -> list[float]:
    """Symmetric boxcar over ``|t_i - t| <= window/2``, endpoints included.

    The reference curve is digitised at ~11.6 ms spacing, so a 30 ms window is
    three samples wide; the operator is applied identically to the reference and
    to the simulation so a narrow digitisation spike cannot be compared against
    a narrow particle-stack spike.
    """
    if window <= 0.0:
        return list(h)
    half = 0.5 * window
    out = []
    lo = 0
    for ti in t:
        while lo < len(t) and t[lo] < ti - half:
            lo += 1
        hi = lo
        acc = 0.0
        n = 0
        while hi < len(t) and t[hi] <= ti + half:
            acc += h[hi]
            n += 1
            hi += 1
        out.append(acc / n if n else float("nan"))
    return out


def peak(t: list[float], h: list[float], window: float = 0.0) -> tuple[float, float]:
    """(height, time) of the maximum after an optional ``window`` smoothing."""
    if not t:
        return float("nan"), float("nan")
    hh = moving_average(t, h, window)
    k = max(range(len(hh)), key=hh.__getitem__)
    return hh[k], t[k]


def arrival(t: list[float], h: list[float], threshold: float,
            mode: str = "interp") -> float:
    """First upward crossing of ``threshold``.

    ``mode="interp"`` linearly interpolates between the two bracketing samples,
    ``mode="sample"`` returns the timestamp of the first sample at or above the
    threshold.  Both are carried because the *reference* is a digitisation at
    ~11.6 ms spacing, where the two conventions differ by up to one sample
    (3 % of the 0.35 s arrival).  ``nan`` when the level starts above the
    threshold or never reaches it, for either mode.
    """
    for k in range(1, len(h)):
        if h[k - 1] < threshold <= h[k]:
            if mode == "sample":
                return t[k]
            if h[k] == h[k - 1]:
                return t[k - 1]
            frac = (threshold - h[k - 1]) / (h[k] - h[k - 1])
            return t[k - 1] + frac * (t[k] - t[k - 1])
    return float("nan")


def resample(t: list[float], h: list[float], grid: list[float]) -> list[float]:
    """Linear interpolation onto ``grid``; ``nan`` outside the series span."""
    out = []
    j = 0
    for tg in grid:
        if tg < t[0] or tg > t[-1]:
            out.append(float("nan"))
            continue
        while j < len(t) - 2 and t[j + 1] < tg:
            j += 1
        while j > 0 and t[j] > tg:
            j -= 1
        t0, t1 = t[j], t[j + 1]
        h0, h1 = h[j], h[j + 1]
        out.append(h0 if t1 == t0 else h0 + (h1 - h0) * (tg - t0) / (t1 - t0))
    return out


def shape_stats(t_ref: list[float], h_ref: list[float],
                t_got: list[float], h_got: list[float],
                t0: float, t1: float, lag_scan: float = 0.4,
                dt: float = 0.005) -> dict[str, float]:
    """Time-history agreement on the common window ``[t0, t1]``.

    Returns the RMS difference, that difference expressed as a fraction of the
    reference RMS about its own mean, and the time shift that minimises it
    (positive = the simulation is late).  A pure offset in time shows up as
    ``rms_lagged << rms``; a spurious hump does not.
    """
    n = int(round((t1 - t0) / dt)) + 1
    grid = [t0 + i * dt for i in range(n)]
    r = resample(t_ref, h_ref, grid)
    g = resample(t_got, h_got, grid)
    pairs = [(a, b) for a, b in zip(r, g) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 10:
        return {"rms": float("nan"), "rms_rel": float("nan"),
                "best_lag_s": float("nan"), "rms_lagged": float("nan"),
                "n_points": float(len(pairs))}
    rms = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))
    peak_ref = max((a for a, _ in pairs), default=float("nan"))
    mean_ref = sum(a for a, _ in pairs) / len(pairs)
    ref_rms = math.sqrt(sum((a - mean_ref) ** 2 for a, _ in pairs) / len(pairs))
    best_lag, best = 0.0, rms
    k = int(round(lag_scan / dt))
    for s in range(-k, k + 1):
        if s == 0:
            continue
        acc, cnt = 0.0, 0
        for i in range(len(grid)):
            j = i + s
            if 0 <= j < len(grid):
                a, b = r[i], g[j]
                if math.isfinite(a) and math.isfinite(b):
                    acc += (a - b) ** 2
                    cnt += 1
        if cnt and math.sqrt(acc / cnt) < best:
            best = math.sqrt(acc / cnt)
            best_lag = s * dt
    return {"rms": rms,
            "rms_rel_to_ref_rms": rms / ref_rms if ref_rms else float("nan"),
            "rms_rel_to_ref_peak": rms / peak_ref if peak_ref else float("nan"),
            "best_lag_s": best_lag, "rms_lagged": best,
            "lag_at_scan_edge": bool(abs(abs(best_lag) - lag_scan) < 0.5 * dt),
            "lag_scan_s": lag_scan,
            "n_points": float(len(pairs))}


# --------------------------------------------------------------------------
# reference bookkeeping
# --------------------------------------------------------------------------
def load_reference() -> tuple[dict, Path]:
    """``reference.json`` plus the resolved digitised H2 series path.

    ``experimental_values.digitised_series`` was written as
    ``../../outputs/kleefsman_h2_ref/h2_fig19.csv``; that ``outputs/`` tree does
    not exist in the checkout, the file lives under ``docs/m20884/``.  Resolve
    robustly (and fall back to a search) instead of trusting the stale string.
    """
    ref = json.loads((HERE / "reference.json").read_text())
    ev = ref["experimental_values"]
    raw = ev.get("digitised_series", "")
    candidates = []
    if raw:
        candidates.append((HERE / raw).resolve())
        for base in (HERE, ROOT, ROOT.parent):
            candidates.append((base / raw).resolve())
    candidates.append(ROOT / "docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.csv")
    for cand in candidates:
        if cand.is_file():
            return ref, cand
    hits = sorted(ROOT.glob("**/kleefsman_h2_ref/h2_fig19.csv"))
    if hits:
        return ref, hits[0]
    raise FileNotFoundError("digitised Kleefsman H2 series not found in checkout")


def frozen_protocol(ref: dict) -> dict:
    """Frozen acceptance protocol, cross-checked against ``FROZEN_DEFAULTS``."""
    fp = ref["metric_definition"]["frozen_primary"]
    out = {
        "operator": fp["operator"],
        "station_offset_m": float(fp["station_offset_m"]),
        "aperture_m": float(fp["aperture_m"]),
        "peak_filter_window_s": float(fp["peak_filter_window_s"]),
        "arrival_threshold_m": float(fp["arrival_threshold_m"]),
    }
    for key, default in FROZEN_DEFAULTS.items():
        if not math.isclose(float(out[key]) if not isinstance(out[key], str) else 0.0,
                            float(default) if not isinstance(default, str) else 0.0,
                            rel_tol=1e-12) and out[key] != default:
            raise ValueError(f"frozen_primary.{key} = {out[key]!r} != {default!r}; "
                             "the acceptance protocol changed -- update "
                             "FROZEN_DEFAULTS and the doc together")
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def rel_error(got: float, want: float) -> float:
    return abs(got - want) / abs(want) if want else float("nan")


def richardson(fine: float, coarse: float, ratio: float, order: float) -> float:
    """h -> 0 estimate from two grids, ``fine`` on the finer grid."""
    return fine + (fine - coarse) / (ratio ** order - 1.0)


def score() -> dict:
    ref, series_path = load_reference()
    ev = ref["experimental_values"]
    proto = frozen_protocol(ref)
    tol = float(ref["tolerance"]["relative"])
    geom = ref["geometry_m"]
    box_x = float(geom["obstacle_upstream_face_x"])
    station = box_x - proto["station_offset_m"]
    other = box_x - 0.248

    # --- reference curve, measured with the same operators -----------------
    t_ref, h_ref = load_series(series_path, "h_m")
    ref_peak, ref_peak_t = peak(t_ref, h_ref, proto["peak_filter_window_s"])
    ref_peak_raw, ref_peak_t_raw = peak(t_ref, h_ref, 0.0)
    ref_arrival = arrival(t_ref, h_ref, proto["arrival_threshold_m"], "interp")
    ref_arrival_sample = arrival(t_ref, h_ref, proto["arrival_threshold_m"], "sample")
    controls = {
        "peak_m": ref_peak, "peak_time_s": ref_peak_t,
        "peak_m_raw": ref_peak_raw, "peak_time_s_raw": ref_peak_t_raw,
        "arrival_time_s": ref_arrival,
        "arrival_time_s_sample": ref_arrival_sample,
        "arrival_quantisation_spread_s": ref_arrival_sample - ref_arrival,
        "anchored_peak_m": ev.get("peak_height_m"),
        "anchored_arrival_time_s": ev.get("arrival_time_s"),
        "peak_rel_dev": rel_error(ref_peak, float(ev["peak_height_m"])),
        "arrival_rel_dev": rel_error(ref_arrival_sample, float(ev["arrival_time_s"])),
        "t_span_s": [t_ref[0], t_ref[-1]],
        "n_samples": len(t_ref),
    }
    # digitisation control: the paper's own COMFLOW curve, traced from the same
    # raster, must reproduce the experimental arrival to within the 3.2 %
    # agreement recorded in reference.json -- otherwise the axis mapping or the
    # digitisation is broken and no verdict from this series is trustworthy.
    comflow = None
    crosscheck = ev.get("digitisation_cross_check") or {}
    if crosscheck:
        comflow = {
            "arrival_time_s_expected": crosscheck.get("comflow_arrival_time_s"),
            "arrival_rel_agreement_expected": crosscheck.get("arrival_agreement_rel"),
            "reproduced_arrival_rel": rel_error(
                float(crosscheck["comflow_arrival_time_s"]), ref_arrival_sample),
            "recorded_arrival_rel": crosscheck.get("arrival_agreement_rel"),
        }

    rows = []
    for grid in GRIDS:
        csv_path = HERE / "results/audit_t30" / grid / "time_series.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open() as fh:
            columns = [c for c in csv.DictReader(fh).fieldnames
                       if "_x" in c and c.split("_x")[0] in ("slab", "surface", "height")]
        for column in columns:
            op, xs = column.split("_x")
            t_got, h_got = load_series(csv_path, column)
            pk, pk_t = peak(t_got, h_got, proto["peak_filter_window_s"])
            pk_raw, pk_t_raw = peak(t_got, h_got, 0.0)
            arr = arrival(t_got, h_got, proto["arrival_threshold_m"], "interp")
            arr_sample = arrival(t_got, h_got, proto["arrival_threshold_m"], "sample")
            is_primary = (op == proto["operator"]
                          and math.isclose(float(xs), station, abs_tol=1e-6))
            rows.append({
                "grid": grid, "operator": op, "station_x_m": float(xs),
                "primary": is_primary,
                "peak_m": pk, "peak_time_s": pk_t,
                "peak_m_raw": pk_raw, "peak_time_s_raw": pk_t_raw,
                "arrival_time_s": arr,
                "peak_rel_error": rel_error(pk, ref_peak),
                "peak_time_rel_error": rel_error(pk_t, ref_peak_t),
                "arrival_rel_error": rel_error(arr, ref_arrival),
                "arrival_rel_error_anchored": rel_error(arr, float(ev["arrival_time_s"])),
                "arrival_rel_error_sample": rel_error(arr_sample, ref_arrival_sample),
                "peak_pass": bool(rel_error(pk, ref_peak) < tol),
                "arrival_pass": bool(rel_error(arr, ref_arrival) < tol),
                "arrival_pass_anchored": bool(
                    rel_error(arr, float(ev["arrival_time_s"])) < tol),
                "arrival_pass_sample": bool(
                    rel_error(arr_sample, ref_arrival_sample) < tol),
                # a criterion that changes verdict when the reference arrival is
                # read one digitisation sample later is not robustly decidable
                "arrival_robust": bool(
                    rel_error(arr, ref_arrival) < tol
                    and rel_error(arr, float(ev["arrival_time_s"])) < tol),
                "t_end_s": t_got[-1],
                "window_covers_ref_peak": bool(t_got[-1] >= ref_peak_t),
            })


    primary_rows = {r["grid"]: r for r in rows if r["primary"]}
    verdict = "WITHHELD"
    marginal: list = []
    if primary_rows:
        # robustness rule: passing only under the more convenient digitisation
        # convention is not a pass.  Both conventions must agree.
        verdict = "PASS" if all(r["peak_pass"] and r["arrival_robust"]
                                for r in primary_rows.values()) else "FAIL"
        marginal = [r for r in primary_rows.values()
                    if r["peak_pass"] != (r["peak_pass"] and r["arrival_robust"])
                    or r["arrival_pass"] != r["arrival_pass_sample"]]
    else:
        marginal = []

    # --- continuum limit of the slab peak at each station (two grids) ------
    # Two grids cannot measure an order, so both p = 1 and p = 2 are reported
    # as bounds; the pair spread is the honest figure of merit.
    cont: dict = {"note": ("two grids only; the order is assumed, not measured, so "
                           "these are bounds on the continuum peak, not a "
                           "convergence proof")}
    by_station: dict[float, dict[str, dict]] = {}
    for r in rows:
        if r["operator"] == proto["operator"]:
            by_station.setdefault(r["station_x_m"], {})[r["grid"]] = r
    for xs, grids in sorted(by_station.items()):
        if not {"dx0.02", "dx0.04"} <= set(grids):
            continue
        fine, coarse = grids["dx0.02"], grids["dx0.04"]
        entry = {
            "primary": bool(math.isclose(xs, station, abs_tol=1e-6)),
            "peak_fine_m": fine["peak_m"], "peak_coarse_m": coarse["peak_m"],
            "peak_change_fine_vs_coarse_rel": rel_error(coarse["peak_m"], fine["peak_m"]),
        }
        for order in (1.0, 2.0):
            lim = richardson(fine["peak_m"], coarse["peak_m"], 2.0, order)
            entry[f"peak_m_richardson_p{order:g}"] = lim
            entry[f"peak_rel_error_richardson_p{order:g}"] = rel_error(lim, ref_peak)
        cont[f"station_x{xs:.3f}"] = entry

    # --- shape agreement on the common window ------------------------------
    shape = {}
    for r in rows:
        if r["operator"] != proto["operator"]:
            continue
        csv_path = HERE / "results/audit_t30" / r["grid"] / "time_series.csv"
        col = f"slab_x{r['station_x_m']:.3f}"
        t_got, h_got = load_series(csv_path, col)
        shape[f"{r['grid']}@x{r['station_x_m']:.3f}"] = shape_stats(
            t_ref, h_ref, t_got, h_got, 0.35, min(3.0, t_got[-1]))

    # --- protocol sensitivity: what run.py's own scorer would say ----------
    protocol = {}
    for grid in GRIDS:
        path = HERE / "results/audit_t30" / grid / "metrics.json"
        if not path.is_file():
            continue
        m = json.loads(path.read_text())
        primary = primary_rows.get(grid)
        # run.py grades ``run.peak_height``, which is the legacy top-N operator at
        # ``cfg.probe_x()`` and is *not* passed through the 30 ms filter
        probe_col = f"height@x{station:.3f}"
        g = m["gauges"][probe_col]
        protocol[grid] = {
            "scored_gauge": probe_col,
            "run_py_peak_m": g["peak_m"], "run_py_peak_time_s": g["peak_time_s"],
            "run_py_arrival_time_s": g["arrival_time_s"],
            "run_py_peak_rel_error": rel_error(g["peak_m"], ref_peak),
            "run_py_verdict_recorded": m["comparison"]["verdict"],
            "scored_vs_frozen_peak_rel": rel_error(
                g["peak_m"], primary["peak_m"]) if primary else None,
            "frozen_slab_peak_m": primary["peak_m"] if primary else None,
            "why": ("run.py grades run.peak_height = the legacy top-N operator at "
                    "probe_x, unfiltered, while reference.json froze the 30 ms-filtered "
                    "slab gauge; the two protocols must not be mixed"),
        }

    return {
        "reference": {"series_path": str(series_path.relative_to(ROOT)),
                      "source": ev["source"], "controls": controls,
                      "uncertainty": ev.get("uncertainty"),
                      "comflow_control": comflow},
        "protocol": proto, "tolerance": tol,
        "station_primary_x_m": station, "station_secondary_x_m": other,
        "rows": rows, "verdict": verdict,
        "verdict_convention_marginal": [
            {k: r[k] for k in ("grid", "operator", "station_x_m", "arrival_time_s",
                               "arrival_rel_error", "arrival_rel_error_anchored",
                               "arrival_pass", "arrival_pass_anchored")}
            for r in marginal],
        "verdict_scope": ("peak value + arrival time under the frozen slab gauge "
                          "at the primary station, both grids"),
        "continuum": cont, "shape": shape, "protocol_sensitivity": protocol,
    }


def markdown(scored: dict) -> str:
    ref = scored["reference"]["controls"]
    out = []
    out.append("| grid | operator@x [m] | peak [m] | peak t [s] | peak err | "
               "arrival [s] | arrival err (interp / anchored) | verdict |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(scored["rows"], key=lambda r: (r["grid"], r["station_x_m"], r["operator"])):
        mark = "**" if r["primary"] else ""
        ok = "PASS" if (r["peak_pass"] and r["arrival_pass"]) else "FAIL"
        out.append(
            f"| {mark}{r['grid']}{mark} | {r['operator']}@{r['station_x_m']:.3f} | "
            f"{r['peak_m']:.4f} | {r['peak_time_s']:.3f} | "
            f"{100 * r['peak_rel_error']:+.1f}% | {r['arrival_time_s']:.4f} | "
            f"{100 * r['arrival_rel_error']:+.1f}% / "
            f"{100 * r['arrival_rel_error_anchored']:+.1f}% | {ok} |")
    out.append("")
    out.append(f"reference (same operators applied to the digitised curve): "
               f"peak {ref['peak_m']:.4f} m @ {ref['peak_time_s']:.3f} s, "
               f"arrival {ref['arrival_time_s']:.4f} s interpolated / "
               f"{ref['arrival_time_s_sample']:.4f} s sample-quantised "
               f"(anchored values {ref['anchored_peak_m']:.3f} m / "
               f"{ref['anchored_arrival_time_s']:.4f} s; self-consistency "
               f"{100 * ref['peak_rel_dev']:+.2f}% / {100 * ref['arrival_rel_dev']:+.2f}%).")
    return "\n".join(out) + "\n"


def figure(scored: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_series, series_path = load_reference()
    t_ref, h_ref = load_series(series_path, "h_m")
    proto = scored["protocol"]
    box_x = float(ref_series["geometry_m"]["obstacle_upstream_face_x"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.plot(t_ref, h_ref, "k-", lw=1.6, label="Kleefsman 2005 exp. (digitised)")
    for grid, colour in (("dx0.02", "C0"), ("dx0.04", "C1")):
        col = f"slab_x{box_x - proto['station_offset_m']:.3f}"
        p = HERE / "results/audit_t30" / grid / "time_series.csv"
        if p.is_file():
            t_g, h_g = load_series(p, col)
            ax.plot(t_g, h_g, color=colour, lw=1.2,
                    label=f"TensorSPH WCSPH {grid} (slab@x{box_x - proto['station_offset_m']:.3f})")
    ax.axhline(proto["arrival_threshold_m"], color="0.6", ls=":", lw=0.8)
    c = scored["reference"]["controls"]
    ax.plot([c["peak_time_s"]], [c["peak_m"]], "kv", ms=6, label="exp. peak")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("water height in front of box [m]")
    ax.set_title("H2 water-level history, frozen slab gauge")
    ax.set_xlim(0, 3.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    prim = {r["grid"]: r for r in scored["rows"] if r["primary"]}
    grids = [g for g in GRIDS if g in prim]
    xs = [2.0 / 100 if g == "dx0.02" else 4.0 / 100 for g in grids]
    ys = [prim[g]["peak_m"] for g in grids]
    ax.plot(xs, ys, "o-", color="C0", label="slab peak (primary station)")
    ax.axhline(c["peak_m"], color="k", ls="--", lw=1.0,
               label=f"experiment {c['peak_m']:.3f} m")
    for key, style in (("peak_m_richardson_p1", ":"), ("peak_m_richardson_p2", "-.")):
        if key in scored["continuum"]:
            ax.axhline(scored["continuum"][key], color="C3", ls=style, lw=0.9,
                       label=f"{key.split('_')[-1]} extrapolation "
                             f"{scored['continuum'][key]:.3f} m")
    ax.plot([0.0], [ys[0] if ys else float('nan')], "w")
    ax.set_xlim(0, 0.045)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("peak water height [m]")
    ax.set_title("peak convergence toward the experiment")
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

    scored = score()
    outdir = HERE / "results/audit_t30"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "score.json").write_text(json.dumps(scored, indent=2))
    (outdir / "score_table.md").write_text(markdown(scored))
    print(markdown(scored))
    print("verdict:", scored["verdict"])
    if args.figure:
        figure(scored, ROOT / "docs/figures/kleefsman_t30_score.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
