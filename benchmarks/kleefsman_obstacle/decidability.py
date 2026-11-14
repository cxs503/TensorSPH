"""Decidability audit of the Kleefsman (2005) obstacle-case acceptance metric.

Why this module exists
----------------------
The case-pool acceptance rule for ``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`` is a
*relative* one: ``|got - ref| / |ref| < 10 %`` for the peak water level in front
of the box and for its arrival time.  A relative rule is only meaningful when
the numerical uncertainty of ``got`` is itself well below the tolerance --
otherwise the verdict is not falsifiable in either direction.

The committed evidence bundle contains several conditions that break that
requirement, and they are *checkable from the stored JSON alone*:

1. **Horizon censoring.**  ``dx = 0.02, c0 = 40 m/s`` stores
   ``peak_time_s = 1.200128`` with ``t_end = 1.2``: the maximum sits on the
   last sample, i.e. the peak was never *observed*, only truncated.  The same
   holds for the ``dx = 0.01`` run (peak at 0.966 s of a 1.0 s horizon).
   Counting a censored value as a peak turns a lower bound into a
   "measurement".
2. **Mixed horizons.**  A ladder that mixes ``t_end = 1.2 s`` with
   ``t_end = 1.0 s`` cannot support an observed order of accuracy.
3. **Acoustic Courant.**  ``dt`` must satisfy ``dt * c0 / (h_factor * dx) <=
   max_courant``.  The stored step count of the ``dx = 0.01`` entry implies a
   step that is roughly 2.5x the limit of the code that produced it.
4. **Compressibility.**  WCSPH keeps ``|drho|/rho_0`` small by construction;
   the stored runs reach 12-14 % at ``c0 = 20 m/s``.  Because the peak height
   reacts to ``c0`` (the c0 family is stored), that error propagates directly
   into the acceptance metric.
5. **Gauge-operator spread.**  The same run gives three different "water
   levels" (top-N particle stack / surface-corrected stack / fixed-aperture
   slab) which differ by up to 21 %.

The module computes, from the committed results, an explicit uncertainty
budget for the acceptance metric and answers one question: *is the 10 %
tolerance decidable with this evidence?*  It is deliberately dependency-free
(no solver import, no numpy) so the checks can be unit tested against synthetic
proof-of-defect inputs.

Usage::

    python benchmarks/kleefsman_obstacle/decidability.py \
        --results benchmarks/kleefsman_obstacle/results \
        --out benchmarks/kleefsman_obstacle/results/audit_protocol
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# defaults of tensorsph.droplet_impact.Config, repeated here so that the audit
# is a pure function of the stored evidence (tests pin these against the solver)
MAX_COURANT = 0.30
H_FACTOR = 1.30

#: a peak found closer than this fraction of the horizon to ``t_end`` is not
#: considered observed (the trace still had an upward trend when the run ended)
CENSOR_MARGIN_FRAC = 0.05
#: and never accept less than this absolute margin [s]
CENSOR_MARGIN_ABS = 0.05
#: standard WCSPH compressibility criterion |drho|/rho_0
COMPRESSION_LIMIT = 0.01
#: case-pool tolerance (see feature_ledger acceptance for this slug)
TOLERANCE = 0.10


# --------------------------------------------------------------------------- #
# pure checks
# --------------------------------------------------------------------------- #
def dt_acoustic(dx: float, sound_speed: float, max_courant: float = MAX_COURANT,
                h_factor: float = H_FACTOR) -> float:
    """Largest stable acoustic step for a grid/sound-speed pair."""
    return max_courant * h_factor * dx / sound_speed


def implied_dt(row: dict) -> float:
    """Mean step actually used: ``simulated_time / steps`` (stored evidence)."""
    steps = row.get("steps")
    sim = row.get("simulated_time_s", row.get("t_end"))
    if not steps:
        return float("nan")
    return float(sim) / float(steps)


def check_horizon(row: dict, margin_frac: float = CENSOR_MARGIN_FRAC,
                  margin_abs: float = CENSOR_MARGIN_ABS) -> dict:
    """Is the stored peak actually *observed* inside the horizon?

    A run whose maximum sits within ``margin`` of ``t_end`` reports a lower
    bound on the peak, not the peak.  The check is on the stored numbers only
    and therefore reproducible from the JSON.
    """
    t_end = float(row.get("t_end", float("nan")))
    peak_t = float(row.get("peak_time_s", float("nan")))
    margin = max(margin_frac * t_end, margin_abs)
    if not (math.isfinite(t_end) and math.isfinite(peak_t)):
        return {"censored": True, "reason": "missing t_end/peak_time", "margin": margin}
    slack = t_end - peak_t
    censored = slack <= margin
    return {
        "censored": bool(censored), "margin_s": margin, "slack_s": slack,
        "peak_time_s": peak_t, "t_end": t_end,
        "reason": ("peak at last sample(s): report as lower bound"
                   if censored else "peak observed with margin"),
    }


def check_courant(row: dict, max_courant: float = MAX_COURANT,
                  h_factor: float = H_FACTOR) -> dict:
    """Recompute the acoustic Courant number from the stored step count."""
    dx = float(row.get("dx", float("nan")))
    c0 = float(row.get("sound_speed", 20.0))
    dt = implied_dt(row)
    if not (math.isfinite(dx) and math.isfinite(dt) and c0 > 0.0):
        return {"ok": False, "reason": "missing dx/dt/c0"}
    limit = dt_acoustic(dx, c0, max_courant, h_factor)
    courant = dt * c0 / (h_factor * dx)
    return {
        "ok": bool(dt <= limit * (1.0 + 1e-9)),
        "dt_implied_s": dt, "dt_acoustic_limit_s": limit,
        "courant": courant, "max_courant": max_courant,
        "ratio_to_limit": dt / limit if limit > 0 else float("inf"),
    }


def check_compression(row: dict, limit: float = COMPRESSION_LIMIT,
                      speed_ref: float | None = None) -> dict:
    """Density error against the WCSPH criterion and the ``c0 >= 10 U`` rule."""
    comp = float(row.get("max_density_compression", float("nan")))
    umax = float(row.get("max_speed", float("nan")))
    c0 = float(row.get("sound_speed", 20.0))
    out = {"compression": comp, "limit": limit, "sound_speed": c0,
           "ok": bool(math.isfinite(comp) and comp <= limit)}
    if math.isfinite(umax) and c0 > 0.0:
        out["mach_max"] = umax / c0
        out["sound_speed_for_10U"] = 10.0 * umax
        out["ten_u_rule_ok"] = bool(c0 >= 10.0 * umax)
    return out


def check_operator_spread(row: dict) -> dict:
    """Peak spread across the three gauge operators of one run."""
    gauges = row.get("gauges") or {}
    out = {}
    for station, tag in ((2.426, "x2.426"), (2.228, "x2.228")):
        vals = {k: float(v["peak_m"]) for k, v in gauges.items()
                if k.endswith("@" + tag) and math.isfinite(float(v.get("peak_m", float("nan"))))}
        if len(vals) < 2:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        mean = sum(vals.values()) / len(vals)
        out[station] = {
            "operators": vals,
            "spread_rel": (hi - lo) / mean if mean else float("nan"),
            "range_m": hi - lo,
        }
    return out


def compressibility_family(rows: list[dict]) -> dict:
    """Fit ``peak = a + b / c0^2`` to one grid's sound-speed family.

    The fit is the cheapest honest extrapolation to the incompressible limit,
    but it is only reported together with its residuals: a family in which the
    peak *increases* with ``c0`` (the stored ``dx = 0.04`` family does) has no
    monotone limit, and that is itself the finding.
    """
    pts = [(float(r["sound_speed"]), float(r["peak_height_m"])) for r in rows
           if r.get("sound_speed") and r.get("peak_height_m") is not None]
    pts = [(c0, p) for c0, p in pts if math.isfinite(c0) and math.isfinite(p)]
    if len(pts) < 3:
        return {"status": "INSUFFICIENT_POINTS", "points": pts}
    xs = [1.0 / (c0 * c0) for c0, _ in pts]
    ys = [p for _, p in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else float("nan")
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    order = [c0 for c0, _ in pts]
    return {
        "status": "OK", "points": pts,
        "a_incompressible_peak_m": a, "b_m2_per_s2": b,
        "residuals_m": resid, "max_abs_residual_m": max(abs(r) for r in resid),
        "monotone_decreasing": all(y2 <= y1 for y1, y2 in zip(ys, ys[1:])),
        "sound_speed_order": order,
        "spread_rel": (max(ys) - min(ys)) / (sum(ys) / n) if n else float("nan"),
    }


def combine(*terms: float) -> float:
    """Root-sum-square of independent uncertainty terms (fractions)."""
    vals = [t for t in terms if t is not None and math.isfinite(t)]
    return math.sqrt(sum(v * v for v in vals)) if vals else float("nan")


def decide(uncertainty: float, tolerance: float = TOLERANCE) -> dict:
    """Can a ``tolerance``-level comparison be decided at this uncertainty?

    A verdict is decidable only when the numerical uncertainty leaves room for
    the tolerance itself; ``uncertainty >= tolerance`` means a measured error
    of 0 % and one of 2 x tolerance are indistinguishable.
    """
    ok = math.isfinite(uncertainty) and uncertainty < tolerance
    return {
        "decidable": bool(ok), "uncertainty_rel": uncertainty,
        "tolerance": tolerance,
        "headroom_rel": tolerance - uncertainty if math.isfinite(uncertainty) else float("nan"),
        "rule": "decidable iff numerical uncertainty < case tolerance",
    }


# --------------------------------------------------------------------------- #
# evidence assembly
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def gauge_rows(results_dir: Path) -> dict[float, dict]:
    """Per-grid gauge operator records from the committed gauge ladder."""
    path = Path(results_dir) / "gauge_ladder" / "ladder.json"
    if not path.exists():
        return {}
    out: dict[float, dict] = {}
    for row in load_json(path).get("rows", []):
        if row.get("gauges"):
            out[float(row["dx"])] = row
    return out


def discover(results_dir: Path) -> dict[str, dict]:
    """Every run record in ``results/`` that carries a grid size and a peak."""
    out: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            row = load_json(path)
        except (ValueError, OSError):
            continue
        if not isinstance(row, dict) or not {"dx", "peak_height_m"} <= set(row):
            continue
        out[path.stem] = row
    return out


def audit(results_dir: Path, tolerance: float = TOLERANCE,
          tags: tuple[str, ...] | None = None) -> dict:
    """Run every check on the committed evidence and build the budget.

    ``tags`` selects a subset of the discovered run records (used by the tests
    to pin the historical bundle without pinning the newest runs).
    """
    results_dir = Path(results_dir)
    report: dict = {"case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
                    "tolerance": tolerance, "grids": {}, "findings": []}
    gauges_by_dx = gauge_rows(results_dir)
    records = discover(results_dir)
    if tags is not None:
        records = {t: records[t] for t in tags if t in records}

    for tag, row in records.items():
        row.setdefault("sound_speed", 20.0)
        row.setdefault("simulated_time_s", row.get("t_end"))
        horizon = check_horizon(row)
        courant = check_courant(row)
        comp = check_compression(row)
        spread = check_operator_spread(gauges_by_dx.get(float(row["dx"]), {}))
        report["grids"][tag] = {
            "dx": row.get("dx"), "t_end": row.get("t_end"),
            "peak_height_m": row.get("peak_height_m"),
            "peak_time_s": row.get("peak_time_s"),
            "arrival_time_s": row.get("arrival_time_s"),
            "horizon": horizon, "courant": courant,
            "compression": comp, "operator_spread": spread,
        }
        if horizon["censored"]:
            report["findings"].append({
                "id": f"CENSORED-{tag}", "severity": "blocking",
                "detail": f"peak at t={horizon['peak_time_s']:.3f} s with "
                          f"t_end={horizon['t_end']:.3f} s (slack "
                          f"{horizon['slack_s']:.3f} s <= margin "
                          f"{horizon['margin_s']:.3f} s): peak is a lower bound",
            })
        if not courant["ok"]:
            report["findings"].append({
                "id": f"COURANT-{tag}", "severity": "blocking",
                "detail": f"implied dt {courant['dt_implied_s']:.3e} s is "
                          f"{courant['ratio_to_limit']:.2f}x the acoustic limit "
                          f"{courant['dt_acoustic_limit_s']:.3e} s "
                          f"(Courant {courant['courant']:.3f} > {courant['max_courant']})",
            })
        if not comp["ok"]:
            report["findings"].append({
                "id": f"COMPRESSION-{tag}", "severity": "major",
                "detail": f"|drho|/rho0 = {comp['compression']*100:.1f}% > "
                          f"{comp['limit']*100:.0f}%; c0/umax = "
                          f"{comp['sound_speed']/row.get('max_speed', float('nan')):.2f} "
                          f"(10U rule wants c0 >= {comp.get('sound_speed_for_10U', float('nan')):.1f} m/s)",
            })
        for station, s in spread.items():
            if s["spread_rel"] > 0.5 * tolerance:
                report["findings"].append({
                    "id": f"GAUGE-{tag}@{station}", "severity": "major",
                    "detail": f"operator peak spread {s['spread_rel']*100:.1f}% of the "
                              f"mean at x={station} m exceeds half the tolerance",
                })

    # ---- protocol uniformity of the uncensored ladder -------------------- #
    rows = []
    for tag, g in report["grids"].items():
        rows.append({"dx": g["dx"], "t_end": g["t_end"],
                     "censored": g["horizon"]["censored"],
                     "peak": g["peak_height_m"], "arrival": g["arrival_time_s"]})
    t_ends = {round(float(r["t_end"]), 6) for r in rows}
    report["ladder"] = {
        "rows": sorted(rows, key=lambda r: -float(r["dx"])),
        "uniform_horizon": len(t_ends) == 1,
        "horizons_s": sorted(t_ends),
        "censored_grids": [r["dx"] for r in rows if r["censored"]],
    }
    if len(t_ends) > 1:
        report["findings"].append({
            "id": "MIXED-HORIZON", "severity": "blocking",
            "detail": f"ladder mixes t_end = {sorted(t_ends)} s; an observed "
                      f"order of accuracy is undefined on it",
        })

    # ---- sound-speed family --------------------------------------------- #
    scan = results_dir / "c0_scan" / "c0_scan_summary.json"
    if scan.exists():
        fam = load_json(scan).get("rows", [])
        fam_by_dx: dict[float, list[dict]] = {}
        for r in fam:
            fam_by_dx.setdefault(float(r["dx"]), []).append(r)
        report["compressibility_family"] = {
            str(dx): compressibility_family(sorted(rs, key=lambda r: r["sound_speed"]))
            for dx, rs in sorted(fam_by_dx.items())
        }
        for dx, fit in report["compressibility_family"].items():
            if fit["status"] != "OK":
                continue
            if not fit["monotone_decreasing"]:
                report["findings"].append({
                    "id": f"C0-NONMONOTONE-dx{dx}", "severity": "major",
                    "detail": "peak does not decrease monotonically with c0 "
                              f"(peaks {[p for _, p in fit['points']]}): the "
                              "c0-dependence stored as 'compressibility effect' is "
                              "not a systematic trend on this grid",
                })

    # ---- uncertainty budget --------------------------------------------- #
    uncensored = [r for r in rows if not r["censored"]]
    peaks = [float(r["peak"]) for r in uncensored
             if r["peak"] is not None and math.isfinite(float(r["peak"]))]
    disc = ((max(peaks) - min(peaks)) / (sum(peaks) / len(peaks))
            if len(peaks) >= 2 else float("nan"))
    fam_spreads = [f["spread_rel"] for f in report.get("compressibility_family", {}).values()
                   if f.get("status") == "OK" and math.isfinite(f.get("spread_rel", float("nan")))]
    comp_term = max(fam_spreads) if fam_spreads else float("nan")
    gauge_terms = [s["spread_rel"] for g in report["grids"].values()
                   for s in g["operator_spread"].values()]
    gauge_term = max(gauge_terms) if gauge_terms else float("nan")
    total = combine(disc, comp_term, gauge_term)
    report["uncertainty_budget"] = {
        "discretisation_rel": disc, "n_uncensored_grids": len(peaks),
        "compressibility_rel": comp_term,
        "gauge_operator_rel": gauge_term,
        "total_rel": total,
        "verdict": decide(total, tolerance),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--tags", nargs="*", default=None,
                    help="restrict the audit to these run records")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.results) / "audit_protocol"
    out.mkdir(parents=True, exist_ok=True)
    report = audit(Path(args.results), tolerance=args.tolerance,
                   tags=tuple(args.tags) if args.tags else None)
    (out / "decidability.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["uncertainty_budget"], indent=2))
    print(json.dumps(report["findings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())