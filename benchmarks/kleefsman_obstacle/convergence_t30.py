"""Three-grid, *measured-order* convergence of the frozen Kleefsman acceptance metric.

Why this file exists
--------------------
``results/audit_t30/score.json`` scores the frozen protocol (slab gauge, 0.08 m
aperture, station = box face - 0.05 m, 30 ms peak filter, 0.02 m arrival
threshold) on the two long runs ``dx0.04`` and ``dx0.02`` (``t_end = 3.0 s``) and
reaches ``FAIL``.  Its continuum block is explicitly labelled

    "two grids only; the order is assumed, not measured, so these are bounds on
     the continuum peak, not a convergence proof"

With two grids every statement about the continuum limit is an assumption: the
peak at the primary station moves **+10.4 %** between ``dx0.04`` and ``dx0.02``,
which is larger than the case tolerance (10 %), so *the acceptance number itself
is not yet separated from its own discretisation error*.  This module closes that
gap with the third grid (``dx0.01``, ``t_end = 3.0 s``), i.e. it answers the only
question that decides the case:

    is the +29 % peak deviation at the primary station a resolution artifact
    (it would collapse as dx -> 0) or a genuine modelling discrepancy
    (it survives at dx -> 0)?

Method (all inputs are committed CSV tables; no number is typed in by hand):

1. score every grid under the frozen protocol with
   :func:`acceptance_frozen.score` (the same function that produces
   ``results/peak_horizon/score.json``), for *both* candidate H2 stations;
2. form the coarse -> fine ladder and get the observed order ``p``, the
   Richardson continuum value and the Roache GCI of the finest grid with
   :mod:`convergence_analysis` (``observed_order`` / ``richardson`` /
   ``gci_fine``); a non-monotone ladder yields ``p = nan`` -- reported as
   ``NOT_CONVERGED`` with the measured spread as the bound, never as a
   convergence claim;
3. apply the ASME V&V 20 comparison-error criterion

       E = S - D,   U_V = sqrt(U_num^2 + U_D^2),   validated  <=>  |E| <= U_V

   with ``S`` the finest-grid (and, separately, the Richardson) peak, ``D`` the
   digitised experiment and ``U_D`` its digitisation uncertainty
   (``reference.json:experimental_values.uncertainty``).  ``|E| > U_V`` means the
   deviation is *not* numerical and must be attributed to the model;
4. report the time of the compared extremum against the experimental peak time,
   because two identical numbers can still be two different physical events.

Outputs (committed):

    results/convergence_t30/convergence.json     full ladder + verdicts
    results/convergence_t30/convergence.md       the table used by the doc
    results/convergence_t30/ladder.png           peak/arrival ladder vs reference
    results/convergence_t30/timeseries.png       the three series vs experiment

Usage::

    python benchmarks/kleefsman_obstacle/convergence_t30.py
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

import acceptance_frozen as af  # noqa: E402
import convergence_analysis as ca  # noqa: E402

OUT = HERE / "results" / "convergence_t30"

#: Candidate H2 stations [m].  ``primary`` is the frozen one (box face - 0.05 m,
#: x = 2.426); ``secondary`` is box face - 0.248 m (x = 2.228), the station the
#: arrival attribution (m21173) shows would remove 52-63 % of the arrival
#: deficit.  The paper does not give the H2 coordinate machine-readably, so both
#: are carried and reported side by side.
STATIONS = {"primary": 2.426, "secondary": 2.228}

#: One CSV per grid, coarse -> fine.  Several drivers write the same frozen
#: gauge columns (``slab_x*``/``height_x*``/``surface_x*``); list the paths in
#: preference order and use the first that exists.  The three grids must share
#: ``t_end`` -- ``check_horizon`` in ``acceptance_frozen`` refuses a censored
#: record, and ``_uniform_horizon`` below refuses a mixed-horizon ladder.
SERIES_CANDIDATES = {
    0.04: ["results/audit_t30/dx0.04/time_series.csv"],
    0.02: ["results/audit_t30/dx0.02/time_series.csv"],
    0.01: ["results/gap_t30/dx0.01_time_series.csv",
           "results/peak_horizon/dx0.01_time_series.csv"],
}

#: Fraction of the experimental value below which a GCI is "small enough" to
#: quote the finest-grid number as the continuum one.  Chosen equal to the case
#: tolerance: if the discretisation uncertainty of the acceptance quantity is
#: already 10 %, the acceptance test cannot be decided by the numbers.
GCI_REFERENCE_BAND = 0.10

#: A compared extremum that sits further than this from the experimental peak
#: time is a *different* event (the same window ``acceptance_frozen`` uses for
#: its event-window peak).
EVENT_HALF_WIDTH_S = af.EVENT_HALF_WIDTH_S


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def resolve_series(dx: float, here: Path = HERE,
                   candidates: dict | None = None) -> Path | None:
    cands = (candidates or SERIES_CANDIDATES)[dx]
    for rel in cands:
        path = here / rel
        if path.exists():
            return path
    return None


def _uniform_horizon(rows: list[dict], tol_s: float = 1e-3) -> dict:
    """Is the ladder's *comparability* protected: equal horizons + event window?

    Two different things can go wrong and they must not be confused:

    * **mixed horizons** -- records that stop at different times.  Harmless for
      the peak *if* every record covers the experimental event window (a peak
      that has already happened is not moved by integrating further), but it
      must be reported, because a maximum taken on a truncated record is a
      lower bound, not a value (``acceptance_frozen`` censors exactly that).
    * **event not covered** -- at least one record stops before the compared
      extremum, which makes the ladder meaningless.
    """
    ends = [r["t_end_s"] for r in rows]
    if any(not math.isfinite(e) for e in ends):
        return {"ok": False, "reason": "unknown t_end", "t_ends": ends}
    out = {"ok": True, "t_end_s": min(ends), "t_ends": ends,
           "uniform": bool(max(ends) - min(ends) <= tol_s)}
    # horizon adequacy: the compared extrema of every grid must sit inside its
    # own record, i.e. not within the peak-filter half width of the last sample
    half = 0.5 * af.FROZEN["peak_window_s"]
    late = [r["dx"] for r in rows
            if not math.isfinite(r["peak_time_s"])
            or r["peak_time_s"] + half >= r["t_end_s"]]
    if late:
        out.update(ok=False, reason="extreme extremum at the record end", dx=late)
    return out


def frozen_ladder(ref: dict, here: Path = HERE,
                  candidates: dict | None = None) -> tuple[dict, dict]:
    """Score every available grid at every station under the frozen protocol."""
    stations: dict[str, dict] = {}
    missing: list[float] = []
    for dx in sorted((candidates or SERIES_CANDIDATES), reverse=True):
        path = resolve_series(dx, here, candidates)
        if path is None:
            missing.append(dx)
            continue
        for label, x in STATIONS.items():
            col = f"{af.FROZEN['operator']}_x{x:.3f}"
            times, values = af.load_series(path, col)
            scored = af.score(times, values, ref, t_end=times[-1])
            row = {
                "dx": dx,
                "station": label,
                "station_x_m": x,
                "column": col,
                "path": str(path.relative_to(here)),
                "t_end_s": times[-1],
                "peak_m": scored["peak_row"]["got"],
                "peak_time_s": scored.get("peak_time_s"),
                "arrival_s": scored["arrival_row"]["got"],
                "peak_rel_error": scored["peak_row"]["rel_error"],
                "arrival_rel_error": scored["arrival_row"]["rel_error"],
                "censored": scored["verdict"] == "NOT_COMPARABLE",
            }
            stations.setdefault(label, {"station_x_m": x, "rows": []})
            stations[label]["rows"].append(row)
    return stations, {"missing_dx": missing}


# --------------------------------------------------------------------------- #
# convergence + validation
# --------------------------------------------------------------------------- #
#: Nominal order of the SPH interior stencil used by the case (m20773 froze the
#: protocol on this solver).  When the *observed* order exceeds it, the GCI is
#: re-evaluated at the nominal order -- Roache's conservative rule: a
#: super-convergent observed order usually means the target quantity is an
#: extremum whose error is not a smooth functional of dx, and the resulting GCI
#: is optimistic.
NOMINAL_ORDER = 2.0


def ladder_stats(rows: list[dict], key: str, safety: float = 1.25,
                 nominal_order: float = NOMINAL_ORDER) -> dict:
    """Observed order / Richardson / GCI for one metric of one station ladder."""
    ordered = sorted(rows, key=lambda r: -r["dx"])
    analysed = ca.analyze_metric([dict(r) for r in ordered], key, safety=safety)
    vals = [r[key] for r in ordered]
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    out = {
        "metric": key,
        "dx": [r["dx"] for r in ordered],
        "values": vals,
        "n_grids": len(ordered),
        # a 2-grid ladder has no order, but it still has (or lacks) monotonicity
        "monotone": bool(diffs) and all(d == 0 or d * diffs[0] > 0 for d in diffs),
        "p": analysed.get("p_finest"),
        "richardson": analysed.get("richardson"),
        "gci_fine": analysed.get("gci_fine"),
        "convergence": analysed.get("verdict"),
        "spread_rel": (max(vals) - min(vals)) / abs(max(vals)) if vals and max(vals) else float("nan"),
    }
    # conservative GCI: never trust an observed order above the nominal one
    p_obs = out["p"]
    r = (out["dx"][-2] / out["dx"][-1]) if len(out["dx"]) >= 2 else float("nan")
    if math.isfinite(p_obs or float("nan")) and p_obs > 0 and math.isfinite(r) and r > 1:
        p_eff = min(p_obs, nominal_order)
        gci_eff = ca.gci_fine(vals[-3], vals[-2], vals[-1], p_eff, r, safety) \
            if len(vals) >= 3 else float("nan")
        out["p_effective"] = p_eff
        out["gci_fine_conservative"] = gci_eff
        out["order_capped"] = bool(p_obs > nominal_order)
    else:
        out["p_effective"] = float("nan")
        out["gci_fine_conservative"] = float("nan")
        out["order_capped"] = False
    out["u_num_rel"] = (out["gci_fine_conservative"]
                        if math.isfinite(out["gci_fine_conservative"] or float("nan"))
                        else out["spread_rel"])
    out["u_num_rel_source"] = ("GCI(p_eff)" if math.isfinite(
        out["gci_fine_conservative"] or float("nan")) else "grid_spread(no order)")
    out["gci_below_reference_band"] = bool(
        math.isfinite(out["u_num_rel"] or float("nan"))
        and out["u_num_rel"] <= GCI_REFERENCE_BAND)
    return out


def validation_uncertainty(u_num: float, u_exp: float) -> float:
    """ASME V&V 20 combined validation uncertainty ``sqrt(u_num^2 + u_exp^2)``."""
    parts = [u for u in (u_num, u_exp) if u is not None and math.isfinite(u)]
    if not parts:
        return float("nan")
    return math.sqrt(sum(u * u for u in parts))


def asme_vv20(sim: float, exp: float, u_num: float, u_exp: float) -> dict:
    """Comparison-error criterion of ASME V&V 20-2009 (sec. 3).

    ``E = S - D``; the simulation is *validated at level* ``U_V`` when
    ``|E| <= U_V``.  Otherwise ``E`` is a modelling error that discretisation
    refinement cannot remove, and the *validation* result is a failure with the
    magnitude of the unmodelled effect quantified by ``|E|``.
    """
    if not all(math.isfinite(v) for v in (sim, exp)):
        return {"E": float("nan"), "U_V": float("nan"), "verdict": "NO_DATA"}
    e = sim - exp
    u_v = validation_uncertainty(u_num, u_exp)
    if not math.isfinite(u_v):
        return {"E": e, "U_V": float("nan"), "verdict": "NO_UNCERTAINTY"}
    # the criterion is |E| <= U_V; the absolute guard keeps a value that sits on
    # the boundary from being flipped by the ~1e-16 relative rounding of E itself
    tol = u_v + 1e-12 * max(1.0, abs(exp))
    return {"E": e, "E_rel": abs(e) / abs(exp) if exp else float("nan"),
            "U_V": u_v, "U_V_rel": u_v / abs(exp) if exp else float("nan"),
            "verdict": "VALIDATED_WITHIN_U_V" if abs(e) <= tol else "MODEL_ERROR_EXCEEDS_U_V"}


def station_verdict(entry: dict, ref: dict) -> dict:
    """Per-station: convergence of the peak, ASME V&V 20, and event identity."""
    exp = ref["experimental_values"]
    rows = entry["rows"]
    if len(rows) < 2:
        return {"verdict": "INSUFFICIENT_GRIDS", "n_grids": len(rows)}
    peak = ladder_stats(rows, "peak_m")
    arrival = ladder_stats(rows, "arrival_s")
    ptime = ladder_stats(rows, "peak_time_s")

    exp_unc = exp.get("uncertainty", {}) or {}
    u_exp_peak = float(exp_unc.get("peak_height_m", 0.0) or 0.0)
    u_exp_arr = float(exp_unc.get("arrival_time_s", 0.0) or 0.0)
    ref_peak = float(exp["peak_height_m"])
    ref_arr = float(exp.get("arrival_time_s_interpolated", exp["arrival_time_s"]))

    # Numerical uncertainty of the finest grid: the GCI when the ladder is
    # monotone; otherwise the measured grid spread, which bounds the
    # discretisation effect without pretending to know an order.
    def u_num_of(st: dict, finest: float) -> tuple[float, str]:
        """Numerical uncertainty of the finest grid [same unit as the metric].

        GCI at the conservative order when an order exists at all; otherwise the
        measured grid spread, which bounds the discretisation effect without
        pretending to know an order.
        """
        rel = st.get("u_num_rel")
        if rel is not None and math.isfinite(rel):
            return rel * abs(finest), st.get("u_num_rel_source", "GCI")
        if math.isfinite(st["spread_rel"]):
            return st["spread_rel"] * abs(finest), "grid_spread(no order)"
        return float("nan"), "none"

    fine = rows[-1] if rows[-1]["dx"] < rows[0]["dx"] else rows[0]
    fine = min(rows, key=lambda r: r["dx"])
    cont_peak = peak["richardson"] if math.isfinite(peak["richardson"] or float("nan")) else float("nan")
    cont_arr = arrival["richardson"] if math.isfinite(arrival["richardson"] or float("nan")) else float("nan")

    u_peak, u_peak_src = u_num_of(peak, fine["peak_m"])
    u_arr, u_arr_src = u_num_of(arrival, fine["arrival_s"])

    out = {
        "n_grids": len(rows),
        "peak": peak,
        "arrival": arrival,
        "peak_time": ptime,
        "u_num_peak_src": u_peak_src,
        "u_num_arrival_src": u_arr_src,
        "vv20_peak_fine": asme_vv20(fine["peak_m"], ref_peak, u_peak, u_exp_peak),
        "vv20_peak_continuum": asme_vv20(cont_peak, ref_peak, 0.0, u_exp_peak)
        if math.isfinite(cont_peak) else {"verdict": "NO_CONTINUUM"},
        "vv20_arrival_fine": asme_vv20(fine["arrival_s"], ref_arr, u_arr, u_exp_arr),
    }
    # event identity: is the compared extremum the same physical event?
    dev = abs(fine["peak_time_s"] - float(exp["peak_time_s"])) if math.isfinite(fine["peak_time_s"]) else float("nan")
    out["peak_event"] = {
        "sim_peak_time_s": fine["peak_time_s"],
        "ref_peak_time_s": float(exp["peak_time_s"]),
        "dev_s": dev,
        "same_event": bool(math.isfinite(dev) and dev <= EVENT_HALF_WIDTH_S),
        "note": "the simulated extremum is the first-impact run-up at the gauge; the "
                "experimental maximum at H2 is the return-wave re-impingement (t = 1.703 s)",
    }
    peak_ok = bool(fine["peak_rel_error"] < float(ref["tolerance"]["relative"]))
    arr_ok = bool(fine["arrival_rel_error"] < float(ref["tolerance"]["relative"]))
    ver = out["vv20_peak_fine"]["verdict"]
    if ver == "MODEL_ERROR_EXCEEDS_U_V":
        # |E| > U_V: the deviation is not numerical *under the U_V we can
        # defend*, which covers both converged and unconverged ladders
        verdict = "PEAK_FAIL_MODEL_ERROR"
    elif peak["convergence"] != "CONVERGED":
        # |E| <= U_V only because U_num is large: this is not a pass
        verdict = "PEAK_NOT_CERTIFIABLE_UNCONVERGED"
    elif ver == "VALIDATED_WITHIN_U_V":
        verdict = "PEAK_VALIDATED" if peak_ok else "PEAK_VALIDATED_BUT_OUTSIDE_TOLERANCE"
    else:
        verdict = "PEAK_INCONCLUSIVE"
    out["verdict"] = verdict
    out["tolerance_pass_finest"] = {"peak": peak_ok, "arrival": arr_ok,
                                    "both": bool(peak_ok and arr_ok)}
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def analyse(ref: dict | None = None, here: Path = HERE,
            candidates: dict | None = None) -> dict:
    ref = ref or json.loads((here / "reference.json").read_text())
    stations, meta = frozen_ladder(ref, here, candidates)
    horizon = {}
    for label, entry in stations.items():
        horizon[label] = _uniform_horizon(entry["rows"]) if entry["rows"] else {"ok": False}
        entry["verdicts"] = station_verdict(entry, ref)
    report = {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "mission": "m21718",
        "protocol": dict(af.FROZEN),
        "reference": {
            "peak_height_m": ref["experimental_values"]["peak_height_m"],
            "peak_time_s": ref["experimental_values"]["peak_time_s"],
            "arrival_time_s": ref["experimental_values"].get(
                "arrival_time_s_interpolated", ref["experimental_values"]["arrival_time_s"]),
            "tolerance_rel": ref["tolerance"]["relative"],
            "u_exp": ref["experimental_values"].get("uncertainty", {}),
        },
        "gci_reference_band": GCI_REFERENCE_BAND,
        "horizon_check": horizon,
        "stations": {k: {"station_x_m": v["station_x_m"], "rows": v["rows"],
                         "verdicts": v["verdicts"]} for k, v in stations.items()},
        "meta": meta,
    }
    report["verdict"] = overall_verdict(report)
    return report


def overall_verdict(report: dict) -> dict:
    """Case-level verdict: depends only on the numbers above."""
    st = report["stations"]
    prim = st.get("primary", {}).get("verdicts", {})
    sec = st.get("secondary", {}).get("verdicts", {})
    if not prim or prim.get("verdict") == "INSUFFICIENT_GRIDS":
        return {"verdict": "INSUFFICIENT_GRIDS",
                "why": "fewer than two grids at the frozen primary station"}
    peak_converged = prim["peak"]["convergence"] == "CONVERGED"
    arrival_converged = prim["arrival"]["convergence"] == "CONVERGED"
    return {
        "verdict": "PEAK_FAIL_RESOLVED" if peak_converged else "PEAK_FAIL_NOT_RESOLVED",
        "peak_convergence_primary": prim["peak"]["convergence"],
        "arrival_convergence_primary": prim["arrival"]["convergence"],
        "peak_vv20_primary": prim["vv20_peak_fine"]["verdict"],
        "peak_vv20_secondary": sec.get("vv20_peak_fine", {}).get("verdict"),
        "peak_event_same_primary": prim["peak_event"]["same_event"],
        "why": (
            "the primary-station peak deviation is measured against a converged "
            "ladder (observed order %s, conservative p_eff %s used for the GCI), "
            "so it can be attributed"
            % ("%.2f" % prim["peak"]["p"] if math.isfinite(prim["peak"]["p"] or float("nan"))
               else "undefined",
               "%.2f" % prim["peak"]["p_effective"]
               if math.isfinite(prim["peak"].get("p_effective") or float("nan"))
               else "undefined")
        ) if peak_converged else (
            "the primary-station ladder does not converge (p = %s); the deviation "
            "cannot be separated from the discretisation error"
            % ("%.2f" % prim["peak"]["p"] if math.isfinite(prim["peak"]["p"] or float("nan"))
               else "undefined")),
        "arrival_note": "arrival ladder converged" if arrival_converged else
                        "arrival ladder is not monotone/converged; its bound is the measured grid spread",
    }


def markdown(report: dict) -> str:
    ref = report["reference"]
    L = ["# Kleefsman (2005) frozen-protocol metric: three-grid convergence + ASME V&V 20", "",
         "Programmatically generated by `benchmarks/kleefsman_obstacle/convergence_t30.py` "
         "(no number below is typed in by hand).", "",
         f"* reference: peak {ref['peak_height_m']:.3f} m @ {ref['peak_time_s']:.3f} s, "
         f"arrival {ref['arrival_time_s']:.4f} s, tolerance {ref['tolerance_rel']:.0%}",
         f"* frozen protocol: `{report['protocol']['operator']}` gauge, "
         f"aperture {report['protocol']['aperture_m']:.2f} m, "
         f"peak filter {report['protocol']['peak_window_s']:.3f} s, "
         f"arrival threshold {report['protocol']['arrival_threshold_m']:.2f} m", ""]
    for label, entry in report["stations"].items():
        L += [f"## Station {label} (x = {entry['station_x_m']:.3f} m)", "",
              "| dx [m] | file | t_end [s] | peak [m] | peak err | peak t [s] | arrival [s] | arrival err |",
              "|---|---|---|---|---|---|---|---|"]
        for r in entry["rows"]:
            L.append("| {dx:g} | `{path}` | {t_end_s:.3f} | {peak_m:.4f} | {pe:.1%} | "
                     "{pt:.3f} | {a:.4f} | {ae:.1%} |".format(
                         dx=r["dx"], path=r["path"], t_end_s=r["t_end_s"],
                         peak_m=r["peak_m"], pe=r["peak_rel_error"],
                         pt=r["peak_time_s"], a=r["arrival_s"], ae=r["arrival_rel_error"]))
        v = entry["verdicts"]
        if v.get("verdict") != "INSUFFICIENT_GRIDS":
            L += ["", f"* peak ladder: values {['%.4f' % x for x in v['peak']['values']]}, "
                      f"monotone={v['peak']['monotone']}, p={v['peak']['p']}, "
                      f"Richardson={v['peak']['richardson']}, GCI={v['peak']['gci_fine']}, "
                      f"convergence={v['peak']['convergence']}",
                  f"* arrival ladder: monotone={v['arrival']['monotone']}, p={v['arrival']['p']}, "
                  f"GCI={v['arrival']['gci_fine']}, convergence={v['arrival']['convergence']}",
                  f"* ASME V&V 20 (finest grid, peak): E={v['vv20_peak_fine']['E']:.4f} m, "
                  f"U_V={v['vv20_peak_fine']['U_V']:.4f} m -> **{v['vv20_peak_fine']['verdict']}**",
                  f"* ASME V&V 20 (Richardson peak): {v['vv20_peak_continuum'].get('verdict')}",
                  f"* ASME V&V 20 (finest grid, arrival): E={v['vv20_arrival_fine']['E']:.4f} s, "
                  f"U_V={v['vv20_arrival_fine']['U_V']:.4f} s -> **{v['vv20_arrival_fine']['verdict']}**",
                  f"* compared extremum is the experimental event: "
                  f"{v['peak_event']['same_event']} "
                  f"(sim {v['peak_event']['sim_peak_time_s']:.3f} s vs ref "
                  f"{v['peak_event']['ref_peak_time_s']:.3f} s)", ""]
    v = report["verdict"]
    L += ["## Case verdict", "", f"**{v['verdict']}** — {v['why']}", "",
          f"* {v['arrival_note']}", ""]
    return "\n".join(L)


def figures(report: dict, outdir: Path = OUT) -> list[Path]:
    import csv as _csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    ref = report["reference"]
    paths = []

    # 1) the three series vs the digitised experiment
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    exp_series = ROOT / "docs" / "m20884" / "outputs" / "kleefsman_h2_ref" / "h2_fig19.csv"
    if exp_series.exists():
        with exp_series.open() as fh:
            rows = list(_csv.DictReader(fh))
        et = [float(r["t_s"]) for r in rows]
        eh = [float(r["h_m"]) for r in rows]
    else:
        et, eh = [], []
    for ax, (label, entry) in zip(axes, report["stations"].items()):
        if et:
            ax.plot(et, eh, "k", lw=1.8, label="Kleefsman 2005 (digitised H2)")
        for r in entry["rows"]:
            col = r["column"]
            times, values = af.load_series(HERE / r["path"], col)
            ax.plot(times, values, lw=1.1, ls="--", label=f"dx={r['dx']:g} {col}")
        ax.axvline(ref["peak_time_s"], color="0.5", ls=":", lw=0.9)
        ax.axhline(ref["peak_height_m"], color="0.5", ls=":", lw=0.9)
        ax.set_xlim(0, max([r["t_end_s"] for r in entry["rows"]] + [2.0]))
        ax.set_ylim(0, 0.65)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("water level at gauge [m]")
        ax.set_title(f"station {label} (x = {entry['station_x_m']:.3f} m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    fig.suptitle("Kleefsman obstacle case: frozen slab gauge vs experiment, three grids")
    fig.tight_layout()
    p = outdir / "timeseries.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2) the ladders with the convergence band
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, key, refv, unit in ((axes[0], "peak_m", ref["peak_height_m"], "m"),
                                (axes[1], "arrival_s", ref["arrival_time_s"], "s")):
        for label, entry in report["stations"].items():
            rows = sorted(entry["rows"], key=lambda r: -r["dx"])
            ax.plot([r["dx"] for r in rows], [r[key] for r in rows], "o-", ms=5,
                    label=f"{label} x={entry['station_x_m']:.3f}")
            v = entry["verdicts"]
            if v.get("verdict") == "INSUFFICIENT_GRIDS":
                continue
            st = v["peak"] if key == "peak_m" else v["arrival"]
            cont = st["richardson"]
            gci = st["gci_fine"]
            if math.isfinite(cont or float("nan")):
                band = (gci * abs(cont)) if math.isfinite(gci or float("nan")) else float("nan")
                if math.isfinite(band):
                    ax.errorbar([min(r["dx"] for r in rows) / 2.0], [cont],
                                yerr=[band], fmt="s", ms=5, capsize=4,
                                label=f"{label} Richardson ± GCI")
                else:
                    ax.plot([min(r["dx"] for r in rows) / 2.0], [cont], "s", ms=5,
                            label=f"{label} Richardson")
        ax.axhline(refv, color="C3", ls="--", lw=1.2,
                   label=f"experiment {refv:.4f} {unit}")
        ax.axhspan(refv * 0.9, refv * 1.1, color="C3", alpha=0.10,
                   label="±10 % tolerance")
        ax.set_xscale("log")
        ax.set_xlabel("dx [m]  (0 -> Richardson extrapolation)")
        ax.set_ylabel(f"{key.replace('_m', ' [m]').replace('_s', ' [s]')}")
        ax.set_title(f"{key}: ladder vs reference")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    p = outdir / "ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    report = analyse()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        report["figures"] = [str(p.relative_to(HERE)) for p in figures(report)]
    (OUT / "convergence.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT / "convergence.md").write_text(markdown(report))
    print(json.dumps({"verdict": report["verdict"],
                      "stations": {k: v["verdicts"].get("verdict")
                                   for k, v in report["stations"].items()},
                      "missing_dx": report["meta"]["missing_dx"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())