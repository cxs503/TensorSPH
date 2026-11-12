"""Machine-checked verdict on the *settling* question of the Laplace read-out.

Why this file exists
--------------------
`verdict_audit.py` grades the 1.5-``t_sigma`` relaxation ladder and therefore
cannot separate two explanations of the same observation:

* the closure variant ``interior donors / 2 layers`` is inside the +/-15 % band
  at every ladder point but **inadmissible** at ``(dx = 2 mm, c0 = 20 m/s)``
  (``v_max/v_ref = 0.388``, final snapshot +142.8 % off) -- so it may not be
  quoted as a capability win; or
* the window is simply too short: the drop is released at ``p = 0`` and must
  compress to ``sigma/R``, which excites a capillary mode of period
  ~ ``t_sigma`` at ``nu = 0.01 m^2/s`` (100x water).

`settling.py` advances the same drop for 6 ``t_sigma`` at the *stable* dt and
records the trace.  This script turns that evidence into a verdict, with every
rule computed here rather than argued in prose:

  P1  provenance -- every graded record must span >= ``MIN_WINDOW`` capillary
      times and must not have been clipped by ``max_steps`` (``relax()`` trades
      dt for step count when that binds, which silently changes the scheme).
  P2  decay -- at a point the short ladder called inadmissible, the spurious
      current must actually fall: ``v_max(late)/v_max(early) < DECAY_MAX``.
      A plateau means the parasitic mode is persistent (scheme defect).
  P3  admissibility -- at every acceptance point ``v_max/v_ref`` over the late
      window < 0.1 (Adami et al. 2012).
  P4  band -- ``|tail40| < 15 %`` at every acceptance point (the owner gate).

Nothing here advances the physics either: it reads the raw files and writes
``settling_matrix.json`` next to ``verdict_matrix.json``, so the document
generator and the test suite quote the same numbers.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/settled_verdict.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "settling_matrix.json"

GATE_PCT = 15.0
SPURIOUS_CURRENT_MAX = 0.1
#: A record shorter than this is not allowed to grade the settling question.
MIN_WINDOW = 4.0
#: v_max(late)/v_max(early) must be below this for the transient to be called decayed.
DECAY_MAX = 0.5
#: Case that carries the band (the only closure variant inside it).
CANDIDATE = "closure_on_interior_layers2"
#: The four points of the m2971 acceptance criterion (dx [m], c0 [m/s]).
ACCEPTANCE_POINTS = ((0.002, 20.0), (0.002, 5.0), (0.0015, 20.0), (0.0015, 5.0))


def _quartile_vmax(trace: list[list[float]]) -> list[float]:
    """Maximum spurious current inside each quarter of the window.

    One number cannot tell a decaying transient from a growing mode; the four
    quartile maxima can, and they are read off the recorded trace only.
    """
    n = len(trace)
    out = []
    for q in range(4):
        sl = trace[int(q * n / 4):max(int(q * n / 4) + 1, int((q + 1) * n / 4))]
        out.append(max(v for _, v in sl) if sl else float("nan"))
    return out


def _drift_percent_per_t_sigma(trace: list[list[float]], dp: float) -> float:
    """Least-squares slope of the late-window core pressure, in % of sigma/R per t_sigma."""
    n = len(trace)
    late = trace[int(0.6 * n):]
    if len(late) < 3:
        return float("nan")
    t = [p[0] for p in late]
    y = [p[1] for p in late]
    mt, my = sum(t) / len(t), sum(y) / len(y)
    sxx = sum((ti - mt) ** 2 for ti in t)
    if sxx == 0:
        return float("nan")
    slope = sum((ti - mt) * (yi - my) for ti, yi in zip(t, y)) / sxx
    return slope / dp * 100.0


def load_points() -> list[dict]:
    points = []
    for path in sorted(RAW.glob("settling_*.json")):
        record = json.loads(path.read_text())
        record["source_file"] = f"raw/{path.name}"
        points.append(record)
    return points


def row(point: dict, case: dict) -> dict:
    dp = point["analytic_dp_Pa"]
    checks = {
        "window": case["t_over_t_sigma_max"] >= MIN_WINDOW,
        "dt_not_clipped": not case.get("dt_was_clipped", False),
        "in_band": abs(case["tail40_rel_error_pct"]) < GATE_PCT,
        "admissible": case["vmax_over_vref_late_max"] < SPURIOUS_CURRENT_MAX,
        "decayed": case["decay_ratio"] < DECAY_MAX,
    }
    return {
        "label": (f"settled <p_core>, {case['case']}, "
                  f"dx = {point['dx_m'] * 1e3:.1f} mm, "
                  f"c0 = {point['sound_speed']:g} m/s"),
        "source": point["source_file"],
        "config": {"dx_m": point["dx_m"], "sound_speed": point["sound_speed"],
                   "closure": case["config"].get("closure", False),
                   "donors": case["config"].get("donors"),
                   "layers": case["config"].get("layers")},
        "window_over_t_sigma": case["t_over_t_sigma_max"],
        "steps": case["steps"],
        "n_particles": case["n_particles"],
        "value_Pa": case["tail40_Pa"],
        "rel_error_pct": case["tail40_rel_error_pct"],
        "rel_error_tail25_pct": case["tail25_rel_error_pct"],
        "final_snapshot_rel_error_pct": case["final_rel_error_pct"],
        "vmax_over_vref_early": case["vmax_over_vref_early20"],
        "vmax_over_vref_late": case["vmax_over_vref_late_max"],
        "decay_ratio": case["decay_ratio"],
        "within_15pct": abs(case["tail40_rel_error_pct"]) < GATE_PCT,
        "within_3pct": abs(case["tail40_rel_error_pct"]) < 3.0,
        "admissible": case["vmax_over_vref_late_max"] < SPURIOUS_CURRENT_MAX,
        "spurious_current_over_limit": case["vmax_over_vref_late_max"] / SPURIOUS_CURRENT_MAX,
        "checks": checks,
        "vmax_quartiles": _quartile_vmax(case["vmax_trace_over_t_sigma"]),
        "p_core_drift_pct_per_t_sigma": _drift_percent_per_t_sigma(
            case["p_core_trace_over_t_sigma"], dp),
        "analytic_dp_Pa": dp,
        "vmax_trace": case["vmax_trace_over_t_sigma"],
        "p_core_trace": case["p_core_trace_over_t_sigma"],
    }


def at_point(rows: list[dict], dx: float, c0: float, case: str) -> dict | None:
    for r in rows:
        c = r["config"]
        if (c["dx_m"], c["sound_speed"]) == (dx, c0):
            if case == CANDIDATE and c["closure"] and c["donors"] == "interior" \
                    and c["layers"] == 2:
                return r
            if case == "closure_off" and not c["closure"]:
                return r
    return None


def main() -> None:
    points = load_points()
    rows: list[dict] = []
    for point in points:
        for case in point["cases"]:
            rows.append(row(point, case))

    # ---- P1: provenance of every graded record
    provenance_fail = [r["label"] for r in rows
                       if not (r["checks"]["window"] and r["checks"]["dt_not_clipped"])]

    # ---- P2: decay at the point the short ladder found inadmissible
    decay_rows = [r for r in rows if r["checks"]["window"]]
    decay_fail = [r["label"] for r in decay_rows if not r["checks"]["decayed"]]

    # ---- P3/P4: the four acceptance points, candidate variant only
    acceptance = []
    for dx, c0 in ACCEPTANCE_POINTS:
        r = at_point(rows, dx, c0, CANDIDATE)
        acceptance.append({
            "dx_m": dx, "sound_speed": c0,
            "recorded": r is not None,
            "value_Pa": None if r is None else r["value_Pa"],
            "rel_error_pct": None if r is None else r["rel_error_pct"],
            "vmax_over_vref_late": None if r is None else r["vmax_over_vref_late"],
            "in_band": None if r is None else r["within_15pct"],
            "admissible": None if r is None else r["admissible"],
            "window_over_t_sigma": None if r is None else r["window_over_t_sigma"],
        })
    missing = [(a["dx_m"], a["sound_speed"]) for a in acceptance if not a["recorded"]]
    band_fail = [a for a in acceptance if a["recorded"] and not a["in_band"]]
    adm_fail = [a for a in acceptance if a["recorded"] and not a["admissible"]]

    # ---- controls: does the closure move the answer at the same point?
    controls = []
    for point in points:
        cand = at_point(rows, point["dx_m"], point["sound_speed"], CANDIDATE)
        off = at_point(rows, point["dx_m"], point["sound_speed"], "closure_off")
        if cand and off:
            controls.append({
                "dx_m": point["dx_m"], "sound_speed": point["sound_speed"],
                "closure_off_rel_error_pct": off["rel_error_pct"],
                "closure_on_rel_error_pct": cand["rel_error_pct"],
                "closure_delta_pct_points": cand["rel_error_pct"] - off["rel_error_pct"],
                "closure_off_vmax": off["vmax_over_vref_late"],
                "closure_on_vmax": cand["vmax_over_vref_late"],
            })

    defects = []
    if provenance_fail:
        defects.append({"id": "provenance", "rows": provenance_fail,
                        "why": (f"record spans < {MIN_WINDOW} t_sigma or dt was clipped by "
                                "max_steps; it may not grade the settling question")})
    if decay_fail:
        defects.append({"id": "spurious-current-does-not-decay", "rows": decay_fail,
                        "why": (f"v_max(late)/v_max(early) >= {DECAY_MAX}: the parasitic "
                                "mode is persistent, so a longer window does not make the "
                                "read-out a static measurement -- a scheme defect, not a "
                                "measurement defect")})
    if missing:
        defects.append({"id": "acceptance-point-not-recorded",
                        "rows": [f"dx={dx * 1e3:.1f} mm, c0={c0:g} m/s" for dx, c0 in missing],
                        "why": "the acceptance ladder is incomplete"})
    if adm_fail:
        defects.append({"id": "inadmissible-at-acceptance-point",
                        "rows": [f"dx={a['dx_m'] * 1e3:.1f} mm, c0={a['sound_speed']:g} m/s "
                                 f"(v_max/v_ref = {a['vmax_over_vref_late']:.3f})"
                                 for a in adm_fail],
                        "why": "spurious current above the Adami limit on the late window"})
    if band_fail:
        defects.append({"id": "out-of-band-at-acceptance-point",
                        "rows": [f"dx={a['dx_m'] * 1e3:.1f} mm, c0={a['sound_speed']:g} m/s "
                                 f"({a['rel_error_pct']:+.1f} %)" for a in band_fail],
                        "why": f"|error| >= {GATE_PCT} %"})

    complete = not missing
    clean = complete and not provenance_fail and not adm_fail and not band_fail
    if clean:
        status = "SETTLED"
        reason = ("every acceptance point is inside the +/-15 % band and admissible on a "
                  f">= {MIN_WINDOW} t_sigma window: the closure read-out is a static Laplace "
                  "measurement at all four points")
    elif not complete:
        status = "INCOMPLETE"
        reason = f"{len(missing)} of {len(ACCEPTANCE_POINTS)} acceptance points not recorded"
        failed_recorded = [a for a in acceptance
                           if a["recorded"] and not (a["in_band"] and a["admissible"])]
        if failed_recorded:
            reason += ("; the criterion is conjunctive and is already unmet at "
                       f"{len(failed_recorded)} of {len(acceptance) - len(missing)} recorded "
                       "points, so the missing point(s) can only add another failure")
    else:
        status = "WITHHELD"
        reason = ("the long window removes the breathing objection but not every acceptance "
                  "point is both in-band and admissible: "
                  + "; ".join(f"dx={a['dx_m'] * 1e3:.1f} mm c0={a['sound_speed']:g} "
                              f"({a['rel_error_pct']:+.1f} %, v_max/v_ref="
                              f"{a['vmax_over_vref_late']:.3f})"
                              for a in acceptance if not (a["in_band"] and a["admissible"])))

    matrix = {
        "benchmark": "laplace_pressure_settling",
        "question": ("is the free-surface-closure Laplace read-out a static measurement, i.e. "
                     "does the spurious current of the breathing drop decay so that an "
                     "admissible, in-band read-out exists at every acceptance point?"),
        "reference": {"sigma_N_per_m": 7.2, "radius_m": 0.02,
                      "analytic_dp_Pa": points[0]["analytic_dp_Pa"] if points else None},
        "rules": {"band_pct": GATE_PCT,
                  "admissibility": f"v_max/v_ref < {SPURIOUS_CURRENT_MAX} on the late window",
                  "spurious_current_limit": SPURIOUS_CURRENT_MAX,
                  "min_window_over_t_sigma": MIN_WINDOW,
                  "decay_ratio_max": DECAY_MAX,
                  "candidate_case": CANDIDATE,
                  "acceptance_points": [{"dx_m": dx, "sound_speed": c0}
                                        for dx, c0 in ACCEPTANCE_POINTS]},
        "readouts": rows,
        "acceptance": acceptance,
        "controls": controls,
        "defects": defects,
        "verdict": {"status": status, "reason": reason,
                    "capability_grade": ("the Laplace pressure jump of this SPH scheme can be "
                                        "graded REAL only when status == SETTLED"),
                    "not_capability_evidence": [
                        r["label"] for r in rows if r["config"]["closure"] is False]},
    }
    OUT.write_text(json.dumps(matrix, indent=2) + "\n")

    print(f"{len(rows)} read-outs from {len(points)} (dx, c0) points")
    for r in rows:
        print(f"  {r['label']:64s} {r['rel_error_pct']:+7.1f} %  "
              f"band={'Y' if r['within_15pct'] else 'n'}  "
              f"adm={'Y' if r['admissible'] else 'n'}  "
              f"decay={r['decay_ratio']:.3f}  window={r['window_over_t_sigma']:.2f}  "
              f"vmaxQ={[round(q, 2) for q in r['vmax_quartiles']]}")
    print("verdict:", status, "-", reason)
    print("defects:", len(defects))
    print("wrote", OUT)


if __name__ == "__main__":
    main()