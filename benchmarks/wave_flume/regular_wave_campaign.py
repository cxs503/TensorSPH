"""Measure the forced harmonic and incident/reflected waves from solver records."""

import argparse
import gzip
import json
import math
from pathlib import Path

from tensorsph.wave_flume import separate_incident_reflected, stability_verdict
from tensorsph.wave_flume_grade import (
    grade_regular_record,
    harmonic_amplitude,
    solver_record_integrity,
)


def assess(record):
    if record.get("failed"):
        return {"passed": False, "error": record["error"]}
    t, series, xs = record["times"], record["probe_eta"], record["probe_x"]
    spec = record["spec"]
    diag = record["diagnostics"]
    c0 = record["config"]["sound_speed"]
    regime = stability_verdict(
        {"max_speed": diag["max_speed"], "max_compression": diag["max_density_compression"]}, c0
    )
    requested = record.get("requested_duration_s", 5.)
    base = dict(
        requested_s=requested,
        first_regime_exit=diag.get("first_regime_exit"),
        simulated_s=diag["simulated_time"],
        requested_clock_reached=diag["simulated_time"] >= requested - 1e-9,
        steps=diag["steps_taken"],
        escaped=diag["escaped"],
        regime=regime,
        record_integrity=solver_record_integrity(record),
    )
    # Secondary common two-cycle window for sensitivity; it does not confer
    # the registered 3–5 s verdict when a run exits just before the final time.
    first_exit = diag.get("first_regime_exit")
    if base["record_integrity"]["passed"] and t[-1] >= 4.9 and (
        regime["stable"] or (first_exit is not None and first_exit["t_s"] > 4.9)
    ):
        base["secondary_2p9_to_4p9_s_height_m"] = [
            harmonic_amplitude(t, eta, spec["omega"], 2.9, 4.9)["height"] for eta in series
        ]
    if t[-1] < 5.0 - 1e-9 or not regime["stable"]:
        return {
            **base,
            "passed": False,
            "height_assessed": False,
            "covers_window": bool(t[-1] >= 5.0 - 1e-9),
        }
    result = grade_regular_record(record, start=3.0, end=5.0)
    if result["failed"] or any("error" in g for g in result.get("gauges", [])):
        return {**base, "passed": False, "covers_window": True, "height_assessed": False,
                "error": result.get("error", "unusable gauge")}
    omega, k = spec["omega"], spec["k"]
    fits = [harmonic_amplitude(t, eta, omega, 3.0, 5.0) for eta in series]
    split = separate_incident_reflected(t, series, xs, omega, k, 3.0, 5.0)
    gauges = []
    for x, fit, eta in zip(xs, fits, series):
        phase_lag = fits[0]["phase"] - fit["phase"]
        phase_error = math.remainder(phase_lag - k * (x - xs[0]), 2 * math.pi)
        heights = [
            harmonic_amplitude(t, eta, omega, start, start + 1.0)["height"] for start in (3.0, 4.0)
        ]
        gauges.append(
            dict(
                x_m=x,
                height_m=fit["height"],
                harmonic_residual=fit["residual_fraction"],
                phase_error_deg=math.degrees(phase_error),
                cycle_heights_m=heights,
            )
        )
    return dict(
        **base,
        covers_window=True,
        height_assessed=True,
        target_height_m=spec["height"],
        window_s=[3.0, 5.0],
        passed=result["passed"],
        gauges=gauges,
        height_error_max_pct=100 * result["worst_error_vs_target"],
        incident_height_m=2 * split["incident_amplitude"],
        reflection_ratio=split["ratio"],
        split_residual=split["relative_residual"],
        max_density_compression=record["diagnostics"]["max_density_compression"],
        peak_mach=record["diagnostics"]["max_speed"] / record["config"]["sound_speed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for path in args.records:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            results[path.name.split(".")[0]] = assess(json.load(f))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
