"""Assemble the wave-tank case results into the campaign verdict matrix.

Reads the per-case JSON produced by ``run.py`` (plus the cross-solver comparison
and the pytest record) and writes ``runs/wave_tank/verdict.json`` together with a
markdown table on stdout. Every number in the report's verdict table comes from
here, so nothing is transcribed by hand.

    python benchmarks/wave_tank/summarise.py
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from tensorsph.wave_tank import detrended_reflection, spatial_wave_number  # noqa: E402

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "wave_tank")
CASES = ("hydrostatic", "dispersion_kh0.5", "dispersion_kh1", "dispersion_kh2",
         "dispersion_kh3", "decay10", "reflection", "reflection_control",
         "ladder_dp0.020", "ladder_dp0.015", "ladder_dp0.010", "ladder_dp0.0075",
         "csweep_c5", "csweep_c20", "piston_transfer", "solitary")
LADDER = (("0.020", "ladder_dp0.020"), ("0.015", "ladder_dp0.015"),
          ("0.010", "ladder_dp0.010"), ("0.0075", "ladder_dp0.0075"))
C0_SWEEP = (("c5", "csweep_c5"), ("c10", "dispersion_kh3"), ("c20", "csweep_c20"))


def load(name: str) -> dict | None:
    path = os.path.join(RUNS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def case_row(payload: dict) -> dict:
    config = payload["config"]
    analysis = payload.get("analysis", {})
    row = {
        "case": payload["case"],
        "kh": config["kh"],
        "dx_over_depth": config["dx_over_depth"],
        "c0_factor": config["c0_factor"],
        "height_over_depth": config["height_over_depth"],
        "dt": config["dt"],
        "steps": payload["run"]["steps_taken"],
        "particles": payload["run"].get("particles"),
        "wall_seconds": payload["run"]["wall_seconds"],
        "max_courant": payload["run"]["max_courant"],
        "max_compression": payload["run"]["max_compression"],
        "mass_error": payload["run"]["mass_error"],
        "volume_error": payload["run"]["volume_error"],
        "escaped": payload["run"]["escaped"],
        "passed": payload.get("passed"),
    }
    if "probe_fits" in analysis:
        fits = analysis["probe_fits"]
        try:
            analysis["dispersion_spatial"] = spatial_wave_number(
                analysis["probe_x"], [fit["height"] for fit in fits],
                [fit["phase"] for fit in fits], analysis["omega_imposed"],
                payload["config"]["depth"], 9.81)
        except ValueError:
            analysis["dispersion_spatial"] = None
    if "dispersion" in analysis and "reflection" in analysis:
        heights = analysis.get("probe_heights") or []
        positions = analysis.get("probe_x") or []
        measured, per_wavelength = detrended_reflection(
            positions, heights, payload["config"]["wavelength"])
        analysis["reflection"].setdefault("R_detrended", measured)
        analysis["reflection"].setdefault("height_ratio_per_wavelength", per_wavelength)
    if "dispersion" in analysis:
        row.update({
            "celerity_measured": analysis["dispersion"]["celerity_measured"],
            "celerity_theory": analysis["dispersion"]["celerity_theory"],
            "celerity_error": analysis["dispersion"]["celerity_error"],
            "k_measured": analysis["dispersion"]["k_measured"],
            "k_theory": analysis["k_theory"],
            "k_error": analysis["dispersion"]["k_error"],
            "celerity_error_spatial":
                (analysis.get("dispersion_spatial") or {}).get("celerity_error"),
            "decay_ratio_per_wavelength":
                (analysis.get("dispersion_spatial") or {}).get(
                    "decay_ratio_per_wavelength"),
            "reflection": analysis["reflection"]["R"],
            "reflection_detrended": analysis["reflection"].get("R_detrended"),
            "height_ratio_per_wavelength":
                analysis["reflection"].get("height_ratio_per_wavelength"),
            "heights": analysis["probe_heights"],
            "decay_ratio": analysis["decay"]["ratio"],
            "transfer_measured": analysis["transfer_measured"],
            "transfer_theory": analysis["transfer_theory"],
        })
    if "interior_mean_abs_error" in analysis:
        row["hydrostatic_error"] = analysis["interior_mean_abs_error"]
        row["band_mean_error"] = analysis["band_mean_error"]
    if "celerity_error" in analysis and "crest_times" in analysis:
        row.update({
            "celerity_measured": analysis["celerity_measured"],
            "celerity_theory": analysis["celerity_theory"],
            "celerity_error": analysis["celerity_error"],
            "waveform_l2": analysis["waveform"]["l2_error"],
            "crest_error": analysis["waveform"]["crest_error"],
        })
    return row


def convergence_rate(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of log(error) against log(dp/h)."""
    values = [(math.log(x), math.log(y)) for x, y in points if y > 0.0]
    if len(values) < 2:
        return None
    mean_x = sum(x for x, _ in values) / len(values)
    mean_y = sum(y for _, y in values) / len(values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in values)
    denominator = sum((x - mean_x) ** 2 for x, _ in values)
    return numerator / denominator if denominator else None


def build() -> dict:
    cases, missing = {}, []
    for name in CASES:
        payload = load(name)
        if payload is None:
            missing.append(name)
            continue
        cases[name] = case_row(payload)

    criteria: dict[str, dict] = {}

    # (a) hydrostatic
    if "hydrostatic" in cases:
        value = cases["hydrostatic"]["hydrostatic_error"]
        criteria["a"] = {"name": "hydrostatic pressure vs rho g h",
                         "value": value, "threshold": 0.01,
                         "passed": value is not None and value <= 0.01,
                         "case": "hydrostatic"}

    # (b) linear dispersion over kh in [0.5, 3]
    dispersion = {name: row for name, row in cases.items()
                  if name.startswith("dispersion_kh")}
    if dispersion:
        for row in dispersion.values():
            row["celerity_error_gate"] = (row["celerity_error_spatial"]
                                          if row.get("celerity_error_spatial") is not None
                                          else row["celerity_error"])
        worst = max(dispersion.items(), key=lambda item: item[1]["celerity_error_gate"])
        criteria["b"] = {
            "name": "linear dispersion celerity error over kh in [0.5, 3]",
            "values": {name: row["celerity_error_gate"] for name, row in dispersion.items()},
            "values_two_probe": {name: row["celerity_error"]
                                 for name, row in dispersion.items()},
            "worst": worst[1]["celerity_error_gate"], "worst_case": worst[0],
            "threshold": 0.03,
            "passed": all(row["celerity_error_gate"] <= 0.03
                          for row in dispersion.values()),
        }

    # (c) ten wavelengths
    if "decay10" in cases:
        value = cases["decay10"]["decay_ratio"]
        criteria["c"] = {"name": "H/H0 after ten wavelengths",
                         "value": value, "threshold": 0.95,
                         "passed": value is not None and value >= 0.95,
                         "case": "decay10",
                         "wavelengths_between": None}

    # (d) solitary wave
    if "solitary" in cases:
        row = cases["solitary"]
        criteria["d"] = {"name": "solitary celerity / waveform / mass",
                         "celerity_error": row["celerity_error"],
                         "waveform_l2": row["waveform_l2"],
                         "mass_error": row["mass_error"],
                         "thresholds": {"celerity": 0.05, "waveform": 0.08, "mass": 0.01},
                         "passed": (row["celerity_error"] <= 0.05
                                    and row["waveform_l2"] <= 0.08
                                    and row["mass_error"] <= 0.01)}

    # (e) reflection
    if "reflection" in cases:
        row = cases["reflection"]
        control = cases.get("reflection_control")
        criteria["e"] = {"name": "reflection coefficient of the sponge layer",
                         "value": row["reflection"], "threshold": 0.05,
                         "value_detrended": row.get("reflection_detrended"),
                         "control_R": None if control is None else control["reflection"],
                         "passed": row["reflection"] <= 0.05}
        if control is not None:
            criteria["e"]["control_passed"] = control["reflection"] >= 0.8

    # (f) cross-solver
    cross = None
    cross_path = os.path.join(RUNS, "cross_solver.json")
    if os.path.exists(cross_path):
        with open(cross_path, encoding="utf-8") as handle:
            cross = json.load(handle)
    criteria["f"] = cross if cross is not None else {
        "name": "cross-solver adjudication against a second wave-tank code",
        "status": "MISSING", "passed": False}

    # (g) resolution ladder
    ladder = [(float(dx), cases[name]["celerity_error"])
              for dx, name in LADDER if name in cases]
    if len(ladder) >= 3:
        errors = [error for _, error in ladder]
        monotone = all(later < earlier for earlier, later in zip(errors, errors[1:]))
        criteria["g"] = {"name": "dispersion error decreases with dp (no plateau)",
                         "points": [{"dp_over_depth": dx, "celerity_error": error}
                                    for dx, error in ladder],
                         "monotone_decreasing": monotone,
                         "observed_order": convergence_rate(ladder),
                         "passed": monotone}

    # (h) pytest
    pytest_path = os.path.join(RUNS, "pytest_summary.txt")
    if os.path.exists(pytest_path):
        with open(pytest_path, encoding="utf-8") as handle:
            text = handle.read()
        criteria["h"] = {"name": "pytest count and status", "record": text.strip(),
                         "passed": "failed" not in text}

    return {"cases": cases, "criteria": criteria, "missing": missing,
            "all_passed": all(item.get("passed") for item in criteria.values())}


def markdown(summary: dict) -> str:
    lines = ["| case | kh | dp/h | c0/√(gh) | H/h | steps | particles | wall s | "
             "c error | R | H/H0 | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in summary["cases"].items():
        lines.append("| {case} | {kh:.2f} | {dx_over_depth:.4f} | {c0_factor:.0f} | "
                     "{height_over_depth:.2f} | {steps} | {particles} | "
                     "{wall_seconds:.0f} | {celerity_error} | {reflection} | "
                     "{decay_ratio} | {passed} |".format(
                         case=name, kh=row["kh"], dx_over_depth=row["dx_over_depth"],
                         c0_factor=row["c0_factor"],
                         height_over_depth=row["height_over_depth"],
                         steps=row["steps"], particles=row["particles"],
                         wall_seconds=row["wall_seconds"],
                         celerity_error=("%.4f" % row["celerity_error"]
                                         if row.get("celerity_error") is not None else "-"),
                         reflection=("%.4f" % row["reflection"]
                                     if row.get("reflection") is not None else "-"),
                         decay_ratio=("%.4f" % row["decay_ratio"]
                                      if row.get("decay_ratio") is not None else "-"),
                         passed=row["passed"]))
    lines.append("")
    for key in sorted(summary["criteria"]):
        item = summary["criteria"][key]
        lines.append(f"* **({key})** {item.get('name')}: passed={item.get('passed')}")
    return "\n".join(lines)


def main() -> int:
    summary = build()
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "verdict.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(markdown(summary))
    if summary["missing"]:
        print("\nmissing cases:", ", ".join(summary["missing"]))
    print("\nall criteria passed:", summary["all_passed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
