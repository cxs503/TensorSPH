"""Cross validation of the TensorSPH DTW regression-baseline protocol.

Reference practice: ``Xiangyu-Hu/SPHinXsys`` ``src/shared/regression_test/``
(DTW distance, generation baselines, ``runtimes.dat`` bookkeeping, the
``1.01 x`` acceptance band and the four-generation convergence rule).  The
sources were read from a local checkout of the upstream repository
(``SPHinXsim/sphinxsys`` @ 5241a89); nothing here is a hand-copied number.

Three arms, all predeclared before the numbers were produced:

A. **Algorithm cross validation.** Our faithful port, the canonical DP and an
   independent shortest-path implementation (``scipy`` dijkstra) are applied to
   the same series pairs.  Criterion: relative deviation < 15% (owner standard);
   expectation for two correct implementations of the same object is < 1e-9.
B. **Protocol arm.** Six generations of the TensorSPH dam-break case with a
   small time-step jitter (0-1%), folded in with ``threshold = 0.01``; then two
   blind tests on copies of the database: a *legit* new run (dt scale 1.002,
   must pass) and a *negative control* (viscosity x50, must be rejected).
   Criterion: the gate must accept every legit generation and reject the
   negative control -- a gate that never fires proves nothing.
C. **Band quirk quantification.** The reference DP zero-initialises the band, so
   an alignment that leaves the band can pick a zero-valued neighbour.  Report
   faithful vs canonical distance on shifted series to size that effect.

Sensitivity (reported, not gated): DTW window {5, 10, 20} and convergence
threshold {1e-2, 1e-3, 1e-4}, recomputed from the same stored series.

Usage:
    PYTHONPATH=src python benchmarks/dtw_regression_protocol/run.py --quick
    PYTHONPATH=src python benchmarks/dtw_regression_protocol/run.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from tensorsph.dam_break import DamBreakConfig
from tensorsph.regression_test import (
    DTWRegressionFailure,
    band_edge_shortcuts,
    dtw_distance,
    dtw_distance_shortest_path,
    dtw_distance_strict,
    run_generation,
)
from tensorsph.regression_test.observe import QUANTITIES, dam_break_observations

XVAL_TOLERANCE = 0.15        # owner standard: >15% is not acceptable
PROTOCOL_THRESHOLD = 0.01    # absolute DTW-distance revision tolerated per generation
WINDOW = 5                   # reference hard-coded half-band width
N_SNAPSHOTS = 20
END_TIME = 0.20              # s
DX = 0.04                    # m
DT_BASE = 1.0e-4             # s
GENERATION_DT_SCALES = (1.0, 1.005, 0.995, 1.010, 0.990, 1.002, 1.004, 0.998)
LEGIT_TEST_SCALE = 1.002
NEGATIVE_CONTROL_VISCOSITY_SCALE = 50.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_config(dt_scale: float = 1.0, viscosity_scale: float = 1.0) -> DamBreakConfig:
    cfg = DamBreakConfig(dx=DX)
    cfg.dt = DT_BASE * dt_scale
    cfg.steps = int(round(END_TIME / cfg.dt))
    cfg.viscosity = cfg.viscosity * viscosity_scale
    # snapshot sampling resolution: the case records snapshots on output steps,
    # so a coarse output interval would make the *sampling* offset (not the
    # physics) dominate the DTW distance between generations.
    cfg.output_every = 10
    return cfg


def relative_deviation(reference: float, candidate: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return abs(candidate - reference) / abs(reference)


def synthetic_families() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Series pairs that exercise the alignment paths the protocol will see."""
    rng = np.random.default_rng(20261113)
    t = np.linspace(0.0, 1.0, 60)
    base = np.sin(2.0 * np.pi * t) + 0.3 * t
    families = {
        "identical": (base, base.copy()),
        "constant_offset": (base, base + 0.05),
        "shift_2": (base, np.roll(base, 2)),
        "shift_8_out_of_band": (base, np.roll(base, 8)),
        "stretch": (base, np.interp(np.linspace(0, 1, 66), t, base)),
        "noise_1pct": (base, base + 0.01 * rng.standard_normal(base.size)),
        "length_plus_8": (base, np.concatenate([base, base[-8:]])),
        "linear_ramp": (t, t ** 2),
    }
    return families


def vector_series() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, 40)
    a = np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    b = np.column_stack([np.cos(2 * np.pi * t + 0.1), np.sin(2 * np.pi * t + 0.1)])
    return a, b


# ---------------------------------------------------------------------------
# arm A: algorithm cross validation
# ---------------------------------------------------------------------------
def arm_algorithm_xval() -> dict:
    rows = []
    families = synthetic_families()
    families["vector_2d"] = vector_series()
    for name, (series_a, series_b) in families.items():
        faithful = dtw_distance(series_a, series_b, window=WINDOW)
        strict = dtw_distance_strict(series_a, series_b, window=WINDOW)
        shortest = dtw_distance_shortest_path(series_a, series_b, window=WINDOW)
        rows.append({
            "family": name,
            "lengths": [len(series_a), len(series_b)],
            "dtw_faithful": faithful,
            "dtw_strict": strict,
            "dtw_shortest_path": shortest,
            "deviation_faithful_vs_strict": relative_deviation(strict, faithful),
            "deviation_strict_vs_shortest_path": relative_deviation(shortest, strict),
        })

    worst = max(row["deviation_strict_vs_shortest_path"] for row in rows)
    return {
        "criterion": f"relative deviation < {100 * XVAL_TOLERANCE:g}%",
        "worst_deviation": worst,
        "passed": bool(worst < XVAL_TOLERANCE),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# arm B: protocol generations
# ---------------------------------------------------------------------------
def observe(dt_scale: float, viscosity_scale: float = 1.0):
    cfg = make_config(dt_scale=dt_scale, viscosity_scale=viscosity_scale)
    # stop just short of the end so that every request is certainly captured
    # before the last integration step
    times = np.linspace(END_TIME / N_SNAPSHOTS, 0.98 * END_TIME, N_SNAPSHOTS)
    return dam_break_observations(cfg, times)


def arm_protocol(artifact_dir: Path) -> dict:
    workdir = artifact_dir / "protocol_database"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    generations = []
    observations = {}
    for index, dt_scale in enumerate(GENERATION_DT_SCALES):
        observed = observe(dt_scale)
        observations[f"g{index}"] = {
            "dt_scale": dt_scale,
            "dt": observed.dt,
            "steps": observed.steps,
            "runtime": observed.runtime,
            "mass_drift": observed.mass_drift,
            "escaped": observed.escaped,
            "series": observed.values.tolist(),
            "times": observed.times,
        }
        report = run_generation(
            workdir, observed.as_rows(), threshold=PROTOCOL_THRESHOLD,
            generate=True, dynamics_name="DamBreak", quantity_names=QUANTITIES,
            window=WINDOW, length_guard=0.2)
        generations.append({
            "index": index,
            "dt_scale": dt_scale,
            "steps": observed.steps,
            "runtime": observed.runtime,
            "report": report,
        })

    # -- blind tests on copies so that the database stays intact -----------
    legit_dir = artifact_dir / "protocol_check_legit"
    negative_dir = artifact_dir / "protocol_check_negative"
    for target in (legit_dir, negative_dir):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(workdir, target)

    legit = observe(LEGIT_TEST_SCALE)
    try:
        legit_report = run_generation(
            legit_dir, legit.as_rows(), threshold=PROTOCOL_THRESHOLD,
            generate=False, dynamics_name="DamBreak", quantity_names=QUANTITIES,
            window=WINDOW, length_guard=0.2)
        legit_outcome = {"rejected": False, "report": legit_report}
    except DTWRegressionFailure as failure:
        legit_outcome = {"rejected": True, "reason": str(failure)}

    negative_observed = observe(1.0, viscosity_scale=NEGATIVE_CONTROL_VISCOSITY_SCALE)
    try:
        negative_report = run_generation(
            negative_dir, negative_observed.as_rows(), threshold=PROTOCOL_THRESHOLD,
            generate=False, dynamics_name="DamBreak", quantity_names=QUANTITIES,
            window=WINDOW, length_guard=0.2)
        negative_outcome = {"rejected": False, "report": negative_report}
    except DTWRegressionFailure as failure:
        negative_outcome = {"rejected": True, "reason": str(failure)}

    converged_at = next(
        (gen["index"] for gen in generations
         if gen["report"].get("converged") == "true"), None)
    return {
        "criterion": "legit generation passes AND negative control is rejected",
        "passed": bool(not legit_outcome["rejected"] and negative_outcome["rejected"]),
        "threshold": PROTOCOL_THRESHOLD,
        "window": WINDOW,
        "generations": generations,
        "converged_at_generation": converged_at,
        "final_state": generations[-1]["report"]["state_after"],
        "legit_test": legit_outcome,
        "negative_control": negative_outcome,
        "observations": observations,
        "negative_control_series": negative_observed.values.tolist(),
    }


# ---------------------------------------------------------------------------
# arm C + sensitivity
# ---------------------------------------------------------------------------
def arm_band_quirk() -> dict:
    rows = []
    for name, (series_a, series_b) in synthetic_families().items():
        diagnostic = band_edge_shortcuts(series_a, series_b, window=WINDOW)
        rows.append({"family": name, **diagnostic})
    return {
        "note": ("faithful = reference recurrence with zero-initialised band; "
                 "strict = canonical constrained optimum"),
        "rows": rows,
        "worst_relative_underestimate": max(
            (row["relative_underestimate"] for row in rows), default=0.0),
    }


def arm_sensitivity(protocol: dict) -> dict:
    """Recompute distances from the stored series; no extra simulations."""
    observations = protocol["observations"]
    order = [f"g{index}" for index in range(len(GENERATION_DT_SCALES))]
    pairs = []
    for index in range(1, len(order)):
        earlier = np.asarray(observations[order[index - 1]]["series"])
        later = np.asarray(observations[order[index]]["series"])
        row = {
            "pair": f"{order[index - 1]}->{order[index]}",
            "dt_scale_step": observations[order[index]]["dt_scale"]
                            - observations[order[index - 1]]["dt_scale"],
        }
        for window in (5, 10, 20):
            for quantity_index, quantity in enumerate(QUANTITIES):
                row[f"window{window}_{quantity}"] = dtw_distance(
                    earlier[:, quantity_index], later[:, quantity_index], window=window)
        # faithful vs canonical on the same real series pair (quirk size on data)
        for quantity_index, quantity in enumerate(QUANTITIES):
            faithful = dtw_distance(earlier[:, quantity_index], later[:, quantity_index],
                                    window=WINDOW)
            strict = dtw_distance_strict(earlier[:, quantity_index], later[:, quantity_index],
                                         window=WINDOW)
            row[f"strict_window{WINDOW}_{quantity}"] = strict
            row[f"deviation_{quantity}"] = relative_deviation(strict, faithful)
        pairs.append(row)

    # threshold sensitivity: how many generations converge for each threshold
    revisions = []
    for generation in protocol["generations"]:
        accepted = np.asarray(generation["report"].get("accepted", [0.0]))
        accepted_new = np.asarray(generation["report"].get("accepted_new", [0.0]))
        revisions.append(np.max(np.abs(accepted_new - accepted)) if accepted.size else 0.0)
    threshold_table = {
        "revision_per_generation": revisions,
        "converged_at": {
            f"{threshold:g}": _converge_index(revisions, threshold)
            for threshold in (1e-2, 1e-3, 1e-4, 1e-5)
        },
    }
    return {"window_table": pairs, "threshold_table": threshold_table}


def _converge_index(revisions, threshold: int | float) -> int | None:
    streak = 0
    for index, revision in enumerate(revisions):
        streak = streak + 1 if revision <= threshold else 0
        if streak >= 4:
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/dtw_regression_protocol")
    parser.add_argument("--quick", action="store_true",
                        help="skip the generation loop (algorithm arms only)")
    arguments = parser.parse_args()

    artifact_dir = Path(arguments.out)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    result = {
        "reference_practice": "Xiangyu-Hu/SPHinXsys src/shared/regression_test/",
        "parameters": {
            "dx_m": DX, "dt_base_s": DT_BASE, "end_time_s": END_TIME,
            "snapshots": N_SNAPSHOTS, "window": WINDOW,
            "length_guard": 0.2, "acceptance_margin": 1.01,
            "repeat_required": 4, "protocol_threshold": PROTOCOL_THRESHOLD,
            "generation_dt_scales": list(GENERATION_DT_SCALES),
            "quantities": list(QUANTITIES),
        },
    }
    result["arm_algorithm_xval"] = arm_algorithm_xval()
    result["arm_band_quirk"] = arm_band_quirk()
    if not arguments.quick:
        result["arm_protocol"] = arm_protocol(artifact_dir)
        result["arm_sensitivity"] = arm_sensitivity(result["arm_protocol"])
    result["elapsed_s"] = time.time() - started

    path = artifact_dir / "xval_report.json"
    path.write_text(json.dumps(result, indent=2, default=float))

    # -- console summary ---------------------------------------------------
    xval = result["arm_algorithm_xval"]
    print(f"arm A  algorithm XVAL: worst deviation {xval['worst_deviation']:.3e} "
          f"(criterion <{XVAL_TOLERANCE:g}) -> {'PASS' if xval['passed'] else 'FAIL'}")
    for row in xval["rows"]:
        print(f"   {row['family']:<22} faithful={row['dtw_faithful']:.6e} "
              f"strict={row['dtw_strict']:.6e} sp={row['dtw_shortest_path']:.6e} "
              f"dev={row['deviation_strict_vs_shortest_path']:.2e}")
    quirk = result["arm_band_quirk"]
    print(f"arm C  band quirk: worst relative underestimate "
          f"{quirk['worst_relative_underestimate']:.2%}")
    if not arguments.quick:
        protocol = result["arm_protocol"]
        for generation in protocol["generations"]:
            report = generation["report"]
            accepted = np.asarray(report.get("accepted_new") or report.get("accepted"))
            print(f"   gen {generation['index']} dt_scale={generation['dt_scale']:.4f} "
                  f"accepted={np.array2string(accepted, precision=6)} "
                  f"converged={report.get('converged')} label={report.get('label_for_repeat')}")
        print(f"arm B  protocol: converged at generation "
              f"{protocol['converged_at_generation']}; legit test "
              f"{'REJECTED' if protocol['legit_test']['rejected'] else 'passed'}; "
              f"negative control "
              f"{'rejected' if protocol['negative_control']['rejected'] else 'PASSED (BAD)'}")
        print(f"   -> {'PASS' if protocol['passed'] else 'FAIL'}")
    print(f"artifacts: {path}  ({result['elapsed_s']:.1f} s)")

    ok = xval["passed"] and (arguments.quick or result["arm_protocol"]["passed"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())