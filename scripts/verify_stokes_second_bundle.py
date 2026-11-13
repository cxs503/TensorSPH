#!/usr/bin/env python3
"""Fail-closed self-consistency check of a published Stokes-second bundle.

The bundle published by ``benchmarks/stokes_second/run.py`` is the evidence, so
it has to be checkable *as shipped* — not re-derived by hand in an audit.  This
module recomputes what the bundle already contains and refuses to bless it when
the pieces disagree:

V1  every artefact listed in ``MANIFEST.json`` exists, and its sha256/bytes match;
V2  every regular file in the bundle directory is listed in ``MANIFEST.json``
    (an unlisted file is an un-hashed artefact);
V3  every figure declared by ``results_ladder.json`` exists on disk and is
    hashed in ``MANIFEST.json`` with the same sha256 (no dangling evidence);
V4  the ladder is complete (``convergence.complete`` and ``resolutions`` equal
    ``required_resolutions``), ``G6_not_growing`` is consistent with the L2
    series, and ``all_gates`` is consistent with the per-run gates;
V5  the graded metrics of every rung are *recomputable from that rung's own
    persisted rows* (rms and worst-row relative error over ``y <= delta``,
    least-squares decay rate) — the numbers in ``metrics`` must be the numbers
    the rows imply;
V6  ``benchmark.json`` agrees with the ladder: same primary value, same rows,
    same verdicts, same repro command;
V8  the manifest's ``runs`` map describes every ``results_*.json`` it hashes,
    and the entry for ``results_ladder.json`` quotes that file's own mode and
    repro command (a manifest written by a later ``--quick`` run over the same
    directory used to describe the ladder with the quick command);
V7  the graded runs were made with the same case configuration (a ladder of
    different cases is not a convergence study).

Usage::

    PYTHONPATH=.:src python scripts/verify_stokes_second_bundle.py [--out DIR]

Exit code 0 = all checks pass, 1 = at least one problem (listed on stdout),
2 = the bundle directory itself is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "benchmarks" / "stokes_second" / "out"

# Metrics that are a pure function of a rung's own persisted rows.
FINGERPRINT = ("viscosity", "omega", "u0", "sound_speed", "h_factor",
               "wall_friction", "delta")


class BundleError(AssertionError):
    """Raised when a published bundle contradicts itself."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rms(values) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def recompute_run(run: dict) -> dict:
    """Recompute the graded metrics from the run's own rows."""
    rows = run["rows"]
    delta = run["delta"]
    layer = [r for r in rows if r["y"] <= delta]
    fit_pts = [r for r in layer if r["amplitude"] > 0]
    slope = float("nan")
    if len(fit_pts) >= 3:
        n = len(fit_pts)
        xs = [r["y"] for r in fit_pts]
        ys = [math.log(r["amplitude"]) for r in fit_pts]
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx else float("nan")
    return {
        "n_rows_stokes_layer": len(layer),
        "amplitude_l2_half_space_pct":
            100.0 * _rms([r["rel_err_half_space"] for r in layer]),
        "amplitude_l2_finite_depth_pct":
            100.0 * _rms([r["rel_err_finite_depth"] for r in layer]),
        "amplitude_worst_row_half_space_pct": 100.0 * max(
            (abs(r["rel_err_half_space"]) for r in layer), default=float("nan")),
        "decay_rate_measured": slope,
        "delta_ratio": (-1.0 / slope) / delta if slope else float("nan"),
    }


def _close(a: float, b: float, rel: float = 5e-5) -> bool:
    if a is None or b is None:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    if b == 0.0:
        return abs(a) <= 1e-9
    return abs(a - b) <= rel * abs(b)


def verify_bundle(out_dir: Path = DEFAULT_OUT) -> dict:
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise BundleError(f"no bundle directory at {out_dir}")
    manifest_path = out_dir / "MANIFEST.json"
    ladder_path = out_dir / "results_ladder.json"
    bench_path = out_dir / "benchmark.json"
    for p in (manifest_path, ladder_path, bench_path):
        if not p.is_file():
            raise BundleError(f"bundle is missing {p.name}")

    manifest = json.loads(manifest_path.read_text())
    ladder = json.loads(ladder_path.read_text())
    bench = json.loads(bench_path.read_text())
    problems: list[str] = []

    # V1  hashes of everything listed
    artifacts = manifest.get("artifacts") or {}
    for name, meta in sorted(artifacts.items()):
        path = out_dir / name
        if not path.is_file():
            problems.append(f"V1 {name}: listed in MANIFEST but not on disk")
            continue
        if sha256(path) != meta.get("sha256"):
            problems.append(f"V1 {name}: sha256 differs from MANIFEST")
        if path.stat().st_size != meta.get("bytes"):
            problems.append(f"V1 {name}: byte count differs from MANIFEST")

    # V2  nothing un-hashed in the bundle
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json" \
                and path.name not in artifacts:
            problems.append(f"V2 {path.name}: present but not hashed in MANIFEST")

    # V3  declared figures exist and are hashed
    for entry in ladder.get("figures") or []:
        path = Path(entry)
        path = path if path.is_absolute() else REPO / path
        if not path.is_file():
            problems.append(f"V3 {entry}: declared by results_ladder.json but "
                            f"missing on disk")
            continue
        meta = artifacts.get(path.name)
        if meta is None:
            problems.append(f"V3 {path.name}: figure not hashed in MANIFEST")
        elif sha256(path) != meta.get("sha256"):
            problems.append(f"V3 {path.name}: figure sha256 differs from MANIFEST")

    # V4  the ladder is a ladder
    runs = ladder.get("runs") or []
    conv = ladder.get("convergence") or {}
    res = sorted(int(r["resolution_delta_over_dx"]) for r in runs)
    required = sorted(int(r) for r in conv.get("required_resolutions") or [])
    if not conv.get("complete"):
        problems.append("V4 convergence.complete is false")
    if required and res != required:
        problems.append(f"V4 resolutions {res} != required {required}")
    l2 = [r["metrics"]["amplitude_l2_half_space_pct"] for r in
          sorted(runs, key=lambda r: r["resolution_delta_over_dx"])]
    if len(l2) >= 2 and l2[-1] > l2[0] + 0.5:
        problems.append(f"V4 G6 would fail: L2 grew {l2[0]} -> {l2[-1]} pp")
    per_run_ok = all(all((r.get("gates") or {}).values()) for r in runs)
    if bool(conv.get("all_gates")) != bool(conv.get("complete")
                                           and conv.get("G6_not_growing")
                                           and per_run_ok):
        problems.append("V4 convergence.all_gates disagrees with its own inputs")

    # V5  metrics recomputable from each rung's rows
    for run in runs:
        got, ref = run["metrics"], recompute_run(run)
        for key, want in ref.items():
            have = got.get(key)
            if isinstance(want, int) and not isinstance(want, bool):
                if have != want:
                    problems.append(f"V5 dx={run['resolution_delta_over_dx']} "
                                    f"{key}: {have} != recomputed {want}")
            elif not _close(float(have), float(want)):
                problems.append(f"V5 dx={run['resolution_delta_over_dx']} "
                                f"{key}: {have} != recomputed {want:.6g}")

    # V6  benchmark.json is the same evidence, summarised
    finest = max(runs, key=lambda r: r["resolution_delta_over_dx"])
    if not _close(bench.get("primary_value_pct"),
                  finest["metrics"]["amplitude_l2_half_space_pct"]):
        problems.append("V6 benchmark.json primary_value_pct != finest metrics")
    if int(bench.get("resolution_delta_over_dx", -1)) != \
            int(finest["resolution_delta_over_dx"]):
        problems.append("V6 benchmark.json graded a different rung than the finest")
    b_rows = {r["quantity"]: r for r in bench.get("rows_of_finest_run") or []}
    for name, row in b_rows.items():
        got = row["got"]
        if name in finest["metrics"] and not _close(got, finest["metrics"][name]):
            problems.append(f"V6 {name}: benchmark row {got} != ladder "
                            f"{finest['metrics'][name]}")
        if bool(row["pass"]) != bool(abs(got - row["reference"]) <=
                                     row["tolerance"] if row["reference"] == 0.0
                                     else abs(got / row["reference"] - 1.0)
                                     * 100.0 <= row["tolerance"]):
            problems.append(f"V6 {name}: pass flag contradicts reference/tolerance")
    if bench.get("repro_cmd") != ladder.get("repro_cmd"):
        problems.append("V6 benchmark repro_cmd != results_ladder repro_cmd")

    # V8  the manifest stays truthful about which run produced what
    runs_map = manifest.get("runs") or {}
    results_files = [n for n in artifacts if n.startswith("results_")
                     and n.endswith(".json")]
    for name in results_files:
        if name not in runs_map:
            problems.append(f"V8 {name}: hashed but absent from MANIFEST.runs")
    entry = runs_map.get("results_ladder.json")
    if entry is None:
        problems.append("V8 MANIFEST.runs has no entry for results_ladder.json")
    else:
        if entry.get("mode") != ladder.get("mode"):
            problems.append(f"V8 MANIFEST.runs ladder mode {entry.get('mode')!r} "
                            f"!= {ladder.get('mode')!r}")
        if entry.get("command") != ladder.get("repro_cmd"):
            problems.append("V8 MANIFEST.runs ladder command != the ladder's own "
                            "repro_cmd (stale header from another mode?)")
    if manifest.get("mode") != ladder.get("mode"):
        problems.append(f"V8 MANIFEST.mode {manifest.get('mode')!r} does not "
                        f"describe the published ladder ({ladder.get('mode')!r})")

    # V7  same case along the ladder
    if runs:
        base = {k: runs[0].get(k) for k in FINGERPRINT}
        for run in runs[1:]:
            other = {k: run.get(k) for k in FINGERPRINT}
            if not all(_close(float(other[k]), float(base[k]), rel=1e-4)
                       for k in FINGERPRINT):
                problems.append("V7 rung dx=%s has a different case "
                                "configuration %s != %s"
                                % (run["resolution_delta_over_dx"], other, base))

    report = {
        "bundle": str(os.path.relpath(out_dir, REPO)),
        "resolutions": res,
        "artifacts": len(artifacts),
        "figures": ladder.get("figures") or [],
        "primary_value_pct": bench.get("primary_value_pct"),
        "checks": ["V1_hashes", "V2_no_unlisted", "V3_figures", "V4_ladder",
                   "V5_metrics_recomputed", "V6_benchmark", "V7_same_case",
                   "V8_manifest_runs"],
        "ok": not problems,
        "problems": problems,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    try:
        report = verify_bundle(args.out)
    except BundleError as exc:
        print(json.dumps({"ok": False, "problems": [str(exc)]}, indent=1))
        return 2
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
