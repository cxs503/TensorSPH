#!/usr/bin/env python
"""WCSPH dam-break / sloshing verification chain -- measurement driver.

Every number the chain reports is produced here and written to
``artifacts/wcsph_chain/*.json``; the gate arithmetic lives in
``tensorsph.wcsph.chain`` and the document is rendered from the artefacts by
``make_doc.py``.  Nothing is retyped by hand.

Sub-commands (each one is an independent, restartable measurement):

    pair                     single particle pair: momentum / energy conservation
    dam2d --dx 0.02          one rung of the 2-D dam-break ladder
    dam3d --dx 0.02          the 3-D dam break
    slosh --freq 1.0         one forcing frequency of the 2-D tank
    tests                    pytest count + coverage of ``tensorsph.wcsph``
    assemble                 combine the artefacts into ``report.json``

The dam-break reference is the independent Lethe transcription of the Martin &
Moyce (1952) experiment, ``benchmarks/dam_break/martin_moyce_1952_crosscheck.json``.
The *bundled* table in the same directory is **not** used: the repository's own
audit (``benchmarks/dam_break/reference_audit.json``) grades it WITHHELD because
its ordinates sit on an exact arithmetic lattice at an exact 0.05 quantum.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.wcsph import chain, kernels                      # noqa: E402
from tensorsph.wcsph.cases import (DamBreakConfig, SloshingConfig, dam_break,
                                   front_at, particle_pair, sloshing)  # noqa: E402
from tensorsph.wcsph.linear_sloshing import LinearSloshing       # noqa: E402

ART = ROOT / "artifacts" / "wcsph_chain"
REFERENCE_2D = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952_crosscheck.json"
GRAVITY = 9.81

#: Non-dimensional instants named by the acceptance criterion.
CRITERION_TIMES = (0.1, 0.2, 0.3)
#: Column geometry of the Martin & Moyce case.
COLUMN_WIDTH = 0.2
COLUMN_HEIGHT = 0.4


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _write(name: str, payload: dict) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)
    return path


def _provenance() -> dict:
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
        except Exception:  # pragma: no cover - git is optional at runtime
            return None

    return {
        "repo": "TensorSPH",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_sha": _git("rev-parse", "HEAD"),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "dtype": "float64",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def tau_of_time(t: float, column_width: float = COLUMN_WIDTH,
                gravity: float = GRAVITY) -> float:
    return t * math.sqrt(2.0 * gravity / column_width)


def load_reference(path: Path = REFERENCE_2D) -> list[dict]:
    """Martin & Moyce front trajectory as ``Z/a`` vs ``tau`` (Z from the edge)."""
    payload = json.loads(path.read_text())
    rows = []
    for point in payload["data"]:
        rows.append({"tau": float(point["tau"]),
                     "Z_over_a": float(point["X_over_a"]) - 1.0})
    return rows


def interpolate(rows: list[dict], tau: float, key: str = "Z_over_a",
                max_extrapolation: float = 0.10) -> float | None:
    """Linear interpolation, with a short linear extrapolation at the ends."""
    taus = [r["tau"] for r in rows]
    vals = [r[key] for r in rows]
    if tau < taus[0] or tau > taus[-1]:
        span = taus[-1] - taus[0]
        if abs(tau - min(max(tau, taus[0]), taus[-1])) > max_extrapolation * span:
            return None
    if tau <= taus[0]:
        return vals[0]
    if tau >= taus[-1]:
        slope = (vals[-1] - vals[-2]) / (taus[-1] - taus[-2])
        return vals[-1] + (tau - taus[-1]) * slope
    for k in range(1, len(taus)):
        if tau <= taus[k]:
            f = (tau - taus[k - 1]) / (taus[k] - taus[k - 1])
            return vals[k - 1] + f * (vals[k] - vals[k - 1])
    return vals[-1]


def reference_at_times(rows: list[dict], times=CRITERION_TIMES,
                       column_width: float = COLUMN_WIDTH) -> dict:
    out = {}
    for t in times:
        tau = tau_of_time(t, column_width)
        out[t] = {"tau": tau, "Z_over_a": interpolate(rows, tau)}
    return out


# --------------------------------------------------------------------------- #
# sub-commands
# --------------------------------------------------------------------------- #

def cmd_pair(args) -> None:
    run = particle_pair(v0=args.v0, c0=args.c0, alpha_av=0.0, steps=args.steps,
                        cfl=args.cfl)
    payload = {
        "case": "particle-pair",
        "provenance": _provenance(),
        "settings": {"v0": args.v0, "c0": args.c0, "alpha_av": 0.0,
                     "dx": 0.01, "steps": args.steps, "cfl": args.cfl,
                     "dtype": "float64"},
        "momentum_drift": run.momentum_drift,
        "energy_drift": run.energy_drift,
        "initial_separation_over_h": run.separation[0] / (1.3 * 0.01),
        "minimum_separation_over_h": min(run.separation) / (1.3 * 0.01),
        "final_separation_over_h": run.separation[-1] / (1.3 * 0.01),
        "times": run.times[::max(1, len(run.times) // 400)],
        "energy": run.energy[::max(1, len(run.energy) // 400)],
        "momentum_norm": [float(m.abs().max()) for m in run.momentum][
            ::max(1, len(run.momentum) // 400)],
        "velocity_mach": args.v0 / args.c0,
    }
    _write("pair.json", payload)


def cmd_dam2d(args) -> None:
    cfg = DamBreakConfig(dx=args.dx, c0=args.c0, alpha_av=args.alpha,
                         t_end=args.seconds, h_adaptive=args.adaptive_h)
    start = time.time()
    run = dam_break(cfg, dim=2)
    wall = time.time() - start
    ref = load_reference()
    payload = {
        "case": "dam_break_2d",
        "provenance": _provenance(),
        "settings": {"dx": args.dx, "c0": args.c0, "alpha_av": args.alpha,
                     "h_factor": cfg.h_factor, "kernel": cfg.kernel,
                     "column_width": cfg.column_width,
                     "column_height": cfg.column_height,
                     "tank_length": cfg.tank_length,
                     "tank_height": cfg.tank_height,
                     "t_end": cfg.t_end, "wall_layers": cfg.wall_layers},
        "diagnostics": {
            "particles": run.particles, "steps": run.steps,
            "simulated_time": run.simulated_time, "dt": run.dt,
            "max_speed": run.max_speed, "mach": run.max_speed / args.c0,
            "max_compression": run.max_compression,
            "max_courant": run.max_courant, "escaped": run.escaped,
            "wall_clock_s": wall,
        },
        "times": run.times,
        "front_max": run.front_max,
        "front_toe": run.front_toe,
        "profiles": {f"{t:g}": v for t, v in run.profiles.items()},
        "profile_edges": run.profile_edges,
        "profile_bins": cfg.profile_bins,
    }
    _write(f"dam2d_dx{args.dx:g}.json", payload)


def cmd_dam3d(args) -> None:
    cfg = DamBreakConfig(dx=args.dx, c0=args.c0, alpha_av=args.alpha,
                         t_end=args.seconds, width=args.width,
                         tank_length=args.tank_length)
    start = time.time()
    run = dam_break(cfg, dim=3)
    wall = time.time() - start
    payload = {
        "case": "dam_break_3d",
        "provenance": _provenance(),
        "settings": {"dx": args.dx, "c0": args.c0, "alpha_av": args.alpha,
                     "column_width": cfg.column_width,
                     "column_height": cfg.column_height,
                     "tank_length": cfg.tank_length,
                     "tank_height": cfg.tank_height, "width": cfg.width,
                     "t_end": cfg.t_end},
        "diagnostics": {
            "particles": run.particles, "steps": run.steps,
            "simulated_time": run.simulated_time, "dt": run.dt,
            "max_speed": run.max_speed, "mach": run.max_speed / args.c0,
            "max_compression": run.max_compression,
            "max_courant": run.max_courant, "escaped": run.escaped,
            "wall_clock_s": wall,
        },
        "times": run.times,
        "front_max": run.front_max,
        "front_toe": run.front_toe,
    }
    _write(f"dam3d_dx{args.dx:g}.json", payload)


def cmd_slosh(args) -> None:
    cfg = SloshingConfig(length=args.length, depth=args.depth, dx=args.dx,
                         amplitude=args.amplitude, frequency=args.freq,
                         c0=args.c0, alpha_av=args.alpha, t_end=args.seconds)
    start = time.time()
    run = sloshing(cfg)
    wall = time.time() - start
    ref = LinearSloshing(length=cfg.length, depth=cfg.depth,
                         amplitude=cfg.amplitude, frequency=cfg.frequency)
    # linear reference sampled on the same time grid
    ref_series = [ref.wall_pressure(cfg.depth, t) for t in run.times]
    settle = args.settle_fraction
    t0 = settle * run.times[-1] if run.times else 0.0
    keep = [k for k, t in enumerate(run.times) if t >= t0]
    payload = {
        "case": "sloshing_2d",
        "provenance": _provenance(),
        "settings": {"length": cfg.length, "depth": cfg.depth, "dx": cfg.dx,
                     "amplitude": cfg.amplitude, "frequency_hz": cfg.frequency,
                     "c0": args.c0, "alpha_av": args.alpha, "t_end": cfg.t_end,
                     "probe_height": cfg.depth},
        "diagnostics": {
            "particles": run.particles, "steps": run.steps,
            "simulated_time": run.simulated_time, "dt": run.dt,
            "max_speed": run.max_speed, "mach": run.max_speed / args.c0,
            "max_compression": run.max_compression,
            "max_courant": run.max_courant, "escaped": run.escaped,
            "wall_clock_s": wall,
        },
        "lines": {"times": run.times, "p_wall_probe": run.p_wall_probe,
                  "eta_wall": run.eta_wall, "p_wall_max": run.p_wall_max,
                  "linear_p_wall_probe": ref_series},
        "windows": {
            "settle_time": t0,
            "sph_peak": max(run.p_wall_probe[k] for k in keep) if keep else 0.0,
            "linear_peak": max(ref_series[k] for k in keep) if keep else 0.0,
            "sph_amplitude": 0.5 * (max(run.p_wall_probe[k] for k in keep)
                                    - min(run.p_wall_probe[k] for k in keep))
            if keep else 0.0,
            "linear_amplitude": 0.5 * (max(ref_series[k] for k in keep)
                                       - min(ref_series[k] for k in keep))
            if keep else 0.0,
            "sph_eta_amplitude": 0.5 * (max(run.eta_wall[k] for k in keep)
                                        - min(run.eta_wall[k] for k in keep))
            if keep else 0.0,
            "linear_eta_amplitude": 0.5 * (
                max(ref.wall_elevation(t) for t in run.times if t >= t0)
                - min(ref.wall_elevation(t) for t in run.times if t >= t0))
            if keep else 0.0,
        },
        "natural_frequency_hz": ref.omega1 / (2.0 * math.pi),
    }
    _write(f"slosh_{args.freq:.2f}Hz.json", payload)


def cmd_tests(args) -> None:
    """Collect + run the new test files and measure their coverage.

    Coverage is driven by ``coverage_run.py`` rather than ``--cov`` because
    torch's C extension cannot be imported under an active tracer.
    """
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob(
        "test_wcsph_*.py"))
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *files],
        cwd=ROOT, capture_output=True, text=True)
    collected = sum(1 for line in collect.stdout.splitlines() if "::" in line)
    ART.mkdir(parents=True, exist_ok=True)
    cov_json = ART / "coverage.json"
    if cov_json.exists():
        cov_json.unlink()
    run = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "wcsph_chain" / "coverage_run.py"),
         str(cov_json), *files], cwd=ROOT, capture_output=True, text=True)
    cov = None
    if cov_json.exists():
        data = json.loads(cov_json.read_text())
        cov = (data.get("totals", {}).get("percent_covered")
                       or data.get("totals", {}).get("percent", 0.0)) / 100.0
    pytest_exit = None
    for line in run.stdout.splitlines():
        if line.startswith("{"):
            pytest_exit = json.loads(line).get("pytest_exit")
    payload = {
        "case": "test-suite",
        "provenance": _provenance(),
        "files": files,
        "collected": collected,
        "pytest_exit": pytest_exit,
        "pytest_tail": (collect.stdout[-1000:] + "\n" + run.stdout[-3000:]),
        "coverage": cov,
        "coverage_source": "coverage_run.py (torch pre-imported)",
    }
    _write("tests.json", payload)
    print(run.stdout[-2000:], flush=True)


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #

def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _profile_l2(a: list[float], b: list[float], lo: float = 0.0,
                hi: float | None = None) -> float:
    """Relative L2 distance between two depth profiles on a shared grid."""
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    aa = torch.tensor(a[:n], dtype=torch.float64)
    bb = torch.tensor(b[:n], dtype=torch.float64)
    if hi is not None:
        keep = torch.linspace(lo, hi, n, dtype=torch.float64) <= hi
        aa, bb = aa[keep], bb[keep]
    den = float((bb ** 2).sum())
    return math.sqrt(float(((aa - bb) ** 2).sum()) / den) if den > 0 else float("inf")


def _triple_grid_order(profiles: list[list[float]]) -> dict:
    """Observed order from three rungs with spacing ``h``, ``h/2``, ``h/4``.

    With ``p`` the order, ``f(h) - f(h/2)`` and ``f(h/2) - f(h/4)`` differ by
    ``2**p``, so ``p = log(|f_h - f_h2| / |f_h2 - f_h4|) / log 2`` -- no
    extrapolated reference is needed.
    """
    coarse, mid, fine = profiles
    n = min(len(coarse), len(mid), len(fine))
    c = torch.tensor(coarse[:n], dtype=torch.float64)
    m = torch.tensor(mid[:n], dtype=torch.float64)
    f = torch.tensor(fine[:n], dtype=torch.float64)
    d1 = float(torch.linalg.vector_norm(m - c))
    d2 = float(torch.linalg.vector_norm(f - m))
    if d2 <= 0.0 or d1 <= 0.0:
        return {"order": None, "difference_coarse": d1, "difference_fine": d2}
    return {"order": math.log(d1 / d2) / math.log(2.0),
            "difference_coarse": d1, "difference_fine": d2}


def cmd_assemble(args) -> None:
    ref_rows = load_reference()
    pair = _load(ART / "pair.json")
    ladder = [_load(p) for p in ART.glob("dam2d_dx*.json")]
    ladder = [r for r in ladder if r]
    ladder.sort(key=lambda r: -r["settings"]["dx"])
    dam3d = None
    for path in sorted(ART.glob("dam3d_dx*.json")):
        dam3d = _load(path)
    slosh_reports = [_load(p) for p in sorted(ART.glob("slosh_*.Hz.json"))]
    slosh_reports = [r for r in slosh_reports if r]
    tests = _load(ART / "tests.json")

    payload: dict = {
        "provenance": _provenance(),
        "reference": {
            "2d": {"path": str(REFERENCE_2D.relative_to(ROOT)),
                   "kind": "Martin & Moyce (1952) front trajectory, "
                           "independent Lethe transcription",
                   "note": "the bundled table is excluded: the repository's own "
                           "audit grades it WITHHELD (exact 0.05 ordinate lattice)"},
            "sloshing": {"kind": "linear potential-flow solution "
                                 "(tensorsph.wcsph.linear_sloshing)",
                         "note": "no digitised tank-pressure experiment is "
                                 "bundled; see docs/benchmarks/wcsph-chain.md"},
        },
    }

    if pair:
        payload["pair"] = {k: pair[k] for k in ("momentum_drift", "energy_drift",
                                                "velocity_mach", "settings",
                                                "initial_separation_over_h",
                                                "minimum_separation_over_h")}

    # ---- 2-D front ------------------------------------------------------ #
    if ladder:
        finest = ladder[-1]
        rows = []
        for t in CRITERION_TIMES:
            tau = tau_of_time(t)
            sim_z = (front_at(finest["times"], finest["front_toe"], t)
                     - COLUMN_WIDTH) / COLUMN_WIDTH
            ref_z = interpolate(ref_rows, tau)
            rows.append({"t": t, "tau": tau, "sim_Z_over_a": sim_z,
                         "ref_Z_over_a": ref_z,
                         "relative_error": chain.relative_error(sim_z, ref_z),
                         "absolute_error": None if ref_z is None else sim_z - ref_z})
        curve = []
        for row in ref_rows:
            if row["tau"] <= 0.0:
                continue
            t = row["tau"] / math.sqrt(2.0 * GRAVITY / COLUMN_WIDTH)
            sim_x = front_at(finest["times"], finest["front_toe"], t)
            if sim_x is None:
                continue
            sim_z = (sim_x - COLUMN_WIDTH) / COLUMN_WIDTH
            curve.append({"tau": row["tau"], "sim": sim_z, "ref": row["Z_over_a"],
                          "error": sim_z - row["Z_over_a"]})
        mae = (sum(abs(c["error"]) for c in curve) / len(curve)) if curve else None
        rel = [abs(r["relative_error"]) for r in rows
               if r["relative_error"] is not None]
        payload["front_2d"] = {
            "reference": "Martin & Moyce 1952 (Lethe transcription) Z/a",
            "dx": finest["settings"]["dx"],
            "rows": rows,
            "max_relative_error": max(rel) if rel else None,
            "mae_over_curve": mae,
            "curve": curve,
            "ladder": [{k: r[k] for k in ("settings", "diagnostics", "times",
                                          "front_toe", "front_max")}
                       for r in ladder],
        }

    # ---- convergence ---------------------------------------------------- #
    if len(ladder) >= 3:
        triples = []
        for coarser, coarse, fine in [(ladder[i], ladder[i + 1], ladder[i + 2])
                                      for i in range(len(ladder) - 2)]:
            t_key = None
            for key in coarser["profiles"]:
                if key in coarse["profiles"] and key in fine["profiles"]:
                    t_key = key
                    break
            if t_key is None:
                continue
            order = _triple_grid_order([coarser["profiles"][t_key],
                                        coarse["profiles"][t_key],
                                        fine["profiles"][t_key]])
            e_coarse = _profile_l2(coarser["profiles"][t_key],
                                   coarse["profiles"][t_key])
            e_fine = _profile_l2(coarse["profiles"][t_key],
                                 fine["profiles"][t_key])
            triples.append({
                "label": f"{coarser['settings']['dx']:g}/"
                         f"{coarse['settings']['dx']:g}/"
                         f"{fine['settings']['dx']:g}",
                "t": float(t_key),
                "order": order["order"],
                "difference_coarse": order["difference_coarse"],
                "difference_fine": order["difference_fine"],
                "l2_coarse_pair": e_coarse,
                "l2_fine_pair": e_fine,
            })
        payload["convergence"] = {"pairs": triples,
                                  "note": "triple-grid order from the archived "
                                          "depth profiles; no extrapolated "
                                          "reference involved"}

    # ---- 3-D front ------------------------------------------------------ #
    if dam3d:
        rows = []
        for t in CRITERION_TIMES:
            tau = tau_of_time(t)
            sim_z = (front_at(dam3d["times"], dam3d["front_toe"], t)
                     - COLUMN_WIDTH) / COLUMN_WIDTH
            ref_z = interpolate(ref_rows, tau)
            rows.append({"t": t, "tau": tau, "sim_Z_over_a": sim_z,
                         "ref_Z_over_a": ref_z,
                         "relative_error": chain.relative_error(sim_z, ref_z),
                         "absolute_error": None if ref_z is None else sim_z - ref_z})
        rel = [abs(r["relative_error"]) for r in rows
               if r["relative_error"] is not None]
        payload["front_3d"] = {
            "reference": "Martin & Moyce 1952 (Lethe transcription) Z/a",
            "dx": dam3d["settings"]["dx"],
            "rows": rows,
            "max_relative_error": max(rel) if rel else None,
            "diagnostics": dam3d["diagnostics"],
            "times": dam3d["times"], "front_toe": dam3d["front_toe"],
        }

    # ---- sloshing ------------------------------------------------------- #
    if slosh_reports:
        rows = []
        for rep in slosh_reports:
            sph = rep["windows"]["sph_peak"]
            lin = rep["windows"]["linear_peak"]
            rows.append({
                "frequency": rep["settings"]["frequency_hz"],
                "sph_peak_pa": sph,
                "linear_peak_pa": lin,
                "relative_error": chain.relative_error(sph, lin),
                "sph_amplitude_pa": rep["windows"]["sph_amplitude"],
                "linear_amplitude_pa": rep["windows"]["linear_amplitude"],
                "sph_eta_amplitude_m": rep["windows"]["sph_eta_amplitude"],
                "linear_eta_amplitude_m": rep["windows"]["linear_eta_amplitude"],
                "natural_frequency_hz": rep["natural_frequency_hz"],
                "reference": "linear potential-flow peak wall pressure",
                "diagnostics": rep["diagnostics"],
            })
        payload["sloshing"] = {
            "rows": rows,
            "reference_kind": "linear potential flow (no digitised experiment)",
            "lines": {f"{r['settings']['frequency_hz']:.2f}":
                      r["lines"] for r in slosh_reports},
        }

    if tests:
        payload["tests"] = {"collected": tests["collected"],
                            "coverage": tests["coverage"],
                            "pytest_exit": tests["pytest_exit"],
                            "files": tests["files"]}

    evaluation = chain.evaluate(payload)
    payload["gates"] = evaluation["gates"]
    payload["summary"] = evaluation["summary"]
    _write("report.json", payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pair")
    p.add_argument("--v0", type=float, default=0.5)
    p.add_argument("--c0", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--cfl", type=float, default=0.1)
    p.set_defaults(func=cmd_pair)

    p = sub.add_parser("dam2d")
    p.add_argument("--dx", type=float, default=0.01)
    p.add_argument("--c0", type=float, default=25.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seconds", type=float, default=0.5)
    p.add_argument("--adaptive-h", action="store_true")
    p.set_defaults(func=cmd_dam2d)

    p = sub.add_parser("dam3d")
    p.add_argument("--dx", type=float, default=0.02)
    p.add_argument("--c0", type=float, default=25.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seconds", type=float, default=0.5)
    p.add_argument("--width", type=float, default=0.4)
    p.add_argument("--tank-length", type=float, default=1.2)
    p.set_defaults(func=cmd_dam3d)

    p = sub.add_parser("slosh")
    p.add_argument("--freq", type=float, default=1.0)
    p.add_argument("--length", type=float, default=1.0)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--dx", type=float, default=0.02)
    p.add_argument("--amplitude", type=float, default=0.01)
    p.add_argument("--c0", type=float, default=25.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--settle-fraction", type=float, default=0.5)
    p.set_defaults(func=cmd_slosh)

    p = sub.add_parser("tests")
    p.set_defaults(func=cmd_tests)

    p = sub.add_parser("assemble")
    p.set_defaults(func=cmd_assemble)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())