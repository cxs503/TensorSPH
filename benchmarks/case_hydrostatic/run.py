"""Case-pool harness for ``CASE-HYDROSTATIC`` (still-water pressure profile).

The case: a rectangular tank of water at rest under gravity must recover

    p(y) = rho0 g (H - y)

on its interior bands, to better than 1 % of the local analytic value.  The
library side of the case lives in :mod:`tensorsph.case_hydrostatic`; this script
only schedules the configurations, writes the run artifacts and aggregates them
into :file:`runs/case_hydrostatic/report.json`.

Measurement, declared before the numbers were read
--------------------------------------------------
* **Gate** (``GATED`` config): ``max interior-band |p - p_ref| / p_ref`` on the
  *time mean* over the last half of the run, free-surface bands excluded by the
  overlap rule (a band is contaminated as soon as its upper edge enters ``2h``
  of the surface).  Limit 1.0 %.
* **Negative control** ``control_g0``: gravity switched off inside the solver
  while the reference keeps ``rho0 g`` -> must read ~-100 %.
* **Sensitivity**: resolution ladder (``dx``), averaging window, sound speed
  (EOS stiffness) -- each run moves exactly one knob.
* **Mechanism diagnostic**: the deterministic ``t = 0`` lattice record, where
  the only error source is the truncation of the summation density next to the
  walls and the free surface.

Usage
-----
    python benchmarks/case_hydrostatic/run.py --config window_8s
    python benchmarks/case_hydrostatic/run.py --config all
    python benchmarks/case_hydrostatic/run.py --config all --static
    python benchmarks/case_hydrostatic/run.py --aggregate --figure

``--config all --static`` is the *whole* declared grid: the time-stepped ladder
first, then the deterministic ``t = 0`` layer (until 2026-11-12 it silently ran
the static layer only, and the following ``--aggregate`` then died with a
``TypeError`` on the missing gate record).  ``--aggregate --figure`` also mirrors
the bundle into :file:`outputs/case_hydrostatic` with a sha256 manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_hydrostatic import (CASE_ID, DEPTH, G, RHO0,  # noqa: E402
                                       TOLERANCE_PCT, run_case,
                                       run_initial_state)

ARTIFACTS = REPO / "runs" / "case_hydrostatic"
RUNS = ARTIFACTS / "runs"
FIGURES = ARTIFACTS / "figures"

#: Shipped copy of the evidence bundle.  ``runs/`` lives outside the delivery
#: whitelist of the platform (``docs/ src/ tests/ examples/ outputs/``), so a
#: bundle left in ``runs/`` never reaches a fresh checkout -- the shipped-evidence
#: tests then red out on every clean tree.  The case therefore ships a copy here
#: under ``outputs/`` plus a sha256 manifest, so the evidence is decidable (and
#: tamper-evident) without re-running the case.
SHIPPED = REPO / "outputs" / "case_hydrostatic"
MANIFEST = "MANIFEST.sha256"

#: One knob moves per entry.  ``window_8s`` is the gate configuration (declared
#: before any result was read): the time mean of a sloshing inviscid pool needs
#: a window long enough that the sampling noise of the oscillation is below the
#: 1 % gate, which the 2 s window is not.
CONFIGS: dict[str, dict] = {
    "window_2s": dict(dx=0.02, tank_width=1.0, seconds=2.0),
    "window_4s": dict(dx=0.02, tank_width=1.0, seconds=4.0),
    "window_8s": dict(dx=0.02, tank_width=1.0, seconds=8.0),
    "dx_004": dict(dx=0.04, tank_width=1.0, seconds=8.0),
    "dx_0025": dict(dx=0.025, tank_width=1.0, seconds=8.0),
    "width_w15": dict(dx=0.02, tank_width=1.5, seconds=4.0),
    "sound_c30": dict(dx=0.02, tank_width=1.0, seconds=4.0, sound_speed=30.0),
    "control_g0": dict(dx=0.04, tank_width=1.0, seconds=2.0, gravity_override=0.0),
}
DEFAULTS = dict(dx=0.02, tank_width=1.0, seconds=8.0, sound_speed=20.0,
                sample_every=5, bands=8)
GATED = "window_8s"


def spec(name: str) -> dict:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; known: {sorted(CONFIGS)}")
    return {**DEFAULTS, **CONFIGS[name]}


def run_one(name: str, static: bool = False) -> dict:
    kw = spec(name)
    if static:
        if "gravity_override" in kw:
            # A single RHS evaluation carries the initial density field, not the
            # gravity/pressure balance: the g=0 control is only meaningful on the
            # time-stepped record, so no static control is produced.
            return {}
        record = run_initial_state(dx=kw["dx"], tank_width=kw["tank_width"],
                                   depth=DEPTH, sound_speed=kw["sound_speed"],
                                   bands=kw["bands"],
                                   gravity_override=kw.get("gravity_override"))
        out = RUNS / f"static_{name}.json"
    else:
        record = run_case(**kw)
        out = RUNS / f"{name}.json"
    record["config_name"] = name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    gate = record["gate"]
    print(f"[{name}{' static' if static else ''}] "
          f"max err {gate['max_band_error_pct']:.4f} % "
          f"({'PASS' if gate['pass'] else 'FAIL'} vs {TOLERANCE_PCT} %), "
          f"{record['config']['fluid_particles']} particles, "
          f"{record.get('wall_seconds', 0.0)} s")
    return record


def aggregate() -> dict:
    names = [n for n in CONFIGS if (RUNS / f"{n}.json").exists()]
    runs = [json.loads((RUNS / f"{n}.json").read_text()) for n in names]
    statics = [json.loads((RUNS / f"static_{n}.json").read_text())
               for n in CONFIGS if (RUNS / f"static_{n}.json").exists()]
    by = {r["config_name"]: r for r in runs}
    gate = by.get(GATED)
    control = by.get("control_g0")

    if gate is None:
        missing = [n for n in CONFIGS if not (RUNS / f"{n}.json").exists()]
        raise SystemExit(
            f"aggregate: the gate config {GATED!r} has no run file under {RUNS}; "
            f"missing configs {missing}. Run "
            "`python benchmarks/case_hydrostatic/run.py --config all --static` "
            "first (time-stepped ladder + t=0 layer).")

    verdict = {
        "case_id": CASE_ID,
        "gate_config": GATED,
        "gate_metric": "max interior-band |p_SPH - p_ref| / p_ref (time mean)",
        "gate_limit_pct": TOLERANCE_PCT,
        "gate_value_pct": gate["gate"]["max_band_error_pct"] if gate else None,
        "gate_pass": gate["gate"]["pass"] if gate else None,
        "negative_control_pct": (control["gate"]["mean_band_error_pct"]
                                 if control else None),
        "negative_control_ok": (bool(abs(control["gate"]["mean_band_error_pct"]) > 80.0)
                                if control else None),
    }

    def ladder(records: list[dict], key: str) -> list[dict]:
        return sorted(
            [{"config": r["config_name"], "columns": r["config"]["columns"],
              "dx": r["config"]["dx"],
              "max_band_error_pct": r["gate"]["max_band_error_pct"],
              "mean_band_error_pct": r["gate"]["mean_band_error_pct"]}
             for r in records if key in r["config_name"] or key == "all"],
            key=lambda row: row["dx"])

    report = {
        "case_id": CASE_ID,
        "title": "Still-water hydrostatic pressure profile p(y) = rho0 g (H - y)",
        "repo": "TensorSPH",
        "verdict": verdict,
        "reference": gate["reference"] if gate else None,
        "gate_run": gate,
        "configs": {n: spec(n) for n in names},
        "runs": runs,
        "static_runs": statics,
        "resolution_ladder_static": [
            {"config": r["config_name"], "columns": r["config"]["columns"],
             "dx": r["config"]["dx"],
             "max_band_error_pct": r["gate"]["max_band_error_pct"]}
            for r in sorted(statics, key=lambda r: -r["config"]["dx"])],
        "window_ladder": [
            {"config": r["config_name"], "seconds": r["config"]["seconds"],
             "samples": r["config"]["samples"],
             "max_band_error_pct": r["gate"]["max_band_error_pct"],
             "mean_band_error_pct": r["gate"]["mean_band_error_pct"],
             "max_fluctuation_pct": r["max_fluctuation_pct"]}
            for r in sorted((r for r in runs if r["config_name"].startswith("window")),
                            key=lambda r: r["config"]["seconds"])],
        "repro": {
            "one_shot": "python -m tensorsph.case_hydrostatic --json-out "
                        "runs/case_hydrostatic/cli_record.json",
            "gate": "python benchmarks/case_hydrostatic/run.py --config window_8s",
            "grid": "python benchmarks/case_hydrostatic/run.py --config all "
                    "--static  # -> runs/case_hydrostatic",
            "figures": "python benchmarks/case_hydrostatic/run.py --aggregate --figure",
            "tests": "python -m pytest -q tests/test_case_hydrostatic.py",
            "shipped": "outputs/case_hydrostatic (mirror of runs/case_hydrostatic"
                       " + MANIFEST.sha256, copied by --aggregate)",
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=1))
    return report


def figure(report: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    made = {}
    gate = report["gate_run"]
    ref = gate["reference"]

    # 1. profile vs analytic
    prof = gate["profile"]
    ys = [b["y"] for b in prof]
    ana = [RHO0 * G * (ref["H"] - y) for y in ys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ana, ys, "k-", label="analytic  $p=\\rho_0 g (H-y)$")
    gated = [b for b in prof if b["gated"]]
    ax.plot([b["pressure"] for b in gated], [b["y"] for b in gated], "o",
            label=f"SPH band mean (gated, max err "
                  f"{gate['gate']['max_band_error_pct']:.3f} %)")
    shelf = [b for b in prof if not b["gated"]]
    ax.plot([b["pressure"] for b in shelf], [b["y"] for b in shelf], "s",
            mfc="none", label="free-surface bands (excluded)")
    ax.set_xlabel("pressure [Pa]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{CASE_ID}: hydrostatic profile ({report['verdict']['gate_config']})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    path = FIGURES / "profile_vs_analytic.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made["profile_vs_analytic"] = str(path.relative_to(REPO))

    # 2. error per band
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([b["band"] for b in prof], [b["rel_error_pct"] for b in prof],
           color=["tab:blue" if b["gated"] else "tab:gray" for b in prof])
    ax.axhline(TOLERANCE_PCT, color="r", ls="--", lw=1, label=f"+{TOLERANCE_PCT} %")
    ax.axhline(-TOLERANCE_PCT, color="r", ls="--", lw=1, label=f"-{TOLERANCE_PCT} %")
    ax.set_xlabel("band index (bottom -> free surface)")
    ax.set_ylabel("(p_SPH - p_ref)/p_ref  [%]")
    ax.set_title("band error; grey = free-surface bands excluded from the gate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = FIGURES / "band_errors.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made["band_errors"] = str(path.relative_to(REPO))

    # 3. sensitivity ladders
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    win = report["window_ladder"]
    if win:
        axes[0].plot([w["seconds"] for w in win],
                     [w["max_band_error_pct"] for w in win], "o-")
        axes[0].set_xlabel("averaging window [s]")
        axes[0].set_ylabel("max band error [%]")
        axes[0].set_title("time-mean convergence")
    stat = report["resolution_ladder_static"]
    if stat:
        axes[1].plot([s["columns"] for s in stat],
                     [s["max_band_error_pct"] for s in stat], "s-")
        axes[1].set_xlabel("columns across the tank (W/dx)")
        axes[1].set_ylabel("max band error [%]")
        axes[1].set_title("static lattice: truncation ladder")
    for ax in axes:
        ax.grid(alpha=0.3)
    path = FIGURES / "sensitivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made["sensitivity"] = str(path.relative_to(REPO))
    return made


def ship(artifacts: Path = ARTIFACTS, dest: Path = SHIPPED) -> dict:
    """Mirror the evidence bundle into the delivery whitelist + sha256 manifest.

    The manifest is ``<sha256>  <bytes>  <relative path>`` per file, sorted, so
    a reader can verify that the shipped numbers are the ones the run produced:
    a hand-edited report, a truncated run file or a half-copied bundle all fail
    the integrity check in ``tests/test_case_hydrostatic.py``.
    """
    import hashlib
    import shutil

    files = sorted(p for p in artifacts.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"ship: nothing to ship under {artifacts}")
    dest.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in files:
        rel = path.relative_to(artifacts)
        data = path.read_bytes()
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {len(data)}  {rel.as_posix()}")
    manifest = dest / MANIFEST
    manifest.write_text("\n".join(lines) + "\n")
    return {"dest": dest.relative_to(REPO).as_posix(),
            "manifest": manifest.relative_to(REPO).as_posix(),
            "files": len(files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="config name, 'all', or 'all-static'")
    parser.add_argument("--static", action="store_true",
                        help="run the deterministic t=0 record instead")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    rc = 0
    if args.config:
        if args.config == "all":
            names = list(CONFIGS)
        else:
            names = [args.config]
        # ``--config all --static`` selects the whole declared grid (time-stepped
        # ladder, then the t = 0 layer); a single named config keeps the old
        # meaning (``--static`` = that config's static record only).
        layers = (False, True) if (args.static and args.config == "all") else (args.static,)
        for static in layers:
            for name in names:
                record = run_one(name, static=static)
                if not record:
                    continue
                if not static and not name.startswith("control") and not record["gate"]["pass"]:
                    rc = 1
    if args.aggregate or args.figure:
        report = aggregate()
        print(f"gate: {report['verdict']['gate_value_pct']:.4f} % "
              f"-> {'PASS' if report['verdict']['gate_pass'] else 'FAIL'}")
        if args.figure:
            for name, path in figure(report).items():
                print(f"figure {name}: {path}")
        shipped = ship()
        print(f"shipped {shipped['files']} files -> {shipped['dest']} "
              f"(manifest {shipped['manifest']})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
