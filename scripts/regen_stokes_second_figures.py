#!/usr/bin/env python3
"""Regenerate the Stokes-second figures *from the published bundle only*.

Why this exists
---------------
``benchmarks/stokes_second/run.py`` draws ``amplitude_decay.png`` and
``field_snapshot.png`` while it runs and hashes them into ``MANIFEST.json``.
The published bundle shipped without the PNGs (binary artefacts were dropped on
the way into the repository) while ``results_ladder.json`` still declared them
and ``docs/verification_stokes_second.md`` still embedded them: two dangling
references and no way to tell which figure belonged to which run.

This script closes that gap without touching a single graded number:

* it reads **only** ``out/results_ladder.json`` (the published ladder),
* writes the two declared figures,
* writes ``out/FIGURES_PROVENANCE.json`` recording the sha256 of the source
  bundle, of this script and of every figure it wrote,
* refreshes ``out/MANIFEST.json``'s ``artifacts`` map with the same rule the
  producer uses (every regular file in ``out/`` except ``MANIFEST.json``),
  leaving ``mode``/``provenance``/``gates``/``command`` untouched.

The figures are a *view* of persisted rows, never a new measurement: the
per-row amplitudes and the finest run's space-time field are copied verbatim
from the ladder.  Only the finest mesh's field was kept in the bundle
(``field_note``), so the snapshot panel shows that rung alone — the label says
so instead of implying three overlaid curves.

Usage::

    PYTHONPATH=.:src python scripts/regen_stokes_second_figures.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "benchmarks" / "stokes_second" / "out"

DECLARED = ("amplitude_decay.png", "field_snapshot.png")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(os.path.relpath(path, REPO))


def figure_payload(ladder: dict) -> dict:
    """Everything the figures need, taken from the ladder and nothing else."""
    runs = ladder["runs"]
    finest = max(runs, key=lambda r: r["resolution_delta_over_dx"])
    return {"runs": runs, "finest": finest}


def make_amplitude_decay(ladder: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensorsph.stokes_second import finite_depth_amplitude

    p = figure_payload(ladder)
    runs, finest = p["runs"], p["finest"]
    delta = finest["delta"]
    height, u0 = finest["height"], finest["u0"]
    nu, omega = finest["viscosity"], finest["omega"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for res in runs:
        y = np.array([r["y_over_delta"] for r in res["rows"]])
        a = np.array([r["amplitude_over_U0"] for r in res["rows"]])
        ax[0].semilogy(y, a, "o-", ms=3, lw=1.0,
                       label=r"SPH $\delta/dx$=%d"
                       % res["resolution_delta_over_dx"])
    yv = np.linspace(0.0, float(finest["height_over_delta"]), 400)
    ax[0].semilogy(yv, np.exp(-yv), "k-", lw=2.0,
                   label=r"$e^{-y/\delta}$ (Landau & Lifshitz \S24)")
    ax[0].semilogy(
        yv, finite_depth_amplitude(yv * delta, height, u0, nu, omega) / u0,
        "k--", lw=1.2, label=r"finite depth (exact, $H=6\delta$)")
    ax[0].set_xlabel(r"$y/\delta$")
    ax[0].set_ylabel(r"$A(y)/U_0$")
    ax[0].set_title("amplitude decay in the Stokes layer")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    for res in runs:
        y = np.array([r["y_over_delta"] for r in res["rows"]])
        e = 100.0 * np.array([r["rel_err_half_space"] for r in res["rows"]])
        ax[1].plot(y, e, "o-", ms=3, lw=1.0,
                   label=r"$\delta/dx$=%d" % res["resolution_delta_over_dx"])
    ax[1].axhspan(-5, 5, color="0.85", label=r"$\pm$5 % gate (G1)")
    ax[1].axvline(1.0, color="k", ls=":", lw=1.0)
    ax[1].set_xlabel(r"$y/\delta$")
    ax[1].set_ylabel(r"$(A_{SPH}-A_{analytic})/A_{analytic}$  [%]")
    ax[1].set_title("amplitude error per row")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "amplitude_decay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_field_snapshot(ladder: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensorsph.stokes_second import finite_depth

    runs = figure_payload(ladder)["runs"]
    with_field = [r for r in runs if isinstance(r.get("field"), dict)]
    if not with_field:
        raise RuntimeError("no run in the bundle retained its field: cannot "
                           "draw the space-time panel honestly")
    field_run = max(with_field, key=lambda r: r["resolution_delta_over_dx"])
    delta = field_run["delta"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.array(field_run["field"]["t"])
    y = np.array(field_run["field"]["y"])
    u = np.array([[np.nan if v is None else v for v in row]
                  for row in field_run["field"]["u"]])
    mesh = ax[0].pcolormesh(t, y / delta, u / field_run["u0"],
                            shading="auto", cmap="RdBu_r")
    cb = fig.colorbar(mesh, ax=ax[0])
    cb.set_label(r"$u_x/U_0$")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel(r"$y/\delta$")
    ax[0].set_title(r"SPH velocity field, $\delta/dx$=%d (measurement window)"
                    % field_run["resolution_delta_over_dx"])
    ax[0].invert_yaxis()

    t_cut = t[-1] - field_run["period"]
    idx = int(np.argmin(np.abs(t - t_cut)))
    for res in with_field:
        yy = np.array(res["field"]["y"]) / res["delta"]
        uu = np.array([[np.nan if v is None else v for v in row]
                       for row in res["field"]["u"]])
        ax[1].plot(uu[:, idx] / res["u0"], yy, "o-", ms=3, lw=1.0,
                   label=r"SPH $\delta/dx$=%d"
                   % res["resolution_delta_over_dx"])
    yv = np.linspace(0, float(field_run["height_over_delta"]), 400)
    ax[1].plot(finite_depth(yv * delta, float(t[idx]),
                            field_run["height"], field_run["u0"],
                            field_run["viscosity"], field_run["omega"])
               / field_run["u0"], yv, "k--", lw=1.6,
               label="finite-depth exact at t=%.3f s" % float(t[idx]))
    ax[1].set_xlabel(r"$u_x/U_0$")
    ax[1].set_ylabel(r"$y/\delta$")
    ax[1].set_title("snapshot profile (fields retained: %d of %d rungs)"
                    % (len(with_field), len(runs)))
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "field_snapshot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def repair_manifest(out_dir: Path, ladder: dict) -> dict:
    """Re-hash every artefact and point the header at the published ladder.

    Two things are repaired, both recoverable from the artefacts themselves:

    * ``artifacts`` — re-hashed with the producer's rule (every regular file in
      ``out/`` except ``MANIFEST.json``), so the two regenerated PNGs and the
      provenance file are covered;
    * the header — the published manifest certified a *ladder* bundle while
      saying ``mode: quick`` with the quick run's command and the pre-fix
      ``c0 = 10`` case block, because a later ``--quick`` run had overwritten
      it.  The ladder's own ``mode``/``repro_cmd``/``finished``/``provenance``
      are copied in; the header that is replaced is kept verbatim under
      ``superseded_headers`` (nothing is deleted).

    ``runs`` maps every ``results_*.json`` to the mode/command/finished *that
    artefact reports about itself*; entries recovered here are marked as such.
    """
    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            artifacts[path.name] = {"sha256": sha256(path),
                                    "bytes": path.stat().st_size}
    manifest["artifacts"] = artifacts

    header_keys = ("mode", "command", "finished", "provenance")
    stale = manifest.get("mode") != ladder.get("mode") or \
        manifest.get("command") != ladder.get("repro_cmd")
    if stale:
        superseded = {k: manifest[k] for k in header_keys if k in manifest}
        manifest.setdefault("superseded_headers", []).append(superseded)
        manifest["mode"] = ladder["mode"]
        manifest["command"] = ladder["repro_cmd"]
        manifest["finished"] = ladder.get("finished")
        manifest["provenance"] = ladder["provenance"]
        manifest["header_repaired"] = (
            "header re-pointed at results_ladder.json: the superseded entry "
            "described a later --quick run (stale mode/command/c0)")

    runs = manifest.get("runs") or {}
    for name in sorted(artifacts):
        if not (name.startswith("results_") and name.endswith(".json")):
            continue
        d = json.loads((out_dir / name).read_text())
        entry = {"mode": d.get("mode"), "command": d.get("repro_cmd"),
                 "finished": d.get("finished")}
        if name not in runs:
            entry["recovered_from_artifact"] = True
            runs[name] = entry
    manifest["runs"] = runs
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="bundle directory holding results_ladder.json")
    args = ap.parse_args(argv)
    out_dir: Path = args.out
    ladder_path = out_dir / "results_ladder.json"
    if not ladder_path.is_file():
        print(f"no published ladder at {ladder_path}", file=sys.stderr)
        return 2
    ladder = json.loads(ladder_path.read_text())
    declared = [Path(f).name for f in ladder.get("figures", [])]
    if sorted(declared) != sorted(DECLARED):
        print(f"declared figures {declared} != {list(DECLARED)}: refusing to "
              f"write a figure the bundle does not claim", file=sys.stderr)
        return 2

    made = [make_amplitude_decay(ladder, out_dir),
            make_field_snapshot(ladder, out_dir)]

    import matplotlib
    provenance = {
        "case": "stokes_second_oscillating_plate",
        "anchor": "CASE-SPH-STOKES-2ND-PROBLEM",
        "reason": ("PNGs were absent from the published bundle while "
                   "results_ladder.json declared them; regenerated here from "
                   "the persisted rows/field only, no new simulation"),
        "source": {"path": _rel(ladder_path), "sha256": sha256(ladder_path)},
        "script": {"path": _rel(Path(__file__)),
                   "sha256": sha256(Path(__file__))},
        "figures": {path.name: {"sha256": sha256(path),
                                "bytes": path.stat().st_size}
                    for path in made},
        "fields_retained": sorted(r["resolution_delta_over_dx"]
                                  for r in ladder["runs"]
                                  if isinstance(r.get("field"), dict)),
        "resolutions": [r["resolution_delta_over_dx"]
                        for r in sorted(ladder["runs"],
                                        key=lambda r: r["resolution_delta_over_dx"])],
        "environment": {"matplotlib": matplotlib.__version__,
                        "numpy": np.__version__,
                        "python": platform.python_version()},
    }
    prov_path = out_dir / "FIGURES_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=1))
    manifest = repair_manifest(out_dir, ladder)
    print(json.dumps({"figures": provenance["figures"],
                      "manifest_artifacts": sorted(manifest["artifacts"])},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
