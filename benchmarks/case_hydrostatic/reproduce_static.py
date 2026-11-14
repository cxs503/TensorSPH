"""Reproduce the *deterministic* ``t = 0`` layer of ``CASE-HYDROSTATIC``.

Why this tool exists
--------------------
The case ships its evidence as text under ``tests/data/case_hydrostatic``
(``report.json`` + every ``runs/*.json`` + a sha256 manifest), so the 1 %
acceptance is decidable on a fresh checkout.  What was *not* decidable there is
**reproducibility**: the only test that re-runs anything
(``test_shipped_evidence_agrees_with_a_regenerated_bundle``) needs a locally
regenerated ``runs/`` bundle, i.e. the full declared grid
(``run.py --config all --static``, ~11 min wall clock).  On a clean tree that
test skips -- a check that never runs where the evidence is actually read.

The time-stepped gate itself is expensive, but the mechanism layer is not: the
``t = 0`` lattice record (:func:`tensorsph.case_hydrostatic.run_initial_state`)
is deterministic (no time integration, one pressure evaluation) and costs ~0.3 s
per configuration.  This tool re-derives **every shipped static record** from
source, compares it band by band with the frozen evidence and writes the
comparison to ``docs/case_hydrostatic_static_repro.json``; the test suite asserts
the same thing (``test_shipped_static_layer_reproduces_on_a_fresh_checkout``), so
CI gets a real reproducibility check instead of a skip.

What is compared
----------------
* the declared configuration (``dx``, tank width, depth, sound speed, bands,
  particle count, ``h``, support width) -- a knob drift fails here;
* every band row: ``y_m``, ``ref_Pa``, ``got_Pa``, ``rel_error_pct``;
* the gate record: ``max_band_error_pct``, ``mean_band_error_pct``,
  ``worst_band`` and the gated / free-surface band partition.

Float comparison uses ``rel=1e-9``: the shipped numbers were produced by the
same code on the same kind of hardware, and the only legitimate difference is
summation order inside the density loop (arch/BLAS), which shows up far below
that.  Anything a developer would call a change (a band value moving in the 6th
digit, a different band partition, a moved gate) fails.

Usage
-----
    python benchmarks/case_hydrostatic/reproduce_static.py            # check
    python benchmarks/case_hydrostatic/reproduce_static.py \
        --json-out docs/case_hydrostatic_static_repro.json

Exit status is 0 only when every shipped static record reproduces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_hydrostatic import run_initial_state  # noqa: E402

SHIPPED = REPO / "tests" / "data" / "case_hydrostatic" / "runs"
REL = 1e-9
FIELDS = ("y_m", "ref_Pa", "got_Pa", "rel_error_pct")


def _static_kwargs(record: dict) -> dict:
    """Knobs of the t = 0 layer, read off the shipped record itself."""
    cfg = record["config"]
    return dict(dx=cfg["dx"], tank_width=cfg["tank_width"], depth=cfg["depth"],
                sound_speed=cfg["sound_speed"], bands=cfg["bands"])


def _close(a: float, b: float, rel: float = REL) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def repro_one(path: Path) -> dict:
    """Re-derive one shipped static record and diff it against the frozen one."""
    shipped = json.loads(path.read_text())
    fresh = run_initial_state(**_static_kwargs(shipped))
    name = shipped.get("config_name") or path.stem

    diffs: list[str] = []
    for key, value in shipped["config"].items():
        other = fresh["config"].get(key)
        if isinstance(value, float):
            if other is None or not _close(value, other):
                diffs.append(f"config.{key}: {value!r} -> {other!r}")
        elif value != other:
            diffs.append(f"config.{key}: {value!r} -> {other!r}")

    old_rows, new_rows = shipped["benchmark"], fresh["benchmark"]
    if len(old_rows) != len(new_rows):
        diffs.append(f"band count: {len(old_rows)} -> {len(new_rows)}")
    else:
        for a, b in zip(old_rows, new_rows):
            if a["band"] != b["band"]:
                diffs.append(f"band id {a['band']} -> {b['band']}")
            for field in FIELDS:
                if not _close(float(a[field]), float(b[field])):
                    diffs.append(f"band {a['band']} {field}: {a[field]!r} -> {b[field]!r}")

    gs, gf = shipped["gate"], fresh["gate"]
    for field in ("max_band_error_pct", "mean_band_error_pct"):
        if not _close(float(gs[field]), float(gf[field])):
            diffs.append(f"gate.{field}: {gs[field]!r} -> {gf[field]!r}")
    for field in ("worst_band", "gated_bands", "free_surface_bands", "limit_pct"):
        if gs.get(field) != gf.get(field):
            diffs.append(f"gate.{field}: {gs.get(field)!r} -> {gf.get(field)!r}")

    return {
        "record": path.name,
        "config_name": name,
        "dx": shipped["config"]["dx"],
        "tank_width": shipped["config"]["tank_width"],
        "sound_speed": shipped["config"]["sound_speed"],
        "fluid_particles": shipped["config"]["fluid_particles"],
        "shipped_max_band_error_pct": gs["max_band_error_pct"],
        "fresh_max_band_error_pct": gf["max_band_error_pct"],
        "shipped_mean_band_error_pct": gs["mean_band_error_pct"],
        "fresh_mean_band_error_pct": gf["mean_band_error_pct"],
        "worst_band": gs["worst_band"],
        "worst_band_y": gs.get("worst_band_y"),
        "gated_bands": gs.get("gated_bands"),
        "reproduced": not diffs,
        "diffs": diffs,
    }


def ladder(records: list[dict]) -> list[dict]:
    """The resolution ladder read on the static layer, cheapest knob first."""
    by_dx: dict[float, dict] = {}
    for rec in records:
        if rec["tank_width"] == 1.0 and rec["sound_speed"] == 20.0:
            by_dx.setdefault(rec["dx"], rec)
    rows = [by_dx[dx] for dx in sorted(by_dx)]
    return [{"dx": r["dx"],
             "fluid_particles": r["fluid_particles"],
             "max_band_error_pct": r["shipped_max_band_error_pct"],
             "worst_band": r["worst_band"],
             "worst_band_y": r["worst_band_y"]} for r in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reproduce_static")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="write the comparison table here")
    args = parser.parse_args(argv)

    paths = sorted(SHIPPED.glob("static_*.json"))
    if not paths:
        print(f"no shipped static records under {SHIPPED}", file=sys.stderr)
        return 2

    rows = [repro_one(p) for p in paths]
    print(f"{'record':18s} {'dx':>6s} {'max err %':>10s} {'fresh %':>10s}  reproduced")
    for r in rows:
        print(f"{r['config_name']:18s} {r['dx']:6.3f} "
              f"{r['shipped_max_band_error_pct']:10.5f} "
              f"{r['fresh_max_band_error_pct']:10.5f}  "
              f"{'yes' if r['reproduced'] else 'NO'}")
    assert rows  # non-empty above
    lad = ladder(rows)
    print("\nstatic truncation ladder (single tank, c = 20 m/s):")
    for r in lad:
        print(f"  dx={r['dx']:.3f} ({r['fluid_particles']:4d} particles): "
              f"{r['max_band_error_pct']:7.4f} %  worst band {r['worst_band']} "
              f"at y={r['worst_band_y']:.4f} m")

    bad = [r for r in rows if not r["reproduced"]]
    for r in bad:
        print(f"\n{r['record']}:")
        for d in r["diffs"][:10]:
            print(f"  - {d}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} shipped static records reproduced")

    if args.json_out is not None:
        payload = {"case_id": "CASE-HYDROSTATIC",
                   "layer": "static_initial_state",
                   "shipped": str(SHIPPED.relative_to(REPO)),
                   "rel_tolerance": REL,
                   "records": rows,
                   "truncation_ladder": lad,
                   "all_reproduced": not bad}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json_out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
