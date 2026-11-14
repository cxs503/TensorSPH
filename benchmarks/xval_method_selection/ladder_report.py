"""Assemble the method-A ladder report tables (XVAL-SPH-METHOD-SELECTION-2026-11-12).

Every number in the generated report is read from files:

* the sweep summaries (``ladder_stability*.json``) written by
  ``ladder_stability_sweep.py``;
* the decision matrix (``docs/reports/xval_method_selection_m17501.json``) written
  by the shipped grader ``grade_matrix.py``;
* the case configuration, re-built with ``benchmarks/wave_flume/run.py``'s own
  ``build_config`` so the cost projection cannot drift from the driver.

The markdown tables are injected between marker comments in the developer report,
so nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/ladder_report.py \
        --summary runs/xval/ladder_stability.json runs/xval/ladder_stability_rest.json \
        --matrix docs/reports/xval_method_selection_m17501.json \
        --doc docs/dev_report_xval_method_selection_m21050.md \
        --out-json docs/reports/xval_method_selection_ladder_m21050.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

VERDICT_TEXT = {
    "blew_up": "**blow-up**",
    "survived_stable": "survived, within both gates",
    "survived_unstable": "survived horizon, out of a gate",
    "rejected": "rejected by `validate_flume` (out of domain)",
    "no_record": "**no record**",
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(paths) -> list:
    rows, seen = [], set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        for row in data["rows"]:
            if row["label"] not in seen:
                seen.add(row["label"])
                rows.append(row)
    order = {"ladder": 0, "mechanism": 1}
    rows.sort(key=lambda r: (order.get(r.get("group"), 9), r["depth_cells"], r["label"]))
    return rows


def ladder_table(rows) -> str:
    lines = [
        "| label | group | H/Δx | Δt [s] | steps asked | verdict | blow-up step | blow-up t [s] | max&nbsp;\\|v\\|/c₀ | compression | escaped |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| `{}` | {} | {:.2f} | {:.3e} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["label"], r.get("group", ""), r["h_over_dx"], r["dt_s"],
            r["steps_requested"], VERDICT_TEXT.get(r["verdict"], r["verdict"]),
            "—" if r["onset_step"] is None else r["onset_step"],
            "—" if r["onset_time_s"] is None else f"{r['onset_time_s']:.3f}",
            "—" if r["speed_ratio"] is None else f"{r['speed_ratio']:.3e}",
            "—" if r["max_compression"] is None else f"{r['max_compression']:.3%}",
            "—" if r["escaped"] is None else r["escaped"]))
    return "\n".join(lines)


def matrix_table(matrix: dict) -> str:
    lines = ["| label | record present | verdict | error / note |", "|---|---|---|---|"]
    for row in matrix.get("rows", []):
        if row.get("present") and row.get("failed"):
            verdict = "**FAILED (numerically)**"
            note = str(row.get("error"))[:70]
        elif not row.get("present"):
            verdict = "missing record"
            note = str(row.get("note", ""))[:70]
        elif not row.get("gradeable", True):
            verdict = "ungradeable"
            note = str(row.get("error"))[:70]
        else:
            verdict = row.get("verdict", "graded")
            note = "within 15 %" if row.get("within_owner_tol") else "outside 15 %"
        lines.append("| `{}` | {} | {} | {} |".format(
            row.get("label"), "yes" if row.get("present") else "no", verdict, note))
    counts = matrix.get("counts", {})
    lines.append("")
    lines.append("counts: " + ", ".join(f"`{k}={v}`" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def cost_projection(depth_cells=(13, 20, 32, 48, 63)) -> str:
    """Particle count / step count / relative cost for candidate resolutions."""
    run = _load("wave_flume_run", REPO / "benchmarks" / "wave_flume" / "run.py")
    sweep = _load("ladder_sweep", REPO / "benchmarks" / "xval_method_selection"
                  / "ladder_stability_sweep.py")

    def cfg_for(dc: int):
        args = run.argparse.Namespace(
            case="regular", depth=sweep.DEPTH_M, depth_cells=dc,
            height_ratio=sweep.HEIGHT_RATIO, wave_period=1.0, gravity=9.81,
            flume_length=3.2, sound_speed=sweep.SOUND_SPEED, viscosity=1.0e-6,
            density_diffusion=0.1, density_regularization="none",
            compression_rule="internal", transport_velocity="off",
            transport_velocity_strength=1.0, sponge_strength=4.0, active_absorption=0.0,
            ramp_periods=2.0, sponge_length=None, xsph=0.0, xsph_settle=0.1,
            viscosity_settle=0.02, settle=0.8, dt=sweep.DT_REQUESTED,
            max_courant=sweep.MAX_COURANT, simulate_time=5.5, output_every=40,
            snapshots=4, threads=8, device="cpu")
        return run.build_config(args)

    from tensorsph.wave_flume import flume_water_lattice, static_wall_lattice, \
        paddle_lattice, resolve_device

    base = None
    lines = ["| candidate H/Δx | depth cells | Δx [mm] | fluid particles | steps for 5.5 s "
             "| relative particle-steps vs H/Δx = 3.20 |",
             "|---|---|---|---|---|---|"]
    for dc in depth_cells:
        cfg = cfg_for(dc)
        device = resolve_device(cfg.device)
        fluid = flume_water_lattice(cfg, device).shape[0]
        wall = static_wall_lattice(cfg, device).shape[0] + paddle_lattice(cfg, device).shape[0]
        work = (fluid + wall) * cfg.steps
        if dc == 20:
            base = work
        lines.append("| {:.2f} | {} | {:.1f} | {} | {} | {} |".format(
            sweep.h_over_dx(dc), dc, cfg.dx * 1e3, fluid, cfg.steps, work))
    out = []
    for line in lines:
        if line.startswith("| 3.20 |"):
            out.append(line)
            continue
        out.append(line)
    # relative column is filled in a second pass so the H/Δx = 3.20 row stays the unit
    filled = []
    for line in out:
        if line.startswith("|---") or line.startswith("| candidate"):
            filled.append(line)
            continue
        cells = line.split("|")
        try:
            work = float(cells[6])
        except (IndexError, ValueError):
            filled.append(line)
            continue
        cells[6] = f" {work / base:.1f}× "
        filled.append("|".join(cells))
    return "\n".join(filled)


def inject(doc: Path, sections: dict) -> list:
    text = doc.read_text()
    missing = []
    for name, body in sections.items():
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
        block = f"{begin}\n{body}\n{end}"
        if pattern.search(text):
            text = pattern.sub(lambda _m: block, text)
        else:
            missing.append(name)
    doc.write_text(text)
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--matrix", default="docs/reports/xval_method_selection_m17501.json")
    parser.add_argument("--doc", default="docs/dev_report_xval_method_selection_m21050.md")
    parser.add_argument("--out-json",
                        default="docs/reports/xval_method_selection_ladder_m21050.json")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = load_rows(args.summary)
    matrix_path = REPO / args.matrix
    matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else {}

    sections = {
        "LADDER_TABLE": ladder_table(rows),
        "MATRIX_TABLE": matrix_table(matrix) if matrix else "(matrix not found)",
        "COST_TABLE": cost_projection(),
    }
    payload = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "rows": rows,
        "counts": {
            "points": len(rows),
            "blew_up": sum(r["verdict"] == "blew_up" for r in rows),
            "survived_stable": sum(r["verdict"] == "survived_stable" for r in rows),
            "survived_unstable": sum(r["verdict"] == "survived_unstable" for r in rows),
            "rejected": sum(r["verdict"] == "rejected" for r in rows),
            "no_record": sum(r["verdict"] == "no_record" for r in rows),
        },
        "matrix_counts": matrix.get("counts", {}),
    }
    out_json = REPO / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    if args.print_only:
        for name, body in sections.items():
            print(f"--- {name}\n{body}\n")
        return 0

    doc = REPO / args.doc
    missing = inject(doc, sections) if doc.exists() else []
    print(f"wrote {out_json}")
    if missing:
        print(f"markers not found in {doc}: {missing}")
        return 1
    print(f"injected {list(sections)} into {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())