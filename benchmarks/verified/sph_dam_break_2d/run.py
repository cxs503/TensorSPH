"""M1 + M3 benchmark: 2-D dam break against measurement, and mass bookkeeping.

Usage (each spacing is a separate process so the benchmark can be split across
nodes; the figures are assembled from the evidence store afterwards)::

    python benchmarks/verified/sph_dam_break_2d/run.py --spacings 0.02
    python benchmarks/verified/sph_dam_break_2d/run.py --spacings 0.01
    python benchmarks/verified/sph_dam_break_2d/run.py --controls
    python benchmarks/verified/sph_dam_break_2d/run.py --figures

``--controls`` runs the ``g = 0`` negative control and the two mass-conservation
runs (closed tank, open tank).  ``--figures`` draws from the database only, so a
plot cannot exist without the run that produced it.

Exit status is non-zero if any gate of the stages that were requested fails.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorsph.verification import conservation, dam_break_2d as m1, report
from tensorsph.verification.evidence import EvidenceStore

HERE = Path(__file__).resolve().parent
STORE = HERE / "evidence.sqlite"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def run_spacing(store: EvidenceStore, dx: float, seconds: float,
                table: dict) -> dict:
    cfg = m1.DamBreak2DConfig(dx=dx, seconds=seconds, sample_interval=0.02)
    result = m1.run(cfg, label=f"dam_break_2d_dx{dx:g}")
    front = m1.front_errors(result, table, key="front_coherent", min_tau=0.5)
    front_all = m1.front_errors(result, table, key="front_coherent")
    layer = m1.front_errors(result, table, key="front_layer", min_tau=0.5)
    height = m1.height_anchor_errors(result)
    ritter = m1.ritter_errors(result)

    m = result["config"]
    limit = 0.10
    front_pass = bool(front["max_rel_error"] <= limit)
    height_pass = bool(height["residual_max_rel_error"] <= limit
                       and height["ballistic_max_rel_error"] <= limit)
    run_id = store.add_run(
        "dam_break_2d",
        {"dx": dx, "seconds": seconds, "column_width": m["column_width"],
         "column_height": m["column_height"], "aspect": m["aspect"],
         "sound_speed": m["sound_speed"], "viscosity": m["viscosity"],
         "dt": result["dt"], "steps": result["steps_taken"],
         "density_renormalization": m["density_renormalization"],
         "surface_pressure_closure": m["surface_pressure_closure"],
         "wall_friction": m["wall_friction"],
         "experimental_reference": table["_path"],
         "experimental_grade": table["_grade"]},
        notes="dam-break free surface vs Martin & Moyce (1952) table "
              f"(grade {table['_grade']}) and two exact height anchors")
    metrics = {
        "front_max_rel_error": dict(value=front["max_rel_error"], unit="1",
                                    gate_limit=limit,
                                    verdict=_verdict(front_pass),
                                    provenance="max over experimental points, "
                                               "tau >= 0.5, coherent front"),
        "front_rms_rel_error": dict(value=front["rms_rel_error"], unit="1"),
        "front_n_points": dict(value=float(front["n_points"]), unit="1"),
        "front_max_rel_error_including_startup": dict(
            value=front_all["max_rel_error"], unit="1"),
        "front_layer_max_rel_error": dict(value=layer["max_rel_error"], unit="1"),
        "height_residual_max_rel_error": dict(
            value=height["residual_max_rel_error"], unit="1", gate_limit=limit,
            verdict=_verdict(height["residual_max_rel_error"] <= limit),
            provenance="h_back vs H for t <= a/(2 c0) (exact Ritter tail)"),
        "height_ballistic_max_rel_error": dict(
            value=height["ballistic_max_rel_error"], unit="1", gate_limit=limit,
            verdict=_verdict(height["ballistic_max_rel_error"] <= limit),
            provenance="h_dam vs H - g t^2/2 for t <= a/c0 (free-fall corner)"),
        "ritter_ratio_mean": dict(value=ritter["ratio_mean"], unit="1"),
        "ritter_ratio_min": dict(value=ritter["ratio_min"], unit="1"),
        "ritter_ratio_max": dict(value=ritter["ratio_max"], unit="1"),
        "gate_front_pass": dict(value=float(front_pass), unit="bool",
                                gate_limit=1.0, verdict=_verdict(front_pass)),
        "gate_height_pass": dict(value=float(height_pass), unit="bool",
                                 gate_limit=1.0, verdict=_verdict(height_pass)),
        "mass_drift_rel": dict(value=abs(result["mass_drift"]) / result["mass_initial"],
                               unit="1"),
        "max_mach": dict(value=result["max_mach"], unit="1"),
        "max_density_compression": dict(value=result["max_density_compression"],
                                        unit="1"),
        "particles": dict(value=float(result["rows"][0]["particles"]), unit="1"),
        "steps": dict(value=float(result["steps_taken"]), unit="1"),
        "simulated_time": dict(value=result["simulated_time"], unit="s"),
    }
    store.add_metrics(run_id, metrics)
    store.add_series(run_id, "front_points", front["points"])
    store.add_series(run_id, "height_series", [
        {"t": r["t"], "tau": r["tau"], "h_back": r["h_back"], "h_dam": r["h_dam"]}
        for r in result["rows"]])
    store.add_series(run_id, "profiles", [
        {"tau": r["tau"], "t": r["t"], "bin": r["profile_bin"],
         "a": m["column_width"], "h": r["profile"]} for r in result["rows"]
        if abs(r["tau"] - round(r["tau"] * 2) / 2) < 0.06 and r["tau"] <= 4.5])
    store.add_series(run_id, "front_curve", {
        "dx": dx, "particles": int(result["rows"][0]["particles"]),
        "tau": [r["tau"] for r in result["rows"]],
        "Z_over_a": [(r["front_coherent"] - m["column_width"]) / m["column_width"]
                     for r in result["rows"]],
        "ritter": [m1.ritter_front(cfg, r["t"]) for r in result["rows"]],
    })
    out = {"dx": dx, "run_id": run_id, "seconds": seconds,
           "front": front, "front_layer": layer, "height": height,
           "ritter": ritter, "config": m, "steps": result["steps_taken"],
           "gates": {"front": front_pass, "height": height_pass}}
    (HERE / f"results_dx{dx:g}.json").write_text(json.dumps(out, indent=1))
    return out


def run_controls(store: EvidenceStore, table: dict, dx: float) -> dict:
    cfg = m1.DamBreak2DConfig(dx=dx)
    ctrl = m1.negative_control_gravity_off(cfg, seconds=0.2)
    gate_spec = ctrl["gate"]
    ctrl_pass = bool(ctrl["front_drift_over_dx"] <= gate_spec["front_drift_over_dx"]
                     and ctrl["height_drift_rel"] <= gate_spec["height_drift_rel"]
                     and ctrl["max_particle_displacement_over_dx"]
                     <= gate_spec["max_particle_displacement_over_dx"])
    run_id = store.add_run("dam_break_2d_gravity_off",
                           {"dx": dx, "seconds": ctrl["seconds"], "gravity": 0.0},
                           notes="negative control: no gravity, column must not collapse")
    store.add_metrics(run_id, {
        "front_drift_over_dx": dict(value=ctrl["front_drift_over_dx"], unit="dx",
                                    gate_limit=gate_spec["front_drift_over_dx"],
                                    verdict=_verdict(ctrl["front_drift_over_dx"]
                                                     <= gate_spec["front_drift_over_dx"])),
        "height_drift_rel": dict(value=ctrl["height_drift_rel"], unit="1",
                                 gate_limit=gate_spec["height_drift_rel"],
                                 verdict=_verdict(ctrl["height_drift_rel"]
                                                  <= gate_spec["height_drift_rel"])),
        "max_particle_displacement_over_dx": dict(
            value=ctrl["max_particle_displacement_over_dx"], unit="dx",
            gate_limit=gate_spec["max_particle_displacement_over_dx"],
            verdict=_verdict(ctrl["max_particle_displacement_over_dx"]
                             <= gate_spec["max_particle_displacement_over_dx"])),
        "gate_pass": dict(value=float(ctrl_pass), unit="bool", gate_limit=1.0,
                          verdict=_verdict(ctrl_pass)),
    })

    # mass conservation: closed tank and open tank (bookkeeping control)
    closed_case = conservation.ConservationCase(dx=dx, open_x=None, seconds=0.45,
                                                sample_every=10)
    closed = conservation.run(closed_case)
    closed_gate = conservation.gate(closed, min_steps=1000)
    closed_id = store.add_run("mass_conservation_dam_break",
                              closed_case.__dict__,
                              notes="closed tank, accounted mass over the run")
    store.add_metrics(closed_id, {
        "relative_loss_accounted": dict(value=closed_gate["measured"], unit="1",
                                        gate_limit=closed_gate["limit"],
                                        verdict=_verdict(closed_gate["pass"])),
        "relative_loss_at_horizon": dict(
            value=closed["measured"]["relative_loss_accounted_at_horizon"]
            if closed["measured"]["relative_loss_accounted_at_horizon"]
            is not None else closed_gate["measured"], unit="1"),
        "steps_run": dict(value=float(closed["measured"]["steps_run"]), unit="1"),
        "gate_pass": dict(value=float(closed_gate["pass"]), unit="bool",
                          gate_limit=1.0, verdict=_verdict(closed_gate["pass"])),
    })
    store.add_series(closed_id, "mass_history", closed["history"])

    open_case = conservation.ConservationCase(dx=dx, open_x=0.40, seconds=0.30,
                                              sample_every=10)
    opened = conservation.run(open_case)
    open_gate = conservation.gate_overflow(opened)
    open_id = store.add_run("mass_conservation_dam_break_open",
                            open_case.__dict__,
                            notes="open boundary at x=0.40 m; forgetting the "
                                  "outflow is the leaky bookkeeping control")
    store.add_metrics(open_id, {
        "relative_loss_accounted": dict(value=open_gate["accounted_loss"], unit="1",
                                        gate_limit=open_gate["limit"],
                                        verdict=_verdict(open_gate["accounted_loss"]
                                                         <= open_gate["limit"])),
        "relative_loss_unaccounted": dict(value=open_gate["unaccounted_loss"],
                                          unit="1", gate_limit=open_gate["min_loss"],
                                          verdict=_verdict(open_gate["unaccounted_loss"]
                                                           >= open_gate["min_loss"])),
        "particles_removed": dict(value=float(open_gate["particles_removed"]),
                                  unit="1"),
        "gate_pass": dict(value=float(open_gate["pass"]), unit="bool",
                          gate_limit=1.0, verdict=_verdict(open_gate["pass"])),
    })
    store.add_series(open_id, "mass_history", opened["history"])

    out = {"gravity_off": ctrl, "gravity_off_pass": ctrl_pass,
           "mass_closed": closed["measured"], "mass_closed_gate": closed_gate,
           "mass_open": opened["measured"], "mass_open_gate": open_gate}
    (HERE / "results_controls.json").write_text(json.dumps(out, indent=1))
    return out


def make_figures(store: EvidenceStore, table: dict) -> list[str]:
    runs = store.query("SELECT run_id, config_json FROM runs WHERE case_name='dam_break_2d'"
                       " ORDER BY run_id")
    if not runs:
        raise SystemExit("no dam-break runs in the store")
    curves = []
    ritter = None
    for row in runs:
        payload = store.query("SELECT payload FROM series WHERE run_id=?"
                             " AND name='front_curve'", (row["run_id"],))
        if not payload:
            continue
        series = json.loads(payload[0]["payload"])
        curves.append(series)
        ritter = series["ritter"]
    finest = runs[-1]["run_id"]
    prof = store.query("SELECT payload FROM series WHERE run_id=? AND name='profiles'",
                       (finest,))
    if not prof:
        raise SystemExit("no profile series; rerun the spacing stage")
    store.add_series(finest, "front", {
        "tau": curves[0]["tau"], "ritter": ritter,
        "fronts": [{"dx": c["dx"], "particles": c["particles"],
                    "tau": c["tau"], "Z_over_a": c["Z_over_a"]} for c in curves]})
    figs = HERE / "figures"
    png = report.figure_dam_break(store, finest, figs / "dam_break_front_profile.png",
                                  table)
    closed = store.query(
        "SELECT run_id FROM runs WHERE case_name='mass_conservation_dam_break'"
        " ORDER BY run_id DESC LIMIT 1")
    opened = store.query(
        "SELECT run_id FROM runs WHERE case_name='mass_conservation_dam_break_open'"
        " ORDER BY run_id DESC LIMIT 1")
    out = [str(png.relative_to(HERE))]
    if closed and opened:
        png2 = report.figure_mass(store, int(closed[0]["run_id"]),
                                  int(opened[0]["run_id"]),
                                  figs / "mass_bookkeeping.png")
        out.append(str(png2.relative_to(HERE)))
    return out


def resolution_check(store: EvidenceStore) -> dict:
    """Self-convergence of the free-surface profile between the two finest spacings.

    Reported, not gated (the campaign's convergence gate is the static kernel
    test in ``benchmarks/verified/sph_convergence``).  The comparison is
    bin-by-bin over the wetted length, so a different wetted *extent* between the
    two spacings enters the norm as well: the number is an upper bound on the
    shape difference, not a clean estimate of it.  It is reported because it
    answers the question a failing front gate raises -- "is this a resolution
    error?" -- and the answer from the front curves themselves is no (they nearly
    coincide in the early window).
    """
    runs = store.query("SELECT run_id, config_json FROM runs WHERE case_name="
                       "'dam_break_2d' ORDER BY run_id")
    by_dx = {}
    for row in runs:
        payload = store.query("SELECT payload FROM series WHERE run_id=?"
                              " AND name='profiles'", (row["run_id"],))
        if not payload:
            continue
        dx = json.loads(row["config_json"]).get("dx")
        by_dx[float(dx)] = json.loads(payload[0]["payload"])
    if len(by_dx) < 2:
        raise SystemExit("need at least two spacings for the resolution check")
    spacings = sorted(by_dx)
    coarse, fine = by_dx[spacings[0]], by_dx[spacings[-1]]
    diffs = []
    for tau in [1.0, 2.0, 3.0]:
        pc = min(coarse, key=lambda r: abs(r["tau"] - tau))
        pf = min(fine, key=lambda r: abs(r["tau"] - tau))
        if abs(pc["tau"] - tau) > 0.2 or abs(pf["tau"] - tau) > 0.2:
            continue
        a = [0.0 if v is None else v for v in pc["h"]]
        b = [0.0 if v is None else v for v in pf["h"]]
        n = min(len(a), len(b))
        num = sum((x - y) ** 2 for x, y in zip(a[:n], b[:n]))
        den = sum(y * y for y in b[:n])
        diffs.append({"tau": float(tau), "dx_coarse": spacings[0],
                      "dx_fine": spacings[-1],
                      "l2_relative": math.sqrt(num / den) if den > 0 else float("nan")})
    mean = sum(d["l2_relative"] for d in diffs) / len(diffs) if diffs else float("nan")
    run_id = store.add_run("dam_break_2d_resolution_check",
                           {"dx_coarse": diffs[0]["dx_coarse"],
                            "dx_fine": diffs[-1]["dx_fine"]},
                           notes="profile self-convergence between the two finest "
                                 "spacings; reported, not gated")
    store.add_metrics(run_id, {
        "mean_profile_l2_between_spacings": dict(value=mean, unit="1",
                                                 provenance="relative L2 of the "
                                                            "surface profile at "
                                                            "tau = 1,2,3, bin-by-bin "
                                                            "over the wetted length; "
                                                            "an upper bound, because "
                                                            "a differing extent also "
                                                            "counts"),
        "max_profile_l2_between_spacings": dict(
            value=max((d["l2_relative"] for d in diffs), default=float("nan")),
            unit="1"),
    })
    store.add_series(run_id, "profile_l2", diffs)
    print(f"resolution check: mean profile L2 {mean:.3f} between dx="
          f"{diffs[0]['dx_coarse']:g} and {diffs[-1]['dx_fine']:g}", flush=True)
    return {"run_id": run_id, "diffs": diffs, "mean": mean}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacings", default="0.02,0.01")
    ap.add_argument("--seconds", type=float, default=0.45)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--resolution-check", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    table = m1.load_experimental()
    store = EvidenceStore(STORE)
    store.add_reference("martin_moyce_1952", table["_path"], table["_grade"],
                        (table.get("integrity_audit") or {}).get("finding", "")[:200])
    failed = []
    if args.resolution_check:
        resolution_check(store)
    if not args.figures and not args.resolution_check:
        for text in args.spacings.split(","):
            if not text.strip():
                continue
            dx = float(text)
            out = run_spacing(store, dx, args.seconds, table)
            print(f"dx={dx:g}: front max rel "
                  f"{out['front']['max_rel_error']:.3f} "
                  f"(rms {out['front']['rms_rel_error']:.3f}), "
                  f"height residual {out['height']['residual_max_rel_error']:.3f}, "
                  f"ballistic {out['height']['ballistic_max_rel_error']:.3f}",
                  flush=True)
            if not (out["gates"]["front"] and out["gates"]["height"]):
                failed.append(f"dx={dx:g}")
    if args.controls:
        ctrl = run_controls(store, table, float(args.spacings.split(",")[0]))
        print(f"gravity-off control pass={ctrl['gravity_off_pass']} "
              f"(front drift {ctrl['gravity_off']['front_drift_over_dx']:.3f} dx), "
              f"mass closed pass={ctrl['mass_closed_gate']['pass']}, "
              f"mass open pass={ctrl['mass_open_gate']['pass']} "
              f"(unaccounted loss {ctrl['mass_open_gate']['unaccounted_loss']:.3f})",
              flush=True)
        if not ctrl["gravity_off_pass"]:
            failed.append("gravity_off")
        if not ctrl["mass_closed_gate"]["pass"]:
            failed.append("mass_closed")
        if not ctrl["mass_open_gate"]["pass"]:
            failed.append("mass_open")
    if args.figures:
        print("figures:", ", ".join(make_figures(store, table)), flush=True)
    store.close()
    if failed:
        print("FAILED gates:", ", ".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
