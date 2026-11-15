"""Does a wall-adhesion traction deliver the set contact angle? (Young, not just Laplace.)

``docs/verification_contact_angle_bc.md`` closed the shipped contact-angle BC as a
physics FAIL: with the admissibility gates satisfied (``R/dx = 13.3``, wall
penetration ``0.00 dx``, spurious current ``0.03``), set angles 60/90/120 deg
relax to 151.4/152.6/154.1 deg, response slope ``-0.025``.  The mechanism is in
the same document: ``apply_contact_angle_boundary`` rotates the near-wall CSF
force *direction* and keeps ``|grad c|``, so the amplitude stays a curvature
force and no per-unit-contact-line traction is ever applied -- and Young's law
``gamma_sv - gamma_sl = sigma cos theta`` *is* a per-unit-length traction.

This harness tests the replacement of that BC by an explicit wall-adhesion
traction (:mod:`tensorsph.contact_angle_isph`) in three arms, all on the same
production solver (``tensorsph.droplet_impact.rhs``), the same instrument
(``benchmarks/contact_angle/run.py::measure_contact_angle``, verified to 1e-5 deg
on analytic interfaces) and the same admissibility gates:

* ``rotate``    -- the shipped BC (baseline; reproduces the FAIL)
* ``young``     -- ``C sigma cos(theta)`` along the wall, away from the liquid
* ``brackbill`` -- ``C sigma (n_wall - cos(theta) n)``, the amplitude form of the
  same literature, as an audited control

Pre-registered gates (fixed before any relaxation is run, asserted in
``tests/test_contact_angle_adhesion_m22237.py``):

``G1`` analytic traction audit: for every ``theta``, ``T / (sigma cos theta)``
    inside ``[0.85, 1.20]`` for the ``young`` model, and the measured calibration
    spread across ``theta`` inside 20 %.
``G2`` relaxation: on admissible runs, ``|theta_meas - theta_set| <= 10 deg``.
``G3`` response: ``d theta_meas / d theta_set`` in ``[0.7, 1.3]`` -- a BC that
    delivers the set angle must have unit slope (the FAIL had ``-0.025``).
``G4`` mechanism: the ``young`` arm's slope must beat the ``rotate`` arm's by at
    least ``0.5``; a traction that does not move the contact line is not a
    contact-angle model.

Usage::

    python benchmarks/contact_angle/run_adhesion.py --audit
    python benchmarks/contact_angle/run_adhesion.py --shard 0 --n 8 --t-end-factor 4
    python benchmarks/contact_angle/run_adhesion.py --merge --verdict
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorsph.contact_angle_isph import (
    DEFAULT_TRACTION_CALIBRATION,
    traction_audit_ladder,
    wall_delta_reference,
)
from tensorsph.droplet_impact import rhs, stable_dt
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "adhesion"

#: Pre-registered gates.  Changing any of these after a run is a different claim.
ANGLE_GATE_DEG = 10.0
SLOPE_BAND = (0.7, 1.3)
TRACTION_RATIO_BAND = (0.85, 1.20)
CALIBRATION_SPREAD_GATE = 0.20
SLOPE_MARGIN_OVER_BASELINE = 0.5
WALL_PENETRATION_GATE_DX = 0.5     # same as run.py / diagnose_bc.py
SPURIOUS_CURRENT_GATE = 0.1        # v_max / sqrt(sigma / (rho R)), Adami et al. 2012

MODELS = ("rotate", "young", "brackbill")
ANGLES_DEG = (30.0, 60.0, 90.0, 120.0)
DEFAULT_DX = (0.003, 0.0015)
T_END_FACTOR = 4.0


def _run_module():
    """``benchmarks/contact_angle/run.py`` -- the instrument and the drop setup."""
    spec = importlib.util.spec_from_file_location("ca_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ca_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def cases(dx_values, models, angles, t_end_factor):
    out = []
    for model in models:
        for dx in dx_values:
            for theta in angles:
                out.append({"model": model, "dx_m": float(dx),
                            "theta_set_deg": float(theta),
                            "t_end_factor": float(t_end_factor)})
    return out


def relax_case(case: dict, device: torch.device) -> dict:
    """Relax one drop and read the contact angle.  One ``CASE {}`` JSON line."""
    ca = _run_module()
    theta = math.radians(case["theta_set_deg"])
    cfg, h, pos, mass, wall, x0 = ca.make_drop_on_wall(theta, device)
    cfg.dx = float(case["dx_m"])
    cfg.contact_angle_model = case["model"]
    dx = cfg.dx

    t0 = time.time()
    dt = float(stable_dt(cfg))
    t_end = float(case["t_end_factor"]) * ca.T_SIGMA
    steps = int(math.ceil(t_end / dt))
    dt = t_end / steps
    vel = torch.zeros_like(pos)
    trace = {"t_over_t_sigma": [], "theta_deg": [], "v_max_over_v_ref": []}
    v_ref = math.sqrt(ca.SIGMA / (ca.RHO0 * ca.RADIUS))
    sample = max(1, steps // 40)
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        _, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample == 0 or step == steps - 1:
            m = ca.measure_contact_angle(pos[:, 0], pos[:, 1], x0, dx)
            trace["t_over_t_sigma"].append((step + 1) * dt / ca.T_SIGMA)
            trace["theta_deg"].append(m["theta_deg"])
            trace["v_max_over_v_ref"].append(float(vel.norm(dim=1).max()) / v_ref)

    finite = [v for v in trace["theta_deg"] if v == v]
    tail = finite[-max(1, len(finite) // 4):]
    theta_meas = sum(tail) / len(tail) if tail else float("nan")
    err = theta_meas - case["theta_set_deg"]
    penetration = max(0.0, -float(pos[:, 1].min())) / dx
    spurious = trace["v_max_over_v_ref"][-1]
    admissible = bool(penetration < WALL_PENETRATION_GATE_DX
                      and spurious < SPURIOUS_CURRENT_GATE
                      and theta_meas == theta_meas)
    row = dict(case)
    row.update({
        "theta_meas_deg": theta_meas,
        "error_deg": err,
        "within_angle_gate": bool(abs(err) <= ANGLE_GATE_DEG) if err == err else False,
        "admissible": admissible,
        "wall_penetration_dx": penetration,
        "spurious_current_ratio": spurious,
        "n_particles": int(pos.shape[0]),
        "steps": steps,
        "dt_s": dt,
        "wallclock_s": time.time() - t0,
        "calibration": DEFAULT_TRACTION_CALIBRATION,
        "trace": trace,
    })
    print("CASE " + json.dumps(row), flush=True)
    return row


def run_audit(dx_values=DEFAULT_DX) -> dict:
    out = {"delta_w_ref": {str(dx): wall_delta_reference(dx) for dx in dx_values},
           "ladders": {}}
    for model in ("young", "brackbill"):
        for dx in dx_values:
            ladder = traction_audit_ladder(dx, angles_deg=ANGLES_DEG, model=model)
            out["ladders"][f"{model}:{dx}"] = ladder
    return out


def _load_rows(pattern: str = "results_*.json") -> list[dict]:
    rows = []
    for path in sorted(OUT.glob(pattern)):
        for row in json.loads(path.read_text())["rows"]:
            rows.append(row)
    return rows


def merge() -> dict:
    rows = _load_rows()
    payload = {"rows": rows, "n_rows": len(rows),
               "gates": {"angle_deg": ANGLE_GATE_DEG, "slope_band": SLOPE_BAND,
                         "traction_ratio_band": TRACTION_RATIO_BAND,
                         "calibration_spread_gate": CALIBRATION_SPREAD_GATE,
                         "slope_margin_over_baseline": SLOPE_MARGIN_OVER_BASELINE,
                         "wall_penetration_gate_dx": WALL_PENETRATION_GATE_DX,
                         "spurious_current_gate": SPURIOUS_CURRENT_GATE},
               "audit": run_audit()}
    (OUT / "adhesion_merged.json").write_text(json.dumps(payload, indent=1))
    return payload


def _slope(rows: list[dict]) -> float:
    """Least-squares slope of theta_meas against theta_set over admissible rows."""
    pts = [(r["theta_set_deg"], r["theta_meas_deg"]) for r in rows if r["admissible"]]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else float("nan")


def verdict(payload: dict) -> dict:
    rows = payload["rows"]
    per_model = {}
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        adm = [r for r in sub if r["admissible"]]
        per_model[model] = {
            "n_runs": len(sub),
            "n_admissible": len(adm),
            "slope": _slope(sub),
            "worst_abs_error_deg": (max(abs(r["error_deg"]) for r in adm)
                                    if adm else float("nan")),
            "all_within_angle_gate": bool(adm) and all(r["within_angle_gate"]
                                                       for r in adm),
            "rows": [{"model": r["model"], "dx_m": r["dx_m"],
                      "theta_set_deg": r["theta_set_deg"],
                      "theta_meas_deg": r["theta_meas_deg"],
                      "error_deg": r["error_deg"],
                      "admissible": r["admissible"],
                      "wall_penetration_dx": r["wall_penetration_dx"],
                      "spurious_current_ratio": r["spurious_current_ratio"]}
                     for r in sub],
        }

    audit = payload["audit"]["ladders"]
    g1_rows = []
    for key, ladder in audit.items():
        if not key.startswith("young"):
            continue
        for row in ladder["rows"]:
            g1_rows.append({"dx_m": row["dx_m"], "theta_deg": row["theta_deg"],
                            "traction_ratio": row["traction_ratio"],
                            "within_band": bool(TRACTION_RATIO_BAND[0]
                                                <= row["traction_ratio"]
                                                <= TRACTION_RATIO_BAND[1])})
    g1 = bool(g1_rows) and all(r["within_band"] for r in g1_rows)
    spread = max((ladder["calibration_spread_rel"]
                  for key, ladder in audit.items() if key.startswith("young")),
                 default=float("nan"))

    young = per_model.get("young", {})
    base = per_model.get("rotate", {})
    gate = {
        "G1_traction_audit": g1,
        "G1_calibration_spread": bool(spread == spread
                                      and spread <= CALIBRATION_SPREAD_GATE),
        "G2_angle_gate": bool(young.get("all_within_angle_gate", False)),
        "G3_unit_slope": bool(young.get("slope") == young.get("slope")
                              and SLOPE_BAND[0] <= young.get("slope", float("nan"))
                              <= SLOPE_BAND[1]),
        "G4_beats_baseline": bool(young.get("slope") == young.get("slope")
                                  and base.get("slope") == base.get("slope")
                                  and young["slope"] - base["slope"]
                                  >= SLOPE_MARGIN_OVER_BASELINE),
    }
    return {"per_model": per_model, "gates": gate,
            "traction_ratio_rows": g1_rows,
            "calibration_spread_rel": spread,
            "verdict": ("PASS" if all(gate.values()) else
                        ("FAIL" if any(v is False for v in gate.values())
                         else "WITHHELD"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--dx", type=float, nargs="*", default=list(DEFAULT_DX))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--angles", type=float, nargs="*", default=list(ANGLES_DEG))
    ap.add_argument("--t-end-factor", type=float, default=T_END_FACTOR)
    ap.add_argument("--tag", default="", help="output tag, so two ladders do not collide")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    if args.audit:
        payload = run_audit(args.dx)
        (OUT / "audit.json").write_text(json.dumps(payload, indent=1))
        print(json.dumps(payload, indent=1)[:2000])
        return

    if args.merge:
        payload = merge()
        print(json.dumps({"n_rows": payload["n_rows"]}, indent=1))
        return

    if args.verdict:
        payload = merge()
        result = verdict(payload)
        (OUT / "verdict.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return

    todo = cases(args.dx, args.models, args.angles, args.t_end_factor)
    if args.shard is not None:
        todo = [c for i, c in enumerate(todo) if i % args.n == args.shard]
    rows = [relax_case(c, torch.device("cpu")) for c in todo]
    tag = (f"results_{args.tag}" if args.tag else "results") + \
        f"_{args.shard if args.shard is not None else 0}.json"
    (OUT / tag).write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()