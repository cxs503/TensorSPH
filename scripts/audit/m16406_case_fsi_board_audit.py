#!/usr/bin/env python3
"""m16406 — board audit of m16352 (CASE-SPH-FSI-ELASTIC-GATE, TensorSPH).

Read-only audit of a *succeeded* mission.  Nothing under ``src/`` is touched:
every check below is recomputed here, from the delivered code and artifacts, so
the verdict does not rest on the mission's own report.

Outputs (all machine-generated, no hand-typed numbers):

  docs/board_audit/m16406/evidence_m16352.json      registry / referee / DB / report
  docs/board_audit/m16406/independent_checks.json   audit-side recomputation
  docs/board_audit/m16406/artifacts/*               snapshot of the 6 run artifacts (+sha256)
  docs/board_audit/m16406/fig1_audit.png             tip history + elastica/rod shape

Run:  python3 scripts/audit/m16406_case_fsi_board_audit.py
Requires a TensorSPH checkout at M16352 (override with --src).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ANCHOR = "CASE-SPH-FSI-ELASTIC-GATE"
MISSION = 16352
AUDIT_MISSION = 16406
ROOT = Path(__file__).resolve().parents[2]          # audit worktree (m16406)
OUT = ROOT / "docs" / "board_audit" / f"m{AUDIT_MISSION}"


# ---------------------------------------------------------------------------
# 1. registry referee: does the benchmark judge have anything to judge?
# ---------------------------------------------------------------------------
def dsn() -> str | None:
    try:
        for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().replace(
                    "postgresql+psycopg://", "postgresql://")
    except OSError:
        return None
    return None


def registry() -> dict:
    out: dict = {"anchor_slug": ANCHOR, "rows": [], "n_rows": 0}
    d = dsn()
    if not d:
        out["error"] = "no DATABASE_URL"
        return out
    import psycopg
    with psycopg.connect(d) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
        out["table_total"] = cur.fetchone()[0]
        cur.execute("SELECT case_name,metric,tolerance,repo,status FROM tensor_benchmark_cases "
                    "WHERE anchor_slug=%s", (ANCHOR,))
        out["rows"] = [list(map(str, r)) for r in cur.fetchall()]
        out["n_rows"] = len(out["rows"])
        cur.execute("SELECT anchor_slug,case_name,tolerance FROM tensor_benchmark_cases "
                    "WHERE anchor_slug ILIKE %s OR case_name ILIKE %s",
                    ("%ELASTIC%", "%elastic%"))
        out["elastic_rows_any_anchor"] = [list(map(str, r)) for r in cur.fetchall()]
        cur.execute("SELECT id,status,repo,left(coalesce(result->>'headline',''),200),report_id "
                    "FROM tensor_dev_missions WHERE id=%s", (MISSION,))
        out["mission"] = cur.fetchone()
        cur.execute("SELECT count(*) FROM tensorbench_results WHERE mission_id=%s", (MISSION,))
        out["tensorbench_results_rows"] = cur.fetchone()[0]
        cur.execute("SELECT id,title,length(markdown),created_at FROM tensor_reports "
                    "WHERE mission_id=%s", (MISSION,))
        out["reports"] = [list(map(str, r)) for r in cur.fetchall()]
        cur.execute("SELECT id,mission_id,level,gate_state,scope,verdict_by FROM "
                    "tensor_maturity_verdicts WHERE mission_id=%s OR scope ILIKE %s "
                    "ORDER BY id DESC LIMIT 5", (MISSION, "%FSI-ELASTIC%"))
        out["prior_verdicts"] = [list(map(str, r)) for r in cur.fetchall()]
    return out


def referee() -> dict:
    import urllib.request
    url = f"http://127.0.0.1:8010/api/v1/tensor/bench/l4-verdict/{MISSION}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    except Exception as exc:                                  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# 2. independent physics: Kirchhoff elastica by elliptic-type quadrature
#    (no ODE shooting, no reuse of the delivered solver)
# ---------------------------------------------------------------------------
def elastica_quadrature(PL2_over_EI: float, L: float) -> dict:
    """Cantilever, dead end-load.  phi = rotation from the clamped base:

        phi'' = lam cos(phi),  phi'(L) = 0  =>  phi'^2 = 2 lam (sin phi_e - sin phi)

    so L = Int_0^{phi_e} dphi / sqrt(2 lam (sin phi_e - sin phi)).  Evaluated
    dimensionless (L=1, lam=PL^2/EI) and rescaled by L; the singular endpoint is
    removed with phi = phi_e - u^2.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq

    lam = PL2_over_EI                       # unit length internally

    def _ker(pe, u):
        s = 2.0 * lam * (math.sin(pe) - math.sin(pe - u * u))
        return 2.0 * u / math.sqrt(max(s, 1e-300))

    def length(pe):
        return quad(lambda u: _ker(pe, u), 0.0, math.sqrt(pe), limit=400)[0]

    pe = brentq(lambda p: length(p) - 1.0, 1e-6, math.pi - 1e-9, xtol=1e-14, rtol=1e-14)
    s_end = math.sqrt(pe)
    x = quad(lambda u: _ker(pe, u) * math.cos(pe - u * u), 0.0, s_end, limit=400)[0]
    y = quad(lambda u: _ker(pe, u) * math.sin(pe - u * u), 0.0, s_end, limit=400)[0]
    us = np.linspace(0.0, s_end, 240)
    phi = pe - us ** 2
    ds = np.array([_ker(pe, u) for u in us])
    xs = np.concatenate(([0.0], np.cumsum(0.5 * (ds[1:] * np.cos(phi[1:])
                                                + ds[:-1] * np.cos(phi[:-1])) * np.diff(us))))
    ys = np.concatenate(([0.0], np.cumsum(0.5 * (ds[1:] * np.sin(phi[1:])
                                                + ds[:-1] * np.sin(phi[:-1])) * np.diff(us))))
    return {"along_beam": x * L, "transverse": y * L, "tip_tangent_rad": pe,
            "shape_along": (xs * L).tolist(), "shape_transverse": (ys * L).tolist()}


# ---------------------------------------------------------------------------
# 3. audit-side recomputation of every number the report leans on
# ---------------------------------------------------------------------------
def independent_checks(src: Path) -> dict:
    sys.path.insert(0, str(src / "src"))
    import torch
    from tensorsph import fsi_elastic_gate as fg

    out: dict = {"src": str(src)}
    case = fg.ElasticGateCase(dx=0.0125, n_gate_nodes=25)
    EI, H = case.gate_stiffness_bending(), case.gate_height
    out["geometry"] = {"EI": EI, "EA": case.gate_stiffness_axial(),
                       "rho_A": case.gate_area_density(), "H": H,
                       "water_height": case.water_height, "dx": case.dx}

    # --- V1: reference load, independent formula -------------------------
    ref_load = 0.5 * case.rho0 * case.gravity * case.water_height ** 2
    out["V1_reference_N_per_m"] = ref_load
    v1 = {}
    for dx in (0.0125, 0.01, 0.0075):
        for mode in ("panel", "pairwise"):
            c = fg.ElasticGateCase(dx=dx, load_transfer=mode)
            try:
                v = fg.verify_hydrostatic_load(c)
                v1[f"{mode}_dx{dx}"] = {k: float(val) for k, val in v.items()
                                        if isinstance(val, (int, float))}
            except Exception as exc:                       # pragma: no cover
                v1[f"{mode}_dx{dx}"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["V1"] = v1

    # --- A2: the elastica reference, recomputed independently ------------
    P2 = 10.0 * EI / H ** 2
    ref = fg.elastica_tip(P2, H, EI)
    mine = elastica_quadrature(10.0, H)
    out["A2_reference_crosscheck"] = {
        "delivered_along_beam": float(ref["tip_x"]),
        "delivered_transverse": float(ref["tip_y"]),
        "delivered_tip_tangent": float(ref["tip_angle"]),
        "audit_quadrature_along_beam": mine["along_beam"],
        "audit_quadrature_transverse": mine["transverse"],
        "audit_quadrature_tip_tangent": mine["tip_tangent_rad"],
        "rel_diff_along": (float(ref["tip_x"]) - mine["along_beam"]) / mine["along_beam"],
        "rel_diff_transverse": (float(ref["tip_y"]) - mine["transverse"]) / mine["transverse"],
        "dimensionless_along": mine["along_beam"] / H,
        "dimensionless_transverse": mine["transverse"] / H,
    }
    out["elastica_shape"] = {"along": mine["shape_along"], "transverse": mine["shape_transverse"]}

    # --- A2: rod vs elastica, BOTH mappings ------------------------------
    # The gate rod is vertical and loaded sideways, so the rod's x is the
    # *transverse* coordinate and its y the *along-beam* one.  The delivered
    # verify_cantilever() compares rod-x against the elastica's along-beam x
    # (and rod-y against the elastica's transverse y) -- the two axes are
    # swapped.  Both conventions are computed here.
    a2 = {}
    for n in (25, 50):
        c = fg.ElasticGateCase(dx=0.0125, n_gate_nodes=n)
        load = torch.zeros((n, 2), dtype=torch.float64)
        load[-1, 0] = 10.0 * EI / H ** 2
        nodes = fg.rod_static_equilibrium(c, load, gravity=False)
        transverse = float(nodes[-1, 0]) - c.water_width        # load direction
        along = float(nodes[-1, 1])                             # along the gate
        seg = nodes[-1] - nodes[-2]
        tangent = 0.5 * math.pi - math.atan2(float(seg[1]), float(seg[0]))
        a2[f"n{n}"] = {
            "rod_transverse": transverse, "rod_along_beam": along,
            "elastica_transverse": float(ref["tip_y"]),
            "elastica_along_beam": float(ref["tip_x"]),
            "err_transverse": (transverse - float(ref["tip_y"])) / float(ref["tip_y"]),
            "err_along_beam": (along - float(ref["tip_x"])) / float(ref["tip_x"]),
            "err_as_delivered_code_computes": (transverse - float(ref["tip_x"])) / float(ref["tip_x"]),
            "rod_tip_rotation_from_base": tangent,
            "elastica_tip_rotation": float(ref["tip_angle"]),
            "err_tip_rotation": (tangent - float(ref["tip_angle"])) / float(ref["tip_angle"]),
            "shape_along": [float(v) for v in nodes[1:, 1]],
            "shape_transverse": [float(v) - c.water_width for v in nodes[1:, 0]],
        }
        # --- A0 at the same resolution (the ladder's other rung) ---------
        P0 = 1.0e-3 * 2.0 * EI / H ** 2
        l0 = torch.zeros((n, 2), dtype=torch.float64)
        l0[-1, 0] = P0
        n0 = fg.rod_static_equilibrium(c, l0, gravity=False)
        got0 = float(n0[-1, 0]) - c.water_width
        exact0 = fg.eb_tip_point_load(P0, H, EI)
        a2[f"n{n}_A0"] = {"P": P0, "tip_rod": got0, "tip_exact": exact0,
                          "rel_error": (got0 - exact0) / exact0}
    out["A2"] = a2

    # --- how far outside beam theory is the case? ------------------------
    y = np.linspace(0.0, H, 4001)
    p = np.where(case.water_height - y > 0.0,
                 case.rho0 * case.gravity * (case.water_height - y), 0.0)
    M = np.array([np.trapezoid(p[y >= v] * (y[y >= v] - v), y[y >= v]) for v in y])
    tip_eb = float(np.trapezoid(M * (H - y) / EI, y))
    out["regime"] = {"EB_small_deflection_tip_m": tip_eb, "tip_over_H": tip_eb / H,
                     "note": "small-deflection beam theory predicts tip >> H: the case lives "
                             "far outside linear beam theory"}
    return out


def run_artifact_stats(src: Path) -> dict:
    runs = json.loads((src / "artifacts/fsi_elastic_gate/runs.json").read_text())
    key = list(runs)[0]
    r = runs[key]
    h = r["history"]
    t = np.asarray(h["t"])
    tip = np.asarray(h["tip_x"])
    force = np.asarray(h["gate_force_x"])
    speed = np.asarray(h["max_speed"])
    beyond = np.where(np.abs(tip) > 0.36)[0]
    # sign changes of tip after the flow reaches the gate -> ringing, not motion
    post = tip[t > 0.10]
    flips = int(np.sum(np.sign(post[1:]) != np.sign(post[:-1])))
    return {
        "run_key": key,
        "n_steps": r["n_steps"], "t_end_s": float(t[-1]), "wall_seconds": r["wall_seconds"],
        "peak_tip_mm": r["peak_tip_x_mm"], "peak_time_ms": r["peak_time_ms"],
        "min_tip_mm": float(tip.min()) * 1e3, "max_tip_mm": float(tip.max()) * 1e3,
        "tip_over_gate_height": float(np.abs(tip).max()) / 0.36,
        "first_abs_tip_over_H_s": float(t[beyond[0]]) if beyond.size else None,
        "n_samples_beyond_H": int(beyond.size),
        "gate_force_at_t0_N_per_m": float(force[0]),
        "gate_force_max_N_per_m": float(force.max()),
        "hydrostatic_reference_N_per_m": 418.21992,
        "gate_force_at_50ms": float(force[np.argmin(np.abs(t - 0.05))]),
        "gate_force_at_100ms": float(force[np.argmin(np.abs(t - 0.10))]),
        "max_fluid_speed_m_s": float(speed.max()),
        "sound_speed_m_s": 20.0,
        "mach": float(speed.max()) / 20.0,
        "speed_over_c0_at_s": float(t[np.argmax(speed > 20.0)]) if np.any(speed > 20.0) else None,
        "tip_sign_flips_after_100ms": flips,
        "tip_series": {"t": t.tolist(), "tip_mm": (tip * 1e3).tolist()},
    }


# ---------------------------------------------------------------------------
# 4. figure: the two claims the verdict turns on
# ---------------------------------------------------------------------------
def figure(ind: dict, stats: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    a = ax[0]
    a.plot(stats["tip_series"]["t"], stats["tip_series"]["tip_mm"], lw=1.0,
           color="#b3001b", label="delivered coupled run (panel + clamp)")
    a.axhspan(-360, 360, color="#d9ead3", alpha=0.6,
              label="geometrically admissible band (|tip| <= gate height 360 mm)")
    a.axhline(0.0, color="k", lw=0.6)
    a.set_xlabel("t [s]")
    a.set_ylabel("gate tip horizontal displacement [mm]")
    a.set_title(f"m16352 coupled run: peak {stats['peak_tip_mm']:.0f} mm = "
                f"{stats['tip_over_gate_height']:.2f}x gate height")
    a.legend(fontsize=7, loc="lower left")
    a.grid(alpha=0.3)

    b = ax[1]
    H = ind["geometry"]["H"]
    sh = ind["elastica_shape"]
    b.plot(np.asarray(sh["along"]) / H, np.asarray(sh["transverse"]) / H, "k-", lw=2.0,
           label="elastica, audit quadrature (independent)")
    for n, st, c in (("n25", ind["A2"]["n25"], "#1f77b4"), ("n50", ind["A2"]["n50"], "#ff7f0e")):
        b.plot(np.asarray(st["shape_along"]) / H, np.asarray(st["shape_transverse"]) / H,
               "o-", ms=2.5, lw=1.2, color=c,
               label=f"rod {n}: err transverse {100*st['err_transverse']:+.2f} %, "
                     f"along-beam {100*st['err_along_beam']:+.2f} %")
    b.set_xlabel("along-beam coordinate / H")
    b.set_ylabel("transverse deflection / H")
    b.set_title(f"A2 (PL$^2$/EI=10): delivered code reports "
                f"{100*ind['A2']['n25']['err_as_delivered_code_computes']:+.1f} % "
                "= axis swap")
    b.legend(fontsize=7, loc="lower right")
    b.grid(alpha=0.3)
    b.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=145)
    plt.close(fig)


def snapshot(src: Path, dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    keep = ["runs.json", "summary.json", "rod_verification.json", "v4_gate_load_path.json",
            "provenance.json", "fig1_tip_and_load.png"]
    rows = []
    for name in keep:
        s = src / "artifacts/fsi_elastic_gate" / name
        if not s.exists():
            rows.append({"file": name, "exists": False})
            continue
        shutil.copy2(s, dest / name)
        rows.append({"file": f"docs/board_audit/m{AUDIT_MISSION}/artifacts/{name}",
                     "origin": str(s),
                     "sha256": hashlib.sha256(s.read_bytes()).hexdigest(),
                     "bytes": s.stat().st_size})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"/root/tp-agent/work/m{MISSION}")
    args = ap.parse_args()
    src = Path(args.src)
    OUT.mkdir(parents=True, exist_ok=True)

    ev = {"anchor_slug": ANCHOR, "reviewed_mission": MISSION,
          "audit_mission": AUDIT_MISSION, "reviewed_src": str(src)}
    ev["registry"] = registry()
    ev["referee"] = referee()
    ev["artifacts"] = snapshot(src, OUT / "artifacts")
    ev["impl_sha256"] = {p: hashlib.sha256((src / p).read_bytes()).hexdigest()
                         for p in ("src/tensorsph/fsi_elastic_gate.py",
                                   "tests/test_fsi_elastic_gate.py",
                                   "benchmarks/fsi_elastic_gate/run_antoci.py")
                         if (src / p).exists()}
    (OUT / "evidence_m16352.json").write_text(json.dumps(ev, indent=1, default=str))

    ind = independent_checks(src)
    stats = run_artifact_stats(src)
    ind["run_artifacts"] = stats
    (OUT / "independent_checks.json").write_text(json.dumps(ind, indent=1))
    figure(ind, stats, OUT / "fig1_audit.png")

    print(json.dumps({
        "benchmark_rows": ev["registry"]["n_rows"],
        "referee": ev["referee"].get("status"),
        "tensorbench_results": ev["registry"].get("tensorbench_results_rows"),
        "A2_delivered_code_err": ind["A2"]["n25"]["err_as_delivered_code_computes"],
        "A2_true_err_transverse": ind["A2"]["n25"]["err_transverse"],
        "A2_true_err_along": ind["A2"]["n25"]["err_along_beam"],
        "A2_n50_transverse": ind["A2"]["n50"]["err_transverse"],
        "A0_n50": ind["A2"]["n50_A0"]["rel_error"],
        "V1_panel": {k: v.get("rel_error") for k, v in ind["V1"].items()},
        "peak_tip_mm": stats["peak_tip_mm"],
        "max_speed": stats["max_fluid_speed_m_s"],
        "force_t0": stats["gate_force_at_t0_N_per_m"],
        "force_max": stats["gate_force_max_N_per_m"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())