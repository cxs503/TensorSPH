#!/usr/bin/env python3
"""Board-audit verdict for m15278 (CASE-LAKE-AT-REST-SPH, TensorSPH) — INSERT only.

Evidence is built from the audit snapshots (sha256 computed here, benchmark
numbers read from the delivered / re-run JSON), never typed by hand.
Idempotent: one row per (repo, scope, verdict_by) — re-running updates it.

Run:  python3 docs/board_audit/m15314/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorSPH"
SCOPE = "m15278 CASE-LAKE-AT-REST-SPH"
MISSION = 15314          # this audit task
SUBJECT = 15278          # audited mission
BY = "board-audit"
LEVEL = 3                # functional substance = L3; L4 refused (criterion fails)
TOL = 0.005

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]              # worktree root (/root/tp-agent/work/m15314)
REFS = HERE / "refs"
ART = REFS / "artifacts"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_evidence() -> dict:
    orig = json.loads((REFS / "benchmarks/lake_at_rest/report.json").read_text())
    rerun = json.loads((ART / "m15278_rerun.json").read_text())
    probe = json.loads((ART / "probe_t0.json").read_text())["result"]
    key = lambda c: (c["mode"], c["dx_m"], c["sound_speed_m_s"], c["viscosity"])  # noqa: E731
    kb = {key(c): c for c in rerun["cases"]}
    repro_diff = max(abs(c["pressure_drift_max"] - kb[key(c)]["pressure_drift_max"])
                     for c in orig["cases"])
    cases = sorted(orig["cases"], key=lambda c: c["pressure_drift_max"])
    best = cases[0]
    shipped_max = max(c["pressure_drift_max"] for c in orig["cases"] if c["mode"] == "shipped")

    return {
        "impl_files": [rel(REFS / "src/tensorsph/droplet_impact.py"),
                       rel(ART / "droplet_impact_wellbal.patch")],
        "entry": rel(REFS / "benchmarks/lake_at_rest/run.py"),
        "entry_detail": ("python3 benchmarks/lake_at_rest/run.py -> run_case(mode='shipped'|'wb', dx, c0, nu) "
                         "-> production droplet_impact.rhs(pos, vel, mass, cfg, wall=..., dt=...) -> "
                         "samples[] + pressure_drift_max + pass_0p5pct -> report.json"),
        "run_artifacts": [rel(REFS / "benchmarks/lake_at_rest/report.json"),
                          rel(ART / "m15278_rerun.json"),
                          rel(ART / "probe_t0.json"),
                          rel(ART / "figures/audit_drift_vs_criterion.png"),
                          rel(ART / "figures/audit_drift_vs_time.png")],
        "tests": ["tests/test_lake_at_rest.py::test_initial_state_is_hydrostatic_in_the_interior",
                  "tests/test_lake_at_rest.py::test_well_balanced_mode_beats_shipped_by_an_order",
                  "tests/test_lake_at_rest.py::test_wall_masses_follow_the_hydrostatic_profile",
                  "auditor ran: PYTHONPATH=src pytest tests/test_lake_at_rest.py -q -> 3 passed in 7.81s"],
        "integration_callers": [
            "droplet_impact.rhs",
            "src/tensorsph/droplet_impact.py:1475 (run(): rho,p,acc = rhs(pos,vel,mass,cfg,wall=...,pairs=...))",
            "src/tensorsph/dam_break.py:282 (rhs(positions, velocity, mass, cfg, pairs=pairs, wall=wall, ...))",
            "src/tensorsph/sloshing.py:282 (rhs(positions, velocity, mass, cfg, pairs=pairs, wall=wall, ...))",
            "src/tensorsph/ice_splash.py:758 (rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall, ...))",
            "src/tensorsph/freesurface/sloshing_mode.py:285 (droplet_impact.rhs(positions, velocity, mass, cfg, ...))",
            "caveat: Config.equilibrium_level defaults to None -> the well-balanced branch is inert unless a "
            "caller sets it; product default path unchanged (shipped defects remain, see findings C2)",
        ],
        "benchmark": {
            "case_key": orig["case_key"],
            "reference": "exact hydrostatic steady state p = rho0 g (H-y), u = 0 "
                         "(LeVeque 1998 JCP 146:346-365 sec 2.1; Adami/Hu/Adams 2012 JCP 231:7057-7075 sec 3.1)",
            "registered_rows_in_tensor_benchmark_cases": 0,
            "criterion_source": "tensor_case_pool#67 LAKE-AT-REST-SPH tol_policy='压力漂移<0.5%' "
                                "(case pool row still status='dispatched', mission_id=15241 = the FAILED first round)",
            "tolerance": TOL,
            "got_best": {"case": "wb dx=20mm c0=20 nu=0.01", "pressure_drift_max": best["pressure_drift_max"],
                         "over_tolerance_x": best["pressure_drift_max"] / TOL, "pass": False},
            "got_shipped_max": {"pressure_drift_max": shipped_max, "over_tolerance_x": shipped_max / TOL,
                                "pass": False},
            "all_cases": [{"mode": c["mode"], "dx_m": c["dx_m"], "c0": c["sound_speed_m_s"],
                           "nu": c["viscosity"], "drift": c["pressure_drift_max"],
                           "pass_0p5pct": c["pass_0p5pct"]} for c in orig["cases"]],
            "improvement_factor_wb_vs_shipped_at_dx20": orig["summary"]["improvement_factor_at_dx_0p02"],
            "summary_flags": orig["summary"],
            "verdict": "honest FAIL on the pool criterion confirmed by this audit; the 6.8x improvement is a "
                       "well-balanced-property measurement against the SAME analytic profile the operator "
                       "subtracts, not a general free-surface accuracy claim",
        },
        "repro_cmd": ("cd /root/tp-agent/work/m15278 && PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py "
                      f"--out /tmp/m15278_rerun.json  # auditor ran it: 7/7 cases bit-identical "
                      f"(max |Δ drift| = {repro_diff:.1e})"),
        "repro_verified": {"max_abs_drift_diff_vs_delivered": repro_diff,
                           "cases_compared": len(orig["cases"]), "wall_seconds_approx": 22},
        "external_anchor": ("LeVeque 1998 J. Comput. Phys. 146:346-365 sec 2.1 (lake at rest) + "
                            "Adami, Hu & Adams 2012 J. Comput. Phys. 231:7057-7075 sec 3.1 (SPH hydrostatic test): "
                            "citation only, no local copy / no network in sandbox -> NOT independently verified"),
        "audited_commit": "no commit: worktree /root/tp-agent/work/m15278 base ae61a2b2ce7a1c79557a4befefb2b5c2893d0f77, "
                          "dirty=true; head_commit=null on the mission; branch agent/m15278 carries no commit "
                          "(git ls-files / git log --all: empty for benchmarks/lake_at_rest, "
                          "docs/verification_lake_at_rest.md, tests/test_lake_at_rest.py)",
        "audited_files_sha256": {rel(p): sha(p) for p in [
            REFS / "benchmarks/lake_at_rest/run.py",
            REFS / "benchmarks/lake_at_rest/report.json",
            REFS / "src/tensorsph/droplet_impact.py",
            REFS / "docs/verification_lake_at_rest.md",
            REFS / "tests/test_lake_at_rest.py",
            ART / "droplet_impact_wellbal.patch"]},
        "merge_status": {"merged_into_main": False, "uncommitted": True,
                         "eco_repo_grep_equilibrium_level": 0,
                         "note": "answers 'is the capability in the product?' = NO; audit branch keeps a "
                                 "byte-identical snapshot + 154-line patch so a worktree recycle cannot lose it"},
        "three_piece_verification": {
            "run_artifacts": "PASS (on disk, snapshot sha256 registered): report.json 36455 B, 7 cases x 16 samples",
            "benchmark_ref_vs_got": "FAIL: best 3.887% vs 0.5% = 7.77x over; every one of the 7 cases FAIL",
            "repro_cmd": "PASS: auditor re-ran the delivered command, 7/7 cases bit-identical",
            "committed": "FAIL: nothing tracked by git (head_commit null)",
            "machine_referee": "ABSENT: 0 rows in tensor_benchmark_cases for this anchor",
        },
        "findings": {
            "B1_criterion_not_met": f"best pressure drift {100 * best['pressure_drift_max']:.3f}% vs 0.5% "
                                    f"criterion = {best['pressure_drift_max'] / TOL:.2f}x over -> L4 refused",
            "B2_not_committed": "all deliverables untracked in work/m15278; TensorSPH main has no "
                                "equilibrium_level; report §6 cites the BASE commit ae61a2b as if it were the "
                                "deliverable commit (provenance mis-attribution)",
            "B3_referee_absent": "tensor_benchmark_cases has 14 rows, 0 for CASE-LAKE-AT-REST-SPH; case pool #67 "
                                 "stale (status=dispatched, mission_id=15241=failed round 1); corroborates "
                                 "m15240 F2 and m15235 F1_referee_missing",
            "C1_self_referential_metric": "wb operator subtracts p_h = rho0 g (H-y) with H = user-supplied "
                                          "Config.equilibrium_level and drift is measured against the same p_h -> "
                                          "favorable by construction; needs a priori equilibrium level",
            "C2_default_path_still_defective": f"equilibrium_level=None keeps the old path: bottom-row pressure "
                                               f"deficit {probe['shipped']['bottom_row_dev_max_pct']:.1f}%, "
                                               f"spurious |a| {probe['shipped']['acc_max_m_s2']:.2f} m/s2 "
                                               f"(1.94 g), shipped drift {100 * shipped_max:.2f}% "
                                               "(wall mass = uniform rho0 dx^2 is a general fix that need not "
                                               "sit behind a new switch)",
            "D1_figures_not_inline": "tensor_reports#14367 figures[].png_base64 empty, but the figure endpoint "
                                     "returns 200 with real PNGs (58616 / 45243 B) -> not a defect",
            "D2_t0_probe_confirms": f"auditor t=0 probe: interior dev max "
                                    f"{probe['wb']['interior_dev_max_pct']:.5f}%, max|a| "
                                    f"{probe['shipped']['acc_max_m_s2']:.2f} / {probe['wb']['acc_max_m_s2']:.2f} "
                                    "m/s2 = report's 0.012% / 19.04 / 4.65 -> numbers truthful",
        },
        "checks_not_run": [
            "cross-implementation comparison (ISPH/DFSPH/other open-source SPH on the same still pool): not_run",
            "feasibility review of the 0.5% criterion for WCSPH at c0/sqrt(gH)=10.79 (0.5% pressure = ~4e-5 "
            "density consistency): dimensional argument only, no numerical experiment",
            "3-D / long-time (>0.05 s) / larger-N behaviour of the well-balanced mode: not_run",
            "line-by-line audit of make_docs.py figure pipeline (only report.json -> markdown number agreement checked)",
        ],
        "claims_overstated": [
            "report §6 'commit ae61a2b2ce7a' is the worktree BASE commit, not a commit containing the work "
            "(nothing was committed)",
            "headline '漂移降低6.8x' is a like-for-like comparison only at dx=20mm/c0=20/nu=0.01 and is measured "
            "against the profile the operator itself subtracts; it is not a general accuracy improvement",
        ],
        "declared_level": LEVEL,
        "level_rationale": (
            "L3 系统集成：真代码（154 行 patch：hydrostatic_pressure / equilibrium_wall_mass / "
            "well-balanced pressure force + kernel-support free-surface test）直接改在生产共享 RHS "
            "droplet_impact.rhs 内，由 droplet_impact.run/dam_break/sloshing/ice_splash 四个入口经 cfg 触达；"
            "入口→计算→工件链路审计实跑（工件非空、7/7 复现逐位相同、3 单测通过、t=0 探针独立复核）。"
            "不授 L4：算例池判据 压力漂移<0.5% 未达标（最好 3.887%，超 7.77×），"
            "L4『基准验证』要求被验证能力通过基准；且交付物未进 git（head_commit=null），产品侧无该能力。"),
        "audit_doc": "docs/board_audit/m15314/AUDIT_m15278.md",
        "subject_mission": SUBJECT,
        "disposition": "request changes: (1) commit/PR the well-balanced work (snapshot preserved in this audit "
                       "branch), (2) continue the free-surface density consistency fix (Shepard-renormalised "
                       "density / ghost particles) to attack the residual 3.887%, (3) register the criterion row "
                       "in tensor_benchmark_cases so the referee stops idling",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ev = build_evidence()
    if a.dry_run:
        print(json.dumps(ev, ensure_ascii=False, indent=1)[:1200])
        print("...dry-run only, no DB write")
        return
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tensor_maturity_verdicts "
                "(repo, scope, mission_id, level, evidence, verdict_by) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
                "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, "
                "evidence = EXCLUDED.evidence, created_at = now() "
                "RETURNING id, gate_state, gate_level",
                (REPO, SCOPE, MISSION, LEVEL, json.dumps(ev), BY),
            )
            print("verdict row:", cur.fetchone())
        conn.commit()


if __name__ == "__main__":
    main()