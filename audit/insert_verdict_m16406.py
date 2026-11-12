#!/usr/bin/env python3
"""m16406 — insert the board-audit maturity verdict for m16352.

Idempotent: the unique key (repo, scope, verdict_by) is upserted, so re-running
this script updates the row instead of duplicating it.

Evidence contract: the same 8 keys the m16318 audit used (impl_files / entry /
run_artifacts / tests / integration_callers / benchmark / repro_cmd /
external_anchor), every value a real pointer that this audit re-verified.
``gate_state`` is left 'pending': /root/tp-agent/maturity_gate.py judges it.

Run: python3 audit/insert_verdict_m16406.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "TensorSPH"
SCOPE = "m16352 CASE-SPH-FSI-ELASTIC-GATE"
MISSION = 16406          # the audit mission (whose id the verdict is filed under)
REVIEWED = 16352         # the audited mission
LEVEL = 2                # honest: impl+entry+artifacts, NO integration caller (see verdict §2)
VERDICT_BY = "board-audit"
OUT = Path(__file__).resolve().parents[1] / "docs" / "board_audit" / "m16406"
AUD = "docs/board_audit/m16406"

EVIDENCE = {
    "impl_files": ["src/tensorsph/fsi_elastic_gate.py"],
    "entry": ["benchmarks/fsi_elastic_gate/run_antoci.py"],
    "run_artifacts": [
        f"{AUD}/artifacts/runs.json",
        f"{AUD}/artifacts/summary.json",
        f"{AUD}/artifacts/rod_verification.json",
        f"{AUD}/artifacts/v4_gate_load_path.json",
        f"{AUD}/artifacts/provenance.json",
        f"{AUD}/artifacts/fig1_tip_and_load.png",
        f"{AUD}/fig1_audit.png",
    ],
    "tests": ["tests/test_fsi_elastic_gate.py"],
    "integration_callers": [
        "NONE beyond its own harness: benchmarks/fsi_elastic_gate/run_antoci.py:30 "
        "imports tensorsph.fsi_elastic_gate; no other TensorSPH module, no CLI entry, "
        "no platform tool calls it, and it is not re-exported from src/tensorsph/__init__.py "
        "=> L3 (integration) not claimed, even though a grep on the module name would hit "
        "the self-harness"
    ],
    "benchmark": (
        "CASE-SPH-FSI-ELASTIC-GATE has 0 rows in tensor_benchmark_cases (table total 32; "
        "anchor ILIKE '%ELASTIC%' only matches TensorDEM Campbell rows) and "
        "tensorbench_results(mission_id=16352)=0 => machine referee "
        "GET /api/v1/tensor/bench/l4-verdict/16352 = CLAIMED_NOT_PROVEN, rows=0, "
        "l4_benchmark_pass=false. Antoci 2007 gate-tip curve WITHHELD (no OA channel; "
        "provenance.json verified), so peak/phase error is not computable. Delivered "
        "coupled got = peak 1427.4 mm @297.6 ms at dx=12.5 mm, which is 3.96x the gate "
        "height and therefore not a physical displacement; |tip|>H from t=0.1148 s "
        "(177/330 samples), 6 sign flips after 100 ms, fluid max speed 71.94 m/s = 3.60 c0, "
        "gate force at t=0 = 0.000 N/m vs 418.220 N/m hydrostatic and peak 26794 N/m = 64x "
        "=> the coupled time history is numerically invalid, L4 denied. Sub-verification "
        "that DOES hold (audit-side recomputation, independent_checks.json): V1 hydrostatic "
        "load transfer +0.240/+0.089/+0.189 % at dx=12.5/10/7.5 mm (legacy pairwise "
        "-28.369/-20.367/-8.317 %); elastica reference cross-checked by independent "
        "elliptic-type quadrature to 2.6e-13 (x/L=0.445004, y/L=0.810609). The delivered "
        "A2 'large-deflection gap +86.8 %' is an axis-swap artefact: with the rod's "
        "transverse/along-beam coordinates paired correctly the errors are +2.55/-4.58 % "
        "(n=25) and +1.28/-2.25 % (n=50), i.e. converged <5 %. A0 at n=50 returns 0.000 m "
        "(-100 %) and is missing from the report table."
    ),
    "repro_cmd": (
        "python -m pytest tests/test_fsi_elastic_gate.py -q -k \"not coupled_short\"   "
        "(audit-side rerun: 7 passed, 1 deselected, 95.46 s) ; "
        "python3 scripts/audit/m16406_case_fsi_board_audit.py   "
        "(regenerates every number in this verdict from the delivered code + artifacts)"
    ),
    "external_anchor": (
        "Antoci, Gallati & Sibilla 2007, Comput. Struct. 85:879-890, "
        "doi:10.1016/j.compstruc.2007.01.002 -- bibliographic record resolves "
        "(Crossref), but the experimental gate-tip curve is NOT retrievable in this "
        "sandbox (Unpaywall best_oa_location=null; 4 direct-PDF channels 403/refused/"
        "timeout, replayed from provenance.json and re-checked by the audit) => the "
        "external anchor is WITHHELD, not met. The internal anchor that IS met: the "
        "Kirchhoff elastica (Bisshopp-Drucker) large-deflection solution, reproduced "
        "independently by the audit to 2.6e-13."
    ),
    # --- audit-specific annotations (not part of the gate's 8-key contract) ---
    "impl_refs": [
        "src/tensorsph/fsi_elastic_gate.py:319-380 gate_load: wall-panel pressure "
        "extrapolation (first-order weighted LS) -- the V1 fix",
        "src/tensorsph/fsi_elastic_gate.py:241-268 rod_energy: clamped-base hinge "
        "(theta0 vs fixed tangent (0,1)) -- the m16214 pin defect",
        "src/tensorsph/fsi_elastic_gate.py:586-617 elastica_tip: Kirchhoff elastica by "
        "shooting (verified to 2.6e-13 by the audit)",
        "src/tensorsph/fsi_elastic_gate.py:745-751 <-- A2 axis swap (F2)",
        "src/tensorsph/fsi_elastic_gate.py:628-680 rod_static_equilibrium: L-BFGS-B "
        "energy minimisation (silently returns the initial point at n=50, F3)",
        "tests/test_fsi_elastic_gate.py:90 <-- 'assert ... or True' (F6)",
        "tests/test_fsi_elastic_gate.py:114-120 coupled guard runs t_end=0.01 s, while "
        "the run fails from t=0.1148 s (F7)",
    ],
    "defects": [
        "L4 unadjudicable: 0 tensor_benchmark_cases rows for the anchor, 0 "
        "tensorbench_results rows, referee CLAIMED_NOT_PROVEN (F1)",
        "A2 reported +86.8 % is a coordinate-axis swap; true error +2.55/-4.58 % at n=25 "
        "and +1.28/-2.25 % at n=50 => the report's 'third gap' conclusion is falsified (F2)",
        "A0 at n=50 returns 0.000 m (-100 %) and is absent from the report table (F3)",
        "coupled run invalid: 3.96x gate height, 31 Hz ringing, Ma=3.6, force 64x "
        "hydrostatic, zero load at t=0 (no hydrostatic initialisation) (F4)",
        "end-to-end V4 not_run; v4_gate_load_path.json rel_error=-97.86 % (F5)",
        "one test assertion is unfalsifiable ('or True') (F6)",
        "the coupled guard stops at 0.01 s, before the failure at 0.1148 s (F7)",
        "report table 4 column misalignment + missing n=50 row (F8)",
        "single-resolution evidence only (dx=12.5 mm; dx=10 mm timed out) (F9)",
    ],
    "reviewed": {
        "mission_id": REVIEWED,
        "repo": REPO,
        "commit": "4af6b11de55b06b864b29cf3d3142b3ad3089f88",
        "branch": "agent/m16352",
        "report_id": 15473,
        "impl_sha256": {
            "src/tensorsph/fsi_elastic_gate.py":
                "a77a0905e3243e12a7fe37f067b04d438ef865e0255f61a7eb0cea85a5c60467",
            "tests/test_fsi_elastic_gate.py":
                "ad2e850dfc90ab5827fbfbae290967d0f05f2c92c9f5b1280a516da65b58a132",
            "benchmarks/fsi_elastic_gate/run_antoci.py":
                "de36100b77054b64c7fcdce81966dfd1670af96574553f9ad6ad17cdc1b74ec8",
        },
    },
    "level_rationale": (
        "L0 docs / L1 tests / L2 implementation all real and re-verified; L3 DENIED: the "
        "only caller is the module's own benchmark driver (self-harness), nothing is "
        "exported from the package and no solver or platform component calls it -- "
        "note this differs from precedent verdict id=464 (m16318) which granted L3 on the "
        "same self-harness pattern; L4 DENIED: no benchmark row + referee "
        "CLAIMED_NOT_PROVEN + reference WITHHELD + the coupled result is numerically "
        "invalid. Disposition: approve with notes (the two defect fixes and the V1/A2 "
        "sub-verifications are real), L4 rework to be spawned separately."
    ),
    "audit_report": {"path": "docs/board_audit/m16406/audit_verdict_m16406.md",
                     "tensor_report_id": 15483,
                     "figures": ["docs/board_audit/m16406/fig1_audit.png",
                                 "docs/board_audit/m16406/fig2_load_transfer.png",
                                 "docs/board_audit/m16406/fig3_gate_force.png"],
                     "gate_preview": "judge(2,evidence,'TensorSPH',16406) -> (2, []) => confirmed"},
    "prior_verdict_on_scope": {"id": 463, "mission_id": 16257, "level": 2,
                               "gate_state": "reopened", "scope": SCOPE},
}


def dsn() -> str:
    for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                         "postgresql://")
    raise SystemExit("no DATABASE_URL")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ev_path = OUT / "verdict_evidence.json"
    ev_path.write_text(json.dumps(EVIDENCE, indent=1, ensure_ascii=False))
    print(f"evidence -> {ev_path} ({ev_path.stat().st_size} bytes)")

    if args.dry_run:
        print(json.dumps({"repo": REPO, "scope": SCOPE, "mission_id": MISSION,
                          "level": LEVEL, "verdict_by": VERDICT_BY}, indent=1))
        return 0

    import psycopg
    sql = """
    INSERT INTO tensor_maturity_verdicts
        (repo, scope, mission_id, level, evidence, verdict_by)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (repo, scope, verdict_by) DO UPDATE
        SET mission_id = EXCLUDED.mission_id,
            level      = EXCLUDED.level,
            evidence   = EXCLUDED.evidence,
            gate_state = 'pending',
            gate_level = NULL,
            gate_note  = NULL
    RETURNING id, level, gate_state
    """
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, (REPO, SCOPE, MISSION, LEVEL, json.dumps(EVIDENCE, ensure_ascii=False),
                          VERDICT_BY))
        row = cur.fetchone()
        conn.commit()
    print(f"verdict id={row[0]} level={row[1]} gate_state={row[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())