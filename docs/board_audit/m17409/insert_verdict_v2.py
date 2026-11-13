#!/usr/bin/env python3
"""m17409 — clean-path, re-insertable board-audit verdict for m17364 (level 3).

Machine contract (maturity_gate): impl_files / entry / run_artifacts / external_anchor must be
*bare* path strings that exist; prose belongs to non-checked keys. Prose-with-parens paths made the
first insert (verdict id 547) unparsable for the gate, so this rebuilds the evidence and upserts.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg

REPO, SCOPE, MISSION, LEVEL, BY = (
    "TensorSPH", "m17364 XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION", 17409, 3, "board-audit")
SRC = "/root/tp-agent/work/m17364"
AUD = Path(__file__).resolve().parent

EVIDENCE = {
    "impl_files": [f"{SRC}/src/tensorsph/compression_regularization.py",
                   f"{SRC}/src/tensorsph/droplet_impact.py"],
    "entry": [f"{SRC}/benchmarks/xval_sphinxsys_compression/run.py",
              f"{SRC}/benchmarks/xval_sphinxsys_compression/figures.py"],
    "run_artifacts": [str(AUD.parent.parent / "artifacts/repro_m17364/records_repro.json"),
                      str(AUD.parent.parent / "artifacts/repro_m17364/figures/fig_band_error.png"),
                      str(AUD.parent.parent / "artifacts/repro_m17364/figures/fig_pressure_profile.png"),
                      str(AUD.parent.parent / "artifacts/repro_m17364/figures/fig_compression_profile.png"),
                      str(AUD.parent.parent / "artifacts/repro_m17364/figures/fig_time_series.png"),
                      f"{SRC}/artifacts/xval_sphinxsys_compression/records_dx004_quick.json"],
    "tests": [
        "tests/test_compression_regularization.py::test_vectorised_matches_literal_transcription",
        "tests/test_compression_regularization.py::test_plain_formulation_equals_the_summation_density",
        "tests/test_compression_regularization.py::test_two_stage_order_reproduces_the_dynamics_list",
        "tests/test_compression_regularization.py::test_solver_density_is_rho0_on_a_reference_lattice",
        "audit re-run: pytest tests/test_compression_regularization.py -q -> 16 passed in 7.16 s",
    ],
    # Honest L3 evidence, but the gate greps /root/tensor-ecosystem/TensorSPH and agent/m17364 is
    # not merged there, so this key is expected to miss and cap the machine verdict at L2.
    "integration_callers": [
        "src/tensorsph/droplet_impact.py:882 (rhs() compression branch, Config switches :227-231); "
        "droplet_impact.rhs is TensorSPH's main WCSPH solver; NOT merged into ECO yet"],
    "benchmark": (
        "anchor has 0 rows in tensor_benchmark_cases (52 total) -> no platform tolerance judge (F2); "
        "validated vs analytic hydrostatic p=rho0*g*(H-y), dx=0.04, 513 steps, interior bands 0-1: "
        "summation -1.401829/-0.681561 %, compression_none -1.426631/-0.72 %, "
        "compression_refmean -9.099796/-8.80 %, compression_refmean_internal -9.156891/-8.85 %; "
        "rho max err 0.0130/0.0132/0.0761/0.0782 %; all four arms fail the repo's own 1 % gate "
        "(disclosed by the author); audit re-run max|delta| = 0.00e+00, 4 PNG sha256 identical"),
    "repro_cmd": (
        "bash docs/board_audit/m17409/repro_m17364.sh  (16 s, verified by this audit; applies the 1-line "
        "argparse fix then re-runs run.py --quick + figures.py and prints sha256 vs delivered). "
        "AS-DELIVERED repro_cmd crashes: TypeError 'str' / 'str' at run.py:222 -> F1, blocks L4"),
    "external_anchor": [str(AUD.parent.parent / "audit_m17364/upstream/density_regularization.hpp"),
                        str(AUD.parent.parent / "audit_m17364/upstream/density_regularization.h"),
                        str(AUD.parent.parent / "audit_m17364/upstream/density_regularization.cpp"),
                        str(AUD.parent.parent / "audit_m17364/upstream/SHA256SUMS.txt")],
    "external_anchor_note": (
        "Xiangyu-Hu/SPHinXsys tree 1612330e44c8947406f24b6756449e80dcd7158e, blobs 5742d3f8/4ecc5924/"
        "244d82b1; independently re-read by this audit: self term W0*Vref unscaled, neighbour term "
        "(*inv_ref)*W_ij*Vref_j, reduce S>1, Kref=1/mean, rho=K*rho0, Internal/FreeSurface=SMAX(S,1)/"
        "FreeStream -> port faithful. Formulation-level anchor only (no experimental magnitudes) => L5 not claimed"),
    "level_notes": {
        "L2": "impl+entry+run_artifacts verified, audit re-ran everything",
        "L3": "wired into droplet_impact.rhs (main WCSPH solver); machine-invisible until agent/m17364 merges",
        "L4_blocked_by": ["F1 delivered repro_cmd TypeError (fixed in audit scratch tree only)",
                          "no multi-resolution convergence: cluster cj1504 still queued, age 26.5 min"],
        "doc_defects": ["param table says 32 wall particles vs 105 in artefacts (F3)",
                        "236 vs 251 lines; '1.4e-16' not reproducible, audit gets 4.44/3.36/6.51e-16 (F4)"],
        "not_run": ["convergence order", "kernel/h-factor sensitivity", "ship-case applicability"],
    },
}


def main() -> int:
    (AUD / "verdict_evidence.json").write_text(json.dumps(EVIDENCE, indent=1, ensure_ascii=False))
    dsn = next(l.split("=", 1)[1].strip().replace("postgresql+psycopg://", "postgresql://")
               for l in open("/root/tensor-ecosystem/TensorPlatform/.env")
               if l.startswith("DATABASE_URL="))
    sql = """
    INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (repo, scope, verdict_by) DO UPDATE
       SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
           evidence = EXCLUDED.evidence, gate_state = 'pending',
           gate_level = NULL, gate_note = NULL
    RETURNING id, level, gate_state
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, (REPO, SCOPE, MISSION, LEVEL, json.dumps(EVIDENCE, ensure_ascii=False), BY))
        row = cur.fetchone()
        conn.commit()
    print(f"verdict id={row[0]} level={row[1]} gate_state={row[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())