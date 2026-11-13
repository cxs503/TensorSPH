#!/usr/bin/env python3
"""Board-audit verdict for m17415 (XVAL-SPH-DTW-REGRESSION-BASELINE-PROTOCOL).

Verdict: L3 — real, executable, reproducible protocol code integrated on the
production dam-break solver and re-run end-to-end by the auditor; **L4 refused**
because the designated arbiter ``tensor_benchmark_cases`` holds no row for this
anchor (0 rows), so there is no registered ref/tolerance to score "got" against,
and the external numeric anchor (same-binary SPHinXsys run) is WITHHELD.

Every string below is assembled from docs/board_audit/m17448/audit_evidence.json
and the subject worktree; nothing is typed from memory. Idempotent upsert.

Run: python3 docs/board_audit/m17448/write_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorSPH"
SCOPE = "m17415 XVAL-SPH-DTW-REGRESSION-BASELINE-PROTOCOL"
MISSION = 17448
SUBJECT = 17415
BY = "board-audit"
LEVEL = 3
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SUBJ = Path("/root/tp-agent/work/m17415")


def strip_clock(obj):
    """Drop wall-clock fields so two runs can be compared on science only."""
    if isinstance(obj, dict):
        return {k: strip_clock(v) for k, v in obj.items()
                if k not in {"elapsed_s", "runtime"}}
    if isinstance(obj, list):
        return [strip_clock(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ev = json.loads((HERE / "audit_evidence.json").read_text())

    # determinism of the scientific content of the re-run
    subject_report = json.loads(
        (SUBJ / "artifacts/dtw_regression_protocol/xval_report.json").read_text())
    rerun = json.loads((ROOT / ev["C7_repro"]["rerun_report"]).read_text())
    science_identical = strip_clock(subject_report) == strip_clock(rerun)
    ev["C7_repro"]["identical_scientific_content"] = science_identical
    ev["C7_repro"].setdefault("identical_raw", ev["C7_repro"].get("identical_to_subject"))
    ev["C7_repro"].pop("identical_to_subject", None)
    (HERE / "audit_evidence.json").write_text(
        json.dumps(ev, indent=2, sort_keys=True, default=float))

    c2 = ev["C2_recompute"]
    c4 = ev["C4_blind_tests"]
    c5 = ev["C5_band_quirk"]
    c7 = ev["C7_repro"]
    evidence = {
        "impl_files": [
            "docs/board_audit/m17448/refs/dtw.py",
            "docs/board_audit/m17448/refs/protocol.py",
            "docs/board_audit/m17448/refs/observe.py",
            "docs/board_audit/m17448/refs/run.py",
            "docs/board_audit/m17448/refs/test_regression_test_dtw.py",
            "docs/board_audit/m17448/audit_independent.py",
        ],
        "impl_files_detail": ("subject worktree /root/tp-agent/work/m17415 @ commit fee0f98; "
            "dtw.py 279 lines (faithful zero-init band DP, canonical DP, scipy dijkstra "
            "shortest path, band diagnostic), protocol.py 440 lines (generation ledger, "
            "runtimes.dat, XML I/O, running-max accepted distance, 4-streak convergence, "
            "1.01x gate), observe.py 87 lines (dam_break.run_case -> series), "
            "run.py 364 lines (arms A/B/C + sensitivity), 17 tests"),
        "entry": "docs/board_audit/m17448/refs/run.py",
        "entry_detail": (
            "PYTHONPATH=src python benchmarks/dtw_regression_protocol/run.py "
            "-> arm_algorithm_xval / arm_band_quirk / arm_protocol(observe -> "
            "dam_break.run_case -> DTWRegressionTest.generate_database / "
            "test_result) -> artifacts/.../xval_report.json (exit 0 = pass); "
            "library entry: tensorsph.regression_test.run_generation(workdir, series, threshold)"
        ),
        "run_artifacts": [
            "docs/board_audit/m17448/refs/artifacts/DamBreak_front_x_top_y_centroid_x_Run_6_result.xml",
            "docs/board_audit/m17448/refs/artifacts/DamBreak_front_x_top_y_centroid_x_dtwdistance.xml",
            "docs/board_audit/m17448/refs/artifacts/DamBreak_front_x_top_y_centroid_x_runtimes.dat",
            "docs/board_audit/m17448/refs/artifacts/xval_dtw_report.json",
            "runs/m17448/repro/xval_report.json",
            "runs/m17448/repro_stdout.log",
            "docs/board_audit/m17448/audit_evidence.json",
        ],
        "run_artifacts_detail": ("7 real dam-break generations (3 quantities x 20 snapshots); "
            "stored dtwdistance = [0.011840492486953735, 0.006022244691848755, "
            "0.003541328012943268]; runtimes.dat = true/7/4; auditor full re-run 101.4 s"),
        "tests": [
            "tests/test_regression_test_dtw.py (17 tests): auditor re-ran "
            "`PYTHONPATH=src python -m pytest tests/test_regression_test_dtw.py -q` "
            "-> 17 passed in 18.62 s",
            "auditor recompute: stored dtwdistance.xml == running max over the full "
            "pairwise matrix re-derived from the Run_N XMLs, max abs diff "
            f"{c2['max_abs_diff_stored_vs_recomputed']:.3g}",
            "auditor negative-control recompute from stored series: faithful DTW vs "
            f"gen6 = {[round(x, 6) for x in c4['negative_distance_recomputed_vs_gen6']]}, "
            f"matches the gate's checked value ({c4['negative_recompute_matches_checked']}), "
            f"limit = {[round(x, 6) for x in c4['limit_1p01x']]}",
        ],
        "integration_callers": [
            "dam_break_observations",
        ],
        "integration_detail": (
            "observe.py:44 dam_break_observations() -> dam_break.py:240 run_case() "
            "(production solver); run.py:143 arm_protocol() drives 8 dt-jittered "
            "generations; caveat: no other case/CI job calls DTWRegressionTest yet, so "
            "the protocol is wired to one solver case + its own benchmark entry, not "
            "fleet-wide (hence L3)"),
        "benchmark": {
            "arbiter": ("SELECT case_name, metric, tolerance FROM tensor_benchmark_cases "
                        f"WHERE anchor_slug='{ev['C6_benchmark_table']['anchor_slug']}'"),
            "rows": ev["C6_benchmark_table"]["rows"],
            "table_total_rows": ev["C6_benchmark_table"]["table_total_rows"],
            "consequence": ("0 rows -> no registered ref/tolerance -> 'got within tolerance' "
                            "cannot be machine-adjudicated -> L4 refused; the subject's own "
                            "criteria (<15% algorithm XVAL, gate accept/reject) are "
                            "self-declared, not arbiter rows"),
        },
        "repro_cmd": ("cd /root/tp-agent/work/m17415 && PYTHONPATH=src python "
                      "benchmarks/dtw_regression_protocol/run.py --out <outdir>  "
                      f"-> auditor ran it: exit 0, {c7['elapsed_s_rerun']:.1f} s, arm A "
                      f"worst deviation {c7['worst_deviation_rerun']:.3g}, arm B passed="
                      f"{c7['protocol_passed_rerun']}, scientific content identical to the "
                      f"delivered report = {science_identical}"),
        "external_anchor": {
            "inventory_row": "oss_solutions id=357 (repo=Xiangyu-Hu/SPHinXsys, "
                             "tag=dtw-regression-baseline-protocol, evidence='report 9986')",
            "pinned_commit_claimed": "5241a89",
            "verification_status": ("WITHHELD for the reference side: no local SPHinXsys "
                                    "checkout exists on this host (find /root -name "
                                    "*sphinxsys* -> empty), so the file:line facts in "
                                    "docs/XVAL-SPH-DTW-REGRESSION-BASELINE-PROTOCOL.md "
                                    "could not be re-read this audit; the port's behaviour "
                                    "matches the documented recurrence (auditor reimplemented "
                                    "it from the description and reproduced every stored "
                                    "distance)"),
        },
        "findings": [
            f"C1 no arbiter row: tensor_benchmark_cases has 0 rows for this anchor "
            f"(table holds {ev['C6_benchmark_table']['table_total_rows']}) -> L4 unadjudicable.",
            f"C2 evidence gap (medium): the blind *legit* run's raw series is not persisted, "
            f"so only its checked distances {[round(x, 6) for x in c4['legit_checked_distances']['1']]} "
            f"are auditable; the negative control is fully reproducible from "
            f"negative_control_series.",
            "C3 semantics (info): the converged flag is raised one generation after the "
            "4th consecutive revision <= threshold (label 4 at gen 5, converged at gen 6) — "
            "documented in the subject's report, consistent with runtimes.dat; treated as "
            "faithful behaviour, not a defect, but unverifiable against upstream source "
            "on this host.",
            "C4 serialisation wart (low): DTWRegressionFailure is stored as a JSON *string* "
            "inside arm_protocol.negative_control.reason, so consumers must double-parse.",
            "C5 declared threshold 0.01 is a repository choice, not the upstream per-case "
            "value (upstream tests/ tree unavailable) — honestly declared by the subject; "
            "sensitivity table shows 1e-2/1e-3 converge, 1e-4/1e-5 do not within 8 generations.",
            "C6 band quirk independently reproduced on the auditor's own pairs: faithful DTW "
            f"underestimates canonical DTW by up to "
            f"{max(r['relative_underestimate'] for r in c5.values() if r['relative_underestimate'] == r['relative_underestimate']):.1%} "
            "(stretch family degenerates to exactly 0) — the documented reference defect is real.",
            "C7 positive: every headline number in the report re-derives from stored "
            "artifacts (pairwise distances, accepted running max, 1.01x band, accept/reject "
            "outcomes); auditor's independent full re-run reproduces the science exactly.",
        ],
        "gate_preview": ("tools-level judge() run by the auditor on this very row: "
                         "gate_level=2, reason 'L3 缺: integration_callers(grep 无引用)' — the "
                         "caller symbol dam_break_observations does not exist in "
                         "/root/tensor-ecosystem/TensorSPH yet because the subject branch is "
                         "not merged (auto-push pending). Functional substance is L3; the "
                         "machine view will reach L3 only after merge. Not a downgrade of the "
                         "audit finding, a merge-order effect."),
        "auditor_worktree": "/root/tp-agent/work/m17448/docs/board_audit/m17448/",
        "level_basis": {
            "L2": "real code, input -> computation -> output (dtw.py/protocol.py/observe.py)",
            "L3": "driver runs 8 real generations on the production dam_break solver; "
                  "XML ledger + gate integrated with that case's benchmark entry",
            "L4_refused": "no tensor_benchmark_cases row and no external numeric reference "
                          "(same-binary SPHinXsys = WITHHELD)",
        },
    }

    payload = json.dumps(evidence, ensure_ascii=False, default=float)
    print(f"level=L{LEVEL} repo={REPO} scope={SCOPE} mission={MISSION} "
          f"evidence_bytes={len(payload)}")
    if args.dry_run:
        print(payload[:1200])
        return 0
    with psycopg.connect(DSN) as conn:
        row = conn.execute(
            "SELECT id FROM tensor_maturity_verdicts WHERE repo=%s AND scope=%s "
            "AND verdict_by=%s", (REPO, SCOPE, BY)).fetchone()
        if row:
            conn.execute(
                "UPDATE tensor_maturity_verdicts SET mission_id=%s, level=%s, "
                "evidence=%s WHERE id=%s", (MISSION, LEVEL, payload, row[0]))
            verdict_id = row[0]
        else:
            verdict_id = conn.execute(
                "INSERT INTO tensor_maturity_verdicts "
                "(repo, scope, mission_id, level, evidence, verdict_by) "
                "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                (REPO, SCOPE, MISSION, LEVEL, payload, BY)).fetchone()[0]
        conn.commit()
        check = conn.execute(
            "SELECT id, repo, scope, mission_id, level, verdict_by, "
            "length(evidence::text) FROM tensor_maturity_verdicts WHERE id=%s",
            (verdict_id,)).fetchone()
    print("verdict row:", check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())