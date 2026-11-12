-- board-audit verdict for m16330 (CASE-SPH-STOKES-2ND-PROBLEM), auditor m16353
-- applied 2026-11-12 as tensor_maturity_verdicts id=475 (gate_state pending -> judge()=L4)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorSPH',
 'm16330 CASE-SPH-STOKES-2ND-PROBLEM',
 16353,
 4,
$json${
 "entry": [
  "benchmarks/stokes_second/run.py",
  "scripts/post_report_m16330.py"
 ],
 "tests": [
  "tests/test_stokes_second.py -k criterion-or-plate_boundary-or-moving: 5 passed in 35.81 s (auditor re-ran 2026-11-12)"
 ],
 "commit": "m16330 branch 0e7f29f, landed on mainline by audit commit 6a84a46 (m16353)",
 "report": "tensor_reports id=15423 (six sections 1-6 verified present; numbers match artefacts)",
 "benchmark": {
  "ref": "Landau & Lifshitz, Fluid Mechanics 2nd English ed. section 24; Macia et al. 2011 Prog. Theor. Phys. 125:1091 doi:10.1143/ptp.125.1091 (context only)",
  "pass": true,
  "got_vs_ref": "G1 amplitude L2 = 0.0101682 % vs tol 5 %; G2 worst row 0.0170754 % vs tol 10 %; G3 decay-rate error -0.0176705 % vs tol 5 %; delta_fit/delta 0.999823 vs 1+-0.05; G4 finite-depth L2 0.0116763 % vs tol 5 %; 18 rows inside layer (>=5); 6/6 benchmark.json rows pass=true; ladder 0.06746/0.01826/0.01021 %",
  "auditor_recompute": "recomputed from stored raw row amplitudes: G1 0.0102073 %, G2 0.0170801 %, G3 -0.0177264 %, delta ratio 0.9998228 -- agrees with the stored metrics to <=0.4 % relative (6-significant-digit rounding of the published rows)",
  "registered_criterion": "tensor_benchmark_cases has NO row for anchor_slug=CASE-SPH-STOKES-2ND-PROBLEM (0 rows, auditor-verified via psql 2026-11-12); the referee used is the mission goal criterion (<5 %, amplitude decay A(y)=U0 exp(-y/delta), >=5 particle rows inside the Stokes layer) plus benchmark.json primary_tolerance_pct=5.0"
 },
 "repro_cmd": "PYTHONPATH=.:src python benchmarks/stokes_second/run.py --ladder --c0 4 --periods 6 --measure-periods 2 --out benchmarks/stokes_second/out  (auditor executed the same entry with --quick on the landed mainline tree: 76.0 s, gates G1-G5 all True; full 3-rung ladder NOT re-run in this audit -- finest rung alone is 853 s)",
 "entry_note": "run.py --quick rerun by auditor on the landed mainline tree: 76.0 s, gates G1..G5 all True; --ladder is the repro_cmd",
 "impl_files": [
  "src/tensorsph/stokes_second.py",
  "src/tensorsph/wall_mirror.py",
  "benchmarks/stokes_second/run.py"
 ],
 "audit_checks": {
  "tests_passed": true,
  "repro_cmd_executed": "partial (--quick)",
  "report_six_sections": true,
  "run_artifacts_exist": true,
  "independent_recompute": true,
  "key_quantity_error_pct": 0.0102,
  "key_quantity_under_3pct": true,
  "manifest_sha256_verified": true,
  "ref_vs_got_within_tolerance": true
 },
 "impl_symbols": {
  "src/tensorsph/wall_mirror.py": "plane_walls (moving no-slip mirror ghosts, plane_vel=(U0 cos wt,0))",
  "src/tensorsph/stokes_second.py": "89 penetration_depth; 96 semi_infinite; 113 finite_depth_complex_amplitude; 186 verify_symbolic; 244 verify_numeric",
  "benchmarks/stokes_second/run.py": "139 drive (WCSPH integration + per-row harmonic fit); 328 evaluate (ref vs got per row); 417 gates (G1..G5)"
 },
 "run_artifacts": [
  "benchmarks/stokes_second/out/results_ladder.json",
  "benchmarks/stokes_second/out/results_sensitivity.json",
  "benchmarks/stokes_second/out/benchmark.json",
  "benchmarks/stokes_second/out/MANIFEST.json",
  "benchmarks/stokes_second/out/amplitude_decay.png",
  "benchmarks/stokes_second/out/field_snapshot.png",
  "docs/verification_stokes_second.md",
  "artifacts/audit_m16353/recomputed.json"
 ],
 "audit_findings": {
  "F2_bookkeeping": "tensor_benchmark_cases has 0 rows for this anchor across the whole case pool (29 anchors / 32 rows exist, this one missing)",
  "L1_audit_limit": "full 3-rung ladder not re-run (cluster job cj1306 still queued); reproducibility supported by independent re-derivation + sha256 + --quick rerun",
  "F3_self_evidence": "m16330 result JSON carries no maturity evidence block (server_quality_audit.gate_valid=false), hence this audit",
  "F1_blocking_fixed": "deliverables existed only on branch agent/m16330; main and ECO/TensorSPH lacked src/tensorsph/stokes_second.py and the 3-rung artefacts -> gate impl_files/run_artifacts would fail; landed verbatim by commit 6a84a46"
 },
 "external_anchor": "Landau & Lifshitz Fluid Mechanics 2nd English ed. section 24 (Stokes second problem); Macia et al. 2011 Prog. Theor. Phys. 125:1091 doi:10.1143/ptp.125.1091",
 "run_artifacts_note": "MANIFEST.json sha256 recomputed by auditor: 5/5 match; results_ladder.json carries raw per-row amplitudes for all 3 rungs (delta/dx=8,12,18)",
 "integration_callers": [
  "from tensorsph.wall_mirror import plane_walls",
  "from tensorsph.droplet_impact import"
 ],
 "integration_callers_note": "wall_mirror.plane_walls is shared with benchmarks/poiseuille_mirror/run.py; the case drives the production WCSPH kernel rhs unmodified"
}$json$::jsonb,
 'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence
RETURNING id, level, gate_state, gate_level;
