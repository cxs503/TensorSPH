-- m15558 verdict for m15408 CASE-POISEUILLE-SPH
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorSPH',
  'm15408 CASE-POISEUILLE-SPH',
  15558,
  4,
  '{"impl_files": ["src/tensorsph/wall_mirror.py", "benchmarks/poiseuille/run.py", "benchmarks/poiseuille_mirror/run.py"], "entry": "benchmarks/poiseuille/run.py", "run_artifacts": ["benchmarks/poiseuille/out/results.json", "benchmarks/poiseuille/out/provenance_audit.json", "benchmarks/poiseuille_mirror/out/results.json"], "tests": ["tests/test_poiseuille.py (7 passed)", "tests/test_poiseuille_mirror_wall.py (10 passed)", "tests/test_poiseuille_claim_provenance.py (11 passed)"], "integration_callers": ["src/tensorsph/wcsph/ (via rhs interface)"], "benchmark": {"u_max_error_max_pct": 0.38, "wall_shear_error_max_pct": 0.44, "all_gates_passed": true, "resolutions": [8, 12, 16]}, "repro_cmd": "PYTHONPATH=src python benchmarks/poiseuille/run.py --resolutions 8 12 16 --t-end 1.0 --t-average 0.2 --out benchmarks/poiseuille/out", "external_anchor": "White, Viscous Fluid Flow 3rd ed. §3-2.2; Batchelor, An Introduction to Fluid Dynamics §4.2"}',
  'board-audit'
);
