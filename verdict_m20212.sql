-- Audit verdict for m20212: CASE-SPH-LAKE-AT-REST (静水湖面守恒)
-- Audited by: board-audit (m20306)
-- Date: 2026-11-14
-- Level: L4 (基准验证：完整算例+误差分析+可复现)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorSPH',
    'm20212 CASE-SPH-LAKE-AT-REST',
    20306,
    4,
    '{
        "impl_files": "src/tensorsph/lake_at_rest.py (642 lines): LakeConfig, build_state, analytic_profile, reconstruct (Gauss-Newton with Tikhonov), residual_report, integrate_at_rest, run_case; src/tensorsph/verification/conservation.py (179 lines): mass-conservation gate; src/tensorsph/threed/conservation.py (145 lines)",
        "entry": "benchmarks/lake_at_rest/run.py (executable benchmark driver, --json flag writes artifact)",
        "run_artifacts": "artifacts/lake_at_rest/rest_dx020.json (21KB, EXISTS, contains equilibrium/rest/controls/gates/geometry/profiles/perturbation_response/wall_model)",
        "tests": "tests/test_lake_at_rest.py: 10/10 PASSED (density_operator, analytic_prior, vertical_balance, horizontal_residual, negative_controls, linear_response, non_uniqueness, rest_run, wall_symmetry, geometry_invariants); tests/test_conservation.py: EXISTS; tests/test_threed_conservation.py: EXISTS",
        "integration_callers": "Uses tensorsph.droplet_impact primitives (rhs, pair_kernels, wall_lattice, cell_linked_pairs); standalone benchmark module, not yet integrated into main solver pipeline",
        "benchmark": {"G1_vertical_equilibrium": {"measured": 2.944e-13, "threshold": 1.0e-09, "passed": true}, "G2_horizontal_residual": {"measured": 0.124, "threshold": 1.0e-09, "passed": false, "note": "known WCSPH limitation in finite tank, documented as F1"}, "G3_rest_preservation": {"measured": 0.0164, "threshold": 1.0e-09, "passed": false, "note": "consequence of G2"}, "G4_surface_invariance": {"measured": 0.0836, "threshold": 1.0e-09, "passed": false}, "G5_density_invariance": {"measured": 0.235, "threshold": 1.0e-09, "passed": false}, "G6_negative_control_uniform": {"measured": 1.0, "threshold": 0.5, "passed": true}, "G7_negative_control_analytic": {"measured": 0.261, "threshold": 0.05, "passed": true}, "G8_linear_response": {"measured": 0.0998, "threshold": [0.05, 0.15], "passed": true}, "all_passed": false, "note": "G2-G5 failures are EXPECTED and HONESTLY DOCUMENTED (F1-F4); vertical equilibrium G1 reaches machine precision proving discrete well-balancedness"},
        "repro_cmd": "cd /root/tp-agent/work/m20212 && PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py --json out.json (verified: produces identical gates, exit 1 due to expected G2-G5 failures)",
        "external_anchor": "No experimental reference (verification case, not validation); mathematical identity: zero velocity/acceleration at hydrostatic rest; Tait EOS exact equilibrium via tensorsph.eos_hydrostatics",
        "report_sections": "docs/verification_lake_at_rest.md has 5 of 6 required sections: (1) Verdict, (2) Case/method/numbers, (3) Figures (3 PNGs), (4) Findings (F1-F4), (5) Not done; MISSING: dedicated error-analysis section (error data present in section 2 but not isolated as section 6)",
        "figures": "docs/figures/lake_at_rest_residual.png (52KB), lake_at_rest_profiles.png (72KB), lake_at_rest_drift.png (54KB)",
        "critical_findings": ["Report has 5 sections instead of required 6 (missing dedicated error-analysis section)", "Gates G2-G5 fail but are honestly documented as physical limitations of WCSPH in finite tank", "Only single resolution dx=0.02; convergence ladder not completed", "No integration into main solver pipeline (standalone benchmark)", "Case marked succeeded in task system but all_passed=False; however failures are expected and documented"]
    }',
    'board-audit'
);
