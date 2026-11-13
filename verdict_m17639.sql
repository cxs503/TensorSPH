-- Audit verdict for m17639: XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION
-- Audited by: board-audit (m17667)
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorSPH',
    'm17639 XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION',
    17667,
    4,
    '{
        "impl_files": "src/tensorsph/transport_velocity_correction.py (~200 lines); src/tensorsph/dam_break.py (TVC integration via tvc_every/tvc_alpha/tvc_scope/tvc_max_shift config)",
        "entry": "benchmarks/xval_sphinx_tvc/run.py",
        "run_artifacts": "artifacts/xval_sphinx_tvc/grid_dx003.json, grid_dx004.json, figure_xval_tvc.png (99KB)",
        "tests": "tests/test_transport_velocity_correction.py: 9/9 passed (transcription match, lattice fixed-point, surface sign, wall factor 2, scope mask, symmetry zero-net, bit-identical when disabled, bounded physical run)",
        "integration_callers": "src/tensorsph/dam_break.py DamBreakConfig with TVC hooks",
        "benchmark": "Martin & Moyce (1952) digitized Z/a(tau); dx=0.03 tvc_a010_all: MAE=0.1425 rel=10.81% gate_abs_pass=True gate_rel_pass=True; dx=0.04 tvc_a010_all: MAE=0.1468 rel=16.96% gate_abs_pass=True gate_rel_pass=False; baseline dx=0.04: MAE=0.591 rel=46.7% FAIL",
        "repro_cmd": "cd /root/tp-agent/work/m17639 && python benchmarks/xval_sphinx_tvc/run.py --resolutions 0.04 0.03 --seconds 0.5",
        "external_anchor": "SPHinXsys commit 09e55cfdd23104e1b1db006a209a906218b3b14e; verbatim upstream files in docs/xval_sphinx_tvc/oss_source/ with sha256 provenance (12 files)",
        "gaps": ["No live binary diff vs SPHinXsys (Boost/TBB unavailable)", "Only 2 resolution levels; 4-level ladder pending (cj1623/cj1624)", "No tensorbench_results row in platform (L4 CLAIMED_NOT_PROVEN)", "dx=0.02 run not completed"]
    }',
    'board-audit'
);
