-- Verdict for m20154 CASE-SPH-OSC-DROP-FREQ (oscillating droplet frequency vs Rayleigh 1879)
-- Audit performed by board-audit (mission 20195)
-- Date: 2026-11-14

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorSPH',
    'm20154 CASE-SPH-OSC-DROP-FREQ',
    20195,
    3,
    '{
        "impl_files": "src/tensorsph/oscillating_drop3d.py (rayleigh_frequency_3d/2d, axisym_drop_geometry, axisym_curvature, axisym_modal_stress, csf3_audit)",
        "entry": "benchmarks/oscillating_drop_freq/run_freq.py (axisym linear-response + 3D lattice audit)",
        "run_artifacts": "benchmarks/oscillating_drop_freq/{report.json, fig_frequency_error.png, fig_noise_floor.png, fig_lattice3d_audit.png, make_report.py}",
        "tests": "tests/test_oscillating_drop_freq.py (17 passed: 8 analytic ref, 5 axisymmetric reduction, 2 damped cosine fit, 2 shipped 3D operator)",
        "integration_callers": "benchmarks/oscillating_droplet/run.py (2D dynamic time-integration, separate from this anchor)",
        "benchmark": {
            "gate_pct": 6.0,
            "n_rows": 4,
            "n_passing": 0,
            "best_error_pct": -221.49,
            "finest_resolution_error_pct": -262.17,
            "gate_pass": false,
            "ref": "Rayleigh 1879 Proc. R. Soc. Lond. 29:71; Lamb 1932 Art. 275: omega_l^2 = l(l-1)(l+2)sigma/(rho R^3), l=2 => 84.853 rad/s",
            "got": "omega_measured = -103.084 rad/s (best, l=2 eps=0.05 R/dx=8); -123.090 rad/s (finest, l=2 eps=0.02 R/dx=12)",
            "failure_modes": [
                "axisym reduction normalization inconsistency: kappa_mean_R ~1.21 vs exact 2.0",
                "curvature estimator mode-2 noise floor: eps_noise=0.0019-0.0061, same order as seed eps=0.02",
                "3D lattice csf3 operator: kappaR=0.77 (should be 2.0), delta_measure=0.33 (should be 1.0), not resolution-convergent"
            ]
        },
        "repro_cmd": "PYTHONPATH=src python benchmarks/oscillating_drop_freq/run_freq.py && PYTHONPATH=src python benchmarks/oscillating_drop_freq/make_report.py && python -m pytest tests/test_oscillating_drop_freq.py -q",
        "external_anchor": "Rayleigh 1879; Lamb 1932 Hydrodynamics 6th ed. Art. 275; R=0.02m sigma=7.2N/m rho=1000kg/m3"
    }',
    'board-audit'
);
