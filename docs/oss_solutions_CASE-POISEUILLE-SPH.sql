-- XVAL-SPH-ROUND2-METHOD-SELECTION-STARTUP-SERIES
-- Method selection for plane Poiseuille startup series validation

INSERT INTO oss_solutions (
    feature_slug,
    repo,
    method,
    evidence_grade,
    external_anchor,
    notes,
    created_at
) VALUES (
    'XVAL-SPH-ROUND2-METHOD-SELECTION-STARTUP-SERIES',
    'TensorSPH',
    'Boussinesq (1868) analytical startup series + ISPH projection solver. '
    'Series: u(y,t) = u_s(y) - (32 u_max/π³) Σ sin(nπy/H)/n³ · exp(-n²π²νt/H²). '
    'Solver: Brookshaw viscous Laplacian + pressure projection.',
    'REAL',
    'Batchelor (1967) §4.2; Morris, Fox & Zhu (1997) JCP 136:214',
    'Max profile L2 error 5.70% at H/dx=8, 0.63% at H/dx=20. '
    'Spatial order 1.76-1.88 (approaching 2nd order). '
    '11 unit tests passing. Benchmark: benchmarks/isph_channel_startup/run.py',
    '2026-11-14'
);
