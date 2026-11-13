-- m17409 board audit — PROPOSED rows for tensor_benchmark_cases (NOT EXECUTED by the auditor).
-- Rationale: F2 in AUDIT_m17364.md — the anchor XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION
-- has 0 rows, so no platform tolerance judge exists. The auditor must not set its own referee,
-- so these are handed to the anchor owner / gate to apply after reviewing the numbers.
-- ref = analytic hydrostatic p(y)=rho0*g*(H-y) evaluated at band centres (case_hydrostatic.pressure_profile),
-- got = audit-reproduced values (docs/artifacts/repro_m17364/records_repro.json, identical to the
-- delivered records_dx004_quick.json), dx=0.04 m, 513 steps, interior bands 0-1 only.
--
-- INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, metric, tolerance, ref_value, got_value)
-- VALUES
--   ('XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION',
--    'hydrostatic_dx004_compression_none', 'max_interior_band_pressure_rel_error_pct', 15.0, 0.0, 1.426631),
--   ('XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION',
--    'hydrostatic_dx004_compression_refmean', 'max_interior_band_pressure_rel_error_pct', 15.0, 0.0, 9.099796);
-- NOTE column list must be adapted to the live schema (verified columns in use elsewhere:
--   case_name, metric, tolerance, anchor_slug; ref/got columns to be confirmed before applying).
SELECT case_name, metric, tolerance
FROM tensor_benchmark_cases
WHERE anchor_slug = 'XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION';  -- audit: returns 0 rows