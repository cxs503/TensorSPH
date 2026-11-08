# Claim-validity gate audit refresh (m6527)

*TensorSPH, verification record. The numbers below are read from
`benchmarks/claim_validity/audit.json` and `benchmarks/verification_ledger/ledger.json`
at generation time; nothing is hand typed.*

## ① Case and problem

The claim-validity gate (`benchmarks/claim_validity/`) sits **before** the tolerance
ladder of `docs/verification_ledger.md`. The ladder answers one question only — *how
far is the number from the reference?* — and cannot answer the question that has to
come first: **was there a run left to measure when the number was taken?** A dead or
blown-up run cannot support a `FAIL` grade (which asserts a measured physics defect);
the honest grade for such a reading is `WITHHELD` (evidence insufficient).

The gate's anti-drift test
`tests/test_claim_validity.py::test_committed_audit_json_is_up_to_date` was **red**:
the committed `audit.json` (31 claims) no longer matched a fresh regeneration (36
claims). Five claims had been added to the ledger after the audit was last generated,
so the gate was not reflecting the current evidence base.

## ② What was done

1. **Root cause.** The ledger gained 5 claims (31 → 36) without regenerating the
   audit. The 5 claims, with their ledger grades:

   | claim_id | ledger grade | artifact |
   |---|---|---|
   | `density_diffusion.decay_rate.worst_of_ladder` | PASS | `benchmarks/density_diffusion_decay/decay_rate` |
   | `density_diffusion.finite_time.amplitude` | PASS | `benchmarks/density_diffusion_decay/decay_rate` |
   | `effective_sigma.sigma_eff_over_sigma.dx5mm` | FAIL | `benchmarks/effective_sigma/out/report.json` |
   | `poiseuille.centre_velocity.finest` | PASS | `benchmarks/poiseuille/out/results.json` |
   | `poiseuille.wall_shear_ratio.finest` | WITHHELD | `benchmarks/poiseuille/out/results.json` |

2. **Fix.** Regenerated `audit.json` + `docs/verification_claim_validity.md` via
   `python benchmarks/claim_validity/validity.py`. The regeneration is deterministic
   (every number is read from the artifacts), so the committed file now equals a fresh
   regeneration exactly.

3. **Verification.** `tests/test_claim_validity.py` is green (14 passed). The change
   touches only the two claim-validity data files; no `src/` code was modified.

## ③ Results

| metric | before | after |
|---|---|---|
| `n_claims` | 31 | 36 |
| `UNPROBED` | 23 | 28 |
| `n_probed` | 8 | 8 (unchanged — the 5 new claims carry no telemetry) |
| anti-drift test | RED | GREEN |

The 5 new claims are all `UNPROBED`: their artifacts carry no containment / liveness /
estimator telemetry, so run validity cannot be asserted from them. That is an honest
state, not a pass.

## ④ Figures

None. This is a pure gate / bookkeeping fix (deterministic data-file regeneration); no
physics was computed, so there is no field quantity or comparison curve to plot.

## ⑤ Notes / limitations

* The 5 new claims are `UNPROBED`. To let the gate grade them, their artifacts need
  telemetry (final-state geometry, energy, multi-estimator) and a probe — a follow-up.
* The full test suite could not be driven to a clean finish on this machine (load ≥ 100
  with concurrent agent pytest). The change is isolated to claim-validity data files,
  and the only test that reads them (`tests/test_claim_validity.py`) is green.
* The gate still only covers 8 of 36 claims; the remaining 28 are `UNPROBED` by
  construction (no telemetry), which the doc states explicitly.

## ⑥ Error determination

The relevant "quantity" is the drift between the committed `audit.json` and a fresh
regeneration:

* **Before:** 5 claims missing (31 vs 36) → anti-drift test RED.
* **After:** 0 drift (committed == fresh, byte-identical after normalisation) →
  anti-drift test GREEN.

No physics quantity is involved, so the 3 % / 15 % ladder does not apply; the relevant
check (drift == 0) is satisfied exactly.
