# CASE-POISEUILLE-SPH / non-Newtonian arm — power-law channel flow

**Status: capability landed; the non-Newtonian accuracy is NOT yet inside the 5 % band.**
Honest verdict first: the Newtonian control (n = 1) reproduces the reference to
0.90 % profile RMS, but the shear-thinning arm (n = 0.5) misses it by
12.4 % at H/dx = 8. The cause is measured and diagnosed below, not hidden.

## 1. Why this arm exists
`CASE-POISEUILLE-SPH` was Newtonian only: `benchmarks/poiseuille`, `poiseuille_mirror`
(mirror walls), `poiseuille_slip` (Navier slip), `case_poiseuille` (Womersley + start-up)
and `case_poiseuille_duct` all pass a **constant** viscosity to the viscous operator, so a
mis-scaled shear stress would be invisible to every one of them. This arm adds the
Ostwald–de Waele fluid `tau = K |du/dy|^(n-1) du/dy` and grades it against the closed form.

## 2. What was added
| file | role |
|---|---|
| `src/tensorsph/rheology.py` | power-law closure (with the `n<1` shear-rate floor and `mu_max` cap), SPH strain-rate estimate, exact plane-channel solution and wall-shear invariants |
| `src/tensorsph/droplet_impact.py` | `rhs(..., nu_field=...)`: optional per-particle kinematic viscosity. Fluid/fluid pair value = arithmetic mean; a fluid/wall pair uses the fluid partner's value (a mirror ghost *is* that particle's image). `nu_field=None` is the previous path bit for bit |
| `benchmarks/case_poiseuille_powerlaw/run.py` | ladder + control driver: per-regime `K` from `consistency_for_umax` (same centre-line speed for every `n`), mirror no-slip walls, steady-state row binning, gates P1–P7 |
| `tests/test_case_poiseuille_powerlaw.py` | 19 tests (see §7) |

## 3. Reference, parameters, provenance
`G = rho g = 9810 Pa/m`, `H = 0.1 m`, `rho0 = 1000 kg/m^3`, target `u_max = 1.2 m/s`.

| n | K (Pa s^n) | nu_wall | nu_cap | u_max exact | tau_w exact |
|---|---|---|---|---|---|
| 0.5 | 57.81 | 0.006813 | 0.03047 | 1.2000 m/s | 490.5 Pa |
| 1.0 | 10.22 | 0.01022 | 0.01022 | 1.2000 m/s | 490.5 Pa |

`u(y) = n/(n+1)(G/K)^(1/n)[(H/2)^((n+1)/n) - |y-H/2|^((n+1)/n)]`,
`tau_w = G H/2` (**independent of n** — a redundant gate). The algebra is re-derived with
sympy in the driver and the momentum balance is re-checked numerically:
`max|residual|/G = 5.4e-11` (n=0.5),
`9.1e-11` (n=1).

## 4. Measured (probe, H/dx = 8, t_end = 0.4 s, columns = 8)
| quantity | n = 0.5 | n = 1.0 | 5 % band |
|---|---|---|---|
| centre line `u_max` | -14.89 % | -2.88 % | fail / pass |
| profile RMS (0.1 <= y/H <= 0.9) | 12.37 % | 0.90 % | fail / pass |
| wall shear `K|du/dy|^n` vs `G H/2` | -17.64 % | -2.65 % | fail / pass |
| `n = 1` closure returns exactly `K` | — | yes | pass |
| steady drift at t = 0.4 s | 0.17 | 0.08 | not settled |

## 5. Diagnosis of the n = 0.5 deficit
The closure is only as good as the strain-rate field it is fed. Measured on a linear-shear
lattice with mirror ghosts (`test_shear_rate_is_linear_in_the_shear_and_row_uniform`): the
estimate is **exactly proportional** to `du/dy` (ratio 2.000000 when the shear doubles) and
row-uniform to 11 %, but it is **biased low by about 46 %** in absolute value — the discrete
kernel sum normalises to `rho0(1+b)` rather than `rho0`, and the truncated near-wall support
removes the rest. For `n = 0.5`, `mu ∝ gamma_dot^(-1/2)`, so a 46 % low `gamma_dot`
over-states `mu` by about 36 %: over-viscous, which is the sign and rough size of the observed
`u_max` (−14.9 %) and wall-shear (−17.6 %) deficits.
**Identified next step, not done here: Shepard-renormalised (first-order consistent) velocity
gradient inside `rheology.shear_rate`; then re-run the ladder.** The Newtonian arm needs no
closure at all, which is why it is already inside the band.

## 6. Not run (explicit)
* The multi-resolution ladder (H/dx = 8, 16 × (0.5, 1.0, 1.5) + closure-off control) was
  **submitted to the cluster as job `cj4514` and had not started** (queue position 75) when
  this note was written; the local cap is < 10 min per command and one rung exceeds it on
  8 threads. P3 (convergence) and P4 (closure-off control) are **not_run**, not passed.
* `n = 1.5` (dilatant) was never executed for the same reason.
* Quantitative claim of the cap on `mu` (the `n<1` regularisation changes the exact profile
  by < 1e-4 of `u_max` in the band it acts on) is analytic only, not yet measured.

## 7. Verification run
`PYTHONPATH=src pytest tests/test_case_poiseuille_powerlaw.py -q` → **19 passed** (6 s):
reference vs momentum balance; wall-shear n-invariance; `n=1` closure exactness; closure
limits/validation; `rhs(nu_field=)` uniform field == constant path **bit for bit**,
viscous-term linearity, rejection of malformed fields; strain-rate proportionality and row
uniformity; rigid-body zero; power-law vs Newtonian shape separation (> 10 % of `u_max`, so
the case cannot be passed by a solver that ignores the closure); one short driver probe.

## 8. Reproduce
```
PYTHONPATH=src python benchmarks/case_poiseuille_powerlaw/run.py \
    --resolutions 8 16 --indices 0.5 1.0 1.5 --t-end 0.8 --t-average 0.25
```
