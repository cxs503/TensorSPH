# CASE-SPH-COUETTE-TRANSIENT — predeclared design (written before any run)

Impulsive start-up of plane Couette flow in TensorSPH: fluid at rest between a
stationary lower wall and an upper wall that is set into tangential motion at
`t = 0` with speed `U`. Channel height `H`, periodic along `x`, no-slip on both
walls, no body force.

Anchor case: `CASE-SPH-COUETTE-TRANSIENT` (算例池 · 第 1 轮).
Reference: Morris, Fox & Zhu, *J. Comput. Phys.* **136** (1997) 214–226, §V
(test problems, plane Couette start-up from rest).

## Decision this benchmark has to support

Does the TensorSPH 2-D WCSPH solver with a **moving wall** (`rhs(..., wall_vel=…)`,
`wall_friction = 1`) reproduce the *transient* Couette start-up profile within
the pool tolerance of **8 %** — and is the agreement caused by the tangential
wall shear and the viscous operator, rather than by the initial condition or by
an averaging window that hides the transient?

## Alternatives

* **A (adopted).** Reuse the moving-wall input of `src/tensorsph/droplet_impact.py:rhs`
  (`wall_vel`, the same input the `wave_flume` wavemaker uses) plus the existing
  steady Couette case machinery (`benchmarks/couette/run.py`), and add only the
  *transient* metric: compare the sampled profile at several instants with the
  exact series of the start-up problem. No new physics, no new case geometry.
* **B (rejected).** Start from the analytic linear profile and report the first
  step. Rejected: that is a restatement of the initial condition; it cannot fail,
  so it carries no evidence. The run starts from rest.
* **C (rejected).** Reuse `benchmarks/couette/run.py` as-is (steady state, gate
  5 % on the time-averaged profile). Rejected as a *substitute*: a steady-state
  average cannot distinguish "the viscous operator has the right rate" from "the
  run was integrated long enough". It is kept as the companion steady check.
* **D (rejected).** Fit an effective viscosity from the measured profile and
  report the fit quality. Rejected: one free parameter fitted to one profile
  absorbs exactly the error being measured.

## Hypothesis

H1: the SPH transient profiles at `t* = νt/H² ≥ 0.1` agree with the exact series
within 8 % (L2 over the interior band) at `H/dx ≥ 16`, the error decreases with
`H/dx`, and the profile is *not* reproduced by a free-slip control.

H0 (falsification): L2 error ≥ 8 % at `t* = 0.1` or `0.2`, or error does not
decrease with resolution, or the free-slip control (no tangential wall shear)
reproduces the same profile — the latter would mean the agreement comes from
something other than the wall condition.

## Reference values — how they are obtained (zero hand-copying)

The start-up problem `u_t = ν u_yy`, `u(y,0)=0`, `u(0,t)=0`, `u(H,t)=U` has the
exact solution (separation of variables; the Fourier series of `-U y/H`)

    u(y,t) = U y/H + (2U/π) Σ_{n≥1} (-1)^n / n · sin(nπy/H) · exp(-n²π²νt/H²)

with the companion exact quantities used as gates

    τ_w(t)/(ρνU/H) = 1 + 2 Σ_{n≥1} (-1)^n exp(-n²π²t*)        (wall shear factor)
    Q(t)/(UH/2)    = 1 - (8/π²) Σ_{k≥0} exp(-(2k+1)²π²t*)/(2k+1)²

All **reference numbers are evaluated in code** (`couette_series`, `exact_scalars`)
from these closed forms — nothing is transcribed by hand into the tables. The
series is independently verified in `tests/test_couette_transient_series.py`
against (i) a Crank–Nicolson finite-difference solution of the same PDE, (ii) the
initial condition `u(y,0)=0`, (iii) the steady limit `u=Uy/H`, and (iv) the
`t→0` wall-shear factor `→0`.

Honest scope note: the Morris et al. (1997) PDF is paywalled and was **not**
readable from this work tree, so no numeric table was transcribed from it; the
case definition (impulsive Couette start-up, series reference) follows that
section, and the tolerance decision is carried by the exact solution, which is
parameter-free. This is reported as a deviation, not hidden.

## Metrics (all machine-computed in `run.py`)

Rows are the lattice bins of `benchmarks/couette/run.py:bin_rows`; the interior
band is `0.10 ≤ y/H ≤ 0.90` (keeps the wall-kernel rows out of the L2 norm; the
full band is reported too).

| id | metric | analytic target |
|----|--------|-----------------|
| M1 | interior L2 profile error `‖u_num-u_ex‖₂/‖u_ex‖₂` at each sampled `t*` | 0 |
| M2 | `L∞` error `/U` at each sampled `t*` | 0 |
| M3 | mass-flux ratio `Q_num/Q_ex(t*)` | 1 |
| M4 | wall-shear ratio `τ_w,num/τ_w,ex(t*)`, `τ = ρν(u₁-u₀)/dx` | 1 |
| M5 | steady-limit profile error at `t* = 0.5` (`‖u-Uy/H‖/U`) | 0 |
| M6 | free-slip negative control (`wall_friction = 0`): flux ratio | ≪ 1 |

## Predeclared gates (do not edit after seeing a result)

    M1(t*)  < 0.08   for every sampled t* ≥ 0.1     ← pool tolerance 8 %
    M3(t*)  within  ±8 %  for every sampled t* ≥ 0.1
    M4(t* = 0.1) within ±8 %
    M6      flux at t* = 0.5  < 20 % of the exact value

Stopping rule / exclusions: no sample is excluded after the fact. `t* = 0.05`
is *reported but not gated* — it is declared here, in advance, as the
boundary-layer-resolving limit (`δ/H ≈ √t* ≈ 0.22`, i.e. ~2 cells at `H/dx=10`).
A run whose `max_mach > 0.1` or which produces NaN is invalid and is reported as
such, not re-tuned.

## Variables, confounders, controls

* **Controlled:** `H`, `U`, `ν`, `ρ₀`, `c₀`, `γ`, `h/dx`, wall layers (3),
  initial condition (rest), lattice layout, fixed `dt = min(0.3h/c₀, 0.125h²/ν)`.
* **Changed:** `H/dx` (ladder), `c₀` (compressibility, a *numerical* confounder
  of the WCSPH viscosity reading), `dt` factor `κ` (temporal resolution).
* **Nuisance variables:** WCSPH acoustic jitter (handled by a declared sampling
  window `Δt* = 0.005`, i.e. 5–20 steps, small against the viscous time),
  particle disorder in `x` (handled by row binning), the wall kernel band
  (handled by the interior band for L2, reported separately for shear).
* **Control:** `wall_friction = 0` free-slip run at the reference resolution.

## Predeclared extension (added before the first full-ladder run)

The ladder in the committed artifacts is `H/dx = 10, 16, 24, 32`.  Level 32 is a
*finer* rung added to make the observed convergence order measurable at the
gated instant; no threshold, band or sample time changed with it.

## Repetitions

The integrand is deterministic (fixed `dt`, no random numbers, no adaptive
stepping), so a single run per configuration is the unit of analysis; the
*replication* axis is the resolution ladder and the parameter sweeps. Sample
sizes: 3 resolutions × 5 sample times, 2 sweeps × 2–3 points.

## Analysis (declared in advance)

Per sample time: L2 and L∞ errors, flux and shear ratios, all reported as tables
plus a `u(y)` overlay figure and a `Q(t)`/`τ_w(t)` transient figure. Convergence:
error vs `H/dx` with the observed order fitted to the two finest points. No
outlier removal, no post-hoc tolerance relaxation.