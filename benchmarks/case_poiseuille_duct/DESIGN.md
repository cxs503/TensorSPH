# CASE-DUCT-POISEUILLE-3D — predeclared design (written before any duct run)

Case-pool entry extending the **算例池·泊肃叶流** family (`CASE-POISEUILLE-SPH`)
from the plane channel to a **rectangular duct**.  Ledger tolerance: analytic
reference, **< 5 %**.

## 1. Why this case is not covered yet

`benchmarks/case_poiseuille` (2-D plane channel, R1 steady/start-up + R2
Womersley), `benchmarks/poiseuille_mirror`, `poiseuille_slip` and
`src/tensorsph/case_poiseuille_morris.py` cover every 2-D regime.  On the 3-D
side the repository verifies a periodic shear-wave decay (no wall), a hydrostatic
column (wall *pressure*, no velocity) and a kernel/operator sweep.  **No 3-D
shear flow with no-slip walls has ever been compared with an analytic
solution.**  A duct closes that gap with one geometry, one body force and a
closed-form reference.

## 2. Problem

```
nu (u_yy + u_zz) = -g0,   0 <= y <= H, 0 <= z <= W,   u = 0 on all four walls
```

    u(y,z) = (16 g0 H^2 W^2 / (nu pi^4)) sum_{m,n odd}
             sin(m pi y/H) sin(n pi z/W) / (m n (m^2 W^2 + n^2 H^2))

Bulk velocity (used for the flow-rate gate):

    ubar = (64 g0 H^2 W^2 / (nu pi^6)) sum_{m,n odd} 1/(m^2 n^2 (m^2 W^2 + n^2 H^2))

Reference provenance: White, *Viscous Fluid Flow* 3rd ed. §3-3; Bruus,
*Theoretical Microfluidics* §3.3 (duct series, same modes).  Limits used as
self-checks: `W/H -> inf` reproduces the plane parabola `g0 (H-y) y/(2 nu)`;
`W = H` gives the classical square-duct ratio `u_c / (g0 H^2/8nu) = 0.5894`.

## 3. Parameters (fixed before the ladder)

| quantity | value |
|---|---|
| `H` | 0.1 m |
| `W/H` | 1.0 (square) and 4.0 (wide) |
| `nu` | 0.01 m2/s |
| `rho0` | 1000 kg/m3, Tait, `c_s = 20 m/s` |
| `g0` | 9.81 m/s2 (streamwise body force) |
| `x` | periodic, 4 columns (flow is `x`-invariant) |
| resolutions | `H/dx = 6, 8, 12, 16` |
| BC | mirror (ghost) no-slip on all four planes, rebuilt every step |
| settle / average | 3 tau, last 25 % averaged, `tau = H^2/(nu pi^2) = 0.101 s` |
| control | `slip="free_slip"` on all four planes |

## 4. Predeclared gates

| gate | metric | threshold |
|---|---|---|
| **D1** | series vs independent FD Poisson solve (96x96), centre velocity | ≤ 1 % |
| **D1b** | same, L2 over the duct | ≤ 1 % |
| **D2** | series truncation `nmax=81` vs `nmax=321` at the centre | ≤ 0.1 % |
| **D3** | centre velocity error at the finest rung (per aspect) | ≤ 5 % |
| **D4** | interior L2 (`0.1 ≤ y/H,z/W ≤ 0.9`) at the finest rung | ≤ 5 % |
| **D5** | bulk velocity (flow rate) at the finest rung | ≤ 5 % |
| **D6** | interior L2 at *every* rung ≤ 5 % and non-increasing | pass |
| **D7** | shape effect: `u_c(square)/(g0 H^2/8nu)` vs 0.5894 | ≤ 5 % |
| **D8** | aspect discrimination: `u_c(wide)/u_c(square)` vs series ratio 1.690 | ≤ 5 % |
| **D9** | free-slip control: centre velocity > 2x the no-slip steady value (no false steady state) | pass |

D7 is the falsification lever: a boundary condition that acts only on the first
pair of walls returns the plane parabola, i.e. **+70 %** on the centre velocity,
which no amount of resolution hides.

## 5. What counts as failure

* Any of D3–D8 failing at the finest rung → honest FAIL with the measured
  numbers, no retuning.  The expected failure mode is WCSPH numerical
  dissipation acting as extra viscosity (interior deficit growing with
  resolution mismatch near the corners).
* D1/D2 failing → the *reference* is wrong and the case is void until fixed.

## 6. Compute reality (recorded before the ladder)

The 3-D operator set evaluates the cubic-spline kernel through several
`torch.where` branches in float64; measured on the control plane,
**1.7 s per time step at 1920 particles** (`H/dx = 8`, 4 columns), i.e. ~43 min
for one rung and ~8x that at `H/dx = 16`.  The full predeclared ladder
(2 aspects x 4 rungs + control) is therefore **out of reach for one mission
slot**; the artifacts shipped with this note grade the reference layer (D1, D1b,
D2) and a **coarse mechanism rung** (`H/dx = 6`), and leave D3–D9 graded from the
committed numbers where available and reported as *not_run* where not.  Nothing
is quoted as passing that was not measured.
