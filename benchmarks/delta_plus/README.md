# Sun δ⁺-SPH in TensorSPH

Use `from tensorsph import delta_plus_step` to advance a 2-D single-phase
liquid. Pass the returned position, physical velocity and density to the next
step. Mass stays fixed. Fixed or moving walls can supply station positions,
velocities and normals. The step returns the force on every wall station in
N per metre of span. The one-step interface also runs without TensorFEM.

## Discretization

- Linear equation of state `p = c0² (rho − rho0)`, evolved continuity, the
  existing gradient-corrected δ diffusion operator (δ = 0.1 by default),
  a symmetric pressure force and the 2-D Wendland C2 kernel at h = 2 Δx.
  The radial artificial viscosity coefficient defaults to α = 0.02.
- Once per physical step, shift positions using Sun's kernel gradient with
  the factor `1 + 0.2 [W(r)/W(Δx)]⁴`, symmetric weight `m_j/(ρ_i+ρ_j)`,
  acoustic CFL `dt c0/h` and reference Mach number `U_ref/c0`. Shortening
  the last step shortens the shift. Shifting does not add physical velocity.
- The smallest eigenvalue of the fluid-only first moment and an outward
  empty-cap test identify exposed particles. Neighbours inside 2h form the
  interface band. The renormalized gradient of the eigenvalue gives the
  normal. Shifting is tangential within the band. Sparse supports (λ < 0.55)
  and unresolved normals in that band are pinned. For immersed walls we
  additionally enforce the nearest station's tangent. Independent normal
  constraints at a contact line pin the positional correction in 2-D.
- This implementation uses a first-order velocity-first acoustic step.
  It extrapolates wall pressure with gravity, reflects normal relative
  velocity for free-slip continuity, and uses a linear acoustic impedance
  term in the wall-normal traction. The impedance and nearest-station
  treatment are TensorSPH's boundary choices for moving, thin walls.
  Fluid and solid receive equal and opposite forces. Prescribed wall
  acceleration is omitted. The implementation covers a single resolution.
  The later negative-pressure TIC switch, adaptive particle refinement and
  2019 consistent transport terms are outside this version.

Method sources: [Sun et al., CMAME 315 (2017), 25–49](https://doi.org/10.1016/j.cma.2016.10.028);
[Sun et al., Applied Ocean Research 72 (2018), §2 and eq. 5](https://doi.org/10.1016/j.apor.2018.01.004).
The [2019 consistent formulation](https://doi.org/10.1016/j.cma.2019.01.045)
introduces further transport terms for long-time integration.

## Analytic free-surface benchmark

Run from a checkout installed with `python -m pip install -e ".[test]"`.

```sh
python benchmarks/delta_plus/elliptical_drop.py --cells 8 12 16
python -m pytest -q tests/test_delta_plus.py
```

A circular unit-radius patch begins with velocity (x, −y), zero gravity,
zero physical viscosity and pressure ρ₀(1−r²)/2. Its incompressible Euler
solution has stretches a and b with ab = 1 and
a′²(1+a⁻⁴) = 2. The benchmark inverts the exact time integral independently
of the SPH solver. It measures the stretches from the mass-weighted second
moments relative to their initial values to remove the starting circle's
lattice bias. The record also reports the outermost-particle stretches.
See [Monaghan, *Simulating Free Surface Flows with SPH* (1994)](https://doi.org/10.1006/jcph.1994.1034).

At t = 0.4 the exact stretches are 1.464774 and 0.682699. The table gives
signed relative errors of the measured stretches. Runs use float64 on CPU,
c₀ = 20, U_ref = 1 and acoustic CFL = 0.20.

| Particles per radius | Fluid particles | Long-axis error | Short-axis error | Peak Mach | Peak density deviation |
|---:|---:|---:|---:|---:|---:|
| 8 | 208 | −0.660% | −0.037% | 0.0590 | 0.294% |
| 12 | 448 | −0.417% | +0.151% | 0.0598 | 0.206% |
| 16 | 812 | −0.279% | +0.229% | 0.0618 | 0.186% |

Both second moments stay within 1% at this time. Refinement reduces the
long-axis error in this sequence. The short-axis error grows in magnitude on
these three rungs. They do not establish a convergence order. [Raw CPU records](ellipse_results.json).
At 12 particles per radius, halving the time step (CFL 0.10) gives errors
of −0.408% and +0.167%. A separate 15 × 15 perturbed lattice check isolates
the shift with zero gravity, pressure and velocity. Over 80 steps the bulk
nearest-neighbour spacing RMS error falls from 0.2391 Δx to 0.1819 Δx while
all physical velocities remain zero. Both checks are reproducible with the
runner (`--cells 12 --courant .1`) or the tests above.

The [coupled Test 19 benchmark](../spheric19/README.md#delta-plus-sph-comparison)
tests a moving gate and an elastic plate against experimental measurements.
