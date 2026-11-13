# Taylor-Green reference: what exactly is being compared against

**Case**: `CASE-TAYLOR-GREEN-SPH` · **Repo**: TensorSPH · generated
2026-11-13T11:52:41Z from `/root/tp-agent/work/m18123/runs/taylor_green/m16664/results.json`.

## 1. Citation

> G. I. Taylor & A. E. Green (1937), *Mechanism of the production of small eddies
> from large ones*, **Proc. R. Soc. Lond. A 158, 499-521**.

> G. K. Batchelor (1967), *An Introduction to Fluid Dynamics*, Cambridge
> University Press, section 3.10 (periodic vortex array).

The field

```
u(x,y,t) = U0 exp(-2 nu k^2 t) sin(kx) cos(ky)
v(x,y,t) = -U0 exp(-2 nu k^2 t) cos(kx) sin(ky)
p(x,y,t) = (rho U0^2 / 4) exp(-4 nu k^2 t) (cos 2kx + cos 2ky)
```

is an exact unsteady solution of the incompressible Navier-Stokes equations, so
the kinetic energy decays as `E(t) = E(0) exp(-4 nu k^2 t)`.  The benchmark target
is the exponent `4 nu k^2`, which for this case is `0.8 pi =
2.5132741 1/s`.

## 2. Machine re-derivation instead of a transcribed table

The owner rule is zero hand-copying, and this work ran offline: the 1937 paper
was **not** re-read page by page.  The reference is therefore re-derived and
verified on every run:

1. `ns_residual()` substitutes the closed form into momentum + continuity with
   autograd on a 48x48 periodic grid.  Measured: scaled momentum residual
   `4.42e-16`, scaled continuity residual
   `1.77e-16` → round-off level.
2. `analytic_decay_rate_from_field()` fits `ln E` of the *analytic* field over the
   same window as the SPH run: `2.5132741` vs
   `2.5132741` 1/s (rel. error `3.5e-14 %`).

Both are asserted by `tests/test_taylor_green_benchmark.py` and stored in the run
artifacts, so the reference cannot drift.

## 3. Scope of the citation

* The paper gives the solution and its decay mechanism; it does **not** tabulate a
  discretisation error.  The 8 % criterion is the case-pool tolerance.
* The SPH discretisation (Morris viscous operator, Tait EOS, minimum-image
  periodicity) is ours; its validity window (`Re <= 5`, `M <= 0.02`) is *measured*
  in the case report, with the numbers that establish it.
