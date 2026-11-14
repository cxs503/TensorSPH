# XVAL — SPHinXsys `sycl-execution-policy-not-transferable` vs TensorSPH

Anchor: `XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE`　|　generated: 2026-11-13T07:03:21Z　|　commit base: `7818cd61ec299dc6cce8a1be86dd433661084226`

## 1. Problem, reference and provenance

**Our side.** TensorSPH has no execution-policy layer at all: every kernel is a
vectorised tensor expression, so a caller cannot share, copy or transfer a
policy -- and cannot control launch geometry either.

**The OSS practice.** SPHinXsys deliberately makes the device execution policy
*non-transferable*. Read from source (nothing transcribed; the harness locates
each fact by regex at run time and stores only line numbers and the sha256 of
the matched text -- see `artifacts/xval_execution_policy/provenance.json`):

- OSS commit `5241a8902019e2b61559460748342dd9c29bd2a9`, checkout available: `True`; facts located: 11/11.
- `compile_time_policy_switch` — src/shared/particle_dynamics/execution/execution_policy.h:77 (sha256 of matched text `bc1dc31502c5ff83…`)
- `main_execution_policy_alias` — src/shared/particle_dynamics/execution/execution_policy.h:78 (sha256 of matched text `1aa326611fd19103…`)
- `device_policy_template` — src/shared/particle_dynamics/execution/execution_policy.h:67 (sha256 of matched text `a727967e71e85fbc…`)
- `execution_instance_copy_deleted` — src/src_sycl/shared/particle_dynamics/implementation_sycl.h:46 (sha256 of matched text `af1143088d6dbe3f…`)
- `execution_instance_assignment_deleted` — src/src_sycl/shared/particle_dynamics/implementation_sycl.h:47 (sha256 of matched text `4cdbdc157489c944…`)
- `singleton_accessor` — src/src_sycl/shared/particle_dynamics/implementation_sycl.h:49 (sha256 of matched text `8f7ef52e762a112e…`)
- `default_device_selector` — src/src_sycl/shared/particle_dynamics/implementation_sycl.h:59 (sha256 of matched text `1819cccaa865b70d…`)
- `work_group_clamp` — src/src_sycl/shared/particle_dynamics/implementation_sycl.h:62 (sha256 of matched text `34279abda29fada8…`)
- `uniform_nd_range_round_up` — src/src_sycl/shared/particle_dynamics/implementation_sycl.h:79 (sha256 of matched text `8e5395301803e4ed…`)
- `tail_mask_in_kernel` — src/src_sycl/shared/common/algorithm_primitive_sycl.h:134 (sha256 of matched text `b1648a6a8cbe95fb…`)
- `device_memory_bound_to_queue` — src/src_sycl/shared/particle_dynamics/implementation_sycl.h:102 (sha256 of matched text `6102605fff7ac400…`)

In words: (a) the policy is a *tag type* chosen by the `SPHINXSYS_USE_SYCL`
compile switch, so it is not a run-time value that can be handed over; (b) the
state it resolves to lives in `ExecutionInstance`, whose copy constructor and
assignment operator are `= delete` and which is reachable only through
`getInstance()`; (c) device memory is allocated and copied through that one
queue, so a device pointer belongs to the instance that made it; (d) the launch
is padded to whole work-groups, which obliges the kernel body to carry an
explicit tail mask.

**What this cross-validation measures.** Two claims of the practice, on a real
SPH work load: (i) the padded-masked launch reproduces the shipped vectorised
summation density for *any* work-group size (the numerics are policy-invariant
even though the policy object is not transferable); (ii) removing the tail mask
breaks the same numbers, so (i) is not vacuous.

## 2. Method and implementation

| artefact | role |
|---|---|
| `src/tensorsph/execution_policy.py` | port: policy tags, frozen main policy, non-copyable `ExecutionInstance` (device + work-group size), replica of `getUniformNdRange`, `iterate_blocks` yielding padded blocks with their mask |
| `src/tensorsph/wcsph/swept.py` | two arms of the same discrete sum: vectorised `index_add` (reference) and the padded masked particle sweep (port), plus a scalar per-particle loop used as a third, code-independent opinion |
| `tests/test_execution_policy.py` | 117 assertions: singleton/non-copyability, OSS round-up formula vs an independent `ceil` re-derivation, block coverage, invariance across 11 work-group sizes, negative control |
| `benchmarks/xval_sycl_execution_policy/run.py` | the measurement harness; writes `results.json`, `provenance.json`, `summary.json`, 4 figures |

Work flow per arm: build the particle set (fluid column + 3 layers of dynamic
boundary particles) once, build the pair list once, then evaluate the summation
density through each arm on the *same* inputs. The launch bound is the fluid
count (the OSS main loop is over the fluid particles; boundary particles carry
their own dynamics), while the pair list still contains the fluid/wall pairs.
Tait pressure is then applied to both density fields, so the comparison also
covers the non-linear amplification of any density defect.

## 3. Parameters (complete, incl. dimensionless groups)

| quantity | value |
|---|---|
| tank / column | [0.8, 0.4] m / [0.2, 0.4] m |
| dx (base) | 200 fluid + 420 boundary particles, 4650 pairs |
| kernel | quintic, h/dx = 1.3 |
| support/dx | 3.9000 |
| particles per support | 15.210 |
| pairs per particle (symmetrised) | 15.00 |
| rho0, c0, gamma (Tait) | 1000.0 kg/m3, 25.0 m/s, 7.0 |
| dimensionless | h/dx, support/dx, particles/support above; Mach and CFL are undefined for a static snapshot (no velocity scale enters) |
| environment | python 3.11.6, torch 2.7.1, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38, 128 cores, 128 torch threads, float64 |
| policy | USE_DEVICE_BACKEND=False -> ParallelPolicy, device=cpu, work-group=64 |

## 4. Cross-validation table (ref vs got, gate check)

Gate: owner standard 15 % (hard), engineering target 1e-12 for a
numerics-neutral refactor (predeclared, both fixed before the run).

| quantity | ref | got | max rel. err | <15 % | <1e-12 | detail |
|---|---|---|---|---|---|---|
| summation density, worst work-group size | vectorised index_add (TensorSPH shipped path) | ported sweep | 0.000e+00 | PASS | PASS | local_size=1 |
| Tait pressure, worst work-group size | Tait law applied to the vectorised density | ported sweep | 0.000e+00 | PASS | PASS | local_size=1 |
| summation density vs scalar per-particle loop | independent scalar loop (no tensor reduction) | ported sweep | 6.418e-16 | PASS | PASS | small cases only |
| negative control (tail mask removed) | vectorised index_add | ported sweep | 6.354e+02 | FAIL | FAIL | local_size=1024, 824 idle work items |

Per work-group size (base resolution):

| work-group | padded | idle | density max | density median | pressure max | control max | control particles >15 % | vectorised ms | sweep ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 200 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0 | 0/200 | 22.4 | 234.5 |
| 3 | 201 | 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.8 | 1/200 | 23.2 | 1227.5 |
| 7 | 203 | 3 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.3 | 1/200 | 29.4 | 484.0 |
| 8 | 200 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0 | 0/200 | 48.4 | 790.8 |
| 16 | 208 | 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 6.2 | 1/200 | 31.6 | 356.1 |
| 64 | 256 | 56 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 43.2 | 1/200 | 32.5 | 171.9 |
| 128 | 256 | 56 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 43.2 | 1/200 | 32.1 | 105.7 |
| 256 | 256 | 56 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 43.2 | 1/200 | 32.1 | 61.7 |
| 1024 | 1024 | 824 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 635.4 | 1/200 | 30.9 | 392.1 |

Third opinion (scalar per-particle loop, small cases): n=620: vectorised 5.69e-16, swept 5.69e-16; n=372: vectorised 6.42e-16, swept 6.42e-16.

Reading: the ported sweep is equal to the shipped reference to the last bit at
every work-group size tested (worst 0.00e+00 at local_size=1, 0 idle work items), and the two independent references agree at round-off, so the
agreement is not a shared-implementation artefact. The control row is the
measured cost of dropping the mask: 1024 work-group gives 824 idle work items, all clamped
onto one particle, and its density error reaches 635.4x (63544.62 %).

## 5. Sensitivity and convergence

**(a) Work-group size (11 values, base resolution).** Density error is
independent of the launch geometry: max 0.00e+00,
i.e. flat, while the padding cost `(ceil(n/l)*l - n)/n` grows like `1/n` and
reaches 412.0 % at local_size=1024.

**(b) Particle count (dx axis).** Density error stays at round-off at every
resolution; padding cost at fixed work-group falls with n, as the OSS design
intends:

| dx | particles | fluid | local 8 idle | local 8 err | local 256 idle | local 256 err | rho/rho0 median |
|---|---|---|---|---|---|---|---|
| 0.03 | 372 | 78 | 2 | 0.00e+00 | 178 | 0.00e+00 | 0.9992 |
| 0.02 | 620 | 200 | 0 | 0.00e+00 | 56 | 0.00e+00 | 1.0000 |
| 0.0125 | 1148 | 512 | 0 | 0.00e+00 | 0 | 0.00e+00 | 1.0000 |

**(c) Cost of the port (torch CPU, best of 3).** The padded sweep is slower
than the vectorised reference by 1.9x at local_size=256 and 52.8x at local_size=3: one host call per work-group, which is the
honest price of the OSS launch discipline on a backend without kernels. On a
GPU the per-work-group overhead amortises over thousands of work items, which
is why the OSS design pays it there; on torch CPU it does not, so TensorSPH
keeps the vectorised path as the default and the sweep as the reference arm.

## 6. Differences, port trade-offs, failure modes, friction

### Difference list (ours vs OSS)

| aspect | SPHinXsys (SYCL) | TensorSPH (this port) |
|---|---|---|
| policy selection | compile-time `#if SPHINXSYS_USE_SYCL` | module constant frozen at import; `set_main_execution_policy` raises |
| policy state | `ExecutionInstance` singleton, copy deleted | same, plus `__reduce__` refuses pickling (a Python process boundary is the analogue of a SYCL context boundary) |
| device choice | `sycl::default_selector_v` | `cuda` if available else `cpu` |
| work-group clamp | `min(max_work_group_size, 64)` | `min(max_threads_per_block, 64)` on CUDA, constant 64 on CPU (torch exposes no work-group notion) |
| padded range | `getUniformNdRange` (conditional round-up) | byte-for-byte replica of the same expression, cross-checked against an independent `ceil` re-derivation |
| tail mask | `if (global_id < loop_bound)` inside the kernel body | mask materialised in `WorkBlock` so a missing mask is visible to a test |
| memory ownership | `malloc_device`/`memcpy` bound to the queue | not ported: torch owns storage and `.to(device)` already refuses cross-device arithmetic |
| loop bound | fluid particles in the main loop | `bound=` parameter, benchmark uses the fluid count; rows beyond it keep the self term |
| boundary condition | dynamic boundary particles with their own dynamics | same particle set, static snapshot, wall mass = rho0 dx^d, no wall pressure extrapolation in this measurement (density-only comparison) |

**Port decision.** Ported: the singleton discipline, the round-up formula, the
padded-and-masked sweep, and the tests that pin them. Not ported: the
compile-time tag (substituted by a frozen module constant -- the same
non-transferability, enforced by an exception instead of the compiler) and the
SYCL memory machinery (torch has no equivalent, and inventing one would be
untestable on this host).

### Failure modes

- **Missing tail mask** (the controlled failure): idle work items land on the
  last particle and its density reaches 635.4x the reference.
- **Empty row hides the bug**: with `fluid_count` pruning and a wall-only tail,
  the clamped particle had no neighbours and the control measured exactly 0.
  The benchmark therefore sweeps the fluid bound, and the test suite keeps a
  wall-free lattice so the control is never vacuous.
- **Sharing the instance is impossible by construction** (copy/deepcopy/pickle
  raise), so no test can silently mutate another arm's launch geometry; a
  caller that wants a different geometry must call `set_work_group_size`
  explicitly, which the OSS API also allows.

### Honest limits

- This compares **an architectural practice**, not a physics result: the
  quantity under test is a numerical identity, so the 15 % gate is trivially
  satisfiable and the informative threshold is the predeclared 1e-12.
- The OSS side was **read, not run**: no SYCL device is available on this host
  (loong64, no `nvidia-smi`, no `icpx` toolchain in the image), so the
  comparison is source-to-implementation, not run-to-run.
- Equality to the last bit is a measured fact *for this summation order*; the
  test gate is 1e-12, which is what a float64 summation may be trusted with.
- No time integration, no wall pressure model, no free-surface closure: the
  density field is a static snapshot with uncalibrated lattice mass (rho/rho0
  range 0.715-1.080),
  which is legitimate for an equivalence test but is not a hydrostatic result.

### Ecosystem friction

- `GET {TENSOR_API_BASE}/tensor/knowledge/search` returns
  `engine_offline` (connection refused on :8021), so the mandatory opening
  knowledge-base check had to be answered from the repository itself
  (`docs/oss_inventory_sph/registry.json`, `gaps.json`) plus
  `/tensor/negative-knowledge/search` (covered count: 0 hits for `sycl`).
- Direct GitHub access from the host is refused (`Connection reset by peer`),
  so the OSS facts were pinned against an existing local checkout; the harness
  degrades to `checkout_available=false` rather than to hand notes.

## 7. Reproduce

```bash
python -m pytest tests/test_execution_policy.py -q
python benchmarks/xval_sycl_execution_policy/run.py
python scripts/xval_execution_policy/render_report.py
```

Figures: `artifacts/xval_execution_policy/fig_*.png` —
- `fig_error_vs_workgroup_size.png` — Max relative density error against the vectorised reference as a function of the padded launch geometry; the ported sweep is flat at round-off, the unmasked control sits above the 15 % gate whenever the launch has idle work items.
- `fig_padding_cost.png` — Idle fraction of the padded launch (ceil(n/l)*l - n)/n for three work-group sizes: the cost the OSS design accepts in exchange for a uniform nd-range, bounded by (l-1)/n.
- `fig_density_field.png` — Left: summation density field of the dam-break column with dynamic boundary particles. Right: relative deviation of the padded masked sweep, on a log scale, for the worst-padded work-group; the deviation stays at round-off everywhere including the clamped last particle.
- `fig_wallclock.png` — Wall clock of the two arms as a function of work-group size (CPU backend): the padded sweep pays per-work-group overhead, which is the honest price of the OSS launch discipline on a host backend.


## 8. Verification follow-up (branch `agent/m19962`, 2026-11-14)

An independent re-measurement of this deliverable is in
`docs/xval_sycl_execution_policy_verification_m19962.md` (harness:
`benchmarks/xval_sycl_execution_policy/verify_convention.py`). Everything above
reproduces, but **one claim above is stated without its precondition** and is
corrected here:

- "the padded-masked launch reproduces the shipped vectorised summation density
  for *any* work-group size" — true for a **uniform** smoothing length, which is
  the only case this report measured. With a spatially varying `h` the shipped
  reference evaluates each pair once with `h[i]` (symmetric) while the sweep
  evaluates every row with its own `h` (one-sided, as the OSS device launch
  must), so the two arms are different operators: measured gap 0 to 44.16 %
  (dx=0.04 m, `h` modulated ±90 %), 5.88 % at dx=0.02 m, `h` modulated ±50 %,
  and 2.02 % at dx=0.04 m, `h` modulated ±10 % — four orders of magnitude above
  the 1e-12 target declared above, and over the 15 % gate in the coarse corner.
- The *launch-geometry* invariance itself is not affected: against a
  convention-matched (per-row-`h`) reference the sweep is bit-identical
  (ULP 0) at every work-group size and every amplitude tested, so the
  padded-masked sweep is sound; only the comparison target was over-general.
- New assertions pin this scope: `tests/test_execution_policy.py` section 5,
  and `summation_density_vectorised_row_h` in `tensorsph.wcsph.swept` provides
  the convention-matched reference.
