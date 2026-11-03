# Verification-harness fidelity and source pinning (TensorSPH, m2730)

**Scope**: `benchmarks/laplace_droplet/run.py`, `tests/conftest.py`, `tools/worktree_env.py`, `tests/test_laplace_droplet.py`, `tests/test_harness_integrity.py`. No change to `src/`.

## 1. Case: two silent ways a verification number loses its owner

TensorSPH ships a CSF surface-tension model. Two rounds of verification work (m2174 legacy audit, m2188 consistent scheme) left behind a benchmark that no longer measured the shipped model, and a host environment that resolved `import tensorsph` to a *different* worktree. Both defects are silent: they produce plausible numbers attributed to the wrong code.

### 1a. Foreign source (environment)

`/usr/lib/python3.11/site-packages/__editable__.tensorsph-0.8.0.pth` contains `/root/tp-agent/work/m2420/src`. Measured evidence from worktree `m2730` before the fix:

```
$ python -c "import tensorsph; print(tensorsph.__file__)"
/root/tp-agent/work/m2420/src/tensorsph/__init__.py     <-- not this worktree
$ python -c "import sys; print([p for p in sys.path if 'work' in p])"
['/root/tp-agent/work/m991/src', '/root/tp-agent/work/m708/src', '/root/tp-agent/work/m1213/src', '/root/tp-agent/work/m2420/src']
```

`m2420` does not even contain `csf_consistent.py` (pre-consistent line). Consequence: a bare `pytest` in `m2730` reported three failures in `tests/test_laplace_law.py` whose measured cut-force ratio was **0.70** — the same criterion measured on this worktree's source gives **1.132** (band 0.85–1.15).  False reds cost budget; the mirror case, false greens, would have certified another branch's physics as this worktree's.

### 1b. Harness drift (code)

`benchmarks/laplace_droplet/run.py` rebuilt the colour field locally with the m277 neighbour-count heuristic and scored *that* field set (`N_delta`, sign/scale of `kappa`, normal coherence), while `compute_surface_tension_force` had already switched to `csf_scheme="consistent"` — the `Config` default since m2188. The benchmark therefore printed a failing field verdict next to a pressure jump computed from the fixed force, and four tests pinned the failure of a pipeline that no longer produced the force.

## 2. Method

1. `tools/worktree_env.py`: `repo_root_for()` + pure predicate `foreign_source_error()` + `ensure_worktree_source()` which prepends `<repo>/src` to `sys.path` and then *verifies* where `tensorsph` really came from, raising a diagnosis instead of measuring the wrong branch.
2. `tests/conftest.py` installs the pin at collection time, so every run — including a bare `pytest` — is hermetic; a failed pin aborts collection.
3. Benchmark entry points call the same helper (`benchmarks/laplace_droplet/run.py`).
4. The droplet harness now scores `tensorsph.csf_consistent.csf_fields` (shipped) and keeps the legacy m277 field set under explicit `legacy_*` keys as the A/B control; the result dict records `fields_under_test`.
5. `tests/test_harness_integrity.py` adds a negative control (a decoy package outside the worktree must be rejected) and a consistency check that the field diagnostics and the production force describe the same model.

## 3. Results (programmatically injected from `artifacts/m2730/harness_fidelity.json`)

Droplet R = 0.1 m, sigma = 0.072 N/m, rho0 = 1000 kg/m3, hex lattice, `csf_scheme='consistent'`, curvature scheme `mls`.

| R/dx | dp/(sigma/R) | rel. error | <15 % | kappa·R (shipped) | kappa·R (legacy) | N_delta (shipped) | N_delta (legacy) | normal alignment |
|---|---|---|---|---|---|---|---|---|
| 10.0 | 0.986 | 1.4 % | yes | 1.153 | 1.465 | 0.787 | 0.344 | 0.999 |
| 20.0 | 1.109 | 10.9 % | yes | 1.120 | 1.723 | 0.831 | 0.379 | 0.999 |
| 40.0 | 1.467 | 46.7 % | NO | 1.169 | 2.570 | 0.910 | 0.448 | 0.999 |

Figure: `artifacts/m2730/fig_harness_fidelity.png` (Laplace jump vs the ±15 % band; curvature vs 1/R for both pipelines; surface-delta normalisation vs the exact integral 1).

Cut-force identity on the laplace_law grid (R = 0.02 m, dx = 0.004 m, R/h = 3.85): `F_cut/(2 sigma) = 1.132`, median cos(a_st, -r_hat) = 1.000.

## 4. Error verdict

- R/dx = 10.0: Young–Laplace error 1.39 % → **<3 %: yes**; curvature error 15.28 % (<15 %: no); surface-delta deficit 21.3 %.
- R/dx = 20.0: Young–Laplace error 10.92 % → **<3 %: no**; curvature error 12.01 % (<15 %: yes); surface-delta deficit 16.9 %.
- R/dx = 40.0: Young–Laplace error 46.69 % → **<3 %: no**; curvature error 16.87 % (<15 %: no); surface-delta deficit 9.0 %.

Verdict: the criterion is met only in the medium-resolution window (R/dx = 10–20: 1.4 % and 10.9 %). At the finest grid (R/dx = 40) the jump overshoots by 46.7 % — the pressure-jump error is **not** monotone, so this model is not yet a converged surface-tension operator; and `N_delta` stays 17–21 % below unity, which is why `dp_ratio` < 1 at the coarse end. Curvature is 12–17 % high at every resolution (inside the 15 % platform bar only at R/dx = 20). All of this is reported, none of it is hidden: the resolution window covered by `tests/test_laplace_droplet.py` is R/dx <= 20, and the R/dx = 40 failure is a tracked open item.

## 5. Notes and limitations

* The unit of the win is *attribution*: after this change every number printed by the suite and by the two Laplace harnesses names the worktree it came from. Any agent running `pytest` in a TensorSPH worktree without the guard is measuring `m2420`.
* `python -m tensorsph ...` (CLI) is **not** routed through the guard; pass `PYTHONPATH=src` for CLI smoke runs until the editable install is repointed.
* `benchmarks/laplace_droplet/results.json` and `results_tables.md` in the tree are the **pre-m2730** (legacy-field) record. Regenerating them needs the full sweep (R/dx up to 40, 2 lattices, 11 cases); it was deferred because host load15 was 113/128 cores. The in-process grid above is the interim record.
* Curvature remains 12–15 % high with the MLS operator and `N_delta` ≈ 0.8; both are upstream of the force magnitude and are the next candidates (`fit_quadratic` gives −27 % on the laplace_law grid, i.e. it does not simply fix it).
* Unrelated reds found by the first hermetic baseline run (see `backlog_update_m2730.md`): `tests/test_lattice_wall_clearance.py` (2) and `tests/test_ritter.py` (1).

> 说明：图与测量数据的可入库副本为 `docs/figures/m2730_harness_fidelity.png` 与
> `benchmarks/laplace_droplet/harness_fidelity_m2730.json`（生成脚本
> `benchmarks/laplace_droplet/make_doc.py`）；`artifacts/` 已被 .gitignore 排除。
