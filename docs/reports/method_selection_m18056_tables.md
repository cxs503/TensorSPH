### 3.1 证据登记表（程序化抽取，每行带源文件与 sha256）

| 方法 | 算例 | 度量 | 实测误差 [%] | 本算例门 [%] | <15% | 过本门 | 来源 |
|---|---|---|---|---|---|---|---|
| `wcsph_delta_sph` | CASE-WAVE-FLUME | max gauge wave-height error vs piston-transfer target (H/dx=3.2, delta=0.1) | 2.968 | 10 | ✔ | ✔ | `docs/reports/wave_flume_numbers.json` |
| `compressible_riemann` | CASE-SPH-SOD-SHOCK-COMPRESSIBLE | max abs relative error over the gated shock quantities (rho*, p*, u*, wave positions) | 0.7768 | 5 | ✔ | ✔ | `benchmarks/case_sod_shock_compressible/result.json` |
| `heat_conduction_operator` | CASE-SPH-CONDUCTION-1D | max normalised error vs the erfc similarity solution | 0.0002181 | 3 | ✔ | ✔ | `benchmarks/case_conduction_1d/result.json` |
| `csf_surface_tension` | CASE-LAPLACE-SPH | Laplace pressure jump rel. error (delta_p vs 2*sigma/R) | 5.599 | 15 | ✔ | ✔ | `benchmarks/laplace_law/validation_report.json` |
| `csf_surface_tension` | CASE-LAPLACE-SPH (consistent CSF) | delta_s-weighted mean curvature rel. error | 23.8 | 15 | ✘ | ✘ | `benchmarks/csf_consistent/validation_report.json` |
| `density_diffusion` | plane density wave decay | max relative error of the decay rate vs the analytic gamma = D k^2 | 1.021 | 3 | ✔ | ✔ | `benchmarks/density_diffusion_decay/decay_rate.json` |
| `density_diffusion` | plane density wave decay (finite run) | final amplitude vs the analytic exp(-D k^2 t) after 100 steps, artefact gate 3.5 % | 0.482 | 3.5 | ✔ | ✔ | `benchmarks/density_diffusion_decay/decay_rate.json` |
| `isph_projection` | CASE-ISPH-CHANNEL (GAP-SPH-ISPH-PROJECTION ladder) | centre-velocity error vs the analytic Poiseuille parabola (H/dx=16, momentum-budget wall-shear record also banked) | 0.8666 | 3 | ✔ | ✔ | `benchmarks/isph_channel/results.json` |
| `compression_regularization` | XVAL-SPHINXSYS-COMPRESSION-DENSITY-REGULARIZATION | max interior-band hydrostatic pressure error vs analytic p = rho0 g (H - y), SPHinXsys reference-mean arm | 5.533 | 1 | ✔ | ✘ | `benchmarks/xval_sphinxsys_compression/records.json` |
| `particle_shifting` | CASE-PARTICLE-SHIFTING-DAMBREAK (B2, shifting on vs off) | dam-break front-position deviation from the unshifted published answer, relative to the 0.2 m column | 8.833 | 10 | ✔ | ✔ | `benchmarks/particle_shifting/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel | interior L2 velocity error vs u=U*y/H (finest rung) | 5.654 | 5 | ✔ | ✘ | `benchmarks/couette/out/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel | wall-shear error at the moving wall, \|1 - tau_SPH/tau_exact\| (the no-slip quantity itself; artefact gate 5 %) | 32.65 | 5 | ✘ | ✘ | `benchmarks/couette/out/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel | cross-channel mass-flux error \|1 - Q_SPH/Q_exact\| (artefact gate 5 %) | 7.159 | 5 | ✔ | ✘ | `benchmarks/couette/out/results.json` |
| `isph_projection` | CASE-ISPH-CHANNEL (GAP-SPH-ISPH-PROJECTION ladder) | momentum-budget wall shear vs rho g H/2 on the finest rung (artefact gate 3 %) | 0.04583 | 3 | ✔ | ✔ | `benchmarks/isph_channel/results.json` |
| `csf_surface_tension` | CASE-LAPLACE-SPH (consistent CSF) | cut-force \|F_SPH - sigma\| error of the consistent-CSF arm | 5.987 | 15 | ✔ | ✔ | `benchmarks/csf_consistent/validation_report.json` |
| `transport_velocity_correction` | XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION (dam break, dx=0.03 m) | mean relative error of the front position Z/a vs the Martin & Moyce (1952) digitised points, alpha=0.10 over all particles | 10.81 | 15 | ✔ | ✔ | `docs/xval_sphinx_tvc/data/grid_dx003.json` |
| `transport_velocity_correction` | XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION (dam break, dx=0.04 m) | same quantity on the coarser rung: records the alpha=0.10 arm is resolution-sensitive (kept in the registry, fails the owner standard) | 16.96 | 15 | ✘ | ✘ | `docs/xval_sphinx_tvc/data/grid_dx004.json` |

登记记录 17 条，覆盖 10 个方法；最坏误差 32.65%。缺失源文件：无。


### 3.2 路由矩阵（候选算例画像 → 选定方法）

| 算例画像 | 选定方法 | 状态 | 最坏实测误差 [%] | 附加项（modifier） |
|---|---|---|---|---|
| regular wave in a flume | wcsph_delta_sph | SELECTED | 2.968 | density_diffusion=SELECTED, compression_regularization=SELECTED |
| Sod shock tube | compressible_riemann | SELECTED | 0.7768 | - |
| capillary droplet (Laplace law) | csf_surface_tension | SELECTED | 5.987 | density_diffusion=SELECTED, compression_regularization=SELECTED |
| planar Couette channel | isph_projection | SELECTED | 0.8666 | - |
| Gresho vortex (smooth inviscid, periodic) | （无） | NONE | - | - |
| 1-D transient conduction | heat_conduction_operator | SELECTED | 0.0002181 | - |
| breaking wave over a slope | wcsph_delta_sph | SELECTED | 2.968 | density_diffusion=SELECTED, transport_velocity_correction=SELECTED, particle_shifting=SELECTED, compression_regularization=SELECTED |

7 个画像中 6 个给出证据背书的方案；与预期不一致者：无。


### 3.3 无证据方法（evidence gate 拦下，附原因）

| 方法 | 族 | 状态 | 文献出处 |
|---|---|---|---|

### 3.4 图

![evidence_registry](docs/figures/method_selection_m18056/evidence_registry.png)

*图 1 证据登记表：每个方法的实测误差（对数轴），虚线=owner 15% 精度标准，蓝色刻度=该算例自己的验收门。*

![routing_matrix](docs/figures/method_selection_m18056/routing_matrix.png)

*图 2 路由矩阵：算例画像 → 选定方法；绿底=证据背书（SELECTED），红底=无可选方法。*

