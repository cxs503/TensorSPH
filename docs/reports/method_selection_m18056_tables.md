### 3.1 证据登记表（程序化抽取，每行带源文件与 sha256）

| 方法 | 算例 | 度量 | 实测误差 [%] | 本算例门 [%] | <15% | 过本门 | 来源 |
|---|---|---|---|---|---|---|---|
| `sun_delta_plus` | ELLIPTICAL-DROP | maximum absolute relative axis-stretch error at t=0.4, R/dx=8,12,16 | 0.6597 | 1 | ✔ | ✔ | `benchmarks/delta_plus/ellipse_results.json` |
| `sun_delta_plus` | SPHERIC-19-ELASTIC-PLATE | largest marker RMSE divided by its measured peak at 5 mm (displacement metric only, density-regime check failed) | 13.87 | 15 | ✔ | ✔ | `benchmarks/delta_plus/spheric19_results.json` |
| `wcsph_delta_sph` | CASE-WAVE-FLUME | max gauge wave-height error vs piston-transfer target (H/dx=3.2, delta=0.1) | 2.968 | 10 | ✔ | ✔ | `docs/reports/wave_flume_numbers.json` |
| `compressible_riemann` | CASE-SPH-SOD-SHOCK-COMPRESSIBLE | max abs relative error over the gated shock quantities (rho*, p*, u*, wave positions) | 0.7768 | 5 | ✔ | ✔ | `benchmarks/case_sod_shock_compressible/result.json` |
| `compressible_riemann` | CASE-SPH-STRONG-SHOCK-CONTACT (1000:1 driver, 10.3x contact jump) | max abs relative error over the gated star-plateau values (rho*_L, rho*_R, p*, u*) and the contact/shock positions | 0.4142 | 5 | ✔ | ✔ | `benchmarks/case_strong_shock_contact/result.json` |
| `heat_conduction_operator` | CASE-SPH-CONDUCTION-1D | max normalised error vs the erfc similarity solution | 0.0002181 | 3 | ✔ | ✔ | `benchmarks/case_conduction_1d/result.json` |
| `heat_conduction_operator` | CASE-SPH-CONDUCTION-2D (Fourier mode decay) | max normalised field error vs exp(-alpha (kx^2+ky^2) t) cos(kx x) cos(ky y) on a periodic square, n=128, RK4 time stepping, 2-D k->0 lattice calibration (gate 3 %) | 0.1504 | 3 | ✔ | ✔ | `benchmarks/case_conduction_2d/result.json` |
| `heat_conduction_operator` | CASE-SPH-CONDUCTION-2D (1-D calibration lifted to 2-D) | same family and resolution, but the 1-D lattice factor (1.0218) applied in 2-D: ~870 % -- O(dx) dimensional mismatch | 834.6 | 3 | ✘ | ✘ | `benchmarks/case_conduction_2d/result.json` |
| `csf_surface_tension` | CASE-LAPLACE-SPH | Laplace pressure jump rel. error (delta_p vs 2*sigma/R) | 5.599 | 15 | ✔ | ✔ | `benchmarks/laplace_law/validation_report.json` |
| `csf_surface_tension` | CASE-LAPLACE-SPH (consistent CSF) | delta_s-weighted mean curvature rel. error | 23.8 | 15 | ✘ | ✘ | `benchmarks/csf_consistent/validation_report.json` |
| `density_diffusion` | plane density wave decay | max relative error of the decay rate vs the analytic gamma = D k^2 | 1.021 | 3 | ✔ | ✔ | `benchmarks/density_diffusion_decay/decay_rate.json` |
| `density_diffusion` | plane density wave decay (finite run) | final amplitude vs the analytic exp(-D k^2 t) after 100 steps, artefact gate 3.5 % | 0.482 | 3.5 | ✔ | ✔ | `benchmarks/density_diffusion_decay/decay_rate.json` |
| `isph_projection` | CASE-ISPH-CHANNEL (GAP-SPH-ISPH-PROJECTION ladder) | centre-velocity error vs the analytic Poiseuille parabola (H/dx=16, momentum-budget wall-shear record also banked) | 0.8666 | 3 | ✔ | ✔ | `benchmarks/isph_channel/results.json` |
| `compression_regularization` | XVAL-SPHINXSYS-COMPRESSION-DENSITY-REGULARIZATION | max interior-band hydrostatic pressure error vs analytic p = rho0 g (H - y), SPHinXsys reference-mean arm | 5.533 | 1 | ✔ | ✘ | `benchmarks/xval_sphinxsys_compression/records.json` |
| `particle_shifting` | CASE-PARTICLE-SHIFTING-DAMBREAK (B2, shifting on vs off) | dam-break front-position deviation from the unshifted published answer, relative to the 0.2 m column | 8.833 | 10 | ✔ | ✔ | `benchmarks/particle_shifting/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel | interior L2 velocity error vs u=U*y/H (finest rung) | 5.654 | 5 | ✔ | ✘ | `benchmarks/couette/out/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel (design-time run) | interior L2 velocity error vs u=U*y/H (finest rung, t_end = 1.2 H^2/nu per DESIGN.md) | 2.933 | 5 | ✔ | ✔ | `benchmarks/couette/out_design/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel | wall-shear error at the moving wall, \|1 - tau_SPH/tau_exact\| (the no-slip quantity itself; artefact gate 5 %) | 32.65 | 5 | ✘ | ✘ | `benchmarks/couette/out/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel (design-time run) | wall-shear error at the moving wall, \|1 - tau_SPH/tau_exact\|, read at t_end = 1.2 H^2/nu per DESIGN.md (finest rung) | 0.4781 | 5 | ✔ | ✔ | `benchmarks/couette/out_design/results.json` |
| `wcsph_wall_no_slip` | CASE-COUETTE-VISCOUS-TIME (steady-state envelope, H/dx=8) | wall-shear error \|1 - tau_SPH/tau_exact\| at the steady end of the viscous-time sweep (longest run whose steady_drift passes the 1 % gate), H/dx=8 | 9.55 | 5 | ✔ | ✘ | `benchmarks/couette/out_design/viscous_time_convergence.json` |
| `wcsph_wall_no_slip` | CASE-COUETTE-VISCOUS-TIME (reading mobility, H/dx=8) | mobility of the wall-shear reading across the admissible viscous-time sweep (max - min, percentage points), H/dx=8: how much the accuracy claim would move if the run were stopped earlier | 18.26 | 5 | ✘ | ✘ | `benchmarks/couette/out_design/viscous_time_convergence.json` |
| `wcsph_wall_no_slip` | CASE-COUETTE-VISCOUS-TIME (steady-state envelope, H/dx=16) | wall-shear error \|1 - tau_SPH/tau_exact\| at the steady end of the viscous-time sweep (longest run whose steady_drift passes the 1 % gate), H/dx=16 | 1.811 | 5 | ✔ | ✔ | `benchmarks/couette/out_design/viscous_time_convergence.json` |
| `wcsph_wall_no_slip` | CASE-COUETTE-VISCOUS-TIME (reading mobility, H/dx=16) | mobility of the wall-shear reading across the admissible viscous-time sweep (max - min, percentage points), H/dx=16: how much the accuracy claim would move if the run were stopped earlier | 0 | 5 | ✔ | ✔ | `benchmarks/couette/out_design/viscous_time_convergence.json` |
| `wcsph_wall_no_slip` | CASE-COUETTE-VISCOUS-TIME (moving wall, H/dx=8) | wall-shear error at the MOVING wall, \|1 - tau_SPH/tau_exact\|, at the steady end of the viscous-time sweep, H/dx=8 (the bottom-wall reading alone does not evidence the whole no-slip condition) | 14.2 | 5 | ✔ | ✘ | `benchmarks/couette/out_design/viscous_time_convergence.json` |
| `wcsph_wall_no_slip` | CASE-COUETTE-VISCOUS-TIME (moving wall, H/dx=16) | wall-shear error at the MOVING wall, \|1 - tau_SPH/tau_exact\|, at the steady end of the viscous-time sweep, H/dx=16 (the bottom-wall reading alone does not evidence the whole no-slip condition) | 17.08 | 5 | ✘ | ✘ | `benchmarks/couette/out_design/viscous_time_convergence.json` |
| `wcsph_wall_no_slip` | plane Couette channel | cross-channel mass-flux error \|1 - Q_SPH/Q_exact\| (artefact gate 5 %) | 7.159 | 5 | ✔ | ✘ | `benchmarks/couette/out/results.json` |
| `wcsph_wall_no_slip` | plane Couette channel (design-time run) | cross-channel mass-flux error \|1 - Q_SPH/Q_exact\| at t_end = 1.2 H^2/nu per DESIGN.md (finest rung) | 4.415 | 5 | ✔ | ✔ | `benchmarks/couette/out_design/results.json` |
| `isph_projection` | CASE-ISPH-CHANNEL (GAP-SPH-ISPH-PROJECTION ladder) | momentum-budget wall shear vs rho g H/2 on the finest rung (artefact gate 3 %) | 0.04583 | 3 | ✔ | ✔ | `benchmarks/isph_channel/results.json` |
| `csf_surface_tension` | CASE-LAPLACE-SPH (consistent CSF) | cut-force \|F_SPH - sigma\| error of the consistent-CSF arm | 5.987 | 15 | ✔ | ✔ | `benchmarks/csf_consistent/validation_report.json` |
| `csf_surface_tension` | CASE-LAPLACE-SPH (quasi-static, time-averaged) | bulk pressure vs vacuum over the 4-6 t_sigma window, worst rung of dx = 3/2/1.5 mm (analytic value sigma/R) | 45.69 | 15 | ✘ | ✘ | `benchmarks/csf_consistent/relax_quasistatic.json` |
| `transport_velocity_correction` | XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION (dam break, dx=0.03 m) | mean relative error of the front position Z/a vs the Martin & Moyce (1952) digitised points, alpha=0.10 over all particles | 10.81 | 15 | ✔ | ✔ | `docs/xval_sphinx_tvc/data/grid_dx003.json` |
| `transport_velocity_correction` | XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION (dam break, dx=0.04 m) | same quantity on the coarser rung: records the alpha=0.10 arm is resolution-sensitive (kept in the registry, fails the owner standard) | 16.96 | 15 | ✘ | ✘ | `docs/xval_sphinx_tvc/data/grid_dx004.json` |

登记记录 32 条，覆盖 11 个方法；最坏误差 834.6%。缺失源文件：无。


### 3.2 路由矩阵（候选算例画像 → 选定方法）

| 算例画像 | 选定方法 | 状态 | 宣称上界 [%] | 精度上界 [%]（整份推荐） | <15% | 附加项（modifier） |
|---|---|---|---|---|---|---|
| regular wave in a flume | wcsph_delta_sph | SELECTED | 2.968 | 5.533 | ✔ | density_diffusion=SELECTED, compression_regularization=SELECTED |
| Sod shock tube | compressible_riemann | SELECTED | 0.7768 | 0.7768 | ✔ | - |
| capillary droplet (Laplace law) | csf_surface_tension | SELECTED | 5.987 | 45.69 | ✘ | density_diffusion=SELECTED, compression_regularization=SELECTED |
| planar Couette channel | isph_projection | SELECTED | 0.8666 | 0.8666 | ✔ | - |
| Gresho vortex (smooth inviscid, periodic) | （无） | NONE | - | - | ✘ | - |
| 1-D transient conduction | heat_conduction_operator | SELECTED | 0.1504 | 0.1504 | ✔ | - |
| breaking wave over a slope | wcsph_delta_sph | SELECTED | 2.968 | 16.96 | ✘ | density_diffusion=SELECTED, transport_velocity_correction=SELECTED, particle_shifting=SELECTED, compression_regularization=SELECTED |

7 个画像中 6 个给出证据背书的方案；与预期不一致者：无。


### 3.3 无证据方法（evidence gate 拦下，附原因）

| 方法 | 族 | 状态 | 文献出处 |
|---|---|---|---|

### 3.5 宣称上界 vs 真实精度上界（逐方法）

| 方法 | 宣称上界 [%]（仅过 15% 的记录） | 真实精度上界 [%]（全部 live accuracy） | 差额 [pp] | 条件数记录 [%]（不计入） | live accuracy 记录数 | <15% |
|---|---|---|---|---|---|---|
| `compressible_riemann` | 0.7768 | 0.7768 | +0 | - | 2 | ✔ |
| `compression_regularization` | 5.533 | 5.533 | +0 | - | 1 | ✔ |
| `csf_surface_tension` | 5.987 | 45.69 | +39.71 | - | 4 | ✘ |
| `density_diffusion` | 1.021 | 1.021 | +0 | - | 2 | ✔ |
| `heat_conduction_operator` | 0.1504 | 0.1504 | +0 | - | 2 | ✔ |
| `isph_projection` | 0.8666 | 0.8666 | +0 | - | 2 | ✔ |
| `particle_shifting` | 8.833 | 8.833 | +0 | - | 1 | ✔ |
| `sun_delta_plus` | 13.87 | 13.87 | +0 | - | 2 | ✔ |
| `transport_velocity_correction` | 10.81 | 16.96 | +6.152 | - | 2 | ✘ |
| `wcsph_delta_sph` | 2.968 | 2.968 | +0 | - | 1 | ✔ |
| `wcsph_wall_no_slip` | 14.2 | 17.08 | +2.873 | 18.26 | 7 | ✘ |

11 个有证据的方法中，3 个的宣称上界**低于**自己最差的活accuracy 记录（最大差额 39.71 pp）；3 个方法的真实精度上界超出 15% owner 标准：`csf_surface_tension`, `transport_velocity_correction`, `wcsph_wall_no_slip`。


### 3.4 图

![evidence_registry](docs/figures/method_selection_m18056/evidence_registry.png)

*图 1 证据登记表：每个方法的实测误差（对数轴），虚线=owner 15% 精度标准，蓝色刻度=该算例自己的验收门。*

![routing_matrix](docs/figures/method_selection_m18056/routing_matrix.png)

*图 2 路由矩阵：算例画像 → 选定方法；绿底=证据背书（SELECTED），红底=无可选方法。*

![claim_bounds](docs/figures/method_selection_m18056/claim_bounds.png)

*图 3 逐方法：圆点=宣称上界（只取过 15% 的记录），菱形=全部 live accuracy 记录的最坏值（绿=在 15% 内，红=超出），灰线=两者差额；虚线=owner 15% 标准。*

