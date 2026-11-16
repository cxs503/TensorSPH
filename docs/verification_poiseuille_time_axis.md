# CASE-POISEUILLE-SPH — 运行时长轴：先否证「dt 致因」，再把认证 Ma 下的长跑点立成可判实验

> 锚点：`CASE-POISEUILLE-SPH`（算例池·泊肃叶流；`feature_ledger` 容差 < 5 %，参考=平面泊肃叶解析解）。
> 本文件由 `benchmarks/poiseuille_mirror/make_time_axis_doc.py` 从 `benchmarks/poiseuille_mirror/out/time_axis.json` 程序化生成，表中每个数字来自工件，无手抄。
> 研究脚本：`benchmarks/poiseuille_mirror/study_time_axis.py`；判定规则在跑之前写进脚本 docstring，跑后未调阈值。

## ① 问题：Ma 包络研究写在自己工件里的欠账

`study_ma_envelope.py` 用 7 个点判定「<5 % 证书是否带 Ma 包络」，结论 `MIXED_MA_TIME`，并在 `out/ma_envelope.json:verdict.open_item` 里留下两条**未被分离**的候选致因：

1. Ma 相关的粒子无序度增长；
2. 为压低 `c_s` 带来的 `dt` 加倍（P3/P4 的 `dt` = 3.25e-4，而 P0 = 1.63e-4）。

被点名的判别实验是 **`c_s = 20`、`t_end = 2.0`**（与 P4 同运行时长，但 Ma 与 dt 保持认证值）。同一份工件因此声明 `psi_index` 只是「事后相关指标，尚无非样本检验，不得当作稳定性判据」。

本研究分两步：**(I) 用已提交的 7 点数据否证候选致因 2**（无需新算例）；**(II) 把认证 Ma/dt 下的运行时长轴立成预声明、可分片执行、可机械判定的实验**。

## ② 方法与参数

平面通道 `0 <= y <= H`，两侧无滑移（mirror/ghost 壁面算子），`x` 周期，体积力 `g` 驱动；解析解 `u(y) = g (H-y) y/(2 nu)`、`u_max = g H²/(8ν)`、`tau_w = rho g H/2`（White, *Viscous Fluid Flow* 3rd ed. §3-2.2；Batchelor §4.2）。

| 量 | 值 | 说明 |
|---|---|---|
| `H` | 0.1 m | 通道高度 |
| `nu` | 0.01 m²/s | 运动粘度（未操纵） |
| `g` | 9.81 m/s² | 体积力（未操纵） |
| `c_s` | 20.0 m/s | 声速（**未操纵**：与 P0 相同，故 Ma 与 dt 均为认证值） |
| `H/dx` | 12 | 锚点 all-row 读数所需的最小分辨率 |
| `dx` | 0.00833 m | `H/(H/dx)` |
| 认证 `dt` | 1.625e-04 s | P0：6154 步 / 1 s |
| `u_max`（解析） | 1.22625 m/s | `g H²/(8ν)` |
| 判据带 | 5.0 % | 算例池既定值，未在事后调整 |
| `Psi` 换算 | 9.022 /s | 本工况退化为 `Psi = 9.02 t_end` |

### 2.1 预声明预测表（跑之前写定；`Psi` 用解析 `u_max`，与实测无关）

| 点 | `t_end` [s] | `Psi`（预声明） | 原位样本带判定 |
|---|---|---|---|
| T1_runlen_2s | 2 | 18.04 | 预测 PASS |
| T2_runlen_4s | 4 | 36.09 | 预测 FAIL |
| T3_runlen_8s | 8 | 72.18 | 预测 FAIL |

### 2.2 预声明判定规则（v1）

* **H_time**：认证 Ma/dt 下仅运行时长即破坏 5 % 带 → T1 FAIL。
* **H_psi**：边界由累积指标 `Psi` 决定（原位样本带 passing ≤ 20.3，failing ≥ 32.6）→ T1 PASS、T2 FAIL。
* **H_ma_dt**：需 Ma（或 dt）高于认证值才失效 → T1 PASS、T2 PASS。

判定：`T1 FAIL→H_TIME`；`T1 PASS,T2 FAIL→H_PSI`；`T1 PASS,T2 PASS(+T3)→H_MA_DT(有界/强)`；其余 → `INCOMPLETE/UNEXPLAINED`。阈值零调整。

## ③ 计算结果

### 3.1 已有 7 点数据已经能判的：`dt` 不是充分致因（否证候选 2）

P3 与 P4 **同为 `c_s = 10`、`dt` 相同**，只差运行时长；若「dt 加倍」是充分致因，两点必须同判。

| 点 | `t_end` [s] | dt [s] | `dt`/认证dt | `Psi` | Ma | Re | 剖面 RMS [%] | 壁剪 [%] | `u_max` err [%] | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| P3_cert_re_high_ma | 1 | 3.250e-04 | 2.00× | 18.05 | 0.123 | 12.26 | 1.251 | 0.367 | +0.010 | PASS |
| P4_cert_re_high_ma_long | 2 | 3.250e-04 | 2.00× | 32.93 | 0.126 | 11.71 | 4.572 | 11.559 | -4.481 | **FAIL** |
| P0_baseline (认证点) | 1 | 1.625e-04 | 1.00× | 9.03 | 0.061 | 12.26 | 1.234 | 0.376 | +0.017 | PASS |

结论（`evidence.dt_candidate = ELIMINATED_AS_SUFFICIENT_CAUSE`）：P3 and P4 share dt = 3.250e-04 s (2.00x the certified dt) and differ only in run length (1 s vs 2 s); P3 passes every gate and P4 fails profile RMS and wall shear. A doubled dt is therefore not sufficient to break the band: within this pair the discriminating variable is the run length (Psi 18.0 -> 32.9).

放大倍数：运行时长 2.00×、`Psi` 1.82×，而剖面 RMS 3.66×、壁剪 31.47×——误差对累积量的放大是超线性的，这正是「累积型」而非「阈值型」失效的指纹。

仍开着的部分：The same run length has never been run at the *certified* Ma and dt, so the failure cannot yet be attributed to accumulated time alone rather than to the higher Ma; that is exactly point T1 (t_end = 2.0 s, c_s = 20), which this module declares and grades.

### 3.2 运行时长轴的三个预声明点

**本文件写作时三个点均无工件**（`verdict.status = NOT_RUN`）。表内不填任何估算值；每个点按 `study_time_axis.py --points <id>` 单独执行，落一份分片 JSON（互相不共享写），再用 `--collect` 合并判定。预计单点成本（4 线程 1 核等效口径）：T1 ≈ 2 min、T2 ≈ 4 min、T3 ≈ 8 min。

图：`docs/figures/poiseuille_time_axis.png` —— (a) 已提交 7 点的 `Psi`–RMS 平面（实心）与三个预声明点的预测 `Psi` 位置（竖虚线，尚未执行）；(b) P3/P4 同 dt 对照（真实数字），运行时长是唯一差别。

## ④ 判定

**status = `NOT_RUN`**

> None of the declared run-length points has an artifact on disk, so this study makes no claim about the run-length axis. What the committed seven-point sweep already settles is recorded separately in existing_evidence(): the doubled-dt candidate is eliminated as a sufficient cause, and the certified-Ma long run (T1) remains the missing discriminator.

## ⑤ 注意点与局限

1. 本研究的**结论分两层**：3.1 的「dt 非充分致因」由已提交数据机械决定，可立即引用；3.2 的运行时长轴在点未执行前**不产生任何结论**（`NOT_RUN`），不得把预测当结果。
2. 候选 1（Ma 驱动的粒子无序度增长）**尚未被否证**：`dt` 被排除只说明「加倍 dt 不足以致失效」，不说明 Ma 无关。要分离二者，缺的正是 T1（同 t_end、不同 Ma/dt 对照走 P3。
3. `Psi` 仍是 7 点事后拟合出的相关量；即便 T1/T2 跑出 H_PSI，也只是**两点夹逼**，不是拟合出的定律，也不可外推到其它几何、核函数、时间积分格式。
4. 与 `study_ma_envelope.py` 同源同设置（同一 `run_point`/`evaluate`/`gates`），系统性偏差（Morris Laplacian 过扩散、壁面偏移 δ/dx ≈ 0.38）在两份研究里同向存在，本研究不消除它。
5. 执行环境限制（实测，记录在案）：本机单次 shell 调用上限 120 s，T1 在 4/6 线程下仍需 ≈115 s 以上，无法在本地取数；集群递交的 1 核作业排队 38 min 未派发（与 Ma 研究遭遇的单核饥饿一致），改以 4 核递交。**排队等待是本研究当前未取到 3.2 数字的唯一原因**，与物理无关。

## ⑥ 误差判定：关键量 vs 基准，逐项是否 <5 %

**无实测数字 → 不做误差判定**（honest：未执行 ≠ 通过）。3.1 的对照表是已提交工件的复核，其逐项误差见该表最后一列；本文件不替工件说话。

## ⑦ 复现

```bash
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_time_axis.py --evidence
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_time_axis.py \
    --points T1_runlen_2s T2_runlen_4s T3_runlen_8s --collect
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_time_axis_doc.py
pytest tests/test_case_poiseuille_time_axis.py -q
```
