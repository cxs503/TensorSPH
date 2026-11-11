# XVAL-SPH-DAMBREAK 第1轮 — honest FAIL（含已建成机制）

## 1. 案例设定 + ref 出处
2-D 溃坝：a=0.2 m，H=0.4 m（H/a=2），水箱 2.0×0.75 m，干床，初始水柱贴左壁。
参考：Martin & Moyce (1952) Phil. Trans. R. Soc. A 244:312-324（本仓已存两份数字化表：
`benchmarks/dam_break/martin_moyce_1952.json`、`..._crosscheck.json`，±0.05 in Z/a）。

## 2. 三方方法与版本
- 我方 TensorSPH（worktree commit base 184ff8d）：WCSPH、Tait γ=7、c0=20 m/s、CubicSpline、h=1.3dx、
  求和密度、Morris 粘性、Adami(2012) eq.11 壁面压力外推、3 层静态壁粒子。
- PySPH：**已装成功** `PySPH 1.0b3.dev0`，git commit `1cd3dea9cfaac7d9f1bda2a2491ce4b42a31dbe5`
  （venv `/tmp/xv2`，系统 numpy 2.4.6/loongarch64；构建链：cyarray 1.2 + compyle(Cython≥3) +
  meson-python/ninja；PyPI 1.0b1/1.0b2 sdist 在 py3.11+Cython3 下不可编译 —— 见 friction）。
  方案选型 `AdamiHuAdamsScheme`（其 `SolidWallPressureBC` 即 Adami eq.11，与我方壁面 BC 同族），
  kernel CubicSpline(dim=2)，alpha=0，几何/壁粒子由本仓 spec 生成后直接喂入 PySPH。
- DualSPHysics：**不可装**（无 CUDA/GPU：`nvcc`/`nvidia-smi` 均不存在，loongarch64 无 CUDA 发行；
  PyPI 无 `dualsphysics` 包（404），源码 clone 后需 GPU 编译）。

## 3. case_spec（一份三用，已落盘）
`artifacts/xval_dambreak/case_spec.json`（`python scripts/xval_dambreak/xval.py spec` 生成）：
dx、h/dx=1.3、c0、γ、ν=1e-6、dt=0.5×Courant、壁层数、采样口径（front=(max x−a)/a，
height=max y/H，τ 网格 0…2.8）逐项显式；两工具都被要求 echo 该 spec 到各自 result 文件以便机器比对。

## 4. 横向对照矩阵：**未取得（WITHHELD）**
本会话未产出任何可用数值结果，故无对照表。原因见 §7。**不得**以任何估计值填充。

## 5. 判据逐条
- 涌浪前沿 x(t) 工具间 <10%：**NOT RUN**（无 PySPH 轨迹）
- 坍塌高度 h(t) 工具间 <15%：**NOT RUN**
- vs Martin & Moyce 1952 <15%：**NOT RUN**（我方运行未在会话预算内收敛完成）

## 6. 收敛性/网格敏感性
未取得（dx=0.01/0.005/0.02 三档运行均超过本会话单次命令预算，见 §7）。

## 7. 失败模式与工件（fail_stage / failure_reason）
fail_stage = `run_arms`
- **我方运行过慢**：dx=0.01、3000 步、~1900 粒子的 `run_case` 在 3 个并发进程竞争时 >10 min 未返回；
  dx=0.02（~250 粒子）在 75 s 内亦未返回 → 疑与本机 4 个 xval 进程 + torch 线程超额订阅有关，未定位。
  `tools` 侧无 traceback，属"未完成"而非崩溃；**如实记为 NOT RUN，不记通过**。
- **PySPH 运行崩溃**：`RuntimeError: ERROR: LinkedListNNPS requires too many cells (-569796944)`
  （`pysph/base/linked_list_nnps.pyx:340` `_count_occupied_cells`），在已按 `setup_properties` 后
  重新钉住 `h`/`rho` 的情况下仍复现 → 疑为 2-D（z 向零厚度）在 Linked-list NNPS 下 z 向格数溢出，
  需换 NNPS（CompressedOctree/CellLinkedList 分块）或给 z 向厚度；未修复，留作下轮。
- DualSPHysics：见 §2（无 GPU），friction 记录见 `artifacts/xval_dambreak/friction.json`。

## 8. bug 判读与生态摩擦
- 已建成的**代码级**判读机制（本轮未能填数）：`scripts/xval_dambreak/xval.py compare` 产出
  `xval_table.json`（tool×metric×got×vs_ref_err×cross_diff）与 `bug_suspects.json`
  （含我方案例默认 `Config.viscosity=0.01 m²/s`（Re≈80，为水的 1e4 倍）vs 水值 1e-6 的对照臂，
  locate 到 `src/tensorsph/dam_break.py`/`droplet_impact.py` —— 该条**已定位未验证**，需跑数确认）。
- 生态摩擦（ecosystem_friction）：①PySPH 官方 PyPI sdist（1.0b1/b2）在 Python 3.11 + Cython 3 下
  无法编译（`UIntArray is not a type identifier` / cyarray 缺失），须走 GitHub main + Cython≥3 +
  先装 cyarray/compyle，且 compyle 又需 meson/ninja；②cyarray 走 meson 构建，PyPI 无 loongarch64 轮子；
  ③`pysplishsplash` 无 loongarch64 轮子且无 sdist（备选第二参照亦不可用）；④DualSPHysics 无 GPU 不可用。

## repro_cmd
```
python scripts/xval_dambreak/xval.py spec
python scripts/xval_dambreak/xval.py ours   --dx 0.01 --nu 1e-6
/tmp/xv2/bin/python scripts/xval_dambreak/xval.py pysph --dx 0.01 --nu 1e-6
python scripts/xval_dambreak/xval.py compare --dx 0.01
python scripts/xval_dambreak/xval.py figures --dx 0.01
```
