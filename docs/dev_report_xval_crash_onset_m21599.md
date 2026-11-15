# m21599 — method A 长时程崩溃：起始时刻分档工具 + 两条取工件通道缺陷取证

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（`feature_ledger` status=in_progress）
**任务类型**: 验证欠账 + 工具（把"长时程必崩"从经验法则变成可量化区间，并修掉取工件通道的行为缺陷）
**前序**: m17501 → m17906 → m20388 → m20569(honest-FAIL) → m20783(通道探针 + 崩溃定位) → m20968 → m21050(稳定性分级曲线)

## ① 案例介绍

### 物理背景

method A = WCSPH + δ-SPH（δ=0.1）+ 桨式造波 + 海绵层，是 `oss_solutions` id=408
（`CASE-WAVE-FLUME`，method-selection-2026-11-12）记录的做法。锚点要求用一阶活塞传递
波高作为解析基准，判定它在业主精度标准 **< 15 %** 下能否复现波高。判级需要一个跑满
`t_end = 5.5 s` 的记录（协议窗 4.0 / 5.0 s）。

### 几何与参数（协议 §4，由 `benchmarks/wave_flume/run.py::build_config` 程序化重建）

| 参数 | 值 |
|---|---|
| 水深 h | 0.25 m |
| 波高比 H/h | 0.16（H = 40 mm） |
| 周期 T | 1.0 s（波长 1.304 m） |
| 槽长 L | 3.192 m（= 13 格深度离散时 `depth/dx = 13`） |
| 声速 c₀ | 30 m/s |
| 运动粘度 ν | 1×10⁻⁶ m²/s |
| δ-SPH 系数 | 0.1 |
| 海绵层 | 强度 4.0 / 长度 0.978 m |
| settle / ramp | 0.8 s / 2 周期 |
| 波高计位置 | x = 0.996 / 1.159 / 1.322 / 1.485 m（首个波到达 0.764 s） |
| 预注册分辨率 | H/Δx = 3.20（20 格）与 2.08（13 格） |

## ② 怎么做的

1. **读账 + 取回在飞作业**：m21050 收尾时递交的两条作业在本单开工时已到终态
   （`cj3150` 剩余分级扫描、`cj3151` 决策单元 1 = 13 格 5.5 s），本单先 `fetch` 取工件。
2. **发现取工件通道的实际行为**（见 ⑤ 缺陷 F1/F2）：`cj3150` 的 stdout 明确打印
   `wrote /data/test002/jobs/cj3150/repo/runs/xval/ladder_stability_rest.json`，
   但 `fetch` 回来的 tar 里**没有** `runs/` 目录 → 记录丢失。两条作业都是 `ref=main`
   递交（未带 `stage_dir`）；m20783 用 `stage_dir` 递交的探针作业（`cj2929`/`cj2962`）
   写入文件则出现在 tar 内。故本单所有后续作业**一律带 `stage_dir`**。
3. **修掉"分档只能重跑整格"的限制**：`run_cell.py` 增加 `--simulate-time` 与
   `common_argv()`（唯一允许的偏离就是时程；几何/分辨率/δ-SPH/海绵/settle 与协议逐字
   一致，`COMMON` 不被就地修改）。
4. **新增分档工具** `benchmarks/xval_method_selection/crash_bisect.py`：对一列时程逐个
   **以子进程**调用 `run_cell.py --simulate-time t`，因此一次 SIGABRT 只带走一个子进程、
   分档继续；每个时程记录 rc、记录是否存在、存活记录的诊断量（步数/模拟时刻/逃逸粒子/
   max|v|/压缩率）与首波到达时刻，输出 `onset_bracket`（最长存活 / 最短致死）。
   无记录一律记 `record=False`，**不把崩溃写成结果**；目录里的旧记录在运行前删除。
5. **守护测试** `tests/test_crash_bisect.py`（12 条）：`common_argv` 只改时程且不改
   `COMMON`、预注册旋钮不漂移、`describe()` 无记录不产生数字、solver 端失败单列、
   **旧记录不得被当成存活**、分档区间计算、子进程命令行携带单元索引。
6. **递交两条带 `stage_dir` 的作业**取真实产物（见 ③ 作业状态）。

## ③ 计算结果（程序化注入，零手抄）

### 作业终态与取回的日志证据

| 作业 | 内容 | 节点 | 队列状态 | 作业日志 rc | 日志末行 | fetch tar 内 `runs/xval/` |
|---|---|---|---|---|---|---|
| `cj3150` | 分级扫描 d16/d20 + 两个机理探针（0.60 s） | n315 | succeeded | **1** | `wrote /data/test002/jobs/cj3150/repo/runs/xval/ladder_stability_rest.json` | **缺失** |
| `cj3151` | 决策单元 1（13 格，5.5 s） | n330 | succeeded | **255** | `free(): corrupted unsorted chunks` | **缺失** |
| `cj3559` | 本单分档（13 格，0.6→5.5 s，6 档，`stage_dir`） | 调度中 | queued（收尾时） | — | — | **not_run** |
| `cj3560` | 分级扫描 d16/d20 + 两探针重跑（`stage_dir`，补工件） | 调度中 | queued（收尾时） | — | — | **not_run** |

`cj3150` 日志中解析出的分级数字（**日志证据，工件未取回**，仅登记不冒充归档）：

| label | H/Δx | 判定 | max&#124;v&#124;/c₀ | 压缩率 |
|---|---|---|---|---|
| `ladder_d16` | 2.56 | survived_stable | 1.0182e-03 | 0.3031 % |
| `ladder_d20` | 3.20 | survived_stable | 8.3932e-04 | 0.3018 % |
| `mech_d20_xsph` | 3.20 | **no_record** | — | — |
| `mech_d20_comp` | 3.20 | survived_stable | 8.1311e-04 | 0.2989 % |

计数：`points=4, blew_up=0, survived_stable=3, survived_unstable=0, rejected=0, no_record=1`。

### 存活证据图（把"长时程必崩"变成区间）

`docs/figures/xval_crash_onset_m21599/survival_evidence_map.png`：横轴为时程（对数），
纵轴为深度离散格数，绿圆=记录写出/跑完，红叉=堆损坏中止，黄色带=**尚未闭合的区间
(0.60 s, 5.5 s]**。所有点都来自上表与 m20783/m21050 的作业日志或本机探针，不含外推。

### 本单取证结论

- **崩溃是时程相关、分辨率无关**：13 格与 20 格、0.05 s 与 0.60 s 在节点上都能跑完
  （`cj2961`/`cj2962`/`cj3150`），而 5.5 s 的单元在 n315/n330 上**全部** SIGABRT
  （`cj2937`–`cj2940`、`cj3151`）。故"长跑必崩"不是分辨率问题，也不是导入/线程环境问题。
- **崩溃点仍在 (0.60 s, 5.5 s] 内**，`cj3559` 的分档（0.6/1.6/2.6/3.6/4.4/5.5 s，逐档
  子进程隔离）是闭合该区间的直接手段；收尾时仍在排队，本单按 **not_run** 记账。

## ④ 结果图

`docs/figures/xval_crash_onset_m21599/survival_evidence_map.png`（存活证据图，见上）。
**没有**波高—解析解比较曲线：本单没有任何一条 5.5 s 记录（`cj3151` 崩溃、`cj3559`
未取回），画波高比较曲线就是伪造证据。

## ⑤ 注意点（本单的实质发现）

- **F1（取工件通道，中危）**：`ref=main` 递交的作业，`fetch` 回的 tar 是**输入树**，
  作业写出的未跟踪文件（`runs/xval/*.json`）不在其中——`cj3150` 的 stdout 证明文件
  确实写成了。后果：作业"成功"但产物永久丢失，历史上有两单因此记 honest-FAIL。
  处置：涉记录类作业一律带 `stage_dir`（m20783 已证该模式产物可回）；建议在
  `run_compute submit` 侧对 `ref != stage_dir` 的作业强制提示。
- **F2（队列终态语义，高危·证据完整性）**：`run_compute status` 对 `cj3150`/`cj3151`
  都报 `succeeded, rc 0`，而作业日志里 rc 分别是 **1** 与 **255**（SIGABRT）。
  即**调度终态忽略了命令退出码**。后果：崩溃的作业在总线上读作成功，若下游只看
  `status` 就会把"没跑完"记成"跑完"。处置：本单全部结论以**作业日志 rc** 为准；
  建议新增总账条目（见 ⑧）。
- **crash_bisect 不是精度工具**：它只回答"能跑多久"，不产出波高；`describe()` 对无记录
  的档位不写任何数值字段（测试 `test_describe_missing_record_is_not_a_result` 钉住）。
- **子进程隔离的必要性**：崩溃是 `SIGABRT`（rc=-6/255），进程内 try/except 拦不住，
  只有子进程隔离才能让分档走完；这也是 `crash_bisect` 与 `ladder_stability_sweep`
  （进程内循环）的本质差别。
- **`--simulate-time` 是唯一允许的偏离**：`test_common_argv_keeps_the_pre_registered_cell_knobs`
  钉住 depth/height-ratio/δ-SPH/声速与 13 格分辨率不随分档漂移。

## ⑥ 误差判定（honest）

关键量：波高 vs 一阶活塞传递解析基准（`tensorsph.wave_flume_grade.wave_height_target`）。

| 关键量 | 与基准误差 | < 3 % | < 15 % |
|---|---|---|---|
| 波高（method A，13 / 20 格） | **无样本**（无一条 5.5 s 记录） | **不可评估** | **不可评估** |
| 崩溃起始时刻 | 区间 (0.60 s, 5.5 s]（`cj3559` 收尾时未回） | — | — |

**明确回答**：本单未证明 < 3 %，也未证明 < 15 %——是"不可评估"，不是通过、不是零误差。
本单的增量是**方法学与证据完整性**：(a) 分档工具 + 12 条守护测试，使崩溃区间可被逐档
闭合；(b) 两条通道缺陷取证（F1 产物丢失、F2 终态误报 succeeded），并把本单全部作业
改成 `stage_dir` 递交；(c) 用作业日志 rc 而非队列状态做判定。

## ⑦ 测试

```
PYTHONPATH=src python -m pytest tests/test_crash_bisect.py \
    tests/test_method_selection.py tests/test_xval_method_selection_matrix.py -q
# 43 passed
```

本机冒烟（真实子进程路径，非 mock）：`crash_bisect.py --cell 1 --times 0.02 0.04` →
两档 `survived rc=0 record=True`，写在 `/tmp/bisect_smoke.json`。

## ⑧ 给下次自找活的建议

1. **先取 `cj3559`/`cj3560`**：`cj3559` 的 `onset_bracket` 直接给出崩溃起始区间；
   拿到后把区间再二分 2 档（例如 t=2.6/3.1）即可定位到 0.5 s 内，再用
   `--progress-every`（尚未实现，建议加在 `run_case` 的步循环里，日志末行即崩溃步）
   把区间收敛到步级。
2. **建议新总账条目** `CASE-WAVE-FLUME-CLUSTER-HEAP-CRASH`（域 TensorSPH；验收：
   n315/n330 上 `run_cell.py 1 --simulate-time 5.5` rc=0 且 `runs/xval/cell_1.json`
   经 `grade_matrix.py` 判级 `present` 由 2/5 升至 5/5）。
3. **建议平台侧条目** `PLATFORM-COMPUTE-FETCH-INPUT-TREE`（域 TensorPlatform；验收：
   `ref=main` 递交的作业写出的未跟踪文件出现在 `fetch` tar 内，或 `submit` 对
   `ref != stage_dir` 给出显式告警）与 `PLATFORM-COMPUTE-TERMINAL-RC`
   （验收：`status` 报出的 rc 与作业日志 rc 一致；不一致时终态为 `failed`）。
   这两条是**共性机制**，按控仓令不由本席动 TensorPlatform。
4. 崩溃若定位到某一步的邻近表重建/逃逸粒子路径，优先在该路径上加不变量断言
   （而非改物理），再重跑 5.5 s 单元——之后才谈 15 %。