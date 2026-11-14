# m20968 — method-selection 决策单元：发散实测 + 记录溯源清单 + JSON 日志回收修复

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（`feature_ledger` status=in_progress）
**任务类型**: 验证欠账（把 method-selection 的决策单元矩阵对解析基准复核）
**上一单**: m20783（honest-FAIL：4 单元 `rc=255`/`free(): corrupted unsorted chunks`，无产物）

## ① 案例介绍

波流水槽（regular wave，方法 A = WCSPH + δ-SPH + XSPH）按锚点预注册几何
（`docs/xval_method_selection_m17501/protocol.md` §4）：

| 参数 | 值 |
|---|---|
| 水深 h | 0.25 m |
| 波高比 H/h | 0.16 |
| 周期 T | 1.0 s |
| 槽长 L | 3.2 m |
| 声速 c₀ | 30 m/s（M ≈ v/c₀ 小） |
| 运动粘度 ν | 1e-6 m²/s |
| δ-SPH 系数 | 0.1（sponge 4.0、settle 0.8 s、ramp 2 周期） |
| 模拟时长 / 分析窗 | 5.5 s / 4.0 s、5.0 s |

判据沿用 owner 精度标准 **<15 %**（`OWNER_TOLERANCE_PCT = 15.0`），参考量为
一阶活塞传递波高（`wave_flume_grade.grade_regular_record`），并叠加稳定性闸门
（`max|v|/c₀`、最大密度压缩）。

四个决策单元（`benchmarks/xval_method_selection/run_cell.py`）：

| # | 标签 | 变化 |
|---|---|---|
| 0 | `D20_h025_A` | H/Δx = 3.20（Δx = 12.5 mm） |
| 1 | `D13_h025_A` | H/Δx = 2.08（Δx = 19.2 mm） |
| 2 | `D20_h025_xsph` | 传播段 XSPH 0.05（机理探针） |
| 3 | `D20_h025_comp` | δ-SPH 关、改用 compression regularization（机理探针） |

## ② 怎么做的

1. **本机成本标定**（只读短测，2/4 线程）：`depth-cells 13`、2158 粒子，
   240 步 11.9 s、2880 步 68.4 s → 约 42 步/s，据此确认整单元（≈2.6e4 步）
   ~10 min 量级，适合递交集群（不指节点，调度器派发）。
2. **递交**：4 个单元各自 8 核 / 7200 s，`stage_dir`=本工作树，
   命令 `PYTHONPATH=src python benchmarks/xval_method_selection/run_cell_dump.py <i> --out runs/xval/cell_<i>.json`
   （`run_cell_dump.py` 把记录以 base64 随日志回传，`fetch` 的 tar 为第二通道）。
   作业号：cell0=`cj3094`、cell1=`cj3095`、cell2=`cj3096`、cell3=`cj3097`。
3. **取证**：`fetch` 取 tar 与 `/root/tp-agent/cluster/jobs/cj30xx.log` 双通道比对；
   记录用 `benchmarks/xval_method_selection/rebuild_records.py` 从日志重建并校验 sha256。
4. **判级**：`benchmarks/xval_method_selection/grade_matrix.py` 重算矩阵
   → `docs/reports/xval_method_selection_m17501.json`（程序化，零手抄）。
5. **新增溯源清单**（本单新增工具）：`record_provenance.py` 把每个记录绑定到
   label / sha256 / 字节数 / 作业号 / 回收通道 → `docs/reports/xval_method_selection_records_manifest.json`。
6. **修复**：`rebuild_records.parse_log` 原先只吃纯文本日志；集群 job 日志是
   JSON（换行被转义为 `\n`），导致**记录明明存在却被判为"没写"**。
   新增 `normalize_log()` 反解一层 JSON 转义，坏摘要仍只报 `problems` 不入库。

## ③ 计算结果（程序化输出）

矩阵（`docs/reports/xval_method_selection_m17501.json`）：

| 标签 | 记录 | 结论 | stability | 15 % 判据 |
|---|---|---|---|---|
| D20_h025_A | `runs/xval/cell_0.json`（sha256 `37c472e0190e…`，168 B） | **FAILED**：`FloatingPointError: non-finite acceleration at step 3890`（t ≈ 0.53 s，仍处 ramp 段） | 发散 | **不可评估**（无波高可测） |
| D13_h025_A | 缺失 | 作业 `cj3095` `rc=255` + `free(): corrupted unsorted chunks`，未写出记录 | — | 不可评估 |
| D20_h025_xsph | 缺失 | `cj3096` 同上 | — | 不可评估 |
| D20_h025_comp | 缺失 | `cj3097` 同上 | — | 不可评估 |
| D20_h020_archive | 归档 | `ungradeable`（可用探头 < 2） | 不稳定 | 不可评估 |

计数：`present=2/5`、`within 15% = 0`。

## ④ 结果图

本单的实测结论是"发散/缺失"，没有任何可绘制到 15 % 比较曲线上的合格波高序列：
`grade_matrix.py` 的 `measurable()` 明确禁止在缺失或发散记录上作图（图会变成
伪造证据），故 **本单不产图**，理由记录在此。溯源清单与矩阵 JSON 代替图作为证据指针。

## ⑤ 注意点

- **环境缺陷复现**：`cj3095/96/97` 在 **n330 与 n315 两个节点**都在打印首行后
  `free(): corrupted unsorted chunks`（`rc=255`）；同一命令在本机（4 线程、2880 步）
  正常。故崩溃属节点侧 glibc/torch 执行环境，与本算例参数无关——这与 m20783 结论一致，
  且**两个节点都能复现**，排除了单节点偶发。
- **cell 0 是真正的物理/数值结果，不是崩溃**：它跑到 step 3890 因非有限加速度自行
  终止并写出 `failed` 记录（`rc=0`，日志 base64 通道可取回），说明"节点崩溃"与
  "算例发散"两种失败在本单被**分开取证**，这正是 m20783 缺的那一步。
- m20783 遗留疑问"fetch 不返回产物"已被排除：记录缺失是因为进程死在写文件之前；
  `run_cell_dump.py` 的日志通道证明了这一点（三份日志只有首行，没有 BEGIN 标记）。
- 归档记录 `wf_D20_unstable.json` 与本单 cell 0 的发散互相印证：方法 A 在 H/Δx = 3.2
  的波槽上不稳，历史记录不是偶然。

## ⑥ 误差判定

关键量（探头波高 vs 活塞传递解析幅值）：

| 单元 | 与基准误差 | 是否 < 3 % | 是否 < 15 % |
|---|---|---|---|
| D20_h025_A | 无（step 3890 发散，分析窗内无样本） | **否（不可评估，诚实 FAIL）** | 不可评估 |
| D13_h025_A / xsph / comp | 无记录（节点崩溃） | 不可评估 | 不可评估 |

**明确回答**：本单没有任何单元给出合格波高，因此 `<3 %`、`<15 %` 两项均**未被证明**。
本单的实质增量是：(a) 把矩阵第 0 行从"记录缺失"升级为"实测发散"（真实结果），
(b) 证明其余三行的阻塞原因是可复现的节点崩溃而非算例设置，(c) 修掉会让这种崩溃
被误读为"没写"的日志解析缺陷。这不是通过，也不是零误差。

## ⑦ 测试

```
PYTHONPATH=src python -m pytest tests/test_rebuild_records_jsonlog.py \
    tests/test_method_selection_cell_provenance.py -q   # 14 passed
PYTHONPATH=src python -m pytest tests/test_method_selection_record_dump.py -q  # 含 6 条，绿
```

新增守护：
- `tests/test_rebuild_records_jsonlog.py`（3 条）：纯文本日志、JSON 转义日志都能取回
  记录；被篡改的 base64 只报 `problems`，不入库。
- `tests/test_method_selection_cell_provenance.py`（11 条）：清单摘要随文件变化、
  缺失记录必须显式标记、声明"已回收"而文件不存在必须报错（m20569 失效模式）、
  作业号必须对得上真实单元目录。

## ⑧ 给下次自找活的建议

1. **节点崩溃是唯一阻塞**：同一 4 单元命令在 n315/n330 均 `rc=255`，本机 rc=0。
   建议总账新增 `CASE-WAVE-FLUME-CLUSTER-HEAP-CRASH`（域 TensorSPH；验收：同一
   `run_cell_dump.py` 命令在两节点 `rc=0` 且日志出现 `>>>BEGIN_RECORD_B64`，
   随后矩阵 `present` 由 2/5 升至 5/5）。
2. 排查方向：`torch` 与节点 glibc 的 `OMP/MKL` 线程绑定（本机与节点唯一差异是核数
   与线程上限）；可先跑 `--threads 1`、`OMP_NUM_THREADS=1` 的同一单元做二分。
3. 记录一律用 `run_cell_dump.py`（日志）+ `fetch` tar 双通道，并立即
   `record_provenance.py` 固化清单——崩溃时才分得清"没写"与"没取回"。