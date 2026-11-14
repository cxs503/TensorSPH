# XVAL-SPH-METHOD-SELECTION-2026-11-12 — 证据回收通道 + 节点崩溃定位（m20783）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（`feature_ledger` status=in_progress）
**任务类型**: 验证欠账（把已合并的 method-selection 功能对解析基准复核）
**上一单**: m20569（honest-FAIL：4 单元 `rc=0` 但产物取不回）

## 1. 本单做了什么

1. **受控探针证伪上一单的经验法则**：用 <1 s 的作业 `cj2929`（`stage_dir`=本工作树）
   证明节点能读到暂存目录本地文件、且 fetch 的 tar **包含作业新写的文件**
   （`cj2929/repo/runs/xval/_job_wrote.txt`）。故"fetch 不返回运行产物"不成立。
2. **新增双通道回收工具**（日志通道 + 现有 tar 通道）：
   - `benchmarks/xval_method_selection/run_cell_dump.py`：原样跑一个单元，成功后把记录
     base64 打印为 `>>>BEGIN_RECORD_B64 <label> <sha256> <bytes>` … `>>>END_RECORD_B64`；
   - `benchmarks/xval_method_selection/rebuild_records.py`：从日志重建记录，逐块校验
     sha256 与字节数，截断/篡改只报 `problems`、不入库；
   - 守护测试 `tests/test_method_selection_record_dump.py`（6 条，全绿）。
3. **按上一单建议重跑 4 个决策单元**（`stage_dir`，8 核/单元）：`cj2937`/`cj2938`（n315）、
   `cj2939`/`cj2940`（n330）。
4. **定位到新的实质性缺陷**：四个单元全部在打印首行后 **`free(): corrupted unsorted
   chunks`、rc=255** 崩溃，节点暂存树内**没有** `runs/xval/cell_*.json`。
5. **对照实验**：同一 `wave_flume/run.py` 短跑（`--simulate-time 0.05`，370 步，5120 粒子）
   在**本机 rc=0**（`runs/xval/tiny_local.json`）。→ 崩溃与算例参数无关，属节点侧执行环境。

## 2. 作业表（程序化取自作业日志与 fetch tar 清单）

| 作业 | 单元 | 节点 | rc | 末尾输出 | fetch tar 内 `runs/xval/` |
|---|---|---|---|---|---|
| cj2929 | 通道探针（写 `_job_wrote.txt`） | n330 | — | 正常 | 含作业写入文件（**通道可用**） |
| cj2937 | 0 `D20_h025_A` | n315 | **255** | `free(): corrupted unsorted chunks` | 只有 `_plumbing_probe.txt` |
| cj2938 | 1 `D13_h025_A` | n315 | **255** | 同上 | 同上 |
| cj2939 | 2 `D20_h025_xsph` | n330 | **255** | 同上 | 同上 |
| cj2940 | 3 `D20_h025_comp` | n330 | **255** | 同上 | 同上 |
| cj2961/cj2962 | 崩溃诊断（分步 import / 线程环境变量） | — | 排队中 | — | not_run |

本机对照：`runs/xval/tiny_local.json`——`steps_taken=370`、`simulated_time=0.050 s`、
`particle_count=5120`、`max|v|=0.112 m/s`、`max_density_compression=4.84e-3`、rc=0。

## 3. 误差判定（honest FAIL）

**本轮没有可判级的波高记录，故 `<15%` 判据本单无法评估**——不是通过、不是零误差，
而是「无证据」。`docs/reports/xval_method_selection_m17501.json` 维持
`counts.present=1`、`counts.passed=0`（未被我改动，避免用无证据的数字污染矩阵）。
上一单遗留的路径问题已闭环（通道已验证可用 + 工具与测试落盘），但物理结论仍为
**WITHHELD**，原因已定位到节点崩溃而非算例设置（本机同命令 rc=0）。

## 4. 注意点

- 作业日志只保留 `output_tail`（约 200 字符），**不适合承载 base64 记录通道**；
  本次记录通道的真正价值是「节点崩溃时能立刻区分『没写出来』与『写了没取回』」。
- `fetch` 在 `running` 期间也可能返回快照，取记录类产物应等到终态。
- 节点崩溃点在本机无法复现，禁止把本机 rc=0 当作节点可用的证据。

## 5. 给下次自找活的建议

1. **先取 `cj2961`/`cj2962`**：若分步 import 就在 `import torch` 崩溃 → 节点 torch/原生库
   问题（升级/换 wheel 或限定 `OMP_NUM_THREADS`）；若短跑崩溃而在本机正常 → 线程绑定/亲和性
   问题，按节点 nproc 与 `--threads` 关系排查。
2. 建议新增总账条目：`CASE-WAVE-FLUME-CLUSTER-HEAP-CRASH`（域 TensorSPH；
   验收：同一 `run_cell.py` 命令在 n315/n330 上 rc=0 且产出可判级 `runs/xval/cell_*.json`；
   随后 `grade_matrix.py` 的 `counts.present` 由 1 升至 5）。
3. 崩溃修好后用 `run_cell_dump.py`（本单新增）递交，产物与日志双通道冗余。
