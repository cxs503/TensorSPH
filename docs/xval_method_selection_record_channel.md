# 集群作业产物回收通道（method-selection 单元 0–3 的欠账根因与修复）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`
**关联**: `docs/xval_method_selection_m20569_dispatch.md`（上一单 honest-FAIL）

## 1. 症状

m20569 把 `benchmarks/xval_method_selection/run_cell.py` 的 4 个决策单元递交集群，
`cj2790` / `cj2793` 返回 `rc=0`，但 `run_compute fetch` 取回的只是仓库检出，
不含节点新写的 `runs/xval/cell_{i}.json`，于是矩阵 `present: false` 无法翻绿，
报告无法分级（`docs/reports/xval_method_selection_m17501.json` 停在
`counts.present = 1`、`counts.passed = 0`）。

## 2. 根因（本单用受控探针证伪了上一单的解释）

上一单的结论是「fetch 不返回运行产物」。这是**过度概括**。本单用一个不到 1 秒的
受控作业（`cj2929`，`cores=2`，`stage_dir=/root/tp-agent/work/m20783`）验证：

| 探针 | 预期 | 实测 |
|---|---|---|
| 节点读到暂存目录里的本地标记 `runs/xval/_plumbing_probe.txt` | 打印 `PROBE_FROM_STAGE: plumbing-test-…` | 命中（`stage_dir` 生效） |
| 节点写入新文件 `runs/xval/_job_wrote.txt` | fetch 的 tar 内应出现该文件 | **命中**：`fetch` 返回 `/root/tp-agent/cluster/fetch/cj2929_n330.tar.gz`（32.9 MB = 节点整棵暂存树），`tar tzf` 列出 `cj2929/repo/runs/xval/_job_wrote.txt` |

结论：**带 `stage_dir` 递交时，fetch 取回的是节点上那棵暂存树的现场快照，包含作业
新写的文件。** 上一单的失败来自递交参数（缺省 `ref=main` ⇒ 节点检出 GitHub main，
作业在检出目录里写 `runs/xval/…`，而那份检出不是本工作树，产物就与调用方失联），
不是 fetch 机制本身。附带条件：**必须等作业进入终态（succeeded/failed）再 fetch**，
`running` 期间取回的只是当时快照（`cj2929` 在 `running` 时 fetch 已能拿到
`_job_wrote.txt`，但记录类产物应在终态后取，以免拿到半截文件）。

## 3. 修复：把记录塞进日志（双通道）

即使产物通道将来再退化，记录也不该只依赖一个通道。本单新增两个小工具：

| 文件 | 作用 |
|---|---|
| `benchmarks/xval_method_selection/run_cell_dump.py` | 原样调用 `run_cell.main`，成功后把 `--out` 记录 base64 打印成 `>>>BEGIN_RECORD_B64 <label> <sha256> <bytes>` … `>>>END_RECORD_B64 <label>`，日志即证据 |
| `benchmarks/xval_method_selection/rebuild_records.py` | 从日志重建记录文件；**逐块校验 sha256 与字节数**，截断/篡改的块报 `problems` 且不入库（不接受"看起来像记录"的残片） |

守护测试 `tests/test_method_selection_record_dump.py`（6 条）钉住：字节级往返、单日志多块、
截断拒绝、篡改拒绝、`run_cell_dump` 的标签集合与 `grade_matrix.MATRIX` 的非 archive 行**完全一致**
（防止日志标签与判级行漂移）、重建写出文件名映射。

## 4. 适用范围 / 注意点

- 该通道对任何「单进程写一个 JSON 记录」的 SPH 单元都适用；记录越大，日志越大
  （base64 约 +33 %），本算例记录 ~1 MB 级，日志可承受。
- 不要用它传二进制大场（压力/速度全网格快照）——那种应该走 `stage_dir` 的 tar 通道。
- 校验和不匹配一律视为「无证据」，按 honest-FAIL 记，不许降级成"近似通过"。

## 5. 补充证据（同一锚点，第二次实测）：单元在节点上崩溃，产物根本不存在

本单按上一单的建议用 `stage_dir`（本地工作树直达节点）重跑 4 个决策单元
（`cj2937`/`cj2938` → n315，`cj2939`/`cj2940` → n330，8 核/单元）。四个作业都在
打印首行后被 **SIGABRT 级堆破坏**打断：

```
cell 0: D20_h025_A -> runs/xval/cell_0.json
free(): corrupted unsorted chunks
rc = 255
```

四份 fetch tar（`cj2937_n315.tar.gz` 等，各 ~33 MB，含节点整棵暂存树）里
`runs/xval/` **只有暂存时就存在的 `_plumbing_probe.txt`，没有任何 `cell_*.json`**：
记录没有被写出来，而不是"写出来没取回来"。因此上一单"fetch 不返回运行产物"的经验
法则要再细一层：

| 情形 | fetch 是否带回作业新写的文件 |
|---|---|
| `stage_dir` + 作业**正常退出**（`cj2929` 探针写 `runs/xval/_job_wrote.txt`） | **是**（tar 内命中） |
| `stage_dir` + 作业**崩溃**（`cj2937`–`cj2940`） | 否——文件从未落盘 |

对照实验：同一条 `benchmarks/wave_flume/run.py` 短跑命令（`--simulate-time 0.05`,
`--threads 4`, 370 步、5120 粒子）在**本机 rc=0**（`runs/xval/tiny_local.json`，
max|v| 0.112 m/s、压缩 4.84e-3），所以崩溃不是算例参数或造波边界的问题，而是节点侧
执行环境（原生库/线程绑定）问题。诊断作业 `cj2961`（`import torch` / `import tensorsph`
/ 短跑分步）与 `cj2962`（`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE`
后短跑）在本单收尾时仍在排队（`queued`，位次 16/17），结果记为 **not_run**，不冒充结论。
