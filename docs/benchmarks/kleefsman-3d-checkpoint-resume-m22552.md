# 3D Kleefsman 算例：检查点/续跑（崩溃不再吞掉整条时程）

**Mission**: m22552 · **分支**: `agent/m22552` · **锚点**: `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`
**仓**: TensorSPH · **类型**: 缺陷修复（验证欠账的阻塞项）

---

## 1. 问题（为什么要做这件事）

算例池锚点的实质增量卡在**一条数据**上：二维模型是 100 % 阻塞（箱宽跨满槽宽），实验是
40.3 % 阻塞（0.403 m 箱 / 1.00 m 槽），主测站峰值 30.1 % 的偏差一直**被归因于阻塞率差异而从未被测过**
（`docs/benchmarks/kleefsman-convergence-t30-m21718.md`、`src/tensorsph/kleefsman_obstacle_3d.py` 模块头
都写明了这一点）。三维通道（`kleefsman_obstacle_3d.py` + `ladder.py` + `make_doc.py`）早已写好，
但 `benchmarks/kleefsman_obstacle_3d/results/ladder.json` 的现状是 **`status = NO_RUNS`**：

| 作业 | dx [m] | threads | 结局 | 死亡时刻 [s] | 死因 |
|---|---|---|---|---|---|
| cj3952 | 0.05 | 16 | ABORT | 1.492 | `free(): corrupted unsorted chunks` |
| cj3953 | 0.04 | 16 | ABORT | 1.825 | 同上 |
| cj3954 | 0.03 | 16 | ABORT | 1.000 | 同上 |
| cj3992 / cj3993 | 0.05 | 1 / 16 | 未启动 | — | 诊断壳 `faulthandler.register(SIGABRT)` 被 CPython 拒绝 |

即：**每一次三维生产 run 都在 1–1.8 s 模拟时间处被 glibc 堆中止杀掉**，而中止发生在 C 分配器里，
Python 层 `except BaseException` 永远看不到它，所以每次重试都从 `t = 0` 重新开始 —— 一条 10 分钟的
作业只换来零数据。锚点的关键测量因此 **WITHHELD**，不是物理算不出来，而是**没有任何一次跑完**。

## 2. 做了什么

不做一次性 hack，而是去掉这类长时程 SPH 作业的**结构性弱点**：状态可持久化、可续跑。

1. `src/tensorsph/kleefsman_obstacle_3d.py`
   * 新增 `checkpoint_fingerprint(cfg)` / `save_checkpoint(...)` / `load_checkpoint(...)`，
     写入走 `tmp` + `os.replace`（原子替换：被杀死在写入中途的进程只会留下上一份或新一份，不会留下半截文件）。
   * `run_case(...)` 新增 `checkpoint_path` / `checkpoint_every` / `resume` 三个参数；
     续跑时恢复的位置、速度、`dt`、已模拟时间、整条水位时程与全部运行诊断量（`max_speed`、
     `max_density_compression`、`escapes`、`clamps`），邻接表在续跑时重建（0.6 s，不值得存）。
   * **`t_end` 不进指纹**：它是停机判据不是物理参数 —— 崩溃总是发生在 `t_end` 之前，续跑必须能用
     更长的时域重入，否则这个功能毫无用处。
   * 跑完后删除状态文件：已完成的网格不允许被重试循环重放。
   * 指纹不匹配时抛 `ValueError` 并列出逐项差异（防止把两套网格混进一条时程）。
2. `benchmarks/kleefsman_obstacle_3d/run.py`
   * 新增 `--checkpoint` / `--checkpoint-every` / `--no-resume`；续跑时 CSV 以追加模式打开，
     避免重复样本，CSV 仍是单调的单一序列。
   * 与 shell 级重试循环配合：`for i in 1..N; do python run.py --checkpoint ... && break; done`
     —— 重试必须在进程外（堆中止会杀掉进程），这与 2D 通道已经建立的「边跑边落盘」策略同源。

## 3. 验证（跑了什么）

```
python -m pytest tests/test_kleefsman_3d_checkpoint.py -q     ->  6 passed in 38.70s
```

新增测试断言的是**不变量**，不是「跑一遍没报错」：

| 测试 | 断言 |
|---|---|
| `test_resume_is_bit_identical_to_an_uninterrupted_run` | 崩溃后从状态文件续跑，`steps`、`simulated_time` 与**不中断的直跑逐位相等**；`times/slab/height/front` 四条序列**逐元素 `==`** |
| `test_a_crashed_run_keeps_the_state_it_reached` | 崩溃留下的状态确实带着已到达的物理状态与已产出的时程，且无 `.tmp` 残留 |
| `test_resume_extends_the_horizon` | 用更长 `t_end` 重入时真的往前走（这是崩溃场景的唯一意义） |
| `test_checkpoint_from_another_configuration_is_refused` | 换 `wall_layers` 后 `load_checkpoint` 与 `run_case` 都拒绝，并点名差异项 |
| `test_a_completed_run_leaves_no_checkpoint` | 完成的网格不留状态文件（重试循环不能重放） |
| `test_checkpoint_carries_its_version` | 状态文件带版本号，读得回 |

「中断」用样本回调抛异常模拟：就求解器而言这与真实堆中止等价（已写下的状态留在盘上、run 永远走不到清理段）。

## 4. 注意点 / 局限（如实说）

1. **我没有定位到堆损坏的根因**。现有证据只支持：它在纯 torch CPU 代码路径上、纯 Python 层不可能产生越界写、
   与线程数/机器无关地以 1–2 s 模拟时间为量级出现。本单交付的是**让它不再致命**（数据可续），
   不是**让它不再发生**。根因定位需要 memcheck 级工具（loongarch64 上不可用）或 torch 内部的 arena 诊断，
   超出单干范围。
2. 单次作业的**总墙钟**不变：崩溃+续跑的总计算量等于一次完整运行，只是不再归零。
3. 续跑是**逐位精确**的（同一 `dt`、同一状态、同一邻接重建顺序），所以拼起来的时程可以当成一次连续运行使用；
   这不是「近似重启」，用时程算出的峰值/到达时刻不需要打折扣。
4. **锚点的物理结论仍未变**：三维阻塞率归因的数值（三维峰值/到达时刻 vs 实验）本单**没有产出** ——
   续跑版生产 run（`cj4230`，dx = 0.05，t_end = 2.6，最多 6 次重试）递交时队列仍有 ~50 个作业在前，
   本单收尾时未回；落地后按
   `python benchmarks/kleefsman_obstacle_3d/run.py ... && python benchmarks/kleefsman_obstacle_3d/ladder.py && python benchmarks/kleefsman_obstacle_3d/make_doc.py`
   即自动纳入，`ladder.json` 会从 `NO_RUNS` 变成实际阶梯。
5. 二维通道的诚实 FAIL（主测站峰值 +30.1 %，`PEAK_FAIL_MODEL_ERROR`）**不因本单改变**，也没有被覆盖。

## 5. 误差判定

本单是工程修复，不产生对实验基准的物理量，故无「vs 基准误差」可算。唯一可量化的是续跑与直跑的差异：

| 量 | 直跑 | 续跑 | 差异 | 是否 < 3 % |
|---|---|---|---|---|
| `steps` | 23 | 23 | 0 | 是（0 %） |
| `simulated_time` [s] | 0.0201750 | 0.0201750 | 0 | 是（0 %） |
| `times` / `slab` / `height` / `front` 序列 | — | — | 逐元素 `==` | 是（0 %） |

（数值取自 `tests/test_kleefsman_3d_checkpoint.py` 的运行，任何一次 `pytest` 都可复算。）

## 6. 产物指针

* 代码：`src/tensorsph/kleefsman_obstacle_3d.py`（`save_checkpoint` / `load_checkpoint` / `run_case`）、
  `benchmarks/kleefsman_obstacle_3d/run.py`（CLI）
* 测试：`tests/test_kleefsman_3d_checkpoint.py`
* 前序证据：`benchmarks/kleefsman_obstacle_3d/results/abort_evidence/`（三次 ABORT 的原始记录与图）