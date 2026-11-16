# 方法选择证据链的「派生件新鲜度」缺陷：公布的严格上界是过期数字（m22608）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ **任务**: m22608（自找活·验证欠账）
**类型**: 证据完整性缺陷修复 + 守卫（adversarial-review 口径）

## ① 案例介绍

锚点的结论不是一句话，而是**一个源 + 四套派生件**：源是
`docs/reports/method_selection_registry.json`（每条记录 = 某方法的某次测量及其误差），
派生件由四个脚本各自生成：

| 生产者 | 产物 | 在本缺陷前后的 `--check` 状态 |
|---|---|---|
| `benchmarks/method_selection/build_registry.py` | registry + matrix + tables + 决策卡片 | 绿（m18056 起有反漂移检查） |
| `benchmarks/method_selection/audit_quantity_coverage.py` | `method_selection_coverage_audit_m20432.{json,md}` + png | **红（陈旧）** |
| `benchmarks/method_selection/audit_decision_fragility.py` | `method_selection_fragility_m21797.json`（留一记录法脆弱性） | 绿（m21797 起有反漂移检查） |
| `benchmarks/method_selection/strict_bound_profiles.py` | `method_selection_strict_bound_m21537.{json,md}` + png | **红（陈旧）** |

在干净检出（merge `b5e0e492`）上实测：两个生产者 `--check` 直接返回非零并打印
`DRIFT` / `is stale: rerun without --check`。这不是我引入的，是**已合并状态**。

物理/工程含义：登记表是「这个方法在多大误差内可用」的唯一账本；派生件是读者真正看到的
卡片（`docs/xval_method_selection_strict_bound_m21537.md` 是 anchor 的公开结论页）。
派生件一陈旧，**公布数字与选择器实际使用的数字就不是同一个数**——而这恰好发生在
「哪个方法能被推荐」这件事上，是最不该悄悄漂移的地方。

## ② 怎么做的

1. 读出干净检出上四个生产者的 `--check` 结果（只读，不动文件）；
2. 对两个红的生产者**不带 `--check` 重跑**，用 `git diff` 逐字段比对差异，判定漂移的
   性质（数值？结论？还是只有计数？）；
3. 判定「是否改变决策」：比较重跑前后的 strict/lenient 模式列与 `counts` 块；
4. 把这一类缺陷（而非这一个实例）关掉：新增
   `tests/test_method_selection_artefact_freshness.py`，用子进程把**四个**生产者全部
   以 `--check` 跑一遍，任何一个陈旧即红；另加一条交叉钉值：coverage 审计里写的
   `registry_records` 必须等于 registry 的实际记录条数；
5. **负控制**（证明守卫真的会响）：把 coverage 审计的两份产物 `git checkout` 回陈旧版本，
   重跑新测试 → 必须 FAIL；再重跑生产者复原 → 必须 PASS。

## ③ 计算结果（程序化注入，`git diff` 原样）

### 3.1 `strict_bound_profiles.py`：公布的毛细上界是过期数字

| 字段（`docs/reports/method_selection_strict_bound_m21537.json`，capillary 剖面） | 陈旧值（已合并） | 重跑值（= 当前 registry） |
|---|---|---|
| `strict bound` 排除理由里的 worst live accuracy record | **23.8 %**（CASE-LAPLACE-SPH (consistent CSF)：`delta_s` 加权平均曲率误差） | **45.69 %**（CASE-LAPLACE-SPH (quasi-static, time-averaged)：4–6 t_sigma 窗口内 bulk 压强对真空，最差栅格 dx = 3/2/1.5 mm） |
| 同一句出现在 `notes` | 23.8 % | 45.69 % |
| MD 表 `capillary droplet (Laplace law)` 行的 bound 列 | 23.80 | 45.69 |
| **该行 lenient/strict 判定** | ✘ / （无） | ✘ / （无） |

差额来源可定位：registry 从 **29 → 30** 条记录（新增的正是上面那条 45.69 % 的 CSF 准静态记录），
`strict_bound_profiles.py` 未被重跑。

### 3.2 `audit_quantity_coverage.py`：记录计数陈旧

| 字段 | 陈旧值 | 重跑值 |
|---|---|---|
| `registry_records` | 29 | 30 |
| `coverage_complete` | True | True（未变） |
| `gated` | 25 | 25（未变） |

### 3.3 守卫的判别力（负控制）

```
$ PYTHONPATH=src pytest tests/test_method_selection_artefact_freshness.py -q -k "coverage or record_count"
FAILED tests/.../test_producer_artefacts_are_a_fresh_build[audit_quantity_coverage.py]   # 退回陈旧产物后
# 重跑生产者复原后：
$ PYTHONPATH=src pytest tests/test_method_selection_artefact_freshness.py -q
5 passed in 28.31s
```

## ④ 注意点（踩过的坑 / 适用范围）

- **这条缺陷不改变任何推荐结论**（诚实记录）：毛细剖面在陈旧与新鲜两种状态下都是
  「strict 模式 WITHHOLD、lenient 模式 SELECTED」，`counts` 块
  `{profiles: 7, lenient_within: 4, strict_within: 5, strict_withheld: 2, excluded_total: 3}`
  未变。所以它是**公布数字错误**（严重度：中），不是**决策错误**（严重度：无）。
  把它写成「发现推荐错了」就是夸大。
- 我**没有**顺手修 `audit_quantity_coverage.py` 自己报的两个疑点
  （`integrity_failures=['x_couette_design']`、`over_owner=[5 项]`）：它们是审计脚本的
  输出内容，属于独立的证据账，不在本单范围；本单只保证「产物是当前源的新鲜构建」。
- 守卫是子进程 + `--check`，因此它检查的是**产物与源一致**，不是**产物正确**。
  产物对不对，仍要靠各自的审计结论与 `tests/test_method_selection_*.py`。
- 新增测试耗时 28 s（四次脚本启动 + 一次 JSON 交叉比对），适合 CI。

## ⑤ 误差判定

| 关键量 | 基准 | 误差 | < 3 % ? |
|---|---|---|---|
| 陈旧产物 `registry_records` 29 vs 真值 30 | registry 实际条数 | 1 条（3.3 %） | **否**（这正是缺陷；修复后 0 %） |
| 陈旧产物「capillary worst live accuracy record」23.8 % vs 真值 45.69 % | registry | **相差 21.89 个百分点 / 相对 47.9 %** | **否**（修复后 0 %） |
| 修复后四个生产者 `--check` | 干净构建 | 0 处漂移 | **是** |

即：修复前该项**明确不满足** 3 %（也不满足 15 %）口径；修复后严密一致（逐字节）。

## ⑥ 未做的事

- 没有把这次漂移追溯成「哪一单加了第 30 条记录却没有重跑两个生产者」的责任链——
  这需要逐 commit 的 `git log -S` 追踪，与本单的「关掉这类缺陷」目标无关。
- 没有改 `audit_quantity_coverage.py` / `strict_bound_profiles.py` 的输出口径（只重跑）。