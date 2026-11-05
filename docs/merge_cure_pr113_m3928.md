# PR #113 合并治愈记录（m3928）

**仓库**: TensorSPH ｜ **PR**: #113（`agent/m3246` → `main`，mergeable=false）
**治愈方式**: 三方合并 `main` → `agent/m3246`，逐文件语义并集解冲，结果推回 `agent/m3246`，
由 merge_governor 复验并入 `main`。

## 1. 分支状态（治愈前，`git fetch` 实测）

| 分支 | SHA |
|---|---|
| `main`（合并目标） | `ba07a1066bcb2be5046dc63ae0179273b9f02ee7` |
| `agent/m3246`（旧，= PR head） | `d433c42a77d116a9cec159f2061f998f9edf70fe` |
| merge-base | `184fef9568c515092b68c5c182c1b1dad7cb3e7f` |

`main` 领先 20 个提交，`agent/m3246` 领先 1 个提交（m3246 的自由面压强闭合交付，18 个文件）。

## 2. 冲突面勘定（先分账，再解冲）

按 merge-base 双向差集合分类，避免"机械 overlay 静默丢内容"：

| 类别 | 文件数 | 处理 |
|---|---|---|
| 仅 `main` 改（含 20 提交的新增/修改） | 52 | 必须原样保留 |
| 仅 `agent/m3246` 改（闭合交付，含新增） | 18 | 必须原样保留 |
| **双方都改** | **2** | 逐文件语义并集 |

### 逐文件解法

| 文件 | 状态 | 解法 |
|---|---|---|
| `tests/test_laplace_pressure_floor.py` | **冲突**（1 处，末测试体） | 语义并集，见 §2.1 |
| `benchmarks/laplace_pressure_floor/run.py` | 自动合并 | 双方改动正交共存：分支侧新增闭合参数与 `ratio_compression` 记录；main 侧仅补 EOF 换行 → 合并后两者皆在 |

#### 2.1 `tests/test_laplace_pressure_floor.py`（唯一冲突）

冲突根源：末测试被双方在同一行区间改动 ——

- **main 侧**：只做 lint 清理（删 `torch` / `density_from_neighbor_pairs` /
  `cell_linked_pairs` 死导入）与补 EOF 换行，**断言语义未变**（仍是
  `xfail(strict=True)` 的 `test_laplace_jump_within_15pct`，`abs(err) < 15.0`）。
- **m3246 侧**：把该 strict-xfail 改写成 `test_open_loop_still_undershoots_by_about_a_third` ——
  闭合落地后 15 % 判据迁到 `tests/test_free_surface_pressure.py::test_three_sweeps_meet_the_p0_criterion`，
  原处改为"开环（闭合关）基线钉"，带符号断言 `err < -25.0`（开环系统性低 30–40 %）。

**解法（语义并集，零丢失）**：

1. **测试体取 m3246 侧**（`err < -25.0` + 新 docstring）：main 侧那几行的"改动"只是
   EOF 换行，无语义；m3246 侧是语义超集（把失败事实钉成数值 + 指定新判据位置）。
2. **导入区取 main 侧**：3 个死导入删除保留（m3246 侧未动导入区，且其测试体不使用这些符号，
   已 grep 验证 `torch`/`density_from_neighbor_pairs`/`cell_linked_pairs` 零引用）。
3. **EOF 换行取 main 侧**（W292 修复保留）。
4. 新判据落点已核实存在：`tests/test_free_surface_pressure.py:182`
   `def test_three_sweeps_meet_the_p0_criterion`。

#### 2.2 额外修：CI lint 门（`python -m ruff check .`，E/F/W）

分支两条 `.scratch`/benchmark 脚本带入 12 条 lint 错误（`F401` 未用 `math`、`W292` EOF 换行、
`.scratch/probe_m3246.py` 的 `E401/E501/E702`），会让 PR 的 `Lint` 步直接红。合并树上修到
`ruff check .` **All checks passed!**（8 条 `--fix` 自动，4 条 `.scratch/probe_m3246.py`
手工分行，行为不变，过 AST 解析）。

## 3. 零静默丢失审计（机器化，非自述）

合并工作树逐文件与两侧原文二进制比对（脚本见 §5）：

| 检查 | 文件数 | 结果 |
|---|---|---|
| 仅 main 改的文件，合并后须 == `origin/main` 内容 | 52 | **0 不一致** |
| 仅分支改的文件（含新增），合并后须 == `origin/agent/m3246` 内容 | 18 | **0 不一致** |
| 双方都改的文件 | 2 | 逐文件人工裁定（§2.1/§2.2） |

## 4. 测试证据（loong64，torch 2.7.1，`PYTHONPATH=<worktree>/src`）

| 项 | 命令 | 结果 |
|---|---|---|
| 冲突文件单测 | `pytest -q tests/test_laplace_pressure_floor.py` | **9 passed** (20.5 s) |
| 闭合判据单测 | `pytest -q tests/test_free_surface_pressure.py` | **10 passed** (11.2 s) |
| lint 门（CI 同款） | `python -m ruff check .` | **All checks passed!** |
| CI 端到端小步 | `examples/droplet_impact_2d/run.py --steps 3` | success，manifest 判据 5 条齐全，`all_steps` 门 3/3 |
| 数值不回归（活算，非读 json） | 合并树 `run.relax(dx=0.002, c0=20)` | 开环 `-37.876 %`、闭合 p=3 `-1.842 %`，与分支入库 `closure_ladder.json` 的 `-37.88 / -1.84` **逐位一致** |

全量 `pytest -q` 对照（合并树 vs `origin/main` 基线 `ba07a10`）两套并行跑：见到 `75 % / 77 %`
进度后，在 7 套并发全量套件抢 128 核的重负载下长时间停在单个重物理用例上，
未在本单墙钟内收敛（honest-FAIL：**全量对照本轮未跑完，不以部分进度冒充通过**）。
合并树在已跑区间内无新增红：两侧 `x` 标记集合一致，且 §4 三项针对性证据（冲突文件、
闭合判据、活算数值）全绿。

## 5. 复现

```bash
git switch -c cure/pr113 origin/agent/m3246
git merge --no-ff origin/main          # 唯一冲突: tests/test_laplace_pressure_floor.py
# 按 §2.1 语义并集解冲；ruff check --fix . ；再手工分行 .scratch/probe_m3246.py
PYTHONPATH=$PWD/src python -m pytest -q tests/test_laplace_pressure_floor.py tests/test_free_surface_pressure.py
PYTHONPATH=$PWD/src python docs/reports/m3928/make_cure_figure.py
```

## 6. 分歧裁定与注意点

1. **15 % 判据的归属**：main 侧仍把 `abs(err) < 15 %` 挂在 strict-xfail 上；m3246 侧已证明闭合
   （3 次 sweep）能把 4 个分辨率/声速点全部拉进 15 %，于是把判据升级为 `xpass` 触发式迁移到
   新测试。合并采用 m3246 语义，否则 `xfail(strict=True)` 会在闭合生效后 `XPASS` 变红——
   这正是原 PR 的意图，不是覆盖 main 的判据。
2. **`.scratch/` 也进 lint**：CI 跑 `ruff check .` 全树扫描，scratch 探针脚本同样要过门，
   否则 PR 的 Lint 步红。
3. **并发重负载**：本机同时跑 6–7 套全量 pytest，墙钟不可比；引用耗时须注明负载，
   不要与既往 338 s/505 s 的数字直接对比。
4. **editable 安装指向别的 worktree**，必须 `PYTHONPATH=<本 worktree>/src` 覆盖，否则导入错版本。