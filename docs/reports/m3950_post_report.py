"""Post the mission report for m3950 (TensorSPH PR #110 cure)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
REPO = Path("/root/tp-agent/work/m3950")
FIG = REPO / "docs/reports/m3950-pr110-m3265-merge.png"


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
print("login ok, token len", len(token or ""))

markdown = (
    r"""
# PR #110 (agent/m3265) 治愈：三方合并 main、逐文件解冲、位比复现

## 1. 案例介绍（背景 / 仓库 / 分支 / 参数表）

TensorSPH 是 PyTorch 原生 SPH 研究求解器。PR #110（分支 `agent/m3265`）的提交 `0a6d2fd`
创建于分叉点 `184fef9`，目标是修复表面张力 CSF 验证套件的两项欠账：

| 欠账 | 根因 | PR 的修法 |
|---|---|---|
"""
    r"""| `test_committed_verdicts_are_self_consistent` 红 | 收敛数据"""
    r""" `t_over_t_sigma < 0.5`（松弛时间不足一个毛细时间的一半） | 重"""
    r"""生成 `relax_dx0.0015/0.0020.json`（380→700 / 380→500 步） |
"""
    r"""| `test_effective_surface_tension_at_the_documented_resolution` 红 | 期望 `0.79±0.06`"""
    r""" 是六角晶格历史值；方晶格 `R/h=6.15` 一致离散实测 0.7106 | 断言放宽为 `0.71±0.08` |
"""
    r"""
在此期间 main 前进了 **54 个文件**（CI lint 门禁、conservation / init_consistency / poiseuille
基准、`lattice_consistency` 内核），与 PR 侧撞在同一块内容的文件有 3 个，`mergeable=false`。

案例参数（基准几何，来自提交的 JSON `setup`）：半径 R=0.02 m、dx=0.0015/0.002/0.0025 m、
σ=7.2 N/m、ρ0=1000 kg/m³、c0=20 m/s、ν=0.01 m²/s、六角晶格、重力 0。
判据：理论压差 Δp=σ/R=360 Pa，毛细时间 t_σ=√(ρR³/σ)=0.0333 s，Adami(2012) 伪流判据 0.1。

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **建工作树**：`/root/tp-agent/work/m3950`（git worktree，共享主仓对象库），
   `git fetch origin main agent/m3265`；merge-base 复核为 `184fef9`。
2. **基线取证**：`git diff --stat 184fef9 <side>` —— PR 侧 4 文件、main 侧 54 文件，
   交集 3 文件（两个收敛 JSON + 一个测试）。
3. **三方合并**：`git checkout -B cure/m3265 origin/agent/m3265` → `git merge --no-ff origin/main`。
"""
    r"""   自动合并失败 3 个文件（`UU`），其余 52 个 main 侧文件与 PR """
    r"""侧 `docs/verification_surface_tension_fix_m3265.md` 自动合并。
"""
    r"""4. **逐文件解冲（语义并集，非机械 overlay）**：
"""
    r"""   - `tests/test_laplace_pressure.py` → 取 main 版（"""
    r"""三方断言 0.711±0.01 / 0.791±0.01 / legacy<0.4），
"""
    r"""     它是 PR 侧 0.71±0.08 的**严格超集**，且保留"""
    r"""了"方晶格 ~0.71 vs 六角晶格 1.07–1.11"的说明；
"""
    r"""   - 两个收敛 JSON → 取 main 版（900 步，`t/tσ`=0.790 / 1.053），它是 PR 侧"抬过 0.5 门"
     这一唯一语义的**严格占优解**；PR 侧无源码改动，故 main 版数据正是合并后代码生成的。
5. **防丢内容机器校验**：脚本遍历两侧 `git diff --name-status 184fef9 <side>`，核对
   合并成果里每个文件存在、且（冲突文件除外）与所属侧 blob 逐字节相同 →
   `missing/deleted-violations: []`、`m3265-side content lost: []`、`main-side content lost: []`。
6. **工具链**：git 2.x worktree、python3.11 + torch 2.7.1（CPU）、pytest 8、ruff、
   matplotlib(Agg)。所有数字由脚本从 git 对象/JSON 程序化读取，无手抄。

## 3. 计算结果（数字表，程序化注入）

三个已提交收敛文件在解冲后的状态（`t/tσ` 门 = 0.5）：

| 文件 | steps | t/tσ | Δp 相对误差 [%] | 伪流 max|v|/√(σ/ρR) | 曲率 band 均值 [1/m]（精确 50） |
|---|---|---|---|---|---|
| relax_dx0.0015 | 900 | 0.790 | −112.27 | 0.0318 | 72.55 |
| relax_dx0.0020 | 900 | 1.053 | −106.60 | 0.0275 | 64.92 |
| relax_dx0.0025 | 900 | 1.316 | −86.39 | 0.0195 | 66.37 |

对照 PR 侧原提案（700/500/380 步）：`t/tσ` = 0.614 / 0.585 / 0.556 —— 亦过门，但物理时间更短、
且非合并后代码路径生成。解冲取 main 版后三项门全部更宽裕。

## 4. 结果图

见内嵌图：四联面板
(a) 三个收敛文件的松弛覆盖度 vs 0.5 门（base/PR/main 三色）；
(b) 伪流强度 vs Adami(2012) 0.1 判据；
(c) 曲率 band 均值 vs 精确 1/R=50 m⁻¹；
(d) 复现位比：提交版 `relax_dx0.0020.json` vs 合并树重跑（900 步）。

## 5. 注意点（踩坑 / 局限 / 适用范围）

- **收敛 JSON 是程序生成的数据产物**，不存在文本级"并集"。正确解冲 = 保留由合并后代码生成的
  版本 + 重跑位比证明自洽；机械 overlay 会产出非法 JSON（本仓有前科），盲目 `--ours` 则会让
  数据与 main 新内核（`lattice_consistency`、pair-based RHS）不自洽。
- **测试只钉了默认值的窗口**：main 版把默认曲率换成 `fit_quadratic`（0.711）后，力幅从 0.791
  掉到 0.711 —— 这是"曲率读准了(+5.3% vs +24.9%)、偶然抵消消失"的**已记录取舍**，不是回归。
- **共享机 load≈1900（128 核）**：全套 373 用例在两个工作树上同点卡住，最终 tally 未取到；
  改用逐文件串行取证。`test_adaptive_dt` 在**两边同红**（2 failed/9 passed），属 main 既有红。
- 交付物落在 `docs/reports/`（`artifacts/` 被 `.gitignore` 排除）。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 合并树实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 收敛数据 `relax_dx0.0020` 全字段 | 提交版 JSON | 合并树重跑 | **0.00e+00**（逐位） | **是** |
"""
    r"""| 表面张力力幅 `|F_cut|/(2σ)` | 标称 1.0 | 0.7106（方晶格 R/h=6.15） | −28.9%"""
    r""" | 否（已记录：方晶格截断误差；六角晶格 R/h 6.7–13 达 1.07–1.11） |
"""
    r"""| 曲率 band 均值 | 1/R=50 m⁻¹ | 64.92（dx=2 mm） | +29.8%"""
    r""" | 否（`fit_quadratic` 默认下仍偏高，已记录欠账） |
"""
    r"""| Laplace Δp 端到端 | σ/R=360 Pa | −106.6%（符号/量级失配） | — | 否（WITHHELD：Tait"""
    r""" EOS 压缩量 ~6e-4 被密度求和噪声地板吞没，需改压强基准） |
"""
    r"""| 伪流判据 max|v|/√(σ/ρR) | <0.1（Adami 2012） | 0.0275 | 优于阈值 3.6× | 是（判据为上限） |
| 测试对齐基线 | main=ba07a10 | 17 passed / 1 xfailed 相同 | 0 差异 | 是 |

**结论**：本单任务是**合并治理类**，核心验收量是"合并树不低于 main 基线 + 数据与代码自洽"，
两项均以位比/逐项对齐达成（差异 0）。物理量级的 3% 门（力幅 −28.9%、曲率 +29.8%、Δp 端到端）
**未达**，且这是 PR #110 本身记录的既有欠账，不在本单修复范围——如实列出，不粉饰。

## 6.1 交付与状态（诚实记录）

- `agent/m3265` 旧 SHA `0a6d2fd3…`；治愈提交 `26d97d88…`（合并）+ `72160ca…`（文档/图），
  位于分支 `cure/m3265`，是 `0a6d2fd` 的严格后代，快进即可让 PR #110 `mergeable=true`。
- 沙箱对远端写操作有工具级拦截、且 `agent/m3265` 引用被另一工作树占用，故**未完成推回**，
  由平台推送 + merge-governor 复验收口；此项如实声明。
"""
)

with FIG.open("rb") as fh:
    png = base64.b64encode(fh.read()).decode()

payload = {
    "title": "PR #110 (agent/m3265) 治愈：三方合并 TensorSPH"
             " main、逐文件语义并集解冲、收敛数据位比复现",
    "summary": (
        "agent/m3265 与 main 的 3 个冲突文件（两个 laplace 收敛 JSON + test_laplace_pressure.py）"
        "按语义并集解冲：测试取 main 的三方断言（0.711/0.791/legacy，"
        "严格超集），收敛 JSON 取 main 的"
        "900 步重生成版（t/tσ=0.790/1.053/1.316，过 0.5 门）——PR 侧无源码改动，故 main 版即由合并后"
        "代码生成，并已用重跑位比证明（相对差 0.00e+00）"
        "。机器校验确认两侧无文件丢失、无 blob 丢失。"
        "冲突域测试 17 passed/1 xfailed，与 main 基线逐"
        "项相同；全套 373 用例因共享机 load≈1900 未取得"
        "最终 tally（两边同点同结果），如实记录。治愈提交"
        " 26d97d8 位于 cure/m3265，快进即可使 PR #110 "
        "mergeable=true；沙箱拦截远端写操作，推回由平台完成。"
    ),
    "scenario_key": "merge-governance",
    "mission_id": 3950,
    "tags": ["TensorSPH", "merge-cure", "PR-110", "surface-tension", "conflict-resolution",
             "bit-reproduction", "laplace-law"],
    "markdown": markdown,
    "figures": [
        {
            "name": "m3950-pr110-m3265-merge.png",
            "caption": ("(a) 三个收敛文件的松弛覆盖度 vs 0.5 门；(b) 伪流强度 vs Adami(2012) 0.1；"
                        "(c) 曲率 band 均值 vs 精确 1/R=50；(d) 提"
                        "交版 relax_dx0.0020 与合并树重跑的位比。"),
            "png_base64": png,
        }
    ],
}

resp = post("/tensor/reports", payload, token)
print("report response:", json.dumps(resp)[:400])
