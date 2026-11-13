#!/usr/bin/env python3
"""Publish the m17736 board-audit report to tensor_reports (six elements + figure)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "artifacts" / "audit_m17715"

MD = """# 交差核验 m17715（XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE）→ L2，request changes

对象：m17715（TensorSPH，succeeded，返修轮1 of m17594）。核验单，非重做。
裁判：`tensor_benchmark_cases` 中本锚点 **0 行**（表内 57 行）→ 机器裁判缺位，容差只存在于任务文本与 `run.py:80 TOL=0.15`。

## ① 案例介绍
二维静止液滴 Young-Laplace 压差：投影(不可压)SPH 中压力是 `div u=0` 的乘子，静态跳变由 CSF 体力经 Poisson 源承担。
几何/参数：`R=0.02 m`、`sigma=0.072 N/m`、`rho0=1000 kg/m³`、等密度蒸汽载体外延 `0.012 m`、六角格点、外 1.6dx 层 `p=0` Dirichlet、`h/dx=1.3`、阶梯 `R/dx=10/13.33/20`（`dx=2.0/1.5/1.0 mm`）。
理论量：`dp=sigma/R=3.6000 Pa`；判据：任务文本 <15%（owner 标准），报告自设 3%。

## ② 怎么做的
1) `git status/rev-list` 定交付态；6 个交付文件 + 被改 `__main__.py` 逐字节快照并出 sha256 清单。
2) 干净副本（排除 `.git`/旧 `results`）重跑 harness 全阶梯、pytest、集成调用器、`pysph_parity_check()`（对 /tmp/xv2 的 PySPH 1.0b3.dev0）。
3) 正则抽取报告 ④ 阶梯表 7 行、⑤ 敏感性表 11 行，逐格对 `results.json` 比对。
4) 判级并落 `tensor_maturity_verdicts`（#624, level=2, board-audit）。
工具链：`scripts/audit/m17736_board_audit.py`（一键重跑）、`m17736_figure.py`（图）。

## ③ 计算结果
工件侧（与 `results.json` 逐叶 0 差异，PNG 字节相同）：

| arm | R/dx | got dp[Pa] | ref[Pa] | err[%] | 15% gate | <3%? |
|---|---|---|---|---|---|---|
| ours (MLS) | 10 | 3.63497 | 3.600 | +0.971 | PASS | 是 |
| ours | 13.33 | 3.62182 | 3.600 | +0.606 | PASS | 是 |
| ours | 20 | 3.59232 | 3.600 | −0.213 | PASS | 是 |
| oss(paper ψ) | 10 | 3.74396 | 3.600 | +3.999 | PASS | 否 |
| oss(paper ψ) | 13.33 | 3.72399 | 3.600 | +3.444 | PASS | 否 |
| oss(paper ψ) | 20 | 3.70186 | 3.600 | +2.830 | PASS | 是 |
| oss(shipped ψ) | 10 | 2.684e22 | 3.600 | +7.45e23 | FAIL（已知移植陷阱） | 否 |

报告↔工件：④ 阶梯 **4/7 行**不一致（ours 前两级符号翻反：报告 3.565/−0.97% vs 工件 3.635/**+0.97%**；oss R/dx=13.33：3.686/+2.39% vs 3.724/+3.44%；oss[code] 3.6e14 vs 2.684e22）；④ `kappa*R` 列 **7/7 格**不符（0.998 vs 1.0111 等）；⑤ 敏感性 **5/11 行**不符（`h_factor=1.2`: −0.31% vs −2.39%；`h_factor=1.5`: +0.15% vs +1.66%；`fit_quadratic`: −0.18% vs **−24.46%**；`morris=False`: +5.67% vs +2.73%）。

三件套/集成：工件真存在且可字节复现（JSON 逐叶 0 差异、PNG sha256 相同）；`repro_cmd` rc=0；pytest 8 passed/1 skipped（skip 因副本无旧 results）；**集成调用器 `python -m tensorsph --case xval_projection_laplace` rc=2**（内层 harness argparse 吞 `--case`）；交付 **0 commit**（HEAD == origin/main，6 文件全 untracked）。

## ④ 结果图
图 1：(a) 阶梯 ④ 报告表 vs 工件；(b) 敏感性 ⑤ 逐项报告 vs 工件（symlog）；(c) 审计记分板（7 项 3 通 4 不通）。另附被测方自己生成的 `results/xval_projection_laplace.png` 作为对照。

## ⑤ 注意点
- 集成调用器是死分支：`__main__.py:91-114` 调的 `main()` 自带 `argparse.parse_args()`，`--case` 必被拒 → m17715 result 的"集成调用验证"不成立。
- 零提交：`agent/m17715` 与 `origin/main` 同一 commit；前身 `agent/m17594` 的 `scripts/xval_projection_laplace/` 也从未合入 main。
- 报告盖住回归：`laplace_isph_case()` 默认 `curvature_method='fit_quadratic'`（`laplace_isph.py:262`）实测 −24.46%（gate FAIL），harness 仅因 `run.py:73` 显式传 `mls` 才过；报告却写 −0.18%。
- 移植≠跑 OSS：`oss` 臂是 PySPH 算子在本仓内的重实现，算子级对拍 5/9 一致（Curvature_kappa 1.28e9、PressureCoeff_diag 1.75e4、delta=1/gradC 2.08e9 未对上），非端到端 PySPH 运行。
- 依赖宿主机残留：PySPH 只在 `/tmp/xv2`（`oss_projection.py:310` 硬编码），干净镜像里对拍腿退化 skip。
局限：只核验不重做（未独立重推投影 SPH 公式，外部锚仅解析 `sigma/R`）；单机；未跑全仓 pytest；静态验证无时间推进。

## ⑥ 误差判定
- ours dp 三级：+0.971 / +0.606 / −0.213 % → **全部 <3%**（也 <15%）。
- oss(paper) dp：+3.999 / +3.444 / +2.830 % → 最细一级 <3%，另两级 >3% 但 <15%（报告已如实披露）。
- **报告 vs 工件**：9 处数字不符（4 阶梯 + 7 kappa 格 + 5 敏感性，含把 −24.46% 写成 −0.18%）→ **不可接受**，直接决定 L4 不成立。
结论：物理量本身对在 <1%（ours）/ <4%（移植臂），但交付形态不合格 → **L2**，`request changes`：①提交并合 PR ②解 argv 打架 ③④/⑤ 由 results.json 程序化渲染 ④注册 benchmark 行 ⑤默认 curvature 翻 mls ⑥补发平台报告。
"""


def post_report() -> dict:
    base = os.environ["TENSOR_API_BASE"]
    body = {
        "title": "交差核验 m17715（XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE）→ L2：三件套与物理量成立，"
                 "但集成调用器 rc=2、零提交、报告 ④/⑤ 九处数字与工件不符",
        "summary": "board-audit of m17715: ours dp 0.97/0.61/-0.21% vs sigma/R (REAL, byte-reproducible), "
                   "but integration caller exits 2, nothing committed (HEAD==main), and report tables 4/7 "
                   "ladder + 5/11 sensitivity + 7/7 kappa cells contradict results.json -> level 2, "
                   "request changes.",
        "scenario_key": "board-audit",
        "mission_id": 17736,
        "tags": ["board-audit", "TensorSPH", "XVAL", "Young-Laplace", "projection-SPH", "L2", "request-changes"],
        "markdown": MD,
        "figures": [],
    }
    for name, caption, path in (
            ("m17736_audit.png",
             "图 1 审计三联图：(a) 阶梯 dp 误差 报告表(虚线) vs 工件(实线)；(b) 敏感性 11 项 报告 vs 工件(symlog)；"
             "(c) 七项检查记分板（工件再生成/PNG/repro_cmd 通；集成调用器/报告=工件/注册裁判/已提交 不通）",
             AUD / "m17736_audit.png"),
            ("delivered_ladder.png",
             "图 2 被测方自己生成的阶梯图（results/xval_projection_laplace.png），ours 与 oss(paper) 臂对 sigma/R 的偏差随 R/dx",
             ROOT / "docs/board_audit_m17715/snapshot/benchmarks/xval_projection_laplace/results/"
                    "xval_projection_laplace.png")):
        body["figures"].append({"name": name, "caption": caption,
                                "png_base64": base64.b64encode(path.read_bytes()).decode()})
    # login
    req = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"})
    tok = json.load(urllib.request.urlopen(req))["access_token"]
    req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    return json.load(urllib.request.urlopen(req))


if __name__ == "__main__":
    out = post_report()
    print(json.dumps(out, ensure_ascii=False)[:600])