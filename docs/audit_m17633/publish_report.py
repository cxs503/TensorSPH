#!/usr/bin/env python3
"""Publish the m17633 audit report to the platform (credentials from env, never inlined).

All numbers in the markdown are pulled from the committed evidence bundle; figures are
base64-embedded from the files produced by this audit.
"""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = pathlib.Path("/root/tp-agent/work/m17633/docs/audit_m17633")
EV = HERE / "evidence"


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": USER, "password": PASS})
tok = tok.get("access_token") or tok.get("token")
print("login ok, token len", len(tok))

J = json.loads((EV / "results_committed_cc48564.json").read_text())
REPRO = json.loads((EV / "repro_identity_check.json").read_text())
VERD = json.loads((EV / "m17633_verdict_evidence.json").read_text())


def arm(prefix):
    rows = [r for r in J["runs"] if r["arm"].startswith(prefix)]
    return sorted(rows, key=lambda r: r["R_over_dx"])


ours, oss = arm("ours[mls"), arm("oss[paper")
code_worst = max(arm("oss[code"), key=lambda r: abs(r["dp_error_pct"]))

rows = []
for o, s in zip(ours, oss):
    rows.append("| {:.2f} | {:.4f} | {:.4f} | {:+.2f} | {} | {:.4f} | {:+.2f} | {} | {:.2f} |".format(
        o["R_over_dx"], o["dp_theory_Pa"], o["dp_measured_Pa"], o["dp_error_pct"],
        "PASS" if abs(o["dp_error_pct"]) <= 15 else "FAIL", s["dp_measured_Pa"], s["dp_error_pct"],
        "PASS" if abs(s["dp_error_pct"]) <= 15 else "FAIL",
        abs(o["dp_measured_Pa"] - s["dp_measured_Pa"]) / o["dp_theory_Pa"] * 100))
ladder = "\n".join(rows)

sens = []
for s in J["sensitivity"]:
    sens.append("| {} | {} | {} | {} |".format(
        s["vary"], s["value"],
        "{:+.2f}".format(s["ours_pct"]) if "ours_pct" in s else "-",
        ("{:.3e}".format(s["oss_pct"]) if abs(s.get("oss_pct", 0)) > 1e3 else "{:+.2f}".format(s["oss_pct"])) if "oss_pct" in s else "-"))
sens_table = "\n".join(sens)

default_block = (EV / "probe_default_kwargs.log").read_text().strip()
band_block = "\n".join(l for l in (EV / "probe_band_sensitivity.log").read_text().splitlines() if l.startswith("band="))

MD = f"""## ① 案例介绍（被审对象）

m17594 是 **XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE**（TensorSPH，branch `agent/m17594`，deliverable commit
**cc48564**）的开放源码交叉验证。物理：二维静止液滴的 **Young–Laplace 压差**。几何：圆滴 `R = 0.02 m`，
外包等密度蒸汽载体至 `R + 0.012 m`，六角格点 `dy = dx·√3/2`，外侧 1.6 dx 层为压力规范 Dirichlet。
参数：`sigma = 0.072 N/m`、`rho0 = 1000 kg/m³`、解析裁判 `dp = sigma(1/R1+1/R2) = sigma/R = {ours[0]['dp_theory_Pa']:.4f} Pa`；
`R/dx` 阶梯 = {', '.join('{:.2f}'.format(r['R_over_dx']) for r in ours)}；`h/dx = 1.3`（敏感带 1.2/1.5）。
物性组：静止无粘无毛细惯性，唯一控制参数是分辨率 `R/dx` 与界面宽比 `h/dx`。

## ② 怎么做的（审计流程，不是重做）

1. **取件**：读 m17594 `tensor_dev_missions.result` + `tensor_reports` id 16542 + `docs/xval_projection_laplace_oss.md` +
   工作树 `/root/tp-agent/work/m17594`，逐项核三件套（run_artifacts / ref-vs-got / repro_cmd）。
2. **基准裁判**：`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE'`
   → **0 行**（本审计受害者，见 §⑤F3）。故改用案例自己预声明的裁判：解析 `dp = sigma/R` + owner 15% 门
   （pytest 里已冻结）+ 3% 严格线。
3. **独立复算**：审计员在自己的 worktree 里**重跑**交付物的三条命令（`run.py`、`oss_ports.py`、`pytest`），
   再对"重跑 JSON vs cc48564 提交 JSON"做逐字段比对（剔除 `timing`/`figure` 两个非物理字段）。
4. **反证探针**：a) 用**出厂默认参数**调用 `laplace_isph_case()`（不传 `curvature_method`）；
   b) 改 `dp` 提取带 `out_band`（怀疑 <3% 是取带假象）。b 被证伪，a 命中真实缺陷。
5. 工具链：psql(tensorplatform) 取裁判行、python3/torch CPU、matplotlib 出审计图。审计在 loongarch64 宿主完成。

## ③ 计算结果（数字表，程序化注入）

`dp` 单位 Pa；error = dp/dp_theory − 1；裁判门 |error| ≤ 15%。

| `R/dx` | ref `sigma/R` | ours `dp`(mls) | ours err % | ours | OSS-port `dp` | OSS err % | OSS | 双路互差 % |
|---|---|---|---|---|---|---|---|---|
{ladder}

复现性：重跑 `run.py` **exit 0 / 14.5 s**；重跑 JSON 与提交 JSON 除 `timing`、`figure` 外
**逐字段完全相同**（`numeric_identity_excluding_timing_and_figure = {REPRO['numeric_identity_excluding_timing_and_figure']}`，
提交 json sha256 = `{REPRO['committed_sha256'][:32]}…`）。pytest **9 passed in 4.61 s**。

敏感性与反证探针（来自提交件 + 审计员实测）：

| probe | value | ours err % | oss err % |
|---|---|---|---|
{sens_table}

审计员实测**出厂默认** `curvature_method="fit_quadratic"`（`laplace_isph.py:263`）：

```
{default_block}
```

审计员实测取带敏感性（`R/dx=20`，`mls`）——**假说被证伪**：

```
{band_block}
```

参考码 `form="code"` 分支的最坏情形：`R/dx = {code_worst['R_over_dx']:.0f}`，`dp = {code_worst['dp_measured_Pa']:.3e} Pa`，
误差 `{code_worst['dp_error_pct']:.3e} %`（即 `1/(V_i V_j)` 量纲性错误，交付物如实报 FAIL）。

## ④ 结果图

**图 1（审计图）**：误差随分辨率阶梯 —— 蓝=被审臂（mls，全部落在 ±3% 带内），紫=PySPH 源码派生臂（系统性
正偏，随分辨率衰减），红=**出厂默认参数路径**（fit_quadratic，−23~−24.5%，跌破 15% 门）。绿带=|err|<3%，
橙带=|err|<15%。数据全部取自提交 JSON 与审计探针日志（`docs/audit_m17633/make_figure.py`）。

**图 2（交付物原图）**：m17594 提交的 fig1（场量/压力剖面 + 与解析解比较），审计逐字节核对
sha256=`dd0586d8…`，与工作树内文件一致——图确实存在且与 JSON 同源。

## ⑤ 注意点（踩坑与局限）

* **F1 高 / 出厂默认即跌门**：`src/tensorsph/laplace_isph.py:263` 默认 `curvature_method="fit_quadratic"`，
  实测 −23.01 / −23.69 / −24.46%（三档全跌 15% 门）；验证过的路径必须显式 `"mls"`（−0.21%）。交付物数字不受
  影响且 §⑤.3 已披露 fit 算子坏，但**默认路径静默给错答案**，属 API 正确性缺陷——本审计只登记不擅改。
* **F2 中 / "与 PySPH 交叉" 是源码派生重实现**：`pysph_parity_check` 9 项里 5 项机器精度重合
  （SmoothedColor 2.2e-16、normal 2.2e-16、gradC 4.8e-7、SY11 delta 2.4e-7、CSF 0.0），
  另 4 项不重合（Curvature_kappa 1.3e9、wij_sum 0.543、PressureCoeff_diag 1.75e4、delta 1/|∇C| 2.1e9）——
  恰是曲率/压力算子本身。故 **3.03% 双路互差是自洽性证据，不能读成"与 PySPH 吻合 3%"**；交付物 §⑥ 已如实列 open item。
* **F3 中 / 裁判未登记**：`tensor_benchmark_cases` 本锚 0 行 → 后续复审无机器裁判，建议登记
  `CASE-LAPLACE-SPH / dp_vs_sigma_over_R_pct / 15`。
* **F4 中 / 交付物未进 main**：`git branch -a --contains cc48564` 仅 `agent/m17594`；审计分支 `79fc925` 无该目录
  → 干净 main 检出跑不了，需合并。
* **F5 低 / 取带假象被证伪**：`out_band` 1.15–1.35R vs 1.25–1.45R → −0.199% vs −0.213%（差 0.014 pp）。
* **F6 低 / "静态平衡"用词过强**：局地动量残差中位/均值 = 0.26/0.28/0.36（ours）、0.42/0.46/0.52（oss），
  是**积分**平衡而非逐点平衡；但独立通量路线 `dp_flux` = −0.60% 与压力路线 −0.21% 一致，说明数字非假象。
* **F7 低 / 环境绑定**：PySPH 仅在遗留 `/tmp/xv2` venv（任务期 github:443 不可达）→ parity 腿换机即失。
* **F8 信息 / 报告齐全**：`tensor_reports` id 16542（14919 字符 + 1 图）、`docs/xval_projection_laplace_oss.md`
  六段齐全，且显式回答 3% 问题。

## ⑥ 误差判定（逐项 vs 基准，显式回答）

| 关键量 | 基准 | got | 误差 | <3% ? | <15% ? |
|---|---|---|---|---|---|
| dp @ R/dx=10 | 3.6000 Pa | {ours[0]['dp_measured_Pa']:.4f} Pa | {ours[0]['dp_error_pct']:+.2f}% | 是 | 是 |
| dp @ R/dx=13.3 | 3.6000 Pa | {ours[1]['dp_measured_Pa']:.4f} Pa | {ours[1]['dp_error_pct']:+.2f}% | 是 | 是 |
| dp @ R/dx=20 | 3.6000 Pa | {ours[2]['dp_measured_Pa']:.4f} Pa | {ours[2]['dp_error_pct']:+.2f}% | 是 | 是 |
| dp @ R/dx=10 (OSS 臂) | 3.6000 Pa | {oss[0]['dp_measured_Pa']:.4f} Pa | {oss[0]['dp_error_pct']:+.2f}% | 否 | 是 |
| dp @ R/dx=20 (OSS 臂) | 3.6000 Pa | {oss[2]['dp_measured_Pa']:.4f} Pa | {oss[2]['dp_error_pct']:+.2f}% | 是 | 是 |
| dp @ R/dx=20 (**出厂默认** fit_quadratic) | 3.6000 Pa | 见 §③ 探针 | −24.46% | **否** | **否** |
| dp_flux（独立通量路线）@ R/dx=20 | 3.6000 Pa | {ours[2]['dp_flux_Pa']:.4f} Pa | {(ours[2]['dp_flux_Pa']/ours[2]['dp_theory_Pa']-1)*100:+.2f}% | 是 | 是 |
| kappa R @ R/dx=20 | 1 | {ours[2]['kappaR_delta_weighted']:.4f} | {ours[2]['kappaR_delta_weighted']-1:+.2%} | 是 | 是 |

**回答：是——被审臂（`mls`）三档全部 <3%（+0.97 / +0.61 / −0.21%），解析裁判为 σ/R，独立于被测代码；
OSS 派生臂仅最细档 <3%（+2.83%），粗档 +4.00% 如实不粉饰。但"出厂默认参数"路径 −24.46%，不满足判据**——
这就是本审计判 level 4（被审臂基准验证成立）同时要求返修 F1（默认参数）的理由。

## 判级与裁决

**Level = 4**（L1 数学/单元验证 → L2 真实现（审计员实跑：格点→CSF→泊松→Δp）→ L4 完整算例+三档误差分析+
6 项敏感性+比特级可复现；L3 仅达 harness/library 级（无时间推进求解器调用方，`relax_projection` 仍发散），
L5 不主张）。**Disposition：approve with notes / level 4 confirmed**；F1、F3、F4 作为返修项登记，
F5/F6 属文档级，F7 属镜像钉版。verdict 已落表 `tensor_maturity_verdicts` id 608（gate_state 待机器裁定）。
"""


def b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode()


payload = {
    "title": "m17633 交差审计：m17594 XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE（level 4 / approve with notes）",
    "summary": ("独立复算 m17594：重跑 run.py 14.5 s、结果与 cc48564 提交 JSON 逐字段相同（仅 timing 变），"
                "pytest 9 passed；解析裁判 dp=sigma/R，被审臂 +0.97/+0.61/-0.21%（三档全<3%）。"
                "反证探针命中真缺陷：出厂默认 curvature_method=fit_quadratic 使 dp 误差 -23.0/-23.7/-24.5%（跌 15% 门）；"
                "取带敏感性假说被证伪（-0.199% vs -0.213%）。判 level 4，返修 F1/F3/F4。"),
    "scenario_key": "board-audit",
    "mission_id": 17633,
    "tags": ["board-audit", "XVAL", "TensorSPH", "projection-SPH", "Young-Laplace", "m17594", "level-4"],
    "markdown": MD,
    "figures": [
        {"name": "AUDIT_m17594_error_ladder.png",
         "caption": "m17633 审计图：dp 误差 vs 解析 σ/R 随分辨率阶梯（蓝=被审 mls 臂，紫=PySPH 源码派生臂，红=出厂默认 fit_quadratic 路径）；绿带 <3%，橙带 <15%",
         "png_base64": b64(HERE / "AUDIT_m17594_error_ladder.png")},
        {"name": "fig1_as_shipped_cc48564.png",
         "caption": "m17594 交付物原图（cc48564 提交，sha256=dd0586d8…，与工作树文件一致）",
         "png_base64": b64(EV / "fig1_as_shipped_cc48564.png")},
    ],
}

r = post("/tensor/reports", payload, tok)
print("report:", json.dumps(r)[:400])