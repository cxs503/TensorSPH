"""Publish the m24140 closure report to the tensor platform (no secrets in argv)."""
import base64
import json
import os
import sys
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = "/root/tp-agent/work/m24140"


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={**{"Content-Type": "application/json"},
            **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def fig(path, caption):
    b64 = base64.b64encode(open(os.path.join(ROOT, path), "rb").read()).decode()
    return {"name": os.path.basename(path), "caption": caption, "png_base64": b64}


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token") or login.get("data", {}).get("token")
assert token, "login failed: %s" % list(login.keys())

# numbers injected programmatically from the committed ledger (zero hand-copying)
sys.path.insert(0, ROOT + "/benchmarks/case_sph_wedge_entry_2d")
sys.path.insert(0, ROOT + "/src")
import footprint_ledger as fl  # noqa: E402

ART = ROOT + "/docs/artifacts/case_sph_wedge_entry_2d_footprint"
rep = json.load(open(ART + "/footprint_ledger.json"))
fresh = fl.build(ART)
g2 = next(g for g in fresh["grids"] if g["tag"] == "dx2mm")
g4 = next(g for g in fresh["grids"] if g["tag"] == "dx4mm")
dr = fresh["designated_readout"]
v = fresh["verdict"]

md = f"""# TensorSPH#1020 重开·管到底第1轮：m21677 footprint 交付物再交付闭合（m24140）

## ① 案例介绍

**物理背景**：2-D 楔形入水（wedge entry）是船舶砰击/上浪冲击的经典理想化问题。
楔角 β=30°、入水速度 V=1 m/s，SPH 求解器在 dx = 4/2/1/0.5 mm 四级网格上各跑
61 帧系综（c0 因子 20，CFL 0.35，α_visc=0.03）。

**被评测量**（m21677 预先声明）：`Cp_Φ = max over sliding windows of the mean
p/q inside a footprint of fixed physical width Φ = 8 mm` —— 即压力传感器实际
报告的量（有限面积平均），而非不可测的点最大值。

**参数表**：

| 参数 | 值 |
|---|---|
| β | 30° |
| V | 1 m/s |
| Φ（传感器足迹） | 8 mm（最粗网格 2 格，最细 16 格） |
| 网格阶梯 | dx = 4 / 2 / 1 / 0.5 mm（已提交工件：dx4mm、dx2mm） |
| 每格系综帧数 | 61 |
| 参考解 | Wagner 解析场（同一 footprint 算子、同一网格） |
| 验收门 | 15%（案例预先声明；平台 3% 标准另行显式回答） |

## ② 怎么做的

原 PR#1020（分支 agent/m21677，13 个文件）于 2026-11-05 被 owner 裁定关闭；
其内容经 #821/m21677、m21953、m22695 等合并已 100% 进入 main（7 个文件逐字节
相同，5 个被后续任务严格扩展）。本单是"管到底"闭合：

1. `make_closure_doc.py` 断言 **13/13** 原 PR#1020 交付物在当前 main 存在；
2. 用 `footprint_ledger.build()` 从原始系综工件**重算** ledger，与已提交的
   `footprint_ledger.json` 逐项核对（verdict + 每格 obs/synth/residual/judgeable）；
3. 程序化生成闭合文档与双联图（零手抄）；4 个新文件、0 个热文件改动
   （未触碰 TensorSPH 核心模块，无与其他开 PR 的路径重叠）。

工具链：pytest（本地短测试）+ matplotlib（Agg）+ git。

## ③ 计算结果（从已提交工件程序化注入）

| 网格 | dx [mm] | Φ [格] | obs Cp | synth Cp | U(bootstrap) | 残差 | 可判 |
|---|---|---|---|---|---|---|---|
| dx4mm | 4 | 2 | {g4['obs_cp']:.4f} | {g4['synth_cp']:.4f} | {g4['uncertainty']:.1%} | {g4['residual']:+.1%} | 否 |
| dx2mm | 2 | 4 | {g2['obs_cp']:.4f} | {g2['synth_cp']:.4f} | {g2['uncertainty']:.1%} | {g2['residual']:+.1%} | 是 |

* **主判级**：**{v['verdict']}** —— {v['reason']}。
* **次级口径一致读数**（m21953，非判级）：可用网格 {', '.join(dr['usable_grids'])}，
  残差 {', '.join(f'{k} {x:+.1%}' for k, x in dr['residuals'].items())}，
  在 15% 门内：**{dr['inside_gate']}**；夹逼目标：**{dr['brackets_target']}**（{dr['note']}）。
* 缺失网格（无已提交工件）：{', '.join(fresh['missing_grids'])}。

## ④ 结果图

图 1（本单生成）：左 = Wagner 解析 p/q 场云图（β=30°，白虚线=峰值轨迹，
红带 = t_end 处 Φ=8mm 足迹窗）；右 = footprint 阶梯比较曲线
（SPH ±bootstrap vs Wagner 同算子 vs Wagner 点最大 / Zhao&Faltinsen COM /
精确相似解三条参考线）。图 2 = m21677 已提交基线图（footprint 阶梯读数 vs 网格）。

## ⑤ 注意点

* 本单是**验证型**交付：不改物理、不改求解器、不改判级规则；PR 只含 4 个新文件。
* dx1mm / dx0.5mm 系综工件未提交，阶梯止于 dx2mm，主判级无法配对"两个最细
  可判网格"，故 NOT_DECIDABLE —— 这是数据缺口，不是算法缺陷。
* dx4mm 网格 U=30.7% > 15%：2 格/足迹低于该算子的分辨率下限，读数被噪声主导。
* 仓库存在 3 个与本案无关的既有收集错误（`import run` 模块名冲突：
  test_tvc_dissipation_attribution / test_wedge_compressibility_study /
  test_xval_tvc_cadence_invariance）与 1 个既有测试污染失败
  （test_case_sph_wedge_entry_2d_literature_regrade::test_driver_refs_derived...），
  均已在干净 main 上复现确认，非本单引入。
* 适用边界：闭合脚本对 ledger 结构敏感（verdict/grids 字段），若后续任务改动
  footprint_ledger.py 的输出 schema，需同步更新 make_closure_doc.py 与测试。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 值 | 基准 | 残差 | <3%? |
|---|---|---|---|---|
| dx2mm footprint Cp vs Wagner 同算子 | {g2['obs_cp']:.4f} | {g2['synth_cp']:.4f} | {g2['residual']:+.1%} | **否** |
| dx2mm footprint Cp vs 指定基准（COM×footprint） | {g2['obs_cp']:.4f} | {dr['designated_point_cp']:.4f} | {dr['residuals']['dx2mm']:+.1%} | **否** |
| dx4mm footprint Cp vs Wagner 同算子 | {g4['obs_cp']:.4f} | {g4['synth_cp']:.4f} | {g4['residual']:+.1%} | **否**（该格不可判，U={g4['uncertainty']:.1%}） |

**结论：无关键量在 3% 以内，不声称 3%。** 案例预先声明的验收门是 15%（非 3%）：
按该门，dx2mm 次级口径一致读数在门内（{dr['residuals']['dx2mm']:+.1%}）；
主判级 NOT_DECIDABLE（仅 1 个可判网格，无法配对两个最细可判网格）。
3% 平台标准未达成，不粉饰、不声称。
"""

payload = {
    "title": "TensorSPH#1020 重开·管到底第1轮：m21677 footprint 交付物再交付闭合（13/13 在 main，ledger 可重算，判级 NOT_DECIDABLE 如实入账）",
    "summary": "原 PR#1020 已裁定关闭；本单程序化验证其 13 个交付物 100% 已在当前 main（7 个逐字节相同、5 个被后续任务严格扩展），footprint ledger 从原始系综工件重算与已提交版本一致，生成闭合文档+双联图+4 个钉住测试；主判级 NOT_DECIDABLE、无关键量 <3%，如实入账。",
    "scenario_key": "case_sph_wedge_entry_2d_footprint",
    "mission_id": 24140,
    "tags": ["TensorSPH", "wedge-entry", "footprint", "closure", "m21677", "re-delivery"],
    "markdown": md,
    "figures": [
        fig("docs/artifacts/case_sph_wedge_entry_2d_footprint/figures/fig_closure_m24140.png",
           "图1（本单生成）：左=Wagner 解析 p/q 场云图（β=30°，白虚线=峰值轨迹，红带=t_end 处 Φ=8mm 足迹窗）；右=footprint 阶梯比较曲线（SPH ±bootstrap vs Wagner 同算子 vs 三条参考线）"),
        fig("docs/artifacts/case_sph_wedge_entry_2d_footprint/figures/fig1_footprint_ladder.png",
           "图2（m21677 已提交基线图）：footprint 阶梯读数 vs 网格"),
    ],
}
out = post("/tensor/reports", payload, token)
print("report posted:", json.dumps(out, ensure_ascii=False)[:300])
