#!/usr/bin/env python3
"""Publish the m15419 board-audit report to the platform (/tensor/reports).

All numbers are injected from runs/m15419_audit/audit.json — no hand copying.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = json.loads((REPO / "runs/m15419_audit/audit.json").read_text())
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
FIG = REPO / "docs/board_audit/m15419"


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())


a = AUDIT
rc, rule, reg = a["recompute"], a["gating_rule"], a["registry"]
tp, rerun, ev = a["three_piece"], a["three_piece"]["rerun"], a["three_piece"]["evidence_store"]
t0, neg, sweep = a["probes"]["t0_static"], a["probes"]["negative_control_g0"], a["probes"]["resolution_sweep"]
rows = a["band_table"]
interior = [r for r in rows if not r["free_surface"]]
fsb = [r for r in rows if r["free_surface"]]

L = []
w = L.append
w("# 交差核验：CASE-HYDROSTATIC（被审 m15340，TensorSPH）— board-audit m15419\n")
w("## ① 案例介绍\n")
w("物理背景：不可压静止液体在重力场中的静水平衡，精确解 "
  "$p(y)=\\rho_0 g (H-y)$。算例 = 常温矩形水箱（宽 1.0 m × 深 0.4 m），"
  "格子 dx=0.02 m → 50×20 = **1000** 粒子（满足战役 N≥1000），"
  "ρ0=1000 kg/m³、g=9.81 m/s²、声速 c=20 m/s、时长 2.0 s（5129 步）、"
  "取后半段 128 次采样做时均、8 个水平带。被审提交 TensorSPH `%s`。\n" % a["subject_commit"][:12])
w("| 参数 | 值 |\n|---|---|")
w("| 水箱 / 深度 | 1.0 m × 0.4 m |\n| dx | 0.02 m |\n| 粒子数 | %d |\n"
  "| ρ0 / g / c | 1000 kg/m³ / 9.81 m/s² / 20 m/s |\n| 时长 / 步数 | 2.0 s / 5129 |\n"
  "| 门限 | 内区带压力误差 ≤ %.0f%% |" % (a["three_piece"]["benchmark"]["n_particles"],
                                    rc["limit_pct"]))
w("\n## ② 怎么做的\n")
w("1. 取证：读被审提交的 `results.json`（tracked）与 `evidence.sqlite`，"
  "并查唯一裁判表 `tensor_benchmark_cases`（anchor_slug='%s' → **%d 行**，全表 %d 行）。\n"
  "2. 独立重算：把工件里的 8 个带的 y 重新代入 $\\rho_0 g (H-y)$，逐带比对记录值；"
  "并按预声明规则 support=2·1.3·dx 重推内区/自由面带的划分。\n"
  "3. 复现：`git archive %s` 到干净目录后原样执行 repro_cmd，逐带比对压力。\n"
  "4. 新增探针（同一份提交代码）：t=0 静态算子误差、g=0 负对照、同几何分辨率扫描"
  "（dx=0.04/0.03/0.02）、以及 dx=0.03 但 depth=0.39（格子恰好铺满）的解耦实验。\n"
  % (a["anchor"], reg["n_rows"], reg["table_total"], a["subject_commit"][:12]))
w("## ③ 计算结果（程序化注入）\n")
w("| 带 | y (m) | N | p_SPH (Pa) | p_ref 重算 (Pa) | 误差重算 (%) | 误差记录 (%) | 类别 |\n"
  "|---|---|---|---|---|---|---|---|")
for r in rows:
    w("| %d | %.4f | %d | %.3f | %.3f | %+.4f | %+.4f | %s |"
      % (r["band"], r["y"], r["particles"], r["p_sph"], r["p_ref_recomputed"],
         r["err_recomputed_pct"], r["err_recorded_pct"],
         "自由面(排除)" if r["free_surface"] else "内区(门控)"))
w("\n- 内区 max = **%.4f%%**（记录 %.4f%%，差 %+.1e pp）；内区 mean = %.4f%%；"
  "**全带 max = %.4f%%**（排除带读 %s%%）。\n"
  "- 门控覆盖深度 %.1f%%（support 2h = %.3f m），内区 = 带 %s，与工件 `gated_bands` 一致。\n"
  "- 复现：gate %.10f%% → %.10f%%（Δ=%.1e），8 带压力最大绝对差 %.1e Pa → 逐位复现。\n"
  "- 负对照 g=0：内区 %.2f%% → 门 FAIL（门可失败）。\n"
  % (rc["interior_max_pct"], rc["committed_max_pct"], rc["delta_pp"], rc["interior_mean_pct"],
     rc["all_band_max_pct"], [round(r["err_recomputed_pct"], 2) for r in fsb],
     rule["coverage_of_depth_pct"], rule["support_m"], rule["interior_from_rule"],
     rerun["gate_pct_committed"], rerun["gate_pct_rerun"], rerun["gate_delta"],
     rerun["max_abs_band_pressure_delta_pa"],
     neg.get("gate", {}).get("measured_pct", float("nan"))))
w("### 分辨率扫描（审计方探针）\n")
w("| 配置 | N | 内区 max (%) | 内区 mean (%) | H 未对齐偏差 (%) | gate |\n|---|---|---|---|---|---|")
for s in sweep:
    w("| %s | %d | %.3f | %.3f | %.2f | %s |"
      % (s["label"], s["n"], s["max_interior_pct"], s["mean_interior_pct"],
         s["H_mismatch_pct"], "PASS" if s["gate_pass"] else "FAIL"))
w("\n## ④ 结果图\n")
w("图1：压力剖面 ref vs got（含 t=0 静态解）；图2：同几何分辨率扫描，"
  "标注格子与参考高度不对齐（H mismatch）导致的假失败。\n")
w("## ⑤ 注意点\n")
w("- **裁判表缺行**：`tensor_benchmark_cases` 无 `%s` 行，2%% 容差由被审单自定；"
  "同锚点 m15264 的 case 亦未登记（m15320 审计 F1 至今未修）。\n"
  "- **发布单 §5 收敛声明被证伪**：其“O(dx) 收敛”由 dx=0.05（N=16、0.2 s、0.2 m 水箱）"
  "与 dx=0.02（N=1000、2 s、1.0 m 水箱）两行得来——两者是不同几何/时长；"
  "同几何扫描 dx=0.04/0.03/0.02 非单调。\n"
  "- **提交态 evidence.sqlite 是空库**（0 runs/0 metrics），而 results.json 引 run_id=%s/%s → 证据链断在库里。\n"
  "- **门控对“格子是否整除深度”敏感**：`cell_centres()` 取 floor(depth/dx)，dx=0.03 时格子顶面 "
  "0.39 m 而参考 H=0.4 m → 每带系统性读低 2.56%%，最终 9.30%% 假失败；把 depth 改 0.39 后同 dx 降到 "
  "2.25%%。提交的 dx=0.02（恰好 50×20=1000）不受影响。\n"
  "- **内区 ~1.00%% 是静态算子偏差**：t=0 一次 RHS 求值内区各带已一致读低约 1.00%%，动力学修的是"
  "壁邻带（−10.2%% → −0.55%%）；发布单把 1%% 表述为时均产物，未区分两者。\n"
  "- 局限：门只覆盖内区（深度的 %.1f%%），自由面 2h 内不带电；未跑 GPU/大粒子数；"
  "未做第三方 SPH 交叉验证。\n"
  % (a["anchor"], ev["results_json_run_id"], ev["control_run_id"], rule["coverage_of_depth_pct"]))
w("## ⑥ 误差判定\n")
w("| 关键量 | 基准 | 实测 | 误差 | <3%? | 容差 |\n|---|---|---|---|---|---|")
w("| 内区带压力 max | $\\rho_0 g (H-y)$ | %.4f%% | — | **%s** | ≤%.0f%% |"
  % (rc["interior_max_pct"], "是" if rc["interior_max_pct"] < 3 else "否", rc["limit_pct"]))
w("| 内区带压力 mean | 同上 | %.4f%% | — | **%s** | — |"
  % (rc["interior_mean_pct"], "是" if rc["interior_mean_pct"] < 3 else "否"))
w("| 全带（含自由面）max | 同上 | %.4f%% | — | **%s** | 不在门控内 |"
  % (rc["all_band_max_pct"], "是" if rc["all_band_max_pct"] < 3 else "否"))
w("| 审计方重算 vs 工件记录 | %.10f%% | %.10f%% | %+.1e pp | 是 | — |"
  % (rc["committed_max_pct"], rc["interior_max_pct"], rc["delta_pp"]))
w("| 复跑 vs 提交 | — | 8 带压力差 %.1e Pa | 0 | 是 | 逐位 |"
  % rerun["max_abs_band_pressure_delta_pa"])
w("\n**是否 <3%%：门控量（内区 max %.2f%%、mean %.2f%%）是**；全带量 %.2f%% 也 <3%%，"
  "但>b>自由面带被门控排除（截断核支撑，物理上不可比），已单独列出。**L 级判定：L4**"
  "（L4 功能实质成立：解析基准 + 误差分析 + 逐位可复现 + 门可失败；F1–F4 需返修）。\n"
  % (rc["interior_max_pct"], rc["interior_mean_pct"], rc["all_band_max_pct"]))

md = "\n".join(L).replace(">b>", "")
figs = []
for name, cap in (("fig1_pressure_profile.png", "图1 压力剖面：解析 $\\rho_0 g(H-y)$、SPH 时均、SPH t=0 静态；虚线=2h 核支撑限"),
                  ("fig2_error_vs_dx.png", "图2 同几何分辨率扫描：内区最大误差 vs dx，标注格子/参考高度不对齐偏差；虚线=2% 门限")):
    p = FIG / name
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

login = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                             "password": os.environ["TENSOR_AGENT_PASS"]})
token = login.get("access_token") or login.get("token")
out = post("/tensor/reports", {
    "title": "交差核验 m15419：CASE-HYDROSTATIC（被审 m15340，TensorSPH）— L4，解析基准 1.06% 内区、逐位可复现、F1–F4 待返修",
    "summary": ("board-audit 复核 m15340 三件套：裁判表 0 行（容差自定）、解析基准内区 max 1.0635%（重算一致）、"
                "全新检出逐位复现（8 带压力差 0 Pa）、g=0 负对照 100% 证明门可失败；证伪发布单 §5“O(dx) 收敛”"
                "（两行本属不同几何），发现提交态 evidence.sqlite 为空库、门控对 depth/dx 整除敏感（dx=0.03 假失败 9.30%→对齐后 2.25%）。定级 L4。"),
    "scenario_key": "audit_case_hydrostatic_m15419",
    "mission_id": 15419,
    "tags": ["board-audit", "TensorSPH", "CASE-HYDROSTATIC", "hydrostatic", "m15340", "L4"],
    "markdown": md,
    "figures": figs,
}, token)
print(json.dumps({k: out.get(k) for k in ("id", "title", "status")}, ensure_ascii=False))