"""Publish the m2502 report with numbers injected from the archived artifact."""
import base64
import io
import json
import math
import os
import urllib.request
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m2502")
art = json.loads((ROOT / "benchmarks/dam_break/resolution_ladder.json").read_text())
rungs = sorted(art["results"], key=lambda r: -r["dx"])

rows = ["| `dx` (m) | 粒子数 | 步数 | `dt` (s) | 墙钟 (s) | exp MAE (`Z/a`) | RMSE | max abs err | 点数 | 落入不确定带 | Mach | 最大密度压缩 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rungs:
    e = r["experimental_error"]
    rows.append(f"| {r['dx']:g} | {r['particles']} | {r['steps_taken']} | {r['dt']:.3e} | "
                f"{r['wall_clock_s']:.1f} | {e['mean_absolute_error']:.4f} | {e['rmse']:.4f} | "
                f"{e['max_absolute_error']:.4f} | {e['points_compared']} | "
                f"{100 * e['within_uncertainty']:.0f} % | {r['mach_number']:.3f} | "
                f"{100 * r['max_density_compression']:.2f} % |")

order_rows = ["| 相邻两级 | 观测阶 `p`（MAE ~ dx^p） |", "|---|---|"]
for a, b in zip(rungs, rungs[1:]):
    p = math.log(a["experimental_error"]["mean_absolute_error"]
                 / b["experimental_error"]["mean_absolute_error"]) / math.log(a["dx"] / b["dx"])
    order_rows.append(f"| {a['dx']:g} -> {b['dx']:g} m | {p:+.4f} |")
ps = [math.log(a["experimental_error"]["mean_absolute_error"]
               / b["experimental_error"]["mean_absolute_error"]) / math.log(a["dx"] / b["dx"])
      for a, b in zip(rungs, rungs[1:])]
mean_p = sum(ps) / len(ps)

fine = rungs[-1]
mae_f = fine["experimental_error"]["mean_absolute_error"]
gate_rel = mae_f / 0.15

png = (ROOT / "benchmarks/dam_break/resolution_ladder.png").read_bytes()
figures = []
try:
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    if im.width > 1200:
        im = im.resize((1200, int(im.height * 1200 / im.width)))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG", optimize=True)
    png = buf.getvalue()
except Exception as exc:  # keep the original if PIL is unavailable
    print("PIL downscale skipped:", exc)

if len(png) > 200_000:
    print("figure too large, dropping:", len(png))
else:
    figures.append({
        "name": "dam_break_resolution_ladder.png",
        "caption": "B2 溃坝前缘轨迹：三级分辨率模拟 vs Martin & Moyce (1952) 实验点 vs Ritter 浅水解（Z/a–tau）。三条曲线几乎重合而实验点系统性超前 —— 残差是偏置而非离散误差。",
        "png_base64": base64.b64encode(png).decode(),
    })

markdown = f"""## ① 案例介绍

**物理背景**：B2 溃坝（dam-break）自由面流。有限水库水柱瞬间释放，前缘沿干床推进，
是 SPH/WCSPH 的标准物理验证案例，也是本仓 `physical_validation` 状态长期悬空的一项。

**实验基准**：Martin & Moyce (1952), *Phil. Trans. R. Soc. A* **244**(882), 312-324,
DOI 10.1098/rsta.1952.0006。仓内已捆绑数字化数据
`benchmarks/dam_break/martin_moyce_1952.json`（20 点，±0.05 in Z/a）。
无量纲化按原文约定：`tau = t·sqrt(2g/a)`，`Z/a = (x_max - a)/a`，`a = 0.2 m`。

**几何与参数**（三级完全相同，只改 `dx`）：

| 量 | 值 |
|---|---|
| 水柱宽 `a` / 高 | 0.2 m / 0.4 m（2:1，与实验同纵横比） |
| 水槽 | 2.0 m 宽，静态壁粒子，自由滑移（`wall_friction = 0`） |
| 水 | `rho0 = 1000 kg/m³`，`c0 = 25 m/s`，`gamma = 7` |
| 密度扩散 | 0（关闭），非自适应 dt，Courant 0.3 |
| 模拟物理时长 | 0.5 s（覆盖全部 20 个实验点，tau ≤ 4.29） |

## ② 怎么做

问题：`docs/benchmarks/dam-break-experimental.md` 只在 **单一粗分辨率** (`dx` = 0.04 m,
50 个流体粒子) 上跑过实验对比，得到 MAE 0.60，随后**外推**预测"按 `dx^1.11` 收敛，
`dx` = 0.01 时 MAE 应降到 0.15–0.20，落在实验不确定带内"。这个预测从未被测量。

做法（最小改动、只增不改）：

1. 用现有 `benchmarks/dam_break/experimental_validation.py` 跑三级阶梯
   `--sizes 0.04,0.02,0.01 --seconds 0.5`，其余参数完全固定 → `benchmarks/dam_break/resolution_ladder.json`；
2. 新增 `benchmarks/dam_break/make_ladder_doc.py`，把 JSON **程序化注入**生成
   `docs/benchmarks/dam-break-resolution-ladder.md`（零手抄，`--check` 可验新鲜度）；
3. 新增 `tests/test_dam_break_resolution_ladder.py`（7 项）钉住每级 MAE、"误差不收敛"这一**结论本身**、
   以及"文档 == 渲染结果"的一致性 —— 将来若有人修好偏置，测试会红并强制改写验证记录；
4. 在 `docs/benchmarks/dam-break-experimental.md` 增补"Correction (measured)"节，
   明确标注原外推预测已被证伪。

**工具链说明（诚实披露）**：本分支 `src/tensorsph/__init__.py` 导出了并不存在的
`apply_contact_angle_boundary`，`import tensorsph` 直接 ImportError（7 个测试模块在收集期即失败）。
该回归的修复正在飞行中（missions 2454/2466/2470 + report 2482）。
为不重复在飞工作、又不至于让本单无法测量，benchmark 用一次性 shim（`.scratch/run_ev.py`，
未入库）只给该**未使用**入口打桩（`contact_angle=None`），数值路径零改动。
`tests/test_dam_break_resolution_ladder.py` 只读归档 JSON、不 import 包，因此对该回归免疫。

## ③ 计算结果

{chr(10).join(rows)}

观测阶（对数斜率，误差应随 `dx → 0` 下降）：

{chr(10).join(order_rows)}

三级平均观测阶 **{mean_p:+.4f}**（若按文档假设的 1.11，`dx` 减半 2.02 倍时 MAE 应减半）。

## ④ 结果图

见内嵌图 `dam_break_resolution_ladder.png`：三条模拟曲线（`dx` = 0.04/0.02/0.01）几乎完全重合，
而实验点在整个 `tau` 区间系统性偏上（模拟前缘落后），Ritter 浅水解在最上方（半无限水库、无摩阻，
符合理论预期）。图本身即"残差不是离散误差"的直接证据。

## ⑤ 注意点 / 局限

1. **这是 2-D 截面模型 vs 3-D 有限宽渠道实验**：自由滑移侧壁、无侧壁拖曳，前缘速度一阶不同 →
   该项误差不随 `dx` 消失。最可能是偏置主因（尚未通过测量分离）。
2. **后墙对称面/瞬时开闸**：2-D 水柱靠壁释放 vs 实验瞬时抽闸，释放瞬态不同。
3. **前缘定义**：取 `x_max`（最前粒子），SPH 前沿偏后约一个 `h` 的分数，该项是分辨率的弱函数。
4. **数字化不确定度**（±0.05 in Z/a）比实测误差小一个量级，不足以解释 0.57。
5. **RMSE 随加密反而上升**（0.6766 → 0.6817 → 0.6994）：说明细网格把实验点未分辨的前沿振荡
   展现出来、而系统偏置不变，这进一步排除"加密即可"的说法。**
6. 墙钟在共享机器上测得（load15 ≈ 240–400，nproc 128），**不是性能测量**，只作可复现性记录。

## ⑥ 误差判定（是否 < 3% / < 15%）

平台标准：关键量与实验/参考值误差 >15% 不可接受。

| 关键量 | 基准 | 实测（最细 `dx` = 0.01） | 相对偏差 | <15%? | <3%? |
|---|---|---|---|---|---|
| 前缘轨迹 MAE | Martin & Moyce (1952) `Z/a` | {mae_f:.4f}（无量纲） | 相对 15% 判据 ≈ **{gate_rel:.1f} 倍** | **否（FAIL）** | 否 |
| 前缘轨迹 RMSE | 同上 | {fine['experimental_error']['rmse']:.4f} | 同上量级 | 否 | 否 |
| 最大绝对偏差 | 同上 | {fine['experimental_error']['max_absolute_error']:.4f} | — | 否 | 否 |
| 落入实验不确定带的点比例 | 100 % 为合格 | {100 * fine['experimental_error']['within_uncertainty']:.0f} % | — | 否 | 否 |

**裁决：B2 的 `physical_validation` 判 FAIL，且不是"分辨率不够"，而是结构性偏置。**

- 数值自洽面全过：质量守恒（步内 0 漂移）、粒子逃逸 0、Mach {fine['mach_number']:.3f} < 0.25、
  最大密度压缩 {100 * fine['max_density_compression']:.2f}% < 10% —— 即 *software_execution = success,
  numerical_verification = pass, physical_validation = fail*，三态必须分开报，不得粉饰。
- 原文档"refinement should bring the error inside the band"的表述已被本测量证伪，已就地更正。

## 给下次自找活的线索

1. **分离偏置来源**（按性价比排序）：(a) 关掉后墙、用无限水槽/周期域重跑，看好多少；
   (b) 加侧壁摩阻 `wall_friction`/Manning 型底摩阻扫描，比较前缘推进；
   (c) 把前缘判据从 `x_max` 改为体积分数等值线（如 5% 液高）再比较 —— 三项都是廉价跑（`dx` = 0.04 一级仅 21 s）。
2. 该仓 `import tensorsph` 在 main 上仍断（`apply_contact_angle_boundary` 缺失，10+ 分支各自"恢复"过同一函数却都未合入）：
   **真正缺的是 API 契约测试**（`__all__` 每个名字必须可解析）。若这条线仍无人收口，下一单可直接加
   `tests/test_public_api.py`，一次性终结这类复发。
3. 归档 JSON + 生成式文档 + "钉住结论本身"的测试是本仓最省预算的验证范式，建议沿用。
"""

report = {
    "title": "TensorSPH B2 溃坝实验验证欠账：Martin & Moyce 分辨率阶梯实测——加密不收敛（MAE 0.599→0.573），"
             "偏置非离散误差，15% 门禁判 FAIL",
    "summary": ("对 TensorSPH 只在单点粗网格上做过的 B2 溃坝 vs Martin & Moyce (1952) 实验对比补齐三级分辨率阶梯"
                f"（dx=0.04/0.02/0.01，50/200/800 粒子）：MAE 0.5988/0.5774/0.5728，观测阶 {mean_p:+.3f}"
                "（文档外推假设 1.11），RMSE 随加密反升（0.6766→0.6994），落入实验不确定带点数 5%→5%→0%。"
                "结论：残差是分辨率无关的结构性偏置（2-D 截面 vs 3-D 渠道实验为主要嫌疑），"
                "原文档『加密即可落入实验带』的外推预测被证伪，B2 physical_validation 判 FAIL；"
                "数值自洽面（质量守恒/无逃逸/Mach 0.13/压缩 1.9%）全过。"),
    "scenario_key": "tensorsph-dam-break-experimental-resolution-ladder",
    "mission_id": 2502,
    "tags": ["TensorSPH", "dam-break", "Martin-Moyce", "validation-debt", "convergence", "honest-FAIL"],
    "markdown": markdown,
    "figures": figures,
}

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = urllib.request.Request(base + "/auth/login",
                               data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                                "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
tok = json.loads(urllib.request.urlopen(login, timeout=30).read())
token = tok.get("access_token") or tok.get("token")
print("payload KB:", len(json.dumps(report)) // 1024)
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(report).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + token}, method="POST")
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    print("REPORT ID:", resp.get("id"), "| keys:", list(resp)[:8])
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read()[:500])