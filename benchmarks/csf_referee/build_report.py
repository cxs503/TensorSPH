"""Build /tmp/report.json for mission 2495 from the referee artefacts."""
import base64
import json
import pathlib

png = pathlib.Path("benchmarks/csf_referee/csf_referee.png").read_bytes()
summary = json.loads(pathlib.Path("benchmarks/csf_referee/summary.json").read_text())
tab = "\n".join(
    f"| {r['label']} | {r['kappa_err_min_abs']:+.1f}% | {r['cut_ratio_dx002']:+.3f} | "
    f"{r['dp_pressure_pct']:+.1f}% | {r['dp_timeavg_pct']:+.1f}% | {r['dp_force_pct']:+.1f}% "
    f"| {r['spurious']:.3f} |" for r in summary)

body = f"""# TensorSPH 表面张力 CSF 修复候选的独立裁判：七个分支，同一把尺子

## 1 案例介绍（物理背景/几何/参数）

TensorSPH 的 CSF 表面张力自 m277 合入后，唯一可量化的物理判据是**二维静止液滴的
Young–Laplace 定律** `Δp = σ/R`：表面张力若幅值或方向错误，液滴内过压立刻偏离解析值。
m2174（2026-11-03）判其 FAIL（Δp −93.6%）并列为 P0；此后**七个未合并分支**
（m2389/m2399/m2420/m2433/m2447/m2451/m2454/m2459）各自声称在修，其中 m2447、m2454
声称已把偏差做到 12% 以内（"门禁通过"）。本报告不新增第八个修复，而是**给这些候选造一把
共用的尺子并给出裁决**：同一算例、同一估计量，谁真修好了、谁只是换了口径。

| 参数 | 值 |
|---|---|
| 液滴半径 R | 0.02 m（六方密排，四周真空，无重力） |
| σ / ρ0 / c0 | 7.2 N/m / 1000 kg·m⁻³ / 20 m·s⁻¹（Tait，γ=7） |
| 粘性 ν | 0.01 m²·s⁻¹ |
| 分辨率 | dx = 3.0 / 2.0 / 1.5 mm（N = 151…585，h = 1.3dx） |
| 时间推进 | 900 步，t = 1.05 t_σ（t_σ = √(ρR³/σ) = 33.3 ms） |
| 解析参考 | Δp = σ/R = 360 Pa；κ = 1/R = 50 m⁻¹；F_cut = −2σ = −14.4 N·m⁻¹ |

## 2 怎么做的（方法/流程/工具链）

**裁判不修改任何 `src/` 文件**——这是本单与七个候选的根本区别。

1. **影子包**：把当前 `src/tensorsph` 复制到 `/tmp/referee/<候选>/`，再用
   `git show agent/<候选>:<file>` 只覆盖该分支改过的 `src/tensorsph/*.py`。
   于是"如果这个分支的 src 改动被合并，用户会得到什么"可以被直接测量，而无需合并。
2. **单一测量工具** `benchmarks/csf_referee/score.py`：只调用公开 API
   （`Config / cell_linked_pairs / density_from_neighbor_pairs / tait_pressure /
   stable_dt / rhs / compute_surface_tension_force`），候选**无法用自己 benchmark 的字段自证**。
3. **三个互不依赖的判据**：
   - **A 切割力恒等式** `F_cut = Σ_{{x>0}} m a_x`，精确值 `−2σκR`，比值即 `κ/(1/R)`（前提 ∫δ_s dn = 1）。
   - **B1 力积分压差** `Δp = Σ ρ(−a·n̂)V/(2πr_c)` —— **候选报告用的就是它**；解析上等于 `σκ`，
     本质是把力再测一遍。
   - **B2 压强场压差** `p(r<0.5R) − p(0.8R<r<0.95R)`（Tait EOS + 求和密度）——这才是
     "求解器是否真的维持了律所要求的过压"；另加 **B2′ 时间平均**（末 25% 步长）以区分偏差与噪声。
   - **C 伪流** `max|v|/√(σ/ρR)`（Adami et al. 2012，判据 < 0.1）。
4. `master` 本身无法 import（见 §5），故其物理以 `master_p0` = master + 两行"让合入段跑得起来"的
   垫片计分，与候选同一起跑线。
5. 工具链：`materialize.sh`（影包）→ `score.py`（每候选 JSON）→ `summarize.py`（表格+图+文档，
   **数字全部程序化注入，无手抄**）。

## 3 计算结果（数字表）

| 候选 | κ 误差（最优 dx） | F_cut/2σ (dx=2mm) | Δp **压强场** | Δp 时间平均 | Δp **力积分** | 伪流 |
|---|---|---|---|---|---|---|
{tab}

**两条主结论**：

- **切割力恒等式被修好了。** 合入基线 `F_cut/2σ = +0.522`（力弱一倍），m2447 = **+1.077**、
  m2454 = **+1.111**；法向全部内指（cos(n,r̂) 中位数 1.000），伪流 0.190 → 0.022–0.037。
  m2174 报的"内向比例仅 51–57%"已不复存在。
- **压强场判据没有任何候选通过。** 最好的是 m2454 的 **−46.6%**，m2447 为 **−87.8%**，
  六个只改接触角/打包的分支停在基线 **−95.5%**。

## 4 结果图

![referee](csf_referee.png)

- **(a)** 同一弛豫末态上三个估计量：m2454 的力积分读 **+15.0%**（"门禁通过"），
  而同一状态的压强场读 **−46.6%**；时间平均未救回（**−38.0%**）→ 残差是**偏差**不是噪声。
- **(b)** 切割力恒等式随 dx：m2447/m2454 贴合 1.0，基线停在 0.52。
- **(c)** m2454 液滴中心压强历程：在 0 附近小幅摆动，从未向 σ/R = 360 Pa 收敛。

## 5 注意点（坑/局限/适用范围）

1. **合入树根本跑不了表面张力**（两处阻塞，均非物理问题）：
   - `src/tensorsph/__init__.py:41` import `apply_contact_angle_boundary`，该函数在
     `src/tensorsph/surface_tension.py` 中不存在 → **`import tensorsph` 直接 ImportError**，
     所有 import 该包的测试模块在 collection 阶段即失败。
   - `src/tensorsph/droplet_impact.py:609` 以 `wall=` 调用力函数，合入版力函数不接受该参数
     → **任何 σ>0 的求解器运行都 TypeError**。
   二者均来自 commit `bec5848`（m2090），其提交信息自述"1 个文件被门禁滤除"：该次改动中
   `surface_tension.py`（+208 行 diff）被合并门禁丢弃，而 `__init__.py` 与
   `droplet_impact.py` 落了盘。这解释了为何同一 bug 被连续六次重攻而 main 上始终看不到修复。
   **属交付管线缺陷；按 2026-11-02 控仓令，本单只记录不改平台。**
2. **不要用力积分冒充 Laplace 验证。** `Σρ(−a·n̂)V/(2πr_c) ≡ σκ`，是力归一化的自证，与判据 A
   同源；二者一致（m2454: +15.0% vs +11.1%）只说明模型内部自洽。
3. **为什么压强跟不上**：定律要求中心-界面密度差 `Δρ/ρ = 9.0e-4`，而自由面求和密度本身就缺
   `1−ρ_band/ρ0 = 0.067`——**比信号大 25 倍**；被 clamp 的 Tait 压强把外层壳压到 p≈0，
   腔内只剩 20 Pa 量级。时间平均无效即为"这不是噪声、是结构性问题"的证据。
4. **局限**：B/C 只跑了一个分辨率（dx=2mm）与一个毛细时间；压强场的分辨率阶梯**未测**，本报告不作声明。
   只计分"默认公开路径"（σ>0 经 `rhs`）；若修复藏在非默认开关后，按设计判为"未改变物理"。
   影包只覆盖 `src/tensorsph`。

## 6 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 最佳候选实测 | 相对误差 | <3%？ | <15%？ |
|---|---|---|---|---|---|
| κ（经切割力） | 50 m⁻¹ | 53.7 m⁻¹（m2447, dx=2mm） | **+7.4%** | ✗ | ✓ |
| F_cut | −14.4 N/m | −15.51 N/m（m2447） | **+7.7%** | ✗ | ✓ |
| Δp（力积分） | 360 Pa | 414 Pa（m2454） | **+15.0%** | ✗ | ✗（临界） |
| Δp（压强场） | 360 Pa | 192 Pa（m2454） | **−46.6%** | ✗ | ✗ |
| 伪流比 | < 0.1 | 0.022（m2454） | — | ✓ | ✓ |

**结论：无一项达到 <3%。** 力归一化类判据进入 15% 门禁；**物理判据（压强场 Δp）仍 FAIL**，
故"表面张力已验证 Young-Laplace"的证据等级维持 **WITHHELD**。

## 7 裁决与建议

- **建议合并 m2447**：唯一在所有 dx 上把切割力恒等式压进 15%（dx=2mm 为 +1.077）、法向全内指、
  伪流比基线安静 25 倍的分支，且携带让包能 import 的接触角边界条件。m2454 为接近的第二名
  （+1.111），但在最细 dx 漂到 +18.7% 且不含接触角 BC。
- 下一步（单干可做）：以 m2447 为基座做**压强场的分辨率阶梯**，并测试 Adami et al. (2010)
  式背景压强/自由面壳层处理，看 Δρ/ρ 信号能否从 6.7% 的自由面亏缺中被分离出来。
"""

rp = {
    "title": "TensorSPH CSF 表面张力的独立裁判：七个修复候选、同一把尺子（压强判据仍 FAIL）",
    "summary": ("用一个只调用公开 API 的裁判工具（benchmarks/csf_referee）给七个未合并的表面张力修复分支"
                "打同一套解析基准：切割力恒等式被 m2447/m2454 修好（F_cut/2σ 0.522→1.077/1.111，"
                "伪流 0.190→0.022），但候选宣称的 Laplace 15% 来自力积分（≡σκ，自我重测），"
                "同一弛豫末态的压强场仍为 −46.6%（m2454）/−87.8%（m2447），时间平均无效说明是偏差非噪声；"
                "另查实合入树因门禁丢弃 bec5848 的一个文件而既不能 import 也不能以 σ>0 运行。"
                "建议合并 m2447，Young-Laplace 物理判据维持 WITHHELD。"),
    "scenario_key": "tensorsph-csf-referee",
    "mission_id": 2495,
    "tags": ["TensorSPH", "surface-tension", "CSF", "Young-Laplace", "verification", "referee"],
    "markdown": body,
    "figures": [{"name": "csf_referee.png",
                 "caption": "(a) 同一末态的三个 Δp 估计量；(b) 切割力恒等式 vs dx；(c) m2454 中心压强未收敛到 σ/R",
                 "png_base64": base64.b64encode(png).decode()}],
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(rp))
print("payload KB", round(len(json.dumps(rp)) / 1024, 1))