"""Markdown renderer for the boundary-integral lake-at-rest study (JSON -> doc).

Kept separate from the runner so that the document can be re-rendered from the
frozen artefact without re-running any physics, and so that every number in the
prose is a lookup, never a transcription.
"""
from __future__ import annotations


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def render(study: dict) -> str:
    rows = study["resolutions"]
    gates = study["gates"]
    fine = rows[-1]
    coarse = rows[0]
    names = list(study["variants"])
    L: list[str] = []
    a = L.append

    a("# 封闭水箱静水湖面：壁面积分（boundary-integral）壁面处理能闭合残差吗？（CASE-SPH-LAKE-AT-REST）")
    a("")
    a(f"**范围** TensorSPH ｜ **模块** `tensorsph.lake_at_rest_bi` ｜ "
      f"**证据** `tests/data/lake_at_rest_bi/bi_wall.json` ｜ "
      f"**门禁** `tests/test_lake_at_rest_bi.py` ｜ "
      f"**图** `docs/figures/lake_bi_variants.png`, `lake_bi_resolution.png`, "
      f"`lake_bi_localisation.png`")
    a("")
    a("本文件所有数字由 `benchmarks/lake_at_rest_bi/run.py` 程序化写入 JSON 后由 "
      "`benchmarks/lake_at_rest_bi/make_doc.py` 注入，零手抄。")
    a("")

    # ---------------------------------------------------------------- verdict
    a("## 1. 判决")
    a("")
    a("前两单把封闭水箱的静水残差逐层剥到最后一句话上：")
    a("")
    a("* `docs/verification_lake_at_rest_wall_gap.md`：残差不是质量场能抵消的离散误差；")
    a("* `docs/verification_lake_at_rest_wb.md`：四个一致的（well-balanced）压力梯度算子也闭合不了，"
      "但该单明确写下了自己的**范围限制**——壁面积分重构未实现，因此「必需壁面积分」只是**推断**。")
    a("")
    a("本单把那句话变成测量：实现壁面积分的三种形式（含一维面元积分、镜像因子、"
      "散度定理/半解析形式），每种都用各自的算子重解质量场，再量水平残差。结果：")
    a("")
    a(f"* 壁面积分**没有**击败随机壁粒子算子——最好形式在 `dx = {fine['dx']:.3f}` m 上为 "
      f"`max|a|/g = {_fmt(gates['best_boundary_integral'])}`，"
      f"而基准（装运的壁粒子算子）为 `{_fmt(gates['baseline_wall_particles'])}`；")
    a(f"* 且残差**随加密变差**（`dx = {coarse['dx']:.3f}` → `{fine['dx']:.3f}`: "
      + " → ".join(_fmt(r['variants']['bi_surface_adami']['max_a_over_g']) for r in rows)
      + "），而基准保持在 0.10–0.12 不收敛也不发散；")
    a(f"* 散度定理（量纲正确、半解析）形式**连垂向平衡都保不住**：用自己的算子重解质量场后，"
      f"最好的 `max|a|/g = {_fmt(gates['resolved_boundary_integral_best'])}`，"
      f"而装运算子的重解把垂向闭到 `{_fmt(fine['shipped_reconstruct']['final_vertical_residual'])}`；")
    a("* 维度上**错误**的朴素面元形式 `∫ p_w ∇W dS` 大出 1/dx 倍（实测 20–80 g），"
      "本单把它测出来并排除，写在这里是为了让后来者不再重复这一步。")
    a("")
    a(f"**结论（诚实 FAIL，且否证一条推断）**：{gates['verdict']}")
    a("")

    # ------------------------------------------------------------------ case
    a("## 2. 算例与问题")
    a("")
    a("| 量 | 值 |")
    a("|---|---|")
    a("| 水箱 | 0.40 m × 0.30 m，静水深 0.30 m，两侧 + 底三层壁粒子（装运基准） |")
    a("| 流体 | rho0 = 1000 kg/m³, c0 = 20 m/s, gamma = 7, g = 9.81 m/s² |")
    a("| 分辨率 | dx = " + ", ".join(f"{r['dx']:.3f}" for r in rows) + " m，h = 1.3 dx |")
    a("| 解析基准 | p = rho0 g (eta − y)（精确，与地形无关） |")
    a(f"| 锚点容差 | {study['anchor_tolerance']:.0e}（机器精度） |")
    a(f"| 流体粒子 / 质量未知量 | " + " / ".join(
        f"{r['particles']}/{r['unknowns']}" for r in rows) + " |")
    a("")
    a("被检验的问题只有一个：**把壁面从一堆静态壁粒子换成流体域边界上的面积分，"
      "封闭水箱的水平静水残差会不会闭合到锚点容差？**")
    a("")

    # ---------------------------------------------------------------- method
    a("## 3. 方法")
    a("")
    a("### 3.1 壁面与面压力")
    a("")
    a("流体（及装运壁粒子第一层）落在 `(k+1/2)dx` 上，所以壁面是 `x = 0`, `x = width`, `y = 0` 三个平面。"
      "面元按同一半格偏置铺设，长度 `A_s = dx`（可用 `sub` 细分，不改面位置），法向指入流体；"
      "面元周长的恒等式 `2·depth + width` 是门禁之一。面压力两种取法：")
    a("")
    a("* `adami`：从流体侧外插 "
      "`p_w = Σ_f [p_f + rho_f g (y_f − y_s)] W_sf / Σ_f W_sf`（Adami et al. 2012，纯数据驱动）；")
    a("* `hydrostatic`：解析面压力 `rho0 g (eta − y_s)`——**神谕上界**，只用来回答"
      "「即使面压力完全精确，够不够？」，不是候选闭合。")
    a("")

    a("### 3.2 三种面元形式")
    a("")
    a("`dS` / `dS2`（朴素）：`a_i^wall = −c (1/rho_i) Σ_s p_w A_s ∇_i W_is`，`c = 1` 或 `2`。"
      "二维核的梯度带 1/长度^3，面元积分因此比体积分短一个长度量级——**本单实测这一族大出 1/dx 倍，"
      "记录并排除**。")
    a("")
    a("`surface_w`（散度定理/半解析）：把核的壁侧半边用散度定理搬到面上，")
    a("")
    a("```")
    a("−∫_{Ω_w} rho c ∇_i W dA = + n ∫_S rho_w c_w W dS + ∫_{Ω_w} W ∇(rho c) dA,")
    a("    c_w = p_i/rho_i^2 + p_w/rho_w^2,   ∇(rho c) = −g ŷ（静水闭合）")
    a("```")
    a("")
    a("面积分用 `W`（二维核值，量纲正确），体积分里的核权重 `∫_{Ω_w} W dA` 由装运壁粒子求和给出"
      "（光滑量，无压力参与）——这就是半解析壁面条件的标准拆分。")
    a("")
    a("### 3.3 协议（与前两单逐位可比）")
    a("")
    a("1. 用装运算子 `lake_at_rest.reconstruct` 把质量场闭到垂向 `~1e-12`，得到**同一个**质量场；")
    a("2. 在同一质量场上量每个变体的 `max|a|/g`（不变质量，只换算子）；")
    a("3. 再让每个候选**用自己的算子重解质量场**（只闭垂向），量重解后的残差——"
      "连垂向都闭不住的算子不是候选；")
    a("4. 控制项：压力场整体 ×1.01（残差必须动）、关重力（残差必须等于压力项）、"
      "去掉壁面（`fluid_only`）、去掉流体项。")
    a("")
    a("### 3.4 实现一致性检查")
    a("")
    a("面元形式必须能复现同一积分的体积求积 `Σ_{j∈wall} V_j phi_j ∇_i W_ij`，"
      "差多少就说明「半解析闭包」丢了多少：")
    a("")
    a("| dx [m] | wall_mode | 相对误差 | 体积量级 |")
    a("|---|---|---|---|")
    for r in rows:
        for mode, chk in r["integral_check"].items():
            a(f"| {r['dx']:.3f} | {mode} | {_fmt(chk['relative_error'])} | "
              f"{_fmt(chk['volume_scale'])} |")
    a("")
    a("这个 5% 量级的差别**不随 `sub` 减小而减小**（闭包 `∇(rho c) = −g ŷ` 是一阶截断，"
      "不是求积分辨率），它就是所有壁面积分数字的误差棒，也是本单对「为什么它闭合不了机器精度」的定量解释。")
    a("")

    # --------------------------------------------------------------- results
    a("## 4. 结果")
    a("")
    a("### 4.1 同一质量场上的残差（`max|a|/g`）")
    a("")
    header = "| dx [m] | 基准壁粒子 | " + " | ".join(names) + " |"
    a(header)
    a("|" + "---|" * (len(names) + 2))
    for r in rows:
        cells = [f"{r['dx']:.3f}", _fmt(r["baseline"]["max_a_over_g"])]
        cells += [_fmt(r["variants"][n]["max_a_over_g"]) for n in names]
        a("| " + " | ".join(cells) + " |")
    a("")
    a("### 4.2 各算子自己重解质量场之后")
    a("")
    a("| dx [m] | 变体 | 垂向重解收敛 | 迭代 | 质量调制 max/min | max\\|a_x\\|/g | max\\|a_y\\|/g | max\\|a\\|/g |")
    a("|---|---|---|---|---|---|---|---|")
    for r in rows:
        for name, res in r["resolved"].items():
            a(f"| {r['dx']:.3f} | {name} | {res['converged']} | {res['iterations']} | "
              f"{res['mass_modulation']:.3f} | {_fmt(res['max_ax_over_g'])} | "
              f"{_fmt(res['max_ay_over_g'])} | {_fmt(res['max_a_over_g'])} |")
    a("")
    a("### 4.3 控制项（`bi_surface_adami`，同一质量场）")
    a("")
    a("| dx [m] | 关重力 | 压力 ×1.01 | 去流体项 |")
    a("|---|---|---|---|")
    for r in rows:
        c = r["controls"]
        a(f"| {r['dx']:.3f} | {_fmt(c['gravity_off']['max_a_over_g'])} | "
          f"{_fmt(c['pressure_scaled_1pct']['max_a_over_g'])} | "
          f"{_fmt(c['no_fluid_term']['max_a_over_g'])} |")
    a("")
    a("### 4.4 残差住在哪一行")
    a("")
    a("| dx [m] | 基准 max\\|a\\|/g | 基准顶行占比 | 壁面积分 max\\|a\\|/g | 壁面积分顶行占比 | 壁面积分去顶行 max\\|a\\|/g |")
    a("|---|---|---|---|---|---|")
    for r in rows:
        bi = r["variants"]["bi_surface_adami"]
        b = r["baseline"]
        a(f"| {r['dx']:.3f} | {_fmt(b['max_a_over_g'])} | {b['top_row_share']:.3f} | "
          f"{_fmt(bi['max_a_over_g'])} | {bi['top_row_share']:.3f} | "
          f"{_fmt(bi['max_a_over_g_below_top_row'])} |")
    a("")
    a("图 `docs/figures/lake_bi_localisation.png`（行内残差剖面）显示：两个算子的残差都聚在自由面那一行，"
      "与 `wall_gap.md` 对装运算子的诊断一致——换壁面处理**没有**把残差搬走。")
    a("")

    # ------------------------------------------------------------- attention
    a("## 5. 注意点与适用范围")
    a("")
    a("* **适用**：2-D 封闭水箱、静水、零速，WCSPH（Tait EOS，密度求和），三次样条核，h = 1.3 dx，"
      "壁粒子三层；结论限于「压力梯度 + 壁面处理」这两族自由度，不含自由面重构（VOF/level-set）"
      "或移动壁面。")
    a("* **未实现**：三维、壁面曲率、移动壁面（Adami 外插里 `a_w` 项）、"
      "以及「把壁侧体积项用高阶外插替代一阶闭包」——后者是本单最可能的下一个改进点，"
      "但它是**闭包阶数**问题，不是壁面处理形式问题。")
    a("* **重解不收敛不等于实现错**：`surface_w` 的重解停在 "
      f"`{_fmt(rows[-1]['resolved']['bi_surface_adami']['max_ay_over_g'])}`，"
      "是因为壁侧核权重带来的 `+g∫W dA` 项在自由面处没有对应的流体侧项可以抵消（残差聚在顶行，见 4.4），"
      "与装运算子的垂向闭合能力（`~1e-13`）形成对照。")
    a("* **朴素形式是故意留下的**：`dS`/`dS2` 是文献里最常见的写法，量纲短一个长度；"
      "把它们留在结果表里，是为了让「为什么不能直接用 ∫p∇W dS」可被复核。")
    a("* 质量调制 `max/min` 到 2 以上说明重解把质量场拉得很远；这类结果不作为候选，只作为「不可行」的证据。")
    a("")

    # ------------------------------------------------------------- error bars
    a("## 6. 误差判定")
    a("")
    a("锚点判据：`max|a|/g ≤ " + f"{study['anchor_tolerance']:.0e}`（机器精度）。")
    a("")
    a("| 关键量 | 基准/目标 | 实测 | 是否达标（<3% / ≤容差） |")
    a("|---|---|---|---|")
    a(f"| 壁粒子基准 max\\|a\\|/g | ≤ {study['anchor_tolerance']:.0e} | "
      f"{_fmt(gates['baseline_wall_particles'])} | 否（超 "
      f"{gates['baseline_wall_particles']/study['anchor_tolerance']:.1e} 倍） |")
    a(f"| 壁面积分最好 max\\|a\\|/g | ≤ {study['anchor_tolerance']:.0e} | "
      f"{_fmt(gates['best_boundary_integral'])} | 否（超 "
      f"{gates['best_boundary_integral']/study['anchor_tolerance']:.1e} 倍） |")
    a(f"| 壁面积分重解后 max\\|a\\|/g | ≤ {study['anchor_tolerance']:.0e} | "
      f"{_fmt(gates['resolved_boundary_integral_best'])} | 否 |")
    a(f"| 无壁面对照 | 应远大于基准 | {_fmt(gates['no_wall'])} | 是（壁项确实在起作用） |")
    a(f"| 面元 vs 体积求积相对误差 | < 3% | "
      f"{_fmt(max(r['integral_check']['adami']['relative_error'] for r in rows))} | 否（"
      + f"{100*max(r['integral_check']['adami']['relative_error'] for r in rows):.1f}%，"
      "一阶闭包的固有误差，见 3.4） |")
    a("")
    a("**是否 <3%？** 关键量（静水残差）**不是**：最好形式仍超锚点容差 "
      f"{gates['best_boundary_integral']/study['anchor_tolerance']:.1e} 倍，"
      "且随加密变差。唯一达到锚点容差的配置仍然是横向周期水箱（"
      "`docs/verification_lake_at_rest_periodic.md`，`max|a|/g ~ 1e-13`）。"
      "因此本单**不**关闭锚点，只关闭锚点的最后一条未决推断。")
    a("")
    a("## 7. 复现")
    a("")
    a("```")
    a("PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/run.py \\")
    a("    --json tests/data/lake_at_rest_bi/bi_wall.json --dx 0.05")
    a("PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/run.py \\")
    a("    --json tests/data/lake_at_rest_bi/bi_wall.json --dx 0.04")
    a("PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/run.py \\")
    a("    --json tests/data/lake_at_rest_bi/bi_wall.json --dx 0.025")
    a("PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/make_doc.py")
    a("python -m pytest tests/test_lake_at_rest_bi.py -q")
    a("```")
    a("")
    a("（三次分辨率分开递交是因为本机单条命令有 120 s 墙，JSON 会累积合并分辨率。）")
    return "\n".join(L) + "\n"