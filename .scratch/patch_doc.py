"""One-shot patch: make_doc.py reads the open-surface matrix for the historical
diagnosis and injects the new closure section (m3501)."""
from pathlib import Path

p = Path("benchmarks/laplace_pressure_floor/make_doc.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, "anchor not unique/found: %r" % old[:60]
    s = s.replace(old, new)


# (1) historical sections read the open-surface matrix
sub(
    'def main():\n    rep = json.loads((HERE / "report.json").read_text())\n    runs = rep["runs"]',
    'def main():\n'
    '    # The diagnostic sections below document the *pre-closure* model (the\n'
    '    # measurement that motivated P0), so they read the open-surface matrix.\n'
    '    # The closure result is injected in section 8, from report.json.\n'
    '    rep = json.loads((HERE / "report_open_surface.json").read_text())\n'
    '    closure = json.loads((HERE / "report.json").read_text())\n'
    '    runs = rep["runs"]',
)

# (2) closure rows + ladder rows, computed from the closure report
sub(
    "    filled = TEMPLATE.format(",
    '    # ---- section 8: the closure result, from report.json ----\n'
    '    c_runs = closure["runs"]\n'
    '    before = {(x["dx_m"], x["sound_speed"]):\n'
    '              x["absolute_time_average_tail"]["rel_error_pct"] for x in runs}\n'
    '    c_rows = []\n'
    '    for run in sorted(c_runs, key=lambda x: (x["dx_m"], -x["sound_speed"])):\n'
    '        after = run["absolute_time_average_tail"]\n'
    '        c_rows.append(\n'
    '            "| {dx:g} | {c0:g} | {b} | **{a}** | {cr:.2f} | {v:.3f} | {p15} | {p3} |"\n'
    '            .format(dx=run["dx_m"], c0=run["sound_speed"],\n'
    '                    b=fmt_pct(before.get((run["dx_m"], run["sound_speed"]),\n'
    '                                         float("nan"))),\n'
    '                    a=fmt_pct(after["rel_error_pct"]),\n'
    '                    cr=((run["density"]["rho_core_over_rho0"] - 1.0)\n'
    '                        / run["density"]["compression_required"]),\n'
    '                    v=run["spurious_current"]["vmax_over_vref_final"],\n'
    '                    p15="是" if after["within_15pct"] else "否",\n'
    '                    p3="是" if after["within_3pct"] else "否"))\n'
    '    c_err = [abs(x["absolute_time_average_tail"]["rel_error_pct"]) for x in c_runs]\n'
    '    c_ladder = "\\n".join(\n'
    '        "| {p} | {e} | {r:.2f} | {v:.3f} |".format(\n'
    '            p=d["closure_passes"], e=fmt_pct(d["tail_rel_error_pct"]),\n'
    '            r=d["compression_fraction"], v=d["spurious_vmax_over_vref"])\n'
    '        for d in closure.get("passes_ladder", []))\n'
    '\n'
    '    filled = TEMPLATE.format(',
)

# (3) new format kwargs
sub(
    '        six_section="\\n".join(six),\n    )',
    '        six_section="\\n".join(six),\n'
    '        c_rows="\\n".join(c_rows),\n'
    '        c_ladder=c_ladder,\n'
    '        c_err_lo=min(c_err), c_err_hi=max(c_err), c_n=len(c_err),\n'
    '        c_min=min(c_err), c_max=max(c_err),\n'
    '        c_pass15=sum(1 for x in c_err if x < 15.0),\n'
    '        c_pass3=sum(1 for x in c_err if x < 3.0),\n'
    '        c_passes=closure["closure"]["closure_passes"],\n'
    '        c_ratio=closure["closure"]["free_surface_rho0_ratio"],\n'
    '    )',
)

# (4) headline verdict note
sub(
    "验收判据和下一步验收门槛都在 §⑥/§⑦ 落定。\n",
    "验收判据和下一步验收门槛都在 §⑥/§⑦ 落定。\n"
    "\n"
    "> **更新（m3501：P0 已落地）**：`src/tensorsph/droplet_impact.py` 现含\n"
    "> `shell_pressure()`（Adami/Ihmsen/Adams 2012 §3.3 的核加权外推 + Ihmsen et al. 2014\n"
    "> 的多趟延长）与 `Config.free_surface_pressure`（默认关，benchmark 显式开启）。\n"
    "> 同一 benchmark 的 tail-avg 误差区间变为 **{c_min:+.1f}%…{c_max:+.1f}%**\n"
    "> （{c_n} 点中 {c_pass15} 点 <15%、{c_pass3} 点 <3%），Laplace 压差从 **WITHHELD**\n"
    "> 升为 **PASS（<15%）**，逐项见 §⑧。\n",
)

# (5) mark P0 as closed in section 7
sub(
    "1. **P0｜自由面压强闭合（C3，唯一门槛项）**：",
    "1. **P0｜自由面压强闭合（C3，唯一门槛项）——已结清（m3501，见 §⑧）**："
    "历史欠账原文如下，实现见 `src/tensorsph/droplet_impact.py::shell_pressure`：",
)

# (6) append sections 8 and 9 inside the template
tail_anchor = "   `p_core - p_shell` 口径；本文件 §③.2 的 bandA 列只作历史对照保留。\n\"\"\")"
new_sections = '''   `p_core - p_shell` 口径；本文件 §③.2 的 bandA 列只作历史对照保留。

## ⑧ P0 落地：自由面压强闭合（m3501）

**机制**：`tait_pressure` 在自由面把压强钳在 0，亏缺壳层（−{def085:.0f}% / −{def095:.0f}%）因此
无法承载 `sigma/R`。闭合把「被密度亏缺标记为自由面」的粒子（`rho/rho0 < {c_ratio:g}`）的压强改为
邻域核加权外推（Adami, Ihmsen & Adams 2012, JCP 231:7057, eq. 12）：

`p_i = max(0, sum_j (p_j + rho_j * g . r_ji) W_ij / sum_j W_ij)`

单趟外推不够：亏缺壳层厚 2–3 层，外层的邻居本身还钳在 0。按 Ihmsen et al. (2014) eq. (12) 做
**{c_passes} 趟延长**（每趟用上一趟的结果重新加权）；内部粒子始终保留自己的 Tait 值，不被修改。
实现：`src/tensorsph/droplet_impact.py::shell_pressure`，开关 `Config.free_surface_pressure`
（**默认关**），趟数 `Config.free_surface_pressure_passes`。

| dx [m] | c0 [m/s] | 闭合前 tail-avg | 闭合后 tail-avg | 内部压缩/所需 | 伪流 | <15% | <3% |
|---|---|---|---|---|---|---|---|
{c_rows}

趟数收敛（参考点 dx={dx_a:g} m, c0={c0_hi:g} m/s）：

| 趟数 | tail-avg 误差 | 内部压缩/所需 | 伪流 |
|---|---|---|---|
{c_ladder}

![closure](benchmarks/laplace_pressure_floor/laplace_pressure_closure.png)

**判读**：

1. 四点全部进 15% 门（{c_pass15}/{c_n}），{c_pass3}/{c_n} 点同时进 3%；误差量级
   {c_min:+.1f}%…{c_max:+.1f}%，不再随 c0 / 分辨率系统漂移。
2. 伪流仍满足 Adami 判据（< 0.1），闭合没有引入新的界面不稳定。
3. **诚实边界**：这**不是**表面张力模型的精度提升——CSF 力与曲率算子一行未动，补的是自由面
   的压强边界条件。`free_surface_pressure` 默认关，因为它会改变所有自由面算例的动力学；
   翻转默认值应由液滴撞池 / dam-break 等下游算例重新验收后决定（见 §⑨）。
4. 剩余最大残差出现在细网格（dx=1.5 mm），与本 benchmark 已记录的口径一致（时间平均窗口
   1.5 t_sigma，呼吸模态未完全衰减），不是新的系统偏差。

## ⑨ 下一步（P0 之后）

1. **把闭合接到下游算例做 A/B**：`benchmarks/dam_break`、`benchmarks/droplet_impact_2d` 在
   `free_surface_pressure=True/False` 下各跑一遍，比较晃荡/飞溅与实验数据；无退化后再考虑翻转默认值。
2. **P1｜初始化质量校准（C2）** 仍未结清（c0=20 时初始虚假压强 {p_init20:.0f} Pa）。
3. **回归门**：`tests/test_laplace_pressure_floor.py` 的 15% 用例已由 `xfail(strict=True)` 转为
   普通断言（闭合开启），历史诊断用例改读 `report_open_surface.json`，两侧都留痕。
""")'''
sub(tail_anchor, new_sections)

p.write_text(s)
print("patched make_doc.py")