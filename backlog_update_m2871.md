# backlog 更新（m2871：delta 函数归一欠账结清 + 一个假达标结论被推翻）

**日期**: 2026-11-03 ｜ **范围**: TensorSPH ｜ 改动：`src/` 增参（默认不变）+ 新基准 + 新测试 + 新文档

## 已结清：`int delta_s dn = 1`（`docs/verification_delta_function_normalisation.md`）

`csf_fields` 新增 `delta_s_scheme`：`"factor2"`（历史，默认）与 `"influid"`
（`|∇c|/(1−c_s)`，`c_s = (1+VW(0))/2 = 0.6165` 由核与体积闭式给出，无拟合参数）。

| 判据 | `factor2` | `influid` |
|---|---|---|
| `int delta_s dn`（dx 0.003/0.002/0.0015） | 0.755/0.751/0.739（−25%） | 0.984/0.979/0.964（−1.6…−3.6%） |
| `F_cut/(2σ)` 力密度 | 0.786/0.713/0.710（−21…−29%） | 1.025/0.930/0.926（+2.5…−7.4%） |
| κ、力方向 | — | **逐位不变**（负控） |

## 推翻的旧结论：0.8.0 线「切割力已进 15% 门」是假的

最外层粒子核和密度 `ρ_shell/ρ0 = 0.797…0.808`（δs 加权），CSF 加速度是 `f/ρ`，
故同一份表面张力被放大 ~25%，恰好抵消 −25% 的 δs 亏损（0.75×1.25≈0.94）。
**切割力恒等式必须在力密度 `f = −σκδ_s n` 上测**，不能在加速度上测（`tests/test_csf_delta_s.py` 已钉死）。

## 仍欠账（优先级不变）

1. P0 端到端 Laplace 压差仍 WITHHELD：`factor2→influid` 只把 Δp 从 −24.25 挪到 +38.45 Pa
   （−106.7%→−89.3%）。根因是弱可压压强噪声地板（360 Pa 只需 6.3e-3 压缩），
   必须动压强基准（背景压强/无散修正/降 c0），与本轮 δs 无关。
2. P0 切换 `csf_delta_s` 默认到 `influid`：需先解决加速度被密度亏损放大 25%，
   并重跑 `csf_consistent`/`csf_curvature_fit`/`laplace_law`/`dam_break*` 全套基准。
3. P1 `surface_color_level` 的 3-D 推广与曲面 `O(hκ)` 修正。
4. 已知非本轮引入的红测试（clean HEAD 亦然）：`test_laplace_droplet`(4)、
   `test_lattice_wall_clearance`(2)、`test_ritter`(1) —— 7 个，属既有欠账。
