# 交差审计 m15164 · XVAL-SPH-DAMBREAK（溃坝三方横向）

审计单 m15244 · verdict_by=board-audit · 2026-11-11
被审计单：m15164（status=succeeded, accepted=true, model=deepseek-v4-flash）
工作树：/root/tp-agent/work/m15164（base 184ff8d，未提交交付物 3 项：docs/dev_report_xval_dambreak_round1.md、scripts/xval_dambreak/、xval_output/）

## ① 案例介绍
2-D 溃坝（干床、初始水柱贴左壁）：a=0.2 m，H=0.4 m（H/a=2），水箱 2.0×0.75 m，
dx=0.01（800 流体粒子+1074 壁粒子），Tait γ=7、c0=20 m/s、ν=1e-6 m²/s、
dt=9.75e-5 s、t_end=0.3 s（3077 步）。参考=Martin & Moyce (1952) Phil. Trans. R. Soc. A
244:312-324（本仓已数字化 20 点 Z/a–τ 表，±0.05 in Z/a，
benchmarks/dam_break/martin_moyce_1952.json）。任务=我方 TensorSPH vs PySPH vs
DualSPHysics 三方同规格横向，查 bug。

## ② 怎么做的（审计方法）
1. 读 m15164 工作树未提交交付物（报告/脚本/工件）与 mission result（tensor_dev_missions.id=15164）。
2. 基准裁判查询：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-SPH-DAMBREAK'` → **0 行**（该锚点从未注册基准判据）。
3. 逐项核验收三件套：run_artifacts 文件存在性/完整性、benchmark ref vs got 数值、repro_cmd 逐条实跑。
4. 判级按 L0-L5 功能实质（附百页报告不改变级别）。
5. verdict 落 tensor_maturity_verdicts，交 maturity_gate 机器复核。

## ③ 计算结果（审计核验表，程序化取证）

| 验收项 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 基准判据注册 | tensor_benchmark_cases 有 XVAL-SPH-DAMBREAK 行 | 0 行（全表 14 锚点无此锚） | **缺** |
| run_artifacts | 三件套之 run 工件 | xval_output/ 27 个 PySPH hdf5（xval_00000…02600，各 270 KB）+ xval.log；xval.info `completed:false`（2600/3077 步被杀）；**无** result_ours_*.json / result_pysph_*.json / xval_table.json | **部分（残缺）** |
| benchmark ref vs got | got 数值在容差内 | **无任何 got 数值**（报告自认 NOT RUN×3 判据） | **缺** |
| repro_cmd 可执行 | 5 条命令逐条实跑 | spec ✓（exit 0，重生成 case_spec.json）；ours ✗（dx=0.02 90 s 未返回 exit 124；另一次 90 s 内跑完仿真但序列化崩 `TypeError: 'float' object is not callable` @ xval.py:245，`mass_drift` 现为 property 非方法）；pysph ✗（复现 `RuntimeError: LinkedListNNPS requires too many cells (-569796944)` @ linked_list_nnps.pyx:340）；compare/figures 无输入文件不可跑 | **不可执行（1/5）** |
| 报告六段 | 齐全 | 报告 8 节齐全且诚实（honest FAIL 自认、WITHHELD 不填数）；但 **tensor_reports 无 m15164 行**（result 自认 401 未发出） | **部分** |
| 三方对照 | 涌浪前沿工具间 <10%、坍塌高度 <15%、vs M&M <15% | 全部 NOT RUN（报告 §5 自认） | **缺** |

## ④ 结果图
图为基准参考线（Martin & Moyce 1952 数字化 20 点，即本应被对照的 ref）：
`artifacts/audit_m15164/fig_martin_moyce_ref.png`。
无场量云图/对比曲线的理由：本单为纯审计（文档/后端核验）任务，且被审计单未产出任何
数值轨迹——无 got 可画，不伪造。

## ⑤ 注意点
- 被审计单报告诚实度高（honest FAIL、不填估计值、friction.json 留痕），但「succeeded+accepted」
  与「零数值结果」并存=结榜口径问题，本 verdict 按功能实质判级。
- xval.py 两处实锤缺陷（审计实跑复现）：
  1) ours 臂 `run.mass_drift()` 调用与现 API（property）不匹配 → 序列化必崩（xval.py:245）；
  2) pysph 臂 LinkedListNNPS 2-D 格数溢出（-569796944 为 int 溢出特征值），re-pin h/rho 未解。
- DualSPHysics 无 GPU 不可装（friction.json 证据链完整）——第二参照缺失，横向带宽 WITHHELD 合理。
- 本机 loongarch64 无 CUDA/无 loongarch 轮子，PyPI sdist 在 py3.11+Cython3 下不可编译——生态摩擦真实。

## ⑥ 误差判定
**无 got 数值 → 误差不可计算，显式回答：无法判定 <3%（也无从谈起 <15% 容差）**。
原因：m15164 未产出任何数值结果（自认 NOT RUN）；且 tensor_benchmark_cases 未注册该锚点判据，
「基准判据唯一裁判」在 DB 侧不存在。case_spec.json 内 gates（10%/15%/15%）仅为工作树内自设，
非 DB 注册判据。

## 判级
**L2（独立功能实现）**：交付物=真代码横向对比 harness（scripts/xval_dambreak/xval.py，26.9 KB，
spec/ours/pysph/compare/figures 五子命令；spec 实跑通过、仿真臂确曾运行出 2600 步工件），
输入→计算→输出链路真实存在。未达 L3（未接入主求解器/平台，独立脚本）、未达 L4（无完整算例
数值+误差分析+可复现——repro_cmd 1/5 可跑、无 got）。honest FAIL 如实入账，不粉饰。

## 复现（审计取证命令）
```
psql -U postgres -d tensorplatform -c "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-SPH-DAMBREAK'"
cd /root/tp-agent/work/m15164
python scripts/xval_dambreak/xval.py spec            # exit 0
timeout 90 python scripts/xval_dambreak/xval.py ours --dx 0.02 --nu 1e-6   # exit 124/TypeError
timeout 300 /tmp/xv2/bin/python scripts/xval_dambreak/xval.py pysph --dx 0.01 --nu 1e-6  # RuntimeError
```
取证日志：artifacts/audit_m15164/repro_runs.log
