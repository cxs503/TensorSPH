# 生态摩擦记录 — m18765（CASE-POISEUILLE-SPH）

| 项 | 现象 | 影响 | 绕行方式 |
|---|---|---|---|
| 集群排队 | `run_compute submit` 递交 cj2020（1 核，3600 s 上限）后 **>30 min 未派发**；同时段在跑 121–142 个作业，先递交的 1 核作业同样滞留（按 core 记账，1 核作业长期排在 16 核作业之后） | 需要 ~160 s 的两次加长算例（P3 原设计 c_s=40、P2 工况 t_end 加倍）无法在合理时间内取回 | 改用设计等价且更省的点（在已认证 Re 上降低 c_s 提高 Ma，步数减半）；未闭合的对照如实写进文档与报告 |
| 本机执行上限 | `run_shell` 硬上限 120 s/命令，单进程组超时即整组收割 | 任何 >120 s（本机 1 核 ≈ 80 s/6154 步）的算例无法一次跑完，也无法分段续算（求解器无 checkpoint 接口） | 每个 ≤80 s 的点单独一次调用；需要 2x 步数的对照放弃 |
| 本机单核吞吐 | 144 粒子 / 6154 步耗时 ~79 s（≈12.8 ms/step）；微基准 threads=1/2/4 → 11.52/11.41/16.18 ms/step | 线程无法加速（小体系算子开销主导，4 线程反而慢 1.4x），只能靠减少步数降成本 | 记入 `out/bench_threads.json`，作为选点依据 |
| 知识库检索 | `/api/v1/tensor/knowledge/search` 返回 `engine_offline`（127.0.0.1:8021 connection refused） | 开工三查中的 capability / oss_solutions 两查无法完成 | 改用 negative-knowledge 检索（可用，TensorSPH 域无 Poiseuille 相关教训，仅 TensorLBM 分层 Poiseuille 的流程性失败）；本单结论因此只依赖仓内工件与解析解 |
| 能力台账 | `/api/v1/capability-upgrade/capabilities?repo=TensorSPH` 返回空列表；`/api/v1/tensor/negative-knowledge/search?keywords=TensorSPH` 返回 422 | 无法核对同域既有能力分级 | 以仓内 `case_verdict.json` / `results.json` 与测试为准 |