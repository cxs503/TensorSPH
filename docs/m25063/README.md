# m25063 — TensorSPH 能力注册表（锚点 CAP-REG-SPH，双试点规范）

## 1. 交付了什么

| 交付物 | 路径 | 说明 |
|---|---|---|
| 注册表模块（微服务式自注册 + 双模式 dump CLI） | `src/tensorsph/registry.py` | `@capability(...)` import 时自注册；`REGISTRY` 单例；`--static` AST 模式不 import 包、免 torch；stdout 单 JSON：`{repo, mode, source_commit, count, summary, capabilities, problems}` |
| 全仓诚实盘点（19 项） | 19 个模块内的 `@capability` 装饰器 | solver 核心 1 / 算例 16 / FSI 1 / 冰 1（含 compressible.sod 子包） |
| 一致性审计（虚标机械复核） | `scripts/m25063_capability_registry_audit.py` → `docs/m25063/registry_audit.{md,json}` | wired/tested/verified/endpoint 四类虚标逐条机械复核，open_findings=0 |
| 平台同步 | `scripts/m25063_capability_sync.py` → `docs/m25063/capability_sync_receipt.json` | `POST /api/v1/tensor/capabilities/sync`：`upserted=19, tombstoned=0` |
| 静态 dump 工件 | `docs/m25063/registry_dump_static.json` | 19 caps，0 error |
| 契约测试 | `tests/test_capability_registry.py`（9 项） | 含「免 torch 静态模式」「静态↔import 逐字段一致」「伪 mature 被本地拦」「cli: endpoint 必须对应真实 --case」 |
| CLI 补线 | `src/tensorsph/__main__.py` 新增 `--case poiseuille_duct` | 让 mature 能力 `sph.case.poiseuille_duct` 的 endpoint 成为真可调用对象 |
| 缺陷修复 | `case_poiseuille_duct.py` `main()`/`run_ladder()` 读 `out['u_centre_analytic']` 顶层键 | 实际键在 `out['profile']['u_centre_analytic']`；原 `--only`/ladder 打印路径 KeyError（注册表指针解析暴露的在案缺陷） |

## 2. 注册表纪律（与平台 422 门同构，同 m23186/m23215 契约）

- `mature` ⇒ 必须 `wired and verified`（服务端 `POST /sync` 同款门；本地注册时即抛 `MaturityDisciplineError`）。
- `verified` ⇒ 必须 `evidence.grade ∈ {ANALYTIC, XVALUE}` 且带 `evidence.max_error_pct`。
- `planned` ⇒ 必须带 `evidence.feature_ledger` 锚，且不得 `verified`。
- `endpoint` ⇒ 只允许 `cli:/route:/api:` 三种 scheme，且只给 `wired=True` 的能力。
- 指针解析：`module_path` / `entry` / `test_path` 全部对工作树求解，指不到就是 `error`。

## 3. 家底盘点（诚实分级，19 项）

- **mature（2，全部 verified+wired，误差带 < 3%）**：
  - `sph.case.hydrostatic`：静水压剖面 p(y)=ρ₀g(H−y)，gate 0.7710% < 1.0% 预声明门（`dev_report_case_hydrostatic_evidence.md`）。
  - `sph.case.poiseuille_duct`：镜壁导管 Poiseuille 对闭式解，中线 +0.38% / 剖面 RMS 2.72% / 壁面剪切 −0.35%（`DELIVERABLE_POISEUILLE_MIRROR.md`，全门 ≤5%）。
- **developing（17）**：WCSPH 核心（droplet impact demo）、stokes_second（Couette 瞬态 8% 门 honest FAIL 在案）、dam_break 2D/3D、sloshing、wave_flume（65.5% 波高误差 honest FAIL 在案）、slamming（Wagner 形状/附加质量律）、gresho、taylor_green、rayleigh_taylor、ritter（Ritter 1962 参考式）、capillary_oscillation（Rayleigh 1917 频率比）、lake_at_rest（静湖守恒锚）、kleefsman_obstacle、FSI2 Turek-Hron、ice_splash、compressible.sod（64% L2 honest FAIL 在案）。
- **planned（0）**：本单不新增 roadmap 项；`planned ⇒ feature_ledger` 规则已预留对账通道。

## 4. 怎么验证的

```bash
PYTHONPATH=src python src/tensorsph/registry.py --static --strict   # 19 caps, 0 error
PYTHONPATH=src python src/tensorsph/registry.py --parity            # static↔import parity ok=True
PYTHONPATH=src python scripts/m25063_capability_registry_audit.py   # findings 0, open 0
PYTHONPATH=src python scripts/m25063_capability_sync.py             # upserted=19 tombstoned=0
PYTHONPATH=src python -m pytest tests/test_capability_registry.py -q  # 9 passed
PYTHONPATH=src python -m tensorsph --case poiseuille_duct --quick   # 新 CLI 线可调用
```

## 5. 注意点 / 局限

1. **`python -m tensorsph.registry` 会 import 包**（Python 语义）；torch-free 主机必须走文件路径 `python src/tensorsph/registry.py --static`（契约测试用假 torch 包 raise ImportError 证明静态模式真的不 import 包）。
2. **wired 机械判据**：src/ 内存在非测试/非 registry 模块按词引用该能力模块基名。`sph.case.poiseuille_duct` 原本未接线（`__init__`/`__main__` 均不引用），本单补 `--case poiseuille_duct` 后才升 mature——endpoint 因此是真可调用对象。
3. **`--case poiseuille_duct --quick`（200 步）是冒烟探针**：t/tau=0.40 未达稳态，u_centre≈0 是诚实的欠积分读数；verified 误差带来自全阶梯运行（`DELIVERABLE_POISEUILLE_MIRROR.md`）。
4. **在案缺陷修复**：`case_poiseuille_duct.main()`/`run_ladder()` 读 `out['u_centre_analytic']` 顶层键，实际键在 `out['profile']` 下——原 `--only` 与 ladder 打印路径 KeyError。注册表指针解析把这条在案缺陷暴露出来，已修。
5. 本单登记 19 项（诚实切片：solver 核心 + 每学科域至少一项）。仓内仍有大量模块未登记（aeration 包、freesurface/、wcsph/、threed/、vv/、ice_thermal/、fsi2_* 其余各件、csf_* 表面张力族等）；纳入是下一位的机械工作（装饰器 + 审计脚本即可）。
6. 误差带口径：hydrostatic 0.7710%（1.0% 预声明门）、poiseuille_duct 2.72%（剖面 RMS，5% 门）——均 < 3%。

## 6. 下一步（供后续单接续）

- 用同一装饰器把 aeration 包（impact_law/bubble_transport/mixture_acoustics）、freesurface/、wcsph/、csf_* 表面张力族、ice_thermal/ 纳入。
- 给 developing 高价值项补解析/外部基准误差带：wave_flume 收敛阶梯（集群在案）、Couette 瞬态 8% 门、Sod 64% L2 根因、Gresho/TGV 对文献误差带——成熟后升 mature。
- 为 TensorSPH 增加真正的 console script（`tensorsph-run --case <x>`），把 endpoint 从 `cli:python -m` 升级成 `cli:tensorsph-run` 契约。
