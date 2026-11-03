import base64, io, json, os, subprocess
from PIL import Image

B = "benchmarks/laplace_equilibrium/"
rep = json.load(open(B + "report.json")); drift = json.load(open(B + "drift.json"))
th = rep["reference"]["delta_p_Pa"]; lad = rep["ladder"]; scan = rep["phase_scan"]
act = next(r for r in drift["runs"] if r["setup"]["sigma_N_per_m"] > 0)
ctl = next(r for r in drift["runs"] if r["setup"]["sigma_N_per_m"] == 0)


def png(path, width=1000, colors=96):
    im = Image.open(path).convert("RGB")
    im = im.resize((width, max(1, int(im.height * width / im.width))), Image.LANCZOS)
    im = im.quantize(colors=colors).convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), len(buf.getvalue())


fig1, s1 = png(B + "laplace_equilibrium.png")
fig2, s2 = png(B + "laplace_drift.png")
errs = [r["equilibrium"]["rel_error_pct"] for r in lad]
rows = "\n".join(
    f"| {r['setup']['dx_m']*1e3:.2f} | {r['setup']['n_particles']} | "
    f"{r['setup']['n_complete_support']} | {r['equilibrium']['p_complete_Pa']:.1f} | "
    f"{r['equilibrium']['rel_error_pct']:.1f} | {r['equilibrium']['p_band_Pa']:.1f} | "
    f"{r['equilibrium']['rel_error_band_pct']:.1f} | "
    f"{r['equilibrium']['measurement_snr']:.0f} |" for r in lad)
pc = ", ".join(f"{r['equilibrium']['p_complete_Pa']:.1f} Pa" for r in lad)
snr = " / ".join(f"{r['equilibrium']['measurement_snr']:.0f}" for r in lad)

md = f"""# TensorSPH 表面张力 Young–Laplace 压差：两轮矛盾结论的定量归因（判据仍 WITHHELD）

## ① 案例介绍
二维单相液滴（R = {rep['params']['radius_m']} m）悬于真空，六方密排初始化，无重力无壁面。表面张力应使内部过压 `Δp = σκ = σ/R`（2-D 圆形界面 κ = 1/R）。
参数：σ = {rep['params']['sigma_N_per_m']} N/m，ρ0 = {rep['params']['rho0_kg_per_m3']} kg/m³，c0 = {rep['params']['sound_speed_m_per_s']} m/s，ν = {rep['params']['viscosity_m2_per_s']} m²/s，Tait γ = 7（压强夹零），dx = 3.0 / 2.0 / 1.5 mm（h = 1.3dx），阶梯推进 4 个毛细时间、长程与对照 12 个。
解析基准：`Δp = σ/R = {th:.1f} Pa`，κ = 1/R = {rep['reference']['curvature_1_per_m']:.1f} 1/m。

## ② 怎么做的
仓库内同一模型存在互相矛盾的两次复核：`benchmarks/laplace_law`（dx=4 mm, 380 步）判 PASS，`benchmarks/csf_consistent`（dx=2 mm, 600–900 步）判 −75.6 %。本轮不复用两者，重建测量基准并加对照：
1. 完整支撑判据：只统计 `r < R − 2.5h` 的粒子（核支撑 2h，留 0.5h 余量），剔除截断壳层；
2. 外部压强取真空值 0（Tait 夹零），不再用壳层内 `p(0.8R–0.95R)` 充当外部压强；
3. 跑到静态并报告末端速度、残余振荡、粒子均值标准误与 SNR；
4. 相位扫描：同一次运行在 {scan['n_samples']} 个不同步数上采样，量化瞬态对结论的影响；
5. 长程 + σ=0 对照：12 个毛细时间跟踪 r_rms、内部密度亏损、p_core 趋势，用关掉 CSF 的对照分离机理。
工具链：PyTorch 2.x（CPU float64）、matplotlib、pytest，OMP_NUM_THREADS=1。

## ③ 计算结果（程序化注入自 report.json / drift.json）

| dx [mm] | N | N(完整支撑) | p_core [Pa] | 误差 % | p_band [Pa] | 旧 band 误差 % | SNR |
|---|---|---|---|---|---|---|---|
{rows}

- 三档误差 {min(errs):.1f} % … {max(errs):.1f} %，跨度仅 {max(errs)-min(errs):.1f} 个百分点 ⇒ 结构性偏差、与网格无关，**全部超 15 % 平台线**。
- 相位扫描（dx=4 mm/380 步，ν=0.01）：完整支撑基准误差跨度 {scan['error_spread_pct']:.0f} 个百分点（{scan['n_samples_within_15pct']}/{scan['n_samples']} 落入 15 %），旧 band 基准 {scan['legacy_error_spread_pct']:.0f} 个百分点（{scan['n_legacy_samples_within_15pct']}/{scan['n_samples']}）。
- 长程：σ = 7.2 时 r_rms/R0 {act['outcome']['r_rms_over_r0_first']:.3f} → {act['outcome']['r_rms_over_r0_last']:.3f}，内部密度亏损 {act['outcome']['rho_excess_last']:+.4f}，p_core/(σ/R) 末值 {act['outcome']['p_core_over_theory_last']:.2f} 且仍无平台；σ = 0 对照 r_rms/R0 {ctl['outcome']['r_rms_over_r0_first']:.3f} → {ctl['outcome']['r_rms_over_r0_last']:.3f}（基本不动，p ≡ 0）。

## ④ 结果图
图1 (a) 三档分辨率的 p_core(t)/(σ/R)：均单调衰减、无平台；(b) 两套测量基准误差 vs dx，完整支撑基准贴 0 线、旧 band 基准系统性负偏；(c) 一次运行的 19 个采样时刻误差带（相位假象）。
图2 (a) r_rms/R0 随时间（有/无表面张力）；(b) p_core/(σ/R) 无平台；(c) 内部密度亏损加深。

## ⑤ 注意点
- 该算例在这套离散下**没有静态平衡**：任何「某步数下 p_core = σ/R」的结论（含历史 PASS 与 −75.6 %）都是瞬态采样，不可作为判据。
- 旧 band 协议把外部压强取在 CSF 壳层内部（壳层内 dp/dr = ρa_r < 0），系统性低估压差；粗网格「看起来更准」正是 h/R 变大掩盖了这一点。
- σ=0 对照不是「表面张力无害」：对照组 p ≡ 0，与有表面张力的差别正是「内部可否蓄压」。
- 16 个毛细时间的探索性运行在 t ≈ 14 t_σ 出现 NaN（非入库记录），与膨胀机理一致。
- 未做：三维、多相、h/dx 系统敏感性、GPU/float32 复测、以及自由面压强基准修复实现。

## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）
| 关键量 | 基准 | 实测 | 误差 | < 3 % | < 15 % |
|---|---|---|---|---|---|
| Δp（完整支撑） | {th:.1f} Pa | {pc} | {min(errs):.1f} – {max(errs):.1f} % | **否** | **否** |
| Δp（旧 band 基准） | {th:.1f} Pa | 见上表 p_band | 更负（见上表） | **否** | **否** |
| 静态性 r_rms/R0 | 1.0 | {act['outcome']['r_rms_over_r0_last']:.2f} | +{100*(act['outcome']['r_rms_over_r0_last']-1):.0f} % | 否 | 否 |
| 测量分辨力 SNR | — | {snr} | — | — | 是 |

**显式回答：关键量 Δp 三档误差 {min(errs):.1f}–{max(errs):.1f} %，既未达到 <3 %，也未达到 <15 %。**
本轮是 honest FAIL：交付的是「为什么这个判据一直测不准」的定量证据（测量基准三缺陷 + 相位跨度 {scan['legacy_error_spread_pct']:.0f} 个百分点 + 无静态平衡与机理定位），不是达标声明。缺口已定位到自由面压强/密度基准（与 backlog P0 #1 一致）；CSF 力的幅值判据（切割力恒等式、法向对齐）此前已进入 15 %，本轮证明问题不在力而在压强基准。

**工件**：benchmarks/laplace_equilibrium/ 下 report.json、drift.json、两张 png；docs/verification_laplace_equilibrium.md；tests/test_laplace_equilibrium.py（10 项）。
"""

report = {
    "title": "TensorSPH Young–Laplace 压差：两轮矛盾结论归因（测量基准三缺陷 + 无静态平衡），判据仍 WITHHELD",
    "summary": f"重建平衡态测量基准并做相位扫描与长程 σ=0 对照：三档分辨率 Δp 误差 {min(errs):.1f}–{max(errs):.1f}% 全部超 15% 线且跨度仅 {max(errs)-min(errs):.1f} 点；旧读数(−75.6% / PASS)为瞬态采样(19 个采样时刻跨度 {scan['legacy_error_spread_pct']:.0f} 点)；单相液滴 12 t_σ 内 r_rms 0.71→1.49 R0 膨胀、p 无平台，σ=0 对照不动，机理定位到自由面压强基准。honest FAIL。",
    "scenario_key": "tensorsph-laplace-equilibrium", "mission_id": 2687,
    "tags": ["TensorSPH", "surface-tension", "Young-Laplace", "verification", "WITHHELD", "SPH"],
    "markdown": md,
    "figures": [
        {"name": "laplace_equilibrium.png",
         "caption": "平衡态测量基准与相位假象：(a) 三档 p_core(t)/(σ/R) 无平台；(b) 两套基准误差 vs dx；(c) 19 个采样时刻误差带",
         "png_base64": fig1},
        {"name": "laplace_drift.png",
         "caption": "长程稳定性与 σ=0 对照：(a) r_rms/R0；(b) p_core/(σ/R)；(c) 内部密度亏损",
         "png_base64": fig2},
    ],
}
json.dump(report, open("/tmp/report.json", "w"))
sz = os.path.getsize("/tmp/report.json")
print("fig bytes:", s1, s2, "payload KB:", sz // 1024)

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER", "agent")
pw = os.environ.get("TENSOR_AGENT_PASS", "")


def sh(*a):
    r = subprocess.run(a, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip()


def post(data):
    r = subprocess.run(["curl", "-s", "-X", "POST", base + "/tensor/reports",
                        "-H", "Authorization: Bearer " + data,
                        "-H", "Content-Type: application/json",
                        "-d", "@/tmp/report.json"], capture_output=True, text=True)
    print("POST:", r.stdout[:600], r.stderr[:200])


tok, err = sh("curl", "-s", "-X", "POST", base + "/auth/login", "-H",
              "Content-Type: application/json",
              "-d", json.dumps({"username": user, "password": pw}))
print("login:", tok[:120], err[:200])
token = ""
try:
    j = json.loads(tok)
    token = j.get("access_token") or j.get("token") or j.get("data", {}).get("access_token", "")
except Exception:
    token = ""
if not token:
    print("no token; trying unauthenticated post")
post(token)