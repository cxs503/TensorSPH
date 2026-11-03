"""Oscillating 2-D drop: the *dynamic* (frequency-domain) test of surface tension.

The static Young-Laplace gate measures the pressure jump ``dp = sigma/R``; it
cannot tell whether the surface-tension force responds with the right
*amplitude in time*.  A slightly deformed inviscid drop oscillates about the
circle, and linear theory fixes its frequency exactly:

    omega^2 = m (m^2 - 1) sigma / (rho R^3)        (2-D cylinder, mode m)

For the quadrupole (ellipse) mode ``m = 2`` this is

    omega^2 = 6 sigma / (rho R^3),      T = 2 pi sqrt(rho R^3 / (6 sigma)).

References
----------
* Rayleigh, Lord (1879), Proc. London Math. Soc. 10, 4 -- capillary
  oscillations of a drop / cylinder.
* Lamb, H. (1932), *Hydrodynamics*, 6th ed., Art. 267 (2-D normal modes).
* Adami, Hu & Adams (2010), JCP 229, 5011 -- surface-tension time-step and
  spurious-current scale ``sqrt(sigma / (rho R))`` used here for reference.

Why this replaces ``benchmarks/capillary_wave``
----------------------------------------------
The plane capillary-wave gate in this repo is dead by construction (see
``docs/verification_droplet_oscillation.md`` §6): it reads ``max|y - y0|``,
which is a positive-definite quantity, so its zero-crossing period finder can
never fire; it integrates 0.2-0.5 s of a 4.2 s period; and with ``gravity=0``,
no periodic boundary and a one-wavelength domain the flat pool contracts under
surface tension instead of oscillating.  A closed drop needs no gravity and no
periodic boundary, so it is well posed on exactly the solver that exists.

Observable
----------
The quadrupole moment of the particle distribution,

    q(t) = (sum m_i (x_i^2 - y_i^2)) / (sum m_i (x_i^2 + y_i^2)),

which for an interface ``r(theta) = R (1 + eps cos 2 theta)`` equals ``2 eps``
(area weighted, to first order in ``eps``).  It uses *every* particle -- no
surface detection, no angular binning, no shell particle -- so the measurement
does not inherit the free-surface detection noise that the static gate fights.
The period is read from the zero crossings of ``q``.

Usage
-----
    python benchmarks/droplet_oscillation/run.py --dx 0.001 --periods 1.6 \\
        --json-out results.json --out-dir <dir>

``--quick`` runs a short, coarse version for CI.  Exit status is non-zero when
the frequency error exceeds ``--max-freq-error`` (default 0.15, the platform
acceptance limit).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from tensorsph.droplet_impact import Config, rhs, stable_dt
from tensorsph.neighbor_search import PairListCache

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))


# --------------------------------------------------------------------------- #
# reference solution
# --------------------------------------------------------------------------- #
def rayleigh_omega(sigma: float, rho: float, radius: float, mode: int = 2) -> float:
    """Linear frequency of mode ``m`` of an inviscid 2-D drop.

    ``omega^2 = m (m^2 - 1) sigma / (rho R^3)``; ``m = 2`` gives
    ``6 sigma / (rho R^3)``.
    """
    return math.sqrt(mode * (mode * mode - 1) * sigma / (rho * radius ** 3))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hexagonal_drop(radius: float, dx: float, rho0: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Hexagonally packed circular drop of exactly ``rho0`` mean density.

    Hex packing gives six nearest neighbours instead of four, which halves the
    lattice bias of the SPH gradients.  The lattice constant is chosen so that
    ``mass / (dx * dy) = rho0``.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    mass = torch.tensor(rho0 * dx * dy, dtype=torch.float64)
    pts: list[tuple[float, float]] = []
    row = 0
    y = -radius
    while y <= radius:
        x = -radius + (0.5 * dx if row % 2 else 0.0)
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        row += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


def elliptically_deform(pos: torch.Tensor, eps: float) -> torch.Tensor:
    """Stretch ``x`` by ``(1+eps)`` and ``y`` by ``1/(1+eps)``.

    The affine map keeps the area (``det = 1``) and therefore the mean density
    exactly, so the initial state carries no compressional energy -- the
    failure mode that invalidated the first generation of drop initial
    conditions in this repo (a lattice discretisation of a deformed circle is
    *not* volume preserving, which injected a ~2.4 kPa spurious core pressure).
    """
    out = pos.clone()
    out[:, 0] *= 1.0 + eps
    out[:, 1] /= 1.0 + eps
    return out


def quadrupole_moment(pos: torch.Tensor, mass: torch.Tensor) -> float:
    """``q = (sum m (x^2 - y^2)) / (sum m (x^2 + y^2))`` -- the m=2 amplitude."""
    x2 = pos[:, 0] * pos[:, 0]
    y2 = pos[:, 1] * pos[:, 1]
    return float((mass * (x2 - y2)).sum() / (mass * (x2 + y2)).sum())


def radius_percentile(pos: torch.Tensor, mass: torch.Tensor, q: float = 0.95) -> float:
    """Mass-weighted radius percentile, used to watch the drop drift."""
    r = torch.linalg.vector_norm(pos, dim=1)
    order = torch.argsort(r)
    cum = torch.cumsum(mass[order], 0) / mass.sum()
    idx = int(torch.searchsorted(cum, torch.tensor(q, dtype=cum.dtype)))
    return float(r[order][min(idx, r.numel() - 1)])


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #
def run_oscillation(cfg: Config, radius: float, eps: float, periods: float,
                    samples_per_period: int = 400):
    """Integrate a deformed drop with zero body force and no walls.

    Returns ``(times, moments, aux)`` where ``moments`` is ``q(t)``.
    """
    omega_ref = rayleigh_omega(cfg.surface_tension_coefficient, cfg.rho0, radius)
    t_ref = 2.0 * math.pi / omega_ref

    dt = min(float(stable_dt(cfg)), t_ref / samples_per_period)
    steps = int(round(periods * t_ref / dt))
    every = max(1, samples_per_period // 40)

    pos, mass = hexagonal_drop(radius, cfg.dx, cfg.rho0)
    n = int(pos.shape[0])
    pos = elliptically_deform(pos, eps)
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h, fluid_count=n)

    q0 = quadrupole_moment(pos, mass)
    r0 = radius_percentile(pos, mass)

    times: list[float] = []
    moments: list[float] = []
    t = 0.0
    started = time.perf_counter()
    for step in range(steps + 1):
        pairs, _ = cache.get(pos)
        _density, _pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs,
                                       wall=None, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        if step % every == 0 or step == steps:
            times.append(t)
            moments.append(quadrupole_moment(pos, mass))
        vel = vel + dt * acc
        pos = pos + dt * vel
        t += dt
    elapsed = time.perf_counter() - started

    aux = {
        "particles": n,
        "dt": dt,
        "steps": steps,
        "wall_clock_s": elapsed,
        "q_initial": q0,
        "eps_initial": 0.5 * q0,
        "r95_initial": r0,
        "r95_final": radius_percentile(pos, mass),
        "vmax_final": float(torch.linalg.vector_norm(vel, dim=1).max()),
    }
    return times, moments, aux


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def period_from_crossings(times: list[float], values: list[float]) -> dict:
    """Period from the zero crossings of a decaying oscillation.

    The signal is de-trended by its mean first (the drop's slow swelling moves
    the mid-level), then crossings are interpolated linearly and the period is
    the mean spacing between *successive same-direction* crossings, i.e. the
    mean distance between crossings ``i`` and ``i+2``.
    """
    if len(values) < 8:
        return {"period": None, "crossings": [], "reason": "too few samples"}
    # The oscillation decays towards a residual level (the drop's slow swelling
    # moves it), so the crossing level is the *tail* mean rather than the mean
    # of the whole window: subtracting the global mean of a damped signal shifts
    # every crossing towards the end of the record and inflates the period.
    tail = values[-max(3, len(values) // 7):]
    level = sum(tail) / len(tail)
    cross: list[float] = []
    for i in range(1, len(values)):
        a, b = values[i - 1] - level, values[i] - level
        if a == 0.0:
            continue
        if a * b < 0.0:
            t0, t1 = times[i - 1], times[i]
            cross.append(t0 - a * (t1 - t0) / (b - a))
    if len(cross) < 3:
        return {"period": None, "crossings": cross,
                "reason": f"only {len(cross)} crossings; need 3 for a period"}
    intervals = [cross[i + 2] - cross[i] for i in range(len(cross) - 2)]
    return {"period": sum(intervals) / len(intervals),
            "crossings": cross,
            "level": level,
            "half_periods": [cross[i + 1] - cross[i] for i in range(len(cross) - 1)]}


def decay_rate(times: list[float], values: list[float]) -> float | None:
    """Amplitude decay constant from the successive extrema of the signal."""
    tail = values[-max(3, len(values) // 7):]
    level = sum(tail) / len(tail)
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(values) - 1):
        a, b, c = (values[i - 1] - level, values[i] - level, values[i + 1] - level)
        if (b - a) * (c - b) < 0.0:
            peaks.append((times[i], abs(b)))
    if len(peaks) < 2:
        return None
    t0, a0 = peaks[0]
    t1, a1 = peaks[-1]
    if a1 <= 0.0 or t1 <= t0:
        return None
    return math.log(a0 / a1) / (t1 - t0)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figure(times, moments, period, omega_ref, aux, out_png: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax.plot(times, moments, "-", lw=1.2, color="#1f4e9c", label="SPH $q(t)$")
    q0 = aux["q_initial"]
    ax.plot(times, [q0 * math.cos(omega_ref * t) for t in times], "--",
            lw=1.1, color="#c0392b",
            label=r"theory $q_0\cos(\omega t)$, $\omega^2=6\sigma/\rho R^3$")
    if period:
        ax.set_title(f"2-D drop, $m=2$:  $T_{{SPH}}$={period:.4f} s, "
                     f"$T_{{theory}}$={2*math.pi/omega_ref:.4f} s")
    else:
        ax.set_title("2-D drop, $m=2$ (no period resolved)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$q=(\sum m(x^2-y^2))/(\sum m(x^2+y^2))$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # amplitude envelope: |q| at successive extrema
    level = sum(moments) / len(moments)
    peaks = [(times[i], abs(moments[i] - level)) for i in range(1, len(moments) - 1)
             if (moments[i] - level - moments[i - 1] + level)
             * (moments[i + 1] - level - moments[i] + level) < 0.0]
    if peaks:
        ax2.semilogy([p[0] for p in peaks], [p[1] for p in peaks], "o-",
                     ms=4, lw=1.0, color="#117a65")
        ax2.set_title("envelope: amplitude decay (viscosity / numerical damping)")
    else:
        ax2.set_title("amplitude envelope (too few extrema)")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel(r"$|q - \langle q\rangle|$ at extrema")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_case(args) -> Config:
    cfg = Config(
        dx=args.dx,
        h_factor=1.3,
        rho0=1000.0,
        sound_speed=args.sound_speed,
        gamma=7.0,
        gravity=0.0,
        viscosity=args.viscosity,
        density_diffusion=0.0,
        surface_tension_coefficient=args.sigma,
        device="cpu",
        boundary="clamp",
        adaptive_dt=False,
        csf_scheme="consistent",
    )
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radius", type=float, default=0.02, help="undisturbed radius [m]")
    p.add_argument("--dx", type=float, default=0.001, help="particle spacing [m]")
    p.add_argument("--eps", type=float, default=0.10, help="initial m=2 deformation")
    p.add_argument("--sigma", type=float, default=0.072, help="surface tension [N/m]")
    p.add_argument("--sound-speed", type=float, default=1.0,
                   help="artificial sound speed [m/s]; see the doc for why it is low")
    p.add_argument("--periods", type=float, default=1.6,
                   help="simulated time in theoretical periods")
    p.add_argument("--samples-per-period", type=int, default=400)
    p.add_argument("--viscosity", type=float, default=0.004,
                   help="kinematic viscosity [m^2/s]; 0 is tensile-unstable here")
    p.add_argument("--max-freq-error", type=float, default=0.15)
    p.add_argument("--quick", action="store_true", help="coarse short run for CI")
    p.add_argument("--json-out", default=None)
    p.add_argument("--out-dir", default=None, help="directory for the JSON + figure")
    p.add_argument("--write-doc", default=None,
                   help="write a markdown verification note with the numbers injected")
    args = p.parse_args()

    if args.quick:
        args.dx = 0.002
        args.periods = 1.15
        args.samples_per_period = 120

    out_dir = args.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_case(args)
    omega_ref = rayleigh_omega(args.sigma, cfg.rho0, args.radius)
    t_ref = 2.0 * math.pi / omega_ref
    v_ref = math.sqrt(args.sigma / (cfg.rho0 * args.radius))

    print(f"theory: omega = {omega_ref:.4f} rad/s, T = {t_ref:.4f} s, "
          f"v_ref = sqrt(sigma/rho R) = {v_ref:.4f} m/s")

    times, moments, aux = run_oscillation(cfg, args.radius, args.eps, args.periods,
                                         args.samples_per_period)
    fit = period_from_crossings(times, moments)
    lam = decay_rate(times, moments)

    error = None
    passed = False
    corrected = None
    if fit["period"]:
        error = abs(fit["period"] - t_ref) / t_ref
        passed = error < args.max_freq_error
        # A damped oscillator crosses zero at the *damped* period
        # T_d = 2 pi / sqrt(omega_0^2 - gamma^2); undo that with the measured
        # envelope decay rate so the number can be compared with the inviscid
        # reference instead of against a damping-dependent constant.
        if lam and lam > 0.0:
            omega_d = 2.0 * math.pi / fit["period"]
            if omega_d > lam:
                corrected = 2.0 * math.pi / math.sqrt(omega_d ** 2 + lam ** 2)
                error = abs(corrected - t_ref) / t_ref
                passed = error < args.max_freq_error

    result = {
        "benchmark": "droplet_oscillation",
        "kind": "dynamic surface-tension gate (frequency domain)",
        "setup": {
            "radius_m": args.radius,
            "dx_m": args.dx,
            "radius_over_dx": args.radius / args.dx,
            "eps_initial": args.eps,
            "sigma_N_per_m": args.sigma,
            "rho0_kg_per_m3": cfg.rho0,
            "sound_speed_m_per_s": cfg.sound_speed,
            "gravity": cfg.gravity,
            "viscosity": cfg.viscosity,
            "csf_scheme": cfg.csf_scheme,
            "periods_simulated": args.periods,
        },
        "reference": {
            "source": "Rayleigh (1879); Lamb (1932) Art. 267, 2-D mode m",
            "formula": "omega^2 = m(m^2-1) sigma / (rho R^3), m=2 -> 6 sigma/(rho R^3)",
            "omega_rad_s": omega_ref,
            "period_s": t_ref,
            "velocity_scale_sqrt_sigma_over_rhoR": v_ref,
        },
        "measured": {
            "period_s": fit["period"],
            "period_s_inviscid_corrected": corrected,
            "crossing_level": fit.get("level"),
            "damping_ratio": (lam / (2.0 * math.pi / fit["period"])
                              if (lam and fit["period"]) else None),
            "frequency_error": error,
            "crossings_s": fit["crossings"],
            "first_quarter_period_s": (fit["crossings"][0]
                                       if fit["crossings"] else None),
            "omega_from_first_crossing": (math.pi / (2.0 * fit["crossings"][0])
                                          if fit["crossings"] else None),
            "error_from_first_crossing": (
                abs(math.pi / (2.0 * fit["crossings"][0]) - omega_ref) / omega_ref
                if fit["crossings"] else None),
            "half_periods_s": fit.get("half_periods"),
            "half_period_spread": (max(fit["half_periods"]) - min(fit["half_periods"]))
            / (2.0 * t_ref) if fit.get("half_periods") else None,
            "decay_rate_per_s": lam,
            "dimensionless_vmax_final": aux["vmax_final"] / v_ref,
            **aux,
            "samples": len(times),
        },
        "pass": passed,
        "threshold": args.max_freq_error,
    }

    json_path = args.json_out or os.path.join(out_dir, "results.json")
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"period: SPH {result['measured']['period_s']} s vs theory {t_ref:.4f} s "
          f"-> error {None if error is None else f'{error*100:.2f}%'} "
          f"({'PASS' if passed else 'FAIL'})")
    print(f"q(t) samples stored: {len(times)}; wall clock {aux['wall_clock_s']:.1f} s")

    png = make_figure(times, moments, fit["period"], omega_ref, aux,
                      os.path.join(out_dir, "droplet_oscillation.png"))
    print(f"figure: {png}")

    with open(os.path.join(out_dir, "trace.json"), "w") as fh:
        json.dump({"t": times, "q": moments}, fh)

    if args.write_doc:
        write_doc(args.write_doc, result)
        print(f"doc: {args.write_doc}")

    return 0 if passed else 1


DOC_TEMPLATE = """# 2-D 液滴振荡（Rayleigh m=2）动态表面张力验证

本文件由 `benchmarks/droplet_oscillation/run.py --write-doc` 生成，所有数字由
`results.json` 程序化注入，无手抄。

## 1. 判据

略去重力、无固壁的二维液滴，把界面扰成 `r(θ)=R(1+ε cos2θ)` 后自由振荡。线性理论
（Rayleigh 1879；Lamb 1932 Art. 267）给出模式 `m` 的角频率

    ω² = m(m²−1)σ/(ρR³)  →  m=2:  ω² = 6σ/(ρR³)

观测量取整个粒子分布的四极矩 `q = Σm(x²−y²)/Σm(x²+y²)`，对上述界面形状
`q = 2ε`（面积加权，一阶）。它用到**每一个**粒子，不做自由面识别、不做角向分箱，
因此不继承静态 Young-Laplace 判据里那套界面探测噪声。周期由 `q(t)` 的零点插值读出。

## 2. 算例与参数

| 量 | 值 |
|---|---|
| 半径 R | {radius_m} m |
| 粒子间距 dx | {dx_m} m （R/dx = {radius_over_dx:.0f}） |
| 初始变形 ε | {eps_initial} |
| 表面张力 σ | {sigma_N_per_m} N/m |
| 密度 ρ0 | {rho0_kg_per_m3} kg/m³ |
| 人工声速 c0 | {sound_speed_m_per_s} m/s |
| 重力 / 粘性 | {gravity} / {viscosity} |
| CSF 离散 | {csf_scheme} |
| 积分时长 | {periods_simulated} 个理论周期 |
| 粒子数 | {particles} |
| 时间步 | {dt:.3e} s × {steps} 步（固定步长，非自适应） |

初值用仿射映射 `x→(1+ε)x, y→y/(1+ε)` 施加变形：**面积严格守恒**，因此不引入压缩能
（首代液滴初值用"圆内格子"，离散单位分解偏高 ~0.6%，制造了数 kPa 的虚假内压，本算例
刻意避开这条老路）。

## 3. 结果

| 判据 | 参考值 | 实测 | 误差 | 判定（<15%） |
|---|---|---|---|---|
| 振荡周期 T | {period_theory:.4f} s | {period_measured} s | {error_str} | **{verdict}** |

独立频率代理（不依赖零交叉配对）：首个零交叉给出四分之一周期
`ω ≈ π/(2 t_cross1)`。本例 `t_cross1` = {first_cross} s → ω = {omega_first} rad/s，
对理论 {omega_theory:.3f} rad/s 的误差 {error_first}。

辅助量（用于判断这个周期是否可信，而不是拿它凑数）：

* 相邻半周期离散度（极差/2T）：{half_period_spread}；实测半周期序列 {half_periods} —— 线性振荡应趋于 0；
  偏大说明信号不纯（形状漂移或伪流）。
* 末态 `max|v|/√(σ/ρR)`：{dimensionless_vmax_final:.4f}
  （Adami 2010 的伪流参考尺度；静态判据的门槛是 0.1）。
* 半径 r95：{r95_initial:.5f} → {r95_final:.5f} m（漂移 {drift_pct:+.2f}%）。
* 包络衰减率 λ：{decay} 1/s（无粘理论应为 0；实测值给出数值耗散的量级）。
* 墙钟：{wall_clock_s:.1f} s（{particles} 粒子 × {steps} 步）。

## 4. 为什么这不是静态判据的重复

静态 Young-Laplace 判据量的是 `Δp = σ/R`，只检验**力的空间平衡**；它即使通过，也无法
排除"力幅值对、但时间响应错"（相位滞后、质量矩阵错、伪流扭曲）这一类缺陷。本判据量的是
**同一个力的时间响应频率**。两者互补：动态判据的频率对 σ_eff 的敏感度是 `ω ∝ √σ`，
即 σ 偏 13% 只会在频率上表现为 6.5%——它检验的是响应关系，不是幅值本身。

## 5. 局限

* 单一分辨率、单一 ε：没有做 ε→0 的外推，也没有做 dx 收敛阶梯；本文件只回答"这一档
  分辨率下频率是否落在 15% 内"，不外推收敛阶。
* 二维柱体（不是三维球）：`m=2` 的 2D 系数是 6σ/(ρR³)，**不要**与三维 Rayleigh
  的 `l=2` 系数 8σ/(ρR³) 混用。
* 积分只覆盖 {periods_simulated} 个周期：m2837 发现的"液滴在 10 t_σ 内膨胀 ~50%"
  这类慢漂移在本时间窗内只体现为 r95 的小幅变化，未做长时程检验。
* 人工声速被压到 {sound_speed_m_per_s} m/s（Mach ≈ {mach:.3f}）以换取步数；声速敏感性
  未做扫描，见 §6。

## 6. 已知空洞：旧的平面毛细波判据不可判

`benchmarks/capillary_wave/run.py` 是一个**永不可能给出测量值**的闸门，三处结构性缺陷：

1. `track_amplitude` 返回 `max|y − y0|`，恒非负；而 `find_period` 只在符号变化处找零点，
   所以周期永远返回 `None`、`pass` 永远 `False`。这是量具本身的符号错误。
2. 默认 `--seconds 0.5`，而 σ=0.072、λ=0.2 m 对应的理论周期是 4.2 s：积分时长只有
   周期的 12%，即使量具修好也读不到一个周期。
3. 物理设定不成立：`gravity=0` + `pool_width` 取成一个波长 + `boundary="clamp"`（且
   `run_capillary_wave` 从未调用边界处理、也没有周期边界）。无重力时自由面的唯一恢复力
   就是表面张力，一块平板液池会朝圆收缩；而周期边界缺失使波长方向的平移不变性破缺。
   平面毛细波因此在现有求解器上**不可良定**，这是把它换成闭合液滴（无需重力、无需周期
   边界）的原因，而不是"换个算例凑通过"。

如需恢复平面毛细波判据，先决条件是求解器支持 x 方向周期边界（或重力占优的
Bond 数区间，但那时 σ 的贡献只占 ω² 的百分之几，判据失去分辨力）。
"""


def write_doc(path: str, result: dict) -> None:
    s = result["setup"]
    m = result["measured"]
    ref = result["reference"]
    period = m["period_s"]
    err = m["frequency_error"]
    drift = ((m["r95_final"] - m["r95_initial"]) / m["r95_initial"] * 100.0
             if m["r95_initial"] else float("nan"))
    mach = m["vmax_final"] / s["sound_speed_m_per_s"] if s["sound_speed_m_per_s"] else float("nan")
    text = DOC_TEMPLATE.format(
        **s,
        particles=m["particles"],
        dt=m["dt"],
        steps=m["steps"],
        period_theory=ref["period_s"],
        period_measured=f"{period:.4f}" if period else "未解析",
        error_str=f"{err*100:+.2f}%" if err is not None else "—",
        verdict=("PASS <15%" if result["pass"] else "FAIL"),
        first_cross=(f"{m['first_quarter_period_s']:.5f}"
                     if m.get("first_quarter_period_s") else "—"),
        omega_first=(f"{m['omega_from_first_crossing']:.3f}"
                     if m.get("omega_from_first_crossing") else "—"),
        omega_theory=ref["omega_rad_s"],
        error_first=(f"{m['error_from_first_crossing']*100:+.2f}%"
                     if m.get("error_from_first_crossing") is not None else "—"),
        half_periods=(f"{[round(h,5) for h in m['half_periods_s']]}"
                      if m.get("half_periods_s") else "—"),
        half_period_spread=(f"{m['half_period_spread']*100:.2f}%"
                            if m.get("half_period_spread") is not None else "—"),
        dimensionless_vmax_final=m["dimensionless_vmax_final"],
        r95_initial=m["r95_initial"],
        r95_final=m["r95_final"],
        drift_pct=drift,
        decay=(f"{m['decay_rate_per_s']:.4f}" if m["decay_rate_per_s"] is not None else "—"),
        wall_clock_s=m["wall_clock_s"],
        mach=mach,
    )
    with open(path, "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    raise SystemExit(main())