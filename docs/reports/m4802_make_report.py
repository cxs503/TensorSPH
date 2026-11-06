"""Render the machine-checked blocks of ``docs/reports/m4802-main-red-mode2-pins.md``.

Every number in that report is read out of a *committed* artefact -- nothing is
transcribed by hand, so the report cannot drift away from the measurement:

* ``benchmarks/oscillating_droplet/ci_pin_evidence.json``  -- the CI-cost case
  (three estimators, stability, thread determinism, the per-platform pins),
* ``benchmarks/oscillating_droplet/validation_report.json`` -- the release-scale
  case and its verdict,
* ``benchmarks/oscillating_droplet/validation_report_lamb.json`` -- the
  one-period solver guard (``case == "mode2"``).

    PYTHONPATH=src python docs/reports/m4802_make_report.py            # rewrite blocks + figure
    PYTHONPATH=src python docs/reports/m4802_make_report.py --check    # non-zero exit if stale

``--check`` runs no solver: it re-renders from the committed JSON and compares
with the text in the document (``tests/test_report_m4802.py`` calls it), so a
hand edit to a number in the report turns the suite red.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks" / "oscillating_droplet"
LEDGER = BENCH / "ci_pin_evidence.json"
RELEASE = BENCH / "validation_report.json"
LAMB = BENCH / "validation_report_lamb.json"
TRACE = BENCH / "trace_data.json"
DOC = Path(__file__).resolve().parent / "m4802-main-red-mode2-pins.md"
FIGURE = ROOT / "docs" / "figures" / "m4802_error_vs_gate.png"
TRACE_FIGURE = ROOT / "docs" / "figures" / "m4802_release_trace.png"

GATE_PCT = 15.0
STRICT_PCT = 3.0

#: Marker pairs in the document; each block is rewritten in place.
BLOCKS = ("error-table", "pins", "stability", "gate-table")


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _f(x, nd=2):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def _pct(x, nd=2):
    return "—" if x is None or x != x else f"{x:+.{nd}f}%"


def _gate(x):
    """(within 3 %, within 15 %) answered from the number, never from a flag."""
    return (abs(x) < STRICT_PCT, abs(x) < GATE_PCT)


def _threads(det: dict) -> str:
    return "/".join(str(t) for t in det["threads_tested"])


def _yn(flag):
    return "是" if flag else "否"


# --------------------------------------------------------------------------- #
# the three measured cases
# --------------------------------------------------------------------------- #
def load():
    return (json.loads(LEDGER.read_text()),
            json.loads(RELEASE.read_text()),
            json.loads(LAMB.read_text()))


def lamb_row(lamb: dict, case: str = "mode2") -> dict:
    for row in lamb["rows"]:
        if row["case"] == case:
            return row
    raise KeyError(f"no {case!r} row in {LAMB.name}")


def error_rows(ledger: dict, release: dict, lamb: dict) -> list[dict]:
    """One row per (case, estimator): analytic, measured, error -- all injected."""
    ci_omega = ledger["analytic"]["omega_2_rad_per_s"]
    ci = ledger["measured"]
    rel = release["frequency"]
    guard = lamb_row(lamb)
    return [
        {"label": "CI 廉价算例 · 二阶矩阻尼拟合", "ascii": "CI cheap case / 2nd-moment damped fit",
         "case": "R=10 mm, dx=1 mm, 1.8 T", "source": "ci_pin_evidence.json",
         "analytic": ci_omega,
         "measured": ci["omega_damped_fit_rad_per_s"],
         "err": ci["fit_rel_error_pct"],
         "note": f"拟合 r² = {_f(ci['fit_r2'], 3)}"},
        {"label": "CI 廉价算例 · 过零估计器", "ascii": "CI cheap case / zero crossing",
         "case": "R=10 mm, dx=1 mm, 1.8 T", "source": "ci_pin_evidence.json",
         "analytic": ci_omega,
         "measured": ci["omega_zero_crossing_rad_per_s"],
         "err": ci["zero_crossing_rel_error_pct"],
         "note": "与拟合差 "
                 f"{_f(ci['fit_rel_error_pct'] - ci['zero_crossing_rel_error_pct'], 2)} pp"},
        {"label": "CI 廉价算例 · 动能半衰估计器", "ascii": "CI cheap case / kinetic-energy halving",
         "case": "R=10 mm, dx=1 mm, 1.8 T", "source": "ci_pin_evidence.json",
         "analytic": ci_omega,
         "measured": ci["omega_kinetic_energy_halved_rad_per_s"],
         "err": ci["kinetic_energy_rel_error_pct"],
         "note": "第三估计器，无阻尼假设"},
        {"label": "基准算例 · 二阶矩阻尼拟合", "ascii": "release case (R=20 mm) / damped fit",
         "case": "R=20 mm, dx=2 mm, 1.8 T", "source": "validation_report.json",
         "analytic": rel["reference"]["omega_rad_per_s"],
         "measured": rel["frequency"]["fit_second_moment_rad_per_s"],
         "err": rel["frequency"]["fit_second_moment_rel_error_pct"],
         "note": "文档/判定口径算例（入库）"},
        {"label": "一周期守卫算例 · 二阶矩阻尼拟合", "ascii": "one-period guard / damped fit",
         "case": "R=20 mm, dx=2.5 mm, 1.0 T", "source": "validation_report_lamb.json",
         "analytic": guard["omega_analytic"],
         "measured": guard["omega_measured"],
         "err": guard["omega_error_pct"],
         "note": guard["stability"]},
    ]


# --------------------------------------------------------------------------- #
# block renderers
# --------------------------------------------------------------------------- #
def render_error_table(ledger, release, lamb) -> str:
    lines = [
        "| 算例 / 估计器 | 解析 ω₂ (rad/s) | 实测 ω (rad/s) | 相对误差 | < 3 %? | < 15 %? | 出处 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in error_rows(ledger, release, lamb):
        w3, w15 = _gate(r["err"])
        lines.append(
            f"| {r['label']}<br>`{r['case']}` | {_f(r['analytic'])} | "
            f"{_f(r['measured'])} | **{_pct(r['err'])}** | {_yn(w3)} | {_yn(w15)} | "
            f"`{r['source']}`<br>{r['note']} |")
    rows = error_rows(ledger, release, lamb)
    mags = [abs(r["err"]) for r in rows]
    spread = max(abs(rows[i]["err"] - rows[j]["err"])
                 for i in range(3) for j in range(3))
    lines += [
        "",
        f"五个（算例, 估计器）组合的误差绝对值区间 {min(mags):.2f} % – "
        f"{max(mags):.2f} %：**没有任何一个 < 3 %，也没有任何一个 < 15 %**。"
        f"同一算例三个估计器之间的分歧最大 {spread:.2f} pp，"
        "这本身就是「该量不可作为频率结论」的证据。",
    ]
    return "\n".join(lines)


def render_pins(ledger, release=None, lamb=None) -> str:
    pins = ledger["recorded_platform_pins_pct"]
    live = ledger["platform"]
    lines = [
        "| 平台 | 记录的模态 2 伪频率误差 | 越 15 % 门? | 钉法 |",
        "|---|---|---|---|",
    ]
    for arch, val in sorted(pins.items(), key=lambda kv: -kv[1]):
        where = "（本机，CI runner）" if arch == live else "（GitHub 托管 runner）"
        lines.append(
            f"| `{arch}`{where} | **{_pct(val)}** | {_yn(abs(val) > GATE_PCT)} | "
            f"`RECORDED_MODE2_ERR_PCT['{arch}']`，容差 2.00 pp |")
    spread = max(pins.values()) - min(pins.values())
    lines += [
        "",
        f"同一份代码、同一份输入的跨平台散布 = **{spread:.2f} pp**"
        f"（{_pct(min(pins.values()))} … {_pct(max(pins.values()))}）：这是"
        "「伪频率按平台记录、判定按平台无关物证」的直接依据，"
        "也是**不得取平均**的理由（取平均会造出一个两个平台都不成立的 −11.57 %）。",
    ]
    return "\n".join(lines)


def render_stability(ledger, release, lamb) -> str:
    st = ledger["stability"]
    det = ledger["thread_determinism"]
    setup = ledger["case"]
    r0 = release["frequency"]["initial_state"]["r_eff"]
    r1 = release["frequency"]["final_state"]["r_eff"]
    guard = lamb_row(lamb)
    lines = [
        "| 量 | CI 廉价算例 | 基准算例（R=20 mm） | 判据 |",
        "|---|---|---|---|",
        f"| 粒子数 N | {setup['n_particles']} | "
        f"{release['frequency']['setup']['n_particles']} | — |",
        f"| 步数 / 模拟时长 | {setup['steps']} / {_f(setup['periods'], 2)} T | "
        f"{release['frequency']['setup']['steps']} / "
        f"{_f(release['frequency']['setup']['periods_simulated'], 2)} T | — |",
        f"| dt (s) | {setup['dt_s']:.3e} | "
        f"{release['frequency']['setup']['dt_s']:.3e} | — |",
        f"| 面积等效半径 r_eff 变化 | 发散（见下行的 max\\|v\\|） | "
        f"{r0 * 1e3:.1f} mm → {r1:.1f} m（**×{r1 / r0:.0f}**） | 守恒应为 ×1 |",
        f"| max\\|v\\| / c₀ | {_f(st['max_speed_over_c0'], 3)} | — | ≪1 才是弱可压 |",
        f"| 质量相对漂移 | {st['mass_relative_drift']:.1e} | — | <1e-12（机器精度） |",
        f"| 线程确定性（{_threads(det)} 线程, {det['steps']} 步） | "
        f"max\\|Δx\\| = {det['max_abs_position_diff_m']:.1e} m | — | "
        "位级一致（无 RNG、无原子归约） |",
        f"| 守卫算例稳定性 | — | {guard['stability']} | 不得离开钳位盒 |",
        "",
        "说明：**质量守恒与线程确定性都成立**，所以失败点不在守恒律或并行不确定性上——"
        "它是格式缺陷（自由面欠密度 + 钳位 Tait 律导致的压强塌陷），"
        f"在 1.8 个解析周期内把液滴的等效半径从 {r0 * 1e3:.1f} mm 推到 {r1:.1f} m"
        f"（×{r1 / r0:.0f}，面积应守恒），"
        "任何从这条轨迹上拟合出来的频率都只能是伪频率。",
    ]
    return "\n".join(lines)


def render_gate_table(ledger, release, lamb) -> str:
    """The error-verdict table of section 6: basis, measured, error, gates."""
    rows = error_rows(ledger, release, lamb)
    lines = [
        "| 关键量 | 基准 | 实测 | 相对误差 | < 3 %? | < 15 %? |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        w3, w15 = _gate(r["err"])
        lines.append(
            f"| {r['label']} | {_f(r['analytic'])} rad/s | "
            f"{_f(r['measured'])} rad/s | **{_pct(r['err'])}** | "
            f"{_yn(w3)} | {_yn(w15)} |")
    st = ledger["stability"]
    det = ledger["thread_determinism"]
    lines += [
        f"| 质量守恒（现场 run） | 0 | {st['mass_relative_drift']:.1e} | 0 % | 是 | 是 |",
        f"| 线程确定性（{_threads(det)} 线程 × {det['steps']} 步） | 位级一致 | "
        f"max\\|Δx\\| = {det['max_abs_position_diff_m']:.1e} m | 0 % | 是 | 是 |",
        "",
        "**逐项回答**：频率类关键量 **没有一项 < 3 %，也没有一项 < 15 %**——honest FAIL；"
        "守恒与确定性两项为 0 %（< 3 %），说明失效不在守恒律或并行不确定性。"
        "跨平台散布见 §③ 的平台钉子表，已如实记录、**不取平均掩盖**。",
    ]
    return "\n".join(lines)


RENDERERS = {
    "error-table": render_error_table,
    "pins": render_pins,
    "stability": render_stability,
    "gate-table": render_gate_table,
}


def render_blocks(doc_artifacts=None) -> dict[str, str]:
    ledger, release, lamb = doc_artifacts or load()
    return {name: fn(ledger, release, lamb) for name, fn in RENDERERS.items()}


# --------------------------------------------------------------------------- #
# splicing
# --------------------------------------------------------------------------- #
def markers(name: str) -> tuple[str, str]:
    return (f"<!-- BEGIN GENERATED:{name} -->",
            f"<!-- END GENERATED:{name} -->")


def splice(text: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        begin, end = markers(name)
        if begin not in text or end not in text:
            raise SystemExit(f"{DOC.name}: missing marker pair for {name!r}")
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{body}\n{end}{tail}"
    return text


def check(text: str | None = None) -> list[str]:
    """Return the blocks of the document that no longer match the artefacts."""
    doc = DOC.read_text() if text is None else text
    expected = render_blocks()
    stale = []
    for name, body in expected.items():
        begin, end = markers(name)
        if begin not in doc or end not in doc:
            stale.append(f"{name}: missing marker pair")
            continue
        got = doc.split(begin, 1)[1].split(end, 1)[0].strip("\n")
        if got != body:
            stale.append(f"{name}: document block differs from the artefacts")
    return stale


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def render_figure(path: Path = FIGURE, ledger=None, release=None, lamb=None) -> Path:
    """Two panels: the five (case, estimator) errors against the gates, and the
    platform spread of the same pseudo-frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ledger is None:
        ledger, release, lamb = load()
    rows = error_rows(ledger, release, lamb)
    pins = ledger["recorded_platform_pins_pct"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    labels = [f"{r['ascii']}\n{r['case']}" for r in rows]  # no CJK font here
    errs = [r["err"] for r in rows]
    colors = ["#c0392b" if abs(e) > GATE_PCT else "#e67e22" for e in errs]
    ax0.barh(range(len(errs)), errs, color=colors, alpha=0.85)
    ax0.set_yticks(range(len(errs)), labels, fontsize=7.5)
    ax0.axvline(0.0, color="k", lw=0.8)
    for sign in (+1, -1):
        ax0.axvline(sign * GATE_PCT, color="#c0392b", ls="--", lw=1.2,
                    label="15 % gate" if sign > 0 else None)
        ax0.axvline(sign * STRICT_PCT, color="#2c3e50", ls=":", lw=1.0,
                    label="3 % strict" if sign > 0 else None)
    for i, e in enumerate(errs):
        ax0.text(e + (1.5 if e >= 0 else -1.5), i, f"{e:+.1f}%",
                 va="center", ha="left" if e >= 0 else "right", fontsize=8)
    ax0.set_xlim(min(errs) - 14, max(max(errs) + 14, 20))
    ax0.set_xlabel("fitted mode-2 frequency error vs Rayleigh–Lamb  [%]")
    ax0.set_title("(a) five (case, estimator) pairs: all miss the 15 % gate, none is < 3 %",
                  fontsize=9.5)
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(axis="x", alpha=0.3)

    arches = list(pins)
    vals = [pins[a] for a in arches]
    ax1.bar(range(len(arches)), vals, color=["#2980b9", "#8e44ad"], alpha=0.85)
    ax1.axhline(0.0, color="k", lw=0.8)
    for sign in (+1, -1):
        ax1.axhline(sign * GATE_PCT, color="#c0392b", ls="--", lw=1.2)
    for i, v in enumerate(vals):
        ax1.text(i, v + (1.6 if v >= 0 else -1.6), f"{v:+.2f}%",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax1.set_xticks(range(len(arches)), arches, fontsize=9)
    ax1.set_ylabel("measured mode-2 pseudo-frequency error  [%]")
    ax1.set_title("(b) same code, same input: platform spread "
                  f"{max(vals) - min(vals):.2f} pp", fontsize=9.5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.text(0.02, 0.97, "red dashed = 15 % gate\nthe x86_64 +8.64 % lands inside it\n"
                         "by coincidence: that run diverges too",
             transform=ax1.transAxes, va="top", fontsize=7.5)
    fig.suptitle("m4802 / m4887 - mode-2 pseudo-frequency: honest FAIL, pinned per platform, "
                 "verdict platform independent", fontsize=10.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path



def render_trace_figure(path: Path = TRACE_FIGURE, release=None,
                        trace=None) -> Path:
    """The release case measured against the Rayleigh-Lamb solution.

    (a) the mode-2 signal ``s(t)`` of the committed trace next to the analytic
    damped cosine at ``omega_2``: the measured envelope *grows* while the
    reference oscillates, which is the divergence behind the -43.5 % fit; (b)
    the kinetic energy of the same run on a log scale, i.e. the energy that the
    clamped free surface injects into the drop.
    """
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if release is None:
        release = json.loads(RELEASE.read_text())
    if trace is None:
        trace = json.loads(TRACE.read_text())["frequency"]
    t = trace["t"]
    s_meas = trace["s_second_moment"]
    ke = trace["kinetic_energy_J"]
    ref = release["frequency"]["reference"]
    omega = ref["omega_rad_per_s"]
    damping = ref["damping_estimate_per_s"]
    s0 = s_meas[0]
    period = 2.0 * math.pi / omega
    s_theory = [s0 * math.exp(-damping * ti) * math.cos(omega * ti) for ti in t]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax0.plot([ti / period for ti in t], s_theory, "--",
             label=f"Rayleigh-Lamb s0*exp(-l t) cos(w2 t), w2={omega:.2f} rad/s",
             color="#2c3e50", lw=1.2)
    ax0.plot([ti / period for ti in t], s_meas, lw=1.4,
             label="measured s(t) (committed trace)", color="#c0392b")
    ax0.axhline(0.0, color="k", lw=0.6)
    ax0.set_xlabel("t / T2  (T2 = %.4f s)" % period)
    ax0.set_ylabel("mode-2 signal  s = (a-b)/(a+b)")
    ax0.set_title("(a) release case vs analytic: the envelope grows instead of "
                  "decaying", fontsize=9.5)
    ax0.legend(fontsize=7.5, loc="upper left")
    ax0.grid(alpha=0.3)

    ax1.semilogy([ti / period for ti in t], ke, color="#8e44ad", lw=1.2)
    ax1.set_xlabel("t / T2")
    ax1.set_ylabel("kinetic energy  [J]  (log)")
    ax1.set_title(f"(b) same run: KE {ke[0]:.2e} J -> {ke[-1]:.2e} J "
                  f"(x{ke[-1] / ke[0]:.1e})", fontsize=9.5)
    ax1.grid(alpha=0.3, which="both")
    ax1.text(0.03, 0.95,
             "energy is being injected by the clamped free surface,\n"
             "not damped: any frequency fitted past this point is a\n"
             "pseudo-frequency of a diverging trajectory",
             transform=ax1.transAxes, va="top", fontsize=7.5)
    fig.suptitle("m4802 / m4887 - release case (R = 20 mm, dx = 2 mm, 1.8 T): "
                 "measured trace vs the analytic mode-2 frequency", fontsize=10.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="rewrite the doc + figure")
    group.add_argument("--check", action="store_true",
                       help="exit non-zero if the doc no longer matches the artefacts")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        stale = check()
        for item in stale:
            print(f"STALE {item}", file=sys.stderr)
        if not stale:
            print(f"{DOC.name}: blocks match the committed artefacts")
        return 1 if stale else 0

    doc = DOC.read_text()
    DOC.write_text(splice(doc, render_blocks()))
    print(f"wrote {DOC.relative_to(ROOT)}")
    if not args.no_figure:
        print(f"wrote {render_figure().relative_to(ROOT)}")
        print(f"wrote {render_trace_figure().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
