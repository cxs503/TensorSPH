"""Render ``docs/verification_lake_at_rest_mechanism.md`` and its figures.

Every number is read out of ``tests/data/lake_at_rest/mechanism.json`` (written
by ``benchmarks/lake_at_rest/mechanism.py``); nothing is typed by hand, so the
document cannot drift away from the run it reports.

    PYTHONPATH=src python3 benchmarks/lake_at_rest/make_mechanism_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "tests" / "data" / "lake_at_rest" / "mechanism.json"
DOC = HERE / "docs" / "verification_lake_at_rest_mechanism.md"
FIG = HERE / "docs"

FAMILY_ORDER = ("reference", "gravity", "sound_speed", "kernel", "geometry",
                "switch")


def load() -> dict:
    if not DATA.is_file():
        raise SystemExit(f"missing artefact {DATA}; run "
                         "benchmarks/lake_at_rest/mechanism.py first")
    return json.loads(DATA.read_text())


def num(value, spec=".3f", missing="n/a"):
    if value is None:
        return missing
    return format(value, spec)


def rate_block(arms: dict) -> str:
    lines = ["| arm | family | what changes | `lambda` [1/s] | `R^2` | "
             "band entered | `max|v|/c0` at stop | mode | `omega` [1/s] | "
             "`max|a|/g` at reconstruction |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, arm in arms.items():
        cfg = arm["config"]
        changes = []
        if arm["family"] == "gravity":
            changes.append(f"`g = {cfg['gravity']:.3f}`")
        if arm["family"] == "sound_speed":
            changes.append(f"`c0 = {cfg['sound_speed']:.1f}`")
        if arm["family"] == "kernel":
            changes.append(f"`h = {cfg['h_factor']:.2f} dx`")
        if arm["family"] == "geometry":
            changes.append(f"`H = {cfg['depth']:.2f} m`")
        if arm["family"] == "switch":
            on = [key for key in ("density_renormalization",
                                  "surface_pressure_closure",
                                  "boundary_consistent_density",
                                  "density_diffusion") if cfg[key]]
            changes.append(", ".join(f"`{key}`" for key in on))
        growth = arm["growth"]
        lines.append(
            f"| `{name}` | {arm['family']} | {', '.join(changes) or 'reference'} "
            f"| {num(growth['lambda_per_s'])} | {num(growth['r2_exp'], '.5f')} "
            f"| {'yes' if arm['entered_band'] else '**no**'} "
            f"| {num(arm['max_v_over_c0'], '.3e')} "
            f"| {arm['mode']['dominant_mode']} "
            f"| {num(arm['oscillation']['omega_per_s'], '.2f')} "
            f"| {num(arm['run']['equilibrium']['a_max_over_g'], '.1e')}"
            f"{'' if arm.get('equilibrium_valid', True) else ' **(no equilibrium)**'} |")
    return "\n".join(lines)


def switch_block(reductions: dict) -> str:
    ratios = reductions.get("switch_ratio") or {}
    if not ratios:
        return "_(no switch arm in this artefact)_"
    lines = ["| switch | `lambda` / `lambda(base)` | verdict |", "|---|---|---|"]
    for name, value in sorted(ratios.items(),
                              key=lambda item: (item[1] is None, item[1])):
        cleared = "**clears the growth**" if value is not None and value < 0.1 \
            else ("no effect" if value is None or value > 0.5 else "partial")
        lines.append(f"| `{name}` | {num(value, '.3f')} | {cleared} |")
    return "\n".join(lines)


def exponent_block(reductions: dict) -> str:
    exponent = reductions.get("exponent") or {}
    lines = ["| family | slope `d ln lambda / d ln x` |", "|---|---|"]
    for key in ("gravity", "sound_speed", "h"):
        lines.append(f"| `{key}` | {num(exponent.get(key), '+.3f')} |")
    lines.append(f"| depth ratio `lambda(H/3) / lambda(H)` "
                 f"| {num(reductions.get('depth_ratio'), '.3f')} |")
    return "\n".join(lines)


def figure_rates(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = payload["arms"]
    names = list(arms)
    values = [arms[name]["growth"]["lambda_per_s"] for name in names]
    colours = {"reference": "#1f4e79", "gravity": "#c55a11",
               "sound_speed": "#548235", "kernel": "#7030a0",
               "geometry": "#bf9000", "switch": "#c00000"}
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    plotted = [(index, value) for index, value in enumerate(values)
               if value is not None]
    ax.bar([names[index] for index, _ in plotted],
           [value for _, value in plotted],
           color=[colours.get(arms[names[index]]["family"], "#808080")
                  for index, _ in plotted])
    for index, value in plotted:
        ax.text(index, value, format(value, ".2f"), ha="center",
                va="bottom", fontsize=7)
    flat = [index for index, value in enumerate(values) if value is None]
    for index in flat:                       # no band entry: never grew
        ax.text(index, 0.4, "never entered\nthe band", ha="center",
                va="bottom", fontsize=6, rotation=90, color="#606060")
    ax.set_ylabel("growth rate `lambda` [1/s]")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title("lake-at-rest departure rate per arm "
                 "(band ${:.0e}$ .. ${:.0e}$)".format(*payload["band"]))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_scaling(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = payload["arms"]
    families = (("gravity", "gravity", "gravity [m/s^2]", "g"),
                ("sound_speed", "sound_speed", "sound speed [m/s]", "c0"),
                ("kernel", "h", "smoothing length h [m]", "h"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    exponent = (payload.get("reductions") or {}).get("exponent") or {}
    for ax, (family, key, label, symbol) in zip(axes, families):
        points = []
        base = arms.get("base")
        if base and base["growth"]["lambda_per_s"]:
            points.append((base["config"][key], base["growth"]["lambda_per_s"]))
        for arm in arms.values():
            if arm["family"] == family and arm["growth"]["lambda_per_s"]:
                points.append((arm["config"][key], arm["growth"]["lambda_per_s"]))
        points = [p for p in points if p[0] > 0.0]
        if not points:
            ax.set_visible(False)
            continue
        xs = [math.log(p[0]) for p in points]
        ys = [math.log(p[1]) for p in points]
        ax.plot(xs, ys, "o-", color="#1f4e79")
        for (x, y) in points:
            ax.annotate(format(y, ".1f"), (math.log(x), math.log(y)),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"ln {symbol}")
        ax.set_ylabel("ln lambda [1/s]")
        ax.set_title(f"slope = {num(exponent.get(key), '+.2f')}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("is the rate set by gravity, by the sound speed, or by the "
                 "kernel?", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_localisation(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = payload["arms"]
    names = [name for name in ("base", "sw_delta", "sw_surfclos")
             if name in arms]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for name in names:
        profile = arms[name]["localisation"]["row_speed_mean"]
        if not profile:
            continue
        height = [index / max(len(profile) - 1, 1) for index in range(len(profile))]
        peak = max(profile) or 1.0
        axes[0].plot([value / peak for value in profile], height, "o-",
                     label=name, markersize=3)
    axes[0].set_xlabel("row-mean |v| / max")
    axes[0].set_ylabel("height fraction (0 = floor, 1 = free surface)")
    axes[0].set_title("where the motion lives", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for name in names:
        series = arms[name]["samples"]
        if not series:
            continue
        axes[1].semilogy([s["t"] for s in series],
                         [s["v_over_c0"] for s in series], label=name, lw=1.0)
    axes[1].axhline(payload["band"][0], ls=":", color="k", lw=0.8)
    axes[1].axhline(payload["band"][1], ls=":", color="k", lw=0.8)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("max |v| / c0")
    axes[1].set_title("departure from rest (band dotted)", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    payload = load()
    arms = payload["arms"]
    reductions = payload.get("reductions") or {}
    verdict = payload.get("verdict") or {}

    FIG.mkdir(parents=True, exist_ok=True)
    figure_rates(payload, FIG / "lake_at_rest_mechanism_rates.png")
    figure_scaling(payload, FIG / "lake_at_rest_mechanism_scaling.png")
    figure_localisation(payload, FIG / "lake_at_rest_mechanism_localisation.png")

    base = arms["base"]
    band = payload["band"]
    lines = [
        "# Lake at rest: what owns the departure (`CASE-SPH-LAKE-AT-REST`)",
        "",
        "> Generated by `benchmarks/lake_at_rest/make_mechanism_doc.py` from",
        "> `tests/data/lake_at_rest/mechanism.json`; every number below is read",
        "> out of that artefact.",
        "",
        "## 1. Question",
        "",
        "`docs/verification_lake_at_rest_horizon.md` measures that the rest state",
        "holds `max|v|/c0 < 1e-9` for `t < t*` and then departs *exponentially*,",
        "and excludes the integrator, the seed, the frozen pair list, one lattice",
        "and water viscosity as the cause. It does not say **which term** the",
        "negative stiffness belongs to. This study asks that question with",
        "scaling arms and shipped switches only -- no new physics, no fitted",
        "constants.",
        "",
        "Reference protocol: `dx = {:g} m`, `H = {:g} m`, `W = {:g} m`,\n"
        "`h = {:.2f} dx`, `c0 = {:g} m/s`, lateral periodicity on, frozen pair\n"
        "list, `{}` particles per arm, exponential band `${:.0e} .. {:.0e}$`.".format(
            base["config"]["dx"], base["config"]["depth"], base["config"]["width"],
            base["config"]["h_factor"], base["config"]["sound_speed"],
            base["run"]["particles"], band[0], band[1]),
        "",
        "## 2. Arms",
        "",
        rate_block(arms),
        "",
        "An arm whose reconstruction did not converge does not start at rest, so",
        "it is **not** a stability arm: `max|a|/g` after the reconstruction is",
        "shown next to every arm, and such arms are excluded from the exponents",
        "and from the switch ratios (they are listed in the verdict).",
        "",
        "`lambda` is the least-squares slope of `ln max|v|/c0` against `t` inside",
        "the declared band (the estimator of `benchmarks/lake_at_rest/band_fit.py`,",
        "the one the horizon study's audit settled on). `n/a` means the arm never",
        "reached the band floor inside the run -- i.e. **no growth was measured**,",
        "which is the strongest statement a finite run can make and is *not* the",
        "same as `lambda = 0`.",
        "",
        "## 3. Scaling laws",
        "",
        exponent_block(reductions),
        "",
        "![scaling](lake_at_rest_mechanism_scaling.png)",
        "",
        "* slope `~ 1/2` on gravity would mean the unstable branch is a gravity",
        "  wave (`omega ~ sqrt(g k tanh kH)`);",
        "* slope `~ 1` on `c0` would mean the rate is an acoustic stiffness",
        "  `c0/h`, i.e. a numerical scale;",
        "* slope `~ 0` on both means the rate is a property of the equilibrium's",
        "  discretisation itself.",
        "",
        "## 4. Do the shipped closures own it?",
        "",
        switch_block(reductions),
        "",
        "## 5. Localisation and temporal character",
        "",
        "![localisation](lake_at_rest_mechanism_localisation.png)",
        "",
        "| arm | KE in top {} rows | KE in bottom {} rows | fastest particle "
        "height | fastest particle in top {} rows |".format(3, 3, 3),
        "|---|---|---|---|---|",
    ]
    for name, arm in arms.items():
        local = arm["localisation"]
        lines.append(
            f"| `{name}` | {num(local['ke_surface_fraction'], '.3f')} "
            f"| {num(local['ke_bottom_fraction'], '.3f')} "
            f"| {num(local['fastest_row_height'], '.3f')} "
            f"| {num(local['fastest_row_top3_share'], '.2f')} |")
    lines += [
        "",
        "## 6. Verdict (rule applied to the measured numbers)",
        "",
        "`p_gravity = {}`, `p_sound_speed = {}`, `p_h = {}`.".format(
            num(verdict.get("p_gravity"), "+.3f"),
            num(verdict.get("p_sound_speed"), "+.3f"),
            num(verdict.get("p_h"), "+.3f")),
        "",
        "**" + str(verdict.get("statement", "n/a")) + "**",
        "",
        "![rates](lake_at_rest_mechanism_rates.png)",
        "",
        "## 7. What this does and does not decide",
        "",
        "* It answers *which family* the unstable mode belongs to (gravity load,",
        "  acoustic stiffness, kernel support) and *whether a shipped closure",
        "  removes it*. Both are decided by numbers above, per the rules declared",
        "  in the module docstring.",
        "* It does **not** change the anchor's verdict. `CASE-SPH-LAKE-AT-REST`",
        "  asks for the analytic hydrostatic identity at machine precision; the",
        "  measured verdict stays FAIL as an unconditional statement, PASS over the",
        "  certified horizon `t*`.",
        "* A finite run cannot prove `lambda = 0`: an arm marked *never entered the",
        "  band* is reported as such, and the claim written down is only the one",
        "  the run supports.",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC} and 3 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())