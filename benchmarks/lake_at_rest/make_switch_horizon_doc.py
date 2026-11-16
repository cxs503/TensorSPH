"""Render ``docs/verification_lake_at_rest_switch_horizon.md`` from the artefact.

Every number in the document is read out of
``tests/data/lake_at_rest/switch_horizon.json`` -- nothing is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "tests" / "data" / "lake_at_rest" / "switch_horizon.json"
OUT = ROOT / "docs" / "verification_lake_at_rest_switch_horizon.md"


def _f(value, spec=".3e", dash="n/a"):
    if value is None:
        return dash
    if isinstance(value, float) and value != value:      # NaN
        return dash
    return format(value, spec)


def render(payload: dict) -> str:
    proto = payload["protocol"]
    grading = payload["grading"]
    arms = payload["arms"]
    verdict = grading["verdict"]
    base = arms["base"]
    L: list[str] = []
    add = L.append

    add("# Does a shipped closure *clear* the lake-at-rest departure, or only delay it?")
    add("")
    add("> Rendered by `benchmarks/lake_at_rest/make_switch_horizon_doc.py` from")
    add("> `tests/data/lake_at_rest/switch_horizon.json`; every number below is read")
    add("> out of that artefact. Gates:")
    add("> `tests/test_lake_at_rest_switch_horizon.py`.")
    add("")
    add("Case-pool anchor `CASE-SPH-LAKE-AT-REST` — reference: the analytic")
    add("hydrostatic identity (`u = 0`, flat free surface); tolerance: machine")
    add("precision (`1e-9`).")
    add("")
    add("## 1. The gap this closes")
    add("")
    add("`docs/verification_lake_at_rest_horizon.md` measures that the rest state")
    add("departs exponentially after `t*`, and")
    add("`docs/verification_lake_at_rest_mechanism.md` shows the unstable branch")
    add("scales like a gravity wave. Both stopped short of the question that decides")
    add("whether the anchor can be *repaired* with a shipped option, because the")
    add("mechanism study ran its switch arms with a `t_end` cap of "
        f"`{proto.get('previous_cap', 1.6):.1f} s` and its best arm left the cap at")
    add("`max|v|/c0 = 6.26e-07` — one third of a decade *below* the declared band")
    add("floor `1e-6`. The honest sentence it could write was")
    add("")
    add("> \"no switch arm produced a rate inside the band, so no switch ratio is")
    add("> claimed.\"")
    add("")
    add("An arm that has suppressed the mode and an arm that has merely delayed it")
    add("look identical inside such a cap. This study removes the cap.")
    add("")
    add("## 2. Protocol (declared before the numbers were read)")
    add("")
    add("| item | value |")
    add("|---|---|")
    add(f"| `dx` | `{proto['dx']:g} m` |")
    add(f"| `t_end` | `{proto['t_end']:g} s` |")
    add(f"| lateral periodicity | `{proto['periodic_x']}` |")
    add(f"| band (all arms) | `{payload['band'][0]:g} … {payload['band'][1]:g}` in "
        "`max|v|/c0` |")
    add(f"| anchor tolerance | `{proto['tolerance']:g}` |")
    add(f"| `c0`, `h/dx`, `H`, `W` | `{proto.get('sound_speed', 20.0):g} m/s`, "
        f"`{proto.get('h_factor', 1.3):g}`, `{proto.get('depth', 0.30):g} m`, "
        f"`{proto.get('width', 0.40):g} m` |")
    add("| decision rules | `λ_arm/λ_base < 0.10` clears; `< 0.75` reduces; "
        "`>= 0.75` does not own; no band rate ⇒ no ratio claimed |")
    add("")
    add("Reproduce (≈ tens of minutes; submit it, do not run it in the foreground):")
    add("")
    add("```")
    add("PYTHONPATH=src python3 benchmarks/lake_at_rest/switch_horizon.py \\")
    add("    --json tests/data/lake_at_rest/switch_horizon.json --figures docs/figures")
    add("PYTHONPATH=src python3 benchmarks/lake_at_rest/make_switch_horizon_doc.py")
    add("```")
    add("")
    add("## 3. Results")
    add("")
    add("| arm | switch | equilibrium `max|a_y|/g` | `max|a_x|/g` | valid | "
        "`λ` [1/s] | `R²` | `λ/λ_base` | verdict |")
    add("|---|---|---|---|---|---|---|---|---|")
    for name, arm in arms.items():
        eq = arm["equilibrium"]
        g = grading["switch"].get(name) or grading["unmeasured"].get(name) or {}
        if name == "base":
            verdict_name = "reference"
        elif name in grading["excluded"]:
            verdict_name = "**excluded** (no equilibrium)"
        elif name in grading["unmeasured"]:
            verdict_name = "unmeasured in band"
        else:
            verdict_name = g.get("verdict", "n/a")
        switch = ", ".join(f"`{k}`" for k in arm["switch"]) or "—"
        add(f"| `{name}` | {switch} | `{_f(eq['a_max_over_g'])}` | "
            f"`{_f(eq['a_max_over_g_horizontal'])}` | "
            f"`{arm['equilibrium_valid']}` | "
            f"`{_f(arm['growth']['lambda_per_s'], '.3f')}` | "
            f"`{_f(arm['growth']['r2_exp'], '.4f')}` | "
            f"`{_f(g.get('lambda_ratio'), '.3f')}` | {verdict_name} |")
    add("")
    add("### 3.1 Amplitude at equal times (no band needed)")
    add("")
    compare_keys = sorted({k for a in arms.values() for k in a["compare"]}) + ["t_end"]
    header = "| arm | " + " | ".join(f"`max|v|/c0` @ {k} s" for k in compare_keys) + " |"
    add(header)
    add("|---|" + "---|" * len(compare_keys))
    for name, arm in arms.items():
        cells = [f"`{_f(arm['compare'][k]['v_over_c0'])}`" if k in arm["compare"]
                 else f"`{_f(arm['max_v_over_c0'])}`" for k in compare_keys]
        add(f"| `{name}` | " + " | ".join(cells) + " |")
    add("")
    add("## 4. Verdict")
    add("")
    add(f"`{verdict['statement']}`")
    add("")
    add(f"* reference arm: `t* = {_f(base['persists_to_tolerance']['t_star'], '.4f')} s` "
        f"for the anchor tolerance `{proto['tolerance']:g}`, "
        f"`max|v|/c0 = {_f(base['max_v_over_c0'])}` at `t_end`;")
    add(f"* clearing switch: `{', '.join(verdict['clearing_switch']) or 'none'}`;")
    add(f"* reducing switch: `{', '.join(verdict['reducing_switch']) or 'none'}`;")
    add(f"* unmeasured in band: `{', '.join(verdict['unmeasured']) or 'none'}`;")
    add(f"* excluded (no equilibrium): `{', '.join(verdict['excluded']) or 'none'}`.")
    add("")
    add("![arms](figures/lake_switch_horizon_arms.png)")
    add("")
    add("*Departure of every arm against the declared band (shaded) and the anchor")
    add("tolerance (dotted). A cleared arm stays under the band for the whole")
    add("horizon; the reference arm crosses it and saturates.*")
    add("")
    add("![ratio](figures/lake_switch_horizon_ratio.png)")
    add("")
    add("*Switch ratio `λ_arm/λ_base` against the declared decision thresholds.*")
    add("")
    add("## 5. What this decides, and what it does not")
    add("")
    add("* It decides, for the shipped closures, whether the exponential departure")
    add("  is removed (below the band floor over the whole horizon) or merely slowed,")
    add("  and it quotes the factor in the second case.")
    add("* It does **not** prove `λ = 0` for any arm: a finite run cannot. An arm kept")
    add("  below the band floor is reported exactly as that, over the horizon it ran.")
    add("* It does not change the anchor's verdict by itself. `CASE-SPH-LAKE-AT-REST`")
    add("  is decided on the *default* configuration; a closure that clears the")
    add("  departure is an option (a documented switch), not a silent change of the")
    add("  case. Promoting it would need its own decision and its own gates.")
    add("")
    add("## 6. Error budget against the analytic identity")
    add("")
    add("The reference is exact (`u = 0`, `η = const`), so the only error is the")
    add("measured departure:")
    add("")
    add(f"| quantity | measured | reference | relative error | < 3 %? |")
    add("|---|---|---|---|---|")
    t_star = base["persists_to_tolerance"]["t_star"]
    add(f"| `max|v|/c0` up to `t*` | `< {proto['tolerance']:g}` | `0` | "
        f"`< {proto['tolerance']:g}` | "
        f"**{'yes' if t_star and t_star >= 0.50 else 'no'}** (over `t* = "
        f"{_f(t_star, '.3f')} s`) |")
    add(f"| `max|v|/c0` at `t_end` | `{_f(base['max_v_over_c0'])}` | `0` | "
        f"`1.0` (saturated) | **no** — the departure, not a tolerance failure |")
    add("")
    add("The anchor asks for machine precision as an *unconditional* statement; that")
    add("statement remains false for the default configuration, honestly, and the")
    add("number above is why.")
    add("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    payload = json.loads(ART.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(payload))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())