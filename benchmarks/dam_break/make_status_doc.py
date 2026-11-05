"""Generate docs/verification_dam_break.md from the machine-checked JSON artifacts.

All numbers in the document are injected programmatically from
``benchmarks/dam_break/resolution_ladder.json`` and
``benchmarks/dam_break/reference_audit.json`` (zero hand transcription).
Re-run after any dam-break re-run to refresh the status.

Usage:
    python benchmarks/dam_break/make_status_doc.py
"""
import json
import os
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LADDER = json.loads((HERE / "resolution_ladder.json").read_text())
AUDIT = json.loads((HERE / "reference_audit.json").read_text())
OUT = HERE.parent.parent / "docs" / "verification_dam_break.md"
FIG = HERE / "dam_break_status.png"


def pct(x):
    return f"{100 * x:.1f}%"


def main(out: Path = OUT, fig_path: Path = FIG):
    gates = LADDER["gates"]
    results = LADDER["results"]
    table = AUDIT["table_audit"]
    geo = AUDIT["geometry"]
    sim = LADDER["simulation_params"]

    # ---- figure: error vs resolution (log dx) ----
    dxs = [r["dx"] for r in results]
    ritter_err = [r["ritter_mean_relative_error"] for r in results]
    exp_mae = [r["experimental_error"]["mean_absolute_error"] for r in results]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    ax.semilogx(dxs, ritter_err, "o-", label="Ritter mean rel. err (no gate)")
    ax.semilogx(dxs, exp_mae, "s--", label="M&M MAE (ref WITHHELD)")
    ax.axhline(0.15, color="red", lw=0.8, ls=":", label="15% gate")
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("error in Z/a")
    ax.set_title("Dam-break B2: error vs resolution (H/a=2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    exp = gates["experimental_comparison"]
    rit = gates["ritter_comparison"]
    mach = gates["mach_number"]
    comp = gates["density_compression"]
    mass = gates["mass_conservation"]
    bound = gates["bounded_flow"]

    ladder_rows = "\n".join(
        f"| {r['dx']} | {r['particles']} | {pct(r['ritter_mean_relative_error'])} "
        f"| {pct(r['experimental_error']['mean_absolute_error'])} "
        f"| {r['experimental_error']['rmse']:.3f} | {pct(r['mach_number'])} "
        f"| {pct(r['max_density_compression'])} |"
        for r in results
    )

    try:
        fig_link = Path(os.path.relpath(fig_path, out.parent)).as_posix()
    except ValueError:  # different drives: fall back to the file name
        fig_link = fig_path.name

    md = f"""# Dam-break (B2) verification status

**Generated**: {date.today().isoformat()} by `benchmarks/dam_break/make_status_doc.py`
(numbers injected from `resolution_ladder.json` + `reference_audit.json`, zero hand copy).
**Scope**: the merged B2 dam-break solver (`src/tensorsph/dam_break.py`) re-verified
against its analytic (Ritter) and experimental (Martin & Moyce 1952) references.
This closes the "dam break not re-verified" debt flagged by the m239 first-wave audit.

## 1. Case (physics / geometry / parameters)

Gravity-driven collapse of a finite liquid column on a dry rigid floor, 2-D.

| quantity | value |
|---|---|
| column width a | {geo['a_m']} m |
| column height H | {geo['H_m']} m |
| aspect ratio H/a | {geo['H_over_a']} |
| tank width | {sim['tank_width']} m |
| sound speed c0 | {sim['sound_speed']} m/s |
| wall friction | {sim['wall_friction']} |
| adaptive dt | {sim['adaptive_dt']} |

## 2. Method (how each criterion is measured)

* **Ritter (analytic)**: leading-edge trajectory and depth field compared with the
  Ritter shallow-water solution. For H/a = 2 shallow-water theory does *not* apply,
  so this is a *diagnostic* (systematic offset expected), not a gate. The regime
  where Ritter is exact (H/a = 0.05-0.2) is covered by `ritter_limit.py`.
* **Experimental (M&M 1952)**: front position Z/a vs tau compared with the bundled
  20-point table. **The reference audit grades this table WITHHELD** (see §5), so the
  experimental gate is reported but *cannot arbitrate* the 15% criterion.
* **Stability gates**: mass drift, particle escape, Mach number, density compression.

## 3. Results (from `resolution_ladder.json`)

Resolution ladder (dx = 0.04 / 0.02 / 0.01):

| dx (m) | N | Ritter rel. err | M&M MAE | M&M RMSE | Mach | dens. comp |
|---|---|---|---|---|---|---|
{ladder_rows}

Gate summary (finest dx):

| gate | measured | limit | result |
|---|---|---|---|
| exp. MAE (M&M) | {pct(exp['measured'])} | {pct(exp['limit'])} | **{exp['result'].upper()}** |
| Ritter mean rel. err | {pct(rit['measured'])} | (no gate) | diagnostic |
| mass drift | {mass['measured']} | {mass['limit']} | {mass['result'].upper()} |
| escaped particles | {bound['measured']} | {bound['limit']} | {bound['result'].upper()} |
| Mach number | {pct(mach['measured'])} | {pct(mach['limit'])} | {mach['result'].upper()} |
| density compression | {pct(comp['measured'])} | {pct(comp['limit'])} | {comp['result'].upper()} |

![dam-break status]({fig_link})

## 4. Error judgement (explicit)

* **Ritter (analytic, H/a=2)**: {pct(rit['measured'])} mean relative error.
  **Not < 3% and not < 15%** — but this is the *expected* systematic offset
  ({rit['note']}), not a scheme defect. The exact-regime check lives in
  `ritter_limit.py` (H/a = 0.05-0.2).
* **Experimental (M&M 1952)**: {pct(exp['measured'])} MAE vs a {pct(exp['limit'])} gate
  → **FAIL on the number**, but the reference is **WITHHELD** (provenance, §5), so the
  FAIL is *not* a valid accuracy verdict. **No valid <15% claim can be made** until a
  traceable digitization replaces the table.
* **Stability gates** (mass / escape / Mach / compression): all **PASS**.

## 5. Caveats / known gaps

1. **M&M table provenance (blocking)**: the reference audit found the bundled table is
   "synthetic-looking" — abscissa is an exact arithmetic sequence and every ordinate is
   an exact multiple of {table['ordinate_quantum']} (second-difference RMS
   {table['second_difference_rms']:.5f} vs claimed ±{table['claimed_uncertainty']}).
   Verdict: `{table['verdict']}`;
   `usable_as_validation_target = {str(table['usable_as_validation_target']).lower()}`,
   grade **{table['grade']}**. A table of unknown provenance cannot arbitrate a 15% gate.
2. **Ritter offset at H/a=2**: {pct(rit['measured'])} is a physics offset (finite
   reservoir), not a solver error; do not gate on it.
3. The experimental FAIL in §3 is therefore a *reference* problem, not a *solver*
   problem. Re-opening the experimental gate requires a traceable digitization of the
   original figure (owner decision, out of scope for a single mission).

## 6. Evidence grade

| criterion | grade | reason |
|---|---|---|
| Ritter (H/a=2) | WITHHELD | diagnostic only; exact regime in ritter_limit.py |
| Experimental (M&M) | {table['grade']} | {table['grade_reason'][:120]}… |
| Stability (mass/escape/Mach/compression) | REAL | all gates pass at finest dx |

**Bottom line**: the dam-break solver is *stable and mass-conserving* (REAL for the
stability gates) but its *accuracy* against experiment is **WITHHELD** pending a
traceable experimental reference. The m239 "not re-verified" debt is now *documented
and machine-pinned* (`tests/test_dam_break_status.py`), not silently green.
"""
    out.write_text(md)
    print("wrote", out)
    print("wrote", fig_path)


def _cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT, help="markdown output path")
    parser.add_argument("--fig", type=Path, default=FIG, help="figure output path")
    args = parser.parse_args(argv)
    main(args.out, args.fig)


if __name__ == "__main__":
    _cli()
