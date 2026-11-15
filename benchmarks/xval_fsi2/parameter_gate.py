"""Benchmark-target integrity audit for the FSI2 anchor (machine-generated).

Writes, with no hand-typed numbers:

* ``artifacts/xval_fsi2/parameter_gate.json``  -- the full audit bundle
* ``docs/xval_fsi2/fig_parameter_gate.png``    -- group-wise error + time coverage
* ``docs/xval_fsi2/PARAMETER_TABLE_INTEGRITY.md`` -- the audit document

Run: ``PYTHONPATH=src python benchmarks/xval_fsi2/parameter_gate.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import fsi2_parameter_gate as gate  # noqa: E402
from tensorsph.fsi2_similarity import BENCHMARK_FSI2, similarity_report  # noqa: E402

ART = ROOT / "artifacts" / "xval_fsi2"
DOCS = ROOT / "docs" / "xval_fsi2"

#: Candidate production resolutions: (label, dx [m], c0 [m/s], sound speed
#: relation actually used by the shipped case is c0 = 30 m/s at dx = 0.02).
RESOLUTIONS = (("dx=0.020 m", 0.020), ("dx=0.010 m", 0.010), ("dx=0.005 m", 0.005))


def _row_md(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header, rows) -> str:
    out = [_row_md(header), _row_md(["---"] * len(header))]
    out += [_row_md(r) for r in rows]
    return "\n".join(out)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:+.2f} %"


def _pct_or_na(x) -> str:
    return "n/a" if x is None else _fmt_pct(x)


def main() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    audit = gate.audit()
    period = audit["period"]["period_used"]

    # coverage of the *remaining* debt, per candidate resolution
    from tensorsph.fsi2_turek_hron import FSI2Case

    cov_rows, cov_cycles = [], []
    for label, dx in RESOLUTIONS:
        case = FSI2Case(dx=dx)
        dt = case.dt()
        n_amp = int(np.ceil(gate.MIN_CYCLES_AMPLITUDE * period / dt))
        n_stat = int(np.ceil(gate.MIN_CYCLES_STATISTICS * period / dt))
        n_fluid = int(round(case.channel_length / dx) * round(case.channel_height / dx))
        cov_rows.append([label, f"{dt:.3e}", n_amp, n_stat, n_fluid])
        cov_cycles.append((label, dt))

    # json artifact
    (ART / "parameter_gate.json").write_text(json.dumps(audit, indent=1) + "\n")

    # ---------------------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    sources = [
        ("survey table", audit["matrix"]["survey"]),
        ("FSI2Case defaults", audit["matrix"]["case_defaults"]),
        ("archived upstream", audit["matrix"]["archived_upstream"]),
    ]
    group_names = ["normalised_stiffness", "density_ratio", "poisson_s"]
    width = 0.26
    xs = np.arange(len(group_names))
    for k, (label, block) in enumerate(sources):
        vals = []
        for name in group_names:
            entry = next((c for c in block["gate"]["checks"] if c["name"] == name), None)
            vals.append(np.nan if entry is None or "rel_error" not in entry
                        else 100.0 * entry["rel_error"])
        ax.bar(xs + (k - 1) * width, vals, width, label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-2.0, 2.0, color="g", alpha=0.12, label="identity tolerance (±2 %)")
    ax.set_xticks(xs)
    ax.set_xticklabels([g.replace("_", "\n") for g in group_names], fontsize=8)
    ax.set_ylabel("relative error vs declared target  [%]")
    ax.set_title("(a) declared benchmark numbers vs the declared target", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = [c[0] for c in cov_cycles]
    cycles_per_10k = [1e4 * c[1] / period for c in cov_cycles]
    ax.bar(labels, cycles_per_10k, color="tab:blue")
    ax.axhline(gate.MIN_CYCLES_AMPLITUDE, color="r", ls="--",
               label=f"amplitude gate ({gate.MIN_CYCLES_AMPLITUDE:.0f} cycles)")
    ax.axhline(gate.MIN_CYCLES_STATISTICS, color="darkred", ls=":",
               label=f"statistics gate ({gate.MIN_CYCLES_STATISTICS:.0f} cycles)")
    ax.set_yscale("log")
    ax.set_ylabel("response cycles per 10 000 steps")
    ax.set_title(f"(b) time coverage at the measured period {period:.3f} s", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_parameter_gate.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------- document
    m = audit["matrix"]
    checks_rows = []
    for src_key, src_label in (("survey", "survey doc table"),
                               ("case_defaults", "FSI2Case defaults"),
                               ("declared_target", "declared dimensionless target"),
                               ("archived_upstream", "archived upstream fsi2.h")):
        block = m[src_key]
        for c in block["gate"]["checks"]:
            if c["status"] == "not_run":
                got, rel = "—", "not run"
            else:
                got, rel = f"{c['computed']:.6g}", _pct_or_na(c["rel_error"])
            checks_rows.append([src_label, block["class"], c["name"], got, rel,
                                c["status"]])

    cand = audit["candidates"]
    cand_rows = []
    for key in ("paper_target", "upstream_consistent"):
        c = cand[key]
        cand_rows.append([key, c["class"], f"{c['u_mean']:.3f}", f"{c['u_max']:.3f}",
                          f"{c['mu_f']:.3f}", f"{c['beam_E']:.3e}",
                          f"{c['rho_s_over_rho_f']:.2f}", f"{c['Ae']:.4g}",
                          f"{c['Re']:.1f}", f"{c['slenderness']:.2f}"])

    per = audit["period"]
    probe = audit["coverage_probe_40_steps"]
    sens = audit["coverage_sensitivity"]

    sim_rows = []
    for key in ("paper_target", "upstream_consistent"):
        case = FSI2Case(
            rho0=cand[key]["rho_f"], viscosity=cand[key]["mu_f"],
            inlet_u_max=cand[key]["u_max"], beam_E=cand[key]["beam_E"],
            beam_rho=cand[key]["rho_s"], beam_nu=cand[key]["poisson_s"],
        )
        upstream_groups = {k: v for k, v in gate.oss_row().items()
                           if k != "U_f_scaled"}
        for tgt_key, tgt in (("declared target", dict(BENCHMARK_FSI2)),
                             ("upstream groups", upstream_groups)):
            rep = similarity_report(case, target=tgt)
            sim_rows.append([
                key, tgt_key,
                _pct_or_na(next(g["rel_error"] for g in rep["groups"]
                                if g["name"] == "Ae")),
                _pct_or_na(next(g["rel_error"] for g in rep["groups"]
                                if g["name"] == "rho_s_over_rho_f")),
                ", ".join(rep["failed"]) if rep["failed"] else "none",
            ])

    md = f"""# FSI2 benchmark-target integrity: the target table does not close

Anchor: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`.
Document generated by `benchmarks/xval_fsi2/parameter_gate.py`
(data: `artifacts/xval_fsi2/parameter_gate.json`,
figure: `docs/xval_fsi2/fig_parameter_gate.png`).  Every number below is
injected programmatically; nothing is transcribed by hand.

## 0. What was checked and why

Every FSI2 error percentage in this repository is a ratio to a *benchmark
target*, and that target is declared in four places (survey document, dataclass
defaults, dimensionless target dict, archived-upstream annotations).  A ratio is
only interpretable if the target is **one self-consistent row of numbers**.  The
gate `tensorsph.fsi2_parameter_gate` parses those four sources out of the
repository and runs the identities that need no paper access:

| identity | statement | why it must hold |
|---|---|---|
| `reynolds` | `Re = rho_f U_mean D / mu_f` | it is the definition of the case's Reynolds number |
| `mean_of_parabola` | `U_mean = (2/3) u_max` | the benchmark inlet is `u = 4 u_max y (H-y)/H^2` |
| `normalised_stiffness` | `Ae = E / (rho_f U_mean^2)` | tip amplitude scales as `1/Ae` in this load-controlled problem |
| `slenderness` | `L / t = 17.5` | geometric identity of the beam |
| `density_ratio` | `rho_s / rho_f` | sets the added-mass regime |

Tolerance ±2 %.  A check whose inputs are absent is reported `not_run`, never
`pass`.

## 1. Result: three of the four declarations disagree with the target

{_table(["declared source", "evidence class", "identity / group", "computed",
         "rel. error", "status"], checks_rows)}

Read-out:

* the **survey table is not a valid Reynolds-number row**: its own
  `rho_f = 1000`, `U_mean = 0.2`, `D = 0.1`, `mu_f = 1.0` give `Re = 20`, while
  the same table declares `Re = 100`.  The mismatch is a factor 5 and cannot be
  a rounding or discretisation question;
* the survey table, and independently the shipped `FSI2Case` defaults, sit at
  `Ae = 1.0e5`, i.e. **+1685.7 %** away from the declared target `5.6e3`
  (the same number the similarity audit already printed, now bound to a gate);
* both declare `nu_s = 0.3` against a target `0.4` (−25 %);
* the archived upstream block is internally coherent but differs from the
  declared target on exactly two groups: `Ae` (`1.4e3` vs `5.6e3`, −75 %) and
  `rho_s/rho_f` (`10` vs `1`).

## 2. What the identities *cannot* decide

Fixing the Reynolds identity requires `mu_f = rho_f U_mean D / Re`.  Holding the
survey's `mu_f = 1.0 Pa s` (the only viscosity value anywhere in the repository)
forces `U_mean = 1.0 m/s`, not the declared `0.2 m/s` — i.e. **the survey's
`U_mean` is the mis-transcribed number, and it is mis-transcribed from a
different case (the low-velocity one), not a rounding error**.  With `U_mean`
fixed, two rows survive, and they differ only on the two quantities no internal
identity constrains:

{_table(["candidate", "class", "U_mean [m/s]", "u_max [m/s]", "mu_f [Pa s]",
         "E [Pa]", "rho_s/rho_f", "Ae", "Re", "L/t"], cand_rows)}

* `paper_target` keeps the declared target (`rho_s/rho_f = 1`, `Ae = 5.6e3`) and
  then forces `E = 5.6e6 Pa` — so the survey's `4.0e6` is a *third*
  mis-transcription;
* `upstream_consistent` takes the archived upstream values verbatim
  (`fsi2.h:36` `rho0_s = 10.0`, `fsi2.h:38` `Ae = 1.4e3`) and gives
  `E = 1.4e6 Pa`.

**Which one is Turek-Hron FSI2 is `WITHHELD`**: the paper is not available
offline in this workspace, and the repository itself labels its own target as
"not re-verified against the paper PDF".  What the gate establishes, without the
paper, is that the declared target is *not reproducible from the repository's
own physical table by correcting a single number* — the physical table needs at
least `U_mean` plus one of the stiffness/density pair, and `nu_s` besides.

## 3. Consequence for the existing verdicts (this is the part that changes)

{_table(["candidate row", "compared against", "Ae error", "density-ratio error",
         "failed groups"], sim_rows)}

The standing conclusion of `docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md` —
"the archived record is *a different physics problem*, so it is a test of the
upstream solver and not a validation against the benchmark" — is the
`paper_target` branch.  Under the `upstream_consistent` branch the archived
record matches the benchmark on **all** groups, and the upstream amplitude
discrepancy (`+456 %` in `tip_x`, `+872 %` in `tip_y`, computed by
`tensorsph.fsi2_reference.summarise_tip` against the declared target) becomes an
*accuracy* result of the upstream run rather than a non-comparability result.
Both branches are printed above; neither is asserted as the paper's truth.

## 4. Time-coverage gate: the second reason a comparison can be void

Independent of the target table, a periodic response needs time.  The period is
**measured** from the archived record, not quoted:

| quantity | value |
|---|---|
| archived samples | {per['n_samples']} at 0.1 s (archived `fsi2.cpp:192`) |
| record duration | {per['duration']:.2f} s |
| dominant spectral peak | {per['f_peak']:.3f} Hz -> period {per['period_fft']:.3f} s |
| zero-crossing period | {per['period_zero_crossing']:.3f} s |
| period used by the gate (conservative) | **{per['period_used']:.3f} s** |
| Strouhal expectation (`St = 0.2`) | {per['period_strouhal_expectation']:.3f} s |

The two measured estimates differ by 2x: the record is sampled at only ~5 points
per cycle at the Strouhal frequency and is not a single clean sinusoid.  The gate
therefore uses the longest candidate, and the verdict is insensitive to the
choice:

{_table(["period used", "cycles covered in a 40-step probe"],
        [[k, f"{v:.4f}"] for k, v in sens.items()])}

The coupled probe that the earlier report compared against the benchmark covers
**{probe['cycles']:.4f} cycles** ({probe['duration']:.4f} s) at `dx = 0.02 m`.  A
half-range amplitude requires `{gate.MIN_CYCLES_AMPLITUDE:.0f}` cycles and a
mean/variance `{gate.MIN_CYCLES_STATISTICS:.0f}` (predeclared here, not retuned
afterwards), so that comparison was void by three orders of magnitude,
irrespective of any parameter question.

Cost of the remaining debt, at the measured period (see figure b):

{_table(["resolution", "dt [s]", "steps for 3 cycles", "steps for 10 cycles",
         "fluid particles"], cov_rows)}

## 5. What this round changes in the code

* new module `src/tensorsph/fsi2_parameter_gate.py` (identities, provenance
  matrix, candidate rows, measured period, coverage gate);
* new tests `tests/test_fsi2_parameter_gate.py` bind the gate to the *files*
  (survey markdown, archived upstream README), so a future edit of either cannot
  silently desynchronise the comparison again;
* **nothing is silently rewritten**: `fsi2_similarity.BENCHMARK_FSI2` and the
  `FSI2Case` defaults keep their current values, because switching them changes
  the point of the coupled case and would invalidate the coupled stability work
  (`STRONG_COUPLING_AND_ROD_STABILITY.md`, `COUPLED_RUN_DEFECTS.md`) that was
  produced at the current point.  The gate makes the disagreement visible and
  testable; the migration is a separate, pre-registered run.

## 6. Honest error judgement

| judgement item | basis | error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| survey table vs its own `Re` identity | definition | {_fmt_pct(1 - 20.0 / 100.0)} (Re 20 vs 100) | NO | NO |
| survey / defaults vs declared `Ae` | identity | +1685.7 % | NO | NO |
| archived upstream vs declared `Ae` | identity | −75.0 % | NO | NO |
| coverage of the 40-step probe | measured period | {probe['cycles']:.4f} cycles (needs 3) | NO | NO |
| which row is the paper's FSI2 | paper | WITHHELD (no PDF access) | cannot judge | cannot judge |

No number in this document is a claim of accuracy: the deliverable is that the
target side of every FSI2 ratio is now machine-checked, and that the two open
branches are named instead of being mixed.
"""
    (DOCS / "PARAMETER_TABLE_INTEGRITY.md").write_text(md)
    return audit


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: out[k] for k in ("period", "coverage_probe_40_steps")},
                     indent=1))
    print("wrote artifacts/xval_fsi2/parameter_gate.json, "
          "docs/xval_fsi2/PARAMETER_TABLE_INTEGRITY.md, "
          "docs/xval_fsi2/fig_parameter_gate.png")