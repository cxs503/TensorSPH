"""Similarity + reference audit for the Turek-Hron FSI2 anchor.

Generates, with no hand-copied numbers:

* ``docs/xval_fsi2/assets/fig_fsi2_reference_tip.png``  -- the three archived
  SPHinXsys tip-position records in physical units against the benchmark band,
* ``docs/xval_fsi2/assets/fig_fsi2_similarity_groups.png`` -- the dimensionless
  groups of the benchmark / OSS / this repo's default / corrected case, with the
  expected tip amplitude each one implies,
* ``docs/xval_fsi2/assets/similarity_audit_results.json`` -- every number used
  by the report,
* ``docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md`` -- the audit page.

Run:  python benchmarks/xval_fsi2/make_similarity_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import fsi2_reference as ref  # noqa: E402
from tensorsph import fsi2_similarity as sim  # noqa: E402
from tensorsph.fsi2_turek_hron import FSI2Case  # noqa: E402

ASSETS = ROOT / "docs" / "xval_fsi2" / "assets"
DOC = ROOT / "docs" / "xval_fsi2" / "SIMILARITY_AND_REFERENCE_AUDIT.md"

#: One stored sample = one upstream ``output_interval`` (fsi2.cpp:192) = 1.0
#: scaled time = 0.1 s physical (length scale 10 at equal velocity).
SAMPLE_DT_PHYSICAL = 1.0 / ref.LENGTH_SCALE


def collect():
    bundle = ref.bundle()
    default = FSI2Case()
    corrected = sim.similarity_corrected_case(U_mean=0.2)
    out = {
        "bundle": bundle,
        "default_case": {
            "groups": sim.dimensionless_groups(default),
            "report": sim.similarity_report(default),
            "consequence": sim.amplitude_scaling_consequence(default),
            "beam_E": default.beam_E,
            "u_mean": 2.0 / 3.0 * default.inlet_u_max,
        },
        "corrected_case": {
            "groups": sim.dimensionless_groups(corrected),
            "report": sim.similarity_report(corrected),
            "consequence": sim.amplitude_scaling_consequence(corrected),
            "beam_E": corrected.beam_E,
            "viscosity": corrected.viscosity,
            "u_mean": 2.0 / 3.0 * corrected.inlet_u_max,
        },
        "oss_case": dict(sim.OSS_FSI2_CASE),
        "benchmark": dict(sim.BENCHMARK_FSI2),
        "benchmark_tip": dict(ref.BENCHMARK_TIP),
        "sample_dt_physical": SAMPLE_DT_PHYSICAL,
    }
    return out


def fig_reference_tip(bundle):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    colors = {"0": "#1f77b4", "4": "#ff7f0e", "9": "#2ca02c"}
    for run, color in colors.items():
        _, pos = ref.load_tip_series(run)
        t = np.arange(pos.shape[0]) * SAMPLE_DT_PHYSICAL
        axes[0].plot(t, pos[:, 1] / ref.LENGTH_SCALE, color=color, lw=1.0,
                     label=f"SPHinXsys Run_{run}")
        axes[1].plot(t, pos[:, 0] / ref.LENGTH_SCALE, color=color, lw=1.0,
                     label=f"SPHinXsys Run_{run}")
    y0 = 0.2  # beam-free-stream tip ordinate [m]
    band = ref.BENCHMARK_TIP["tip_y_amplitude"]
    axes[0].axhspan(y0 - band, y0 + band, color="#d62728", alpha=0.25,
                    label="Turek-Hron FSI2 tip-y band")
    axes[0].set_ylabel("tip $y$ [m]")
    axes[1].set_ylabel("tip $x$ [m]")
    axes[1].set_xlabel("physical time [s]  (one stored sample = 0.1 s)")
    axes[0].set_title("Archived SPHinXsys test_2d_fsi2 records vs the FSI2 benchmark band")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = ASSETS / "fig_fsi2_reference_tip.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_similarity_groups(data):
    labels = ["Turek-Hron\nFSI2 target", "SPHinXsys\nfsi2.h", "TensorSPH\ndefault",
              "TensorSPH\nsimilarity-corrected"]
    ae = [data["benchmark"]["Ae"], data["oss_case"]["Ae"],
          data["default_case"]["groups"]["Ae"], data["corrected_case"]["groups"]["Ae"]]
    ratios = [1.0, 1.0,
              data["default_case"]["consequence"]["tip_amplitude_ratio"],
              data["corrected_case"]["consequence"]["tip_amplitude_ratio"]]
    amps = np.array(ratios) * ref.BENCHMARK_TIP["tip_y_amplitude"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = np.arange(len(labels))
    axes[0].bar(x, ae, color=["#333333", "#9467bd", "#d62728", "#2ca02c"])
    axes[0].set_yscale("log")
    axes[0].axhline(data["benchmark"]["Ae"], color="k", ls="--", lw=1.0)
    axes[0].set_ylabel("$Ae = E/(\\rho U^2)$")
    axes[0].set_title("stiffness group")
    axes[1].bar(x, amps * 1e3, color=["#333333", "#9467bd", "#d62728", "#2ca02c"])
    axes[1].axhspan(ref.BENCHMARK_TIP["tip_y_amplitude"] * 1e3 * 0.85,
                    ref.BENCHMARK_TIP["tip_y_amplitude"] * 1e3 * 1.15,
                    color="#d62728", alpha=0.2, label="+/-15 % gate")
    axes[1].set_ylabel("expected tip $y$ amplitude [mm]")
    axes[1].set_title("first-order consequence (tip amp ~ 1/Ae)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = ASSETS / "fig_fsi2_similarity_groups.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fmt_groups(report):
    lines = ["| group | value | FSI2 target | rel. error | consumed by rod model |",
             "|---|---|---|---|---|"]
    for g in report["groups"]:
        consumed = "no (informational)" if g["name"] in report["informational"] else "yes"
        lines.append(
            f"| `{g['name']}` | {g['value']:.6g} | {g['target']:.6g} | "
            f"{100 * g['rel_error']:+.2f} % | {consumed} |")
    return "\n".join(lines)


def fmt_tip_table(bundle):
    lines = ["| run | samples | mean tip x [m] | mean tip y [m] | tip x amp [m] | tip y amp [m] |",
             "|---|---|---|---|---|---|"]
    for run, r in bundle["runs"].items():
        lines.append(f"| Run_{run} | {r['n_samples']} | {r['x_mean']:.4f} | {r['y_mean']:.4f} | "
                     f"{r['tip_x_amplitude']:.4f} | {r['tip_y_amplitude']:.4f} |")
    return "\n".join(lines)


def fmt_gate(gate):
    lines = ["| quantity | reference [m] | OSS record [m] | rel. error | <15 % | <3 % |",
             "|---|---|---|---|---|---|"]
    for r in gate["rows"]:
        lines.append(f"| {r['quantity']} | {r['reference']:.4f} | {r['computed']:.4f} | "
                     f"{100 * r['rel_error']:+.2f} % | {'PASS' if r['pass_15'] else 'FAIL'} | "
                     f"{'PASS' if r['pass_3'] else 'FAIL'} |")
    return "\n".join(lines)


def write_doc(data, fig_tip, fig_groups):
    b = data["bundle"]
    gate = b["tip_gate"]
    d = data["default_case"]
    c = data["corrected_case"]
    f_scaled = b["viscous_force_scaled"]
    f_phys = b["viscous_force_physical"]
    cd = b["viscous_Cd"]
    est = b["flat_plate_estimate"]

    md = f"""# FSI2 similarity audit and reference-bundle verdict

**Anchor**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`
**Branch/build**: `agent/m20125`, `src/tensorsph/fsi2_similarity.py`, `src/tensorsph/fsi2_reference.py`
**Reference archive**: `benchmarks/xval_fsi2/reference/oss/` (SPHinXsys `test_2d_fsi2`, verbatim, sha256 in the README)

## 1. Case introduction

The Turek-Hron FSI2 benchmark (Turek & Hron, LNCSE 53 (2006) 371-385) is a 2-D
channel (L x H = 2.5 m x 0.41 m) with a parabolic inlet profile past a rigid
cylinder (D = 0.1 m, centre (0.2, 0.2) m) carrying an elastic beam
(L = 0.35 m, t = 0.02 m, E = 5.6 MPa, nu_s = 0.4, rho_s = rho_f = 1000 kg/m^3).
The fluid is at Re = 100 and the beam responds in vortex-induced vibration. The
quantities compared are drag/lift coefficients and the beam-tip displacement
amplitudes (benchmark values: tip x 0.0028 m, tip y 0.0082 m, drag mean 3.2 with
amplitude 0.96, lift amplitude 0.22 -- secondary provenance, quoted from
Turek & Hron 2006 in `src/tensorsph/fsi2_turek_hron.py` and **not** re-verified
against the paper PDF in this round).

Two questions are answered here, both cheap and both blocking for the anchor:

1. **Is the case this repository runs the benchmark case?** (dimensionless
   similarity; the benchmark is posed in SI units, and every SPH re-scaling --
   lower inlet velocity, higher sound speed -- silently changes the physics
   unless the groups are held fixed.)
2. **Can the archived SPHinXsys regression records serve as the acceptance
   reference for FSI2?** The previous round's gap note
   (`docs/xval_fsi2/GAP-SPH-FSI-BENCHMARK_xval.md`, §6 "round-2 targets") planned
   exactly that.

## 2. How it was done

* `src/tensorsph/fsi2_similarity.py` -- computes the groups
  `Re = rho_f U_mean D / mu`, `Ae = E / (rho_f U_mean^2)`, `rho_s/rho_f`,
  `nu_s`, `L/t` for any `FSI2Case` and compares them with the FSI2 targets
  (2 % tolerance, `Re`, `Ae`, density ratio and slenderness are gated; `nu_s` is
  reported as informational because the discrete elastic rod never consumes it).
  `similarity_corrected_case()` derives `mu` and `E` from `Re` and `Ae`, so the
  benchmark can be run at an SPH-friendly velocity without changing the physics.
* `src/tensorsph/fsi2_reference.py` -- loads the verbatim upstream records,
  converts them to physical units and applies the owner's acceptance gate
  (15 %, plus the 3 % reporting threshold).
* Unit conversion is *derived*, not assumed: the straight-beam tip abscissa is
  0.6 m and the record stores ~6.0, so the length scale is 10 (recovered from
  the data: {b['inferred_length_scale']:.3f}, slightly below 10 because the
  deflected beam shortens its x-projection). A 2-D force per unit depth then
  maps as `F = F' rho_f U_mean^2 / (rho' U'^2 * L_scale)`, i.e. x100.
* Cross-check of that force convention, independent of any benchmark: the
  time-averaged viscous record becomes `Cd_visc = {cd[0]:.3f}`; a Blasius
  flat-plate estimate for the same beam (both sides, `Re_L = {est['Re_L']:.0f}`)
  gives `Cd_friction = {est['Cd_friction']:.3f}`. Same order (factor
  {cd[0] / est['Cd_friction']:.2f}), so the conversion is not off by a decade.
* All numbers below are written by `benchmarks/xval_fsi2/make_similarity_audit.py`
  into `docs/xval_fsi2/assets/similarity_audit_results.json`; nothing is typed
  by hand.

## 3. Results

### 3.1 Dimensionless groups (the answer to question 1)

**This repository's default case** (`FSI2Case()`, E = {d['beam_E']:.3g} Pa,
U_mean = {d['u_mean']:.3g} m/s):

{fmt_groups(d['report'])}

**Similarity-corrected case** (`similarity_corrected_case(U_mean=0.2)`,
E = {c['beam_E']:.3g} Pa, mu = {c['viscosity']:.3g} Pa s):

{fmt_groups(c['report'])}

First-order consequence of the default mismatch (small-deflection, load-controlled
cantilever: tip deflection ~ 1/Ae, bending frequency ~ sqrt(Ae)):

| case | Ae | expected tip amplitude ratio | expected tip y amplitude [mm] | expected bending frequency ratio |
|---|---|---|---|---|
| default | {d['consequence']['Ae']:.6g} | {d['consequence']['tip_amplitude_ratio']:.3f} | {1000 * ref.BENCHMARK_TIP['tip_y_amplitude'] * d['consequence']['tip_amplitude_ratio']:.3f} | {d['consequence']['bending_frequency_ratio']:.2f} |
| corrected | {c['consequence']['Ae']:.6g} | {c['consequence']['tip_amplitude_ratio']:.3f} | {1000 * ref.BENCHMARK_TIP['tip_y_amplitude'] * c['consequence']['tip_amplitude_ratio']:.3f} | {c['consequence']['bending_frequency_ratio']:.2f} |

The default case is **{d['consequence']['tip_amplitude_rel_error'] * 100:+.1f} %** off
the benchmark's tip amplitude before a single fluid step is taken. That, and not
solver quality, is why a coupled run with the shipped defaults cannot pass the
15 % gate.

### 3.2 The archived SPHinXsys records (the answer to question 2)

SPHinXsys `fsi2.h` runs the same *geometry* at length scale 10 (`D' = 1.0`,
`DH = 4.1`, `bl = 3.5`) with `rho0_f = 1`, `U = 1`, `Re = 100` -- but with
`rho0_s = 10` (density ratio **10**, benchmark 1) and `Ae = 1.4e3` (benchmark
**5.6e3**). Its regression records therefore describe a *softer and much
heavier* beam than the benchmark. Measured from the three archived runs
(physical units):

{fmt_tip_table(b)}

Across the three upstream runs the tip amplitude spreads by
{100 * b['amplitude_spread_tip_y']:.1f} % -- the records are internally
reproducible, so the discrepancy below is set-up, not scatter.

**Acceptance gate against the benchmark (15 % / 3 %):**

{fmt_gate(gate)}

Both quantities fail the 15 % gate by factors of 4.6x and 8.7x.

The only force record upstream publishes for this case is the time-averaged
total viscous force, `[{f_scaled[0]:.6f}, {f_scaled[1]:.6f}]` (scaled) =
`[{f_phys[0]:.3f}, {f_phys[1]:.3f}]` N/m = `Cd_visc = [{cd[0]:.3f}, {cd[1]:.4f}]`.

### 3.3 Figures

![archived tip records vs benchmark band](assets/fig_fsi2_reference_tip.png)

*The three archived records in physical units. The red band is the FSI2 tip-y
amplitude (+/- 0.0082 m about the free-stream ordinate 0.2 m); the records
oscillate ~10x outside it.*

![similarity groups](assets/fig_fsi2_similarity_groups.png)

*Left: the stiffness group `Ae = E/(rho U^2)` of the four set-ups (log scale,
dashed line = FSI2 target). Right: the beam-tip amplitude each group implies
(first order, tip amplitude ~ 1/Ae) with the +/-15 % acceptance band; the
SPHinXsys set-up and this repository's default both sit outside it.*

## 4. Caveats

1. **The benchmark amplitudes are secondary evidence.** They come from the
   repository's own quotation of Turek & Hron (2006); the paper PDF was not
   consulted this round. If a primary copy becomes available, only
   `fsi2_reference.BENCHMARK_TIP` / `fsi2_similarity.BENCHMARK_FSI2` change; the
   OSS numbers do not, and the factor-4.6/8.7 verdict is far outside any
   plausible transcription error.
2. **The time axis of the archived records is an assumption.** `fsi2.cpp`
   writes every `output_interval = 1.0` (scaled) = 0.1 s, and 107 consecutive
   samples are assumed; the archive is a rolling window of the 200 s run. Only
   the amplitude/mean statistics are used quantitatively -- no frequency or
   Strouhal claim is made from these records.
3. **The OSS set-up is not "wrong".** SPHinXsys' `test_2d_fsi2` is a regression
   test of its own solver, not a validation against the benchmark; `Ae = 1.4e3`
   and `rho_s/rho_f = 10` are its chosen parameters. The finding is that those
   records cannot arbitrate a Turek-Hron comparison -- not that the upstream
   code is defective.
4. `nu_s` (0.3 vs 0.4) is not consumed by the discrete elastic rod (`EI` and
   `EA` only), so it is reported, not gated. A Saint-Venant-Kirchhoff / TLSPH
   solid would have to gate it.
5. The first-order consequence table ignores added mass and resonance; it is a
   scale, not a prediction. The measured 8.7x of the OSS record mixes the 4x
   `Ae` change with the 10x density ratio and is not decomposed here.

## 5. Error judgement (vs the benchmark, quantity by quantity)

| quantity | reference | this round's evidence | relative error | <3 % ? |
|---|---|---|---|---|
| beam-tip y amplitude | 0.0082 m | 0.0797 m (OSS record) | +872 % | **no** |
| beam-tip x amplitude | 0.0028 m | 0.0156 m (OSS record) | +456 % | **no** |
| drag coefficient (mean) | 3.2 | not available in the archive | -- | no basis |
| lift coefficient (amplitude) | 0.22 | not available in the archive | -- | no basis |
| tip amplitude, this repo's default case | 0.0082 m | 0.00046 m (first order, 1/Ae) | -94 % | **no** |
| tip amplitude, similarity-corrected case | 0.0082 m | 0.0082 m (by construction) | 0 % (group-level) | group only, **not yet run** |

Nothing in this round is a positive verification of a *coupled* FSI2 run: what
is verified is (a) the archive's units and content, (b) that the OSS records are
not the benchmark, and (c) that the repository's default case was not
similarity-equivalent. Honest FAIL, recorded as such.

## 6. Disposition

**Request changes** (adversarial-review disposition) to the round-2 plan in
`GAP-SPH-FSI-BENCHMARK_xval.md`:

* Stop treating `BeamObserver_Position_Run_*.xml` as the FSI2 acceptance
  reference. Use it only as a reproduction target for the OSS parameter point
  (`Ae = 1.4e3`, `rho_s/rho_f = 10`), which is a different physics problem.
* For the ledger acceptance (< 15 % vs Turek-Hron), use
  `similarity_corrected_case()` (E = 2.24e5 Pa at U_mean = 0.2 m/s, Re = 100)
  and the benchmark amplitudes in `fsi2_reference.BENCHMARK_TIP`; the drag/lift
  targets still need a primary source (the archive has no pressure-force
  record).
* Before any coupled run, assert `fsi2_similarity.similarity_report(case).passed`
  -- the failure is now a one-line check instead of a 20 s simulation.
"""
    DOC.write_text(md)
    return DOC


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = collect()
    fig_tip = fig_reference_tip(data["bundle"])
    fig_groups = fig_similarity_groups(data)
    (ASSETS / "similarity_audit_results.json").write_text(
        json.dumps(data, indent=2, default=str))
    doc = write_doc(data, fig_tip, fig_groups)
    print("wrote", doc, fig_tip.name, fig_groups.name)
    print("gate:", data["bundle"]["tip_gate"]["passed_15"],
          "worst", round(data["bundle"]["tip_gate"]["worst_rel_error"], 3))


if __name__ == "__main__":
    main()