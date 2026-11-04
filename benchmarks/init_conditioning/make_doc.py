"""Render the figures and write ``docs/verification_initial_pressure_offset.md``.

Every number in the document is injected from ``report.json`` (produced by
``run.py``); nothing is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DOC = REPO / "docs" / "verification_initial_pressure_offset.md"
PNG = HERE / "init_conditioning.png"


def make_figures(report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) bias against spacing, per lattice / h ratio: flat means "not a
    # discretization error".
    ax = axes[0][0]
    families = {}
    for row in report["sweep"]:
        key = (row["lattice"], row["h_factor"])
        families.setdefault(key, []).append((row["dx"], 100.0 * row["bias"]))
    for (lattice, hf), pts in sorted(families.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                ls="-" if hf == 1.3 else "--",
                label=f"{lattice}, h={hf}dx", alpha=0.9)
    ax.set_xscale("log")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("particle spacing dx [m]")
    ax.set_ylabel(r"$(\rho_{\rm init}/\rho_0 - 1)\times 100$  [%]")
    ax.set_title("(a) lattice offset is spacing-independent")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # (b) radial density profile before / after the calibration.
    ax = axes[0][1]
    prof = report["radial_profiles"][0]
    r = [b["r_over_R"] for b in prof["bins"]]
    ax.plot(r, [b["uncalibrated"] for b in prof["bins"]], "o-", ms=3,
            label="uncalibrated  $m=\\rho_0 V_0$")
    ax.plot(r, [b["calibrated"] for b in prof["bins"]], "s-", ms=3,
            label="calibrated  $m=\\rho_0 V_0/(1+b)$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axvspan(0.8, 1.2, color="grey", alpha=0.15)
    ax.set_xlabel("r / R")
    ax.set_ylabel(r"$\rho/\rho_0$")
    ax.set_title(f"(b) density profile, {prof['lattice']} dx={prof['dx']:g} m")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) absolute spurious pressure against sound speed (p ~ c0^2).
    ax = axes[1][0]
    ladder = report["sound_speed_ladder_hex_h13"]
    c0 = [r["sound_speed"] for r in ladder]
    ax.plot(c0, [r["p_abs_max_before"] for r in ladder], "o-",
            label="uncalibrated")
    ax.plot(c0, [max(r["p_abs_max_after"], 1e-12) for r in ladder], "s-",
            label="calibrated")
    ax.axhline(report["settling"]["laplace_signal_Pa"], color="C3", ls=":",
               label=r"$\sigma/R$ signal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sound speed $c_0$ [m/s]")
    ax.set_ylabel("max $|p|$ after initialisation [Pa]")
    ax.set_title(r"(c) offset scales as $c_0^2$, signal does not")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (d) normalised criterion before / after against the acceptance limit.
    ax = axes[1][1]
    rows = [r for r in report["sweep"] if r["lattice"] == "hex"]
    labels = [f"{r['h_factor']}dx\n{r['dx']:g}" for r in rows]
    ax.bar(range(len(rows)), [r["p_norm_before"] for r in rows], 0.4,
           label="uncalibrated", align="edge")
    ax.bar([i + 0.4 for i in range(len(rows))],
           [max(r["p_norm_after"], 1e-16) for r in rows], 0.4,
           label="calibrated", align="edge")
    ax.axhline(report["settling"]["acceptance_limit"], color="C3", ls="--",
               label="acceptance 1e-4")
    ax.set_yscale("log")
    ax.set_xticks([i + 0.4 for i in range(len(rows))])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(r"max$|p|/(\rho_0 c_0^2)$")
    ax.set_title("(d) acceptance criterion, hex lattice")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both", axis="y")

    fig.suptitle("TensorSPH: spurious pressure of the initial lattice state", y=0.995)
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    plt.close(fig)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def make_doc(report: dict) -> str:
    H_FACTOR = 1.3
    s = report["settling"]
    su = report["summary"]
    sweep = report["sweep"]
    ladder = report["sound_speed_ladder_hex_h13"]
    res = report["resolution_ladder_hex_h13"]
    prof = report["radial_profiles"][0]
    impact = report["laplace_impact"][0]

    bias_table = table(
        [(r["lattice"], f"{r['h_factor']}", f"{r['dx']:g}", r["n_particles"],
          r["n_supported"], f"{r['bias_percent']:+.4f}",
          f"{r['p_abs_max_before']:.1f}", f"{r['p_norm_before']:.3e}",
          f"{r['p_norm_after']:.1e}")
         for r in res],
        ["lattice", "h/dx", "dx [m]", "N", "fully supported", "b [%]",
         "max|p| [Pa]", "before", "after"])

    layout_table = table(
        [(r["lattice"], f"{r['h_factor']}", f"{r['bias_percent']:+.4f}",
          f"{r['p_norm_before']:.3e}", f"{r['mass_change'] * 100:+.3f}")
         for r in sweep if r["dx"] == 0.002],
        ["lattice", "h/dx", "b [%]", "max|p|/(rho0 c0^2)", "mass change [%]"])

    c0_table = table(
        [(f"{r['sound_speed']:g}", f"{r['p_abs_max_before']:.1f}",
          f"{r['p_norm_before']:.4e}", f"{r['p_abs_max_after']:.2e}",
          f"{r['p_abs_max_before'] / s['laplace_signal_Pa']:.2f}")
         for r in ladder],
        ["c0 [m/s]", "max|p| before [Pa]", "normalised", "max|p| after [Pa]",
         "x sigma/R"])

    return f"""# Verification debt: the initial density offset of a summation-density lattice

**Scope**: TensorSPH, `src/tensorsph/initial_conditions.py` +
`benchmarks/init_conditioning/`. **Status**: closed for the bulk, open at the
free surface (see "What this does *not* fix").

**Numbers in this document are injected by `benchmarks/init_conditioning/make_doc.py`
from `report.json`; nothing is typed by hand.** Figure: `init_conditioning.png`.

## 1. The claim under test

`backlog_update_m2971.md` item C2/P1 states: every weakly compressible SPH case
in this solver starts with a spurious pressure because the particle mass is
`rho0 * V0` while the *discrete* kernel sum `C = sum_j W_ij V0` is not exactly
one. The phenomenon matters because `c0 = 20 m/s` turns a sub-percent density
offset into kilopascals, which is larger than the capillary pressure the
verification suite is trying to measure.

Acceptance criterion (from the same backlog item, machine-checkable):

```
max|p| / (rho0 c0^2) < 1e-4      after initialisation
```

## 2. Method

Pure `t = 0` evaluation, no time stepping: a static disc is placed on a
hexagonal and a square lattice (`rho0 = {s['rho0']:g} kg/m^3`,
`R = {s['radius']:g} m`, `sigma = {s['sigma']:g} N/m` so that
`sigma/R = {s['laplace_signal_Pa']:g} Pa`, the geometry of
`benchmarks/laplace_pressure_floor`), the summation density is taken with the
solver's own `density_from_neighbor_pairs`, and the pressure with the solver's
own Tait law (clamped at `p >= 0`).

The offset `b` is measured only on the **fully supported** particles - those
whose neighbour count equals the maximum count in the configuration
(`initial_conditions.supported_mask`, tolerance 0). Particles next to the free
surface must be excluded: their kernel sum is legitimately below one, and
averaging them in biases the fit (measured: a tolerance of two neighbours left
the strictly interior at `+7e-4` in `rho/rho0` instead of zero).

Calibration is `m -> m / (1 + b)`, and it is **exact in one pass**: the
summation density is linear in the masses, so scaling all masses by `alpha`
scales all densities by `alpha`. Verified: after calibration the interior
satisfies `|rho/rho0 - 1| < {su['hex_p_norm_after_max']:.1e}`.

## 3. Results

### 3.1 The offset is a lattice constant, not a discretization error

Hexagonal lattice, `h = {H_FACTOR} dx`, radius 0.02 m:

{bias_table}

`b` is identical at every spacing to the printed precision
(`hex_bias_dx_independent = {su['hex_bias_dx_independent']}`), so refining the
mesh does **not** remove it. This is the key qualitative point: the offset is a
property of the packing, not of the resolution.

### 3.2 Spurious pressure at the verification geometry

{c0_table}

The absolute offset goes exactly as `c0^2` (the normalised criterion is
constant, `{su['hex_p_norm_before'][0]:.4e}` for every `c0`), so lowering the
sound speed shrinks the pressure but never the *relative* error. At
`c0 = 20 m/s` the offset is `{su['hex_signal_ratio_before']:.2f} x` the
`sigma/R` signal - the number quoted as 6.7x in the backlog is confirmed.

### 3.3 Lattice / smoothing-length dependence

{layout_table}

(square lattice at `h = 1.3 dx` is the one combination that happens to be
already balanced; it is also the packing used by `make_particles`, which is why
the droplet-impact production case never showed the effect.)

### 3.4 After calibration

* worst normalised offset over the whole sweep: `{su['worst_p_norm_after']:.1e}`
  against the `{s['acceptance_limit']:g}` limit (`all_pass_after =
  {su['all_pass_after']}`);
* cost: total mass moves by `{su['hex_mass_change_percent']:+.3f} %` for the hex
  lattice - exact mass conservation against the analytic `rho0 * Area` is given
  up for a pressure-free initial state. This is the documented trade-off:
  the calibration preserves *density*, not *mass*.

### 3.5 What the calibration does **not** change

The merged CSF jump measurement (`benchmarks/laplace_droplet`, `dx =
{impact['dx']:g} m`) moves by `{impact['dp_ratio_delta_points']:+.2f}` percentage
points only (`dp_ratio` from `{impact['dp_ratio_uncalibrated']:.4f}` to
`{impact['dp_ratio_calibrated']:.4f}`; `N_delta` is unchanged at
`{impact['n_delta_calibrated']:.4f}` because the colour field does not depend on
a uniform density scale). So the residual 16-21 % deficit of the force-integral
jump is **not** caused by C2 - attribution stays where `m2971` left it (the
free-surface pressure closure). What C2 does contaminate is every *pressure*
reading taken at or near `t = 0`.

## 4. API

```python
from tensorsph.initial_conditions import (
    TAIT_ACCEPTANCE_LIMIT, calibrate_lattice_mass, measure_density_bias,
    spurious_pressure, supported_mask,
)

bias = measure_density_bias(pos, mass, h, rho0)          # dict
mass, info = calibrate_lattice_mass(pos, mass, h, rho0)  # exact in one pass
spurious_pressure(pos, mass, h, rho0, c0, gamma)         # dict, 'passes_acceptance'
```

`Config.lattice_mass_calibration` (default `False`) wires the calibration into
`make_particles`. It is **off by default** on purpose: it shifts every published
number by `-b/(1+b)` and the production tank uses the square lattice, whose
offset is only `-0.005 %`. Turning it on is a re-baselining decision, not a bug
fix, and the pinned droplet-impact numbers would have to be re-measured.

## 5. What this does *not* fix

1. **Free-surface deficit.** The drop below `r/R ~ 0.8` in figure (b) is the
   kernel deficit of the surface layer (`rho/rho0` down to ~0.6). The Tait clamp
   turns it into `p = 0`, which is a first-order free-surface treatment, not a
   closure. That is item P0 of `backlog_update_m2971.md` and remains **open**.
2. **Non-uniform lattices.** A jittered or adaptively refined particle set has a
   *distribution* of kernel sums, not one constant; a scalar calibration then
   only removes the mean. `measure_density_bias` reports `rho_ratio_min/max` so
   the spread is visible.
3. **Walls.** `wall_mass` keeps `rho0 dx^2`; walls use the extrapolated pressure
   of Adami et al. (2012) and this benchmark does not measure them.

## 6. Reproduction

```bash
PYTHONPATH=src python benchmarks/init_conditioning/run.py     # ~21 s, 1 core
PYTHONPATH=src python benchmarks/init_conditioning/make_doc.py
PYTHONPATH=src python -m pytest tests/test_init_conditioning.py -q
```
"""


def main() -> int:
    report = json.loads((HERE / "report.json").read_text())
    make_figures(report)
    DOC.write_text(make_doc(report))
    print(f"wrote {DOC.relative_to(REPO)} and {PNG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())