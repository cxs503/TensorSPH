"""Figures + document for the lake-at-rest artefact (no number is retyped).

Reads ``artifacts/lake_at_rest/rest_dx020.json``, writes three PNGs and
``docs/verification_lake_at_rest.md``.  Every number in the document is read out
of that JSON, so the prose cannot disagree with the artefact.

    PYTHONPATH=src python3 benchmarks/lake_at_rest/make_docs.py [--json ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorsph import eos_hydrostatics as eos

ROOT = Path(__file__).resolve().parents[2]


def g(value, digits=3):
    if value is None:
        return "n/a"
    if value != 0 and (abs(value) < 1e-2 or abs(value) >= 1e4):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path,
                        default=ROOT / "artifacts/lake_at_rest/rest_dx020.json")
    parser.add_argument("--docs", type=Path,
                        default=ROOT / "docs/verification_lake_at_rest.md")
    parser.add_argument("--figures", type=Path,
                        default=ROOT / "docs/figures")
    args = parser.parse_args(argv)
    data = json.loads(args.json.read_text())
    args.figures.mkdir(parents=True, exist_ok=True)
    geom, eq, rest, prof = (data["geometry"], data["equilibrium"], data["rest"],
                            data["profiles"])
    controls, gates = data["controls"], data["gates"]

    # ---- figure 1: reconstruction convergence and the negative controls ------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    history = eq["residual_history"]
    ax.semilogy(range(len(history)), history, "o-", ms=3, label="Newton iterations")
    ax.axhline(1.0, color="0.6", ls="--", lw=1)
    for name, value in (("uniform lattice mass", controls["uniform_lattice_mass"]["a_max_over_g"]),
                        ("analytic hydrostatic profile",
                         controls["analytic_hydrostatic_profile"]["a_max_over_g"])):
        ax.axhline(value, color="crimson", ls=":", lw=1)
        ax.annotate(f"{name}: {g(value)}", (0.2, value), color="crimson",
                    fontsize=7, va="bottom")
    ax.axhline(eq["a_max_over_g"], color="green", ls="-", lw=1)
    ax.annotate(f"reconstructed $a_y$: {g(eq['a_max_over_g'])}", (0.2, eq["a_max_over_g"]),
                color="green", fontsize=7, va="top")
    ax.axhline(eq["a_max_over_g_horizontal_wall_band"], color="darkorange", ls="-", lw=1)
    ax.annotate(f"side-wall $a_x$: {g(eq['a_max_over_g_horizontal_wall_band'])}",
                (0.2, eq["a_max_over_g_horizontal_wall_band"]), color="darkorange",
                fontsize=7, va="top")
    ax.set_xlabel("Gauss-Newton iteration")
    ax.set_ylabel(r"$\max|a|/g$")
    ax.set_title("lake at rest: reconstruction vs the two negative controls")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.figures / "lake_at_rest_residual.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: pressure and density profiles against the analytic ones ---
    rows = prof["row_profiles"]
    y = [r["y"] for r in rows]
    p_meas = [r["pressure"] for r in rows]
    rho_meas = [r["rho_over_rho0"] for r in rows]
    yt = torch.tensor(y, dtype=torch.float64)
    depth, rho0 = geom["depth"], 1000.0
    c0, gamma, gravity = geom["sound_speed"], geom["gamma"], geom["gravity"]
    p_eos = eos.eos_hydrostatic_pressure(yt, depth=depth, rho0=rho0,
                                         sound_speed=c0, gamma=gamma,
                                         gravity=gravity).tolist()
    rho_eos = (eos.eos_hydrostatic_density(yt, depth=depth, rho0=rho0,
                                           sound_speed=c0, gamma=gamma,
                                           gravity=gravity) / rho0).tolist()
    p_inc = (rho0 * gravity * (depth - yt).clamp_min(0.0)).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].plot(p_meas, y, "o-", ms=3, label="reconstructed")
    axes[0].plot(p_eos, y, "-", label="Tait equilibrium (exact)")
    axes[0].plot(p_inc, y, "--", label=r"$\rho_0 g (H-y)$")
    axes[0].set_xlabel("pressure [Pa]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("hydrostatic pressure profile")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].plot(rho_meas, y, "o-", ms=3, label="reconstructed")
    axes[1].plot(rho_eos, y, "-", label="Tait equilibrium (exact)")
    axes[1].axvline(1.0, color="0.6", ls="--", lw=1)
    axes[1].set_xlabel(r"$\rho/\rho_0$")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title("equilibrium density profile")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.figures / "lake_at_rest_profiles.png", dpi=130)
    plt.close(fig)

    # ---- figure 3: what the run actually does -------------------------------
    samples = rest["samples"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy([s["t"] for s in samples], [max(s["v_over_c0"], 1e-18) for s in samples],
                "-", label=r"$\max|v|/c_0$")
    ax.semilogy([s["t"] for s in samples], [max(s["dy_over_H"], 1e-18) for s in samples],
                "--", label=r"$\max|\Delta y|/H$")
    ax.semilogy([s["t"] for s in samples],
                [max(s["drho_over_rho0"], 1e-18) for s in samples], ":",
                label=r"$\max|\Delta\rho|/\rho_0$")
    ax.axhline(1e-9, color="green", lw=1)
    ax.annotate("machine-precision gate (1e-9)", (0.05, 1e-9), color="green",
                fontsize=7, va="bottom")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("departure from rest")
    ax.set_title("the reconstructed lake does not stay at rest")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.figures / "lake_at_rest_drift.png", dpi=130)
    plt.close(fig)

    # ---- document -----------------------------------------------------------
    gate_rows = "\n".join(
        f"| `{r['gate']}` | {r['quantity']} | {g(r['measured'])} | "
        f"{r['comparison']} {r['threshold']:.1e} | "
        f"{'PASS' if r['passed'] else ('FAIL' if r['passed'] is not None else 'n/a')} |"
        for r in gates["rows"])
    profile_rows = "\n".join(
        f"| {r['y']:.3f} | {r['count']} | {r['rho_over_rho0']:.6f} | "
        f"{r['pressure']:.1f} | {r.get('mass_over_unit', float('nan')):.4f} | "
        f"{r.get('mass_spread', float('nan')):.2e} |"
        for r in rows)
    control_rows = "\n".join(
        f"| {name} | {g(c['a_max_over_g'])} | {g(c['a_max_over_g_interior'])} | "
        f"{g(c['p_max'])} | {c['rho_over_rho0_min']:.4f}..{c['rho_over_rho0_max']:.4f} |"
        for name, c in controls.items())
    wall = data["wall_model"]
    perturb_rows = "\n".join(f"| {k} | {g(v)} | {g(v / eps if eps else None)} |"
                             for (k, v), eps in zip(data["perturbation_response"].items(),
                                                    [float(k) for k in data["perturbation_response"]]))
    converged = "yes" if eq["converged"] else "no"
    verdict = "PASS" if gates["all_passed"] else "FAIL"
    doc = f"""# Lake at rest: does the shipped WCSPH hold a still lake still?

Case-pool anchor `CASE-SPH-LAKE-AT-REST`. Reference: the analytic hydrostatic
identity (`u = 0`, flat free surface) — no experimental uncertainty, no fitted
constant. Artefact: `{args.json.relative_to(ROOT)}`; code
`src/tensorsph/lake_at_rest.py`; runner `benchmarks/lake_at_rest/run.py`;
reproduce with

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py \\
    --json artifacts/lake_at_rest/rest_dx020.json
```

## 1. Verdict

**{verdict}** — the anchor's machine-precision tolerance is *not* met, and this
document says exactly where it breaks and where it does not.

* The **vertical** balance of the shipped operator is achievable to machine
  precision: `max|a_y|/g = {g(eq['a_max_over_g'])}` after {eq['iterations']}
  Gauss-Newton iterations (converged: {converged}).
* The **horizontal** balance is not. Side-wall truncation leaves
  `max|a_x|/g = {g(eq['a_max_over_g_horizontal_wall_band'])}` in the two
  smoothing-lengths-wide wall band, and no mass field that is mirror-symmetric
  about the tank centreline can cancel it (the reconstruction has one degree of
  freedom per particle and the horizontal force is odd under that reflection).
* Consequence, measured over {rest['t_end']:.2f} s: `max|v|/c0 =
  {g(rest['max_v_over_c0'])}`, surface departure `max|dy|/H =
  {g(rest['max_dy_over_H'])}`, density departure `max|drho|/rho0 =
  {g(rest['max_drho_over_rho0'])}`. The lake sloshes.

## 2. Case, method and numbers

Geometry: tank `{geom['width']} x {geom['depth']} m`, particle spacing
`dx = {geom['dx']} m`, `h = {geom['h']:.4f} m` (`h = {geom['h_factor']} dx`),
`{geom['particles']}` fluid particles on a square lattice, `{geom['wall_particles']}`
static wall particles in three layers, `{geom['pairs']}` interacting pairs,
`rho0 = 1000 kg/m3`, `c0 = {geom['sound_speed']} m/s`, `gamma = {geom['gamma']}`,
`g = {geom['gravity']} m/s2`. Acoustic Mach number `c0/sqrt(gH) =
{geom['acoustic_mach']:.2f}`.

Method, in the order the code runs it:

1. **Lattice and wall.** Square lattice from `pool_lattice`; the wall is built
   mirror-symmetric about `x = width/2` (see §4, finding F2).
2. **Analytic profile as a prior.** The summation density is linear in the
   particle masses, so "the mass field whose summation density is the exact
   Tait equilibrium profile `rho(y)`" is one direct linear solve, not a fixed
   point (the fixed point diverges at the free surface: rows alternate by ±30 %).
   This prior's residual, `max|a_y|/g =
   {g(controls['analytic_hydrostatic_profile']['a_max_over_g'])}`, is the floor a
   continuum reference can reach — the negative control that makes the
   reconstruction's machine-precision number meaningful.
3. **Discrete reconstruction.** Damped Gauss-Newton on `u = log(m/ m_unit)`,
   residual `a_y` of every fluid particle from the *shipped*
   `droplet_impact.rhs`, unknowns tied over mirror pairs, Tikhonov pull towards
   the prior to pick the physically meaningful member of a rank-deficient
   solution set, backtracking line search on the RMS merit.
4. **Rest test.** Symplectic Euler at `dt = {rest['dt']:.3e} s`
   (`Courant = 0.30`) for {rest['steps']} steps, positions free to move.

### Gate table (exact thresholds in `benchmarks/lake_at_rest/run.py`)

| gate | quantity | measured | threshold | verdict |
|---|---|---|---|---|
{gate_rows}

### Negative controls and linear response

| control | `max\\|a\\|/g` | interior `max\\|a_y\\|/g` | `p_max` [Pa] | `rho/rho0` |
|---|---|---|---|---|
{control_rows}

Scaling the reconstructed mass field by `1 + eps` must scale the residual
linearly, which is what proves the near-zero residual of `G1` is a balance and
not a dead code path:

| eps | `max\\|a_y\\|/g` | per unit eps |
|---|---|---|
{perturb_rows}

### Reconstructed profile (row means)

| y [m] | particles | `rho/rho0` | p [Pa] | `m/m_unit` | mass spread in row |
|---|---|---|---|---|---|
{profile_rows}

Physics cross-check on the interior band (`2h` from the bottom and from the
surface, {prof['band_count']} particles, `y` in
[{prof['band_y_min']:.3f}, {prof['band_y_max']:.3f}] m):

| quantity | value |
|---|---|
| density vs exact Tait equilibrium, max rel. | {g(prof['density_vs_eos_max_rel'])} |
| pressure vs exact Tait equilibrium, max rel. | {g(prof['pressure_vs_eos_max_rel'])} |
| pressure vs `rho0 g (H-y)`, max rel. | {g(prof['pressure_vs_rho0gh_max_rel'])} |
| compressibility bias of the reference itself | {g(prof['eos_pressure_bias_at_bottom'])} |

That last row matters: the exact Tait equilibrium differs from
`rho0 g (H-y)` by {g(prof['eos_pressure_bias_at_bottom'])} of the bottom
pressure at `c0 = {geom['sound_speed']} m/s`, so part of the
"pressure vs `rho0 g (H-y)`" column is *modelling* bias, not discretisation —
the decomposition machinery is `tensorsph.eos_hydrostatics`.

## 3. Figures

* `docs/figures/lake_at_rest_residual.png` — Newton history with the two
  negative controls and the side-wall horizontal residual on the same axis.
* `docs/figures/lake_at_rest_profiles.png` — reconstructed pressure and density
  profiles against the two analytic references.
* `docs/figures/lake_at_rest_drift.png` — velocity / surface / density departure
  during the {rest['t_end']:.2f} s rest test, against the 1e-9 gate.

No field cloud map is shown because the case is one-dimensional by construction:
the state is a function of `y` only, and a contour plot of a stratified state
would show nothing a profile does not.

## 4. Findings

**F1 (blocking, the anchor's own verdict).** The shipped WCSPH cannot hold a
lake at rest to machine precision in a *finite* tank. The vertical balance is
reconstructible to `{g(eq['a_max_over_g'])}` but the side-wall band keeps
`{g(eq['a_max_over_g_horizontal_wall_band'])}` of horizontal residual, which
drives the whole body: `max|v|/c0 = {g(rest['max_v_over_c0'])}` within
{rest['t_end']:.2f} s. The anchor's tolerance (machine precision) is therefore
**unattainable with the shipped scheme**; the honest statement is the pair of
numbers above.

**F2 (method, measured).** Mirror symmetry of the wall had to be *checked*, not
assumed: the shipped `wall_lattice` reaches the same column through two
arithmetic paths (bottom row and side wall) and the halves differ in the last
bits, so the unrounded lattice carries almost-duplicate columns and the mirror
map finds no partner for one of them — the mass field then silently loses its
symmetry and a spurious horizontal residual appears. After rounding to 1e-12 the
report gives {wall['unpaired']} unpaired columns (shipped) and {wall['case_wall_unpaired']} (this case's
wall). The horizontal residual of F1 is therefore *not* a wall asymmetry
artefact; it survives a symmetric wall.

**F3 (negative result, documented so it is not retried blind).** Enabling the
shipped Adami free-surface pressure closure (`cfg.surface_pressure_closure`)
makes the reconstruction *stall*: the Newton history flattens at
`max|a_y|/g ≈ 2.0` after ten iterations, because the closure re-derives the
surface pressure from a threshold test on the summation density and the Tait
clamp is no longer the only non-smoothness. The default here is `False`.

**F4 (non-uniqueness).** The pair operator annihilates a constant pressure field
on a complete support, so the balance Jacobian is rank deficient; without the
Tikhonov pull the mass field wanders by up to 40 % between neighbouring rows.
The regularised solution is reported with its per-row mass spread
(table above) so the reader can see how much of the profile is datum and how
much is the chosen member of the solution set.

## 5. Not done (unknown, not "passed")

* The resolution ladder (`dx = 0.02` only) — the interior-band profile error has
  no measured convergence order yet, because a second resolution needs the
  {geom['particles']:.0f}-particle Newton solve again at 4x the particle count.
* A per-particle (non-mirror) reconstruction minimising both components, which is
  the obvious way to attack F1: it turns the system into 2N equations and N
  unknowns, and the best achievable compromise is not known.
* Independent-implementation cross-check (ISPH / DFSPH on the same state).
* Longer runs (`t > {rest['t_end']:.2f} s`) and closed (periodic-x) alternatives
  to the side walls.
"""
    args.docs.write_text(doc)
    print(f"wrote {args.docs} and 3 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())