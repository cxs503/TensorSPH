"""Does the recorded mode-2 frequency measure capillary physics, or a blow-up?

Context
-------
``benchmarks/oscillating_droplet`` releases an elliptical 2-D drop with a seeded
m = 2 deformation and fits a damped cosine to the second moment of the particle
distribution.  The fitted frequency is compared with the inviscid Rayleigh
frequency ``omega^2 = sigma m (m^2 - 1) / (rho R^3)``.  The recorded verdict in
``docs/verification_oscillating_droplet.md`` is FAIL (-43.5 % at R = 2 cm), and
``tests/test_oscillating_droplet.py`` pins a cheap release case (R = 1 cm) at
-31.8 % on loong64 but +8.6 % on x86_64 CI for identical code; that two-fold
disagreement is booked there as "the mode-2 fit is architecture-fragile" and
blocks the re-baseline.

This script tests a different explanation, using *saved* traces only (no new SPH
computation, so it is cheap and load-safe):

    the simulated cases are numerically unstable -- kinetic energy grows by
    5-12 decades inside the analysed window -- so the fitted frequency is a
    property of the fit window and of the round-off-seeded growth rate, not of
    the capillary wave.

Inputs
------
``benchmarks/oscillating_droplet/trace_data.json``
    (t, second moment, kinetic energy) of the documented reference case
    (R = 2 cm, dx = 2 mm, 1.8 periods).
``benchmarks/oscillating_droplet_instability/release_case_trace.json``
    same trace for the cheap CI pin case (R = 1 cm, dx = 1 mm), produced by
    ``probe_release_case.py``.

Outputs
-------
``report.json`` (numbers injected into the doc by ``make_doc.py``) +
``oscillating_droplet_instability.png``.

Usage:  python benchmarks/oscillating_droplet_instability/run.py
        python benchmarks/oscillating_droplet_instability/probe_release_case.py   # 1st
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT / "benchmarks" / "oscillating_droplet"

#: energy growth relative to the pre-growth baseline that counts as "diverged"
DIVERGENCE_FACTOR = 10.0
#: fit windows, in reference periods
WINDOW_PERIODS = (0.5, 0.75, 1.0, 1.8)
#: relative perturbation levels for the conditioning study
PERTURBATIONS = (1e-6, 1e-4, 1e-2)
PERTURBATION_SEEDS = (0, 1)


def _bench():
    spec = importlib.util.spec_from_file_location("oscillating_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Case:
    """One recorded oscillation case: setup, reference frequency, raw trace."""

    def __init__(self, name: str, setup: dict, omega_ref: float,
                 t, s, ke, source: str):
        self.name = name
        self.setup = setup
        self.omega_ref = float(omega_ref)
        self.period = 2.0 * math.pi / self.omega_ref
        self.t = np.asarray(t, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.ke = np.abs(np.asarray(ke, dtype=float))
        self.source = source

    def rel(self, path: Path) -> str:
        return str(path.relative_to(ROOT))


def load_cases() -> list[Case]:
    ref = json.loads((BENCH / "validation_report.json").read_text())["frequency"]
    tr = json.loads((BENCH / "trace_data.json").read_text())["frequency"]
    cases = [Case("reference_R2cm", ref["setup"], ref["reference"]["omega_rad_per_s"],
                  tr["t"], tr["s_second_moment"], tr["kinetic_energy_J"],
                  "benchmarks/oscillating_droplet/trace_data.json")]
    probe = HERE / "release_case_trace.json"
    if probe.exists():
        p = json.loads(probe.read_text())
        cases.append(Case("release_R1cm", p["setup"], p["reference"]["omega_rad_per_s"],
                          p["trace"]["t"], p["trace"]["s_second_moment"],
                          p["trace"]["kinetic_energy_J"],
                          "benchmarks/oscillating_droplet_instability/"
                          "release_case_trace.json"))
    return cases


def ellipse_perimeter(a: float, b: float, n: int = 20001) -> float:
    """Perimeter of the half-axes (a, b) ellipse, by numeric quadrature."""
    phi = np.linspace(0.0, 2.0 * math.pi, n)
    dphi = phi[1] - phi[0]
    integrand = np.sqrt(a * a * np.sin(phi) ** 2 + b * b * np.cos(phi) ** 2)
    return float(np.sum(integrand) * dphi - 0.5 * dphi * (integrand[0] + integrand[-1]))


def available_energy(case: Case) -> dict:
    """Surface energy released when the ellipse relaxes to a circle.

    ``E = sigma * (P_ellipse - P_circle)`` is a *hard* upper bound on the kinetic
    energy any energy-conserving solver can produce from this initial state, so
    ``KE_max / E > 1`` is proof that the scheme created energy from nothing --
    independent of how the frequency is estimated.
    """
    radius = float(case.setup["radius_m"])
    eps = float(case.setup["initial_eps"])
    sigma = float(case.setup["sigma_N_per_m"])
    a, b = radius * (1.0 + eps), radius / (1.0 + eps)
    p_ellipse = ellipse_perimeter(a, b)
    p_circle = 2.0 * math.pi * radius
    mass = float(case.setup["rho0"]) * math.pi * radius ** 2
    return {
        "semi_major_m": a,
        "semi_minor_m": b,
        "perimeter_ellipse_m": p_ellipse,
        "perimeter_circle_m": p_circle,
        "surface_energy_total_J": sigma * p_ellipse,
        "available_energy_J": sigma * (p_ellipse - p_circle),
        "linear_kinetic_energy_scale_J":
            0.5 * mass * (eps * case.omega_ref * radius) ** 2,
    }


def divergence_diagnostics(case: Case, energy: dict) -> dict:
    """Energy budget of the recorded window and the onset of the blow-up."""
    t, ke = case.t, case.ke
    onset_idx = int(np.argmax(ke > DIVERGENCE_FACTOR * energy["linear_kinetic_energy_scale_J"]))
    onset = float(t[onset_idx])
    budget = energy["available_energy_J"]
    return {
        "case": case.name,
        "period_s": case.period,
        "simulated_time_s": float(t[-1]),
        "simulated_periods": float(t[-1] / case.period),
        "kinetic_energy_first_J": float(ke[0]),
        "kinetic_energy_max_J": float(ke.max()),
        "kinetic_energy_over_available": float(ke.max() / budget),
        "kinetic_energy_over_linear_scale": float(
            ke.max() / energy["linear_kinetic_energy_scale_J"]),
        "energy_creation_decades": float(math.log10(ke.max() / budget)),
        "divergence_onset_s": onset,
        "divergence_onset_periods": float(onset / case.period),
        "divergence_onset_fraction_of_window": float(onset / t[-1]),
        "second_moment_min_m2": float(case.s.min()),
        "second_moment_max_m2": float(case.s.max()),
        "diverged": bool(ke.max() > budget),
    }


def window_study(case: Case, bench, energy: dict) -> list[dict]:
    """Re-fit the same trace on prefixes of the recorded window."""
    budget = energy["available_energy_J"]
    rows = []
    for frac in WINDOW_PERIODS:
        t_end = frac * case.period
        mask = case.t <= t_end
        ts, ss = list(case.t[mask]), list(case.s[mask])
        if len(ts) < 20:
            continue
        fit = bench.damped_cosine_fit(ts, ss, case.omega_ref)
        zc = bench.zero_crossing_period(ts, ss)
        rows.append({
            "window_periods": frac,
            "window_s": t_end,
            "n_samples": len(ts),
            "fit_omega_rad_per_s": float(fit["omega"]),
            "fit_rel_error_pct": float((fit["omega"] - case.omega_ref)
                                       / case.omega_ref * 100.0),
            "fit_r2": float(fit.get("r2", float("nan"))),
            "zero_crossing_rel_error_pct": float(
                (zc["omega"] - case.omega_ref) / case.omega_ref * 100.0),
            "fit_lambda_per_s": float(fit["lambda_per_s"]),
            "kinetic_energy_at_window_end_J": float(case.ke[mask][-1]),
            "kinetic_energy_over_available_at_window_end": float(
                case.ke[mask][-1] / budget),
            "energy_growth_in_window": float(case.ke[mask][-1]
                                             / case.ke[mask][0]),
        })
    return rows


def conditioning_study(case: Case, bench, window_periods: float = 1.0) -> dict:
    """Spread of the fitted omega under relative perturbations of the trace.

    A different floating-point reduction order (thread count, SIMD width,
    ``index_add`` order) changes the *trajectory*, not the recording; adding a
    relative perturbation of size ``eps`` to the second-moment series is the
    cheapest conservative proxy for the resulting change of the fitted
    quantity.  If the fit is already ill-conditioned at ``eps`` = 1e-6 it cannot
    survive an architecture change; if it is well conditioned there, the blame
    for the arch spread lies with the solution (divergence), not the fit.
    """
    mask = case.t <= window_periods * case.period
    ts, ss = case.t[mask], case.s[mask]
    base = bench.damped_cosine_fit(list(ts), list(ss), case.omega_ref)
    base_err = (base["omega"] - case.omega_ref) / case.omega_ref * 100.0
    rows = []
    for eps in PERTURBATIONS:
        errs = []
        for seed in PERTURBATION_SEEDS:
            rng = np.random.default_rng(seed)
            noisy = ss * (1.0 + eps * rng.standard_normal(ss.shape))
            fit = bench.damped_cosine_fit(list(ts), list(noisy), case.omega_ref)
            errs.append((fit["omega"] - case.omega_ref) / case.omega_ref * 100.0)
        rows.append({
            "relative_perturbation": eps,
            "errors_pct": [float(e) for e in errs],
            "spread_pct": float(max(errs) - min(errs)),
            "max_abs_shift_pct": float(max(abs(e - base_err) for e in errs)),
        })
    return {
        "case": case.name,
        "window_periods": window_periods,
        "unperturbed_rel_error_pct": float(base_err),
        "unperturbed_r2": float(base.get("r2", float("nan"))),
        "rows": rows,
        "worst_spread_pct": float(max(r["spread_pct"] for r in rows)),
    }


def plot(cases: list[Case], windows: dict[str, list[dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.3))
    colours = {"reference_R2cm": "C3", "release_R1cm": "C0"}
    for case in cases:
        c = colours.get(case.name, "C1")
        ax[0].semilogy(case.t / case.period, case.ke, c, lw=1.2, label=case.name)
        ax[1].plot(np.asarray([w["window_periods"] for w in windows[case.name]]),
                   np.asarray([w["fit_rel_error_pct"] for w in windows[case.name]]),
                   "o-", color=c, label=case.name)
    ax[0].set_xlabel("t / reference period")
    ax[0].set_ylabel("kinetic energy [J]")
    ax[0].set_title("both recorded cases gain 5-12 decades of KE")
    ax[0].legend(fontsize=8)
    ax[1].axhspan(-15.0, 15.0, color="0.85", zorder=0, label="15 % gate")
    ax[1].set_xlabel("fit window [reference periods]")
    ax[1].set_ylabel("fitted frequency error [%]")
    ax[1].set_title("the 'measured' frequency is a function of the window")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    bench = _bench()
    cases = load_cases()
    energies = {c.name: available_energy(c) for c in cases}
    divergences = [divergence_diagnostics(c, energies[c.name]) for c in cases]
    windows = {c.name: window_study(c, bench, energies[c.name]) for c in cases}
    conditioning = conditioning_study(cases[0], bench)
    report = {
        "benchmark": "oscillating_droplet_instability",
        "question": ("is the recorded mode-2 frequency a measurement of the "
                     "capillary wave, or a property of the fit window over a "
                     "diverging solution?"),
        "trace_sources": {c.name: c.source for c in cases},
        "setup": {c.name: c.setup for c in cases},
        "omega_reference_rad_per_s": {c.name: c.omega_ref for c in cases},
        "energy_budget": energies,
        "divergence": divergences,
        "window_study": windows,
        "conditioning": conditioning,
        "figures": ["oscillating_droplet_instability.png"],
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=1))
    plot(cases, windows, HERE / "oscillating_droplet_instability.png")
    print(json.dumps({"divergence": divergences, "windows": windows,
                      "conditioning": conditioning}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())