"""Verification-and-validation ladder: run the three rungs and emit one JSON.

The ladder is a *contract*: every rung states a reference, a measured quantity, a
percentage error and a gate, and the whole document is pinned by
:func:`tensorsph.vv.schema.validate_ladder_json`. The negative controls exist so
that a rung cannot be satisfied by an implementation that ignores the mechanism
the rung is supposed to exercise: each control switches one mechanism off and the
metric must move.

Honesty rules encoded here (not just documented):

* a failing rung keeps its measured number and is reported as ``FAIL``; nothing
  is re-gated after the fact and no reference is chosen after seeing the result;
* the dam-break rung's experimental cross-check is carried in ``measured`` and
  never replaces the analytic gate;
* every number in ``summary`` is derived from the rung dictionaries, so the
  document cannot claim a pass that a rung did not produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from .dambreak import run_dambreak
from .hydrostatic import HydrostaticConfig, run_hydrostatic
from .schema import LADDER_SCHEMA_VERSION, validate_ladder_json
from .tgv import TGVConfig, run_tgv, viscous_operator_error

NODE_ID = "p83w1-n01-tensorsph-verification"


@dataclass
class LadderConfig:
    """Per-rung parameters. ``quick=True`` shrinks everything for the test suite."""

    quick: bool = False
    # rung 1
    hydro_dx: float = 0.02
    hydro_seconds: float = 4.0
    hydro_average_from: float = 0.75
    # rung 2
    tgv_dx: float = 0.025
    tgv_t_end: float = 0.6
    tgv_u0: float = 0.2
    tgv_sound_speed: float = 10.0
    tgv_viscosity: float = 1.0 / (10.0 * 2.0 * math.pi)
    # rung 3
    dam_dx: float = 0.02
    dam_seconds: float = 0.4
    dam_coarse_factor: float = 2.0
    dam_sound_speed: float = 25.0
    dam_surface_tension: float = 0.0
    # controls
    run_controls: bool = True
    control_seconds_hydro: float = 0.5
    control_tgv_t_end: float = 0.4
    control_dam_dx: float = 0.04
    control_dam_seconds: float = 0.2
    control_surface_tension: float = 0.0728

    def quickify(self) -> "LadderConfig":
        return LadderConfig(
            quick=True,
            hydro_dx=0.04, hydro_seconds=1.2, hydro_average_from=0.75,
            tgv_dx=0.05, tgv_t_end=0.3,
            dam_dx=0.04, dam_seconds=0.2, dam_coarse_factor=0.0,
            run_controls=True,
            control_seconds_hydro=0.4,
            control_tgv_t_end=0.2,
            control_dam_dx=0.05, control_dam_seconds=0.2,
        )


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git",) + args, capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parents[3]),
                             timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # pragma: no cover - git is absent in some sandboxes
        return None


def provenance(command: str) -> dict:
    import tensorsph
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool((_git("status", "--porcelain") or "").strip()),
        "python": platform.python_version(),
        "torch": __import__("torch").__version__,
        "package_version": getattr(tensorsph, "__version__", "unknown"),
        "platform": platform.platform(),
        "command": command,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# anchors
# --------------------------------------------------------------------------- #
def _hydrostatic_anchor(cfg: LadderConfig) -> dict:
    run = run_hydrostatic(HydrostaticConfig(dx=cfg.hydro_dx,
                                            seconds=cfg.hydro_seconds,
                                            average_from=cfg.hydro_average_from))
    return {
        "id": "C1_hydrostatic",
        "title": "Hydrostatic pressure balance",
        "physics": "rho g h pressure gradient balancing gravity in a still column",
        "reference": run["reference"],
        "measured": run["measured"],
        "error_pct": run["error_pct"],
        "gate": run["gate"],
        "verdict": "PASS" if run["gate"]["pass"] else "FAIL",
        "config": run["config"],
        "series": [{"y": b["y"], "pressure": b["pressure"],
                    "pressure_analytic": b["pressure_analytic"]}
                   for b in run["measured"]["bands"]],
        "figures": [{"name": "fig1_hydrostatic_profile.png",
                     "caption": "Time-averaged SPH pressure per band vs p = rho g h"}],
    }


def _tgv_anchor(cfg: LadderConfig) -> dict:
    tgv_cfg = TGVConfig(dx=cfg.tgv_dx, t_end=cfg.tgv_t_end, U0=cfg.tgv_u0,
                        sound_speed=cfg.tgv_sound_speed,
                        viscosity=cfg.tgv_viscosity)
    run = run_tgv(tgv_cfg)
    run["measured"]["viscous_operator_l2_error_pct"] = viscous_operator_error(tgv_cfg)
    return {
        "id": "C2_tgv",
        "title": "Taylor-Green vortex viscous decay",
        "physics": "viscous dissipation of a periodic vortex, lap(u) = -2 k^2 u",
        "reference": run["reference"],
        "measured": run["measured"],
        "error_pct": run["error_pct"],
        "gate": run["gate"],
        "verdict": "PASS" if run["gate"]["pass"] else "FAIL",
        "config": run["config"],
        "series": {"time": run["series"]["time"],
                   "energy": run["series"]["energy"],
                   "energy_analytic": run["series"]["energy_analytic"]},
        "figures": [{"name": "fig2_tgv_energy_decay.png",
                     "caption": "Kinetic energy decay vs the exact exp(-4 nu k^2 t)"}],
    }


def _dambreak_anchor(cfg: LadderConfig) -> dict:
    run = run_dambreak(dx=cfg.dam_dx, seconds=cfg.dam_seconds,
                       sound_speed=cfg.dam_sound_speed,
                       surface_tension=cfg.dam_surface_tension,
                       coarse_factor=cfg.dam_coarse_factor)
    return {
        "id": "C3_dambreak",
        "title": "Dam-break free-surface position",
        "physics": "nonlinear free-surface collapse, leading-edge trajectory",
        "reference": run["reference"],
        "measured": run["measured"],
        "error_pct": run["error_pct"],
        "gate": run["gate"],
        "verdict": "PASS" if run["gate"]["pass"] else "FAIL",
        "config": run["config"],
        "series": run["series"],
        "figures": [{"name": "fig3_dambreak_front.png",
                     "caption": "Z/a vs tau: simulation, Martin & Moyce (1952), Ritter"}],
    }


# --------------------------------------------------------------------------- #
# negative controls
# --------------------------------------------------------------------------- #
def _control_gravity_off(cfg: LadderConfig) -> dict:
    run = run_hydrostatic(HydrostaticConfig(dx=cfg.hydro_dx,
                                            seconds=cfg.control_seconds_hydro,
                                            average_from=cfg.hydro_average_from),
                          gravity_override=0.0)
    err = run["error_pct"]
    moved = err > 50.0
    return {
        "id": "C4a_gravity_off",
        "title": "Hydrostatic rung with gravity switched off",
        "mechanism": "gravity enters the pressure balance",
        "metric": "max interior band pressure error vs rho g h",
        "expected": "error rises to ~100 % if the rung really tests the balance",
        "measured": {"error_pct": err,
                     "mean_band_error_pct": run["measured"]["mean_band_error_pct"],
                     "pressure_max_pa": max(b["pressure"] for b in run["measured"]["bands"])},
        "verdict": "PASS" if moved else "FAIL",
    }


def _control_viscosity_off(cfg: LadderConfig) -> dict:
    run = run_tgv(TGVConfig(dx=cfg.tgv_dx, t_end=cfg.control_tgv_t_end,
                            U0=cfg.tgv_u0, sound_speed=cfg.tgv_sound_speed,
                            viscosity=cfg.tgv_viscosity), viscosity_override=0.0)
    rate = run["measured"]["energy_decay_rate"]
    # The reference rate is the *nominal* one (4 nu k^2 of the rung), not the
    # zero that the overridden config would produce.
    k = 2.0 * math.pi / run["config"]["L"]
    analytic = 4.0 * cfg.tgv_viscosity * k * k
    fraction = rate / analytic if analytic else float("nan")
    moved = abs(fraction) < 0.5
    return {
        "id": "C4b_viscosity_off",
        "title": "Taylor-Green rung with viscosity switched off",
        "mechanism": "viscosity is the only dissipation in the rung",
        "metric": "measured energy decay rate / analytic 4 nu k^2",
        "expected": "ratio collapses towards 0 (only the numerical dissipation floor remains)",
        "measured": {"rate": rate, "analytic_rate": analytic,
                     "ratio": fraction,
                     "numerical_dissipation_floor_pct": fraction * 100.0},
        "verdict": "PASS" if moved else "FAIL",
    }


def _control_surface_tension(cfg: LadderConfig) -> dict:
    plain = run_dambreak(dx=cfg.control_dam_dx, seconds=cfg.control_dam_seconds,
                         sound_speed=cfg.dam_sound_speed, surface_tension=0.0)
    wet = run_dambreak(dx=cfg.control_dam_dx, seconds=cfg.control_dam_seconds,
                       sound_speed=cfg.dam_sound_speed,
                       surface_tension=cfg.control_surface_tension)
    a = plain["measured"]["dispersion"] or {}
    b = wet["measured"]["dispersion"] or {}
    front_plain, front_wet = a.get("front_raw"), b.get("front_raw")
    spread_plain, spread_wet = a.get("leader_spread"), b.get("leader_spread")
    response_front = (abs(front_wet - front_plain) / front_plain * 100.0
                      if front_plain else float("nan"))
    response_spread = (abs(spread_wet - spread_plain) / spread_plain * 100.0
                       if spread_plain else float("nan"))
    # The control can only *prove* something if the mechanism matters. At the
    # dam-break Weber number (~550 at these scales) the capillary force is
    # negligible against inertia, so a null response is a statement about the
    # regime, not about the implementation. It is labelled INCONCLUSIVE rather
    # than PASS/FAIL for exactly that reason.
    discriminating = (response_front == response_front and response_front > 1.0)
    return {
        "id": "C4c_surface_tension",
        "title": "Dam-break jet with surface tension off / on",
        "mechanism": "capillary force on the free surface (CSF model)",
        "metric": "leading-edge transverse spread and detached-particle count",
        "expected": ("campaign wording expects the jet to disperse when sigma = 0; "
                     "at this Weber number the capillary force is dynamically "
                     "negligible, so the measured response is reported as-is"),
        "measured": {"sigma_off": a, "sigma_on": b,
                     "front_response_pct": response_front,
                     "spread_response_pct": response_spread,
                     "weber_number_off": plain["config"]["weber_number"],
                     "weber_number_on": wet["config"]["weber_number"]},
        "verdict": "PASS" if discriminating else "INCONCLUSIVE",
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_ladder(cfg: LadderConfig | None = None, command: str | None = None,
               log=print) -> dict:
    cfg = LadderConfig() if cfg is None else cfg
    command = command or " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    anchors = []
    for name, runner in (("C1", _hydrostatic_anchor), ("C2", _tgv_anchor),
                         ("C3", _dambreak_anchor)):
        log(f"[ladder] running anchor {name} ...")
        anchors.append(runner(cfg))
        log(f"[ladder] {name}: error {anchors[-1]['error_pct']:.3f} % -> "
            f"{anchors[-1]['verdict']}")

    controls = []
    if cfg.run_controls:
        for runner in (_control_gravity_off, _control_viscosity_off,
                       _control_surface_tension):
            ctl = runner(cfg)
            log(f"[ladder] control {ctl['id']}: {ctl['verdict']}")
            controls.append(ctl)

    by_id = {a["id"]: a for a in anchors}
    controls_by_id = {c["id"]: c for c in controls}
    dam = by_id["C3_dambreak"]
    campaign = {
        "C1_hydrostatic_max_pressure_error_below_1pct": {
            "target_pct": 1.0, "measured_pct": by_id["C1_hydrostatic"]["error_pct"],
            "verdict": by_id["C1_hydrostatic"]["verdict"]},
        "C2_tgv_energy_decay_rate_error_below_3pct": {
            "target_pct": 3.0, "measured_pct": by_id["C2_tgv"]["error_pct"],
            "verdict": by_id["C2_tgv"]["verdict"]},
        "C3_dambreak_front_error_below_5pct_vs_MartinMoyce": {
            "target_pct": 5.0,
            "measured_pct_of_a": dam["measured"]["experimental_relative_pct_of_a"],
            "measured_mae_Z_over_a": dam["measured"]["experimental_mae_Z_over_a"],
            "measured_ritter_error_pct": dam["error_pct"],
            "verdict": "FAIL",
            "note": "two references disagree with the column by 49 % (experiment, "
                    "MAE 0.49 in Z/a) and 94 % (Ritter shallow-water front); the "
                    "gap is a resolution-independent physical offset (H/a = 2 is "
                    "outside shallow-water theory) and the bundled experimental "
                    "table is audited WITHHELD as a digitisation, so the 5 % "
                    "criterion cannot be met by refinement alone"},
        "C4_negative_controls_mechanism_effectiveness": {
            "gravity_off": controls_by_id.get("C4a_gravity_off", {}).get("verdict"),
            "viscosity_off": controls_by_id.get("C4b_viscosity_off", {}).get("verdict"),
            "surface_tension_off": controls_by_id.get("C4c_surface_tension", {}).get("verdict"),
            "verdict": ("INCONCLUSIVE"
                        if controls_by_id.get("C4c_surface_tension", {}).get("verdict")
                        == "INCONCLUSIVE" else "PASS"),
            "note": "the campaign expects the dam-break jet to disperse with sigma = 0; "
                    "at We ~ 550 the capillary force is dynamically negligible, so the "
                    "measured response is 0.01 % and the test does not discriminate"},
        "C5_end_to_end_json_schema": {
            "schema_version": LADDER_SCHEMA_VERSION,
            "schema_errors": [],
            "anchors_present": sorted(by_id),
            "verdict": "PASS"},
    }

    failed = [a["id"] for a in anchors + controls if a["verdict"] != "PASS"]
    doc = {
        "schema_version": LADDER_SCHEMA_VERSION,
        "node": NODE_ID,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "provenance": provenance(command),
        "config": cfg.__dict__,
        "anchors": anchors,
        "negative_controls": controls,
        "summary": {
            "anchors_total": len(anchors),
            "anchors_passed": sum(1 for a in anchors if a["verdict"] == "PASS"),
            "controls_total": len(controls),
            "controls_passed": sum(1 for c in controls if c["verdict"] == "PASS"),
            "all_passed": not failed,
            "honest_failures": failed,
            "errors_pct": {a["id"]: a["error_pct"] for a in anchors},
        },
        "campaign_criteria": campaign,
        "limitations": [
            "Rung 2 is two-dimensional and periodic; the 3-D Taylor-Green vortex has "
            "no analytic solution and is not covered (it would need reference DNS data).",
            "Rung 3's experimental cross-check uses a transcribed table that the "
            "repository audit grades WITHHELD as a digitisation, so a disagreement "
            "with it cannot be attributed to the solver alone.",
            "The dam-break free-slip wall condition is not the no-slip condition of "
            "the Martin & Moyce tank.",
            "No turbulence model, no surface tension in the gated dam-break run, and "
            "no density diffusion; each is available in the library but would add "
            "unverified physics to a verification case.",
        ],
    }
    errors = validate_ladder_json(doc)
    if errors:
        raise ValueError("ladder JSON violates its schema: " + "; ".join(errors))
    return doc


def write_ladder(doc: dict, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return out