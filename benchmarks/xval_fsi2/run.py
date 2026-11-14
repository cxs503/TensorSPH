"""Cross-validation of the fluid->solid load transfer against the OSS convention.

Predeclared design (written before the numbers were produced):

Decision
    Should TensorSPH adopt the SPHinXsys ``ForceFromFluid`` convention as its
    production FSI load transfer, keep its own panel (WLS) operator, or neither?
Alternatives
    A ``oss``            -- one-sided summation with Adami wall-pressure
                            extrapolation, fsi2 configuration (NoRiemann,
                            NoKernelCorrection).
    B ``oss_riemann``    -- same with the acoustic Riemann dissipation switched on.
    C ``panel``          -- TensorSPH's kernel-weighted plane fit (m16352).
    D ``pairwise_legacy``-- the pre-m16352 symmetric Adami term.
    E analytic reference -- the ground truth of each case.
Metric       relative error of the total load component along the analytic
             direction, ``(F_got - F_ref)/|F_ref|``, plus the orthogonal leakage.
Thresholds   |err| < 15 % required (owner accuracy standard), < 3 % is the
             target; support may be declared on the whole dx ladder or the case
             is reported as not transferred.
Stopping     no threshold is retuned after the run; every operator is reported
             for every resolution whether it passes or not.
Environment  torch CPU float64, deterministic lattices, no solver in the loop.
             One case-family (end-to-end) additionally uses a solved WCSPH field.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.fsi_force_from_fluid import (ForceFromFluidConfig,  # noqa: E402
                                            OSS_PROVENANCE,
                                            panel_pressure_force,
                                            pairwise_pressure_force,
                                            pressure_force_from_fluid,
                                            pressure_force_from_fluid_normalised,
                                            one_sided_viscous_truncation_factor,
                                            viscous_force_from_fluid,
                                            wall_shear_from_gradient)
import cases as C  # noqa: E402

ART = ROOT / "artifacts" / "xval_fsi2"
DOCS = ROOT / "docs" / "xval_fsi2"


def _cfg(dx: float, case: C.LoadCase, riemann: bool, correction: bool = False,
         kernel: str = "wendland") -> ForceFromFluidConfig:
    return ForceFromFluidConfig(
        dim=2, h=case.h, kernel=kernel, riemann=riemann,
        rho0_f=1.0, c0_f=1.0, kernel_correction=correction)


def _along(force: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """(got, ref, orthogonal leakage) resolved along the analytic direction."""
    direction = ref / ref.norm().clamp_min(1.0e-30)
    got_along = float(force @ direction)
    got_perp = float((force - got_along * direction).norm())
    return got_along, float(ref.norm()), got_perp


def evaluate(case: C.LoadCase) -> dict:
    row = {"case": case.name, "dx": case.dx, "fluid_state": case.fluid_state,
           "source": case.source, "description": case.description,
           "reference": case.reference_label,
           "reference_force": [float(v) for v in case.reference_force],
           "n_solid": int(case.pos_solid.shape[0]),
           "n_fluid": int(case.pos_fluid.shape[0]), "operators": {}}

    if case.name in ("couette_shear", "curved_disk_shear"):
        f = viscous_force_from_fluid(
            case.pos_solid, case.vol_solid, case.vel_solid,
            case.pos_fluid, case.vol_fluid, case.vel_fluid, case.mu_fluid,
            _cfg(case.dx, case, riemann=False))
        row["operators"]["oss_viscous"] = _score(f, case.reference_force)
        f = wall_shear_from_gradient(
            case.pos_solid, case.vol_solid, case.normal_solid,
            case.pos_fluid, case.vol_fluid, case.vel_fluid, case.mu_fluid,
            case.vel_solid, _cfg(case.dx, case, riemann=False))
        row["operators"]["gradient_corrected"] = _score(f, case.reference_force)
        # independent continuum prediction of the OSS deficit for this kernel
        row["oss_viscous_continuum_factor"] = one_sided_viscous_truncation_factor(
            case.h, 2, "wendland", regularisation=0.01, spacing=case.dx)
        return row

    for name, kwargs in (("oss", dict(riemann=False)),
                         ("oss_riemann", dict(riemann=True))):
        f = pressure_force_from_fluid(
            case.pos_solid, case.vol_solid, case.normal_solid,
            torch.zeros_like(case.pos_solid), case.vel_solid,
            case.pos_fluid, case.vol_fluid, case.pressure_fluid, case.rho_fluid,
            case.vel_fluid, case.acc_ext_fluid, _cfg(case.dx, case, **kwargs))
        row["operators"][name] = _score(f, case.reference_force)

    f = pressure_force_from_fluid_normalised(
        case.pos_solid, case.vol_solid, case.normal_solid,
        torch.zeros_like(case.pos_solid), case.vel_solid,
        case.pos_fluid, case.vol_fluid, case.pressure_fluid, case.rho_fluid,
        case.vel_fluid, case.acc_ext_fluid, _cfg(case.dx, case, riemann=False))
    row["operators"]["oss_normalised"] = _score(f, case.reference_force)

    f = panel_pressure_force(case.pos_solid, case.vol_solid, case.normal_solid,
                             case.pos_fluid, case.vol_fluid, case.pressure_fluid,
                             _cfg(case.dx, case, riemann=False))
    row["operators"]["panel"] = _score(f, case.reference_force)

    f = pairwise_pressure_force(case.pos_solid, case.vol_solid, case.pos_fluid,
                                case.vol_fluid, case.pressure_fluid, case.rho_fluid,
                                _cfg(case.dx, case, riemann=False),
                                rho0_solid=case.rho_fluid[:1].expand(
                                    case.pos_solid.shape[0]))
    row["operators"]["pairwise_legacy"] = _score(f, case.reference_force)
    return row


def _score(force: torch.Tensor, ref: torch.Tensor) -> dict:
    force = force.sum(dim=0) if force.dim() == 2 else force
    got, ref_mag, perp = _along(force, ref)
    err = (got - ref_mag) / ref_mag
    return {"force": [float(v) for v in force], "got": got,
            "ref": ref_mag, "error": float(err),
            "orthogonal": perp, "passes_15": bool(abs(err) < 0.15),
            "passes_3": bool(abs(err) < 0.03)}


# --------------------------------------------------------------------------- #
# Sensitivity knobs (predeclared: kernel and h/spacing ratio)
# --------------------------------------------------------------------------- #
def sensitivity(case_name: str, dx: float) -> list[dict]:
    out = []
    for hf, kernel in ((1.3, "wendland"), (1.3, "quintic"), (1.15, "wendland")):
        build = C.CASE_BUILDERS[case_name]
        case = build(dx=dx, h_factor=hf)
        f = pressure_force_from_fluid(
            case.pos_solid, case.vol_solid, case.normal_solid,
            torch.zeros_like(case.pos_solid), case.vel_solid,
            case.pos_fluid, case.vol_fluid, case.pressure_fluid, case.rho_fluid,
            case.vel_fluid, case.acc_ext_fluid,
            _cfg(dx, case, riemann=False, kernel=kernel))
        s = _score(f, case.reference_force)
        s.update({"h_factor": hf, "kernel": kernel, "case": case.name, "dx": dx})
        out.append(s)
    return out


def main() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    rows, timings = [], {}
    for name in ("vertical_wall", "horizontal_gate", "couette_shear",
                 "fsi2_insert_linear", "curved_disk_shear"):
        build = C.CASE_BUILDERS[name]
        for dx in C.dx_ladder(name):
            t0 = time.time()
            case = build(dx=dx)
            rows.append(evaluate(case))
            timings[f"{name}@{dx}"] = time.time() - t0
            r = rows[-1]
            print(f"{name:20s} dx={dx:<7g} " + "  ".join(
                f"{k}:{(v['error']*100):+7.2f}%" for k, v in r["operators"].items()),
                  flush=True)
    sens = sensitivity("fsi2_insert_linear", 0.05) + \
        sensitivity("horizontal_gate", 0.025)
    thickness = []
    for name in ("vertical_wall", "horizontal_gate"):
        for layers in (1, 3):
            case = C.CASE_BUILDERS[name](dx=0.05, layers=layers)
            f = pressure_force_from_fluid(
                case.pos_solid, case.vol_solid, case.normal_solid,
                torch.zeros_like(case.pos_solid), case.vel_solid,
                case.pos_fluid, case.vol_fluid, case.pressure_fluid, case.rho_fluid,
                case.vel_fluid, case.acc_ext_fluid, _cfg(case.dx, case, riemann=False))
            s_oss = _score(f, case.reference_force)
            fp = panel_pressure_force(case.pos_solid, case.vol_solid,
                                      case.normal_solid, case.pos_fluid,
                                      case.vol_fluid, case.pressure_fluid,
                                      _cfg(case.dx, case, riemann=False))
            s_pan = _score(fp, case.reference_force)
            thickness.append({"case": name, "dx": 0.05, "layers": layers,
                              "oss_error": s_oss["error"], "panel_error": s_pan["error"],
                              "pairwise_error": _score(pairwise_pressure_force(
                                  case.pos_solid, case.vol_solid, case.pos_fluid,
                                  case.vol_fluid, case.pressure_fluid, case.rho_fluid,
                                  _cfg(case.dx, case, riemann=False),
                                  rho0_solid=case.rho_fluid[:1].expand(
                                      case.pos_solid.shape[0])),
                                  case.reference_force)["error"]})
            print(f"thickness {name:16s} layers={layers} oss={s_oss['error']*100:+7.2f}% "
                  f"panel={s_pan['error']*100:+7.2f}%", flush=True)
    payload = {"provenance": OSS_PROVENANCE, "rows": rows, "sensitivity": sens,
               "thickness": thickness,
               "timings_s": timings,
               "parameters": {"kernel_default": "wendland", "h_factor": 1.3,
                              "c2_oss": "fsi2: NoRiemann, NoKernelCorrection, "
                                        "solid h/spacing=1.15 & refinement=2.0"
                                        " (fsi2.cpp:37)"}}
    (ART / "results.json").write_text(json.dumps(payload, indent=2))
    print("wrote", ART / "results.json")
    return payload


if __name__ == "__main__":
    main()
