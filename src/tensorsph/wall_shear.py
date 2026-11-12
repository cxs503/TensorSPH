"""Wall-shear estimators for wall-bounded SPH channel flows.

Why this module exists
----------------------
A plane channel driven by a uniform body force ``g_x = g`` of height ``H`` with
two no-slip walls has *two* independent observables of the wall shear, and they
are not interchangeable:

1. **Drag from the momentum budget** (no near-wall stencil).  The only
   x-momentum sink of the discrete system is the fluid-wall friction, so the
   mass-averaged streamwise velocity ``<u>`` obeys

   .. math:: \\frac{d\\langle u\\rangle}{dt} = g - \\frac{D}{M},
             \\qquad \\frac{D}{M} = \\frac{2 \\tau_w}{\\rho H}

   hence

   .. math:: \\tau_w = \\tfrac{1}{2} \\rho H \\left(g - \\dot{\\langle u\\rangle}\\right).

   In a steady channel ``\\dot{\\langle u\\rangle}\\to 0`` and the wall *must*
   remove exactly what the body force injects.  This estimator uses no kernel,
   no spacing and no near-wall row, so it is insensitive to how badly the wall
   layer is resolved.

2. **Near-wall velocity gradient** ``\\tau_w = \\rho\\nu\\,\\mathrm{d}u/\\mathrm{d}y|_0``
   read off a mean profile.  On a *resolved* wall layer this is the textbook
   definition; on an SPH wall layer of one particle spacing the *stencil* decides
   the answer, and the bias has nothing to do with the wall model.  Measured here
   (``tests/test_wall_shear_estimators.py``) and on the committed
   plane-Poiseuille ladder: a one-spacing one-sided secant on a smeared wall row
   reads 30-50 % **low** (that is the family of the published ``-42 %`` /
   ``-49.7 %`` numbers), a parabola fit that includes the wall row reads
   ``-18 %`` at ``H/dx = 8..16``, and the same fit over the interior band
   ``0.25 <= y/H <= 0.75`` reads within a few percent.  Details:
   ``docs/verification_poiseuille_wall_shear_estimator.md`` and
   ``docs/verification_poiseuille_wall_shear_library_gate.md``.

Consequences for acceptance gates
---------------------------------
A gate *named* "wall shear budget" must read observable 1, otherwise it fails a
correct wall model for a reason that lives in the post-processing.  Observable 2
is still reported -- it is what an experimenter with probes would measure -- but
as a *diagnostic*, explicitly not a pass/fail criterion while the wall layer
spans one spacing.

Estimator algebra is verified on closed-form data in
``tests/test_wall_shear_estimators.py`` (constant trace: ``\\tau_w`` exact to
machine precision; known ramp: exact; analytic parabola: ``<1e-12`` relative),
and against the committed plane-Poiseuille artefacts.

Nothing in this module advances an SPH state: every entry point is pure
post-processing of a stored trace or a mean profile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: **Window length** fraction of the run used for the drift fit, counted back
#: from ``t_end``: the window is ``[(1 - fraction) * t_end, t_end]``.  The flow
#: has to be through the initial transient, so 40 % of the run is used.
#:
#: Beware a naming collision with the legacy script
#: ``benchmarks/poiseuille/wall_shear_estimators.py``: its ``fraction`` is a
#: **start** fraction (``t_lo = fraction * t_end``), so *the same number selects
#: a different window* -- its ``0.4`` is the last 60 % of the run, i.e. exactly
#: this module's ``0.6``.  ``tests/test_wall_shear_window_audit.py`` pins both
#: conventions on one trace so the collision cannot be silently re-introduced,
#: and the sensitivity of the reading to the window is reported by
#: :func:`window_sensitivity` (see ``docs/audit_verdict_m16581.md``).
DEFAULT_WINDOW_FRACTION = 0.4

#: Post-transient windows (as **window length** fractions, see above) used to
#: report how much the momentum-budget reading moves with the fitting window.
#: All of them lie after the initial spin-up, so the spread is the estimator's
#: own uncertainty -- not a statement about the physics.
DEFAULT_SENSITIVITY_FRACTIONS = (0.2, 0.3, 0.4, 0.5)


@dataclass
class DriftRate:
    """Least-squares slope of a trace quantity, with its own uncertainty."""

    slope: float
    stderr: float
    residual_rms: float
    r_squared: float
    n_samples: int
    t_lo: float
    t_hi: float

    def as_dict(self) -> dict:
        return {
            "slope": self.slope,
            "stderr": self.stderr,
            "residual_rms": self.residual_rms,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
            "t_lo": self.t_lo,
            "t_hi": self.t_hi,
        }


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def drift_rate(trace, t_end: float | None = None,
               fraction: float = DEFAULT_WINDOW_FRACTION,
               key: str = "u_mean") -> DriftRate:
    """Least-squares slope ``d<u>/dt`` over the last ``fraction`` of a trace.

    Parameters
    ----------
    trace
        Sequence of samples, each a mapping with ``"t"`` and ``key``.  Samples
        with a non-finite value are dropped (a diverged run then raises a clear
        error instead of returning a silently meaningless slope).
    t_end
        End time of the run; defaults to the largest ``t`` in ``trace``.
    fraction
        Window length as a fraction of ``t_end``, counted back from ``t_end``.
    key
        Which trace field carries the mass-averaged streamwise velocity.

    Raises
    ------
    ValueError
        If fewer than two finite samples fall inside the window.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    t = np.array([float(p["t"]) for p in trace], dtype=float)
    u = np.array([float(p[key]) for p in trace], dtype=float)
    if t.size == 0:
        raise ValueError("trace is empty")
    t_stop = float(t_end) if t_end is not None else float(np.max(t))
    t_lo = (1.0 - fraction) * t_stop
    keep = (t >= t_lo) & (t <= t_stop) & np.isfinite(u) & np.isfinite(t)
    t_w, u_w = t[keep], u[keep]
    n = int(t_w.size)
    if n < 2:
        raise ValueError(
            f"need >=2 finite samples in window [{t_lo:.6g}, {t_stop:.6g}], got {n}"
        )
    t_mean = float(np.mean(t_w))
    u_mean = float(np.mean(u_w))
    sxx = float(np.sum((t_w - t_mean) ** 2))
    if sxx <= 0.0:
        raise ValueError("degenerate window: all sample times identical")
    slope = float(np.sum((t_w - t_mean) * (u_w - u_mean)) / sxx)
    intercept = u_mean - slope * t_mean
    residual = u_w - (intercept + slope * t_w)
    dof = n - 2
    resid_var = float(np.sum(residual ** 2)) / dof if dof > 0 else float("nan")
    stderr = math.sqrt(resid_var / sxx) if dof > 0 else float("nan")
    var_u = float(np.sum((u_w - u_mean) ** 2))
    r_squared = 1.0 - float(np.sum(residual ** 2)) / var_u if var_u > 0.0 else float("nan")
    return DriftRate(
        slope=slope,
        stderr=stderr,
        residual_rms=float(np.sqrt(np.mean(residual ** 2))),
        r_squared=r_squared,
        n_samples=n,
        t_lo=float(t_w[0]),
        t_hi=float(t_w[-1]),
    )


def momentum_budget_wall_shear(trace, height: float, body_force: float,
                               rho0: float = 1000.0,
                               t_end: float | None = None,
                               fraction: float = DEFAULT_WINDOW_FRACTION,
                               drift_gain: float = 1.0,
                               sensitivity_fractions=DEFAULT_SENSITIVITY_FRACTIONS
                               ) -> dict:
    """Wall shear implied by the x-momentum budget of the whole channel.

    ``tau_w = (1/2) rho0 H (g - d<u>/dt)``.

    ``drift_gain`` calibrates the *measurement* of ``d<u>/dt``, not the physics:
    it is the ratio ``(measured drift of the free-slip control) / g``, which for
    a run with no wall friction must be ``1`` by construction.  Any departure
    from 1 is the gain error of the finite-window slope estimator, and dividing
    the measured drift by it removes that bias.  Default ``1.0`` = uncalibrated.

    Returns a dict with the estimator value, the momentum-balance constant
    ``rho0 g H / 2``, the closure ``D/(M g)`` and the relative error in percent.
    ``available`` is ``False`` (all values ``nan``) when the trace carries no
    usable drift, so callers cannot mistake a missing measurement for a pass.

    ``fraction`` is a **window length** fraction (``t_lo = (1 - fraction)
    t_end``).  ``sensitivity_fractions`` additionally reports the same reading
    over other post-transient windows and the resulting band
    (``window_sensitivity``), because a single window cannot show how much of
    the quoted number is the estimator's own bias; pass an empty sequence to
    skip the extra fits.
    """
    required = 0.5 * rho0 * body_force * height
    try:
        drift = drift_rate(trace, t_end=t_end, fraction=fraction)
    except (ValueError, KeyError) as exc:
        return {
            "estimator": "momentum_budget",
            "available": False,
            "reason": str(exc),
            "tau_wall": float("nan"),
            "tau_wall_required": required,
            "ratio": float("nan"),
            "momentum_closure": float("nan"),
            "rel_error_pct": float("nan"),
            "drift": float("nan"),
            "drift_stderr": float("nan"),
            "n_samples": 0,
            "window_length_fraction": fraction,
            "window_sensitivity": _empty_sensitivity(sensitivity_fractions),
        }
    if not (drift_gain > 0.0):
        raise ValueError(f"drift_gain must be positive, got {drift_gain!r}")
    drag_per_mass = body_force - drift.slope / drift_gain
    tau_wall = 0.5 * rho0 * height * drag_per_mass
    return {
        "estimator": "momentum_budget",
        "available": True,
        "tau_wall": tau_wall,
        "tau_wall_required": required,
        "ratio": tau_wall / required if required else float("nan"),
        "momentum_closure": drag_per_mass / body_force if body_force else float("nan"),
        "rel_error_pct": 100.0 * (tau_wall - required) / required if required else float("nan"),
        "drift": drift.slope,
        "drift_stderr": drift.stderr,
        "drift_r_squared": drift.r_squared,
        "n_samples": drift.n_samples,
        "window": [drift.t_lo, drift.t_hi],
        "window_length_fraction": fraction,
        "drift_gain": drift_gain,
        "window_sensitivity": window_sensitivity(
            trace, height, body_force, rho0, t_end=t_end,
            fractions=sensitivity_fractions, drift_gain=drift_gain),
    }


def _empty_sensitivity(fractions) -> dict:
    nan = float("nan")
    return {
        "fractions": [float(f) for f in fractions],
        "definitions": "fraction = window length / run length, window = [(1-f) t_end, t_end]",
        "per_fraction": {},
        "rel_error_pct_band": [nan, nan],
        "worst_abs_rel_error_pct": nan,
        "spread_pct_points": nan,
    }


def window_sensitivity(trace, height: float, body_force: float,
                       rho0: float = 1000.0, t_end: float | None = None,
                       fractions=DEFAULT_SENSITIVITY_FRACTIONS,
                       drift_gain: float = 1.0) -> dict:
    """Spread of the momentum-budget wall shear over alternative drift windows.

    The estimator returns ``tau_w = (1/2) rho H (g - d<u>/dt)``; on a run whose
    mean velocity is *almost* steady, ``tau_w`` is dominated by the residual
    transient inside the fitting window, so the quoted number is only as
    trustworthy as the window is short.  This function re-fits the same trace
    over every window length in ``fractions`` and reports

    * ``rel_error_pct_band`` -- ``[min, max]`` of the per-window errors,
    * ``worst_abs_rel_error_pct`` -- the conservative single number a gate
      should compare against its limit,
    * ``spread_pct_points`` -- ``max - min``, the estimator's own uncertainty.

    All windows must be post-transient: a window that reaches back into the
    spin-up is not a steady-state reading at all (on the committed ISPH ladder
    a full-run window reads ``-3.6 %``, entirely from the start-up ramp).
    Windows with fewer than two usable samples are skipped and reported as
    ``None`` instead of being silently dropped.
    """
    per: dict[str, dict] = {}
    errors: list[float] = []
    for f in fractions:
        try:
            est = momentum_budget_wall_shear(
                trace, height, body_force, rho0, t_end=t_end, fraction=f,
                drift_gain=drift_gain, sensitivity_fractions=())
        except (ValueError, KeyError) as exc:  # pragma: no cover - defensive
            per[f"{f:g}"] = {"available": False, "reason": str(exc)}
            continue
        entry = {
            "available": bool(est["available"]),
            "n_samples": est["n_samples"],
            "drift": est["drift"],
            "rel_error_pct": est["rel_error_pct"],
        }
        per[f"{f:g}"] = entry
        if est["available"] and math.isfinite(est["rel_error_pct"]):
            errors.append(float(est["rel_error_pct"]))
    if errors:
        band = [min(errors), max(errors)]
        spread = band[1] - band[0]
        worst = max(abs(v) for v in errors)
    else:
        band = [float("nan"), float("nan")]
        spread = float("nan")
        worst = float("nan")
    return {
        "fractions": [float(f) for f in fractions],
        "definitions": "fraction = window length / run length, window = [(1-f) t_end, t_end]",
        "per_fraction": per,
        "rel_error_pct_band": band,
        "worst_abs_rel_error_pct": worst,
        "spread_pct_points": spread,
    }


def profile_wall_shear(profile, height: float, viscosity: float,
                       rho0: float = 1000.0, lo: float = 0.0,
                       hi: float = 1.0, body_force: float = 0.0) -> dict:
    """Wall shear from a quadratic least-squares fit of the mean profile.

    ``u = c0 + c1 y + c2 y^2`` over the rows with ``lo <= y/H <= hi``; the
    no-slip wall at ``y = 0`` then has ``du/dy = c1`` and
    ``tau_w = rho0 nu c1``.  For an exact parabola the fit returns the analytic
    value to machine precision.  With a smeared wall row inside the band the fit
    is biased (``-18 %`` on the committed ladder, direction set by how the smear
    redistributes velocity), which is why this reading is a *diagnostic* and not
    the acceptance gate.
    """
    rows = [p for p in profile if lo <= float(p["y_over_H"]) <= hi]
    if len(rows) < 3:
        return {
            "estimator": "profile_parabola",
            "available": False,
            "reason": f"need >=3 profile rows in band, got {len(rows)}",
            "n_rows": len(rows),
            "du_dy_wall": float("nan"),
            "tau_wall": float("nan"),
        }
    y = np.array([p["y"] for p in rows], dtype=float)
    u = np.array([p["u_mean"] for p in rows], dtype=float)
    design = np.stack((np.ones_like(y), y, y * y), axis=1)
    coef, *_ = np.linalg.lstsq(design, u, rcond=None)
    c0, c1, c2 = (float(v) for v in coef)
    residual = design @ coef - u
    curvature = -c2
    return {
        "estimator": "profile_parabola",
        "available": True,
        "n_rows": len(rows),
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "du_dy_wall": c1,
        "tau_wall": rho0 * viscosity * c1,
        "nu_eff": effective_viscosity(body_force, curvature),
        "fit_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "u_centre_fit": c0 + c1 * (height / 2.0) + c2 * (height ** 2 / 4.0),
    }


def effective_viscosity(body_force: float, curvature: float) -> float:
    """Viscosity implied by a parabola of curvature ``c2`` (``u'' = 2 c2``).

    Steady balance ``nu u'' = -g`` with ``u'' = 2 c2`` gives ``nu = g / (2 c2)``.
    With ``body_force == 0`` (no driving force) or a non-concave fit the return
    value is ``nan``: curvature alone says nothing about ``nu``.
    """
    if not math.isfinite(curvature) or curvature <= 0.0 or body_force == 0.0:
        return float("nan")
    return body_force / (2.0 * curvature)


__all__ = [
    "DEFAULT_SENSITIVITY_FRACTIONS",
    "DEFAULT_WINDOW_FRACTION",
    "DriftRate",
    "effective_viscosity",
    "drift_rate",
    "momentum_budget_wall_shear",
    "profile_wall_shear",
    "window_sensitivity",
]
