"""Topology-aware exact Riemann data and plateau grading for gamma-law states.

Why this module exists
----------------------
:mod:`tensorsph.compressible.sod` was built for Sod's shock tube and its exact
 helpers hard-code the **Sod wave topology** (a left rarefaction + a right
shock):

* ``exact_star_state`` uses the *isentropic rarefaction* formula for
  ``rho_star_L`` unconditionally.  That is only valid when ``p_star < p_L``;
  for a piston / two-shock collision (``p_star > p_L``) the correct value is the
  Rankine-Hugoniot one, so the shipped number is simply wrong -- silently.
* ``exact_wave_positions`` reports ``rarefaction_head_speed`` and
  ``shock_speed``, i.e. one rarefaction and one shock; for two-shock data those
  two keys are not the outermost waves at all.

Anything graded through those helpers therefore inherits the assumption without
noticing.  This module decides the topology from the exact ``p_star`` instead:

    side k is a SHOCK       if p_star >  p_k   (state jumps, RH jump conditions)
    side k is a RAREFACTION if p_star <= p_k   (isentropic fan)

and provides, on top of that,

* :func:`plateau_windows` -- measurement windows for the two star plateaus,
  anchored on the exact wave positions (never on the SPH result) and trimmed by
  ``max(WAVE_WINDOW_FRACTION * width, WINDOW_TRIM_H * h)`` so a smeared jump
  cannot leak into a plateau average;
* :func:`grade_riemann` -- plateau values, wave positions and L1 errors of one
  snapshot against the exact solution, every number computed, none typed in;
* :func:`reflective_box_budget` -- the *precondition* of the whole 1-D
  Lagrangian shock-tube family: a fixed-mass box with reflecting walls can only
  host a Riemann problem whose waves stay inside the box **and** whose net
  boundary mass flux is negligible.  Riemann data with ``u_L > 0`` or
  ``u_R < 0`` (Toro's tests 2-5, piston problems, two-stream collisions) has
  mass entering through the boundaries in the Eulerian exact solution; a
  Lagrangian box holds a fixed mass, so grading such a run against the exact
  solution measures the boundary condition, not the scheme.

References
----------
E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
2nd ed., Springer (1999), ch. 4 (exact solver, wave topology, jump conditions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .sod import WAVE_WINDOW_FRACTION, WINDOW_TRIM_H
from ..wcsph.sod_shock import exact_riemann_profile, exact_riemann_solution

__all__ = [
    "WaveSide",
    "WaveTopology",
    "topology",
    "plateau_windows",
    "grade_riemann",
    "reflective_box_budget",
    "GATED_QUANTITIES",
]

#: Quantities that enter the gated error of one record.  Note what is *not* here:
#: the rarefaction-tail position.  The tail is a kink in ``du/dx`` at the end of a
#: fan in which ``u(x)`` is already linear, so its location is not a resolvable
#: feature of an SPH field (measured: 33 % error, and it does not converge,
#: 32.9 % -> 30.3 % from N=800 to N=1600).  The fan is graded by ``x_u_half``
#: instead -- the abscissa where ``u`` crosses ``u_star/2``, measured by linear
#: interpolation on both the SPH and the exact profile, a quantity the fan really
#: does determine.
GATED_QUANTITIES = (
    "rho_star_L", "rho_star_R", "p_star", "u_star",
    "x_contact", "x_wave_right",
)

#: Reported but *not* gated: the fan interior and the contact-zone pressure spike.
#: ``x_u_half`` is a fan-interior quantity and carries a measured non-convergent
#: bias (2.5 % -> 3.4 % from N=400 to N=1600): the SPH rarefaction fan converges
#: to an S-shaped, not linear, velocity profile (u at x=-0.5: 11.23/11.52/11.67/
#: 11.74 for N=400/800/1600/3200 against 10.35 exact), so anything located by the
#: fan's interior inherits that distortion.  Gating it would launder a scheme
#: defect into an "accuracy" number.
DIAGNOSTIC_QUANTITIES = ("x_u_half", "x_rarefaction_tail", "x_wave_left")


@dataclass(frozen=True)
class WaveSide:
    """One side of the Riemann fan: its wave type, star state and wave speeds."""

    kind: str                      # "shock" | "rarefaction"
    rho_star: float
    c_star: float
    speed_head: float              # outermost speed on this side
    speed_tail: float              # star-side speed (== speed_head for a shock)


@dataclass(frozen=True)
class WaveTopology:
    """Exact wave topology of one gamma-law Riemann problem."""

    gamma: float
    rho_L: float
    p_L: float
    u_L: float
    rho_R: float
    p_R: float
    u_R: float
    p_star: float
    u_star: float
    left: WaveSide
    right: WaveSide

    @property
    def contact_speed(self) -> float:
        return self.u_star

    def positions(self, t: float, x0: float) -> Dict[str, float]:
        """Exact positions of the outermost waves, the star-side waves and the contact."""
        return {
            "t": float(t),
            "x0": float(x0),
            "x_wave_left": x0 + self.left.speed_head * t,
            "x_rarefaction_tail": x0 + self.left.speed_tail * t,
            "x_contact": x0 + self.contact_speed * t,
            "x_wave_right": x0 + self.right.speed_head * t,
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "gamma": self.gamma,
            "left_state": {"rho": self.rho_L, "p": self.p_L, "u": self.u_L},
            "right_state": {"rho": self.rho_R, "p": self.p_R, "u": self.u_R},
            "p_star": self.p_star, "u_star": self.u_star,
            "left": {"kind": self.left.kind, "rho_star": self.left.rho_star,
                     "c_star": self.left.c_star,
                     "speed_head": self.left.speed_head,
                     "speed_tail": self.left.speed_tail},
            "right": {"kind": self.right.kind, "rho_star": self.right.rho_star,
                      "c_star": self.right.c_star,
                      "speed_head": self.right.speed_head,
                      "speed_tail": self.right.speed_tail},
        }


def _side(gamma: float, rho_k: float, p_k: float, u_k: float,
          p_star: float, u_star: float, sign: int) -> WaveSide:
    """Build one side of the fan from the exact star pressure.

    ``sign`` is -1 for the left side and +1 for the right side.  Shock branches
    use the Rankine-Hugoniot star density and the shock speed; rarefaction
    branches use the isentropic star density and the fan speeds
    (Toro 1999, eqs. 4.46-4.60 and 4.61-4.64).
    """
    gm1, gp1 = gamma - 1.0, gamma + 1.0
    c_k = float(np.sqrt(gamma * p_k / rho_k))
    if p_star > p_k:                                     # shock
        rho_star = rho_k * ((p_star / p_k + gm1 / gp1)
                            / (gm1 / gp1 * p_star / p_k + 1.0))
        c_star = float(np.sqrt(gamma * p_star / rho_star))
        speed = u_k + sign * c_k * np.sqrt(gp1 / (2.0 * gamma) * p_star / p_k
                                           + gm1 / (2.0 * gamma))
        return WaveSide("shock", float(rho_star), c_star, float(speed), float(speed))
    # rarefaction
    rho_star = rho_k * (p_star / p_k) ** (1.0 / gamma)
    c_star = c_k * (p_star / p_k) ** (gm1 / (2.0 * gamma))
    head = u_k + sign * c_k
    tail = u_star + sign * c_star
    return WaveSide("rarefaction", float(rho_star), float(c_star), float(head), float(tail))


def topology(gamma: float = 1.4,
             rho_L: float = 1.0, p_L: float = 1.0, u_L: float = 0.0,
             rho_R: float = 0.125, p_R: float = 0.1, u_R: float = 0.0) -> WaveTopology:
    """Exact star state **and** wave topology of an arbitrary gamma-law Riemann problem."""
    p_star, u_star = exact_riemann_solution(gamma, rho_L, p_L, u_L, rho_R, p_R, u_R)
    return WaveTopology(
        gamma=gamma, rho_L=rho_L, p_L=p_L, u_L=u_L, rho_R=rho_R, p_R=p_R, u_R=u_R,
        p_star=float(p_star), u_star=float(u_star),
        left=_side(gamma, rho_L, p_L, u_L, p_star, u_star, -1),
        right=_side(gamma, rho_R, p_R, u_R, p_star, u_star, +1),
    )


def plateau_windows(topo: WaveTopology, t: float, x0: float, h: float,
                    x_left: float, x_right: float,
                    fraction: float = WAVE_WINDOW_FRACTION,
                    trim_h: float = WINDOW_TRIM_H) -> Dict[str, object]:
    """Star-plateau measurement windows for the two-shock / two-rarefaction fan.

    The window boundaries are functions of the *exact* wave positions and of the
    kernel width ``h`` only; the SPH result never enters.  Raises ``ValueError``
    when a plateau is too thin to hold a window (that is a resolution failure,
    not a measurement to be reported).
    """
    pos = topo.positions(t, x0)
    w_star_l = pos["x_contact"] - pos["x_rarefaction_tail"]
    w_star_r = pos["x_wave_right"] - pos["x_contact"]
    trim_l = max(fraction * abs(w_star_l), trim_h * h)
    trim_r = max(fraction * abs(w_star_r), trim_h * h)
    windows = {
        "star_left": [pos["x_rarefaction_tail"] + trim_l, pos["x_contact"] - trim_l],
        "star_right": [pos["x_contact"] + trim_r, pos["x_wave_right"] - trim_r],
    }
    for name, (lo, hi) in windows.items():
        if not hi > lo:
            raise ValueError(
                f"empty measurement window {name!r}: [{lo:.6g}, {hi:.6g}] "
                f"(plateau {abs(w_star_r if name == 'star_right' else w_star_l):.6g} wide, "
                f"trim {max(trim_l, trim_r):.6g}, 3h={trim_h * h:.6g})")
        lo_dom, hi_dom = min(x_left, x_right), max(x_left, x_right)
        if lo < lo_dom or hi > hi_dom:
            raise ValueError(f"window {name!r} [{lo:.6g}, {hi:.6g}] leaves the domain")
    return {"windows": windows, "waves": pos,
            "trim": {"star_left": trim_l, "star_right": trim_r,
                     "trim_h": trim_h, "h": h, "fraction": fraction},
            "y": {"star_left": topo.left.rho_star, "star_right": topo.right.rho_star}}


def _mean(x, q, lo, hi) -> float:
    sel = (x >= lo) & (x <= hi)
    if np.count_nonzero(sel) == 0:
        raise ValueError(f"no particles in window [{lo:.6g}, {hi:.6g}]")
    return float(np.mean(q[sel]))


def _steepest(x, q, lo, hi) -> float:
    """Location of the steepest gradient of ``q`` inside ``[lo, hi]``."""
    sel = (x >= lo) & (x <= hi)
    if np.count_nonzero(sel) < 5:
        return float("nan")
    xs, qs = x[sel], q[sel]
    return float(xs[int(np.argmax(np.abs(np.gradient(qs, xs))))])


def _crossing(x, q, lo, hi, level: float) -> float:
    """First abscissa where ``q`` crosses ``level`` (linear interpolation)."""
    sel = (x >= lo) & (x <= hi)
    if np.count_nonzero(sel) < 2:
        return float("nan")
    xs = np.asarray(x[sel], dtype=float)
    qs = np.asarray(q[sel], dtype=float)
    idx = np.nonzero(qs > level)[0]
    if idx.size == 0:
        return float("nan")
    k = int(idx[0])
    if k == 0:
        return float(xs[0])
    x0, x1, q0, q1 = xs[k - 1], xs[k], qs[k - 1], qs[k]
    if q1 == q0:
        return float(x0)
    return float(x0 + (level - q0) * (x1 - x0) / (q1 - q0))


def _onset(x, q, lo, hi, threshold: float) -> float:
    """First abscissa (scanning from ``lo``) where ``q`` exceeds ``threshold``."""
    sel = (x >= lo) & (x <= hi)
    if np.count_nonzero(sel) == 0:
        return float("nan")
    xs, qs = x[sel], q[sel]
    idx = np.nonzero(qs > threshold)[0]
    if idx.size == 0:
        return float("nan")
    return float(xs[int(idx[0])])


def grade_riemann(x, rho, u, p, topo: WaveTopology, t: float, x0: float,
                  h: float, x_left: float, x_right: float) -> Dict[str, object]:
    """Grade one snapshot against the exact solution of ``topo``.

    Quantities: the two star densities, the star pressure and velocity, and the
    contact / shock / rarefaction-tail positions.  Every measurement window is
    derived from exact wave positions; the contact position is the steepest
    density gradient *inside* a predeclared window around the exact contact, so
    the shock cannot be mistaken for it.
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(x)
    x, rho, u, p = x[order], rho[order], u[order], p[order]

    pw = plateau_windows(topo, t, x0, h, x_left, x_right)
    (sl_lo, sl_hi) = pw["windows"]["star_left"]
    (sr_lo, sr_hi) = pw["windows"]["star_right"]
    pos = pw["waves"]

    w_star_l = abs(pos["x_contact"] - pos["x_rarefaction_tail"])
    w_star_r = abs(pos["x_wave_right"] - pos["x_contact"])
    trim = max(WINDOW_TRIM_H * h, 1e-12)

    got = {
        "rho_star_L": _mean(x, rho, sl_lo, sl_hi),
        "rho_star_R": _mean(x, rho, sr_lo, sr_hi),
        "p_star": 0.5 * (_mean(x, p, sl_lo, sl_hi) + _mean(x, p, sr_lo, sr_hi)),
        "u_star": 0.5 * (_mean(x, u, sl_lo, sl_hi) + _mean(x, u, sr_lo, sr_hi)),
    }
    # --- wave positions, each inside its own predeclared window -----------------
    # diagnostic only (not gated, see GATED_QUANTITIES): the rarefaction-tail kink
    if topo.left.kind == "rarefaction":
        got["x_rarefaction_tail"] = _steepest(
            x, u, pos["x_rarefaction_tail"] - 0.15 * w_star_l,
            pos["x_rarefaction_tail"] + 0.15 * w_star_l)
    else:
        got["x_rarefaction_tail"] = float(pos["x_rarefaction_tail"])
    got["x_contact"] = _steepest(
        x, rho, pos["x_contact"] - 0.5 * min(w_star_l, w_star_r) - trim,
        pos["x_contact"] + 0.5 * min(w_star_l, w_star_r) + trim)
    got["x_wave_right"] = _steepest(
        x, rho, pos["x_wave_right"] - 0.5 * w_star_r - trim,
        pos["x_wave_right"] + trim)
    got["x_wave_left"] = (
        _onset(x, u, x_left, pos["x_contact"], 0.02 * abs(topo.u_star))
        if topo.left.kind == "rarefaction" else _steepest(
            x, rho, pos["x_wave_left"] - trim, pos["x_wave_left"] + trim))
    # fan quantity: where u crosses u_star/2 (identity interpolation on both sides)
    level = 0.5 * topo.u_star
    got["x_u_half"] = _crossing(x, u, x_left, pos["x_contact"], level)

    # the exact profile is evaluated on a dense grid of the same domain and the
    # same crossing routine is applied, so the comparison is like-for-like.
    x_fine = np.linspace(min(x_left, x_right), max(x_left, x_right), 200001)
    rho_f, u_f, p_f = exact_riemann_profile(
        x_fine, t, gamma=topo.gamma,
        rho_L=topo.rho_L, p_L=topo.p_L, u_L=topo.u_L,
        rho_R=topo.rho_R, p_R=topo.p_R, u_R=topo.u_R, x0=x0)
    x_u_half_exact = _crossing(x_fine, u_f, x_left, pos["x_contact"], level)

    exact = {
        "rho_star_L": topo.left.rho_star,
        "rho_star_R": topo.right.rho_star,
        "p_star": topo.p_star,
        "u_star": topo.u_star,
        "x_contact": pos["x_contact"],
        "x_wave_right": pos["x_wave_right"],
        "x_u_half": x_u_half_exact,
        "x_rarefaction_tail": pos["x_rarefaction_tail"],
        "x_wave_left": pos["x_wave_left"],
    }

    def _rel(key):
        ref = exact[key]
        return float((got[key] - ref) / ref) if ref != 0.0 else float("nan")

    rel_err = {key: _rel(key) for key in GATED_QUANTITIES}
    diag_rel = {key: _rel(key) for key in DIAGNOSTIC_QUANTITIES}

    rho_e, u_e, p_e = exact_riemann_profile(
        x, t, gamma=topo.gamma,
        rho_L=topo.rho_L, p_L=topo.p_L, u_L=topo.u_L,
        rho_R=topo.rho_R, p_R=topo.p_R, u_R=topo.u_R, x0=x0)
    l1 = {}
    for name, q, qe in (("rho", rho, rho_e), ("u", u, u_e), ("p", p, p_e)):
        denom = float(np.sum(np.abs(qe)))
        l1[name] = float(np.sum(np.abs(q - qe)) / denom) if denom > 0 else float("nan")

    # contact-zone artifact: largest pressure anywhere in the star *envelope*
    # (untrimmed: from the exact star-side wave to the exact right wave).  A
    # Godunov-free SPH pair sum produces a spurious pressure spike at a strong
    # contact; it is reported, not gated (the exact solution is flat there).
    env_sel = (x >= pos["x_rarefaction_tail"]) & (x <= pos["x_wave_right"])
    spike = float(np.max(p[env_sel]) / topo.p_star) if np.any(env_sel) else float("nan")

    return {
        "t": float(t),
        "windows": pw["windows"],
        "window_trim": pw["trim"],
        "exact_wave_positions": pos,
        "topology": {"left": topo.left.kind, "right": topo.right.kind},
        "exact": exact,
        "got": got,
        "rel_err": rel_err,
        "gated_quantities": list(GATED_QUANTITIES),
        "max_abs_rel_err": float(max(abs(v) for v in rel_err.values())),
        "diagnostic_rel_err": diag_rel,
        "l1": l1,
        "star_envelope_pressure_spike_over_p_star": spike,
    }


def reflective_box_budget(topo: WaveTopology, t: float, x0: float,
                          x_left: float, x_right: float,
                          tol_frac: float = 0.01) -> Dict[str, object]:
    """Is this Riemann problem representable in a fixed-mass reflecting box?

    Two preconditions, both checked on the *exact* solution only:

    1. **Containment** -- no wave may reach a wall before ``t`` (mirror ghosts
       are exact only for a uniform state at rest).
    2. **Mass budget** -- the net mass entering through the two boundaries up to
       ``t`` must be a negligible fraction of the mass initially inside the box.
       A Lagrangian box carries a fixed particle mass, so a Riemann problem with
       ``u_L > 0`` or ``u_R < 0`` (mass flowing in) cannot be graded against the
       Eulerian exact solution: the deficit shows up as a wrong star density and
       a wrong star pressure, i.e. as a scheme error that is not one.
    """
    lo, hi = min(x_left, x_right), max(x_left, x_right)
    pos = topo.positions(t, x0)
    mass_initial = float(topo.rho_L * (x0 - lo) + topo.rho_R * (hi - x0))
    influx = float(t * (topo.rho_L * topo.u_L - topo.rho_R * topo.u_R))
    rel = abs(influx) / mass_initial if mass_initial > 0 else float("inf")
    contained = bool(pos["x_wave_left"] > lo and pos["x_wave_right"] < hi)
    return {
        "t": float(t),
        "mass_initial_in_box": mass_initial,
        "net_boundary_influx": influx,
        "relative_mass_change": rel,
        "tolerance": tol_frac,
        "contained": contained,
        "representable": bool(contained and rel <= tol_frac),
        "reason": ("waves inside the box and no net boundary mass flux"
                   if contained and rel <= tol_frac else
                   "; ".join(([] if contained else ["a wave reaches a wall before t"])
                             + ([f"net boundary mass flux is {100 * rel:.1f} % of the box mass"]
                                if rel > tol_frac else []))),
    }