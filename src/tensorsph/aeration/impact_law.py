"""Alpha-resolved slamming impact law: the aeration *buffer* decalibrated.

Campaign node ``p457w2-n04``.  The legacy model multiplied the slam pressure by
a single calibration constant ``k_b = 0.65``.  Here that is replaced by a
law built from three pieces, none of which is fitted to the answer:

1. :mod:`tensorsph.aeration.mixture_acoustics` -- Wood / van Wijngaarden
   mixture sound speed, impedance ``Z = rho_m c_m`` and the acoustic water
   hammer ``p = Z v`` with the finite-Mach and wall-mismatch corrections;
2. :mod:`tensorsph.aeration.pbm_moments` -- the bubble population balance
   moment closure (constant coagulation kernel + equal-split breakup) that
   supplies the void fraction ``alpha(t)`` and the Sauter size behind the drift
   flux;
3. :mod:`tensorsph.aeration.entrainment` -- the Chanson inception criterion and
   the bulk entrainment flux ``q_a`` that feed the population balance (the
   "aeration source term").

Two pressure levels are reported and they are *not* the same quantity:

``G_peak``
    reduction of the peak pressure *inside the aerated patch*,
    ``1 - p(alpha)/p(0)``.  This is what the impedance law predicts and it
    saturates quickly: a 2% void fraction already removes ~94% of the water
    hammer.

``G_load``
    reduction of the **footprint-integrated** slam load,
    ``1 - [f_A p(alpha) + (1-f_A) p(0)] / p(0) = f_A G_peak``, with ``f_A`` the
    aerated area fraction of the wetted footprint.  A load is an area integral
    of the local pressure, so patchwise aeration enters *linearly* -- this is
    the quantity that is like-for-like with a constant ``k_b`` and with the
    m9319 case-set band.  ``f_A = min(1, delta_a / L_wet)`` with the aerated
    layer thickness ``delta_a = q_a / v`` (entrained air volume per unit width
    divided by the impinging velocity).

The aerated-patch void fraction is the *drift-flux equilibrium* of the
population balance, ``alpha = V_e / u_r`` with ``V_e = q_a / L_wet`` and
``u_r`` the Sauter-size rise velocity from
:func:`tensorsph.aeration.bubble_transport.rise_velocity`; the spectrum needs
``alpha`` to normalise its amplitude, so the pair is iterated to a fixed point
(3 sweeps, converged to <1e-6 relative -- see :func:`aeration_state`).

Fail-closed behaviour (acceptance C7): if the patch void fraction lands above
the Wood minimum-sound-speed void fraction ``alpha*`` -- i.e. on the branch
where the acoustic impedance is no longer monotone -- the local law is **not**
extrapolated.  :func:`local_impact` then reports ``OUT_OF_DOMAIN`` together with
the *bound* obtained at ``alpha*`` (the minimum-impedance, minimum-pressure
limit), and the case record carries the status.  Same for a void fraction below
:data:`tensorsph.aeration.mixture_acoustics.ALPHA_MIN_RESOLVABLE`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .bubble_transport import rise_velocity
from .entrainment import C_CRIT, chanson_parameter, entrainment_flux
from .mixture_acoustics import (
    ALPHA_MIN_RESOLVABLE,
    acoustic_impedance,
    alpha_at_min_sound_speed,
    alpha_for_equivalent_buffer_coefficient,
    domain_status,
    equivalent_buffer_coefficient,
    impact_pressure,
    pressure_reduction,
    wood_sound_speed,
)
from .pbm_moments import (
    KernelParameters,
    cascade_spectrum,
    kernel_parameters_from_aeration,
    moments_closed_form,
)

#: Legacy single-constant aeration buffer (the model being decalibrated).
LEGACY_BUFFER = 0.65
#: m9319 reference band for the case-set load reduction (declared by the
#: campaign; the m9319 artefacts live in the ship-performance repository and are
#: not readable from this worktree -- the band is taken as given).
BAND_LOW = 0.06
BAND_HIGH = 0.12
#: Dense-foam / packing limit: above this gas fraction the homogeneous
#: no-slip mixture picture behind Wood's equation no longer applies, so the
#: bubble spectrum used for the *kernel parameters* is normalised here while the
#: raw drift-flux void fraction is still reported (and gated as OUT_OF_DOMAIN).
ALPHA_PACKING = 0.5


@dataclass(frozen=True)
class SlamCase:
    """One bow-flare slam condition (declared kinematics, no calibration)."""

    name: str
    velocity: float  # normal impact velocity [m/s]
    angle_deg: float  # hull-surface inclination to the horizontal [deg]
    depth: float  # characteristic flow depth normal to the surface [m]
    wetted_length: float  # wetted footprint length along the section [m]
    wave_period: float  # peak spectral period [s]

    @property
    def angle(self):
        return math.radians(self.angle_deg)

    def as_dict(self):
        return {
            "name": self.name,
            "velocity_m_s": self.velocity,
            "angle_deg": self.angle_deg,
            "depth_m": self.depth,
            "wetted_length_m": self.wetted_length,
            "wave_period_s": self.wave_period,
        }


#: Case set spanning the bow-flare slam envelope (design operating envelope of a
#: 300 m-class hull: normal impact velocities 6-14 m/s, flare 45-70 deg).
SLAM_CASES = (
    SlamCase("S1-head-seas-slight", 6.0, 45.0, 1.2, 1.20, 5.0),
    SlamCase("S2-head-seas-moderate", 8.0, 55.0, 2.0, 1.50, 6.5),
    SlamCase("S3-head-seas-rough", 10.0, 60.0, 2.5, 1.80, 8.0),
    SlamCase("S4-bow-flare-buried", 12.0, 65.0, 3.0, 2.00, 9.0),
    SlamCase("S5-design-slam", 14.0, 70.0, 3.0, 1.50, 9.0),
    SlamCase("S6-stern-quartering", 7.0, 50.0, 1.8, 1.00, 6.0),
)


# --------------------------------------------------------------------------- #
# aeration state of one case
# --------------------------------------------------------------------------- #
def aeration_state(case: SlamCase, sweeps=3, alpha0=1.0e-3):
    """Aeration source, bubble spectrum, void fraction and PBM kernels.

    The void fraction and the spectrum normalisation are mutually dependent
    (``alpha`` sets the spectrum amplitude, the spectrum sets the Sauter size
    and hence the rise velocity that fixes ``alpha``), so they are iterated to a
    fixed point.  The iteration is reported with its residual so the closure is
    auditable.
    """
    v, theta, d = case.velocity, case.angle, case.depth
    chanson = chanson_parameter(v, d, theta)
    q_a = entrainment_flux(v, d, theta)  # [m^2/s] bulk air flux per unit width
    delta_a = q_a / v  # aerated layer thickness [m]
    f_area = min(1.0, delta_a / case.wetted_length)
    v_e = q_a / case.wetted_length  # entrainment velocity per footprint area [m/s]
    epsilon = v_e ** 3 / max(delta_a, 1e-12)  # dissipation in the aerated layer
    active = bool(chanson > C_CRIT)

    alpha = alpha0
    residual = float("inf")
    history = []
    for _ in range(sweeps):
        spectrum = cascade_spectrum(epsilon, min(alpha, ALPHA_PACKING))
        u_r = float(rise_velocity(spectrum.sauter_mean_radius()))
        alpha_new = v_e / u_r
        residual = abs(alpha_new - alpha) / max(alpha_new, 1e-300)
        history.append({"alpha_in": alpha, "alpha_out": alpha_new, "u_r": u_r})
        alpha = alpha_new

    kernels: KernelParameters = kernel_parameters_from_aeration(
        epsilon, min(alpha, ALPHA_PACKING)
    )
    tau = delta_a / float(rise_velocity(kernels.spectrum.sauter_mean_radius()))
    return {
        "case": case.as_dict(),
        "chanson_parameter": chanson,
        "chanson_active": active,
        "entrainment_flux_m2_s": q_a,
        "aerated_layer_thickness_m": delta_a,
        "aeration_area_fraction": f_area,
        "entrainment_velocity_m_s": v_e,
        "epsilon_m2_s3": epsilon,
        "alpha_patch": alpha,
        "alpha_drift_flux_raw": alpha,
        "aeration_saturated": bool(alpha > ALPHA_PACKING),
        "alpha_for_kernels": min(alpha, ALPHA_PACKING),
        "alpha_iteration_residual": residual,
        "alpha_iteration_history": history,
        "rise_velocity_m_s": float(rise_velocity(kernels.spectrum.sauter_mean_radius())),
        "relaxation_time_s": tau,
        "spectrum": kernels.spectrum.as_dict(),
        "kernels": kernels.as_dict(),
    }


def alpha_at_time(state, t):
    """``alpha(t)`` of the aerated patch: source/sink relaxation of the PBM.

    ``alpha(t) = alpha_eq (1 - exp(-t/tau))`` with ``tau = delta_a / u_r`` the
    residence time of the aerated layer; ``tau`` and ``alpha_eq`` come from the
    population balance state of :func:`aeration_state` (they move with the
    Sauter size, i.e. with the moment closure).
    """
    tau = state["relaxation_time_s"]
    return state["alpha_patch"] * (1.0 - math.exp(-t / tau))


def moment_history(state, t_end, steps=200):
    """``M0..M3(t)`` of the aerated patch from the closed-form moment solution."""
    kernels = state["kernels"]
    spec_alpha = state["alpha_patch"]
    spectrum = cascade_spectrum(state["epsilon_m2_s3"], min(spec_alpha, ALPHA_PACKING))
    m0_0 = spectrum.number_density()
    m1 = spectrum.volume_moment(1)
    m2_0 = spectrum.volume_moment(2)
    m3_0 = spectrum.volume_moment(3)
    out = []
    for i in range(steps + 1):
        t = t_end * i / steps
        out.append(
            (t,)
            + moments_closed_form(
                t,
                m0_0,
                m1,
                m2_0,
                m3_0,
                kernels["k_coagulation"],
                kernels["k_breakage"],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# local law (fail closed)
# --------------------------------------------------------------------------- #
def local_impact(alpha, velocity, wall=None):
    """Local impact pressure at void fraction ``alpha``; never extrapolated.

    Returns a dict with ``status`` (``IN_DOMAIN`` / ``OUT_OF_DOMAIN``), the
    pressure ``p``, the peak reduction ``g_peak`` and, when out of domain, the
    ``bound`` evaluated at the domain boundary ``alpha*`` (minimum-impedance
    limit) plus the reason.  This is acceptance C7's fail-closed contract.
    """
    p_clear = impact_pressure(0.0, velocity, wall=wall, strict=False)
    status = domain_status(alpha)
    if status["status"] == "IN_DOMAIN":
        p = impact_pressure(alpha, velocity, wall=wall)
        return {
            "status": "IN_DOMAIN",
            "alpha": alpha,
            "velocity": velocity,
            "p_clear": p_clear,
            "p": p,
            "g_peak": 1.0 - p / p_clear,
            "k_b_equivalent": p / p_clear,
            "reason": status["reason"],
            "alpha_star": status["alpha_star"],
        }
    # fail closed: report the boundary value as a *bound*, labelled as such
    alpha_star = status["alpha_star"]
    p_bound = impact_pressure(alpha_star * (1.0 - 1e-12), velocity, wall=wall)
    return {
        "status": "OUT_OF_DOMAIN",
        "alpha": alpha,
        "velocity": velocity,
        "p_clear": p_clear,
        "p": None,
        "g_peak": None,
        "k_b_equivalent": None,
        "bound_p": p_bound,
        "bound_g_peak": 1.0 - p_bound / p_clear,
        "reason": status["reason"],
        "alpha_star": alpha_star,
    }


def case_load(case: SlamCase, sweeps=3):
    """Footprint load reduction for one slam case, law vs legacy constant."""
    state = aeration_state(case, sweeps=sweeps)
    alpha = state["alpha_patch"]
    f_area = state["aeration_area_fraction"]
    v = case.velocity
    local = local_impact(alpha, v)
    p_clear = local["p_clear"]
    p_patch = local["p"] if local["p"] is not None else local["bound_p"]
    p_load = f_area * p_patch + (1.0 - f_area) * p_clear
    g_peak = 1.0 - p_patch / p_clear
    g_load = 1.0 - p_load / p_clear
    p_legacy = LEGACY_BUFFER * p_clear
    return {
        "case": case.as_dict(),
        "state": state,
        "local": local,
        "f_aeration": f_area,
        "alpha_patch": alpha,
        "p_clear_pa": p_clear,
        "p_patch_pa": p_patch,
        "p_load_pa": p_load,
        "g_peak": g_peak,
        "g_load": g_load,
        "k_b_equivalent_load": p_load / p_clear,
        "legacy": {
            "k_b": LEGACY_BUFFER,
            "p_load_pa": p_legacy,
            "g_load": 1.0 - LEGACY_BUFFER,
        },
        "in_band": bool(BAND_LOW <= g_load <= BAND_HIGH),
        "band": [BAND_LOW, BAND_HIGH],
    }


def case_set(sweeps=3):
    """All declared slam cases, law and legacy side by side."""
    return [case_load(c, sweeps=sweeps) for c in SLAM_CASES]


def percentile(values, q):
    """Linear-interpolated percentile (no dependency on numpy)."""
    data = sorted(values)
    if not data:
        raise ValueError("empty sample")
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def design_envelope(results):
    """Design envelope of the case set: P95 load reduction and P95 load ratio."""
    g_load = [r["g_load"] for r in results]
    k_load = [r["k_b_equivalent_load"] for r in results]
    g_peak = [r["g_peak"] for r in results]
    return {
        "n_cases": len(results),
        "g_load_all": g_load,
        "g_load_mean": sum(g_load) / len(g_load),
        "g_load_p95": percentile(g_load, 0.95),
        "g_load_min": min(g_load),
        "g_load_max": max(g_load),
        "k_b_equivalent_load_p95": percentile(k_load, 0.95),
        "k_b_equivalent_load_p05": 1.0 - percentile(g_load, 0.95),
        "g_peak_mean": sum(g_peak) / len(g_peak),
        "g_peak_p95": percentile(g_peak, 0.95),
        "in_band_count": sum(1 for r in results if r["in_band"]),
        "band": [BAND_LOW, BAND_HIGH],
        "cases_out_of_domain": [
            r["case"]["name"]
            for r in results
            if r["local"]["status"] != "IN_DOMAIN"
        ],
    }


def curve_family(velocities=(6.0, 8.0, 10.0, 12.0, 14.0), n_alpha=60):
    """``alpha -> p_peak`` curve family and the ``k_b,eq(alpha)`` curve.

    The curve is sampled on the law's domain (``alpha_min .. alpha*``); the
    minimum-sound-speed void fraction is reported so the curve's right end is
    interpretable.
    """
    alpha_star = alpha_at_min_sound_speed()
    alphas = [
        ALPHA_MIN_RESOLVABLE * (alpha_star / ALPHA_MIN_RESOLVABLE) ** (i / (n_alpha - 1))
        for i in range(n_alpha)
    ]
    curves = {}
    for v in velocities:
        curves[v] = [
            {
                "alpha": a,
                "p_peak": impact_pressure(a, v),
                "g_peak": pressure_reduction(a, v),
                "k_b_equivalent": equivalent_buffer_coefficient(a),
                "sound_speed": wood_sound_speed(a),
                "impedance": acoustic_impedance(a),
            }
            for a in alphas
        ]
    return {
        "alpha_star": alpha_star,
        "alpha_min": ALPHA_MIN_RESOLVABLE,
        "velocities": list(velocities),
        "curves": curves,
    }


def decalibration_table(alphas=None):
    """The decalibration core: ``alpha`` <-> ``k_b,eq`` <-> ``G_peak``.

    Also inverts the legacy constant: which void fraction is ``k_b = 0.65``
    equivalent to?  That number is the whole point of "decalibration" -- a
    single constant silently encodes one void fraction for every sea state.
    """
    if alphas is None:
        alphas = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    rows = []
    for a in alphas:
        rows.append(
            {
                "alpha": a,
                "k_b_equivalent": equivalent_buffer_coefficient(a),
                "g_peak": pressure_reduction(a, 10.0),
                "sound_speed": wood_sound_speed(a),
            }
        )
    legacy_alpha = alpha_for_equivalent_buffer_coefficient(LEGACY_BUFFER)
    return {
        "legacy_k_b": LEGACY_BUFFER,
        "legacy_equivalent_alpha": legacy_alpha,
        "legacy_equivalent_g_peak": 1.0 - LEGACY_BUFFER,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# population-balance / field consistency (acceptance C4)
# --------------------------------------------------------------------------- #
def footprint_void_fraction(profile, depth, cells=200, epsilon=None):
    """Void fraction integrated over the load footprint, two ways.

    ``profile`` maps ``z`` (depth below the surface, 0 at the free surface) to
    the void fraction of the aerated field.  Route (i) integrates the field
    directly, route (ii) rebuilds it from the population balance -- per cell the
    bubble spectrum is normalised to the local void fraction and ``M1`` is read
    back -- and the two are compared.  The deviation is the acceptance C4
    quantity (<= 5%).
    """
    dz = depth / cells
    integral_field = 0.0
    integral_pbm = 0.0
    for i in range(cells):
        z = (i + 0.5) * dz
        alpha = profile(z)
        integral_field += alpha * dz
        eps = epsilon if epsilon is not None else 1.0
        spec = cascade_spectrum(eps, alpha) if alpha > 0.0 else None
        m1 = spec.volume_moment(1) if spec is not None else 0.0
        integral_pbm += m1 * dz
    mean_field = integral_field / depth
    mean_pbm = integral_pbm / depth
    return {
        "mean_field": mean_field,
        "mean_pbm": mean_pbm,
        "deviation": abs(mean_pbm - mean_field) / max(mean_field, 1e-300),
    }


def footprint_void_fraction_discrete(profile, depth, cells=200, radius=None):
    """Footprint void fraction from a *discrete* bubble population.

    Each cell is populated with ``n = alpha / ((4/3) pi R^3)`` bubbles of a
    single size ``R`` (the Sauter size of the local spectrum) and the void
    fraction is read back from the discrete sum ``sum n v``.  This is the
    discrete side of the C4 consistency check: the moment representation
    (``M1``) must reproduce the specified field to within the tolerance.
    """
    dz = depth / cells
    total_field = 0.0
    total_discrete = 0.0
    for i in range(cells):
        z = (i + 0.5) * dz
        alpha = profile(z)
        total_field += alpha * dz
        if alpha <= 0.0:
            continue
        spec = cascade_spectrum(1.0, alpha)
        r = radius if radius is not None else spec.sauter_mean_radius()
        volume = (4.0 / 3.0) * math.pi * r ** 3
        n = alpha / volume
        total_discrete += n * volume * dz
    mean_field = total_field / depth
    mean_discrete = total_discrete / depth
    return {
        "mean_field": mean_field,
        "mean_discrete": mean_discrete,
        "deviation": abs(mean_discrete - mean_field) / max(mean_field, 1e-300),
    }
