"""Free-surface air entrainment, bubble transport and aerated drag (P165W2-N04).

Three loosely coupled pieces make up the model chain:

``entrainment``
    Chanson (1997) inception criterion for self-aeration of a free surface on a
    sloping chute, ``V sin(theta) / sqrt(g d cos(theta)) > C_crit``.

``bubble_transport``
    Dilute bubbly-phase transport: bubble rise velocity (Stokes limit, rigid
    sphere drag balance, Mendelson wave theory), an Eulerian advective-diffusion
    solver for the local void fraction, a Lagrangian bubble-cloud integrator, and
    the SPH summation estimate that maps bubble volumes onto a void fraction.

``mixture_acoustics``
    Wood / van Wijngaarden mixture sound speed, acoustic impedance and the
    alpha-resolved impact pressure (with the finite-Mach and wall-mismatch
    corrections plus an explicit, fail-closed validity domain).

``pbm_moments``
    Bubble population balance moment closure (constant coagulation kernel +
    equal-split breakup) that supplies the void fraction and the Sauter size,
    with kernels derived from the aeration source term.

``impact_law``
    The decalibrated aeration buffer: ``p_peak(alpha, v)`` replaces the single
    constant ``k_b``, together with the slamming case set, the design envelope
    and the ``k_b,eq(alpha)`` equivalence curve.

``friction``
    Frictional resistance of the aerated mixture: Chisholm two-phase multiplier
    and a log-law (Wood 1991 form) closure, plus the reduction rate each implies
    at the same mixture flux.
"""

from .entrainment import (
    CAVITY_FRACTION,
    C_CRIT,
    aeration_active,
    chanson_parameter,
    derived_crit_constant,
    entrainment_flux,
    inception_depth,
    inception_froude,
    inception_froude_normal,
    inception_velocity,
    onset_velocity_by_bisection,
)
from .bubble_transport import (
    AIR_20C,
    WATER_20C,
    analytic_void_fraction,
    drag_balance_rise_velocity,
    erf_anchor,
    l2_relative_error,
    mendelson_rise_velocity,
    rise_velocity,
    schiller_naumann_cd,
    solve_void_fraction_field,
    stokes_rise_velocity,
    transport_bubbles,
    void_fraction_from_bubbles,
    void_fraction_histogram,
    y90_for_diffusion,
)
from .friction import (
    aerated_friction_factor,
    chisholm_multiplier,
    chisholm_reduction_rate,
    friction_factor_loglaw,
    lockhart_martinelli_x,
    loglaw_reduction_rate,
    wood_kappa_effective,
)

__all__ = [
    "C_CRIT",
    "CAVITY_FRACTION",
    "aeration_active",
    "chanson_parameter",
    "derived_crit_constant",
    "entrainment_flux",
    "inception_depth",
    "inception_froude",
    "inception_froude_normal",
    "inception_velocity",
    "onset_velocity_by_bisection",
    "AIR_20C",
    "WATER_20C",
    "analytic_void_fraction",
    "drag_balance_rise_velocity",
    "erf_anchor",
    "l2_relative_error",
    "mendelson_rise_velocity",
    "rise_velocity",
    "schiller_naumann_cd",
    "solve_void_fraction_field",
    "stokes_rise_velocity",
    "transport_bubbles",
    "void_fraction_from_bubbles",
    "void_fraction_histogram",
    "y90_for_diffusion",
    "aerated_friction_factor",
    "chisholm_multiplier",
    "chisholm_reduction_rate",
    "friction_factor_loglaw",
    "lockhart_martinelli_x",
    "loglaw_reduction_rate",
    "wood_kappa_effective",
]

from .mixture_acoustics import (
    AIR,
    ALPHA_MIN_RESOLVABLE,
    ALPHA_VW_MAX,
    OutOfDomain,
    STEEL,
    WATER,
    acoustic_impedance,
    alpha_at_min_sound_speed,
    alpha_for_equivalent_buffer_coefficient,
    domain_status,
    equivalent_buffer_coefficient,
    impact_pressure,
    low_frequency_ratio,
    mach_correction,
    min_sound_speed,
    mixture_compressibility,
    mixture_density,
    mixture_gamma,
    pressure_reduction,
    van_wijngaarden_sound_speed,
    wall_mismatch_factor,
    water_hammer_pressure,
    wood_sound_speed,
)
from .pbm_moments import (
    BubbleSpectrum,
    KernelParameters,
    breakage_frequency,
    cascade_spectrum,
    discrete_moment_rates,
    hinze_radius,
    integrate_moments,
    kernel_parameters_from_aeration,
    moment_rates,
    moments_closed_form,
    saffman_turner_kernel,
    sauter_mean_diameter,
)
from .impact_law import (
    BAND_HIGH,
    BAND_LOW,
    LEGACY_BUFFER,
    SLAM_CASES,
    SlamCase,
    aeration_state,
    alpha_at_time,
    case_load,
    case_set,
    curve_family,
    decalibration_table,
    design_envelope,
    local_impact,
    moment_history,
    percentile,
)

__all__ += [
    "AIR",
    "ALPHA_MIN_RESOLVABLE",
    "ALPHA_VW_MAX",
    "OutOfDomain",
    "STEEL",
    "WATER",
    "acoustic_impedance",
    "alpha_at_min_sound_speed",
    "alpha_for_equivalent_buffer_coefficient",
    "domain_status",
    "equivalent_buffer_coefficient",
    "impact_pressure",
    "low_frequency_ratio",
    "mach_correction",
    "min_sound_speed",
    "mixture_compressibility",
    "mixture_density",
    "mixture_gamma",
    "pressure_reduction",
    "van_wijngaarden_sound_speed",
    "wall_mismatch_factor",
    "water_hammer_pressure",
    "wood_sound_speed",
    "BubbleSpectrum",
    "KernelParameters",
    "breakage_frequency",
    "cascade_spectrum",
    "discrete_moment_rates",
    "hinze_radius",
    "integrate_moments",
    "kernel_parameters_from_aeration",
    "moment_rates",
    "moments_closed_form",
    "saffman_turner_kernel",
    "sauter_mean_diameter",
    "BAND_HIGH",
    "BAND_LOW",
    "LEGACY_BUFFER",
    "SLAM_CASES",
    "SlamCase",
    "aeration_state",
    "alpha_at_time",
    "case_load",
    "case_set",
    "curve_family",
    "decalibration_table",
    "design_envelope",
    "local_impact",
    "moment_history",
    "percentile",
]
