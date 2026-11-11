"""Acceptance tests for campaign node p457w2-n04 (aeration buffer decalibration).

The node's acceptance criteria C1--C7 are mapped one-to-one onto tests:

======  ==============================================================
C1      Wood closed form vs 60-digit mpmath reference (<=1e-6),
        minimum-sound-speed point closed form vs numerical minimisation
C2      ``alpha -> 0`` and ``alpha -> 1`` limits of the impact pressure
        (residual <= 0.1 %)
C3      PBM moment closure: closed form vs RK4 (<=1 %), closure exactness
        vs the direct double-sum collision integral, ``M1`` conservation,
        ``M0`` decay, ``M2`` single-rate exponential evolution
C4      PBM void fraction vs the specified field (deviation <= 5 %)
C5      case set with ``k_b = 0.65`` replaced by ``Z(alpha)``: load-reduction
        envelope inside the m9319 [6 %, 12 %] band
C6      monotonicity / boundedness of ``p_peak(alpha)`` on ``alpha < alpha*``
C7      fail-closed: ``alpha > alpha*`` or below the resolution floor ->
        OUT_OF_DOMAIN (never extrapolated)
======  ==============================================================

Everything here is a pure-function / small-ODE check (no SPH run), so the file
stays fast.
"""
import math

import mpmath as mp
import pytest

from tensorsph.aeration import impact_law as law
from tensorsph.aeration import mixture_acoustics as mx
from tensorsph.aeration import pbm_moments as pbm


# --------------------------------------------------------------------------- #
# C1 -- Wood closed form and the minimum-sound-speed point
# --------------------------------------------------------------------------- #
def test_c1_wood_matches_high_precision_reference():
    """Pointwise ``c_m(alpha)`` vs a 60-digit arbitrary-precision evaluation."""
    mp.mp.dps = 60
    rho_l = mp.mpf("998.2")
    rho_g = mp.mpf("1.204")
    c_l = mp.mpf("1483.0")
    c_g = mp.mpf("343.0")
    worst = 0.0
    for i in range(1, 200):
        alpha = mp.mpf(i) / 200
        rho_m = alpha * rho_g + (1 - alpha) * rho_l
        comp = alpha / (rho_g * c_g**2) + (1 - alpha) / (rho_l * c_l**2)
        ref = 1 / mp.sqrt(rho_m * comp)
        got = mx.wood_sound_speed(float(alpha))
        worst = max(worst, abs(got - float(ref)) / float(ref))
    assert worst <= 1.0e-6, f"max relative deviation {worst:.3e}"


def test_c1_bulk_modulus_route_agrees():
    """Independent code path: ``c = sqrt(K_m/rho_m)`` with ``K_m`` the harmonic
    bulk modulus (catches algebra/coding slips, not floating point)."""
    for i in range(1, 100):
        alpha = i / 100.0
        k_g = mx.AIR["rho"] * mx.AIR["c"] ** 2
        k_l = mx.WATER["rho"] * mx.WATER["c"] ** 2
        k_m = 1.0 / (alpha / k_g + (1.0 - alpha) / k_l)
        ref = math.sqrt(k_m / mx.mixture_density(alpha))
        assert abs(mx.wood_sound_speed(alpha) - ref) / ref <= 1.0e-12


def test_c1_minimum_sound_speed_point_is_stationary():
    """``dc_m/dalpha = 0`` at the closed-form ``alpha*``; ``c_min`` ~ 23.8 m/s."""
    a_star = mx.alpha_at_min_sound_speed()
    c_min = mx.min_sound_speed()
    h = 1.0e-6
    deriv = (
        mx.wood_sound_speed(a_star + h) - mx.wood_sound_speed(a_star - h)
    ) / (2.0 * h)
    assert abs(deriv) <= 1.0e-3, deriv
    # the closed form must beat any neighbouring void fraction
    for da in (-1e-3, -1e-4, 1e-4, 1e-3):
        assert mx.wood_sound_speed(a_star + da) > c_min
    # magnitude check on the physical value quoted in the brief (~20 m/s)
    assert 15.0 <= c_min <= 30.0, c_min
    assert abs(c_min - 23.81) <= 0.2, c_min
    assert 0.45 <= a_star <= 0.55, a_star


def test_c1_van_wijngaarden_is_the_dilute_limit_of_wood():
    """van Wijngaarden is exact as ``alpha -> 0`` and agrees to ``O(alpha)``.

    Two ingredients separate it from Wood: (i) ``rho_m -> rho_l`` (an ``O(alpha)``
    density term) and (ii) the gas compressibility written as ``gamma_g p``
    instead of ``rho_g c_g^2``.  With the *same* gas compressibility the two
    must differ by the density term only, ~``alpha``/2 in relative terms."""
    assert abs(mx.van_wijngaarden_sound_speed(0.0) - mx.WATER["c"]) <= 1.0e-6
    c_g_eff = math.sqrt(mx.AIR["gamma"] * mx.P_ATM / mx.AIR["rho"])
    for alpha in (1e-4, 1e-3, 1e-2):
        c_wood = mx.wood_sound_speed(alpha, c_g=c_g_eff)
        c_vw = mx.van_wijngaarden_sound_speed(alpha)
        assert abs(c_wood - c_vw) / c_wood <= 2.0 * alpha
    # with the *tabulated* air sound speed the two differ by the 0.15 %
    # inconsistency between rho_g c_g^2 and gamma_g p -- a declared, reported
    # difference, not a modelling error
    for alpha in (1e-4, 1e-3, 1e-2):
        c_wood = mx.wood_sound_speed(alpha)
        c_vw = mx.van_wijngaarden_sound_speed(alpha)
        assert abs(c_wood - c_vw) / c_wood <= 2.0e-3 + 1.2 * alpha


# --------------------------------------------------------------------------- #
# C2 -- two limits of the impact law
# --------------------------------------------------------------------------- #
def test_c2_clear_water_limit_residual_below_0_1_percent():
    for v in (1.0, 6.0, 10.0, 14.0, 25.0):
        p = mx.impact_pressure(0.0, v, strict=False)
        ref = mx.WATER["rho"] * mx.WATER["c"] * v
        assert abs(p - ref) / ref <= 1.0e-3
        assert abs(p - ref) / ref <= 1.0e-12  # in fact exact to machine precision


def test_c2_all_air_limit_residual_below_0_1_percent():
    for v in (1.0, 6.0, 10.0, 14.0, 25.0):
        p = mx.impact_pressure(1.0, v, strict=False)
        ref = mx.AIR["rho"] * mx.AIR["c"] * v
        assert abs(p - ref) / ref <= 1.0e-3
        assert abs(p - ref) / ref <= 1.0e-12


def test_c2_wall_mismatch_recovers_rigid_limit():
    """The mismatch factor is 1 for an infinitely stiff wall and < 1 otherwise."""
    z_steel = mx.STEEL["rho"] * mx.STEEL["c"]
    f = mx.wall_mismatch_factor(0.0, mx.STEEL["rho"], mx.STEEL["c"])
    assert 0.96 < f < 1.0
    assert abs(f - z_steel / (z_steel + mx.acoustic_impedance(0.0))) <= 1.0e-15
    # a hundredfold stiffer wall is closer to rigid than steel
    assert mx.wall_mismatch_factor(0.0, 100 * mx.STEEL["rho"], mx.STEEL["c"]) > f


# --------------------------------------------------------------------------- #
# C3 -- PBM moment closure
# --------------------------------------------------------------------------- #
def _params():
    """Kernels of a declared aeration state (epsilon = 1 m^2/s^3, alpha = 1e-3).

    Deliberately taken from the module, not hand-typed: K_c ~ 3.6e-9 m^3/s and
    K_b ~ 0.27 1/s, i.e. a M0 time scale of a few seconds, which is what makes
    the RK4 reference resolvable."""
    k = pbm.kernel_parameters_from_aeration(1.0, 1.0e-3)
    return dict(k_coagulation=k.k_coagulation, k_breakage=k.k_breakage)


def _initial_moments():
    spec = pbm.cascade_spectrum(1.0, 1.0e-3)
    return (
        spec.number_density(),
        spec.volume_moment(1),
        spec.volume_moment(2),
        spec.volume_moment(3),
    )


def test_c3_closed_form_matches_rk4_reference():
    m0, m1, m2, m3 = _initial_moments()
    kw = _params()
    t_end = 20.0
    history = pbm.integrate_moments(m0, m1, m2, m3, t_end=t_end, steps=4000, **kw)
    worst = 0.0
    scales = (m0, m1, m2, m3)
    for t, numeric in history[::200]:
        closed = pbm.moments_closed_form(t, m0, m1, m2, m3, **kw)
        for a, b, s in zip(numeric, closed, scales):
            worst = max(worst, abs(a - b) / max(abs(b), 1e-6 * abs(s)))
    assert worst <= 1.0e-2, f"closed form vs RK4 max residual {worst:.3e}"
    assert worst <= 1.0e-6, f"the ODE scheme should be near machine accurate: {worst:.3e}"


def test_c3_moment_rates_equal_the_direct_double_sum():
    """Closure exactness: the closed moment rates reproduce the discrete
    collision/breakage integrals for an arbitrary (non self-similar) spectrum."""
    volumes = [1e-10, 3e-10, 8e-10, 2e-9, 6e-9, 1.5e-8, 4e-8]
    densities = [3e12, 4e11, 5e10, 6e9, 7e8, 8e7, 9e6]
    k_c, k_b = 5.0e-5, 0.2
    direct = pbm.discrete_moment_rates(volumes, densities, k_c, k_b)
    m0 = sum(densities)
    m1 = sum(n * v for n, v in zip(densities, volumes))
    m2 = sum(n * v * v for n, v in zip(densities, volumes))
    m3 = sum(n * v**3 for n, v in zip(densities, volumes))
    closed = pbm.moment_rates(m0, m1, m2, m3, k_c, k_b)
    # reference scale for the cancellation-free comparison
    scale = k_c * m0 * m1 + k_b * m0
    for k, (a, b) in enumerate(zip(direct, closed)):
        assert abs(a - b) <= 1.0e-9 * scale, (k, a, b)


def test_c3_m0_decays_m1_conserved_m2_relaxes_exponentially():
    m0, m1, m2, m3 = _initial_moments()
    kw = _params()
    t_end = 100.0
    trajectory = pbm.integrate_moments(m0, m1, m2, m3, t_end=t_end, steps=4000, **kw)
    assert trajectory[-1][1][1] == pytest.approx(m1, rel=1e-12)  # M1 conserved
    assert trajectory[-1][1][0] < m0  # M0 decays
    m0_eq = 2.0 * kw["k_breakage"] / kw["k_coagulation"]
    assert trajectory[-1][1][0] == pytest.approx(m0_eq, rel=1e-3)
    # M2 relaxes with the single rate K_b/2 towards 2 K_c M1^2 / K_b
    m2_inf = 2.0 * kw["k_coagulation"] * m1**2 / kw["k_breakage"]
    t1, t2 = 2.0, 4.0
    y1 = pbm.moments_closed_form(t1, m0, m1, m2, m3, **kw)[2]
    y2 = pbm.moments_closed_form(t2, m0, m1, m2, m3, **kw)[2]
    rate = math.log((y1 - m2_inf) / (y2 - m2_inf)) / (t2 - t1)
    assert rate == pytest.approx(0.5 * kw["k_breakage"], rel=1e-9)
    assert y2 != y1


def test_c3_m2_grows_exponentially_when_started_below_equilibrium():
    """The campaign sheet's "M2 exponential growth" branch is the other sign:
    start below the coagulation/breakage equilibrium and M2 grows with rate
    K_b/2.  Pinned so the sign convention is not a matter of interpretation."""
    kw = _params()
    m1 = 1.0e-3
    m2_inf = 2.0 * kw["k_coagulation"] * m1**2 / kw["k_breakage"]
    m2_0 = 0.5 * m2_inf
    m0, _, _, m3 = _initial_moments()
    y1 = pbm.moments_closed_form(1.0, m0, m1, m2_0, m3, **kw)[2]
    y2 = pbm.moments_closed_form(3.0, m0, m1, m2_0, m3, **kw)[2]
    assert y2 > y1 > m2_0
    rate = math.log((y1 - m2_inf) / (y2 - m2_inf)) / 2.0
    assert rate == pytest.approx(0.5 * kw["k_breakage"], rel=1e-9)


def test_c3_kernels_follow_from_the_aeration_state_not_from_fitting():
    """Kernels are functions of (epsilon, alpha) only; doubling epsilon must move
    K_c as sqrt(epsilon) (Saffman-Turner) and must not change with the band."""
    k1 = pbm.kernel_parameters_from_aeration(1.0, 1.0e-3)
    k4 = pbm.kernel_parameters_from_aeration(4.0, 1.0e-3)
    # K_c ~ sqrt(epsilon) at a *fixed* spectrum (Saffman-Turner); the full
    # epsilon-dependence also moves the Hinze scale, so the ratio is not exactly
    # 2 -- it must merely be monotone and between 1 and 2
    assert 1.0 < k4.k_coagulation / k1.k_coagulation < 2.0
    assert pbm.saffman_turner_kernel(4.0, k1.spectrum) / pbm.saffman_turner_kernel(
        1.0, k1.spectrum
    ) == pytest.approx(2.0, rel=1e-12)
    assert k1.spectrum.volume_moment(1) == pytest.approx(1.0e-3, rel=1e-12)
    assert 1e-12 < k1.k_coagulation < 1e-3
    assert 1e-6 < k1.k_breakage < 1e3
    # the void fraction is carried by the spectrum amplitude, not by guesswork
    spec = pbm.cascade_spectrum(1.0, 3.0e-4)
    assert spec.volume_moment(1) == pytest.approx(3.0e-4, rel=1e-12)


# --------------------------------------------------------------------------- #
# C4 -- alpha from the PBM vs the specified field
# --------------------------------------------------------------------------- #
def test_c4_pbm_void_fraction_matches_the_specified_field():
    alpha_s, scale = 2.0e-3, 0.05

    def profile(z):
        return alpha_s * math.exp(-z / scale)

    out = law.footprint_void_fraction(profile, depth=0.4, cells=400)
    assert out["deviation"] <= 0.05, out
    assert out["deviation"] <= 1.0e-12, out


def test_c4_discrete_population_reproduces_the_field():
    def profile(z):
        return 1.0e-3 * math.exp(-z / 0.05)

    out = law.footprint_void_fraction_discrete(profile, depth=0.3, cells=300)
    assert out["deviation"] <= 0.05, out


# --------------------------------------------------------------------------- #
# C5 -- case set: legacy constant vs the Z(alpha) law
# --------------------------------------------------------------------------- #
def test_c5_case_set_envelope_inside_the_m9319_band():
    results = law.case_set()
    envelope = law.design_envelope(results)
    assert envelope["n_cases"] == len(law.SLAM_CASES)
    # design envelope (P95) of the footprint load reduction inside [6 %, 12 %]
    assert law.BAND_LOW <= envelope["g_load_p95"] <= law.BAND_HIGH, envelope
    # every case must be accounted for (in-domain or explicitly gated)
    for r in results:
        assert r["local"]["status"] in ("IN_DOMAIN", "OUT_OF_DOMAIN")
        assert 0.0 <= r["g_load"] <= 1.0
        assert r["k_b_equivalent_load"] == pytest.approx(1.0 - r["g_load"], rel=1e-12)


def test_c5_legacy_constant_is_not_alpha_consistent():
    """Decalibration: the single constant k_b = 0.65 hides a *fixed* void
    fraction, so across the case set it is neither conservative nor uniform."""
    results = law.case_set()
    g_load = [r["g_load"] for r in results]
    assert max(g_load) - min(g_load) > 0.05  # the law's spread >> a single factor
    table = law.decalibration_table()
    assert 1.0e-5 < table["legacy_equivalent_alpha"] < 1.0e-3
    # the implied void fraction reproduces the legacy coefficient exactly
    assert mx.equivalent_buffer_coefficient(table["legacy_equivalent_alpha"]) == pytest.approx(
        law.LEGACY_BUFFER, rel=1e-9
    )


# --------------------------------------------------------------------------- #
# C6 -- monotonicity and boundedness
# --------------------------------------------------------------------------- #
def test_c6_pressure_decreases_monotonically_on_the_domain():
    a_star = mx.alpha_at_min_sound_speed()
    previous = None
    alpha = mx.ALPHA_MIN_RESOLVABLE
    alphas = []
    while alpha < a_star * 0.999:
        alphas.append(alpha)
        alpha *= 1.15
    for v in (6.0, 10.0, 14.0):
        previous = None
        for a in alphas:
            p = mx.impact_pressure(a, v)
            if previous is not None:
                assert p < previous, (v, a)
            previous = p
    # boundedness: 0 < p(alpha) <= p(0) and Z above its minimum
    p_min = mx.impact_pressure(a_star * 0.99, 10.0)
    assert 0.0 < p_min <= mx.impact_pressure(0.0, 10.0, strict=False)
    assert mx.acoustic_impedance(0.5) > mx.acoustic_impedance(a_star)


def test_c6_equivalent_buffer_coefficient_monotone_and_bounded():
    ks = [mx.equivalent_buffer_coefficient(a) for a in (1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.4)]
    for left, right in zip(ks, ks[1:]):
        assert right < left
    assert all(0.0 < k <= 1.0 for k in ks)


# --------------------------------------------------------------------------- #
# C7 -- fail closed
# --------------------------------------------------------------------------- #
def test_c7_above_alpha_star_is_out_of_domain():
    a_star = mx.alpha_at_min_sound_speed()
    for alpha in (a_star * (1.0 + 1e-9), a_star * 1.001, 0.75, 1.0, 3.0):
        status = mx.domain_status(alpha)
        assert status["status"] == "OUT_OF_DOMAIN"
        with pytest.raises(mx.OutOfDomain):
            mx.impact_pressure(alpha, 10.0)
        local = law.local_impact(alpha, 10.0)
        assert local["status"] == "OUT_OF_DOMAIN"
        assert local["p"] is None and local["g_peak"] is None
        assert local["bound_p"] is not None and local["bound_p"] < local["p_clear"]


def test_c7_below_resolution_floor_is_out_of_domain():
    for alpha in (0.0, 1e-12, -1.0):
        assert mx.domain_status(alpha)["status"] == "OUT_OF_DOMAIN"
        with pytest.raises((mx.OutOfDomain, ValueError)):
            mx.impact_pressure(alpha, 10.0)


def test_c7_van_wijngaarden_fails_closed_outside_its_dilute_domain():
    with pytest.raises(mx.OutOfDomain):
        mx.van_wijngaarden_sound_speed(mx.ALPHA_VW_MAX * 2)
    # and it is *not* used as a stand-in for Wood at large alpha: at alpha=1 the
    # dilute limit would return 11.9 m/s instead of the gas sound speed
    with pytest.raises(mx.OutOfDomain):
        mx.van_wijngaarden_sound_speed(1.0)


# --------------------------------------------------------------------------- #
# coupling / plumbing
# --------------------------------------------------------------------------- #
def test_impact_law_exports_through_the_aeration_package():
    from tensorsph.aeration import (  # noqa: F401
        OutOfDomain,
        impact_pressure,
        wood_sound_speed,
        alpha_at_min_sound_speed,
        alpha_at_time,
        case_load,
        case_set,
        design_envelope,
        moment_history,
    )


def test_alpha_at_time_follows_the_population_balance_relaxation():
    state = law.aeration_state(law.SLAM_CASES[1])
    tau = state["relaxation_time_s"]
    a_eq = state["alpha_patch"]
    assert law.alpha_at_time(state, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert law.alpha_at_time(state, tau) == pytest.approx(a_eq * (1.0 - math.exp(-1.0)), rel=1e-12)
    assert law.alpha_at_time(state, 10 * tau) == pytest.approx(a_eq, rel=1e-3)


def test_moment_history_starts_at_the_spectrum_moments_and_conserves_m1():
    state = law.aeration_state(law.SLAM_CASES[0])
    hist = law.moment_history(state, t_end=5.0, steps=50)
    t0, m0_0, m1_0, m2_0, m3_0 = hist[0]
    assert t0 == 0.0
    expected_alpha = min(state["alpha_patch"], law.ALPHA_PACKING)
    assert m1_0 == pytest.approx(expected_alpha, rel=1e-9)
    assert hist[-1][2] == pytest.approx(m1_0, rel=1e-12)  # M1 conserved
    assert hist[-1][1] < m0_0  # M0 decays under coagulation

def test_curve_family_is_bounded_by_the_clear_water_hammer():
    family = law.curve_family(velocities=(8.0,), n_alpha=20)
    p_clear = mx.WATER["rho"] * mx.WATER["c"] * 8.0
    for point in family["curves"][8.0]:
        assert 0.0 < point["p_peak"] <= p_clear
        assert 0.0 < point["impedance"]
        assert point["sound_speed"] >= mx.min_sound_speed()


def test_percentile_helper():
    assert law.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == pytest.approx(3.0)
    assert law.percentile([1.0], 0.95) == 1.0
    assert law.percentile(list(range(101)), 0.95) == pytest.approx(95.0)
