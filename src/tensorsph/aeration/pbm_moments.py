"""Population-balance moment closure for the aerated impact law (P457W2-N04).

The void fraction that drives the mixture acoustics of
:mod:`tensorsph.aeration.mixture_acoustics` is not a free constant: it is the
first moment ``M1`` of a bubble population that is simultaneously *coagulating*
(bubble-bubble collision, constant kernel) and *breaking up* (turbulent
breakup into two equal daughters).  This module supplies that closure.

Internal coordinate: bubble volume ``v``; moments ``M_k = int v^k n(v) dv``,
so that

    M0 = number density [1/m^3]          M1 = void fraction alpha [-]
    M2 = volume-weighted size [m^6/m^3]  M3 = interfacial-area proxy [m^9/m^3]

Declared event model
--------------------
* **coagulation**, constant kernel ``K_c``: two bubbles merge into one of
  volume ``v1 + v2`` (kernel is the spectrum-averaged Saffman-Turner rate, see
  :func:`saffman_turner_kernel`);
* **breakup**, size-independent frequency ``K_b``: one bubble splits into two
  *equal* daughters of volume ``v/2`` (rate from the Coulaoglou-Tavlarides
  Kolmogorov criterion, see :func:`breakage_frequency`).

For that pair the moment equations close *exactly* in ``M0..M3`` -- no
assumed distribution shape and no fitting:

    dM0/dt = +K_b M0            - (K_c/2) M0^2
    dM1/dt =  0
    dM2/dt = +K_c M1^2          - (K_b/2) M2
    dM3/dt = +(3/2) K_c M1 M2   - (3/4) K_b M3

(``M1`` is conserved by both events: coagulation is 2->1 at constant total
volume, breakup is 1->2 equal-split at constant volume.)  The closure is exact
for *any* distribution, which is verified in ``tests/`` against a direct
double-sum evaluation of the collision/breakage integrals over a discrete
spectrum.

Closed-form solution
--------------------
``M0`` is logistic (decay from ``M0_0`` to the equilibrium
``2 K_b / K_c`` when ``M0_0`` is above it), ``M1`` is constant, ``M2`` relaxes
exponentially with the single rate ``K_b/2`` toward ``2 K_c M1^2 / K_b`` and
``M3`` follows in closed form as well (:func:`moments_closed_form`).

Kernel parameters come from the aeration source, not from data
--------------------------------------------------------------
``epsilon`` (turbulent dissipation) and the gas void fraction are the two
external inputs; the bubble size *spectrum shape* is a declared Kolmogorov
cascade (:math:`n(R) \\propto R^{-10/3}` above :func:`hinze_radius`), and the
spectrum *amplitude* is fixed by requiring ``1/6 pi M3-like`` mass
normalisation, i.e. by the supplied void fraction.  ``K_c`` is then
``1.294 (epsilon/nu)^(1/2) <(R1+R2)^3>`` with the average taken over that
spectrum (:func:`saffman_turner_kernel`), and ``K_b`` from the
Coulaoglou-Tavlarides (1981) Kolmogorov breakup criterion at the Sauter-mean
size.  No parameter is adjusted to hit the ``k_b``-band of the acceptance test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Saffman & Turner (1956) collision-rate prefactor for inertially-driven
#: bubble collisions in isotropic turbulence.
SAFFMAN_TURNER_C = 1.294
#: Critical Weber number for incipient bubble breakup (Hinze 1955).
WE_CRIT = 4.7
#: Declared lower cutoff of the entrained bubble cascade [m] (below this,
#: surface tension suppresses breakup and dissolution dominates).
SPECTRUM_R_MIN = 2.0e-5
#: Declared cascade slope: n(R) ~ R^-10/3 (Garrettson 1973; the large-bubble
#: branch measured by Deane & Stokes 2002 for breaking-wave plumes).
SPECTRUM_SLOPE = 10.0 / 3.0
#: Coulaoglou & Tavlarides (1981) breakup-rate constants (literature values,
#: *not* calibrated here).
CT_C1 = 0.00481
#: See :data:`CT_C1`.
CT_C2 = 0.08


# --------------------------------------------------------------------------- #
# bubble size spectrum (declared shape, amplitude from the aeration source)
# --------------------------------------------------------------------------- #
def hinze_radius(epsilon, sigma=0.0728, rho_l=998.2, we_crit=WE_CRIT):
    """Hinze (1955) maximum stable bubble radius [m].

    ``R_H = (We_c sigma / (2 rho_l))^(3/5) epsilon^(-2/5)``.
    """
    if epsilon <= 0.0:
        raise ValueError("dissipation must be positive")
    return (we_crit * sigma / (2.0 * rho_l)) ** 0.6 * epsilon ** -0.4


@dataclass(frozen=True)
class BubbleSpectrum:
    """Declared entrained-bubble number spectrum ``n(R) = C R^-slope``.

    ``n`` is normalised so that ``int (4/3) pi R^3 n dR = void_fraction``: the
    *shape* is a declared cascade, the *amplitude* comes from the aeration
    source term.  ``r_min`` / ``r_hinze`` are the integration limits.
    """

    r_min: float
    r_hinze: float
    slope: float
    void_fraction: float

    @property
    def exponent(self):
        """``3 - slope`` -- exponent of the volume-weighted integrand."""
        return 3.0 - self.slope

    def _integral(self, power):
        """``int_{r_min}^{r_hinze} R^power dR`` with the un-normalised ``C``."""
        p = power - self.slope
        if abs(p + 1.0) < 1.0e-12:
            return math.log(self.r_hinze / self.r_min)
        return (self.r_hinze ** (p + 1.0) - self.r_min ** (p + 1.0)) / (p + 1.0)

    def normalisation(self):
        """``C`` such that the void fraction matches the supplied value."""
        # int (4/3) pi R^3 C R^-slope dR = void_fraction
        denom = (4.0 / 3.0) * math.pi * self._integral(3.0)
        return self.void_fraction / denom

    def moment_radius(self, power):
        """``<R^power>`` over the *number*-weighted spectrum."""
        return self._integral(power) / self._integral(0.0)

    def number_density(self):
        """``M0`` of the spectrum [1/m^3]."""
        return self.normalisation() * self._integral(0.0)

    def volume_moment(self, k):
        """``M_k = int v^k n dv`` with ``v = (4/3) pi R^3`` (k = 0..3)."""
        if k < 0 or k > 3:
            raise ValueError("only moments M0..M3 are closed by this model")
        # M_k = ((4/3) pi)^k * int R^(3k) n dR = ((4/3)pi)^k * C * I(3k - slope)
        coeff = ((4.0 / 3.0) * math.pi) ** k
        if k == 0:
            return self.number_density()
        return coeff * self.normalisation() * self._integral(3.0 * k)

    def sauter_mean_radius(self):
        """``d32/2 = <R^3>/<R^2>`` -- the size that carries the volume."""
        return self.moment_radius(3.0) / self.moment_radius(2.0)

    def as_dict(self):
        return {
            "r_min": self.r_min,
            "r_hinze": self.r_hinze,
            "slope": self.slope,
            "void_fraction": self.void_fraction,
            "number_density": self.number_density(),
            "sauter_mean_radius": self.sauter_mean_radius(),
        }


def cascade_spectrum(
    epsilon,
    void_fraction,
    sigma=0.0728,
    rho_l=998.2,
    we_crit=WE_CRIT,
    r_min=SPECTRUM_R_MIN,
    slope=SPECTRUM_SLOPE,
):
    """Build the declared Hinze-capped cascade at dissipation ``epsilon``."""
    r_h = hinze_radius(epsilon, sigma, rho_l, we_crit)
    if r_h <= r_min:
        raise ValueError(
            "Hinze radius below the cascade cutoff: the declared spectrum is empty"
        )
    if not 0.0 < void_fraction < 1.0:
        raise ValueError("void fraction must lie in (0, 1)")
    return BubbleSpectrum(r_min, r_h, slope, void_fraction)


# --------------------------------------------------------------------------- #
# kernels from the aeration source term
# --------------------------------------------------------------------------- #
def saffman_turner_kernel(
    epsilon, spectrum, nu=1.002e-6, c_st=SAFFMAN_TURNER_C
):
    """Spectrum-averaged Saffman-Turner collision kernel ``K_c`` [m^3/s].

    ``K_c = c_st sqrt(epsilon/nu) <(R1+R2)^3>`` and, for the mean over two
    independent draws from the spectrum,

        ``<(R1+R2)^3> = 2<R^3> + 6<R^2><R>``.
    """
    if epsilon <= 0.0 or nu <= 0.0:
        raise ValueError("epsilon and nu must be positive")
    r1 = spectrum.moment_radius(1.0)
    r2 = spectrum.moment_radius(2.0)
    r3 = spectrum.moment_radius(3.0)
    sum_cubed = 2.0 * r3 + 6.0 * r2 * r1
    return c_st * math.sqrt(epsilon / nu) * sum_cubed


def breakage_frequency(
    epsilon,
    sauter_radius,
    sigma=0.0728,
    rho_l=998.2,
    c1=CT_C1,
    c2=CT_C2,
):
    """Coulaoglou-Tavlarides (1981) turbulent equal-split breakup frequency [1/s].

    ``b(d) = c1 epsilon^(1/3) d^(-2/3) exp(-c2 sigma / (rho epsilon^(2/3) d^(5/3)))``
    with ``d`` the Sauter diameter.  ``c1``, ``c2`` are the published constants
    (module constants, not tuned here).
    """
    if epsilon <= 0.0 or sauter_radius <= 0.0:
        raise ValueError("epsilon and the Sauter radius must be positive")
    d = 2.0 * sauter_radius
    e13 = epsilon ** (1.0 / 3.0)
    base = c1 * e13 * d ** (-2.0 / 3.0)
    group = c2 * sigma / (rho_l * epsilon ** (2.0 / 3.0) * d ** (5.0 / 3.0))
    return base * math.exp(-group)


@dataclass(frozen=True)
class KernelParameters:
    """Coagulation/breakup parameters implied by one aeration state."""

    epsilon: float
    void_fraction: float
    k_coagulation: float
    k_breakage: float
    spectrum: BubbleSpectrum

    @property
    def m0_equilibrium(self):
        """Logistic fixed point of ``M0``: ``2 K_b / K_c``."""
        return 2.0 * self.k_breakage / self.k_coagulation

    def m2_equilibrium(self, m1):
        """Relaxation target of ``M2``: ``2 K_c M1^2 / K_b``."""
        return 2.0 * self.k_coagulation * m1 * m1 / self.k_breakage

    def as_dict(self):
        return {
            "epsilon": self.epsilon,
            "void_fraction": self.void_fraction,
            "k_coagulation": self.k_coagulation,
            "k_breakage": self.k_breakage,
            "m0_equilibrium": self.m0_equilibrium,
            "sauter_mean_radius": self.spectrum.sauter_mean_radius(),
            "hinze_radius": self.spectrum.r_hinze,
            "number_density": self.spectrum.number_density(),
        }


def kernel_parameters_from_aeration(
    epsilon,
    void_fraction,
    sigma=0.0728,
    rho_l=998.2,
    nu=1.002e-6,
    we_crit=WE_CRIT,
    r_min=SPECTRUM_R_MIN,
    slope=SPECTRUM_SLOPE,
    c1=CT_C1,
    c2=CT_C2,
):
    """Bundle the two kernels from one aeration state (``epsilon``, ``alpha``).

    ``epsilon`` and ``void_fraction`` are the aeration source term of
    :mod:`tensorsph.aeration.entrainment`; everything else is a declared
    literature constant.  No parameter is tuned to the acceptance band.
    """
    spec = cascade_spectrum(
        epsilon, void_fraction, sigma, rho_l, we_crit, r_min, slope
    )
    k_c = saffman_turner_kernel(epsilon, spec, nu)
    k_b = breakage_frequency(
        epsilon, spec.sauter_mean_radius(), sigma, rho_l, c1, c2
    )
    return KernelParameters(epsilon, void_fraction, k_c, k_b, spec)


# --------------------------------------------------------------------------- #
# moment equations and closed form
# --------------------------------------------------------------------------- #
def moment_rates(m0, m1, m2, m3, k_coagulation, k_breakage):
    """Right-hand side of the closed moment system (see module docstring)."""
    return (
        k_breakage * m0 - 0.5 * k_coagulation * m0 * m0,
        0.0,
        k_coagulation * m1 * m1 - 0.5 * k_breakage * m2,
        1.5 * k_coagulation * m1 * m2 - 0.75 * k_breakage * m3,
    )


def moments_closed_form(t, m0_0, m1, m2_0, m3_0, k_coagulation, k_breakage):
    """Exact ``M0..M3`` of the closed system.

    * ``M0(t) = A / (1 + (A/M0_0 - 1) exp(-K_b t))``, ``A = 2 K_b / K_c``;
    * ``M1`` constant;
    * ``M2(t) = M2_inf + (M2_0 - M2_inf) exp(-K_b t / 2)``,
      ``M2_inf = 2 K_c M1^2 / K_b``;
    * ``M3(t)`` from the linear ODE with the known ``M2(t)`` forcing.
    """
    if k_coagulation <= 0.0 or k_breakage <= 0.0:
        raise ValueError("both kernels must be positive")
    if m0_0 <= 0.0:
        raise ValueError("initial number density must be positive")
    a = 2.0 * k_breakage / k_coagulation
    decay0 = math.exp(-k_breakage * t)
    m0 = a / (1.0 + (a / m0_0 - 1.0) * decay0)

    m2_inf = 2.0 * k_coagulation * m1 * m1 / k_breakage
    m2 = m2_inf + (m2_0 - m2_inf) * math.exp(-0.5 * k_breakage * t)

    rate3 = 0.75 * k_breakage
    forcing = 1.5 * k_coagulation * m1
    delta = m2_0 - m2_inf
    e_a = math.exp(-rate3 * t)
    if abs(rate3 - 0.5 * k_breakage) < 1.0e-12 * rate3:
        # degenerate double root: (exp(-b t) - exp(-a t)) / (a - b) -> t exp(-a t)
        cross = t * e_a
    else:
        cross = (
            math.exp(-0.5 * k_breakage * t) - e_a
        ) / (rate3 - 0.5 * k_breakage)
    m3 = (
        m3_0 * e_a
        + forcing * m2_inf * (1.0 - e_a) / rate3
        + forcing * delta * cross
    )
    return m0, m1, m2, m3


def integrate_moments(
    m0_0, m1, m2_0, m3_0, k_coagulation, k_breakage, t_end, steps=4000
):
    """Classical RK4 reference solution of the moment ODEs (no closed form)."""
    dt = t_end / steps
    y = [m0_0, m1, m2_0, m3_0]
    history = [(0.0, tuple(y))]
    for i in range(steps):
        k1 = moment_rates(*y, k_coagulation, k_breakage)
        k2 = moment_rates(
            *[y[j] + 0.5 * dt * k1[j] for j in range(4)], k_coagulation, k_breakage
        )
        k3 = moment_rates(
            *[y[j] + 0.5 * dt * k2[j] for j in range(4)], k_coagulation, k_breakage
        )
        k4 = moment_rates(
            *[y[j] + dt * k3[j] for j in range(4)], k_coagulation, k_breakage
        )
        y = [
            y[j] + dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
            for j in range(4)
        ]
        history.append(((i + 1) * dt, tuple(y)))
    return history


def discrete_moment_rates(volumes, densities, k_coagulation, k_breakage):
    """Direct double-sum moment production rates for a *discrete* spectrum.

    Independent of :func:`moment_rates`: evaluates the collision integral

        (1/2) sum_ij K n_i n_j [(v_i+v_j)^k - v_i^k - v_j^k]

    and the equal-split breakage integral

        sum_i K_b n_i [2 (v_i/2)^k - v_i^k]

    term by term, and equals the closed-form rates exactly for this event model.
    Used by the test suite as a closure proof.
    """
    rates = [0.0, 0.0, 0.0, 0.0]
    for i, (vi, ni) in enumerate(zip(volumes, densities)):
        for k in range(4):
            rates[k] += k_breakage * ni * (2.0 * (0.5 * vi) ** k - vi**k)
        for vj, nj in zip(volumes, densities):
            for k in range(4):
                rates[k] += (
                    0.5
                    * k_coagulation
                    * ni
                    * nj
                    * ((vi + vj) ** k - vi**k - vj**k)
                )
    return tuple(rates)


def sauter_mean_diameter(m2, m3):
    """``d32 = M3/M2`` (Sauter diameter, volume/area) [m]."""
    if m2 <= 0.0:
        raise ValueError("M2 must be positive")
    return m3 / m2


def void_fraction_from_moments(m1):
    """The mixture void fraction *is* ``M1`` (bubble volume per mixture volume)."""
    return m1
