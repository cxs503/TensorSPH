"""Linear potential-flow solution for a forced rectangular tank (2-D).

This is the reference the sloshing rung is measured against.  It is the exact
solution of the *linearised* initial/boundary value problem, evaluated in
closed form per sloshing mode, so it carries no fitted constant and no
iteration:

Tank ``0 <= x <= L``, still-water level ``y = H`` (the floor is ``y = 0``),
walls at rest in the tank frame, apparent horizontal body acceleration
``a_x(t) = -A omega^2 sin(omega t)`` (i.e. the tank is shaken with
``x_g = A sin omega t``).

With the velocity potential expanded as
``phi = sum_n q_n(t) cos(k_n x) cosh(k_n (y' + H)) / cosh(k_n H)``,
``y' = y - H`` and ``k_n = n pi / L``, the free-surface and wall conditions give
one forced oscillator per mode (n >= 1):

    q_n'' + omega_n^2 q_n = A omega^3 c_n cos(omega t),
    omega_n^2 = g k_n tanh(k_n H),
    c_n = 2 L ((-1)^n - 1) / (n^2 pi^2),

whose zero-initial-condition solution is closed form.  Free-surface elevation
and pressure follow from mass conservation (``int eta dx = 0``) and from the
linearised momentum equation:

    eta(x,t) = -[ sum_n q_n'(t) cos(k_n x) + a_x(t) (x - L/2) ] / g
    p(x,y,t) = rho g (eta(x,t) - y')
               - rho sum_n q_n'(t) cos(k_n x)
                 [ cosh(k_n (y' + H)) / cosh(k_n H) - 1 ]

Known limits that the tests pin: for ``omega -> 0`` the elevation reduces to the
rigid-body tilt ``-a_x (x - L/2) / g``; the first peak of the resonance curve is
at ``omega_1``; the pressure reduces to ``rho g (eta - y')`` at the wall for
long waves.

**Validity.** Linear, inviscid, irrotational, small amplitude, no wave breaking,
no damping.  Close to ``omega_1`` the response grows without bound (a real tank
is limited by damping and by nonlinearity), so the reference is quoted only with
that caveat -- see ``docs/benchmarks/wcsph-chain.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LinearSloshing:
    length: float = 1.0
    depth: float = 0.5
    amplitude: float = 0.01
    frequency: float = 1.0
    gravity: float = 9.81
    modes: int = 40
    rho0: float = 1000.0

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.frequency

    def omega_n(self, n: int) -> float:
        k = n * math.pi / self.length
        return math.sqrt(self.gravity * k * math.tanh(k * self.depth))

    @property
    def omega1(self) -> float:
        return self.omega_n(1)

    def acceleration(self, t: float) -> float:
        """Apparent body acceleration of the fluid in the tank frame."""
        return -self.amplitude * self.omega ** 2 * math.sin(self.omega * t)

    def _mode(self, n: int, t: float) -> tuple[float, float]:
        """``(q_n, q_n')`` for mode ``n`` at time ``t``."""
        omega = self.omega
        wn = self.omega_n(n)
        k = n * math.pi / self.length
        c = 2.0 * self.length * ((-1.0) ** n - 1.0) / (n * n * math.pi ** 2)
        force = self.amplitude * omega ** 3 * c
        det = wn * wn - omega * omega
        if abs(det) <= 1.0e-9 * wn * wn:
            # resonant limit (zero damping): q = F t sin(wt) / (2 w)
            q = force * t * math.sin(omega * t) / (2.0 * omega)
            qdot = force * (math.sin(omega * t)
                            + omega * t * math.cos(omega * t)) / (2.0 * omega)
            return q, qdot
        q = force / det * (math.cos(omega * t) - math.cos(wn * t))
        qdot = force / det * (-omega * math.sin(omega * t)
                              + wn * math.sin(wn * t))
        del k
        return q, qdot

    def mode_sum(self, t: float) -> list[tuple[int, float, float]]:
        """``[(n, k_n, q_n'(t))]`` -- everything eta/pressure need."""
        return [(n, n * math.pi / self.length, self._mode(n, t)[1])
                for n in range(1, self.modes + 1)]

    def elevation(self, x: float, t: float) -> float:
        """Free-surface elevation above the still-water level ``y = H``."""
        total = 0.0
        for _, k, qdot in self.mode_sum(t):
            total += qdot * math.cos(k * x)
        total += self.acceleration(t) * (x - 0.5 * self.length)
        return -total / self.gravity

    def pressure(self, x: float, y: float, t: float) -> float:
        """Absolute pressure at ``(x, y)``; ``y`` is measured from the floor."""
        yp = y - self.depth
        eta = self.elevation(x, t)
        dynamic = 0.0
        for _, k, qdot in self.mode_sum(t):
            shape = (math.cosh(k * (yp + self.depth))
                     / math.cosh(k * self.depth) - 1.0)
            dynamic += qdot * math.cos(k * x) * shape
        return self.rho0 * (self.gravity * (eta - yp) - dynamic)

    def wall_pressure(self, y: float, t: float) -> float:
        return self.pressure(self.length, y, t)

    def wall_elevation(self, t: float) -> float:
        return self.elevation(self.length, t)

    def response_amplitude(self, periods: float = 1.0) -> float:
        """Rigid-tilt-plus-first-mode amplitude estimate (diagnostic only)."""
        t_end = periods * 2.0 * math.pi / self.omega
        samples = [self.wall_elevation(t_end * k / 400.0) for k in range(401)]
        return 0.5 * (max(samples) - min(samples))


__all__ = ["LinearSloshing"]