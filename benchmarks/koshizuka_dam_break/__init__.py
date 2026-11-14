"""Koshizuka & Oka (1996) dam-break pressure peak benchmark.

Reference:
    Koshizuka, S. & Oka, Y. (1996). Moving-particle semi-implicit method for
    fragmentation of incompressible fluid. *Nuclear Science and Engineering*,
    123(3), 421-434. DOI: 10.13182/NSE96-A24205.

Geometry (all lengths in metres, tank frame, floor at y = 0):
    tank length:       0.584
    tank height:       0.14  (or 0.2 in some variants; we use 0.14 per Fig. 10)
    column width (a):  0.14
    column height:     0.28  (2a)
    probe location:    x = 0.584 m (right wall), y = 0.024 m from floor

Reference pressure peak (from Koshizuka & Oka 1996, Fig. 15):
    P_max ≈ 2500–3000 Pa at the probe (y = 0.024 m)
    The exact value depends on resolution and probe averaging; we use
    2700 Pa as the central reference with ±12% tolerance per the task spec.

Scope:
    * This benchmark measures the wall pressure time history at a probe on the
      right wall and extracts the peak pressure.
    * The reference value is taken from Koshizuka & Oka (1996) Figure 15.
    * Tolerance: <12% error vs reference.
"""
