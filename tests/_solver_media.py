"""
Media, geometries and small predicates shared by the solver test modules.

``tests/test_cylindrical_solver.py`` grew to 26,376 lines and 741 test
cases before it was split into the six ``tests/test_solver_*.py``
modules. Almost everything in it was already local to one section --
the split found only eighteen names used across the seams, which is
what made the seams real rather than arbitrary.

Those eighteen live here. Two of the modules referred to each other's
definitions in both directions (``test_solver_branches`` used the
Schmitt & Cheng stack defined among the figure tests, and
``test_solver_figures`` used the A.2 casing and cement defined among
the branch tests), so importing across test modules would have been a
cycle. A shared module is the way out, and it is small on purpose:
anything used by one module belongs in that module.
"""

from __future__ import annotations

import numpy as np

from fwap.cylindrical_solver import BoreholeLayer

# ----------------------------------------------------------------------
# The slow formation the n=1, VTI and cased sections all build on
# ----------------------------------------------------------------------

SLOW_VP = 2200.0
SLOW_VS = 800.0
SLOW_RHO = 2200.0
SLOW_VF = 1500.0
SLOW_RHO_F = 1000.0
SLOW_A = 0.1


def _stoneley_lf_truth(vs, rho, vf, rho_f):
    """White (1983) eq. 5.42 closed form: S_ST^2 = 1/V_f^2 + rho_f/mu."""
    mu = rho * vs**2
    return float(np.sqrt(1.0 / vf**2 + rho_f / mu))


# ----------------------------------------------------------------------
# Roadmap A.2: the cased geometry the coverage work was measured on
# ----------------------------------------------------------------------

_A2_CASING = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
_A2_CEMENT = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)


def _gap_sweep_layers(ratio: float, vs: float = 800.0) -> tuple[BoreholeLayer, ...]:
    """One annulus of the stiffness sweep A.9's gap was recorded on."""
    vs_layer = ratio * vs
    return (BoreholeLayer(vp=2.2 * vs_layer, vs=vs_layer, rho=2000.0, thickness=0.04),)


# ----------------------------------------------------------------------
# Sinha's published configurations, and the quadrupole's fast formation
# ----------------------------------------------------------------------

_LC_SLOW = dict(vp=1890.0, vs=508.0, rho=2054.0, vf=1500.0, rho_f=1000.0, a=0.1016)
_PR_SINHA_FAST = dict(
    vp=3658.0, vs=2032.0, rho=2350.0, vf=1500.0, rho_f=1000.0, a=0.1016
)
_QUAD_FAST = dict(vp=4500.0, vs=2600.0, rho=2400.0)
_QUAD_FLUID = dict(vf=1500.0, rho_f=1000.0)


# ----------------------------------------------------------------------
# The marcher's descent predicate
# ----------------------------------------------------------------------

_MARCHER_STEP_UP_SLACK = 1.0e-3


def _descends(velocity: np.ndarray) -> bool:
    """True if ``velocity`` never rises by more than the marcher's slack."""
    if velocity.size < 2:
        return True
    return bool(np.all(np.diff(velocity) <= velocity[:-1] * _MARCHER_STEP_UP_SLACK))


# ----------------------------------------------------------------------
# Schmitt & Cheng (1987) figs 20 / 21: the cased hole
# ----------------------------------------------------------------------

_SC87_SLOW = dict(vp=2751.0, vs=1201.0, rho=2100.0, vf=1500.0, rho_f=1000.0)
_SC87_CASING = BoreholeLayer(vp=6098.0, vs=3354.0, rho=7500.0, thickness=0.0102)
_SC87_HOLE = 0.10


def _sc87_cement(vs: float, rho: float, thickness: float) -> BoreholeLayer:
    return BoreholeLayer(vp=2823.0, vs=vs, rho=rho, thickness=thickness)


def _sc87_stack(vs: float, rho: float, thickness: float):
    """``(radius, layers)`` for one cement, inside a 10 cm original hole."""
    radius = _SC87_HOLE - _SC87_CASING.thickness - thickness
    return radius, (_SC87_CASING, _sc87_cement(vs, rho, thickness))


# ----------------------------------------------------------------------
# Thomsen (1986) table 1
# ----------------------------------------------------------------------


def _green_river_shale_stiffness() -> dict[str, float]:
    """Thomsen (1986) table 1 Green River shale -> the five VTI
    stiffnesses. ``V_Sv`` = 1768 m/s against a 1500 m/s fluid, so this
    is the fast-formation TI regime; ``V_Sh`` = 2062 m/s."""
    vp0, vs0, rho = 3292.0, 1768.0, 2075.0
    eps, delta, gamma = 0.195, -0.220, 0.180
    c33, c44 = rho * vp0**2, rho * vs0**2
    c13 = np.sqrt(2 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2) - c44
    return dict(
        c11=c33 * (1.0 + 2.0 * eps),
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c44 * (1.0 + 2.0 * gamma),
        rho=rho,
    )
