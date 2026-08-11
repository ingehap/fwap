"""
Closed-form physics oracles for validating dispersion estimates.

These are analytic limits -- independent of the cylindrical-Biot modal
determinant solver -- against which a *trained* forward surrogate (and, here,
the vendored formulas themselves) can be checked. They are lifted from the
oracle assertions in the core repo's ``tests/test_cylindrical_solver.py`` so
sonic_ml can validate without importing test modules.

References
----------
* White, J. E. (1983). *Underground Sound*, eq. 5.42 (Stoneley low-frequency
  slowness).
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes*, ch. 4
  (flexural low-/high-frequency asymptotes).
"""

from __future__ import annotations

import numpy as np
from fwap.cylindrical import rayleigh_speed


def stoneley_lf_slowness(vs: float, rho: float, vf: float, rho_f: float) -> float:
    r"""
    Stoneley low-frequency phase slowness (White 1983, eq. 5.42).

    .. math::

        S_{ST}^2 = \frac{1}{V_f^2} + \frac{\rho_f}{\mu},
        \qquad \mu = \rho V_s^2

    The ``f -> 0`` limit of the Stoneley (tube-wave) slowness; the modal
    solver's ``stoneley_dispersion`` converges to this at ~10 Hz.

    Parameters
    ----------
    vs : float
        Formation shear velocity (m/s).
    rho : float
        Formation bulk density (kg/m^3).
    vf : float
        Borehole-fluid velocity (m/s).
    rho_f : float
        Borehole-fluid density (kg/m^3).

    Returns
    -------
    float
        Low-frequency Stoneley slowness (s/m).
    """
    mu = rho * vs**2
    return float(np.sqrt(1.0 / vf**2 + rho_f / mu))


def flexural_lf_slowness(vs: float) -> float:
    """
    Dipole flexural low-frequency phase slowness: ``1 / vs``.

    As ``f -> 0`` the flexural wavelength dwarfs the borehole and the mode
    propagates at the formation shear speed.

    Parameters
    ----------
    vs : float
        Formation shear velocity (m/s).

    Returns
    -------
    float
        ``1 / vs`` (s/m).
    """
    if vs <= 0.0:
        raise ValueError("vs must be positive")
    return 1.0 / vs


def flexural_hf_slowness(vp: float, vs: float) -> float:
    """
    Dipole flexural high-frequency phase slowness: ``1 / V_R(vp, vs)``.

    At high frequency the flexural mode localizes at the borehole wall and
    propagates as a surface wave, so this returns the (vacuum-loaded)
    Rayleigh slowness as a reference value.

    **It is not a bound on the mode.** The true high-frequency limit is the
    fluid-loaded Scholte speed, and the gap is not small: for a
    4000/2300 m/s formation against a 1500 m/s fluid, Scholte is 1470 m/s
    against a Rayleigh speed of 2116 -- about 30 %. A dipole flexural
    curve descends through ``V_R`` partway up the band and keeps going.
    Treating this value as a ceiling is what fwap's roadmap A.2 turned out
    to be, and it is why ``flexural_dispersion`` returned a trapped mode
    instead of the fundamental over most of the fast-formation band.

    Parameters
    ----------
    vp : float
        Formation compressional velocity (m/s), ``> vs``.
    vs : float
        Formation shear velocity (m/s).

    Returns
    -------
    float
        ``1 / rayleigh_speed(vp, vs)`` (s/m), strictly greater than ``1 / vs``.
    """
    return 1.0 / rayleigh_speed(vp, vs)
