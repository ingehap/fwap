"""
Elastic surface-wave speeds and cylindrical-mode dispersion models.

Part 3 of Mari et al. (1994) treats the borehole flexural mode with a
phenomenological dispersion law in :mod:`fwap.synthetic`
(:func:`dipole_flexural_dispersion`). That model's high-frequency
asymptote is ``1.25 * shear_slowness``, a round number chosen for
convenience rather than physics. This module provides a
Poisson-ratio-dependent replacement via the Rayleigh surface-wave
speed, which is the correct high-frequency limit for a vacuum-loaded
free surface.

For a fluid-filled borehole the true high-frequency limit is the
Scholte interface-wave speed at the fluid-solid boundary, which is a
few percent below the Rayleigh speed; the fluid-loading correction
depends on the density ratio and is usually small
(Paillet & Cheng, 1991, Ch. 4). A full cylindrical-Biot modal
determinant solver (Schmitt, 1988) is future work -- :mod:`fwap`
deliberately ships only models whose assumptions are explicit and
whose published limits match to within percent-level precision.

Functions
---------
* :func:`rayleigh_speed`
* :func:`flexural_dispersion_physical`

References
----------
* Rayleigh, Lord (1885). On waves propagating along the plane
  surface of an elastic solid. *Proceedings of the London
  Mathematical Society* 17, 4-11.
* Viktorov, I. A. (1967). *Rayleigh and Lamb Waves: Physical Theory
  and Applications.* Plenum (closed-form Rayleigh approximation).
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
  Boreholes*, Chapter 4. CRC Press (cylindrical-mode theory).
* Schmitt, D. P. (1988). Shear-wave logging in elastic formations.
  *Journal of the Acoustical Society of America* 84(6), 2230-2244.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import brentq


def rayleigh_speed(vp: float, vs: float) -> float:
    r"""
    Rayleigh surface-wave speed for a vacuum-loaded elastic half-space.

    Solves the classical Rayleigh equation

    .. math::

        \left(2 - \xi\right)^2
        \;=\;
        4 \sqrt{\left(1 - \xi \cdot (V_s/V_p)^2\right)\,(1 - \xi)}

    for :math:`\xi = (V_R / V_s)^2 \in (0, 1)` by bracketing the
    non-trivial root with :func:`scipy.optimize.brentq`. Returns
    :math:`V_R = V_s \sqrt{\xi}`.

    Typical values: for Poisson's ratio :math:`\nu = 0.25`
    (``Vp/Vs = sqrt(3)``) the ratio ``V_R / V_s`` is approximately
    0.9194; for :math:`\nu = 0.33` it is approximately 0.9325.

    Parameters
    ----------
    vp : float
        Compressional wave speed (m/s). Must satisfy ``vp > vs``.
    vs : float
        Shear wave speed (m/s). Must be positive.

    Returns
    -------
    float
        Rayleigh wave speed (m/s), always strictly less than ``vs``.

    Raises
    ------
    ValueError
        If ``vs <= 0`` or ``vp <= vs`` (physically invalid for an
        isotropic solid).

    References
    ----------
    * Rayleigh (1885), Proc. London Math. Soc. 17, 4-11.
    """
    if vs <= 0.0:
        raise ValueError("vs must be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")

    a2 = (vs / vp) ** 2

    def F(xi: float) -> float:
        return ((2.0 - xi) ** 2
                - 4.0 * np.sqrt((1.0 - xi * a2) * (1.0 - xi)))

    # F(0) = 0 (trivial root at zero speed). F dips negative and
    # crosses zero at the Rayleigh root somewhere in (~0.75, ~0.98)
    # for all physically reasonable Poisson's ratios. A bracket of
    # (0.1, 0.9999) reliably contains the non-trivial root.
    xi = brentq(F, 0.1, 0.9999, xtol=1.0e-12)
    return float(vs * np.sqrt(xi))


def flexural_dispersion_physical(vp: float,
                                 vs: float,
                                 a_borehole: float = 0.1,
                                 ) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Dipole flexural dispersion with a Rayleigh-speed high-frequency
    asymptote.

    Returns a callable ``s(f)`` that interpolates between the
    low-frequency limit (formation shear slowness ``1/vs``) and the
    high-frequency limit (the Rayleigh slowness ``1/V_R``, computed
    by :func:`rayleigh_speed` from ``vp`` and ``vs``) with the
    two-parameter form

    .. math::

        s(f) \;=\; s_\text{low} + (s_\text{high} - s_\text{low})
                 \, \frac{x^2}{1 + x^2}

    where :math:`x = f / f_c` and :math:`f_c = V_s / (2 \pi a)` is
    the geometric cut-off frequency. This is the same shape as the
    phenomenological :func:`fwap.synthetic.dipole_flexural_dispersion`
    but with a physically-grounded ``s_high`` that depends on the
    Poisson's ratio instead of the round-number ``1.25 / vs``.

    The interpolation form matches the smoothed-step model in
    Mari et al. (1994), Part 3, eq. (3.18); the Rayleigh asymptote
    and Scholte correction are derived from the cylindrical-mode
    dispersion relations of Paillet & Cheng (1991), Ch. 4, eqs.
    (4.27) and (4.31).

    Caveats
    -------
    The Rayleigh speed is the vacuum-loaded free-surface limit. For a
    fluid-filled borehole the correct asymptote is the *Scholte*
    speed at the fluid-solid interface, which is a few percent lower.
    The fluid-loading correction depends on the fluid-to-solid
    density ratio and the fluid speed; for typical brine-filled
    boreholes it reduces the high-frequency slowness by roughly
    ``1 - 3 %`` relative to the Rayleigh limit.

    For shear-slowness estimation accurate to better than a few
    percent in the high-frequency regime, the full cylindrical-Biot
    modal determinant (Schmitt, 1988; Paillet & Cheng, 1991) should
    be solved -- that remains future work in :mod:`fwap`.

    Parameters
    ----------
    vp : float
        Compressional wave speed of the formation (m/s).
    vs : float
        Shear wave speed of the formation (m/s).
    a_borehole : float, default 0.1
        Borehole radius (m).

    Returns
    -------
    Callable[[ndarray], ndarray]
        Array-in / array-out mapping of frequency (Hz) to phase
        slowness (s/m).

    See Also
    --------
    fwap.synthetic.dipole_flexural_dispersion :
        The phenomenological 1.25 factor model. Use this one when you
        want a Poisson-grounded asymptote.
    """
    s_low = 1.0 / vs
    v_rayleigh = rayleigh_speed(vp, vs)
    s_high = 1.0 / v_rayleigh
    fc = vs / (2.0 * np.pi * a_borehole)

    def s_of_f(f: np.ndarray) -> np.ndarray:
        x = np.asarray(f) / fc
        return s_low + (s_high - s_low) * (x ** 2) / (1.0 + x ** 2)

    return s_of_f
