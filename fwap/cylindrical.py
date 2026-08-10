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
* :func:`scholte_speed`
* :func:`tube_wave_speed`
* :func:`leaky_radiation_attenuation`
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
* Schmitt, D. P. (1988). Shear wave logging in elastic formations.
  *Journal of the Acoustical Society of America* 84(6), 2215-2229.
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
        return (2.0 - xi) ** 2 - 4.0 * np.sqrt((1.0 - xi * a2) * (1.0 - xi))

    # F(0) = 0 (trivial root at zero speed). F dips negative and
    # crosses zero at the Rayleigh root somewhere in (~0.75, ~0.98)
    # for all physically reasonable Poisson's ratios. A bracket of
    # (0.1, 0.9999) reliably contains the non-trivial root.
    xi = brentq(F, 0.1, 0.9999, xtol=1.0e-12)
    return float(vs * np.sqrt(xi))


def scholte_speed(
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
) -> float:
    r"""
    Scholte interface-wave speed for a **plane** fluid/solid interface.

    Solves the classical fluid-loaded secular equation

    .. math::

        \left(2 - \frac{c^2}{V_s^2}\right)^2
        - 4\sqrt{1 - \frac{c^2}{V_p^2}}\sqrt{1 - \frac{c^2}{V_s^2}}
        + \frac{\rho_f}{\rho}\,\frac{c^4}{V_s^4}\,
          \frac{\sqrt{1 - c^2/V_p^2}}{\sqrt{1 - c^2/V_f^2}}
        \;=\; 0

    for the root :math:`c < \min(V_s, V_f)`, where all three radicals are
    real and the wave is bound to the interface.

    Why this function exists
    ------------------------
    It is an **independent oracle for the cylindrical solver**. The
    borehole Stoneley mode of :func:`fwap.stoneley_dispersion` approaches
    this speed as the wavelength becomes short compared with the borehole
    radius, at which point the borehole wall looks flat. Because that
    limit is computed here from a *different* equation --- a plane
    interface, no Bessel functions, no modal determinant --- agreement
    between the two is a real cross-check rather than the solver
    confirming itself. See ``tests/test_cylindrical.py`` for the
    convergence test that uses it.

    Notes
    -----
    Two things about the equation are easy to get wrong, and both are
    checked by the test suite:

    * **The sign of the fluid-loading term is not determined by the
      light-fluid limit.** Both signs reduce to the Rayleigh equation as
      ``rho_f -> 0``, so that limit cannot discriminate them. The sign is
      fixed instead by requiring a root to exist below
      ``min(V_s, V_f)``: with the opposite sign the left-hand side is
      negative throughout and never crosses zero.
    * The root is *not* generally near the Rayleigh speed. When
      ``V_f < V_s`` the Rayleigh speed can lie above ``min(V_s, V_f)``
      and so outside the admissible range entirely.

    Parameters
    ----------
    vp : float
        Solid compressional speed (m/s); must exceed ``vs``.
    vs : float
        Solid shear speed (m/s); must be positive.
    rho : float
        Solid density (kg/m^3); must be positive.
    vf : float
        Fluid speed (m/s); must be positive.
    rho_f : float
        Fluid density (kg/m^3); must be positive.

    Returns
    -------
    float
        Scholte speed (m/s), strictly less than ``min(vs, vf)``.

    Raises
    ------
    ValueError
        If any input is non-positive, if ``vp <= vs``, or if no root can
        be bracketed below ``min(vs, vf)``.

    References
    ----------
    * Scholte, J. G. (1947). The range of existence of Rayleigh and
      Stoneley waves. *Geophys. Suppl. Mon. Not. R. Astron. Soc.*, 5(5),
      120-126.
    * Brekhovskikh, L. M., & Godin, O. A. (1990). *Acoustics of Layered
      Media I.* Springer.
    """
    if vs <= 0.0 or rho <= 0.0 or vf <= 0.0 or rho_f <= 0.0:
        raise ValueError("vs, rho, vf and rho_f must all be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")

    ceiling = min(vs, vf)

    def secular(c: float) -> float:
        radical_p = np.sqrt(1.0 - (c / vp) ** 2)
        radical_s = np.sqrt(1.0 - (c / vs) ** 2)
        radical_f = np.sqrt(1.0 - (c / vf) ** 2)
        rayleigh = (2.0 - (c / vs) ** 2) ** 2 - 4.0 * radical_p * radical_s
        loading = (rho_f / rho) * (c / vs) ** 4 * radical_p / radical_f
        return rayleigh + loading

    # The loading term diverges as c -> min(vs, vf), so the upper end is
    # approached rather than reached. Scan for the bracket: the root sits
    # close to the ceiling for a heavy fluid and near the Rayleigh speed
    # for a light one, which is too wide a span to hard-code a bracket.
    grid = np.linspace(0.05 * ceiling, ceiling * (1.0 - 1.0e-12), 4096)
    values = np.array([secular(float(c)) for c in grid])
    finite = np.isfinite(values)
    for i in range(grid.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if np.sign(values[i]) != np.sign(values[i + 1]):
            return float(
                brentq(secular, float(grid[i]), float(grid[i + 1]), xtol=1.0e-13)
            )
    raise ValueError(  # pragma: no cover - unreachable for valid media
        "no Scholte root below min(vs, vf); check the input media"
    )


def tube_wave_speed(
    vs: float,
    rho: float,
    *,
    vf: float,
    rho_f: float,
) -> float:
    r"""
    Low-frequency (tube-wave) limit of the borehole Stoneley speed.

    The White (1983) closed form

    .. math::

        S_T^2 \;=\; \frac{1}{V_f^2} \;+\; \frac{\rho_f}{\mu},
        \qquad \mu = \rho V_s^2,

    returned as a speed :math:`V_T = 1 / S_T`. Note what is *absent*:
    the borehole radius. In this limit the wavelength is long compared
    with the hole, the fluid column responds quasi-statically against
    the formation's shear stiffness, and the geometry drops out
    entirely --- which is itself a testable prediction, and a sharper
    one than the value.

    Why this function exists
    ------------------------
    It is the **low-frequency counterpart to :func:`scholte_speed`**.
    Between them the two pin both ends of the borehole Stoneley
    dispersion curve against closed forms: this one as
    :math:`f \to 0`, the Scholte speed as :math:`f \to \infty`.

    **A caveat on independence, which matters for how much the
    agreement is worth.** Unlike :func:`scholte_speed`, this formula is
    not wholly outside the solver: ``_stoneley_kz_bracket`` uses it to
    place the upper end of its search bracket. The bracket is a
    factor-of-two-wide window and cannot force agreement to the ~1e-7
    that is actually observed, and the tests deliberately locate the
    root by scanning 40x wider than the solver's own bracket so the
    estimate plays no part. But the check is weaker than the Scholte
    one and is documented as such rather than being presented as a
    fully independent tie.

    Validity floor
    --------------
    A tube wave is a *bound* mode, so it must be slower than the
    formation shear wave. Requiring :math:`V_T < V_s` gives

    .. math::

        V_s \;>\; V_f \sqrt{1 - \rho_f / \rho},

    below which this formula returns a speed above :math:`V_s`, which
    describes no bound mode --- and, measured, no bound root exists
    there: the borehole Stoneley merges with the shear branch point at
    a finite frequency and ceases to exist below it. Scanning the modal
    determinant across a window far wider than the solver's bracket
    finds no sign change at all. The closed form predicts where the
    cylindrical solver stops converging to within 1 % across formation
    and fluid densities giving floors from 960 to 1255 m/s.

    This bites in practice: for brine (``rho_f`` 1000) in a
    2200 kg/m^3 formation the floor is 1108 m/s, which is an ordinary
    slow formation. It is the reason
    :func:`fwap.vs_from_stoneley_slow_formation` --- whose whole
    purpose is slow formations --- has a lower limit on the shear
    velocities it can report, documented there in terms of the
    Stoneley slowness a caller actually measures.

    Parameters
    ----------
    vs : float
        Formation shear velocity (m/s). Must be positive and above the
        validity floor above.
    rho : float
        Formation bulk density (kg/m^3). Must exceed ``rho_f``.
    vf : float
        Borehole-fluid velocity (m/s); must be positive.
    rho_f : float
        Borehole-fluid density (kg/m^3); must be positive.

    Returns
    -------
    float
        Tube-wave speed (m/s), strictly less than both ``vf`` and
        ``vs``.

    Raises
    ------
    ValueError
        If any input is non-positive, if ``rho <= rho_f`` (the
        formation cannot be lighter than the mud in it), or if ``vs``
        is at or below the validity floor, where the formula describes
        no bound mode.

    See Also
    --------
    scholte_speed : The high-frequency end of the same curve.
    fwap.vs_from_stoneley_slow_formation : Inverts this relation for
        ``V_S`` from a measured Stoneley slowness.

    References
    ----------
    * White, J. E. (1983). *Underground Sound: Application of Seismic
      Waves.* Elsevier, sect. 5.5.
    * Norris, A. N. (1990). The speed of a tube wave. *J. Acoust. Soc.
      Am.* 87(1), 414-417.
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes.* CRC Press, Ch. 3.
    """
    if vs <= 0.0 or rho <= 0.0 or vf <= 0.0 or rho_f <= 0.0:
        raise ValueError("vs, rho, vf and rho_f must all be positive")
    if rho <= rho_f:
        raise ValueError(
            f"require rho > rho_f; got rho={rho}, rho_f={rho_f}. A formation "
            "lighter than the borehole fluid has no bound tube wave."
        )

    floor = vf * np.sqrt(1.0 - rho_f / rho)
    if vs <= floor:
        raise ValueError(
            f"vs={vs} is at or below the tube-wave validity floor "
            f"vf*sqrt(1 - rho_f/rho) = {floor:.4g} m/s. Below it the White "
            "formula returns a speed above vs, which is not a bound mode; "
            "the borehole Stoneley wave ceases to exist at low frequency "
            "in such formations."
        )

    return float(1.0 / np.sqrt(1.0 / vf**2 + rho_f / (rho * vs**2)))


def _fluid_solid_reflection(
    c: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
) -> np.ndarray:
    """
    Plane-wave reflection coefficient, fluid on elastic solid.

    Evaluated at horizontal slowness ``1 / c``, i.e. for the plane wave
    whose trace velocity along the interface is ``c``. See
    :func:`leaky_radiation_attenuation` for the formula and references.

    Parameters
    ----------
    c : ndarray
        Trace velocity along the interface (m/s).
    vp, vs, rho, vf, rho_f : float
        Solid and fluid media, as in
        :func:`leaky_radiation_attenuation`.

    Returns
    -------
    ndarray
        Complex reflection coefficient, same shape as ``c``. Its modulus
        is exactly 1 wherever ``vf < c <= vs`` (both formation waves
        evanescent, so nothing radiates) and less than 1 for
        ``vs < c < vp``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_f = np.sqrt((1.0 - (vf / c) ** 2).astype(complex))
        cos_p = np.sqrt((1.0 - (vp / c) ** 2).astype(complex))
        cos_s = np.sqrt((1.0 - (vs / c) ** 2).astype(complex))

        gamma = (vs / c) ** 2
        z_f = rho_f * vf / cos_f
        z_s = rho * vp * (1.0 - 2.0 * gamma) ** 2 / cos_p + (
            4.0 * rho * vs * gamma * (1.0 - gamma) / cos_s
        )
        return np.asarray((z_s - z_f) / (z_s + z_f))


def leaky_radiation_attenuation(
    phase_velocity: np.ndarray | float,
    freq: np.ndarray | float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> np.ndarray:
    r"""
    Ray estimate of a borehole leaky mode's radiation attenuation.

    Models the ``n=0`` leaky mode as a plane wave bouncing inside the
    fluid column: it crosses the borehole from wall to wall through the
    axis (a transverse path of ``2a``), reflects once, and repeats. The
    reflection is lossy whenever the formation shear wave is
    propagating, so the mode decays along ``z`` at

    .. math::

        \operatorname{Im} k_z
        \;=\; \frac{-\ln |R(c)| \; k_f}{2 a \, k_z},
        \qquad
        k_z = \frac{\omega}{c}, \quad
        k_f = \sqrt{\frac{\omega^2}{V_f^2} - k_z^2}

    where :math:`R` is the textbook plane-wave reflection coefficient
    for a fluid on an elastic solid at horizontal slowness :math:`1/c`,

    .. math::

        R = \frac{Z_s - Z_f}{Z_s + Z_f}, \qquad
        Z_f = \frac{\rho_f V_f}{\sqrt{1 - V_f^2/c^2}},

    .. math::

        Z_s = \frac{\rho V_p \left(1 - 2V_s^2/c^2\right)^2}
                   {\sqrt{1 - V_p^2/c^2}}
            + \frac{4 \rho V_s \,(V_s^2/c^2)(1 - V_s^2/c^2)}
                   {\sqrt{1 - V_s^2/c^2}}.

    Why this function exists
    ------------------------
    It is an **independent oracle for the leaky-mode solver**, in the
    same spirit as :func:`scholte_speed`. Nothing here touches the
    cylindrical modal determinant: there are no Bessel functions and no
    root finding, only a plane-wave reflection coefficient and a
    bounce-rate. Comparing it with the ``attenuation_per_meter`` field
    of :func:`fwap.pseudo_rayleigh_modal_dispersion` therefore checks
    the solver against different physics rather than against itself.

    What the comparison actually shows, measured over a 4-30 kHz band
    for borehole radii 0.07-0.15 m and fast formations with
    ``V_S`` 1700-2800 m/s (see ``tests/test_cylindrical.py``):

    * The estimate is the **right size and carries the right
      geometry**. Solver-to-estimate ratios stay inside roughly
      0.37-1.9 for a brine-like fluid.
    * There is a **stable systematic offset**: the median ratio is
      0.57-0.71 across every one of those cases, so the solver is
      consistently *less* attenuating than this estimate by a factor
      near 0.6. That offset is reported rather than corrected --- no
      derivation here accounts for it, and folding an empirical
      constant into the formula would turn an independent oracle into
      a fit.
    * The residual scatter about that offset is a **transverse
      resonance** the single-bounce picture omits. The solver's
      attenuation oscillates with frequency, and the peak spacing
      satisfies ``spacing * a = const`` to about 6 % over that range of
      radii --- the ``2a`` round trip assumed above, arrived at
      independently from the solver's own output.

    So this is a **order-of-magnitude and scaling check**, not a
    precision one. It is strong enough to catch a wrong power of
    frequency, a wrong radius dependence, or an attenuation that is
    orders out; it would not catch a 30 % error.

    Parameters
    ----------
    phase_velocity : ndarray or float
        Mode phase velocity ``c`` (m/s), of any shape broadcastable
        against ``freq``. Finite entries must satisfy
        ``vf < c < vp``. ``NaN`` entries propagate to ``NaN``, so the
        ``1 / slowness`` of a :class:`fwap.BoreholeMode` can be passed
        in directly.
    freq : ndarray or float
        Frequency (Hz), broadcastable against ``phase_velocity``. Must
        be strictly positive.
    vp, vs, rho : float
        Formation P-wave velocity (m/s), S-wave velocity (m/s) and bulk
        density (kg/m^3). Require ``vp > vs > 0`` and ``rho > 0``.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3). Require
        ``vs > vf > 0`` and ``rho_f > 0`` (fast formation).
    a : float
        Borehole radius (m); must be positive.

    Returns
    -------
    ndarray
        Radiation attenuation (1/m, the spatial decay rate of mode
        amplitude in the ``+z`` direction), broadcast to the common
        shape of ``phase_velocity`` and ``freq``. Exactly ``0.0``
        wherever ``c <= vs``: both formation waves are then evanescent,
        ``|R| = 1`` identically, and the mode is bound rather than
        leaky.

    Raises
    ------
    ValueError
        If any medium parameter is non-positive, if ``vp <= vs``, if
        ``vs <= vf`` (slow formation --- no leaky-S window exists), if
        any frequency is non-positive, or if any finite
        ``phase_velocity`` falls outside ``(vf, vp)``.

    See Also
    --------
    scholte_speed : The analogous plane-interface oracle for the bound
        Stoneley mode's high-frequency limit.
    fwap.pseudo_rayleigh_modal_dispersion : The cylindrical leaky-mode
        solver this function is used to check.

    References
    ----------
    * Brekhovskikh, L. M., & Godin, O. A. (1990). *Acoustics of Layered
      Media I.* Springer. (Fluid/solid plane-wave reflection.)
    * Aki, K., & Richards, P. G. (2002). *Quantitative Seismology*
      (2nd ed.). University Science Books, sect. 5.2.
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes.* CRC Press, sect. 4.4. (Borehole leaky modes.)
    """
    if vs <= 0.0 or rho <= 0.0 or vf <= 0.0 or rho_f <= 0.0 or a <= 0.0:
        raise ValueError("vs, rho, vf, rho_f and a must all be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")
    if vs <= vf:
        raise ValueError(
            "require vs > vf (fast formation); a slow formation has no leaky-S window"
        )

    c = np.asarray(phase_velocity, dtype=float)
    f = np.asarray(freq, dtype=float)
    if np.any(f <= 0.0):
        raise ValueError("freq must be strictly positive")

    c, f = np.broadcast_arrays(c, f)
    finite = np.isfinite(c)
    if np.any(finite & ((c <= vf) | (c >= vp))):
        raise ValueError(
            f"phase_velocity must lie in ({vf}, {vp}) m/s where the fluid "
            "wave propagates and the formation P wave does not"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        reflection = np.abs(
            _fluid_solid_reflection(c, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f)
        )

        omega = 2.0 * np.pi * f
        k_z = omega / c
        k_f = np.sqrt(np.maximum((omega / vf) ** 2 - k_z**2, 0.0))
        attenuation = -np.log(reflection) * k_f / (2.0 * a * k_z)

    # Below V_S nothing radiates: |R| is 1 as an algebraic identity, so
    # the attenuation is exactly zero. It is imposed here rather than
    # left to the arithmetic because |R| comes out of a complex division
    # and lands on 1 only to a rounding error, which would otherwise
    # leak a ~1e-16 attenuation into a mode that is bound.
    attenuation = np.where(c <= vs, 0.0, attenuation)
    return np.where(np.isfinite(c), np.maximum(attenuation, 0.0), np.nan)


def flexural_dispersion_physical(
    vp: float,
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
        return s_low + (s_high - s_low) * (x**2) / (1.0 + x**2)

    return s_of_f


def flexural_dispersion_vti_physical(
    vp: float,
    vsv: float,
    vsh: float,
    a_borehole: float = 0.1,
) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Phenomenological VTI dipole-flexural dispersion law.

    Generalises :func:`flexural_dispersion_physical` to a VTI
    formation by anchoring the two-parameter rational
    interpolation between physically-correct low- and high-
    frequency asymptotes:

    * **Low-frequency limit.** The dipole-flexural mode at
      :math:`f \to 0` has wavelength much larger than the borehole
      radius, so it sees the formation only through its
      vertically-propagating shear stiffness :math:`C_{44} =
      \rho V_{Sv}^2`. The low-frequency phase slowness is therefore
      :math:`1/V_{Sv}` (Ellefsen, Cheng & Toksoz 1991, sect. III.B;
      Sinha, Norris & Chang 1994, eq. 14).
    * **High-frequency limit.** As :math:`f \to \infty` the
      flexural mode localises within ~one wavelength of the
      borehole wall and propagates as a Rayleigh-like surface wave
      whose speed is governed by the *horizontal* shear modulus
      :math:`C_{66} = \rho V_{Sh}^2` (the SH polarisation of the
      surface motion). The high-frequency phase slowness is
      :math:`1/V_R(V_P, V_{Sh})` from
      :func:`rayleigh_speed`.

    Between the two asymptotes the dispersion is interpolated with
    the same rational form
    :math:`s(f) = s_{\text{low}} + (s_{\text{high}}-s_{\text{low}})
    \cdot x^2/(1+x^2)` used in
    :func:`flexural_dispersion_physical`, with the geometric cut-
    off :math:`f_c = V_{Sv} / (2\pi a)` set by the *low-frequency*
    velocity. For an isotropic formation
    (:math:`V_{Sh} = V_{Sv} \Rightarrow \gamma = 0`) the function
    reduces exactly to :func:`flexural_dispersion_physical(vp, vsv,
    a_borehole)`.

    Caveats
    -------
    * **Phenomenological**, not the full Schmitt (1989) VTI
      cylindrical-mode determinant. The two asymptotes are
      physically correct in their respective limits, but the
      transition shape is a smoothed-step approximation of the
      true root of the modal determinant. For dispersion-curve
      bias estimates within the typical dipole-sonic band
      (~1-6 kHz) the approximation is in line with
      Ellefsen-Cheng-Toksöz-grade weak-anisotropy expansions; for
      quantitative full-band inversion (e.g. matching dispersion-
      corrected STC to better than 1-2 % :math:`V_{Sv}`) the full
      modal solver is required.
    * Vacuum-loaded Rayleigh asymptote (no Scholte / fluid-loading
      correction). For a brine-filled borehole through a typical
      VTI shale the Rayleigh-Scholte difference is at the
      few-percent level (Paillet & Cheng 1991, ch. 4).
    * Single-layer VTI; mudcake / altered-zone radial layering not
      modelled.

    Parameters
    ----------
    vp : float
        Vertical P-wave velocity (m/s). For VTI media this is
        :math:`V_P = \sqrt{C_{33}/\rho}`.
    vsv : float
        Vertical shear velocity (m/s):
        :math:`V_{Sv} = \sqrt{C_{44}/\rho}`. Must satisfy
        ``vp > vsv``.
    vsh : float
        Horizontal shear velocity (m/s):
        :math:`V_{Sh} = \sqrt{C_{66}/\rho}`. Must satisfy
        ``vp > vsh``. Equal to ``vsv`` for an isotropic formation;
        :math:`V_{Sh} > V_{Sv}` for typical VTI shales (Thomsen
        :math:`\gamma > 0`).
    a_borehole : float, default 0.1
        Borehole radius (m).

    Returns
    -------
    Callable[[ndarray], ndarray]
        Array-in / array-out mapping of frequency (Hz) to phase
        slowness (s/m). Drop-in for the ``dispersion_family``
        argument of :func:`fwap.dispersion.dispersive_stc` and
        :func:`fwap.dispersion.dispersive_pseudo_rayleigh_stc`.

    Raises
    ------
    ValueError
        If ``vsv`` or ``vsh`` is non-positive, or if either is
        :math:`\ge` ``vp`` (an isotropic-rock physical bound that
        the VTI Christoffel system also respects under standard
        positivity of the Hooke tensor).

    See Also
    --------
    flexural_dispersion_physical : Isotropic special case;
        equivalent to this function when ``vsh == vsv``.
    fwap.anisotropy.vti_moduli_from_logs : Sonic-derived VTI
        moduli; supplies the ``(vp, vsv, vsh)`` triple from
        monopole + dipole + Stoneley + density logs.

    References
    ----------
    * Ellefsen, K. J., Cheng, C. H., & Toksoz, M. N. (1991).
      Effects of anisotropy upon the resonances of normal modes
      in a borehole. *J. Acoust. Soc. Am.* 89(6), 2597-2616.
      Section III.B gives the long-wavelength flexural-mode limit
      :math:`s_{low} = 1/V_{Sv}` for VTI formations.
    * Sinha, B. K., Norris, A. N., & Chang, S. K. (1994).
      Borehole flexural modes in anisotropic formations.
      *Geophysics* 59(7), 1037-1052. Eq. 14 confirms the same
      low-frequency limit and gives the high-frequency
      surface-wave-on-borehole-wall asymptote that motivates
      :math:`V_R(V_P, V_{Sh})`.
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes*, ch. 4. CRC Press (cylindrical-mode dispersion;
      Rayleigh / Scholte asymptotic forms).
    """
    if vsv <= 0.0:
        raise ValueError("vsv must be positive")
    if vsh <= 0.0:
        raise ValueError("vsh must be positive")
    if vp <= vsv:
        raise ValueError("require vp > vsv")
    if vp <= vsh:
        raise ValueError("require vp > vsh")

    s_low = 1.0 / vsv
    # High-f Rayleigh asymptote with horizontal shear setting the
    # surface-wave SH polarisation speed. rayleigh_speed validates
    # the vp > vs constraint internally.
    v_rayleigh = rayleigh_speed(vp, vsh)
    s_high = 1.0 / v_rayleigh
    # Geometric cut-off set by the low-frequency velocity (the
    # dispersion turns over when wavelength ~ borehole radius, and
    # at low f the wave propagates at V_Sv).
    fc = vsv / (2.0 * np.pi * a_borehole)

    def s_of_f(f: np.ndarray) -> np.ndarray:
        x = np.asarray(f) / fc
        return s_low + (s_high - s_low) * (x**2) / (1.0 + x**2)

    return s_of_f
