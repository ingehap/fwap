"""
Vertical-wellbore stability: Kirsch wall stresses, Mohr-Coulomb
shear-breakout pressure, Hubbert-Willis tensile-breakdown pressure,
and the :func:`safe_mud_weight_window` wrapper that bundles the
two pressures with the :class:`MudWeightWindow` result.

The :class:`MudWeightWindow` dataclass is shared with
:func:`fwap.geomechanics.inclined.inclined_safe_mud_weight_window`.

References
----------
* Kirsch, G. (1898). Die Theorie der Elastizitat und die Bedurfnisse
  der Festigkeitslehre. *Zeitschrift des Vereines Deutscher
  Ingenieure* 42, 797-807.
* Hubbert, M. K., & Willis, D. G. (1957). Mechanics of hydraulic
  fracturing. *AIME Petroleum Transactions* 210, 153-168.
* Zoback, M. D. (2007). *Reservoir Geomechanics*, Sections 6.2-6.4.
  Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _mohr_coulomb_q(friction_angle_deg: float) -> float:
    r"""Mohr-Coulomb stress-ratio :math:`q = (1+\sin\phi)/(1-\sin\phi)`.

    Shared by :func:`mohr_coulomb_breakout_pressure` and
    :func:`fwap.geomechanics.inclined.inclined_breakout_pressure`.
    Caller is responsible for validating ``friction_angle_deg`` is in
    ``(-90, 90)``.
    """
    phi = np.deg2rad(friction_angle_deg)
    sin_phi = np.sin(phi)
    return float((1.0 + sin_phi) / (1.0 - sin_phi))


def kirsch_wall_stresses(
    sigma_v: np.ndarray,
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    *,
    azimuth_deg: np.ndarray,
    mud_pressure: float | np.ndarray = 0.0,
    poisson: float | np.ndarray = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Kirsch (1898) borehole-wall stresses for a vertical well.

    Stress concentration around a circular hole drilled vertically
    through a homogeneous, isotropic, elastic medium under far-field
    horizontal stresses :math:`\sigma_H` (max), :math:`\sigma_h`
    (min), and vertical stress :math:`\sigma_v`. At the borehole
    wall (``r = a``), at azimuth :math:`\theta` measured from the
    :math:`\sigma_H` direction:

    .. math::

        \sigma_{\theta\theta}(\theta) &=
            \sigma_H + \sigma_h
            - 2 (\sigma_H - \sigma_h)\cos(2\theta) - P_w,
        \\
        \sigma_{zz}(\theta) &=
            \sigma_v - 2\nu(\sigma_H - \sigma_h)\cos(2\theta),
        \\
        \sigma_{rr} &= P_w.

    The shear stresses :math:`\sigma_{r\theta}, \sigma_{rz},
    \sigma_{\theta z}` vanish at the wall when the well axis is
    aligned with one of the principal stress directions, which is
    the convention used here. The three stresses returned are then
    principal stresses of the local stress tensor at the wall.

    Special azimuths:

    * :math:`\theta = 0\degree` (in the :math:`\sigma_H` direction):
      :math:`\sigma_{\theta\theta} = 3\sigma_h - \sigma_H - P_w`
      (least compressive; tensile-failure / fracture-initiation
      azimuth).
    * :math:`\theta = 90\degree` (in the :math:`\sigma_h`
      direction): :math:`\sigma_{\theta\theta} = 3\sigma_H -
      \sigma_h - P_w` (most compressive; shear-failure / breakout
      azimuth).

    Parameters
    ----------
    sigma_v : scalar or ndarray
        Vertical (overburden) stress (Pa). Use
        :func:`overburden_stress`.
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa). Convention:
        ``sigma_H >= sigma_h``; the function does not enforce this
        because the user may pass either as the "long" or "short"
        principal direction.
    azimuth_deg : scalar or ndarray
        Azimuth (degrees) measured from the :math:`\sigma_H`
        direction.
    mud_pressure : scalar or ndarray, default 0.0
        Wellbore (mud) pressure :math:`P_w` (Pa). Default 0.0
        models a dry hole.
    poisson : scalar or ndarray, default 0.25
        Poisson's ratio (dimensionless). Enters the
        :math:`\sigma_{zz}` formula via the plane-strain
        coupling between horizontal stress deviator and axial
        stress. Default 0.25 is typical for sandstones.

    Returns
    -------
    (sigma_theta, sigma_z, sigma_r) : tuple of ndarrays
        Hoop, axial, and radial stresses at the wall (Pa),
        broadcast to the common shape of the inputs. All three
        are total stresses (no pore-pressure subtraction); pass
        through ``sigma - alpha * P_p`` for effective stresses.

    See Also
    --------
    mohr_coulomb_breakout_pressure : Critical mud pressure for
        shear breakout, derived from the Kirsch hoop stress at the
        breakout azimuth.

    References
    ----------
    * Kirsch, E. G. (1898). Die Theorie der Elastizitaet und die
      Beduerfnisse der Festigkeitslehre. *Z. Verein. Deutsch.
      Ing.* 42, 797-807.
    * Jaeger, J. C., Cook, N. G. W., & Zimmerman, R. W. (2007).
      *Fundamentals of Rock Mechanics*, 4th ed., Chapter 8
      (borehole-stress analysis).
    """
    theta = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    sH = np.asarray(sigma_H, dtype=float)
    sh = np.asarray(sigma_h, dtype=float)
    sv = np.asarray(sigma_v, dtype=float)
    Pw = np.asarray(mud_pressure, dtype=float)
    nu = np.asarray(poisson, dtype=float)

    cos2 = np.cos(2.0 * theta)
    deviator = sH - sh
    sigma_theta = sH + sh - 2.0 * deviator * cos2 - Pw
    sigma_z = sv - 2.0 * nu * deviator * cos2
    sigma_r = np.broadcast_to(Pw, sigma_theta.shape).astype(float).copy()
    return sigma_theta, sigma_z, sigma_r


def mohr_coulomb_breakout_pressure(
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    pore_pressure: np.ndarray,
    ucs: np.ndarray,
    *,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
) -> np.ndarray:
    r"""
    Mohr-Coulomb shear-breakout mud pressure for a vertical well.

    Returns the minimum mud pressure :math:`P_w^{\,\mathrm{crit}}`
    below which Mohr-Coulomb shear failure initiates at the
    breakout azimuth (perpendicular to :math:`\sigma_H`). For
    :math:`P_w < P_w^{\,\mathrm{crit}}` the wellbore wall fails in
    shear, leading to wellbore breakout / enlargement /
    eventually collapse.

    Derivation
    ----------
    At the breakout azimuth, the Kirsch hoop stress is
    :math:`\sigma_{\theta\theta} = 3\sigma_H - \sigma_h - P_w`
    (most compressive); the radial stress is
    :math:`\sigma_{rr} = P_w` (least compressive). For a vertical
    well in a normal-fault stress regime where
    :math:`\sigma_{\theta\theta} > \sigma_{zz} > \sigma_{rr}`,
    these are the maximum and minimum principal stresses at the
    wall. Pass through effective stresses by subtracting
    :math:`\alpha P_p`:

    .. math::

        \sigma_1' &= \sigma_{\theta\theta} - \alpha P_p
                  = 3\sigma_H - \sigma_h - P_w - \alpha P_p,
        \\
        \sigma_3' &= \sigma_{rr} - \alpha P_p
                  = P_w - \alpha P_p.

    Apply the Mohr-Coulomb failure criterion in principal-stress
    form
    :math:`\sigma_1' = q\,\sigma_3' + \mathrm{UCS}` where
    :math:`q = (1+\sin\phi)/(1-\sin\phi)` for friction angle
    :math:`\phi`. Solving for :math:`P_w`:

    .. math::

        P_w^{\,\mathrm{crit}} \;=\;
            \frac{3\sigma_H \;-\; \sigma_h
                  \;+\; (q - 1)\,\alpha P_p
                  \;-\; \mathrm{UCS}}{1 + q}.

    Sensitivity (typical regimes with :math:`q > 1`):

    * Higher horizontal stress anisotropy
      (:math:`3\sigma_H - \sigma_h`): higher
      :math:`P_w^{\,\mathrm{crit}}` (more support needed).
    * Higher pore pressure: higher
      :math:`P_w^{\,\mathrm{crit}}` (the rock is weaker in
      effective stress).
    * Higher UCS or friction angle (stronger rock): lower
      :math:`P_w^{\,\mathrm{crit}}`.

    Parameters
    ----------
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa).
    pore_pressure : scalar or ndarray
        Pore pressure :math:`P_p` (Pa). Use
        :func:`pore_pressure_eaton` to estimate from sonic data.
    ucs : scalar or ndarray
        Unconfined compressive strength :math:`\mathrm{UCS}` (Pa).
        Use :func:`unconfined_compressive_strength` from a sonic
        log, or pass a measured value.
    friction_angle_deg : float, default 30.0
        Internal friction angle :math:`\phi` (degrees). Typical
        ranges: 25-35 for shales, 30-40 for sandstones, 35-45 for
        limestones. Set to 0 for the cohesion-only (Tresca) limit.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient. ``1.0`` is the textbook
        upper bound for a soft frame; tight rocks may be 0.7-0.8.

    Returns
    -------
    ndarray
        Critical mud pressure :math:`P_w^{\,\mathrm{crit}}` (Pa)
        for shear breakout, broadcast to the common shape of the
        inputs. Negative values indicate the rock is strong
        enough to remain stable even with negative wellbore
        pressure (i.e. shear-failure-free); in practice, the
        actual mud pressure should be at least
        :math:`P_p` to balance pore pressure regardless of the
        Mohr-Coulomb result.

    Raises
    ------
    ValueError
        If ``friction_angle_deg`` is not in the open interval
        ``(-90, 90)`` (which keeps :math:`\cos\phi > 0` so
        :math:`q` is finite and positive).

    Notes
    -----
    The formula assumes:

    * Vertical well, vertical principal stress (normal-fault
      regime). Strike-slip and reverse-fault regimes need a
      different :math:`\sigma_1, \sigma_3` identification at
      the wall and are not handled here.
    * :math:`\sigma_{\theta\theta} > \sigma_{zz}` at the
      breakout azimuth, which is the typical case but can fail
      in regimes where :math:`\sigma_v` greatly exceeds
      :math:`\sigma_H`. Callers in non-typical regimes should
      use :func:`kirsch_wall_stresses` directly and apply the
      Mohr-Coulomb criterion to the actual maximum principal
      stress.
    * No tensile failure (fracture initiation, the upper bound
      of the safe mud-weight window). The companion tensile-
      breakdown calculation is a planned follow-up.

    See Also
    --------
    kirsch_wall_stresses : Underlying primitive that gives the
        wall stresses at any azimuth.
    unconfined_compressive_strength : Sonic-derived UCS estimate
        suitable as the ``ucs`` input.
    pore_pressure_eaton : Sonic-derived pore-pressure estimate
        suitable as the ``pore_pressure`` input.
    closure_stress : Closure stress (the lower bound of the safe
        mud-weight window when the limiting failure is tensile).

    References
    ----------
    * Mohr, O. (1900). Welche Umstaende bedingen die
      Elastizitaetsgrenze und den Bruch eines Materiales? *Z.
      Verein. Deutsch. Ing.* 44, 1524-1530.
    * Coulomb, C. A. (1776). Essai sur une application des
      regles de maximis et minimis a quelques problemes de
      statique relatifs a l'architecture. *Mem. Acad. Sci. Paris*
      7, 343-382.
    * Zoback, M. D. (2007). *Reservoir Geomechanics.* Cambridge
      University Press, Chapter 6.
    * Jaeger, J. C., Cook, N. G. W., & Zimmerman, R. W. (2007).
      *Fundamentals of Rock Mechanics*, 4th ed., Section 8.6.
    """
    if not (-90.0 < friction_angle_deg < 90.0):
        raise ValueError("friction_angle_deg must be in (-90, 90)")

    sH = np.asarray(sigma_H, dtype=float)
    sh = np.asarray(sigma_h, dtype=float)
    Pp = np.asarray(pore_pressure, dtype=float)
    UCS = np.asarray(ucs, dtype=float)

    q = _mohr_coulomb_q(friction_angle_deg)

    return (3.0 * sH - sh + (q - 1.0) * biot_alpha * Pp - UCS) / (1.0 + q)


def tensile_breakdown_pressure(
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    pore_pressure: np.ndarray,
    *,
    tensile_strength: float | np.ndarray = 0.0,
    biot_alpha: float = 1.0,
) -> np.ndarray:
    r"""
    Tensile-failure mud pressure (fracture initiation) for a vertical well.

    Returns the maximum mud pressure :math:`P_w^{\,\mathrm{break}}`
    above which tensile failure (fracture initiation) starts at the
    breakdown azimuth (in the :math:`\sigma_H` direction). Mud
    pressures above this limit open hydraulic fractures at the wall
    -- the standard "lost circulation" / "leak-off" scenario.

    Derivation
    ----------
    At the breakdown azimuth (:math:`\theta = 0` from
    :math:`\sigma_H`), the Kirsch hoop stress is
    :math:`\sigma_{\theta\theta} = 3\sigma_h - \sigma_H - P_w`
    (the LEAST compressive of the three wall stresses). Tensile
    failure occurs when the effective hoop stress drops below
    :math:`-T` (negative = tension; :math:`T` = tensile strength):

    .. math::

        \sigma_{\theta\theta} - \alpha P_p \;\le\; -T,

    so the maximum mud pressure that keeps the wall in compression is

    .. math::

        P_w^{\,\mathrm{break}} \;=\;
            3\sigma_h - \sigma_H + T - \alpha P_p.

    This is the Hubbert-Willis (1957) fracture-initiation pressure
    for a vertical well aligned with the principal stress axes.

    Sensitivity:

    * Higher :math:`\sigma_h` (stronger wall confinement): higher
      :math:`P_w^{\,\mathrm{break}}` (more pressure needed before
      tension overcomes compression).
    * Higher :math:`\sigma_H`: lower
      :math:`P_w^{\,\mathrm{break}}` (the wall is already pre-
      tensioned by horizontal stress anisotropy).
    * Higher :math:`P_p`: lower
      :math:`P_w^{\,\mathrm{break}}` (effective stress is reduced).
    * Higher tensile strength :math:`T`: higher
      :math:`P_w^{\,\mathrm{break}}` (the rock can carry some
      tension before failing).

    Parameters
    ----------
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa).
    pore_pressure : scalar or ndarray
        Pore pressure :math:`P_p` (Pa).
    tensile_strength : scalar or ndarray, default 0.0
        Tensile strength :math:`T` (Pa). Default 0.0 is the
        conservative case (no tensile strength); typical sandstones
        carry 1-5% of UCS in tension. Many petroleum-engineering
        treatments stick with the default zero because a single
        crack can dominate even when the bulk rock has tensile
        strength.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.

    Returns
    -------
    ndarray
        Tensile-breakdown mud pressure :math:`P_w^{\,\mathrm{break}}`
        (Pa). Mud pressures above this limit open fractures at the
        wall.

    See Also
    --------
    mohr_coulomb_breakout_pressure : The lower bound of the safe
        mud-weight window (shear-failure threshold).
    safe_mud_weight_window : Convenience wrapper returning both
        bounds plus a drillability flag.

    Notes
    -----
    Same vertical-well, principal-stress-aligned, drained-elastic
    assumptions as :func:`kirsch_wall_stresses` and
    :func:`mohr_coulomb_breakout_pressure`. In strike-slip and
    reverse-fault stress regimes the breakdown azimuth and formula
    change; this function does not handle those cases.

    References
    ----------
    * Hubbert, M. K., & Willis, D. G. (1957). Mechanics of
      hydraulic fracturing. *Trans. AIME* 210, 153-168.
    * Zoback, M. D. (2007). *Reservoir Geomechanics.* Cambridge
      University Press, Section 6.6.
    """
    sH = np.asarray(sigma_H, dtype=float)
    sh = np.asarray(sigma_h, dtype=float)
    Pp = np.asarray(pore_pressure, dtype=float)
    T = np.asarray(tensile_strength, dtype=float)
    return 3.0 * sh - sH + T - biot_alpha * Pp


@dataclass
class MudWeightWindow:
    r"""
    Output of :func:`safe_mud_weight_window`.

    The two pressure bounds that frame the safe mud-weight window
    for a vertical well, plus convenience properties for the
    window width and a per-depth drillability flag.

    Attributes
    ----------
    breakout_pressure : ndarray
        Lower bound (Pa); mud pressures below this trigger
        Mohr-Coulomb shear breakout at the borehole wall. Output
        of :func:`mohr_coulomb_breakout_pressure`.
    breakdown_pressure : ndarray
        Upper bound (Pa); mud pressures above this trigger tensile
        failure / fracture initiation. Output of
        :func:`tensile_breakdown_pressure`.

    Properties
    ----------
    width : ndarray
        ``breakdown_pressure - breakout_pressure`` (Pa). The mud-
        weight margin available for drilling.
    is_drillable : ndarray of bool
        ``True`` where ``width > 0`` (a non-empty safe window
        exists). ``False`` where the breakout limit exceeds the
        breakdown limit -- the well cannot be drilled in the
        chosen geometry without casing or stress-state
        intervention.
    """

    breakout_pressure: np.ndarray
    breakdown_pressure: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.breakdown_pressure - self.breakout_pressure

    @property
    def is_drillable(self) -> np.ndarray:
        return self.width > 0


def safe_mud_weight_window(
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    pore_pressure: np.ndarray,
    ucs: np.ndarray,
    *,
    tensile_strength: float | np.ndarray = 0.0,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
) -> MudWeightWindow:
    r"""
    Both mud-weight bounds (shear breakout + tensile breakdown).

    Convenience wrapper that calls
    :func:`mohr_coulomb_breakout_pressure` and
    :func:`tensile_breakdown_pressure` with consistent inputs and
    returns the two pressures bundled in a :class:`MudWeightWindow`
    dataclass.

    The "safe" mud-weight window is the closed interval
    ``[breakout_pressure, breakdown_pressure]``: mud pressures in
    this range avoid both shear failure (collapse) at the borehole
    wall and tensile failure (lost circulation). Pressures outside
    either bound trigger the corresponding failure mode.

    Per-depth diagnostic: if the window has zero or negative width
    at a particular depth (``breakout > breakdown``), the well
    cannot be drilled in the supplied geometry without casing,
    drilling-fluid-additive intervention, or a different well
    trajectory.

    Parameters
    ----------
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa).
    pore_pressure : scalar or ndarray
        Pore pressure (Pa).
    ucs : scalar or ndarray
        Unconfined compressive strength (Pa). Drives the breakout
        bound.
    tensile_strength : scalar or ndarray, default 0.0
        Tensile strength (Pa). Drives the breakdown bound.
    friction_angle_deg : float, default 30.0
        Internal friction angle (degrees) for the Mohr-Coulomb
        breakout calculation.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.

    Returns
    -------
    MudWeightWindow
        Dataclass with ``breakout_pressure`` and
        ``breakdown_pressure`` arrays plus ``width`` and
        ``is_drillable`` properties.

    See Also
    --------
    mohr_coulomb_breakout_pressure : The lower-bound primitive.
    tensile_breakdown_pressure : The upper-bound primitive.
    """
    P_breakout = mohr_coulomb_breakout_pressure(
        sigma_H,
        sigma_h,
        pore_pressure,
        ucs,
        friction_angle_deg=friction_angle_deg,
        biot_alpha=biot_alpha,
    )
    P_breakdown = tensile_breakdown_pressure(
        sigma_H,
        sigma_h,
        pore_pressure,
        tensile_strength=tensile_strength,
        biot_alpha=biot_alpha,
    )
    return MudWeightWindow(
        breakout_pressure=np.asarray(P_breakout, dtype=float),
        breakdown_pressure=np.asarray(P_breakdown, dtype=float),
    )
