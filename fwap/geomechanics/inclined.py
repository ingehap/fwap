"""
Inclined-wellbore stability: generalised Kirsch wall stresses for
arbitrarily oriented wells plus the Mohr-Coulomb shear-breakout /
Hubbert-Willis tensile-breakdown bisection scans.

For each candidate mud pressure, the wall stresses are computed at
``n_azimuth`` points around the borehole; the worst-azimuth failure
margin is bracketed and refined with ``scipy.optimize.brentq``.
Vertical-well closed forms are in :mod:`fwap.geomechanics.vertical`;
both modules agree to azimuth-grid resolution at zero inclination.

References
----------
* Hiramatsu, Y., & Oka, Y. (1962). Stress around a shaft or level
  excavated in ground with a three-dimensional stress state.
  *Memoirs of the Faculty of Engineering, Kyoto University* 24, 56-76.
* Fairhurst, C. (1968). Methods of determining in-situ rock
  stresses at great depths. Technical Report TRI-68, Missouri
  River Division Corps of Engineers, Omaha, Nebraska.
* Zoback, M. D. (2007). *Reservoir Geomechanics*, Sections 6.5-6.6.
  Cambridge University Press.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from fwap.geomechanics.vertical import MudWeightWindow


def _stress_rotation_to_well_frame(
    inclination_deg: float, azimuth_deg: float
) -> np.ndarray:
    r"""
    3x3 rotation matrix from principal-stress (PS) to well-aligned frame.

    PS axes: :math:`X = \sigma_H` direction, :math:`Y = \sigma_h`
    direction (both horizontal), :math:`Z = \sigma_v` direction
    (vertical, downward positive). Well axes: :math:`Z_W` along the
    well axis (down the hole). The well inclination ``iota`` is the
    angle from :math:`\sigma_v` (vertical) to the well axis; the
    well azimuth ``phi`` is the angle from :math:`\sigma_H` to the
    horizontal projection of the well axis (clockwise looking down).

    Standard parameterisation (Fjaer et al. 2008, Box 4.1): the
    high-side direction :math:`X_W` lies in the plane containing
    the well axis and the vertical axis, pointing toward the
    high side of the inclined wellbore.

    Returns the rotation matrix ``R`` such that the stress tensor
    transforms as ``sigma_W = R @ sigma_PS @ R.T``.

    For a vertical well (``iota = 0``), ``R`` reduces to the
    identity rotated by ``phi`` about the vertical axis -- the
    standard "well aligned with horizontal stress directions" case.
    """
    iota = np.deg2rad(inclination_deg)
    phi = np.deg2rad(azimuth_deg)
    cos_i, sin_i = np.cos(iota), np.sin(iota)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    # X_W (high side): in the (Z_W, Z_PS) plane.
    # Y_W: perpendicular to that plane, completing right-handed.
    # Z_W (well axis): down the hole.
    R = np.array(
        [
            [cos_i * cos_p, cos_i * sin_p, -sin_i],
            [-sin_p, cos_p, 0.0],
            [sin_i * cos_p, sin_i * sin_p, cos_i],
        ]
    )
    return R


def inclined_wellbore_wall_stresses(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    azimuth_around_wall_deg: np.ndarray,
    mud_pressure: float = 0.0,
    poisson: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Generalized Kirsch wall stresses for an arbitrarily inclined well.

    Returns the four wall stress components at each azimuth around
    the borehole (measured from the high side of the inclined well)
    after rotating the principal-stress tensor into well-aligned
    coordinates. Generalises :func:`kirsch_wall_stresses` (which
    is the vertical-well special case) to wells of any orientation.

    Two-step computation:

    1. **Stress rotation**: build the rotation matrix from
       principal-stress (PS) to well-aligned (W) coordinates from
       the well inclination and azimuth, then transform the
       diagonal :math:`\sigma_\mathrm{PS} =
       \mathrm{diag}(\sigma_H, \sigma_h, \sigma_v)` into the
       generally-non-diagonal :math:`\sigma_W` with components
       :math:`\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy},
       \sigma_{xz}, \sigma_{yz}`.

    2. **Generalized Kirsch (Hiramatsu-Oka 1962, Fairhurst 1968)**:
       at the wall, parameterised by azimuth :math:`\theta` from
       the high-side :math:`X_W` axis,

       .. math::

           \sigma_{\theta\theta}(\theta) &=
               \sigma_{xx} + \sigma_{yy}
               - 2(\sigma_{xx} - \sigma_{yy})\cos 2\theta
               - 4\sigma_{xy}\sin 2\theta
               - P_w,
           \\
           \sigma_{zz}(\theta) &=
               \sigma_{zz}^{(W)}
               - 2\nu(\sigma_{xx} - \sigma_{yy})\cos 2\theta
               - 4\nu\sigma_{xy}\sin 2\theta,
           \\
           \sigma_{\theta z}(\theta) &=
               2(-\sigma_{xz}\sin\theta + \sigma_{yz}\cos\theta),
           \\
           \sigma_{rr} &= P_w.

       The :math:`\sigma_{\theta z}` shear component is the new
       feature at non-vertical wells: it is identically zero for
       a vertical well (because the off-diagonal :math:`\sigma_{xz}`
       and :math:`\sigma_{yz}` rotation components vanish) and the
       formulas reduce to the standard Kirsch in
       :func:`kirsch_wall_stresses`.

    The four returned stresses are NOT all principal stresses --
    :math:`\sigma_{rr}` is, but :math:`\sigma_{\theta\theta},
    \sigma_{zz}, \sigma_{\theta z}` are coupled by the off-
    diagonal shear. Use the eigenvalues of the
    :math:`\begin{pmatrix}\sigma_{\theta\theta} & \sigma_{\theta z}
    \\ \sigma_{\theta z} & \sigma_{zz}\end{pmatrix}` 2x2 sub-block
    (or the helper inside :func:`inclined_breakout_pressure`) to
    extract the principal stresses.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Vertical and the two horizontal principal stresses (Pa).
    well_inclination_deg : float
        Angle from vertical to the well axis (deg). 0 = vertical
        well; 90 = horizontal well.
    well_azimuth_deg : float
        Angle from sigma_H direction to the horizontal projection
        of the well axis (deg, clockwise looking down). Only
        relevant when ``well_inclination_deg > 0``.
    azimuth_around_wall_deg : ndarray
        Azimuth grid around the borehole (deg, measured from the
        high-side X_W axis). Use ``np.linspace(0, 360, n,
        endpoint=False)`` for a full sweep.
    mud_pressure : float, default 0.0
        Wellbore pressure :math:`P_w` (Pa).
    poisson : float, default 0.25
        Poisson's ratio.

    Returns
    -------
    (sigma_theta, sigma_z, sigma_theta_z, sigma_r) : tuple of ndarrays
        Wall-stress components at each azimuth (Pa). Each array
        has the shape of ``azimuth_around_wall_deg``.

    See Also
    --------
    kirsch_wall_stresses : The vertical-well special case.
    inclined_breakout_pressure : The minimum-mud-pressure
        consumer of this primitive.

    References
    ----------
    * Hiramatsu, Y., & Oka, Y. (1962). Stress around a shaft or
      level excavated in ground with a three-dimensional stress
      state. *Memoirs of the Faculty of Engineering, Kyoto Univ.*
      24, 56-76.
    * Fairhurst, C. (1968). Methods of determining in-situ rock
      stresses at great depths. *Tech. Rept. ARSE 1-68*, US Army
      Corps of Engineers.
    * Fjaer, E., Holt, R. M., Horsrud, P., Raaen, A. M., & Risnes,
      R. (2008). *Petroleum Related Rock Mechanics*, 2nd ed.
      Elsevier, Chapter 4 and Box 4.1 (the rotation matrix
      parameterisation used here).
    """
    sigma_PS = np.diag([float(sigma_H), float(sigma_h), float(sigma_v)])
    R = _stress_rotation_to_well_frame(well_inclination_deg, well_azimuth_deg)
    sigma_W = R @ sigma_PS @ R.T

    s_xx = sigma_W[0, 0]
    s_yy = sigma_W[1, 1]
    s_zz_W = sigma_W[2, 2]
    s_xy = sigma_W[0, 1]
    s_xz = sigma_W[0, 2]
    s_yz = sigma_W[1, 2]

    theta = np.deg2rad(np.asarray(azimuth_around_wall_deg, dtype=float))
    cos_2t = np.cos(2.0 * theta)
    sin_2t = np.sin(2.0 * theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    sigma_theta = (
        s_xx + s_yy - 2.0 * (s_xx - s_yy) * cos_2t - 4.0 * s_xy * sin_2t - mud_pressure
    )
    sigma_z = (
        s_zz_W - 2.0 * poisson * (s_xx - s_yy) * cos_2t - 4.0 * poisson * s_xy * sin_2t
    )
    sigma_theta_z = 2.0 * (-s_xz * sin_t + s_yz * cos_t)
    sigma_r = np.full_like(sigma_theta, float(mud_pressure))

    return sigma_theta, sigma_z, sigma_theta_z, sigma_r


def _wall_principal_stresses(
    sigma_theta: np.ndarray,
    sigma_z: np.ndarray,
    sigma_theta_z: np.ndarray,
    sigma_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three principal stresses at each wall azimuth.

    The radial stress ``sigma_r`` is already principal (no shear in
    the r direction at the wall). The other two principal stresses
    are eigenvalues of the (theta, z) 2x2 sub-block:

        | sigma_theta   sigma_theta_z |
        | sigma_theta_z sigma_z       |

    Returns ``(sigma_r, lambda_plus, lambda_minus)`` where
    ``lambda_plus >= lambda_minus`` are the eigenvalues.
    """
    half_sum = 0.5 * (sigma_theta + sigma_z)
    half_diff = 0.5 * (sigma_theta - sigma_z)
    disc = np.sqrt(half_diff**2 + sigma_theta_z**2)
    return sigma_r, half_sum + disc, half_sum - disc


def inclined_breakout_pressure(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    pore_pressure: float,
    ucs: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
    poisson: float = 0.25,
    n_azimuth: int = 180,
) -> float:
    r"""
    Mohr-Coulomb breakout pressure for an arbitrarily inclined well.

    Generalises :func:`mohr_coulomb_breakout_pressure` (which
    closes the formula at the vertical-well breakout azimuth) to
    inclined wells where the failure azimuth around the borehole
    must be searched numerically. Returns the minimum mud pressure
    :math:`P_w^\mathrm{crit}` such that the Mohr-Coulomb failure
    condition :math:`\sigma_1' \le q\,\sigma_3' + \mathrm{UCS}` is
    satisfied at every azimuth around the wall.

    Algorithm:

    1. For a candidate :math:`P_w`, compute the four wall stresses
       (:func:`inclined_wellbore_wall_stresses`) at
       ``n_azimuth`` points around the borehole.
    2. Diagonalise the 2x2 :math:`(\theta, z)` sub-block to get
       the principal stresses at each azimuth (plus the trivial
       :math:`\sigma_{rr} = P_w` principal stress).
    3. Subtract :math:`\alpha P_p` for effective stresses.
    4. Compute the Mohr-Coulomb failure margin
       :math:`\Phi(\theta, P_w) = \sigma_1' - q\,\sigma_3' -
       \mathrm{UCS}` at each azimuth; take the maximum over
       :math:`\theta`. :math:`\Phi_\mathrm{max}(P_w) > 0` means
       the wall fails somewhere; :math:`\Phi_\mathrm{max}(P_w)
       \le 0` means stable.
    5. Bisect (``scipy.optimize.brentq``) for the smallest
       :math:`P_w` such that :math:`\Phi_\mathrm{max}(P_w) = 0`.

    For a vertical well (``well_inclination_deg = 0``), the result
    must agree with :func:`mohr_coulomb_breakout_pressure` to
    within the azimuth-grid resolution.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Principal stresses (Pa).
    pore_pressure : float
        Pore pressure (Pa).
    ucs : float
        Unconfined compressive strength (Pa).
    well_inclination_deg : float
        Well inclination from vertical (deg).
    well_azimuth_deg : float
        Well azimuth from sigma_H direction (deg).
    friction_angle_deg : float, default 30.0
        Internal friction angle (deg).
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.
    poisson : float, default 0.25
        Poisson's ratio (used for the sigma_zz wall-stress
        component, which depends on it).
    n_azimuth : int, default 180
        Number of azimuth-around-wall sample points (1-deg
        resolution by default). Increase for higher precision in
        the worst-azimuth search.

    Returns
    -------
    float
        Critical mud pressure (Pa). Mud pressures below this
        value trigger Mohr-Coulomb shear breakout somewhere on
        the wall.

    Raises
    ------
    ValueError
        If ``friction_angle_deg`` is not in ``(-90, 90)``; if
        ``n_azimuth`` is less than 8; or if the bracket search
        cannot find a sign change for ``brentq`` (typical cause:
        the wall is unconditionally stable or unconditionally
        unstable in the chosen geometry).

    See Also
    --------
    inclined_wellbore_wall_stresses : The wall-stress primitive.
    mohr_coulomb_breakout_pressure : The vertical-well closed-form
        special case.
    """
    if not (-90.0 < friction_angle_deg < 90.0):
        raise ValueError("friction_angle_deg must be in (-90, 90)")
    if n_azimuth < 8:
        raise ValueError("n_azimuth must be at least 8")

    phi = np.deg2rad(friction_angle_deg)
    sin_phi = np.sin(phi)
    q = (1.0 + sin_phi) / (1.0 - sin_phi)
    alpha_Pp = biot_alpha * pore_pressure
    azimuths = np.linspace(0.0, 360.0, n_azimuth, endpoint=False)

    def phi_max(P_w: float) -> float:
        s_t, s_z, s_tz, s_r = inclined_wellbore_wall_stresses(
            sigma_v,
            sigma_H,
            sigma_h,
            well_inclination_deg=well_inclination_deg,
            well_azimuth_deg=well_azimuth_deg,
            azimuth_around_wall_deg=azimuths,
            mud_pressure=P_w,
            poisson=poisson,
        )
        sr, l_p, l_m = _wall_principal_stresses(s_t, s_z, s_tz, s_r)
        principal = np.stack([sr - alpha_Pp, l_p - alpha_Pp, l_m - alpha_Pp])
        sigma_1 = np.max(principal, axis=0)
        sigma_3 = np.min(principal, axis=0)
        violation = sigma_1 - q * sigma_3 - ucs
        return float(np.max(violation))

    # Coarse-grid scan: phi_max(P_w) is positive in two regimes
    # (low P_w: shear breakout; very high P_w: tensile breakdown
    # -- our criterion treats negative sigma_3 as a Mohr-Coulomb
    # violation too) and negative in the stable window in between.
    # We want the FIRST sign change going from low to high P_w
    # -- the shear breakout boundary.
    P_max_scan = 1.5 * max(sigma_v, sigma_H, sigma_h)
    P_grid = np.linspace(0.0, P_max_scan, 50)
    phi_grid = np.array([phi_max(P) for P in P_grid])
    sign_changes = np.where(np.diff(np.sign(phi_grid)))[0]
    if len(sign_changes) == 0:
        if phi_grid[0] <= 0.0:
            # Wall is unconditionally stable at P_w = 0; no breakout.
            return 0.0
        raise ValueError(
            "Could not bracket a sign change of the Mohr-Coulomb failure "
            f"margin in P_w in [0, {P_max_scan:.2e}]. The wall is "
            "unconditionally unstable in the chosen geometry; the well "
            "cannot be drilled without casing or stress-state intervention."
        )
    i = int(sign_changes[0])
    return float(brentq(phi_max, P_grid[i], P_grid[i + 1], xtol=1.0e-3))


def inclined_breakdown_pressure(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    pore_pressure: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    tensile_strength: float = 0.0,
    biot_alpha: float = 1.0,
    poisson: float = 0.25,
    n_azimuth: int = 180,
) -> float:
    r"""
    Tensile-breakdown mud pressure for an arbitrarily inclined well.

    Generalises :func:`tensile_breakdown_pressure` (closed form at
    the vertical-well breakdown azimuth) to inclined wells where
    the failure azimuth must be searched numerically. Returns the
    maximum mud pressure :math:`P_w^\mathrm{break}` such that the
    minimum effective principal stress at every azimuth around the
    wall stays above :math:`-T` (no tensile failure).

    Algorithm:

    1. For a candidate :math:`P_w`, compute the four wall stresses
       (:func:`inclined_wellbore_wall_stresses`) at ``n_azimuth``
       points around the borehole.
    2. Diagonalise the 2x2 :math:`(\theta, z)` sub-block to get
       the two non-radial principal stresses (the smaller is
       :math:`\lambda_-`); the radial principal stress is
       :math:`\sigma_{rr} = P_w` itself.
    3. Subtract :math:`\alpha P_p` for effective stresses and take
       the minimum over all azimuths and over the three
       principal-stress candidates -- this is the worst-azimuth
       most-tensile effective stress.
    4. Tensile failure occurs when this minimum is below
       :math:`-T`. The breakdown pressure is the smallest
       :math:`P_w` at which that condition is just satisfied.

    Increasing :math:`P_w` always decreases :math:`\sigma_{\theta
    \theta}`, which can only decrease :math:`\lambda_-` (the
    smaller eigenvalue), so the worst-azimuth minimum eigenvalue
    is monotonically non-increasing in :math:`P_w`. This makes the
    breakdown bisection well-conditioned.

    For a vertical well (``well_inclination_deg = 0``), the
    result must agree with :func:`tensile_breakdown_pressure` to
    within the azimuth-grid resolution.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Principal stresses (Pa).
    pore_pressure : float
        Pore pressure (Pa).
    well_inclination_deg : float
        Well inclination from vertical (deg).
    well_azimuth_deg : float
        Well azimuth from sigma_H direction (deg).
    tensile_strength : float, default 0.0
        Tensile strength :math:`T` (Pa). Default 0 is the
        conservative (no tensile strength) case.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.
    poisson : float, default 0.25
        Poisson's ratio.
    n_azimuth : int, default 180
        Number of azimuth-around-wall sample points.

    Returns
    -------
    float
        Tensile-breakdown mud pressure (Pa). Mud pressures above
        this value trigger tensile failure / fracture initiation
        somewhere on the wall. May be negative if the wall is in
        tension at zero mud pressure -- in which case no positive
        mud pressure is "safe" and the well is undrillable in the
        chosen geometry without intervention.

    See Also
    --------
    inclined_wellbore_wall_stresses : The wall-stress primitive.
    inclined_breakout_pressure : The shear-failure lower bound.
    tensile_breakdown_pressure : The vertical-well closed-form
        special case.
    """
    if n_azimuth < 8:
        raise ValueError("n_azimuth must be at least 8")

    alpha_Pp = biot_alpha * pore_pressure
    azimuths = np.linspace(0.0, 360.0, n_azimuth, endpoint=False)

    def margin(P_w: float) -> float:
        """Margin against tensile failure: positive = stable.

        Convention follows the vertical-well
        :func:`tensile_breakdown_pressure`: tensile failure
        criterion applies to the smallest eigenvalue of the
        :math:`(\\theta, z)` 2x2 sub-block at the wall, NOT to
        the radial principal stress :math:`\\sigma_{rr} = P_w`.
        Including :math:`\\sigma_{rr}` would always trigger
        failure under positive pore pressure (since
        :math:`\\sigma_{rr,\\mathrm{eff}} = P_w - \\alpha P_p` is
        easily negative) and would not match the standard
        Hubbert-Willis fracture-initiation interpretation.
        """
        s_t, s_z, s_tz, s_r = inclined_wellbore_wall_stresses(
            sigma_v,
            sigma_H,
            sigma_h,
            well_inclination_deg=well_inclination_deg,
            well_azimuth_deg=well_azimuth_deg,
            azimuth_around_wall_deg=azimuths,
            mud_pressure=P_w,
            poisson=poisson,
        )
        _, _, l_m = _wall_principal_stresses(s_t, s_z, s_tz, s_r)
        # Worst-azimuth most-tensile eigenvalue of the (theta, z)
        # sub-block, after effective-stress correction.
        sigma_3 = float(np.min(l_m)) - alpha_Pp
        return sigma_3 + tensile_strength

    # margin(0) is typically positive (no tension at zero mud);
    # margin(very_high) is negative (tensile failure dominates).
    P_low = 0.0
    P_high = 2.0 * max(sigma_v, sigma_H, sigma_h, abs(pore_pressure))
    f_low = margin(P_low)
    f_high = margin(P_high)
    n_expand = 0
    while f_low * f_high > 0.0 and n_expand < 8:
        P_high *= 1.5
        f_high = margin(P_high)
        n_expand += 1
    if f_low <= 0.0:
        # Wall is in tension at zero mud pressure; no positive
        # mud weight prevents tensile failure. Return P_low (= 0)
        # as the boundary; callers should detect this via
        # MudWeightWindow.is_drillable when paired with a breakout
        # pressure.
        return 0.0
    if f_low * f_high > 0.0:
        raise ValueError(
            "Could not bracket a sign change of the tensile-failure "
            f"margin between P_w={P_low} and P_w={P_high}. The wall is "
            "either unconditionally stable in tension (no tensile "
            "breakdown possible) or there is a numerical issue."
        )
    return float(brentq(margin, P_low, P_high, xtol=1.0e-3))


def inclined_safe_mud_weight_window(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    pore_pressure: float,
    ucs: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    tensile_strength: float = 0.0,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
    poisson: float = 0.25,
    n_azimuth: int = 180,
) -> MudWeightWindow:
    r"""
    Both mud-weight bounds for an arbitrarily inclined well.

    Convenience wrapper that calls
    :func:`inclined_breakout_pressure` and
    :func:`inclined_breakdown_pressure` with consistent inputs and
    returns the two pressures bundled in a :class:`MudWeightWindow`
    (the same dataclass used by the vertical-well counterpart
    :func:`safe_mud_weight_window`).

    The "safe" mud-weight window is the closed interval
    ``[breakout_pressure, breakdown_pressure]``: pressures in this
    range avoid both shear failure (collapse) and tensile failure
    (lost circulation) at every azimuth around the inclined
    borehole. ``MudWeightWindow.is_drillable`` flags whether such
    a window exists.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Principal stresses (Pa).
    pore_pressure : float
        Pore pressure (Pa).
    ucs : float
        Unconfined compressive strength (Pa). Drives the breakout
        bound.
    well_inclination_deg, well_azimuth_deg : float
        Well orientation.
    tensile_strength : float, default 0.0
        Tensile strength (Pa). Drives the breakdown bound.
    friction_angle_deg : float, default 30.0
        Internal friction angle (deg).
    biot_alpha : float, default 1.0
        Biot coefficient.
    poisson : float, default 0.25
        Poisson's ratio.
    n_azimuth : int, default 180
        Azimuth-around-wall scan resolution.

    Returns
    -------
    MudWeightWindow
        Dataclass with ``breakout_pressure`` and
        ``breakdown_pressure`` (both 0-D arrays of float for
        scalar inputs) plus ``width`` and ``is_drillable``
        properties. Note that for inclined wells the bounds are
        always scalar; vector inputs are not supported here
        because the per-orientation worst-azimuth scan does not
        vectorise across depths cleanly. Loop in the caller for
        per-depth inclined-well analysis.

    See Also
    --------
    safe_mud_weight_window : The vertical-well counterpart, which
        does vectorise across depth arrays.
    inclined_breakout_pressure : The lower-bound primitive.
    inclined_breakdown_pressure : The upper-bound primitive.
    """
    P_breakout = inclined_breakout_pressure(
        sigma_v,
        sigma_H,
        sigma_h,
        pore_pressure,
        ucs,
        well_inclination_deg=well_inclination_deg,
        well_azimuth_deg=well_azimuth_deg,
        friction_angle_deg=friction_angle_deg,
        biot_alpha=biot_alpha,
        poisson=poisson,
        n_azimuth=n_azimuth,
    )
    P_breakdown = inclined_breakdown_pressure(
        sigma_v,
        sigma_H,
        sigma_h,
        pore_pressure,
        well_inclination_deg=well_inclination_deg,
        well_azimuth_deg=well_azimuth_deg,
        tensile_strength=tensile_strength,
        biot_alpha=biot_alpha,
        poisson=poisson,
        n_azimuth=n_azimuth,
    )
    return MudWeightWindow(
        breakout_pressure=np.asarray(P_breakout, dtype=float),
        breakdown_pressure=np.asarray(P_breakdown, dtype=float),
    )
