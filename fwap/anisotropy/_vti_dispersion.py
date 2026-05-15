"""
VTI elastic tensors and dispersion: Backus averaging of
layered isotropic stacks into an effective VTI tensor, plus
Christoffel phase and group velocities (qP, qSV, SH).

References
----------
* Backus, G. E. (1962). Long-wave elastic anisotropy
  produced by horizontal layering. *J. Geophys. Res.* 67(11),
  4427-4440.
* Tsvankin, I. (2001). *Seismic signatures and analysis of
  reflection data in anisotropic media*. Pergamon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BackusResult:
    r"""
    Effective VTI elastic tensor from Backus (1962) averaging.

    Output of :func:`backus_average`. The five independent VTI
    elastic constants in Voigt notation, plus the volume-weighted
    effective density. The symmetry axis is :math:`x_3` (vertical),
    matching the standard VTI convention used elsewhere in
    ``fwap.anisotropy``.

    Layer-parallel components ``c11`` and ``c66`` are arithmetic
    volume averages (Voigt-like upper bounds); layer-perpendicular
    components ``c33`` and ``c44`` are harmonic volume averages
    (Reuss-like lower bounds). The cross-coupling component ``c13``
    is the standard Backus combination of ``lambda / (lambda + 2 mu)``
    weighted averages.

    Attributes
    ----------
    c11 : float
        In-plane P-wave modulus :math:`\rho V_{P,h}^2` (Pa) for
        propagation parallel to the layering.
    c13 : float
        Off-axis cross-coupling elastic constant (Pa).
    c33 : float
        Vertical P-wave modulus :math:`\rho V_P^2` (Pa) for
        propagation perpendicular to the layering.
    c44 : float
        Vertical shear modulus :math:`\rho V_{Sv}^2` (Pa); SV-wave
        with vertical propagation.
    c66 : float
        Horizontal shear modulus :math:`\rho V_{Sh}^2` (Pa); SH-wave
        with horizontal propagation.
    rho : float
        Volume-weighted effective density (kg/m^3).
    """

    c11: float
    c13: float
    c33: float
    c44: float
    c66: float
    rho: float


def backus_average(
    thickness: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> BackusResult:
    r"""
    Backus (1962) long-wavelength average of a layered isotropic stack.

    Homogenises a sequence of N isotropic layers into a single
    transversely-isotropic (VTI) effective medium with vertical
    symmetry axis. Valid in the long-wavelength limit
    (wavelength :math:`\gg` total stack thickness); typical use
    is upscaling thinly-bedded sonic-log intervals to seismic
    resolution.

    Per-layer Lame parameters are computed from the inputs:

    .. math::

        \mu_i &= \rho_i\,V_{S,i}^2,
        \\
        M_i &= \rho_i\,V_{P,i}^2 \;=\; \lambda_i + 2\mu_i,
        \\
        \lambda_i &= M_i - 2\mu_i.

    Volume fractions :math:`\phi_i = h_i / \sum_j h_j` weight the
    arithmetic and harmonic averages :math:`\langle X \rangle =
    \sum_i \phi_i X_i`. The five effective VTI elastic constants
    are (Backus 1962; Mavko et al. 2009 Section 1.5):

    .. math::

        C_{33} &= 1 \,/\, \langle 1/M \rangle,
        \\
        C_{13} &= \langle \lambda/M \rangle \;\big/\;
                  \langle 1/M \rangle,
        \\
        C_{11} &= \langle M - \lambda^2/M \rangle
                  + \langle \lambda/M \rangle^2 \;\big/\;
                    \langle 1/M \rangle,
        \\
        C_{44} &= 1 \,/\, \langle 1/\mu \rangle,
        \\
        C_{66} &= \langle \mu \rangle.

    The effective density is the arithmetic volume average
    :math:`\rho_\mathrm{eff} = \langle \rho \rangle`.

    Parameters
    ----------
    thickness : ndarray, shape (n_layers,)
        Per-layer thickness (m). Must be strictly positive (zero-
        thickness layers are not allowed; drop them upstream). The
        absolute scale does not matter; only volume fractions
        ``thickness / sum(thickness)`` enter the result.
    vp : ndarray, shape (n_layers,)
        Per-layer P-wave velocity (m/s). Strictly positive.
    vs : ndarray, shape (n_layers,)
        Per-layer S-wave velocity (m/s). Strictly positive and less
        than the corresponding ``vp``.
    rho : ndarray, shape (n_layers,)
        Per-layer mass density (kg/m^3). Strictly positive.

    Returns
    -------
    BackusResult
        Five independent VTI elastic constants (Pa) plus the
        volume-weighted effective density (kg/m^3). Use
        :func:`thomsen_gamma` on ``c44, c66`` for shear anisotropy
        and the standard Thomsen formulas on the full set
        (``epsilon = (c11 - c33) / (2 c33)``,
        ``delta = ((c13 + c44)^2 - (c33 - c44)^2) /
        (2 c33 (c33 - c44))``) for the full Thomsen triple.

    Raises
    ------
    ValueError
        If any input array is empty, has shape mismatching the
        others, or contains a non-positive value; or if any
        ``vs >= vp`` (the isotropic-layer constraint that keeps
        :math:`\lambda + 2\mu > 0` and :math:`\mu > 0`).

    Notes
    -----
    Long-wavelength regime: the Backus average represents the
    layered stack as a *single* effective TI medium. It is exact
    for vertically-propagating waves whose wavelength is much
    larger than the stack thickness; for waves with wavelength
    comparable to layer thicknesses, the stack acts as a periodic
    medium with dispersion (Bragg scattering) and Backus is no
    longer applicable.

    Layer-parallel vs layer-perpendicular limits:

    * ``C_{66} = \langle \mu \rangle`` is an arithmetic volume
      average (Voigt-like upper bound). The SH wave parallel to
      the layering experiences the *stiffest* bulk-mu pathway.
    * ``C_{44} = 1 / \langle 1/\mu \rangle`` is a harmonic volume
      average (Reuss-like lower bound). The SV wave vertical to
      the layering experiences the *most-compliant* path.
    * The Voigt-Reuss inequality :math:`C_{66} \ge C_{44}`
      always holds with equality iff every layer has the same
      :math:`\mu` -- i.e. ``gamma >= 0`` always for any layered
      stack of isotropic layers, with ``gamma = 0`` only in the
      degenerate identical-layer case. This is one consequence
      that the test suite checks.

    For an isotropic stack (all layers identical), the result
    reduces to the per-layer isotropic moduli:
    ``C_{11} = C_{33} = lambda + 2 mu``,
    ``C_{13} = lambda``, ``C_{44} = C_{66} = mu``.

    See Also
    --------
    thomsen_gamma : Thomsen :math:`\gamma` from ``c44, c66``.
    vti_moduli_from_logs : Per-depth VTI moduli from a sonic +
        density log (the inverse direction: log -> moduli).
    fwap.rockphysics.reuss_average : Isotropic Reuss bound (the
        layer-perpendicular average direction in spirit, but
        applied to bulk modulus rather than the full tensor).
    fwap.rockphysics.voigt_average : Isotropic Voigt bound
        (analogous to ``c66``).

    References
    ----------
    * Backus, G. E. (1962). Long-wave elastic anisotropy produced
      by horizontal layering. *J. Geophys. Res.* 67(11),
      4427-4440.
    * Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock
      Physics Handbook*, 2nd ed., Section 1.5. Cambridge
      University Press.
    * Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics*
      51(10), 1954-1966 (Thomsen-parameter conventions used by
      callers of this function).
    """
    h = np.asarray(thickness, dtype=float)
    Vp = np.asarray(vp, dtype=float)
    Vs = np.asarray(vs, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)

    if h.ndim != 1:
        raise ValueError("thickness must be 1-D")
    if Vp.shape != h.shape or Vs.shape != h.shape or rho_arr.shape != h.shape:
        raise ValueError(
            "thickness, vp, vs, rho must all be 1-D arrays of the same length"
        )
    if h.size == 0:
        raise ValueError("at least one layer required")
    if np.any(h <= 0):
        raise ValueError("thickness must be strictly positive")
    if np.any(Vp <= 0) or np.any(Vs <= 0) or np.any(rho_arr <= 0):
        raise ValueError("vp, vs, rho must all be strictly positive")
    if np.any(Vs >= Vp):
        raise ValueError("require vs < vp on every layer")

    phi = h / np.sum(h)

    mu = rho_arr * Vs**2
    M = rho_arr * Vp**2  # lambda + 2 mu
    lam = M - 2.0 * mu

    inv_M = 1.0 / M
    inv_mu = 1.0 / mu
    lam_over_M = lam / M
    lam_sq_over_M = lam * lam_over_M  # = lambda^2 / M

    avg_inv_M = float(np.sum(phi * inv_M))
    avg_inv_mu = float(np.sum(phi * inv_mu))
    avg_lam_over_M = float(np.sum(phi * lam_over_M))
    avg_M_minus_lam_sq_over_M = float(np.sum(phi * (M - lam_sq_over_M)))
    avg_mu = float(np.sum(phi * mu))
    avg_rho = float(np.sum(phi * rho_arr))

    c33 = 1.0 / avg_inv_M
    c13 = avg_lam_over_M / avg_inv_M
    c11 = avg_M_minus_lam_sq_over_M + (avg_lam_over_M**2) / avg_inv_M
    c44 = 1.0 / avg_inv_mu
    c66 = avg_mu

    return BackusResult(
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=avg_rho,
    )


# ---------------------------------------------------------------------
# VTI phase velocities (Tsvankin 2001 / Christoffel)
# ---------------------------------------------------------------------


def vti_phase_velocities(
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    *,
    phase_angle_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Phase velocities of the three modes in a VTI medium.

    Christoffel-determinant solution for the phase velocities of
    the three plane-wave modes (quasi-P, quasi-SV, SH) in a
    transversely-isotropic medium with vertical symmetry axis,
    propagating at phase angle :math:`\theta` from the symmetry
    axis (so :math:`\theta = 0` is vertical propagation,
    :math:`\theta = \pi/2` is horizontal). For the standard VTI
    plane (:math:`x_1`-:math:`x_3`) the in-plane qP and qSV modes
    decouple from the out-of-plane SH mode; the qP and qSV
    velocities are the two roots of the quadratic Christoffel
    determinant in :math:`v^2`.

    Tsvankin (2001) eq. 1.41:

    .. math::

        v_{qP}^{\,2}(\theta), v_{qSV}^{\,2}(\theta) \;=\;
            \frac{1}{2\rho}\bigg[
                (C_{11} + C_{44})\sin^2\theta
                + (C_{33} + C_{44})\cos^2\theta
                \pm \sqrt{D(\theta)}
            \bigg],

    where the discriminant is

    .. math::

        D(\theta) \;=\;
            \big[(C_{11} - C_{44})\sin^2\theta
                 - (C_{33} - C_{44})\cos^2\theta\big]^2
            \;+\; 4\,(C_{13} + C_{44})^2\,\sin^2\theta\,\cos^2\theta.

    The plus sign gives qP; the minus sign gives qSV. The SH mode
    is decoupled and has the simpler form

    .. math::

        v_{SH}^{\,2}(\theta) \;=\;
            \frac{C_{44}\cos^2\theta + C_{66}\sin^2\theta}{\rho}.

    Limit checks (verified by the test suite):

    * Vertical propagation :math:`(\theta = 0)`:
      :math:`v_{qP} = \sqrt{C_{33}/\rho}`,
      :math:`v_{qSV} = v_{SH} = \sqrt{C_{44}/\rho}` (vertical
      shear velocities degenerate).
    * Horizontal propagation :math:`(\theta = \pi/2)`:
      :math:`v_{qP} = \sqrt{C_{11}/\rho}`,
      :math:`v_{qSV} = \sqrt{C_{44}/\rho}`,
      :math:`v_{SH} = \sqrt{C_{66}/\rho}`.
    * Isotropic limit (:math:`C_{11} = C_{33}`, :math:`C_{44} =
      C_{66}`, :math:`C_{13} = C_{11} - 2C_{44}`):
      :math:`v_{qP} = \sqrt{(C_{33})/\rho}` for all
      :math:`\theta`, and :math:`v_{qSV} = v_{SH}` for all
      :math:`\theta`.

    Parameters
    ----------
    c11, c13, c33, c44, c66 : float
        The five independent VTI elastic constants (Pa). The
        natural source is :func:`backus_average` for layered
        media, or :func:`vti_moduli_from_logs` for sonic-derived
        per-depth values.
    rho : float
        Mass density (kg/m^3).
    phase_angle_rad : scalar or ndarray
        Phase angle :math:`\theta` (radians) measured from the
        symmetry axis (vertical). Use ``np.linspace(0, np.pi/2,
        91)`` for a 1-degree grid over a quadrant.

    Returns
    -------
    (v_qP, v_qSV, v_SH) : tuple of ndarrays
        Phase velocities (m/s) of the three modes at each input
        angle, broadcast to the shape of ``phase_angle_rad``.

    Raises
    ------
    ValueError
        If ``rho`` is non-positive; if any elastic constant is
        non-positive; or if the qSV discriminant goes negative
        (would indicate non-physical input violating the strong-
        ellipticity constraint of the VTI tensor).

    See Also
    --------
    backus_average : Computes the five VTI elastic constants from
        a layered isotropic stack -- the natural input for this
        function.
    vti_moduli_from_logs : Sonic-derived per-depth VTI moduli.
    flexural_dispersion_vti_physical : Borehole-flexural-mode
        dispersion in a VTI formation. Uses a different
        velocity-based parameterisation (``vsv, vsh``).

    Notes
    -----
    The function returns *phase* velocities (the speed of a
    constant-phase plane). Group velocities (the speed of energy
    propagation, which is what determines wavefront shapes) follow
    from the phase velocities by

    .. math::

        v_g(\theta) = \sqrt{v_p^{\,2}(\theta)
                            + (\partial v_p/\partial\theta)^2},

    with the group angle :math:`\psi` given by
    :math:`\tan\psi = \tan\theta + (1/v_p)\,(\partial v_p
    /\partial\theta)\,/\,(1 - \tan\theta\,(1/v_p)\,
    (\partial v_p/\partial\theta))`. Group-velocity calculation
    is a planned follow-up.

    The decoupling of SH from qP/qSV is specific to propagation
    in the symmetry plane (:math:`x_1`-:math:`x_3`); for off-plane
    propagation the SH mode mixes with qSV. This function assumes
    :math:`\phi = 0` (in-plane propagation), which is the
    convention used throughout :mod:`fwap.anisotropy`.

    References
    ----------
    * Tsvankin, I. (2001). *Seismic Signatures and Analysis of
      Reflection Data in Anisotropic Media.* Pergamon, eq. 1.41.
    * Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics*
      51(10), 1954-1966.
    * Carcione, J. M. (2014). *Wave Fields in Real Media*, 3rd
      ed., Section 1.4. Elsevier (Christoffel-determinant
      derivation in standard form).
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    for name, val in [
        ("c11", c11),
        ("c33", c33),
        ("c44", c44),
        ("c66", c66),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be positive")

    theta = np.asarray(phase_angle_rad, dtype=float)
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2

    # Tsvankin 2001 eq. 1.41
    inner = ((c11 - c44) * sin2 - (c33 - c44) * cos2) ** 2 + 4.0 * (
        c13 + c44
    ) ** 2 * sin2 * cos2
    if np.any(inner < 0.0):
        raise ValueError(
            "Christoffel discriminant went negative; check that "
            "the VTI elastic constants satisfy strong ellipticity "
            "(C33 * C11 - C13^2 > 0, etc.)."
        )
    disc = np.sqrt(inner)

    sum_term = (c11 + c44) * sin2 + (c33 + c44) * cos2
    v_qP = np.sqrt((sum_term + disc) / (2.0 * rho))
    v_qSV = np.sqrt((sum_term - disc) / (2.0 * rho))
    v_SH = np.sqrt((c44 * cos2 + c66 * sin2) / rho)

    return v_qP, v_qSV, v_SH


@dataclass
class VtiGroupVelocities:
    r"""
    Output of :func:`vti_group_velocities`.

    Group velocity (the speed of energy / wavefront propagation)
    and group angle (the direction of energy propagation, generally
    different from the phase-angle direction in anisotropic media)
    for the three VTI modes, on the same phase-angle grid the
    function was called with.

    Attributes
    ----------
    v_qP, v_qSV, v_SH : ndarray
        Group velocities (m/s) for the three modes at each phase
        angle in the input grid.
    psi_qP, psi_qSV, psi_SH : ndarray
        Group angles (radians, measured from the vertical
        :math:`x_3` axis) for the three modes. In an isotropic
        medium ``psi == phase_angle_rad``; in an anisotropic
        medium they differ except at the symmetry-aligned angles
        :math:`\theta = 0, \pi/2` where the wavefront is locally
        aligned with the symmetry direction.
    """

    v_qP: np.ndarray
    v_qSV: np.ndarray
    v_SH: np.ndarray
    psi_qP: np.ndarray
    psi_qSV: np.ndarray
    psi_SH: np.ndarray


def vti_group_velocities(
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    *,
    phase_angle_rad: np.ndarray,
) -> VtiGroupVelocities:
    r"""
    Group velocities and group angles for the three VTI modes.

    Group velocity (the speed of energy propagation, which
    determines wavefront shapes) follows from the phase velocity
    and its derivative with respect to phase angle:

    .. math::

        v_{g,x}(\theta) &= v_p(\theta) \sin\theta
                          + \frac{\partial v_p}{\partial\theta}
                            \cos\theta,
        \\
        v_{g,z}(\theta) &= v_p(\theta) \cos\theta
                          - \frac{\partial v_p}{\partial\theta}
                            \sin\theta,
        \\
        |v_g(\theta)| &= \sqrt{v_p^{\,2}(\theta)
                               + \big(\partial v_p / \partial\theta\big)^2},

    with the group angle :math:`\psi` (measured from the symmetry
    axis) given by :math:`\tan\psi = v_{g,x} / v_{g,z}`. In an
    isotropic medium :math:`\partial v_p / \partial\theta = 0`,
    so :math:`v_g = v_p` and :math:`\psi = \theta`.

    The :math:`\partial v_p / \partial\theta` derivative is
    computed numerically via :func:`numpy.gradient` (central
    differences in the interior, one-sided at the grid endpoints).
    This is conventional for wavefront-plotting workflows and
    avoids the algebraic complexity of the analytic Tsvankin-2001
    closed form. For high-accuracy boundary derivatives at
    :math:`\theta = 0` and :math:`\pi/2`, supply a grid with
    additional points slightly outside the quadrant of interest.

    Parameters
    ----------
    c11, c13, c33, c44, c66 : float
        Five independent VTI elastic constants (Pa).
    rho : float
        Mass density (kg/m^3).
    phase_angle_rad : ndarray, shape (n,)
        Strictly-increasing 1-D phase-angle grid (radians) with at
        least two points. Typical use:
        ``np.linspace(0, np.pi/2, 91)``.

    Returns
    -------
    VtiGroupVelocities
        Dataclass with the three group velocities and three group
        angles, all the same shape as ``phase_angle_rad``.

    Raises
    ------
    ValueError
        If ``phase_angle_rad`` has fewer than two points or is not
        strictly increasing; otherwise the same physical-input
        validations as :func:`vti_phase_velocities`.

    See Also
    --------
    vti_phase_velocities : The phase-velocity primitive this
        function differentiates.

    Notes
    -----
    The :math:`v_g(\theta), \psi(\theta)` relation is implicit:
    the group velocity is parameterised by the *phase* angle, not
    by the group angle directly. Plotting the wavefront in
    Cartesian coordinates uses the group angle for direction and
    the group magnitude for distance:

    .. code-block:: python

        out = vti_group_velocities(...)
        x = out.v_qP * np.sin(out.psi_qP)
        z = out.v_qP * np.cos(out.psi_qP)

    For a unit-time wavefront, this gives the Cartesian
    coordinates of the wavefront tip; sample finely enough in
    :math:`\theta` to resolve any qSV cuspidal triplications,
    which can occur in strongly-anellipsoidal VTI media.

    References
    ----------
    * Tsvankin, I. (2001). *Seismic Signatures and Analysis of
      Reflection Data in Anisotropic Media.* Pergamon, Section
      1.3 (group velocity and group angle in anisotropic media).
    * Carcione, J. M. (2014). *Wave Fields in Real Media*, 3rd
      ed., Section 1.4. Elsevier.
    """
    theta = np.asarray(phase_angle_rad, dtype=float)
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError("phase_angle_rad must be a 1-D array with at least 2 points")
    if not np.all(np.diff(theta) > 0):
        raise ValueError("phase_angle_rad must be strictly increasing")

    v_qP, v_qSV, v_SH = vti_phase_velocities(
        c11,
        c13,
        c33,
        c44,
        c66,
        rho,
        phase_angle_rad=theta,
    )

    dv_qP = np.gradient(v_qP, theta)
    dv_qSV = np.gradient(v_qSV, theta)
    dv_SH = np.gradient(v_SH, theta)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    def _group(v_p: np.ndarray, dv_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v_g_x = v_p * sin_t + dv_p * cos_t
        v_g_z = v_p * cos_t - dv_p * sin_t
        v_g = np.hypot(v_g_x, v_g_z)
        # arctan2 with (x, z) gives the angle measured from z, i.e.
        # from the vertical symmetry axis -- the standard "group
        # angle from symmetry" convention.
        psi = np.arctan2(v_g_x, v_g_z)
        return v_g, psi

    v_g_P, psi_P = _group(v_qP, dv_qP)
    v_g_SV, psi_SV = _group(v_qSV, dv_qSV)
    v_g_SH, psi_SH = _group(v_SH, dv_SH)

    return VtiGroupVelocities(
        v_qP=v_g_P,
        v_qSV=v_g_SV,
        v_SH=v_g_SH,
        psi_qP=psi_P,
        psi_qSV=psi_SV,
        psi_SH=psi_SH,
    )
