"""
VTI elastic-moduli inversion: vertical-well summary
(C33, C44, C66, gamma from monopole P + dipole S + Stoneley
low-f) and walkaway-VSP slowness-polarization inversion for
Thomsen epsilon and delta.

References
----------
* Tsvankin, I. (2001). *Seismic signatures and analysis of
  reflection data in anisotropic media*. Pergamon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fwap.anisotropy._thomsen import (
    stoneley_horizontal_shear_modulus_corrected,
    thomsen_gamma,
    thomsen_gamma_from_logs,
)


def c33_from_p_pick(
    slowness_p: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    r"""
    Vertical P-wave modulus :math:`C_{33}` from monopole P slowness
    and bulk density.

    For a vertical well in a (possibly transversely-isotropic)
    formation, the monopole-derived compressional head wave samples
    the **vertically-incident** P-wave modulus

    .. math::

        C_{33} \;=\; \rho \, V_P^{\,2}
                \;=\; \rho \,/\, S_P^{\,2}.

    Combined with :math:`C_{44}` (from the dipole shear log) and
    :math:`C_{66}` (from the Stoneley low-frequency tube wave via
    :func:`stoneley_horizontal_shear_modulus`), this gives the three
    of the five VTI elastic constants that a vertical-well sonic
    acquisition can recover. The remaining two (:math:`C_{11}`,
    :math:`C_{13}`) need horizontal-P or off-axis-S measurements
    (walkaway VSP, cross-well, oblique-incidence VSP).

    Parameters
    ----------
    slowness_p : ndarray or float
        Per-depth monopole P slowness (s/m). Typically the
        ``"P"`` mode slowness from :func:`fwap.picker.pick_modes`.
        Must be strictly positive.
    rho : ndarray or float
        Per-depth formation bulk density (kg/m^3) from the bulk-
        density log (typically the ``RHOB`` curve). Must be
        strictly positive.

    Returns
    -------
    ndarray
        :math:`C_{33}` (Pa), broadcast to the common shape of the
        inputs.

    Raises
    ------
    ValueError
        If any input is non-positive.

    See Also
    --------
    vti_moduli_from_logs : Bundles C33 + C44 + C66 + gamma in one
        call from monopole P + dipole S + Stoneley + density.
    """
    s_p = np.asarray(slowness_p, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    if np.any(s_p <= 0):
        raise ValueError("slowness_p must be strictly positive")
    if np.any(rho_arr <= 0):
        raise ValueError("rho must be strictly positive")
    return rho_arr / (s_p * s_p)


@dataclass
class VtiModuli:
    r"""
    Output of :func:`vti_moduli_from_logs`.

    The three off-diagonal elastic constants a vertical-well sonic
    + density acquisition can recover, plus the corresponding
    velocities and the Thomsen shear-anisotropy parameter.

    The remaining two Thomsen parameters
    :math:`\epsilon = (C_{11} - C_{33}) / (2 C_{33})` and
    :math:`\delta = ((C_{13} + C_{44})^2 - (C_{33} - C_{44})^2)
    / (2 C_{33} (C_{33} - C_{44}))` are *not* fields here -- they
    cannot be recovered from a single vertical-well sonic record
    and need horizontal-P or off-axis-S measurements (walkaway VSP,
    cross-well, oblique-incidence VSP).

    Attributes
    ----------
    c33 : ndarray
        Vertical P-wave modulus :math:`\rho V_P^2` (Pa). From the
        monopole P pick.
    c44 : ndarray
        Vertical shear modulus :math:`\rho V_{Sv}^2` (Pa). From the
        dipole shear log.
    c66 : ndarray
        Horizontal shear modulus (Pa). From the Stoneley low-
        frequency tube-wave inversion (White 1983 / Norris 1990).
    gamma : ndarray
        Thomsen shear-anisotropy parameter
        :math:`\gamma = (C_{66} - C_{44}) / (2 C_{44})`. ``0`` for
        an isotropic formation; positive for typical VTI shales.
    vp : ndarray
        Vertical P-wave velocity :math:`V_P = \sqrt{C_{33}/\rho}`
        (m/s).
    vsv : ndarray
        Vertical shear velocity :math:`V_{Sv} = \sqrt{C_{44}/\rho}`
        (m/s).
    vsh : ndarray
        Horizontal shear velocity :math:`V_{Sh} = \sqrt{C_{66}/\rho}`
        (m/s). Equal to :math:`V_{Sv}` for an isotropic formation;
        :math:`V_{Sh} > V_{Sv}` for typical VTI shales.
    """

    c33: np.ndarray
    c44: np.ndarray
    c66: np.ndarray
    gamma: np.ndarray
    vp: np.ndarray
    vsv: np.ndarray
    vsh: np.ndarray


def vti_moduli_from_logs(
    slowness_p: np.ndarray,
    slowness_dipole: np.ndarray,
    slowness_stoneley: np.ndarray,
    rho: np.ndarray,
    *,
    rho_fluid: float,
    v_fluid: float,
    correct_for_p_modulus: bool = True,
) -> VtiModuli:
    r"""
    Vertical-well VTI elastic-moduli summary from a sonic + density
    log set.

    One-call wrapper that combines:

    - :math:`C_{33} = \rho V_P^2` from the monopole P slowness via
      :func:`c33_from_p_pick`;
    - :math:`C_{44} = \rho V_{Sv}^2` from the dipole shear slowness;
    - :math:`C_{66}` from the Stoneley low-frequency tube wave -- by
      default the Tang & Cheng (2004) §5.4 form via
      :func:`stoneley_horizontal_shear_modulus_corrected`, falling
      back to the rigid-formation White (1983) form via
      :func:`stoneley_horizontal_shear_modulus` when
      ``correct_for_p_modulus=False``;
    - the Thomsen shear-anisotropy parameter
      :math:`\gamma = (C_{66} - C_{44}) / (2 C_{44})` via
      :func:`thomsen_gamma`;
    - the corresponding vertical and horizontal shear / compressional
      velocities :math:`V_P`, :math:`V_{Sv}`, :math:`V_{Sh}`.

    For an isotropic formation :math:`V_{Sh} = V_{Sv}` and
    :math:`\gamma = 0`; positive :math:`\gamma` (and
    :math:`V_{Sh} > V_{Sv}`) flags VTI behaviour. The Workflow-3
    deliverable in Mari et al. (1994), Part 3 lists this triple --
    "shear anisotropy, mechanical properties and fracture
    indicators from the flexural wave" -- as the dipole-sonic
    output; this wrapper produces the shear-anisotropy half of it
    in one call.

    Out of scope
    ------------
    The remaining two Thomsen parameters
    (:math:`\epsilon, \delta`) need horizontal-P or off-axis-S
    measurements that a vertical-well sonic acquisition cannot
    provide. Walkaway-VSP or cross-well processing is the standard
    route; both are outside fwap's scope today and are flagged in
    :file:`docs/roadmap_old.m`.

    Parameters
    ----------
    slowness_p : ndarray
        Per-depth monopole P slowness (s/m).
    slowness_dipole : ndarray
        Per-depth dipole shear slowness (s/m).
    slowness_stoneley : ndarray
        Per-depth low-frequency Stoneley slowness (s/m).
    rho : ndarray
        Per-depth formation bulk density (kg/m^3).
    rho_fluid : float
        Borehole-fluid density (kg/m^3).
    v_fluid : float
        Borehole-fluid acoustic velocity (m/s).
    correct_for_p_modulus : bool, default True
        Apply the Tang & Cheng (2004) §5.4 finite-formation-impedance
        correction on the Stoneley → :math:`C_{66}` inversion.
        Recommended (and the default) because the monopole P pick
        and density log are already required arguments. Pass
        ``False`` to recover the literal White (1983) reading; the
        :math:`\gamma` returned in that mode matches
        :func:`thomsen_gamma_from_logs` exactly. The two modes
        typically differ by 5–15 % in :math:`C_{66}` (and
        correspondingly in :math:`\gamma`) -- larger in slow VTI
        shales, smaller in fast carbonates.

    Returns
    -------
    VtiModuli
        ``c33``, ``c44``, ``c66`` (Pa), ``gamma`` (-), and the
        derived ``vp``, ``vsv``, ``vsh`` (m/s); all per-depth and
        broadcast to the common input shape.

    Raises
    ------
    ValueError
        Same conditions as :func:`c33_from_p_pick`,
        :func:`thomsen_gamma_from_logs`, and -- when
        ``correct_for_p_modulus=True`` --
        :func:`stoneley_horizontal_shear_modulus_corrected`
        (additionally requires the formation P-wave modulus to
        exceed the fluid bulk modulus).

    See Also
    --------
    thomsen_gamma_from_logs :
        Returns just the (C44, C66, gamma) triple when the monopole
        P pick or density log isn't available; uses the uncorrected
        White (1983) C66 inversion exclusively.
    stoneley_horizontal_shear_modulus_corrected :
        The Tang & Cheng (2004) C66 inversion used here when
        ``correct_for_p_modulus=True``.
    fwap.geomechanics.geomechanics_indices :
        Companion one-call wrapper for the geomechanical indices
        (brittleness, fracability, UCS, closure stress, sand
        stability) on top of :class:`~fwap.rockphysics.ElasticModuli`.
    """
    s_p = np.asarray(slowness_p, dtype=float)
    s_d = np.asarray(slowness_dipole, dtype=float)
    s_st = np.asarray(slowness_stoneley, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    c33 = c33_from_p_pick(s_p, rho_arr)
    if correct_for_p_modulus:
        if np.any(s_d <= 0):
            raise ValueError("slowness_dipole must be strictly positive")
        c44 = rho_arr / (s_d * s_d)
        c66 = stoneley_horizontal_shear_modulus_corrected(
            slowness_stoneley=s_st,
            rho=rho_arr,
            slowness_p=s_p,
            rho_fluid=rho_fluid,
            v_fluid=v_fluid,
        )
        gamma = thomsen_gamma(c44, c66)
        vp = np.sqrt(c33 / rho_arr)
        vsv = np.sqrt(c44 / rho_arr)
        vsh = np.sqrt(c66 / rho_arr)
        return VtiModuli(
            c33=c33,
            c44=c44,
            c66=c66,
            gamma=gamma,
            vp=vp,
            vsv=vsv,
            vsh=vsh,
        )
    gamma_res = thomsen_gamma_from_logs(
        slowness_dipole=s_d,
        slowness_stoneley=s_st,
        rho=rho_arr,
        rho_fluid=rho_fluid,
        v_fluid=v_fluid,
    )
    vp = np.sqrt(c33 / rho_arr)
    vsv = np.sqrt(gamma_res.c44 / rho_arr)
    vsh = np.sqrt(gamma_res.c66 / rho_arr)
    return VtiModuli(
        c33=c33,
        c44=gamma_res.c44,
        c66=gamma_res.c66,
        gamma=gamma_res.gamma,
        vp=vp,
        vsv=vsv,
        vsh=vsh,
    )


# ---------------------------------------------------------------------
# Thomsen epsilon / delta from walkaway-VSP slowness-polarization
# inversion (Tier 2 VTI roadmap)
# ---------------------------------------------------------------------


@dataclass
class ThomsenEpsilonDeltaResult:
    r"""
    Output of :func:`thomsen_epsilon_delta_from_walkaway_vsp`.

    Attributes
    ----------
    epsilon : float
        Thomsen P-wave anisotropy parameter
        :math:`\epsilon = (C_{11} - C_{33}) / (2 C_{33})`. ``0`` for
        an isotropic formation; positive for typical VTI shales
        (horizontal P faster than vertical P).
    delta : float
        Thomsen near-vertical anisotropy parameter
        :math:`\delta = ((C_{13} + C_{44})^2 - (C_{33} - C_{44})^2)
        / (2 C_{33} (C_{33} - C_{44}))`. Controls near-vertical
        P-wave reflection moveout. ``0`` for isotropic; can be
        positive or negative in VTI shales.
    vp0 : float
        Vertical P-wave velocity (m/s) used in the inversion -- the
        sonic-derived value passed in by the caller.
    residual_rms : float
        Root-mean-square residual of the joint (V_phase, polarization
        angle) least-squares fit. Has the units of the joint
        residual vector (mixed -- treat as a relative quality
        score; an order-of-magnitude smaller than ``epsilon`` /
        ``delta`` themselves indicates a clean fit).
    n_shots : int
        Number of walkaway-VSP shots used in the inversion.
    """

    epsilon: float
    delta: float
    vp0: float
    residual_rms: float
    n_shots: int


def thomsen_epsilon_delta_from_walkaway_vsp(
    slowness_vectors: np.ndarray,
    polarization_vectors: np.ndarray,
    vp0: float,
) -> ThomsenEpsilonDeltaResult:
    r"""
    Thomsen :math:`\epsilon` / :math:`\delta` from walkaway-VSP
    slowness-polarization measurements.

    Closes the VTI-roadmap gap that
    :func:`vti_moduli_from_logs` flags as out-of-scope: the two
    Thomsen parameters that a vertical-well sonic acquisition
    cannot recover (:math:`\epsilon`, :math:`\delta`) but a
    walkaway-VSP at the same depth can (Miller & Spencer 1994;
    Horne & Leaney 2000).

    **Minimum extra data** beyond the sonic logs already in
    :mod:`fwap`: per shot, the 2-D P-wave slowness vector
    :math:`\mathbf{p} = (p_x, p_z)` measured at the downhole
    geophone (typically from the array slope of the picked first
    arrivals) plus the 2-D P-wave polarization unit vector
    :math:`\mathbf{u} = (u_x, u_z)` (the eigenvector of the
    3C particle-motion covariance at the picked first break). The
    sonic monopole P log supplies the vertical P velocity
    :math:`V_{P0}` directly -- pass it as ``vp0``.

    Inversion (Thomsen 1986 weak-anisotropy linearisation)
    --------------------------------------------------------
    For a P-wave in a VTI medium with a vertical symmetry axis,
    the per-shot phase velocity and polarization-deviation angle
    are linear in :math:`\epsilon`, :math:`\delta`:

    .. math::

        \frac{V_{\mathrm{phase}}(\theta)}{V_{P0}} - 1
        \;\approx\; \delta \sin^2\theta \cos^2\theta
                \;+\; \epsilon \sin^4\theta,

    .. math::

        \psi_u(\theta) - \theta
        \;\approx\; \epsilon \sin(2\theta)
                \;+\; \tfrac{1}{2}(\delta - \epsilon) \sin(4\theta),

    where :math:`\theta = \arctan(p_x / p_z)` is the slowness-vector
    phase angle from vertical and :math:`\psi_u =
    \arctan(u_x / u_z)` is the polarization angle. Stacking both
    equations across all :math:`N` shots gives a :math:`2N \times 2`
    linear system that is solved via :func:`numpy.linalg.lstsq`.

    Assumptions
    -----------
    * Weak-anisotropy regime: :math:`|\epsilon|, |\delta| \lesssim
      0.3`. Beyond that the linearisation is biased; the exact
      Christoffel inversion is needed.
    * Single-layer VTI between source and receiver (no
      dipping-layer / azimuthal-anisotropy corrections).
    * The polarization vectors are unit-magnitude P-wave first-
      motion estimates; magnitude is ignored (only the direction
      enters the inversion).
    * The slowness-vector horizontal component ``p_x`` is signed
      (same convention as the offset direction). The polarization
      ``u_x`` carries the same sign as ``p_x`` for a physically
      reasonable P-wave first motion.

    Out of scope
    ------------
    The same data gives the third Thomsen parameter :math:`\gamma`
    only with a converted (P-to-S) S-wave polarization measurement
    at each shot, which is uncommon in routine walkaway VSP. For
    :math:`\gamma`, use the sonic-only
    :func:`thomsen_gamma_from_logs` instead -- the dipole + Stoneley
    tracks are much more reliably available than P-to-S converted
    waves at oblique incidence.

    Parameters
    ----------
    slowness_vectors : ndarray, shape (n_shots, 2)
        Per-shot ``[p_x, p_z]`` slowness components (s/m). Must
        have positive :math:`|p|` everywhere; ``p_z`` should be
        strictly positive (down-going wave).
    polarization_vectors : ndarray, shape (n_shots, 2)
        Per-shot ``[u_x, u_z]`` polarization components. Magnitude
        is irrelevant -- the function uses only the direction.
        Must be non-zero everywhere.
    vp0 : float
        Vertical P-wave velocity (m/s). The standard source is the
        sonic monopole pick: ``vp0 = sqrt(c33 / rho)`` with
        ``c33`` from :func:`c33_from_p_pick`. Must be strictly
        positive.

    Returns
    -------
    ThomsenEpsilonDeltaResult
        ``epsilon``, ``delta``, ``vp0`` (echoed), ``residual_rms``,
        ``n_shots``.

    Raises
    ------
    ValueError
        If ``vp0 <= 0``, the input arrays are mis-shaped or have
        zero-length, or any per-shot slowness / polarization vector
        is zero.

    See Also
    --------
    thomsen_gamma_from_logs : The sonic-only :math:`\gamma`
        inversion that this function complements; together they
        give all three Thomsen parameters.
    vti_moduli_from_logs : The vertical-well-sonic VTI summary
        whose ``epsilon`` / ``delta`` slots are filled by this
        function when a walkaway VSP is also available.

    References
    ----------
    * Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics*
      51(10), 1954-1966.
    * Miller, D. E., & Spencer, C. (1994). An exact inversion for
      anisotropic moduli from phase-slowness data. *J. Geophys.
      Res.* 99(B11), 21651-21657.
    * Horne, S., & Leaney, S. (2000). Polarization and slowness
      component inversion for TI anisotropy. *Geophysical
      Prospecting* 48(4), 779-788.
    * Tsvankin, I. (2012). *Seismic Signatures and Analysis of
      Reflection Data in Anisotropic Media*, 3rd ed., Chapter 1
      (weak-anisotropy linearisations). SEG.
    """
    if vp0 <= 0:
        raise ValueError("vp0 must be strictly positive")
    p = np.asarray(slowness_vectors, dtype=float)
    u = np.asarray(polarization_vectors, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(
            f"slowness_vectors must have shape (n_shots, 2); got {p.shape}"
        )
    if u.shape != p.shape:
        raise ValueError(
            "polarization_vectors must have the same shape as "
            f"slowness_vectors; got {u.shape} vs {p.shape}"
        )
    n_shots = p.shape[0]
    if n_shots < 1:
        raise ValueError("at least one shot is required")
    p_norm = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    u_norm = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
    if np.any(p_norm <= 0):
        raise ValueError("every slowness vector must be non-zero")
    if np.any(u_norm <= 0):
        raise ValueError("every polarization vector must be non-zero")

    theta = np.arctan2(p[:, 0], p[:, 1])  # phase angle from z
    psi_u = np.arctan2(u[:, 0], u[:, 1])  # polarization angle
    v_phase = 1.0 / p_norm

    # Velocity equation: V_phase / V_P0 - 1 = epsilon sin^4 theta
    #                                       + delta sin^2 theta cos^2 theta
    sin2_t = np.sin(theta) ** 2
    cos2_t = np.cos(theta) ** 2
    rhs_v = v_phase / vp0 - 1.0
    coef_eps_v = sin2_t**2  # sin^4 theta
    coef_del_v = sin2_t * cos2_t  # sin^2 theta cos^2 theta

    # Polarization equation: psi_u - theta = epsilon sin(2 theta)
    #                                       + (delta - epsilon)/2 sin(4 theta)
    sin_2t = np.sin(2.0 * theta)
    sin_4t = np.sin(4.0 * theta)
    rhs_p = psi_u - theta
    coef_eps_p = sin_2t - 0.5 * sin_4t  # eps coefficient
    coef_del_p = 0.5 * sin_4t  # delta coefficient

    A = np.empty((2 * n_shots, 2), dtype=float)
    A[:n_shots, 0] = coef_eps_v
    A[:n_shots, 1] = coef_del_v
    A[n_shots:, 0] = coef_eps_p
    A[n_shots:, 1] = coef_del_p
    y = np.concatenate([rhs_v, rhs_p])

    m, *_ = np.linalg.lstsq(A, y, rcond=None)
    epsilon = float(m[0])
    delta = float(m[1])

    residual = A @ m - y
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    return ThomsenEpsilonDeltaResult(
        epsilon=epsilon,
        delta=delta,
        vp0=float(vp0),
        residual_rms=residual_rms,
        n_shots=n_shots,
    )
