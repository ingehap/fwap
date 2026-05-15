"""
Thomsen gamma (VTI vertical-symmetry-axis anisotropy) from
combined dipole shear (-> C44) and Stoneley low-frequency
tube wave (-> C66).

References
----------
* Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics*
  51(10), 1954-1966.
* White, J. E. (1983). *Underground sound*. Elsevier.
* Tang, X.-M. & Cheng, A. (2004). *Quantitative borehole
  acoustic methods*. Elsevier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThomsenGammaResult:
    r"""
    Output of :func:`thomsen_gamma_from_logs`.

    Attributes
    ----------
    c44 : ndarray
        Vertical shear modulus :math:`C_{44} = \rho V_{Sv}^2` (Pa).
        Derived from the dipole shear log (vertically-propagating
        shear with horizontal polarization in a vertical well).
    c66 : ndarray
        Horizontal shear modulus :math:`C_{66}` (Pa). Derived from
        the Stoneley low-frequency tube-wave inversion (White 1983;
        Norris 1990 for the VTI extension).
    gamma : ndarray
        Thomsen (1986) shear-anisotropy parameter
        :math:`\gamma = (C_{66} - C_{44}) / (2 C_{44})`. ``0`` for
        an isotropic formation; positive in typical VTI shales
        (horizontal shear stiffer than vertical shear).
    """

    c44: np.ndarray
    c66: np.ndarray
    gamma: np.ndarray


def stoneley_horizontal_shear_modulus(
    slowness_stoneley: np.ndarray,
    *,
    rho_fluid: float,
    v_fluid: float,
) -> np.ndarray:
    r"""
    Horizontal shear modulus :math:`C_{66}` from low-frequency
    Stoneley slowness (White 1983 tube-wave formula).

    For a fluid-filled borehole through a (possibly transversely-
    isotropic) elastic formation, the low-frequency Stoneley phase
    slowness :math:`S_{ST}` satisfies (White 1983, eq. 5.42; Norris
    1990 generalises the result to VTI media):

    .. math::

        S_{ST}^2 \;=\; S_f^2 \;+\; \frac{\rho_f}{C_{66}},

    where :math:`S_f = 1/V_f` is the borehole-fluid slowness and
    :math:`\rho_f` is the fluid density. The relevant shear modulus
    is :math:`C_{66}` (the horizontal shear modulus); for an
    isotropic formation :math:`C_{66} = \mu = \rho V_S^2`. Solving
    for :math:`C_{66}`:

    .. math::

        C_{66} \;=\; \frac{\rho_f}{S_{ST}^2 - S_f^2}.

    Assumptions
    -----------
    * Low-frequency limit (tube wave well below the dipole-flexural
      and pseudo-Rayleigh cutoff frequencies). Above the cutoffs
      Tang-Cheng (2004) eqs. 5.19-5.22 give the corrected, dispersive
      form.
    * Inviscid borehole fluid; centred tool; circular borehole cross-
      section. Eccentricity and tool-mode effects fold into a
      multiplicative correction that fwap does not currently apply.
    * Hard formation (formation shear impedance >> fluid impedance).
      Mudcake / casing layers are ignored; for a cased hole the
      multilayered radial form (Tang & Cheng 2004 §2) is required.

    Parameters
    ----------
    slowness_stoneley : ndarray or float
        Per-depth Stoneley-wave slowness (s/m). Must be strictly
        greater than ``1 / v_fluid`` everywhere -- the Stoneley wave
        is always slower than the unconfined fluid wave because the
        formation loads it.
    rho_fluid : float
        Borehole-fluid density (kg/m^3). Brine ~ 1000-1100; oil ~
        800-900; gas/foam << 1000.
    v_fluid : float
        Borehole-fluid acoustic velocity (m/s). Brine ~ 1500; oil ~
        1300; gas << 1000.

    Returns
    -------
    ndarray
        :math:`C_{66}` (Pa), broadcast to the shape of
        ``slowness_stoneley``.

    Raises
    ------
    ValueError
        If ``rho_fluid`` or ``v_fluid`` is non-positive, or if any
        ``slowness_stoneley`` value is at or below the fluid slowness
        (the inversion is undefined there).

    References
    ----------
    * White, J. E. (1983). *Underground Sound: Application of Seismic
      Waves.* Elsevier (sect. 5.5: tube waves).
    * Norris, A. N. (1990). The speed of a tube wave. *J. Acoust.
      Soc. Am.* 87(1), 414-417.
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, sect. 5.4 (Stoneley-wave inversion
      for C66 in VTI formations).
    """
    if rho_fluid <= 0.0:
        raise ValueError("rho_fluid must be strictly positive")
    if v_fluid <= 0.0:
        raise ValueError("v_fluid must be strictly positive")
    s_st = np.asarray(slowness_stoneley, dtype=float)
    s_f2 = 1.0 / (v_fluid * v_fluid)
    diff = s_st * s_st - s_f2
    if np.any(diff <= 0.0):
        raise ValueError(
            "slowness_stoneley must exceed 1 / v_fluid everywhere "
            "(Stoneley wave is slower than the unconfined fluid wave); "
            f"got min slowness {float(np.min(s_st)):.3e} s/m, fluid "
            f"slowness {1.0 / v_fluid:.3e} s/m."
        )
    return rho_fluid / diff


def stoneley_horizontal_shear_modulus_corrected(
    slowness_stoneley: np.ndarray,
    rho: np.ndarray,
    slowness_p: np.ndarray,
    *,
    rho_fluid: float,
    v_fluid: float,
) -> np.ndarray:
    r"""
    Tang & Cheng (2004) §5.4 finite-formation-impedance correction
    on the White (1983) Stoneley → :math:`C_{66}` inversion.

    The simple :func:`stoneley_horizontal_shear_modulus` formula
    (White 1983)

    .. math::

        S_{ST}^2 \;=\; \frac{1}{V_f^2} \;+\; \frac{\rho_f}{C_{66}}

    treats the formation as **rigid** against radial tube-wave
    displacement. In reality the Stoneley wave couples weakly to the
    formation P-mode -- the radial pressure oscillation pumps fluid
    into the formation slightly, even at zero matrix permeability,
    which softens the *effective* shear modulus felt by the tube
    wave. Tang & Cheng (2004), sect. 5.4 (eq. 5.31) give the
    refinement:

    .. math::

        S_{ST}^2 \;=\; \frac{1}{V_f^2}
                      \;+\; \frac{\rho_f}{C_{66,\mathrm{eff}}},
        \qquad
        C_{66,\mathrm{eff}} \;=\; C_{66} \,
            \left(1 \;-\; \frac{\rho_f V_f^{\,2}}{\rho V_P^{\,2}}\right).

    Solving for :math:`C_{66}`:

    .. math::

        C_{66} \;=\;
        \frac{\rho_f}{(S_{ST}^2 - 1/V_f^2)}
        \,/\, \left(1 - \frac{\rho_f V_f^{\,2}}{\rho V_P^{\,2}}\right).

    The correction factor is
    :math:`1 / (1 - \rho_f V_f^{\,2}/(\rho V_P^{\,2}))`. For brine
    in a typical sandstone (:math:`V_P` ≈ 4500 m/s) it is ~1.05;
    for slow VTI shales (:math:`V_P` ≈ 2500-3000 m/s) it can reach
    1.10–1.20 and matters operationally for the recovered shear
    anisotropy. In the rigid-formation limit
    (:math:`V_P \to \infty`) the correction vanishes and the formula
    reduces to :func:`stoneley_horizontal_shear_modulus`.

    Parameters
    ----------
    slowness_stoneley : ndarray or float
        Per-depth low-frequency Stoneley slowness (s/m). Must
        satisfy ``slowness_stoneley > 1 / v_fluid`` everywhere.
    rho : ndarray or float
        Per-depth formation bulk density (kg/m^3). Must be strictly
        positive.
    slowness_p : ndarray or float
        Per-depth monopole P slowness (s/m), giving the formation
        P-wave modulus :math:`\rho V_P^2 = \rho / S_P^2`. Must be
        strictly positive.
    rho_fluid : float
        Borehole-fluid density (kg/m^3).
    v_fluid : float
        Borehole-fluid acoustic velocity (m/s).

    Returns
    -------
    ndarray
        Corrected :math:`C_{66}` (Pa), broadcast to the common shape
        of the inputs.

    Raises
    ------
    ValueError
        If any input is non-positive, if any Stoneley slowness is at
        or below the fluid slowness, or if the formation P-wave
        modulus :math:`\rho V_P^2` does not exceed the fluid bulk
        modulus :math:`\rho_f V_f^2` (the correction factor would
        otherwise be non-positive -- physically a vanishing or
        inverted radial impedance contrast and outside the model's
        scope).

    See Also
    --------
    stoneley_horizontal_shear_modulus :
        The uncorrected (White 1983) form. Equivalent to this
        function in the :math:`V_P \to \infty` limit; faster and
        the right choice when the monopole P pick or density log is
        unavailable.

    References
    ----------
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, Section 5.4 (Stoneley-wave
      inversion for the horizontal shear modulus in VTI
      formations).
    * Norris, A. N. (1990). The speed of a tube wave. *J. Acoust.
      Soc. Am.* 87(1), 414-417 (general radial-impedance form).
    * White, J. E. (1983). *Underground Sound: Application of
      Seismic Waves.* Elsevier, sect. 5.5 (rigid-formation limit).
    """
    if rho_fluid <= 0.0:
        raise ValueError("rho_fluid must be strictly positive")
    if v_fluid <= 0.0:
        raise ValueError("v_fluid must be strictly positive")
    s_st = np.asarray(slowness_stoneley, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    s_p = np.asarray(slowness_p, dtype=float)
    if np.any(rho_arr <= 0):
        raise ValueError("rho must be strictly positive")
    if np.any(s_p <= 0):
        raise ValueError("slowness_p must be strictly positive")
    s_f2 = 1.0 / (v_fluid * v_fluid)
    diff = s_st * s_st - s_f2
    if np.any(diff <= 0.0):
        raise ValueError(
            "slowness_stoneley must exceed 1 / v_fluid everywhere "
            "(Stoneley wave is slower than the unconfined fluid wave); "
            f"got min slowness {float(np.min(s_st)):.3e} s/m, fluid "
            f"slowness {1.0 / v_fluid:.3e} s/m."
        )
    # Formation P-wave modulus rho * V_P^2 = rho / s_p^2 must exceed
    # the fluid bulk modulus rho_f * V_f^2 so the correction factor
    # 1 - (rho_f V_f^2)/(rho V_P^2) stays positive.
    rho_vp2 = rho_arr / (s_p * s_p)
    rho_f_vf2 = rho_fluid * v_fluid * v_fluid
    factor = 1.0 - rho_f_vf2 / rho_vp2
    if np.any(factor <= 0.0):
        raise ValueError(
            "formation P-wave modulus rho * V_P^2 must exceed the "
            "fluid bulk modulus rho_fluid * v_fluid^2 everywhere "
            "(the Tang & Cheng (2004) correction factor "
            "1 - rho_f V_f^2 / (rho V_P^2) must stay positive); got "
            f"min rho * V_P^2 = {float(np.min(rho_vp2)):.3e} Pa "
            f"vs rho_f * V_f^2 = {rho_f_vf2:.3e} Pa."
        )
    return rho_fluid / (diff * factor)


def thomsen_gamma(c44: np.ndarray, c66: np.ndarray) -> np.ndarray:
    r"""
    Thomsen (1986) shear-anisotropy parameter from C44 and C66.

    .. math::

        \gamma \;=\; \frac{C_{66} - C_{44}}{2\, C_{44}}.

    For a vertically-symmetric (VTI) medium :math:`\gamma` measures
    the difference between horizontally- and vertically-polarised
    shear stiffness; ``0`` for an isotropic formation, positive for
    typical layered shales.

    Parameters
    ----------
    c44, c66 : scalar or ndarray
        Vertical and horizontal shear moduli (Pa). Both must be
        strictly positive everywhere.

    Returns
    -------
    ndarray
        Thomsen :math:`\gamma`, broadcast to the common shape.

    Raises
    ------
    ValueError
        If either modulus is non-positive anywhere.

    References
    ----------
    Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics*
    51(10), 1954-1966.
    """
    c44_arr = np.asarray(c44, dtype=float)
    c66_arr = np.asarray(c66, dtype=float)
    if np.any(c44_arr <= 0):
        raise ValueError("c44 must be strictly positive")
    if np.any(c66_arr <= 0):
        raise ValueError("c66 must be strictly positive")
    return (c66_arr - c44_arr) / (2.0 * c44_arr)


def thomsen_gamma_from_logs(
    slowness_dipole: np.ndarray,
    slowness_stoneley: np.ndarray,
    rho: np.ndarray,
    *,
    rho_fluid: float,
    v_fluid: float,
) -> ThomsenGammaResult:
    r"""
    Thomsen :math:`\gamma` from combined dipole + Stoneley sonic logs.

    Combines two complementary sonic measurements that together fix
    both off-diagonal shear moduli of a VTI formation:

    * **Dipole shear log** -- the low-frequency dipole flexural mode
      travels at the formation vertical shear speed
      :math:`V_{Sv} = \sqrt{C_{44}/\rho}`, so

      .. math::

          C_{44} \;=\; \rho \, / \, S_{S,\mathrm{dipole}}^{\,2}.

    * **Stoneley tube wave** -- inverted via the White (1983) /
      Norris (1990) low-frequency formula in
      :func:`stoneley_horizontal_shear_modulus` to give the
      horizontal shear modulus :math:`C_{66}`.

    Their ratio gives Thomsen :math:`\gamma` per
    :func:`thomsen_gamma`:

    .. math::

        \gamma \;=\; \frac{C_{66} - C_{44}}{2\, C_{44}}.

    This is the standard sonic-only VTI shear-anisotropy estimator
    that Tang & Cheng (2004), sect. 5.4 list as a Workflow-3
    deliverable for the dipole-sonic + Stoneley combination. It
    does *not* recover the P-wave Thomsen parameters
    :math:`\epsilon, \delta` -- those need additional measurements
    (cross-well or VSP) since a vertical-well sonic record samples
    only the vertical P-wave slowness.

    Parameters
    ----------
    slowness_dipole : ndarray
        Per-depth dipole shear slowness (s/m). Typically
        :attr:`fwap.picker.ModePick.slowness` for the ``"S"`` mode
        gathered across depths, or the ``"DTS"`` column of a LAS
        log written via :func:`fwap.picker.track_to_log_curves`
        (after converting from us/ft back to s/m).
    slowness_stoneley : ndarray
        Per-depth Stoneley-wave slowness (s/m). Typically the
        ``"Stoneley"`` mode, low-frequency band.
    rho : ndarray
        Per-depth formation bulk density (kg/m^3) from the bulk-
        density log (typically the ``RHOB`` curve).
    rho_fluid : float
        Borehole-fluid density (kg/m^3); see
        :func:`stoneley_horizontal_shear_modulus`.
    v_fluid : float
        Borehole-fluid acoustic velocity (m/s).

    Returns
    -------
    ThomsenGammaResult
        ``c44`` (Pa), ``c66`` (Pa), and ``gamma`` (dimensionless),
        all per-depth and broadcast to the common input shape.

    Raises
    ------
    ValueError
        Same conditions as :func:`stoneley_horizontal_shear_modulus`
        and :func:`thomsen_gamma`, plus rejection of non-positive
        slownesses or densities.
    """
    s_d = np.asarray(slowness_dipole, dtype=float)
    s_st = np.asarray(slowness_stoneley, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    if np.any(s_d <= 0):
        raise ValueError("slowness_dipole must be strictly positive")
    if np.any(s_st <= 0):
        raise ValueError("slowness_stoneley must be strictly positive")
    if np.any(rho_arr <= 0):
        raise ValueError("rho must be strictly positive")
    c44 = rho_arr / (s_d * s_d)
    c66 = stoneley_horizontal_shear_modulus(
        s_st,
        rho_fluid=rho_fluid,
        v_fluid=v_fluid,
    )
    gamma = thomsen_gamma(c44, c66)
    return ThomsenGammaResult(c44=c44, c66=c66, gamma=gamma)
