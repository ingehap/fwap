"""
Cross-dipole Alford rotation for shear-wave azimuthal anisotropy.

Note
----
Cross-dipole Alford rotation is **not** treated in Mari et al. (1994).
This module is an extension that complements the dipole-flexural
processing of :mod:`fwap.dispersion`: a four-component (XX, XY, YX, YY)
shear measurement is rotated into the (fast, slow) shear frame so that
each component carries one polarization.

References
----------
* Alford, R. M. (1986). Shear data in the presence of azimuthal
  anisotropy. *56th SEG Annual International Meeting, Expanded
  Abstracts*, 476-479.
* Esmersoy, C., Koster, K., Williams, M., Boyd, A., & Kane, M. (1994).
  Dipole shear anisotropy logging. *64th SEG Annual International
  Meeting, Expanded Abstracts*, 1139-1142.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlfordResult:
    """
    Output of :func:`alford_rotation`.

    Attributes
    ----------
    angle : float
        Fast-shear azimuth measured from the X-dipole firing direction
        (radians, in ``(-pi/2, pi/2]``).
    fast, slow : ndarray, shape (n_samples,)
        Rotated fast-shear and slow-shear diagonal components.
    cross_energy_ratio : float
        ``(xy' + yx') energy / total energy`` after rotation; zero
        means perfect diagonalisation.
    """

    angle: float
    fast: np.ndarray
    slow: np.ndarray
    cross_energy_ratio: float


def alford_rotation(
    xx: np.ndarray, xy: np.ndarray, yx: np.ndarray, yy: np.ndarray
) -> AlfordResult:
    """
    Cross-dipole Alford rotation: find the rotation angle that
    minimises cross-component energy.

    Given the 2x2 time-domain cross-dipole tensor
    ``[[xx, xy], [yx, yy]]``, rotate to fast/slow shear axes. The
    closed-form minimiser of the sum of squared off-diagonal
    components over rotation angle ``theta`` is

        tan(4 theta) = (A + B) / (C - D) = 2 A / (C - D)

    where
        A = <xx - yy, xy + yx>,
        B = <xy + yx, xx - yy>  (= A for real-valued traces),
        C = <xx - yy, xx - yy>,
        D = <xy + yx, xy + yx>.

    The simplification ``A + B = 2 A`` is exact for real-valued
    cross-dipole components; for the general (complex) case keep
    ``(A + B) / (C - D)``.

    In practice both ``theta`` and ``theta + pi/2`` are stationary
    points -- we pick the one that puts the *slower* shear on the
    ``y'`` axis (by integrated arrival time) for the conventional
    labelling; the caller can swap if the other convention is needed.

    Parameters
    ----------
    xx, xy, yx, yy : ndarray
        The four components of the cross-dipole tensor at a single
        depth. All four arrays must have the same shape. Typically
        one-dimensional (``(n_samples,)``), but any shape is
        accepted; time-integrated inner products are taken over the
        flattened array.

    Returns
    -------
    AlfordResult
        See the :class:`AlfordResult` dataclass for field-by-field
        descriptions.

    References
    ----------
    Alford, R. M. (1986). Shear data in the presence of azimuthal
    anisotropy. *56th SEG Annual Meeting*, Expanded Abstracts,
    476-479.
    Esmersoy, C., et al. (1994). Dipole shear anisotropy logging.
    *64th SEG Annual Meeting*, Expanded Abstracts, 1139-1142.
    """
    xx = np.asarray(xx, dtype=float)
    xy = np.asarray(xy, dtype=float)
    yx = np.asarray(yx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    if not (xx.shape == xy.shape == yx.shape == yy.shape):
        raise ValueError("all four components must have the same shape")

    diff = xx - yy
    sum_ = xy + yx
    num = 2.0 * np.sum(diff * sum_)
    den = float(np.sum(diff * diff) - np.sum(sum_ * sum_))

    theta = 0.25 * np.arctan2(num, den)

    def rotate(th: float):
        c, s = np.cos(th), np.sin(th)
        f = c * c * xx + s * c * (xy + yx) + s * s * yy
        sl = s * s * xx - s * c * (xy + yx) + c * c * yy
        x_y = c * s * (yy - xx) + c * c * xy - s * s * yx
        y_x = c * s * (yy - xx) - s * s * xy + c * c * yx
        return f, sl, x_y, y_x

    fast_a, slow_a, xy_a, yx_a = rotate(theta)
    fast_b, slow_b, xy_b, yx_b = rotate(theta + np.pi / 2)
    # Pick the orientation whose diagonal has the earlier-arriving
    # component on the fast axis. Use centre-of-energy time on the
    # fast trace as the proxy.
    t = np.arange(fast_a.size)
    ea = float(np.sum(fast_a**2))
    eb = float(np.sum(fast_b**2))
    if ea < 1e-30 and eb < 1e-30:
        chosen = (theta, fast_a, slow_a, xy_a, yx_a)
    else:
        t_a = np.sum(t * fast_a**2) / (ea + 1e-30)
        t_b = np.sum(t * fast_b**2) / (eb + 1e-30)
        if t_a <= t_b:
            chosen = (theta, fast_a, slow_a, xy_a, yx_a)
        else:
            chosen = (theta + np.pi / 2, fast_b, slow_b, xy_b, yx_b)

    th, fast, slow, xy_r, yx_r = chosen
    # Fold to (-pi/2, pi/2].
    th = (th + np.pi / 2) % np.pi - np.pi / 2

    cross_en = float(np.sum(xy_r**2) + np.sum(yx_r**2))
    total_en = np.sum(fast**2) + np.sum(slow**2) + cross_en
    ratio = float(cross_en / (total_en + 1e-30))

    return AlfordResult(angle=float(th), fast=fast, slow=slow, cross_energy_ratio=ratio)


@dataclass
class StressAnisotropyEstimate:
    r"""
    Petrophysical labelling of a cross-dipole Alford rotation.

    Re-frames the angle and energy-ratio outputs of
    :func:`alford_rotation` in terms of the three Workflow-3
    quantities the book (Mari et al. 1994, Part 3) lists for the
    dipole-sonic deliverable: stress direction, anisotropy strength,
    and a flexural-fracture indicator. The numerics are the same as
    in :class:`AlfordResult`; this dataclass just adds the
    petrophysical interpretation and a couple of derived metrics
    (splitting-time delay, anisotropy strength as a relative L2
    norm) that the raw Alford output does not surface directly.

    Attributes
    ----------
    max_horizontal_stress_azimuth : float
        Fast-shear azimuth, in radians, in ``(-pi/2, pi/2]``.
        Conventionally aligned with the maximum horizontal stress
        :math:`\sigma_{H,\max}` for stress-induced anisotropy, with
        the strike of natural fractures for fracture-induced
        anisotropy, and with the bedding orientation for intrinsic
        anisotropy in laminated formations. The mechanism that
        dominates is *not* identifiable from a single Alford
        rotation -- the user picks the interpretation given image
        logs, mud-loss data, or other ancillary measurements.
    min_horizontal_stress_azimuth : float
        Slow-shear azimuth = ``max_horizontal_stress_azimuth + pi/2``,
        folded back into ``(-pi/2, pi/2]``.
    splitting_time_delay : float
        Time delay (s) between the fast and slow shear arrivals,
        estimated as the lag of the cross-correlation peak between
        the rotated ``fast`` and ``slow`` waveforms (positive ⇔
        slow trails fast, which is the physical case). The classic
        anisotropy strength metric :math:`\gamma = \Delta t /
        t_\text{travel}` follows directly from this quantity.
    anisotropy_strength : float
        Dimensionless ``[0, 1]`` measure of how much fast and slow
        shear differ, defined as
        :math:`\|f - s\| / \sqrt{2\,(\|f\|^2 + \|s\|^2)}`. ``0`` for
        identical waveforms (isotropic medium); approaches ``1`` only
        as :math:`s \to -f` (sign-flipped pair), and equals
        :math:`1/\sqrt{2} \approx 0.707` for orthogonal equal-energy
        waveforms. Time-shift-blind, so reports anisotropy even on a
        fast / slow pair that differ only in arrival time.
    fracture_indicator : float
        Heuristic flexural-wave fracture proxy in ``[0, 1]``:
        :math:`\text{rotation\_quality} \times
        \text{anisotropy\_strength}`. High values require both
        a clean rotation (so the anisotropy is well-resolved) and
        a strong fast/slow contrast (so the medium is genuinely
        anisotropic). Cannot distinguish stress-induced from
        fracture-induced anisotropy on its own; intended as a
        *flag* track to be cross-checked against a borehole image
        log or a Stoneley-permeability indicator before being
        labelled as fractures.
    rotation_quality : float
        ``1 - cross_energy_ratio`` from :class:`AlfordResult`. ``1``
        means the rotation perfectly diagonalises the cross-dipole
        tensor; lower values flag depths where the two-shear
        Alford model itself fits poorly (e.g. multi-mode
        interference, off-axis arrivals, low-SNR data).
    alford : AlfordResult
        The underlying Alford rotation, kept for callers that need
        the rotated waveforms (``fast`` / ``slow``) themselves.
    """

    max_horizontal_stress_azimuth: float
    min_horizontal_stress_azimuth: float
    splitting_time_delay: float
    anisotropy_strength: float
    fracture_indicator: float
    rotation_quality: float
    alford: AlfordResult


def _splitting_time_delay(fast: np.ndarray, slow: np.ndarray, dt: float) -> float:
    """Cross-correlation lag (s) of slow vs fast.

    Positive return value means the slow trace trails the fast
    trace, which is the physical convention for shear-wave
    splitting (slow shear arrives later).
    """
    a = np.asarray(fast, dtype=float).ravel()
    b = np.asarray(slow, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    if np.allclose(a, 0.0) or np.allclose(b, 0.0):
        return 0.0
    corr = np.correlate(b, a, mode="full")
    lags = np.arange(-(a.size - 1), b.size)
    peak = int(np.argmax(np.abs(corr)))
    return float(lags[peak]) * float(dt)


def _anisotropy_strength(fast: np.ndarray, slow: np.ndarray) -> float:
    """Relative L2 difference between fast and slow shear, in [0, 1]."""
    a = np.asarray(fast, dtype=float).ravel()
    b = np.asarray(slow, dtype=float).ravel()
    diff_norm2 = float(np.sum((a - b) ** 2))
    sum_norm2 = float(np.sum(a * a) + np.sum(b * b))
    if sum_norm2 <= 0.0:
        return 0.0
    # ``sqrt(2)`` factor caps the metric at 1: the maximum value of
    # ||a - b||^2 / (||a||^2 + ||b||^2) for non-trivial a, b is 2
    # (achieved when ``a == -b``).
    return float(np.sqrt(diff_norm2 / sum_norm2 / 2.0))


def stress_anisotropy_from_alford(
    alford: AlfordResult,
    dt: float,
) -> StressAnisotropyEstimate:
    r"""
    Re-label an :class:`AlfordResult` in petrophysical terms.

    Computes the splitting-time delay and the relative-L2
    anisotropy strength from the rotated ``fast`` / ``slow``
    waveforms, packages them with the orthogonal stress azimuths
    and a fracture-indicator heuristic, and returns the result as
    a :class:`StressAnisotropyEstimate`. See that dataclass's
    docstring for the per-field semantics and caveats; the
    short version is that ``max_horizontal_stress_azimuth`` is the
    Alford fast-shear angle re-labelled with the conventional
    stress-direction interpretation.

    Parameters
    ----------
    alford : AlfordResult
        Output of :func:`alford_rotation` (or
        :func:`alford_rotation_from_tensor`) at one depth.
    dt : float
        Time sampling interval (s) of the rotated traces. Used
        only for the splitting-time-delay computation.

    Returns
    -------
    StressAnisotropyEstimate
    """
    sigma_max_az = float(alford.angle)
    # Orthogonal direction folded into (-pi/2, pi/2].
    sigma_min_az = float(((sigma_max_az + np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2)
    delay = _splitting_time_delay(alford.fast, alford.slow, dt)
    strength = _anisotropy_strength(alford.fast, alford.slow)
    rotation_quality = max(0.0, 1.0 - float(alford.cross_energy_ratio))
    fracture = float(rotation_quality * strength)
    return StressAnisotropyEstimate(
        max_horizontal_stress_azimuth=sigma_max_az,
        min_horizontal_stress_azimuth=sigma_min_az,
        splitting_time_delay=delay,
        anisotropy_strength=strength,
        fracture_indicator=fracture,
        rotation_quality=rotation_quality,
        alford=alford,
    )


def alford_rotation_from_tensor(tensor: np.ndarray) -> AlfordResult:
    """
    Cross-dipole Alford rotation from a packed ``(2, 2, n_samples)``
    tensor.

    Thin adapter over :func:`alford_rotation` for callers that already
    carry the cross-dipole measurement as a single array with the two
    dipole-source / two dipole-receiver axes as the leading two
    dimensions.

    Parameters
    ----------
    tensor : ndarray, shape ``(2, 2, n_samples)`` (or any broadcastable
        shape ``(2, 2, ...)``)
        ``tensor[0, 0]`` is XX (X-source -> X-receiver),
        ``tensor[0, 1]`` is XY, ``tensor[1, 0]`` is YX,
        ``tensor[1, 1]`` is YY.

    Returns
    -------
    AlfordResult

    Raises
    ------
    ValueError
        If ``tensor.shape[:2] != (2, 2)``.
    """
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape[:2] != (2, 2):
        raise ValueError(f"tensor must have shape (2, 2, ...); got {tensor.shape}")
    return alford_rotation(tensor[0, 0], tensor[0, 1], tensor[1, 0], tensor[1, 1])


# ---------------------------------------------------------------------
# VTI (vertical-symmetry-axis) anisotropy: Thomsen gamma from
# combined dipole shear (-> C44) + Stoneley low-f tube wave (-> C66)
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Vertical-well VTI moduli summary
# ---------------------------------------------------------------------


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
    :file:`docs/roadmap.md`.

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


# ---------------------------------------------------------------------
# Backus averaging: layered isotropic media -> effective VTI tensor
# ---------------------------------------------------------------------


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
        ("c11", c11), ("c33", c33), ("c44", c44), ("c66", c66),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be positive")

    theta = np.asarray(phase_angle_rad, dtype=float)
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2

    # Tsvankin 2001 eq. 1.41
    inner = (
        ((c11 - c44) * sin2 - (c33 - c44) * cos2) ** 2
        + 4.0 * (c13 + c44) ** 2 * sin2 * cos2
    )
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
