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


def alford_rotation(xx: np.ndarray, xy: np.ndarray,
                    yx: np.ndarray, yy: np.ndarray
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
        f  = c * c * xx + s * c * (xy + yx) + s * s * yy
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
    ea = float(np.sum(fast_a ** 2))
    eb = float(np.sum(fast_b ** 2))
    if ea < 1e-30 and eb < 1e-30:
        chosen = (theta, fast_a, slow_a, xy_a, yx_a)
    else:
        t_a = np.sum(t * fast_a ** 2) / (ea + 1e-30)
        t_b = np.sum(t * fast_b ** 2) / (eb + 1e-30)
        if t_a <= t_b:
            chosen = (theta, fast_a, slow_a, xy_a, yx_a)
        else:
            chosen = (theta + np.pi / 2, fast_b, slow_b, xy_b, yx_b)

    th, fast, slow, xy_r, yx_r = chosen
    # Fold to (-pi/2, pi/2].
    th = (th + np.pi / 2) % np.pi - np.pi / 2

    cross_en = float(np.sum(xy_r ** 2) + np.sum(yx_r ** 2))
    total_en = (np.sum(fast ** 2) + np.sum(slow ** 2) + cross_en)
    ratio = float(cross_en / (total_en + 1e-30))

    return AlfordResult(angle=float(th), fast=fast, slow=slow,
                        cross_energy_ratio=ratio)


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


def _splitting_time_delay(fast: np.ndarray,
                          slow: np.ndarray,
                          dt: float) -> float:
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


def stress_anisotropy_from_alford(alford: AlfordResult,
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
    sigma_min_az = float(
        ((sigma_max_az + np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2
    )
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
        raise ValueError(
            "tensor must have shape (2, 2, ...); got "
            f"{tensor.shape}"
        )
    return alford_rotation(tensor[0, 0], tensor[0, 1],
                           tensor[1, 0], tensor[1, 1])


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
            f"slowness {1.0/v_fluid:.3e} s/m."
        )
    return rho_fluid / diff


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
    s_d  = np.asarray(slowness_dipole,   dtype=float)
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
        s_st, rho_fluid=rho_fluid, v_fluid=v_fluid,
    )
    gamma = thomsen_gamma(c44, c66)
    return ThomsenGammaResult(c44=c44, c66=c66, gamma=gamma)
