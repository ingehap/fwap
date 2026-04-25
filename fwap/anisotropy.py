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
        :math:`\text{rotation_quality} \times
        \text{anisotropy_strength}`. High values require both
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
