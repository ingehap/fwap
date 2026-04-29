"""
Attenuation (quality factor Q) estimation from array-sonic gathers.

Provides two complementary estimators: the centroid-frequency-shift
method (:func:`centroid_frequency_shift_Q`) and the spectral-ratio /
log-slope method (:func:`spectral_ratio_Q`).

Note
----
Q estimation is **not** covered in Mari et al. (1994); this module is an
extension included for completeness. It belongs to the same broader
"borehole as small-scale seismic experiment" philosophy that the book
champions, but the algorithms come from the surface-seismic literature.

References
----------
* Quan, Y., & Harris, J. M. (1997). Seismic attenuation tomography
  using the frequency shift method. *Geophysics* 62(3), 895-905.
* Bath, M. (1974). *Spectral Analysis in Geophysics.* Developments in
  Solid Earth Geophysics 7, Elsevier (spectral-ratio Q estimator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AttenuationMethod = Literal["centroid", "spectral_ratio"]

import numpy as np


@dataclass
class AttenuationResult:
    """
    Output of :func:`centroid_frequency_shift_Q` /
    :func:`spectral_ratio_Q`.

    Attributes
    ----------
    q : float
        Quality factor. Higher ``Q`` = less attenuation.
    q_sigma : float
        1-sigma uncertainty on ``Q`` from the regression.
    method : str
        ``"centroid"`` or ``"spectral_ratio"``.
    diagnostic : dict
        Fitted parameters and intermediate quantities useful for QC.
    """

    q: float
    q_sigma: float
    method: AttenuationMethod
    diagnostic: dict[str, np.ndarray] = field(default_factory=dict)


def _windowed_spectrum(
    data: np.ndarray, dt: float, t_start: np.ndarray, window_length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tapered window starting at ``t_start[i]`` on trace i."""
    n_rec, n_samp = data.shape
    L = max(2, int(round(window_length / dt)))
    i0 = np.clip(np.round(t_start / dt).astype(int), 0, n_samp - L)
    # Hann taper reduces spectral leakage, important for centroid fits.
    taper = np.hanning(L)
    chunks = np.empty((n_rec, L))
    for i in range(n_rec):
        chunks[i] = data[i, i0[i] : i0[i] + L] * taper
    spec = np.fft.rfft(chunks, axis=1)
    freqs = np.fft.rfftfreq(L, d=dt)
    return spec, freqs


def centroid_frequency_shift_Q(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slowness: float,
    window_length: float = 5.0e-4,
    f_range: tuple[float, float] = (2000.0, 30000.0),
    pick_intercept: float = 0.0,
) -> AttenuationResult:
    """
    Attenuation (``Q``) from centroid-frequency shift along the array.

    For a Gaussian source spectrum with centroid ``fc0`` and variance
    ``sigma_f**2``, the centroid at travel time ``t`` decays linearly
    with ``t`` under constant-``Q`` attenuation:

        fc(t) = fc0 - pi * sigma_f**2 * t / Q

    so ``Q = -pi * sigma_f**2 / slope`` where the slope comes from a
    weighted linear fit of ``fc`` against travel time across receivers.

    Assumptions
    -----------
    The closed form above requires (a) a (locally) Gaussian source
    spectrum whose variance is approximately constant across the array
    and (b) a constant-``Q`` attenuation model over the band of
    interest. Deviations from either -- e.g., a source with structured
    side-lobes, or frequency-dependent ``Q`` -- are the dominant source
    of systematic error. See Quan & Harris (1997), Section 3.

    Parameters
    ----------
    pick_intercept : float, default 0.0
        Absolute record time at which the first-breaking wave passes
        the ``offset = 0`` reference. Only shifts the x-axis of the QC
        plot in ``diagnostic['t']``; the fitted slope and derived ``Q``
        depend solely on travel time differences across the array.

    References
    ----------
    Quan, Y., & Harris, J. M. (1997). Seismic attenuation tomography
    using the frequency shift method. *Geophysics* 62(3), 895-905.
    """
    # t_travel is the travel-time axis used by the attenuation model;
    # the constant pick_intercept cancels out of the slope fit but is
    # retained so that diagnostic['t'] aligns with the absolute record
    # time for QC plotting.
    t_travel = pick_intercept + offsets * slowness
    spec, freqs = _windowed_spectrum(data, dt, t_travel, window_length)
    band = (freqs >= f_range[0]) & (freqs <= f_range[1])
    f_band = freqs[band]
    power = (np.abs(spec[:, band])) ** 2
    total = power.sum(axis=1) + 1e-30
    fc = (power * f_band).sum(axis=1) / total
    fvar = ((power * f_band**2).sum(axis=1) / total) - fc**2
    fvar = np.clip(fvar, 1.0, None)  # clamp against degenerate windows
    sigma_f2 = float(np.mean(fvar))

    # Weighted LS fit fc = a*t + b with weights = total power.
    # Apply sqrt(w) to A and fc so lstsq minimises sum_i w_i * resid_i**2
    # (the canonical WLS objective). Multiplying by w directly would
    # weight by w_i**2 instead.
    w = total / total.max()
    sqrt_w = np.sqrt(w)
    A = np.stack([t_travel, np.ones_like(t_travel)], axis=1)
    A_w = A * sqrt_w[:, None]
    fc_w = sqrt_w * fc
    m, *_ = np.linalg.lstsq(A_w, fc_w, rcond=None)
    slope, intercept = m

    # Standard error of the slope. cov = sigma_res2 * (A^T W A)^(-1);
    # A_w.T @ A_w == A^T diag(w) A by construction.
    resid = fc - (slope * t_travel + intercept)
    dof = max(1, t_travel.size - 2)
    sigma_res2 = np.sum(w * resid**2) / dof
    cov = sigma_res2 * np.linalg.pinv(A_w.T @ A_w)
    slope_sigma = float(np.sqrt(max(cov[0, 0], 0.0)))

    if slope >= 0:
        q = float("inf")
        q_sigma = float("inf")
    else:
        q = -np.pi * sigma_f2 / slope
        q_sigma = np.pi * sigma_f2 * slope_sigma / slope**2

    return AttenuationResult(
        q=float(q),
        q_sigma=float(q_sigma),
        method="centroid",
        diagnostic=dict(
            t=t_travel,
            fc=fc,
            sigma_f2=np.array(sigma_f2),
            slope=np.array(slope),
            intercept=np.array(intercept),
        ),
    )


def spectral_ratio_Q(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slowness: float,
    window_length: float = 5.0e-4,
    f_range: tuple[float, float] = (3000.0, 20000.0),
    reference: int = 0,
    pick_intercept: float = 0.0,
) -> AttenuationResult:
    """
    Attenuation from the classical spectral-ratio (log-slope) method.

    Under constant-``Q`` attenuation the log amplitude ratio between
    two traces at travel times ``t1`` and ``t2`` is linear in
    frequency with slope ``-pi*(t2 - t1)/Q``:

        log(|A2(f)| / |A1(f)|) = -pi*(t2-t1)/Q * f + const.

    A weighted LS fit of ``log_ratio`` vs ``f`` across frequency and
    receivers yields ``Q``. This is the Bath (1974) estimator.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
        Array-sonic gather.
    dt : float
        Sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m).
    slowness : float
        Formation slowness of the analysed arrival (s/m); used to
        compute travel-time differences across the array.
    window_length : float, default 5e-4
        Length of the Hann-tapered window (s) used for the local
        spectrum at each receiver.
    f_range : (float, float), default (3000, 20000)
        Inclusive frequency band (Hz) over which the log-ratio fit is
        performed.
    reference : int, default 0
        Index of the receiver used as the denominator ``A_1`` in the
        log ratio. The fit combines all other receivers against this
        reference.
    pick_intercept : float, default 0.0
        Absolute record time at which the first-breaking wave passes
        ``offset = 0``; affects only the absolute window placement.

    Returns
    -------
    AttenuationResult
        With ``method = "spectral_ratio"``. Returns ``q = NaN`` when
        no frequency bin has positive amplitude at every receiver.

    References
    ----------
    Bath, M. (1974). *Spectral Analysis in Geophysics.* Developments
    in Solid Earth Geophysics 7, Elsevier.
    """
    t_travel = pick_intercept + offsets * slowness
    spec, freqs = _windowed_spectrum(data, dt, t_travel, window_length)
    band = (freqs >= f_range[0]) & (freqs <= f_range[1])
    f_band = freqs[band]
    amp = np.abs(spec[:, band])
    ref_amp = amp[reference]
    eps = 1e-12
    mask = (ref_amp > eps) & np.all(amp > eps, axis=0)
    if not mask.any():
        return AttenuationResult(
            q=float("nan"), q_sigma=float("nan"), method="spectral_ratio", diagnostic={}
        )
    f_used = f_band[mask]

    # Build the stacked system: for each receiver i != reference,
    # log(A_i / A_ref) = -pi*(t_i - t_ref)*f / Q + c_i.
    # Unknowns: [1/Q, c_1, c_2, ..., c_{n_rec-1}].
    idx_others = [i for i in range(data.shape[0]) if i != reference]
    blocks = []
    y_all = []
    w_all = []
    for k, i in enumerate(idx_others):
        dt_i = t_travel[i] - t_travel[reference]
        y = np.log(amp[i, mask] / ref_amp[mask])
        row = np.zeros((f_used.size, 1 + len(idx_others)))
        row[:, 0] = -np.pi * dt_i * f_used
        row[:, 1 + k] = 1.0
        blocks.append(row)
        y_all.append(y)
        w_all.append(np.sqrt(amp[i, mask] * ref_amp[mask]))

    A = np.vstack(blocks)
    y = np.concatenate(y_all)
    w = np.concatenate(w_all)
    # Apply sqrt(w) so lstsq minimises sum_i w_i_norm * resid_i**2;
    # multiplying by w directly would weight by w**2.
    w_norm = w / w.max()
    sqrt_w = np.sqrt(w_norm)
    A_w = A * sqrt_w[:, None]
    y_w = sqrt_w * y
    m, *_ = np.linalg.lstsq(A_w, y_w, rcond=None)
    inv_q = m[0]
    resid = y - A @ m
    dof = max(1, y.size - m.size)
    sigma_res2 = np.sum(w_norm * resid**2) / dof
    cov = sigma_res2 * np.linalg.pinv(A_w.T @ A_w)
    inv_q_sigma = float(np.sqrt(max(cov[0, 0], 0.0)))

    if inv_q <= 0:
        q = float("inf")
        q_sigma = float("inf")
    else:
        q = 1.0 / inv_q
        q_sigma = inv_q_sigma / inv_q**2

    return AttenuationResult(
        q=float(q),
        q_sigma=float(q_sigma),
        method="spectral_ratio",
        diagnostic=dict(freqs=f_used, inv_q=np.array(inv_q)),
    )
