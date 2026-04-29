"""
Dipole flexural processing: dispersion estimation and dispersive STC.

Implements the dipole-sonic algorithms of Part 3 of the book. Because
the borehole flexural mode is dispersive, ordinary STC biases shear
slowness high; this module provides three remedies:
narrow-band STC (:func:`narrow_band_stc`), per-frequency phase-slowness
estimation (:func:`phase_slowness_from_f_k`,
:func:`phase_slowness_matrix_pencil`), and dispersion-corrected STC
(:func:`dispersive_stc`).

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 3. Editions Technip,
  Paris. ISBN 978-2-7108-0664-6.
* Kimball, C. V. (1998). Shear slowness measurement by dispersive
  processing of the borehole flexural mode. *Geophysics* 63(2),
  337-344.
* Ekstrom, M. P. (1995). Dispersion estimation from borehole acoustic
  arrays using a modified matrix pencil algorithm. *29th Asilomar
  Conference on Signals, Systems and Computers*, 449-453.
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes*,
  Chapter 4. CRC Press.
* Schmitt, D. P. (1988). Shear-wave logging in elastic formations.
  *Journal of the Acoustical Society of America* 84(6), 2230-2244.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

PhaseSlownessMethod = Literal["frequency_unwrap", "spatial_unwrap"]

import numpy as np
from scipy.signal import butter, sosfiltfilt

from fwap._common import logger
from fwap.coherence import STCResult, stc
from fwap.synthetic import pseudo_rayleigh_dispersion


def _batched_wls_phase_slope(
    phase_block: np.ndarray,
    amp_block: np.ndarray,
    x: np.ndarray,
    f_out: np.ndarray,
    amp_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a weighted least-squares ``phi = slope * x + b`` at every
    frequency in one vectorised pass and return (slowness, quality).

    The per-frequency ``numpy.linalg.lstsq`` loop in the older code
    spent almost all its time in Python-call overhead. The closed-form
    WLS solution is inexpensive to batch across the frequency axis
    because the design matrix depends only on ``x`` (receiver offsets),
    which is constant.

    Parameters
    ----------
    phase_block : ndarray, shape (n_rec, n_f)
        Unwrapped (or wrapped; caller's choice) phase vs (receiver,
        frequency).
    amp_block : ndarray, shape (n_rec, n_f)
        Amplitude spectrum at the same (receiver, frequency) grid.
        Per-frequency weights are ``amp_block[:, j] / amp_block[:, j].max()``.
    x : ndarray, shape (n_rec,)
        Receiver offset axis (m), typically ``offsets - offsets[0]``.
    f_out : ndarray, shape (n_f,)
        Frequency axis (Hz); frequencies <= 0 produce ``NaN`` slowness.
    amp_norm : ndarray, shape (n_f,)
        Band-normalised amplitude weight used to modulate the per-
        frequency fit quality.

    Returns
    -------
    slow : ndarray, shape (n_f,)
        ``-slope / (2 pi f)`` at each in-band frequency.
    qual : ndarray, shape (n_f,)
        ``exp(-(rmse/pi)^2) * amp_norm``.
    """
    n_rec, n_f = phase_block.shape
    slow = np.zeros(n_f, dtype=float)
    qual = np.zeros(n_f, dtype=float)

    col_max = amp_block.max(axis=0)  # (n_f,)
    valid = col_max > 1e-12
    if not valid.any():
        return slow, qual

    # Per-column weights (n_rec, n_f); columns with zero max stay 0.
    w = np.zeros_like(amp_block)
    w[:, valid] = amp_block[:, valid] / col_max[valid]

    # Weighted sums along the receiver axis -- closed-form WLS.
    S_w = w.sum(axis=0)  # (n_f,)
    S_wx = (w * x[:, None]).sum(axis=0)
    S_wxx = (w * (x[:, None] ** 2)).sum(axis=0)
    S_wy = (w * phase_block).sum(axis=0)
    S_wxy = (w * x[:, None] * phase_block).sum(axis=0)
    denom = S_w * S_wxx - S_wx**2
    safe = valid & (np.abs(denom) > 1e-30)
    slope = np.zeros(n_f, dtype=float)
    intercept = np.zeros(n_f, dtype=float)
    slope[safe] = (S_w[safe] * S_wxy[safe] - S_wx[safe] * S_wy[safe]) / denom[safe]
    intercept[safe] = (S_wy[safe] - slope[safe] * S_wx[safe]) / S_w[safe]

    # Residual RMSE per frequency.
    fit = slope[None, :] * x[:, None] + intercept[None, :]
    resid = phase_block - fit
    num = (w * resid**2).sum(axis=0)
    den = np.where(S_w > 0, S_w, 1.0)
    rmse = np.sqrt(np.clip(num / den, 0.0, None))
    q_fit = np.exp(-((rmse / np.pi) ** 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        slow = np.where(safe & (f_out > 0), -slope / (2.0 * np.pi * f_out), np.nan)
    slow = np.where(valid & ~np.isfinite(slow) & (f_out > 0), 0.0, slow)
    qual = np.where(valid, q_fit * amp_norm, 0.0)
    return slow, qual


def bandpass(
    data: np.ndarray, dt: float, f_lo: float, f_hi: float, order: int = 4
) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass along the time axis.

    Parameters
    ----------
    data : ndarray
        Input gather; filtering is applied along ``axis=-1``.
    dt : float
        Sampling interval (s).
    f_lo, f_hi : float
        Lower and upper corner frequencies (Hz). Values are clipped
        to the open interval ``(0, Nyquist)`` before designing the
        filter.
    order : int, default 4
        Filter order per pass. The effective order is doubled by the
        zero-phase forward-backward application (``sosfiltfilt``).

    Returns
    -------
    ndarray
        Filtered data, same shape as ``data``.
    """
    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = max(f_lo / nyq, 1.0e-6)
    high = min(f_hi / nyq, 0.999)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def narrow_band_stc(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    f_lo: float = 1500.0,
    f_hi: float = 4000.0,
    **stc_kwargs,
) -> STCResult:
    """
    Band-pass the gather, then run STC -> lower-bias shear slowness.

    Convenience wrapper that applies :func:`bandpass` to ``data`` and
    forwards the result (together with ``dt`` and ``offsets``) to
    :func:`fwap.coherence.stc`. Keyword arguments in ``stc_kwargs`` are
    passed through unchanged to :func:`stc`.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
    dt : float
        Sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m).
    f_lo, f_hi : float
        Band-pass corner frequencies (Hz); see :func:`bandpass`.
    **stc_kwargs
        Forwarded to :func:`fwap.coherence.stc` (e.g. ``slowness_range``,
        ``n_slowness``, ``window_length``, ``time_step``).

    Returns
    -------
    STCResult
    """
    return stc(bandpass(data, dt, f_lo, f_hi), dt=dt, offsets=offsets, **stc_kwargs)


@dataclass
class DispersionCurve:
    """
    Phase-slowness dispersion curve.

    Attributes
    ----------
    freq : ndarray, shape (n_f,)
        Frequencies at which the curve is evaluated (Hz).
    slowness : ndarray, shape (n_f,)
        Estimated phase slowness (s/m) at each frequency. ``NaN`` /
        ``0`` where the estimator was undefined (typically below the
        source low-frequency cutoff).
    quality : ndarray, shape (n_f,)
        Fit-quality weight in ``[0, 1]`` -- the product of a phase-fit
        residual score and a band-normalised amplitude weight.
        Consumers of this curve (:func:`shear_slowness_from_dispersion`
        and plotting code) should mask or weight by this value.
    """

    freq: np.ndarray
    slowness: np.ndarray
    quality: np.ndarray


def phase_slowness_from_f_k(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    f_range: tuple[float, float] = (500.0, 8000.0),
    method: PhaseSlownessMethod = "frequency_unwrap",
) -> DispersionCurve:
    """
    Estimate phase slowness vs frequency by weighted LS phase-vs-offset.

    ``method``
    ----------
    ``"frequency_unwrap"`` (default)
        Cross-correlate each trace against the reference (first) trace
        in the frequency domain, then unwrap the cross-spectrum phase
        along the **frequency** axis per trace. Because the cross
        spectrum of trace 0 with itself is identically real, the
        unwrap is anchored; trace-to-trace 2*pi ambiguities are
        eliminated.
    ``"spatial_unwrap"``
        Unwrap across receivers at each frequency. Fails when the
        total phase swing across the aperture exceeds pi, i.e. above
        roughly ``1 / (2 * aperture * s)`` Hz; prefer
        ``"frequency_unwrap"`` unless replicating a specific reference
        output.

    Validity band
    -------------
    The frequency-domain unwrap is itself valid provided
    ``|dphi/df * df| < pi``, i.e. the arrival time is less than half
    the record length. That holds for typical sonic data.
    """
    n_rec, n_samp = data.shape
    spec = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(n_samp, d=dt)
    band = (freqs >= f_range[0]) & (freqs <= f_range[1])
    f_out = freqs[band]
    slow = np.zeros_like(f_out)
    qual = np.zeros_like(f_out)
    x = offsets - offsets[0]

    if method == "frequency_unwrap":
        # Cross-spectrum with trace 0 removes the per-trace DC phase.
        # Then unwrap along frequency per trace -- but only within the
        # range where the trace amplitude is a meaningful fraction of
        # its peak. Below that threshold, quantisation noise produces
        # spurious 2*pi branch jumps that propagate into every
        # higher-frequency bin.
        ref = 0
        cross = spec * np.conj(spec[ref])
        amp = np.abs(spec)
        phase_rel = np.zeros_like(cross, dtype=float)
        amp_thresh_frac = 0.05  # 5% of peak per trace
        for i in range(n_rec):
            a_i = amp[i]
            if a_i.max() < 1e-12:
                continue
            strong = a_i > amp_thresh_frac * a_i.max()
            # Unwrap only the strong-SNR run. Weak bins keep the raw
            # wrapped phase -- they're amplitude-down-weighted anyway.
            raw = np.angle(cross[i])
            unwrapped = raw.copy()
            if strong.any():
                ks = np.where(strong)[0]
                unwrapped[ks[0] : ks[-1] + 1] = np.unwrap(raw[ks[0] : ks[-1] + 1])
            phase_rel[i] = unwrapped
        phase_block = phase_rel[:, band]
        amp_block = amp[:, band]
        amp_mean = amp_block.mean(axis=0)
        amp_norm = (amp_mean / (amp_mean.max() + 1e-30)) if amp_mean.size else amp_mean
        slow, qual = _batched_wls_phase_slope(
            phase_block, amp_block, x, f_out, amp_norm
        )

    elif method == "spatial_unwrap":
        amp_all = np.abs(spec)
        spec_band = spec[:, band]
        amp_block = amp_all[:, band]
        amp_mean = amp_block.mean(axis=0)
        amp_norm = (amp_mean / (amp_mean.max() + 1e-30)) if amp_mean.size else amp_mean
        # Unwrap each frequency column along the receiver axis in one
        # vectorised call instead of a Python loop over frequencies.
        phase_block = np.unwrap(np.angle(spec_band), axis=0)
        slow, qual = _batched_wls_phase_slope(
            phase_block, amp_block, x, f_out, amp_norm
        )
    else:
        raise ValueError("method must be 'frequency_unwrap' or 'spatial_unwrap'")

    return DispersionCurve(freq=f_out, slowness=slow, quality=qual)


def phase_slowness_matrix_pencil(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    f_range: tuple[float, float] = (500.0, 8000.0),
) -> DispersionCurve:
    """
    Single-mode phase slowness via a matrix-pencil / ESPRIT-style
    estimator at each frequency.

    At a given ``f`` the spatial samples ``X_i = A exp(2*pi*i*f*s*x_i)``
    form a geometric progression along a uniform array. Stacking two
    shifted windows and taking the generalised eigenvalue yields
    ``exp(2*pi*i*f*s*dx)``; the slowness follows. Bypasses phase
    unwrapping entirely (Ekstroem, 1995; related to Prony and MUSIC).

    Caveats
    -------
    Requires a uniform receiver spacing; assumes a single dominant
    spatial mode per frequency. For a two-mode problem one should use
    a true matrix-pencil algorithm with pencil parameter ``L > 1``.
    """
    n_rec, n_samp = data.shape
    if n_rec < 3:
        raise ValueError("matrix pencil needs >= 3 receivers")
    dx_samples = np.diff(offsets)
    dx = float(dx_samples.mean())
    if not np.allclose(dx_samples, dx, rtol=1e-3):
        raise ValueError("matrix-pencil estimator requires uniform receiver spacing")

    spec = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(n_samp, d=dt)
    band = (freqs >= f_range[0]) & (freqs <= f_range[1])
    f_out = freqs[band]
    slow = np.zeros_like(f_out)
    qual = np.zeros_like(f_out)

    spec_band = spec[:, band]  # (n_rec, n_f)
    amp_mean = np.abs(spec_band).mean(axis=0)
    amp_norm = (amp_mean / (amp_mean.max() + 1e-30)) if amp_mean.size else amp_mean

    # Pencil ratio at every frequency in one pass.
    # For each column col of spec_band:
    #     num_j = sum_i conj(col[i]) * col[i+1]
    #     den_j = sum_i conj(col[i]) * col[i]
    # and z_j = num_j / den_j encodes exp(-2 pi i f s dx) for a single
    # +s mode. The frequency-dimension loop was one Python-level call
    # per bin; here we stack them all as a (n_rec-1, n_f) inner product.
    x0 = spec_band[:-1, :]  # (n_rec-1, n_f)
    x1 = spec_band[1:, :]
    num = np.sum(np.conj(x0) * x1, axis=0)  # (n_f,)
    den = np.sum(np.conj(x0) * x0, axis=0)  # (n_f,)
    safe = np.abs(den) >= 1e-15
    z = np.zeros_like(num)
    z[safe] = num[safe] / den[safe]
    mag = np.abs(z)
    angle = np.angle(z)

    positive_f = f_out > 0
    valid = safe & positive_f & (mag >= 1e-12)
    # col[i+1] / col[i] ~ exp(-2 pi i f s dx) for a +s mode, so
    # s = -angle / (2 pi f dx). Bins that fail the validity mask keep
    # the zero default.
    slow[valid] = -angle[valid] / (2.0 * np.pi * f_out[valid] * dx)
    q_circle = np.clip(1.0 - np.abs(mag - 1.0), 0.0, 1.0)
    qual = np.where(valid, q_circle * amp_norm, 0.0)

    return DispersionCurve(freq=f_out, slowness=slow, quality=qual)


def shear_slowness_from_dispersion(
    curve: DispersionCurve,
    f_lo: float = 500.0,
    f_hi: float = 2500.0,
    quality_threshold: float = 0.8,
) -> float:
    """
    Quality-weighted mean of ``s(f)`` in the low-frequency asymptote
    band (Kimball, 1998, eq. 14).

    If no dispersion points in ``[f_lo, f_hi]`` meet
    ``quality_threshold`` the call falls back to the unweighted set of
    finite points in the same band and emits a ``logging.warning`` so
    the caller can see that their quality gate was dropped.
    """
    mask = (
        (curve.freq >= f_lo)
        & (curve.freq <= f_hi)
        & (curve.quality >= quality_threshold)
        & np.isfinite(curve.slowness)
    )
    if not mask.any():
        logger.warning(
            "shear_slowness_from_dispersion: no points pass "
            "quality_threshold=%g in band [%g, %g] Hz; falling back to "
            "finite points only.",
            quality_threshold,
            f_lo,
            f_hi,
        )
        mask = (curve.freq >= f_lo) & (curve.freq <= f_hi) & np.isfinite(curve.slowness)
        if not mask.any():
            return float("nan")
    w = np.clip(curve.quality[mask], 1e-3, 1.0)
    s = curve.slowness[mask]
    return float(np.sum(w * s) / np.sum(w))


def dispersive_stc(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    dispersion_family: Callable[[float], Callable[[np.ndarray], np.ndarray]],
    shear_slowness_range: tuple[float, float] = (150e-6, 600e-6),
    n_slowness: int = 91,
    f_range: tuple[float, float] = (500.0, 6000.0),
    window_length: float = 1.5e-3,
    time_step: int = 4,
    min_energy_fraction: float = 1.0e-8,
) -> STCResult:
    """
    Dispersive STC in the spirit of Kimball (1998).

    For each trial shear slowness ``s_shear``, the expected phase
    slowness ``s_phase(f) = dispersion_family(s_shear)(f)`` is computed
    and used to back-project every frequency bin before windowed
    semblance. A true flexural arrival is collapsed to zero relative
    delay at the correct ``s_shear``, removing the high-frequency bias
    that plagues ordinary STC on dispersive modes.

    Parameters
    ----------
    dispersion_family
        Callable that maps a candidate shear slowness to a phase-
        slowness function ``s_phase(f)``. The returned callable **must
        accept a NumPy array** of in-band frequencies and return a
        same-shape array of slownesses (s/m). For the phenomenological
        flexural model in this module::

            lambda s: dipole_flexural_dispersion(vs=1/s, a_borehole=...)

    Returns
    -------
    STCResult
        ``slowness`` holds the *shear* slowness axis (not the phase
        slowness -- the whole point of the transform).

    References
    ----------
    Kimball, C. V. (1998). Shear slowness measurement by dispersive
    processing of the borehole flexural mode. *Geophysics* 63(2),
    337-344, eqs. 5-10.
    """
    n_rec, n_samp = data.shape
    s_shear_axis = np.linspace(*shear_slowness_range, n_slowness)

    spec = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(n_samp, d=dt)
    band = (freqs >= f_range[0]) & (freqs <= f_range[1])

    L = max(2, int(round(window_length / dt)))
    t_idx = np.arange(0, n_samp - L + 1, time_step)
    time = t_idx * dt
    n_t = t_idx.size

    gather_rms2 = float(np.mean(data**2) + 1e-30)
    den_floor = min_energy_fraction * n_rec * L * gather_rms2

    rel_off = offsets - offsets[0]
    rho = np.empty((n_slowness, n_t), dtype=float)
    # Per-cell stack amplitude, same definition as fwap.coherence.stc
    # (RMS of the per-trace stack contribution); see STCResult.
    amp = np.empty((n_slowness, n_t), dtype=float)

    # Pre-compute the constant factor shared across all slowness trials:
    # tau_ij = rel_off[i] * s_phase(freqs[j]) is a rank-1 outer product,
    # so the full per-receiver phase matrix is
    #   phase[i, j] = exp(2*pi*i * freqs[j] * rel_off[i] * s_phase[j])
    # We factor out the ``freqs[j] * rel_off[i]`` grid and multiply in
    # s_phase(freqs) at each trial.
    freq_off = 2.0 * np.pi * freqs[None, :] * rel_off[:, None]  # (n_rec, n_f)
    band_mask = band[None, :]  # (1, n_f)
    # freqs[0] == 0 is always masked out of the band, but guard anyway.
    band_pos = band & (freqs > 0)

    for k, s_shear in enumerate(s_shear_axis):
        s_phase_fn = dispersion_family(s_shear)
        s_of_f = np.zeros_like(freqs)
        s_of_f[band_pos] = s_phase_fn(freqs[band_pos])
        # shifted_spec = spec * exp(+2*pi*i * freq_off * s_phase), with
        # out-of-band bins zeroed (double as leakage guard).
        phase = np.exp(1j * freq_off * s_of_f[None, :])
        shifted_spec = np.where(band_mask, spec * phase, 0.0)
        shifted = np.fft.irfft(shifted_spec, n=n_samp, axis=1)

        windows = np.lib.stride_tricks.sliding_window_view(
            shifted, window_shape=L, axis=-1
        )
        if time_step != 1:
            windows = windows[:, ::time_step]
        windows = windows[:, :n_t]

        stack = windows.sum(axis=0)
        num = (stack * stack).sum(axis=-1)
        den = n_rec * (windows * windows).sum(axis=(0, -1))
        rho_k = np.full(n_t, np.nan, dtype=float)
        mask = den > den_floor
        rho_k[mask] = num[mask] / den[mask]
        rho[k] = rho_k
        amp_k = np.full(n_t, np.nan, dtype=float)
        amp_k[mask] = np.sqrt(num[mask] / L) / n_rec
        amp[k] = amp_k

    return STCResult(
        slowness=s_shear_axis,
        time=time,
        coherence=rho,
        window_length=window_length,
        amplitude=amp,
    )


def dispersive_pseudo_rayleigh_stc(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    *,
    v_fluid: float = 1500.0,
    a_borehole: float = 0.1,
    shear_slowness_range: tuple[float, float] = (130e-6, 500e-6),
    n_slowness: int = 91,
    f_range: tuple[float, float] = (3000.0, 12000.0),
    window_length: float = 1.0e-3,
    time_step: int = 4,
    min_energy_fraction: float = 1.0e-8,
) -> STCResult:
    r"""
    Dispersive STC for the pseudo-Rayleigh / guided trapped mode.

    Direct analogue of :func:`dispersive_stc` applied to the
    pseudo-Rayleigh dispersion law of
    :func:`fwap.synthetic.pseudo_rayleigh_dispersion`. For each trial
    formation shear slowness ``s_shear``, the expected phase slowness

    .. math::

        s_\mathrm{phase}(f; s_\mathrm{shear}) \;=\;
        s_\mathrm{shear} \;+\; (s_\mathrm{fluid} - s_\mathrm{shear})\,
        \frac{(f / f_c)^2}{1 + (f / f_c)^2},

    with :math:`s_\mathrm{fluid} = 1/V_\mathrm{fluid}` and
    :math:`f_c = V_s / (2\pi a)`, is used to back-project every
    frequency bin before the windowed semblance. A true pseudo-Rayleigh
    arrival collapses to zero relative delay at the correct
    ``s_shear``, removing the high-frequency bias that plagues plain
    STC for guided modes -- the same trick :func:`dispersive_stc`
    uses for the dipole flexural mode.

    The output ``slowness`` axis holds the **formation shear**
    slowness (i.e., the low-frequency cutoff of the pseudo-Rayleigh
    branch, equal to :math:`1/V_s`), not the per-frequency phase
    slowness. Pick the maximum-coherence row of the surface to recover
    formation Vs from a pseudo-Rayleigh-dominated gather.

    Existence constraint
    --------------------
    Pseudo-Rayleigh exists only in **fast formations**
    (:math:`V_s > V_\mathrm{fluid}`). The trial shear-slowness range
    is therefore required to satisfy
    ``shear_slowness_range[1] < 1 / v_fluid``; the wrapper raises
    ``ValueError`` for ranges that would step into the slow-formation
    regime where the dispersion law is undefined.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
        Real-valued multichannel gather.
    dt : float
        Sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m).
    v_fluid : float, default 1500.0
        Borehole-fluid acoustic velocity (m/s). Sets the high-
        frequency asymptote of the pseudo-Rayleigh phase slowness.
    a_borehole : float, default 0.1
        Borehole radius (m). Sets the cutoff frequency
        :math:`f_c = V_s / (2 \pi a)` of the dispersion law.
    shear_slowness_range : (float, float), default (130 us/m, 500 us/m)
        Lower and upper trial formation shear slowness (s/m). Must
        be strictly positive, ordered, and bounded above by
        ``1 / v_fluid``.
    n_slowness : int, default 91
        Number of trial shear-slowness points.
    f_range : (float, float), default (3000, 12000) Hz
        Inclusive frequency band over which the dispersion correction
        is applied. The pseudo-Rayleigh branch only exists above its
        low-frequency cutoff :math:`f_c`; for typical sonic geometry
        (a = 0.1 m, Vs ~ 2-5 km/s) :math:`f_c` falls in the 3-8 kHz
        band, motivating the higher-frequency default relative to
        :func:`dispersive_stc` (which targets the low-frequency
        flexural mode).
    window_length : float, default 1e-3
        STC time-window length (s). Pseudo-Rayleigh wavetrains are
        typically more impulsive than the flexural mode, so a 1 ms
        window suffices to bracket the dominant lobe at the default
        frequency band.
    time_step : int, default 4
        Stride of window starts in samples.
    min_energy_fraction : float, default 1e-8
        Energy floor relative to gather RMS; below this the
        coherence cell is reported as ``NaN``.

    Returns
    -------
    STCResult
        ``slowness`` is the formation shear-slowness axis; a true
        pseudo-Rayleigh arrival peaks at the correct :math:`1/V_s`
        on this axis (the whole point of the transform). All other
        fields follow :class:`fwap.coherence.STCResult` conventions.

    Raises
    ------
    ValueError
        If ``shear_slowness_range[1] >= 1 / v_fluid`` (the trial
        range would leave the pseudo-Rayleigh existence domain),
        if the range is mis-ordered, or if any bound is non-positive.

    See Also
    --------
    fwap.dispersion.dispersive_stc :
        The dipole-flexural-mode equivalent. The two routines share
        the same back-projection / semblance machinery; only the
        per-mode dispersion law differs.
    fwap.synthetic.pseudo_rayleigh_dispersion :
        The phenomenological dispersion law used here.
    """
    if shear_slowness_range[0] <= 0.0:
        raise ValueError("shear_slowness_range[0] must be positive")
    if shear_slowness_range[0] >= shear_slowness_range[1]:
        raise ValueError(
            "require shear_slowness_range[0] < shear_slowness_range[1]; got "
            f"{shear_slowness_range}"
        )
    s_fluid = 1.0 / v_fluid
    if shear_slowness_range[1] >= s_fluid:
        raise ValueError(
            f"shear_slowness_range[1] must be < 1/v_fluid "
            f"({s_fluid:.3e} s/m); pseudo-Rayleigh only exists in fast "
            f"formations (vs > v_fluid). Got "
            f"{shear_slowness_range[1]:.3e} s/m."
        )

    def _family(s_shear: float) -> Callable[[np.ndarray], np.ndarray]:
        return pseudo_rayleigh_dispersion(
            vs=1.0 / s_shear,
            v_fluid=v_fluid,
            a_borehole=a_borehole,
        )

    return dispersive_stc(
        data,
        dt=dt,
        offsets=offsets,
        dispersion_family=_family,
        shear_slowness_range=shear_slowness_range,
        n_slowness=n_slowness,
        f_range=f_range,
        window_length=window_length,
        time_step=time_step,
        min_energy_fraction=min_energy_fraction,
    )


# ---------------------------------------------------------------------
# Stress-vs-intrinsic anisotropy from fast/slow flexural dispersion
# (Sinha & Kostek 1996; Tang & Cheng 2004 sect. 5.3)
# ---------------------------------------------------------------------


@dataclass
class FlexuralDispersionDiagnosis:
    r"""
    Output of :func:`classify_flexural_anisotropy`.

    Attributes
    ----------
    classification : str
        One of ``"isotropic"``, ``"intrinsic"``, ``"stress_induced"``,
        ``"ambiguous"``.
    delta_low : float
        Mean :math:`s_b(f) - s_a(f)` over the low-frequency band
        (s/m). ``NaN`` if no quality-passing samples land in the
        band.
    delta_high : float
        Same over the high-frequency band.
    crossover_frequency : float or None
        Frequency (Hz) of the **first** zero-crossing of
        :math:`s_b(f) - s_a(f)` inside the bracket
        ``[f_low_band[0], f_high_band[1]]``. ``None`` when no
        crossover is detected (intrinsic / isotropic / ambiguous
        cases) or when no sign flip is observed inside the bracket.
    reasons : tuple of str
        Human-readable description of how the classification was
        reached. Useful for QC / display.
    """

    classification: str
    delta_low: float
    delta_high: float
    crossover_frequency: float | None
    reasons: tuple[str, ...]


def classify_flexural_anisotropy(
    curve_a: DispersionCurve,
    curve_b: DispersionCurve,
    *,
    quality_threshold: float = 0.5,
    min_anisotropy: float = 5.0e-6,
    f_low_band: tuple[float, float] = (1000.0, 2500.0),
    f_high_band: tuple[float, float] = (4000.0, 8000.0),
) -> FlexuralDispersionDiagnosis:
    r"""
    Discriminate stress-induced vs intrinsic flexural anisotropy
    from two dispersion curves.

    Following Sinha & Kostek (1996, *Geophysics* 61(6), 1899-1907) and
    Tang & Cheng (2004, sect. 5.3), the two cross-dipole flexural
    dispersion curves of a stress-anisotropic formation **cross over**
    in frequency: the low-frequency flexural mode samples ~1-2
    wavelengths radially into the formation -- primarily the far-field
    rock fabric -- while the high-frequency mode samples ~0.1-0.5
    wavelengths radially -- primarily the near-wellbore stress
    concentration. When the bulk fabric is isotropic but the borehole
    is loaded by an anisotropic far-field stress, the two regions see
    *different* anisotropy, so :math:`\Delta s(f) = s_b(f) - s_a(f)`
    changes sign between the bands. Intrinsic anisotropy (aligned
    fractures, layered shale fabric) shows no such crossover -- the
    same direction is slow at every frequency.

    Classification logic
    --------------------
    Compute :math:`\Delta s_\mathrm{low}` and :math:`\Delta s_\mathrm{high}`
    as the means of :math:`\Delta s(f)` over ``f_low_band`` and
    ``f_high_band`` respectively, restricted to bins where both
    curves have ``quality >= quality_threshold`` and a finite
    slowness. Then:

    - both bands :math:`< \mathrm{min\_anisotropy}` in magnitude
      :math:`\rightarrow` ``"isotropic"``;
    - both bands above threshold with the **same** sign
      :math:`\rightarrow` ``"intrinsic"``;
    - both bands above threshold with **opposite** signs
      :math:`\rightarrow` ``"stress_induced"`` (and the crossover
      frequency is reported);
    - any other combination (one band missing, only one band
      anisotropic) :math:`\rightarrow` ``"ambiguous"``.

    Parameters
    ----------
    curve_a, curve_b : DispersionCurve
        The two flexural dispersion curves -- typically obtained by
        running :func:`phase_slowness_from_f_k` on the two rotated
        cross-dipole components (XX' and YY' in the Alford-rotated
        frame). The two curves must share the same ``freq`` axis to
        within a small relative tolerance.
    quality_threshold : float, default 0.5
        Per-bin quality floor on each curve; bins below this on
        either curve are excluded from the band averages.
    min_anisotropy : float, default 5e-6 s/m
        Magnitude of :math:`\Delta s` below which a band is treated
        as isotropic. The default ~5 us/m corresponds to ~1.5 us/ft.
    f_low_band : (float, float), default (1000, 2500) Hz
        Low-frequency averaging band (samples deep into the
        formation).
    f_high_band : (float, float), default (4000, 8000) Hz
        High-frequency averaging band (samples near the borehole
        wall).

    Returns
    -------
    FlexuralDispersionDiagnosis
        Classification plus the diagnostic numbers used to reach it.

    Raises
    ------
    ValueError
        If the two curves have mismatched ``freq`` axes, or the
        bands are mis-ordered / overlap.

    See Also
    --------
    phase_slowness_from_f_k :
        Produces the input dispersion curves from raw cross-dipole
        sonic data.
    fwap.anisotropy.alford_rotation :
        Cross-dipole rotation that defines the fast / slow
        components from the raw XX/XY/YX/YY tensor.

    References
    ----------
    * Sinha, B. K., & Kostek, S. (1996). Stress-induced azimuthal
      anisotropy in borehole flexural waves. *Geophysics* 61(6),
      1899-1907.
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, Section 5.3 (stress-induced vs
      intrinsic anisotropy discrimination).
    """
    if f_low_band[0] >= f_low_band[1]:
        raise ValueError("f_low_band must be (lo, hi) with lo < hi")
    if f_high_band[0] >= f_high_band[1]:
        raise ValueError("f_high_band must be (lo, hi) with lo < hi")
    if f_low_band[1] >= f_high_band[0]:
        raise ValueError(
            "f_low_band and f_high_band must not overlap or touch; got "
            f"low={f_low_band}, high={f_high_band}"
        )
    if curve_a.freq.shape != curve_b.freq.shape:
        raise ValueError("dispersion curves must share the same freq axis")
    if curve_a.freq.size > 0 and not np.allclose(
        curve_a.freq, curve_b.freq, rtol=1.0e-6, atol=0.0
    ):
        raise ValueError("dispersion curves must share the same freq axis")

    f = curve_a.freq
    delta = curve_b.slowness - curve_a.slowness
    valid = (
        np.isfinite(curve_a.slowness)
        & np.isfinite(curve_b.slowness)
        & (curve_a.quality >= quality_threshold)
        & (curve_b.quality >= quality_threshold)
    )

    def _band_mean(lo: float, hi: float) -> float:
        m = valid & (f >= lo) & (f <= hi)
        if not m.any():
            return float("nan")
        return float(delta[m].mean())

    delta_low = _band_mean(*f_low_band)
    delta_high = _band_mean(*f_high_band)

    reasons: list[str] = []

    if not (np.isfinite(delta_low) and np.isfinite(delta_high)):
        if not np.isfinite(delta_low):
            reasons.append(f"low-f band {f_low_band} has no quality-passing samples")
        if not np.isfinite(delta_high):
            reasons.append(f"high-f band {f_high_band} has no quality-passing samples")
        return FlexuralDispersionDiagnosis(
            classification="ambiguous",
            delta_low=delta_low,
            delta_high=delta_high,
            crossover_frequency=None,
            reasons=tuple(reasons),
        )

    is_low_aniso = abs(delta_low) > min_anisotropy
    is_high_aniso = abs(delta_high) > min_anisotropy

    if not is_low_aniso and not is_high_aniso:
        return FlexuralDispersionDiagnosis(
            classification="isotropic",
            delta_low=delta_low,
            delta_high=delta_high,
            crossover_frequency=None,
            reasons=(
                f"|delta_s| below {min_anisotropy:.1e} s/m in both bands "
                f"(low={delta_low:.2e}, high={delta_high:.2e})",
            ),
        )

    if not (is_low_aniso and is_high_aniso):
        which = "high" if is_low_aniso else "low"
        return FlexuralDispersionDiagnosis(
            classification="ambiguous",
            delta_low=delta_low,
            delta_high=delta_high,
            crossover_frequency=None,
            reasons=(
                f"{which}-f band below min_anisotropy "
                f"(low={delta_low:.2e}, high={delta_high:.2e})",
            ),
        )

    if delta_low * delta_high > 0:
        return FlexuralDispersionDiagnosis(
            classification="intrinsic",
            delta_low=delta_low,
            delta_high=delta_high,
            crossover_frequency=None,
            reasons=(
                f"delta_s has same sign in both bands "
                f"(low={delta_low:.2e}, high={delta_high:.2e}); "
                f"no dispersion crossover",
            ),
        )

    # Opposite signs => stress-induced; locate the first crossover by
    # linear interpolation between the first pair of valid samples
    # whose delta_s differs in sign. Restrict the search to the
    # bracket spanning the two band means, [f_low_band[0],
    # f_high_band[1]] -- the meaningful crossover sits between them,
    # and a sign flip at frequencies outside that bracket would be a
    # spurious noise zero-crossing rather than the band-to-band
    # transition the classification is reporting.
    bracket = valid & (f >= f_low_band[0]) & (f <= f_high_band[1])
    f_valid = f[bracket]
    delta_valid = delta[bracket]
    crossover_freq: float | None = None
    if f_valid.size >= 2:
        sign_changes = np.where(np.diff(np.sign(delta_valid)) != 0)[0]
        if sign_changes.size > 0:
            i = int(sign_changes[0])
            f1, f2 = float(f_valid[i]), float(f_valid[i + 1])
            d1, d2 = float(delta_valid[i]), float(delta_valid[i + 1])
            if d2 != d1:
                crossover_freq = f1 - d1 * (f2 - f1) / (d2 - d1)
            else:
                crossover_freq = 0.5 * (f1 + f2)

    return FlexuralDispersionDiagnosis(
        classification="stress_induced",
        delta_low=delta_low,
        delta_high=delta_high,
        crossover_frequency=crossover_freq,
        reasons=(
            f"delta_s sign flips between bands "
            f"(low={delta_low:.2e}, high={delta_high:.2e})",
        ),
    )
