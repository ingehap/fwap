"""
Slowness-time coherence (STC / semblance).

Implements the coherence surface that the rule-based picker in
:mod:`fwap.picker` consumes. STC is the workhorse of Part 1 of the book
(automatic multi-wave log production).

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 1. Editions Technip,
  Paris. ISBN 978-2-7108-0664-6.
* Kimball, C. V., & Marzetta, T. L. (1984). Semblance processing of
  borehole acoustic array data. *Geophysics* 49(3), 274-281.
* Neidell, N. S., & Taner, M. T. (1971). Semblance and other coherency
  measures for multichannel data. *Geophysics* 36(3), 482-497.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from fwap._common import _phase_shift


@dataclass
class STCResult:
    """
    Slowness-time coherence surface returned by :func:`stc`.

    Attributes
    ----------
    slowness : ndarray, shape (n_slowness,)
        Trial slownesses (s/m), increasing.
    time : ndarray, shape (n_time,)
        Start time of each STC window relative to the start of the
        record (s).
    coherence : ndarray, shape (n_slowness, n_time)
        Semblance value in ``[0, 1]``; ``NaN`` for windows whose total
        energy falls below the ``min_energy_fraction`` floor passed to
        :func:`stc`.
    window_length : float
        STC time-window length actually used (s). Kept on the result
        so downstream picker code (see :func:`fwap.picker.pick_modes`
        and :func:`fwap.picker.track_modes`) can derive a sensible
        default time scale.
    amplitude : ndarray or None, shape (n_slowness, n_time)
        Per-cell stack amplitude (same units as the input data),
        defined as the RMS over the STC time window of the
        per-trace stack ``(sum_i x_i_aligned) / N_traces``. For a
        single coherent mode of amplitude ``A`` on every trace this
        equals the RMS of ``A`` (so a unit-amplitude sine on every
        trace gives ``amplitude = 1/sqrt(2)``). ``NaN`` at the same
        bins where ``coherence`` is ``NaN``. ``None`` on results
        from STC variants that have not yet been extended to track
        amplitude (e.g. the dispersion-corrected STC of
        :func:`fwap.dispersion.dispersive_stc`).
    """

    slowness: np.ndarray
    time: np.ndarray
    coherence: np.ndarray
    window_length: float
    amplitude: np.ndarray | None = None


def semblance(window: np.ndarray) -> float:
    """
    Semblance of co-aligned windowed traces.

    Parameters
    ----------
    window : ndarray, shape (N, L)
        ``N`` traces of length ``L``, already time-aligned on the trial
        slowness.

    Returns
    -------
    float
        Semblance in ``[0, 1]`` (1 = perfectly coherent). Returns
        ``NaN`` when the denominator is zero (an all-zero window).

    Notes
    -----
    Semblance (Neidell & Taner, 1971, *Geophysics* 36(3), 482-497) is
    the ratio of stack power to sum-of-traces power:

        rho = ( sum_t (sum_i x_i(t))**2 )
              --------------------------
              N * sum_t sum_i x_i(t)**2

    which lies in [0, 1] with 1 for perfectly coherent traces. Callers
    needing an energy floor should filter windows upstream -- the
    per-window cost of that check in the inner loop is identical, and
    the appropriate threshold depends on the caller.
    """
    stack = np.sum(window, axis=0)
    num = float(np.sum(stack * stack))
    den = window.shape[0] * np.sum(window * window)
    if den <= 0.0:
        return float("nan")
    return float(num / den)


def stc(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slowness_range: tuple[float, float] = (50e-6, 500e-6),
    n_slowness: int = 181,
    window_length: float = 4.0e-4,
    time_step: int = 1,
    min_energy_fraction: float = 1.0e-8,
) -> STCResult:
    """
    Slowness-time coherence map (Kimball & Marzetta, 1984).

    Trace alignment is done in the frequency domain (a fractional-sample
    phase shift per slowness) so that slowness resolution is a true
    function of the slowness grid, not quantised by ``dt / dx``. One
    rFFT / iRFFT pair per slowness plus a single
    ``np.lib.stride_tricks.sliding_window_view`` stack gives a fully
    vectorised implementation.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
    dt : float
        Sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m).
    slowness_range : (float, float)
        Lower and upper slowness to scan (s/m).
    n_slowness : int
        Number of slowness samples in the grid.
    window_length : float
        STC time-window length (s).
    time_step : int
        Stride of window starts in samples.
    min_energy_fraction : float
        Windows with energy below this fraction of the gather RMS are
        returned as ``NaN`` in the coherence map.
    """
    n_rec, n_samp = data.shape
    if offsets.size != n_rec:
        raise ValueError("offsets must have length n_rec")
    if n_slowness < 2:
        raise ValueError("n_slowness must be >= 2")

    s_min, s_max = slowness_range
    slowness = np.linspace(s_min, s_max, n_slowness)
    L = max(2, int(round(window_length / dt)))

    # Sliding-window start indices
    t_idx = np.arange(0, n_samp - L + 1, time_step)
    time = t_idx * dt
    n_t = t_idx.size

    # FFT each trace once
    spec = np.fft.rfft(data, axis=1)
    f = np.fft.rfftfreq(n_samp, d=dt)

    # Threshold on denominator based on gather RMS
    gather_rms2 = float(np.mean(data**2) + 1e-30)
    den_floor = min_energy_fraction * n_rec * L * gather_rms2

    rel_off = offsets - offsets[0]
    rho = np.empty((n_slowness, n_t), dtype=float)
    # Amplitude log: per-trace RMS of the stack. ``num`` is the
    # already-windowed stack-power sum, so ``sqrt(num / L) / N_traces``
    # gives the RMS of one trace's contribution to the stack; for a
    # single coherent mode of unit amplitude on every trace this
    # returns 1/sqrt(2) (the RMS of a unit sine).
    amp = np.empty((n_slowness, n_t), dtype=float)

    for k, s in enumerate(slowness):
        tau = rel_off * s
        shifted = np.fft.irfft(_phase_shift(spec, f, tau), n=n_samp, axis=1)

        # windows[i, j, :] is trace i starting at sample t_idx[j], length L
        windows = np.lib.stride_tricks.sliding_window_view(
            shifted, window_shape=L, axis=-1
        )
        if time_step != 1:
            windows = windows[:, ::time_step]
        windows = windows[:, :n_t]

        stack = windows.sum(axis=0)  # (n_t, L)
        num = (stack * stack).sum(axis=-1)  # (n_t,)
        den = n_rec * (windows * windows).sum(axis=(0, -1))  # (n_t,)
        mask = den > den_floor
        rho_k = np.full(n_t, np.nan, dtype=float)
        rho_k[mask] = num[mask] / den[mask]
        rho[k] = rho_k
        amp_k = np.full(n_t, np.nan, dtype=float)
        amp_k[mask] = np.sqrt(num[mask] / L) / n_rec
        amp[k] = amp_k

    return STCResult(
        slowness=slowness,
        time=time,
        coherence=rho,
        window_length=window_length,
        amplitude=amp,
    )


def find_peaks(
    result: STCResult,
    threshold: float = 0.5,
    min_separation_s: float = 1.0e-4,
    min_separation_slow: float = 1.5e-5,
) -> np.ndarray:
    """
    Local-maxima picker on an STC coherence surface.

    Uses ``scipy.ndimage.maximum_filter`` to find each cell equal to
    its neighbourhood maximum in a single vectorised pass.

    Returns
    -------
    peaks : ndarray, shape (n_peaks, n_cols)
        Rows are ``[slowness, time, coherence]`` -- or
        ``[slowness, time, coherence, amplitude]`` when the STC
        result carries an ``amplitude`` map -- sorted by descending
        coherence.
    """
    rho = np.nan_to_num(result.coherence, nan=-np.inf)
    s_axis = result.slowness
    t_axis = result.time
    n_cols = 4 if result.amplitude is not None else 3
    if rho.size == 0:
        return np.empty((0, n_cols))

    ds = s_axis[1] - s_axis[0] if s_axis.size > 1 else 1.0
    dt = t_axis[1] - t_axis[0] if t_axis.size > 1 else 1.0
    ws = max(1, int(round(min_separation_slow / ds)))
    wt = max(1, int(round(min_separation_s / dt)))
    neighbourhood = (2 * ws + 1, 2 * wt + 1)

    local_max = maximum_filter(rho, size=neighbourhood, mode="nearest")
    mask = (rho == local_max) & (rho >= threshold)
    si, ti = np.where(mask)
    if si.size == 0:
        return np.empty((0, n_cols))
    cols = [s_axis[si], t_axis[ti], rho[si, ti]]
    if result.amplitude is not None:
        amp = np.nan_to_num(result.amplitude, nan=0.0)
        cols.append(amp[si, ti])
    peaks = np.column_stack(cols)
    order = np.argsort(-peaks[:, 2])
    return peaks[order]
