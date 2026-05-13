"""
Wavelet-shape and onset-polarity expert rules (post-pick filters).

Two further book-listed expert rules (Mari et al. 1994, Part 1)
applied **after** the slowness / coherence / time-order picker
finishes: a wavelet-shape goodness-of-fit gate (cross-correlation
against a Ricker template at the prior centre frequency) and an
onset-polarity gate (sign of the dominant excursion of the
moveout-aligned stack).

Both gates are opt-in -- callers reach for
:func:`filter_picks_by_shape` (single depth) or
:func:`filter_track_by_shape` (multi-depth) explicitly. The
underlying primitives :func:`onset_polarity` and
:func:`wavelet_shape_score` are also exposed for direct use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fwap._common import _phase_shift
from fwap.picker._types import DEFAULT_PRIORS, DepthPicks, ModePick


def _align_and_stack(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slowness: float,
    stc_window_start: float,
    stc_window_length: float,
    analysis_factor: float = 2.0,
) -> np.ndarray:
    """Frequency-domain align + per-trace average over an analysis window.

    Returns the per-trace mean of the moveout-aligned waveforms over
    a window of width ``analysis_factor * stc_window_length`` centred
    on the **midpoint** of the STC window (i.e. on
    ``stc_window_start + stc_window_length / 2``). Centring -- rather
    than starting -- on the STC window's midpoint matters for the
    polarity / shape gates because ``pick.time`` is a window-*start*
    time; a window started there may capture only a sidelobe of the
    underlying wavelet, whose true centroid sits roughly at the STC
    window's centre. The widened analysis window is also clipped to
    the available data range, so picks at the very edge of the
    record still produce a usable stack.
    """
    n_rec, n_samp = data.shape
    spec = np.fft.rfft(data, axis=1)
    f = np.fft.rfftfreq(n_samp, d=dt)
    rel_off = offsets - offsets[0]
    tau = rel_off * slowness
    shifted = np.fft.irfft(_phase_shift(spec, f, tau), n=n_samp, axis=1)
    L_analysis = max(2, int(round(analysis_factor * stc_window_length / dt)))
    centre_sample = int(round((stc_window_start + stc_window_length / 2.0) / dt))
    j0 = max(0, min(n_samp - L_analysis, centre_sample - L_analysis // 2))
    window = shifted[:, j0 : j0 + L_analysis]
    return window.mean(axis=0)


def onset_polarity(stack: np.ndarray) -> int:
    """Sign of the largest-absolute sample in a stacked waveform.

    Returns ``+1`` when the dominant excursion is positive, ``-1``
    when negative, and ``0`` for an all-zero input. The book's "onset
    polarity" expert rule (Mari et al. 1994, Part 1) gates picks
    against an expected first-motion sign convention; the dominant
    excursion of an STC-window-stacked pulse-like wavelet is the
    main lobe at the prior centre frequency, so its sign is the
    natural per-pick polarity readout.

    Caveat -- pulse-like wavelets only
    ----------------------------------
    This heuristic is only meaningful for *pulse-like* wavelets
    where one excursion clearly dominates the others (Ricker P /
    Ricker S, in the canonical 1994 monopole gather). For
    multi-cycle wavetrains -- Gabor / Stoneley -- the dominant
    excursion can be either a peak or a trough depending on the
    sub-sample alignment of the analysis window, so the polarity
    readout is not stable. In practice the polarity gate in
    :data:`DEFAULT_PRIORS` should only be enabled on the impulsive
    modes (P, S); leave Stoneley and PseudoRayleigh at
    ``polarity=0`` (the default = "ignore").
    """
    if stack.size == 0:
        return 0
    j = int(np.argmax(np.abs(stack)))
    val = float(stack[j])
    if val > 0.0:
        return 1
    if val < 0.0:
        return -1
    return 0


def wavelet_shape_score(
    stack: np.ndarray,
    dt: float,
    f0: float,
) -> float:
    """Absolute Pearson correlation of a stacked window vs a Ricker(f0).

    Returns a score in ``[0, 1]``: ``1.0`` for a stacked window
    that is exactly a Ricker at the prior centre frequency
    ``f0`` (modulo amplitude and time shift), ``0.0`` for an
    uncorrelated waveform. The Ricker template is centred on the
    location of ``stack``'s largest-absolute sample so the
    correlation is invariant to sub-window jitter of the picked
    arrival time.

    The score is **polarity-blind** -- it returns the absolute
    correlation -- so it can be combined orthogonally with
    :func:`onset_polarity` for a separate sign check.
    """
    n = stack.size
    if n < 2:
        return 0.0
    j_peak = int(np.argmax(np.abs(stack)))
    t = (np.arange(n) - j_peak) * dt
    a = (np.pi * f0 * t) ** 2
    template = (1.0 - 2.0 * a) * np.exp(-a)

    s = stack - stack.mean()
    tmpl = template - template.mean()
    s_norm = float(np.sqrt(np.sum(s * s)))
    t_norm = float(np.sqrt(np.sum(tmpl * tmpl)))
    if s_norm == 0.0 or t_norm == 0.0:
        return 0.0
    return float(abs(np.sum(s * tmpl)) / (s_norm * t_norm))


def _filter_one_depth(
    picks: dict[str, ModePick],
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    priors: dict[str, dict[str, float]],
    window_length: float,
    analysis_factor: float,
) -> dict[str, ModePick]:
    """Apply per-mode polarity / shape gates to one depth's picks."""
    out: dict[str, ModePick] = {}
    for name, pick in picks.items():
        prior = priors.get(name, {})
        polarity_expected = int(prior.get("polarity", 0))
        shape_min = float(prior.get("shape_match_min", 0.0))
        if polarity_expected == 0 and shape_min <= 0.0:
            # Neither gate enabled for this mode -- keep the pick.
            out[name] = pick
            continue

        stack = _align_and_stack(
            data,
            dt,
            offsets,
            pick.slowness,
            pick.time,
            window_length,
            analysis_factor=analysis_factor,
        )

        if polarity_expected != 0:
            actual = onset_polarity(stack)
            if actual != polarity_expected:
                continue  # polarity mismatch -- drop pick

        if shape_min > 0.0:
            f0 = prior.get("f0")
            if f0 is None:
                raise ValueError(
                    f"prior for {name!r} sets shape_match_min={shape_min} "
                    f"but no `f0` (Hz) -- the wavelet-shape gate needs the "
                    f"per-mode centre frequency"
                )
            score = wavelet_shape_score(stack, dt, float(f0))
            if score < shape_min:
                continue  # shape mismatch -- drop pick

        out[name] = pick
    return out


def filter_picks_by_shape(
    picks: dict[str, ModePick],
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    *,
    priors: dict[str, dict[str, float]] | None = None,
    window_length: float = 4.0e-4,
    analysis_factor: float = 2.0,
) -> dict[str, ModePick]:
    """
    Drop picks whose stacked waveform fails the polarity / shape rules.

    Implements two of the book's expert-rule layers (Mari et al.
    1994, Part 1) as a post-pick filter, applied to the dict of
    per-mode picks returned by :func:`pick_modes`. Picks whose mode
    in ``priors`` declares neither rule are passed through unchanged.

    Two opt-in rules per mode (read from each mode's prior dict):

    * ``polarity`` (``+1`` / ``-1`` / ``0``): expected sign of the
      stacked window's largest-absolute sample. ``0`` (the default)
      disables the gate.
    * ``shape_match_min`` (float in ``[0, 1]``): minimum absolute
      Pearson correlation between the stacked window and a Ricker
      template at the prior's ``f0`` (Hz). ``0.0`` disables the
      gate. When enabled, the prior **must** also carry an ``f0``
      key.

    Parameters
    ----------
    picks : dict from str to ModePick
        Per-mode picks at one depth, e.g. the output of
        :func:`pick_modes`.
    data : ndarray, shape (n_rec, n_samples)
        The same gather the picks were derived from. Used to
        re-stack the moveout-aligned window at each pick's
        ``(slowness, time)``.
    dt : float
        Sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m).
    priors : dict, optional
        Per-mode prior windows, with optional ``polarity``,
        ``shape_match_min`` and ``f0`` keys driving the gates.
        Defaults to :data:`DEFAULT_PRIORS`, which declares neither
        gate so the default-priors call is a no-op pass-through.
    window_length : float, default 4e-4
        STC time-window length (s) used to produce the picks. The
        polarity / shape gates analyse a window centred on the STC
        window's midpoint -- ``pick.time + window_length / 2`` --
        because ``pick.time`` is a window-start time and a window
        started there can capture a sidelobe rather than the main
        wavelet lobe.
    analysis_factor : float, default 2.0
        Width of the analysis window as a multiple of
        ``window_length``. The default ``2.0`` gives the polarity /
        shape gates one STC-window of context on each side of the
        STC window's centre, which is enough to bracket a Ricker's
        main lobe plus its near sidelobes for the typical sonic
        prior frequencies (3-15 kHz).

    Returns
    -------
    dict from str to ModePick
        A new dict containing only the picks that passed every
        enabled gate. The original ``picks`` dict is not mutated.
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    return _filter_one_depth(
        picks, data, dt, offsets, priors, window_length, analysis_factor
    )


def filter_track_by_shape(
    track_picks: Sequence[DepthPicks],
    datas: Sequence[np.ndarray],
    dt: float,
    offsets: np.ndarray,
    *,
    priors: dict[str, dict[str, float]] | None = None,
    window_length: float = 4.0e-4,
    analysis_factor: float = 2.0,
) -> list[DepthPicks]:
    """Apply :func:`filter_picks_by_shape` per-depth across a track.

    The multi-depth analogue of :func:`filter_picks_by_shape`: the
    same polarity / shape gates run once per depth against the
    matching per-depth gather. ``track_picks`` and ``datas`` must
    have the same length.

    Parameters
    ----------
    track_picks : sequence of DepthPicks
        Output of :func:`track_modes`, :func:`viterbi_pick`, or
        :func:`viterbi_pick_joint`.
    datas : sequence of ndarray, shape (n_rec, n_samples)
        One gather per depth, in the same order as ``track_picks``.
    dt, offsets : as in :func:`filter_picks_by_shape`.
    priors, window_length : as in :func:`filter_picks_by_shape`.

    Returns
    -------
    list of DepthPicks
        Filtered track. ``DepthPicks`` instances are new; the
        ``picks`` dicts are filtered copies of the originals.
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    if len(track_picks) != len(datas):
        raise ValueError(
            f"track_picks and datas must have the same length; got "
            f"{len(track_picks)} and {len(datas)}"
        )
    out: list[DepthPicks] = []
    for dp, data in zip(track_picks, datas):
        filt = _filter_one_depth(
            dp.picks, data, dt, offsets, priors, window_length, analysis_factor
        )
        out.append(DepthPicks(depth=dp.depth, picks=filt))
    return out
