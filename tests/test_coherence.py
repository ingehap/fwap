"""STC + semblance tests."""

from __future__ import annotations

import numpy as np

from fwap._common import US_PER_FT
from fwap.coherence import find_peaks, semblance, stc
from fwap.synthetic import (
    ArrayGeometry,
    monopole_formation_modes,
    synthesize_gather,
)


def test_semblance_identical_traces_is_one():
    """Perfectly coherent traces have semblance 1."""
    tr = np.sin(np.linspace(0, 4 * np.pi, 128))
    window = np.tile(tr, (8, 1))
    assert semblance(window) == 1.0


def test_semblance_zero_window_is_nan():
    """All-zero input returns NaN rather than dividing by zero."""
    out = semblance(np.zeros((4, 32)))
    assert np.isnan(out)


def test_stc_peak_at_formation_p_slowness():
    """STC coherence peak on a noiseless P/S/St gather sits at 1/Vp."""
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=2048)
    data = synthesize_gather(geom, monopole_formation_modes(Vp, Vs, Vst),
                             noise=0.01, seed=0)
    res = stc(data, dt=geom.dt, offsets=geom.offsets,
              slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
              n_slowness=181, window_length=4.0e-4, time_step=2)
    peaks = find_peaks(res, threshold=0.5)
    # At least one peak must be within 5 us/ft of each truth slowness.
    for v in (Vp, Vs, Vst):
        s_true = 1.0 / v
        close = np.min(np.abs(peaks[:, 0] - s_true)) / US_PER_FT
        assert close < 5.0, f"no peak near 1/{v:.0f} (closest {close:.1f} us/ft)"


def test_stc_amplitude_field_shape_and_finiteness():
    """STC populates ``amplitude`` with the same shape as ``coherence``."""
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=2048)
    data = synthesize_gather(geom, monopole_formation_modes(Vp, Vs, Vst),
                             noise=0.01, seed=0)
    res = stc(data, dt=geom.dt, offsets=geom.offsets,
              slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
              n_slowness=181, window_length=4.0e-4, time_step=2)
    assert res.amplitude is not None
    assert res.amplitude.shape == res.coherence.shape
    # NaN bins must coincide between coherence and amplitude (both
    # trip the same min_energy_fraction floor).
    assert np.array_equal(np.isnan(res.amplitude), np.isnan(res.coherence))
    finite = ~np.isnan(res.amplitude)
    assert np.all(res.amplitude[finite] >= 0.0)


def test_stc_amplitude_recovers_planted_amplitude_for_unit_sine():
    """A unit-amplitude sine on every trace should have amplitude
    ~1/sqrt(2) (RMS of a unit sine) at the right slowness."""
    n_rec, n_samp, dt = 8, 2048, 1.0e-5
    dx = 0.1524
    offsets = np.arange(n_rec) * dx
    p0 = 1.5e-4
    f0 = 5_000.0
    t = np.arange(n_samp) * dt
    A = 1.0
    data = np.zeros((n_rec, n_samp))
    for i in range(n_rec):
        data[i] = A * np.sin(2.0 * np.pi * f0 * (t - p0 * offsets[i]))
    res = stc(data, dt=dt, offsets=offsets,
              slowness_range=(0.5e-4, 3.0e-4), n_slowness=181,
              window_length=4.0e-4, time_step=4)
    assert res.amplitude is not None
    # Find the (slowness, time) cell with the highest coherence.
    rho = np.nan_to_num(res.coherence)
    i_max, j_max = np.unravel_index(np.argmax(rho), rho.shape)
    # At the right slowness, semblance should be close to 1 and the
    # per-trace stack RMS should be close to 1/sqrt(2) for a unit sine.
    assert rho[i_max, j_max] > 0.99
    assert abs(res.amplitude[i_max, j_max] - 1.0 / np.sqrt(2.0)) < 0.05


def test_find_peaks_returns_amplitude_when_present():
    """find_peaks returns 4-column rows (slow, time, coh, amp) when the
    STC result carries amplitude, and 3-column rows when it doesn't."""
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=2048)
    data = synthesize_gather(geom, monopole_formation_modes(),
                             noise=0.01, seed=0)
    res = stc(data, dt=geom.dt, offsets=geom.offsets,
              slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
              n_slowness=121, window_length=4.0e-4, time_step=2)
    peaks_with_amp = find_peaks(res, threshold=0.5)
    assert peaks_with_amp.shape[1] == 4
    assert np.all(peaks_with_amp[:, 3] >= 0.0)

    # Strip amplitude to simulate a legacy / dispersion-style result
    # and confirm the 3-column path still works.
    res.amplitude = None
    peaks_no_amp = find_peaks(res, threshold=0.5)
    assert peaks_no_amp.shape[1] == 3
