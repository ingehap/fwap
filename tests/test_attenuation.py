"""Attenuation (Q) tests."""

from __future__ import annotations

import numpy as np

from fwap.attenuation import centroid_frequency_shift_Q, spectral_ratio_Q
from fwap.synthetic import ArrayGeometry, ricker


def _attenuated_gather(Q_true=50.0, Vp=4000.0, f0=15_000.0, noise=0.02, seed=4):
    geom = ArrayGeometry(n_rec=12, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    t = geom.t
    t0 = 2.0e-4
    n = geom.n_samples
    freqs = np.fft.rfftfreq(n, d=geom.dt)
    data = np.zeros((geom.n_rec, n))
    rng = np.random.default_rng(seed)
    for i, off in enumerate(geom.offsets):
        tt = t0 + off / Vp
        src = ricker(t, f0=f0, t0=tt)
        S = np.fft.rfft(src)
        S = S * np.exp(-np.pi * freqs * (off / Vp) / Q_true)
        data[i] = np.fft.irfft(S, n=n)
    rms = np.sqrt(np.mean(data**2)) + 1e-12
    data += rng.normal(scale=noise * rms, size=data.shape)
    return geom, data, t0, Vp, Q_true


def test_centroid_Q_recovers_order_of_magnitude():
    """Centroid-shift Q is within roughly a factor-2 of the planted value.

    The centroid-shift estimator is biased by the non-Gaussianity of
    the Ricker source spectrum and by the finite window length; for
    synthetic ringing-free data we expect the estimate to track truth
    to within a factor of ~2, which is the accuracy claimed in the
    literature for this SNR regime (Quan & Harris, 1997, Section 4).
    """
    geom, data, t0, Vp, Q_true = _attenuated_gather()
    res = centroid_frequency_shift_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / Vp,
        window_length=4.0e-4,
        f_range=(5_000.0, 30_000.0),
        pick_intercept=t0,
    )
    assert res.method == "centroid"
    assert 0.5 * Q_true < res.q < 2.0 * Q_true


def test_spectral_ratio_Q_positive_and_finite():
    """Spectral-ratio Q returns a positive finite value on attenuated data."""
    geom, data, t0, Vp, Q_true = _attenuated_gather()
    res = spectral_ratio_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / Vp,
        window_length=4.0e-4,
        f_range=(5_000.0, 25_000.0),
        pick_intercept=t0,
    )
    assert res.method == "spectral_ratio"
    assert np.isfinite(res.q) and res.q > 0.0
    # The estimator is noisier than the centroid method for a Ricker
    # source; only require the right sign and order of magnitude.
    assert 0.25 * Q_true < res.q < 4.0 * Q_true


def test_no_attenuation_gives_very_large_q():
    """With zero attenuation the slope -> 0 and Q -> infinity."""
    geom, data, t0, Vp, _ = _attenuated_gather(Q_true=1.0e12, noise=0.0, seed=10)
    res = centroid_frequency_shift_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / Vp,
        window_length=4.0e-4,
        f_range=(5_000.0, 30_000.0),
        pick_intercept=t0,
    )
    # Expect Q >> 1000 (effectively infinity) in the noise-free limit.
    assert res.q > 1.0e3 or np.isinf(res.q)
