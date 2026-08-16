"""Attenuation (Q) tests."""

from __future__ import annotations

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Causality (Kramers-Kronig) and the Q estimators.
#
# `_attenuated_gather` above multiplies the spectrum by exp(-pi f t / Q) and
# leaves the phase alone. That waveform is **acausal**: constant-Q amplitude
# loss without the velocity dispersion that must accompany it violates the
# Kramers-Kronig relation between attenuation and dispersion, and the result
# has energy arriving before the geometric arrival time.
#
# It is not a bug in the estimators, both of which use |S(f)| only. But the
# accuracy those tests demonstrate is accuracy on a signal that cannot exist,
# so the physical case is worth covering separately -- the estimators window in
# time, and dispersion reshapes the waveform inside that window.
# ---------------------------------------------------------------------------

_KOLSKY_REFERENCE_HZ = 15_000.0


def _causal_attenuated_gather(Q_true=50.0, Vp=4000.0, f0=15_000.0, noise=0.02, seed=4):
    """As ``_attenuated_gather`` but Kramers-Kronig consistent.

    Adds the Kolsky-Futterman velocity dispersion that must accompany
    constant-Q attenuation, ``c(f) = c_0 [1 + ln(f/f_ref) / (pi Q)]``. High
    frequencies travel faster and arrive earlier, which is what removes the
    pre-arrival energy the amplitude-only model has.
    """
    geom = ArrayGeometry(n_rec=12, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    t = geom.t
    t0 = 2.0e-4
    n = geom.n_samples
    freqs = np.fft.rfftfreq(n, d=geom.dt)
    data = np.zeros((geom.n_rec, n))
    rng = np.random.default_rng(seed)
    epsilon = 1.0 / (np.pi * Q_true)
    for i, off in enumerate(geom.offsets):
        travel = off / Vp
        spectrum = np.fft.rfft(ricker(t, f0=f0, t0=t0 + travel))
        spectrum = spectrum * np.exp(-np.pi * freqs * travel / Q_true)
        phase = np.where(
            freqs > 0.0,
            2.0
            * np.pi
            * freqs
            * travel
            * epsilon
            * np.log(np.maximum(freqs, 1.0e-9) / _KOLSKY_REFERENCE_HZ),
            0.0,
        )
        data[i] = np.fft.irfft(spectrum * np.exp(1j * phase), n=n)
    rms = np.sqrt(np.mean(data**2)) + 1e-12
    data += rng.normal(scale=noise * rms, size=data.shape)
    return geom, data, t0, Vp, Q_true


def _pre_arrival_energy_fraction(trace, t, arrival, guard):
    before = t < (arrival - guard)
    return float(np.sum(trace[before] ** 2) / np.sum(trace**2))


def test_the_dispersion_free_gather_is_acausal_and_the_kolsky_one_is_not():
    """Establishes that the two synthetics differ in the way claimed.

    Measured rather than asserted from the algebra, because the sign of the
    dispersion phase is easy to get backwards -- and getting it backwards
    makes the signal *more* acausal, which is how the error was caught. The
    causal gather must show far less pre-arrival energy than the
    amplitude-only one.
    """
    geom, plain, t0, vp, _ = _attenuated_gather(noise=0.0)
    _, causal, _, _, _ = _causal_attenuated_gather(noise=0.0)

    offset = geom.offsets[-1]
    arrival = t0 + offset / vp
    guard = 3.0 / 15_000.0

    plain_leak = _pre_arrival_energy_fraction(plain[-1], geom.t, arrival, guard)
    causal_leak = _pre_arrival_energy_fraction(causal[-1], geom.t, arrival, guard)

    assert causal_leak < 0.05 * plain_leak, (plain_leak, causal_leak)


def test_q_estimators_survive_a_causal_waveform():
    """Both estimators still recover Q on a physically realisable signal.

    They read |S(f)| only, so dispersion reaches them solely through the time
    window -- but that route is real, and it moves the answer by roughly a
    third. Measured on the causal gather: the centroid estimate lands nearer
    the planted value than on the acausal one, and the spectral-ratio estimate
    improves from about 2.3x high to about 1.6x high. So the existing tests
    understate both estimators rather than flattering them.
    """
    geom, data, t0, vp, q_true = _causal_attenuated_gather()

    centroid = centroid_frequency_shift_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / vp,
        window_length=4.0e-4,
        f_range=(5_000.0, 30_000.0),
        pick_intercept=t0,
    )
    ratio = spectral_ratio_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / vp,
        window_length=4.0e-4,
        f_range=(5_000.0, 30_000.0),
        pick_intercept=t0,
    )

    assert 0.5 * q_true < centroid.q < 2.0 * q_true, centroid.q
    assert np.isfinite(ratio.q) and ratio.q > 0.0
    assert ratio.q < 3.0 * q_true, ratio.q


def test_causality_changes_the_recovered_q_materially():
    """The two synthetics are not interchangeable, which is why both exist.

    If dispersion made no difference the causal gather would be redundant.
    It moves the recovered Q by tens of percent through the windowing alone.
    """
    kwargs = dict(window_length=4.0e-4, f_range=(5_000.0, 30_000.0))

    geom_a, acausal, t0, vp, _ = _attenuated_gather()
    geom_c, causal, _, _, _ = _causal_attenuated_gather()

    def estimate(data, geom):
        return centroid_frequency_shift_Q(
            data,
            dt=geom.dt,
            offsets=geom.offsets,
            slowness=1.0 / vp,
            pick_intercept=t0,
            **kwargs,
        ).q

    q_acausal = estimate(acausal, geom_a)
    q_causal = estimate(causal, geom_c)
    assert abs(q_causal / q_acausal - 1.0) > 0.10, (q_acausal, q_causal)


# ----------------------------------------------------------------------
# W1 group A: what the centroid estimator does on its own ideal input
#
# `test_centroid_Q_recovers_order_of_magnitude` allows a factor of two
# and attributes the error to "the non-Gaussianity of the Ricker source
# spectrum and the finite window length". Both halves of that are
# testable by removing them: plant a source whose spectrum *is*
# Gaussian, and use a window ten times the pulse. The error does not go
# away, so the stated explanation is not the cause.
# ----------------------------------------------------------------------


def _gaussian_source_gather(
    q_true: float,
    vp: float = 4000.0,
    fc0: float = 15000.0,
    sigma_f: float = 4000.0,
    t0: float = 2.0e-3,
    n_samples: int = 8192,
):
    """A gather satisfying the centroid method's assumptions exactly.

    The source spectrum is Gaussian -- not Ricker -- and the only thing
    applied along the array is ``exp(-pi f t / Q)``, which is the
    constant-``Q`` model the closed form is derived from. For this
    input ``fc(t) = fc0 - pi sigma_f^2 t / Q`` holds analytically, so a
    faithful implementation should return ``q_true``.
    """
    geom = ArrayGeometry(
        n_rec=12, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=n_samples
    )
    freqs = np.fft.rfftfreq(n_samples, d=geom.dt)
    envelope = np.exp(-((freqs - fc0) ** 2) / (2.0 * sigma_f**2))
    data = np.zeros((geom.n_rec, n_samples))
    for i, offset in enumerate(geom.offsets):
        travel = offset / vp
        spec = (
            envelope
            * np.exp(-2j * np.pi * freqs * (t0 + travel))
            * np.exp(-np.pi * freqs * travel / q_true)
        )
        data[i] = np.fft.irfft(spec, n=n_samples)
    return geom, data, t0, vp, sigma_f


def _centroid_on_gaussian(q_true: float, window_length: float = 4.0e-3):
    geom, data, t0, vp, _ = _gaussian_source_gather(q_true)
    return centroid_frequency_shift_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / vp,
        window_length=window_length,
        f_range=(1000.0, 40000.0),
        pick_intercept=t0,
    )


def test_the_centroid_bias_is_not_the_ricker_and_is_not_the_window():
    """Remove both stated causes; the error stays.

    With a genuinely Gaussian source and a 4 ms window -- ten times the
    0.4 ms the other test uses, and long past where the answer stops
    moving -- the recovered ``Q`` is still wrong, by an amount that
    depends on ``Q`` and changes sign:

        Q =  25 -> +8.6 %
        Q =  50 -> +8.2 %
        Q = 100 -> +5.7 %
        Q = 200 -> +0.3 %
        Q = 400 -> -9.1 %

    A sign change rules out both stated explanations. Non-Gaussianity
    and truncation are one-directional biases on a fixed source; they
    cannot flip with ``Q`` when the source is held constant.

    Asserted as measured. These are the numbers to re-baseline against
    if the estimator is ever corrected.
    """
    measured = {25.0: 0.086, 50.0: 0.082, 100.0: 0.057, 200.0: 0.003, 400.0: -0.091}
    for q_true, expected in measured.items():
        result = _centroid_on_gaussian(q_true)
        bias = (result.q - q_true) / q_true
        assert bias == pytest.approx(expected, abs=0.004), (q_true, bias)

    # And it is converged in window length, so it is not truncation.
    short = _centroid_on_gaussian(50.0, window_length=4.0e-3)
    long = _centroid_on_gaussian(50.0, window_length=6.0e-2)
    assert (long.q - short.q) / short.q == pytest.approx(0.0, abs=0.001)


def test_the_variance_estimate_carries_most_of_the_centroid_bias():
    """Where the error actually enters, since ``Q`` is proportional to it.

    ``Q = -pi sigma_f^2 / slope``, so an error in ``sigma_f^2`` passes
    straight through. Against a planted source variance of
    ``4000^2 = 1.6e7``, the estimator reports **1.11 to 1.13 times**
    that, near enough independent of ``Q`` -- a systematic +12 %.

    Substituting the true variance into the same fitted slope does not
    fix things, it moves the error to the other side and makes it grow
    with ``Q``: -2.5 % at 25, -20 % at 400. So there are two systematic
    effects here of opposite sign, and the near-zero total near
    ``Q ~ 200`` is them cancelling rather than either being small.

    That is worth stating plainly because a single number quoted from a
    validation at one ``Q`` would look excellent and generalise to
    nothing.
    """
    planted_variance = 4000.0**2

    ratios, substituted = {}, {}
    for q_true in (25.0, 50.0, 100.0, 200.0, 400.0):
        result = _centroid_on_gaussian(q_true)
        variance = float(result.diagnostic["sigma_f2"])
        slope = float(result.diagnostic["slope"])
        ratios[q_true] = variance / planted_variance
        substituted[q_true] = (-np.pi * planted_variance / slope - q_true) / q_true

    assert all(1.10 < r < 1.14 for r in ratios.values()), ratios
    assert np.ptp(list(ratios.values())) < 0.02, ratios

    # The residual, with the variance error removed, is one-signed and
    # grows -- the opposite sense to the total bias above.
    assert substituted[25.0] == pytest.approx(-0.025, abs=0.01)
    assert substituted[400.0] == pytest.approx(-0.195, abs=0.02)
    assert substituted[400.0] < substituted[25.0]
