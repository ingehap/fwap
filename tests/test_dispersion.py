"""Dispersion estimator tests."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fwap.dispersion import (
    DispersionCurve,
    bandpass,
    dispersive_pseudo_rayleigh_stc,
    phase_slowness_from_f_k,
    phase_slowness_matrix_pencil,
    shear_slowness_from_dispersion,
)
from fwap.synthetic import (
    ArrayGeometry,
    Mode,
    dipole_flexural_dispersion,
    pseudo_rayleigh_dispersion,
    synthesize_gather,
)


def _flex_gather(Vs=2500.0, seed=7):
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=2.0e-5, n_samples=2048)
    disp = dipole_flexural_dispersion(vs=Vs, a_borehole=0.1)
    mode = Mode(name="Flex", slowness=1.0 / Vs, f0=4000.0,
                amplitude=1.0, dispersion=disp)
    data = synthesize_gather(geom, [mode], noise=0.01, seed=seed)
    return geom, data


def test_bandpass_preserves_in_band_and_attenuates_out_of_band():
    """Butterworth band-pass attenuates a sinusoid outside the pass band."""
    dt = 5.0e-6
    n = 4096
    t = np.arange(n) * dt
    sig_in  = np.sin(2 * np.pi * 3000.0 * t)     # in band
    sig_out = np.sin(2 * np.pi * 30000.0 * t)    # well above pass
    data = np.stack([sig_in, sig_out])
    filt = bandpass(data, dt, f_lo=1000.0, f_hi=5000.0, order=4)
    rms_in  = np.sqrt(np.mean(filt[0] ** 2))
    rms_out = np.sqrt(np.mean(filt[1] ** 2))
    assert rms_in  > 0.5
    assert rms_out < 0.05


def test_phase_slowness_matrix_pencil_recovers_constant_slowness():
    """A non-dispersive mode gives a flat slowness = 1/Vp in the estimator.

    The matrix-pencil estimator unwraps phase implicitly, but it is
    still limited by the spatial-Nyquist frequency
    ``f_alias = 1 / (2 * s * dx)`` above which the trace-to-trace phase
    increment exceeds pi. We pick Vp, dx, and the fit band so that
    f_alias sits well above the fit band upper edge.
    """
    Vp = 5500.0
    dr = 0.1524
    s_true = 1.0 / Vp
    f_alias = 1.0 / (2.0 * s_true * dr)   # ~ 18 kHz here
    assert f_alias > 10_000.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=dr,
                         dt=1.0e-5, n_samples=2048)
    mode = Mode(name="P", slowness=s_true, f0=6000.0, amplitude=1.0)
    data = synthesize_gather(geom, [mode], noise=0.005, seed=2)
    curve = phase_slowness_matrix_pencil(
        data, dt=geom.dt, offsets=geom.offsets,
        f_range=(2500.0, 10000.0))
    mask = curve.quality > 0.3
    assert mask.any()
    s_mean = np.average(curve.slowness[mask], weights=curve.quality[mask])
    assert abs(s_mean - s_true) / s_true < 0.05


def test_phase_slowness_from_f_k_methods_agree():
    """Frequency-unwrap and spatial-unwrap give consistent slownesses."""
    geom, data = _flex_gather()
    c_fu = phase_slowness_from_f_k(
        data, dt=geom.dt, offsets=geom.offsets,
        f_range=(500.0, 4000.0), method="frequency_unwrap")
    c_su = phase_slowness_from_f_k(
        data, dt=geom.dt, offsets=geom.offsets,
        f_range=(500.0, 4000.0), method="spatial_unwrap")
    # Quality-weighted means within 5% of each other.
    mask = (c_fu.quality > 0.3) & (c_su.quality > 0.3)
    if mask.any():
        mu_fu = np.average(c_fu.slowness[mask], weights=c_fu.quality[mask])
        mu_su = np.average(c_su.slowness[mask], weights=c_su.quality[mask])
        assert abs(mu_fu - mu_su) / abs(mu_fu) < 0.05


def test_shear_slowness_fallback_warns(caplog):
    """Fallback when no points pass quality_threshold emits a warning."""
    # Pathological curve: every quality value is below the threshold.
    curve = DispersionCurve(
        freq=np.array([1000.0, 2000.0, 3000.0]),
        slowness=np.array([4.0e-4, 4.1e-4, 4.2e-4]),
        quality=np.array([0.1, 0.1, 0.1]),
    )
    with caplog.at_level(logging.WARNING, logger="fwap"):
        s = shear_slowness_from_dispersion(
            curve, f_lo=500.0, f_hi=4000.0, quality_threshold=0.8)
    assert np.isfinite(s)
    assert any("quality_threshold" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------
# dispersive_pseudo_rayleigh_stc
# ---------------------------------------------------------------------


def _pr_gather(Vs=2500.0, v_fluid=1500.0, a_borehole=0.1, f0=8000.0,
               seed=11):
    """Single-mode pseudo-Rayleigh gather for the dispersive STC tests."""
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=2048)
    disp = pseudo_rayleigh_dispersion(vs=Vs, v_fluid=v_fluid,
                                      a_borehole=a_borehole)
    mode = Mode(name="PR", slowness=1.0 / Vs, f0=f0,
                amplitude=1.0, intercept=0.0, dispersion=disp)
    data = synthesize_gather(geom, [mode], noise=0.02, seed=seed)
    return geom, data


def test_dispersive_pr_stc_recovers_formation_shear_slowness():
    """Peak coherence on the dispersive STC sits at the planted 1/Vs."""
    Vs = 2500.0
    geom, data = _pr_gather(Vs=Vs)
    res = dispersive_pseudo_rayleigh_stc(
        data, dt=geom.dt, offsets=geom.offsets,
        v_fluid=1500.0, a_borehole=0.1,
        shear_slowness_range=(200e-6, 500e-6),
        n_slowness=61, f_range=(4000.0, 12000.0),
    )
    coh = np.nan_to_num(res.coherence, nan=0.0)
    s_peak, _ = np.unravel_index(int(np.argmax(coh)), coh.shape)
    s_recovered = res.slowness[s_peak]
    s_truth = 1.0 / Vs
    # 5% relative tolerance is comfortable for an 8-receiver gather
    # at the default noise level.
    assert abs(s_recovered - s_truth) / s_truth < 0.05


def test_dispersive_pr_stc_outperforms_plain_stc_on_dispersive_arrival():
    """Plain STC biases slowness high; dispersion-corrected STC does not."""
    from fwap.coherence import stc as plain_stc

    Vs = 2500.0
    geom, data = _pr_gather(Vs=Vs)
    res_disp = dispersive_pseudo_rayleigh_stc(
        data, dt=geom.dt, offsets=geom.offsets,
        v_fluid=1500.0, a_borehole=0.1,
        shear_slowness_range=(200e-6, 500e-6),
        n_slowness=61, f_range=(4000.0, 12000.0),
    )
    res_plain = plain_stc(
        data, dt=geom.dt, offsets=geom.offsets,
        slowness_range=(200e-6, 500e-6),
        n_slowness=61, window_length=1.0e-3, time_step=4,
    )
    s_truth = 1.0 / Vs
    s_disp  = res_disp.slowness[
        int(np.argmax(np.nan_to_num(res_disp.coherence,  nan=0.0).max(axis=1)))]
    s_plain = res_plain.slowness[
        int(np.argmax(np.nan_to_num(res_plain.coherence, nan=0.0).max(axis=1)))]
    # Both estimators should produce something, but the dispersion-
    # corrected one should be at least as close to truth as plain STC.
    assert abs(s_disp - s_truth) <= abs(s_plain - s_truth) + 1.0e-6


def test_dispersive_pr_stc_returns_stcresult_with_correct_axes():
    """Output dataclass shape matches the requested grid."""
    geom, data = _pr_gather()
    res = dispersive_pseudo_rayleigh_stc(
        data, dt=geom.dt, offsets=geom.offsets,
        shear_slowness_range=(200e-6, 500e-6),
        n_slowness=33, f_range=(4000.0, 10000.0),
        time_step=4,
    )
    assert res.slowness.shape == (33,)
    assert res.coherence.shape[0] == 33
    assert res.coherence.shape[1] == res.time.size
    assert res.amplitude is not None
    assert res.amplitude.shape == res.coherence.shape
    assert res.window_length == pytest.approx(1.0e-3)


def test_dispersive_pr_stc_rejects_range_above_fluid_slowness():
    """A trial range that crosses 1/v_fluid is unphysical (slow formation)."""
    geom, data = _pr_gather()
    with pytest.raises(ValueError, match="v_fluid"):
        dispersive_pseudo_rayleigh_stc(
            data, dt=geom.dt, offsets=geom.offsets,
            v_fluid=1500.0,
            # 800 us/m corresponds to vs = 1250 m/s < v_fluid; pseudo-
            # Rayleigh does not exist here.
            shear_slowness_range=(200e-6, 800e-6),
            n_slowness=21,
        )


def test_dispersive_pr_stc_rejects_inverted_range():
    """min >= max is rejected with a clear ValueError."""
    geom, data = _pr_gather()
    with pytest.raises(ValueError, match="shear_slowness_range"):
        dispersive_pseudo_rayleigh_stc(
            data, dt=geom.dt, offsets=geom.offsets,
            shear_slowness_range=(500e-6, 200e-6),
            n_slowness=21,
        )


def test_dispersive_pr_stc_rejects_non_positive_lower_bound():
    """Zero or negative slowness is rejected."""
    geom, data = _pr_gather()
    with pytest.raises(ValueError, match="positive"):
        dispersive_pseudo_rayleigh_stc(
            data, dt=geom.dt, offsets=geom.offsets,
            shear_slowness_range=(0.0, 500e-6),
            n_slowness=21,
        )


# ---------------------------------------------------------------------
# classify_flexural_anisotropy (stress vs intrinsic vs isotropic)
# ---------------------------------------------------------------------


def _curve(freq, slowness, quality=None):
    """Helper: build a DispersionCurve with broadcast quality."""
    freq = np.asarray(freq, dtype=float)
    slowness = np.asarray(slowness, dtype=float)
    if quality is None:
        quality = np.ones_like(freq)
    else:
        quality = np.broadcast_to(quality, freq.shape).astype(float)
    return DispersionCurve(freq=freq, slowness=slowness, quality=quality)


def test_classify_flexural_isotropic():
    """Two identical curves -> isotropic classification."""
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 50)
    s = 4.0e-4 * np.ones_like(f)
    diag = classify_flexural_anisotropy(_curve(f, s), _curve(f, s.copy()))
    assert diag.classification == "isotropic"
    assert abs(diag.delta_low) < 5e-6
    assert abs(diag.delta_high) < 5e-6
    assert diag.crossover_frequency is None


def test_classify_flexural_intrinsic_constant_offset():
    """Slow curve constantly slower than fast -> intrinsic."""
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 50)
    s_fast = 4.0e-4 * np.ones_like(f)
    s_slow = s_fast + 1.5e-5  # 15 us/m, comfortably above min_anisotropy
    diag = classify_flexural_anisotropy(_curve(f, s_fast), _curve(f, s_slow))
    assert diag.classification == "intrinsic"
    assert diag.delta_low > 0 and diag.delta_high > 0
    assert diag.crossover_frequency is None


def test_classify_flexural_stress_induced_crossover():
    """Curves whose Δs flips sign across the band -> stress_induced."""
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 100)
    f_cross = 3000.0
    s_fast = 4.0e-4 * np.ones_like(f)
    # Slope chosen so |Δs| is well above 5 us/m default threshold in
    # both bands: at f=1000 (low band), Δs = -4e-5; at f=8000
    # (high band), Δs = +1e-4.
    s_slow = s_fast + 2.0e-8 * (f - f_cross)
    diag = classify_flexural_anisotropy(_curve(f, s_fast), _curve(f, s_slow))
    assert diag.classification == "stress_induced"
    assert diag.delta_low < 0
    assert diag.delta_high > 0
    assert diag.crossover_frequency is not None
    # The synthetic crossover sits at 3000 Hz; the first sign-change
    # interpolation should land within ~1 sample spacing of that
    # (~95 Hz on this 100-point grid).
    assert abs(diag.crossover_frequency - f_cross) < 200.0


def test_classify_flexural_ambiguous_when_one_band_quiet():
    """Anisotropic at low-f, flat at high-f -> ambiguous."""
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 50)
    s_fast = 4.0e-4 * np.ones_like(f)
    s_slow = s_fast.copy()
    # 20 us/m offset, but only in the low-f band (1-2.5 kHz).
    low_mask = (f >= 1000.0) & (f <= 2500.0)
    s_slow[low_mask] += 2.0e-5
    diag = classify_flexural_anisotropy(_curve(f, s_fast), _curve(f, s_slow))
    assert diag.classification == "ambiguous"
    # Low band carries the anisotropy; high band does not.
    assert abs(diag.delta_low) > 5e-6
    assert abs(diag.delta_high) < 5e-6


def test_classify_flexural_ambiguous_when_no_quality_samples():
    """All quality below threshold -> ambiguous with informative reason."""
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 50)
    s = 4.0e-4 * np.ones_like(f)
    diag = classify_flexural_anisotropy(
        _curve(f, s, quality=0.1),
        _curve(f, s.copy(), quality=0.1),
    )
    assert diag.classification == "ambiguous"
    assert any("no quality-passing" in r for r in diag.reasons)


def test_classify_flexural_quality_threshold_filters_out_noise():
    """Bins with low quality on one curve are excluded from the bands."""
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 50)
    s_fast = 4.0e-4 * np.ones_like(f)
    s_slow = s_fast + 1.5e-5
    # Inject a spurious negative jump only in the high-f band, but
    # mark those bins as low quality on curve_a.
    spurious_mask = (f >= 4000.0) & (f <= 8000.0)
    s_fast_noisy = s_fast.copy()
    s_fast_noisy[spurious_mask] = 4.0e-4 + 5.0e-5  # would force fast > slow
    q_a = np.ones_like(f)
    q_a[spurious_mask] = 0.1  # well below default threshold = 0.5
    diag = classify_flexural_anisotropy(
        _curve(f, s_fast_noisy, quality=q_a),
        _curve(f, s_slow),
    )
    # High-f band is now empty (excluded); should be ambiguous.
    assert diag.classification == "ambiguous"
    assert not np.isfinite(diag.delta_high)


def test_classify_flexural_rejects_mismatched_freq_axes():
    """Different freq axes are a caller error."""
    import pytest
    from fwap.dispersion import classify_flexural_anisotropy
    f1 = np.linspace(500.0, 10000.0, 50)
    f2 = np.linspace(500.0, 10000.0, 60)
    s1 = 4.0e-4 * np.ones_like(f1)
    s2 = 4.0e-4 * np.ones_like(f2)
    with pytest.raises(ValueError, match="freq axis"):
        classify_flexural_anisotropy(_curve(f1, s1), _curve(f2, s2))


def test_classify_flexural_rejects_overlapping_bands():
    """f_low_band and f_high_band must not overlap."""
    import pytest
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 50)
    s = 4.0e-4 * np.ones_like(f)
    with pytest.raises(ValueError, match="overlap"):
        classify_flexural_anisotropy(
            _curve(f, s), _curve(f, s.copy()),
            f_low_band=(1000.0, 5000.0), f_high_band=(4000.0, 8000.0),
        )


def test_classify_flexural_rejects_inverted_bands():
    """lo >= hi within a band is a caller error."""
    import pytest
    from fwap.dispersion import classify_flexural_anisotropy
    f = np.linspace(500.0, 10000.0, 50)
    s = 4.0e-4 * np.ones_like(f)
    with pytest.raises(ValueError, match="lo < hi"):
        classify_flexural_anisotropy(
            _curve(f, s), _curve(f, s.copy()),
            f_low_band=(2500.0, 1000.0), f_high_band=(4000.0, 8000.0),
        )
