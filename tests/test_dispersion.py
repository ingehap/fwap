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
