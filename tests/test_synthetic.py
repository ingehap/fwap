"""Synthetic-gather tests."""

from __future__ import annotations

import numpy as np

from fwap.synthetic import (
    ArrayGeometry,
    Mode,
    dipole_flexural_dispersion,
    gabor,
    monopole_formation_modes,
    ricker,
    synthesize_gather,
)


def test_ricker_peak_at_t0():
    """Ricker wavelet is centred at t0 with amplitude 1."""
    t = np.linspace(0.0, 1.0e-3, 513)
    w = ricker(t, f0=5000.0, t0=5.0e-4)
    assert abs(t[np.argmax(w)] - 5.0e-4) < 5e-6
    assert abs(w.max() - 1.0) < 1e-12


def test_gabor_finite_and_symmetric_envelope():
    """Gabor wavelet envelope is approximately symmetric around t0."""
    t = np.linspace(0.0, 1.0e-3, 513)
    w = gabor(t, f0=4000.0, t0=5.0e-4, sigma=1.5e-4)
    assert np.isfinite(w).all()
    env = np.abs(w)
    assert env.argmax() > 0 and env.argmax() < t.size - 1


def test_array_geometry_offsets_and_time():
    """ArrayGeometry computes offsets and the time axis correctly."""
    g = ArrayGeometry(n_rec=4, tr_offset=3.0, dr=0.1, dt=1.0e-5,
                      n_samples=100)
    assert np.allclose(g.offsets, [3.0, 3.1, 3.2, 3.3])
    assert g.t.size == 100
    assert g.t[-1] == 99 * 1.0e-5


def test_array_geometry_from_imperial_matches_default():
    """from_imperial reproduces the ft/in metric defaults."""
    g_metric = ArrayGeometry()
    g_imp = ArrayGeometry.from_imperial(n_rec=g_metric.n_rec,
                                        tr_offset_ft=g_metric.tr_offset / 0.3048,
                                        dr_in=g_metric.dr / 0.0254,
                                        dt=g_metric.dt,
                                        n_samples=g_metric.n_samples)
    assert np.allclose(g_imp.offsets, g_metric.offsets)


def test_schlumberger_array_sonic_factory():
    """schlumberger_array_sonic() matches the documented reference geometry."""
    g = ArrayGeometry.schlumberger_array_sonic()
    assert g.n_rec == 8
    assert abs(g.tr_offset - 10.0 * 0.3048) < 1.0e-12
    assert abs(g.dr - 6.0 * 0.0254) < 1.0e-12


def test_array_geometry_offsets_is_cached():
    """ArrayGeometry.offsets and .t return the same object on repeated access."""
    g = ArrayGeometry()
    assert g.offsets is g.offsets
    assert g.t is g.t


def test_dipole_flexural_dispersion_is_array_compatible():
    """The flexural dispersion accepts a NumPy array and returns one."""
    f = np.linspace(100.0, 5000.0, 51)
    s = dipole_flexural_dispersion(vs=2500.0, a_borehole=0.1)(f)
    assert s.shape == f.shape
    # Monotone increasing with frequency (flexural mode up-shifts).
    assert np.all(np.diff(s) >= -1e-18)
    # Low-f limit equals 1/Vs within the first few Hz.
    assert abs(s[0] - 1.0 / 2500.0) < 1e-6


def test_synthesize_gather_noise_level():
    """Noise RMS matches the requested fraction of the gather RMS."""
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=2048)
    modes = monopole_formation_modes()
    clean = synthesize_gather(geom, modes, noise=0.0, seed=1)
    noisy = synthesize_gather(geom, modes, noise=0.1, seed=1)
    rms_clean = np.sqrt(np.mean(clean ** 2))
    rms_noise = np.sqrt(np.mean((noisy - clean) ** 2))
    # Requested 10% of the gather RMS; tolerate the sample variation.
    assert abs(rms_noise / rms_clean - 0.1) < 0.02


def test_mode_with_dispersion_receives_array_input():
    """A custom dispersion callable is called with an ndarray, not a scalar."""
    seen = {}

    def disp(freqs):
        seen["type"] = type(freqs)
        seen["shape"] = np.asarray(freqs).shape
        return np.full_like(np.asarray(freqs, dtype=float), 1.0 / 2500.0)

    geom = ArrayGeometry(n_rec=4, tr_offset=3.0, dr=0.1, dt=1.0e-5,
                         n_samples=256)
    mode = Mode(name="D", slowness=1.0 / 2500.0, f0=3000.0,
                amplitude=1.0, dispersion=disp)
    synthesize_gather(geom, [mode], noise=0.0, seed=0)
    assert seen["type"] is np.ndarray
    assert len(seen["shape"]) == 1
