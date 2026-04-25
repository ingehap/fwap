"""Tests for the sign-convention helper in fwap._common.

``_phase_shift`` is the single source of truth for the moveout sign
used by STC, dispersive STC, wave separation, and azimuthal detilt.
A silent flip here would corrupt every one of those results, so this
test pins the convention explicitly.
"""

from __future__ import annotations

import numpy as np

from fwap._common import _phase_shift


def test_phase_shift_advances_delta_arrival():
    """A positive tau advances the spike by tau seconds (earlier in time).

    Under NumPy's forward-FFT convention, multiplying the spectrum by
    ``exp(+2*pi*i*f*tau)`` shifts a delta at t0 to a delta at t0 - tau.
    This test builds a delta on two traces, applies a positive shift,
    and verifies that the peak moves to the expected earlier sample.
    """
    n = 256
    dt = 1.0e-5
    t0 = 100 * dt
    tau = 30 * dt
    data = np.zeros((2, n))
    data[:, int(round(t0 / dt))] = 1.0
    spec = np.fft.rfft(data, axis=1)
    f = np.fft.rfftfreq(n, d=dt)
    shifted = np.fft.irfft(
        _phase_shift(spec, f, np.array([tau, tau])), n=n, axis=1)
    peak = int(np.argmax(shifted[0]))
    expected = int(round((t0 - tau) / dt))
    assert peak == expected


def test_phase_shift_round_trip_is_identity():
    """phase_shift(+tau) then phase_shift(-tau) returns the input."""
    rng = np.random.default_rng(0)
    n = 512
    dt = 1.0e-5
    x = rng.standard_normal((4, n))
    spec = np.fft.rfft(x, axis=1)
    f = np.fft.rfftfreq(n, d=dt)
    tau = np.array([0.0, 1.0e-4, -5.0e-5, 3.7e-5])
    fwd = _phase_shift(spec, f, tau)
    back = _phase_shift(fwd, f, -tau)
    x_round = np.fft.irfft(back, n=n, axis=1)
    # Nyquist-bin phase is forced real, so tolerate the usual floor.
    data_rms = np.sqrt(np.mean(x ** 2))
    err_rms = np.sqrt(np.mean((x_round - x) ** 2))
    assert err_rms / data_rms < 1.0e-10


def test_plotting_legacy_aliases_still_importable():
    """fwap._plotting._wiggle / _savefig keep working for backwards compat."""
    from fwap import _plotting, plotting
    assert _plotting._wiggle is plotting.wiggle_plot
    assert _plotting._savefig is plotting.save_figure


def test_shared_logger_is_single_instance():
    """Every module that logs uses the same ``fwap`` logger."""
    import logging

    import fwap
    from fwap import _common, cli, demos, dispersion, plotting
    # All imports return the same logger object.
    shared = _common.logger
    assert fwap.logger is shared
    assert plotting.logger is shared
    assert cli.logger is shared
    assert demos.logger is shared
    assert dispersion.logger is shared
    # And it is the canonical "fwap" logger.
    assert shared is logging.getLogger("fwap")
