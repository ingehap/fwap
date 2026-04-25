"""LWD phenomenological-layer tests.

Verifies the collar-mode synthesis + slowness-band notch + picker
chain on a synthetic LWD gather.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwap._common import US_PER_FT
from fwap.coherence import find_peaks, stc
from fwap.lwd import (
    DEFAULT_COLLAR_FREQUENCY_HZ,
    DEFAULT_COLLAR_SLOWNESS_S_PER_M,
    lwd_collar_mode,
    notch_slowness_band,
    synthesize_lwd_gather,
)
from fwap.picker import DEFAULT_PRIORS, pick_modes
from fwap.synthetic import (
    ArrayGeometry,
    Mode,
    monopole_formation_modes,
    synthesize_gather,
)


def _geom():
    return ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=2048)


# ---------------------------------------------------------------------
# lwd_collar_mode
# ---------------------------------------------------------------------


def test_lwd_collar_mode_fields_match_documented_defaults():
    """Default factory returns a Mode with the documented properties."""
    m = lwd_collar_mode()
    assert m.name == "Collar"
    assert m.slowness == pytest.approx(DEFAULT_COLLAR_SLOWNESS_S_PER_M)
    assert m.f0 == pytest.approx(DEFAULT_COLLAR_FREQUENCY_HZ)
    assert m.wavelet == "gabor"
    # Default collar slowness sits in the published 80-130 us/ft
    # band for the steel-collar arrival.
    slow_us_per_ft = m.slowness / US_PER_FT
    assert 80.0 <= slow_us_per_ft <= 130.0


def test_lwd_collar_mode_overrides_propagate():
    """Caller-supplied slowness / frequency / amplitude reach the Mode."""
    m = lwd_collar_mode(apparent_slowness=2.5e-4, f0=8000.0,
                        amplitude=2.0, sigma=2.0e-4, intercept=1.0e-3)
    assert m.slowness == pytest.approx(2.5e-4)
    assert m.f0 == pytest.approx(8000.0)
    assert m.amplitude == pytest.approx(2.0)
    assert m.sigma == pytest.approx(2.0e-4)
    assert m.intercept == pytest.approx(1.0e-3)


# ---------------------------------------------------------------------
# synthesize_lwd_gather
# ---------------------------------------------------------------------


def test_synthesize_lwd_gather_appends_collar_to_modes():
    """LWD gather differs from the formation-only gather by a collar
    arrival -- the difference should peak at the collar slowness on
    an STC of the difference."""
    geom = _geom()
    formation = monopole_formation_modes()
    data_clean = synthesize_gather(geom, formation, noise=0.0, seed=0)
    data_lwd   = synthesize_lwd_gather(geom, formation, noise=0.0, seed=0,
                                       collar_amplitude=1.5)
    diff = data_lwd - data_clean
    surf = stc(diff, dt=geom.dt, offsets=geom.offsets,
               slowness_range=(50e-6, 360.0 * US_PER_FT),
               n_slowness=121, window_length=4.0e-4, time_step=2)
    coh = np.nan_to_num(surf.coherence, nan=0.0)
    # The peak coherence sits at the collar slowness.
    s_peak = surf.slowness[int(np.argmax(coh.max(axis=1)))]
    s_truth = DEFAULT_COLLAR_SLOWNESS_S_PER_M
    assert abs(s_peak - s_truth) / s_truth < 0.15


def test_synthesize_lwd_gather_seed_reproducibility():
    """Same seed -> identical gathers."""
    geom = _geom()
    formation = monopole_formation_modes()
    a = synthesize_lwd_gather(geom, formation, seed=42)
    b = synthesize_lwd_gather(geom, formation, seed=42)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------
# notch_slowness_band
# ---------------------------------------------------------------------


def test_notch_slowness_band_attenuates_in_band_amplitude():
    """A monochromatic in-band arrival is attenuated in amplitude;
    the out-of-band arrival is preserved (up to the slant-stack
    point-spread constant). Coherence (semblance) is normalised so
    it's resilient to amplitude changes -- amplitude is the right
    metric for notch effectiveness."""
    geom = _geom()
    # Two non-dispersive modes: one inside the notch, one outside.
    in_band = Mode("in_band",  slowness=3.0e-4, f0=10_000.0, amplitude=1.0)
    out_band = Mode("out_band", slowness=1.0e-4, f0=10_000.0, amplitude=1.0)
    data = synthesize_gather(geom, [in_band, out_band], noise=0.0, seed=0)
    notched = notch_slowness_band(
        data, dt=geom.dt, offsets=geom.offsets,
        slow_min=2.5e-4, slow_max=3.5e-4,
        n_slowness=121, taper_width=0.15,
    )
    # STC both records and compare per-cell amplitude in / out of the
    # notched band.
    surf_orig = stc(data, dt=geom.dt, offsets=geom.offsets,
                    slowness_range=(50e-6, 5.0e-4),
                    n_slowness=181, window_length=4.0e-4, time_step=2)
    surf_notched = stc(notched, dt=geom.dt, offsets=geom.offsets,
                       slowness_range=(50e-6, 5.0e-4),
                       n_slowness=181, window_length=4.0e-4, time_step=2)
    amp_orig    = np.nan_to_num(surf_orig.amplitude,    nan=0.0)
    amp_notched = np.nan_to_num(surf_notched.amplitude, nan=0.0)
    s_axis = surf_orig.slowness
    in_mask  = (s_axis >= 2.7e-4) & (s_axis <= 3.3e-4)
    out_mask = (s_axis >= 0.5e-4) & (s_axis <= 1.5e-4)
    amp_ratio_in  = amp_notched[in_mask].max()  / amp_orig[in_mask].max()
    amp_ratio_out = amp_notched[out_mask].max() / amp_orig[out_mask].max()
    # In-band amplitude attenuated to ~50% of original (the
    # tau-p-adjoint back-end is non-unitary; deeper rejection would
    # need an LSQR inverse, which does not commute with masking).
    # Out-of-band preserved within 10% of original via the subtract-
    # the-in-band route (out-of-grid signals pass through unchanged).
    assert amp_ratio_in < 0.6
    assert amp_ratio_out > 0.85


def test_notch_slowness_band_rejects_inverted_range():
    """slow_min >= slow_max is a caller error."""
    geom = _geom()
    data = synthesize_gather(geom, monopole_formation_modes(), noise=0.0, seed=0)
    with pytest.raises(ValueError, match="slow_min < slow_max"):
        notch_slowness_band(data, geom.dt, geom.offsets,
                            slow_min=3.0e-4, slow_max=2.0e-4)


def test_notch_slowness_band_rejects_non_positive_lower():
    """Zero or negative slow_min is unphysical."""
    geom = _geom()
    data = synthesize_gather(geom, monopole_formation_modes(), noise=0.0, seed=0)
    with pytest.raises(ValueError, match="slow_min < slow_max"):
        notch_slowness_band(data, geom.dt, geom.offsets,
                            slow_min=0.0, slow_max=3.0e-4)


# ---------------------------------------------------------------------
# End-to-end: pick P / S / Stoneley despite a collar arrival, after
# rejection. This is the whole point of the phenomenological layer.
# ---------------------------------------------------------------------


def test_lwd_collar_rejection_recovers_formation_picks():
    """Plain STC on a collar-contaminated gather peaks at the collar
    slowness; after notching out the collar slowness band, the formation
    P / S / Stoneley are the dominant peaks again and the picker
    recovers them within the usual 10 us/ft tolerance.

    The notch attenuates amplitude (the residual collar stays narrow-
    band-coherent at lower amplitude), so what matters is whether the
    notched gather still ranks the formation modes ahead of the
    residual collar in the picker's combined coherence + earliness
    score. The default 'scored' selection rule does -- the time
    penalty against the residual collar's earliness keeps it from
    outscoring the cleaner formation S in the same prior window.
    """
    geom = _geom()
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    formation = monopole_formation_modes(vp=Vp, vs=Vs, v_stoneley=Vst)
    # Plant a collar arrival between the formation P and S in apparent
    # slowness so it actively contaminates both windows.
    collar_slow = 1.0 / 3300.0   # ~92 us/ft (P=68 us/ft, S=122 us/ft)
    data = synthesize_lwd_gather(
        geom, formation,
        collar_amplitude=1.0, collar_slowness=collar_slow,
        noise=0.03, seed=7,
    )
    # Notch a wide band around the known collar slowness; the slant-
    # stack point-spread function smears narrow-band arrivals into
    # adjacent slownesses, so a +/- 8 % notch is too narrow to
    # suppress the collar enough for the picker.
    notch_lo = collar_slow * 0.85
    notch_hi = collar_slow * 1.15
    cleaned = notch_slowness_band(
        data, dt=geom.dt, offsets=geom.offsets,
        slow_min=notch_lo, slow_max=notch_hi,
        n_slowness=181, taper_width=0.15,
    )
    surf = stc(cleaned, dt=geom.dt, offsets=geom.offsets,
               slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
               n_slowness=181, window_length=4.0e-4, time_step=2)
    # Use 3-mode priors -- the test synthetic does not carry a
    # pseudo-Rayleigh arrival, and including the PR window in the
    # picker would let a spurious peak between S and Stoneley
    # promote-rule-block the Stoneley pick on this small synthetic.
    three_mode_priors = {m: DEFAULT_PRIORS[m] for m in ("P", "S", "Stoneley")}
    picks = pick_modes(surf, priors=three_mode_priors, threshold=0.4)
    assert {"P", "S", "Stoneley"}.issubset(set(picks))
    tol = 10.0 * US_PER_FT
    assert abs(picks["P"].slowness        - 1.0 / Vp)  < tol
    assert abs(picks["S"].slowness        - 1.0 / Vs)  < tol
    assert abs(picks["Stoneley"].slowness - 1.0 / Vst) < tol


def test_lwd_collar_appears_as_strongest_peak_before_rejection():
    """Sanity-check the contamination: on a contaminated gather the
    STC peak with the highest coherence is at (or near) the collar
    slowness, *not* at any formation mode. This justifies the
    rejection step in the previous test."""
    geom = _geom()
    formation = monopole_formation_modes(vp=4500.0, vs=2500.0, v_stoneley=1400.0)
    collar_slow = 1.0 / 3300.0
    data = synthesize_lwd_gather(
        geom, formation,
        collar_amplitude=2.5, collar_slowness=collar_slow,
        noise=0.03, seed=7,
    )
    surf = stc(data, dt=geom.dt, offsets=geom.offsets,
               slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
               n_slowness=181, window_length=4.0e-4, time_step=2)
    peaks = find_peaks(surf, threshold=0.5)
    assert peaks.size > 0
    # The highest-coherence peak in the unrejected gather is the
    # collar arrival, which lies in [80, 130] us/ft.
    top = peaks[0]   # find_peaks returns peaks sorted by descending coherence
    top_us_per_ft = top[0] / US_PER_FT
    assert 80.0 <= top_us_per_ft <= 130.0
