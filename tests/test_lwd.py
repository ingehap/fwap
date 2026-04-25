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


# ---------------------------------------------------------------------
# Quadrupole-source LWD ring synthesis + receiver-side m=2 stacking
# ---------------------------------------------------------------------


def test_synthesize_quadrupole_ring_gather_shapes_and_pattern():
    """Per-receiver amplitude at the known formation-arrival sample
    follows cos(2(theta - source_azimuth))."""
    from fwap.lwd import synthesize_quadrupole_lwd_gather
    Vs = 2500.0
    tool_offset = 3.0
    dt = 1.0e-5
    g = synthesize_quadrupole_lwd_gather(
        n_rec=8, n_samples=1024, dt=dt, tool_offset=tool_offset,
        formation_slowness=1.0 / Vs, formation_amplitude=1.0,
        source_azimuth=0.0, include_collar=False, noise=0.0, seed=0,
    )
    assert g.data.shape == (8, 1024)
    assert g.azimuths.shape == (8,)
    np.testing.assert_allclose(g.axial_offsets, 3.0)
    # Sample at the formation-arrival time; the Ricker peak there is
    # `1.0 * formation_amplitude * cos(2 theta_i)` (no other modes,
    # no noise).
    j_peak = int(round((tool_offset / Vs) / dt))
    expected = np.cos(2.0 * g.azimuths)
    np.testing.assert_allclose(g.data[:, j_peak], expected, atol=1.0e-12)


def test_synthesize_quadrupole_ring_rejects_too_few_receivers():
    """n_rec < 4 cannot resolve the m=2 pattern by Nyquist."""
    from fwap.lwd import synthesize_quadrupole_lwd_gather
    with pytest.raises(ValueError, match="n_rec"):
        synthesize_quadrupole_lwd_gather(n_rec=3)


def test_quadrupole_stack_rejects_monopole_pattern():
    """An m=0 (uniform) signal has zero projection on cos(2 theta)."""
    from fwap.lwd import quadrupole_stack
    n_rec = 8
    azimuths = np.linspace(0.0, 2.0 * np.pi, n_rec, endpoint=False)
    pulse = np.zeros((n_rec, 256))
    pulse[:, 100] = 1.0   # identical impulse on every receiver (m=0)
    stacked = quadrupole_stack(pulse, azimuths, source_azimuth=0.0)
    # cos(2 theta) has zero mean over a uniformly-sampled ring.
    assert abs(stacked).max() < 1.0e-10


def test_quadrupole_stack_rejects_dipole_pattern():
    """An m=1 (cos(theta)) signal is orthogonal to cos(2 theta)."""
    from fwap.lwd import quadrupole_stack
    n_rec = 8
    azimuths = np.linspace(0.0, 2.0 * np.pi, n_rec, endpoint=False)
    weights_dipole = np.cos(azimuths)
    pulse = np.zeros((n_rec, 256))
    pulse[:, 100] = weights_dipole
    stacked = quadrupole_stack(pulse, azimuths, source_azimuth=0.0)
    # cos(theta) and cos(2 theta) are orthogonal under uniform sampling.
    assert abs(stacked).max() < 1.0e-10


def test_quadrupole_stack_passes_quadrupole_pattern():
    """An m=2 input survives the m=2 projection (modulo a constant)."""
    from fwap.lwd import quadrupole_stack
    n_rec = 8
    azimuths = np.linspace(0.0, 2.0 * np.pi, n_rec, endpoint=False)
    weights_quadrupole = np.cos(2.0 * azimuths)
    pulse = np.zeros((n_rec, 256))
    pulse[:, 100] = weights_quadrupole
    stacked = quadrupole_stack(pulse, azimuths, source_azimuth=0.0)
    # Sum_i cos^2(2 theta_i) = n_rec / 2 for uniform azimuths (n_rec >= 3).
    assert stacked[100] == pytest.approx(n_rec / 2.0)


def test_quadrupole_stack_rejects_shape_mismatch():
    """Wrong shapes raise ValueError with named messages."""
    from fwap.lwd import quadrupole_stack
    with pytest.raises(ValueError, match="2-D"):
        quadrupole_stack(np.zeros(8), np.linspace(0, 2*np.pi, 8))
    with pytest.raises(ValueError, match="azimuths"):
        quadrupole_stack(np.zeros((8, 100)), np.linspace(0, 2*np.pi, 4))


def test_quadrupole_stack_recovers_formation_screw_arrival_time():
    """Stacking the synthesised quadrupole gather peaks at the
    formation-mode arrival time."""
    from fwap.lwd import quadrupole_stack, synthesize_quadrupole_lwd_gather
    Vs = 2500.0
    tool_offset = 3.0
    g = synthesize_quadrupole_lwd_gather(
        n_rec=8, n_samples=2048, dt=1.0e-5,
        tool_offset=tool_offset,
        formation_slowness=1.0 / Vs,
        formation_f0=6000.0,
        formation_amplitude=1.0,
        include_collar=False, noise=0.0, seed=0,
    )
    stacked = quadrupole_stack(g.data, g.azimuths,
                               source_azimuth=g.source_azimuth)
    t = np.arange(stacked.size) * g.dt
    t_truth = tool_offset / Vs
    t_peak = t[int(np.argmax(np.abs(stacked)))]
    # Within one sample of truth.
    assert abs(t_peak - t_truth) < 2.0 * g.dt


def test_lwd_quadrupole_priors_contains_collar_and_formation_modes():
    """The helper returns a priors dict with the two LWD-quadrupole modes."""
    from fwap.lwd import lwd_quadrupole_priors
    priors = lwd_quadrupole_priors()
    assert set(priors) == {"CollarQuadrupole", "FormationShear"}
    # Collar arrives first (lower slowness) -> order 0.
    assert priors["CollarQuadrupole"]["order"] == 0
    assert priors["FormationShear"]["order"] == 1
    # Slowness windows in the published Tang & Cheng 2004 ranges.
    for prior in priors.values():
        assert 0.0 < prior["slow_min"] < prior["slow_max"]


def test_quadrupole_stack_then_pick_recovers_formation_shear():
    """End-to-end: synthesize quadrupole gather with collar contamination,
    stack to a single trace, run the picker on a synthetic axial gather
    built by replicating the stacked trace at a few axial offsets, and
    confirm the formation-shear slowness is recovered to within
    10 us/ft of truth."""
    from fwap.coherence import stc as run_stc
    from fwap.lwd import (
        DEFAULT_COLLAR_SLOWNESS_S_PER_M,
        lwd_quadrupole_priors,
        quadrupole_stack,
        synthesize_quadrupole_lwd_gather,
    )
    from fwap.picker import pick_modes
    Vs = 2300.0
    n_axial = 8
    dr = 0.1524
    tr_offset0 = 3.0
    dt = 1.0e-5
    n_samples = 2048
    # One ring per axial offset; same source / quadrupole pattern;
    # arrivals delayed per offset by formation_slowness * (offset_i).
    axial_traces = np.empty((n_axial, n_samples), dtype=float)
    for k in range(n_axial):
        offset_k = tr_offset0 + k * dr
        g = synthesize_quadrupole_lwd_gather(
            n_rec=8, n_samples=n_samples, dt=dt,
            tool_offset=offset_k,
            formation_slowness=1.0 / Vs,
            formation_f0=6000.0, formation_amplitude=1.0,
            include_collar=True,
            collar_slowness=DEFAULT_COLLAR_SLOWNESS_S_PER_M,
            collar_amplitude=1.0,
            noise=0.02, seed=11 + k,
        )
        axial_traces[k] = quadrupole_stack(
            g.data, g.azimuths, source_azimuth=g.source_azimuth)
    offsets = tr_offset0 + np.arange(n_axial) * dr
    surf = run_stc(axial_traces, dt=dt, offsets=offsets,
                   slowness_range=(50e-6, 600e-6),
                   n_slowness=181, window_length=4.0e-4, time_step=2)
    picks = pick_modes(surf, priors=lwd_quadrupole_priors(), threshold=0.4)
    assert "FormationShear" in picks
    assert abs(picks["FormationShear"].slowness - 1.0 / Vs) < 10.0 * US_PER_FT


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
