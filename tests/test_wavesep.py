"""Wave separation tests."""

from __future__ import annotations

import numpy as np

from fwap.synthetic import (
    ArrayGeometry,
    monopole_formation_modes,
    synthesize_gather,
)
from fwap.wavesep import (
    apply_moveout,
    fk_filter,
    fk_forward,
    fk_inverse,
    sequential_kl_separation,
    svd_project,
    tau_p_adjoint,
    tau_p_filter,
    tau_p_forward,
    tau_p_inverse,
    unapply_moveout,
)


def _monopole_gather(seed=42):
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    data = synthesize_gather(
        geom, monopole_formation_modes(Vp, Vs, Vst), noise=0.02, seed=seed
    )
    return geom, data, Vp, Vs, Vst


def test_fk_round_trip_is_identity():
    """fk_forward o fk_inverse reproduces the input."""
    geom, data, *_ = _monopole_gather()
    spec, f, k = fk_forward(data, geom.dt, geom.dr)
    back = fk_inverse(spec, n_samples=data.shape[1])
    assert np.allclose(back, data, atol=1.0e-10)


def test_apply_unapply_moveout_is_identity():
    """unapply_moveout undoes apply_moveout up to the Nyquist-bin floor.

    The round trip goes ``data -> rfft -> phase -> irfft -> rfft -> -phase
    -> irfft``. For even-length inputs the Nyquist bin of the rfft is
    forced real, so any imaginary phase kick applied there is silently
    zeroed, which puts a finite floor on the round-trip accuracy. The
    RMS error should still be small compared to the gather RMS.
    """
    geom, data, Vp, *_ = _monopole_gather()
    flat = apply_moveout(data, geom.dt, geom.offsets, 1.0 / Vp)
    back = unapply_moveout(flat, geom.dt, geom.offsets, 1.0 / Vp)
    data_rms = np.sqrt(np.mean(data**2))
    err_rms = np.sqrt(np.mean((back - data) ** 2))
    assert err_rms / data_rms < 1.0e-3


def test_fk_filter_attenuates_out_of_band_slowness():
    """f-k pass-band around 1/Vp keeps the P arrival, suppresses Stoneley."""
    geom, data, Vp, Vs, Vst = _monopole_gather()
    p_only = fk_filter(
        data,
        geom.dt,
        geom.dr,
        slow_min=1.0 / 5500,
        slow_max=1.0 / 3600,
        taper_width=0.3,
    )
    # Compare energy in the first quarter of the record (P zone) vs
    # last quarter (Stoneley zone).
    n = data.shape[1]
    q = n // 4
    p_p_zone = np.sum(p_only[:, :q] ** 2)
    p_s_zone = np.sum(p_only[:, -q:] ** 2)
    assert p_p_zone > 5.0 * p_s_zone


def test_svd_project_isolates_coherent_mode():
    """svd_project at the true P slowness recovers a coherent P arrival.

    The decomposition must satisfy two properties:

    1. ``coh + resid == data`` exactly (the SVD reconstruction is
       unitary).
    2. The coherent part correlates strongly with a synthetic gather
       that contains only the P mode, i.e. ``svd_project`` really did
       pick out the P-arrival waveform rather than producing a
       scrambled rank-1 fit. This is a clean-mode property; in a
       multi-mode gather a higher-amplitude, non-flat mode can still
       dominate the rank-1 approximation, so the test uses a
       P-only-plus-noise gather to verify the algorithm's ideal-case
       behaviour. ``sequential_kl_separation`` is the way to peel
       interfering modes in the multi-mode case.
    """
    from fwap.synthetic import ArrayGeometry, Mode, synthesize_gather

    Vp = 4500.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    p_mode = Mode("P", slowness=1.0 / Vp, f0=15000.0, amplitude=1.0)
    data = synthesize_gather(geom, [p_mode], noise=0.2, seed=0)
    p_only_truth = synthesize_gather(geom, [p_mode], noise=0.0, seed=0)

    coh, resid = svd_project(data, geom.dt, geom.offsets, 1.0 / Vp, rank=1)
    # (1) exact reconstruction
    assert np.allclose(coh + resid, data, atol=1.0e-10)

    # (2) coh correlates with the noise-free P synthetic.
    correl = np.mean(
        [np.corrcoef(coh[i], p_only_truth[i])[0, 1] for i in range(geom.n_rec)]
    )
    assert correl > 0.85, f"P correlation only {correl:.2f}"


def test_sequential_kl_separation_sum_equals_input():
    """Component sum + residual = input (exact decomposition)."""
    geom, data, Vp, Vs, Vst = _monopole_gather()
    comps, resid = sequential_kl_separation(
        data, geom.dt, geom.offsets, slownesses=[1.0 / Vp, 1.0 / Vs, 1.0 / Vst], rank=1
    )
    reconstructed = sum(comps, np.zeros_like(data)) + resid
    assert np.allclose(reconstructed, data, atol=1.0e-10)


def test_svd_project_n_keep_alias_matches_rank():
    """The deprecated ``n_keep`` keyword still produces the same output."""
    geom, data, Vp, _, _ = _monopole_gather()
    coh_rank, _ = svd_project(data, geom.dt, geom.offsets, 1.0 / Vp, rank=2)
    coh_alias, _ = svd_project(data, geom.dt, geom.offsets, 1.0 / Vp, n_keep=2)
    assert np.allclose(coh_rank, coh_alias, atol=1.0e-12)


def test_fk_filter_passband_orientation():
    """fk_filter passes ``+s`` energy when the pass-band contains +s.

    Build a single forward-propagating arrival at physical slowness
    ``s_true > 0`` and check that a pass-band around ``+s_true``
    preserves most of the energy. If the sign convention were wrong
    (``S = +k/f`` instead of ``S = -k/f``) this filter would place
    the pass-band on the wrong side of the ``k`` axis and reject
    almost everything.
    """
    from fwap.synthetic import ArrayGeometry, Mode

    s_true = 1.0 / 4000.0
    geom = ArrayGeometry(n_rec=16, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    mode = Mode(name="P", slowness=s_true, f0=8000.0, amplitude=1.0)
    data = synthesize_gather(geom, [mode], noise=0.0, seed=0)
    keep = fk_filter(
        data,
        geom.dt,
        geom.dr,
        slow_min=0.8 * s_true,
        slow_max=1.2 * s_true,
        taper_width=0.3,
    )
    e_keep = np.sum(keep**2)
    e_in = np.sum(data**2)
    assert e_keep / e_in > 0.4


def test_fk_filter_rejects_zero_slowness_band():
    """A narrow pass-band far from the true slowness rejects all energy."""
    geom, data, Vp, *_ = _monopole_gather()
    # Pass-band centered far below any physical sonic slowness.
    filt = fk_filter(
        data, geom.dt, geom.dr, slow_min=1.0e-7, slow_max=3.0e-7, taper_width=0.3
    )
    e_filt = np.sum(filt**2)
    e_in = np.sum(data**2)
    assert e_filt / e_in < 0.01


def test_fk_filter_rejects_inverted_slowness_range():
    """fk_filter raises ValueError when slow_max <= slow_min."""
    import pytest

    geom, data, *_ = _monopole_gather()
    with pytest.raises(ValueError):
        fk_filter(data, geom.dt, geom.dr, slow_min=3.0e-4, slow_max=2.0e-4)
    with pytest.raises(ValueError):
        fk_filter(data, geom.dt, geom.dr, slow_min=-1.0e-4, slow_max=2.0e-4)


# ---------------------------------------------------------------------
# tau-p / slant-stack / linear Radon
# ---------------------------------------------------------------------


def test_tau_p_forward_shape_and_peak_localisation():
    """A single plane wave produces a sharp peak at the right (tau, p)."""
    n_rec, n_samp, dt = 12, 1024, 1.0e-5
    dx = 0.1524
    offsets = np.arange(n_rec) * dx
    p0 = 1.5e-4  # s/m
    intercept = 0.005

    t = np.arange(n_samp) * dt
    f0 = 5_000.0
    data = np.zeros((n_rec, n_samp))
    for i in range(n_rec):
        a = (np.pi * f0 * (t - p0 * offsets[i] - intercept)) ** 2
        data[i] = (1.0 - 2.0 * a) * np.exp(-a)

    slownesses = np.linspace(0.5e-4, 3.0e-4, 256)
    panel = tau_p_forward(data, dt, offsets, slownesses)
    assert panel.shape == (256, n_samp)

    # Peak in (slowness, tau).
    i_peak, j_peak = np.unravel_index(np.argmax(np.abs(panel)), panel.shape)
    assert abs(slownesses[i_peak] - p0) < 5e-7  # < 0.1 us/ft
    assert abs(j_peak * dt - intercept) < 1.0e-4  # < 0.1 ms


def test_tau_p_inverse_round_trip_is_identity():
    """tau_p_inverse(tau_p_forward(d)) reproduces d to ~0.1% on a clean
    monopole gather."""
    geom, data, *_ = _monopole_gather()
    from fwap._common import US_PER_FT

    slownesses = np.linspace(20.0 * US_PER_FT, 360.0 * US_PER_FT, 256)
    panel = tau_p_forward(data, geom.dt, geom.offsets, slownesses)
    recon = tau_p_inverse(panel, geom.dt, geom.offsets, slownesses)
    err = np.sqrt(np.mean((recon - data) ** 2)) / np.sqrt(np.mean(data**2))
    assert err < 5.0e-3, f"round-trip relative err {err:.3e} too large"


def test_tau_p_adjoint_shape():
    """tau_p_adjoint returns the same (n_rec, n_samples) shape as input."""
    geom, data, *_ = _monopole_gather()
    from fwap._common import US_PER_FT

    slownesses = np.linspace(20.0 * US_PER_FT, 360.0 * US_PER_FT, 128)
    panel = tau_p_forward(data, geom.dt, geom.offsets, slownesses)
    adj = tau_p_adjoint(panel, geom.dt, geom.offsets, slownesses)
    assert adj.shape == data.shape


def test_tau_p_filter_isolates_each_mode_by_slowness():
    """tau_p_filter on a P/S/Stoneley gather: each band's STC peak slowness
    matches the planted mode."""
    from fwap._common import US_PER_FT
    from fwap.coherence import stc

    geom, data, Vp, Vs, Vst = _monopole_gather()

    bands = {
        "P": (50.0 * US_PER_FT, 85.0 * US_PER_FT),
        "S": (100.0 * US_PER_FT, 150.0 * US_PER_FT),
        "Stoneley": (200.0 * US_PER_FT, 240.0 * US_PER_FT),
    }
    # Vp / Vs / Vst from the canonical gather aren't used past
    # band-edge selection -- the assertion is that the dominant
    # filtered slowness lands in-band, not on the truth.
    _ = (Vp, Vs, Vst)
    for name, (s_lo, s_hi) in bands.items():
        filt = tau_p_filter(
            data, geom.dt, geom.offsets, s_lo, s_hi, n_slowness=181, taper_width=0.2
        )
        res = stc(
            filt,
            dt=geom.dt,
            offsets=geom.offsets,
            slowness_range=(30.0 * US_PER_FT, 360.0 * US_PER_FT),
            n_slowness=121,
            window_length=4.0e-4,
            time_step=2,
        )
        rho = np.nan_to_num(res.coherence)
        i_max, _ = np.unravel_index(np.argmax(rho), rho.shape)
        s_recovered = res.slowness[i_max]
        # The filtered output's dominant slowness has to fall inside
        # the pass-band, not necessarily on the truth -- the slant-
        # stack point-spread function can shift the peak by a few
        # us/ft toward the band centre.
        assert s_lo - 5.0 * US_PER_FT <= s_recovered <= s_hi + 5.0 * US_PER_FT, (
            f"{name}: filtered peak at {s_recovered / US_PER_FT:.1f} us/ft "
            f"outside pass-band [{s_lo / US_PER_FT:.0f}, {s_hi / US_PER_FT:.0f}]"
        )


def test_tau_p_filter_rejects_inverted_slowness_range():
    """tau_p_filter raises ValueError when slow_max <= slow_min."""
    import pytest

    geom, data, *_ = _monopole_gather()
    with pytest.raises(ValueError):
        tau_p_filter(data, geom.dt, geom.offsets, slow_min=3.0e-4, slow_max=2.0e-4)
    with pytest.raises(ValueError):
        tau_p_filter(data, geom.dt, geom.offsets, slow_min=-1.0e-4, slow_max=2.0e-4)


def test_tau_p_forward_supports_non_uniform_offsets():
    """tau_p does not require uniform offsets, unlike fk_filter."""
    n_rec, n_samp, dt = 8, 1024, 1.0e-5
    # Deliberately non-uniform spacing (jittered linear).
    rng = np.random.default_rng(0)
    offsets = np.linspace(3.0, 4.0, n_rec) + rng.normal(0, 0.01, n_rec)
    p0 = 1.5e-4
    t = np.arange(n_samp) * dt
    f0 = 5_000.0
    data = np.zeros((n_rec, n_samp))
    for i in range(n_rec):
        a = (np.pi * f0 * (t - p0 * offsets[i] - 0.005)) ** 2
        data[i] = (1.0 - 2.0 * a) * np.exp(-a)

    slownesses = np.linspace(0.5e-4, 3.0e-4, 256)
    panel = tau_p_forward(data, dt, offsets, slownesses)
    i_peak, _ = np.unravel_index(np.argmax(np.abs(panel)), panel.shape)
    assert abs(slownesses[i_peak] - p0) < 5e-7


# ----------------------------------------------------------------------
# W1 group A: how well the slowness filter separates, as a number
# ----------------------------------------------------------------------


def test_tau_p_filter_selectivity_is_quantified_in_both_directions():
    """``test_tau_p_filter_isolates_each_mode_by_slowness`` says it
    isolates. This says by how much, and that a wider window is worse.

    The synthetic is exactly linear -- a two-mode gather equals the sum
    of its one-mode gathers to ``0.0`` -- so each pure mode is available
    as a reference and the filter's output can be correlated against
    both.

    Measured on P at 222 us/m against Stoneley at 690 us/m:

        passband 190-260 us/m  ->  0.94 with P,  0.14 with Stoneley
        passband 150-300 us/m  ->  0.88 with P,  0.20 with Stoneley
        passband 120-350 us/m  ->  0.85 with P,  0.26 with Stoneley

    Selectivity degrades monotonically as the band widens, which is the
    slant stack's point-spread function showing up as a number rather
    than as a caveat. The complementary band passes Stoneley (0.87) and
    rejects P (0.23), so this is not a statement about one mode being
    easier.
    """
    from fwap.synthetic import ArrayGeometry, Mode, synthesize_gather

    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    fast = Mode(name="P", slowness=1.0 / 4500.0, intercept=2.0e-4, f0=10000.0)
    slow = Mode(name="ST", slowness=1.0 / 1450.0, intercept=7.0e-4, f0=10000.0)

    only_fast = synthesize_gather(geom, [fast], noise=0.0, seed=0)
    only_slow = synthesize_gather(geom, [slow], noise=0.0, seed=0)
    both = synthesize_gather(geom, [fast, slow], noise=0.0, seed=0)

    # The reference the whole test rests on.
    assert np.abs(both - (only_fast + only_slow)).max() == 0.0

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))

    selectivity = {}
    for lo, hi in ((190.0e-6, 260.0e-6), (150.0e-6, 300.0e-6), (120.0e-6, 350.0e-6)):
        kept = tau_p_filter(both, geom.dt, geom.offsets, lo, hi)
        selectivity[hi - lo] = (
            correlation(kept, only_fast),
            correlation(kept, only_slow),
        )

    widths = sorted(selectivity)
    assert selectivity[widths[0]][0] > 0.93
    assert selectivity[widths[0]][1] < 0.15
    # Wider band, worse on both counts -- monotonically.
    for narrow, wide in zip(widths, widths[1:]):
        assert selectivity[wide][0] < selectivity[narrow][0]
        assert selectivity[wide][1] > selectivity[narrow][1]

    # The other side of the band works too.
    kept_slow = tau_p_filter(both, geom.dt, geom.offsets, 600.0e-6, 780.0e-6)
    assert correlation(kept_slow, only_slow) > 0.85
    assert correlation(kept_slow, only_fast) < 0.25
