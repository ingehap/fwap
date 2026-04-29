"""Intercept-time inversion tests."""

from __future__ import annotations

import numpy as np

from fwap.tomography import (
    assemble_observations_from_picks,
    build_design_matrix,
    build_design_matrix_segmented,
    solve_intercept_time,
)


def _build_synthetic_picks(
    n_depth=40, n_rec=8, Vp=4500.0, delay_peak=2.0e-5, noise=0.0, seed=0
):
    rng = np.random.default_rng(seed)
    dz = 0.1524
    tr_offset = 3.0
    dr = dz
    z = np.arange(n_depth) * dz
    offsets = tr_offset + np.arange(n_rec) * dr
    zc = z.mean()
    delay = delay_peak * np.exp(-0.5 * ((z - zc) / (4 * dz)) ** 2)
    tt = np.zeros((n_depth, n_rec))
    for j, zs in enumerate(z):
        for k, x in enumerate(offsets):
            z_rec = zs + x
            idx = int(np.clip(round((z_rec - z[0]) / dz), 0, n_depth - 1))
            tt[j, k] = (1.0 / Vp) * x + delay[j] + delay[idx]
    if noise > 0:
        tt += rng.normal(scale=noise, size=tt.shape)
    return z, offsets, tt, delay, Vp


def test_solve_intercept_time_recovers_background_slowness():
    """Synthetic data with known Vp, delays, round-trips through the solver."""
    z, offsets, tt, delay, Vp = _build_synthetic_picks()
    travel_times, off_vec, src_idx, rec_idx, n_d, depth_axis = (
        assemble_observations_from_picks(z, offsets, tt)
    )
    r = solve_intercept_time(
        travel_times,
        off_vec,
        src_idx,
        rec_idx,
        n_d,
        depth_axis=depth_axis,
        mean_delay_zero=True,
        smooth_s=5.0e3,
        smooth_src=1.0e3,
        smooth_rec=1.0e3,
        delay_l2=1.0e2,
        method="midpoint",
    )
    # Mean inverted slowness should match truth to better than 2%.
    # (The discretised midpoint design matrix has a small systematic
    # bias vs the true continuous ray path at long offsets.)
    assert abs(np.mean(r.slowness) - 1.0 / Vp) / (1.0 / Vp) < 0.02
    assert r.rms_residual < 1.0e-5


def test_mean_delay_zero_pins_each_block_independently():
    """Both delay blocks sum to ~0 independently after the split constraint."""
    z, offsets, tt, delay, Vp = _build_synthetic_picks(noise=1e-6, seed=1)
    travel_times, off_vec, src_idx, rec_idx, n_d, depth_axis = (
        assemble_observations_from_picks(z, offsets, tt)
    )
    r = solve_intercept_time(
        travel_times,
        off_vec,
        src_idx,
        rec_idx,
        n_d,
        depth_axis=depth_axis,
        mean_delay_zero=True,
        smooth_s=1.0e3,
        smooth_src=0.0,
        smooth_rec=0.0,
        delay_l2=0.0,
        method="midpoint",
    )
    # Each block mean should be zero to ~1e-9 s.
    assert abs(r.delay_src.mean()) < 1.0e-9
    assert abs(r.delay_rec.mean()) < 1.0e-9


def test_segmented_design_matrix_matches_midpoint_on_uniform_grid():
    """On a fine grid the two design matrices give comparable slownesses."""
    z, offsets, tt, delay, Vp = _build_synthetic_picks(n_depth=60)
    travel_times, off_vec, src_idx, rec_idx, n_d, depth_axis = (
        assemble_observations_from_picks(z, offsets, tt)
    )
    r_mp = solve_intercept_time(
        travel_times,
        off_vec,
        src_idx,
        rec_idx,
        n_d,
        depth_axis=depth_axis,
        mean_delay_zero=True,
        smooth_s=5.0e3,
        smooth_src=1.0e3,
        smooth_rec=1.0e3,
        delay_l2=1.0e2,
        method="midpoint",
    )
    r_seg = solve_intercept_time(
        travel_times,
        off_vec,
        depth_axis[src_idx],
        depth_axis[rec_idx],
        n_d,
        depth_axis=depth_axis,
        mean_delay_zero=True,
        smooth_s=5.0e3,
        smooth_src=1.0e3,
        smooth_rec=1.0e3,
        delay_l2=1.0e2,
        method="segmented",
    )
    assert abs(np.mean(r_mp.slowness) - np.mean(r_seg.slowness)) < 5.0e-6


def test_build_design_matrix_shapes():
    """build_design_matrix returns the advertised shapes."""
    n_depth = 12
    n_obs = 30
    G, d = build_design_matrix(
        travel_times=np.arange(n_obs, dtype=float) * 1e-5,
        offsets=np.full(n_obs, 3.0),
        src_depth_idx=np.zeros(n_obs, dtype=int),
        rec_depth_idx=np.arange(n_obs, dtype=int) % n_depth,
        n_depth=n_depth,
    )
    assert G.shape == (n_obs, 3 * n_depth)
    assert d.shape == (n_obs,)


def test_build_design_matrix_segmented_preserves_slowness_column_sum():
    """Segmented rows sum (over cells) to total offset for each observation."""
    cell_depths = np.linspace(0.0, 10.0, 21)
    n_obs = 5
    src = np.array([0.1, 2.0, 3.5, 7.0, 9.9])
    rec = np.array([1.0, 4.5, 5.0, 9.0, 9.95])
    tt = np.zeros(n_obs)
    G, _ = build_design_matrix_segmented(tt, src, rec, cell_depths)
    n_cells = cell_depths.size
    slowness_block = G[:, :n_cells]
    # Each row's row-sum equals |src - rec| (with clipping at edges).
    expected = np.abs(rec - src)
    assert np.allclose(slowness_block.sum(axis=1), expected, atol=1.0e-12)


def test_segmented_ray_exactly_on_cell_boundary_rounds_up():
    """A depth exactly on an interior edge lands in the higher cell.

    The docstring pins this rounding direction; any change would
    silently shift every boundary pick one cell.
    """
    cell_depths = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Source exactly on the edge between cell 1 and cell 2 (z = 1.5).
    # With dz = 1.0, cell centres are z = 0, 1, 2, 3, 4 and edges are
    # -0.5, 0.5, 1.5, 2.5, 3.5, 4.5. Depth 1.5 is on the edge between
    # cell centre 1 and cell centre 2, and must land in cell 2.
    src = np.array([1.5])
    rec = np.array([3.0])
    tt = np.zeros(1)
    G, _ = build_design_matrix_segmented(tt, src, rec, cell_depths)
    n_cells = cell_depths.size
    src_indicator = G[0, n_cells : 2 * n_cells]
    assert np.argmax(src_indicator) == 2
    assert src_indicator[2] == 1.0


def test_segmented_rays_outside_grid_contribute_zero():
    """Portions of the ray outside the cell grid contribute nothing."""
    cell_depths = np.linspace(2.0, 5.0, 7)  # covers [1.75, 5.25]
    n_cells = cell_depths.size
    # Ray from 0.5 (below grid) to 4.0 (in grid). Only the portion
    # inside [1.75, 4.0] should land on the slowness block.
    src = np.array([0.5])
    rec = np.array([4.0])
    tt = np.zeros(1)
    G, _ = build_design_matrix_segmented(tt, src, rec, cell_depths)
    slowness_row_sum = G[0, :n_cells].sum()
    # Expected in-grid overlap: 4.0 - 1.75 = 2.25 m
    assert abs(slowness_row_sum - 2.25) < 1.0e-12
    # The source-delay indicator is clipped to the nearest in-grid
    # cell (cell 0 at depth 2.0).
    assert np.argmax(G[0, n_cells : 2 * n_cells]) == 0


def test_segmented_rejects_non_uniform_grid():
    """A non-uniformly-spaced cell axis must raise ValueError."""
    import pytest

    cell_depths = np.array([0.0, 1.0, 2.1, 3.0])  # not uniform
    with pytest.raises(ValueError, match="uniformly spaced"):
        build_design_matrix_segmented(
            np.zeros(1), np.array([0.5]), np.array([2.5]), cell_depths
        )


def test_segmented_rejects_reversed_grid():
    """A non-increasing cell axis must raise ValueError."""
    import pytest

    cell_depths = np.array([3.0, 2.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        build_design_matrix_segmented(
            np.zeros(1), np.array([0.5]), np.array([2.5]), cell_depths
        )


def test_segmented_rejects_too_few_cells():
    """A cell axis with fewer than 2 points must raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match=">= 2"):
        build_design_matrix_segmented(
            np.zeros(1), np.array([0.5]), np.array([2.5]), np.array([1.0])
        )


# ---------------------------------------------------------------------
# Delay -> altered-zone thickness conversion
# ---------------------------------------------------------------------


def test_delay_to_altered_zone_known_values():
    """First-order conversion: delay = 2 * h * (s_alt - s_virgin)."""
    from fwap.tomography import delay_to_altered_zone_thickness

    # 20 us delay, 1/4500 vs 1/3500 s/m. s_alt - s_virgin = 6.35e-5 s/m.
    # Expected thickness = 20e-6 / (2 * 6.35e-5) = 0.1575 m.
    thickness = delay_to_altered_zone_thickness(
        delay=20.0e-6,
        slowness_virgin=1.0 / 4500.0,
        slowness_altered=1.0 / 3500.0,
    )
    expected = 20.0e-6 / (2.0 * (1.0 / 3500.0 - 1.0 / 4500.0))
    assert abs(float(thickness) - expected) < 1.0e-12


def test_delay_to_altered_zone_broadcasts_over_depth():
    """Array input produces a same-shape array output."""
    from fwap.tomography import delay_to_altered_zone_thickness

    delay = np.linspace(0.0, 30e-6, 11)
    out = delay_to_altered_zone_thickness(
        delay, slowness_virgin=1.0 / 4500.0, slowness_altered=1.0 / 3500.0
    )
    assert out.shape == delay.shape
    assert np.all(out >= 0.0)


def test_delay_to_altered_zone_clips_negative():
    """Negative delays (inversion noise) map to zero thickness."""
    from fwap.tomography import delay_to_altered_zone_thickness

    delay = np.array([-5e-6, 0.0, 10e-6, 20e-6])
    out = delay_to_altered_zone_thickness(
        delay, slowness_virgin=1.0 / 4500.0, slowness_altered=1.0 / 3500.0
    )
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] > 0.0
    assert out[3] > out[2]


def test_delay_to_altered_zone_rejects_inverted_slownesses():
    """s_altered <= s_virgin raises ValueError."""
    import pytest

    from fwap.tomography import delay_to_altered_zone_thickness

    with pytest.raises(ValueError, match="slowness_altered"):
        delay_to_altered_zone_thickness(
            delay=10e-6, slowness_virgin=1.0 / 3500.0, slowness_altered=1.0 / 4500.0
        )
    with pytest.raises(ValueError, match="slowness_altered"):
        delay_to_altered_zone_thickness(
            delay=10e-6, slowness_virgin=1.0 / 4000.0, slowness_altered=1.0 / 4000.0
        )


def test_delay_to_altered_zone_velocity_contrast_round_trip():
    """The velocity-contrast dual is the algebraic inverse of the
    thickness conversion: round-tripping through both must reproduce
    the original delay."""
    from fwap.tomography import (
        delay_to_altered_zone_thickness,
        delay_to_altered_zone_velocity_contrast,
    )

    delay = 20.0e-6
    s_virgin = 1.0 / 4500.0
    s_altered = 1.0 / 3500.0
    h = delay_to_altered_zone_thickness(delay, s_virgin, s_altered)
    contrast = delay_to_altered_zone_velocity_contrast(delay, h)
    # contrast = s_altered - s_virgin
    assert abs(float(contrast) - (s_altered - s_virgin)) < 1.0e-15
    # And: 2 * h * contrast must reproduce the original delay.
    assert abs(2.0 * float(h) * float(contrast) - delay) < 1.0e-15


def test_delay_to_altered_zone_velocity_contrast_rejects_nonpositive_thickness():
    """thickness must be strictly positive at every depth."""
    import pytest

    from fwap.tomography import delay_to_altered_zone_velocity_contrast

    with pytest.raises(ValueError, match="strictly positive"):
        delay_to_altered_zone_velocity_contrast(delay=10e-6, thickness=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        delay_to_altered_zone_velocity_contrast(delay=10e-6, thickness=-0.05)
    with pytest.raises(ValueError, match="strictly positive"):
        delay_to_altered_zone_velocity_contrast(
            delay=np.array([10e-6, 20e-6]), thickness=np.array([0.05, 0.0])
        )


def test_delay_to_altered_zone_velocity_contrast_clips_negative_delay():
    """Negative delays map to zero contrast (matches the thickness fn)."""
    from fwap.tomography import delay_to_altered_zone_velocity_contrast

    delay = np.array([-5e-6, 0.0, 10e-6])
    out = delay_to_altered_zone_velocity_contrast(delay, thickness=0.05)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] > 0.0


def test_altered_zone_estimate_with_thickness_anchor():
    """Pinning thickness yields the right slowness contrast and altered slowness."""
    from fwap.tomography import AlteredZoneEstimate, altered_zone_estimate

    delay = np.array([10e-6, 20e-6, 30e-6])
    s_virgin = 1.0 / 4500.0
    h = 0.05  # 5 cm halo
    est = altered_zone_estimate(delay, s_virgin, thickness=h)
    assert isinstance(est, AlteredZoneEstimate)
    # delay = 2 * h * contrast holds depth-by-depth.
    assert np.allclose(2.0 * est.thickness * est.slowness_contrast, delay)
    # slowness_altered = s_virgin + contrast.
    assert np.allclose(est.slowness_altered, s_virgin + est.slowness_contrast)
    # Thickness broadcasts to the delay shape.
    assert est.thickness.shape == delay.shape
    assert np.allclose(est.thickness, h)


def test_altered_zone_estimate_with_slowness_altered_anchor():
    """Pinning slowness_altered reproduces the existing thickness conversion."""
    from fwap.tomography import (
        altered_zone_estimate,
        delay_to_altered_zone_thickness,
    )

    delay = np.array([10e-6, 20e-6, 30e-6])
    s_virgin = 1.0 / 4500.0
    s_altered = 1.0 / 3500.0
    est = altered_zone_estimate(delay, s_virgin, slowness_altered=s_altered)
    expected_h = delay_to_altered_zone_thickness(delay, s_virgin, s_altered)
    assert np.allclose(est.thickness, expected_h)
    assert np.allclose(est.slowness_altered, s_altered)
    assert np.allclose(est.slowness_contrast, s_altered - s_virgin)


def test_altered_zone_estimate_rejects_both_or_neither_anchor():
    """The (h, Δs) pair is under-determined from a single delay; the
    helper must demand exactly one anchor."""
    import pytest

    from fwap.tomography import altered_zone_estimate

    s_virgin = 1.0 / 4500.0
    with pytest.raises(ValueError, match="under-determined"):
        altered_zone_estimate(delay=10e-6, slowness_virgin=s_virgin)
    with pytest.raises(ValueError, match="under-determined"):
        altered_zone_estimate(
            delay=10e-6,
            slowness_virgin=s_virgin,
            thickness=0.05,
            slowness_altered=1.0 / 3500.0,
        )


def test_altered_zone_estimate_per_depth_thickness_anchor():
    """A per-depth thickness anchor yields a per-depth contrast."""
    from fwap.tomography import altered_zone_estimate

    delay = np.array([10e-6, 20e-6, 30e-6])
    s_virgin = 1.0 / 4500.0
    h = np.array([0.02, 0.05, 0.10])
    est = altered_zone_estimate(delay, s_virgin, thickness=h)
    expected_contrast = delay / (2.0 * h)
    assert np.allclose(est.slowness_contrast, expected_contrast)
    assert np.allclose(est.thickness, h)
