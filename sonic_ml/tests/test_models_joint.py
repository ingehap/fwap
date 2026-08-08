"""Tests for joint multi-depth inversion (torch-gated).

Like the single-frame inversion tests, these pin *mechanism*, not accuracy: the
CI fixture trains a surrogate on ~32 samples, so any assertion about how much
depth coupling improves an answer would be measuring the fixture. The measured
comparison against the independent control lives in the module docstring and the
changelog, quoted with the surrogate and profiles it came from.

What is asserted here is what must not silently break: that the synthetic log is
inside the surrogate's training support and physically admissible, that the
penalty actually smooths, that lambda selection cannot see the truth, and that
the in-bed/at-boundary split is a real decomposition rather than a relabelling.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.inversion import InversionResult  # noqa: E402
from sonic_ml.models.joint import (  # noqa: E402
    DepthProfile,
    bed_vs_boundary_mae,
    contact_precision,
    invert_joint,
    no_skill_contact_precision,
    profile_mae,
    select_lambda,
    smooth_independent,
    synthesize_profile,
)
from sonic_ml.models.train import train_forward  # noqa: E402
from sonic_ml.split import stratified_split  # noqa: E402


@pytest.fixture(scope="module")
def surrogate(bundle):
    trained, _ = train_forward(
        bundle, split=stratified_split(bundle, seed=0), epochs=25, seed=0
    )
    return trained


@pytest.fixture(scope="module")
def profile(bundle):
    return synthesize_profile(bundle, n_depths=12, mean_bed_frames=4.0, seed=3)


# ------------------------------------------------------------------
# synthesize_profile
# ------------------------------------------------------------------


def test_profile_shapes_and_depth_axis(bundle, profile):
    assert profile.params.shape == (12, len(profile.active_names))
    assert profile.slowness.shape == (12, len(bundle.mode_names), bundle.freq.size)
    assert profile.clean_slowness.shape == profile.slowness.shape
    assert np.all(np.diff(profile.depths) > 0.0)
    assert profile.n_depths == 12


def test_profile_stays_inside_the_training_support(bundle, profile):
    """Out-of-range beds would measure extrapolation, not depth coupling."""
    for name in profile.active_names:
        column = bundle.param(name)
        values = profile.value(name)
        assert values.min() >= column.min() - 1e-9
        assert values.max() <= column.max() + 1e-9


def test_profile_respects_the_solver_constraint(profile):
    """vp > vs everywhere, or the modal solvers would refuse the formation."""
    assert np.all(profile.value("vp") > profile.value("vs"))


def test_beds_are_piecewise_constant_and_change_only_at_boundaries(profile):
    changed = np.where(np.any(np.diff(profile.params, axis=0) != 0.0, axis=1))[0] + 1
    np.testing.assert_array_equal(changed, profile.bed_boundaries)
    assert profile.n_beds == profile.bed_boundaries.size + 1


def test_gradation_softens_contacts_without_moving_the_beds(bundle):
    """The unfavourable case for a contact-preserving penalty, built honestly.

    Ramping must change only *how* a contact is reached, not which beds the log
    contains or where the contacts are --- otherwise the sharp and gradational
    runs are not a matched pair and comparing penalties across them proves
    nothing.
    """
    kw = dict(n_depths=30, mean_bed_frames=8.0, noise_us_per_ft=0.0, seed=7)
    sharp = synthesize_profile(bundle, **kw)
    ramped = synthesize_profile(bundle, gradation_frames=4, **kw)

    np.testing.assert_array_equal(sharp.bed_boundaries, ramped.bed_boundaries)
    steps_sharp = np.abs(np.diff(sharp.value("vs")))
    steps_ramped = np.abs(np.diff(ramped.value("vs")))
    assert steps_ramped.max() < steps_sharp.max()
    # the change is spread over more transitions, not removed
    assert (steps_ramped > 1.0).sum() > (steps_sharp > 1.0).sum()
    assert np.isfinite(ramped.slowness).any(axis=(1, 2)).all()


def test_zero_gradation_is_the_blocky_default(bundle):
    kw = dict(n_depths=20, mean_bed_frames=6.0, noise_us_per_ft=0.0, seed=4)
    np.testing.assert_array_equal(
        synthesize_profile(bundle, **kw).params,
        synthesize_profile(bundle, gradation_frames=0, **kw).params,
    )


def test_noise_free_profile_is_exactly_the_forward_model(bundle):
    quiet = synthesize_profile(bundle, n_depths=6, noise_us_per_ft=0.0, seed=1)
    np.testing.assert_array_equal(quiet.slowness, quiet.clean_slowness)


def test_noise_perturbs_data_without_inventing_absent_modes(bundle):
    """NaN means 'this mode is not there'; noise must not turn that into data."""
    noisy = synthesize_profile(bundle, n_depths=6, noise_us_per_ft=3.0, seed=1)
    np.testing.assert_array_equal(
        np.isfinite(noisy.slowness), np.isfinite(noisy.clean_slowness)
    )
    finite = np.isfinite(noisy.clean_slowness)
    assert np.any(noisy.slowness[finite] != noisy.clean_slowness[finite])


def test_profile_is_reproducible_and_seed_dependent(bundle):
    a = synthesize_profile(bundle, n_depths=8, seed=5)
    b = synthesize_profile(bundle, n_depths=8, seed=5)
    c = synthesize_profile(bundle, n_depths=8, seed=6)
    np.testing.assert_array_equal(a.params, b.params)
    assert not np.array_equal(a.params, c.params)


def test_profile_rejects_a_degenerate_interval(bundle):
    with pytest.raises(ValueError, match="at least 2"):
        synthesize_profile(bundle, n_depths=1)


def test_profile_rejects_a_solver_count_mismatch(bundle):
    from fwap import stoneley_dispersion

    with pytest.raises(ValueError, match="solvers for"):
        synthesize_profile(bundle, n_depths=4, solvers=(stoneley_dispersion,))


def test_profile_value_rejects_an_unknown_name(profile):
    with pytest.raises(KeyError, match="not in this profile"):
        profile.value("porosity")


def test_boundary_mask_widens_with_halo(profile):
    narrow = profile.boundary_mask(halo=1)
    wide = profile.boundary_mask(halo=2)
    assert wide.sum() >= narrow.sum()
    assert np.all(wide[narrow])


# ------------------------------------------------------------------
# invert_joint
# ------------------------------------------------------------------


def test_joint_inversion_shapes_and_contract(surrogate, profile):
    result = invert_joint(surrogate, profile.slowness, lam=1e-3, steps=30, n_starts=1)
    assert result.method == "joint"
    assert result.active_names == surrogate.param_std.active_names
    assert result.params.shape == (profile.n_depths, len(result.active_names))
    assert np.all(np.isfinite(result.params))
    assert np.all(result.data_misfit >= 0.0)


def test_penalty_smooths_the_recovered_profile(surrogate, profile):
    """The whole mechanism: a larger penalty must produce a flatter log.

    Measured as total absolute frame-to-frame change, normalised per parameter
    so the four columns are comparable.
    """

    def roughness(result):
        scale = np.abs(result.params).mean(axis=0)
        return float(np.abs(np.diff(result.params, axis=0) / scale).sum())

    warm = invert_joint(
        surrogate, profile.slowness, lam=0.0, steps=30, n_starts=1
    ).params
    loose = invert_joint(
        surrogate, profile.slowness, lam=0.0, steps=60, warm_start=warm
    )
    tight = invert_joint(
        surrogate, profile.slowness, lam=1.0, steps=60, warm_start=warm
    )
    assert roughness(tight) < roughness(loose)


def test_zero_penalty_keeps_optimising_the_data_misfit(surrogate, profile):
    """lam=0 is the control, so it must be plain independent optimisation."""
    warm = invert_joint(
        surrogate, profile.slowness, lam=0.0, steps=20, n_starts=1
    ).params
    short = invert_joint(surrogate, profile.slowness, lam=0.0, steps=1, warm_start=warm)
    long = invert_joint(
        surrogate, profile.slowness, lam=0.0, steps=200, warm_start=warm
    )
    assert long.data_misfit.mean() < short.data_misfit.mean()


def test_joint_inversion_is_reproducible(surrogate, profile):
    kw = dict(lam=1e-2, steps=25, n_starts=1)
    a = invert_joint(surrogate, profile.slowness, **kw)
    b = invert_joint(surrogate, profile.slowness, **kw)
    np.testing.assert_array_equal(a.params, b.params)


def test_joint_inversion_accepts_a_warm_start_verbatim(surrogate, profile):
    """Zero steps must return the warm start, or a lambda sweep is not controlled."""
    warm = invert_joint(
        surrogate, profile.slowness, lam=0.0, steps=10, n_starts=1
    ).params
    kept = invert_joint(surrogate, profile.slowness, lam=0.5, steps=0, warm_start=warm)
    np.testing.assert_allclose(kept.params, warm, rtol=1e-5)


def test_joint_inversion_rejects_bad_input(surrogate, profile):
    with pytest.raises(ValueError, match=r"\(D, M, F\)"):
        invert_joint(surrogate, profile.slowness[0], lam=0.0, steps=1)
    with pytest.raises(ValueError, match="at least 2 depth frames"):
        invert_joint(surrogate, profile.slowness[:1], lam=0.0, steps=1)
    with pytest.raises(ValueError, match="non-negative"):
        invert_joint(surrogate, profile.slowness, lam=-1.0, steps=1)
    with pytest.raises(ValueError, match="tv_eps must be positive"):
        invert_joint(surrogate, profile.slowness, lam=0.1, tv_eps=0.0, steps=1)
    with pytest.raises(ValueError, match="penalty must be"):
        invert_joint(surrogate, profile.slowness, lam=0.1, penalty="l1", steps=1)


# ------------------------------------------------------------------
# The bed-boundary-aware (TV) penalty
# ------------------------------------------------------------------


def test_tv_is_nearly_indifferent_to_how_change_is_distributed():
    """The defining property, checked on the penalty itself, not through a fit.

    Both profiles below contain the same *total* change. L2 charges four times
    as much when it arrives as one contact instead of four gentle steps, which
    is a standing incentive to erase contacts. TV is within a few percent of
    indifferent --- and indifference, not a preference for jumps, is what lets a
    contact survive.
    """
    from sonic_ml.models.joint import _roughness_term

    l2 = _roughness_term("l2", 0.05)
    tv = _roughness_term("tv", 0.05)
    one_jump = torch.tensor([[0.0], [0.0], [4.0], [4.0], [4.0]])
    spread = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])

    assert float(l2(one_jump) / l2(spread)) == pytest.approx(4.0, rel=1e-4)
    assert float(tv(one_jump) / tv(spread)) == pytest.approx(1.0, abs=0.05)


def test_tv_reduces_to_l2_behaviour_for_a_large_epsilon():
    """The two penalties are a family, not rivals, and eps is the dial."""
    from sonic_ml.models.joint import _roughness_term

    z = torch.tensor([[0.0], [0.02], [0.01], [0.03]])
    l2 = _roughness_term("l2", 0.0)(z)
    # for |diff| << eps, pseudo-Huber ~= |diff|^2 / (2 eps)
    big_eps = 50.0
    tv = _roughness_term("tv", big_eps)(z)
    assert tv == pytest.approx(l2 / (2 * big_eps), rel=1e-3)


def test_both_penalties_smooth_but_tv_keeps_more_structure(surrogate, profile):
    """At a weight strong enough to visibly smooth, TV must retain more jumps."""

    def roughness(result):
        scale = np.abs(result.params).mean(axis=0)
        return np.abs(np.diff(result.params, axis=0) / scale).sum(axis=1)

    warm = invert_joint(
        surrogate, profile.slowness, lam=0.0, steps=30, n_starts=1
    ).params
    kw = dict(lam=0.3, steps=80, warm_start=warm)
    flat = invert_joint(surrogate, profile.slowness, penalty="l2", **kw)
    kept = invert_joint(surrogate, profile.slowness, penalty="tv", **kw)
    # both smooth relative to the warm start
    start = InversionResult(
        params=warm,
        active_names=surrogate.param_std.active_names,
        data_misfit=np.zeros(profile.n_depths),
        n_starts=1,
        method="joint",
    )
    assert roughness(flat).sum() < roughness(start).sum()
    assert roughness(kept).sum() < roughness(start).sum()
    # but TV concentrates what is left into fewer, larger transitions
    assert roughness(kept).max() > roughness(flat).max()


def test_penalty_choice_changes_the_answer(surrogate, profile):
    warm = invert_joint(
        surrogate, profile.slowness, lam=0.0, steps=20, n_starts=1
    ).params
    kw = dict(lam=0.1, steps=60, warm_start=warm)
    a = invert_joint(surrogate, profile.slowness, penalty="l2", **kw)
    b = invert_joint(surrogate, profile.slowness, penalty="tv", **kw)
    assert not np.allclose(a.params, b.params)


# ------------------------------------------------------------------
# contact_precision
# ------------------------------------------------------------------


def test_perfect_recovery_finds_every_contact(profile):
    """A profile recovered exactly must score 1.0, or the metric is broken."""
    exact = InversionResult(
        params=profile.params.copy(),
        active_names=profile.active_names,
        data_misfit=np.zeros(profile.n_depths),
        n_starts=1,
        method="joint",
    )
    assert contact_precision(exact, profile, tolerance=0) == pytest.approx(1.0)


def test_a_flat_log_finds_nothing_real(profile):
    """A constant profile has no jumps; its picks are arbitrary, not contacts."""
    flat = InversionResult(
        params=np.repeat(profile.params[:1], profile.n_depths, axis=0),
        active_names=profile.active_names,
        data_misfit=np.zeros(profile.n_depths),
        n_starts=1,
        method="joint",
    )
    score = contact_precision(flat, profile, tolerance=0)
    assert 0.0 <= score <= 1.0


def test_no_skill_precision_is_the_reachable_bar(profile):
    """It must be a real fraction strictly inside (0, 1] on a bedded log."""
    bar = no_skill_contact_precision(profile, tolerance=1)
    assert 0.0 < bar <= 1.0
    # widening the tolerance can only make blind guessing easier
    assert no_skill_contact_precision(profile, tolerance=2) >= bar


def test_a_profile_without_contacts_has_nothing_to_score(bundle):
    single = synthesize_profile(
        bundle, n_depths=4, mean_bed_frames=500.0, noise_us_per_ft=0.0, seed=2
    )
    assert single.n_beds == 1
    exact = InversionResult(
        params=single.params.copy(),
        active_names=single.active_names,
        data_misfit=np.zeros(single.n_depths),
        n_starts=1,
        method="joint",
    )
    assert np.isnan(contact_precision(exact, single))
    assert np.isnan(no_skill_contact_precision(single))


def test_contact_metrics_reject_a_negative_tolerance(profile):
    exact = InversionResult(
        params=profile.params.copy(),
        active_names=profile.active_names,
        data_misfit=np.zeros(profile.n_depths),
        n_starts=1,
        method="joint",
    )
    with pytest.raises(ValueError, match="non-negative"):
        contact_precision(exact, profile, tolerance=-1)
    with pytest.raises(ValueError, match="non-negative"):
        no_skill_contact_precision(profile, tolerance=-1)


# ------------------------------------------------------------------
# select_lambda
# ------------------------------------------------------------------


def test_selection_returns_a_candidate_and_scores_all_of_them(surrogate, profile):
    grid = (0.0, 1e-3, 1e-1)
    picked = select_lambda(
        surrogate, profile.slowness, grid, steps=20, n_starts=1, seed=0
    )
    assert picked.lam in grid
    assert picked.candidates == grid
    assert picked.holdout_misfit.shape == (3,)
    assert np.all(np.isfinite(picked.holdout_misfit))
    assert picked.n_holdout > 0
    assert picked.selected_coupling == (picked.lam > 0.0)


def test_selection_cannot_see_the_truth(surrogate, profile):
    """Scrambling the true parameters must not change the chosen lambda.

    This is the property that makes the selection reportable: it is a function
    of the observed curves alone, so the lambda it picks is one an operator
    could have picked without knowing the answer.
    """
    grid = (0.0, 1e-3, 1e-1)
    kw = dict(steps=20, n_starts=1, seed=0)
    honest = select_lambda(surrogate, profile.slowness, grid, **kw)
    lied_to = dataclasses.replace(profile, params=profile.params[::-1] * 1.5)
    scrambled = select_lambda(surrogate, lied_to.slowness, grid, **kw)
    assert honest.lam == scrambled.lam
    np.testing.assert_array_equal(honest.holdout_misfit, scrambled.holdout_misfit)


# ------------------------------------------------------------------
# smooth_independent -- the control
# ------------------------------------------------------------------


def _fake(params: np.ndarray, names: tuple[str, ...]) -> InversionResult:
    return InversionResult(
        params=params,
        active_names=names,
        data_misfit=np.zeros(params.shape[0]),
        n_starts=1,
        method="surrogate",
    )


def test_unit_window_is_the_identity():
    """Width 1 must be a no-op, so it is a usable 'no smoothing' grid point."""
    values = np.array([[1.0, 9.0], [2.0, 8.0], [30.0, 7.0]])
    out = smooth_independent(_fake(values, ("vp", "vs")), 1)
    np.testing.assert_array_equal(out.params, values)
    assert out.method == "smoothed"
    assert out.active_names == ("vp", "vs")


def test_smoothing_averages_the_window_and_truncates_at_the_ends():
    """Edges are truncated, not padded --- no value outside the log is invented."""
    values = np.array([[0.0], [3.0], [6.0], [9.0], [12.0]])
    out = smooth_independent(_fake(values, ("vp",)), 3)
    # ends average 2 samples, interior averages 3
    np.testing.assert_allclose(out.params[:, 0], [1.5, 3.0, 6.0, 9.0, 10.5])


def test_smoothing_preserves_a_constant_log():
    """A truncated window must not bias the ends of an already-flat profile."""
    values = np.full((7, 2), 4.2)
    out = smooth_independent(_fake(values, ("vp", "vs")), 5)
    np.testing.assert_allclose(out.params, values)


def test_wider_windows_are_monotonically_smoother():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 1)) * 100.0

    def roughness(p):
        return float(np.abs(np.diff(p, axis=0)).sum())

    widths = [1, 3, 5, 9]
    scores = [
        roughness(smooth_independent(_fake(values, ("vp",)), w).params) for w in widths
    ]
    assert scores == sorted(scores, reverse=True)


def test_smoothing_rejects_a_degenerate_window():
    with pytest.raises(ValueError, match="at least 1"):
        smooth_independent(_fake(np.zeros((3, 1)), ("vp",)), 0)


def test_selection_rejects_bad_arguments(surrogate, profile):
    with pytest.raises(ValueError, match="at least one candidate"):
        select_lambda(surrogate, profile.slowness, (), steps=5)
    with pytest.raises(ValueError, match="holdout_frac"):
        select_lambda(surrogate, profile.slowness, (0.0,), holdout_frac=1.5, steps=5)


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


def _perfect(profile: DepthProfile, offset: float = 0.0) -> InversionResult:
    return InversionResult(
        params=profile.params + offset,
        active_names=profile.active_names,
        data_misfit=np.zeros(profile.n_depths),
        n_starts=1,
        method="joint",
    )


def test_profile_mae_matches_a_hand_computation(profile):
    assert profile_mae(_perfect(profile, 25.0), profile, "vs") == pytest.approx(25.0)
    assert profile_mae(_perfect(profile), profile, "vs") == pytest.approx(0.0)


def test_bed_and_boundary_errors_reconstruct_the_total(surrogate, profile):
    """The split must be a partition of the profile, not a re-weighting."""
    result = invert_joint(surrogate, profile.slowness, lam=1e-2, steps=30, n_starts=1)
    at_boundary = profile.boundary_mask(halo=1)
    assert at_boundary.any() and not at_boundary.all()
    for name in profile.active_names:
        in_bed, boundary = bed_vs_boundary_mae(result, profile, name)
        blended = (
            in_bed * (~at_boundary).sum() + boundary * at_boundary.sum()
        ) / profile.n_depths
        assert blended == pytest.approx(profile_mae(result, profile, name))


def test_single_bed_profile_has_no_boundary_to_score(bundle):
    quiet = synthesize_profile(
        bundle, n_depths=4, mean_bed_frames=500.0, noise_us_per_ft=0.0, seed=2
    )
    assert quiet.n_beds == 1
    in_bed, boundary = bed_vs_boundary_mae(_perfect(quiet, 10.0), quiet, "vs")
    assert in_bed == pytest.approx(10.0)
    assert np.isnan(boundary)
