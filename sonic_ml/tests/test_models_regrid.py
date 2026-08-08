"""Tests for M5f: re-gridding evaluation and casing-ring augmentation.

The re-gridding tests exist to keep an easy claim honest. M5c reports that the
operator is *self-consistent* under re-gridding; these check the quantity that
actually matters -- accuracy against the solver on a grid the model never saw --
and pin that the interpolation control is computed alongside it, so
"the operator can be evaluated at new frequencies" can never be reported without
the number that can refute its usefulness.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.augment import (  # noqa: E402
    Augmentation,
    CasingRingAugmentation,
    GatherAugmentation,
)
from sonic_ml.models.cased_inverse import (  # noqa: E402
    cased_target_mae,
    train_cased_inverse,
)
from sonic_ml.models.cased_train import train_cased_operator  # noqa: E402
from sonic_ml.models.regrid import (  # noqa: E402
    RegridScore,
    evaluate_regridding,
    format_regrid_score,
    true_curves_on_grid,
)
from sonic_ml.split import stratified_split  # noqa: E402


@pytest.fixture(scope="module")
def cased_split(cased_bundle):
    return stratified_split(cased_bundle, seed=0)


@pytest.fixture(scope="module")
def operator(cased_bundle, cased_split):
    trained, _ = train_cased_operator(
        cased_bundle, split=cased_split, epochs=20, seed=0
    )
    return trained


# ------------------------------------------------------------------
# true_curves_on_grid -- the ground truth that makes off-grid claims checkable
# ------------------------------------------------------------------


def test_true_curves_reproduce_the_stored_labels_exactly(cased_bundle):
    """On the dataset's own grid the reconstruction must be bit-identical.

    The generator produced ``bundle.slowness`` with the same solver and the same
    stored inputs, so anything other than exact equality means the solver call is
    being rebuilt wrongly -- and every off-grid number would inherit that error.
    """
    idx = np.arange(min(4, cased_bundle.n_samples))
    got = true_curves_on_grid(cased_bundle, idx, cased_bundle.freq)
    np.testing.assert_array_equal(got, cased_bundle.slowness[idx])


def test_true_curves_shape_on_an_unseen_grid(cased_bundle):
    idx = np.arange(3)
    fine = np.linspace(cased_bundle.freq[0], cased_bundle.freq[-1], 97)
    got = true_curves_on_grid(cased_bundle, idx, fine)
    assert got.shape == (3, 1, 97)
    assert np.isfinite(got).any()


def test_true_curves_rejects_open_hole(bundle):
    with pytest.raises(ValueError, match="cased-hole"):
        true_curves_on_grid(bundle, np.arange(2), bundle.freq)


def test_true_curves_rejects_multi_mode(cased_bundle):
    two_mode = dataclasses.replace(cased_bundle, mode_names=("Stoneley", "flexural"))
    with pytest.raises(ValueError, match="single-mode"):
        true_curves_on_grid(two_mode, np.arange(2), cased_bundle.freq)


# ------------------------------------------------------------------
# evaluate_regridding
# ------------------------------------------------------------------


def test_evaluate_regridding_reports_all_three_numbers(
    operator, cased_bundle, cased_split
):
    fine = np.linspace(
        cased_bundle.freq[0], cased_bundle.freq[-1], 2 * cased_bundle.n_freq - 1
    )
    score = evaluate_regridding(operator, cased_bundle, cased_split.test, fine)
    assert isinstance(score, RegridScore)
    assert score.n_train_grid == cased_bundle.n_freq
    assert score.n_eval_grid == fine.size
    for value in (score.training_grid_mae, score.zero_shot_mae, score.interpolated_mae):
        assert np.isfinite(value) and value >= 0.0


def test_evaluate_regridding_on_the_training_grid_matches_its_own_baseline(
    operator, cased_bundle, cased_split
):
    """Evaluated on the training grid, zero-shot and interpolation coincide."""
    score = evaluate_regridding(
        operator, cased_bundle, cased_split.test, cased_bundle.freq
    )
    assert score.zero_shot_mae == pytest.approx(score.training_grid_mae, rel=1e-9)
    assert score.interpolated_mae == pytest.approx(score.training_grid_mae, rel=1e-9)


def test_regrid_score_verdict_property_and_formatting():
    beats = RegridScore(24, 93, 0.25, 0.10, 0.30)
    loses = RegridScore(24, 93, 0.25, 0.35, 0.25)
    assert beats.zero_shot_beats_interpolation
    assert not loses.zero_shot_beats_interpolation
    assert "beat the interpolation control" in format_regrid_score(beats)
    assert "did NOT beat" in format_regrid_score(loses)
    # The control is always printed, whichever way the verdict falls.
    for score in (beats, loses):
        assert "(control)" in format_regrid_score(score)


# ------------------------------------------------------------------
# CasingRingAugmentation
# ------------------------------------------------------------------


def test_casing_ring_conforms_to_the_augmentation_protocol():
    assert isinstance(CasingRingAugmentation(), Augmentation)
    assert isinstance(GatherAugmentation(), Augmentation)  # unchanged by M5f


def test_casing_ring_adds_energy_without_changing_shape(cased_bundle):
    ring = CasingRingAugmentation()
    g = cased_bundle.gather[0]
    out = ring.apply(g, np.random.default_rng(0))
    assert out.shape == g.shape
    assert np.all(np.isfinite(out))
    assert not np.allclose(out, g)


def test_casing_ring_is_reproducible_given_a_seed(cased_bundle):
    ring = CasingRingAugmentation()
    g = cased_bundle.gather[0]
    a = ring.apply(g, np.random.default_rng(7))
    b = ring.apply(g, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_casing_ring_amplitude_is_independent_of_the_sample(cased_bundle):
    """The ring carries no bond information -- amplitude comes from the RNG only.

    Coupling ring amplitude to the bond index would manufacture a CBL-like
    signal, and a model 'recovering' bond from it would only be recovering a
    hard-coded relationship. This pins the design decision.
    """
    ring = CasingRingAugmentation(amplitude_range=(1.0, 1.0))  # deterministic
    # Same RNG draw on two different gathers scales with each gather's own RMS
    # and nothing else, so the injected ring is identical up to that factor.
    residuals = []
    for i in (0, 1):
        g = cased_bundle.gather[i]
        added = ring.apply(g, np.random.default_rng(0)) - g
        rms = float(np.sqrt(np.mean(g**2)))
        residuals.append(added / rms)
    np.testing.assert_allclose(residuals[0], residuals[1], rtol=1e-9, atol=1e-12)


def test_casing_ring_arrives_before_the_stoneley(cased_bundle):
    """The ring is a fast, early arrival -- well separated from the Stoneley."""
    ring = CasingRingAugmentation()
    g = np.zeros_like(cased_bundle.gather[0])
    added = ring.apply(g, np.random.default_rng(0))
    # With a zero input the RMS is ~0, so scale is degenerate; check timing only.
    peak_sample = int(np.argmax(np.abs(added[0])))
    expected = ring.tr_offset * ring.slowness / ring.dt
    assert abs(peak_sample - expected) < 5


def test_casing_ring_trains_through_the_cased_inverse(cased_bundle, cased_split):
    """It drops into the existing `augment=` slot with no dataset changes."""
    trained, history = train_cased_inverse(
        cased_bundle,
        split=cased_split,
        epochs=3,
        augment=CasingRingAugmentation(),
        seed=0,
    )
    assert len(history) == 3
    assert all(np.isfinite(h) for h in history)
    mae = cased_target_mae(trained, cased_bundle, cased_split.test, "bond_index")
    assert np.isfinite(mae)
