"""Tests for surrogate-in-the-loop inversion (torch-gated).

These pin *mechanism*, not accuracy. The CI fixture trains a surrogate on ~32
samples, so an accuracy assertion would measure the fixture rather than the
method; the measured accuracy comparison against the true-solver control lives
in the module docstring and the changelog, quoted with the dataset it came from.

What is asserted here is the part that must not silently break: that the search
actually reduces the data misfit, that the two routes agree on what they are
solving for, that results are reproducible, and that a caller cannot mistake one
parameter's column for another's.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.dataset import SlownessNormalizer  # noqa: E402
from sonic_ml.models.forward import (  # noqa: E402
    ForwardSurrogate,
    TrainedForwardSurrogate,
)
from sonic_ml.models.inversion import (  # noqa: E402
    InversionResult,
    inversion_mae,
    invert_with_solver,
    invert_with_surrogate,
    no_skill_mae,
)
from sonic_ml.models.train import train_forward  # noqa: E402
from sonic_ml.normalize import Standardizer  # noqa: E402
from sonic_ml.split import stratified_split  # noqa: E402


@pytest.fixture(scope="module")
def split(bundle):
    return stratified_split(bundle, seed=0)


@pytest.fixture(scope="module")
def surrogate(bundle, split):
    """A briefly-trained forward surrogate (module-scoped: training is the cost)."""
    trained, _ = train_forward(bundle, split=split, epochs=25, seed=0)
    return trained


@pytest.fixture(scope="module")
def observed(bundle, split):
    return bundle.slowness[split.test[:3]]


# ------------------------------------------------------------------
# InversionResult
# ------------------------------------------------------------------


def test_result_value_lookup_and_unknown_name():
    result = InversionResult(
        params=np.array([[1.0, 2.0], [3.0, 4.0]]),
        active_names=("vp", "vs"),
        data_misfit=np.array([0.1, 0.2]),
        n_starts=2,
        method="surrogate",
    )
    np.testing.assert_array_equal(result.value("vs"), [2.0, 4.0])
    with pytest.raises(KeyError, match="not inverted for"):
        result.value("rho")


# ------------------------------------------------------------------
# invert_with_surrogate
# ------------------------------------------------------------------


def test_surrogate_inversion_shapes_and_names(surrogate, observed):
    result = invert_with_surrogate(surrogate, observed, n_starts=2, steps=40)
    assert result.method == "surrogate"
    assert result.active_names == surrogate.param_std.active_names
    assert result.params.shape == (observed.shape[0], len(result.active_names))
    assert result.data_misfit.shape == (observed.shape[0],)
    assert np.all(np.isfinite(result.params))
    assert np.all(result.data_misfit >= 0.0)


def test_surrogate_inversion_reduces_the_data_misfit(surrogate, observed):
    """Optimising must actually fit the data better than the starting point.

    The starting point is the prior mean, so a search that achieved nothing
    would return this misfit unchanged.
    """
    from sonic_ml.models.inversion import invert_with_surrogate as invert

    barely = invert(surrogate, observed, n_starts=1, steps=1)
    properly = invert(surrogate, observed, n_starts=1, steps=200)
    assert np.median(properly.data_misfit) < np.median(barely.data_misfit)


def test_surrogate_inversion_is_reproducible(surrogate, observed):
    a = invert_with_surrogate(surrogate, observed, n_starts=2, steps=30, seed=0)
    b = invert_with_surrogate(surrogate, observed, n_starts=2, steps=30, seed=0)
    np.testing.assert_array_equal(a.params, b.params)
    np.testing.assert_array_equal(a.data_misfit, b.data_misfit)


def test_more_starts_never_worsens_the_misfit(surrogate, observed):
    """Extra starts are kept only when they fit better, so misfit is monotone."""
    one = invert_with_surrogate(surrogate, observed, n_starts=1, steps=60, seed=0)
    three = invert_with_surrogate(surrogate, observed, n_starts=3, steps=60, seed=0)
    assert np.all(three.data_misfit <= one.data_misfit + 1e-9)


def test_surrogate_inversion_rejects_wrong_shape(surrogate, bundle):
    with pytest.raises(ValueError, match=r"\(N, M, F\)"):
        invert_with_surrogate(surrogate, bundle.slowness[0], n_starts=1, steps=2)


def test_surrogate_inversion_ignores_absent_frequencies(surrogate, observed):
    """NaN entries are excluded from the misfit, not imputed as data."""
    blanked = observed.copy()
    blanked[:, :, -5:] = np.nan  # blank the top of the band
    result = invert_with_surrogate(surrogate, blanked, n_starts=1, steps=40)
    assert np.all(np.isfinite(result.data_misfit))
    assert np.all(np.isfinite(result.params))


# ------------------------------------------------------------------
# invert_with_solver -- the control
# ------------------------------------------------------------------


def test_solver_inversion_matches_the_surrogate_contract(bundle, split, surrogate):
    """Both routes must solve for the same parameters, or the comparison lies."""
    indices = split.test[:2]
    # Small budget: each residual evaluation runs the real modal solver.
    result = invert_with_solver(bundle, indices, n_starts=1, max_nfev=30)
    assert result.method == "solver"
    assert result.active_names == surrogate.param_std.active_names
    assert result.params.shape == (indices.size, len(result.active_names))
    assert np.all(np.isfinite(result.params))
    assert np.all(result.data_misfit >= 0.0)


def test_solver_inversion_respects_dataset_bounds(bundle, split):
    indices = split.test[:2]
    result = invert_with_solver(bundle, indices, n_starts=1, max_nfev=30)
    for name in result.active_names:
        column = bundle.param(name)
        recovered = result.value(name)
        assert np.all(recovered >= column.min() - 1e-9)
        assert np.all(recovered <= column.max() + 1e-9)


def test_solver_inversion_rejects_empty_active_set(bundle, split):
    with pytest.raises(ValueError, match="no active parameters"):
        invert_with_solver(bundle, split.test[:1], active_names=(), max_nfev=10)


# ------------------------------------------------------------------
# Scoring helpers
# ------------------------------------------------------------------


def test_inversion_mae_matches_a_hand_computation(bundle, split):
    indices = split.test[:3]
    truth = bundle.param("vs")[indices]
    result = InversionResult(
        params=(truth + 100.0)[:, None],
        active_names=("vs",),
        data_misfit=np.zeros(indices.size),
        n_starts=1,
        method="surrogate",
    )
    assert inversion_mae(result, bundle, indices, "vs") == pytest.approx(100.0)


def test_no_skill_mae_is_the_training_mean_baseline(bundle, split):
    reference = bundle.param("vs")[split.train].mean()
    expected = np.abs(bundle.param("vs")[split.test] - reference).mean()
    got = no_skill_mae(bundle, split.train, split.test, "vs")
    assert got == pytest.approx(expected)


def test_no_skill_mae_is_a_real_bar_not_a_formality(bundle, split):
    """It must be finite and positive, or 'beats no-skill' means nothing."""
    for name in ("vp", "vs", "rho"):
        value = no_skill_mae(bundle, split.train, split.test, name)
        assert np.isfinite(value) and value > 0.0


# ------------------------------------------------------------------
# The surrogate must be the thing being inverted
# ------------------------------------------------------------------


def test_inversion_depends_on_the_trained_weights(bundle, split, surrogate, observed):
    """An untrained surrogate must give a different answer.

    If it did not, the result would be coming from the initialisation or the
    normalisers rather than from the learned forward model.
    """
    param_std = Standardizer.fit(bundle.params[split.train], bundle.param_names)
    untrained = TrainedForwardSurrogate(
        ForwardSurrogate(
            param_std.n_active, bundle.n_modes, bundle.n_freq, width=16, depth=1
        ),
        param_std,
        SlownessNormalizer.fit(bundle.slowness[split.train]),
        bundle.mode_names,
    )
    from_trained = invert_with_surrogate(surrogate, observed, n_starts=1, steps=50)
    from_untrained = invert_with_surrogate(untrained, observed, n_starts=1, steps=50)
    assert not np.allclose(from_trained.params, from_untrained.params)
