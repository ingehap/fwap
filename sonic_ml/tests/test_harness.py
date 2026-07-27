"""Tests for sonic_ml.bench.harness."""

from __future__ import annotations

import numpy as np

from sonic_ml.bench import Predictor, Scorecard, StubPredictor, evaluate


def test_perfect_predictor_scores_zero_error(bundle, geom):
    sc = evaluate(StubPredictor(None), bundle, geom, seed=0, n_boot=200)
    assert isinstance(sc, Scorecard)
    assert sc.predictor == "stub"
    assert sc.parameter == "vs"
    assert sc.per_regime["all"].median_abs_error == 0.0
    assert sc.per_regime["all"].n_finite == bundle.n_samples


def test_constant_predictor_error_matches_hand_computation(bundle, geom):
    const = 2000.0
    sc = evaluate(StubPredictor(const), bundle, geom, seed=0, n_boot=200)
    expected = float(np.median(np.abs(const - bundle.param("vs"))))
    assert sc.per_regime["all"].median_abs_error == expected


def test_regimes_present_and_partition(bundle, geom):
    sc = evaluate(StubPredictor(2500.0), bundle, geom, seed=0, n_boot=100)
    # "all" is always present; slow/fast present iff that regime has samples.
    assert "all" in sc.per_regime
    regime_n = sum(sc.per_regime[r].n for r in ("slow", "fast") if r in sc.per_regime)
    assert regime_n == sc.per_regime["all"].n == bundle.n_samples


def test_bootstrap_ci_is_reproducible(bundle, geom):
    a = evaluate(StubPredictor(2200.0), bundle, geom, seed=7, n_boot=200)
    b = evaluate(StubPredictor(2200.0), bundle, geom, seed=7, n_boot=200)
    for name in a.per_regime:
        assert a.per_regime[name].ci_low == b.per_regime[name].ci_low
        assert a.per_regime[name].ci_high == b.per_regime[name].ci_high
    # CI brackets the point estimate.
    r = a.per_regime["all"]
    assert r.ci_low <= r.median_abs_error <= r.ci_high


def test_indices_subset_scored(bundle, geom):
    idx = np.arange(5)
    sc = evaluate(StubPredictor(None), bundle, geom, indices=idx, n_boot=50)
    assert sc.per_regime["all"].n == 5


def test_nan_predictions_counted_not_crashed(bundle, geom):
    class AllNaN:
        name = "all_nan"

        def predict_vs(self, bundle, geom, indices):
            return np.full(np.asarray(indices).size, np.nan)

    pred = AllNaN()
    assert isinstance(pred, Predictor)  # runtime-checkable protocol
    sc = evaluate(pred, bundle, geom, n_boot=50)
    assert sc.per_regime["all"].n_finite == 0
    assert np.isnan(sc.per_regime["all"].median_abs_error)
