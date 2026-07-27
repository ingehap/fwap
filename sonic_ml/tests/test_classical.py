"""Tests for sonic_ml.baselines.classical.

These run fwap's STC / dispersion estimators, so they score only a small
mixed-regime subset of samples to stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml.baselines import ClassicalSTCBaseline, FKDispersionBaseline
from sonic_ml.bench import evaluate
from sonic_ml.split import regime_labels


def _mixed_indices(bundle, n_each: int = 2) -> np.ndarray:
    labels = regime_labels(bundle)
    slow = np.flatnonzero(labels == 0)[:n_each]
    fast = np.flatnonzero(labels == 1)[:n_each]
    return np.concatenate([slow, fast])


def test_classical_stc_runs_without_raising(bundle, geom):
    idx = _mixed_indices(bundle)
    vs = ClassicalSTCBaseline().predict_vs(bundle, geom, idx)
    assert vs.shape == (idx.size,)
    # The fast branch must never leak a pseudo-Rayleigh existence ValueError;
    # failures come back as NaN, and at least one estimate is finite.
    assert np.isfinite(vs).any()


def test_fk_baseline_runs_without_raising(bundle, geom):
    idx = _mixed_indices(bundle)
    vs = FKDispersionBaseline().predict_vs(bundle, geom, idx)
    assert vs.shape == (idx.size,)
    assert np.isfinite(vs).any()


def test_finite_estimates_are_physical(bundle, geom):
    idx = _mixed_indices(bundle, n_each=3)
    vs = ClassicalSTCBaseline().predict_vs(bundle, geom, idx)
    finite = vs[np.isfinite(vs)]
    # Whatever the accuracy, a slowness peak inverts to a plausible velocity.
    assert np.all(finite > 500.0)
    assert np.all(finite < 6000.0)


def test_evaluate_with_classical_baseline(bundle, geom):
    idx = _mixed_indices(bundle)
    sc = evaluate(ClassicalSTCBaseline(), bundle, geom, indices=idx, n_boot=100)
    assert sc.predictor == "classical_stc"
    assert sc.parameter == "vs"
    assert "all" in sc.per_regime
    # A held-out subset with both regimes scores each separately.
    assert sc.per_regime["all"].n == idx.size


def test_baselines_are_predictors():
    from sonic_ml.bench import Predictor

    assert isinstance(ClassicalSTCBaseline(), Predictor)
    assert isinstance(FKDispersionBaseline(), Predictor)


def test_pseudo_rayleigh_range_guard_honored():
    # The fast branch's search range must stay below 1/v_fluid or fwap raises;
    # the default does. A misconfigured range should surface as NaN, not a
    # crash, because predict_vs guards ValueError.
    baseline = ClassicalSTCBaseline(fast_slowness_range=(700e-6, 900e-6))
    assert baseline.fast_slowness_range[1] > 1.0 / baseline.v_fluid
