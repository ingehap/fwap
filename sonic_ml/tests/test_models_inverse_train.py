"""Tests for sonic_ml.models.inverse_train (torch-gated)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.bench import Predictor, Scorecard, evaluate  # noqa: E402
from sonic_ml.geometry import default_geometry  # noqa: E402
from sonic_ml.models.inverse_train import (  # noqa: E402
    InversePredictor,
    main,
    parameter_mae,
    residual_zscore_std,
    train_inverse,
)
from sonic_ml.split import DataSplit, stratified_split  # noqa: E402


def test_train_runs_and_nll_decreases(bundle):
    split = stratified_split(bundle, seed=0)
    trained, history = train_inverse(
        bundle, split=split, epochs=10, channels=(8, 16), batch_size=16, seed=0
    )
    assert len(history) == 10
    assert history[-1] < history[0]


def test_overfit_a_batch(bundle):
    idx = np.arange(6)
    split = DataSplit(train=idx, val=idx, test=idx[:0])
    trained, history = train_inverse(
        bundle,
        split=split,
        epochs=150,
        channels=(16, 32),
        batch_size=6,
        lr=2e-3,
        seed=0,
    )
    assert history[-1] < history[0]
    # The net memorizes Vs on the tiny batch to well under the prior spread.
    assert parameter_mae(trained, bundle, idx, "vs") < 150.0


def test_inverse_predictor_plugs_into_harness(bundle):
    split = stratified_split(bundle, seed=0)
    trained, _ = train_inverse(bundle, split=split, epochs=6, channels=(8, 16), seed=0)
    predictor = InversePredictor(trained)
    assert isinstance(predictor, Predictor)  # satisfies the M1 protocol
    geom = default_geometry(bundle)
    sc = evaluate(predictor, bundle, geom, indices=split.val, n_boot=100)
    assert isinstance(sc, Scorecard)
    assert sc.predictor == "inverse_net"
    assert sc.per_regime["all"].n == split.val.size


def test_anti_leakage_prediction_ignores_slowness(bundle):
    # The inverse net must never use the dispersion label: mutating slowness
    # to garbage leaves its Vs prediction unchanged.
    split = stratified_split(bundle, seed=0)
    trained, _ = train_inverse(bundle, split=split, epochs=4, channels=(8, 16), seed=0)
    predictor = InversePredictor(trained)
    geom = default_geometry(bundle)
    idx = split.val
    ref = predictor.predict_vs(bundle, geom, idx)
    garbled = dataclasses.replace(
        bundle, slowness=np.full_like(bundle.slowness, np.nan)
    )
    got = predictor.predict_vs(garbled, geom, idx)
    np.testing.assert_array_equal(ref, got)


def test_training_is_reproducible(bundle):
    split = stratified_split(bundle, seed=0)
    a, ha = train_inverse(bundle, split=split, epochs=5, channels=(8, 16), seed=1)
    b, hb = train_inverse(bundle, split=split, epochs=5, channels=(8, 16), seed=1)
    np.testing.assert_allclose(ha, hb, rtol=0, atol=1e-6)
    pa, _ = a.predict(bundle.gather[:4])
    pb, _ = b.predict(bundle.gather[:4])
    np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-6)


def test_metrics_finite(bundle):
    split = stratified_split(bundle, seed=0)
    trained, _ = train_inverse(bundle, split=split, epochs=5, channels=(8, 16), seed=0)
    val = split.val if split.val.size else split.train
    assert np.isfinite(parameter_mae(trained, bundle, val, "vs"))
    assert np.isfinite(residual_zscore_std(trained, bundle, val, "vs"))


def test_cli_writes_checkpoint(tmp_path, npz_path):
    out = tmp_path / "inv.pt"
    rc = main(
        [
            "--npz",
            npz_path,
            "--out",
            str(out),
            "--epochs",
            "3",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    assert out.exists()
    from sonic_ml.models.inverse import TrainedInverseNet

    TrainedInverseNet.load(str(out))
