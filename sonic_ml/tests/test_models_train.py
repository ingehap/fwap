"""Tests for sonic_ml.models.train (torch-gated)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.train import (  # noqa: E402
    main,
    presence_auc,
    slowness_rmse,
    train_forward,
)
from sonic_ml.split import stratified_split  # noqa: E402


def test_train_runs_and_loss_decreases(bundle):
    split = stratified_split(bundle, seed=0)
    trained, history = train_forward(
        bundle, split=split, epochs=12, width=32, depth=2, batch_size=16, seed=0
    )
    assert len(history) == 12
    # Training loss should fall meaningfully over the run.
    assert history[-1] < history[0]


def test_overfit_a_batch(bundle):
    # A model with enough steps must drive the loss on a handful of samples
    # near zero -- proof the training loop actually learns.
    from sonic_ml.split import DataSplit

    idx = np.arange(4)
    split = DataSplit(train=idx, val=idx, test=idx[:0])
    trained, history = train_forward(
        bundle,
        split=split,
        epochs=250,
        width=64,
        depth=2,
        batch_size=4,
        lr=2e-3,
        seed=0,
    )
    assert history[-1] < 0.2 * history[0]
    rmse = slowness_rmse(trained, bundle, idx)
    assert rmse < 5e-5  # s/m; near the solver on the memorized batch


def test_metrics_are_finite_and_bounded(bundle):
    split = stratified_split(bundle, seed=0)
    trained, _ = train_forward(bundle, split=split, epochs=8, width=32, depth=2, seed=0)
    val = split.val if split.val.size else split.train
    rmse = slowness_rmse(trained, bundle, val)
    auc = presence_auc(trained, bundle, val)
    assert rmse >= 0.0 and np.isfinite(rmse)
    assert np.isnan(auc) or (0.0 <= auc <= 1.0)


def test_training_is_reproducible(bundle):
    split = stratified_split(bundle, seed=0)
    a, ha = train_forward(bundle, split=split, epochs=5, width=32, depth=2, seed=1)
    b, hb = train_forward(bundle, split=split, epochs=5, width=32, depth=2, seed=1)
    np.testing.assert_allclose(ha, hb, rtol=0, atol=1e-6)
    pa, _ = a.predict(bundle.params[:4])
    pb, _ = b.predict(bundle.params[:4])
    np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-6)


def test_cli_writes_checkpoint(tmp_path, npz_path):
    out = tmp_path / "m.pt"
    rc = main(
        [
            "--npz",
            npz_path,
            "--out",
            str(out),
            "--epochs",
            "3",
            "--width",
            "16",
            "--depth",
            "1",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    assert out.exists()
    # Loads back as a usable surrogate.
    from sonic_ml.models.forward import TrainedForwardSurrogate

    TrainedForwardSurrogate.load(str(out))
