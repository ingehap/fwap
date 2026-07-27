"""Tests for sonic_ml.models.lwd -- the low-latency LWD variant (torch-gated)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.inverse import InverseNet, TrainedInverseNet  # noqa: E402
from sonic_ml.models.inverse_train import InversePredictor  # noqa: E402
from sonic_ml.models.lwd import (  # noqa: E402
    LWD_COMPACT_CHANNELS,
    LatencyAccuracy,
    build_lwd_inverse_net,
    count_parameters,
    format_latency_accuracy,
    latency_accuracy_report,
    measure_latency_ms,
    train_lwd_inverse,
)
from sonic_ml.normalize import Standardizer  # noqa: E402

# ------------------------------------------------------------------
# Depthwise-separable conv block
# ------------------------------------------------------------------


def test_separable_block_reduces_params():
    # Same architecture, only the block type differs: the separable variant
    # must have strictly fewer parameters.
    dense = InverseNet(8, 4, channels=(16, 32), kernel=5, hidden=32, separable=False)
    sep = InverseNet(8, 4, channels=(16, 32), kernel=5, hidden=32, separable=True)
    assert count_parameters(sep) < count_parameters(dense)
    assert dense.hparams["separable"] is False
    assert sep.hparams["separable"] is True


def test_separable_net_output_shapes():
    model = build_lwd_inverse_net(8, 4)
    x = torch.zeros(3, 8, 512)
    mean, log_var = model(x)
    assert mean.shape == (3, 4)
    assert log_var.shape == (3, 4)


def test_separable_net_accepts_truncated_record():
    # AdaptiveAvgPool over the time axis makes the net length-agnostic, so a
    # shorter (lower-latency) record still produces the right output shape.
    model = build_lwd_inverse_net(8, 4)
    for t in (128, 512, 2048):
        mean, log_var = model(torch.zeros(2, 8, t))
        assert mean.shape == (2, 4) and log_var.shape == (2, 4)


# ------------------------------------------------------------------
# Compact preset
# ------------------------------------------------------------------


def test_build_lwd_inverse_net_is_compact():
    full = InverseNet(8, 4)  # full default (32, 64, 128) dense
    compact = build_lwd_inverse_net(8, 4)
    assert count_parameters(compact) < count_parameters(full)
    assert compact.hparams["separable"] is True
    assert tuple(compact.hparams["channels"]) == LWD_COMPACT_CHANNELS


# ------------------------------------------------------------------
# Checkpoint round-trip + backward compatibility
# ------------------------------------------------------------------


def _param_std(bundle) -> Standardizer:
    return Standardizer.fit(bundle.params, bundle.param_names)


def _held_out_idx(bundle) -> np.ndarray:
    """A small held-out index slice for scoring (last few samples)."""
    return np.arange(bundle.n_samples)[-5:]


def test_separable_checkpoint_roundtrips(bundle, tmp_path):
    param_std = _param_std(bundle)
    trained = TrainedInverseNet(
        build_lwd_inverse_net(bundle.gather.shape[1], param_std.n_active), param_std
    )
    ref_p, ref_s = trained.predict(bundle.gather[:3])
    path = tmp_path / "lwd.pt"
    trained.save(str(path))
    loaded = TrainedInverseNet.load(str(path))
    assert loaded.model.hparams["separable"] is True
    got_p, got_s = loaded.predict(bundle.gather[:3])
    np.testing.assert_allclose(got_p, ref_p, rtol=0, atol=0)
    np.testing.assert_allclose(got_s, ref_s, rtol=0, atol=0)


def test_legacy_dense_checkpoint_loads_without_separable_key(bundle, tmp_path):
    # A pre-separable checkpoint is a dense model whose hparams lack the key.
    # Loading it must default to a dense net (no error) and predict identically.
    param_std = _param_std(bundle)
    dense = TrainedInverseNet(
        InverseNet(bundle.gather.shape[1], param_std.n_active, channels=(16, 32)),
        param_std,
    )
    path = tmp_path / "dense.pt"
    dense.save(str(path))
    ref_p, _ = dense.predict(bundle.gather[:3])

    ckpt = torch.load(str(path), weights_only=True)
    del ckpt["hparams"]["separable"]  # simulate a pre-separable checkpoint
    legacy_path = tmp_path / "legacy.pt"
    torch.save(ckpt, legacy_path)

    legacy = TrainedInverseNet.load(str(legacy_path))
    assert legacy.model.hparams["separable"] is False
    got_p, _ = legacy.predict(bundle.gather[:3])
    np.testing.assert_allclose(got_p, ref_p, rtol=0, atol=0)


# ------------------------------------------------------------------
# Training + prediction
# ------------------------------------------------------------------


def test_train_lwd_inverse_trains_and_predicts(bundle):
    trained, history = train_lwd_inverse(bundle, epochs=3, seed=0)
    assert len(history) == 3
    assert all(np.isfinite(h) for h in history)
    assert trained.model.hparams["separable"] is True
    # Scored through the same M1 predictor adapter as every other model.
    pred = InversePredictor(trained)
    vs = pred.predict_vs(bundle, object(), np.arange(4))  # geom unused by the net
    assert vs.shape == (4,)
    assert np.all(np.isfinite(vs))


def test_train_lwd_inverse_reproducible(bundle):
    a, ha = train_lwd_inverse(bundle, epochs=2, seed=0)
    b, hb = train_lwd_inverse(bundle, epochs=2, seed=0)
    assert ha == hb
    pa, _ = a.predict(bundle.gather[:3])
    pb, _ = b.predict(bundle.gather[:3])
    np.testing.assert_allclose(pa, pb, rtol=0, atol=0)


# ------------------------------------------------------------------
# Benchmark utilities
# ------------------------------------------------------------------


def test_count_parameters_matches_manual():
    model = InverseNet(4, 2, channels=(8,), kernel=3, hidden=8)
    manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count_parameters(model) == int(manual)


def test_measure_latency_positive(bundle):
    param_std = _param_std(bundle)
    trained = TrainedInverseNet(
        build_lwd_inverse_net(bundle.gather.shape[1], param_std.n_active), param_std
    )
    lat = measure_latency_ms(trained, bundle.gather, repeats=5, warmup=1)
    assert np.isfinite(lat) and lat > 0.0


def test_latency_accuracy_report_rows_and_format(bundle):
    param_std = _param_std(bundle)
    n_rec = bundle.gather.shape[1]
    full = TrainedInverseNet(InverseNet(n_rec, param_std.n_active), param_std)
    compact = TrainedInverseNet(
        build_lwd_inverse_net(n_rec, param_std.n_active), param_std
    )
    rows = latency_accuracy_report(
        {"full": full, "lwd_compact": compact},
        bundle,
        _held_out_idx(bundle),
        repeats=5,
    )
    assert [r.name for r in rows] == ["full", "lwd_compact"]
    assert all(isinstance(r, LatencyAccuracy) for r in rows)
    for r in rows:
        assert r.n_params > 0
        assert np.isfinite(r.latency_ms_per_sample) and r.latency_ms_per_sample > 0
        assert np.isfinite(r.vs_mae_m_s)
    # The compact net has strictly fewer parameters than the full default net
    # (a deterministic, hardware-independent claim -- unlike wall-clock latency).
    params = {r.name: r.n_params for r in rows}
    assert params["lwd_compact"] < params["full"]

    table = format_latency_accuracy(rows)
    assert "lwd_compact" in table and "vs MAE" in table
