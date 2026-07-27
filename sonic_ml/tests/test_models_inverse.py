"""Tests for sonic_ml.models.inverse / inverse_dataset (torch-gated)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.inverse import InverseNet, TrainedInverseNet  # noqa: E402
from sonic_ml.models.inverse_dataset import (  # noqa: E402
    InverseDataset,
    normalize_gathers,
)
from sonic_ml.normalize import Standardizer  # noqa: E402


def test_normalize_gathers_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(4, 8, 256)) * 5.0 + 3.0
    z = normalize_gathers(g)
    assert z.shape == g.shape
    for i in range(g.shape[0]):
        assert abs(z[i].mean()) < 1e-6
        assert abs(z[i].std() - 1.0) < 1e-6


def test_inverse_dataset_items(bundle):
    idx = np.arange(bundle.n_samples)
    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    ds = InverseDataset(bundle, idx, param_std)
    assert len(ds) == bundle.n_samples
    x, y = ds[0]
    assert x.shape == (bundle.gather.shape[1], bundle.gather.shape[2])
    assert y.shape == (param_std.n_active,)
    assert x.dtype == torch.float32


def test_inverse_dataset_does_not_use_slowness(bundle):
    # The inverse input must be gather-only; slowness must not leak in.
    import dataclasses

    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    idx = np.arange(bundle.n_samples)
    ref = InverseDataset(bundle, idx, param_std).x
    garbled = dataclasses.replace(bundle, slowness=np.zeros_like(bundle.slowness))
    got = InverseDataset(garbled, idx, param_std).x
    assert torch.equal(ref, got)


def test_inverse_net_output_shapes():
    model = InverseNet(8, 4, channels=(8, 16), kernel=5)
    x = torch.zeros(3, 8, 512)
    mean, log_var = model(x)
    assert mean.shape == (3, 4)
    assert log_var.shape == (3, 4)


def test_inverse_net_deterministic_given_seed():
    from sonic_ml.determinism import seed_everything

    seed_everything(0)
    a = InverseNet(8, 4, channels=(8, 16))
    seed_everything(0)
    b = InverseNet(8, 4, channels=(8, 16))
    x = torch.randn(2, 8, 512)
    a.eval()
    b.eval()
    with torch.no_grad():
        ma, _ = a(x)
        mb, _ = b(x)
    assert torch.equal(ma, mb)


def _trained(bundle) -> TrainedInverseNet:
    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    model = InverseNet(bundle.gather.shape[1], param_std.n_active, channels=(8, 16))
    return TrainedInverseNet(model, param_std)


def test_predict_shapes_and_positive_std(bundle):
    trained = _trained(bundle)
    params, std = trained.predict(bundle.gather[:4])
    assert params.shape == (4, trained.param_std.n_active)
    assert std.shape == (4, trained.param_std.n_active)
    assert np.all(np.isfinite(params))
    assert np.all(std > 0.0)


def test_checkpoint_roundtrip(bundle, tmp_path):
    trained = _trained(bundle)
    ref_p, ref_s = trained.predict(bundle.gather[:4])
    path = tmp_path / "inv.pt"
    trained.save(str(path))
    loaded = TrainedInverseNet.load(str(path))
    got_p, got_s = loaded.predict(bundle.gather[:4])
    np.testing.assert_allclose(got_p, ref_p, rtol=0, atol=0)
    np.testing.assert_allclose(got_s, ref_s, rtol=0, atol=0)
    assert loaded.active_names == trained.active_names
