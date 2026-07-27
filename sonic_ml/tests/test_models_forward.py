"""Tests for sonic_ml.models.forward (torch-gated)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.dataset import SlownessNormalizer  # noqa: E402
from sonic_ml.models.forward import (  # noqa: E402
    ForwardSurrogate,
    TrainedForwardSurrogate,
)
from sonic_ml.normalize import Standardizer  # noqa: E402


def test_forward_output_shapes():
    model = ForwardSurrogate(4, 2, 48, width=32, depth=2)
    x = torch.zeros(5, 4)
    slowness, presence = model(x)
    assert slowness.shape == (5, 2, 48)
    assert presence.shape == (5, 2)


def test_forward_is_deterministic_given_seed():
    from sonic_ml.determinism import seed_everything

    seed_everything(0)
    a = ForwardSurrogate(4, 2, 16, width=32, depth=2)
    seed_everything(0)
    b = ForwardSurrogate(4, 2, 16, width=32, depth=2)
    x = torch.randn(3, 4)
    with torch.no_grad():
        sa, _ = a(x)
        sb, _ = b(x)
    assert torch.equal(sa, sb)


def _trained(bundle) -> TrainedForwardSurrogate:
    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    slow_norm = SlownessNormalizer.fit(bundle.slowness)
    model = ForwardSurrogate(
        param_std.n_active, bundle.n_modes, bundle.n_freq, width=16, depth=1
    )
    return TrainedForwardSurrogate(model, param_std, slow_norm, bundle.mode_names)


def test_predict_shapes_and_finite(bundle):
    trained = _trained(bundle)
    slowness, presence = trained.predict(bundle.params[:4])
    assert slowness.shape == (4, bundle.n_modes, bundle.n_freq)
    assert presence.shape == (4, bundle.n_modes)
    assert np.isfinite(slowness).all()
    assert np.all((presence >= 0.0) & (presence <= 1.0))


def test_checkpoint_roundtrip(bundle, tmp_path):
    trained = _trained(bundle)
    ref, ref_p = trained.predict(bundle.params[:4])
    path = tmp_path / "ckpt.pt"
    trained.save(str(path))
    loaded = TrainedForwardSurrogate.load(str(path))
    got, got_p = loaded.predict(bundle.params[:4])
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)
    np.testing.assert_allclose(got_p, ref_p, rtol=0, atol=0)
    assert loaded.mode_names == trained.mode_names
    assert loaded.param_std.active_names == trained.param_std.active_names
