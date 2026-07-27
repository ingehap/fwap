"""Tests for sonic_ml.models.dataset (torch-gated)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.dataset import ForwardDataset, SlownessNormalizer  # noqa: E402
from sonic_ml.normalize import Standardizer  # noqa: E402


def test_slowness_normalizer_roundtrip(bundle):
    norm = SlownessNormalizer.fit(bundle.slowness)
    assert norm.mean.shape == (bundle.n_modes,)
    assert norm.std.shape == (bundle.n_modes,)
    z = norm.transform(bundle.slowness)
    back = norm.inverse(z)
    finite = np.isfinite(bundle.slowness)
    np.testing.assert_allclose(back[finite], bundle.slowness[finite], rtol=1e-9)


def test_slowness_normalizer_standardizes_finite_entries(bundle):
    norm = SlownessNormalizer.fit(bundle.slowness)
    z = norm.transform(bundle.slowness)
    for m in range(bundle.n_modes):
        col = z[:, m, :][np.isfinite(z[:, m, :])]
        if col.size > 1:
            assert abs(col.mean()) < 1e-6
            assert abs(col.std() - 1.0) < 1e-6


def test_forward_dataset_items(bundle):
    idx = np.arange(bundle.n_samples)
    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    slow_norm = SlownessNormalizer.fit(bundle.slowness)
    ds = ForwardDataset(bundle, idx, param_std, slow_norm)
    assert len(ds) == bundle.n_samples
    x, target, mask, presence = ds[0]
    assert x.shape == (param_std.n_active,)
    assert target.shape == (bundle.n_modes, bundle.n_freq)
    assert mask.shape == (bundle.n_modes, bundle.n_freq)
    assert presence.shape == (bundle.n_modes,)
    assert x.dtype == torch.float32
    assert mask.dtype == torch.bool
    # Targets are finite everywhere (masked-out NaNs were zeroed).
    assert bool(torch.isfinite(target).all())


def test_forward_dataset_mask_matches_finite(bundle):
    idx = np.arange(bundle.n_samples)
    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    slow_norm = SlownessNormalizer.fit(bundle.slowness)
    ds = ForwardDataset(bundle, idx, param_std, slow_norm)
    mask = ds.mask.numpy()
    np.testing.assert_array_equal(mask, np.isfinite(bundle.slowness))
