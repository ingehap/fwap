"""Tests for sonic_ml.models.augment and its dataset/training integration."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.augment import GatherAugmentation  # noqa: E402
from sonic_ml.models.inverse_dataset import InverseDataset  # noqa: E402
from sonic_ml.models.inverse_train import train_inverse  # noqa: E402
from sonic_ml.normalize import Standardizer  # noqa: E402
from sonic_ml.split import stratified_split  # noqa: E402


def test_apply_is_reproducible_and_changes_gather():
    g = np.random.default_rng(0).normal(size=(8, 256))
    aug = GatherAugmentation(snr_db_range=(10.0, 20.0), amp_jitter=0.1)
    a = aug.apply(g, np.random.default_rng(1))
    b = aug.apply(g, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)  # same rng -> identical
    assert not np.allclose(a, g)  # actually augmented
    assert a.shape == g.shape


def test_apply_is_identity_when_disabled():
    g = np.random.default_rng(0).normal(size=(4, 64))
    aug = GatherAugmentation(snr_db_range=None, amp_jitter=0.0)
    np.testing.assert_array_equal(aug.apply(g, np.random.default_rng(0)), g)


def test_snr_is_roughly_achieved():
    g = np.sin(np.linspace(0.0, 50.0, 2048))[None, :].repeat(8, axis=0)
    aug = GatherAugmentation(snr_db_range=(20.0, 20.0), amp_jitter=0.0)  # fixed 20 dB
    out = aug.apply(g, np.random.default_rng(0))
    noise = out - g
    snr = 10.0 * np.log10((g**2).mean() / (noise**2).mean())
    assert abs(snr - 20.0) < 2.0


def test_dataset_augment_is_stochastic_but_preserves_clean_x(bundle):
    ps = Standardizer.fit(bundle.params, bundle.param_names)
    idx = np.arange(bundle.n_samples)
    aug = GatherAugmentation(snr_db_range=(5.0, 15.0))
    ds = InverseDataset(bundle, idx, ps, augment=aug, augment_seed=0)
    x_a, _ = ds[0]
    x_b, _ = ds[0]
    assert not torch.equal(x_a, x_b)  # stochastic per access
    # the precomputed non-augmented view still matches a plain dataset
    plain = InverseDataset(bundle, idx, ps)
    assert torch.equal(ds.x, plain.x)


def test_train_with_augment_runs_and_is_reproducible(bundle):
    split = stratified_split(bundle, seed=0)
    aug = GatherAugmentation(snr_db_range=(5.0, 25.0))
    a, ha = train_inverse(
        bundle, split=split, epochs=5, channels=(8, 16), seed=1, augment=aug
    )
    b, hb = train_inverse(
        bundle, split=split, epochs=5, channels=(8, 16), seed=1, augment=aug
    )
    np.testing.assert_allclose(ha, hb, rtol=0, atol=1e-6)
    pa, _ = a.predict(bundle.gather[:4])
    pb, _ = b.predict(bundle.gather[:4])
    np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-6)
