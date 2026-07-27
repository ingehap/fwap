"""Tests for sonic_ml.mask."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml import mask
from sonic_ml.loader import load_npz


def test_finite_mask(npz_path):
    b = load_npz(npz_path)
    np.testing.assert_array_equal(mask.finite_mask(b.slowness), np.isfinite(b.slowness))


def test_presence_differs_from_isfinite_any():
    # A mode that is present but whose curve is entirely NaN, and a mode that
    # is absent but has one finite sample: isfinite(...).any() gets BOTH wrong,
    # mode_in_gather gets both right. This is the whole reason the two are kept
    # separate.
    slowness = np.array(
        [
            [[np.nan, np.nan, np.nan], [1e-4, np.nan, np.nan]],
        ]
    )  # (N=1, M=2, F=3)
    mode_in_gather = np.array([[True, False]])  # mode 0 present, mode 1 absent

    presence = mask.presence_labels(mode_in_gather)
    isfinite_any = np.isfinite(slowness).any(axis=-1)

    np.testing.assert_array_equal(presence, mode_in_gather)
    # The naive derivation disagrees on both modes.
    assert not np.array_equal(presence, isfinite_any)


def test_inverse_frequency_weights_balance():
    labels = np.array([0, 0, 0, 1])  # 3 slow, 1 fast
    w = mask.inverse_frequency_weights(labels)
    assert w.shape == (2,)
    # rarer class weighs more; weighted class sizes are equal
    assert w[1] > w[0]
    np.testing.assert_allclose(w[0] * 3, w[1] * 1)


def test_inverse_frequency_weights_num_classes_and_absent():
    labels = np.array([0, 0])
    w = mask.inverse_frequency_weights(labels, num_classes=3)
    assert w.shape == (3,)
    assert w[1] == 0.0 and w[2] == 0.0  # absent classes weigh 0


def test_inverse_frequency_weights_validates():
    with pytest.raises(ValueError):
        mask.inverse_frequency_weights(np.array([], dtype=int))
    with pytest.raises(ValueError):
        mask.inverse_frequency_weights(np.array([-1, 0]))


def test_masked_mean():
    vals = np.array([1.0, 2.0, np.nan])
    m = np.array([True, True, False])
    assert mask.masked_mean(vals, m) == 1.5
    assert np.isnan(mask.masked_mean(vals, np.zeros(3, dtype=bool)))
    with pytest.raises(ValueError):
        mask.masked_mean(vals, np.array([True, False]))
