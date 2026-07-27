"""Tests for sonic_ml.split."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml.loader import load_npz
from sonic_ml.split import DataSplit, regime_labels, stratified_split


def test_split_is_a_partition(npz_path):
    b = load_npz(npz_path)
    sp = stratified_split(b, seed=0)
    allidx = np.concatenate([sp.train, sp.val, sp.test])
    np.testing.assert_array_equal(np.sort(allidx), np.arange(b.n_samples))
    # disjoint
    assert len(set(allidx.tolist())) == b.n_samples


def test_split_is_reproducible(npz_path):
    b = load_npz(npz_path)
    a = stratified_split(b, seed=3)
    c = stratified_split(b, seed=3)
    np.testing.assert_array_equal(a.train, c.train)
    np.testing.assert_array_equal(a.val, c.val)
    np.testing.assert_array_equal(a.test, c.test)


def test_regime_balance_preserved(npz_path):
    b = load_npz(npz_path)
    labels = regime_labels(b)
    # Only meaningful when both regimes are present.
    if len(np.unique(labels)) < 2:
        pytest.skip("dataset has a single regime")
    sp = stratified_split(b, fractions=(0.6, 0.2, 0.2), seed=0)
    overall = labels.mean()
    for part in (sp.train, sp.val, sp.test):
        if part.size:
            # each split's fast-fraction is within 0.25 of the overall mix
            assert abs(labels[part].mean() - overall) < 0.25


def test_regime_labels_use_per_sample_vf(npz_path):
    b = load_npz(npz_path)
    labels = regime_labels(b)
    expected = (b.param("vs") >= b.param("vf")).astype(int)
    np.testing.assert_array_equal(labels, expected)


def test_fraction_validation(npz_path):
    b = load_npz(npz_path)
    with pytest.raises(ValueError):
        stratified_split(b, fractions=(0.5, 0.4, 0.4))  # sums to 1.3
    with pytest.raises(ValueError):
        stratified_split(b, fractions=(0.0, 0.5, 0.5))  # non-positive


def test_json_round_trip():
    sp = DataSplit(
        train=np.array([0, 2, 4]),
        val=np.array([1]),
        test=np.array([3]),
    )
    back = DataSplit.from_json(sp.to_json())
    np.testing.assert_array_equal(back.train, sp.train)
    np.testing.assert_array_equal(back.val, sp.val)
    np.testing.assert_array_equal(back.test, sp.test)


def test_write_read_file(tmp_path):
    sp = DataSplit(np.array([0, 1]), np.array([2]), np.array([3]))
    p = tmp_path / "split.json"
    sp.write(str(p))
    back = DataSplit.read(str(p))
    np.testing.assert_array_equal(back.train, sp.train)
