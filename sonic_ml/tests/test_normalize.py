"""Tests for sonic_ml.normalize."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml.loader import load_npz
from sonic_ml.normalize import Standardizer


def test_constant_columns_are_dropped(npz_path):
    b = load_npz(npz_path)
    std = Standardizer.fit(b.params, b.param_names)
    # vf and rho_f are fixed by the default priors -> zero variance -> dropped.
    assert "vf" not in std.active_names
    assert "rho_f" not in std.active_names
    assert std.active_names == ("vp", "vs", "rho", "a")
    assert std.n_active == 4


def test_transform_shape_and_no_nans(npz_path):
    b = load_npz(npz_path)
    std = Standardizer.fit(b.params, b.param_names)
    z = std.transform(b.params)
    assert z.shape == (b.n_samples, std.n_active)
    assert np.all(np.isfinite(z))  # no divide-by-zero from constant columns


def test_transform_is_standardized(npz_path):
    b = load_npz(npz_path)
    std = Standardizer.fit(b.params, b.param_names)
    z = std.transform(b.params)
    # active columns have ~zero mean, ~unit std
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-9)


def test_inverse_transform_round_trip(npz_path):
    b = load_npz(npz_path)
    std = Standardizer.fit(b.params, b.param_names)
    z = std.transform(b.params)
    raw_active = std.inverse_transform(z)
    active_idx = [b.param_names.index(n) for n in std.active_names]
    np.testing.assert_allclose(raw_active, b.params[:, active_idx])


def test_fit_validates_columns():
    with pytest.raises(ValueError):
        Standardizer.fit(np.zeros((5, 3)), ("a", "b"))  # 3 cols, 2 names
    with pytest.raises(ValueError):
        Standardizer.fit(np.zeros(5), ("a",))  # not 2-D


def test_all_constant_drops_everything():
    params = np.tile([1500.0, 1000.0], (4, 1))
    std = Standardizer.fit(params, ("vf", "rho_f"))
    assert std.n_active == 0
    assert std.transform(params).shape == (4, 0)
