"""Tests for sonic_ml.determinism.

The pure-NumPy behaviour is always exercised; the torch-specific paths are
skipped when torch is not installed (they run in the ml.yml CI job).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from sonic_ml import determinism


def test_seed_everything_makes_numpy_reproducible():
    determinism.seed_everything(123)
    a = np.random.rand(5)
    determinism.seed_everything(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_sets_pythonhashseed():
    determinism.seed_everything(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_pin_threads_sets_env():
    determinism.pin_threads(2)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"


def test_worker_init_fn_seeds_numpy_without_torch():
    # Runs regardless of torch; two calls with the same worker id reseed
    # numpy identically.
    determinism.worker_init_fn(0)
    a = np.random.rand(3)
    determinism.worker_init_fn(0)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_torch_available_is_bool():
    assert isinstance(determinism.torch_available(), bool)


def test_seed_everything_with_torch():
    torch = pytest.importorskip("torch")
    determinism.seed_everything(321)
    a = torch.rand(4)
    determinism.seed_everything(321)
    b = torch.rand(4)
    assert torch.equal(a, b)
