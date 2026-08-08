"""Shared fixtures for the sonic_ml test suite."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml import gen_shim


@pytest.fixture(scope="session")
def freq() -> np.ndarray:
    """A coarse shared frequency grid (fast forward solves)."""
    return gen_shim.generator().default_freq_grid(n_freq=48)


@pytest.fixture(scope="session")
def stacked(freq: np.ndarray) -> dict[str, np.ndarray]:
    """A stacked dataset dict (as ``save_npz`` would write) for 40 samples."""
    gen = gen_shim.generator()
    return gen.stack_dataset(gen_shim.generate(40, seed=0, freq=freq))


@pytest.fixture(scope="session")
def npz_path(tmp_path_factory: pytest.TempPathFactory, freq: np.ndarray) -> str:
    """Path to a small on-disk dataset shared across read-only tests."""
    p = tmp_path_factory.mktemp("data") / "ds.npz"
    gen_shim.build_npz(str(p), n=40, seed=0, freq=freq)
    return str(p)


@pytest.fixture(scope="session")
def bundle(npz_path: str):
    """The shared dataset loaded into a DatasetBundle."""
    from sonic_ml import load_npz

    return load_npz(npz_path)


@pytest.fixture(scope="session")
def cased_npz_path(tmp_path_factory: pytest.TempPathFactory, freq: np.ndarray) -> str:
    """Path to a small on-disk *cased-hole* dataset (schema v4)."""
    gen = gen_shim.generator()
    p = tmp_path_factory.mktemp("cased") / "cased.npz"
    gen.save_npz(str(p), gen.generate_cased_dataset(40, seed=0, freq=freq))
    return str(p)


@pytest.fixture(scope="session")
def cased_bundle(cased_npz_path: str):
    """The shared cased-hole dataset loaded into a DatasetBundle."""
    from sonic_ml import load_npz

    return load_npz(cased_npz_path)


@pytest.fixture(scope="session")
def geom(bundle):
    """Default acquisition geometry reconstructed for the shared dataset."""
    from sonic_ml import default_geometry

    return default_geometry(bundle)
