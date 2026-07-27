"""Tests for sonic_ml.loader."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml import gen_shim
from sonic_ml.loader import (
    SUPPORTED_SCHEMA_VERSIONS,
    DatasetBundle,
    SchemaError,
    UnsupportedSchemaVersionError,
    load_npz,
)


def test_load_reads_metadata_not_hardcoded(npz_path):
    b = load_npz(npz_path)
    assert isinstance(b, DatasetBundle)
    # N/M/F are read from metadata.
    assert b.n_samples == b.params.shape[0]
    assert b.n_modes == len(b.mode_names) == b.slowness.shape[1]
    assert b.n_freq == b.freq.shape[0] == b.slowness.shape[2]
    assert b.param_names == ("vp", "vs", "rho", "vf", "rho_f", "a")
    assert b.schema_version in SUPPORTED_SCHEMA_VERSIONS


def test_dtypes_and_shapes(npz_path):
    b = load_npz(npz_path)
    n, m, f = b.n_samples, b.n_modes, b.n_freq
    assert b.params.shape == (n, len(b.param_names))
    assert b.slowness.shape == (n, m, f)
    assert b.mode_in_gather.shape == (n, m)
    assert b.gather.shape[0] == n and b.gather.ndim == 3
    assert b.params.dtype == np.float64
    assert b.mode_in_gather.dtype == np.bool_


def test_param_lookup_and_unknown(npz_path):
    b = load_npz(npz_path)
    np.testing.assert_array_equal(b.param("vs"), b.params[:, 1])
    with pytest.raises(KeyError):
        b.param("not_a_param")


def test_finite_mask_matches_isfinite(npz_path):
    b = load_npz(npz_path)
    np.testing.assert_array_equal(b.finite_mask(), np.isfinite(b.slowness))


def test_missing_key_raises(tmp_path, stacked):
    bad = {k: v for k, v in stacked.items() if k != "mode_names"}
    p = tmp_path / "missing.npz"
    np.savez_compressed(p, **bad)
    with pytest.raises(SchemaError):
        load_npz(str(p))


def test_unsupported_schema_version_raises(tmp_path, stacked):
    bad = dict(stacked)
    bad["schema_version"] = np.asarray(999, dtype=np.int64)
    p = tmp_path / "future.npz"
    np.savez_compressed(p, **bad)
    with pytest.raises(UnsupportedSchemaVersionError):
        load_npz(str(p))


def test_inconsistent_shape_raises(tmp_path, stacked):
    bad = dict(stacked)
    # Drop a mode row from slowness only -> inconsistent with mode_names.
    bad["slowness"] = bad["slowness"][:, :1, :]
    p = tmp_path / "ragged.npz"
    np.savez_compressed(p, **bad)
    with pytest.raises(SchemaError):
        load_npz(str(p))


def test_allow_pickle_false_is_used(tmp_path):
    # An object-dtype array can only be saved/loaded with pickle; the loader
    # must refuse it (allow_pickle=False), proving it never unpickles data.
    p = tmp_path / "pickled.npz"
    obj = np.array([{"x": 1}], dtype=object)
    np.savez(p, params=obj)
    with pytest.raises((ValueError, SchemaError)):
        load_npz(str(p))
