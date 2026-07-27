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


# ------------------------------------------------------------------
# Schema v2: acquisition geometry
# ------------------------------------------------------------------


def test_v2_bundle_carries_geometry(npz_path):
    b = load_npz(npz_path)
    assert b.schema_version >= 2  # geometry present since v2
    assert b.geometry is not None
    assert b.geometry.n_rec == b.gather.shape[1]
    assert b.geometry.n_samples == b.gather.shape[2]
    assert b.geometry.dt > 0.0


def test_v1_file_loads_with_none_geometry(tmp_path, stacked):
    # A legacy v1 file (no geometry keys, schema_version=1) still loads,
    # exposing geometry=None -- backward compatibility.
    legacy = {k: v for k, v in stacked.items() if k not in ("dt", "tr_offset", "dr")}
    legacy["schema_version"] = np.asarray(1, dtype=np.int64)
    p = tmp_path / "v1.npz"
    np.savez_compressed(p, **legacy)
    b = load_npz(str(p))
    assert b.schema_version == 1
    assert b.geometry is None


def test_v2_missing_geometry_key_raises(tmp_path, stacked):
    bad = {k: v for k, v in stacked.items() if k != "dr"}  # keep schema v3
    p = tmp_path / "v2_nogeom.npz"
    np.savez_compressed(p, **bad)
    with pytest.raises(SchemaError):
        load_npz(str(p))


# ------------------------------------------------------------------
# Schema v3: leaky-mode attenuation
# ------------------------------------------------------------------


def test_v3_bundle_carries_attenuation(npz_path):
    b = load_npz(npz_path)
    assert b.schema_version >= 3  # attenuation present since v3
    assert b.attenuation is not None
    assert b.attenuation.shape == b.slowness.shape


def test_v2_file_loads_with_none_attenuation(tmp_path, stacked):
    # A v2 file (geometry present, no attenuation) still loads.
    legacy = {k: v for k, v in stacked.items() if k != "attenuation"}
    legacy["schema_version"] = np.asarray(2, dtype=np.int64)
    p = tmp_path / "v2.npz"
    np.savez_compressed(p, **legacy)
    b = load_npz(str(p))
    assert b.schema_version == 2
    assert b.attenuation is None
    assert b.geometry is not None  # v2 still carries geometry


def test_v3_missing_attenuation_raises(tmp_path, stacked):
    bad = {k: v for k, v in stacked.items() if k != "attenuation"}  # keep v3
    p = tmp_path / "v3_noatt.npz"
    np.savez_compressed(p, **bad)
    with pytest.raises(SchemaError):
        load_npz(str(p))


# ------------------------------------------------------------------
# Schema v4: cased-hole annulus + bond label
# ------------------------------------------------------------------


def test_open_hole_v4_bundle_has_empty_layers(npz_path):
    # The default (open-hole) dataset is v4: layer arrays present but empty,
    # bond_index all-NaN, and is_cased False.
    b = load_npz(npz_path)
    assert b.schema_version >= 4
    assert b.layer_params is not None
    assert b.layer_params.shape == (b.n_samples, 0, 4)
    assert b.layer_names == ()
    assert b.bond_index is not None
    assert not np.isfinite(b.bond_index).any()
    assert b.is_cased is False


def test_cased_v4_bundle_carries_layers_and_bond(tmp_path, freq):
    gen = gen_shim.generator()
    p = tmp_path / "cased.npz"
    gen.save_npz(str(p), gen.generate_cased_dataset(6, seed=0, freq=freq))
    b = load_npz(str(p))
    assert b.schema_version == 4
    assert b.is_cased is True
    assert b.layer_names == ("casing", "cement")
    assert b.layer_params is not None
    assert b.layer_params.shape == (b.n_samples, 2, 4)
    assert b.bond_index is not None
    assert b.bond_index.shape == (b.n_samples,)
    assert np.all(np.isfinite(b.bond_index))
    assert np.all((b.bond_index >= 0.0) & (b.bond_index <= 1.0))
    assert b.mode_names == ("Stoneley",)


def test_v3_file_loads_with_none_cased_fields(tmp_path, stacked):
    # A v3 file (no cased keys) still loads, exposing layer_params/bond None.
    legacy = {
        k: v
        for k, v in stacked.items()
        if k not in ("layer_params", "layer_names", "bond_index")
    }
    legacy["schema_version"] = np.asarray(3, dtype=np.int64)
    p = tmp_path / "v3.npz"
    np.savez_compressed(p, **legacy)
    b = load_npz(str(p))
    assert b.schema_version == 3
    assert b.layer_params is None
    assert b.bond_index is None
    assert b.layer_names == ()
    assert b.is_cased is False


def test_v4_missing_cased_key_raises(tmp_path, stacked):
    bad = {k: v for k, v in stacked.items() if k != "bond_index"}  # keep v4
    p = tmp_path / "v4_nobond.npz"
    np.savez_compressed(p, **bad)
    with pytest.raises(SchemaError):
        load_npz(str(p))
