"""
Frozen on-disk contract for the surrogate dataset ``.npz``.

``scripts/gen_surrogate_dataset.py`` is a path-imported script (not part of
the public ``fwap`` API and so not covered by ``scripts/check_public_api.py``),
and the ``.npz`` it writes is the sole hand-off to any downstream
machine-learning consumer. This module pins that layout -- the exact key set,
the :data:`PARAM_NAMES` column order, per-array dtypes and shapes, and the
``schema_version`` -- so a breaking core-side change (a reordered/renamed key
or parameter column, a changed dtype, an added axis) fails CI *here* rather
than silently mislabelling training data downstream.

Any intentional change to the layout must bump ``SCHEMA_VERSION`` in the
generator and update the assertions below in the same commit -- mirroring the
frozen-public-API guard's "change the contract and its test together" rule.

Pure NumPy, no ML dependency: this test runs in the default ``.[dev]`` CI job.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "gen_surrogate_dataset.py"
)
_spec = importlib.util.spec_from_file_location("gen_surrogate_dataset", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
# Register before exec so dataclasses can resolve the module by name when
# ``from __future__ import annotations`` defers annotation evaluation.
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)

# The frozen contract. Update these in lockstep with a SCHEMA_VERSION bump.
EXPECTED_KEYS = {
    "params",
    "slowness",
    "gather",
    "mode_in_gather",
    "freq",
    "param_names",
    "mode_names",
    "schema_version",
    # geometry scalars (schema v2): make the gather self-describing.
    "dt",
    "tr_offset",
    "dr",
}
EXPECTED_PARAM_NAMES = ("vp", "vs", "rho", "vf", "rho_f", "a")
EXPECTED_SCHEMA_VERSION = 2
# Default fwap.ArrayGeometry scalars the generator stamps unless overridden.
EXPECTED_DEFAULT_GEOMETRY = {"dt": 1.0e-5, "tr_offset": 3.0, "dr": 0.1524}


def _small_grid() -> np.ndarray:
    # Coarse grid keeps the forward solves fast in the test suite.
    return gen.default_freq_grid(n_freq=48)


def _dataset(n: int = 3):
    freq = _small_grid()
    samples = gen.generate_dataset(n, seed=0, freq=freq)
    return samples, freq


# ------------------------------------------------------------------
# Constants stay in sync with the pinned contract
# ------------------------------------------------------------------


def test_module_constants_match_contract():
    """The generator's own constants equal the frozen expectations."""
    assert gen.PARAM_NAMES == EXPECTED_PARAM_NAMES
    assert gen.SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION


# ------------------------------------------------------------------
# In-memory stack_dataset layout
# ------------------------------------------------------------------


def test_stack_dataset_key_set_is_frozen():
    samples, _ = _dataset()
    stacked = gen.stack_dataset(samples)
    assert set(stacked) == EXPECTED_KEYS


def test_stack_dataset_schema_version_value_and_dtype():
    samples, _ = _dataset()
    stacked = gen.stack_dataset(samples)
    sv = stacked["schema_version"]
    assert sv.shape == ()  # 0-d scalar
    assert np.issubdtype(sv.dtype, np.integer)
    assert int(sv) == EXPECTED_SCHEMA_VERSION == gen.SCHEMA_VERSION


def test_stack_dataset_param_names_order_is_frozen():
    samples, _ = _dataset()
    stacked = gen.stack_dataset(samples)
    assert tuple(stacked["param_names"]) == EXPECTED_PARAM_NAMES


# ------------------------------------------------------------------
# Round-tripped .npz: dtypes, shapes, and cross-key consistency
# ------------------------------------------------------------------


def test_npz_roundtrip_dtypes_and_shapes(tmp_path):
    samples, freq = _dataset(n=3)
    out = tmp_path / "contract.npz"
    gen.save_npz(str(out), samples)

    # allow_pickle=False: the contract must be plain arrays, never pickled
    # Python objects (a security- and portability-relevant guarantee).
    with np.load(out, allow_pickle=False) as data:
        assert set(data.files) == EXPECTED_KEYS

        n = len(samples)
        n_modes = len(gen.DEFAULT_MODES)
        n_f = freq.size

        # dtypes
        assert data["params"].dtype == np.float64
        assert data["slowness"].dtype == np.float64
        assert data["gather"].dtype == np.float64
        assert data["freq"].dtype == np.float64
        assert data["mode_in_gather"].dtype == np.bool_
        assert data["param_names"].dtype.kind in ("U", "S")
        assert data["mode_names"].dtype.kind in ("U", "S")
        assert np.issubdtype(data["schema_version"].dtype, np.integer)
        for key in EXPECTED_DEFAULT_GEOMETRY:
            assert np.issubdtype(data[key].dtype, np.floating)

        # shapes
        assert data["params"].shape == (n, len(EXPECTED_PARAM_NAMES))
        assert data["slowness"].shape == (n, n_modes, n_f)
        assert data["gather"].shape[0] == n
        assert data["gather"].ndim == 3  # (N, n_rec, n_samples)
        assert data["mode_in_gather"].shape == (n, n_modes)
        assert data["freq"].shape == (n_f,)
        assert data["param_names"].shape == (len(EXPECTED_PARAM_NAMES),)
        assert data["mode_names"].shape == (n_modes,)
        assert data["schema_version"].shape == ()
        for key in EXPECTED_DEFAULT_GEOMETRY:
            assert data[key].shape == ()  # 0-d geometry scalars

        # pinned values / ordering
        assert tuple(data["param_names"]) == EXPECTED_PARAM_NAMES
        assert tuple(data["mode_names"]) == tuple(
            spec.name for spec in gen.DEFAULT_MODES
        )
        assert int(data["schema_version"]) == EXPECTED_SCHEMA_VERSION
        np.testing.assert_array_equal(data["freq"], freq)
        # geometry scalars round-trip the default ArrayGeometry, and together
        # with the gather's (n_rec, n_samples) fully specify acquisition.
        for key, value in EXPECTED_DEFAULT_GEOMETRY.items():
            assert float(data[key]) == pytest.approx(value)


def test_npz_consumer_reads_shape_from_metadata_not_hardcoded(tmp_path):
    """A downstream loader can recover N, M, and n_f from the arrays alone."""
    samples, freq = _dataset(n=2)
    out = tmp_path / "meta.npz"
    gen.save_npz(str(out), samples)
    with np.load(out, allow_pickle=False) as data:
        n_modes = data["mode_names"].shape[0]
        n_params = data["param_names"].shape[0]
        # slowness axis 1 is the mode axis; params axis 1 is the param axis.
        assert data["slowness"].shape[1] == n_modes
        assert data["mode_in_gather"].shape[1] == n_modes
        assert data["params"].shape[1] == n_params
        assert data["slowness"].shape[2] == freq.size
