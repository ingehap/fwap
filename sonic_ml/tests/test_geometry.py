"""Tests for sonic_ml.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml.geometry import default_geometry


def test_prefers_stored_geometry(bundle):
    # Schema v2 datasets carry geometry; default_geometry returns it verbatim.
    assert bundle.geometry is not None
    geom = default_geometry(bundle)
    assert geom is bundle.geometry
    assert geom.n_rec == bundle.gather.shape[1]
    assert geom.n_samples == bundle.gather.shape[2]
    assert geom.dt == pytest.approx(1.0e-5)
    assert geom.offsets.shape == (geom.n_rec,)


def test_v1_fallback_reconstructs_defaults(bundle):
    # A legacy v1 bundle (no stored geometry) falls back to default reconstruction.
    import dataclasses

    v1 = dataclasses.replace(bundle, geometry=None, schema_version=1)
    geom = default_geometry(v1)
    assert geom.n_rec == bundle.gather.shape[1]
    assert geom.n_samples == bundle.gather.shape[2]
    assert geom.dt == pytest.approx(1.0e-5)


def test_v1_fallback_rejects_non_3d():
    class Fake:
        gather = np.zeros((3, 8))  # 2-D
        geometry = None

    with pytest.raises(ValueError):
        default_geometry(Fake())  # type: ignore[arg-type]
