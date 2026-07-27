"""Tests for sonic_ml.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml.geometry import default_geometry


def test_default_geometry_matches_gather_shape(bundle):
    geom = default_geometry(bundle)
    assert geom.n_rec == bundle.gather.shape[1]
    assert geom.n_samples == bundle.gather.shape[2]
    # Generator defaults are carried through.
    assert geom.dt == pytest.approx(1.0e-5)
    assert geom.offsets.shape == (geom.n_rec,)


def test_default_geometry_rejects_non_3d(bundle):
    class Fake:
        gather = np.zeros((3, 8))  # 2-D

    with pytest.raises(ValueError):
        default_geometry(Fake())  # type: ignore[arg-type]
