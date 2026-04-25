"""Edge-case tests: small receiver counts, empty inputs, boundary values."""

from __future__ import annotations

import numpy as np
import pytest

from fwap.coherence import stc
from fwap.dispersion import phase_slowness_matrix_pencil
from fwap.synthetic import (
    ArrayGeometry,
    Mode,
    synthesize_gather,
)
from fwap.wavesep import apply_moveout, fk_forward, fk_inverse


def _tiny_gather(n_rec, Vp=4500.0, f0=8000.0, seed=0):
    geom = ArrayGeometry(n_rec=n_rec, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=512)
    mode = Mode(name="P", slowness=1.0 / Vp, f0=f0, amplitude=1.0)
    data = synthesize_gather(geom, [mode], noise=0.01, seed=seed)
    return geom, data


def test_matrix_pencil_rejects_fewer_than_three_receivers():
    """matrix pencil needs >= 3 receivers."""
    geom, data = _tiny_gather(n_rec=2)
    with pytest.raises(ValueError, match=">= 3 receivers"):
        phase_slowness_matrix_pencil(
            data, dt=geom.dt, offsets=geom.offsets,
            f_range=(2000.0, 8000.0))


def test_stc_handles_two_receivers():
    """STC still produces a finite surface on a two-receiver gather."""
    geom, data = _tiny_gather(n_rec=2)
    res = stc(data, dt=geom.dt, offsets=geom.offsets,
              n_slowness=31, window_length=2.0e-4, time_step=4)
    finite = np.isfinite(res.coherence)
    assert finite.any()
    assert np.nanmax(res.coherence) <= 1.0 + 1e-9
    assert np.nanmin(res.coherence[finite]) >= 0.0 - 1e-9


def test_apply_moveout_rejects_wrong_offsets_length():
    """apply_moveout validates that offsets has length n_rec."""
    geom, data = _tiny_gather(n_rec=4)
    with pytest.raises(ValueError, match="length n_rec"):
        apply_moveout(data, geom.dt,
                      offsets=np.array([0.1, 0.2]),    # wrong length
                      slowness=1.0e-4)


def test_fk_round_trip_on_minimal_gather():
    """fk forward/inverse work on the minimal 2-trace gather."""
    geom, data = _tiny_gather(n_rec=2)
    spec, f, k = fk_forward(data, geom.dt, geom.dr)
    assert spec.shape == (2, geom.n_samples // 2 + 1)
    assert f.shape == (geom.n_samples // 2 + 1,)
    assert k.shape == (2,)
    back = fk_inverse(spec, n_samples=geom.n_samples)
    assert np.allclose(back, data, atol=1.0e-10)


def test_stc_rejects_offsets_length_mismatch():
    """stc raises ValueError when offsets does not match n_rec."""
    geom, data = _tiny_gather(n_rec=4)
    with pytest.raises(ValueError, match="length n_rec"):
        stc(data, dt=geom.dt, offsets=np.array([0.1, 0.2, 0.3]),
            n_slowness=11, window_length=2.0e-4)


def test_stc_rejects_n_slowness_lt_2():
    """stc requires at least two slowness samples."""
    geom, data = _tiny_gather(n_rec=4)
    with pytest.raises(ValueError, match="n_slowness"):
        stc(data, dt=geom.dt, offsets=geom.offsets,
            n_slowness=1, window_length=2.0e-4)
