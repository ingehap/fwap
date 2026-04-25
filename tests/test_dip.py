"""Dip / azimuth tests."""

from __future__ import annotations

import numpy as np
import pytest

from fwap.dip import estimate_dip, synthesize_azimuthal_arrival


@pytest.mark.parametrize("dip_deg, az_deg", [
    (15.0,   30.0),
    (35.0,   60.0),
    (50.0,  -45.0),
])
def test_estimate_dip_recovers_planted(dip_deg, az_deg):
    """Grid + refinement recover the planted (dip, azimuth) within 2 deg."""
    data, dt, ax_off, az, a, slow = synthesize_azimuthal_arrival(
        n_rec=8, n_samples=1024, dt=2.0e-5,
        tool_radius=0.08, slowness=1.0 / 4000.0,
        dip=np.deg2rad(dip_deg), azimuth=np.deg2rad(az_deg),
        f0=8000.0, noise=0.02, seed=3)
    result = estimate_dip(data, dt=dt, axial_offsets=ax_off, azimuths=az,
                          tool_radius=a, slowness=slow,
                          dip_range=(0.0, np.deg2rad(70.0)),
                          n_dip=31, n_az=72, refine=True)
    assert abs(np.rad2deg(result.dip) - dip_deg) < 2.0
    # Azimuth error in the (-180, 180] fold.
    daz = np.rad2deg(result.azimuth) - az_deg
    daz = ((daz + 180.0) % 360.0) - 180.0
    assert abs(daz) < 2.0


def test_estimate_dip_refined_flag_set_when_improved():
    """refined=True iff Nelder-Mead improves on the grid maximum."""
    data, dt, ax_off, az, a, slow = synthesize_azimuthal_arrival(
        n_rec=8, n_samples=1024, dt=2.0e-5,
        dip=np.deg2rad(35.0), azimuth=np.deg2rad(45.0),
        noise=0.02, seed=5)
    result = estimate_dip(data, dt=dt, axial_offsets=ax_off, azimuths=az,
                          tool_radius=a, slowness=slow,
                          dip_range=(0.0, np.deg2rad(70.0)),
                          n_dip=11, n_az=12, refine=True)
    # With such a coarse grid the refinement should almost always help.
    assert result.refined


def test_axial_offsets_is_ignored():
    """Passing different axial_offsets must not affect the result."""
    data, dt, ax_off, az, a, slow = synthesize_azimuthal_arrival(
        n_rec=8, n_samples=1024, seed=1)
    r1 = estimate_dip(data, dt=dt, axial_offsets=np.zeros_like(ax_off),
                      azimuths=az, tool_radius=a, slowness=slow,
                      n_dip=11, n_az=12, refine=False)
    r2 = estimate_dip(data, dt=dt, axial_offsets=np.arange(ax_off.size) * 0.1,
                      azimuths=az, tool_radius=a, slowness=slow,
                      n_dip=11, n_az=12, refine=False)
    assert r1.dip == r2.dip
    assert r1.azimuth == r2.azimuth


def test_synthesize_azimuthal_arrival_returns_named_tuple():
    """Return value is an AzimuthalGather with both tuple and field access."""
    from fwap.dip import AzimuthalGather
    result = synthesize_azimuthal_arrival(n_rec=6, n_samples=256, seed=0)
    # Named-tuple is-a tuple.
    assert isinstance(result, AzimuthalGather)
    assert isinstance(result, tuple)
    assert len(result) == 6
    # Tuple-unpacking still works.
    data, dt, ax_off, az, a, slow = result
    # Field access also works.
    assert result.data is data
    assert result.dt == dt
    assert result.tool_radius == a
    assert result.slowness == slow
