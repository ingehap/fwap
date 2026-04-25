"""Alford rotation tests."""

from __future__ import annotations

import numpy as np

from fwap.anisotropy import alford_rotation
from fwap.synthetic import ricker


def _rotated_tensor(true_angle_rad, vs_fast=2600.0, vs_slow=2400.0,
                    noise=0.005, seed=0):
    """Build an (xx, xy, yx, yy) cross-dipole tensor with a planted angle."""
    rng = np.random.default_rng(seed)
    n_samp = 1024
    dt = 2.0e-5
    t = np.arange(n_samp) * dt
    offset = 3.5
    fast = ricker(t, 3000.0, t0=offset / vs_fast)
    slow = 0.85 * ricker(t, 3000.0, t0=offset / vs_slow)
    c, s = np.cos(true_angle_rad), np.sin(true_angle_rad)
    xx = c * c * fast + s * s * slow
    yy = s * s * fast + c * c * slow
    xy = c * s * (fast - slow)
    yx = c * s * (fast - slow)
    for arr in (xx, xy, yx, yy):
        arr += rng.normal(scale=noise * np.max(np.abs(arr)), size=arr.shape)
    return xx, xy, yx, yy


def test_alford_recovers_planted_angle():
    """Planted angle in (-pi/2, pi/2] is recovered within ~1 degree."""
    for deg in (-60.0, -30.0, 0.0, 15.0, 45.0, 75.0):
        xx, xy, yx, yy = _rotated_tensor(np.deg2rad(deg))
        res = alford_rotation(xx, xy, yx, yy)
        err_deg = abs(np.rad2deg(res.angle) - deg)
        # Angle is returned modulo pi; +/-90 deg maps to the same
        # orientation. Fold the error into [0, 90] before testing.
        err_deg = min(err_deg, 180.0 - err_deg)
        assert err_deg < 1.5, f"deg={deg} recovered with {err_deg:.2f} deg error"


def test_alford_cross_energy_small_after_rotation():
    """Cross-component residual energy is a small fraction after rotation.

    The synthetic has 0.5% noise on each component, so the cross-energy
    floor is dominated by the noise contribution; we only check that
    the ratio is small compared to the diagonal energies.
    """
    xx, xy, yx, yy = _rotated_tensor(np.deg2rad(30.0))
    res = alford_rotation(xx, xy, yx, yy)
    assert 0.0 <= res.cross_energy_ratio < 1.0e-2


def test_alford_shape_mismatch_raises():
    """Mismatched component shapes raise ValueError."""
    import pytest
    xx = np.zeros(100)
    yy = np.zeros(100)
    xy = np.zeros(101)
    yx = np.zeros(100)
    with pytest.raises(ValueError):
        alford_rotation(xx, xy, yx, yy)


def test_alford_rotation_from_tensor_matches_four_arg_form():
    """Tensor adapter produces the same result as the 4-arg function."""
    from fwap.anisotropy import alford_rotation_from_tensor
    xx, xy, yx, yy = _rotated_tensor(np.deg2rad(30.0))
    res_args = alford_rotation(xx, xy, yx, yy)
    tensor = np.stack([np.stack([xx, xy]), np.stack([yx, yy])])
    assert tensor.shape == (2, 2, xx.size)
    res_tensor = alford_rotation_from_tensor(tensor)
    assert abs(res_args.angle - res_tensor.angle) < 1.0e-12
    assert abs(res_args.cross_energy_ratio
               - res_tensor.cross_energy_ratio) < 1.0e-12
    assert np.allclose(res_args.fast, res_tensor.fast)
    assert np.allclose(res_args.slow, res_tensor.slow)


def test_alford_rotation_from_tensor_rejects_wrong_shape():
    """tensor adapter raises ValueError when the first two dims are not (2, 2)."""
    import pytest

    from fwap.anisotropy import alford_rotation_from_tensor
    with pytest.raises(ValueError, match="shape"):
        alford_rotation_from_tensor(np.zeros((3, 2, 128)))
    with pytest.raises(ValueError, match="shape"):
        alford_rotation_from_tensor(np.zeros((128,)))


# ---------------------------------------------------------------------
# StressAnisotropyEstimate -- petrophysical labelling of Alford output
# ---------------------------------------------------------------------


def test_stress_anisotropy_recovers_planted_azimuth_and_orthogonal():
    """max-H stress azimuth = planted angle; min-H is orthogonal."""
    from fwap.anisotropy import (
        StressAnisotropyEstimate,
        stress_anisotropy_from_alford,
    )
    angle = np.deg2rad(30.0)
    xx, xy, yx, yy = _rotated_tensor(angle)
    res = alford_rotation(xx, xy, yx, yy)
    est = stress_anisotropy_from_alford(res, dt=2.0e-5)
    assert isinstance(est, StressAnisotropyEstimate)
    assert abs(est.max_horizontal_stress_azimuth - angle) < np.deg2rad(1.0)
    # Orthogonality, with min-H folded into (-pi/2, pi/2].
    diff = (est.max_horizontal_stress_azimuth
            - est.min_horizontal_stress_azimuth)
    # Must be ±pi/2 modulo pi.
    assert abs(abs(diff) - np.pi / 2) < 1.0e-9


def test_stress_anisotropy_splitting_time_matches_planted_delay():
    """splitting_time_delay ≈ offset/Vs_slow - offset/Vs_fast."""
    from fwap.anisotropy import stress_anisotropy_from_alford
    vs_fast, vs_slow = 2600.0, 2400.0
    offset = 3.5
    expected_delay = offset / vs_slow - offset / vs_fast    # ~112 us
    xx, xy, yx, yy = _rotated_tensor(np.deg2rad(30.0),
                                      vs_fast=vs_fast, vs_slow=vs_slow)
    res = alford_rotation(xx, xy, yx, yy)
    est = stress_anisotropy_from_alford(res, dt=2.0e-5)
    # Within one sample (dt=20 us) of the analytical splitting delay.
    assert abs(est.splitting_time_delay - expected_delay) < 2.0e-5
    # Sign must be positive: slow trails fast.
    assert est.splitting_time_delay > 0.0


def test_stress_anisotropy_strength_is_zero_on_isotropic_medium():
    """fast == slow ⇒ anisotropy_strength == 0 and splitting delay == 0."""
    from fwap.anisotropy import stress_anisotropy_from_alford
    # Build a truly isotropic tensor: same Vs *and* same amplitude on
    # both rotated axes (the ``_rotated_tensor`` helper scales slow by
    # 0.85, so it is never quite isotropic). Pick an arbitrary
    # rotation angle to confirm the labelling tolerates any frame.
    n_samp, dt = 1024, 2.0e-5
    t = np.arange(n_samp) * dt
    wavelet = ricker(t, 3000.0, t0=3.5 / 2500.0)
    angle = np.deg2rad(20.0)
    c, s = np.cos(angle), np.sin(angle)
    xx_iso = c * c * wavelet + s * s * wavelet
    yy_iso = s * s * wavelet + c * c * wavelet
    xy_iso = c * s * (wavelet - wavelet)
    yx_iso = c * s * (wavelet - wavelet)
    res = alford_rotation(xx_iso, xy_iso, yx_iso, yy_iso)
    est = stress_anisotropy_from_alford(res, dt=dt)
    assert est.anisotropy_strength < 1.0e-9
    assert abs(est.splitting_time_delay) < 1.0e-12
    # Fracture indicator is the product, so it must also be ~0.
    assert est.fracture_indicator < 1.0e-9


def test_stress_anisotropy_strength_in_unit_interval_and_grows_with_contrast():
    """anisotropy_strength stays in [0, 1] and increases with vs contrast."""
    from fwap.anisotropy import stress_anisotropy_from_alford
    angle = np.deg2rad(20.0)
    res_small = alford_rotation(*_rotated_tensor(angle,
                                                  vs_fast=2510.0, vs_slow=2490.0))
    res_large = alford_rotation(*_rotated_tensor(angle,
                                                  vs_fast=2700.0, vs_slow=2300.0))
    est_small = stress_anisotropy_from_alford(res_small, dt=2.0e-5)
    est_large = stress_anisotropy_from_alford(res_large, dt=2.0e-5)
    for est in (est_small, est_large):
        assert 0.0 <= est.anisotropy_strength <= 1.0
        assert 0.0 <= est.fracture_indicator <= 1.0
        assert 0.0 <= est.rotation_quality <= 1.0
    assert est_large.anisotropy_strength > est_small.anisotropy_strength


def test_stress_anisotropy_rotation_quality_matches_alford():
    """rotation_quality = 1 - cross_energy_ratio."""
    from fwap.anisotropy import stress_anisotropy_from_alford
    res = alford_rotation(*_rotated_tensor(np.deg2rad(15.0)))
    est = stress_anisotropy_from_alford(res, dt=2.0e-5)
    assert abs(est.rotation_quality - (1.0 - res.cross_energy_ratio)) < 1.0e-12
    # Underlying Alford output is preserved on the dataclass.
    assert est.alford is res
