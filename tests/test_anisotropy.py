"""Alford rotation tests."""

from __future__ import annotations

import numpy as np

from fwap.anisotropy import alford_rotation
from fwap.synthetic import ricker


def _rotated_tensor(
    true_angle_rad, vs_fast=2600.0, vs_slow=2400.0, noise=0.005, seed=0
):
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
    assert abs(res_args.cross_energy_ratio - res_tensor.cross_energy_ratio) < 1.0e-12
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
    diff = est.max_horizontal_stress_azimuth - est.min_horizontal_stress_azimuth
    # Must be ±pi/2 modulo pi.
    assert abs(abs(diff) - np.pi / 2) < 1.0e-9


def test_stress_anisotropy_splitting_time_matches_planted_delay():
    """splitting_time_delay ≈ offset/Vs_slow - offset/Vs_fast."""
    from fwap.anisotropy import stress_anisotropy_from_alford

    vs_fast, vs_slow = 2600.0, 2400.0
    offset = 3.5
    expected_delay = offset / vs_slow - offset / vs_fast  # ~112 us
    xx, xy, yx, yy = _rotated_tensor(np.deg2rad(30.0), vs_fast=vs_fast, vs_slow=vs_slow)
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
    res_small = alford_rotation(*_rotated_tensor(angle, vs_fast=2510.0, vs_slow=2490.0))
    res_large = alford_rotation(*_rotated_tensor(angle, vs_fast=2700.0, vs_slow=2300.0))
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


# ---------------------------------------------------------------------
# Thomsen gamma (VTI shear anisotropy from dipole + Stoneley)
# ---------------------------------------------------------------------


def test_stoneley_c66_round_trips_through_white_formula():
    """Plant C66, build the matching Stoneley slowness, recover C66."""
    import pytest

    from fwap.anisotropy import stoneley_horizontal_shear_modulus

    rho_f, v_f = 1000.0, 1500.0
    c66_planted = 1.0e10  # 10 GPa, typical sandstone
    # White (1983): S_ST^2 = 1/V_f^2 + rho_f / C66
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66_planted)
    c66 = stoneley_horizontal_shear_modulus(s_st, rho_fluid=rho_f, v_fluid=v_f)
    assert c66 == pytest.approx(c66_planted, rel=1.0e-12)


def test_stoneley_c66_vector_input():
    """Per-depth Stoneley slowness gives a per-depth C66."""
    from fwap.anisotropy import stoneley_horizontal_shear_modulus

    rho_f, v_f = 1000.0, 1500.0
    c66_planted = np.array([5.0e9, 1.0e10, 2.0e10])
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66_planted)
    c66 = stoneley_horizontal_shear_modulus(s_st, rho_fluid=rho_f, v_fluid=v_f)
    assert c66.shape == (3,)
    np.testing.assert_allclose(c66, c66_planted, rtol=1.0e-12)


def test_stoneley_c66_rejects_slowness_below_fluid_slowness():
    """Stoneley slowness must exceed fluid slowness; reject otherwise."""
    import pytest

    from fwap.anisotropy import stoneley_horizontal_shear_modulus

    rho_f, v_f = 1000.0, 1500.0
    s_f = 1.0 / v_f
    with pytest.raises(ValueError, match="v_fluid"):
        stoneley_horizontal_shear_modulus(s_f, rho_fluid=rho_f, v_fluid=v_f)
    with pytest.raises(ValueError, match="v_fluid"):
        stoneley_horizontal_shear_modulus(0.5 * s_f, rho_fluid=rho_f, v_fluid=v_f)


def test_stoneley_c66_rejects_non_positive_fluid_params():
    """rho_fluid and v_fluid must be strictly positive."""
    import pytest

    from fwap.anisotropy import stoneley_horizontal_shear_modulus

    with pytest.raises(ValueError, match="rho_fluid"):
        stoneley_horizontal_shear_modulus(8.0e-4, rho_fluid=0.0, v_fluid=1500.0)
    with pytest.raises(ValueError, match="v_fluid"):
        stoneley_horizontal_shear_modulus(8.0e-4, rho_fluid=1000.0, v_fluid=-1.0)


def test_thomsen_gamma_zero_for_isotropic_inputs():
    """C44 == C66 -> gamma == 0."""
    from fwap.anisotropy import thomsen_gamma

    g = thomsen_gamma(c44=1.0e10, c66=1.0e10)
    assert g == 0.0


def test_thomsen_gamma_positive_for_horizontal_stiffer_than_vertical():
    """Typical VTI shale: C66 > C44 -> gamma > 0."""
    import pytest

    from fwap.anisotropy import thomsen_gamma

    g = thomsen_gamma(c44=8.0e9, c66=1.2e10)
    # (1.2e10 - 8e9) / (2 * 8e9) = 4e9 / 1.6e10 = 0.25
    assert g == pytest.approx(0.25)


def test_thomsen_gamma_negative_when_horizontal_softer():
    """Pathological / unusual case: C66 < C44 -> gamma < 0 (allowed)."""
    from fwap.anisotropy import thomsen_gamma

    g = thomsen_gamma(c44=1.0e10, c66=8.0e9)
    assert g < 0


def test_thomsen_gamma_rejects_non_positive_moduli():
    """C44 or C66 <= 0 raises ValueError."""
    import pytest

    from fwap.anisotropy import thomsen_gamma

    with pytest.raises(ValueError, match="c44"):
        thomsen_gamma(c44=0.0, c66=1.0e10)
    with pytest.raises(ValueError, match="c66"):
        thomsen_gamma(c44=1.0e10, c66=-1.0)


def test_thomsen_gamma_from_logs_recovers_planted_anisotropy():
    """End-to-end: plant Vsv, C66, rho; recover gamma from the formulas."""
    import pytest

    from fwap.anisotropy import thomsen_gamma_from_logs

    rho = 2400.0
    rho_f, v_f = 1000.0, 1500.0
    Vsv = 2500.0  # vertical shear velocity
    s_dipole = 1.0 / Vsv
    c44_truth = rho * Vsv**2
    c66_truth = 1.3 * c44_truth
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66_truth)
    res = thomsen_gamma_from_logs(
        slowness_dipole=s_dipole,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    # gamma = (1.3 c44 - c44) / (2 c44) = 0.15
    assert res.c44 == pytest.approx(c44_truth, rel=1.0e-12)
    assert res.c66 == pytest.approx(c66_truth, rel=1.0e-12)
    assert res.gamma == pytest.approx(0.15, rel=1.0e-12)


def test_thomsen_gamma_from_logs_vector_inputs_broadcast():
    """Per-depth arrays in -> per-depth arrays out, all aligned."""
    from fwap.anisotropy import thomsen_gamma_from_logs

    n = 4
    rho_f, v_f = 1000.0, 1500.0
    rho = np.full(n, 2400.0)
    Vsv = np.array([2400.0, 2500.0, 2600.0, 2700.0])
    s_dipole = 1.0 / Vsv
    c44 = rho * Vsv**2
    c66 = c44 * np.array([1.0, 1.1, 1.2, 1.3])  # increasing anisotropy
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66)
    res = thomsen_gamma_from_logs(
        slowness_dipole=s_dipole,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    assert res.gamma.shape == (n,)
    assert np.all(np.diff(res.gamma) > 0)
    np.testing.assert_allclose(
        res.gamma, [0.0, 0.05, 0.10, 0.15], rtol=1.0e-12, atol=1.0e-12
    )


def test_thomsen_gamma_from_logs_rejects_non_positive_inputs():
    """Negative or zero slowness / density is rejected with a clear message."""
    import pytest

    from fwap.anisotropy import thomsen_gamma_from_logs

    with pytest.raises(ValueError, match="slowness_dipole"):
        thomsen_gamma_from_logs(
            slowness_dipole=0.0,
            slowness_stoneley=8.0e-4,
            rho=2400.0,
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )
    with pytest.raises(ValueError, match="slowness_stoneley"):
        thomsen_gamma_from_logs(
            slowness_dipole=4.0e-4,
            slowness_stoneley=-1.0,
            rho=2400.0,
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )
    with pytest.raises(ValueError, match="rho"):
        thomsen_gamma_from_logs(
            slowness_dipole=4.0e-4,
            slowness_stoneley=8.0e-4,
            rho=0.0,
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )


def test_thomsen_gamma_from_logs_round_trips_through_write_las(tmp_path):
    """gamma_from_logs output is LAS-ready: C44 / C66 / GAMMA mnemonics."""
    from fwap.anisotropy import thomsen_gamma_from_logs
    from fwap.io import read_las, write_las

    n = 4
    depth = np.linspace(1000.0, 1003.0, n)
    rho_f, v_f = 1000.0, 1500.0
    rho = np.full(n, 2400.0)
    Vsv = np.full(n, 2500.0)
    s_dipole = 1.0 / Vsv
    c44 = rho * Vsv**2
    c66 = c44 * np.linspace(1.0, 1.3, n)
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66)
    res = thomsen_gamma_from_logs(
        slowness_dipole=s_dipole,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    curves = {"C44": res.c44, "C66": res.c66, "GAMMA": res.gamma}
    path = str(tmp_path / "gamma.las")
    write_las(path, depth, curves, well_name="VTI")
    loaded = read_las(path)
    assert loaded.units["C44"] == "Pa"
    assert loaded.units["C66"] == "Pa"
    assert loaded.units["GAMMA"] == ""
    np.testing.assert_allclose(loaded.curves["GAMMA"], res.gamma, rtol=0, atol=1.0e-3)


# ---------------------------------------------------------------------
# c33_from_p_pick + vti_moduli_from_logs
# ---------------------------------------------------------------------


def test_c33_from_p_pick_round_trips_through_rho_Vp_squared():
    """C33 = rho * Vp^2 = rho / S_P^2."""
    import pytest

    from fwap.anisotropy import c33_from_p_pick

    rho = 2400.0
    Vp = 4500.0
    S_P = 1.0 / Vp
    c33 = c33_from_p_pick(S_P, rho)
    assert c33 == pytest.approx(rho * Vp**2, rel=1.0e-12)


def test_c33_from_p_pick_vector_input_broadcasts():
    """Per-depth slowness + density gives per-depth C33."""
    from fwap.anisotropy import c33_from_p_pick

    rho = np.array([2300.0, 2400.0, 2500.0])
    Vp = np.array([4400.0, 4500.0, 4600.0])
    S_P = 1.0 / Vp
    c33 = c33_from_p_pick(S_P, rho)
    np.testing.assert_allclose(c33, rho * Vp**2, rtol=1.0e-12)


def test_c33_from_p_pick_rejects_non_positive_inputs():
    """Slowness and density must both be strictly positive."""
    import pytest

    from fwap.anisotropy import c33_from_p_pick

    with pytest.raises(ValueError, match="slowness_p"):
        c33_from_p_pick(0.0, 2400.0)
    with pytest.raises(ValueError, match="rho"):
        c33_from_p_pick(2.0e-4, -1.0)


def test_vti_moduli_from_logs_recovers_planted_isotropic_case():
    """C66 == C44 -> gamma = 0, Vsh == Vsv exactly."""
    import pytest

    from fwap.anisotropy import vti_moduli_from_logs

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs = 4500.0, 2500.0
    # Plant an isotropic formation: pick a Stoneley slowness whose
    # *Tang & Cheng (2004)* corrected C66 matches C44 (the function
    # defaults to the corrected inversion).
    c44 = rho * Vs**2
    factor = 1.0 - rho_f * v_f**2 / (rho * Vp**2)
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / (c44 * factor))
    out = vti_moduli_from_logs(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    assert out.c33 == pytest.approx(rho * Vp**2, rel=1.0e-12)
    assert out.c44 == pytest.approx(c44, rel=1.0e-12)
    assert out.c66 == pytest.approx(c44, rel=1.0e-12)
    assert out.gamma == pytest.approx(0.0, abs=1.0e-12)
    assert out.vp == pytest.approx(Vp, rel=1.0e-12)
    assert out.vsv == pytest.approx(Vs, rel=1.0e-12)
    assert out.vsh == pytest.approx(Vs, rel=1.0e-12)


def test_vti_moduli_from_logs_planted_vti_case():
    """C66 = 1.3 C44 -> gamma = 0.15; Vsh > Vsv."""
    import pytest

    from fwap.anisotropy import vti_moduli_from_logs

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs = 4500.0, 2500.0
    c44 = rho * Vs**2
    c66 = 1.3 * c44
    # Plant via the corrected forward (default).
    factor = 1.0 - rho_f * v_f**2 / (rho * Vp**2)
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / (c66 * factor))
    out = vti_moduli_from_logs(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    assert out.gamma == pytest.approx(0.15, rel=1.0e-12)
    assert out.vsh > out.vsv
    # Vsh = sqrt(C66/rho) = sqrt(1.3) * Vsv
    assert out.vsh == pytest.approx(np.sqrt(1.3) * Vs, rel=1.0e-12)


def test_vti_moduli_from_logs_vector_inputs_broadcast():
    """Per-depth arrays in -> per-depth fields out, all aligned."""
    from fwap.anisotropy import vti_moduli_from_logs

    n = 4
    rho_f, v_f = 1000.0, 1500.0
    rho = np.full(n, 2400.0)
    Vp = np.array([4400.0, 4500.0, 4600.0, 4700.0])
    Vs = np.array([2400.0, 2500.0, 2600.0, 2700.0])
    c44 = rho * Vs**2
    c66 = c44 * np.array([1.0, 1.1, 1.2, 1.3])
    # Plant via the corrected forward (default).
    factor = 1.0 - rho_f * v_f**2 / (rho * Vp**2)
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / (c66 * factor))
    out = vti_moduli_from_logs(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    for fld in (out.c33, out.c44, out.c66, out.gamma, out.vp, out.vsv, out.vsh):
        assert fld.shape == (n,)
    np.testing.assert_allclose(
        out.gamma, [0.0, 0.05, 0.10, 0.15], rtol=1.0e-12, atol=1.0e-12
    )
    # Vsh / Vsv ratio is sqrt(C66 / C44).
    np.testing.assert_allclose(out.vsh / out.vsv, np.sqrt(c66 / c44), rtol=1.0e-12)


def test_vti_moduli_from_logs_internal_consistency():
    """gamma matches (c66 - c44) / (2 c44); velocities match sqrt(C/rho)."""
    from fwap.anisotropy import vti_moduli_from_logs

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs = 4500.0, 2500.0
    c66 = 1.2 * rho * Vs**2
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66)
    out = vti_moduli_from_logs(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    np.testing.assert_allclose(
        out.gamma, (out.c66 - out.c44) / (2.0 * out.c44), rtol=1.0e-12
    )
    np.testing.assert_allclose(out.vp, np.sqrt(out.c33 / rho), rtol=1.0e-12)
    np.testing.assert_allclose(out.vsv, np.sqrt(out.c44 / rho), rtol=1.0e-12)
    np.testing.assert_allclose(out.vsh, np.sqrt(out.c66 / rho), rtol=1.0e-12)


def test_vti_moduli_from_logs_rejects_non_positive_inputs():
    """All slownesses and density must be strictly positive."""
    import pytest

    from fwap.anisotropy import vti_moduli_from_logs

    with pytest.raises(ValueError, match="slowness_p"):
        vti_moduli_from_logs(
            slowness_p=0.0,
            slowness_dipole=4.0e-4,
            slowness_stoneley=8.0e-4,
            rho=2400.0,
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )
    with pytest.raises(ValueError, match="slowness_dipole"):
        vti_moduli_from_logs(
            slowness_p=2.0e-4,
            slowness_dipole=-1.0,
            slowness_stoneley=8.0e-4,
            rho=2400.0,
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )
    with pytest.raises(ValueError, match="slowness_stoneley"):
        vti_moduli_from_logs(
            slowness_p=2.0e-4,
            slowness_dipole=4.0e-4,
            slowness_stoneley=0.0,
            rho=2400.0,
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )
    with pytest.raises(ValueError, match="rho"):
        vti_moduli_from_logs(
            slowness_p=2.0e-4,
            slowness_dipole=4.0e-4,
            slowness_stoneley=8.0e-4,
            rho=0.0,
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )


def test_vti_moduli_from_logs_round_trips_through_write_las(tmp_path):
    """C33 / C44 / C66 / GAMMA / VP / VSV / VSH mnemonics carry units."""
    from fwap.anisotropy import vti_moduli_from_logs
    from fwap.io import read_las, write_las

    n = 4
    depth = np.linspace(1000.0, 1003.0, n)
    rho_f, v_f = 1000.0, 1500.0
    rho = np.full(n, 2400.0)
    Vp = np.full(n, 4500.0)
    Vs = np.full(n, 2500.0)
    c44 = rho * Vs**2
    c66 = c44 * np.linspace(1.0, 1.3, n)
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66)
    out = vti_moduli_from_logs(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    curves = {
        "C33": out.c33,
        "C44": out.c44,
        "C66": out.c66,
        "GAMMA": out.gamma,
        "VP": out.vp,
        "VSV": out.vsv,
        "VSH": out.vsh,
    }
    path = str(tmp_path / "vti.las")
    write_las(path, depth, curves, well_name="VTI_FULL")
    loaded = read_las(path)
    assert loaded.units["C33"] == "Pa"
    assert loaded.units["C44"] == "Pa"
    assert loaded.units["C66"] == "Pa"
    assert loaded.units["GAMMA"] == ""
    assert loaded.units["VP"] == "m/s"
    assert loaded.units["VSV"] == "m/s"
    assert loaded.units["VSH"] == "m/s"


# ---------------------------------------------------------------------
# Tang & Cheng (2004) sect. 5.4 finite-impedance correction on
# stoneley_horizontal_shear_modulus
# ---------------------------------------------------------------------


def test_stoneley_c66_corrected_round_trips_through_forward_model():
    """Plant C66 + V_P, build the corrected forward S_ST, recover C66."""
    import pytest

    from fwap.anisotropy import stoneley_horizontal_shear_modulus_corrected

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp = 4500.0
    c66_planted = 1.5e10
    s_p = 1.0 / Vp
    factor = 1.0 - rho_f * v_f**2 / (rho * Vp**2)
    c66_eff = c66_planted * factor
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66_eff)
    c66 = stoneley_horizontal_shear_modulus_corrected(
        slowness_stoneley=s_st, rho=rho, slowness_p=s_p, rho_fluid=rho_f, v_fluid=v_f
    )
    assert c66 == pytest.approx(c66_planted, rel=1.0e-12)


def test_stoneley_c66_corrected_exceeds_uncorrected_for_finite_vp():
    """For finite V_P the corrected C66 is strictly greater than the
    White (1983) reading of the same observed slowness; the ratio
    matches the closed-form factor 1/(1 - rho_f V_f^2 / (rho V_P^2))."""
    import pytest

    from fwap.anisotropy import (
        stoneley_horizontal_shear_modulus,
        stoneley_horizontal_shear_modulus_corrected,
    )

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp = 4500.0
    s_p = 1.0 / Vp
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / 1.0e10)
    c66_white = stoneley_horizontal_shear_modulus(s_st, rho_fluid=rho_f, v_fluid=v_f)
    c66_corr = stoneley_horizontal_shear_modulus_corrected(
        slowness_stoneley=s_st, rho=rho, slowness_p=s_p, rho_fluid=rho_f, v_fluid=v_f
    )
    expected_ratio = 1.0 / (1.0 - rho_f * v_f**2 / (rho * Vp**2))
    assert c66_corr > c66_white
    assert c66_corr / c66_white == pytest.approx(expected_ratio, rel=1.0e-12)


def test_stoneley_c66_corrected_reduces_to_white_in_rigid_limit():
    """V_P -> very large => correction factor -> 1 => corrected == White."""
    from fwap.anisotropy import (
        stoneley_horizontal_shear_modulus,
        stoneley_horizontal_shear_modulus_corrected,
    )

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp = 1.0e8  # absurdly fast formation; correction factor ~ 1 - 1e-13
    s_p = 1.0 / Vp
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / 1.0e10)
    c66_white = stoneley_horizontal_shear_modulus(s_st, rho_fluid=rho_f, v_fluid=v_f)
    c66_corr = stoneley_horizontal_shear_modulus_corrected(
        slowness_stoneley=s_st, rho=rho, slowness_p=s_p, rho_fluid=rho_f, v_fluid=v_f
    )
    np.testing.assert_allclose(c66_corr, c66_white, rtol=1.0e-10)


def test_stoneley_c66_corrected_correction_grows_with_slow_formation():
    """Slow VTI shales (V_P ~ 2500 m/s) get a larger correction
    (~1.10-1.20x) than fast carbonates (V_P ~ 6000 m/s, ~1.02x)."""
    from fwap.anisotropy import (
        stoneley_horizontal_shear_modulus,
        stoneley_horizontal_shear_modulus_corrected,
    )

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / 1.0e10)
    ratios = []
    for Vp in (2500.0, 3500.0, 4500.0, 6000.0):
        s_p = 1.0 / Vp
        c66_white = stoneley_horizontal_shear_modulus(
            s_st, rho_fluid=rho_f, v_fluid=v_f
        )
        c66_corr = stoneley_horizontal_shear_modulus_corrected(
            slowness_stoneley=s_st,
            rho=rho,
            slowness_p=s_p,
            rho_fluid=rho_f,
            v_fluid=v_f,
        )
        ratios.append(c66_corr / c66_white)
    # Slower formations get larger correction factors.
    for r1, r2 in zip(ratios[:-1], ratios[1:]):
        assert r1 > r2
    # 2500 m/s shale: 5-25 % correction; 6000 m/s carbonate: 1-3 %.
    assert 1.05 < ratios[0] < 1.25
    assert 1.005 < ratios[-1] < 1.05


def test_stoneley_c66_corrected_vector_inputs_broadcast():
    """Per-depth inputs broadcast to per-depth outputs."""
    from fwap.anisotropy import stoneley_horizontal_shear_modulus_corrected

    n = 4
    rho_f, v_f = 1000.0, 1500.0
    rho = np.full(n, 2400.0)
    Vp = np.array([3000.0, 3500.0, 4000.0, 4500.0])
    c66_planted = np.linspace(1.0e10, 2.0e10, n)
    factor = 1.0 - rho_f * v_f**2 / (rho * Vp**2)
    c66_eff = c66_planted * factor
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / c66_eff)
    s_p = 1.0 / Vp
    c66 = stoneley_horizontal_shear_modulus_corrected(
        slowness_stoneley=s_st, rho=rho, slowness_p=s_p, rho_fluid=rho_f, v_fluid=v_f
    )
    np.testing.assert_allclose(c66, c66_planted, rtol=1.0e-12)


def test_stoneley_c66_corrected_rejects_unphysical_p_modulus():
    """rho V_P^2 <= rho_f V_f^2 makes the correction factor non-positive
    -- rejected explicitly with a named error."""
    import pytest

    from fwap.anisotropy import stoneley_horizontal_shear_modulus_corrected

    rho_f, v_f = 1000.0, 1500.0
    # Choose Vp so that rho*Vp^2 == rho_f*Vf^2 exactly => factor = 0.
    rho = 1000.0
    Vp = 1500.0
    s_p = 1.0 / Vp
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / 5.0e9)
    with pytest.raises(ValueError, match="P-wave modulus"):
        stoneley_horizontal_shear_modulus_corrected(
            slowness_stoneley=s_st,
            rho=rho,
            slowness_p=s_p,
            rho_fluid=rho_f,
            v_fluid=v_f,
        )


def test_stoneley_c66_corrected_rejects_non_positive_inputs():
    """All slownesses + densities + fluid params must be positive."""
    import pytest

    from fwap.anisotropy import stoneley_horizontal_shear_modulus_corrected

    base = dict(
        slowness_stoneley=8.0e-4,
        rho=2400.0,
        slowness_p=2.0e-4,
        rho_fluid=1000.0,
        v_fluid=1500.0,
    )
    with pytest.raises(ValueError, match="rho_fluid"):
        stoneley_horizontal_shear_modulus_corrected(**{**base, "rho_fluid": 0.0})
    with pytest.raises(ValueError, match="v_fluid"):
        stoneley_horizontal_shear_modulus_corrected(**{**base, "v_fluid": -1.0})
    with pytest.raises(ValueError, match="rho"):
        stoneley_horizontal_shear_modulus_corrected(**{**base, "rho": 0.0})
    with pytest.raises(ValueError, match="slowness_p"):
        stoneley_horizontal_shear_modulus_corrected(**{**base, "slowness_p": -1.0})


def test_vti_moduli_default_uses_corrected_c66():
    """vti_moduli_from_logs(...) defaults to the corrected C66; the
    gamma it returns matches the corrected helper, not the White one."""
    from fwap.anisotropy import (
        stoneley_horizontal_shear_modulus_corrected,
        thomsen_gamma,
        vti_moduli_from_logs,
    )

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs = 4500.0, 2500.0
    c44_planted = rho * Vs**2
    c66_planted = 1.3 * c44_planted
    factor = 1.0 - rho_f * v_f**2 / (rho * Vp**2)
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / (c66_planted * factor))
    out = vti_moduli_from_logs(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    np.testing.assert_allclose(out.c66, c66_planted, rtol=1.0e-12)
    np.testing.assert_allclose(
        out.gamma, thomsen_gamma(c44_planted, c66_planted), rtol=1.0e-12
    )


def test_vti_moduli_correct_for_p_modulus_false_matches_white_helper():
    """The legacy uncorrected mode (correct_for_p_modulus=False) gives
    exactly the same gamma as thomsen_gamma_from_logs."""
    import pytest

    from fwap.anisotropy import (
        thomsen_gamma_from_logs,
        vti_moduli_from_logs,
    )

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs = 4500.0, 2500.0
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / (1.3 * rho * Vs**2))
    out_white = vti_moduli_from_logs(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
        correct_for_p_modulus=False,
    )
    ref = thomsen_gamma_from_logs(
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    assert out_white.c44 == pytest.approx(ref.c44, rel=1.0e-12)
    assert out_white.c66 == pytest.approx(ref.c66, rel=1.0e-12)
    assert out_white.gamma == pytest.approx(ref.gamma, rel=1.0e-12)


def test_vti_moduli_corrected_and_white_diverge_for_typical_inputs():
    """Corrected gamma > White gamma by ~5-10 % for typical sandstone."""
    import pytest

    from fwap.anisotropy import vti_moduli_from_logs

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs = 4500.0, 2500.0
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / (1.3 * rho * Vs**2))
    common = dict(
        slowness_p=1.0 / Vp,
        slowness_dipole=1.0 / Vs,
        slowness_stoneley=s_st,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    corr = vti_moduli_from_logs(correct_for_p_modulus=True, **common)
    whte = vti_moduli_from_logs(correct_for_p_modulus=False, **common)
    # Corrected C66 / White C66 = 1 / (1 - rho_f V_f^2 / (rho V_P^2))
    expected_ratio = 1.0 / (1.0 - rho_f * v_f**2 / (rho * Vp**2))
    assert corr.c66 / whte.c66 == pytest.approx(expected_ratio, rel=1.0e-12)
    # gamma is monotonic in C66 at fixed C44 -> corrected gamma is
    # also greater than White gamma here.
    assert corr.gamma > whte.gamma


# ---------------------------------------------------------------------
# Walkaway-VSP slowness-polarization inversion (Tier 2 VTI)
# ---------------------------------------------------------------------


def _synth_walkaway_vsp(
    theta_deg,
    vp0,
    epsilon,
    delta,
    *,
    polarization_noise_rad=0.0,
    slowness_noise_rel=0.0,
    seed=0,
):
    """Build (slowness_vectors, polarization_vectors) for a list of
    phase angles via the Thomsen weak-anisotropy forward formulas."""
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    sin2_t = np.sin(theta) ** 2
    cos2_t = np.cos(theta) ** 2
    # Phase velocity (Thomsen weak-anisotropy P-wave).
    v_phase = vp0 * (1.0 + delta * sin2_t * cos2_t + epsilon * sin2_t**2)
    # Polarization-deviation angle: psi_u - theta = eps sin(2t)
    #                                              + (delta - eps) sin(4t) / 2
    sin_2t = np.sin(2.0 * theta)
    sin_4t = np.sin(4.0 * theta)
    psi_u = theta + (epsilon * sin_2t + 0.5 * (delta - epsilon) * sin_4t)

    # Slowness vector p = (1/V) * (sin theta, cos theta).
    p = np.column_stack([np.sin(theta) / v_phase, np.cos(theta) / v_phase])
    # Polarization unit vector u = (sin psi_u, cos psi_u).
    u = np.column_stack([np.sin(psi_u), np.cos(psi_u)])

    if slowness_noise_rel > 0:
        p = p * (1.0 + slowness_noise_rel * rng.standard_normal(p.shape))
    if polarization_noise_rad > 0:
        # Rotate each polarization vector by a small angle.
        dpsi = polarization_noise_rad * rng.standard_normal(theta.size)
        u_rot = np.column_stack(
            [
                u[:, 0] * np.cos(dpsi) + u[:, 1] * np.sin(dpsi),
                -u[:, 0] * np.sin(dpsi) + u[:, 1] * np.cos(dpsi),
            ]
        )
        u = u_rot
    return p, u


def test_thomsen_eps_delta_round_trips_through_forward_model():
    """Plant epsilon / delta / V_P0; build synthetic walkaway VSP via
    the forward formulas; recover epsilon and delta to floating-point
    precision in the noise-free case."""
    import pytest

    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    vp0 = 4500.0
    eps_truth, delta_truth = 0.15, 0.08
    theta_deg = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    p, u = _synth_walkaway_vsp(theta_deg, vp0, eps_truth, delta_truth)
    res = thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=vp0)
    assert res.epsilon == pytest.approx(eps_truth, rel=1.0e-10)
    assert res.delta == pytest.approx(delta_truth, rel=1.0e-10)
    assert res.vp0 == vp0
    assert res.n_shots == 5
    assert res.residual_rms < 1.0e-12


def test_thomsen_eps_delta_recovers_isotropic():
    """epsilon = delta = 0 round-trips on an isotropic synthetic."""
    import pytest

    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    vp0 = 4500.0
    theta_deg = np.array([10.0, 20.0, 30.0, 40.0])
    p, u = _synth_walkaway_vsp(theta_deg, vp0, 0.0, 0.0)
    res = thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=vp0)
    assert res.epsilon == pytest.approx(0.0, abs=1.0e-12)
    assert res.delta == pytest.approx(0.0, abs=1.0e-12)


def test_thomsen_eps_delta_under_noise_recovers_within_tolerance():
    """Modest synthetic noise (1 % slowness, 0.5 deg polarization)
    still recovers epsilon and delta to within 0.02."""
    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    vp0 = 4500.0
    eps_truth, delta_truth = 0.15, 0.08
    theta_deg = np.linspace(5.0, 50.0, 20)
    p, u = _synth_walkaway_vsp(
        theta_deg,
        vp0,
        eps_truth,
        delta_truth,
        slowness_noise_rel=0.01,
        polarization_noise_rad=np.deg2rad(0.5),
        seed=42,
    )
    res = thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=vp0)
    assert abs(res.epsilon - eps_truth) < 0.02
    assert abs(res.delta - delta_truth) < 0.02
    assert res.residual_rms < 0.1


def test_thomsen_eps_delta_separates_eps_and_delta():
    """Plant epsilon high / delta low and the inversion separates them
    rather than mixing -- the velocity equation alone has the
    sin^4 / sin^2 cos^2 angular dependence that breaks degeneracy
    above ~30 degrees."""
    import pytest

    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    vp0 = 4500.0
    theta_deg = np.array([10.0, 25.0, 40.0, 55.0])
    p, u = _synth_walkaway_vsp(theta_deg, vp0, epsilon=0.20, delta=-0.05)
    res = thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=vp0)
    assert res.epsilon == pytest.approx(0.20, rel=1.0e-10)
    assert res.delta == pytest.approx(-0.05, rel=1.0e-9)


def test_thomsen_eps_delta_rejects_non_positive_vp0():
    import pytest

    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    p = np.array([[0.1, 0.9]]) / 4500.0
    u = np.array([[0.1, 0.9]])
    with pytest.raises(ValueError, match="vp0"):
        thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=0.0)
    with pytest.raises(ValueError, match="vp0"):
        thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=-1.0)


def test_thomsen_eps_delta_rejects_misshaped_inputs():
    import pytest

    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    # Wrong second-dim size.
    with pytest.raises(ValueError, match="slowness_vectors"):
        thomsen_epsilon_delta_from_walkaway_vsp(
            np.zeros((3, 3)), np.zeros((3, 3)), vp0=4500.0
        )
    # Slowness and polarization shapes disagree.
    with pytest.raises(ValueError, match="polarization_vectors"):
        thomsen_epsilon_delta_from_walkaway_vsp(
            np.array([[1.0, 1.0]]) / 4500.0,
            np.array([[1.0, 1.0], [0.5, 0.5]]),
            vp0=4500.0,
        )


def test_thomsen_eps_delta_rejects_zero_vectors():
    import pytest

    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    with pytest.raises(ValueError, match="slowness vector"):
        thomsen_epsilon_delta_from_walkaway_vsp(
            np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), vp0=4500.0
        )
    with pytest.raises(ValueError, match="polarization vector"):
        thomsen_epsilon_delta_from_walkaway_vsp(
            np.array([[1.0e-4, 9.0e-5]]), np.array([[0.0, 0.0]]), vp0=4500.0
        )


def test_thomsen_eps_delta_polarization_magnitude_does_not_matter():
    """Scaling polarization_vectors by an arbitrary positive factor
    leaves the result unchanged (only the direction enters)."""
    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    vp0 = 4500.0
    theta_deg = np.array([10.0, 20.0, 30.0, 40.0])
    p, u = _synth_walkaway_vsp(theta_deg, vp0, 0.18, 0.10)
    res_unit = thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=vp0)
    res_scaled = thomsen_epsilon_delta_from_walkaway_vsp(p, 17.3 * u, vp0=vp0)
    np.testing.assert_allclose(res_scaled.epsilon, res_unit.epsilon, rtol=1.0e-12)
    np.testing.assert_allclose(res_scaled.delta, res_unit.delta, rtol=1.0e-12)


def test_thomsen_eps_delta_minimum_two_shots_exactly_determined():
    """Two shots give an exactly-determined 4x2 system; solution
    matches the truth and residual_rms ~ 0."""
    import pytest

    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    vp0 = 4500.0
    theta_deg = np.array([15.0, 35.0])
    p, u = _synth_walkaway_vsp(theta_deg, vp0, 0.12, 0.04)
    res = thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=vp0)
    assert res.epsilon == pytest.approx(0.12, rel=1.0e-10)
    assert res.delta == pytest.approx(0.04, rel=1.0e-10)
    assert res.residual_rms < 1.0e-12


def test_thomsen_eps_delta_preserves_n_shots():
    from fwap.anisotropy import thomsen_epsilon_delta_from_walkaway_vsp

    vp0 = 4500.0
    theta_deg = np.linspace(10.0, 50.0, 7)
    p, u = _synth_walkaway_vsp(theta_deg, vp0, 0.15, 0.08)
    res = thomsen_epsilon_delta_from_walkaway_vsp(p, u, vp0=vp0)
    assert res.n_shots == 7


# =====================================================================
# Backus (1962) layered-medium averaging
# =====================================================================


def _berea_layer():
    """Berea-sandstone-like single layer."""
    return dict(thickness=2.0, vp=3500.0, vs=2000.0, rho=2200.0)


# ---------------------------------------------------------------------
# Isotropic limit: single layer == per-layer moduli
# ---------------------------------------------------------------------


def test_backus_single_layer_recovers_isotropic_moduli():
    """One-layer input must give back the layer's isotropic Lame
    moduli: c11 = c33 = lambda + 2 mu; c44 = c66 = mu; c13 = lambda."""
    from fwap.anisotropy import backus_average

    layer = _berea_layer()
    out = backus_average(
        thickness=np.array([layer["thickness"]]),
        vp=np.array([layer["vp"]]),
        vs=np.array([layer["vs"]]),
        rho=np.array([layer["rho"]]),
    )
    mu = layer["rho"] * layer["vs"] ** 2
    M = layer["rho"] * layer["vp"] ** 2
    lam = M - 2.0 * mu

    assert abs(out.c11 - M) / M < 1.0e-12
    assert abs(out.c33 - M) / M < 1.0e-12
    assert abs(out.c13 - lam) / lam < 1.0e-12
    assert abs(out.c44 - mu) / mu < 1.0e-12
    assert abs(out.c66 - mu) / mu < 1.0e-12
    assert out.rho == layer["rho"]


def test_backus_uniform_stack_equals_single_layer():
    """A stack of N identical layers equals the single-layer result
    -- the volume averages all collapse to the layer's value."""
    from fwap.anisotropy import backus_average

    layer = _berea_layer()
    n = 7
    out_uniform = backus_average(
        thickness=np.full(n, layer["thickness"]),
        vp=np.full(n, layer["vp"]),
        vs=np.full(n, layer["vs"]),
        rho=np.full(n, layer["rho"]),
    )
    out_single = backus_average(
        thickness=np.array([layer["thickness"]]),
        vp=np.array([layer["vp"]]),
        vs=np.array([layer["vs"]]),
        rho=np.array([layer["rho"]]),
    )
    for field in ("c11", "c13", "c33", "c44", "c66", "rho"):
        a = getattr(out_uniform, field)
        b = getattr(out_single, field)
        assert abs(a - b) / max(abs(b), 1.0) < 1.0e-12, field


def test_backus_thickness_scale_is_irrelevant():
    """Multiplying every thickness by a constant doesn't change the
    averages -- only volume *fractions* matter, not absolute scale."""
    from fwap.anisotropy import backus_average

    h = np.array([1.0, 0.5, 2.0])
    vp = np.array([3500.0, 2500.0, 4000.0])
    vs = np.array([2000.0, 1200.0, 2400.0])
    rho = np.array([2200.0, 2300.0, 2400.0])

    out_a = backus_average(thickness=h, vp=vp, vs=vs, rho=rho)
    out_b = backus_average(thickness=100.0 * h, vp=vp, vs=vs, rho=rho)
    for field in ("c11", "c13", "c33", "c44", "c66", "rho"):
        a = getattr(out_a, field)
        b = getattr(out_b, field)
        assert abs(a - b) / max(abs(b), 1.0) < 1.0e-12, field


# ---------------------------------------------------------------------
# Voigt-Reuss inequalities and induced anisotropy
# ---------------------------------------------------------------------


def test_backus_two_layer_produces_positive_thomsen_gamma():
    """Backus-averaging two layers with different shear moduli must
    give c66 > c44 (Voigt-Reuss inequality), i.e. Thomsen
    gamma > 0. Equality only in the degenerate uniform-layer case."""
    from fwap.anisotropy import backus_average, thomsen_gamma

    out = backus_average(
        thickness=np.array([1.0, 1.0]),
        vp=np.array([3500.0, 2500.0]),
        vs=np.array([2000.0, 1200.0]),
        rho=np.array([2200.0, 2300.0]),
    )
    assert out.c66 > out.c44
    gamma = thomsen_gamma(out.c44, out.c66)
    assert gamma > 0.0


def test_backus_horizontal_p_modulus_at_least_vertical():
    """Layered medium has c11 >= c33 always: the horizontal P-mode
    sees the stiffer-bulk-modulus path. Equality only in the
    uniform-layer case."""
    from fwap.anisotropy import backus_average

    out = backus_average(
        thickness=np.array([1.0, 1.0]),
        vp=np.array([3500.0, 2500.0]),
        vs=np.array([2000.0, 1200.0]),
        rho=np.array([2200.0, 2300.0]),
    )
    assert out.c11 >= out.c33


def test_backus_density_is_arithmetic_volume_average():
    """rho_eff = sum(phi_i * rho_i) for thickness-weighted phi_i."""
    from fwap.anisotropy import backus_average

    h = np.array([2.0, 1.0, 1.0])
    rho = np.array([2200.0, 2400.0, 2300.0])
    expected = np.sum(h * rho) / np.sum(h)

    out = backus_average(
        thickness=h,
        vp=np.array([3500.0, 4000.0, 3000.0]),
        vs=np.array([2000.0, 2400.0, 1500.0]),
        rho=rho,
    )
    assert abs(out.rho - expected) < 1.0e-9


# ---------------------------------------------------------------------
# Hand-derived two-layer numerical check
# ---------------------------------------------------------------------


def test_backus_two_layer_hand_derived_values():
    """Hand-computed two-layer case: equal volumes of layer A
    (mu_A = 1e10 Pa, M_A = 3e10 Pa, lam_A = 1e10) and layer B
    (mu_B = 4e9 Pa, M_B = 2e10 Pa, lam_B = 1.2e10)."""
    from fwap.anisotropy import backus_average

    rho_a, mu_a, M_a = 2500.0, 1.0e10, 3.0e10
    rho_b, mu_b, M_b = 2000.0, 4.0e9, 2.0e10
    vs_a = np.sqrt(mu_a / rho_a)
    vp_a = np.sqrt(M_a / rho_a)
    vs_b = np.sqrt(mu_b / rho_b)
    vp_b = np.sqrt(M_b / rho_b)
    lam_a = M_a - 2.0 * mu_a  # 1e10
    lam_b = M_b - 2.0 * mu_b  # 1.2e10

    out = backus_average(
        thickness=np.array([1.0, 1.0]),
        vp=np.array([vp_a, vp_b]),
        vs=np.array([vs_a, vs_b]),
        rho=np.array([rho_a, rho_b]),
    )

    avg_inv_M = 0.5 * (1.0 / M_a + 1.0 / M_b)
    avg_inv_mu = 0.5 * (1.0 / mu_a + 1.0 / mu_b)
    avg_lam_over_M = 0.5 * (lam_a / M_a + lam_b / M_b)
    avg_M_minus_lam2_over_M = 0.5 * (M_a - lam_a**2 / M_a + M_b - lam_b**2 / M_b)
    avg_mu = 0.5 * (mu_a + mu_b)

    c33_expected = 1.0 / avg_inv_M
    c13_expected = avg_lam_over_M / avg_inv_M
    c11_expected = avg_M_minus_lam2_over_M + avg_lam_over_M**2 / avg_inv_M
    c44_expected = 1.0 / avg_inv_mu
    c66_expected = avg_mu

    assert abs(out.c33 - c33_expected) / c33_expected < 1.0e-9
    assert abs(out.c13 - c13_expected) / c13_expected < 1.0e-9
    assert abs(out.c11 - c11_expected) / c11_expected < 1.0e-9
    assert abs(out.c44 - c44_expected) / c44_expected < 1.0e-9
    assert abs(out.c66 - c66_expected) / c66_expected < 1.0e-9


# ---------------------------------------------------------------------
# Positive-definiteness of the resulting tensor
# ---------------------------------------------------------------------


def test_backus_result_satisfies_positive_definite_tensor():
    """The 6x6 Voigt elastic matrix of the result must be positive
    definite -- a thermodynamic stability requirement for any
    physically realisable medium. For a VTI tensor with
    {C11, C13, C33, C44, C66}, this reduces to:
    C44 > 0, C66 > 0, C33 > 0, C11 > 0, and the determinant of the
    {C11, C13; C13, C33} 2x2 sub-block is positive."""
    from fwap.anisotropy import backus_average

    rng = np.random.default_rng(seed=42)
    n_layers = 5
    rho = rng.uniform(2000.0, 2600.0, n_layers)
    vs = rng.uniform(1000.0, 2500.0, n_layers)
    vp = vs * rng.uniform(1.5, 2.5, n_layers)
    h = rng.uniform(0.1, 5.0, n_layers)

    out = backus_average(thickness=h, vp=vp, vs=vs, rho=rho)
    assert out.c44 > 0
    assert out.c66 > 0
    assert out.c33 > 0
    assert out.c11 > 0
    assert out.c11 * out.c33 - out.c13**2 > 0


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_backus_rejects_empty_input():
    """Empty input arrays raise ValueError -- nothing to average."""
    import pytest

    from fwap.anisotropy import backus_average

    with pytest.raises(ValueError, match="at least one layer"):
        backus_average(
            thickness=np.array([]),
            vp=np.array([]),
            vs=np.array([]),
            rho=np.array([]),
        )


def test_backus_rejects_shape_mismatch():
    """Mismatched array lengths raise ValueError."""
    import pytest

    from fwap.anisotropy import backus_average

    with pytest.raises(ValueError, match="same length"):
        backus_average(
            thickness=np.array([1.0, 1.0]),
            vp=np.array([3500.0]),
            vs=np.array([2000.0]),
            rho=np.array([2200.0]),
        )


def test_backus_rejects_non_positive_inputs():
    """Zero or negative thickness, vp, vs, or rho raises."""
    import pytest

    from fwap.anisotropy import backus_average

    base = dict(
        thickness=np.array([1.0]),
        vp=np.array([3500.0]),
        vs=np.array([2000.0]),
        rho=np.array([2200.0]),
    )
    with pytest.raises(ValueError, match="thickness"):
        backus_average(**{**base, "thickness": np.array([0.0])})
    with pytest.raises(ValueError, match="vp, vs, rho"):
        backus_average(**{**base, "vp": np.array([0.0])})
    with pytest.raises(ValueError, match="vp, vs, rho"):
        backus_average(**{**base, "vs": np.array([-1.0])})
    with pytest.raises(ValueError, match="vp, vs, rho"):
        backus_average(**{**base, "rho": np.array([0.0])})


def test_backus_rejects_vs_ge_vp():
    """vs >= vp on any layer is unphysical (lambda + 2 mu would not
    be positive); raise rather than produce garbage."""
    import pytest

    from fwap.anisotropy import backus_average

    with pytest.raises(ValueError, match="vs < vp"):
        backus_average(
            thickness=np.array([1.0]),
            vp=np.array([2000.0]),
            vs=np.array([2500.0]),  # vs > vp
            rho=np.array([2200.0]),
        )


# =====================================================================
# vti_phase_velocities (Tsvankin 2001 eq. 1.41)
# =====================================================================


def _berea_vti():
    """Backus-derived VTI elastic constants (Pa) from a synthetic
    shale/sand alternation. The qP/qSV/SH velocities computed from
    these constants exhibit the standard VTI features (qP epsilon
    > 0, qSV bulge near 45 deg, SH gamma > 0)."""
    return dict(
        c11=2.063e10,
        c13=8.307e9,
        c33=1.875e10,
        c44=4.813e9,
        c66=6.056e9,
        rho=2250.0,
    )


# ---------------------------------------------------------------------
# Vertical and horizontal limits
# ---------------------------------------------------------------------


def test_vti_phase_at_vertical_recovers_axial_moduli():
    """At theta = 0, v_qP = sqrt(C33/rho), and v_qSV = v_SH =
    sqrt(C44/rho) (the vertical-shear degeneracy)."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    vP, vSV, vSH = vti_phase_velocities(**p, phase_angle_rad=np.array([0.0]))
    assert abs(float(vP[0]) - np.sqrt(p["c33"] / p["rho"])) < 1.0e-6
    assert abs(float(vSV[0]) - np.sqrt(p["c44"] / p["rho"])) < 1.0e-6
    assert abs(float(vSH[0]) - np.sqrt(p["c44"] / p["rho"])) < 1.0e-6


def test_vti_phase_at_horizontal_recovers_in_plane_moduli():
    """At theta = pi/2, v_qP = sqrt(C11/rho), v_qSV = sqrt(C44/rho),
    v_SH = sqrt(C66/rho)."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    vP, vSV, vSH = vti_phase_velocities(
        **p,
        phase_angle_rad=np.array([np.pi / 2]),
    )
    assert abs(float(vP[0]) - np.sqrt(p["c11"] / p["rho"])) < 1.0e-6
    assert abs(float(vSV[0]) - np.sqrt(p["c44"] / p["rho"])) < 1.0e-6
    assert abs(float(vSH[0]) - np.sqrt(p["c66"] / p["rho"])) < 1.0e-6


# ---------------------------------------------------------------------
# Isotropic limit: all three velocities are constant in theta
# ---------------------------------------------------------------------


def test_vti_phase_isotropic_limit_constant_in_angle():
    """An isotropic medium has C11 = C33, C44 = C66, C13 = C11 -
    2*C44. All three phase velocities then become constant in theta;
    qSV and SH are equal (S-wave isotropy)."""
    from fwap.anisotropy import vti_phase_velocities

    mu = 8.0e9
    M = 27.0e9
    lam = M - 2.0 * mu
    rho = 2200.0
    theta = np.linspace(0.0, np.pi / 2, 13)
    vP, vSV, vSH = vti_phase_velocities(
        c11=M,
        c13=lam,
        c33=M,
        c44=mu,
        c66=mu,
        rho=rho,
        phase_angle_rad=theta,
    )
    np.testing.assert_allclose(vP, np.sqrt(M / rho), rtol=1.0e-12)
    np.testing.assert_allclose(vSV, np.sqrt(mu / rho), rtol=1.0e-12)
    np.testing.assert_allclose(vSH, np.sqrt(mu / rho), rtol=1.0e-12)


# ---------------------------------------------------------------------
# Anisotropy signatures
# ---------------------------------------------------------------------


def test_vti_phase_v_qSV_equals_v_SH_at_vertical():
    """The vertical S-wave is degenerate (qSV polarisation is in
    the propagation plane, SH polarisation is perpendicular but
    both see the same C44 stiffness for vertical propagation)."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    _, vSV, vSH = vti_phase_velocities(**p, phase_angle_rad=np.array([0.0]))
    assert abs(float(vSV[0]) - float(vSH[0])) < 1.0e-6


def test_vti_phase_v_qSV_equals_v_qSV_at_horizontal_for_C44():
    """At pi/2, qSV propagates with vertical-shear stiffness C44
    (its polarisation direction at horizontal propagation is the
    vertical x_3 axis), independent of C66."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    _, vSV, _ = vti_phase_velocities(
        **p,
        phase_angle_rad=np.array([np.pi / 2]),
    )
    assert abs(float(vSV[0]) - np.sqrt(p["c44"] / p["rho"])) < 1.0e-6


def test_vti_phase_v_SH_increases_when_C66_larger_than_C44():
    """Positive Thomsen gamma (C66 > C44) means v_SH(pi/2) >
    v_SH(0). The Berea-VTI test fixture has gamma > 0."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    _, _, vSH = vti_phase_velocities(
        **p,
        phase_angle_rad=np.array([0.0, np.pi / 2]),
    )
    # gamma > 0 means the horizontal SH is faster than the vertical.
    assert float(vSH[1]) > float(vSH[0])


def test_vti_phase_v_qP_at_horizontal_above_vertical_for_positive_epsilon():
    """C11 > C33 (positive Thomsen epsilon) means horizontal qP is
    faster than vertical qP. The Berea-VTI fixture has epsilon > 0."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    vP, _, _ = vti_phase_velocities(
        **p,
        phase_angle_rad=np.array([0.0, np.pi / 2]),
    )
    assert float(vP[1]) > float(vP[0])


# ---------------------------------------------------------------------
# Output shape and broadcasting
# ---------------------------------------------------------------------


def test_vti_phase_output_shapes_match_input_angle_grid():
    """Each of the three velocity arrays has the same shape as the
    input phase_angle_rad grid."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 91)
    vP, vSV, vSH = vti_phase_velocities(**p, phase_angle_rad=theta)
    assert vP.shape == theta.shape
    assert vSV.shape == theta.shape
    assert vSH.shape == theta.shape


def test_vti_phase_scalar_input_returns_scalar_output():
    """Scalar input produces scalar output (numpy 0-d arrays)."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    vP, vSV, vSH = vti_phase_velocities(**p, phase_angle_rad=0.5)
    # Result of np.sqrt on a 0-d array is a 0-d array.
    assert vP.ndim == 0
    assert vSV.ndim == 0
    assert vSH.ndim == 0


# ---------------------------------------------------------------------
# Round-trip with Backus
# ---------------------------------------------------------------------


def test_vti_phase_consumes_backus_output_directly():
    """Run Backus on a layered stack, feed the result into
    vti_phase_velocities, confirm the velocity surfaces are
    well-defined and have the expected anisotropy signatures."""
    from fwap.anisotropy import backus_average, vti_phase_velocities

    out = backus_average(
        thickness=np.array([1.0, 1.0]),
        vp=np.array([3500.0, 2500.0]),
        vs=np.array([2000.0, 1200.0]),
        rho=np.array([2200.0, 2300.0]),
    )
    theta = np.linspace(0.0, np.pi / 2, 19)
    vP, vSV, vSH = vti_phase_velocities(
        c11=out.c11,
        c13=out.c13,
        c33=out.c33,
        c44=out.c44,
        c66=out.c66,
        rho=out.rho,
        phase_angle_rad=theta,
    )
    # All velocities are real and positive.
    assert np.all(np.isfinite(vP))
    assert np.all(np.isfinite(vSV))
    assert np.all(np.isfinite(vSH))
    assert np.all(vP > vSV)  # qP always faster than qSV in stable VTI
    # Expected ordering: SH and qSV degenerate at vertical, separate
    # for theta > 0 with v_SH > v_qSV (positive Thomsen gamma).
    assert vSH[-1] > vSV[-1]


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_vti_phase_rejects_non_positive_density():
    """Zero or negative density raises."""
    import pytest

    from fwap.anisotropy import vti_phase_velocities

    with pytest.raises(ValueError, match="rho"):
        vti_phase_velocities(
            c11=2e10,
            c13=8e9,
            c33=2e10,
            c44=5e9,
            c66=6e9,
            rho=0.0,
            phase_angle_rad=np.array([0.0]),
        )


def test_vti_phase_rejects_non_positive_elastic_constants():
    """Zero or negative c11/c33/c44/c66 raise (c13 is allowed
    negative in degenerate cases, but negative diagonal moduli
    violate physical-positivity constraints)."""
    import pytest

    from fwap.anisotropy import vti_phase_velocities

    base = dict(
        c11=2e10,
        c13=8e9,
        c33=2e10,
        c44=5e9,
        c66=6e9,
        rho=2400.0,
        phase_angle_rad=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="c11"):
        vti_phase_velocities(**{**base, "c11": 0.0})
    with pytest.raises(ValueError, match="c33"):
        vti_phase_velocities(**{**base, "c33": -1.0})
    with pytest.raises(ValueError, match="c44"):
        vti_phase_velocities(**{**base, "c44": 0.0})
    with pytest.raises(ValueError, match="c66"):
        vti_phase_velocities(**{**base, "c66": -1.0})


# =====================================================================
# vti_group_velocities (numerical differentiation of phase velocity)
# =====================================================================


# ---------------------------------------------------------------------
# Isotropic limit: group velocity exactly equals phase velocity
# ---------------------------------------------------------------------


def test_vti_group_isotropic_limit_equals_phase():
    """In an isotropic medium dv_p/dtheta = 0, so group velocity
    equals phase velocity at every angle and group angle equals
    phase angle. Numerical differentiation reproduces this to
    floating-point precision."""
    from fwap.anisotropy import vti_group_velocities, vti_phase_velocities

    mu, M, rho = 8.0e9, 27.0e9, 2200.0
    lam = M - 2.0 * mu
    theta = np.linspace(0.0, np.pi / 2, 19)

    vP_p, vSV_p, vSH_p = vti_phase_velocities(
        c11=M,
        c13=lam,
        c33=M,
        c44=mu,
        c66=mu,
        rho=rho,
        phase_angle_rad=theta,
    )
    out = vti_group_velocities(
        c11=M,
        c13=lam,
        c33=M,
        c44=mu,
        c66=mu,
        rho=rho,
        phase_angle_rad=theta,
    )
    np.testing.assert_allclose(out.v_qP, vP_p, rtol=1.0e-10)
    np.testing.assert_allclose(out.v_qSV, vSV_p, rtol=1.0e-10)
    np.testing.assert_allclose(out.v_SH, vSH_p, rtol=1.0e-10)
    np.testing.assert_allclose(out.psi_qP, theta, atol=1.0e-12)
    np.testing.assert_allclose(out.psi_qSV, theta, atol=1.0e-12)
    np.testing.assert_allclose(out.psi_SH, theta, atol=1.0e-12)


# ---------------------------------------------------------------------
# Output dataclass contract
# ---------------------------------------------------------------------


def test_vti_group_returns_VtiGroupVelocities_with_six_fields():
    """The output dataclass holds three velocities and three group
    angles, all with the same shape as the input grid."""
    from fwap.anisotropy import VtiGroupVelocities, vti_group_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 19)
    out = vti_group_velocities(**p, phase_angle_rad=theta)
    assert isinstance(out, VtiGroupVelocities)
    for name in ("v_qP", "v_qSV", "v_SH", "psi_qP", "psi_qSV", "psi_SH"):
        arr = getattr(out, name)
        assert arr.shape == theta.shape
        assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------
# Group velocity magnitude has the right shape
# ---------------------------------------------------------------------


def test_vti_group_qP_velocity_close_to_phase_at_endpoints():
    """At theta = 0 and pi/2 the wavefront is locally aligned with
    a symmetry direction so v_g should be very close to v_p (small
    numerical-differentiation boundary error). Spot-check a 1-deg
    grid: 0.5% tolerance is well within the boundary one-sided
    difference."""
    from fwap.anisotropy import vti_group_velocities, vti_phase_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 91)  # 1-deg grid

    vP_p, _, _ = vti_phase_velocities(**p, phase_angle_rad=theta)
    out = vti_group_velocities(**p, phase_angle_rad=theta)

    rel_err_0 = abs(out.v_qP[0] - vP_p[0]) / vP_p[0]
    rel_err_pi2 = abs(out.v_qP[-1] - vP_p[-1]) / vP_p[-1]
    assert rel_err_0 < 5.0e-3
    assert rel_err_pi2 < 5.0e-3


def test_vti_group_psi_at_theta_zero_is_zero():
    """At vertical phase propagation, energy flows along the
    symmetry axis, so the group angle is zero (psi(0) = 0)."""
    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 91)
    out = vti_group_velocities(**p, phase_angle_rad=theta)
    # Boundary one-sided difference at theta=0 may give a small
    # nonzero psi; allow a 1-deg tolerance on the 1-deg grid.
    assert abs(out.psi_qP[0]) < np.deg2rad(1.0)
    assert abs(out.psi_SH[0]) < np.deg2rad(1.0)


def test_vti_group_psi_at_theta_pi_over_2_is_pi_over_2():
    """At horizontal phase propagation, energy also flows along a
    symmetry direction (the x_1 axis), so the group angle equals
    the phase angle pi/2."""
    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 91)
    out = vti_group_velocities(**p, phase_angle_rad=theta)
    assert abs(out.psi_qP[-1] - np.pi / 2) < np.deg2rad(1.0)
    assert abs(out.psi_SH[-1] - np.pi / 2) < np.deg2rad(1.0)


# ---------------------------------------------------------------------
# Anisotropy signature: group angle differs from phase angle off-axis
# ---------------------------------------------------------------------


def test_vti_group_psi_differs_from_phase_for_anisotropic_medium():
    """At an interior phase angle (e.g. theta=pi/4) the group and
    phase angles differ in any anisotropic medium. Magnitude of the
    difference depends on the anellipticity."""
    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 91)
    out = vti_group_velocities(**p, phase_angle_rad=theta)

    idx_45 = np.argmin(np.abs(theta - np.pi / 4))
    diff_qSV = out.psi_qSV[idx_45] - theta[idx_45]
    diff_SH = out.psi_SH[idx_45] - theta[idx_45]
    # The Berea-VTI fixture has positive gamma, so SH group angle
    # is larger than phase at theta = pi/4 (energy refracted toward
    # the faster horizontal direction).
    assert diff_SH > 0
    # qSV behaviour depends on the sign of the qSV anellipticity
    # parameter; the Berea-VTI fixture has the typical sign that
    # gives qSV group angle SMALLER than phase at theta = pi/4.
    assert diff_qSV < 0


# ---------------------------------------------------------------------
# Wavefront-shape sanity: group velocity is finite and positive
# ---------------------------------------------------------------------


def test_vti_group_velocities_strictly_positive():
    """All three group-velocity arrays must be strictly positive
    everywhere on a sensible phase-angle grid -- no zero crossings
    or near-zeros that would correspond to triplication / cuspidal
    behaviour for the Berea-VTI fixture (which is mild enough to
    avoid cusps)."""
    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 91)
    out = vti_group_velocities(**p, phase_angle_rad=theta)
    assert np.all(out.v_qP > 0)
    assert np.all(out.v_qSV > 0)
    assert np.all(out.v_SH > 0)


def test_vti_group_qP_above_qSV_everywhere():
    """qP is always faster than qSV in stable VTI media -- a
    consequence of the strong-ellipticity constraint on the
    elastic tensor."""
    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    theta = np.linspace(0.0, np.pi / 2, 91)
    out = vti_group_velocities(**p, phase_angle_rad=theta)
    assert np.all(out.v_qP > out.v_qSV)


# ---------------------------------------------------------------------
# Input validation on the phase-angle grid
# ---------------------------------------------------------------------


def test_vti_group_rejects_one_point_grid():
    """A single-point phase-angle grid cannot be differentiated."""
    import pytest

    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    with pytest.raises(ValueError, match="at least 2 points"):
        vti_group_velocities(**p, phase_angle_rad=np.array([0.0]))


def test_vti_group_rejects_non_increasing_grid():
    """Non-increasing or unsorted phase-angle grids raise."""
    import pytest

    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    with pytest.raises(ValueError, match="strictly increasing"):
        vti_group_velocities(
            **p,
            phase_angle_rad=np.array([0.5, 0.3, 0.1]),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        vti_group_velocities(
            **p,
            phase_angle_rad=np.array([0.1, 0.1, 0.5]),
        )


def test_vti_group_rejects_non_1d_grid():
    """A multi-D phase-angle array doesn't make sense for the
    underlying np.gradient call."""
    import pytest

    from fwap.anisotropy import vti_group_velocities

    p = _berea_vti()
    grid = np.array([[0.0, 0.5], [1.0, 1.5]])
    with pytest.raises(ValueError, match="1-D"):
        vti_group_velocities(**p, phase_angle_rad=grid)


# ---------------------------------------------------------------------
# Round-trip with backus_average + plotting use case
# ---------------------------------------------------------------------


def test_vti_group_wavefront_cartesian_coordinates_form_smooth_curve():
    """The natural plotting use: convert (v_g, psi) into Cartesian
    wavefront coordinates and confirm the resulting curve is
    monotonically winding around the origin with no kinks. Spot-
    check on a Backus-derived VTI medium."""
    from fwap.anisotropy import backus_average, vti_group_velocities

    b = backus_average(
        thickness=np.array([1.0, 1.0]),
        vp=np.array([3500.0, 2500.0]),
        vs=np.array([2000.0, 1200.0]),
        rho=np.array([2200.0, 2300.0]),
    )
    theta = np.linspace(0.0, np.pi / 2, 91)
    out = vti_group_velocities(
        c11=b.c11,
        c13=b.c13,
        c33=b.c33,
        c44=b.c44,
        c66=b.c66,
        rho=b.rho,
        phase_angle_rad=theta,
    )
    # Wavefront tip in Cartesian (x_1, x_3) for the qP mode.
    x = out.v_qP * np.sin(out.psi_qP)
    z = out.v_qP * np.cos(out.psi_qP)
    # x is monotonically increasing as theta sweeps 0 -> pi/2.
    assert np.all(np.diff(x) > 0)
    # z is monotonically decreasing.
    assert np.all(np.diff(z) < 0)


# ----------------------------------------------------------------------
# A.11 phase 0: where the VTI radial solve gives up, and what it costs
# ----------------------------------------------------------------------

#: Thomsen (1986) table 1, as ``(V_P0, V_S0, rho, epsilon, delta, gamma)``.
_THOMSEN_TABLE_1 = {
    "Green River shale": (3292.0, 1768.0, 2075.0, 0.195, -0.220, 0.180),
    "Mesaverde shale(5)": (3794.0, 2074.0, 2560.0, 0.189, 0.204, 0.175),
    "Mesaverde sandstone": (4529.0, 2703.0, 2460.0, 0.033, 0.040, 0.019),
    "Pierre shale": (2074.0, 869.0, 2250.0, 0.110, 0.090, 0.165),
    "Taylor sandstone": (3368.0, 1829.0, 2500.0, 0.110, -0.035, 0.255),
    "Dog Creek shale": (1875.0, 826.0, 2000.0, 0.225, 0.100, 0.345),
}


def _thomsen_stiffness(entry: tuple[float, ...]) -> dict[str, float]:
    vp0, vs0, rho, eps, delta, gamma = entry
    c33, c44 = rho * vp0**2, rho * vs0**2
    c13 = np.sqrt(max(2 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2, 0.0)) - c44
    return dict(
        c11=c33 * (1.0 + 2.0 * eps),
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c44 * (1.0 + 2.0 * gamma),
        rho=rho,
    )


def _christoffel_discriminant(kz: float, omega: float, s: dict[str, float]) -> float:
    r = s["rho"] * omega * omega
    a_eff = s["c11"] * s["c44"]
    b_eff = (s["c11"] + s["c44"]) * r - (
        s["c11"] * s["c33"] + s["c44"] * s["c44"] - (s["c13"] + s["c44"]) ** 2
    ) * kz * kz
    c_eff = s["c44"] * s["c33"] * kz**4 - (s["c44"] + s["c33"]) * r * kz * kz + r * r
    return float(b_eff * b_eff - 4.0 * a_eff * c_eff)


def test_the_vti_radial_solve_gives_up_where_the_discriminant_turns_negative():
    """Measured limitation, recorded so the fix has a target.

    ``_radial_wavenumbers_vti`` returns ``(nan, nan, nan)`` when the
    Christoffel quadratic's discriminant goes negative, on the stated
    grounds that complex roots are "not physical in the bound regime".
    Complex-conjugate ``alpha^2`` pairs are a real feature of TI media
    rather than an error state, so that early return costs a genuine
    part of the bound window -- on two of Thomsen's six table-1 media,
    most of it.

    This is A.11 phase 0. It pins the size of the gap rather than
    asserting the mode continues past it: showing that needs the
    conjugate-pair handling itself, which is phase 2, and this test is
    what phase 2 has to move.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti

    omega = 2.0 * np.pi * 8000.0
    clean, truncated = [], {}
    for name, entry in _THOMSEN_TABLE_1.items():
        s = _thomsen_stiffness(entry)
        v_sv = np.sqrt(s["c44"] / s["rho"])
        speeds = np.linspace(700.0, v_sv * 0.999, 700)
        negative = np.array(
            [_christoffel_discriminant(omega / c, omega, s) < 0.0 for c in speeds]
        )
        if not negative.any():
            clean.append(name)
            continue
        cutoff = speeds[negative].max()
        truncated[name] = (cutoff, float(negative.mean()))
        # Where the discriminant is negative the solve reports nothing ...
        below = _radial_wavenumbers_vti(omega / (cutoff * 0.99), omega, **s)
        assert all(np.isnan(x) for x in below), (name, below)
        # ... and just above it the three wavenumbers are finite and real.
        above = _radial_wavenumbers_vti(omega / (cutoff * 1.02), omega, **s)
        assert all(np.isfinite(x) and x > 0.0 for x in above), (name, above)

    assert set(truncated) == {"Mesaverde shale(5)", "Mesaverde sandstone"}, truncated
    assert len(clean) == 4, clean
    # The cost is most of the window on both, not a sliver at the edge.
    assert truncated["Mesaverde shale(5)"][1] > 0.70, truncated
    assert truncated["Mesaverde sandstone"][1] > 0.50, truncated


def test_the_bound_vti_flexural_branch_now_crosses_that_cutoff():
    """A.11 phase 3 moved this: the branch used to stop at the cutoff.

    Before phase 3 ``flexural_dispersion_vti`` descended 2071 -> 1783
    m/s over 3-6 kHz on Mesaverde shale(5) and returned ``NaN`` from
    7 kHz up, stopping just above the discriminant cutoff near 1759 and
    far above every physical edge (``V_f`` is 1500). With the conjugate
    columns recombined it runs the whole band, and the four frequencies
    that already worked are unchanged to the digit.
    """
    from fwap import flexural_dispersion_vti

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Mesaverde shale(5)"])
    freq = np.arange(3000.0, 20000.0, 1000.0)
    c = (
        1.0
        / flexural_dispersion_vti(freq, **s, vf=1500.0, rho_f=1000.0, a=0.10).slowness
    )

    assert np.isfinite(c).all(), c

    # The bound-regime values are a regression guard: phase 3 must not
    # have moved what already worked.
    for freq_hz, expected in (
        (3000.0, 2070.94),
        (4000.0, 2042.40),
        (5000.0, 1934.83),
        (6000.0, 1782.65),
    ):
        got = c[freq == freq_hz][0]
        assert abs(got - expected) < 0.01, (freq_hz, got, expected)

    # Monotone descent, with the step shrinking -- a dispersion curve,
    # not a scatter. The pre-phase-3 mechanical change (complex columns
    # through ``det(M.real)``) produced 1500 -> 1470 -> nan -> 817 ->
    # 1470 here, which this rejects.
    steps = np.diff(c)
    assert np.all(steps < 0.0), c
    assert np.all(np.diff(steps[3:]) > 0.0), steps
    # It descends past the cutoff the discriminant sets.
    omega = 2.0 * np.pi * 8000.0
    speeds = np.linspace(700.0, np.sqrt(s["c44"] / s["rho"]) * 0.999, 700)
    negative = np.array(
        [_christoffel_discriminant(omega / v, omega, s) < 0.0 for v in speeds]
    )
    assert c.min() < speeds[negative].max(), (c.min(), speeds[negative].max())


def test_the_recovered_vti_branch_is_continuous_in_the_anisotropy_homotopy():
    """The oracle for phase 3, and the reason it is trustworthy at all.

    The isotropic limit **cannot** validate this: there the Christoffel
    discriminant is identically the perfect square
    ``A^2 (p^2 - s^2)^2``, so the conjugate regime never arises and the
    oracle that anchored #131, #132 and phase 2 has nothing to say.

    What replaces it is a homotopy. Scaling Thomsen's parameters from 0
    to their Mesaverde shale(5) values sweeps the medium from isotropic
    -- where an independent solver, ``flexural_dispersion``, gives the
    answer -- to the one whose window was truncated. The conjugate
    region grows with ``t`` and swallows the mode near ``t = 0.55``, so
    before phase 3 the family broke there. The recovered branch must
    continue it *smoothly*: a different mode, or the same mode on the
    wrong branch, would show as a kink exactly at that crossing.
    """
    from fwap import flexural_dispersion, flexural_dispersion_vti

    vp0, vs0, rho = 3794.0, 2074.0, 2560.0
    eps0, delta0, gamma0 = 0.189, 0.204, 0.175
    c33, c44 = rho * vp0**2, rho * vs0**2

    def stiffness(t: float) -> dict[str, float]:
        eps, delta, gamma = eps0 * t, delta0 * t, gamma0 * t
        c13 = np.sqrt(max(2 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2, 0.0)) - c44
        return dict(
            c11=c33 * (1 + 2 * eps),
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c44 * (1 + 2 * gamma),
            rho=rho,
        )

    freq = np.array([8000.0])
    fluid = dict(vf=1500.0, rho_f=1000.0, a=0.10)
    speeds = np.array(
        [
            1.0 / flexural_dispersion_vti(freq, **stiffness(t), **fluid).slowness[0]
            for t in np.arange(0.0, 1.001, 0.05)
        ]
    )
    assert np.isfinite(speeds).all(), speeds

    # Anchored at the isotropic end by a different code path entirely.
    isotropic = (
        1.0 / flexural_dispersion(freq, vp=vp0, vs=vs0, rho=rho, **fluid).slowness[0]
    )
    assert abs(speeds[0] - isotropic) < 1e-9, (speeds[0], isotropic)

    # Smooth all the way across, including the t ~ 0.55 crossing where
    # the conjugate region takes the mode and the family used to end.
    steps = np.diff(speeds)
    assert np.all(steps > 0.0), steps
    assert np.all(np.diff(steps) < 0.0), steps
    assert np.max(np.abs(np.diff(steps))) < 0.2, steps


def test_the_isotropic_limit_cannot_reach_the_conjugate_regime():
    """Why phase 3 needed a new oracle, pinned as a fact.

    With ``c11 = c33``, ``c44 = c66`` and ``c13 = c11 - 2 c44`` the
    Christoffel quadratic factors as ``A (x - p^2)(x - s^2)``, so its
    discriminant is ``A^2 (p^2 - s^2)^2`` -- a perfect square, never
    negative. No isotropic medium at any frequency produces the
    conjugate pair, so the isotropic oracle cannot test the path phase 3
    opens.
    """
    rng = np.random.default_rng(0)
    worst, negative = 0.0, 0
    for _ in range(2000):
        rho = rng.uniform(1800.0, 2900.0)
        vs = rng.uniform(600.0, 3200.0)
        vp = vs * rng.uniform(1.45, 2.4)
        mu = rho * vs * vs
        lam = rho * vp * vp - 2 * mu
        s = dict(
            c11=lam + 2 * mu,
            c13=lam,
            c33=lam + 2 * mu,
            c44=mu,
            c66=mu,
            rho=rho,
        )
        omega = 2.0 * np.pi * rng.uniform(2000.0, 20000.0)
        kz = omega / rng.uniform(500.0, 4000.0)
        disc = _christoffel_discriminant(kz, omega, s)
        if disc < 0.0:
            negative += 1
        predicted = (
            s["c11"]
            * s["c44"]
            * ((kz**2 - (omega / vp) ** 2) - (kz**2 - (omega / vs) ** 2))
        ) ** 2
        worst = max(worst, abs(disc - predicted) / max(abs(predicted), 1e-30))
    assert negative == 0, negative
    assert worst < 1e-9, worst


# ----------------------------------------------------------------------
# A.11 phase 2: the radial solve, without the real-root restriction
#
# Nothing consumes this yet -- the row builders still cast their
# formation Bessels to float, so the 77 % / 57 % window phase 0
# measured is not recovered until phase 3 flips them. What is
# established here is that the values are right where the old solver
# gives up, checked against the governing equation rather than against
# the solver that could not produce them.
# ----------------------------------------------------------------------


def _christoffel_residual(alpha: complex, kz: complex, omega: float, s) -> float:
    """``|A x^2 + B x + C| / scale`` at ``x = alpha^2``: how well a
    returned wavenumber satisfies the quadratic it came from."""
    r = s["rho"] * omega * omega
    a_eff = s["c11"] * s["c44"]
    b_eff = (s["c11"] + s["c44"]) * r - (
        s["c11"] * s["c33"] + s["c44"] * s["c44"] - (s["c13"] + s["c44"]) ** 2
    ) * kz * kz
    c_eff = s["c44"] * s["c33"] * kz**4 - (s["c44"] + s["c33"]) * r * kz * kz + r * r
    x = complex(alpha) ** 2
    scale = max(abs(a_eff * x * x), abs(b_eff * x), abs(c_eff), 1.0)
    return float(abs(a_eff * x * x + b_eff * x + c_eff) / scale)


def test_the_complex_radial_solve_matches_the_real_one_where_both_are_defined():
    """Additive, not a replacement: identical wherever the old one answers."""
    from fwap.cylindrical_solver._bessel import (
        _radial_wavenumbers_vti,
        _radial_wavenumbers_vti_complex,
    )

    worst, compared = 0.0, 0
    for entry in _THOMSEN_TABLE_1.values():
        s = _thomsen_stiffness(entry)
        v_sv = np.sqrt(s["c44"] / s["rho"])
        for freq in (3000.0, 8000.0, 15000.0):
            omega = 2.0 * np.pi * freq
            for c in np.linspace(760.0, v_sv * 0.999, 60):
                real = _radial_wavenumbers_vti(omega / c, omega, **s)
                if not all(np.isfinite(x) for x in real):
                    continue
                got = _radial_wavenumbers_vti_complex(omega / c, omega, **s)
                for u, v in zip(real, got):
                    assert abs(complex(v).imag) < 1e-12, (u, v)
                    worst = max(worst, abs(u - complex(v)) / max(abs(u), 1e-30))
                    compared += 1
    assert compared > 1000, compared
    assert worst < 1e-11, worst


def test_the_complex_radial_solve_answers_where_the_real_one_gives_up():
    """The conjugate-pair regime, checked against the governing equation.

    This is the claim phase 0 could not make: on the two media whose
    discriminant turns negative the solve now returns a conjugate pair
    that satisfies the Christoffel quadratic to ~1e-13, over exactly
    the window the real solver reports as ``NaN``. Both members decay
    (``Re(alpha) >= 0``), so they are admissible bound-mode columns.
    """
    from fwap.cylindrical_solver._bessel import (
        _radial_wavenumbers_vti,
        _radial_wavenumbers_vti_complex,
    )

    omega = 2.0 * np.pi * 8000.0
    for name in ("Mesaverde shale(5)", "Mesaverde sandstone"):
        s = _thomsen_stiffness(_THOMSEN_TABLE_1[name])
        v_sv = np.sqrt(s["c44"] / s["rho"])
        conjugate, worst = 0, 0.0
        for c in np.linspace(760.0, v_sv * 0.999, 300):
            kz = omega / c
            if all(np.isfinite(x) for x in _radial_wavenumbers_vti(kz, omega, **s)):
                continue  # the real solver handles this one
            qp, qsv, sh = _radial_wavenumbers_vti_complex(kz, omega, **s)
            assert abs(qp - np.conj(qsv)) < 1e-9 * abs(qp), (name, c, qp, qsv)
            for alpha in (qp, qsv, sh):
                assert complex(alpha).real >= -1e-12, (name, c, alpha)
            # Only qP and qSV are roots of the quadratic; SH comes from
            # its own closed form and is checked against that instead.
            worst = max(
                worst,
                _christoffel_residual(qp, kz, omega, s),
                _christoffel_residual(qsv, kz, omega, s),
            )
            sh_expected = np.sqrt(
                complex((s["c44"] * kz * kz - s["rho"] * omega**2) / s["c66"])
            )
            assert abs(sh - sh_expected) < 1e-9 * max(abs(sh_expected), 1.0), (
                name,
                c,
                sh,
            )
            conjugate += 1
        assert conjugate > 100, (name, conjugate)
        assert worst < 1e-10, (name, worst)


def test_the_complex_radial_solve_is_continuous_through_the_branch_point():
    """The roots merge and split rather than jumping.

    ``disc = 0`` is a square-root branch point: approached from the
    real side the two roots converge, and past it they separate into a
    conjugate pair with the same real part. A labelling that swapped
    them, or a branch that flipped sign, would show as a step here.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Mesaverde shale(5)"])
    omega = 2.0 * np.pi * 8000.0
    lo, hi = 1700.0, 1800.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _christoffel_discriminant(omega / mid, omega, s) < 0.0:
            lo = mid
        else:
            hi = mid

    offsets = (5.0, 2.0, 1.0, 0.2, 0.05, -0.05, -0.2, -1.0, -2.0, -5.0)
    previous, steps = None, []
    for d in offsets:
        qp, qsv, _ = _radial_wavenumbers_vti_complex(omega / (hi + d), omega, **s)
        if d < 0.0:
            assert abs(qp - np.conj(qsv)) < 1e-9 * abs(qp), (d, qp, qsv)
            assert abs(qp.imag) > 0.0, (d, qp)
        else:
            assert abs(qp.imag) < 1e-9 and abs(qsv.imag) < 1e-9, (d, qp, qsv)
            assert qp.real >= qsv.real, (d, qp, qsv)
        if previous is not None:
            steps.append(abs(qp - previous[0]) + abs(qsv - previous[1]))
        previous = (qp, qsv)
    # Small and smooth across the crossing, not a jump at one point.
    assert max(steps) < 1.0, steps
    assert max(steps) < 8.0 * min(steps), steps


def test_the_complex_radial_solve_reduces_to_p_and_s_and_handles_complex_kz():
    """Isotropic limit, and the leaky-side arithmetic.

    At isotropic constants the three wavenumbers must be the ordinary
    ``p``, ``s``, ``s``. Complex ``k_z`` is not a supported solver
    regime -- no driver reaches it and the radiating branch is not
    offered -- but the arithmetic must not fall over, since phase 4 is
    where it becomes load-bearing.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex

    vp, vs, rho = 3658.0, 2032.0, 2350.0
    c44 = rho * vs**2
    c11 = rho * vp**2
    iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
    for freq, c in ((6000.0, 1700.0), (11000.0, 1400.0)):
        omega = 2.0 * np.pi * freq
        kz = omega / c
        qp, qsv, sh = _radial_wavenumbers_vti_complex(kz, omega, **iso)
        assert abs(qp - np.sqrt(kz**2 - (omega / vp) ** 2)) < 1e-12
        assert abs(qsv - np.sqrt(kz**2 - (omega / vs) ** 2)) < 1e-12
        assert abs(sh - np.sqrt(kz**2 - (omega / vs) ** 2)) < 1e-12

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Mesaverde shale(5)"])
    omega = 2.0 * np.pi * 8000.0
    for c in (1500.0, 1800.0, 2000.0):
        for damping in (0.01, 0.1, 0.5, 1.0):
            kz = omega / c + 1j * damping
            qp, qsv, sh = _radial_wavenumbers_vti_complex(kz, omega, **s)
            for alpha in (qp, qsv, sh):
                assert np.isfinite(complex(alpha))
                assert complex(alpha).real >= -1e-12, (c, damping, alpha)
            assert _christoffel_residual(qp, kz, omega, s) < 1e-10
            assert _christoffel_residual(qsv, kz, omega, s) < 1e-10


def test_the_qsv_column_is_the_qp_column_conjugated_and_scaled():
    """The structural fact the recombination rests on.

    ``_recombine_conjugate_columns`` replaces the qP / qSV pair with a
    divided difference, which is only a change of basis if the two
    columns really are one function evaluated at the two roots. They
    are: the builders differ by a factor ``k_z``, so where the roots
    are conjugate ``col_qSV = k_z conj(col_qP)`` exactly. Checked on
    the raw rows, before recombination.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex
    from fwap.cylindrical_solver._vti import (
        _modal_row1_at_a_n1_vti,
        _modal_row2_at_a_n1_vti,
        _modal_row3_at_a_n1_vti,
        _modal_row4_at_a_n1_vti,
    )

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Mesaverde shale(5)"])
    kw = dict(**s, vf=1500.0, rho_f=1000.0, a=0.10)
    omega = 2.0 * np.pi * 8000.0

    checked, worst = 0, 0.0
    for c in (1200.0, 1400.0, 1600.0, 1700.0, 1750.0):
        kz = omega / c
        qp, qsv, _ = _radial_wavenumbers_vti_complex(kz, omega, **s)
        if qp.imag == 0.0:
            continue
        assert abs(qp - qsv.conjugate()) < 1e-9 * abs(qp), (c, qp, qsv)
        raw = np.vstack(
            [
                _modal_row1_at_a_n1_vti(kz, omega, **kw),
                _modal_row2_at_a_n1_vti(kz, omega, **kw),
                _modal_row3_at_a_n1_vti(kz, omega, **kw),
                _modal_row4_at_a_n1_vti(kz, omega, **kw),
            ]
        )
        col_qP, col_qSV = raw[:, 1], raw[:, 2]
        residual = np.linalg.norm(col_qSV - kz * np.conj(col_qP))
        worst = max(worst, residual / max(np.linalg.norm(col_qSV), 1e-30))
        checked += 1
    assert checked >= 4, checked
    assert worst < 1e-12, worst


# ----------------------------------------------------------------------
# A.11 phase 4: radiating branches, and the oracle that checks them
# ----------------------------------------------------------------------


def test_the_leaky_window_keeps_the_qp_qsv_pair_separable():
    """A.11 phase 0 deferred this; the open window makes it answerable.

    Phase 0 measured the branch-exchange region over a scan reaching
    ``1.6 V_P`` and found ``Re(disc)`` sign changes on three media, but
    could not say whether they mattered: the window a leaky search
    visits is ``(V_Sv, V_P0)``, and on two media the solver could not
    even reach it. It does now, and the pair stays well separated
    there on all six -- so continuity assignment suffices and no
    polarisation criterion is needed.
    """
    for name, entry in _THOMSEN_TABLE_1.items():
        s = _thomsen_stiffness(entry)
        v_sv = np.sqrt(s["c44"] / s["rho"])
        v_p = np.sqrt(s["c33"] / s["rho"])
        worst = np.inf
        for freq in (4000.0, 8000.0, 14000.0):
            omega = 2.0 * np.pi * freq
            for c in np.linspace(v_sv * 1.002, v_p * 0.998, 40):
                for damping in (0.0, 0.3, 0.7, 1.0):
                    disc = _christoffel_discriminant(omega / c + 1j * damping, omega, s)
                    scale = abs((s["c11"] + s["c44"]) * s["rho"] * omega**2) ** 2
                    worst = min(worst, abs(disc) / scale)
        assert worst > 1e-6, (name, worst)


def test_the_radiating_branch_is_per_wave_and_does_not_reorder_the_pair():
    """qP and qSV share a quadratic but not a sheet.

    Over ``V_Sv < c < V_P`` the qSV wave radiates while qP is still
    evanescent -- the ordinary leaky configuration -- so the flags must
    be independent. And the labelling must survive it: ``alpha_qP`` is
    the larger ``alpha^2``, which in this window means the *positive*
    square even though the other root has the larger ``|alpha|``.
    Ordering on ``alpha`` instead silently swaps the two waves, which
    is what this did until phase 4 reached the window.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Green River shale"])
    v_sv = np.sqrt(s["c44"] / s["rho"])
    omega = 2.0 * np.pi * 8000.0

    for c in (1200.0, 1600.0, 1900.0, 2200.0, 2600.0, 3000.0, 3200.0):
        leaky = c > v_sv
        qp, qsv, sh = _radial_wavenumbers_vti_complex(
            omega / c, omega, **s, radiating=(False, leaky, leaky)
        )
        assert (qp**2).real >= (qsv**2).real - 1e-6, (c, qp, qsv)
        if leaky:
            assert qsv.imag > 0.0, (c, qsv)
            assert abs(qp.imag) < 1e-12, (c, qp)

    # Flagging one wave leaves the others exactly alone.
    for c in (2200.0, 3000.0):
        base = _radial_wavenumbers_vti_complex(omega / c, omega, **s)
        split = _radial_wavenumbers_vti_complex(
            omega / c, omega, **s, radiating=(False, True, False)
        )
        assert base[0] == split[0], (c, base[0], split[0])
        assert base[2] == split[2], (c, base[2], split[2])


def test_the_leaky_vti_determinant_reproduces_the_isotropic_one():
    """The oracle for phase 4, and it is a real one.

    Unlike the conjugate regime -- which the isotropic limit cannot
    reach at all -- the *leaky* regime is reachable isotropically, so
    the VTI determinant with radiating branches can be checked against
    `_modal_determinant_n1_complex`, an independent implementation.

    They agree exactly, up to the recombination's own Jacobian: the
    column change of basis scales the determinant by ``-1/(split k_z)``
    with ``split = alpha_qP - alpha_qSV``, which is non-vanishing here
    and so moves no root. Dividing it out leaves a ratio of exactly
    ``-1`` at every sampled velocity.

    This test caught two real errors while being written: the branch
    flags were mapped in the wrong order (`_detect_leaky_branches`
    returns ``(leaky_F, leaky_p, leaky_s)`` -- fluid first), and the
    qP/qSV labels were swapping above ``V_Sv``.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex
    from fwap.cylindrical_solver._leaky import _detect_leaky_branches
    from fwap.cylindrical_solver._n1_isotropic import (
        _modal_determinant_n1_complex,
    )
    from fwap.cylindrical_solver._vti import _modal_matrix_n1_vti

    vp, vs, rho = 3658.0, 2032.0, 2350.0
    vf, rho_f, a = 1500.0, 1000.0, 0.10
    c44, c11 = rho * vs * vs, rho * vp * vp
    iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
    omega = 2.0 * np.pi * 9000.0

    ratios = []
    for c in (2200.0, 2500.0, 2800.0, 3100.0, 3400.0, 3600.0):
        kz = complex(omega / c)
        _, leaky_p, leaky_s = _detect_leaky_branches(kz, omega, vp, vs, vf)
        assert (leaky_p, leaky_s) == (False, True), (c, leaky_p, leaky_s)
        qp, qsv, _ = _radial_wavenumbers_vti_complex(
            kz, omega, **iso, radiating=(leaky_p, leaky_s, leaky_s)
        )
        matrix = _modal_matrix_n1_vti(
            kz.real,
            omega,
            **iso,
            vf=vf,
            rho_f=rho_f,
            a=a,
            radiating=(leaky_p, leaky_s, leaky_s),
        )
        vti = complex(np.linalg.det(matrix))
        isotropic = complex(
            _modal_determinant_n1_complex(
                kz,
                omega,
                vp=vp,
                vs=vs,
                rho=rho,
                vf=vf,
                rho_f=rho_f,
                a=a,
                leaky_p=leaky_p,
                leaky_s=leaky_s,
            )
        )
        ratios.append(vti / isotropic * (qp - qsv) * kz)

    adjusted = np.array(ratios)
    assert np.allclose(adjusted, -1.0, rtol=1e-9, atol=1e-9), adjusted


def test_a_complex_kz_is_refused_only_when_no_wave_radiates():
    """The refusal narrowed rather than disappeared.

    This used to reject every complex ``k_z``, naming the fluid column
    as the gap. The fluid column now takes one, so what is left is the
    genuinely meaningless case: a complex ``k_z`` with every wave on
    the bound branch describes a field that decays in ``r`` and grows
    along ``z``, which is not a mode. That is still refused, and
    saying which waves radiate is what makes the call well posed.
    """
    import pytest

    from fwap.cylindrical_solver._vti import _modal_matrix_n1_vti

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Green River shale"])
    fluid = dict(vf=1500.0, rho_f=1000.0, a=0.10)
    omega = 2.0 * np.pi * 8000.0

    with pytest.raises(NotImplementedError, match="radiating"):
        _modal_matrix_n1_vti(complex(20.0, 0.3), omega, **s, **fluid)

    # Naming a radiating wave makes it well posed, and it answers.
    matrix = _modal_matrix_n1_vti(
        complex(20.0, 0.3), omega, **s, **fluid, radiating=(False, True, True)
    )
    assert matrix.shape == (4, 4)
    assert np.isfinite(matrix).all()
    assert abs(complex(np.linalg.det(matrix))) > 0.0

    # A real k_z is unaffected either way.
    bound = _modal_matrix_n1_vti(20.0, omega, **s, **fluid)
    assert np.isfinite(bound).all()


def test_the_fluid_column_needs_no_outgoing_form_and_takes_a_complex_kz():
    """The fluid was the last blocker, and it wanted a smaller fix.

    "Write the outgoing fluid form" was the wrong framing. The fluid
    occupies ``0 <= r <= a``, so its condition is regularity at the
    origin rather than radiation at infinity, and ``I_n`` -- entire for
    integer ``n`` -- supplies that at any complex argument. All the
    leaky case needed was for ``F_f^2`` to be allowed to go complex.

    The branch of ``F_f`` does not matter to the modes either: the
    fluid enters only rows 1 and 2, as ``(F_f I_0 - I_1/a)`` and
    ``-I_1``, and ``I_0`` is even in its argument while ``I_1`` is odd,
    so both combinations are odd in ``F_f``. Flipping the branch
    negates the whole column, and with it the determinant, moving no
    root.
    """
    from scipy import special

    from fwap.cylindrical_solver._vti import _fluid_bessels_n1_vti

    omega, vf, a = 2.0 * np.pi * 9000.0, 1500.0, 0.10

    # The real path is bit-identical to what it was.
    for c in (900.0, 1200.0, 1499.0, 1501.0, 1800.0, 2400.0):
        real = _fluid_bessels_n1_vti(omega / c, omega, vf, a)
        via_complex = _fluid_bessels_n1_vti(complex(omega / c, 0.0), omega, vf, a)
        assert real == via_complex, (c, real, via_complex)

    for c in (1200.0, 2400.0):
        for damping in (0.05, 0.5, 1.5):
            F_f, i0, i1 = _fluid_bessels_n1_vti(
                complex(omega / c, damping), omega, vf, a
            )
            assert np.isfinite(complex(F_f))
            assert np.isfinite(complex(i0)) and np.isfinite(complex(i1))

            # I_0 even, I_1 odd, so each row entry flips sign with the
            # branch and the column flips as a whole.
            i0_flipped = complex(special.iv(0, -F_f * a))
            i1_flipped = complex(special.iv(1, -F_f * a))
            assert abs(i0_flipped - i0) < 1e-9 * max(abs(i0), 1.0)
            assert abs(i1_flipped + i1) < 1e-9 * max(abs(i1), 1.0)

            row1 = F_f * i0 - i1 / a
            row1_flipped = (-F_f) * i0_flipped - i1_flipped / a
            assert abs(row1_flipped + row1) < 1e-9 * max(abs(row1), 1.0)


def test_the_leaky_vti_determinant_matches_the_isotropic_one_at_complex_kz():
    """Phase 4's oracle, now over the regime a driver would search.

    The real-``k_z`` version of this test passed while the labelling
    was still swapping qP and qSV at a complex ``k_z``, so it did not
    on its own establish the leaky path. This does: over ``c`` in
    [2200, 3600] and ``Im(k_z)`` in [0, 1.5] the VTI determinant
    reproduces `_modal_determinant_n1_complex` up to the recombination
    Jacobian, to ~1e-14.

    It fails at ``3.4`` rather than ``1e-14`` if the qP/qSV ordering
    reverts to ``|alpha|``: above roughly ``1.2 V_Sv`` the radiating
    root has the larger magnitude, so the two waves swap.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex
    from fwap.cylindrical_solver._leaky import _detect_leaky_branches
    from fwap.cylindrical_solver._n1_isotropic import (
        _modal_determinant_n1_complex,
    )
    from fwap.cylindrical_solver._vti import _modal_matrix_n1_vti

    vp, vs, rho = 3658.0, 2032.0, 2350.0
    vf, rho_f, a = 1500.0, 1000.0, 0.10
    c44, c11 = rho * vs * vs, rho * vp * vp
    iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
    omega = 2.0 * np.pi * 9000.0

    ratios = []
    for c in (2200.0, 2600.0, 3000.0, 3400.0, 3600.0):
        for damping in (0.0, 0.02, 0.10, 0.35, 0.80, 1.5):
            kz = complex(omega / c, damping)
            _, leaky_p, leaky_s = _detect_leaky_branches(kz, omega, vp, vs, vf)
            qp, qsv, _ = _radial_wavenumbers_vti_complex(
                kz, omega, **iso, radiating=(leaky_p, leaky_s, leaky_s)
            )
            matrix = _modal_matrix_n1_vti(
                kz,
                omega,
                **iso,
                vf=vf,
                rho_f=rho_f,
                a=a,
                radiating=(leaky_p, leaky_s, leaky_s),
            )
            isotropic = complex(
                _modal_determinant_n1_complex(
                    kz,
                    omega,
                    vp=vp,
                    vs=vs,
                    rho=rho,
                    vf=vf,
                    rho_f=rho_f,
                    a=a,
                    leaky_p=leaky_p,
                    leaky_s=leaky_s,
                )
            )
            ratios.append(complex(np.linalg.det(matrix)) / isotropic * (qp - qsv) * kz)

    assert len(ratios) == 30
    assert np.abs(np.array(ratios) + 1.0).max() < 1e-11, ratios


def test_the_qp_qsv_ordering_is_on_the_squares_at_a_complex_kz_too():
    """The rule is exact, not a heuristic.

    In the isotropic limit ``alpha_p^2 - alpha_s^2`` is
    ``(omega/V_s)^2 - (omega/V_p)^2`` -- a positive real constant that
    does not depend on ``k_z`` at all -- so ordering the pair on
    ``Re(alpha^2)`` labels them correctly for a complex ``k_z`` exactly
    as for a real one. Ordering on ``|alpha|`` agrees only below about
    ``1.2 V_Sv``, which is why the real-window tests missed it.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex

    vp, vs, rho = 3658.0, 2032.0, 2350.0
    c44, c11 = rho * vs * vs, rho * vp * vp
    iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
    omega = 2.0 * np.pi * 9000.0

    disagreements = 0
    for c in (2200.0, 2600.0, 3000.0, 3400.0):
        for damping in (0.02, 0.35, 0.80):
            kz = complex(omega / c, damping)
            qp, qsv, _ = _radial_wavenumbers_vti_complex(
                kz, omega, **iso, radiating=(False, True, True)
            )
            p = np.sqrt(complex(kz**2 - (omega / vp) ** 2))
            if p.real < 0:
                p = -p
            s = np.sqrt(complex(kz**2 - (omega / vs) ** 2))
            if s.imag < 0:
                s = -s
            assert abs(qp - p) < 1e-9 * max(abs(p), 1.0), (c, damping, qp, p)
            assert abs(qsv - s) < 1e-9 * max(abs(s), 1.0), (c, damping, qsv, s)
            if abs(qp) < abs(qsv):
                disagreements += 1
    # The magnitude rule really does disagree over this window -- if it
    # did not, this test would be vacuous.
    assert disagreements >= 8, disagreements


# ----------------------------------------------------------------------
# A.11 phase 5: is there a leaky n=1 mode to drive?
# ----------------------------------------------------------------------


def _winding_number(fn, re_lo, re_hi, im_lo, im_hi, n=240):
    """Roots of ``fn`` inside the rectangle, by the argument principle.

    The loop is closed **before** unwrapping. Unwrapping first and then
    summing differences around the closed cycle telescopes to exactly
    zero for any input, which is a bug that looks like "no roots
    anywhere" and is why the control below exists.
    """
    points = []
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_lo + (re_hi - re_lo) * t, im_lo))
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_hi, im_lo + (im_hi - im_lo) * t))
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_hi + (re_lo - re_hi) * t, im_hi))
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_lo, im_hi + (im_lo - im_hi) * t))
    values = np.array([fn(z) for z in points])
    if not np.all(np.isfinite(values)) or np.any(values == 0):
        return None
    phase = np.unwrap(np.angle(np.concatenate([values, values[:1]])))
    return float((phase[-1] - phase[0]) / (2.0 * np.pi))


def test_the_winding_instrument_finds_a_root_it_should():
    """The control, without which a null survey means nothing.

    A first version of this returned ``0`` for every input, including a
    box drawn round a root whose location was already known -- and the
    VTI survey below was briefly read as "no leaky mode exists" on the
    strength of it.
    """
    from fwap import flexural_dispersion
    from fwap.cylindrical_solver._n1_isotropic import (
        _modal_determinant_n1_complex,
    )

    vp, vs, rho = 3658.0, 2032.0, 2350.0
    fluid = dict(vf=1500.0, rho_f=1000.0, a=0.10)
    omega = 2.0 * np.pi * 8000.0
    speed = (
        1.0
        / flexural_dispersion(
            np.array([8000.0]), vp=vp, vs=vs, rho=rho, **fluid
        ).slowness[0]
    )
    kz = omega / speed

    def bound(z):
        return complex(
            _modal_determinant_n1_complex(
                z, omega, vp=vp, vs=vs, rho=rho, **fluid, leaky_p=False, leaky_s=False
            )
        )

    around = _winding_number(bound, kz - 0.6, kz + 0.6, -0.35, 0.35)
    empty = _winding_number(bound, kz + 2.0, kz + 4.0, -0.35, 0.35)
    assert around is not None and abs(around - 1.0) < 0.05, around
    assert empty is not None and abs(empty) < 0.05, empty


def test_the_open_hole_leaky_dipole_window_is_empty_isotropic_and_vti_alike():
    """Phase 5's finding: there is no mode there for a driver to march.

    Counted with the instrument the test above validates, the window
    ``V_S < c < V_P`` at ``Im(k_z) > 0`` holds **no** ``n = 1`` root --
    and it holds none for the *isotropic* determinant either, which is
    the mature path. So this is a fact about the open-hole dipole
    problem rather than something the VTI assembly is missing.

    Real-axis ``Im(det)`` sign changes are not evidence against this.
    Green River shows exactly one per frequency across 3-18 kHz, at
    velocities that rise, jump, descend and split -- no coherent
    branch, and no root: ``|det|`` never vanishes at any of them.

    Where the repo *does* record leaky dipole modes is behind casing
    (A.9), where the steel lifts the mode above a slow formation's
    shear speed. That is the cased VTI path, which is not built.
    """
    from fwap.cylindrical_solver._leaky import _detect_leaky_branches
    from fwap.cylindrical_solver._n1_isotropic import (
        _modal_determinant_n1_complex,
    )
    from fwap.cylindrical_solver._vti import _modal_determinant_n1_vti_complex

    fluid = dict(vf=1500.0, rho_f=1000.0, a=0.10)

    # Isotropic control, fast and slow.
    for vp, vs, rho in ((3658.0, 2032.0, 2350.0), (2400.0, 900.0, 2100.0)):
        for freq in (4000.0, 8000.0):
            omega = 2.0 * np.pi * freq

            def isotropic(z, omega=omega, vp=vp, vs=vs, rho=rho):
                _, leaky_p, leaky_s = _detect_leaky_branches(
                    z, omega, vp, vs, fluid["vf"]
                )
                return complex(
                    _modal_determinant_n1_complex(
                        z,
                        omega,
                        vp=vp,
                        vs=vs,
                        rho=rho,
                        **fluid,
                        leaky_p=leaky_p,
                        leaky_s=leaky_s,
                    )
                )

            count = _winding_number(
                isotropic,
                omega / (vp * 0.997),
                omega / (vs * 1.003),
                0.001,
                1.5,
            )
            assert count is not None and abs(count) < 0.05, (vp, freq, count)

    # VTI, on the media whose windows are widest.
    for name in ("Green River shale", "Pierre shale"):
        s = _thomsen_stiffness(_THOMSEN_TABLE_1[name])
        v_sv = np.sqrt(s["c44"] / s["rho"])
        v_p = np.sqrt(s["c33"] / s["rho"])
        for freq in (4000.0, 8000.0):
            omega = 2.0 * np.pi * freq

            def vti(z, omega=omega, s=s, v_p=v_p, v_sv=v_sv):
                _, leaky_p, leaky_s = _detect_leaky_branches(
                    z, omega, v_p, v_sv, fluid["vf"]
                )
                return complex(
                    _modal_determinant_n1_vti_complex(
                        z,
                        omega,
                        **s,
                        **fluid,
                        radiating=(leaky_p, leaky_s, leaky_s),
                    )
                )

            count = _winding_number(
                vti,
                omega / (v_p * 0.997),
                omega / (v_sv * 1.003),
                0.001,
                1.5,
            )
            assert count is not None and abs(count) < 0.05, (name, freq, count)


def test_the_cased_leaky_dipole_exists_and_anisotropy_would_move_it():
    """A.12 groundwork: the target for a cased VTI solver is real.

    Two things a cased VTI build depends on, measured before building
    it. First, the mode exists: A.9's isotropic cased leaky dipole
    converges across 3-15 kHz for slow formations matching the
    vertical velocities of Thomsen's slow media, sitting well above
    ``V_S`` with a real attenuation.

    Second, anisotropy would move it by enough to matter. Running the
    isotropic solver at ``V_Sv`` and again at ``V_Sh`` brackets where a
    VTI answer must fall, and the bracket is **1.6 to 8.9 %** -- far
    above the 0.21-0.27 % at which the cased curves are tied to Schmitt
    & Cheng figures 20 and 21. A cased VTI solver would therefore be
    resolving a real effect rather than a rounding difference.
    """
    from fwap import flexural_dispersion_layered
    from fwap.cylindrical_solver._dataclasses import BoreholeLayer

    steel = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7850.0, thickness=0.01)
    cement = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.02)
    freq = np.arange(3000.0, 15001.0, 2000.0)
    fluid = dict(vf=1500.0, rho_f=1000.0, a=0.10)

    spreads = []
    for name in ("Pierre shale", "Dog Creek shale"):
        vp0, vs0, rho, _, _, gamma = _THOMSEN_TABLE_1[name]
        v_sh = vs0 * np.sqrt(1.0 + 2.0 * gamma)

        curves = {}
        for tag, vs in (("sv", vs0), ("sh", v_sh)):
            result = flexural_dispersion_layered(
                freq,
                vp=vp0,
                vs=vs,
                rho=rho,
                **fluid,
                layers=(steel, cement),
            )
            slowness = np.asarray(result.slowness)
            assert np.isfinite(slowness).all(), (name, tag, slowness)
            curves[tag] = 1.0 / slowness

        # The mode is leaky: comfortably above the formation shear speed.
        assert (curves["sv"] > vs0 * 1.2).all(), (name, curves["sv"], vs0)
        spreads.extend(100.0 * (curves["sh"] - curves["sv"]) / curves["sv"])

    spreads = np.array(spreads)
    assert (spreads > 0.0).all(), spreads
    assert spreads.min() > 1.0, spreads.min()
    assert spreads.max() > 6.0, spreads.max()


# ----------------------------------------------------------------------
# A.12: the u_theta / u_z derivation the cased stack needs
# ----------------------------------------------------------------------


def test_the_vti_polarisation_ratio_has_the_right_isotropic_limits():
    """The sharp analytic check on ``gamma``.

    ``gamma`` is fixed by the axial equation of motion. Two isotropic
    limits pin it with no freedom at all: at the P root it must be
    exactly ``1``, recovering ``u = grad phi``; at the S root exactly
    ``alpha^2 / k_z^2``, which is the Hansen form ``u = curl curl(chi
    z)`` the isotropic assembly already uses. A sign error, or the
    wrong stiffness in the numerator, breaks one or both.
    """
    from fwap.cylindrical_solver._vti import _vti_polarisation_ratio

    vp, vs, rho = 3658.0, 2032.0, 2350.0
    mu = rho * vs * vs
    lam = rho * vp * vp - 2 * mu
    stiff = dict(c13=lam, c33=lam + 2 * mu, c44=mu, rho=rho)

    for freq in (4000.0, 9000.0, 16000.0):
        omega = 2.0 * np.pi * freq
        for c in (1200.0, 1700.0, 2400.0):
            kz = omega / c
            alpha_p = np.sqrt(complex(kz**2 - (omega / vp) ** 2))
            alpha_s = np.sqrt(complex(kz**2 - (omega / vs) ** 2))

            gamma_p = _vti_polarisation_ratio(alpha_p, kz, omega, **stiff)
            assert abs(gamma_p - 1.0) < 1e-12, (freq, c, gamma_p)

            gamma_s = _vti_polarisation_ratio(alpha_s, kz, omega, **stiff)
            expected = alpha_s**2 / kz**2
            assert abs(gamma_s - expected) < 1e-12 * max(abs(expected), 1.0), (
                freq,
                c,
                gamma_s,
                expected,
            )


def test_the_formation_displacement_columns_reproduce_the_validated_row():
    """The tie to code that is already trusted.

    ``u_r`` is the one component the open-hole assembly already had, in
    ``_modal_row1_at_a_n1_vti``. The new columns must reproduce those
    entries exactly, which is what makes ``u_theta`` and ``u_z`` -- the
    two the layered stack needs and which never existed here -- an
    extension rather than a reimplementation.

    It also pins the per-column normalisation: qP carries ``-1``, qSV
    ``-k_z`` and SH ``+i``. Those are conventions of the existing
    assembly, applied to whole columns, so they cancel out of any
    determinant.
    """
    from fwap.cylindrical_solver._vti import (
        _formation_displacements_n1_vti,
        _modal_row1_at_a_n1_vti,
    )

    a = 0.10
    for name in ("Green River shale", "Mesaverde sandstone", "Dog Creek shale"):
        s = _thomsen_stiffness(_THOMSEN_TABLE_1[name])
        for freq in (5000.0, 11000.0):
            omega = 2.0 * np.pi * freq
            for c in (1100.0, 1500.0):
                kz = omega / c
                row = _modal_row1_at_a_n1_vti(
                    kz, omega, **s, vf=1500.0, rho_f=1000.0, a=a
                )
                columns = _formation_displacements_n1_vti(kz, omega, **s, r=a)
                if not np.isfinite(columns).all():
                    continue
                for j in range(3):
                    assert abs(columns[0, j] - row[j + 1]) < 1e-9 * max(
                        abs(row[j + 1]), 1.0
                    ), (name, freq, c, j, columns[0, j], row[j + 1])

            # SH is curl(psi z): purely horizontal, no axial component.
            columns = _formation_displacements_n1_vti(omega / 1300.0, omega, **s, r=a)
            assert columns[2, 2] == 0.0, (name, freq, columns[2, 2])


def test_the_formation_columns_satisfy_the_vti_equations_of_motion():
    """The independent physics check: the columns are solutions.

    ``div sigma + rho omega^2 u`` is formed from the returned
    displacements by fourth-order finite differences in ``r``, with the
    strains, VTI constitutive law and cylindrical divergence written
    out here rather than taken from the module. Nothing in this test
    shares algebra with the code it checks, so agreement is evidence
    rather than a tautology.
    """
    from fwap.cylindrical_solver._vti import _formation_displacements_n1_vti

    n = 1
    r0, h = 0.12, 0.12e-4

    def residual(s, kz, omega, column):
        c11, c13, c33 = s["c11"], s["c13"], s["c33"]
        c44, c66, rho = s["c44"], s["c66"], s["rho"]
        c12 = c11 - 2 * c66

        def u(r):
            return _formation_displacements_n1_vti(kz, omega, **s, r=r)[:, column]

        def stress(r):
            ur, ut, uz = u(r)
            d = (-u(r + 2 * h) + 8 * u(r + h) - 8 * u(r - h) + u(r - 2 * h)) / (12 * h)
            e_rr, e_tt, e_zz = d[0], ur / r + (1j * n / r) * ut, 1j * kz * uz
            e_rt = (1j * n / r) * ur + d[1] - ut / r
            e_rz = 1j * kz * ur + d[2]
            e_tz = (1j * n / r) * uz + 1j * kz * ut
            return np.array(
                [
                    c11 * e_rr + c12 * e_tt + c13 * e_zz,
                    c12 * e_rr + c11 * e_tt + c13 * e_zz,
                    c13 * (e_rr + e_tt) + c33 * e_zz,
                    c66 * e_rt,
                    c44 * e_rz,
                    c44 * e_tz,
                ]
            )

        sig = stress(r0)
        dsig = (
            -stress(r0 + 2 * h)
            + 8 * stress(r0 + h)
            - 8 * stress(r0 - h)
            + stress(r0 - 2 * h)
        ) / (12 * h)
        s_rr, s_tt, s_zz, s_rt, s_rz, s_tz = sig
        ur, ut, uz = u(r0)
        res = (
            abs(
                dsig[0]
                + (1j * n / r0) * s_rt
                + 1j * kz * s_rz
                + (s_rr - s_tt) / r0
                + rho * omega**2 * ur
            ),
            abs(
                dsig[3]
                + (1j * n / r0) * s_tt
                + 1j * kz * s_tz
                + 2 * s_rt / r0
                + rho * omega**2 * ut
            ),
            abs(
                dsig[4]
                + (1j * n / r0) * s_tz
                + 1j * kz * s_zz
                + s_rz / r0
                + rho * omega**2 * uz
            ),
        )
        scale = max(abs(dsig[0]), abs(s_rr / r0), abs(rho * omega**2 * ur), 1e-300)
        return max(res) / scale

    checked = 0
    for name in ("Green River shale", "Mesaverde sandstone", "Dog Creek shale"):
        s = _thomsen_stiffness(_THOMSEN_TABLE_1[name])
        omega = 2.0 * np.pi * 8000.0
        for c in (1200.0, 1600.0):
            kz = omega / c
            if not np.isfinite(
                _formation_displacements_n1_vti(kz, omega, **s, r=r0)
            ).all():
                continue
            for column in range(3):
                assert residual(s, kz, omega, column) < 1e-6, (name, c, column)
                checked += 1
    assert checked >= 12, checked


def test_the_n0_real_reduction_was_already_a_valid_basis():
    """Recorded because it looked like a bug and was not.

    The ``n = 0`` determinant reduces over ``M.real`` while the qP /
    qSV columns carry imaginary parts of the same order as the real
    ones -- 44 % of them on Mesaverde shale(5) where the Stoneley mode
    sits. That reads as throwing away half of two independent
    solutions, which is exactly the defect A.11 phase 3 fixed at
    ``n = 1``.

    It is not the same defect. At ``n = 0`` the two columns are
    proportional, ``col_qSV = lambda conj(col_qP)``, so taking real
    parts still spans their plane whenever ``Im(lambda) != 0``, and the
    two determinants differ by exactly the scalar
    ``Im(lambda) Im(alpha_qP)``. The roots never moved. This pins that
    identity so the reasoning is not re-litigated from the 44 % alone.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex
    from fwap.cylindrical_solver._vti import (
        _modal_row1_at_a_vti,
        _modal_row2_at_a_vti,
        _modal_row3_at_a_vti,
        _recombine_conjugate_columns_n0,
    )

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Mesaverde shale(5)"])
    kw = dict(**s, vf=1500.0, rho_f=1000.0, a=0.10)
    omega = 2.0 * np.pi * 8000.0

    checked = 0
    for c in (1150.0, 1250.0, 1350.0, 1430.0, 1500.0):
        kz = omega / c
        alpha_qP, _, _ = _radial_wavenumbers_vti_complex(kz, omega, **s)
        matrix = np.vstack(
            [
                _modal_row1_at_a_vti(kz, omega, **kw),
                _modal_row2_at_a_vti(kz, omega, **kw),
                _modal_row3_at_a_vti(kz, omega, **kw),
            ]
        )
        if not np.isfinite(matrix).all() or alpha_qP.imag == 0.0:
            continue

        column, other = matrix[:, 1], matrix[:, 2]
        conjugate = np.conj(column)
        pivot = int(np.argmax(np.abs(conjugate)))
        lam = other[pivot] / conjugate[pivot]
        # Proportional to machine precision -- this is what makes the
        # real reduction a change of basis rather than a loss.
        assert np.allclose(other, lam * conjugate, rtol=1e-12, atol=0.0)
        # And genuinely complex, so the 44 % is real, not noise.
        assert np.max(np.abs(matrix.imag)) > 0.1 * np.max(np.abs(matrix.real))

        recombined = _recombine_conjugate_columns_n0(matrix, alpha_qP)
        assert not np.array_equal(recombined, matrix)
        ratio = float(np.linalg.det(matrix.real)) / float(
            np.linalg.det(recombined.real)
        )
        predicted = lam.imag * alpha_qP.imag
        assert abs(ratio - predicted) < 1e-8 * max(abs(predicted), 1.0), (
            c,
            ratio,
            predicted,
        )
        assert abs(predicted) > 1e-3, (c, predicted)
        checked += 1

    assert checked >= 4, checked


def test_the_vti_stoneley_curve_is_unchanged_by_the_n0_recombination():
    """The consequence of the identity above, stated as a regression.

    Since the two determinants differ by a non-zero scalar, the
    ``n = 0`` VTI dispersion curve must be exactly what it was, and the
    isotropic limit must still reproduce `stoneley_dispersion` to the
    bit.
    """
    from fwap import stoneley_dispersion, stoneley_dispersion_vti

    freq = np.arange(3000.0, 16001.0, 1000.0)
    fluid = dict(vf=1500.0, rho_f=1000.0, a=0.10)

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Mesaverde shale(5)"])
    speeds = 1.0 / stoneley_dispersion_vti(freq, **s, **fluid).slowness
    assert np.isfinite(speeds).all(), speeds
    assert abs(speeds.min() - 1410.71) < 0.05, speeds.min()
    assert abs(speeds.max() - 1448.76) < 0.05, speeds.max()

    vp, vs, rho = 3658.0, 2032.0, 2350.0
    c44, c11 = rho * vs * vs, rho * vp * vp
    iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
    a = stoneley_dispersion_vti(freq, **iso, **fluid).slowness
    b = stoneley_dispersion(freq, vp=vp, vs=vs, rho=rho, **fluid).slowness
    assert np.array_equal(a, b), np.nanmax(np.abs(a - b))


# ----------------------------------------------------------------------
# A.12: the 10x10 cased VTI assembly
# ----------------------------------------------------------------------

_CASED_LAYER = dict(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.02)


def _cased_fluid():
    from fwap.cylindrical_solver._dataclasses import BoreholeLayer

    return dict(vf=1500.0, rho_f=1000.0, a=0.10, layer=BoreholeLayer(**_CASED_LAYER))


def test_the_layered_matrix_isolates_the_formation_to_three_columns():
    """The structural fact the substitution rests on, measured.

    The formation occupies columns 6, 7 and 10 and appears only in rows
    5-10 -- the six continuity conditions at ``r = b``. Every other
    entry is *bit-identical* whatever the formation is, so the
    substep-F.2.a.5 phase rescale does not couple the layer block to
    formation parameters and the three columns can be replaced without
    disturbing anything else. If that ever stops holding, swapping in
    VTI columns silently corrupts the rescale, so it is checked rather
    than assumed.
    """
    import fwap.cylindrical_solver._n1_layered as layered
    from fwap.cylindrical_solver._vti import (
        _LAYERED_N1_FORMATION_COLUMNS,
    )

    builders = [getattr(layered, f"_layered_n1_row{i}_at_a") for i in (1, 2, 3, 4)]
    builders += [getattr(layered, f"_layered_n1_row{i}_at_b") for i in range(5, 11)]
    fluid = _cased_fluid()
    omega = 2.0 * np.pi * 8000.0
    kz = omega / 1300.0

    def build(vp, vs, rho):
        return np.vstack(
            [b(kz, omega, vp=vp, vs=vs, rho=rho, **fluid) for b in builders]
        )

    first = build(3000.0, 1700.0, 2300.0)
    second = build(4200.0, 2400.0, 2600.0)
    differs = np.abs(first - second) > 1e-12 * np.maximum(
        np.maximum(np.abs(first), np.abs(second)), 1e-300
    )

    formation_columns = sorted(j for j in range(10) if differs[:, j].any())
    assert tuple(formation_columns) == _LAYERED_N1_FORMATION_COLUMNS, formation_columns
    formation_rows = sorted(i for i in range(10) if differs[i].any())
    assert formation_rows == [4, 5, 6, 7, 8, 9], formation_rows
    others = [j for j in range(10) if j not in formation_columns]
    assert np.array_equal(first[:, others], second[:, others])


def test_each_layered_row_carries_the_calibrated_formation_quantity():
    """The row -> quantity map, and that its factors are per row.

    Rows 5-10 carry ``u_r``, ``u_theta``, ``u_z``, ``sigma_rr``,
    ``sigma_r_theta``, ``sigma_rz`` with factors ``1, i, i, 1, -1, -1``.
    What matters is not the values but that each factor is **constant
    across the three columns**: that is what says the isotropic
    assembly and the VTI columns already share a per-column
    normalisation, so no column rescale is needed. A per-column
    discrepancy would produce a determinant that still looks plausible.
    """
    import fwap.cylindrical_solver._n1_layered as layered
    from fwap.cylindrical_solver._vti import (
        _LAYERED_N1_FORMATION_COLUMNS,
        _LAYERED_N1_FORMATION_ROWS,
        _formation_displacements_n1_vti,
        _modal_row2_at_a_n1_vti,
        _modal_row3_at_a_n1_vti,
        _modal_row4_at_a_n1_vti,
    )

    vp, vs, rho = 3400.0, 1900.0, 2400.0
    c44, c11 = rho * vs * vs, rho * vp * vp
    iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
    fluid = _cased_fluid()
    b = fluid["a"] + fluid["layer"].thickness
    omega = 2.0 * np.pi * 8000.0
    kz = omega / 1300.0

    builders = [getattr(layered, f"_layered_n1_row{i}_at_a") for i in (1, 2, 3, 4)]
    builders += [getattr(layered, f"_layered_n1_row{i}_at_b") for i in range(5, 11)]
    matrix = np.vstack(
        [bld(kz, omega, vp=vp, vs=vs, rho=rho, **fluid) for bld in builders]
    )

    displacements = _formation_displacements_n1_vti(kz, omega, **iso, r=b)
    traction = dict(**iso, vf=fluid["vf"], rho_f=fluid["rho_f"], a=b)
    quantities = {
        "u_r": displacements[0],
        "u_theta": displacements[1],
        "u_z": displacements[2],
        "sigma_rr": _modal_row2_at_a_n1_vti(kz, omega, **traction)[1:4],
        # Row 3 is sigma_r_theta and row 4 is sigma_rz -- see
        # test_the_vti_traction_rows_are_not_named_in_the_obvious_order.
        "sigma_rtheta": _modal_row3_at_a_n1_vti(kz, omega, **traction)[1:4],
        "sigma_rz": _modal_row4_at_a_n1_vti(kz, omega, **traction)[1:4],
    }

    for offset, (name, factor) in enumerate(_LAYERED_N1_FORMATION_ROWS):
        row = 4 + offset
        mine = quantities[name]
        for slot, column in enumerate(_LAYERED_N1_FORMATION_COLUMNS):
            expected = factor * mine[slot]
            got = matrix[row, column]
            assert abs(got - expected) < 1e-9 * max(abs(got), 1.0), (
                name,
                row,
                column,
                got,
                expected,
            )


def test_the_cased_vti_matrix_reproduces_the_isotropic_one_exactly():
    """The oracle, and it is the strongest in this line of work.

    At isotropic stiffnesses the VTI 10x10 must *be* the isotropic
    10x10 -- and it is, determinant ratio ``1 + 0j`` rather than merely
    proportional. `_modal_determinant_n1_layered` is itself tied to
    Schmitt & Cheng figures 20 and 21 at 0.21-0.27 %, so this inherits
    an external tie that no open-hole VTI path has.
    """
    from fwap.cylindrical_solver._n1_layered import _modal_determinant_n1_layered
    from fwap.cylindrical_solver._vti import _modal_matrix_n1_layered_vti

    fluid = _cased_fluid()
    ratios, checked = [], 0
    for vp, vs, rho in ((3400.0, 1900.0, 2400.0), (4500.0, 2700.0, 2450.0)):
        c44, c11 = rho * vs * vs, rho * vp * vp
        iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
        for freq in (5000.0, 9000.0, 14000.0):
            for c in (1100.0, 1350.0):
                omega = 2.0 * np.pi * freq
                kz = omega / c
                vti = complex(
                    np.linalg.det(
                        _modal_matrix_n1_layered_vti(kz, omega, **iso, **fluid)
                    )
                )
                isotropic = _modal_determinant_n1_layered(
                    kz, omega, vp=vp, vs=vs, rho=rho, **fluid
                )
                if not np.isfinite(isotropic) or isotropic == 0.0:
                    continue
                ratios.append(vti / isotropic)
                checked += 1
    assert checked >= 10, checked
    assert np.abs(np.array(ratios) - 1.0).max() < 1e-11, ratios


def test_the_cased_vti_determinant_has_the_isotropic_roots():
    """Roots, not values -- the recombination rescales the determinant.

    `_modal_determinant_n1_layered_vti` recombines, so it differs from
    the isotropic determinant by that Jacobian; dividing it out leaves
    exactly ``-1``. What has to hold for the solver is that the roots
    coincide, and on a fast formation they do to four decimals.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti_complex
    from fwap.cylindrical_solver._n1_layered import _modal_determinant_n1_layered
    from fwap.cylindrical_solver._vti import _modal_determinant_n1_layered_vti

    fluid = _cased_fluid()
    vp, vs, rho = 3400.0, 1900.0, 2400.0
    c44, c11 = rho * vs * vs, rho * vp * vp
    iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)

    adjusted = []
    for freq in (5000.0, 9000.0, 14000.0):
        omega = 2.0 * np.pi * freq
        for c in (1050.0, 1250.0, 1400.0):
            kz = omega / c
            vti = _modal_determinant_n1_layered_vti(kz, omega, **iso, **fluid)
            isotropic = _modal_determinant_n1_layered(
                kz, omega, vp=vp, vs=vs, rho=rho, **fluid
            )
            if not np.isfinite(isotropic) or isotropic == 0.0:
                continue
            qp, qsv, _ = _radial_wavenumbers_vti_complex(kz, omega, **iso)
            adjusted.append(vti / isotropic * (qp - qsv) * kz)
    assert len(adjusted) >= 6, len(adjusted)
    assert np.abs(np.array(adjusted) + 1.0).max() < 1e-10, adjusted

    omega = 2.0 * np.pi * 9000.0
    speeds = np.linspace(950.0, 1450.0, 900)

    def root_of(fn):
        values = np.array([fn(omega / c) for c in speeds])
        finite = np.isfinite(values)
        sign = np.sign(values[finite])
        idx = np.where(np.diff(sign) != 0)[0]
        out = []
        for i in idx:
            x0, x1 = speeds[finite][i], speeds[finite][i + 1]
            y0, y1 = values[finite][i], values[finite][i + 1]
            out.append(x0 - y0 * (x1 - x0) / (y1 - y0))
        return out

    vti_roots = root_of(
        lambda kz: _modal_determinant_n1_layered_vti(kz, omega, **iso, **fluid)
    )
    iso_roots = root_of(
        lambda kz: _modal_determinant_n1_layered(
            kz, omega, vp=vp, vs=vs, rho=rho, **fluid
        )
    )
    assert len(vti_roots) == len(iso_roots) == 1, (vti_roots, iso_roots)
    assert abs(vti_roots[0] - iso_roots[0]) < 1e-3, (vti_roots, iso_roots)


def test_the_vti_traction_rows_are_not_named_in_the_obvious_order():
    """Row 3 is ``sigma_r_theta`` and row 4 is ``sigma_rz``.

    The reverse looks right from the row numbering and was assumed
    while wiring the cased assembly. It is wrong, and the calibration
    against the layered stack could not catch it: that matched *values*
    into the correct slots, so the determinant was right while the
    labels on it were not.

    Settled against the constitutive law instead, with both stresses
    built directly from the returned displacements. The pairing below
    gives a clean per-column ratio of ``-i``; the swapped pairing gives
    ratios spanning a factor of sixty.
    """
    from fwap.cylindrical_solver._vti import (
        _formation_displacements_n1_vti,
        _modal_row3_at_a_n1_vti,
        _modal_row4_at_a_n1_vti,
    )

    s = _thomsen_stiffness(_THOMSEN_TABLE_1["Green River shale"])
    omega = 2.0 * np.pi * 8000.0
    n, r = 1, 0.10
    h = r * 1e-4

    for c in (1200.0, 1500.0):
        kz = omega / c

        def u(radius, kz=kz):
            return _formation_displacements_n1_vti(kz, omega, **s, r=radius)

        here = u(r)
        d = (-u(r + 2 * h) + 8 * u(r + h) - 8 * u(r - h) + u(r - 2 * h)) / (12 * h)
        sigma_rz = s["c44"] * (1j * kz * here[0] + d[2])
        sigma_rtheta = s["c66"] * ((1j * n / r) * here[0] + d[1] - here[1] / r)

        fluid = dict(vf=1500.0, rho_f=1000.0, a=r)
        row3 = _modal_row3_at_a_n1_vti(kz, omega, **s, **fluid)[1:4]
        row4 = _modal_row4_at_a_n1_vti(kz, omega, **s, **fluid)[1:4]

        for row, expected in ((row3, sigma_rtheta), (row4, sigma_rz)):
            ratio = np.array([row[j] / expected[j] for j in range(3)])
            assert np.allclose(ratio, ratio[0], rtol=1e-9, atol=0.0), (c, ratio)
            assert abs(ratio[0] + 1j) < 1e-9, (c, ratio[0])

        # And the swapped pairing is nowhere near constant.
        swapped = np.array([row3[j] / sigma_rz[j] for j in range(3)])
        assert np.max(np.abs(swapped - swapped[0])) > 0.5 * np.abs(swapped).max()


def test_the_leaky_cased_vti_determinant_matches_the_isotropic_one():
    """The radiating flags reach the formation through the cased stack.

    The layered real-``k_z`` path cannot express a complex ``k_z`` at
    all, so the leaky case goes through
    `_modal_determinant_n1_cased_complex` with the VTI formation block
    substituted for the isotropic one. Everything else -- fluid, every
    layer, the propagator, the real-axis branch handling -- is the
    isotropic machinery untouched.

    Only the formation takes radiating branches, and that is physical
    rather than a shortcut: the fluid and the layers occupy bounded
    annuli and carry both Bessel families, so their condition is
    regularity; the half-space is the only part that can carry energy
    away.

    Checked at genuinely complex ``k_z`` with ``leaky_s`` active, which
    is the configuration the slow-formation cased dipole actually sits
    in.
    """
    from fwap.cylindrical_solver._cased import (
        _modal_determinant_n1_cased_complex,
    )
    from fwap.cylindrical_solver._dataclasses import BoreholeLayer
    from fwap.cylindrical_solver._leaky import _detect_leaky_branches
    from fwap.cylindrical_solver._vti import (
        _modal_determinant_n1_cased_vti_complex,
    )

    layers = (
        BoreholeLayer(vp=5900.0, vs=3200.0, rho=7850.0, thickness=0.01),
        BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.02),
    )
    fluid = dict(vf=1500.0, rho_f=1000.0, a=0.10, layers=layers)
    omega = 2.0 * np.pi * 8000.0

    ratios, leaky_seen = [], 0
    for vp, vs, rho in ((2500.0, 800.0, 2200.0), (2074.0, 869.0, 2250.0)):
        c44, c11 = rho * vs * vs, rho * vp * vp
        iso = dict(c11=c11, c13=c11 - 2 * c44, c33=c11, c44=c44, c66=c44, rho=rho)
        for c in (1200.0, 1350.0):
            for damping in (0.0, 0.3, 1.0):
                kz = complex(omega / c, damping)
                _, leaky_p, leaky_s = _detect_leaky_branches(kz, omega, vp, vs, 1500.0)
                leaky_seen += int(leaky_s)
                vti = _modal_determinant_n1_cased_vti_complex(
                    kz, omega, **iso, **fluid, radiating=(leaky_p, leaky_s, leaky_s)
                )
                isotropic = _modal_determinant_n1_cased_complex(
                    kz,
                    omega,
                    vp=vp,
                    vs=vs,
                    rho=rho,
                    **fluid,
                    leaky_p=leaky_p,
                    leaky_s=leaky_s,
                )
                if not np.isfinite(vti) or not np.isfinite(isotropic):
                    continue
                if isotropic == 0.0:
                    continue
                ratios.append(vti / isotropic)

    assert len(ratios) >= 10, len(ratios)
    assert leaky_seen >= 10, leaky_seen
    assert np.abs(np.array(ratios) - 1.0).max() < 1e-11, ratios


def test_the_cased_formation_block_rejects_a_wrong_shape():
    """The injected block is checked, not trusted.

    `_modal_determinant_n1_cased_complex` indexes the block by row and
    column; a wrongly shaped one would either raise deep inside the
    assembly or, worse, broadcast.
    """
    import pytest

    from fwap.cylindrical_solver._cased import (
        _modal_determinant_n1_cased_complex,
    )
    from fwap.cylindrical_solver._dataclasses import BoreholeLayer

    layers = (BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.02),)
    with pytest.raises(ValueError, match=r"\(6, 3\)"):
        _modal_determinant_n1_cased_complex(
            complex(20.0, 0.1),
            2.0 * np.pi * 8000.0,
            vp=2500.0,
            vs=800.0,
            rho=2200.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.10,
            layers=layers,
            formation_block=np.zeros((3, 6), dtype=complex),
        )
