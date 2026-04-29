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
        c11=2.063e10, c13=8.307e9, c33=1.875e10,
        c44=4.813e9, c66=6.056e9, rho=2250.0,
    )


# ---------------------------------------------------------------------
# Vertical and horizontal limits
# ---------------------------------------------------------------------


def test_vti_phase_at_vertical_recovers_axial_moduli():
    """At theta = 0, v_qP = sqrt(C33/rho), and v_qSV = v_SH =
    sqrt(C44/rho) (the vertical-shear degeneracy)."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    vP, vSV, vSH = vti_phase_velocities(
        **p, phase_angle_rad=np.array([0.0])
    )
    assert abs(float(vP[0]) - np.sqrt(p["c33"] / p["rho"])) < 1.0e-6
    assert abs(float(vSV[0]) - np.sqrt(p["c44"] / p["rho"])) < 1.0e-6
    assert abs(float(vSH[0]) - np.sqrt(p["c44"] / p["rho"])) < 1.0e-6


def test_vti_phase_at_horizontal_recovers_in_plane_moduli():
    """At theta = pi/2, v_qP = sqrt(C11/rho), v_qSV = sqrt(C44/rho),
    v_SH = sqrt(C66/rho)."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    vP, vSV, vSH = vti_phase_velocities(
        **p, phase_angle_rad=np.array([np.pi / 2]),
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
        c11=M, c13=lam, c33=M, c44=mu, c66=mu, rho=rho,
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
    _, vSV, vSH = vti_phase_velocities(
        **p, phase_angle_rad=np.array([0.0])
    )
    assert abs(float(vSV[0]) - float(vSH[0])) < 1.0e-6


def test_vti_phase_v_qSV_equals_v_qSV_at_horizontal_for_C44():
    """At pi/2, qSV propagates with vertical-shear stiffness C44
    (its polarisation direction at horizontal propagation is the
    vertical x_3 axis), independent of C66."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    _, vSV, _ = vti_phase_velocities(
        **p, phase_angle_rad=np.array([np.pi / 2]),
    )
    assert abs(float(vSV[0]) - np.sqrt(p["c44"] / p["rho"])) < 1.0e-6


def test_vti_phase_v_SH_increases_when_C66_larger_than_C44():
    """Positive Thomsen gamma (C66 > C44) means v_SH(pi/2) >
    v_SH(0). The Berea-VTI test fixture has gamma > 0."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    _, _, vSH = vti_phase_velocities(
        **p, phase_angle_rad=np.array([0.0, np.pi / 2]),
    )
    # gamma > 0 means the horizontal SH is faster than the vertical.
    assert float(vSH[1]) > float(vSH[0])


def test_vti_phase_v_qP_at_horizontal_above_vertical_for_positive_epsilon():
    """C11 > C33 (positive Thomsen epsilon) means horizontal qP is
    faster than vertical qP. The Berea-VTI fixture has epsilon > 0."""
    from fwap.anisotropy import vti_phase_velocities

    p = _berea_vti()
    vP, _, _ = vti_phase_velocities(
        **p, phase_angle_rad=np.array([0.0, np.pi / 2]),
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
        c11=out.c11, c13=out.c13, c33=out.c33,
        c44=out.c44, c66=out.c66, rho=out.rho,
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
            c11=2e10, c13=8e9, c33=2e10, c44=5e9, c66=6e9, rho=0.0,
            phase_angle_rad=np.array([0.0]),
        )


def test_vti_phase_rejects_non_positive_elastic_constants():
    """Zero or negative c11/c33/c44/c66 raise (c13 is allowed
    negative in degenerate cases, but negative diagonal moduli
    violate physical-positivity constraints)."""
    import pytest

    from fwap.anisotropy import vti_phase_velocities

    base = dict(c11=2e10, c13=8e9, c33=2e10, c44=5e9, c66=6e9, rho=2400.0,
                phase_angle_rad=np.array([0.0]))
    with pytest.raises(ValueError, match="c11"):
        vti_phase_velocities(**{**base, "c11": 0.0})
    with pytest.raises(ValueError, match="c33"):
        vti_phase_velocities(**{**base, "c33": -1.0})
    with pytest.raises(ValueError, match="c44"):
        vti_phase_velocities(**{**base, "c44": 0.0})
    with pytest.raises(ValueError, match="c66"):
        vti_phase_velocities(**{**base, "c66": -1.0})
