"""Geomechanics-indices tests."""

from __future__ import annotations

import numpy as np
import pytest

from fwap.geomechanics import (
    GeomechanicsIndices,
    RICKMAN_E_MAX_PA,
    RICKMAN_E_MIN_PA,
    RICKMAN_NU_MAX,
    RICKMAN_NU_MIN,
    brittleness_index_rickman,
    closure_stress,
    fracability_index,
    geomechanics_indices,
    overburden_stress,
    sand_stability_indicator,
    unconfined_compressive_strength,
)
from fwap.rockphysics import elastic_moduli


# ---------------------------------------------------------------------
# brittleness_index_rickman
# ---------------------------------------------------------------------


def test_brittleness_endpoints_clip_to_unit_interval():
    """Stiff & low-Poisson saturates to 1; soft & high-Poisson to 0."""
    bi_high = brittleness_index_rickman(young_pa=1.0e12, poisson=0.05)
    bi_low  = brittleness_index_rickman(young_pa=1.0e8,  poisson=0.45)
    assert bi_high == pytest.approx(1.0)
    assert bi_low  == pytest.approx(0.0)


def test_brittleness_midpoint_is_half():
    """Midpoint of both Rickman bands gives BI = 0.5."""
    e_mid = 0.5 * (RICKMAN_E_MIN_PA + RICKMAN_E_MAX_PA)
    nu_mid = 0.5 * (RICKMAN_NU_MIN + RICKMAN_NU_MAX)
    bi = brittleness_index_rickman(young_pa=e_mid, poisson=nu_mid)
    assert bi == pytest.approx(0.5)


def test_brittleness_high_e_is_more_brittle_than_low_e():
    """At fixed Poisson, increasing E increases brittleness."""
    bi_a = brittleness_index_rickman(2.0e10, 0.25)
    bi_b = brittleness_index_rickman(6.0e10, 0.25)
    assert bi_b > bi_a


def test_brittleness_low_poisson_is_more_brittle_than_high_poisson():
    """At fixed E, lower Poisson is more brittle."""
    bi_low_nu  = brittleness_index_rickman(4.0e10, 0.18)
    bi_high_nu = brittleness_index_rickman(4.0e10, 0.38)
    assert bi_low_nu > bi_high_nu


def test_brittleness_array_input_broadcasts():
    """Vector inputs return a vector output, same length."""
    e_mid = 0.5 * (RICKMAN_E_MIN_PA + RICKMAN_E_MAX_PA)
    nu_mid = 0.5 * (RICKMAN_NU_MIN + RICKMAN_NU_MAX)
    e = np.array([RICKMAN_E_MIN_PA, e_mid, RICKMAN_E_MAX_PA])
    nu = np.array([RICKMAN_NU_MAX, nu_mid, RICKMAN_NU_MIN])
    bi = brittleness_index_rickman(e, nu)
    assert bi.shape == (3,)
    # First sample is at the (E_min, nu_max) corner -> BI = 0.
    # Last  sample is at the (E_max, nu_min) corner -> BI = 1.
    # Middle sample is at both midpoints       -> BI = 0.5.
    np.testing.assert_allclose(bi, [0.0, 0.5, 1.0])


def test_brittleness_rejects_inverted_bounds():
    """e_max <= e_min and nu_max <= nu_min are caller errors."""
    with pytest.raises(ValueError):
        brittleness_index_rickman(3.0e10, 0.25, e_min_pa=8.0e10, e_max_pa=1.0e10)
    with pytest.raises(ValueError):
        brittleness_index_rickman(3.0e10, 0.25, nu_min=0.40, nu_max=0.15)


def test_fracability_index_equals_brittleness_for_default_inputs():
    """Sonic-only fracability is the Rickman BI."""
    e = np.linspace(1.0e10, 8.0e10, 5)
    nu = np.linspace(0.40, 0.15, 5)
    np.testing.assert_allclose(
        fracability_index(e, nu),
        brittleness_index_rickman(e, nu),
    )


# ---------------------------------------------------------------------
# closure_stress
# ---------------------------------------------------------------------


def test_closure_stress_dry_rock_collapses_to_eaton_form():
    """With P_p = 0, sigma_h = nu/(1-nu) * sigma_v exactly."""
    nu = 0.25
    sigma_v = 50.0e6
    sh = closure_stress(nu, sigma_v)
    expected = (nu / (1.0 - nu)) * sigma_v
    assert sh == pytest.approx(expected)


def test_closure_stress_with_pore_pressure_uses_effective_stress():
    """sigma_h - alpha P_p == nu/(1-nu) * (sigma_v - alpha P_p)."""
    nu = 0.30
    sigma_v = 60.0e6
    pp = 25.0e6
    alpha = 0.8
    sh = closure_stress(nu, sigma_v, pore_pressure_pa=pp, biot_alpha=alpha)
    eff_v = sigma_v - alpha * pp
    expected = (nu / (1.0 - nu)) * eff_v + alpha * pp
    assert sh == pytest.approx(expected)


def test_closure_stress_increases_with_poisson():
    """At fixed sigma_v, higher Poisson => higher closure stress."""
    sigma_v = 50.0e6
    sh_low_nu  = closure_stress(0.18, sigma_v)
    sh_high_nu = closure_stress(0.35, sigma_v)
    assert sh_high_nu > sh_low_nu


def test_closure_stress_rejects_unphysical_poisson():
    """Poisson >= 1 is rejected."""
    with pytest.raises(ValueError):
        closure_stress(1.0, 50.0e6)
    with pytest.raises(ValueError):
        closure_stress(np.array([0.25, 1.5]), 50.0e6)


def test_closure_stress_array_inputs_broadcast():
    """Per-depth Poisson and sigma_v arrays broadcast correctly."""
    nu = np.array([0.20, 0.25, 0.30])
    sigma_v = np.array([40.0e6, 50.0e6, 60.0e6])
    sh = closure_stress(nu, sigma_v)
    assert sh.shape == (3,)
    np.testing.assert_allclose(
        sh, (nu / (1.0 - nu)) * sigma_v
    )


# ---------------------------------------------------------------------
# unconfined_compressive_strength
# ---------------------------------------------------------------------


def test_ucs_lacy_zero_modulus_gives_zero_ucs():
    """E = 0 => UCS = 0 (the Lacy correlation has no intercept term)."""
    ucs = unconfined_compressive_strength(0.0)
    assert ucs == pytest.approx(0.0)


def test_ucs_lacy_monotonic_in_young():
    """UCS strictly increases with Young's modulus over the realistic band."""
    e = np.linspace(5.0e9, 8.0e10, 10)
    ucs = unconfined_compressive_strength(e)
    assert np.all(np.diff(ucs) > 0)


def test_ucs_lacy_typical_sandstone_in_realistic_range():
    """E ~ 10 GPa (mid sandstone) gives UCS in the 10-100 MPa band."""
    ucs = unconfined_compressive_strength(1.0e10)
    assert 1.0e7 < ucs < 1.0e8


def test_ucs_rejects_negative_modulus():
    with pytest.raises(ValueError):
        unconfined_compressive_strength(-1.0e9)


def test_ucs_rejects_unknown_model():
    with pytest.raises(ValueError):
        unconfined_compressive_strength(1.0e10, model="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# sand_stability_indicator
# ---------------------------------------------------------------------


def test_sand_stability_threshold_default_is_5_gpa():
    """Below 5 GPa is sand-prone (False); at/above is stable (True)."""
    flag = sand_stability_indicator(np.array([1.0e9, 5.0e9, 1.0e10]))
    np.testing.assert_array_equal(flag, [False, True, True])


def test_sand_stability_custom_threshold():
    flag = sand_stability_indicator(np.array([2.0e9, 4.0e9]),
                                    threshold_pa=3.0e9)
    np.testing.assert_array_equal(flag, [False, True])


# ---------------------------------------------------------------------
# overburden_stress
# ---------------------------------------------------------------------


def test_overburden_constant_density_is_linear_in_depth():
    """sigma_v(z) = rho * g * (z - z0) for a constant-density column."""
    z = np.linspace(0.0, 1000.0, 11)
    rho = np.full_like(z, 2400.0)
    sigma = overburden_stress(z, rho)
    g = 9.80665
    expected = 2400.0 * g * (z - z[0])
    np.testing.assert_allclose(sigma, expected, rtol=1e-12)


def test_overburden_seeded_with_surface_value():
    """surface_value_pa is added to every depth's stress."""
    z = np.array([100.0, 200.0])
    rho = np.array([2400.0, 2400.0])
    sigma = overburden_stress(z, rho, surface_value_pa=2.0e6)
    assert sigma[0] == pytest.approx(2.0e6)
    g = 9.80665
    assert sigma[1] == pytest.approx(2.0e6 + 2400.0 * g * 100.0)


def test_overburden_rejects_non_increasing_depth():
    with pytest.raises(ValueError):
        overburden_stress(np.array([0.0, 0.0, 1.0]),
                          np.array([2400.0, 2400.0, 2400.0]))
    with pytest.raises(ValueError):
        overburden_stress(np.array([1.0, 0.0]),
                          np.array([2400.0, 2400.0]))


def test_overburden_rejects_negative_density():
    with pytest.raises(ValueError):
        overburden_stress(np.array([0.0, 1.0]), np.array([-1.0, 2400.0]))


def test_overburden_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        overburden_stress(np.array([0.0, 1.0, 2.0]),
                          np.array([2400.0, 2400.0]))


def test_overburden_empty_input_returns_empty():
    sigma = overburden_stress(np.array([]), np.array([]))
    assert sigma.shape == (0,)


# ---------------------------------------------------------------------
# geomechanics_indices (one-call wrapper)
# ---------------------------------------------------------------------


def test_geomechanics_indices_no_overburden_omits_closure_stress():
    """Without sigma_v_pa the closure_stress field is None."""
    moduli = elastic_moduli(vp=4500.0, vs=2500.0, rho=2400.0)
    out = geomechanics_indices(moduli)
    assert isinstance(out, GeomechanicsIndices)
    assert out.closure_stress is None
    # Shape consistency for the always-emitted fields.
    assert np.shape(out.brittleness) == np.shape(moduli.young)
    assert np.shape(out.fracability) == np.shape(moduli.young)
    assert np.shape(out.ucs)         == np.shape(moduli.young)
    assert np.shape(out.sand_stability) == np.shape(moduli.young)


def test_geomechanics_indices_with_overburden_returns_closure_stress():
    """sigma_v_pa supplied => closure stress is broadcast to the moduli shape."""
    n = 5
    vp = np.full(n, 4500.0)
    vs = np.full(n, 2500.0)
    rho = np.full(n, 2400.0)
    moduli = elastic_moduli(vp=vp, vs=vs, rho=rho)
    sigma_v = np.linspace(20.0e6, 40.0e6, n)
    out = geomechanics_indices(moduli, sigma_v_pa=sigma_v)
    assert out.closure_stress is not None
    assert out.closure_stress.shape == (n,)
    np.testing.assert_allclose(
        out.closure_stress,
        closure_stress(moduli.poisson, sigma_v),
    )


def test_geomechanics_indices_brittleness_matches_standalone_call():
    """Wrapper agrees with the standalone Rickman function."""
    moduli = elastic_moduli(
        vp=np.array([4000.0, 4500.0, 5000.0]),
        vs=np.array([2000.0, 2500.0, 3000.0]),
        rho=np.array([2400.0, 2400.0, 2400.0]),
    )
    out = geomechanics_indices(moduli)
    np.testing.assert_allclose(
        out.brittleness,
        brittleness_index_rickman(moduli.young, moduli.poisson),
    )


def test_geomechanics_indices_sand_stability_matches_standalone_call():
    """Wrapper's sand-stability flag matches the standalone shear-mu gate."""
    moduli = elastic_moduli(
        vp=np.array([3000.0, 4500.0]),
        vs=np.array([1000.0, 2500.0]),
        rho=np.array([2200.0, 2400.0]),
    )
    out = geomechanics_indices(moduli)
    np.testing.assert_array_equal(
        out.sand_stability,
        sand_stability_indicator(moduli.mu),
    )


def test_geomechanics_indices_pipes_into_track_to_log_curves_path(tmp_path):
    """End-to-end: moduli -> indices -> LAS round-trip via _FWAP_UNITS."""
    from fwap.io import read_las, write_las

    n = 4
    depth = np.linspace(1000.0, 1003.0, n)
    moduli = elastic_moduli(
        vp=np.full(n, 4500.0),
        vs=np.full(n, 2500.0),
        rho=np.full(n, 2400.0),
    )
    sigma_v = np.linspace(20.0e6, 23.0e6, n)
    out = geomechanics_indices(moduli, sigma_v_pa=sigma_v,
                               pore_pressure_pa=10.0e6)
    curves = {
        "BRIT": out.brittleness,
        "FRAC": out.fracability,
        "UCS":  out.ucs,
        "SH":   out.closure_stress,
        "SV":   sigma_v,
        "SAND": out.sand_stability.astype(float),
    }
    path = str(tmp_path / "geomech.las")
    write_las(path, depth, curves, well_name="GEO")
    loaded = read_las(path)
    assert loaded.units["UCS"] == "Pa"
    assert loaded.units["SH"]  == "Pa"
    assert loaded.units["BRIT"] == ""
    np.testing.assert_allclose(loaded.curves["BRIT"], out.brittleness,
                               rtol=0, atol=1e-3)
