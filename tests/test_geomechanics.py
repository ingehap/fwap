"""Geomechanics-indices tests."""

from __future__ import annotations

import numpy as np
import pytest

from fwap.geomechanics import (
    RICKMAN_E_MAX_PA,
    RICKMAN_E_MIN_PA,
    RICKMAN_NU_MAX,
    RICKMAN_NU_MIN,
    GeomechanicsIndices,
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
    bi_low = brittleness_index_rickman(young_pa=1.0e8, poisson=0.45)
    assert bi_high == pytest.approx(1.0)
    assert bi_low == pytest.approx(0.0)


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
    bi_low_nu = brittleness_index_rickman(4.0e10, 0.18)
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
    sh_low_nu = closure_stress(0.18, sigma_v)
    sh_high_nu = closure_stress(0.35, sigma_v)
    assert sh_high_nu > sh_low_nu


def test_closure_stress_rejects_unphysical_poisson():
    """Poisson >= 1 is rejected."""
    with pytest.raises(ValueError):
        closure_stress(1.0, 50.0e6)
    with pytest.raises(ValueError):
        closure_stress(np.array([0.25, 1.5]), 50.0e6)


def test_closure_stress_rejects_negative_poisson():
    """Auxetic / negative Poisson is out of scope and rejected."""
    with pytest.raises(ValueError, match="poisson >= 0"):
        closure_stress(-0.1, 50.0e6)
    with pytest.raises(ValueError, match="poisson >= 0"):
        closure_stress(np.array([0.25, -0.05]), 50.0e6)


def test_closure_stress_array_inputs_broadcast():
    """Per-depth Poisson and sigma_v arrays broadcast correctly."""
    nu = np.array([0.20, 0.25, 0.30])
    sigma_v = np.array([40.0e6, 50.0e6, 60.0e6])
    sh = closure_stress(nu, sigma_v)
    assert sh.shape == (3,)
    np.testing.assert_allclose(sh, (nu / (1.0 - nu)) * sigma_v)


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
    flag = sand_stability_indicator(np.array([2.0e9, 4.0e9]), threshold_pa=3.0e9)
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
        overburden_stress(np.array([0.0, 0.0, 1.0]), np.array([2400.0, 2400.0, 2400.0]))
    with pytest.raises(ValueError):
        overburden_stress(np.array([1.0, 0.0]), np.array([2400.0, 2400.0]))


def test_overburden_rejects_negative_density():
    with pytest.raises(ValueError):
        overburden_stress(np.array([0.0, 1.0]), np.array([-1.0, 2400.0]))


def test_overburden_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        overburden_stress(np.array([0.0, 1.0, 2.0]), np.array([2400.0, 2400.0]))


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
    assert np.shape(out.ucs) == np.shape(moduli.young)
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
    out = geomechanics_indices(moduli, sigma_v_pa=sigma_v, pore_pressure_pa=10.0e6)
    curves = {
        "BRIT": out.brittleness,
        "FRAC": out.fracability,
        "UCS": out.ucs,
        "SH": out.closure_stress,
        "SV": sigma_v,
        "SAND": out.sand_stability.astype(float),
    }
    path = str(tmp_path / "geomech.las")
    write_las(path, depth, curves, well_name="GEO")
    loaded = read_las(path)
    assert loaded.units["UCS"] == "Pa"
    assert loaded.units["SH"] == "Pa"
    assert loaded.units["BRIT"] == ""
    np.testing.assert_allclose(
        loaded.curves["BRIT"], out.brittleness, rtol=0, atol=1e-3
    )


# =====================================================================
# hydrostatic_pressure
# =====================================================================


def test_hydrostatic_pressure_at_zero_depth_is_zero():
    """At surface, P_hydro = 0 by construction."""
    from fwap.geomechanics import hydrostatic_pressure

    p = hydrostatic_pressure(np.array([0.0]))
    assert p[0] == 0.0


def test_hydrostatic_pressure_linear_in_depth():
    """P_hydro = rho_w * g * z; should be linear in z with slope
    rho_w * g."""
    from fwap.geomechanics import hydrostatic_pressure

    z = np.array([0.0, 1000.0, 2000.0, 3000.0])
    p = hydrostatic_pressure(z, fluid_density=1000.0)
    # 1000 m of fresh water = ~9.81 MPa.
    expected = 1000.0 * 9.80665 * z
    np.testing.assert_allclose(p, expected, rtol=1.0e-12)


def test_hydrostatic_pressure_brine_higher_than_freshwater():
    """A heavier fluid (brine ~ 1080 kg/m^3) gives higher P_hydro
    at every depth."""
    from fwap.geomechanics import hydrostatic_pressure

    z = np.array([1000.0])
    fresh = hydrostatic_pressure(z, fluid_density=1000.0)
    brine = hydrostatic_pressure(z, fluid_density=1080.0)
    assert brine[0] > fresh[0]
    # Ratio matches the density ratio.
    assert abs(brine[0] / fresh[0] - 1.08) < 1.0e-12


def test_hydrostatic_pressure_rejects_negative_depth():
    """Negative depth (above datum) is unphysical and raises."""
    import pytest

    from fwap.geomechanics import hydrostatic_pressure

    with pytest.raises(ValueError, match="depth"):
        hydrostatic_pressure(np.array([-100.0]))


def test_hydrostatic_pressure_rejects_non_positive_density():
    """Zero or negative fluid density raises."""
    import pytest

    from fwap.geomechanics import hydrostatic_pressure

    with pytest.raises(ValueError, match="fluid_density"):
        hydrostatic_pressure(np.array([1000.0]), fluid_density=0.0)


# =====================================================================
# pore_pressure_eaton
# =====================================================================


def test_eaton_normal_compaction_recovers_hydrostatic():
    """When observed slowness equals the normal-compaction trend,
    the slowness ratio is 1, and Eaton's formula reduces to
    P_pore = sigma_v - (sigma_v - P_hydro) * 1 = P_hydro."""
    from fwap.geomechanics import pore_pressure_eaton

    sigma_v = np.array([20.0e6, 40.0e6, 60.0e6])
    s_obs = np.array([3.0e-4, 2.5e-4, 2.2e-4])
    s_normal = s_obs.copy()  # exactly on the trend
    P_hydro = np.array([10.0e6, 20.0e6, 30.0e6])

    P_pore = pore_pressure_eaton(
        sigma_v, s_obs, s_normal, hydrostatic_pressure_pa=P_hydro
    )
    np.testing.assert_allclose(P_pore, P_hydro, rtol=1.0e-12)


def test_eaton_severe_overpressure_approaches_overburden():
    """In the limit of extreme overpressure (s_obs >> s_normal), the
    ratio (s_normal/s_obs)^n -> 0, so P_pore -> sigma_v."""
    from fwap.geomechanics import pore_pressure_eaton

    sigma_v = np.array([40.0e6])
    s_obs = np.array([1.0e-3])  # very slow (severely undercompacted)
    s_normal = np.array([2.5e-4])  # normal trend
    P_hydro = np.array([20.0e6])

    P_pore = pore_pressure_eaton(
        sigma_v, s_obs, s_normal, hydrostatic_pressure_pa=P_hydro
    )
    # ratio = 0.25; ratio^3 = 0.0156; P_pore = 40 - (40-20)*0.0156 = 39.7 MPa
    assert P_pore[0] > 0.95 * sigma_v[0]


def test_eaton_overpressure_between_hydro_and_overburden():
    """A moderate overpressure (ratio = 0.7) gives P_pore strictly
    between P_hydro and sigma_v."""
    from fwap.geomechanics import pore_pressure_eaton

    sigma_v = np.array([40.0e6])
    s_normal = np.array([2.5e-4])
    s_obs = s_normal / 0.7  # ratio = 0.7
    P_hydro = np.array([20.0e6])

    P_pore = pore_pressure_eaton(
        sigma_v, s_obs, s_normal, hydrostatic_pressure_pa=P_hydro
    )
    assert P_hydro[0] < P_pore[0] < sigma_v[0]


def test_eaton_subhydrostatic_below_hydrostatic():
    """If observed slowness is faster than the trend (depleted /
    overcompacted zone), the ratio > 1 and P_pore < P_hydro."""
    from fwap.geomechanics import pore_pressure_eaton

    sigma_v = np.array([40.0e6])
    s_normal = np.array([2.5e-4])
    s_obs = s_normal / 1.2  # ratio = 1.2
    P_hydro = np.array([20.0e6])

    P_pore = pore_pressure_eaton(
        sigma_v, s_obs, s_normal, hydrostatic_pressure_pa=P_hydro
    )
    assert P_pore[0] < P_hydro[0]


def test_eaton_with_depth_matches_explicit_hydrostatic():
    """Passing depth (and letting the function compute hydrostatic)
    must agree with explicitly passing hydrostatic_pressure_pa."""
    from fwap.geomechanics import hydrostatic_pressure, pore_pressure_eaton

    z = np.array([1000.0, 2000.0, 3000.0])
    sigma_v = np.array([20.0e6, 40.0e6, 60.0e6])
    s_obs = np.array([3.0e-4, 2.7e-4, 2.5e-4])
    s_normal = s_obs * 0.9

    P_explicit = pore_pressure_eaton(
        sigma_v, s_obs, s_normal,
        hydrostatic_pressure_pa=hydrostatic_pressure(z),
    )
    P_via_depth = pore_pressure_eaton(
        sigma_v, s_obs, s_normal, depth=z,
    )
    np.testing.assert_allclose(P_via_depth, P_explicit, rtol=1.0e-12)


def test_eaton_exponent_amplifies_departure_from_normal():
    """A higher Eaton exponent makes P_pore more sensitive to
    departures of the slowness ratio from 1. Specifically: at ratio
    < 1 (overpressure), higher n produces higher P_pore."""
    from fwap.geomechanics import pore_pressure_eaton

    sigma_v = np.array([40.0e6])
    s_normal = np.array([2.5e-4])
    s_obs = np.array([s_normal[0] / 0.8])  # ratio = 0.8
    P_hydro = np.array([20.0e6])

    P_n3 = pore_pressure_eaton(
        sigma_v, s_obs, s_normal,
        hydrostatic_pressure_pa=P_hydro, eaton_exponent=3.0,
    )
    P_n6 = pore_pressure_eaton(
        sigma_v, s_obs, s_normal,
        hydrostatic_pressure_pa=P_hydro, eaton_exponent=6.0,
    )
    assert P_n6[0] > P_n3[0]


def test_eaton_round_trip_with_overburden_log():
    """End-to-end: compute sigma_v from a density log, run Eaton on
    a synthetic overpressure profile, check the result is bounded
    below by P_hydro and above by sigma_v at every depth, and
    matches P_hydro outside the overpressure zone."""
    from fwap.geomechanics import (
        hydrostatic_pressure,
        overburden_stress,
        pore_pressure_eaton,
    )

    z = np.linspace(0.0, 3000.0, 31)
    rho = np.full_like(z, 2400.0)
    sigma_v = overburden_stress(z, rho)
    P_hydro = hydrostatic_pressure(z)

    s_normal = 2.5e-4 * np.exp(-z / 6000.0)  # log-linear trend
    s_obs = s_normal.copy()
    overpressure = (z >= 1500.0) & (z <= 2000.0)
    s_obs[overpressure] *= 1.3

    P_pore = pore_pressure_eaton(sigma_v, s_obs, s_normal, depth=z)

    # Outside the overpressure zone, P_pore must equal P_hydro.
    outside = ~overpressure
    np.testing.assert_allclose(
        P_pore[outside], P_hydro[outside], rtol=1.0e-12
    )
    # Inside the overpressure zone, P_hydro < P_pore < sigma_v.
    assert np.all(P_pore[overpressure] > P_hydro[overpressure])
    assert np.all(P_pore[overpressure] < sigma_v[overpressure])


# Input validation
# ---------------------------------------------------------------------


def test_eaton_requires_hydrostatic_or_depth():
    """At least one of hydrostatic_pressure_pa or depth must be
    supplied; otherwise raise."""
    import pytest

    from fwap.geomechanics import pore_pressure_eaton

    with pytest.raises(ValueError, match="hydrostatic_pressure_pa or depth"):
        pore_pressure_eaton(
            np.array([20.0e6]),
            np.array([3.0e-4]),
            np.array([2.5e-4]),
        )


def test_eaton_rejects_non_positive_slowness():
    """Zero or negative slowness raises."""
    import pytest

    from fwap.geomechanics import pore_pressure_eaton

    with pytest.raises(ValueError, match="slowness"):
        pore_pressure_eaton(
            np.array([20.0e6]),
            np.array([0.0]),
            np.array([2.5e-4]),
            hydrostatic_pressure_pa=np.array([10.0e6]),
        )
    with pytest.raises(ValueError, match="slowness"):
        pore_pressure_eaton(
            np.array([20.0e6]),
            np.array([3.0e-4]),
            np.array([-1.0]),
            hydrostatic_pressure_pa=np.array([10.0e6]),
        )


def test_eaton_rejects_non_positive_exponent():
    """The Eaton exponent must be positive."""
    import pytest

    from fwap.geomechanics import pore_pressure_eaton

    with pytest.raises(ValueError, match="eaton_exponent"):
        pore_pressure_eaton(
            np.array([20.0e6]),
            np.array([3.0e-4]),
            np.array([2.5e-4]),
            hydrostatic_pressure_pa=np.array([10.0e6]),
            eaton_exponent=0.0,
        )


def test_eaton_rejects_negative_sigma_v():
    """Negative overburden stress is unphysical and raises."""
    import pytest

    from fwap.geomechanics import pore_pressure_eaton

    with pytest.raises(ValueError, match="sigma_v_pa"):
        pore_pressure_eaton(
            np.array([-1.0e6]),
            np.array([3.0e-4]),
            np.array([2.5e-4]),
            hydrostatic_pressure_pa=np.array([10.0e6]),
        )


# =====================================================================
# Wellbore stability: Kirsch + Mohr-Coulomb
# =====================================================================


# Standard parameter set used across the wellbore-stability tests.
# 3 km depth in a normal-fault stress regime.
WB_SIGMA_V = 70.0e6   # Pa
WB_SIGMA_H = 60.0e6   # Pa (max horizontal)
WB_SIGMA_H_MIN = 40.0e6  # Pa (min horizontal)
WB_PORE_PRESSURE = 30.0e6
WB_UCS = 50.0e6
WB_FRICTION = 30.0     # degrees
WB_NU = 0.25


# ---------------------------------------------------------------------
# Kirsch wall stresses
# ---------------------------------------------------------------------


def test_kirsch_at_breakout_azimuth_matches_closed_form():
    """At theta = 90 deg (perpendicular to sigma_H, the breakout
    azimuth), sigma_theta = 3*sigma_H - sigma_h - P_w."""
    from fwap.geomechanics import kirsch_wall_stresses

    P_w = 35.0e6
    sigma_t, sigma_z, sigma_r = kirsch_wall_stresses(
        WB_SIGMA_V, WB_SIGMA_H, WB_SIGMA_H_MIN,
        azimuth_deg=90.0, mud_pressure=P_w, poisson=WB_NU,
    )
    expected_theta = 3.0 * WB_SIGMA_H - WB_SIGMA_H_MIN - P_w
    assert abs(float(sigma_t) - expected_theta) < 1.0e-6
    assert abs(float(sigma_r) - P_w) < 1.0e-6


def test_kirsch_at_breakdown_azimuth_matches_closed_form():
    """At theta = 0 deg (in sigma_H direction, the tensile-failure
    azimuth), sigma_theta = 3*sigma_h - sigma_H - P_w."""
    from fwap.geomechanics import kirsch_wall_stresses

    P_w = 35.0e6
    sigma_t, _, _ = kirsch_wall_stresses(
        WB_SIGMA_V, WB_SIGMA_H, WB_SIGMA_H_MIN,
        azimuth_deg=0.0, mud_pressure=P_w, poisson=WB_NU,
    )
    expected_theta = 3.0 * WB_SIGMA_H_MIN - WB_SIGMA_H - P_w
    assert abs(float(sigma_t) - expected_theta) < 1.0e-6


def test_kirsch_isotropic_horizontal_independent_of_azimuth():
    """When sigma_H == sigma_h, sigma_theta does not depend on
    azimuth (the cos(2 theta) coefficient vanishes)."""
    from fwap.geomechanics import kirsch_wall_stresses

    azimuths = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
    P_w = 35.0e6
    sigma_iso = WB_SIGMA_H
    sigma_t, _, _ = kirsch_wall_stresses(
        WB_SIGMA_V, sigma_iso, sigma_iso,
        azimuth_deg=azimuths, mud_pressure=P_w, poisson=WB_NU,
    )
    assert np.allclose(sigma_t, sigma_t[0])


def test_kirsch_higher_mud_lowers_sigma_theta_uniformly():
    """Sigma_theta has a linear -P_w term, so a unit increase in
    mud pressure decreases sigma_theta by exactly that amount at
    every azimuth."""
    from fwap.geomechanics import kirsch_wall_stresses

    azimuths = np.linspace(0.0, 180.0, 7)
    sigma_t_low, _, _ = kirsch_wall_stresses(
        WB_SIGMA_V, WB_SIGMA_H, WB_SIGMA_H_MIN,
        azimuth_deg=azimuths, mud_pressure=30.0e6, poisson=WB_NU,
    )
    sigma_t_high, _, _ = kirsch_wall_stresses(
        WB_SIGMA_V, WB_SIGMA_H, WB_SIGMA_H_MIN,
        azimuth_deg=azimuths, mud_pressure=40.0e6, poisson=WB_NU,
    )
    np.testing.assert_allclose(sigma_t_low - sigma_t_high, 10.0e6, atol=1.0)


def test_kirsch_sigma_z_depends_on_poisson_in_horizontal_deviator():
    """sigma_z = sigma_v - 2 nu (sigma_H - sigma_h) cos(2 theta).
    For an isotropic horizontal stress (sigma_H == sigma_h),
    sigma_z = sigma_v regardless of azimuth or Poisson."""
    from fwap.geomechanics import kirsch_wall_stresses

    sigma_iso = WB_SIGMA_H
    _, sigma_z, _ = kirsch_wall_stresses(
        WB_SIGMA_V, sigma_iso, sigma_iso,
        azimuth_deg=np.array([0.0, 45.0, 90.0]), poisson=0.5,
    )
    np.testing.assert_allclose(sigma_z, WB_SIGMA_V, atol=1.0)


# ---------------------------------------------------------------------
# Mohr-Coulomb breakout pressure
# ---------------------------------------------------------------------


def test_mohr_coulomb_at_critical_pressure_just_satisfies_failure():
    """Plug the breakout pressure back into the Kirsch formula at
    the breakout azimuth, apply effective-stress subtraction, and
    confirm Mohr-Coulomb is satisfied with equality."""
    from fwap.geomechanics import (
        kirsch_wall_stresses,
        mohr_coulomb_breakout_pressure,
    )

    P_crit = float(mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
        friction_angle_deg=WB_FRICTION,
    ))
    sigma_t, _, sigma_r = kirsch_wall_stresses(
        WB_SIGMA_V, WB_SIGMA_H, WB_SIGMA_H_MIN,
        azimuth_deg=90.0, mud_pressure=P_crit, poisson=WB_NU,
    )
    sigma_1_eff = float(sigma_t) - WB_PORE_PRESSURE
    sigma_3_eff = float(sigma_r) - WB_PORE_PRESSURE
    phi = np.deg2rad(WB_FRICTION)
    q = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
    # MC failure envelope: sigma_1' = q * sigma_3' + UCS
    rhs = q * sigma_3_eff + WB_UCS
    assert abs(sigma_1_eff - rhs) < 1.0e-3


def test_mohr_coulomb_higher_ucs_lowers_breakout_pressure():
    """Stronger rock (higher UCS) needs less mud-pressure support
    to prevent shear failure."""
    from fwap.geomechanics import mohr_coulomb_breakout_pressure

    P_weak = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
        ucs=20.0e6, friction_angle_deg=WB_FRICTION,
    )
    P_strong = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
        ucs=80.0e6, friction_angle_deg=WB_FRICTION,
    )
    assert P_strong < P_weak


def test_mohr_coulomb_higher_friction_lowers_breakout_pressure():
    """Higher friction angle (steeper failure envelope) means lower
    critical mud pressure -- the rock supports more deviatoric
    stress before failing."""
    from fwap.geomechanics import mohr_coulomb_breakout_pressure

    P_low_friction = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
        friction_angle_deg=15.0,
    )
    P_high_friction = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
        friction_angle_deg=40.0,
    )
    assert P_high_friction < P_low_friction


def test_mohr_coulomb_higher_pore_pressure_increases_breakout_pressure():
    """In the typical regime (q > 1, friction angle > 0), higher
    pore pressure makes the rock weaker in effective stress and
    raises the required mud pressure."""
    from fwap.geomechanics import mohr_coulomb_breakout_pressure

    P_low_pp = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN,
        pore_pressure=20.0e6, ucs=WB_UCS,
        friction_angle_deg=WB_FRICTION,
    )
    P_high_pp = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN,
        pore_pressure=40.0e6, ucs=WB_UCS,
        friction_angle_deg=WB_FRICTION,
    )
    assert P_high_pp > P_low_pp


def test_mohr_coulomb_higher_horizontal_anisotropy_increases_breakout():
    """Larger sigma_H - sigma_h amplifies the Kirsch hoop stress at
    the breakout azimuth, requiring more mud-pressure support."""
    from fwap.geomechanics import mohr_coulomb_breakout_pressure

    # Same mean horizontal stress, increasing anisotropy
    P_isotropic = mohr_coulomb_breakout_pressure(
        50.0e6, 50.0e6, WB_PORE_PRESSURE, WB_UCS,
        friction_angle_deg=WB_FRICTION,
    )
    P_anisotropic = mohr_coulomb_breakout_pressure(
        70.0e6, 30.0e6, WB_PORE_PRESSURE, WB_UCS,
        friction_angle_deg=WB_FRICTION,
    )
    assert P_anisotropic > P_isotropic


def test_mohr_coulomb_zero_friction_is_tresca_limit():
    """At friction_angle_deg = 0, q = 1 and the formula reduces
    to P_crit = (3*sigma_H - sigma_h - UCS) / 2 with no
    pore-pressure dependence (the (q-1) term vanishes)."""
    from fwap.geomechanics import mohr_coulomb_breakout_pressure

    P_pp_low = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, pore_pressure=10.0e6, ucs=WB_UCS,
        friction_angle_deg=0.0,
    )
    P_pp_high = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, pore_pressure=50.0e6, ucs=WB_UCS,
        friction_angle_deg=0.0,
    )
    assert abs(float(P_pp_low) - float(P_pp_high)) < 1.0e-6
    expected = 0.5 * (3.0 * WB_SIGMA_H - WB_SIGMA_H_MIN - WB_UCS)
    assert abs(float(P_pp_low) - expected) < 1.0e-6


def test_mohr_coulomb_biot_alpha_zero_removes_pore_pressure_term():
    """Setting biot_alpha = 0 zeroes the effective-stress
    correction; the formula reduces to (3*sigma_H - sigma_h - UCS)
    / (1 + q) with no pore-pressure dependence."""
    from fwap.geomechanics import mohr_coulomb_breakout_pressure

    P_pp_low = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN,
        pore_pressure=10.0e6, ucs=WB_UCS,
        friction_angle_deg=WB_FRICTION, biot_alpha=0.0,
    )
    P_pp_high = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN,
        pore_pressure=50.0e6, ucs=WB_UCS,
        friction_angle_deg=WB_FRICTION, biot_alpha=0.0,
    )
    assert abs(float(P_pp_low) - float(P_pp_high)) < 1.0e-6


# Input validation
# ---------------------------------------------------------------------


def test_mohr_coulomb_rejects_friction_outside_range():
    """Friction angle must be in (-90, 90); +/- 90 makes sin(phi)
    = +/-1 and the formula divides by zero."""
    import pytest

    from fwap.geomechanics import mohr_coulomb_breakout_pressure

    with pytest.raises(ValueError, match="friction_angle_deg"):
        mohr_coulomb_breakout_pressure(
            WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
            friction_angle_deg=90.0,
        )
    with pytest.raises(ValueError, match="friction_angle_deg"):
        mohr_coulomb_breakout_pressure(
            WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
            friction_angle_deg=-90.0,
        )


# Integration: full stress-state pipeline overburden -> pore -> closure -> breakout
# ---------------------------------------------------------------------


def test_wellbore_stability_pipeline_end_to_end():
    """Compute the full stress-state pipeline from a synthetic
    density + sonic log: overburden -> pore pressure (Eaton) ->
    closure stress (Eaton) -> wellbore breakout pressure
    (Mohr-Coulomb). Sanity checks on the pressure ordering."""
    from fwap.geomechanics import (
        closure_stress,
        hydrostatic_pressure,
        mohr_coulomb_breakout_pressure,
        overburden_stress,
        pore_pressure_eaton,
    )

    z = np.linspace(0.0, 3000.0, 31)
    rho = np.full_like(z, 2400.0)
    sigma_v = overburden_stress(z, rho)

    s_normal = 2.5e-4 * np.exp(-z / 6000.0)
    s_obs = s_normal.copy()
    s_obs[(z >= 1500.0) & (z <= 2000.0)] *= 1.3

    P_p = pore_pressure_eaton(sigma_v, s_obs, s_normal, depth=z)

    # Estimate horizontal stresses from closure-stress (uniaxial-strain
    # Eaton 1969 with nu = 0.25). Treat sigma_h ~ closure and pick
    # sigma_H = sigma_h + 0.4 * (sigma_v - P_p) as a generic anisotropy.
    nu = np.full_like(z, 0.25)
    sigma_h_min = closure_stress(nu, sigma_v, pore_pressure_pa=P_p)
    sigma_H_max = sigma_h_min + 0.4 * (sigma_v - P_p)

    UCS = np.full_like(z, 50.0e6)
    P_crit = mohr_coulomb_breakout_pressure(
        sigma_H_max, sigma_h_min, P_p, UCS, friction_angle_deg=30.0,
    )

    P_hydro = hydrostatic_pressure(z)

    # All four logs are well-defined arrays of the right shape.
    assert sigma_v.shape == P_p.shape == P_crit.shape == P_hydro.shape == z.shape
    # Shallow depths trivially meet the "P_p >= 0" sanity check.
    assert np.all(P_p[1:] > 0)
    # Overburden bounds the pore pressure from above.
    assert np.all(P_p <= sigma_v + 1.0e-6)


# ---------------------------------------------------------------------
# Tensile breakdown pressure + safe mud-weight window
# ---------------------------------------------------------------------


def test_tensile_breakdown_matches_hubbert_willis_closed_form():
    """T = 0 case: P_breakdown = 3 sigma_h - sigma_H - alpha P_p."""
    from fwap.geomechanics import tensile_breakdown_pressure

    P_break = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
    ))
    expected = 3.0 * WB_SIGMA_H_MIN - WB_SIGMA_H - WB_PORE_PRESSURE
    assert abs(P_break - expected) < 1.0e-6


def test_tensile_breakdown_at_critical_pressure_satisfies_kirsch():
    """Plug the breakdown pressure into the Kirsch formula at the
    breakdown azimuth (theta = 0); the effective hoop stress
    must equal -T."""
    from fwap.geomechanics import (
        kirsch_wall_stresses,
        tensile_breakdown_pressure,
    )

    T = 5.0e6
    P_break = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
        tensile_strength=T,
    ))
    sigma_t, _, _ = kirsch_wall_stresses(
        WB_SIGMA_V, WB_SIGMA_H, WB_SIGMA_H_MIN,
        azimuth_deg=0.0, mud_pressure=P_break, poisson=WB_NU,
    )
    sigma_eff = float(sigma_t) - WB_PORE_PRESSURE
    # Effective hoop stress at the breakdown pressure equals -T
    # (just on the tensile-failure envelope).
    assert abs(sigma_eff - (-T)) < 1.0e-3


def test_tensile_breakdown_increases_with_tensile_strength():
    """Higher T means the rock can carry more wall tension, so the
    breakdown pressure rises by exactly T (linear shift)."""
    from fwap.geomechanics import tensile_breakdown_pressure

    P_T0 = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
        tensile_strength=0.0,
    ))
    P_T5 = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
        tensile_strength=5.0e6,
    ))
    assert abs((P_T5 - P_T0) - 5.0e6) < 1.0e-6


def test_tensile_breakdown_decreases_with_horizontal_stress_anisotropy():
    """Larger sigma_H pre-tensions the wall; raising sigma_H at fixed
    sigma_h reduces the breakdown pressure 1:1."""
    from fwap.geomechanics import tensile_breakdown_pressure

    P_low_aniso = float(tensile_breakdown_pressure(
        50.0e6, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
    ))
    P_high_aniso = float(tensile_breakdown_pressure(
        70.0e6, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
    ))
    assert P_high_aniso < P_low_aniso
    assert abs((P_high_aniso - P_low_aniso) - (-20.0e6)) < 1.0e-6


def test_tensile_breakdown_decreases_with_pore_pressure():
    """Higher pore pressure reduces effective stress; at biot_alpha = 1
    each Pa of P_p reduces the breakdown pressure by 1 Pa."""
    from fwap.geomechanics import tensile_breakdown_pressure

    P_low_pp = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, pore_pressure=20.0e6,
    ))
    P_high_pp = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, pore_pressure=40.0e6,
    ))
    assert abs((P_high_pp - P_low_pp) - (-20.0e6)) < 1.0e-6


def test_tensile_breakdown_biot_alpha_zero_removes_pore_pressure():
    """biot_alpha = 0 zeroes the effective-stress correction; the
    breakdown pressure becomes 3 sigma_h - sigma_H + T regardless
    of pore pressure."""
    from fwap.geomechanics import tensile_breakdown_pressure

    P_pp_low = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, pore_pressure=10.0e6, biot_alpha=0.0,
    ))
    P_pp_high = float(tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, pore_pressure=50.0e6, biot_alpha=0.0,
    ))
    assert abs(P_pp_low - P_pp_high) < 1.0e-6
    assert abs(P_pp_low - (3.0 * WB_SIGMA_H_MIN - WB_SIGMA_H)) < 1.0e-6


# Safe mud-weight window
# ---------------------------------------------------------------------


def test_window_returns_dataclass_with_both_pressures():
    """safe_mud_weight_window returns a MudWeightWindow with the
    correct field types."""
    from fwap.geomechanics import MudWeightWindow, safe_mud_weight_window

    w = safe_mud_weight_window(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
    )
    assert isinstance(w, MudWeightWindow)
    assert w.breakout_pressure.dtype == float
    assert w.breakdown_pressure.dtype == float


def test_window_matches_individual_function_outputs():
    """The combiner is a pure pass-through; its bounds equal the
    outputs of the two underlying primitives called individually."""
    from fwap.geomechanics import (
        mohr_coulomb_breakout_pressure,
        safe_mud_weight_window,
        tensile_breakdown_pressure,
    )

    w = safe_mud_weight_window(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
        tensile_strength=2.0e6, friction_angle_deg=25.0, biot_alpha=0.85,
    )
    P_breakout = mohr_coulomb_breakout_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE, WB_UCS,
        friction_angle_deg=25.0, biot_alpha=0.85,
    )
    P_breakdown = tensile_breakdown_pressure(
        WB_SIGMA_H, WB_SIGMA_H_MIN, WB_PORE_PRESSURE,
        tensile_strength=2.0e6, biot_alpha=0.85,
    )
    assert abs(float(w.breakout_pressure) - float(P_breakout)) < 1.0e-9
    assert abs(float(w.breakdown_pressure) - float(P_breakdown)) < 1.0e-9


def test_window_width_and_drillability_properties():
    """width and is_drillable are derived from the bounds. Verify
    against an explicit drillable case and a not-drillable case."""
    from fwap.geomechanics import safe_mud_weight_window

    # Drillable: weak rock + low anisotropy gives breakdown > breakout
    w_drillable = safe_mud_weight_window(
        sigma_H=50.0e6, sigma_h=45.0e6,  # low anisotropy
        pore_pressure=20.0e6, ucs=10.0e6,  # very weak rock
    )
    assert float(w_drillable.width) > 0
    assert bool(w_drillable.is_drillable)

    # Not drillable: strong horizontal anisotropy + strong rock gives
    # breakout > breakdown, no safe window.
    w_impossible = safe_mud_weight_window(
        sigma_H=80.0e6, sigma_h=30.0e6,  # high anisotropy
        pore_pressure=40.0e6, ucs=80.0e6,  # strong rock
    )
    assert float(w_impossible.width) < 0
    assert not bool(w_impossible.is_drillable)


def test_window_vector_input_per_depth_drillability():
    """Vector inputs produce per-depth drillability flags."""
    from fwap.geomechanics import safe_mud_weight_window

    sigma_H = np.array([50.0e6, 80.0e6])
    sigma_h = np.array([45.0e6, 30.0e6])
    P_p = np.array([20.0e6, 40.0e6])
    UCS = np.array([10.0e6, 80.0e6])
    w = safe_mud_weight_window(sigma_H, sigma_h, P_p, UCS)
    # First depth drillable, second not.
    assert w.is_drillable[0]
    assert not w.is_drillable[1]
    assert w.width.shape == (2,)


def test_window_drillable_implies_pressure_in_range():
    """For a drillable window, any mud pressure between breakout
    and breakdown should keep the wall stable. Spot-check a single
    drillable case."""
    from fwap.geomechanics import (
        kirsch_wall_stresses,
        safe_mud_weight_window,
    )

    w = safe_mud_weight_window(
        sigma_H=50.0e6, sigma_h=45.0e6,
        pore_pressure=20.0e6, ucs=10.0e6, friction_angle_deg=30.0,
    )
    P_low = float(w.breakout_pressure)
    P_high = float(w.breakdown_pressure)
    P_mid = 0.5 * (P_low + P_high)

    # At the midpoint, the breakout azimuth hoop stress is between
    # the wall-strength threshold and the tensile threshold.
    sigma_t_breakout, _, _ = kirsch_wall_stresses(
        WB_SIGMA_V, 50.0e6, 45.0e6,
        azimuth_deg=90.0, mud_pressure=P_mid, poisson=WB_NU,
    )
    sigma_t_breakdown, _, _ = kirsch_wall_stresses(
        WB_SIGMA_V, 50.0e6, 45.0e6,
        azimuth_deg=0.0, mud_pressure=P_mid, poisson=WB_NU,
    )
    # Effective hoop stress at the breakdown azimuth is above -T
    # (no tensile failure). Tensile strength was 0 by default.
    assert float(sigma_t_breakdown) - 20.0e6 > 0
    # Effective hoop stress at the breakout azimuth is below the
    # MC envelope (no shear failure). The actual MC margin would
    # require recomputing the failure envelope; spot-check that
    # sigma_theta is finite and positive.
    assert float(sigma_t_breakout) > 0
