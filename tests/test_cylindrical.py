"""Surface-wave speed and cylindrical-dispersion tests."""

from __future__ import annotations

import numpy as np
import pytest

from fwap.cylindrical import flexural_dispersion_physical, rayleigh_speed

# ------------------------------------------------------------------
# Rayleigh speed
# ------------------------------------------------------------------


def test_rayleigh_speed_poisson_0_25():
    """For Poisson = 0.25 (Vp/Vs = sqrt(3)), V_R/Vs is approximately 0.9194.

    The exact ratio is the positive real root of the Rayleigh cubic;
    published tabulated values give 0.9194 at nu = 0.25 (see
    Viktorov 1967 Table 1-1). We check to 3 decimal places.
    """
    vs = 2500.0
    vp = vs * np.sqrt(3.0)
    v_r = rayleigh_speed(vp, vs)
    assert abs(v_r / vs - 0.9194) < 1.0e-3


def test_rayleigh_speed_poisson_0_33():
    """Higher Poisson's ratio gives a slightly higher V_R/Vs ratio.

    For nu = 1/3 (common for shales), Vp/Vs = 2 and the Rayleigh
    ratio is ~0.9325.
    """
    vs = 2500.0
    vp = 2.0 * vs
    v_r = rayleigh_speed(vp, vs)
    assert abs(v_r / vs - 0.9325) < 1.0e-3


def test_rayleigh_speed_monotone_in_poisson():
    """V_R/Vs increases monotonically with Poisson's ratio."""
    vs = 2500.0
    # Sweep Poisson from 0.1 to 0.45 (Vp/Vs from 1.5 to ~3.3).
    ratios = np.linspace(1.5, 3.0, 6)
    v_r_over_vs = np.array([
        rayleigh_speed(r * vs, vs) / vs for r in ratios
    ])
    assert np.all(np.diff(v_r_over_vs) > 0)


def test_rayleigh_speed_always_below_vs():
    """V_R is always strictly less than V_s."""
    for vp_ratio in [1.5, 1.7, 2.0, 2.5, 3.0]:
        v_r = rayleigh_speed(vp_ratio * 2500.0, 2500.0)
        assert v_r < 2500.0


def test_rayleigh_speed_rejects_invalid_inputs():
    """vs <= 0 or vp <= vs raise ValueError."""
    with pytest.raises(ValueError):
        rayleigh_speed(vp=4000.0, vs=0.0)
    with pytest.raises(ValueError):
        rayleigh_speed(vp=4000.0, vs=-10.0)
    with pytest.raises(ValueError):
        rayleigh_speed(vp=2000.0, vs=2500.0)   # vp < vs
    with pytest.raises(ValueError):
        rayleigh_speed(vp=2500.0, vs=2500.0)   # vp == vs


# ------------------------------------------------------------------
# Physical flexural dispersion
# ------------------------------------------------------------------


def test_flexural_dispersion_low_frequency_equals_shear_slowness():
    """At f -> 0 the flexural slowness equals 1 / vs."""
    vp, vs = 4500.0, 2500.0
    s_of_f = flexural_dispersion_physical(vp, vs, a_borehole=0.1)
    s0 = float(s_of_f(np.array([1.0e-3]))[0])   # essentially zero frequency
    assert abs(s0 - 1.0 / vs) / (1.0 / vs) < 1.0e-4


def test_flexural_dispersion_high_frequency_approaches_rayleigh():
    """At f -> inf the flexural slowness approaches the Rayleigh slowness."""
    vp, vs = 4500.0, 2500.0
    v_r = rayleigh_speed(vp, vs)
    s_of_f = flexural_dispersion_physical(vp, vs, a_borehole=0.1)
    s_inf = float(s_of_f(np.array([1.0e8]))[0])   # very high frequency
    assert abs(s_inf - 1.0 / v_r) / (1.0 / v_r) < 1.0e-3


def test_flexural_dispersion_monotone():
    """Slowness is monotonically increasing with frequency."""
    vp, vs = 4500.0, 2500.0
    s_of_f = flexural_dispersion_physical(vp, vs, a_borehole=0.1)
    f = np.linspace(10.0, 50_000.0, 201)
    s = s_of_f(f)
    assert np.all(np.diff(s) >= 0.0)


def test_flexural_dispersion_high_asymptote_is_poisson_dependent():
    """Physical model yields different s_high for different Poisson's ratios.

    The phenomenological model in fwap.synthetic uses a fixed 1.25
    factor regardless of Vp/Vs; the physical one should vary.
    """
    vs = 2500.0
    # nu = 0.25  -> Vp/Vs = sqrt(3) ~= 1.732
    # nu = 0.40  -> Vp/Vs = sqrt(3) ... wait, general relation:
    # Vp/Vs = sqrt(2(1-nu) / (1-2*nu)). For nu=0.40, Vp/Vs ~= 2.449.
    s_025 = flexural_dispersion_physical(vs * np.sqrt(3.0), vs, 0.1)(
        np.array([1.0e8]))[0]
    s_040 = flexural_dispersion_physical(vs * 2.449, vs, 0.1)(
        np.array([1.0e8]))[0]
    # Different by at least 0.5%.
    assert abs(float(s_025) - float(s_040)) / float(s_025) > 5.0e-3


def test_flexural_dispersion_shape_form():
    """Dispersion returned by the callable matches input frequency shape."""
    s_of_f = flexural_dispersion_physical(4500.0, 2500.0, 0.1)
    f = np.linspace(100.0, 5000.0, 51)
    s = s_of_f(f)
    assert s.shape == f.shape
    assert np.all(np.isfinite(s))


def test_flexural_dispersion_is_below_phenomenological_1_25_factor():
    """The Rayleigh-grounded high-f asymptote is below the 1.25 round number.

    For nu = 0.25 the Rayleigh ratio is ~0.9194, so the high-frequency
    slowness is ``1/0.9194/Vs ~= 1.088/Vs`` -- below the phenomenological
    ``1.25/Vs``. This is the whole point of the physics-grounded model.
    """
    from fwap.synthetic import dipole_flexural_dispersion
    vp, vs = 4500.0, 2500.0
    s_physical = float(
        flexural_dispersion_physical(vp, vs, 0.1)(np.array([1.0e8]))[0]
    )
    s_phenom = float(
        dipole_flexural_dispersion(vs, 0.1)(np.array([1.0e8]))[0]
    )
    assert s_physical < s_phenom
    assert s_physical > 1.0 / vs    # but still a real flexural dispersion
