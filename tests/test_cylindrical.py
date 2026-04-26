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


# ---------------------------------------------------------------------
# flexural_dispersion_vti_physical
# ---------------------------------------------------------------------


def test_flexural_dispersion_vti_reduces_to_isotropic_when_vsh_eq_vsv():
    """Vsh == Vsv -> identical to flexural_dispersion_physical."""
    from fwap.cylindrical import (
        flexural_dispersion_physical,
        flexural_dispersion_vti_physical,
    )
    vp, vs = 4500.0, 2500.0
    a = 0.1
    f = np.linspace(500.0, 12000.0, 41)
    s_iso = flexural_dispersion_physical(vp, vs, a)(f)
    s_vti = flexural_dispersion_vti_physical(vp, vs, vs, a)(f)
    np.testing.assert_allclose(s_vti, s_iso, rtol=1.0e-12)


def test_flexural_dispersion_vti_low_freq_limit_is_inverse_vsv():
    """f -> 0: phase slowness approaches 1/Vsv."""
    import pytest
    from fwap.cylindrical import flexural_dispersion_vti_physical
    vp, vsv, vsh = 4500.0, 2500.0, 2700.0   # gamma > 0 typical shale
    s_of_f = flexural_dispersion_vti_physical(vp, vsv, vsh, 0.1)
    s_low = float(s_of_f(np.array([1.0]))[0])
    assert s_low == pytest.approx(1.0 / vsv, rel=1.0e-6)


def test_flexural_dispersion_vti_high_freq_limit_is_inverse_rayleigh_vsh():
    """f -> infinity: phase slowness approaches 1/V_R(vp, vsh)."""
    import pytest
    from fwap.cylindrical import (
        flexural_dispersion_vti_physical,
        rayleigh_speed,
    )
    vp, vsv, vsh = 4500.0, 2500.0, 2700.0
    s_of_f = flexural_dispersion_vti_physical(vp, vsv, vsh, 0.1)
    s_high = float(s_of_f(np.array([1.0e9]))[0])
    expected = 1.0 / rayleigh_speed(vp, vsh)
    assert s_high == pytest.approx(expected, rel=1.0e-6)


def test_flexural_dispersion_vti_high_freq_uses_horizontal_shear():
    """At high frequency, Vsh sets the surface-wave speed:
    increasing Vsh at fixed Vsv decreases the high-f asymptote."""
    from fwap.cylindrical import flexural_dispersion_vti_physical
    vp, vsv = 4500.0, 2500.0
    f_hi = np.array([1.0e8])
    s_isotropic = float(
        flexural_dispersion_vti_physical(vp, vsv, vsv, 0.1)(f_hi)[0])
    s_vti = float(
        flexural_dispersion_vti_physical(vp, vsv, 1.2 * vsv, 0.1)(f_hi)[0])
    assert s_vti < s_isotropic


def test_flexural_dispersion_vti_low_freq_uses_vertical_shear():
    """At low frequency, the dispersion still anchors to 1/Vsv even
    when Vsh changes."""
    import pytest
    from fwap.cylindrical import flexural_dispersion_vti_physical
    vp, vsv = 4500.0, 2500.0
    f_lo = np.array([1.0])
    s_a = float(
        flexural_dispersion_vti_physical(vp, vsv, vsv, 0.1)(f_lo)[0])
    s_b = float(
        flexural_dispersion_vti_physical(vp, vsv, 1.3 * vsv, 0.1)(f_lo)[0])
    assert s_a == pytest.approx(s_b, rel=1.0e-6)


def test_flexural_dispersion_vti_vector_input_broadcasts():
    """Frequency arrays produce same-shape slowness arrays."""
    from fwap.cylindrical import flexural_dispersion_vti_physical
    f = np.linspace(100.0, 10_000.0, 25)
    s = flexural_dispersion_vti_physical(4500.0, 2500.0, 2700.0, 0.1)(f)
    assert s.shape == f.shape
    assert np.all(np.isfinite(s))


def test_flexural_dispersion_vti_monotonic_dispersion_curve():
    """For a strongly anisotropic shale where Vsh is much larger
    than Vsv, the high-f Rayleigh limit (~ 0.92 * Vsh) beats the
    low-f Vsv anchor and the dispersion is monotonically
    decreasing in f."""
    from fwap.cylindrical import flexural_dispersion_vti_physical
    f = np.linspace(500.0, 20000.0, 50)
    s = flexural_dispersion_vti_physical(4500.0, 2500.0, 2900.0, 0.1)(f)
    assert np.all(np.diff(s) < 0.0)


def test_flexural_dispersion_vti_rejects_unphysical_velocities():
    """vp <= vsv or vp <= vsh, or non-positive shear -> ValueError."""
    import pytest
    from fwap.cylindrical import flexural_dispersion_vti_physical
    with pytest.raises(ValueError, match="vsv"):
        flexural_dispersion_vti_physical(4500.0, 0.0, 2700.0, 0.1)
    with pytest.raises(ValueError, match="vsh"):
        flexural_dispersion_vti_physical(4500.0, 2500.0, -1.0, 0.1)
    with pytest.raises(ValueError, match="vp > vsv"):
        flexural_dispersion_vti_physical(2500.0, 2500.0, 2700.0, 0.1)
    with pytest.raises(ValueError, match="vp > vsh"):
        flexural_dispersion_vti_physical(2600.0, 2500.0, 2700.0, 0.1)


def test_flexural_dispersion_vti_works_with_dispersive_stc():
    """The VTI law is drop-in for dispersive_stc: synthesize a
    flexural arrival with this dispersion law, then dispersive_stc
    using the same family recovers the formation Vsv."""
    from fwap.coherence import find_peaks
    from fwap.cylindrical import flexural_dispersion_vti_physical
    from fwap.dispersion import dispersive_stc
    from fwap.synthetic import (
        ArrayGeometry, Mode, synthesize_gather,
    )
    Vsv, Vsh = 2500.0, 2700.0
    vp = 4500.0
    a_borehole = 0.1
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=2.0e-5, n_samples=2048)
    disp_truth = flexural_dispersion_vti_physical(vp, Vsv, Vsh, a_borehole)
    mode = Mode("Flex", slowness=1.0/Vsv, f0=4000.0, amplitude=1.0,
                dispersion=disp_truth)
    data = synthesize_gather(geom, [mode], noise=0.02, seed=11)

    def family(s_shear: float):
        # Match the planted Vsh by holding the Vsh/Vsv ratio fixed
        # across the scan.
        ratio = Vsh / Vsv
        vsv_trial = 1.0 / s_shear
        vsh_trial = ratio * vsv_trial
        return flexural_dispersion_vti_physical(
            vp, vsv_trial, vsh_trial, a_borehole)

    # Slowness range chosen so Vsh = ratio * Vsv stays below Vp =
    # 4500 m/s across the entire scan (the family validates
    # vp > vsh internally).
    surf = dispersive_stc(
        data, dt=geom.dt, offsets=geom.offsets,
        dispersion_family=family,
        shear_slowness_range=(260e-6, 600e-6),
        n_slowness=81, f_range=(500.0, 6000.0),
        window_length=1.5e-3, time_step=4,
    )
    peaks = find_peaks(surf, threshold=0.5)
    assert peaks.size > 0
    s_recovered = peaks[0, 0]
    # Within 5% of planted 1/Vsv.
    assert abs(s_recovered - 1.0 / Vsv) / (1.0 / Vsv) < 0.05
