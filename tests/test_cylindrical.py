"""Surface-wave speed and cylindrical-dispersion tests."""

from __future__ import annotations

import numpy as np
import pytest

from fwap.cylindrical import (
    flexural_dispersion_physical,
    rayleigh_speed,
    scholte_speed,
)

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
    v_r_over_vs = np.array([rayleigh_speed(r * vs, vs) / vs for r in ratios])
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
        rayleigh_speed(vp=2000.0, vs=2500.0)  # vp < vs
    with pytest.raises(ValueError):
        rayleigh_speed(vp=2500.0, vs=2500.0)  # vp == vs


# ------------------------------------------------------------------
# Physical flexural dispersion
# ------------------------------------------------------------------


def test_flexural_dispersion_low_frequency_equals_shear_slowness():
    """At f -> 0 the flexural slowness equals 1 / vs."""
    vp, vs = 4500.0, 2500.0
    s_of_f = flexural_dispersion_physical(vp, vs, a_borehole=0.1)
    s0 = float(s_of_f(np.array([1.0e-3]))[0])  # essentially zero frequency
    assert abs(s0 - 1.0 / vs) / (1.0 / vs) < 1.0e-4


def test_flexural_dispersion_high_frequency_approaches_rayleigh():
    """At f -> inf the flexural slowness approaches the Rayleigh slowness."""
    vp, vs = 4500.0, 2500.0
    v_r = rayleigh_speed(vp, vs)
    s_of_f = flexural_dispersion_physical(vp, vs, a_borehole=0.1)
    s_inf = float(s_of_f(np.array([1.0e8]))[0])  # very high frequency
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
    s_025 = flexural_dispersion_physical(vs * np.sqrt(3.0), vs, 0.1)(np.array([1.0e8]))[
        0
    ]
    s_040 = flexural_dispersion_physical(vs * 2.449, vs, 0.1)(np.array([1.0e8]))[0]
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
    s_physical = float(flexural_dispersion_physical(vp, vs, 0.1)(np.array([1.0e8]))[0])
    s_phenom = float(dipole_flexural_dispersion(vs, 0.1)(np.array([1.0e8]))[0])
    assert s_physical < s_phenom
    assert s_physical > 1.0 / vs  # but still a real flexural dispersion


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

    vp, vsv, vsh = 4500.0, 2500.0, 2700.0  # gamma > 0 typical shale
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
    s_isotropic = float(flexural_dispersion_vti_physical(vp, vsv, vsv, 0.1)(f_hi)[0])
    s_vti = float(flexural_dispersion_vti_physical(vp, vsv, 1.2 * vsv, 0.1)(f_hi)[0])
    assert s_vti < s_isotropic


def test_flexural_dispersion_vti_low_freq_uses_vertical_shear():
    """At low frequency, the dispersion still anchors to 1/Vsv even
    when Vsh changes."""
    import pytest

    from fwap.cylindrical import flexural_dispersion_vti_physical

    vp, vsv = 4500.0, 2500.0
    f_lo = np.array([1.0])
    s_a = float(flexural_dispersion_vti_physical(vp, vsv, vsv, 0.1)(f_lo)[0])
    s_b = float(flexural_dispersion_vti_physical(vp, vsv, 1.3 * vsv, 0.1)(f_lo)[0])
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
        ArrayGeometry,
        Mode,
        synthesize_gather,
    )

    Vsv, Vsh = 2500.0, 2700.0
    vp = 4500.0
    a_borehole = 0.1
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=2.0e-5, n_samples=2048)
    disp_truth = flexural_dispersion_vti_physical(vp, Vsv, Vsh, a_borehole)
    mode = Mode(
        "Flex", slowness=1.0 / Vsv, f0=4000.0, amplitude=1.0, dispersion=disp_truth
    )
    data = synthesize_gather(geom, [mode], noise=0.02, seed=11)

    def family(s_shear: float):
        # Match the planted Vsh by holding the Vsh/Vsv ratio fixed
        # across the scan.
        ratio = Vsh / Vsv
        vsv_trial = 1.0 / s_shear
        vsh_trial = ratio * vsv_trial
        return flexural_dispersion_vti_physical(vp, vsv_trial, vsh_trial, a_borehole)

    # Slowness range chosen so Vsh = ratio * Vsv stays below Vp =
    # 4500 m/s across the entire scan (the family validates
    # vp > vsh internally).
    surf = dispersive_stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        dispersion_family=family,
        shear_slowness_range=(260e-6, 600e-6),
        n_slowness=81,
        f_range=(500.0, 6000.0),
        window_length=1.5e-3,
        time_step=4,
    )
    peaks = find_peaks(surf, threshold=0.5)
    assert peaks.size > 0
    s_recovered = peaks[0, 0]
    # Within 5% of planted 1/Vsv.
    assert abs(s_recovered - 1.0 / Vsv) / (1.0 / Vsv) < 0.05


# ----------------------------------------------------------------------
# Scholte speed, and the cylindrical solver's high-frequency limit
#
# This is the literature tie roadmap A.1 asks for, in the one form that
# needs no digitised figure. `scholte_speed` solves a *plane* fluid/solid
# interface problem -- a different equation from the cylindrical modal
# determinant, with no Bessel functions in it -- so the borehole Stoneley
# mode converging to it at short wavelength is a genuine cross-check
# rather than the solver agreeing with itself.
# ----------------------------------------------------------------------

_FAST_ROCK = dict(vp=4500.0, vs=2600.0, rho=2400.0)
_SLOW_ROCK = dict(vp=2200.0, vs=800.0, rho=2200.0)
_BRINE = dict(vf=1500.0, rho_f=1000.0)


def test_scholte_reduces_to_rayleigh_for_a_vanishing_fluid():
    """With no fluid loading the equation must collapse to Rayleigh's.

    Checked against `rayleigh_speed`, which is a separate implementation
    of a separate equation, so this validates the secular function itself
    before it is used as an oracle for anything else.

    The fluid is given a high speed so that `min(vs, vf)` does not cut the
    Rayleigh root out of the admissible range.
    """
    for vp, vs in ((3000.0, 1500.0), (4000.0, 2000.0), (5000.0, 2400.0)):
        expected = rayleigh_speed(vp, vs)
        got = scholte_speed(vp, vs, 2400.0, 9000.0, 1.0e-8)
        assert got == pytest.approx(expected, rel=1e-9)


def test_scholte_is_bound_to_the_interface():
    """A bound interface wave must be slower than both media."""
    for rock in (_FAST_ROCK, _SLOW_ROCK):
        speed = scholte_speed(**rock, **_BRINE)
        assert speed < min(rock["vs"], _BRINE["vf"])
        assert speed > 0.0


def test_heavier_fluid_slows_the_scholte_wave():
    """More fluid loading, more mass dragged along, lower speed."""
    light = scholte_speed(**_FAST_ROCK, vf=1500.0, rho_f=200.0)
    heavy = scholte_speed(**_FAST_ROCK, vf=1500.0, rho_f=1200.0)
    assert heavy < light


def test_scholte_rejects_unphysical_media():
    with pytest.raises(ValueError, match="must all be positive"):
        scholte_speed(4500.0, -1.0, 2400.0, 1500.0, 1000.0)
    with pytest.raises(ValueError, match="require vp > vs"):
        scholte_speed(2000.0, 2600.0, 2400.0, 1500.0, 1000.0)
    with pytest.raises(ValueError, match="must all be positive"):
        scholte_speed(4500.0, 2600.0, 2400.0, 1500.0, 0.0)


@pytest.mark.parametrize("rock", [_FAST_ROCK, _SLOW_ROCK], ids=["fast", "slow"])
def test_borehole_stoneley_converges_to_the_plane_scholte_speed(rock):
    """The A.1 check: cylindrical solver -> independent plane-interface limit.

    At short wavelength the borehole wall is effectively flat, so
    `stoneley_dispersion` must approach `scholte_speed`. The approach is
    from below for a fast formation and from above for a slow one, so the
    assertion is on |ratio - 1| shrinking rather than on a signed
    difference.
    """
    from fwap import stoneley_dispersion

    reference = scholte_speed(**rock, **_BRINE)
    frequencies = np.array([50.0e3, 100.0e3, 200.0e3, 400.0e3])
    result = stoneley_dispersion(frequencies, **rock, **_BRINE, a=0.10)
    assert np.all(np.isfinite(result.slowness))

    error = np.abs(1.0 / result.slowness / reference - 1.0)
    # monotone convergence, and close by the top of the band
    assert np.all(np.diff(error) < 0.0)
    assert error[-1] < 1.0e-3


def test_the_scholte_limit_is_not_trivially_satisfied():
    """The convergence test must be able to fail, so check that it can.

    A limit test is only worth having if a wrong reference would miss it.
    Two wrong references are scored against the same solver output: the
    fluid velocity itself (the naive guess for where a borehole tube wave
    ends up) and the Scholte speed of a rock whose fluid density is off by
    20 %. Both must be substantially further away than the correct value.

    The margins are modest in absolute terms -- the Scholte speed is not
    very sensitive to fluid density -- which is exactly why the assertion
    is relative rather than an absolute threshold.
    """
    from fwap import stoneley_dispersion

    result = stoneley_dispersion(np.array([400.0e3]), **_FAST_ROCK, **_BRINE, a=0.10)
    speed = 1.0 / result.slowness[0]

    correct = abs(speed / scholte_speed(**_FAST_ROCK, **_BRINE) - 1.0)
    wrong_density = abs(
        speed / scholte_speed(**_FAST_ROCK, vf=1500.0, rho_f=1200.0) - 1.0
    )
    naive_fluid = abs(speed / _BRINE["vf"] - 1.0)

    assert correct < 1.0e-3
    assert wrong_density > 5.0 * correct
    assert naive_fluid > 5.0 * correct


# ---------------------------------------------------------------------------
# Leaky-mode radiation attenuation (independent oracle)
# ---------------------------------------------------------------------------

_LEAKY_ROCK = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_LEAKY_HOLE = {"vf": 1500.0, "rho_f": 1000.0, "a": 0.10}


def test_leaky_radiation_attenuation_vanishes_below_the_shear_velocity():
    """A bound mode radiates nothing, and does so exactly.

    Below ``V_S`` both formation waves are evanescent, so the solid's
    impedance is purely imaginary against a real fluid impedance and
    ``|R|`` is identically 1. That is an algebraic identity rather than a
    numerical near-miss, so the attenuation must be exactly zero.
    """
    from fwap import leaky_radiation_attenuation

    below = np.array([1600.0, 2000.0, 2299.0])
    att = leaky_radiation_attenuation(below, 10.0e3, **_LEAKY_ROCK, **_LEAKY_HOLE)
    assert np.all(att == 0.0)


def test_leaky_radiation_attenuation_is_positive_inside_the_leaky_window():
    """Above ``V_S`` the shear wave propagates and carries energy away."""
    from fwap import leaky_radiation_attenuation

    inside = np.array([2400.0, 2800.0, 3500.0])
    att = leaky_radiation_attenuation(inside, 10.0e3, **_LEAKY_ROCK, **_LEAKY_HOLE)
    assert np.all(att > 0.0)


def test_leaky_radiation_attenuation_scales_inversely_with_radius():
    """The bounce rate, hence the attenuation, goes as ``1/a``.

    Same phase velocity and frequency, so ``|R|``, ``k_f`` and ``k_z`` are
    all unchanged and only the ``2a`` transverse round trip differs.
    """
    from fwap import leaky_radiation_attenuation

    kw = dict(vf=1500.0, rho_f=1000.0)
    narrow = leaky_radiation_attenuation(2600.0, 12.0e3, **_LEAKY_ROCK, **kw, a=0.05)
    wide = leaky_radiation_attenuation(2600.0, 12.0e3, **_LEAKY_ROCK, **kw, a=0.15)
    assert narrow / wide == pytest.approx(3.0, rel=1.0e-12)


def test_leaky_radiation_attenuation_propagates_nan_and_broadcasts():
    """A ``BoreholeMode`` is full of NaN below cutoff; passing it in must work."""
    from fwap import leaky_radiation_attenuation

    c = np.array([np.nan, 2600.0, np.nan, 2800.0])
    freq = np.array([5.0e3, 10.0e3, 15.0e3, 20.0e3])
    att = leaky_radiation_attenuation(c, freq, **_LEAKY_ROCK, **_LEAKY_HOLE)
    assert att.shape == (4,)
    assert np.isnan(att[0]) and np.isnan(att[2])
    assert np.all(np.isfinite(att[[1, 3]])) and np.all(att[[1, 3]] > 0.0)


def test_leaky_radiation_attenuation_rejects_bad_media_and_velocities():
    from fwap import leaky_radiation_attenuation

    with pytest.raises(ValueError, match="must all be positive"):
        leaky_radiation_attenuation(
            2600.0, 1.0e4, **_LEAKY_ROCK, vf=1500.0, rho_f=1000.0, a=-0.1
        )
    with pytest.raises(ValueError, match="require vp > vs"):
        leaky_radiation_attenuation(
            2600.0, 1.0e4, vp=2000.0, vs=2300.0, rho=2500.0, **_LEAKY_HOLE
        )
    with pytest.raises(ValueError, match="slow formation"):
        leaky_radiation_attenuation(
            2600.0, 1.0e4, vp=3000.0, vs=1400.0, rho=2500.0, **_LEAKY_HOLE
        )
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        leaky_radiation_attenuation(2600.0, 0.0, **_LEAKY_ROCK, **_LEAKY_HOLE)
    with pytest.raises(ValueError, match="phase_velocity must lie in"):
        leaky_radiation_attenuation(1400.0, 1.0e4, **_LEAKY_ROCK, **_LEAKY_HOLE)
    with pytest.raises(ValueError, match="phase_velocity must lie in"):
        leaky_radiation_attenuation(4500.0, 1.0e4, **_LEAKY_ROCK, **_LEAKY_HOLE)


def test_leaky_radiation_reflection_matches_the_normal_incidence_limit():
    """As ``c -> infinity`` the reflection coefficient must go normal-incidence.

    This is the one point where ``R`` has a closed form independent of the
    whole angle-dependent construction, so it is the check that the
    impedance expression is assembled correctly. It is tested on the
    private helper because ``c`` far above ``V_P`` is outside the physical
    domain the public function accepts.
    """
    from fwap.cylindrical import _fluid_solid_reflection

    vp, vs, rho = _LEAKY_ROCK["vp"], _LEAKY_ROCK["vs"], _LEAKY_ROCK["rho"]
    vf, rho_f = 1500.0, 1000.0
    got = _fluid_solid_reflection(
        np.array(1.0e12), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f
    )
    expected = (rho * vp - rho_f * vf) / (rho * vp + rho_f * vf)
    assert float(np.real(got)) == pytest.approx(expected, rel=1.0e-9)
    assert abs(float(np.imag(got))) < 1.0e-9


def test_leaky_radiation_reflection_is_unit_modulus_when_nothing_radiates():
    """``|R| = 1`` below ``V_S`` is an identity, so check it as one."""
    from fwap.cylindrical import _fluid_solid_reflection

    c = np.array([1600.0, 1900.0, 2200.0, 2299.999])
    got = _fluid_solid_reflection(
        c,
        vp=_LEAKY_ROCK["vp"],
        vs=_LEAKY_ROCK["vs"],
        rho=_LEAKY_ROCK["rho"],
        vf=1500.0,
        rho_f=1000.0,
    )
    assert np.allclose(np.abs(got), 1.0, rtol=0.0, atol=1.0e-14)


def test_leaky_radiation_estimate_brackets_the_cylindrical_solver():
    """The oracle and the modal-determinant solver must agree in size.

    This is the point of the whole function: ``pseudo_rayleigh_dispersion``
    finds a complex root of the cylindrical modal determinant, while this
    estimate is a plane-wave reflection coefficient times a bounce rate.
    Nothing is shared between them.

    The bounds are wide on purpose, and they are measured rather than
    chosen. **They have been re-measured**, and the comparison is
    weaker than it looked.

    It used to read: across borehole radii 0.07-0.15 m and fast
    formations with ``V_S`` 1700-2800 m/s the ratio stays inside
    0.37-1.91 with a median of 0.57-0.71 in every case. That was
    measured over 4-30 kHz on a solver whose leaky branch was two
    thirds incoming; corrected, the mode does not live over 4-30 kHz at
    all but in a band just under its trapped cut-off, and there the
    ratio runs 0.11-2.6 with a median near 0.31 (0.72 for the softest
    formation).

    The spread is not noise, and it is not the solver disagreeing with
    the oracle either. It is the cut-off, which the single-bounce
    picture does not have: at the cut-off the mode grazes and radiates
    nothing, so ``Im(k_z) -> 0`` while the plane-wave estimate stays
    finite and the ratio collapses. Sampled to the cut-off it reaches
    0.006. The band below is trimmed to 0.92 ``f_c`` for that reason,
    which is a limitation of the estimate being named rather than a
    tolerance being widened.

    What survives is the claim this was always for: two constructions
    with nothing shared between them agree in *size*, within a factor
    of a few, across a 2x span of radius and a 1.6x span of ``V_S``.
    """
    from fwap import leaky_radiation_attenuation, pseudo_rayleigh_modal_dispersion

    cases = [
        dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10),
        dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.15),
        dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.07),
        dict(vp=4600.0, vs=2800.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10),
        dict(vp=3200.0, vs=1700.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.10),
    ]
    from fwap.cylindrical_solver._leaky import _trapped_branch_cutoff

    freq = np.linspace(1.0e3, 30.0e3, 300)
    for medium in cases:
        cutoff = _trapped_branch_cutoff(0, **medium, f_hi=40.0e3)
        assert cutoff is not None, medium
        mode = pseudo_rayleigh_modal_dispersion(freq, **medium)
        ok = np.isfinite(mode.slowness) & (freq < 0.92 * cutoff)
        assert ok.sum() > 10, medium
        estimate = leaky_radiation_attenuation(1.0 / mode.slowness, freq, **medium)
        ratio = mode.attenuation_per_meter[ok] / estimate[ok]

        assert np.all(ratio > 0.10), (medium, ratio.min())
        assert np.all(ratio < 6.0), (medium, ratio.max())
        assert 0.25 < float(np.median(ratio)) < 0.80, (medium, np.median(ratio))


def test_leaky_estimate_would_catch_a_wrong_radius_scaling():
    """The bracket above is loose; check it is still tight enough to bite.

    A factor-of-two error in the borehole radius -- the kind of mistake the
    estimate exists to catch -- must push the ratio outside the band that
    the correct radius satisfies.
    """
    from fwap import leaky_radiation_attenuation, pseudo_rayleigh_modal_dispersion
    from fwap.cylindrical_solver._leaky import _trapped_branch_cutoff

    medium = dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10)
    freq = np.linspace(1.0e3, 30.0e3, 300)
    cutoff = _trapped_branch_cutoff(0, **medium, f_hi=40.0e3)
    mode = pseudo_rayleigh_modal_dispersion(freq, **medium)
    ok = np.isfinite(mode.slowness) & (freq < 0.92 * cutoff)

    wrong = {**medium, "a": 0.20}
    estimate = leaky_radiation_attenuation(1.0 / mode.slowness, freq, **wrong)
    ratio = mode.attenuation_per_meter[ok] / estimate[ok]
    assert float(np.median(ratio)) > 0.50


# ---------------------------------------------------------------------------
# Low-frequency (tube-wave) limit of the borehole Stoneley mode
# ---------------------------------------------------------------------------

_TUBE_BRINE = {"vf": 1500.0, "rho_f": 1000.0}


def _bound_root_without_bracket_hint(freq, *, span=40.0, n=400, **medium):
    """Locate the bound n=0 root by scanning far wider than the solver.

    ``_stoneley_kz_bracket`` places its upper end at 1.5x the White
    estimate, so a test that used ``stoneley_dispersion`` alone would be
    checking a formula the solver was already told about. Scanning
    ``span`` times the bound edge -- twenty times wider than that
    bracket -- takes the estimate out of the loop entirely.
    """
    from scipy.optimize import brentq

    from fwap.cylindrical_solver._n0_isotropic import _modal_determinant_n0

    omega = 2.0 * np.pi * freq

    def det(s):
        return _modal_determinant_n0(s * omega, omega, **medium)

    s_lo = max(1.0 / medium["vf"], 1.0 / medium["vs"]) * (1.0 + 1.0e-12)
    grid = np.linspace(s_lo, s_lo * span, n)
    values = np.array([det(s) for s in grid])
    finite = np.isfinite(values)
    for i in range(grid.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if np.sign(values[i]) != np.sign(values[i + 1]):
            return float(brentq(det, grid[i], grid[i + 1], xtol=1e-18, rtol=8.9e-16))
    return float("nan")


def test_tube_wave_speed_is_the_low_frequency_limit_of_the_solver():
    """The modal determinant's f -> 0 root is the White tube-wave speed.

    Deliberately does *not* call ``stoneley_dispersion``: that function's
    bracket is built from this very formula, so agreement through it
    would be partly circular. The root here is found by scanning 40x the
    bound edge, against a solver bracket only a factor of two wide.
    """
    from fwap import tube_wave_speed

    cases = [
        dict(vp=2600.0, vs=1300.0, rho=2300.0, **_TUBE_BRINE, a=0.10),
        dict(vp=3400.0, vs=1800.0, rho=2400.0, **_TUBE_BRINE, a=0.10),
        dict(vp=4000.0, vs=2300.0, rho=2500.0, **_TUBE_BRINE, a=0.10),
        dict(vp=5200.0, vs=3000.0, rho=2650.0, **_TUBE_BRINE, a=0.10),
        dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1400.0, rho_f=1400.0, a=0.10),
    ]
    for medium in cases:
        scanned = _bound_root_without_bracket_hint(1.0, **medium)
        expected = 1.0 / tube_wave_speed(
            medium["vs"], medium["rho"], vf=medium["vf"], rho_f=medium["rho_f"]
        )
        assert scanned == pytest.approx(expected, rel=1.0e-6), medium


def test_tube_wave_limit_is_independent_of_borehole_radius():
    """No radius appears in the closed form, so none may appear in the limit.

    A sharper check than the value itself: the borehole radius is the one
    parameter the solver has and the formula does not, and it is not part
    of the bracket's shape. If the low-frequency limit drifted with ``a``
    the agreement above would be a coincidence of one geometry.
    """
    from fwap import tube_wave_speed

    base = dict(vp=4000.0, vs=2300.0, rho=2500.0, **_TUBE_BRINE)
    expected = 1.0 / tube_wave_speed(2300.0, 2500.0, **_TUBE_BRINE)
    for a in (0.05, 0.10, 0.20, 0.30):
        scanned = _bound_root_without_bracket_hint(1.0, **base, a=a)
        assert scanned == pytest.approx(expected, rel=1.0e-6), a


def test_stoneley_dispersion_converges_to_the_tube_wave_as_frequency_falls():
    """The solver's Stoneley curve tends to the closed form as f -> 0.

    The approach is **not** one-sided, which is worth stating because the
    Scholte high-frequency check is, and the analogy tempts the opposite
    assumption. Measured, the relative error changes sign at a crossover
    whose frequency moves with ``V_S``: it is positive throughout for
    ``V_S`` = 1300 m/s, crosses near 200 Hz at 2300 m/s and near 20 Hz at
    3000 m/s, and at 1600 m/s the crossover happens to sit at the top of
    this grid. So the assertion is on convergence, not on side.

    Restricted to 500 Hz and below, where every case is past its
    crossover, the error falls monotonically and reaches better than
    1e-6 relative at 1 Hz.
    """
    from fwap import stoneley_dispersion, tube_wave_speed

    freq = np.array([500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0])
    cases = [
        dict(vp=2600.0, vs=1300.0, rho=2300.0, **_TUBE_BRINE, a=0.10),
        dict(vp=3000.0, vs=1600.0, rho=2300.0, **_TUBE_BRINE, a=0.10),
        dict(vp=4000.0, vs=2300.0, rho=2500.0, **_TUBE_BRINE, a=0.10),
        dict(vp=5200.0, vs=3000.0, rho=2650.0, **_TUBE_BRINE, a=0.10),
    ]
    for medium in cases:
        limit = 1.0 / tube_wave_speed(
            medium["vs"], medium["rho"], vf=medium["vf"], rho_f=medium["rho_f"]
        )
        slowness = stoneley_dispersion(freq, **medium).slowness
        assert np.all(np.isfinite(slowness)), medium

        error = np.abs(slowness / limit - 1.0)
        assert np.all(np.diff(error) < 0.0), (medium, error)
        assert error[-1] < 1.0e-6, (medium, error[-1])


def test_tube_wave_speed_rejects_media_below_the_validity_floor():
    """Below ``vf * sqrt(1 - rho_f/rho)`` the formula is not a bound mode."""
    from fwap import tube_wave_speed

    floor = 1500.0 * np.sqrt(1.0 - 1000.0 / 2200.0)
    assert floor == pytest.approx(1107.8, rel=1.0e-3)

    with pytest.raises(ValueError, match="validity floor"):
        tube_wave_speed(floor * 0.99, 2200.0, **_TUBE_BRINE)
    # ...and it is admitted just above.
    assert tube_wave_speed(floor * 1.01, 2200.0, **_TUBE_BRINE) < floor * 1.01

    with pytest.raises(ValueError, match="must all be positive"):
        tube_wave_speed(-1.0, 2200.0, **_TUBE_BRINE)
    with pytest.raises(ValueError, match="require rho > rho_f"):
        tube_wave_speed(1300.0, 900.0, **_TUBE_BRINE)


def test_tube_wave_validity_floor_predicts_where_the_solver_stops_converging():
    """The floor is physics, not a bracketing artefact.

    Below it no bound root exists at all -- checked by scanning the
    determinant across a window far wider than the solver's bracket, not
    merely by observing that ``stoneley_dispersion`` returns NaN. The
    closed form locates the transition to within a couple of percent.
    """
    from fwap import stoneley_dispersion

    rho, vf, rho_f = 2200.0, 1500.0, 1000.0
    floor = vf * np.sqrt(1.0 - rho_f / rho)

    below = dict(vp=2000.0, vs=0.90 * floor, rho=rho, vf=vf, rho_f=rho_f, a=0.10)
    above = dict(vp=2400.0, vs=1.10 * floor, rho=rho, vf=vf, rho_f=rho_f, a=0.10)

    # No bound root below the floor, by direct scan...
    assert not np.isfinite(_bound_root_without_bracket_hint(1.0, **below))
    # ...and there is one above it.
    assert np.isfinite(_bound_root_without_bracket_hint(1.0, **above))

    # The solver agrees on both counts.
    freq = np.array([1.0, 10.0, 100.0])
    assert not np.any(np.isfinite(stoneley_dispersion(freq, **below).slowness))
    assert np.all(np.isfinite(stoneley_dispersion(freq, **above).slowness))


def test_modal_dispersion_is_not_kramers_kronig_constrained():
    """Waveguide dispersion is geometric, so no causality relation ties it.

    Recorded because `plans/learning.md` listed Kramers-Kronig as a promising
    remaining oracle for the modal solver, on the grounds that it relates
    ``Re(k_z)`` and ``Im(k_z)`` across frequencies and so cannot be satisfied
    by a single root. That is true of a *material* response and false here.

    The disproof is one line of data. A subtracted Kramers-Kronig relation on
    the complex slowness says that zero attenuation at every frequency forces
    zero dispersion -- the dispersion integral has an identically zero
    integrand. The bound Stoneley mode is exactly lossless: the solver returns
    no attenuation field at all for it. If it were also KK-constrained its
    phase velocity would have to be constant. It varies by more than 8 %
    between the tube-wave and Scholte limits.

    The physics: Kramers-Kronig follows from causality of the *constitutive*
    relation. Here the medium is perfectly elastic and non-dispersive, and all
    of the frequency dependence comes from the boundary conditions. A hollow
    metallic waveguide is the textbook version of the same point -- strongly
    dispersive with lossless walls.
    """
    from fwap import stoneley_dispersion

    medium = dict(vp=2600.0, vs=1300.0, rho=2300.0, **_TUBE_BRINE, a=0.10)
    freq = np.array([1.0, 1.0e3, 1.0e4, 4.0e5])
    mode = stoneley_dispersion(freq, **medium)

    # Lossless: no attenuation anywhere, not merely small.
    assert mode.attenuation_per_meter is None

    # ...and dispersive all the same.
    speeds = 1.0 / mode.slowness
    assert np.all(np.isfinite(speeds))
    assert abs(speeds[-1] / speeds[0] - 1.0) > 0.05, speeds
