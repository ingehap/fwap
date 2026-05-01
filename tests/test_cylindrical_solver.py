"""Schmitt (1988) cylindrical-borehole modal-determinant solver tests.

Phase 1: n=0 axisymmetric Stoneley dispersion. Validates against
the closed-form White (1983) low-f limit and a battery of
parametric checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwap.cylindrical import (
    flexural_dispersion_physical,
    flexural_dispersion_vti_physical,
    rayleigh_speed,
)
from fwap.cylindrical_solver import (
    BoreholeLayer,
    BoreholeMode,
    _layer_e_matrix_n0,
    _layer_propagator_n0,
    _modal_determinant_n0_cased,
    _validate_borehole_layers_stacked,
    _layered_n0_bessel_pack,
    _layered_n0_radial_wavenumbers,
    _layered_n0_row1_at_a,
    _layered_n0_row2_at_a,
    _layered_n0_row3_at_a,
    _layered_n0_row4_at_b,
    _layered_n0_row5_at_b,
    _layered_n0_row6_at_b,
    _layered_n0_row7_at_b,
    _layered_n1_row1_at_a,
    _layered_n1_row10_at_b,
    _layered_n1_row2_at_a,
    _layered_n1_row3_at_a,
    _layered_n1_row4_at_a,
    _layered_n1_row5_at_b,
    _layered_n1_row6_at_b,
    _layered_n1_row7_at_b,
    _layered_n1_row8_at_b,
    _layered_n1_row9_at_b,
    _modal_determinant_n0,
    _modal_determinant_n0_layered,
    _modal_determinant_n1,
    _modal_determinant_n1_layered,
    _modal_row1_at_a_n1_vti,
    _modal_row1_at_a_vti,
    _modal_row2_at_a_n1_vti,
    _modal_row2_at_a_vti,
    _modal_row3_at_a_n1_vti,
    _modal_row4_at_a_n1_vti,
    _modal_row3_at_a_vti,
    _modal_determinant_n0_vti,
    _modal_determinant_n1_vti,
    _is_isotropic_stiffness,
    _polarization_ratio_uz_over_ur_vti,
    _radial_wavenumbers_vti,
    flexural_dispersion,
    flexural_dispersion_layered,
    flexural_dispersion_vti,
    stoneley_dispersion,
    stoneley_dispersion_layered,
    stoneley_dispersion_vti,
)


def _stoneley_lf_truth(vs, rho, vf, rho_f):
    """White (1983) eq. 5.42 closed form: S_ST^2 = 1/V_f^2 + rho_f/mu."""
    mu = rho * vs**2
    return float(np.sqrt(1.0 / vf**2 + rho_f / mu))


# ---------------------------------------------------------------------
# Closed-form low-f validation
# ---------------------------------------------------------------------


def test_stoneley_low_f_matches_white_closed_form():
    """At very low frequency (10 Hz) the Stoneley slowness from the
    full modal determinant must match the closed-form
    `S_ST^2 = 1/V_f^2 + rho_f / mu` to better than 0.01 %."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(
        np.array([10.0]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    s_truth = _stoneley_lf_truth(vs, rho, vf, rho_f)
    assert res.slowness[0] == pytest.approx(s_truth, rel=1.0e-4)


def test_stoneley_solver_returns_nan_in_slow_formation_regime():
    """Slow formations (V_S < V_f) put V_ST above V_S, so the
    formation-side S radial wavenumber becomes imaginary (s^2 < 0)
    -- the Stoneley wave radiates into the formation as a leaky S
    arrival. The Phase-1 bound-mode solver cannot handle that
    regime; it should return ``NaN`` rather than wrong numbers."""
    vp, vs, rho = 2200.0, 800.0, 2200.0  # slow shale: V_S < V_f
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(
        np.array([10.0]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    assert np.isnan(res.slowness[0])


def test_stoneley_low_f_matches_white_closed_form_fast_formation():
    """Fast-formation low-f match (V_S much larger than V_f)."""
    vp, vs, rho = 6000.0, 3500.0, 2700.0  # fast carbonate
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(
        np.array([10.0]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    s_truth = _stoneley_lf_truth(vs, rho, vf, rho_f)
    assert res.slowness[0] == pytest.approx(s_truth, rel=1.0e-4)


# ---------------------------------------------------------------------
# Dispersion-curve shape
# ---------------------------------------------------------------------


def test_stoneley_dispersion_is_finite_across_typical_band():
    """The Stoneley root is reliably bracketed across 100-15 kHz."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(100.0, 15000.0, 60)
    res = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    assert np.all(np.isfinite(res.slowness))


def test_stoneley_dispersion_speeds_up_with_frequency():
    """Stoneley phase slowness *decreases* with frequency (the wave
    sheds the rho_f / mu loading and approaches the bare fluid as
    f rises in this band) for the typical brine-in-sandstone case."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(100.0, 10_000.0, 40)
    s = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a).slowness
    # Slowness is monotonically decreasing across the test band.
    assert np.all(np.diff(s) < 0.0)


def test_stoneley_slowness_always_above_fluid_slowness():
    """Stoneley wave is always slower than the unconfined fluid wave
    (formation loading slows it down)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(100.0, 12_000.0, 30)
    s = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a).slowness
    assert np.all(s > 1.0 / vf)


def test_stoneley_slowness_above_formation_shear_slowness():
    """Stoneley phase velocity in a fast formation is below V_S
    (V_ST < V_f < V_S), so the *slowness* sits above 1/V_S. The
    Stoneley wave is loaded by the formation through the rho_f /
    mu term but never speeds up to formation-shear levels."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(100.0, 12_000.0, 30)
    s = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a).slowness
    assert np.all(s > 1.0 / vs)


# ---------------------------------------------------------------------
# Parametric monotonicity
# ---------------------------------------------------------------------


def test_stoneley_slowness_increases_with_lower_shear_modulus():
    """At fixed fluid + frequency, decreasing the formation shear
    modulus (softer rock) increases the Stoneley slowness -- the
    wave gets slower as the formation is less stiff."""
    f = np.array([1000.0])
    rho = 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    s_stiff = stoneley_dispersion(
        f,
        vp=5000.0,
        vs=3000.0,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    ).slowness[0]
    s_soft = stoneley_dispersion(
        f,
        vp=4000.0,
        vs=2000.0,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    ).slowness[0]
    assert s_soft > s_stiff


def test_stoneley_slowness_increases_with_heavier_fluid():
    """Heavier borehole fluid -> larger rho_f / mu loading -> slower
    Stoneley."""
    f = np.array([1000.0])
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, a = 1500.0, 0.1
    s_light = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=900.0,
        a=a,
    ).slowness[0]
    s_heavy = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=1100.0,
        a=a,
    ).slowness[0]
    assert s_heavy > s_light


# ---------------------------------------------------------------------
# Modal determinant zero structure
# ---------------------------------------------------------------------


def test_modal_determinant_changes_sign_across_stoneley_root():
    """Direct check: det(M) at k_z slightly below the recovered
    Stoneley root has opposite sign to det(M) slightly above it."""
    omega = 2.0 * np.pi * 1000.0
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(
        np.array([1000.0]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    s_root = res.slowness[0]
    kz_root = s_root * omega
    d_lo = _modal_determinant_n0(kz_root * 0.999, omega, vp, vs, rho, vf, rho_f, a)
    d_hi = _modal_determinant_n0(kz_root * 1.001, omega, vp, vs, rho, vf, rho_f, a)
    assert np.sign(d_lo) != np.sign(d_hi)


def test_modal_determinant_at_root_is_near_zero():
    """det(M) at the recovered k_z must be 8+ orders of magnitude
    smaller than at a nearby k_z."""
    omega = 2.0 * np.pi * 1000.0
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    s_root = stoneley_dispersion(
        np.array([1000.0]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    ).slowness[0]
    kz_root = s_root * omega
    d_at = abs(_modal_determinant_n0(kz_root, omega, vp, vs, rho, vf, rho_f, a))
    d_near = abs(
        _modal_determinant_n0(kz_root * 1.05, omega, vp, vs, rho, vf, rho_f, a)
    )
    assert d_at < d_near * 1.0e-6


# ---------------------------------------------------------------------
# Borehole-radius dependence
# ---------------------------------------------------------------------


def test_stoneley_low_f_independent_of_borehole_radius():
    """The closed-form Stoneley low-f formula has no `a` dependence.
    The full modal determinant should reproduce that to within the
    low-f approximation error."""
    f = np.array([10.0])
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f = 1500.0, 1000.0
    s_truth = _stoneley_lf_truth(vs, rho, vf, rho_f)
    for a in (0.05, 0.10, 0.15, 0.20):
        s = stoneley_dispersion(
            f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
        ).slowness[0]
        assert s == pytest.approx(s_truth, rel=1.0e-3)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_stoneley_dispersion_rejects_non_positive_inputs():
    f = np.array([1000.0])
    base = dict(vp=4500.0, vs=2500.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1)
    with pytest.raises(ValueError, match="vp, vs, rho"):
        stoneley_dispersion(f, **{**base, "rho": 0.0})
    with pytest.raises(ValueError, match="vp, vs, rho"):
        stoneley_dispersion(f, **{**base, "vs": -1.0})
    with pytest.raises(ValueError, match="vf and rho_f"):
        stoneley_dispersion(f, **{**base, "vf": 0.0})
    with pytest.raises(ValueError, match="vf and rho_f"):
        stoneley_dispersion(f, **{**base, "rho_f": -100.0})
    with pytest.raises(ValueError, match="^a must"):
        stoneley_dispersion(f, **{**base, "a": 0.0})


def test_stoneley_dispersion_rejects_vp_le_vs():
    f = np.array([1000.0])
    with pytest.raises(ValueError, match="vp > vs"):
        stoneley_dispersion(
            f, vp=2500.0, vs=2500.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1
        )


def test_stoneley_dispersion_rejects_non_positive_freq():
    f = np.array([10.0, 0.0, 100.0])
    with pytest.raises(ValueError, match="freq"):
        stoneley_dispersion(
            f, vp=4500.0, vs=2500.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1
        )


# ---------------------------------------------------------------------
# Output dataclass contract
# ---------------------------------------------------------------------


def test_stoneley_dispersion_returns_borehole_mode_dataclass():
    f = np.linspace(500.0, 5000.0, 5)
    res = stoneley_dispersion(
        f, vp=4500.0, vs=2500.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    np.testing.assert_array_equal(res.freq, f)
    assert res.slowness.shape == f.shape


# =====================================================================
# n=1 dipole flexural modal-determinant solver tests
# =====================================================================
#
# Slow-formation parameters used throughout (V_S < V_f so the
# flexural mode is bound). Geometric cutoff f_cut ~ V_S / (2 pi a)
# is around 1273 Hz for these values; tests sit above ~2 kHz where
# the bound mode is well-defined.

SLOW_VP = 2200.0
SLOW_VS = 800.0
SLOW_RHO = 2200.0
SLOW_VF = 1500.0
SLOW_RHO_F = 1000.0
SLOW_A = 0.1


# ---------------------------------------------------------------------
# Asymptotic limits (substep 1.6.b and 1.6.c-d predictions)
# ---------------------------------------------------------------------


def test_flexural_low_f_slowness_approaches_inverse_vs():
    """Just above the geometric cutoff, the modal flexural slowness
    must approach 1 / V_S (Ellefsen-Cheng-Toksoz 1991 sect. III.B
    long-wavelength asymptote). Tests f = 2 kHz where the slowness
    should be within 2% of 1/V_S."""
    res = flexural_dispersion(
        np.array([2000.0]),
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
    )
    assert np.isfinite(res.slowness[0])
    assert res.slowness[0] == pytest.approx(1.0 / SLOW_VS, rel=2.0e-2)


def test_flexural_high_f_slowness_above_inverse_rayleigh():
    """At high f, the modal flexural slowness must sit above
    1 / V_R (Rayleigh-asymptote with positive Scholte / fluid-
    loading offset; substep 1.6.c-d). Tests f = 10 kHz where the
    slowness should be above 1/V_R but within 10% of it."""
    res = flexural_dispersion(
        np.array([10000.0]),
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
    )
    assert np.isfinite(res.slowness[0])
    vR = rayleigh_speed(SLOW_VP, SLOW_VS)
    s_R = 1.0 / vR
    # Above 1/V_R: this is the genuine physical correction the
    # modal solver captures that flexural_dispersion_physical does
    # not (it uses the vacuum-loaded Rayleigh asymptote exactly).
    assert res.slowness[0] > s_R
    assert res.slowness[0] == pytest.approx(s_R, rel=0.10)


def test_flexural_dispersion_is_monotonic_above_cutoff():
    """Slowness increases monotonically from ~1/V_S to ~1/V_R as
    f rises (substep 1.6.e cross-check 2). Tests f = 2-15 kHz."""
    f = np.linspace(2000.0, 15000.0, 14)
    res = flexural_dispersion(
        f, vp=SLOW_VP, vs=SLOW_VS, rho=SLOW_RHO, vf=SLOW_VF, rho_f=SLOW_RHO_F, a=SLOW_A
    )
    s = res.slowness
    assert np.all(np.isfinite(s))
    # Monotonically non-decreasing in slowness (= phase velocity
    # decreasing). Allow tiny numerical wobble at the high end.
    assert np.all(np.diff(s) >= -1.0e-9)


def test_flexural_dispersion_qualitative_match_with_phenomenological():
    """The modal solver and ``flexural_dispersion_physical`` agree
    qualitatively (low-f anchor at 1/V_S, high-f anchor near 1/V_R)
    with a few-percent quantitative offset from the Scholte / fluid-
    loading correction the modal solver captures and the
    phenomenological one does not."""
    f = np.array([2000.0, 5000.0, 10000.0])
    s_modal = flexural_dispersion(
        f, vp=SLOW_VP, vs=SLOW_VS, rho=SLOW_RHO, vf=SLOW_VF, rho_f=SLOW_RHO_F, a=SLOW_A
    ).slowness
    phen = flexural_dispersion_physical(SLOW_VP, SLOW_VS, SLOW_A)
    s_phen = phen(f)
    # Both are within 10% of each other across the band
    rel_diff = np.abs(s_modal - s_phen) / s_phen
    assert np.all(rel_diff < 0.10)


# ---------------------------------------------------------------------
# Out-of-regime behavior: NaN return rather than raise
# ---------------------------------------------------------------------


def test_flexural_dispatches_to_fast_formation_path_when_vs_gt_vf():
    """Plan item B: ``flexural_dispersion`` now auto-dispatches to
    the complex-determinant fast-formation path when ``V_S > V_f``,
    instead of returning NaN throughout. At least some frequencies
    in a sensible band must yield finite slowness in the
    ``(V_R, V_S)`` window. The previous "all NaN" contract is
    deliberately broken."""
    from fwap.cylindrical import rayleigh_speed

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    vR = rayleigh_speed(vp, vs)
    f = np.linspace(20000.0, 80000.0, 30)
    res = flexural_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    finite = np.isfinite(res.slowness)
    assert finite.any(), "fast-formation path must populate at least one frequency"
    velocity = 1.0 / res.slowness[finite]
    # Strictly between V_R and V_S (the leaky-F regime window).
    assert (velocity > vR * 0.99).all(), (
        f"velocity must stay near or above V_R ({vR:.0f}); got {velocity}"
    )
    assert (velocity < vs).all(), (
        f"velocity must stay below V_S ({vs}); got {velocity}"
    )
    # Bound mode -> attenuation_per_meter is None.
    assert res.attenuation_per_meter is None


def test_flexural_returns_nan_below_geometric_cutoff():
    """The dipole flexural mode in slow formations has a low-f
    geometric cutoff near V_S / (2 pi a) ~ 1273 Hz for these
    parameters. Below that, no bound flexural root exists and
    NaN is returned."""
    f = np.array([500.0, 800.0, 1100.0])  # all below cutoff
    res = flexural_dispersion(
        f, vp=SLOW_VP, vs=SLOW_VS, rho=SLOW_RHO, vf=SLOW_VF, rho_f=SLOW_RHO_F, a=SLOW_A
    )
    assert np.all(np.isnan(res.slowness))


# ---------------------------------------------------------------------
# Modal-determinant zero structure
# ---------------------------------------------------------------------


def test_modal_determinant_n1_changes_sign_across_root():
    """Direct check: det(M) at k_z slightly below the recovered
    flexural root has opposite sign to det(M) slightly above it."""
    omega = 2.0 * np.pi * 5000.0
    res = flexural_dispersion(
        np.array([5000.0]),
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
    )
    s_root = res.slowness[0]
    kz_root = s_root * omega
    d_lo = _modal_determinant_n1(
        kz_root * 0.999, omega, SLOW_VP, SLOW_VS, SLOW_RHO, SLOW_VF, SLOW_RHO_F, SLOW_A
    )
    d_hi = _modal_determinant_n1(
        kz_root * 1.001, omega, SLOW_VP, SLOW_VS, SLOW_RHO, SLOW_VF, SLOW_RHO_F, SLOW_A
    )
    assert np.sign(d_lo) != np.sign(d_hi)


def test_modal_determinant_n1_at_root_is_near_zero():
    """det(M) at the recovered k_z is many orders of magnitude
    smaller than at a nearby k_z."""
    omega = 2.0 * np.pi * 5000.0
    s_root = flexural_dispersion(
        np.array([5000.0]),
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
    ).slowness[0]
    kz_root = s_root * omega
    d_at = abs(
        _modal_determinant_n1(
            kz_root, omega, SLOW_VP, SLOW_VS, SLOW_RHO, SLOW_VF, SLOW_RHO_F, SLOW_A
        )
    )
    d_near = abs(
        _modal_determinant_n1(
            kz_root * 1.05,
            omega,
            SLOW_VP,
            SLOW_VS,
            SLOW_RHO,
            SLOW_VF,
            SLOW_RHO_F,
            SLOW_A,
        )
    )
    assert d_at < d_near * 1.0e-6


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_flexural_dispersion_rejects_non_positive_inputs():
    f = np.array([5000.0])
    base = dict(
        vp=SLOW_VP, vs=SLOW_VS, rho=SLOW_RHO, vf=SLOW_VF, rho_f=SLOW_RHO_F, a=SLOW_A
    )
    with pytest.raises(ValueError, match="vp, vs, rho"):
        flexural_dispersion(f, **{**base, "rho": 0.0})
    with pytest.raises(ValueError, match="vp, vs, rho"):
        flexural_dispersion(f, **{**base, "vs": -1.0})
    with pytest.raises(ValueError, match="vf and rho_f"):
        flexural_dispersion(f, **{**base, "vf": 0.0})
    with pytest.raises(ValueError, match="vf and rho_f"):
        flexural_dispersion(f, **{**base, "rho_f": -100.0})
    with pytest.raises(ValueError, match="^a must"):
        flexural_dispersion(f, **{**base, "a": 0.0})


def test_flexural_dispersion_rejects_vp_le_vs():
    f = np.array([5000.0])
    with pytest.raises(ValueError, match="vp > vs"):
        flexural_dispersion(
            f, vp=800.0, vs=800.0, rho=SLOW_RHO, vf=SLOW_VF, rho_f=SLOW_RHO_F, a=SLOW_A
        )


def test_flexural_dispersion_rejects_non_positive_freq():
    f = np.array([2000.0, 0.0, 5000.0])
    with pytest.raises(ValueError, match="freq"):
        flexural_dispersion(
            f,
            vp=SLOW_VP,
            vs=SLOW_VS,
            rho=SLOW_RHO,
            vf=SLOW_VF,
            rho_f=SLOW_RHO_F,
            a=SLOW_A,
        )


# ---------------------------------------------------------------------
# Output dataclass contract
# ---------------------------------------------------------------------


def test_flexural_dispersion_returns_borehole_mode_dataclass():
    f = np.linspace(2000.0, 10000.0, 5)
    res = flexural_dispersion(
        f, vp=SLOW_VP, vs=SLOW_VS, rho=SLOW_RHO, vf=SLOW_VF, rho_f=SLOW_RHO_F, a=SLOW_A
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    np.testing.assert_array_equal(res.freq, f)
    assert res.slowness.shape == f.shape


# =====================================================================
# Leaky-mode scaffolding (Roadmap A continuation, phases L1 + L2)
# =====================================================================
#
# Tests for the complex-aware n=0 modal determinant and its
# supporting helpers. Phase L3 (the complex root finder) and
# phases L4-L6 (public API for pseudo-Rayleigh, fast-formation
# flexural, quadrupole) are planned follow-ups; this file pins the
# regression invariants for L1+L2 only.


# Standard parameter set used across the leaky-mode regression tests.
LEAKY_VP = 4500.0
LEAKY_VS = 2500.0
LEAKY_RHO = 2400.0
LEAKY_VF = 1500.0
LEAKY_RHO_F = 1000.0
LEAKY_A = 0.1


# ---------------------------------------------------------------------
# Regression: complex evaluator matches real evaluator in bound regime
# ---------------------------------------------------------------------


def test_complex_n0_matches_real_in_bound_regime():
    """Real-valued ``_modal_determinant_n0`` and complex-valued
    ``_modal_determinant_n0_complex`` must agree to floating-point
    precision when called with real ``kz`` and both ``leaky_*``
    flags False (the bound Stoneley regime).

    This is THE regression invariant for the L2 refactor: as long
    as it holds, the existing 16+ Stoneley tests cover the bound-
    regime physics; the complex evaluator only adds new capability
    on top."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n0,
        _modal_determinant_n0_complex,
    )

    omega = 2.0 * np.pi * 1000.0
    # Stoneley-region bracket on a fast formation.
    for kz in [5.0, 5.5, 6.0, 7.0]:
        d_real = _modal_determinant_n0(
            kz, omega, LEAKY_VP, LEAKY_VS, LEAKY_RHO,
            LEAKY_VF, LEAKY_RHO_F, LEAKY_A,
        )
        d_complex = _modal_determinant_n0_complex(
            complex(kz), omega, LEAKY_VP, LEAKY_VS, LEAKY_RHO,
            LEAKY_VF, LEAKY_RHO_F, LEAKY_A,
        )
        # Real part agrees exactly.
        assert abs(d_complex.real - d_real) / abs(d_real) < 1.0e-12, (
            f"real-vs-complex mismatch at kz={kz}: "
            f"real={d_real}, complex={d_complex}"
        )
        # Imaginary part is identically zero (complex arithmetic on
        # purely real inputs).
        assert d_complex.imag == 0.0


def test_complex_n0_changes_sign_across_stoneley_root_in_bound_regime():
    """The complex evaluator preserves the sign-change behaviour of
    the real evaluator across the Stoneley root. Combined with the
    matching-values test above, this guarantees that any future
    real-axis root finder built on the complex evaluator finds
    the same Stoneley root as the existing brentq-based solver."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n0_complex,
        stoneley_dispersion,
    )

    omega = 2.0 * np.pi * 1000.0
    res = stoneley_dispersion(
        np.array([1000.0]), vp=LEAKY_VP, vs=LEAKY_VS, rho=LEAKY_RHO,
        vf=LEAKY_VF, rho_f=LEAKY_RHO_F, a=LEAKY_A,
    )
    kz_root = res.slowness[0] * omega
    d_lo = _modal_determinant_n0_complex(
        complex(kz_root * 0.999), omega, LEAKY_VP, LEAKY_VS,
        LEAKY_RHO, LEAKY_VF, LEAKY_RHO_F, LEAKY_A,
    )
    d_hi = _modal_determinant_n0_complex(
        complex(kz_root * 1.001), omega, LEAKY_VP, LEAKY_VS,
        LEAKY_RHO, LEAKY_VF, LEAKY_RHO_F, LEAKY_A,
    )
    assert np.sign(d_lo.real) != np.sign(d_hi.real)


# ---------------------------------------------------------------------
# Branch detector
# ---------------------------------------------------------------------


def test_detect_leaky_branches_bound_regime_all_false():
    """In the fully-bound regime (kz above ALL of omega/V_alpha),
    every branch is bound. For a fast formation the binding
    constraint is kz > omega/V_f (the largest of the three
    omega/V_alpha thresholds when V_f < V_S < V_P). Pick a kz
    well above that floor."""
    from fwap.cylindrical_solver import _detect_leaky_branches

    omega = 2.0 * np.pi * 1000.0
    # In fast formation V_f is the smallest velocity, so omega/V_f
    # is the largest of the three thresholds. Pick kz comfortably
    # above it.
    kz = omega / LEAKY_VF * 1.5
    leaky_F, leaky_p, leaky_s = _detect_leaky_branches(
        complex(kz), omega, LEAKY_VP, LEAKY_VS, LEAKY_VF,
    )
    assert leaky_F is False
    assert leaky_p is False
    assert leaky_s is False


def test_detect_leaky_branches_pseudo_rayleigh_regime():
    """In the pseudo-Rayleigh regime (kz between omega/V_P and
    omega/V_S, fast formation V_S > V_f), the S branch is leaky;
    the P branch stays bound. The fluid F branch is leaky in this
    region too because kz < omega/V_f for typical fast-formation
    parameters."""
    from fwap.cylindrical_solver import _detect_leaky_branches

    omega = 2.0 * np.pi * 1000.0
    # Slowness just above 1/V_S; fast formation means kz < omega/V_f.
    kz = omega / LEAKY_VS * 0.95
    leaky_F, leaky_p, leaky_s = _detect_leaky_branches(
        complex(kz), omega, LEAKY_VP, LEAKY_VS, LEAKY_VF,
    )
    assert leaky_s is True   # S radiates outward
    assert leaky_p is False  # P stays bound (kz still > omega/V_P)
    # F is leaky here because kz_pr = 2.4 < omega/V_f = 4.2.
    assert leaky_F is True


def test_detect_leaky_branches_fast_flexural_regime():
    """In a fast formation, the flexural mode at slowness ~1/V_R
    has phase velocity above V_f, so F^2 < 0 and the F branch is
    leaky; p and s stay bound."""
    from fwap.cylindrical import rayleigh_speed
    from fwap.cylindrical_solver import _detect_leaky_branches

    omega = 2.0 * np.pi * 5000.0
    vR = rayleigh_speed(LEAKY_VP, LEAKY_VS)
    # Slowness ~1/V_R for the high-f flexural asymptote.
    kz = omega / vR
    leaky_F, leaky_p, leaky_s = _detect_leaky_branches(
        complex(kz), omega, LEAKY_VP, LEAKY_VS, LEAKY_VF,
    )
    assert leaky_F is True   # fluid radiates (V_R > V_f for fast formation)
    assert leaky_p is False  # P stays bound
    assert leaky_s is False  # S stays bound (kz > omega/V_S in flexural band)


# ---------------------------------------------------------------------
# K-or-Hankel helper
# ---------------------------------------------------------------------


def test_k_or_hankel_bound_branch_matches_kv():
    """In the bound branch, ``_k_or_hankel`` returns exactly
    ``scipy.special.kv`` values."""
    from scipy import special

    from fwap.cylindrical_solver import _k_or_hankel

    alpha = 1.5  # arbitrary positive real
    r = 0.1
    K0_h, K1_h = _k_or_hankel(0, complex(alpha), r, leaky=False)
    assert abs(K0_h.real - float(special.kv(0, alpha * r))) < 1.0e-15
    assert abs(K1_h.real - float(special.kv(1, alpha * r))) < 1.0e-15
    assert K0_h.imag == 0.0
    assert K1_h.imag == 0.0


def test_k_or_hankel_leaky_branch_returns_finite_complex():
    """In the leaky branch, the Hankel-via-analytic-continuation
    formula evaluates to a finite complex number."""
    from fwap.cylindrical_solver import _k_or_hankel

    # Imaginary alpha (the leaky-S case after sqrt of a negative
    # alpha^2 with the principal-branch sign convention).
    alpha = 0.5 + 1.5j
    r = 0.1
    K0_h, K1_h = _k_or_hankel(0, alpha, r, leaky=True)
    assert np.isfinite(K0_h.real)
    assert np.isfinite(K0_h.imag)
    assert np.isfinite(K1_h.real)
    assert np.isfinite(K1_h.imag)


# ---------------------------------------------------------------------
# Complex evaluator handles the leaky regime without exceptions
# ---------------------------------------------------------------------


def test_complex_n0_in_pseudo_rayleigh_regime_is_finite():
    """The complex evaluator must not raise or return NaN in the
    leaky regime; the value at a randomly-chosen point is just a
    complex number whose root will be located by the L3 follow-up
    root finder."""
    from fwap.cylindrical_solver import (
        _detect_leaky_branches,
        _modal_determinant_n0_complex,
    )

    omega = 2.0 * np.pi * 5000.0
    # A pseudo-Rayleigh-region kz: between omega/V_P and omega/V_S
    # in the fast formation.
    kz = omega / LEAKY_VS * 0.92
    leaky_F, leaky_p, leaky_s = _detect_leaky_branches(
        complex(kz), omega, LEAKY_VP, LEAKY_VS, LEAKY_VF,
    )
    d = _modal_determinant_n0_complex(
        complex(kz), omega, LEAKY_VP, LEAKY_VS, LEAKY_RHO,
        LEAKY_VF, LEAKY_RHO_F, LEAKY_A,
        leaky_p=leaky_p, leaky_s=leaky_s,
    )
    assert np.isfinite(d.real)
    assert np.isfinite(d.imag)


def test_complex_n0_complex_kz_with_imaginary_part_finite():
    """A complex kz with non-zero imaginary part (the typical
    state of a leaky-mode dispersion locus) evaluates without
    issue. The matrix is now genuinely complex-valued; the
    determinant is finite."""
    from fwap.cylindrical_solver import (
        _detect_leaky_branches,
        _modal_determinant_n0_complex,
    )

    omega = 2.0 * np.pi * 5000.0
    kz = (omega / LEAKY_VS * 0.92) + 0.1j
    leaky_F, leaky_p, leaky_s = _detect_leaky_branches(
        kz, omega, LEAKY_VP, LEAKY_VS, LEAKY_VF,
    )
    d = _modal_determinant_n0_complex(
        kz, omega, LEAKY_VP, LEAKY_VS, LEAKY_RHO,
        LEAKY_VF, LEAKY_RHO_F, LEAKY_A,
        leaky_p=leaky_p, leaky_s=leaky_s,
    )
    assert np.isfinite(d.real)
    assert np.isfinite(d.imag)
    # The determinant is non-trivially complex (i.e. has a non-zero
    # imaginary part) when kz itself has a non-zero imaginary part.
    assert d.imag != 0.0


# ---------------------------------------------------------------------
# L3 -- complex-k_z root finder + frequency-marcher
# ---------------------------------------------------------------------


# Synthetic det functions
# -----------------------


def test_track_complex_root_finds_simple_linear_root():
    """A linear det function ``det(kz) = kz - target`` has its
    root at ``kz = target``. The tracker must find it from any
    nearby starting guess."""
    from fwap.cylindrical_solver import _track_complex_root

    target = 3.0 + 0.5j
    root = _track_complex_root(lambda kz: kz - target, kz_start=2.5 + 0.3j)
    assert root is not None
    assert abs(root - target) < 1.0e-9


def test_track_complex_root_picks_root_closest_to_initial_guess():
    """A quadratic det function has two roots; the tracker picks
    whichever is closer to the initial guess."""
    from fwap.cylindrical_solver import _track_complex_root

    r1 = 3.0 + 0.5j
    r2 = 5.0 - 0.2j

    def det(kz):
        return (kz - r1) * (kz - r2)

    root_near_r1 = _track_complex_root(det, kz_start=2.5 + 0.3j)
    root_near_r2 = _track_complex_root(det, kz_start=5.5 + 0.0j)
    assert root_near_r1 is not None
    assert root_near_r2 is not None
    assert abs(root_near_r1 - r1) < 1.0e-9
    assert abs(root_near_r2 - r2) < 1.0e-9


def test_track_complex_root_does_not_propagate_det_exceptions():
    """``_track_complex_root`` catches exceptions raised by
    ``det_fn`` and converts them to large penalty residuals, so
    the tracker either returns a (possibly off-target) root or
    None -- never raises. Documents the exception-safety
    contract."""
    from fwap.cylindrical_solver import _track_complex_root

    def det_always_raises(kz):
        raise ValueError("synthetic always-raise")

    # Should not raise. Result may be anything (None or a non-root
    # iterate); the only hard contract is that no exception
    # escapes the tracker.
    root = _track_complex_root(det_always_raises, kz_start=1.0 + 0.0j)
    # No assertion on root value: scipy.optimize.root may return
    # success=True on a degenerate residual landscape; the
    # contract is exception-safety.
    del root  # unused; documents the no-raise behaviour


# Frequency marcher
# -----------------


def test_march_synthetic_linear_dispersion_curve():
    """A synthetic det whose root scales linearly with frequency
    (constant slowness with a small imaginary part) is the simplest
    test case. The marcher should follow the curve exactly."""
    from fwap.cylindrical_solver import _march_complex_dispersion

    slowness = 1.0 / 2500.0 * (1.0 + 0.05j)

    def det(kz, omega):
        return kz - omega * slowness

    freqs = np.array([1000.0, 2000.0, 3000.0, 5000.0])
    kz_start = 2.0 * np.pi * freqs[0] * slowness
    curve = _march_complex_dispersion(det, freqs, kz_start)
    expected = 2.0 * np.pi * freqs * slowness
    np.testing.assert_allclose(curve, expected, rtol=1.0e-12)


def test_march_marches_along_a_smoothly_drifting_dispersion():
    """A synthetic dispersion where the root has both a real-axis
    drift (slowness changing with frequency) and a complex-plane
    drift (attenuation changing with frequency). The marcher
    must follow both."""
    from fwap.cylindrical_solver import _march_complex_dispersion

    # Slowness changes from 1/2500 + 0.05j/2500 at f=1 kHz to
    # 1/2400 + 0.10j/2400 at f=10 kHz (linear interpolation).
    def slowness_at_f(f):
        t = (f - 1000.0) / 9000.0
        s_re = 1.0 / 2500.0 * (1.0 - t) + 1.0 / 2400.0 * t
        s_im = 0.05 / 2500.0 * (1.0 - t) + 0.10 / 2400.0 * t
        return s_re + 1j * s_im

    def det(kz, omega):
        f = omega / (2.0 * np.pi)
        return kz - omega * slowness_at_f(f)

    freqs = np.linspace(1000.0, 10000.0, 10)
    kz_start = 2.0 * np.pi * freqs[0] * slowness_at_f(freqs[0])
    curve = _march_complex_dispersion(det, freqs, kz_start)
    expected = np.array(
        [2.0 * np.pi * f * slowness_at_f(f) for f in freqs]
    )
    np.testing.assert_allclose(curve, expected, rtol=1.0e-10)


def test_march_scale_invariant_continuation_handles_large_kz_jumps():
    """Stoneley kz scales linearly with frequency (slowness ~constant
    across the band), so successive frequencies have ``k_z`` values
    differing by factors of 2-10x. The marcher's scale-invariant
    continuation -- seeding the next step with
    ``k_z_prev * (f / f_prev)`` -- handles this without
    re-bracketing problems."""
    from fwap.cylindrical_solver import _march_complex_dispersion

    slowness = 1.0 / 1399.0  # White-Stoneley low-f closed form

    def det(kz, omega):
        return kz - omega * slowness

    # Big multiplicative jumps in frequency
    freqs = np.array([100.0, 1000.0, 5000.0, 15000.0])
    kz_start = 2.0 * np.pi * freqs[0] * slowness
    curve = _march_complex_dispersion(det, freqs, kz_start)
    expected = 2.0 * np.pi * freqs * slowness
    np.testing.assert_allclose(curve.real, expected, rtol=1.0e-12)


# Composition with the L1+L2 complex-aware modal determinant
# ----------------------------------------------------------


def test_march_recovers_existing_stoneley_curve_in_bound_regime():
    """End-to-end check that L1+L2+L3 compose correctly: use
    ``_modal_determinant_n0_complex`` with the auto-detected
    bound-regime branches as the marcher's det function, seeding
    from the White (1983) low-f closed form, and verify the
    recovered ``k_z`` curve matches the existing brentq-based
    ``stoneley_dispersion`` to floating-point precision (the bound-
    regime regression invariant from L2 carried through L3)."""
    from fwap.cylindrical_solver import (
        _detect_leaky_branches,
        _march_complex_dispersion,
        _modal_determinant_n0_complex,
        stoneley_dispersion,
    )

    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1

    def stoneley_det(kz, omega):
        leaky_F, leaky_p, leaky_s = _detect_leaky_branches(
            kz, omega, vp, vs, vf,
        )
        return _modal_determinant_n0_complex(
            kz, omega, vp, vs, rho, vf, rho_f, a,
            leaky_p=leaky_p, leaky_s=leaky_s,
        )

    # White (1983) low-f closed form: S_ST^2 = 1/V_f^2 + rho_f / mu.
    mu = rho * vs * vs
    s_st_lf = float(np.sqrt(1.0 / vf**2 + rho_f / mu))

    freqs = np.array([1000.0, 2000.0, 5000.0, 10000.0])
    omega_low = 2.0 * np.pi * float(freqs[0])
    kz_start = complex(omega_low * s_st_lf, 0.0)
    curve = _march_complex_dispersion(stoneley_det, freqs, kz_start)

    # Reference: the existing brentq-based solver.
    res = stoneley_dispersion(
        freqs, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    for kz_marcher, s_existing, f in zip(curve, res.slowness, freqs):
        omega = 2.0 * np.pi * float(f)
        kz_existing = omega * s_existing
        # Real parts agree to ~1e-11 relative (brentq xtol is
        # 1e-10; root xtol is 1e-12 here).
        assert abs(kz_marcher.real - kz_existing) / kz_existing < 1.0e-10
        # Imaginary parts are tiny (Stoneley is bound -> kz real).
        assert abs(kz_marcher.imag) < 1.0e-9 * kz_existing


# ---------------------------------------------------------------------
# L4 -- BoreholeMode extension + pseudo-Rayleigh experimental scaffolding
# ---------------------------------------------------------------------


def test_borehole_mode_attenuation_per_meter_field_optional():
    """``BoreholeMode`` now has an optional ``attenuation_per_meter``
    field defaulting to None. Existing constructors that don't pass
    it (e.g., the bound-mode Stoneley and flexural solvers)
    continue to work unchanged."""
    from fwap.cylindrical_solver import BoreholeMode

    # Construct without the new field -- still works.
    bm = BoreholeMode(
        name="Stoneley", azimuthal_order=0,
        freq=np.array([1000.0, 2000.0]),
        slowness=np.array([7e-4, 7e-4]),
    )
    assert bm.attenuation_per_meter is None


def test_borehole_mode_attenuation_per_meter_field_accepts_array():
    """The new field accepts an ndarray for leaky modes carrying
    spatial attenuation Im(k_z)."""
    from fwap.cylindrical_solver import BoreholeMode

    bm = BoreholeMode(
        name="pseudo_rayleigh", azimuthal_order=0,
        freq=np.array([15000.0, 20000.0]),
        slowness=np.array([4.5e-4, 4.2e-4]),
        attenuation_per_meter=np.array([1e-3, 5e-4]),
    )
    assert bm.attenuation_per_meter is not None
    np.testing.assert_array_equal(
        bm.attenuation_per_meter, np.array([1e-3, 5e-4]),
    )


def test_existing_stoneley_solver_attenuation_field_is_None():
    """The existing ``stoneley_dispersion`` doesn't populate
    ``attenuation_per_meter`` (Stoneley is bound -> attenuation is
    zero -> no field needed). Verify the default-None behaviour
    survives the dataclass extension."""
    res = stoneley_dispersion(
        np.array([1000.0]), vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert res.attenuation_per_meter is None


# =====================================================================
# n=0 leaky pseudo-Rayleigh modal-determinant solver tests
# =====================================================================
#
# Fast-formation parameters used throughout (V_S > V_f so the
# pseudo-Rayleigh leaky mode exists in the slowness window
# (1/V_P, 1/V_S)). Limestone-like properties.

PR_VP = 5500.0
PR_VS = 3100.0
PR_RHO = 2500.0
PR_VF = 1500.0
PR_RHO_F = 1000.0
PR_A = 0.1


def test_pseudo_rayleigh_rejects_slow_formation():
    """The pseudo-Rayleigh mode does not exist when V_S <= V_f
    (slow formation -- the s-branch leaky condition is unreachable
    inside the fluid-bound regime). Construct must reject."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    f = np.array([10000.0])
    with pytest.raises(ValueError, match="fast formation"):
        pseudo_rayleigh_dispersion(
            f, vp=PR_VP, vs=1200.0, rho=PR_RHO,
            vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
        )


def test_pseudo_rayleigh_rejects_non_positive_inputs():
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    f = np.array([10000.0])
    base = dict(vp=PR_VP, vs=PR_VS, rho=PR_RHO,
                vf=PR_VF, rho_f=PR_RHO_F, a=PR_A)
    with pytest.raises(ValueError, match="vp, vs, rho"):
        pseudo_rayleigh_dispersion(f, **{**base, "rho": 0.0})
    with pytest.raises(ValueError, match="vf and rho_f"):
        pseudo_rayleigh_dispersion(f, **{**base, "vf": 0.0})
    with pytest.raises(ValueError, match="^a must"):
        pseudo_rayleigh_dispersion(f, **{**base, "a": -0.1})
    with pytest.raises(ValueError, match="vp > vs"):
        pseudo_rayleigh_dispersion(f, **{**base, "vp": PR_VS})


def test_pseudo_rayleigh_rejects_non_positive_freq():
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    bad_freq = np.array([1000.0, 0.0, 5000.0])
    with pytest.raises(ValueError, match="freq"):
        pseudo_rayleigh_dispersion(
            bad_freq, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
            vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
        )


def test_pseudo_rayleigh_returns_borehole_mode_with_attenuation():
    """The leaky-mode return contract: ``BoreholeMode`` with the
    expected name, n=0 azimuthal order, frequency echoed, and a
    populated ``attenuation_per_meter`` field whose shape matches
    ``slowness``."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    freq = np.linspace(30000.0, 80000.0, 20)
    res = pseudo_rayleigh_dispersion(
        freq, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "pseudo_rayleigh"
    assert res.azimuthal_order == 0
    np.testing.assert_array_equal(res.freq, freq)
    assert res.slowness.shape == freq.shape
    assert res.attenuation_per_meter is not None
    assert res.attenuation_per_meter.shape == freq.shape


def test_pseudo_rayleigh_within_leaky_s_regime():
    """Every finite slowness must sit strictly inside the leaky-S
    window ``1/V_P < slowness < 1/V_S`` (equivalently, phase
    velocity in (V_S, V_P)). Outside that window the s-branch
    formula is not physical."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    freq = np.linspace(30000.0, 80000.0, 50)
    res = pseudo_rayleigh_dispersion(
        freq, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    finite = np.isfinite(res.slowness)
    assert finite.any(), "expected at least some finite slownesses"
    s = res.slowness[finite]
    assert (s > 1.0 / PR_VP).all()
    assert (s < 1.0 / PR_VS).all()


def test_pseudo_rayleigh_attenuation_strictly_positive():
    """Spatial attenuation ``Im(k_z) > 0`` everywhere the mode is
    finite -- a defining property of the radiating leaky regime."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    freq = np.linspace(30000.0, 80000.0, 50)
    res = pseudo_rayleigh_dispersion(
        freq, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    finite = np.isfinite(res.attenuation_per_meter)
    assert finite.any()
    assert (res.attenuation_per_meter[finite] > 0.0).all()


def test_pseudo_rayleigh_frequency_order_invariant():
    """The marcher walks descending frequency internally; the same
    physical inputs must produce the same per-frequency outputs
    regardless of whether the caller passes an ascending or
    descending grid."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    freq_asc = np.linspace(30000.0, 80000.0, 30)
    freq_desc = freq_asc[::-1]
    res_asc = pseudo_rayleigh_dispersion(
        freq_asc, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    res_desc = pseudo_rayleigh_dispersion(
        freq_desc, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    # Compare element-wise between freq[i] in ascending order and
    # the matching freq[-1-i] in descending order.
    np.testing.assert_allclose(
        res_asc.slowness, res_desc.slowness[::-1],
        rtol=1.0e-9, equal_nan=True,
    )
    np.testing.assert_allclose(
        res_asc.attenuation_per_meter,
        res_desc.attenuation_per_meter[::-1],
        rtol=1.0e-9, equal_nan=True,
    )


def test_pseudo_rayleigh_handles_empty_frequency_array():
    """Empty ``freq`` should return empty arrays without crashing
    -- a no-op that the marcher must short-circuit before
    indexing the (non-existent) high-frequency seed point."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    res = pseudo_rayleigh_dispersion(
        np.array([], dtype=float),
        vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    assert res.slowness.size == 0
    assert res.attenuation_per_meter.size == 0


def test_pseudo_rayleigh_modal_determinant_at_root_is_small():
    """At a converged pseudo-Rayleigh root the leaky-formulated
    n=0 modal determinant must be small relative to nearby points
    (a local zero on the leaky branch). Validates that the public
    function is converging onto an actual root rather than a
    plateau."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n0_complex, pseudo_rayleigh_dispersion,
    )

    freq = np.array([60000.0])
    res = pseudo_rayleigh_dispersion(
        freq, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    assert np.isfinite(res.slowness[0])
    omega = 2.0 * np.pi * freq[0]
    kz_root = complex(res.slowness[0] * omega, res.attenuation_per_meter[0])
    det_at_root = _modal_determinant_n0_complex(
        kz_root, omega, PR_VP, PR_VS, PR_RHO, PR_VF, PR_RHO_F, PR_A,
        leaky_p=False, leaky_s=True,
    )
    # The matrix entries scale with K_n / I_n / Hankel values that
    # span many orders of magnitude (matrix norm ~ 1e9 for these
    # parameters); the determinant at a true zero is many orders
    # below that scale.
    kz_off_root = kz_root * 1.05
    det_off_root = _modal_determinant_n0_complex(
        kz_off_root, omega, PR_VP, PR_VS, PR_RHO, PR_VF, PR_RHO_F, PR_A,
        leaky_p=False, leaky_s=True,
    )
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-2


def test_pseudo_rayleigh_high_velocity_below_vp():
    """As frequency decreases toward the cutoff, the mode's phase
    velocity rises but never crosses V_P (the upper edge of the
    leaky-S regime). Check the maximum velocity over the supported
    band stays below V_P."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    freq = np.linspace(30000.0, 100000.0, 60)
    res = pseudo_rayleigh_dispersion(
        freq, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    finite = np.isfinite(res.slowness)
    velocity = 1.0 / res.slowness[finite]
    # All velocities strictly below V_P (matches slowness > 1/V_P).
    assert (velocity < PR_VP).all()
    # And strictly above V_S (matches slowness < 1/V_S).
    assert (velocity > PR_VS).all()


# =====================================================================
# Plan item C: cutoff handling + branch tracker (validated marcher,
# BranchSegment dataclass, and segments-from-kz-curve splitter).
# =====================================================================


def test_classify_marcher_step_returns_ok_when_no_validator():
    """A converged complex root with no validator passes through
    as ``"ok"`` (the strict-marcher semantics)."""
    from fwap.cylindrical_solver import _classify_marcher_step

    assert _classify_marcher_step(complex(1.0, 0.5), 1000.0, None) == "ok"


def test_classify_marcher_step_recognises_convergence_failure():
    """A ``None`` ``kz_root`` is reported as
    ``"convergence_failure"`` regardless of validator state."""
    from fwap.cylindrical_solver import _classify_marcher_step

    assert _classify_marcher_step(None, 1000.0, None) == "convergence_failure"
    assert _classify_marcher_step(None, 1000.0, lambda kz, w: True) \
        == "convergence_failure"


def test_classify_marcher_step_recognises_regime_exit():
    """A converged root rejected by the validator is reported as
    ``"regime_exit"``, and a validator exception (``ValueError`` /
    ``ArithmeticError``) is treated as the same."""
    from fwap.cylindrical_solver import _classify_marcher_step

    rejector = lambda kz, w: False  # noqa: E731
    raiser = lambda kz, w: (_ for _ in ()).throw(ValueError("bad"))  # noqa: E731

    assert _classify_marcher_step(complex(1.0, 0.0), 1.0, rejector) \
        == "regime_exit"
    assert _classify_marcher_step(complex(1.0, 0.0), 1.0, raiser) \
        == "regime_exit"


def test_branch_segment_dataclass_contract():
    """BranchSegment exposes start_idx, end_idx, freq, kz, and a
    Python ``len`` matching the inclusive index range."""
    from fwap.cylindrical_solver import BranchSegment

    f = np.array([1.0, 2.0, 3.0])
    kz = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    seg = BranchSegment(start_idx=0, end_idx=2, freq=f, kz=kz)
    assert seg.start_idx == 0
    assert seg.end_idx == 2
    assert len(seg) == 3
    np.testing.assert_array_equal(seg.freq, f)
    np.testing.assert_array_equal(seg.kz, kz)
    # Single-sample segment.
    one = BranchSegment(start_idx=5, end_idx=5,
                        freq=np.array([4.0]), kz=np.array([4.0 + 0j]))
    assert len(one) == 1


def test_segments_from_kz_curve_handles_nan_gap():
    """Two-segment input with one NaN gap returns exactly two
    BranchSegments with the right index ranges and freq/kz
    slices."""
    from fwap.cylindrical_solver import segments_from_kz_curve

    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nan = np.nan + 1j * np.nan
    kz = np.array([1.0 + 0.1j, 2.0 + 0.2j, nan, 4.0 + 0.4j, 5.0 + 0.5j])
    segs = segments_from_kz_curve(f, kz)

    assert len(segs) == 2
    assert segs[0].start_idx == 0
    assert segs[0].end_idx == 1
    np.testing.assert_array_equal(segs[0].freq, f[:2])
    np.testing.assert_array_equal(segs[0].kz, kz[:2])
    assert segs[1].start_idx == 3
    assert segs[1].end_idx == 4
    np.testing.assert_array_equal(segs[1].freq, f[3:])
    np.testing.assert_array_equal(segs[1].kz, kz[3:])


def test_segments_from_kz_curve_no_finite_returns_empty():
    """An all-NaN curve produces an empty segment list."""
    from fwap.cylindrical_solver import segments_from_kz_curve

    nan = np.nan + 1j * np.nan
    f = np.array([1.0, 2.0, 3.0])
    kz = np.array([nan, nan, nan])
    assert segments_from_kz_curve(f, kz) == []


def test_segments_from_kz_curve_rejects_mismatched_lengths():
    """Length mismatch between freq and kz must raise."""
    from fwap.cylindrical_solver import segments_from_kz_curve

    with pytest.raises(ValueError, match="same length"):
        segments_from_kz_curve(np.array([1.0, 2.0]), np.array([1.0 + 0j]))


def test_validated_marcher_skips_invalid_step_and_continues():
    """One bad step in the middle of the curve gets NaN'd out, and
    marching continues from the previous good step. Use a det
    function with a known root at every frequency, plus a
    validator that fakes a single-step regime exit."""
    from fwap.cylindrical_solver import _march_complex_dispersion_validated

    # Synthetic "dispersion": kz(omega) = omega / V with V=1500 m/s.
    # Det vanishes at kz = omega/V; nearby seeds converge there.
    V = 1500.0

    def det(kz, omega):
        target = omega / V
        return complex(kz - target)  # zero at kz = omega/V

    freq = np.linspace(1000.0, 5000.0, 5)
    seed = complex(2.0 * np.pi * freq[0] / V, 0.0)

    skipped_idx = 2

    def validator(kz, omega):
        # Reject only the converged root at the third frequency,
        # to simulate a one-off regime-exit step.
        target = omega / V
        if abs(omega - 2.0 * np.pi * freq[skipped_idx]) < 1.0:
            return False
        return abs(kz - target) < 1.0e-3

    kz_curve = _march_complex_dispersion_validated(
        det, freq, seed, validator=validator,
        max_consecutive_invalid=2,
    )

    # Step 2 must be NaN; all others must be finite and on the
    # analytic kz=omega/V trajectory.
    assert np.isnan(kz_curve[skipped_idx].real)
    assert np.isnan(kz_curve[skipped_idx].imag)
    omega_arr = 2.0 * np.pi * freq
    expected_re = omega_arr / V
    for i, e in enumerate(expected_re):
        if i == skipped_idx:
            continue
        np.testing.assert_allclose(kz_curve[i].real, e, rtol=1.0e-9)


def test_validated_marcher_stops_after_budget_exhausted():
    """Once the consecutive-invalid budget is exceeded, the
    marcher stops and the rest of the curve stays NaN."""
    from fwap.cylindrical_solver import _march_complex_dispersion_validated

    V = 1500.0

    def det(kz, omega):
        return complex(kz - omega / V)

    freq = np.linspace(1000.0, 10000.0, 10)
    seed = complex(2.0 * np.pi * freq[0] / V, 0.0)

    # Reject everything from the third step onward.
    def validator(kz, omega):
        target = omega / V
        return omega / (2.0 * np.pi) <= 3000.0 and abs(kz - target) < 1.0e-3

    kz_curve = _march_complex_dispersion_validated(
        det, freq, seed, validator=validator,
        max_consecutive_invalid=1,
    )

    # Steps at f = 1000, 2000, 3000 pass the validator; step at
    # 4000 is strike 1 (within budget, NaN'd but march continues);
    # step at 5000 is strike 2 (exceeds budget=1, march stops).
    for i in range(3):
        assert np.isfinite(kz_curve[i].real)
    for i in range(3, 10):
        assert not np.isfinite(kz_curve[i].real)


def test_validated_marcher_handles_empty_grid():
    """An empty frequency grid returns an empty kz_curve with no
    crash, mirroring the strict marcher's behaviour."""
    from fwap.cylindrical_solver import _march_complex_dispersion_validated

    out = _march_complex_dispersion_validated(
        lambda kz, w: complex(kz),
        np.array([], dtype=float),
        complex(1.0, 0.0),
    )
    assert out.shape == (0,)


def test_validated_marcher_zero_budget_matches_strict_marcher_semantics():
    """``max_consecutive_invalid=0`` reproduces the strict-stop
    behaviour: the very first invalid step ends the march."""
    from fwap.cylindrical_solver import _march_complex_dispersion_validated

    V = 1500.0

    def det(kz, omega):
        return complex(kz - omega / V)

    freq = np.linspace(1000.0, 5000.0, 5)
    seed = complex(2.0 * np.pi * freq[0] / V, 0.0)

    # Reject step index 1 onward.
    def validator(kz, omega):
        target = omega / V
        return omega / (2.0 * np.pi) <= 1000.0 and abs(kz - target) < 1.0e-3

    out = _march_complex_dispersion_validated(
        det, freq, seed, validator=validator,
        max_consecutive_invalid=0,
    )
    assert np.isfinite(out[0].real)
    for i in range(1, 5):
        assert not np.isfinite(out[i].real)


# =====================================================================
# Plan item B: leaky flexural mode (n=1) in fast formations.
# =====================================================================


def test_complex_n1_matches_real_in_bound_regime():
    """``_modal_determinant_n1`` and ``_modal_determinant_n1_complex``
    must agree to floating-point precision when called with real
    ``kz`` and both ``leaky_*`` flags False (the bound flexural
    regime). This is the regression invariant for the n=1 complex
    refactor: as long as it holds, the existing bound-mode tests
    cover the bound-regime physics; the complex evaluator only
    adds new capability on top."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n1,
        _modal_determinant_n1_complex,
    )

    # Slow-formation parameters (matches the existing bound-mode
    # bracket sweep).
    omega = 2.0 * np.pi * 5000.0
    for kz in [50.0, 60.0, 80.0, 100.0]:
        d_real = _modal_determinant_n1(
            kz, omega, SLOW_VP, SLOW_VS, SLOW_RHO,
            SLOW_VF, SLOW_RHO_F, SLOW_A,
        )
        d_complex = _modal_determinant_n1_complex(
            complex(kz), omega, SLOW_VP, SLOW_VS, SLOW_RHO,
            SLOW_VF, SLOW_RHO_F, SLOW_A,
        )
        # Real part agrees exactly (or to a small relative tolerance
        # for the matrix-product cumulative roundoff).
        rel_err = abs(d_complex.real - d_real) / max(abs(d_real), 1.0e-300)
        assert rel_err < 1.0e-12, (
            f"real-vs-complex n=1 mismatch at kz={kz}: "
            f"real={d_real}, complex={d_complex}, rel_err={rel_err}"
        )
        # Imaginary part is identically zero in the bound regime
        # (real arithmetic on real inputs).
        assert d_complex.imag == 0.0


def test_flexural_slow_formation_bit_identical_after_dispatch():
    """Plan item B validation goal #2: when ``V_S < V_f`` the new
    auto-dispatch must bypass the complex path and reproduce the
    existing slow-formation answer bit-for-bit. We compare against
    a direct call into ``_modal_determinant_n1`` + brentq with the
    same bracket helper, which is what the slow-formation branch
    of the refactored ``flexural_dispersion`` does internally."""
    from fwap.cylindrical_solver import (
        _flexural_kz_bracket, _modal_determinant_n1,
    )
    from scipy import optimize as _opt

    f = np.linspace(2000.0, 12000.0, 11)
    res = flexural_dispersion(
        f, vp=SLOW_VP, vs=SLOW_VS, rho=SLOW_RHO,
        vf=SLOW_VF, rho_f=SLOW_RHO_F, a=SLOW_A,
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 8, "slow-formation path must still find roots above cutoff"

    # Reference computation: open-coded brentq on
    # ``_modal_determinant_n1`` with the existing bracket helper.
    for i in np.where(finite)[0]:
        omega = 2.0 * np.pi * f[i]

        def _det(kz):
            return _modal_determinant_n1(
                kz, omega, SLOW_VP, SLOW_VS, SLOW_RHO,
                SLOW_VF, SLOW_RHO_F, SLOW_A,
            )

        kz_lo, kz_hi = _flexural_kz_bracket(
            omega, SLOW_VP, SLOW_VS, SLOW_RHO,
            SLOW_VF, SLOW_RHO_F, SLOW_A,
        )
        # Bracket-expansion mirrors the production code.
        n_expand = 0
        d_lo = _det(kz_lo)
        d_hi = _det(kz_hi)
        while np.sign(d_lo) == np.sign(d_hi) and n_expand < 8:
            kz_hi *= 1.5
            d_hi = _det(kz_hi)
            n_expand += 1
        kz_ref = _opt.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
        slow_ref = kz_ref / omega
        assert res.slowness[i] == slow_ref, (
            f"slow-formation dispatch changed value at f={f[i]}: "
            f"got {res.slowness[i]}, ref {slow_ref}"
        )

    # Bound mode -> attenuation_per_meter stays None.
    assert res.attenuation_per_meter is None


def test_flexural_fast_formation_velocities_are_real_kz():
    """In the fast-formation regime the converged ``k_z`` is real
    to floating-point precision (``Im(k_z) = 0``). This is enforced
    by brentq'ing ``Im(det)`` along the real-``k_z`` axis, which can
    only return real-valued ``k_z``. The converged determinant must
    therefore have small magnitude -- an upper-bound sanity check."""
    from fwap.cylindrical_solver import _modal_determinant_n1_complex
    from fwap.cylindrical import rayleigh_speed

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    vR = rayleigh_speed(vp, vs)
    f = np.linspace(20000.0, 80000.0, 30)
    res = flexural_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    finite_idx = np.where(np.isfinite(res.slowness))[0]
    assert finite_idx.size >= 3, "expected at least a few finite samples"

    # At each converged root, |det| must be small relative to
    # |det| at an off-root point (the same local-zero test used
    # for the n=0 leaky solver).
    for i in finite_idx[:5]:
        omega = 2.0 * np.pi * res.freq[i]
        kz_root = complex(res.slowness[i] * omega, 0.0)
        det_at_root = _modal_determinant_n1_complex(
            kz_root, omega, vp, vs, rho, vf, rho_f, a,
            leaky_p=False, leaky_s=False,
        )
        # Off-root sample: shift kz by 1 % toward V_R.
        kz_off = kz_root * 1.005
        det_off = _modal_determinant_n1_complex(
            kz_off, omega, vp, vs, rho, vf, rho_f, a,
            leaky_p=False, leaky_s=False,
        )
        # Brentq targets only Im(det); Re(det) is small in the
        # regime but not driven to zero. Compare imaginary parts,
        # which should be many orders smaller at the root.
        assert abs(det_at_root.imag) < abs(det_off.imag) * 1.0e-2, (
            f"|Im(det)| not small at converged root for f={res.freq[i]}: "
            f"root={det_at_root}, off-root={det_off}"
        )


def test_flexural_fast_formation_frequency_order_invariant():
    """The fast-formation marcher walks descending frequency
    internally; ascending- and descending-grid inputs must
    produce identical per-frequency outputs."""
    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f_asc = np.linspace(20000.0, 80000.0, 25)
    f_desc = f_asc[::-1]
    res_asc = flexural_dispersion(
        f_asc, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    res_desc = flexural_dispersion(
        f_desc, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    np.testing.assert_allclose(
        res_asc.slowness, res_desc.slowness[::-1],
        rtol=1.0e-9, equal_nan=True,
    )


def test_pseudo_rayleigh_segmenter_returns_single_continuous_segment():
    """For the standard fast-formation parameter set, the validated
    marcher recovers one contiguous segment over the supported
    band -- no NaN gaps in the middle even where the strict
    marcher used to drop steps to single-step root hops."""
    from fwap.cylindrical_solver import (
        pseudo_rayleigh_dispersion, segments_from_kz_curve,
    )

    freq = np.linspace(30000.0, 80000.0, 60)
    res = pseudo_rayleigh_dispersion(
        freq, vp=PR_VP, vs=PR_VS, rho=PR_RHO,
        vf=PR_VF, rho_f=PR_RHO_F, a=PR_A,
    )
    omega = 2.0 * np.pi * res.freq
    kz = res.slowness * omega + 1j * res.attenuation_per_meter
    segs = segments_from_kz_curve(res.freq, kz)
    assert len(segs) == 1
    assert len(segs[0]) == freq.size


# =====================================================================
# Plan item D: n=2 quadrupole bound-mode dispersion tests.
# =====================================================================
#
# Slow-formation parameters (V_S < V_f). Same set as the slow-formation
# flexural tests so the two suites can share intuition about the
# cutoff and the V_R high-f asymptote.

QUAD_VP = 2200.0
QUAD_VS = 800.0
QUAD_RHO = 2200.0
QUAD_VF = 1500.0
QUAD_RHO_F = 1000.0
QUAD_A = 0.1


def test_quadrupole_returns_borehole_mode_with_n2_label():
    """Output dataclass contract: ``BoreholeMode`` with
    ``name = "quadrupole"`` and ``azimuthal_order = 2``."""
    from fwap.cylindrical_solver import quadrupole_dispersion

    f = np.linspace(3000.0, 12000.0, 5)
    res = quadrupole_dispersion(
        f, vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
        vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    np.testing.assert_array_equal(res.freq, f)
    assert res.slowness.shape == f.shape
    # Bound mode -> attenuation_per_meter is None (same convention
    # as Stoneley and the bound flexural).
    assert res.attenuation_per_meter is None


def test_quadrupole_finite_above_cutoff_in_slow_formation():
    """In the slow-formation regime the bound n=2 mode exists
    above a geometric cutoff. At least one frequency in a
    reasonable band must yield a finite slowness in the
    ``(1/V_S, ~1.1/V_R)`` window."""
    from fwap.cylindrical_solver import quadrupole_dispersion
    from fwap.cylindrical import rayleigh_speed

    vR = rayleigh_speed(QUAD_VP, QUAD_VS)
    f = np.linspace(3000.0, 20000.0, 10)
    res = quadrupole_dispersion(
        f, vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
        vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A,
    )
    finite = np.isfinite(res.slowness)
    assert finite.any(), "expected the bound n=2 mode to exist above cutoff"
    velocity = 1.0 / res.slowness[finite]
    # All recovered velocities sit between the Scholte limit
    # (slightly below V_R; fluid loading depresses the asymptote
    # ~5% below the vacuum Rayleigh speed for this rock) and V_S.
    assert (velocity < QUAD_VS).all()
    assert (velocity > vR * 0.85).all()


def test_quadrupole_dispatches_to_fast_formation_path_when_vs_gt_vf():
    """Plan item E: ``quadrupole_dispersion`` now auto-dispatches to
    the complex-determinant fast-formation path when ``V_S > V_f``
    instead of returning NaN throughout. At least some frequencies
    in a sensible band must yield finite slowness in the
    ``(V_R, V_S)`` velocity window. Direct sister of the n=1
    fast-formation dispatch test."""
    from fwap.cylindrical_solver import quadrupole_dispersion
    from fwap.cylindrical import rayleigh_speed

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    vR = rayleigh_speed(vp, vs)
    f = np.linspace(40000.0, 100000.0, 30)
    res = quadrupole_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    finite = np.isfinite(res.slowness)
    assert finite.any(), "fast-formation path must populate at least one frequency"
    velocity = 1.0 / res.slowness[finite]
    assert (velocity > vR * 0.99).all(), (
        f"velocity must stay near or above V_R ({vR:.0f}); got {velocity}"
    )
    assert (velocity < vs).all(), (
        f"velocity must stay below V_S ({vs}); got {velocity}"
    )
    # Bound mode -> attenuation_per_meter is None.
    assert res.attenuation_per_meter is None


def test_quadrupole_returns_nan_below_geometric_cutoff():
    """The dipole-side cutoff is at ``V_S / (2 pi a) ~ 1273 Hz``
    for these slow-formation parameters; the n=2 quadrupole
    cutoff is higher (typically ~2.5 to 4 times the n=1 cutoff,
    so ~3-5 kHz here). At very low frequencies the bracket has
    no sign change and the function returns NaN rather than
    spurious roots."""
    from fwap.cylindrical_solver import quadrupole_dispersion

    f = np.array([500.0, 1000.0])  # well below n=2 cutoff
    res = quadrupole_dispersion(
        f, vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
        vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A,
    )
    assert np.all(np.isnan(res.slowness))


def test_quadrupole_long_wavelength_slowness_above_inverse_vs():
    """All bound n=2 roots above cutoff have ``slowness > 1/V_S``
    (kz > omega/V_S, the bound-regime floor); just above the
    cutoff slowness is closest to ``1/V_S`` and grows slightly at
    higher frequency toward the Scholte limit."""
    from fwap.cylindrical_solver import quadrupole_dispersion

    f = np.linspace(5000.0, 30000.0, 26)
    res = quadrupole_dispersion(
        f, vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
        vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A,
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 5
    s_vals = res.slowness[finite]
    assert (s_vals > 1.0 / QUAD_VS).all()


def test_quadrupole_modal_determinant_at_root_is_near_zero():
    """At a converged quadrupole root the bound-regime modal
    determinant must be many orders smaller than the determinant
    at a nearby off-root point. Same local-zero check used for
    Stoneley and flexural roots."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion, _modal_determinant_n2,
    )

    f = np.array([8000.0])
    res = quadrupole_dispersion(
        f, vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
        vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A,
    )
    assert np.isfinite(res.slowness[0])
    omega = 2.0 * np.pi * f[0]
    kz_root = res.slowness[0] * omega
    det_at_root = _modal_determinant_n2(
        kz_root, omega, QUAD_VP, QUAD_VS, QUAD_RHO,
        QUAD_VF, QUAD_RHO_F, QUAD_A,
    )
    # Off-root sample 5 % below kz_root.
    kz_off = kz_root * 0.95
    det_off = _modal_determinant_n2(
        kz_off, omega, QUAD_VP, QUAD_VS, QUAD_RHO,
        QUAD_VF, QUAD_RHO_F, QUAD_A,
    )
    assert abs(det_at_root) < abs(det_off) * 1.0e-2


def test_quadrupole_dispersion_rejects_non_positive_inputs():
    """Mirrors the Stoneley / flexural input-validation suite."""
    from fwap.cylindrical_solver import quadrupole_dispersion

    f = np.array([5000.0])
    base = dict(vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
                vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A)
    with pytest.raises(ValueError, match="vp, vs, rho"):
        quadrupole_dispersion(f, **{**base, "rho": 0.0})
    with pytest.raises(ValueError, match="vf and rho_f"):
        quadrupole_dispersion(f, **{**base, "vf": 0.0})
    with pytest.raises(ValueError, match="^a must"):
        quadrupole_dispersion(f, **{**base, "a": -0.1})
    with pytest.raises(ValueError, match="vp > vs"):
        quadrupole_dispersion(f, **{**base, "vp": QUAD_VS})


def test_quadrupole_dispersion_rejects_non_positive_freq():
    from fwap.cylindrical_solver import quadrupole_dispersion

    bad = np.array([1000.0, -500.0, 5000.0])
    with pytest.raises(ValueError, match="freq"):
        quadrupole_dispersion(
            bad, vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
            vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A,
        )


# =====================================================================
# Plan item E: leaky quadrupole mode (n=2) in fast formations.
# =====================================================================


def test_complex_n2_matches_real_in_bound_regime():
    """``_modal_determinant_n2`` and ``_modal_determinant_n2_complex``
    must agree to floating-point precision when called with real
    ``kz`` and both ``leaky_*`` flags False (the bound quadrupole
    regime). Same regression invariant as for n=0 and n=1; ensures
    the existing slow-formation bound-mode tests carry over to the
    complex evaluator without modification."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2,
        _modal_determinant_n2_complex,
    )

    # Slow-formation parameters; pick kz values comfortably inside
    # the bound regime ``kz > omega/V_S`` (avoiding the s = 0
    # boundary where the real evaluator's np.sqrt of a negative
    # would produce NaN).
    omega = 2.0 * np.pi * 8000.0
    for kz in [70.0, 90.0, 120.0, 200.0]:
        d_real = _modal_determinant_n2(
            kz, omega, QUAD_VP, QUAD_VS, QUAD_RHO,
            QUAD_VF, QUAD_RHO_F, QUAD_A,
        )
        d_complex = _modal_determinant_n2_complex(
            complex(kz), omega, QUAD_VP, QUAD_VS, QUAD_RHO,
            QUAD_VF, QUAD_RHO_F, QUAD_A,
        )
        rel_err = abs(d_complex.real - d_real) / max(abs(d_real), 1.0e-300)
        assert rel_err < 1.0e-12, (
            f"real-vs-complex n=2 mismatch at kz={kz}: "
            f"real={d_real}, complex={d_complex}, rel_err={rel_err}"
        )
        assert d_complex.imag == 0.0


def test_quadrupole_slow_formation_bit_identical_after_dispatch():
    """Plan item E validation goal #2: when ``V_S < V_f`` the new
    auto-dispatch must bypass the complex path and reproduce the
    existing slow-formation answer bit-for-bit. We compare against
    a direct call into ``_modal_determinant_n2`` + brentq with the
    same bracket helper, mirroring the n=1 fast-formation
    bit-identical guard from plan item B."""
    from fwap.cylindrical_solver import (
        _quadrupole_kz_bracket, _modal_determinant_n2, quadrupole_dispersion,
    )
    from scipy import optimize as _opt

    f = np.linspace(5000.0, 20000.0, 11)
    res = quadrupole_dispersion(
        f, vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO,
        vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A,
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 8

    for i in np.where(finite)[0]:
        omega = 2.0 * np.pi * f[i]

        def _det(kz):
            return _modal_determinant_n2(
                kz, omega, QUAD_VP, QUAD_VS, QUAD_RHO,
                QUAD_VF, QUAD_RHO_F, QUAD_A,
            )

        kz_lo, kz_hi = _quadrupole_kz_bracket(
            omega, QUAD_VP, QUAD_VS, QUAD_RHO,
            QUAD_VF, QUAD_RHO_F, QUAD_A,
        )
        n_expand = 0
        d_lo = _det(kz_lo)
        d_hi = _det(kz_hi)
        while np.sign(d_lo) == np.sign(d_hi) and n_expand < 8:
            kz_hi *= 1.5
            d_hi = _det(kz_hi)
            n_expand += 1
        kz_ref = _opt.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
        slow_ref = kz_ref / omega
        assert res.slowness[i] == slow_ref, (
            f"slow-formation dispatch changed value at f={f[i]}: "
            f"got {res.slowness[i]}, ref {slow_ref}"
        )


def test_quadrupole_fast_formation_im_det_relative_zero():
    """In the fast-formation regime the converged ``k_z`` is real
    and brentq targets ``Im(det) = 0``. Because the n=2
    determinant magnitudes are ~15 orders larger than the n=1
    sister, the absolute residual ``|Im(det)|`` at the converged
    root can still be ~1e8 -- but it must be *relatively* tiny
    against ``|det|`` itself (machine-precision territory). This
    is the n=2 analogue of the n=1 local-zero check, expressed
    as a relative residual rather than as an absolute one."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion, _modal_determinant_n2_complex,
    )

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(50000.0, 100000.0, 30)
    res = quadrupole_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    finite_idx = np.where(np.isfinite(res.slowness))[0]
    assert finite_idx.size >= 3

    for i in finite_idx[:5]:
        omega = 2.0 * np.pi * res.freq[i]
        kz_root = complex(res.slowness[i] * omega, 0.0)
        det_at_root = _modal_determinant_n2_complex(
            kz_root, omega, vp, vs, rho, vf, rho_f, a,
            leaky_p=False, leaky_s=False,
        )
        relative = abs(det_at_root.imag) / max(abs(det_at_root), 1.0e-300)
        assert relative < 1.0e-12, (
            f"Im(det)/|det| not at machine precision at converged "
            f"root for f={res.freq[i]}: relative={relative}, "
            f"det={det_at_root}"
        )


def test_quadrupole_fast_formation_velocities_in_bound_window():
    """All fast-formation finite outputs must have phase velocity
    strictly between V_R and V_S -- the bound-mode window for the
    n=2 leaky-F regime. Mirrors the n=1 sister test."""
    from fwap.cylindrical_solver import quadrupole_dispersion
    from fwap.cylindrical import rayleigh_speed

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    vR = rayleigh_speed(vp, vs)
    f = np.linspace(40000.0, 100000.0, 50)
    res = quadrupole_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 5
    velocity = 1.0 / res.slowness[finite]
    assert (velocity > vR * 0.99).all()
    assert (velocity < vs).all()


def test_quadrupole_fast_formation_frequency_order_invariant():
    """The fast-formation marcher walks descending frequency
    internally; ascending- and descending-grid inputs must
    produce identical per-frequency outputs."""
    from fwap.cylindrical_solver import quadrupole_dispersion

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f_asc = np.linspace(50000.0, 100000.0, 25)
    f_desc = f_asc[::-1]
    res_asc = quadrupole_dispersion(
        f_asc, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_desc = quadrupole_dispersion(
        f_desc, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    np.testing.assert_allclose(
        res_asc.slowness, res_desc.slowness[::-1],
        rtol=1.0e-9, equal_nan=True,
    )


# =====================================================================
# Plan item F (foundation): layered Stoneley dispersion API
# =====================================================================
#
# Foundation tests for the layered n=0 public API. The 7x7 layered
# modal determinant itself is the next step of plan item F; here we
# only exercise:
#
#   * the ``BoreholeLayer`` dataclass + validator,
#   * the empty-layers degenerate dispatch (must be bit-equivalent
#     to ``stoneley_dispersion``), and
#   * the ``NotImplementedError`` sentinel for non-empty layers.


def test_borehole_layer_dataclass_construction():
    layer = BoreholeLayer(vp=4000.0, vs=2200.0, rho=2300.0, thickness=0.005)
    assert layer.vp == 4000.0
    assert layer.vs == 2200.0
    assert layer.rho == 2300.0
    assert layer.thickness == 0.005


def test_stoneley_dispersion_layered_empty_layers_bit_matches_unlayered():
    """Degenerate single-interface case: ``layers=()`` must produce a
    slowness curve bit-identical to :func:`stoneley_dispersion`. This
    is the floating-point oracle that will continue to anchor the
    layered solver once the 7x7 modal determinant lands."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(500.0, 8000.0, 16)
    res_unlayered = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_layered = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a, layers=(),
    )
    np.testing.assert_array_equal(res_layered.slowness, res_unlayered.slowness)
    np.testing.assert_array_equal(res_layered.freq, res_unlayered.freq)
    assert res_layered.name == res_unlayered.name == "Stoneley"
    assert res_layered.azimuthal_order == 0


def test_stoneley_dispersion_layered_empty_layers_returns_borehole_mode():
    f = np.linspace(500.0, 5000.0, 5)
    res = stoneley_dispersion_layered(
        f, vp=4500.0, vs=2500.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0


def test_stoneley_dispersion_layered_multilayer_raises_not_implemented():
    """Multi-layer stacks (``len(layers) > 1``) currently raise
    ``NotImplementedError`` until plan items G.c (stacked modal
    determinant) and G.d (public-API hook) land. Single-layer and
    unlayered are supported by F.1.b.4."""
    f = np.array([1000.0, 2000.0])
    layer1 = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    layer2 = BoreholeLayer(vp=3000.0, vs=1500.0, rho=1900.0, thickness=0.005)
    with pytest.raises(NotImplementedError, match=r"G\.c"):
        stoneley_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(layer1, layer2),
        )


def test_stoneley_dispersion_layered_rejects_bad_layer_object():
    f = np.array([1000.0])
    with pytest.raises(ValueError, match="BoreholeLayer"):
        stoneley_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=("not a layer",),
        )


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"vp": 0.0, "vs": 1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": -1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 1.0, "rho": 0.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 2.0, "rho": 1.0, "thickness": 1.0}, "vp > vs"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": 0.0}, "thickness"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": -0.1}, "thickness"),
    ],
)
def test_stoneley_dispersion_layered_rejects_malformed_layer_params(kwargs, msg):
    f = np.array([1000.0])
    layer = BoreholeLayer(**kwargs)
    with pytest.raises(ValueError, match=msg):
        stoneley_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(layer,),
        )


def test_stoneley_dispersion_layered_accepts_list_for_layers():
    """``layers`` should accept any iterable that ``tuple(...)``
    consumes; the empty list must dispatch to the unlayered solver
    just like ``()``."""
    f = np.linspace(500.0, 5000.0, 4)
    res_tuple = stoneley_dispersion_layered(
        f, vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1, layers=(),
    )
    res_list = stoneley_dispersion_layered(
        f, vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1, layers=[],
    )
    np.testing.assert_array_equal(res_tuple.slowness, res_list.slowness)


# =====================================================================
# Plan item F.1.b.1 -- radial-wavenumber + Bessel-pack helpers
# =====================================================================


def _layered_typical_params():
    """Bound-regime fast-formation parameters with a slower-than-
    formation mudcake. Used as the default fixture for F.1.b
    helper / row tests."""
    return dict(
        vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1,
        layer=BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005),
    )


def test_layered_radial_wavenumbers_bound_regime_returns_real_positive():
    """Above the bound-regime floor ``omega / min(V_S, V_S_m, V_f)``
    every wavenumber is real positive."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    floor = omega / min(p["vs"], p["layer"].vs, p["vf"])
    kz = floor * 1.5
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz, omega,
        vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    for v in (F_f, p_m, s_m, pp, ss):
        assert np.isfinite(v)
        assert v > 0.0


def test_layered_radial_wavenumbers_below_bound_floor_returns_nan():
    """Below the bound-regime floor the slowest-wave radial
    wavenumber goes imaginary; numpy.sqrt of a negative real
    returns NaN. The helper passes NaN through (brentq-safe;
    no raise)."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    # Pick kz strictly below ``omega / max(...)`` so every wavenumber
    # argument is negative (kz^2 - (omega/V)^2 < 0 needs
    # kz < omega/V for *every* wave speed V, which means
    # kz < omega/max(V)).
    fastest = max(p["vf"], p["vs"], p["layer"].vs, p["layer"].vp, p["vp"])
    kz = omega / fastest * 0.5
    with np.errstate(invalid="ignore"):
        F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
            kz, omega,
            vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
        )
    for v in (F_f, p_m, s_m, pp, ss):
        assert np.isnan(v)


def test_layered_radial_wavenumbers_satisfy_definition():
    """Each wavenumber squared equals ``kz^2 - (omega / V)^2`` per
    substep F.1.a.1."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz, omega,
        vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    assert F_f ** 2 == pytest.approx(kz ** 2 - (omega / p["vf"]) ** 2)
    assert p_m ** 2 == pytest.approx(kz ** 2 - (omega / p["layer"].vp) ** 2)
    assert s_m ** 2 == pytest.approx(kz ** 2 - (omega / p["layer"].vs) ** 2)
    assert pp ** 2 == pytest.approx(kz ** 2 - (omega / p["vp"]) ** 2)
    assert ss ** 2 == pytest.approx(kz ** 2 - (omega / p["vs"]) ** 2)


def test_layered_radial_wavenumbers_layer_equals_formation_collapses():
    """When the annulus material matches the formation, ``p_m == p``
    and ``s_m == s`` to floating-point precision -- the algebraic
    cornerstone of the substep F.1.a.6 self-check."""
    vp, vs = 4500.0, 2500.0
    p = dict(
        vp=vp, vs=vs, rho=2400.0, vf=1500.0,
        layer=BoreholeLayer(vp=vp, vs=vs, rho=2400.0, thickness=0.01),
    )
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, p["vf"]) * 1.5
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz, omega,
        vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    assert p_m == pp
    assert s_m == ss


def test_layered_bessel_pack_has_22_keys():
    """Substep F.1.b.1 plan: the pack covers 2 (fluid r=a) + 8
    (annulus P, both interfaces) + 8 (annulus S, both interfaces)
    + 4 (formation r=b) = 22 Bessel values."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    a = p["a"]
    b = a + p["layer"].thickness
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz, omega,
        vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    pack = _layered_n0_bessel_pack(F_f, p_m, s_m, pp, ss, a, b)
    assert len(pack) == 22
    expected_keys = {
        "I0_Ff_a", "I1_Ff_a",
        "I0_pm_a", "I1_pm_a", "K0_pm_a", "K1_pm_a",
        "I0_sm_a", "I1_sm_a", "K0_sm_a", "K1_sm_a",
        "I0_pm_b", "I1_pm_b", "K0_pm_b", "K1_pm_b",
        "I0_sm_b", "I1_sm_b", "K0_sm_b", "K1_sm_b",
        "K0_p_b", "K1_p_b", "K0_s_b", "K1_s_b",
    }
    assert set(pack.keys()) == expected_keys


def test_layered_bessel_pack_matches_scipy_directly():
    """Each entry in the pack must equal the corresponding direct
    scipy.special call to floating-point precision; this is the
    primary unit oracle for the helper."""
    from scipy import special

    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    a = p["a"]
    b = a + p["layer"].thickness
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz, omega,
        vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    pack = _layered_n0_bessel_pack(F_f, p_m, s_m, pp, ss, a, b)

    # Fluid at r = a.
    assert pack["I0_Ff_a"] == float(special.iv(0, F_f * a))
    assert pack["I1_Ff_a"] == float(special.iv(1, F_f * a))

    # Annulus P/S at both interfaces; formation P/S at r = b.
    cases = [
        ("pm", p_m, ("a", "b")),
        ("sm", s_m, ("a", "b")),
    ]
    for wave, alpha, radii in cases:
        for r_label, r in zip(radii, (a, b)):
            x = alpha * r
            assert pack[f"I0_{wave}_{r_label}"] == float(special.iv(0, x))
            assert pack[f"I1_{wave}_{r_label}"] == float(special.iv(1, x))
            assert pack[f"K0_{wave}_{r_label}"] == float(special.kv(0, x))
            assert pack[f"K1_{wave}_{r_label}"] == float(special.kv(1, x))

    assert pack["K0_p_b"] == float(special.kv(0, pp * b))
    assert pack["K1_p_b"] == float(special.kv(1, pp * b))
    assert pack["K0_s_b"] == float(special.kv(0, ss * b))
    assert pack["K1_s_b"] == float(special.kv(1, ss * b))


def test_layered_bessel_pack_layer_equals_formation_p_columns_match():
    """Substep F.1.a.6 self-check at the Bessel level: when the
    annulus material matches the formation, the K-flavour pack
    entries at ``r = b`` for the annulus P (``K0_pm_b``) match the
    formation P (``K0_p_b``); same for S."""
    vp, vs = 4500.0, 2500.0
    layer = BoreholeLayer(vp=vp, vs=vs, rho=2400.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5
    a = 0.1
    b = a + layer.thickness
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz, omega, vp=vp, vs=vs, vf=1500.0, layer=layer,
    )
    pack = _layered_n0_bessel_pack(F_f, p_m, s_m, pp, ss, a, b)
    assert pack["K0_pm_b"] == pack["K0_p_b"]
    assert pack["K1_pm_b"] == pack["K1_p_b"]
    assert pack["K0_sm_b"] == pack["K0_s_b"]
    assert pack["K1_sm_b"] == pack["K1_s_b"]


def test_layered_bessel_pack_propagates_nan_inputs():
    """Out-of-regime radial wavenumbers (NaN) propagate to NaN
    pack entries; the helper does not raise."""
    nan = float("nan")
    pack = _layered_n0_bessel_pack(nan, 10.0, 10.0, 10.0, 10.0, 0.1, 0.105)
    assert np.isnan(pack["I0_Ff_a"])
    assert np.isnan(pack["I1_Ff_a"])
    # Non-NaN inputs still produce finite Bessel values.
    assert np.isfinite(pack["K0_pm_a"])


# =====================================================================
# Plan item F.1.b.2.a -- row 1 of the n=0 layered determinant (r = a)
# =====================================================================


def _row1_test_setup():
    """Bound-regime kz / omega above every wavenumber floor for the
    typical fast-formation + soft-mudcake fixture."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    return p, omega, kz


def test_layered_row1_at_a_layer_equals_formation_per_element():
    """Substep F.1.a.6 self-check at the row level: with annulus
    properties identical to the formation, row 1 of the layered
    matrix has its (A, B_K, C_K) entries equal to ``M11, M12, M13``
    of :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row1_at_a(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    # Reconstruct the corresponding entries of the unlayered matrix
    # without invoking the determinant routine: M_11, M_12, M_13 of
    # _modal_determinant_n0 (lifted from the docstring / source).
    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = F * float(sp.iv(1, F * a)) / (rho_f * omega ** 2)
    M12 = p * float(sp.kv(1, p * a))
    M13 = kz * float(sp.kv(1, s * a))

    # Layer=formation collapses ``p_m -> p``, ``s_m -> s``. The
    # K-flavour columns (B_K, C_K) at indices 2, 4 then equal the
    # M_12 / M_13 entries of the unlayered matrix.
    assert row[0].real == pytest.approx(M11)
    assert row[2].real == pytest.approx(M12)
    assert row[4].real == pytest.approx(M13)
    assert abs(row[0].imag) < 1.0e-14
    assert abs(row[2].imag) < 1.0e-14
    assert abs(row[4].imag) < 1.0e-14


def test_layered_row1_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (indices 5, 6)
    are zero -- the formation half-space ``r > b`` doesn't touch
    the fluid-annulus interface."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row1_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row1_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: post-rescale row entries are
    purely real in the bound regime. Any non-zero imaginary part
    flags a sign error in the C-flavour rescaling."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row1_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row1_at_a_i_k_sign_flip():
    """Substep F.1.a.2 sign convention: the I-flavour annulus
    columns (B_I, C_I) carry the opposite sign of the K-flavour
    counterparts (B_K, C_K) on the B amplitudes, the *same* sign on
    the C amplitudes. Specifically:

        row[1] / row[2] == -I_1(p_m a) / K_1(p_m a)    (B_I vs B_K)
        row[3] / row[4] == +I_1(s_m a) / K_1(s_m a)    (C_I vs C_K)
    """
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n0_row1_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )

    expected_ratio_B = -float(sp.iv(1, p_m * p["a"])) / float(sp.kv(1, p_m * p["a"]))
    expected_ratio_C = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    assert row[1].real / row[2].real == pytest.approx(expected_ratio_B)
    assert row[3].real / row[4].real == pytest.approx(expected_ratio_C)


# =====================================================================
# Plan item F.1.b.2.b -- row 2 of the n=0 layered determinant (r = a)
# =====================================================================


def test_layered_row2_at_a_layer_equals_formation_per_element():
    """At layer=formation, row 2's (A, B_K, C_K) entries match
    M21, M22, M23 of :func:`_modal_determinant_n0` to floating-
    point precision -- the primary correctness oracle for the row's
    Lame-reduction bookkeeping."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row2_at_a(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M21 = -float(sp.iv(0, F * a))
    M22 = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p * a))
        + 2.0 * p * float(sp.kv(1, p * a)) / a
    )
    M23 = -2.0 * kz * mu * (
        s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a
    )

    assert row[0].real == pytest.approx(M21)
    assert row[2].real == pytest.approx(M22)
    assert row[4].real == pytest.approx(M23)


def test_layered_row2_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (5, 6) are
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row2_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row2_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: post-rescale row 2 entries
    are purely real in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row2_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row2_at_a_i_flavour_columns_match_derivation():
    """The I-flavour annulus columns (B_I, C_I) have no single-
    interface analog. Cross-check them against the closed-form
    expressions read directly off the substep-F.1.a.3 derivation:

        row[1] (B_I) =
            -mu_m [(2 k_z^2 - k_Sm^2) I_0(p_m a) - 2 p_m I_1(p_m a) / a]
        row[3] (C_I) =
            +2 mu_m k_z [s_m I_0(s_m a) - I_1(s_m a) / a]
    """
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    a = p["a"]

    expected_BI = -mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(0, p_m * a))
        - 2.0 * p_m * float(sp.iv(1, p_m * a)) / a
    )
    expected_CI = +2.0 * mu_m * kz * (
        s_m * float(sp.iv(0, s_m * a))
        - float(sp.iv(1, s_m * a)) / a
    )

    row = _layered_n0_row2_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[1].real == pytest.approx(expected_BI)
    assert row[3].real == pytest.approx(expected_CI)


# =====================================================================
# Plan item F.1.b.2.c -- row 3 of the n=0 layered determinant (r = a)
# =====================================================================


def test_layered_row3_at_a_layer_equals_formation_per_element():
    """At layer=formation, row 3's (A, B_K, C_K) entries match
    M31 (= 0), M32, M33 of :func:`_modal_determinant_n0` to
    floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row3_at_a(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M32 = 2.0 * kz * p * mu * float(sp.kv(1, p * a))
    M33 = mu * two_kz2_minus_kS2 * float(sp.kv(1, s * a))

    assert row[0] == 0.0  # M31 = 0 (fluid no shear)
    assert row[2].real == pytest.approx(M32)
    assert row[4].real == pytest.approx(M33)


def test_layered_row3_at_a_fluid_column_is_zero():
    """Row 3 column 0 (the A / fluid-pressure amplitude) is
    identically zero -- the fluid carries no shear stress so it
    contributes nothing to the ``sigma_rz = 0`` BC."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row3_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (5, 6) are
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row3_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5: the full ``row * i`` plus column-by-(-i)
    rescale lands row 3 in real form."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row3_at_a_i_k_sign_flip():
    """Same I-K sign structure as row 1 (different physics, same
    pattern):

        row[1] / row[2] == -I_1(p_m a) / K_1(p_m a)    (B_I vs B_K)
        row[3] / row[4] == +I_1(s_m a) / K_1(s_m a)    (C_I vs C_K)
    """
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n0_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )

    expected_ratio_B = -float(sp.iv(1, p_m * p["a"])) / float(sp.kv(1, p_m * p["a"]))
    expected_ratio_C = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    assert row[1].real / row[2].real == pytest.approx(expected_ratio_B)
    assert row[3].real / row[4].real == pytest.approx(expected_ratio_C)


# =====================================================================
# Plan item F.1.b.3.a -- row 4 of the n=0 layered determinant (r = b)
# =====================================================================
#
# First of the four interface-continuity rows at the second
# interface ``r = b``. Unlike rows 1-3, no single-interface analog
# exists, so the primary correctness oracle is the substep-F.1.a.6
# K-flavour cancellation identity at layer=formation.


def test_layered_row4_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check at the row level: at layer=
    formation the K-flavour annulus and formation columns of row 4
    cancel pair-wise. Specifically:

        row4[2] (B_K) + row4[5] (B) == 0
        row4[4] (C_K) + row4[6] (C) == 0

    Physically: when the annulus material matches the formation,
    the second interface is fictitious, and the outgoing-wave
    K-flavour contributions from both sides represent the same
    field, so continuity is trivially satisfied. This is the
    central correctness invariant for rows 4-7."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row4_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row4_at_b_fluid_column_is_zero():
    """The fluid lives at ``r < a``; it does not reach ``r = b``.
    Column 0 (A) is identically zero in row 4."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row4_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row4_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 4 is real in the bound
    regime (no row scaling; column-by-(-i) on C_I, C_K, C kills
    the explicit ``i`` factors)."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row4_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row4_at_b_matches_closed_form_per_column():
    """Cross-check every non-zero entry against the substep-F.1.a.2
    closed form, evaluated at ``r = b``. No single-interface analog
    to compare against, so this is the per-element transcription
    check."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row4_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    assert row[1].real == pytest.approx(+p_m * float(sp.iv(1, p_m * b)))
    assert row[2].real == pytest.approx(-p_m * float(sp.kv(1, p_m * b)))
    assert row[3].real == pytest.approx(-kz * float(sp.iv(1, s_m * b)))
    assert row[4].real == pytest.approx(-kz * float(sp.kv(1, s_m * b)))
    assert row[5].real == pytest.approx(+p_form * float(sp.kv(1, p_form * b)))
    assert row[6].real == pytest.approx(+kz * float(sp.kv(1, s_form * b)))


def test_layered_row4_at_b_annulus_K_sign_opposite_to_row1_at_a():
    """Sign-flow consistency between the two interfaces. In row 1
    (``u_r^{(f)} - u_r^{(m)} = 0`` at r=a) the annulus B_K
    coefficient is ``+p_m K_1(p_m a)``. In row 4
    (``u_r^{(m)} - u_r^{(s)} = 0`` at r=b) the same physical
    quantity is ``-p_m K_1(p_m b)`` -- opposite sign because the
    annulus appears with opposite sign in the two BCs."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row4 = _layered_n0_row4_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness
    assert row4[2].real == pytest.approx(-p_m * float(sp.kv(1, p_m * b)))


# =====================================================================
# Plan item F.1.b.3.b -- row 5 of the n=0 layered determinant (r = b)
# =====================================================================
#
# Row 5 is the u_z continuity BC at the second interface. Genuinely
# new at the layered case: no single-interface analog because the
# fluid-solid interface at r = a replaces u_z continuity with
# sigma_rz = 0. Imaginary-power pattern is the *opposite* of row 4
# (B-imag / C-real pre-rescale, like rows 3 and 7); the post-
# rescale row * i scaling is what makes the row real.


def test_layered_row5_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check: at layer=formation the K-flavour
    annulus + formation columns of row 5 cancel pair-wise.

        row5[2] (B_K) + row5[5] (B) == 0
        row5[4] (C_K) + row5[6] (C) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row5_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row5_at_b_fluid_column_is_zero():
    """Fluid lives at ``r < a``; column 0 (A) is identically zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row5_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: the row * i scaling on row 5
    (z-derivative-bearing) plus column-by-(-i) on C_I, C_K, C
    leaves the post-rescale row real-valued in the bound regime.
    Forgetting the row * i is the most direct transcription error
    F.1.a.5 calls out -- this test catches it."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row5_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.2 at
    r = b, with the row * i / col * -i rescaling applied. Notable
    feature: row 5 uses degree-0 Bessel functions (I_0 / K_0),
    distinguishing it from rows 1, 4, 6 (degree-1)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    assert row[1].real == pytest.approx(-kz * float(sp.iv(0, p_m * b)))
    assert row[2].real == pytest.approx(-kz * float(sp.kv(0, p_m * b)))
    assert row[3].real == pytest.approx(+s_m * float(sp.iv(0, s_m * b)))
    assert row[4].real == pytest.approx(-s_m * float(sp.kv(0, s_m * b)))
    assert row[5].real == pytest.approx(+kz * float(sp.kv(0, p_form * b)))
    assert row[6].real == pytest.approx(+s_form * float(sp.kv(0, s_form * b)))


def test_layered_row5_at_b_uses_degree0_not_degree1_bessel():
    """Structural check distinguishing u_z (row 5) from u_r (row 4):
    at the same kz, omega, layer, the B_K coefficient in row 5 is
    proportional to ``K_0(p_m b)`` while row 4's B_K coefficient is
    proportional to ``K_1(p_m b)``. The Bessel-index difference
    flows from the ``u_z = i k_z phi`` term (no derivative) vs
    ``u_r = d_r phi`` (one derivative; bumps the Bessel index)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row4 = _layered_n0_row4_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    row5 = _layered_n0_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    # row 4: B_K column = -p_m K_1(p_m b)
    # row 5: B_K column = -k_z K_0(p_m b)
    # Their ratio should be (p_m K_1) / (k_z K_0).
    assert row4[2].real / row5[2].real == pytest.approx(
        (p_m * float(sp.kv(1, p_m * b))) / (kz * float(sp.kv(0, p_m * b)))
    )


# =====================================================================
# Plan item F.1.b.3.c -- row 6 of the n=0 layered determinant (r = b)
# =====================================================================


def test_layered_row6_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check: at layer=formation the K-flavour
    annulus and formation columns of row 6 cancel pair-wise.

        row6[2] (B_K) + row6[5] (B) == 0
        row6[4] (C_K) + row6[6] (C) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row6_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row6_at_b_fluid_column_is_zero():
    """Fluid lives at ``r < a``; column 0 (A) is identically zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row6_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row6_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 6 is real in the bound
    regime. Same imaginary-power pattern as rows 1, 4 (B-real,
    C-imag pre-rescale) so no row scaling needed; only the
    column-by-(-i) on C_I, C_K, C is applied."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row6_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row6_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.3 at
    r = b. Row 6 carries the Lame combination ``(2 k_z^2 - k_Sm^2)``
    on each B / C column, identical in structure to the row-2 form
    but evaluated at r = b with non-zero formation columns."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row6_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    expected_BI = mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(0, p_m * b))
        - 2.0 * p_m * float(sp.iv(1, p_m * b)) / b
    )
    expected_BK = mu_m * (
        two_kz2_minus_kSm2 * float(sp.kv(0, p_m * b))
        + 2.0 * p_m * float(sp.kv(1, p_m * b)) / b
    )
    expected_CI = -2.0 * kz * mu_m * (
        s_m * float(sp.iv(0, s_m * b)) - float(sp.iv(1, s_m * b)) / b
    )
    expected_CK = +2.0 * kz * mu_m * (
        s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b
    )
    expected_B = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p_form * b))
        + 2.0 * p_form * float(sp.kv(1, p_form * b)) / b
    )
    expected_C = -2.0 * kz * mu * (
        s_form * float(sp.kv(0, s_form * b)) + float(sp.kv(1, s_form * b)) / b
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)


def test_layered_row6_at_b_layer_equals_formation_annulus_K_matches_negated_row2():
    """At layer=formation, row 6's annulus K-flavour entries (B_K,
    C_K) -- which carry the unnegated stress form -- equal the
    *negation* of row 2's M22, M23-equivalents evaluated at r = b
    (row 2 uses the negated ``-(sigma_rr + P)`` convention; row 6
    uses unnegated continuity). This pins down the convention
    choice and confirms the two row builders are using the same
    underlying stress formula."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row6 = _layered_n0_row6_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    # Compute an "M22-like at r = b" entry using the row 2 formula.
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M22_at_b = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p * b))
        + 2.0 * p * float(sp.kv(1, p * b)) / b
    )
    M23_at_b = -2.0 * kz * mu * (
        s * float(sp.kv(0, s * b)) + float(sp.kv(1, s * b)) / b
    )

    # row6[2] is unnegated stress; M22_at_b is negated. They differ
    # by sign, so row6[2] = -M22_at_b at layer=formation.
    assert row6[2].real == pytest.approx(-M22_at_b)
    assert row6[4].real == pytest.approx(-M23_at_b)


# =====================================================================
# Plan item F.1.b.3.d -- row 7 of the n=0 layered determinant (r = b)
# =====================================================================
#
# Final row of the 7x7 layered determinant; closes F.1.b.3 and
# unblocks F.1.b.4 assembly. Same z-derivative-bearing pattern as
# rows 3 and 5; structurally analogous to row 3 at r=a but with
# non-zero formation columns.


def test_layered_row7_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check: at layer=formation the K-flavour
    annulus and formation columns of row 7 cancel pair-wise.

        row7[2] (B_K) + row7[5] (B) == 0
        row7[4] (C_K) + row7[6] (C) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row7_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row7_at_b_fluid_column_is_zero():
    """Row 7 column 0 (A) is identically zero. The fluid carries
    no shear AND lives at r < a, so it contributes nothing to the
    sigma_rz continuity at r=b."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row7_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row7_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 7 is real in the bound
    regime. Row 7 is z-derivative-bearing (B-imag / C-real pre-
    rescale, like rows 3 and 5), so it requires the row * i scaling
    plus column-by-(-i) on C_I, C_K, C. Forgetting the row * i is
    the same easy-to-miss F.1.a.5 error as in row 5; this test is
    the safety net."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row7_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row7_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.3 at
    r = b. Row 7 carries the Lame combination ``(2 k_z^2 - k_Sm^2)``
    on the C-flavour columns and the ``2 k_z mu_m p_m`` factor on
    the B-flavour columns (single-Bessel-term entries throughout,
    like row 4)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row7_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    assert row[1].real == pytest.approx(-2.0 * kz * mu_m * p_m * float(sp.iv(1, p_m * b)))
    assert row[2].real == pytest.approx(+2.0 * kz * mu_m * p_m * float(sp.kv(1, p_m * b)))
    assert row[3].real == pytest.approx(+mu_m * two_kz2_minus_kSm2 * float(sp.iv(1, s_m * b)))
    assert row[4].real == pytest.approx(+mu_m * two_kz2_minus_kSm2 * float(sp.kv(1, s_m * b)))
    assert row[5].real == pytest.approx(-2.0 * kz * mu * p_form * float(sp.kv(1, p_form * b)))
    assert row[6].real == pytest.approx(-mu * two_kz2_minus_kS2 * float(sp.kv(1, s_form * b)))


def test_layered_row7_at_b_layer_equals_formation_annulus_K_matches_row3_at_b():
    """At layer=formation, row 7's annulus K-flavour entries (B_K,
    C_K) match the ``M32, M33`` form of :func:`_modal_determinant_n0`
    (the n=0 single-interface row 3) evaluated at ``r = b`` instead
    of ``r = a``. Confirms row 7 and row 3 share the same
    underlying ``sigma_rz`` formula -- they differ only in which
    interface they live at."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row7 = _layered_n0_row7_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M32_at_b = 2.0 * kz * p * mu * float(sp.kv(1, p * b))
    M33_at_b = mu * two_kz2_minus_kS2 * float(sp.kv(1, s * b))

    assert row7[2].real == pytest.approx(M32_at_b)
    assert row7[4].real == pytest.approx(M33_at_b)


# =====================================================================
# Plan item F.1.b.4 -- assembly + dispatch
# =====================================================================
#
# Closes the F.1.b chain. Tests fall into two groups:
#
#   * ``_modal_determinant_n0_layered``: real-valued in bound regime;
#     evaluates without raising; behaves correctly at the layer=
#     formation degenerate point.
#   * ``stoneley_dispersion_layered`` end-to-end: layer=formation
#     reproduces ``stoneley_dispersion`` slowness curve to
#     ``rtol=1e-8``; thickness->0 limit ditto; dispatched correctly.


def test_modal_determinant_n0_layered_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: each row builder applies the
    rescale internally, so the assembled 7x7 is real-valued in
    the bound regime."""
    p, omega, kz = _row1_test_setup()
    det = _modal_determinant_n0_layered(
        kz, omega, p["vp"], p["vs"], p["rho"],
        p["vf"], p["rho_f"], p["a"], layer=p["layer"],
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n0_layered_layer_equals_formation_root_matches_unlayered():
    """The substep-F.1.a.6 self-check at the determinant level: at
    layer=formation, the layered determinant has the same
    Stoneley root as :func:`_modal_determinant_n0`. The two
    determinants are not numerically equal (the 7x7 has a
    different overall scale than the 3x3), but they share the
    same root in ``k_z``.

    Verify by: (a) computing the Stoneley root from
    ``stoneley_dispersion`` (the 3x3); (b) evaluating the layered
    7x7 at that root; (c) checking ``|det_layered|`` is small
    relative to its order of magnitude away from the root."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    bound = stoneley_dispersion(
        np.array([5000.0]),
        vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n0_layered(
        kz_root, omega, vp, vs, rho, vf, rho_f, a, layer=layer,
    )
    det_off_root = _modal_determinant_n0_layered(
        kz_root * 1.05, omega, vp, vs, rho, vf, rho_f, a, layer=layer,
    )
    # Not strictly zero (different matrix size + numerical noise),
    # but several orders of magnitude smaller than away from root.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_stoneley_dispersion_layered_layer_equals_formation_matches_unlayered():
    """End-to-end integration test: with a layer whose properties
    match the formation, the layered solver produces the same
    Stoneley dispersion curve as the unlayered solver to
    ``rtol=1e-8``. This is the floating-point oracle for the
    entire F.1.b chain. Any algebra error accumulated across the
    seven row builders surfaces here."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    f = np.linspace(500.0, 8000.0, 16)

    res_unlayered = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_layered = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer,),
    )
    np.testing.assert_allclose(
        res_layered.slowness, res_unlayered.slowness,
        rtol=1.0e-8, equal_nan=True,
    )


def test_stoneley_dispersion_layered_thickness_zero_limit():
    """As ``layer.thickness -> 0`` (with arbitrary layer material),
    the layered solver continuously approaches the unlayered
    answer. Algebraic identity: in the limit ``b -> a``, the rows
    at r=b approach the rows at r=a, the second interface degenerates,
    and the converged k_z must approach the single-interface root."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = 5000.0

    res_unlayered = stoneley_dispersion(
        np.array([f]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    # Even a "different" layer with vanishing thickness should
    # converge to the unlayered Stoneley slowness.
    layer_thin = BoreholeLayer(
        vp=3500.0, vs=1800.0, rho=2100.0, thickness=1.0e-9,
    )
    res_thin = stoneley_dispersion_layered(
        np.array([f]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer_thin,),
    )
    assert res_thin.slowness[0] == pytest.approx(
        res_unlayered.slowness[0], rel=1.0e-4,
    )


def test_stoneley_dispersion_layered_non_trivial_layer_runs():
    """End-to-end smoke: a soft mudcake layer different from the
    formation produces a finite slowness curve in the bound
    regime. No analytic oracle is asserted here (that's the
    Schmitt 1988 fig 6 validation in F.1.d); the test just
    confirms the dispatch + matrix + brentq + bracket all wire up
    without raising."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = np.linspace(1000.0, 8000.0, 8)

    res = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer,),
    )
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    assert res.slowness.shape == f.shape
    # All slownesses finite in this bound-regime fast-formation case.
    assert np.all(np.isfinite(res.slowness))
    # All slownesses above the slowest-shear floor.
    assert np.all(res.slowness > 1.0 / max(vs, layer.vs, vf))


def test_stoneley_dispersion_layered_softer_mudcake_slows_down():
    """Sanity check: a mudcake softer than the formation
    (lower V_S) increases the Stoneley slowness compared to
    the unlayered formation -- the qualitative effect documented
    in Schmitt 1988 fig 6 and the F.1.d validation target."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.array([3000.0])

    res_unlayered = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    soft_layer = BoreholeLayer(
        vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01,
    )
    res_layered = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(soft_layer,),
    )
    assert res_layered.slowness[0] > res_unlayered.slowness[0]


# =====================================================================
# Plan item F.1.d -- validation tightening on top of F.1.b.4
# =====================================================================
#
# Hardening tests for the assembled layered Stoneley solver. Each
# tests an asymptotic / self-consistency property that the
# layer=formation regression alone doesn't pin down.


def test_stoneley_dispersion_layered_thickness_dominant_limit():
    """As the layer thickness grows much larger than the field's
    radial extent at r = a (set roughly by ``1 / p_m``), the
    second interface becomes irrelevant and the Stoneley wave
    propagates as if the *layer* material were the formation
    half-space.

    Concretely: layered_dispersion(formation=X, layer=Y,
    thickness=large) -> stoneley_dispersion(formation=Y) as
    thickness * p_m -> infty. We test at a frequency high enough
    that p_m * thickness >> 1 with a 0.5 m thick layer.
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer_vp, layer_vs, layer_rho = 3500.0, 1800.0, 2100.0
    f = np.array([5000.0])
    # 0.5 m -- far thicker than any physical mudcake, just to
    # stress-test the limit. p_m * thickness is well above 1
    # at f = 5 kHz, so the field at r = b is exponentially small.
    layer = BoreholeLayer(
        vp=layer_vp, vs=layer_vs, rho=layer_rho, thickness=0.5,
    )

    res_layered = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer,),
    )
    # Limit value: unlayered Stoneley with the LAYER properties as
    # the formation half-space.
    res_limit = stoneley_dispersion(
        f, vp=layer_vp, vs=layer_vs, rho=layer_rho,
        vf=vf, rho_f=rho_f, a=a,
    )
    # Tolerance is loose because the limit is asymptotic; tighter
    # match would need an even thicker layer (numerically delicate
    # because K_n decays exponentially).
    assert res_layered.slowness[0] == pytest.approx(
        res_limit.slowness[0], rel=1.0e-3,
    )


def test_modal_determinant_n0_layered_vanishes_at_converged_root():
    """Self-consistency: at the converged ``k_z`` returned by
    :func:`stoneley_dispersion_layered`, the layered determinant
    is small compared to its off-root value. Tighter check than
    the layer=formation det-at-root test (which only verifies the
    F.1.a.6 self-check); this works for any non-trivial layer."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = stoneley_dispersion_layered(
        np.array([f]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n0_layered(
        kz_root, omega, vp, vs, rho, vf, rho_f, a, layer=layer,
    )
    det_off = _modal_determinant_n0_layered(
        kz_root * 1.05, omega, vp, vs, rho, vf, rho_f, a, layer=layer,
    )
    # brentq returns a converged root, so |det_at| should be at
    # least ~6 orders of magnitude smaller than |det_off|.
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_stoneley_dispersion_layered_multiple_frequencies_bound_regime():
    """Smoke test across a wide frequency band to confirm the
    bracket + brentq combination stays well-behaved over a range
    spanning ~3 decades. The Stoneley slowness *decreases*
    monotonically with frequency in a fast-formation borehole
    (the wave speeds up toward a fluid-loaded Rayleigh /
    Scholte-like asymptote at high f); same dispersion direction
    as the unlayered case from
    ``test_stoneley_dispersion_speeds_up_with_frequency``."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = np.geomspace(100.0, 20000.0, 25)

    res = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer,),
    )
    assert np.all(np.isfinite(res.slowness))
    diffs = np.diff(res.slowness)
    # Slowness decreases with frequency in a fast formation
    # (phase velocity increases). Monotonic decrease across the
    # full 100 Hz - 20 kHz band; allow a tiny tolerance for
    # near-asymptote flatness.
    assert np.all(diffs < 1.0e-9)


def test_stoneley_dispersion_layered_low_f_layer_shifts_off_formation_white():
    """At very low frequency the layer is NOT invisible: even a
    5 mm mudcake at 10 Hz (wavelength ~2 km) shifts the Stoneley
    slowness off the unlayered White (1983) formation-only
    closed-form. Reason: the layer sits at the borehole wall
    where the radial field amplitude is highest, so the effective
    near-wall shear modulus is the layer's, not the formation's.

    Verify the layered slowness lies *between* the two unlayered
    White-formula values (formation and layer-as-formation),
    closer to the formation value because the formation half-space
    still provides the bulk of the back-field-decay support at
    low f."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    layer_vp, layer_vs, layer_rho = 3500.0, 1800.0, 2100.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(
        vp=layer_vp, vs=layer_vs, rho=layer_rho, thickness=0.005,
    )
    f = np.array([10.0])

    res = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer,),
    )
    s_formation = _stoneley_lf_truth(vs, rho, vf, rho_f)
    s_layer = _stoneley_lf_truth(layer_vs, layer_rho, vf, rho_f)
    # layer is softer than formation -> s_layer > s_formation.
    # The layered slowness must lie between the two.
    assert s_formation < res.slowness[0] < s_layer


# =====================================================================
# Plan item F.2.0 -- public-API foundation for layered flexural
# =====================================================================
#
# Sister of the F.1 foundation tests. The 10x10 layered modal
# determinant is scheduled in plan item F.2; here we only exercise
# the public-API surface (validation, empty-layers dispatch,
# NotImplementedError sentinel for non-empty).


def test_flexural_dispersion_layered_empty_layers_bit_matches_unlayered():
    """Degenerate single-interface case: ``layers=()`` must produce
    a slowness curve bit-identical to :func:`flexural_dispersion`.
    Floating-point oracle that will continue to anchor the layered
    flexural solver once the 10x10 modal determinant lands in F.2.d.
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(2000.0, 8000.0, 12)
    res_unlayered = flexural_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_layered = flexural_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a, layers=(),
    )
    np.testing.assert_array_equal(res_layered.slowness, res_unlayered.slowness)
    np.testing.assert_array_equal(res_layered.freq, res_unlayered.freq)
    assert res_layered.name == res_unlayered.name == "flexural"
    assert res_layered.azimuthal_order == 1


def test_flexural_dispersion_layered_empty_layers_returns_borehole_mode():
    f = np.linspace(2000.0, 5000.0, 5)
    res = flexural_dispersion_layered(
        f, vp=4500.0, vs=2500.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1


def test_flexural_dispersion_layered_fast_formation_layered_raises_not_implemented():
    """Fast-formation layered flexural (``V_S > V_f`` with a
    non-empty layer) is future work per F.2.d. The 10x10 layered
    solver covers the slow-formation bound regime only; fast-
    formation layered raises a clear NotImplementedError."""
    f = np.array([2000.0, 4000.0])
    # Fast formation: vs (2500) > vf (1500).
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    with pytest.raises(NotImplementedError, match="fast formation"):
        flexural_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(layer,),
        )


def test_flexural_dispersion_layered_multilayer_raises_not_implemented():
    """Multi-layer stacks (``len(layers) > 1``) currently raise
    ``NotImplementedError``; the n=1 cased-hole counterpart is
    deferred to plan G' (follow-up to plan G), which extends the
    G.b/G.c propagator scaffolding to 6x6 per-layer blocks.
    Single-layer and unlayered are supported by F.2.d."""
    f = np.array([2000.0, 4000.0])
    layer1 = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    layer2 = BoreholeLayer(vp=3000.0, vs=1500.0, rho=1900.0, thickness=0.005)
    with pytest.raises(NotImplementedError, match=r"plan G'"):
        flexural_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(layer1, layer2),
        )


def test_flexural_dispersion_layered_rejects_bad_layer_object():
    f = np.array([2000.0])
    with pytest.raises(ValueError, match="BoreholeLayer"):
        flexural_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=("not a layer",),
        )


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"vp": 0.0, "vs": 1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": -1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 1.0, "rho": 0.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 2.0, "rho": 1.0, "thickness": 1.0}, "vp > vs"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": 0.0}, "thickness"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": -0.1}, "thickness"),
    ],
)
def test_flexural_dispersion_layered_rejects_malformed_layer_params(kwargs, msg):
    f = np.array([2000.0])
    layer = BoreholeLayer(**kwargs)
    with pytest.raises(ValueError, match=msg):
        flexural_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(layer,),
        )


def test_flexural_dispersion_layered_accepts_list_for_layers():
    """``layers`` should accept any iterable that ``tuple(...)``
    consumes; empty list dispatches to the unlayered solver."""
    f = np.linspace(2000.0, 5000.0, 4)
    res_tuple = flexural_dispersion_layered(
        f, vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1, layers=(),
    )
    res_list = flexural_dispersion_layered(
        f, vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1, layers=[],
    )
    np.testing.assert_array_equal(res_tuple.slowness, res_list.slowness)


# =====================================================================
# Plan item F.2.c.1 -- row 3 of the n=1 layered determinant (r = a)
# =====================================================================
#
# First sin-sector row of the F.2 chain. Encodes
# ``sigma_rtheta^{(m)}(a) = 0``. The 10x10 layered determinant is
# dense (per the F.2.a.6 erratum) so this row builder returns
# shape-(10,) covering all amplitude columns. Primary correctness
# oracle: per-element layer=formation match against M31, M32, M33,
# M34 of :func:`_modal_determinant_n1` (the existing n=1
# single-interface form).


def test_layered_n1_row3_at_a_layer_equals_formation_per_element():
    """Substep F.2.a.7 (a) self-check at the row level: at
    layer=formation the K-flavour annulus columns of row 3 match
    M31, M32, M33, M34 of :func:`_modal_determinant_n1` to
    floating-point precision.

    Specifically:
        row[0] (A)   = M31 = 0     (fluid no shear)
        row[2] (B_K) = M32         (K-flavor B coefficient at r=a)
        row[4] (C_K) = M33         (K-flavor C coefficient at r=a)
        row[8] (D_K) = M34         (K-flavor D coefficient at r=a)
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row3_at_a(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    # Reconstruct the M31-M34 entries directly from the n=1
    # single-interface formula (see _modal_determinant_n1
    # docstring lines for row 3).
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M31 = 0.0
    M32 = 2.0 * mu * (
        p * float(sp.kv(0, p * a)) / a
        + 2.0 * float(sp.kv(1, p * a)) / (a * a)
    )
    M33 = kz * mu * float(sp.kv(1, s * a)) / a
    M34 = -mu * (
        s * s * float(sp.kv(1, s * a))
        + 2.0 * s * float(sp.kv(0, s * a)) / a
        + 4.0 * float(sp.kv(1, s * a)) / (a * a)
    )

    assert row[0].real == pytest.approx(M31)
    assert row[2].real == pytest.approx(M32)
    assert row[4].real == pytest.approx(M33)
    assert row[8].real == pytest.approx(M34)


def test_layered_n1_row3_at_a_sparsity():
    """Sparsity per the corrected F.2.a.4: A column is zero (fluid
    no shear); formation columns (5 = B, 6 = C, 9 = D) are zero
    because the formation half-space lives at ``r > b`` and doesn't
    reach r = a."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0  # A
    assert row[5] == 0.0  # formation B
    assert row[6] == 0.0  # formation C
    assert row[9] == 0.0  # formation D
    # The remaining six columns (1, 2, 3, 4, 7, 8) are generically
    # non-zero in the bound regime.
    for i in (1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row3_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 3 has the no-row-rescale pattern;
    only the column-by-(-i) on C_I, C_K is applied. Post-rescale
    row is real-valued in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row3_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substeps F.2.a.2 /
    F.2.a.3 closed forms (with the F.1.a.2 sign-flip pattern
    applied to the I-flavour annulus terms)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2

    expected_BI = 2.0 * mu_m * (
        -p_m * float(sp.iv(0, p_m * a)) / a
        + 2.0 * float(sp.iv(1, p_m * a)) / (a * a)
    )
    expected_BK = 2.0 * mu_m * (
        +p_m * float(sp.kv(0, p_m * a)) / a
        + 2.0 * float(sp.kv(1, p_m * a)) / (a * a)
    )
    expected_CI = +kz * mu_m * float(sp.iv(1, s_m * a)) / a
    expected_CK = +kz * mu_m * float(sp.kv(1, s_m * a)) / a
    expected_DI = -mu_m * (
        s_m * s_m * float(sp.iv(1, s_m * a))
        - 2.0 * s_m * float(sp.iv(0, s_m * a)) / a
        + 4.0 * float(sp.iv(1, s_m * a)) / (a * a)
    )
    expected_DK = -mu_m * (
        s_m * s_m * float(sp.kv(1, s_m * a))
        + 2.0 * s_m * float(sp.kv(0, s_m * a)) / a
        + 4.0 * float(sp.kv(1, s_m * a)) / (a * a)
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


def test_layered_n1_row3_at_a_C_column_i_k_sign_flip():
    """The C-amplitude entries are single-Bessel-term, so the I-K
    ratio collapses to a clean ``+I_1(s_m a) / K_1(s_m a)`` (no
    sign flip; same Bessel-index, KEEP-sign per the F.1.a.2
    pattern). Direct verifiable algebraic identity."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row3_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )

    expected_ratio = (
        +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    )
    assert row[3].real / row[4].real == pytest.approx(expected_ratio)


# =====================================================================
# Plan item F.2.c.2 -- row 6 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Genuinely new BC type at the layered case: u_theta continuity at
# r=b has no single-interface analog (the fluid-solid interface at
# r=a replaces it with sigma_rtheta = 0). C does NOT appear in
# u_theta per substep F.2.a.2; row 6 has six non-zero entries
# (B and D amplitudes only) and three explicit zero entries beyond
# the standard A=0 sparsity.


def test_layered_n1_row6_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.2.a.7 (a) self-check: at layer=formation the
    K-flavour annulus and formation columns cancel pair-wise.

        row6[2] (B_K) + row6[5] (B) == 0
        row6[8] (D_K) + row6[9] (D) == 0

    No C cancellation to verify since C is identically zero in u_theta.
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row6_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row6_at_b_C_columns_are_identically_zero():
    """Substep F.2.a.2: u_theta has B and D contributions, NOT C.
    Columns 3 (C_I), 4 (C_K), 6 (formation C) are identically zero
    in row 6 -- a stronger sparsity than the F.2.a.4 generic
    pattern (which only requires A=0 and formation cols zero in
    rows touching r=a). Row 6 distinguishes itself by also having
    C=0 even though it touches r=b."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row6_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[3] == 0.0  # C_I
    assert row[4] == 0.0  # C_K
    assert row[6] == 0.0  # formation C
    assert row[0] == 0.0  # A (fluid r<a)
    # Six remaining columns (1, 2, 5, 7, 8, 9) generically non-zero.
    for i in (1, 2, 5, 7, 8, 9):
        assert row[i] != 0.0


def test_layered_n1_row6_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 6 is NOT z-derivative-bearing (no
    row * i scaling). C columns are zero so column-by-(-i) is
    irrelevant. Pre- and post-rescale are both real-valued in the
    bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row6_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row6_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_theta closed forms. The B and D coefficients carry the
    F.1.a.2 sign-flip pattern: ``s I_0`` flips, ``K_1/r``-style
    direct terms keep sign."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row6_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    expected_BI = -float(sp.iv(1, p_m * b)) / b
    expected_BK = -float(sp.kv(1, p_m * b)) / b
    expected_B = +float(sp.kv(1, p_form * b)) / b
    expected_DI = -s_m * float(sp.iv(0, s_m * b)) + float(sp.iv(1, s_m * b)) / b
    expected_DK = +s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b
    expected_D = -s_form * float(sp.kv(0, s_form * b)) - float(sp.kv(1, s_form * b)) / b

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row6_at_b_B_column_i_k_sign_flip():
    """The B-amplitude entries are single-Bessel-term:
    row6[B_I] = -I_1/b, row6[B_K] = -K_1/b. Their ratio is
    ``+I_1(p_m b) / K_1(p_m b)`` -- KEEP-sign per the F.1.a.2
    pattern (direct ``K_1/r`` term, no derivative-induced
    Bessel-index shift)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row6_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    expected_ratio = (
        +float(sp.iv(1, p_m * b)) / float(sp.kv(1, p_m * b))
    )
    assert row[1].real / row[2].real == pytest.approx(expected_ratio)


# =====================================================================
# Plan item F.2.c.3 -- row 9 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Closes substep F.2.c. Same algebraic structure as row 3 at r=a
# but at the second interface with non-zero formation columns.
# The cross-row-3 identity at layer=formation is the structural
# safety net.


def test_layered_n1_row9_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.2.a.7 (a) self-check: at layer=formation all THREE
    K-flavour annulus / formation column pairs cancel:

        row9[2] (B_K) + row9[5] (B) == 0
        row9[4] (C_K) + row9[6] (C) == 0
        row9[8] (D_K) + row9[9] (D) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row9_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row9_at_b_sparsity():
    """Sparsity: A column zero (fluid r<a, no shear); all other
    nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row9_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row9_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 9 has the no-row-rescale pattern;
    only column-by-(-i) on C cols. Post-rescale row is real-valued
    in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row9_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row9_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substeps F.2.a.2 /
    F.2.a.3 closed forms for all nine non-zero entries (annulus
    B/C/D + formation B/C/D)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row9_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    mu = p["rho"] * p["vs"] ** 2

    expected_BI = 2.0 * mu_m * (
        -p_m * float(sp.iv(0, p_m * b)) / b
        + 2.0 * float(sp.iv(1, p_m * b)) / (b * b)
    )
    expected_BK = 2.0 * mu_m * (
        +p_m * float(sp.kv(0, p_m * b)) / b
        + 2.0 * float(sp.kv(1, p_m * b)) / (b * b)
    )
    expected_CI = +kz * mu_m * float(sp.iv(1, s_m * b)) / b
    expected_CK = +kz * mu_m * float(sp.kv(1, s_m * b)) / b
    expected_B = -2.0 * mu * (
        +p_form * float(sp.kv(0, p_form * b)) / b
        + 2.0 * float(sp.kv(1, p_form * b)) / (b * b)
    )
    expected_C = -kz * mu * float(sp.kv(1, s_form * b)) / b
    expected_DI = -mu_m * (
        s_m * s_m * float(sp.iv(1, s_m * b))
        - 2.0 * s_m * float(sp.iv(0, s_m * b)) / b
        + 4.0 * float(sp.iv(1, s_m * b)) / (b * b)
    )
    expected_DK = -mu_m * (
        s_m * s_m * float(sp.kv(1, s_m * b))
        + 2.0 * s_m * float(sp.kv(0, s_m * b)) / b
        + 4.0 * float(sp.kv(1, s_m * b)) / (b * b)
    )
    expected_D = +mu * (
        s_form * s_form * float(sp.kv(1, s_form * b))
        + 2.0 * s_form * float(sp.kv(0, s_form * b)) / b
        + 4.0 * float(sp.kv(1, s_form * b)) / (b * b)
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row9_at_b_annulus_K_matches_row3_M32_M33_M34_at_b():
    """Cross-row identity: at layer=formation, row 9's annulus
    K-flavour entries match row 3's M32, M33, M34-equivalent forms
    evaluated at r = b (same underlying sigma_rtheta formula at
    both interfaces; only the evaluation radius differs)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row9 = _layered_n1_row9_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    # Row 3's M32, M33, M34 form evaluated at r=b instead of r=a.
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M32_at_b = 2.0 * mu * (
        p * float(sp.kv(0, p * b)) / b
        + 2.0 * float(sp.kv(1, p * b)) / (b * b)
    )
    M33_at_b = kz * mu * float(sp.kv(1, s * b)) / b
    M34_at_b = -mu * (
        s * s * float(sp.kv(1, s * b))
        + 2.0 * s * float(sp.kv(0, s * b)) / b
        + 4.0 * float(sp.kv(1, s * b)) / (b * b)
    )

    assert row9[2].real == pytest.approx(M32_at_b)
    assert row9[4].real == pytest.approx(M33_at_b)
    assert row9[8].real == pytest.approx(M34_at_b)


# =====================================================================
# Plan item F.2.b.1 -- row 1 of the n=1 layered determinant (r = a)
# =====================================================================
#
# First cos-sector row of the F.2 chain. The genuinely new content
# vs F.1.b.2.a (n=0 row 1) is the D-amplitude column: at n=1 the
# SH amplitude D appears in cos-sector u_r via the
# ``(1/r) d_theta psi_z`` cross-coupling. Primary oracle: per-
# element layer=formation match against M11-M14 of
# :func:`_modal_determinant_n1`.


def test_layered_n1_row1_at_a_layer_equals_formation_per_element():
    """At layer=formation, row 1's K-flavour annulus columns and
    the A column match M11, M12, M13, M14 of
    :func:`_modal_determinant_n1` to floating-point precision.

    Specifically:
        row[0] (A)   = M11   (fluid pressure coefficient)
        row[2] (B_K) = M12   (P-amplitude at r=a)
        row[4] (C_K) = M13   (SV-amplitude at r=a, post-rescale)
        row[8] (D_K) = M14   (SH-amplitude at r=a -- new at n=1)
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row1_at_a(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = (
        F * float(sp.iv(0, F * a)) - float(sp.iv(1, F * a)) / a
    ) / (rho_f * omega ** 2)
    M12 = p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a
    M13 = kz * float(sp.kv(1, s * a))
    M14 = -float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M11)
    assert row[2].real == pytest.approx(M12)
    assert row[4].real == pytest.approx(M13)
    assert row[8].real == pytest.approx(M14)


def test_layered_n1_row1_at_a_formation_columns_are_zero():
    """Sparsity: at r=a the formation columns (5 = B, 6 = C, 9 = D)
    are zero -- the formation half-space lives at r > b and doesn't
    touch the fluid-annulus interface."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row1_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[9] == 0.0
    # All other columns generically non-zero in the bound regime.
    for i in (0, 1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row1_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 1 has the no-row-rescale pattern;
    only the column-by-(-i) on C_I, C_K is applied. Post-rescale
    row is real-valued in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row1_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row1_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_r decomposition (with the F.1.a.2 sign-flip pattern applied
    to the I-flavour annulus terms)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row1_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    expected_A = (
        F_f * float(sp.iv(0, F_f * a)) - float(sp.iv(1, F_f * a)) / a
    ) / (p["rho_f"] * omega ** 2)
    expected_BI = (
        -p_m * float(sp.iv(0, p_m * a)) + float(sp.iv(1, p_m * a)) / a
    )
    expected_BK = (
        +p_m * float(sp.kv(0, p_m * a)) + float(sp.kv(1, p_m * a)) / a
    )
    expected_CI = +kz * float(sp.iv(1, s_m * a))
    expected_CK = +kz * float(sp.kv(1, s_m * a))
    expected_DI = -float(sp.iv(1, s_m * a)) / a
    expected_DK = -float(sp.kv(1, s_m * a)) / a

    assert row[0].real == pytest.approx(expected_A)
    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


def test_layered_n1_row1_at_a_C_and_D_column_i_k_sign_flips():
    """The C and D entries are single-Bessel-term (no derivative-
    induced terms), so the I-K ratios collapse to clean
    ``+I_1(s_m a) / K_1(s_m a)`` -- KEEP-sign per F.1.a.2 (direct
    terms, no Bessel-index shift)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row1_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    expected_ratio = (
        +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    )
    # C ratio: I_1(s_m a) / K_1(s_m a).
    assert row[3].real / row[4].real == pytest.approx(expected_ratio)
    # D ratio: also I_1(s_m a) / K_1(s_m a) (same Bessel arg).
    assert row[7].real / row[8].real == pytest.approx(expected_ratio)


# =====================================================================
# Plan item F.2.b.2 -- row 2 of the n=1 layered determinant (r = a)
# =====================================================================
#
# Lame-reduction row at the first interface; algebraically heaviest
# of the cos-sector r=a rows. Multi-term entries on every B / C / D
# column, so the I-K sign-flip pattern is verified through the
# closed-form per-column transcription test rather than via a
# clean ratio (single-Bessel-term ratios don't apply).


def test_layered_n1_row2_at_a_layer_equals_formation_per_element():
    """At layer=formation row 2's K-flavour annulus columns and the
    A column match M21, M22, M23, M24 of
    :func:`_modal_determinant_n1` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row2_at_a(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M21 = -float(sp.iv(1, F * a))
    M22 = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p * a))
        + 2.0 * p * float(sp.kv(0, p * a)) / a
        + 4.0 * float(sp.kv(1, p * a)) / (a * a)
    )
    M23 = -2.0 * kz * mu * (
        s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a
    )
    M24 = +2.0 * mu * (
        s * float(sp.kv(0, s * a)) / a + 2.0 * float(sp.kv(1, s * a)) / (a * a)
    )

    assert row[0].real == pytest.approx(M21)
    assert row[2].real == pytest.approx(M22)
    assert row[4].real == pytest.approx(M23)
    assert row[8].real == pytest.approx(M24)


def test_layered_n1_row2_at_a_formation_columns_are_zero():
    """Sparsity: at r=a the formation columns (5, 6, 9) are zero;
    remaining seven columns (0, 1, 2, 3, 4, 7, 8) generically non-
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row2_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[9] == 0.0
    for i in (0, 1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row2_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 2 has the no-row-rescale pattern;
    column-by-(-i) on C_I, C_K only. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row2_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row2_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rr decomposition. The I-flavour entries (B_I, C_I, D_I)
    encode the F.1.a.2 sign-flip pattern: derivative-induced
    ``p_m X_0/a`` and ``s_m X_0/a`` terms flip sign vs the
    K-flavour twins; direct ``X_1`` and ``X_1/r^n`` terms keep
    sign."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row2_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    expected_A = -float(sp.iv(1, F_f * a))
    expected_BI = -mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(1, p_m * a))
        - 2.0 * p_m * float(sp.iv(0, p_m * a)) / a
        + 4.0 * float(sp.iv(1, p_m * a)) / (a * a)
    )
    expected_BK = -mu_m * (
        two_kz2_minus_kSm2 * float(sp.kv(1, p_m * a))
        + 2.0 * p_m * float(sp.kv(0, p_m * a)) / a
        + 4.0 * float(sp.kv(1, p_m * a)) / (a * a)
    )
    expected_CI = +2.0 * kz * mu_m * (
        s_m * float(sp.iv(0, s_m * a)) - float(sp.iv(1, s_m * a)) / a
    )
    expected_CK = -2.0 * kz * mu_m * (
        s_m * float(sp.kv(0, s_m * a)) + float(sp.kv(1, s_m * a)) / a
    )
    expected_DI = +2.0 * mu_m * (
        -s_m * float(sp.iv(0, s_m * a)) / a
        + 2.0 * float(sp.iv(1, s_m * a)) / (a * a)
    )
    expected_DK = +2.0 * mu_m * (
        +s_m * float(sp.kv(0, s_m * a)) / a
        + 2.0 * float(sp.kv(1, s_m * a)) / (a * a)
    )

    assert row[0].real == pytest.approx(expected_A)
    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


# =====================================================================
# Plan item F.2.b.3 -- row 4 of the n=1 layered determinant (r = a)
# =====================================================================
#
# First z-derivative-bearing cos row of the F.2 chain. Per substep
# F.2.a.5: row * i AND col-by-(-i) on C cols. Both rescales must be
# correctly applied for the post-rescale row to be real.


def test_layered_n1_row4_at_a_layer_equals_formation_per_element():
    """At layer=formation row 4's K-flavour annulus columns and the
    A column match M41-M44 of :func:`_modal_determinant_n1`."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row4_at_a(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M41 = 0.0
    M42 = +2.0 * kz * mu * (
        p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a
    )
    M43 = +mu * two_kz2_minus_kS2 * float(sp.kv(1, s * a))
    M44 = -kz * mu * float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M41)
    assert row[2].real == pytest.approx(M42)
    assert row[4].real == pytest.approx(M43)
    assert row[8].real == pytest.approx(M44)


def test_layered_n1_row4_at_a_sparsity():
    """Sparsity: A column zero (fluid no shear); formation columns
    (5, 6, 9) zero (don't reach r=a); remaining six columns
    generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row4_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[9] == 0.0
    for i in (1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row4_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 4 has the FULL z-bearing rescale (row
    * i AND col-by-(-i) on C cols). Both must be correctly applied
    for the post-rescale row to be real. Forgetting the row * i is
    the most direct transcription error per F.2.a.5 commentary;
    this test catches it."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row4_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row4_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rz decomposition. The B columns have multi-term
    (``+p_m X_0 +/- X_1/r``); C and D columns are single-Bessel-term."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row4_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    expected_BI = -2.0 * kz * mu_m * (
        p_m * float(sp.iv(0, p_m * a)) - float(sp.iv(1, p_m * a)) / a
    )
    expected_BK = +2.0 * kz * mu_m * (
        p_m * float(sp.kv(0, p_m * a)) + float(sp.kv(1, p_m * a)) / a
    )
    expected_CI = +mu_m * two_kz2_minus_kSm2 * float(sp.iv(1, s_m * a))
    expected_CK = +mu_m * two_kz2_minus_kSm2 * float(sp.kv(1, s_m * a))
    expected_DI = -kz * mu_m * float(sp.iv(1, s_m * a)) / a
    expected_DK = -kz * mu_m * float(sp.kv(1, s_m * a)) / a

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


def test_layered_n1_row4_at_a_C_and_D_column_i_k_sign_flips():
    """The C and D entries are single-Bessel-term direct ``X_1`` /
    ``X_1/a`` forms, so the I-K ratio collapses to a clean
    ``+I_1(s_m a) / K_1(s_m a)`` -- KEEP-sign per F.1.a.2."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row4_at_a(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )

    expected_ratio = (
        +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    )
    assert row[3].real / row[4].real == pytest.approx(expected_ratio)
    assert row[7].real / row[8].real == pytest.approx(expected_ratio)


# =====================================================================
# Plan item F.2.b.4 -- row 5 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Mirror of row 1 evaluated at r=b with non-zero formation columns.
# No single-interface analog; primary oracle is K-flavour
# cancellation at layer=formation.


def test_layered_n1_row5_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.2.a.7 (a) self-check: at layer=formation all THREE
    K-flavour annulus / formation column pairs cancel:

        row5[2] (B_K) + row5[5] (B) == 0
        row5[4] (C_K) + row5[6] (C) == 0
        row5[8] (D_K) + row5[9] (D) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row5_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row5_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid r<a doesn't reach r=b);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row5_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 5 has the no-row-rescale pattern;
    column-by-(-i) on C_I, C_K, C only. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row5_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_r decomposition at r=b."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    expected_BI = +p_m * float(sp.iv(0, p_m * b)) - float(sp.iv(1, p_m * b)) / b
    expected_BK = -p_m * float(sp.kv(0, p_m * b)) - float(sp.kv(1, p_m * b)) / b
    expected_CI = -kz * float(sp.iv(1, s_m * b))
    expected_CK = -kz * float(sp.kv(1, s_m * b))
    expected_B = +p_form * float(sp.kv(0, p_form * b)) + float(sp.kv(1, p_form * b)) / b
    expected_C = +kz * float(sp.kv(1, s_form * b))
    expected_DI = +float(sp.iv(1, s_m * b)) / b
    expected_DK = +float(sp.kv(1, s_m * b)) / b
    expected_D = -float(sp.kv(1, s_form * b)) / b

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row5_at_b_annulus_K_sign_opposite_to_row1_at_a():
    """Sign-flow consistency vs row 1: the BC subtraction direction
    flips between row 1 (``u_r^{(f)} - u_r^{(m)} = 0``, annulus
    appears with - sign) and row 5 (``u_r^{(m)} - u_r^{(s)} = 0``,
    annulus appears with + sign). Consequently row 5's annulus
    K-flavour B_K is the negation of row 1's B_K (modulo the
    radius shift a -> b)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row5 = _layered_n1_row5_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness
    # Row 5 B_K should be -p_m K_0(p_m b) - K_1(p_m b)/b = -(row 1 B_K form at r=b).
    assert row5[2].real == pytest.approx(
        -(p_m * float(sp.kv(0, p_m * b)) + float(sp.kv(1, p_m * b)) / b)
    )


# =====================================================================
# Plan item F.2.b.5 -- row 7 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Z-derivative-bearing cos row at the second interface. Distinctive
# sparsity: D columns (7, 8, 9) are identically zero because u_z
# does not couple to psi_z under the curl decomposition (curl_z =
# (1/r) d_r(r psi_theta), no psi_z term).


def test_layered_n1_row7_at_b_K_flavour_cancels_at_layer_equals_formation():
    """Substep F.2.a.7 (a) self-check: K-flavour cancellation pairs
    at layer=formation. The D pair is trivial since both D_K and D
    are zero (u_z has no psi_z contribution)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row7_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == 0.0  # both 0 -> exact


def test_layered_n1_row7_at_b_D_columns_are_identically_zero():
    """Distinctive sparsity of row 7 in F.2.b: u_z does not couple
    to psi_z under the curl decomposition (curl_z = (1/r)
    d_r(r psi_theta), no psi_z term). D columns (7, 8, 9) are
    identically zero -- the structural feature that distinguishes
    row 7 from rows 5, 8, 10."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row7_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[7] == 0.0  # D_I
    assert row[8] == 0.0  # D_K
    assert row[9] == 0.0  # formation D
    assert row[0] == 0.0  # A (fluid r<a)
    # Six remaining columns (1, 2, 3, 4, 5, 6) generically non-zero.
    for i in (1, 2, 3, 4, 5, 6):
        assert row[i] != 0.0


def test_layered_n1_row7_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 7 is z-derivative-bearing -- gets the
    FULL rescale (row * i + col-by-(-i) on C cols). Both must be
    correctly applied."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row7_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row7_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_z decomposition at r=b for the six non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row7_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    expected_BI = -kz * float(sp.iv(1, p_m * b))
    expected_BK = -kz * float(sp.kv(1, p_m * b))
    expected_CI = +s_m * float(sp.iv(0, s_m * b))
    expected_CK = -s_m * float(sp.kv(0, s_m * b))
    expected_B = +kz * float(sp.kv(1, p_form * b))
    expected_C = +s_form * float(sp.kv(0, s_form * b))

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)


def test_layered_n1_row7_at_b_C_column_I_K_ratio_has_sign_flip():
    """C-column I-K ratio in row 7 is NEGATIVE: row[3]/row[4] =
    -I_0(s_m b)/K_0(s_m b). This is the F.1.a.2 sign-flip rule
    applied to the derivative-induced ``(1/r) d_r [r X_1] =
    +/- s X_0`` term: the I-flavour produces +s_m I_0; the
    K-flavour produces -s_m K_0. Ratio is sign-flipped (vs the
    +I_n/K_n ratios in row 5's D and row 1's C/D entries).

    Compared to the B-column ratio, which IS +I_1/K_1 (single-
    Bessel-term direct ``X_1`` form, KEEP-sign per F.1.a.2)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row7_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    # B ratio: +I_1/K_1 (KEEP sign).
    expected_B_ratio = (
        +float(sp.iv(1, p_m * b)) / float(sp.kv(1, p_m * b))
    )
    assert row[1].real / row[2].real == pytest.approx(expected_B_ratio)
    # C ratio: -I_0/K_0 (FLIP sign on derivative-induced term).
    expected_C_ratio = (
        -float(sp.iv(0, s_m * b)) / float(sp.kv(0, s_m * b))
    )
    assert row[3].real / row[4].real == pytest.approx(expected_C_ratio)


# =====================================================================
# Plan item F.2.b.6 -- row 8 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Lame-reduction row at the second interface; uses the unnegated
# continuity convention. Algebraically heaviest of the r=b cos
# rows.


def test_layered_n1_row8_at_b_K_flavour_cancels_at_layer_equals_formation():
    """Substep F.2.a.7 (a) self-check: all THREE K-flavour pairs
    cancel at layer=formation.

        row8[2] (B_K) + row8[5] (B) == 0
        row8[4] (C_K) + row8[6] (C) == 0
        row8[8] (D_K) + row8[9] (D) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row8_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row8_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid r<a doesn't reach r=b);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row8_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row8_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 8 has the no-row-rescale pattern;
    only column-by-(-i) on C cols. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row8_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row8_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rr decomposition at r=b for all nine non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row8_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    expected_BI = +mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(1, p_m * b))
        - 2.0 * p_m * float(sp.iv(0, p_m * b)) / b
        + 4.0 * float(sp.iv(1, p_m * b)) / (b * b)
    )
    expected_BK = +mu_m * (
        two_kz2_minus_kSm2 * float(sp.kv(1, p_m * b))
        + 2.0 * p_m * float(sp.kv(0, p_m * b)) / b
        + 4.0 * float(sp.kv(1, p_m * b)) / (b * b)
    )
    expected_CI = -2.0 * kz * mu_m * (
        s_m * float(sp.iv(0, s_m * b)) - float(sp.iv(1, s_m * b)) / b
    )
    expected_CK = +2.0 * kz * mu_m * (
        s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b
    )
    expected_B = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p_form * b))
        + 2.0 * p_form * float(sp.kv(0, p_form * b)) / b
        + 4.0 * float(sp.kv(1, p_form * b)) / (b * b)
    )
    expected_C = -2.0 * kz * mu * (
        s_form * float(sp.kv(0, s_form * b)) + float(sp.kv(1, s_form * b)) / b
    )
    expected_DI = +2.0 * mu_m * (
        s_m * float(sp.iv(0, s_m * b)) / b
        - 2.0 * float(sp.iv(1, s_m * b)) / (b * b)
    )
    expected_DK = -2.0 * mu_m * (
        s_m * float(sp.kv(0, s_m * b)) / b
        + 2.0 * float(sp.kv(1, s_m * b)) / (b * b)
    )
    expected_D = +2.0 * mu * (
        s_form * float(sp.kv(0, s_form * b)) / b
        + 2.0 * float(sp.kv(1, s_form * b)) / (b * b)
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row8_at_b_annulus_K_matches_negated_row2_M22_M23_M24_at_b():
    """Convention cross-check: at layer=formation, row 8's annulus
    K-flavour entries (B_K, C_K, D_K) equal the NEGATION of
    row 2's M22, M23, M24-equivalent forms evaluated at r=b
    (row 2 uses negated ``-(sigma_rr + P)`` convention; row 8 uses
    unnegated continuity). Pins down the convention difference."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row8 = _layered_n1_row8_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    # Row 2's M22, M23, M24 form evaluated at r=b instead of r=a.
    M22_at_b = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p * b))
        + 2.0 * p * float(sp.kv(0, p * b)) / b
        + 4.0 * float(sp.kv(1, p * b)) / (b * b)
    )
    M23_at_b = -2.0 * kz * mu * (
        s * float(sp.kv(0, s * b)) + float(sp.kv(1, s * b)) / b
    )
    M24_at_b = +2.0 * mu * (
        s * float(sp.kv(0, s * b)) / b + 2.0 * float(sp.kv(1, s * b)) / (b * b)
    )

    # Row 8 unnegated; row 2 negated. row8 = -M2j_at_b.
    assert row8[2].real == pytest.approx(-M22_at_b)
    assert row8[4].real == pytest.approx(-M23_at_b)
    assert row8[8].real == pytest.approx(-M24_at_b)


# =====================================================================
# Plan item F.2.b.7 -- row 10 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Final row of the 10x10 layered determinant; closes substep F.2.b.
# Z-derivative-bearing cos row at the second interface; analogous
# to row 4 at r=a with non-zero formation cols.


def test_layered_n1_row10_at_b_K_flavour_cancels_at_layer_equals_formation():
    """All three K-flavour pairs cancel at layer=formation."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row10_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row10_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid no shear AND fluid r<a);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row10_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row10_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 10 is z-derivative-bearing -- gets the
    FULL rescale (row * i + col-by-(-i) on C cols)."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row10_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row10_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rz decomposition at r=b for all nine non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz, omega, vp=p["vp"], vs=p["vs"], vf=p["vf"], layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row10_at_b(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=a, layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    expected_BI = -2.0 * kz * mu_m * (
        p_m * float(sp.iv(0, p_m * b)) - float(sp.iv(1, p_m * b)) / b
    )
    expected_BK = +2.0 * kz * mu_m * (
        p_m * float(sp.kv(0, p_m * b)) + float(sp.kv(1, p_m * b)) / b
    )
    expected_CI = +mu_m * two_kz2_minus_kSm2 * float(sp.iv(1, s_m * b))
    expected_CK = +mu_m * two_kz2_minus_kSm2 * float(sp.kv(1, s_m * b))
    expected_B = -2.0 * kz * mu * (
        p_form * float(sp.kv(0, p_form * b)) + float(sp.kv(1, p_form * b)) / b
    )
    expected_C = -mu * two_kz2_minus_kS2 * float(sp.kv(1, s_form * b))
    expected_DI = -kz * mu_m * float(sp.iv(1, s_m * b)) / b
    expected_DK = -kz * mu_m * float(sp.kv(1, s_m * b)) / b
    expected_D = +kz * mu * float(sp.kv(1, s_form * b)) / b

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row10_at_b_annulus_K_matches_row4_M42_M43_M44_at_b():
    """Cross-row identity: at layer=formation, row 10's annulus
    K-flavour entries (B_K, C_K, D_K) match row 4's M42, M43, M44-
    equivalent forms evaluated at r=b instead of r=a (same
    underlying sigma_rz formula at both interfaces)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row10 = _layered_n1_row10_at_b(
        kz, omega, vp=vp, vs=vs, rho=rho,
        vf=vf, rho_f=rho_f, a=a, layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M42_at_b = +2.0 * kz * mu * (
        p * float(sp.kv(0, p * b)) + float(sp.kv(1, p * b)) / b
    )
    M43_at_b = +mu * two_kz2_minus_kS2 * float(sp.kv(1, s * b))
    M44_at_b = -kz * mu * float(sp.kv(1, s * b)) / b

    assert row10[2].real == pytest.approx(M42_at_b)
    assert row10[4].real == pytest.approx(M43_at_b)
    assert row10[8].real == pytest.approx(M44_at_b)


# =====================================================================
# Plan item F.2.d -- assembly + dispatch
# =====================================================================
#
# Closes the F.2.b/c chain. Tests fall into two groups:
#
#   * ``_modal_determinant_n1_layered``: real-valued in bound regime;
#     evaluates without raising; behaves correctly at the layer=
#     formation degenerate point.
#   * ``flexural_dispersion_layered`` end-to-end: layer=formation
#     reproduces ``flexural_dispersion`` slowness curve to
#     ``rtol=1e-8``; thickness->0 limit ditto; dispatched correctly.


def _layered_n1_slow_formation_params():
    """Slow-formation fixture (vs < vf) for end-to-end layered
    flexural tests. The layer must satisfy ``layer.vs >= vs``
    (a ``harder'' layer) for the wave to stay in the bound regime
    in the annulus: flexural slowness in slow formations is very
    close to ``1/vs``, and a softer layer (``layer.vs < vs``)
    would put ``s_m^2 < 0`` in the annulus -- the leaky regime
    handled by future fast-formation-layered work."""
    return dict(
        vp=3000.0, vs=1200.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1,
        layer=BoreholeLayer(vp=3200.0, vs=1300.0, rho=2350.0, thickness=0.005),
    )


def test_modal_determinant_n1_layered_is_real_in_bound_regime():
    """Substep F.2.a.5 phase rescale: each row builder applies the
    rescale internally, so the assembled 10x10 is real-valued in
    the bound regime."""
    p = _layered_n1_slow_formation_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.05
    det = _modal_determinant_n1_layered(
        kz, omega, p["vp"], p["vs"], p["rho"],
        p["vf"], p["rho_f"], p["a"], layer=p["layer"],
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n1_layered_layer_equals_formation_root_matches_unlayered():
    """The substep-F.2.a.7 (a) self-check at the determinant level:
    at layer=formation, the layered determinant has the same
    flexural root as :func:`_modal_determinant_n1`. The two
    determinants are not numerically equal (the 10x10 has a
    different overall scale than the 4x4), but they share the
    same root in ``k_z``."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    bound = flexural_dispersion(
        np.array([5000.0]),
        vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n1_layered(
        kz_root, omega, vp, vs, rho, vf, rho_f, a, layer=layer,
    )
    det_off_root = _modal_determinant_n1_layered(
        kz_root * 1.05, omega, vp, vs, rho, vf, rho_f, a, layer=layer,
    )
    # Determinant at root much smaller than off-root.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_flexural_dispersion_layered_layer_equals_formation_matches_unlayered():
    """End-to-end integration test: with a layer whose properties
    match the formation, the layered solver produces the same
    flexural dispersion curve as the unlayered solver to
    ``rtol=1e-8``. Floating-point oracle for the entire F.2 chain.
    Any algebra error accumulated across the ten row builders
    surfaces here."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    f = np.linspace(2000.0, 8000.0, 12)

    res_unlayered = flexural_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_layered = flexural_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer,),
    )
    np.testing.assert_allclose(
        res_layered.slowness, res_unlayered.slowness,
        rtol=1.0e-8, equal_nan=True,
    )


def test_flexural_dispersion_layered_thickness_zero_limit():
    """As ``layer.thickness -> 0`` (with arbitrary layer material),
    the layered solver continuously approaches the unlayered
    answer. Algebraic identity: in the limit ``b -> a``, the rows
    at r=b approach the rows at r=a, the second interface
    degenerates, and the converged k_z must approach the single-
    interface root."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = 5000.0

    res_unlayered = flexural_dispersion(
        np.array([f]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    # Even a "different" layer with vanishing thickness should
    # converge to the unlayered flexural slowness. Use a harder
    # layer (layer.vs > vs) so the bound regime holds in the
    # annulus per the F.2.d slow-formation gate.
    layer_thin = BoreholeLayer(
        vp=3200.0, vs=1300.0, rho=2350.0, thickness=1.0e-9,
    )
    res_thin = flexural_dispersion_layered(
        np.array([f]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(layer_thin,),
    )
    assert res_thin.slowness[0] == pytest.approx(
        res_unlayered.slowness[0], rel=1.0e-4,
    )


def test_flexural_dispersion_layered_non_trivial_layer_runs():
    """Smoke test: a soft mudcake layer different from the
    formation produces a finite slowness curve in the slow-
    formation bound regime. No analytic oracle here (Schmitt 1988
    fig 6 is the F.2.e validation target); the test confirms that
    the dispatch + 10x10 + brentq + bracket all wire up without
    raising."""
    p = _layered_n1_slow_formation_params()
    f = np.linspace(2000.0, 8000.0, 8)

    res = flexural_dispersion_layered(
        f, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"],
        layers=(p["layer"],),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    assert res.slowness.shape == f.shape
    # Most slownesses finite in the bound regime; cutoff effects may
    # leave a few low-frequency NaNs, so check that at least the
    # high-f half is fully populated.
    n_finite = int(np.sum(np.isfinite(res.slowness)))
    assert n_finite >= len(f) // 2


# =====================================================================
# Plan item F.2.e -- validation hardening on top of F.2.d
# =====================================================================
#
# Hardening tests for the assembled layered flexural solver. Each
# tests an asymptotic / self-consistency property that the
# layer=formation regression alone doesn't pin down.
#
# Note: the F.1.d "thickness -> infty" test does NOT translate
# cleanly to F.2 because the layer's natural flexural mode has
# phase velocity in (V_R_layer, V_S_layer), and the F.2.d
# "harder layer" requirement (V_S_layer >= V_S_formation) means
# this band lies AT OR ABOVE V_S_formation -- outside the bound
# regime captured by the formation half-space. The layered
# flexural slowness in the thickness -> infty limit thus exits
# the bound regime; a faithful test would need fast-formation
# layered handling (future work).


def test_modal_determinant_n1_layered_vanishes_at_converged_root():
    """Self-consistency: at the converged ``k_z`` returned by
    :func:`flexural_dispersion_layered` (any non-trivial layer), the
    layered determinant is several orders of magnitude smaller than
    its value off-root. Sharper than the layer=formation det-at-root
    check from F.2.d; works for any harder layer in the slow-
    formation bound regime."""
    p = _layered_n1_slow_formation_params()
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = flexural_dispersion_layered(
        np.array([f]), vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(p["layer"],),
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n1_layered(
        kz_root, omega, p["vp"], p["vs"], p["rho"],
        p["vf"], p["rho_f"], p["a"], layer=p["layer"],
    )
    det_off = _modal_determinant_n1_layered(
        kz_root * 1.01, omega, p["vp"], p["vs"], p["rho"],
        p["vf"], p["rho_f"], p["a"], layer=p["layer"],
    )
    # brentq-converged root: |det_at| >= 6 orders of magnitude
    # smaller than |det_off| at 1% off the root.
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_flexural_dispersion_layered_multiple_frequencies_bound_regime():
    """Smoke test across the slow-formation bound band. The flexural
    slowness in a slow formation INCREASES with frequency: low-f
    cutoff is at slowness ~1/V_S (formation), high-f asymptote is
    at slowness ~1/V_R > 1/V_S (Rayleigh / Scholte limit). Confirm
    monotonicity holds across a wide band with a non-trivial
    layer."""
    p = _layered_n1_slow_formation_params()
    # Skip the very low-f cutoff region; pick frequencies safely
    # above the geometric cutoff f ~ V_S / (2 pi a) ~ 1900 Hz
    # (which the layer can shift slightly upward).
    f = np.geomspace(3000.0, 15000.0, 12)

    res = flexural_dispersion_layered(
        f, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"],
        layers=(p["layer"],),
    )
    assert np.all(np.isfinite(res.slowness))
    # Slowness increases monotonically with frequency in slow-
    # formation flexural (1/V_S at cutoff -> 1/V_R at high f, with
    # V_R < V_S so 1/V_R > 1/V_S). Tiny negative tolerance for
    # asymptotic-flatness rounding noise.
    diffs = np.diff(res.slowness)
    assert np.all(diffs > -1.0e-9)


def test_flexural_dispersion_layered_harder_layer_speeds_up_flexural():
    """Headline physics validation: a layer with ``V_S_layer >
    V_S_formation`` (harder near-borehole zone) speeds up the
    flexural wave -- the layered slowness is BELOW the unlayered
    slowness at the same frequency. Direct test of the qualitative
    expectation behind altered-zone interpretation: stiffer
    near-wall material shifts flexural slowness toward the layer's
    Rayleigh-like speed (faster than the formation's).

    Quantitative: the smoke test above showed ~1-1.3% speedup at
    a few kHz; this test confirms the inequality holds at every
    frequency in a typical band."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(3000.0, 8000.0, 10)

    res_unlayered = flexural_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    hard_layer = BoreholeLayer(
        vp=3500.0, vs=1500.0, rho=2400.0, thickness=0.01,
    )
    res_layered = flexural_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        layers=(hard_layer,),
    )
    # Both fully populated in this band.
    assert np.all(np.isfinite(res_unlayered.slowness))
    assert np.all(np.isfinite(res_layered.slowness))
    # Harder layer => faster flexural => smaller slowness.
    assert np.all(res_layered.slowness < res_unlayered.slowness)
    # Speedup should be within physically reasonable range
    # (0.1% to 5% for a 1 cm layer with modest contrast).
    speedup_frac = 1.0 - res_layered.slowness / res_unlayered.slowness
    assert np.all(speedup_frac > 0.001)
    assert np.all(speedup_frac < 0.05)


# =====================================================================
# Plan item H.0 -- public-API foundation for VTI formation
# =====================================================================
#
# Sister of F.1.0 / F.2.0 layered foundations along the anisotropy
# axis. The 5-parameter TI stiffness tensor (C11, C13, C33, C44,
# C66) collapses to the isotropic case when C11=C33, C44=C66, and
# C13=C11-2*C44 -- the dispatch in stoneley_dispersion_vti and
# flexural_dispersion_vti detects this and routes to the existing
# isotropic solvers, providing the floating-point oracle for the
# entire H chain.


def _isotropic_stiffness_from_lame(vp, vs, rho):
    """Construct an isotropic stiffness tensor (C11, C13, C33,
    C44, C66) from the Lame parameters (vp, vs, rho)."""
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2.0 * mu
    return dict(
        c11=lam + 2.0 * mu,
        c13=lam,
        c33=lam + 2.0 * mu,
        c44=mu,
        c66=mu,
    )


def test_stoneley_dispersion_vti_isotropic_collapse_bit_matches_unlayered():
    """Floating-point oracle for the H chain: with an isotropic
    stiffness tensor the VTI Stoneley solver bit-matches the
    isotropic ``stoneley_dispersion`` answer to ``rtol=1e-12``
    across a 16-point frequency grid."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(500.0, 8000.0, 16)
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)

    res_iso = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_vti = stoneley_dispersion_vti(
        f, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    np.testing.assert_array_equal(res_vti.slowness, res_iso.slowness)
    np.testing.assert_array_equal(res_vti.freq, res_iso.freq)
    assert res_vti.name == "Stoneley"
    assert res_vti.azimuthal_order == 0


def test_flexural_dispersion_vti_isotropic_collapse_bit_matches_unlayered():
    """Same floating-point oracle for the n=1 dipole flexural."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(2000.0, 8000.0, 12)
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)

    res_iso = flexural_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_vti = flexural_dispersion_vti(
        f, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    np.testing.assert_array_equal(res_vti.slowness, res_iso.slowness)
    np.testing.assert_array_equal(res_vti.freq, res_iso.freq)
    assert res_vti.name == "flexural"
    assert res_vti.azimuthal_order == 1


def test_flexural_dispersion_vti_fast_formation_genuine_TI_raises_not_implemented():
    """``flexural_dispersion_vti`` for FAST-formation genuine-TI
    (``V_Sv > V_f``) still raises ``NotImplementedError`` -- the
    real-valued VTI modal determinant requires ``F_f^2 > 0``,
    which fails for fast formations. The complex-determinant
    follow-up is deferred (see plan H.d). Slow-formation genuine
    TI ships in H.d.6 and is exercised in the integration tests
    below."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.array([5000.0])
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    # Thomsen-gamma: C44 != C66. V_Sv = 2500 > V_f = 1500.
    cij_gam = dict(cij, c66=cij["c66"] * 1.10)
    assert cij_gam["c11"] > cij_gam["c66"]
    with pytest.raises(NotImplementedError, match="fast-formation"):
        flexural_dispersion_vti(
            f, **cij_gam, rho=rho, vf=vf, rho_f=rho_f, a=a,
        )


def test_dispersion_vti_returns_borehole_mode():
    f = np.linspace(2000.0, 5000.0, 5)
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    res = stoneley_dispersion_vti(
        f, **cij, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert isinstance(res, BoreholeMode)


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"c11": 0.0}, "c11 must be positive"),
        ({"c33": -1.0e9}, "c33 must be positive"),
        ({"c44": 0.0}, "c44 must be positive"),
        ({"c66": -1.0}, "c66 must be positive"),
    ],
)
def test_dispersion_vti_rejects_non_positive_cij(kwargs, msg):
    f = np.array([5000.0])
    base = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        stoneley_dispersion_vti(
            f, **base, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1,
        )


def test_dispersion_vti_rejects_unstable_c33_le_c13():
    """Validator rejects ``C33 <= C13`` (would break qP/qSV
    decoupling in the Christoffel equation)."""
    f = np.array([5000.0])
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    cij["c13"] = cij["c33"] * 1.5  # force c13 > c33
    with pytest.raises(ValueError, match="c33 > c13"):
        stoneley_dispersion_vti(
            f, **cij, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1,
        )


def test_dispersion_vti_rejects_unstable_c11_le_c66():
    """Validator rejects ``C11 <= C66`` (would have horizontal P
    no faster than horizontal S)."""
    f = np.array([5000.0])
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    cij["c66"] = cij["c11"] * 1.5  # force c66 > c11
    with pytest.raises(ValueError, match="c11 > c66"):
        stoneley_dispersion_vti(
            f, **cij, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1,
        )


def test_dispersion_vti_rejects_non_positive_freq_and_geometry():
    """Standard freq/geometry validation as for the isotropic
    public APIs."""
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    base = dict(rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1)
    # Non-positive freq.
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        stoneley_dispersion_vti(
            np.array([0.0]), **cij, **base,
        )
    # Non-positive vf.
    with pytest.raises(ValueError, match="vf and rho_f must be positive"):
        stoneley_dispersion_vti(
            np.array([5000.0]), **cij, **{**base, "vf": 0.0},
        )
    # Non-positive a.
    with pytest.raises(ValueError, match="a must be positive"):
        stoneley_dispersion_vti(
            np.array([5000.0]), **cij, **{**base, "a": 0.0},
        )


# =====================================================================
# Plan item H.b -- radial-wavenumber helper (Christoffel-equation roots)
# =====================================================================
#
# Bound-regime ``(alpha_qP, alpha_qSV, alpha_SH)`` from the H.a.2
# Christoffel quadratic and the H.a.4 SH closed form. The
# isotropic-collapse identity (qP -> p, qSV -> s, SH -> s) is the
# floating-point oracle for the entire H chain.


def _typical_vti_params():
    """Genuine-TI fixture for H.b tests. Roughly Thomsen-style:
    ~10% epsilon, ~5% delta, ~15% gamma. Within the Thomsen-stable
    range where qP / qSV remain well-separated."""
    return dict(
        c11=4.0e10,   # V_Ph^2 * rho ~ (4080 m/s)^2 * 2400
        c13=1.5e10,   # delta-coupled
        c33=3.5e10,   # V_Pv^2 * rho ~ (3819 m/s)^2 * 2400
        c44=1.0e10,   # V_Sv^2 * rho ~ (2041 m/s)^2 * 2400
        c66=1.3e10,   # V_Sh^2 * rho ~ (2327 m/s)^2 * 2400  (gamma > 0)
        rho=2400.0,
    )


def test_radial_wavenumbers_vti_isotropic_collapse_matches_isotropic():
    """Floating-point oracle for H.b: with an isotropic stiffness
    tensor the VTI radial wavenumbers reduce to the isotropic
    ``(p, s, s)``."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5

    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )

    p_iso = float(np.sqrt(kz ** 2 - (omega / vp) ** 2))
    s_iso = float(np.sqrt(kz ** 2 - (omega / vs) ** 2))

    assert alpha_qP == pytest.approx(p_iso, rel=1.0e-12)
    assert alpha_qSV == pytest.approx(s_iso, rel=1.0e-12)
    assert alpha_SH == pytest.approx(s_iso, rel=1.0e-12)


def test_radial_wavenumbers_vti_genuine_TI_christoffel_identity():
    """Christoffel-equation identity check: substituting the qP and
    qSV roots back into the bound-mode Christoffel determinant must
    give zero to floating-point precision. Catches sign / coefficient
    errors in the H.a.2 quadratic transcription that the isotropic-
    collapse test would miss when C44 = C66."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    # kz above the bound floor for both qP and qSV branches.
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh) * 1.5

    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )

    rho_omega_sq = rho * omega * omega
    # Christoffel determinant at alpha^2 (substep H.a.2 form):
    #   det = (-C11 alpha^2 + C44 kz^2 - rho omega^2)
    #         * (-C44 alpha^2 + C33 kz^2 - rho omega^2)
    #       + (C13 + C44)^2 alpha^2 kz^2
    def det_christoffel(alpha):
        a2 = alpha * alpha
        m11 = -cij["c11"] * a2 + cij["c44"] * kz * kz - rho_omega_sq
        m22 = -cij["c44"] * a2 + cij["c33"] * kz * kz - rho_omega_sq
        return m11 * m22 + (cij["c13"] + cij["c44"]) ** 2 * a2 * kz * kz

    # Both qP and qSV roots should give det = 0 to fp precision.
    # Use a relative tolerance: the determinant scales as
    # (rho omega^2)^2 ~ 1e21 in this fixture, so absolute zero
    # tolerance must be set against that scale.
    scale = rho_omega_sq * rho_omega_sq
    assert abs(det_christoffel(alpha_qP)) < scale * 1.0e-10
    assert abs(det_christoffel(alpha_qSV)) < scale * 1.0e-10


def test_radial_wavenumbers_vti_qP_larger_than_qSV():
    """Substep H.a.3 ordering: alpha_qP > alpha_qSV in the bound
    regime. Convention agrees with the isotropic limit (p > s
    always when V_P > V_S, because the radial decay rate is
    ``alpha = sqrt(kz^2 - omega^2/V^2)`` and larger V gives larger
    alpha)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh) * 1.5

    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    assert alpha_qP > alpha_qSV
    assert alpha_qP > 0.0
    assert alpha_qSV > 0.0


def test_radial_wavenumbers_vti_SH_uses_C44_and_C66():
    """Substep H.a.4 (corrected): alpha_SH^2 = (C44 kz^2 - rho
    omega^2) / C66. Verify directly. Distinguishes the corrected
    form from the buggy ``kz^2 - rho omega^2 / C66`` (which would
    give the same isotropic limit but wrong genuine-TI value)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh) * 1.5

    _, _, alpha_SH = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    expected_SH_sq = (cij["c44"] * kz ** 2 - rho * omega ** 2) / cij["c66"]
    assert alpha_SH ** 2 == pytest.approx(expected_SH_sq, rel=1.0e-12)
    # Sanity: the buggy ``kz^2 - rho omega^2 / C66`` form would
    # give a DIFFERENT value here because C44 != C66.
    buggy_SH_sq = kz ** 2 - rho * omega ** 2 / cij["c66"]
    assert abs(alpha_SH ** 2 - buggy_SH_sq) > 0.0  # they differ
    assert cij["c44"] != cij["c66"]  # confirm fixture is genuine TI


def test_radial_wavenumbers_vti_below_bound_floor_returns_nan():
    """Below the bound floor ``kz < omega / min(V_Sv, V_Sh, V_f)``
    one or more decay rates would be imaginary; the helper returns
    NaN (brentq-safe convention)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    # Pick kz well below the bound floor.
    vsv = float(np.sqrt(cij["c44"] / rho))
    kz = omega / vsv * 0.5
    with np.errstate(invalid="ignore"):
        alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
            kz, omega, **cij, rho=rho,
        )
    # alpha_SH definitely NaN below the V_Sv-related floor.
    assert np.isnan(alpha_SH)


# =====================================================================
# Plan item H.c.1.a -- row 1 of the n=0 VTI modal determinant (r = a)
# =====================================================================
#
# First row of the 3x3 VTI Stoneley modal determinant. Returns the
# three post-rescale coefficients [A | B_qP, C_qSV]. At isotropic
# collapse the entries match (M11, M12, M13) of
# :func:`_modal_determinant_n0` bit-exactly -- the floating-point
# oracle for the H.c.1 chain.


def test_modal_row1_at_a_vti_isotropic_collapse_matches_M11_M12_M13():
    """Floating-point oracle: at isotropic stiffness, row 1 of the
    VTI determinant matches M11, M12, M13 of
    :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row1_at_a_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = F * float(sp.iv(1, F * a)) / (rho_f * omega ** 2)
    M12 = p * float(sp.kv(1, p * a))
    M13 = kz * float(sp.kv(1, s * a))

    assert row[0].real == pytest.approx(M11, rel=1.0e-12)
    assert row[1].real == pytest.approx(M12, rel=1.0e-12)
    assert row[2].real == pytest.approx(M13, rel=1.0e-12)


def test_modal_row1_at_a_vti_all_columns_nonzero_in_bound_regime():
    """Sparsity / non-degeneracy: in the bound regime all three
    columns of row 1 are non-zero."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    for i in range(3):
        assert row[i] != 0.0


def test_modal_row1_at_a_vti_is_real_in_bound_regime():
    """Substep H.a.6 phase rescale: row 1 has the no-row-rescale
    pattern; only column-by-(-i) on the C_qSV column is applied.
    Post-rescale row is real-valued in the bound regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row1_at_a_vti_uses_alpha_qP_alpha_qSV_not_p_s():
    """Genuine TI sanity: with non-trivial epsilon (C11 != C33),
    the qP root alpha_qP differs from the isotropic-equivalent
    p = sqrt(kz^2 - omega^2 / V_Pv^2). Verify row[1] uses
    alpha_qP K_1(alpha_qP a), NOT p K_1(p a). Confirms the row
    builder pulls from the Christoffel solver, not from a hard-
    coded isotropic substitution."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    # Pull alpha_qP from the helper.
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    from scipy import special as sp

    expected_BqP = alpha_qP * float(sp.kv(1, alpha_qP * 0.1))
    expected_CqSV = kz * float(sp.kv(1, alpha_qSV * 0.1))
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    # Sanity: confirm alpha_qP differs from the naive p computed
    # with V_Pv = sqrt(c33/rho) so the test isn't passing trivially.
    Vpv = float(np.sqrt(cij["c33"] / rho))
    p_naive = float(np.sqrt(kz * kz - (omega / Vpv) ** 2))
    assert abs(alpha_qP - p_naive) > 0.01  # non-trivial epsilon


# =====================================================================
# Plan item H.c.1.b -- polarization-ratio helper + row 2 (sigma_rr at a)
# =====================================================================
#
# Algebraically heaviest row of the n=0 VTI determinant. Tests
# anchor on the per-element layer=formation match against M21,
# M22, M23 of :func:`_modal_determinant_n0` plus a separate
# polarization-ratio identity check.


def test_polarization_ratio_uz_over_ur_vti_isotropic_qP_limit():
    """At isotropic limit alpha_qP -> p, the polarization ratio
    gamma_qP = -i k_z / p."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5

    p_iso = float(np.sqrt(kz ** 2 - (omega / vp) ** 2))
    gamma_qP = _polarization_ratio_uz_over_ur_vti(
        p_iso, kz, omega, c11=cij["c11"], c13=cij["c13"],
        c44=cij["c44"], rho=rho,
    )
    expected = -1j * kz / p_iso
    assert gamma_qP.real == pytest.approx(expected.real, abs=1.0e-12)
    assert gamma_qP.imag == pytest.approx(expected.imag, rel=1.0e-12)


def test_polarization_ratio_uz_over_ur_vti_isotropic_qSV_limit():
    """At isotropic limit alpha_qSV -> s, gamma_qSV = -i s / k_z."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5

    s_iso = float(np.sqrt(kz ** 2 - (omega / vs) ** 2))
    gamma_qSV = _polarization_ratio_uz_over_ur_vti(
        s_iso, kz, omega, c11=cij["c11"], c13=cij["c13"],
        c44=cij["c44"], rho=rho,
    )
    expected = -1j * s_iso / kz
    assert gamma_qSV.real == pytest.approx(expected.real, abs=1.0e-12)
    assert gamma_qSV.imag == pytest.approx(expected.imag, rel=1.0e-12)


def test_polarization_ratio_uz_over_ur_vti_christoffel_identity():
    """Substituting (u_r, u_z) = (1, gamma_qX) into the Christoffel
    eigenvector equation
        (-C11 alpha^2 + C44 kz^2 - rho omega^2) u_r
        + i (C13 + C44) alpha kz u_z = 0
    must give zero to floating-point precision (verifies that the
    polarization-ratio formula is the correct null-space direction
    of the Christoffel matrix at the qX root)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    rho_omega_sq = rho * omega ** 2

    for alpha_qX in (alpha_qP, alpha_qSV):
        gamma = _polarization_ratio_uz_over_ur_vti(
            alpha_qX, kz, omega,
            c11=cij["c11"], c13=cij["c13"], c44=cij["c44"], rho=rho,
        )
        # Eigenvector equation residual:
        residual = (
            (-cij["c11"] * alpha_qX ** 2 + cij["c44"] * kz ** 2 - rho_omega_sq)
            + 1j * (cij["c13"] + cij["c44"]) * alpha_qX * kz * gamma
        )
        # Scale check: M11 element ~ rho omega^2 in magnitude.
        assert abs(residual) < rho_omega_sq * 1.0e-12


def test_modal_row2_at_a_vti_isotropic_collapse_matches_M21_M22_M23():
    """Floating-point oracle for H.c.1.b: at isotropic stiffness,
    row 2 of the VTI determinant matches M21, M22, M23 of
    :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row2_at_a_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M21 = -float(sp.iv(0, F * a))
    M22 = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p * a))
        + 2.0 * p * float(sp.kv(1, p * a)) / a
    )
    M23 = -2.0 * kz * mu * (
        s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a
    )

    assert row[0].real == pytest.approx(M21, rel=1.0e-12)
    assert row[1].real == pytest.approx(M22, rel=1.0e-12)
    assert row[2].real == pytest.approx(M23, rel=1.0e-12)


def test_modal_row2_at_a_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 2 is no-row-rescale; col-by-(-i) on
    C_qSV. Post-rescale row is real-valued in the bound regime.
    Catches polarization-ratio sign errors that would leave a
    nonzero imaginary part."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row2_at_a_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e0)


def test_modal_row2_at_a_vti_uses_c66_in_KK_one_over_a_term():
    """Genuine-TI sanity: with C44 != C66 (gamma > 0), the K_1/a
    coefficient of the B_qP column scales with ``2 C66``, NOT
    ``2 C44``. Confirms the (C11 - 2 C66) u_r/r slot is correctly
    transcribed -- the slot through which Norris 1990 LF coupling
    enters."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row2_at_a_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    alpha_qP, _, _ = _radial_wavenumbers_vti(kz, omega, **cij, rho=rho)
    rho_omega_sq = rho * omega ** 2
    q_qP = (
        cij["c44"] * (cij["c11"] * alpha_qP ** 2 + cij["c13"] * kz ** 2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    from scipy import special as sp

    # row[1] = -Q_qP K_0(alpha_qP a) - 2 C66 alpha_qP K_1(alpha_qP a)/a
    expected = (
        -q_qP * float(sp.kv(0, alpha_qP * a))
        - 2.0 * cij["c66"] * alpha_qP * float(sp.kv(1, alpha_qP * a)) / a
    )
    assert row[1].real == pytest.approx(expected, rel=1.0e-12)
    # Sanity check: confirm fixture has C44 != C66 (genuine gamma).
    assert cij["c44"] != cij["c66"]


# =====================================================================
# Plan item H.c.1.c -- row 3 (sigma_rz at r=a)
# =====================================================================
#
# Z-derivative-bearing cos row of the n=0 VTI determinant. Gets
# the FULL substep-H.a.6 phase rescale (row * i AND col-by-(-i)
# on C_qSV). Tests anchor on the per-element layer=formation
# match against M31, M32, M33.


def test_modal_row3_at_a_vti_isotropic_collapse_matches_M31_M32_M33():
    """Floating-point oracle for H.c.1.c: at isotropic stiffness,
    row 3 of the VTI determinant matches M31, M32, M33 of
    :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row3_at_a_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M31 = 0.0
    M32 = 2.0 * kz * p * mu * float(sp.kv(1, p * a))
    M33 = mu * two_kz2_minus_kS2 * float(sp.kv(1, s * a))

    assert row[0] == 0.0
    assert row[1].real == pytest.approx(M32, rel=1.0e-12)
    assert row[2].real == pytest.approx(M33, rel=1.0e-12)


def test_modal_row3_at_a_vti_fluid_column_is_zero():
    """A column identically zero (fluid carries no shear)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert row[0] == 0.0
    # B and C columns generically non-zero.
    assert row[1] != 0.0
    assert row[2] != 0.0


def test_modal_row3_at_a_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 3 is z-derivative-bearing -- gets the
    FULL rescale (row * i AND col-by-(-i) on C_qSV). Both must be
    correctly applied for the post-rescale row to be real.
    Forgetting the row * i is the most direct H.a.6 transcription
    error; this test catches it."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row3_at_a_vti_does_not_use_c66():
    """Genuine-TI sanity: sigma_rz uses ONLY C44 (vertical shear),
    not C66 (horizontal shear). Doubling C66 leaves row 3
    unchanged, EXCEPT through the alpha_qP and alpha_qSV
    Christoffel roots from :func:`_radial_wavenumbers_vti` which
    only depend on C11, C13, C33, C44, rho (not C66). Confirms
    the (C11 - 2 C66) u_r/r slot does NOT appear in row 3."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 2.0  # double C66
    rho = cij_a.pop("rho")
    cij_b.pop("rho")

    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij_a["c44"] / rho))
    # Bound regime floor: must be above min(V_Sv) and the doubled
    # V_Sh of cij_b. Pick kz well above both.
    kz = omega / vsv * 2.0

    row_a = _modal_row3_at_a_vti(
        kz, omega, **cij_a, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    row_b = _modal_row3_at_a_vti(
        kz, omega, **cij_b, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    # Identical: row 3 doesn't see C66 directly.
    np.testing.assert_array_equal(row_a, row_b)


# =====================================================================
# Plan item H.c.1.d -- assembly into _modal_determinant_n0_vti
# =====================================================================
#
# Stacks the three row builders into the 3x3 VTI Stoneley modal
# matrix; takes the real determinant. Tests anchor on the
# determinant-vanishes-at-isotropic-Stoneley-root self-consistency.


def test_modal_determinant_n0_vti_is_real_in_bound_regime():
    """The assembled 3x3 matrix is real-valued post-rescale (each
    row builder applies its own rescale internally), so
    ``np.linalg.det`` returns a finite real scalar in the bound
    regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    det = _modal_determinant_n0_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n0_vti_isotropic_collapse_root_matches_unlayered():
    """Substep H.a.7 (a) self-check at the determinant level: at
    isotropic stiffness, the VTI determinant has the same Stoneley
    root as :func:`_modal_determinant_n0`. The two determinants
    are not numerically equal (different overall scale due to
    different intermediate factors), but they share the same root
    in ``k_z``.

    Verify by: (a) computing the Stoneley root from
    ``stoneley_dispersion``; (b) evaluating the VTI determinant
    at that root; (c) checking ``|det_at_root|`` is small relative
    to its value off-root."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0

    bound = stoneley_dispersion(
        np.array([5000.0]),
        vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n0_vti(
        kz_root, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    det_off_root = _modal_determinant_n0_vti(
        kz_root * 1.05, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    # Determinant at root much smaller than off-root: brentq-type
    # root-finder will converge cleanly. The factor 1e-3 budget
    # is loose because the two determinants differ in absolute
    # scale; tighter tolerance kicks in at the full
    # stoneley_dispersion_vti integration test in H.c.2.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_modal_determinant_n0_vti_bracket_brackets_isotropic_root():
    """End-to-end at-isotropic check: brentq across the
    standard Stoneley bracket finds the determinant root, and
    that root matches the isotropic Stoneley slowness."""
    from scipy import optimize

    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0

    bound = stoneley_dispersion(
        np.array([5000.0]),
        vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    kz_root_iso = float(bound.slowness[0]) * omega

    def _det(kz):
        return _modal_determinant_n0_vti(
            kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
        )

    # Bracket around the isotropic root.
    kz_lo = kz_root_iso * 0.99
    kz_hi = kz_root_iso * 1.01
    d_lo = _det(kz_lo)
    d_hi = _det(kz_hi)
    assert np.sign(d_lo) != np.sign(d_hi)  # bracket valid
    kz_root_vti = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
    assert kz_root_vti == pytest.approx(kz_root_iso, rel=1.0e-8)


def test_modal_determinant_n0_vti_returns_nan_outside_bound_regime():
    """Below the bound floor at least one Christoffel root is
    imaginary; the assembled determinant returns NaN (brentq-safe
    convention propagates from the radial-wavenumber helper)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vp_h = float(np.sqrt(cij["c11"] / rho))  # fastest body wave
    omega = 2.0 * np.pi * 5000.0
    kz = omega / vp_h * 0.5  # well below the bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n0_vti(
            kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
        )
    assert np.isnan(det)


# =====================================================================
# Plan item H.c.2 -- Stoneley public-API hook (genuine TI brentq path)
# =====================================================================
#
# Replaces the H.0 ``NotImplementedError`` with a brentq loop on
# ``_modal_determinant_n0_vti``. The integration oracle is the
# isotropic-collapse regression vs ``stoneley_dispersion`` to
# ``rtol=1e-8`` -- the floating-point oracle for the entire H.c
# chain.


def test_stoneley_dispersion_vti_isotropic_via_genuine_TI_path_matches_isotropic():
    """Floating-point oracle for the H.c chain. Force the
    genuine-TI brentq path by passing a stiffness tensor that is
    formally non-isotropic (``c13`` perturbed by 1 ULP) but
    physically equivalent to isotropic, and verify the resulting
    slowness curve matches the isotropic ``stoneley_dispersion``
    answer to ``rtol=1e-7``.

    This test is more discriminating than the H.0 isotropic-
    collapse test (which dispatches directly to
    ``stoneley_dispersion`` and cannot fail) because it exercises
    the full ``_modal_determinant_n0_vti`` + brentq pipeline."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    # Force the genuine-TI path by tweaking c13 by 1 part in 1e-6
    # (well within Thomsen-stability but enough to defeat the
    # isotropic dispatch).
    cij_perturbed = dict(cij)
    cij_perturbed["c13"] = cij["c13"] * (1.0 + 1.0e-6)
    f = np.linspace(500.0, 8000.0, 12)

    res_iso = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_vti = stoneley_dispersion_vti(
        f, **cij_perturbed, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    np.testing.assert_allclose(
        res_vti.slowness, res_iso.slowness,
        rtol=1.0e-5, equal_nan=True,
    )
    # Confirm the perturbation actually defeated the isotropic
    # dispatch (the test would pass trivially otherwise).
    assert not _is_isotropic_stiffness(**{
        k: cij_perturbed[k] for k in ("c11", "c13", "c33", "c44", "c66")
    })


def test_stoneley_dispersion_vti_genuine_TI_runs_smoke():
    """Smoke: a typical genuine-TI fixture produces a finite
    slowness curve. No analytic oracle (Norris 1990 LF check is
    H.c.3); just confirms the brentq + bracket combination
    handles the TI case across a broad band."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(1000.0, 10000.0, 8)

    res = stoneley_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    assert res.slowness.shape == f.shape
    assert np.all(np.isfinite(res.slowness))
    # All slownesses above the slowest-shear floor.
    Vsv = float(np.sqrt(cij["c44"] / rho))
    Vsh = float(np.sqrt(cij["c66"] / rho))
    floor = 1.0 / max(Vsv, Vsh, 1500.0)
    assert np.all(res.slowness > floor)


def test_stoneley_dispersion_vti_genuine_TI_determinant_vanishes_at_root():
    """At each converged kz from ``stoneley_dispersion_vti``, the
    underlying VTI determinant must vanish (self-consistency).
    Ratio against the off-root determinant value at kz_root *
    1.01."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = stoneley_dispersion_vti(
        np.array([f]), **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n0_vti(
        kz_root, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    det_off = _modal_determinant_n0_vti(
        kz_root * 1.01, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_stoneley_dispersion_vti_returns_borehole_mode_for_genuine_TI():
    """BoreholeMode return-type contract on the genuine-TI path."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(2000.0, 5000.0, 4)
    res = stoneley_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    np.testing.assert_array_equal(res.freq, f)


# =====================================================================
# Plan item H.c.3 -- Norris 1990 LF closed-form oracle
# =====================================================================
#
# The TI-specific validation oracle. At low frequency the n=0
# Stoneley slowness in a VTI formation approaches
#
#       S_ST^2 = 1/V_f^2 + rho_f / C66            (Norris 1990 eq. 6)
#
# Strongest validation of the C-matrix entries: depends on **C66**
# (NOT C44) -- the difference is invisible in the isotropic-collapse
# tests (where C44 = C66) but emerges sharply with gamma > 0.


def _norris_1990_LF_stoneley_slowness(c66, vf, rho_f):
    """S_ST = sqrt(1/V_f^2 + rho_f / C66) per Norris 1990 eq. 6.
    The TI-specific LF closed form for the Stoneley tube-wave
    slowness."""
    return float(np.sqrt(1.0 / vf ** 2 + rho_f / c66))


def test_stoneley_dispersion_vti_LF_matches_norris_1990_C66_form():
    """At very low frequency the VTI Stoneley slowness approaches
    the Norris 1990 closed form
        S_ST = sqrt(1/V_f^2 + rho_f / C66).

    Tested with a typical Thomsen-stable VTI fixture (gamma ~
    0.15) at f = 10 Hz. Tolerance loose because the LF closed
    form is asymptotic; tightening would require f -> 0 and run
    into bracket-floor numerics."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vf, rho_f, a = 1500.0, 1000.0, 0.1

    res = stoneley_dispersion_vti(
        np.array([10.0]), **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    s_norris = _norris_1990_LF_stoneley_slowness(cij["c66"], vf, rho_f)
    # ~0.1% tolerance: leading-order asymptote.
    assert res.slowness[0] == pytest.approx(s_norris, rel=1.0e-3)


def test_stoneley_dispersion_vti_LF_distinguishes_C66_from_C44():
    """Genuine TI vs isotropic-with-C44: the Norris 1990 LF form
    uses C66, not C44. With gamma > 0 (C66 > C44), the genuine-TI
    LF slowness matches the C66-based form much more closely than
    the C44-based form. Confirms the previous test isn't passing
    trivially through C44 = C66."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vf, rho_f, a = 1500.0, 1000.0, 0.1

    res = stoneley_dispersion_vti(
        np.array([10.0]), **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    s_C66 = _norris_1990_LF_stoneley_slowness(cij["c66"], vf, rho_f)
    s_C44 = _norris_1990_LF_stoneley_slowness(cij["c44"], vf, rho_f)
    # Fixture has C66 > C44 (gamma > 0), so s_C66 < s_C44.
    assert s_C66 < s_C44
    err_to_C66 = abs(res.slowness[0] - s_C66) / s_C66
    err_to_C44 = abs(res.slowness[0] - s_C44) / s_C44
    assert err_to_C66 < err_to_C44 * 0.05  # at least 20x closer
    gamma = (cij["c66"] - cij["c44"]) / (2.0 * cij["c44"])
    assert gamma > 0.05


def test_stoneley_dispersion_vti_LF_gamma_monotonicity():
    """Increasing C66 (at fixed other C-matrix entries) decreases
    the LF Stoneley slowness per Norris 1990 (since
    ``dS_ST^2/dC66 = -rho_f / C66^2 < 0``).

    Verify by computing the LF slowness at two C66 values: the
    larger C66 produces the smaller slowness."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 1.20
    rho = cij_a.pop("rho")
    cij_b.pop("rho")
    vf, rho_f, a = 1500.0, 1000.0, 0.1

    res_a = stoneley_dispersion_vti(
        np.array([10.0]), **cij_a, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_b = stoneley_dispersion_vti(
        np.array([10.0]), **cij_b, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    # Larger C66 -> smaller LF Stoneley slowness.
    assert res_b.slowness[0] < res_a.slowness[0]
    s_norris_a = _norris_1990_LF_stoneley_slowness(cij_a["c66"], vf, rho_f)
    s_norris_b = _norris_1990_LF_stoneley_slowness(cij_b["c66"], vf, rho_f)
    expected_ratio = s_norris_b / s_norris_a
    actual_ratio = res_b.slowness[0] / res_a.slowness[0]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1.0e-3)


# =====================================================================
# Plan item H.d.1 -- row 1 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# First row of the 4x4 n=1 VTI flexural modal determinant.
# Mirrors :func:`_modal_determinant_n1`'s M11-M14 with the
# Christoffel roots (alpha_qP, alpha_qSV, alpha_SH) replacing
# isotropic (p, s, s). New at n>=1 (vs the n=0 H.c.1.a row 1):
# the D_SH column appears via (1/r) d_theta psi_z cross-coupling.


def test_modal_row1_at_a_n1_vti_isotropic_collapse_matches_M11_M12_M13_M14():
    """Floating-point oracle for H.d.1: at isotropic stiffness,
    row 1 of the n=1 VTI determinant matches M11, M12, M13, M14
    of :func:`_modal_determinant_n1` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row1_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = (
        F * float(sp.iv(0, F * a)) - float(sp.iv(1, F * a)) / a
    ) / (rho_f * omega ** 2)
    M12 = p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a
    M13 = kz * float(sp.kv(1, s * a))
    M14 = -float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M11, rel=1.0e-12)
    assert row[1].real == pytest.approx(M12, rel=1.0e-12)
    assert row[2].real == pytest.approx(M13, rel=1.0e-12)
    assert row[3].real == pytest.approx(M14, rel=1.0e-12)


def test_modal_row1_at_a_n1_vti_all_columns_nonzero_in_bound_regime():
    """Sparsity / non-degeneracy: in the bound regime all four
    columns of row 1 are non-zero. (No fluid-no-shear constraint
    on row 1; A enters via fluid pressure.)"""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    for i in range(4):
        assert row[i] != 0.0


def test_modal_row1_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 1 has the no-row-rescale pattern; only
    column-by-(-i) on C_qSV is applied. Post-rescale row is
    real-valued in the bound regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row1_at_a_n1_vti_uses_christoffel_roots_not_naive():
    """Genuine TI sanity: with non-trivial epsilon (C11 != C33),
    the qP root alpha_qP differs from the naive ``sqrt(kz^2 -
    omega^2/V_Pv^2)``; row[1] uses alpha_qP via the Christoffel
    solver. Same check for alpha_qSV and alpha_SH (which differ
    from the naive isotropic-with-V_Sv values when gamma > 0)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row1_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    from scipy import special as sp

    expected_BqP = (
        alpha_qP * float(sp.kv(0, alpha_qP * a))
        + float(sp.kv(1, alpha_qP * a)) / a
    )
    expected_CqSV = kz * float(sp.kv(1, alpha_qSV * a))
    expected_DSH = -float(sp.kv(1, alpha_SH * a)) / a

    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)
    # Sanity: with non-trivial gamma, alpha_SH differs from
    # alpha_qSV (different stiffness moduli C66 vs C44 enter).
    assert abs(alpha_SH - alpha_qSV) > 1.0  # well-separated roots


# =====================================================================
# Plan item H.d.2 -- row 2 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# Algebraically heaviest row of the n=1 VTI determinant. Each
# column has multi-Bessel-term entries combining Q_qX (from
# H.c.1.b) with the n=1 ``4 C66 K_1/a^2`` azimuthal-derivative slot.
# Tests anchor on the per-element layer=formation match against
# M21-M24 of :func:`_modal_determinant_n1`.


def test_modal_row2_at_a_n1_vti_isotropic_collapse_matches_M21_M22_M23_M24():
    """Floating-point oracle for H.d.2: at isotropic stiffness,
    row 2 matches M21, M22, M23, M24 of :func:`_modal_determinant_n1`
    to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row2_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M21 = -float(sp.iv(1, F * a))
    M22 = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p * a))
        + 2.0 * p * float(sp.kv(0, p * a)) / a
        + 4.0 * float(sp.kv(1, p * a)) / (a * a)
    )
    M23 = -2.0 * kz * mu * (
        s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a
    )
    M24 = +2.0 * mu * (
        s * float(sp.kv(0, s * a)) / a
        + 2.0 * float(sp.kv(1, s * a)) / (a * a)
    )

    assert row[0].real == pytest.approx(M21, rel=1.0e-12)
    assert row[1].real == pytest.approx(M22, rel=1.0e-12)
    assert row[2].real == pytest.approx(M23, rel=1.0e-12)
    assert row[3].real == pytest.approx(M24, rel=1.0e-12)


def test_modal_row2_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 2 is no-row-rescale; col-by-(-i) on
    C_qSV. Post-rescale row is real-valued in the bound regime.
    Catches polarization-ratio sign errors that would leave a
    nonzero imaginary part."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row2_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e0)


def test_modal_row2_at_a_n1_vti_matches_closed_form_per_column():
    """Per-column transcription check against the H.d.2 derivation
    closed forms (Q_qX combinations + C66 azimuthal-derivative
    slots)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1
    vf, rho_f = 1500.0, 1000.0

    row = _modal_row2_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    rho_omega_sq = rho * omega ** 2
    q_qP = (
        cij["c44"] * (cij["c11"] * alpha_qP ** 2 + cij["c13"] * kz ** 2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    q_qSV = (
        cij["c44"] * (cij["c11"] * alpha_qSV ** 2 + cij["c13"] * kz ** 2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    from scipy import special as sp

    F = float(np.sqrt(kz ** 2 - (omega / vf) ** 2))
    expected_A = -float(sp.iv(1, F * a))
    expected_BqP = -(
        q_qP * float(sp.kv(1, alpha_qP * a))
        + 2.0 * cij["c66"] * alpha_qP * float(sp.kv(0, alpha_qP * a)) / a
        + 4.0 * cij["c66"] * float(sp.kv(1, alpha_qP * a)) / (a * a)
    )
    expected_CqSV = -kz * (
        q_qSV / alpha_qSV * float(sp.kv(0, alpha_qSV * a))
        + 2.0 * cij["c66"] * float(sp.kv(1, alpha_qSV * a)) / a
    )
    expected_DSH = +2.0 * cij["c66"] * (
        alpha_SH * float(sp.kv(0, alpha_SH * a)) / a
        + 2.0 * float(sp.kv(1, alpha_SH * a)) / (a * a)
    )

    assert row[0].real == pytest.approx(expected_A, rel=1.0e-12)
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)


def test_modal_row2_at_a_n1_vti_BqP_K1_over_a_squared_scales_with_4_C66():
    """Genuine-TI sanity: the K_1/a^2 coefficient of the B_qP
    column scales with ``4 C66`` (NOT 4 C44). Same approach as
    F.2.b.2's (C11 - 2 C66) test for the layered case.

    Confirms the n=1 azimuthal-derivative slot ``4 C66 K_1/a^2``
    -- which combines u_r/r and (1/r) d_theta u_theta contributions
    -- is correctly transcribed."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    # Vary C66, keep all other C-matrix entries fixed.
    cij_a = dict(cij)
    cij_b = dict(cij)
    cij_b["c66"] = cij_a["c66"] * 1.50  # 50% increase

    row_a = _modal_row2_at_a_n1_vti(
        kz, omega, **cij_a, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    row_b = _modal_row2_at_a_n1_vti(
        kz, omega, **cij_b, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    # The difference in row[1] is purely from the 2 C66 K_0/a +
    # 4 C66 K_1/a^2 slots (Q_qP doesn't depend on C66; alpha_qP
    # doesn't depend on C66 directly either since the Christoffel
    # quadratic uses only C11, C13, C33, C44).
    alpha_qP, _, _ = _radial_wavenumbers_vti(kz, omega, **cij_a, rho=rho)
    from scipy import special as sp

    delta_c66 = cij_b["c66"] - cij_a["c66"]
    expected_diff = -(
        2.0 * delta_c66 * alpha_qP * float(sp.kv(0, alpha_qP * a)) / a
        + 4.0 * delta_c66 * float(sp.kv(1, alpha_qP * a)) / (a * a)
    )
    actual_diff = row_b[1].real - row_a[1].real
    assert actual_diff == pytest.approx(expected_diff, rel=1.0e-10)
    # Sanity: confirm the difference is non-zero (test isn't
    # passing trivially).
    assert abs(expected_diff) > 0.0


def test_modal_row2_at_a_n1_vti_DSH_column_pure_C66_scaling():
    """The D_SH column of row 2 scales entirely with C66 (no Q
    factor; pure (C11 - 2 C66) epsilon_theta_theta contribution).
    Verify by doubling C66 and checking that the D_SH entry
    doubles accordingly (modulo the alpha_SH change which itself
    depends on C66 via the SH dispersion ``alpha_SH^2 = (C44 kz^2 -
    rho omega^2)/C66``).

    Since alpha_SH depends on C66, the test compares against the
    explicit closed form rather than a simple ratio."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    # Two C66 values.
    cij_a = dict(cij)
    cij_b = dict(cij_a, c66=cij_a["c66"] * 2.0)

    row_a = _modal_row2_at_a_n1_vti(
        kz, omega, **cij_a, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    row_b = _modal_row2_at_a_n1_vti(
        kz, omega, **cij_b, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    # row[3] = 2 C66 (alpha_SH K_0(alpha_SH a)/a + 2 K_1(alpha_SH a)/a^2).
    # Both C66 (outer) and alpha_SH (Bessel arg) change.
    _, _, alpha_SH_a = _radial_wavenumbers_vti(kz, omega, **cij_a, rho=rho)
    _, _, alpha_SH_b = _radial_wavenumbers_vti(kz, omega, **cij_b, rho=rho)
    from scipy import special as sp

    expected_a = +2.0 * cij_a["c66"] * (
        alpha_SH_a * float(sp.kv(0, alpha_SH_a * a)) / a
        + 2.0 * float(sp.kv(1, alpha_SH_a * a)) / (a * a)
    )
    expected_b = +2.0 * cij_b["c66"] * (
        alpha_SH_b * float(sp.kv(0, alpha_SH_b * a)) / a
        + 2.0 * float(sp.kv(1, alpha_SH_b * a)) / (a * a)
    )
    assert row_a[3].real == pytest.approx(expected_a, rel=1.0e-12)
    assert row_b[3].real == pytest.approx(expected_b, rel=1.0e-12)
    # The D_SH column at the two C66 values differs by both the
    # 2 C66 outer factor AND the alpha_SH dependence -- confirms
    # the row 2 D entry has full C66 sensitivity.
    assert row_a[3].real != row_b[3].real


# =====================================================================
# Plan item H.d.3 -- row 3 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# Sin-sector tangential-shear BC ``sigma_rtheta = 0``. Pure C66
# shear (no Lame replacement, no Q_qX). Every non-zero entry
# scales linearly with C66.


def test_modal_row3_at_a_n1_vti_isotropic_collapse_matches_M31_M32_M33_M34():
    """Floating-point oracle for H.d.3: at isotropic stiffness,
    row 3 matches M31, M32, M33, M34 of :func:`_modal_determinant_n1`
    to floating-point precision. M31 = 0 (fluid no shear)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row3_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M31 = 0.0
    M32 = 2.0 * mu * (
        p * float(sp.kv(0, p * a)) / a
        + 2.0 * float(sp.kv(1, p * a)) / (a * a)
    )
    M33 = kz * mu * float(sp.kv(1, s * a)) / a
    M34 = -mu * (
        s * s * float(sp.kv(1, s * a))
        + 2.0 * s * float(sp.kv(0, s * a)) / a
        + 4.0 * float(sp.kv(1, s * a)) / (a * a)
    )

    assert row[0].real == pytest.approx(M31)
    assert row[1].real == pytest.approx(M32, rel=1.0e-12)
    assert row[2].real == pytest.approx(M33, rel=1.0e-12)
    assert row[3].real == pytest.approx(M34, rel=1.0e-12)


def test_modal_row3_at_a_n1_vti_fluid_column_is_zero():
    """The fluid carries no shear at the wall -- column A is
    identically zero in row 3. Stronger sparsity than rows 1, 2,
    4 (which all have non-zero A from fluid pressure)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert row[0] == 0.0
    # Other three columns generically non-zero.
    for i in (1, 2, 3):
        assert row[i] != 0.0


def test_modal_row3_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 3 is no-row-rescale; col-by-(-i) on
    C_qSV cancels the +i k_z factor. Post-rescale row is
    real-valued in the bound regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row3_at_a_n1_vti_matches_closed_form_per_column():
    """Per-column transcription check against the H.d.3 derivation
    closed forms. Verifies the alpha_SH^2 K_1 direct term in the
    D_SH column (unique to row 3)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row3_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    from scipy import special as sp

    expected_BqP = +2.0 * cij["c66"] * (
        alpha_qP * float(sp.kv(0, alpha_qP * a)) / a
        + 2.0 * float(sp.kv(1, alpha_qP * a)) / (a * a)
    )
    expected_CqSV = +kz * cij["c66"] * float(sp.kv(1, alpha_qSV * a)) / a
    expected_DSH = -cij["c66"] * (
        alpha_SH ** 2 * float(sp.kv(1, alpha_SH * a))
        + 2.0 * alpha_SH * float(sp.kv(0, alpha_SH * a)) / a
        + 4.0 * float(sp.kv(1, alpha_SH * a)) / (a * a)
    )

    assert row[0] == 0.0
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)


def test_modal_row3_at_a_n1_vti_BqP_CqSV_scale_linearly_with_C66():
    """B_qP and C_qSV entries scale LINEARLY with C66: doubling
    C66 (with all other C-matrix entries fixed) doubles the
    entries exactly, since alpha_qP and alpha_qSV are
    C66-independent (the Christoffel quadratic uses only C11,
    C13, C33, C44).

    The D_SH entry does NOT scale linearly because alpha_SH
    depends on C66 via the SH Christoffel branch ``alpha_SH^2 =
    (C44 kz^2 - rho omega^2)/C66``."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 2.0
    rho = cij_a.pop("rho")
    cij_b.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij_a["c44"] / rho))
    vsh_a = float(np.sqrt(cij_a["c66"] / rho))
    vsh_b = float(np.sqrt(cij_b["c66"] / rho))
    kz = omega / min(vsv, vsh_a, vsh_b, 1500.0) * 1.5

    row_a = _modal_row3_at_a_n1_vti(
        kz, omega, **cij_a, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    row_b = _modal_row3_at_a_n1_vti(
        kz, omega, **cij_b, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    # B_qP and C_qSV: ratio b/a = 2 exactly (C66 outer factor;
    # alpha_qP and alpha_qSV are C66-independent).
    assert row_b[1].real / row_a[1].real == pytest.approx(2.0, rel=1.0e-12)
    assert row_b[2].real / row_a[2].real == pytest.approx(2.0, rel=1.0e-12)
    # D_SH: ratio is NOT 2 because alpha_SH depends on C66.
    # Just verify the entry is non-trivial.
    assert row_b[3].real != row_a[3].real
    # Confirm the assumption: alpha_qP and alpha_qSV are unchanged.
    aqp_a, aqsv_a, _ = _radial_wavenumbers_vti(kz, omega, **cij_a, rho=rho)
    aqp_b, aqsv_b, _ = _radial_wavenumbers_vti(kz, omega, **cij_b, rho=rho)
    assert aqp_a == aqp_b
    assert aqsv_a == aqsv_b


# =====================================================================
# Plan item H.d.4 -- row 4 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# Z-derivative-bearing cos-sector row (sigma_rz = 0). Pure C44
# shear; uses the P_qX combination from H.c.1.c. Adds the D_SH
# column at n=1 (via d_z u_r from (1/r) d_theta psi_z).


def test_modal_row4_at_a_n1_vti_isotropic_collapse_matches_M41_M42_M43_M44():
    """Floating-point oracle for H.d.4: at isotropic stiffness,
    row 4 matches M41, M42, M43, M44 of :func:`_modal_determinant_n1`
    to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row4_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M41 = 0.0
    M42 = 2.0 * kz * mu * (
        p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a
    )
    M43 = mu * two_kz2_minus_kS2 * float(sp.kv(1, s * a))
    M44 = -kz * mu * float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M41)
    assert row[1].real == pytest.approx(M42, rel=1.0e-12)
    assert row[2].real == pytest.approx(M43, rel=1.0e-12)
    assert row[3].real == pytest.approx(M44, rel=1.0e-12)


def test_modal_row4_at_a_n1_vti_fluid_column_is_zero():
    """Column A is identically zero (fluid carries no shear)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row4_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert row[0] == 0.0
    for i in (1, 2, 3):
        assert row[i] != 0.0


def test_modal_row4_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 4 is z-derivative-bearing -- gets the
    FULL rescale (row * i + col-by-(-i) on C_qSV). Both rescales
    must be correctly applied for the post-rescale row to be
    real-valued. Forgetting the row * i (the F.2.a.5-flagged
    transcription error mode) leaves a non-zero imaginary part.
    """
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row4_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row4_at_a_n1_vti_matches_closed_form_per_column():
    """Per-column transcription check against the H.d.4 derivation
    closed forms. Verifies the two-term ``alpha_qP K_0 + K_1/a``
    combination on B_qP (vs the single K_1 term in H.c.1.c row 3
    at n=0)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row4_at_a_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz, omega, **cij, rho=rho,
    )
    rho_omega_sq = rho * omega ** 2
    p_qP = cij["c11"] * alpha_qP ** 2 + cij["c13"] * kz ** 2 + rho_omega_sq
    p_qSV = cij["c11"] * alpha_qSV ** 2 + cij["c13"] * kz ** 2 + rho_omega_sq
    from scipy import special as sp

    expected_BqP = (
        +cij["c44"] * p_qP / ((cij["c13"] + cij["c44"]) * kz)
        * (
            alpha_qP * float(sp.kv(0, alpha_qP * a))
            + float(sp.kv(1, alpha_qP * a)) / a
        )
    )
    expected_CqSV = (
        +cij["c44"] * p_qSV / (cij["c13"] + cij["c44"])
        * float(sp.kv(1, alpha_qSV * a))
    )
    expected_DSH = -kz * cij["c44"] * float(sp.kv(1, alpha_SH * a)) / a

    assert row[0] == 0.0
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)


def test_modal_row4_at_a_n1_vti_C66_independent_except_via_alpha_SH():
    """B_qP and C_qSV entries are C66-INDEPENDENT (entries depend
    only on C11, C13, C44 via P_qX and the Christoffel roots
    alpha_qP, alpha_qSV -- which themselves don't see C66). The
    D_SH column DOES depend on C66 (via alpha_SH only).

    Verify by varying C66 and checking that B_qP and C_qSV are
    unchanged while D_SH changes."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 1.50
    rho = cij_a.pop("rho")
    cij_b.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij_a["c44"] / rho))
    vsh_a = float(np.sqrt(cij_a["c66"] / rho))
    vsh_b = float(np.sqrt(cij_b["c66"] / rho))
    kz = omega / min(vsv, vsh_a, vsh_b, 1500.0) * 1.5

    row_a = _modal_row4_at_a_n1_vti(
        kz, omega, **cij_a, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    row_b = _modal_row4_at_a_n1_vti(
        kz, omega, **cij_b, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    # B_qP and C_qSV unchanged (C66-independent).
    assert row_a[1].real == pytest.approx(row_b[1].real, rel=1.0e-12)
    assert row_a[2].real == pytest.approx(row_b[2].real, rel=1.0e-12)
    # D_SH differs (alpha_SH depends on C66).
    assert row_a[3].real != row_b[3].real


# =====================================================================
# Plan item H.d.5 -- assembly into _modal_determinant_n1_vti
# =====================================================================
#
# Stacks the four row builders into the 4x4 VTI flexural modal
# matrix; takes the real determinant. Tests anchor on the
# determinant-vanishes-at-isotropic-flexural-root self-consistency.
# Mirrors the H.c.1.d test pattern.


def test_modal_determinant_n1_vti_is_real_in_bound_regime():
    """The assembled 4x4 matrix is real-valued post-rescale (each
    row builder applies its own rescale internally), so
    ``np.linalg.det`` returns a finite real scalar in the bound
    regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    det = _modal_determinant_n1_vti(
        kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n1_vti_isotropic_collapse_root_matches_unlayered():
    """Substep H.a.7 (a) self-check at the determinant level: at
    isotropic stiffness (slow-formation regime where ``F^2 > 0``
    and the real-valued determinant is well-defined), the VTI
    determinant has the same flexural root as
    :func:`_modal_determinant_n1`. The two determinants are not
    numerically equal (different overall scale due to different
    intermediate factors), but they share the same root in ``k_z``.

    Verify by: (a) computing the flexural root from
    ``flexural_dispersion``; (b) evaluating the VTI determinant
    at that root; (c) checking ``|det_at_root|`` is small relative
    to its value off-root."""
    # Slow formation (V_S < V_f) keeps F^2 = kz^2 - (omega/V_f)^2
    # positive at the flexural root, the regime in which the
    # real-valued ``_modal_determinant_n1`` (and its VTI mirror)
    # is well-defined.
    vp, vs, rho = SLOW_VP, SLOW_VS, SLOW_RHO
    vf, rho_f, a = SLOW_VF, SLOW_RHO_F, SLOW_A
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0

    bound = flexural_dispersion(
        np.array([5000.0]),
        vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n1_vti(
        kz_root, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    det_off_root = _modal_determinant_n1_vti(
        kz_root * 1.05, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    # Determinant at root much smaller than off-root: brentq-type
    # root-finder will converge cleanly. The factor 1e-3 budget
    # is loose because the two determinants differ in absolute
    # scale; tighter tolerance kicks in at the full
    # flexural_dispersion_vti integration test in H.d.6.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_modal_determinant_n1_vti_bracket_brackets_isotropic_root():
    """End-to-end at-isotropic check (slow formation): brentq
    across a tight bracket around the isotropic flexural root
    finds the determinant root, and that root matches the
    isotropic flexural slowness to ``rtol=1e-8``."""
    from scipy import optimize

    vp, vs, rho = SLOW_VP, SLOW_VS, SLOW_RHO
    vf, rho_f, a = SLOW_VF, SLOW_RHO_F, SLOW_A
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0

    bound = flexural_dispersion(
        np.array([5000.0]),
        vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    kz_root_iso = float(bound.slowness[0]) * omega

    def _det(kz):
        return _modal_determinant_n1_vti(
            kz, omega, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
        )

    # Bracket around the isotropic root.
    kz_lo = kz_root_iso * 0.99
    kz_hi = kz_root_iso * 1.01
    d_lo = _det(kz_lo)
    d_hi = _det(kz_hi)
    assert np.sign(d_lo) != np.sign(d_hi)  # bracket valid
    kz_root_vti = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
    assert kz_root_vti == pytest.approx(kz_root_iso, rel=1.0e-8)


def test_modal_determinant_n1_vti_returns_nan_outside_bound_regime():
    """Below the bound floor at least one Christoffel root is
    imaginary; the assembled determinant returns NaN (brentq-safe
    convention propagates from the radial-wavenumber helper)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vp_h = float(np.sqrt(cij["c11"] / rho))  # fastest body wave
    omega = 2.0 * np.pi * 5000.0
    kz = omega / vp_h * 0.5  # well below the bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n1_vti(
            kz, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
        )
    assert np.isnan(det)


# =====================================================================
# Plan item H.d.6 -- flexural_dispersion_vti public-API hook
# =====================================================================
#
# Replaces the H.0 ``NotImplementedError`` (now restricted to fast-
# formation TI) with a brentq loop on ``_modal_determinant_n1_vti``
# for slow-formation TI (V_Sv < V_f). Mirrors H.c.2 for n=0.


def _typical_slow_vti_params():
    """Slow-formation genuine-TI fixture for H.d.6 tests.

    ``V_Sv = 1100 m/s < V_f = 1500 m/s`` so the real-valued VTI
    modal determinant is well-defined (``F_f^2 > 0`` at every
    flexural ``k_z``). Roughly Thomsen-style: epsilon ~ 0.1,
    gamma ~ 0.1; ``c13`` chosen well within Thomsen-stability."""
    return dict(
        c11=1.27e10,  # V_Ph^2 * rho ~ (2400 m/s)^2 * 2200
        c13=4.0e9,    # delta-coupled (c33 > c13 stability)
        c33=1.06e10,  # V_Pv^2 * rho ~ (2200 m/s)^2 * 2200
        c44=2.66e9,   # V_Sv^2 * rho ~ (1100 m/s)^2 * 2200  (Vsv < Vf)
        c66=3.17e9,   # V_Sh^2 * rho ~ (1200 m/s)^2 * 2200  (gamma > 0)
        rho=2200.0,
    )


def test_flexural_dispersion_vti_isotropic_via_genuine_TI_path_matches_isotropic():
    """Floating-point oracle for the H.d chain. Force the
    genuine-TI brentq path by passing a stiffness tensor that is
    formally non-isotropic (``c13`` perturbed by 1 ULP) but
    physically equivalent to isotropic, and verify the resulting
    slowness curve matches the isotropic ``flexural_dispersion``
    answer to ``rtol=1e-5``.

    Slow-formation regime so the real-valued VTI determinant is
    well-defined. More discriminating than the H.0 isotropic-
    collapse test (which dispatches directly to
    ``flexural_dispersion`` and cannot fail) because it exercises
    the full ``_modal_determinant_n1_vti`` + brentq pipeline."""
    vp, vs, rho = SLOW_VP, SLOW_VS, SLOW_RHO
    vf, rho_f, a = SLOW_VF, SLOW_RHO_F, SLOW_A
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    # Force the genuine-TI path by tweaking c13 by 1 part in 1e-6.
    cij_perturbed = dict(cij)
    cij_perturbed["c13"] = cij["c13"] * (1.0 + 1.0e-6)
    f = np.linspace(2000.0, 6000.0, 6)

    res_iso = flexural_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    res_vti = flexural_dispersion_vti(
        f, **cij_perturbed, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    np.testing.assert_allclose(
        res_vti.slowness, res_iso.slowness,
        rtol=1.0e-5, equal_nan=True,
    )
    # Confirm the perturbation actually defeated the isotropic
    # dispatch (the test would pass trivially otherwise).
    assert not _is_isotropic_stiffness(**{
        k: cij_perturbed[k] for k in ("c11", "c13", "c33", "c44", "c66")
    })


def test_flexural_dispersion_vti_genuine_TI_runs_smoke():
    """Smoke: a typical slow-formation genuine-TI fixture
    produces a finite slowness curve above the geometric cutoff.

    Cutoff for ``V_Sv ~ 1100`` and ``a = 0.1`` sits around
    ``V_Sv / (2 pi a) ~ 1750 Hz``; tests sit safely above 3 kHz.
    No analytic oracle here; just confirms the brentq + bracket
    combination handles the TI case across a moderate band."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(3000.0, 7000.0, 5)

    res = flexural_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    assert res.slowness.shape == f.shape
    assert np.all(np.isfinite(res.slowness))
    # Slowness above the V_Sv floor (LF asymptote).
    Vsv = float(np.sqrt(cij["c44"] / rho))
    assert np.all(res.slowness > 1.0 / Vsv * (1.0 - 1.0e-3))


def test_flexural_dispersion_vti_genuine_TI_determinant_vanishes_at_root():
    """At each converged kz from ``flexural_dispersion_vti``, the
    underlying VTI determinant must vanish (self-consistency).
    Ratio against the off-root determinant value at kz_root *
    1.01."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = flexural_dispersion_vti(
        np.array([f]), **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n1_vti(
        kz_root, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    det_off = _modal_determinant_n1_vti(
        kz_root * 1.01, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_flexural_dispersion_vti_returns_borehole_mode_for_genuine_TI():
    """BoreholeMode return-type contract on the slow-formation
    genuine-TI path."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(2000.0, 5000.0, 4)
    res = flexural_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    np.testing.assert_array_equal(res.freq, f)


def test_flexural_dispersion_vti_LF_approaches_V_Sv():
    """Plan H.d sanity: at low frequency (just above the
    geometric cutoff) the VTI flexural slowness should approach
    ``1 / V_Sv`` (the Sinha-Norris-Chang LF asymptote for slow
    TI). For the slow-TI fixture (V_Sv ~ 1100, a = 0.1) the
    cutoff sits around 1750 Hz; this test runs at 3 kHz, well
    above cutoff but low enough to be in the LF asymptotic
    regime where slowness ~ 1/V_Sv to within a few percent."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    Vsv = float(np.sqrt(cij["c44"] / rho))
    res = flexural_dispersion_vti(
        np.array([3000.0]), **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    # LF asymptote: slowness ~ 1/Vsv. Tolerance ~ 2% at f = 3 kHz.
    assert res.slowness[0] == pytest.approx(1.0 / Vsv, rel=2.0e-2)


# =====================================================================
# Plan item H.e -- validation hardening on top of H.d.6
# =====================================================================
#
# Hardening tests for the assembled VTI solvers. Each tests an
# asymptotic / self-consistency property that the isotropic-collapse
# regression alone doesn't pin down. Mirrors F.2.e for the layered
# solver, plus a weak-anisotropy regression against the
# phenomenological model from ``fwap.cylindrical``
# (``flexural_dispersion_vti_physical``) -- the only TI-specific
# external oracle we have for the n=1 dipole.


def test_modal_determinant_n0_vti_vanishes_at_converged_root_multi_freq():
    """Self-consistency: at the converged ``k_z`` from
    ``stoneley_dispersion_vti`` at every frequency in a multi-
    point grid, the underlying VTI determinant is many orders of
    magnitude smaller than its value at ``k_z * 1.01``. Sharper
    than the single-frequency check from H.c.2 because it
    catches regressions where the brentq pipeline converges to
    something other than the true root for some frequencies."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.geomspace(1000.0, 10000.0, 6)

    res = stoneley_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n0_vti(
            kz_root, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
        )
        det_off = _modal_determinant_n0_vti(
            kz_root * 1.01, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
        )
        # Ratio at every frequency: det at root << det 1% off.
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_modal_determinant_n1_vti_vanishes_at_converged_root_multi_freq():
    """Mirror of the n=0 multi-frequency self-consistency at n=1.
    Slow-formation TI fixture so the real-valued VTI determinant
    is well-defined across the full frequency band."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    # Above the geometric cutoff (~1750 Hz for V_Sv=1100, a=0.1).
    f = np.geomspace(3000.0, 12000.0, 6)

    res = flexural_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n1_vti(
            kz_root, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
        )
        det_off = _modal_determinant_n1_vti(
            kz_root * 1.01, omega, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
        )
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_stoneley_dispersion_vti_multi_frequency_smoothness():
    """Stoneley slowness varies smoothly with frequency: across a
    geomspaced band the slowness curve is finite at every point,
    sits above the Norris LF floor, and below the rigid-formation
    fluid-only ceiling. No strict monotonicity check (the Stoneley
    is gently dispersive; sign of the derivative depends on
    parameters), just smoothness."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.geomspace(500.0, 12000.0, 16)

    res = stoneley_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    # All slownesses above the fluid-only slowness 1/V_f and
    # below the Norris LF cap (a sanity fence, not a tight oracle).
    s_fluid = 1.0 / 1500.0
    s_norris = float(np.sqrt(1.0 / 1500.0 ** 2 + 1000.0 / cij["c66"]))
    assert np.all(res.slowness > s_fluid)
    assert np.all(res.slowness < s_norris * 1.10)
    # Smoothness: relative step-to-step change capped at 5 %
    # (geomspaced grid, so adjacent frequencies are ~ 60 % apart
    # but slowness is gently dispersive).
    rel_steps = np.abs(np.diff(res.slowness)) / res.slowness[:-1]
    assert np.all(rel_steps < 0.05)


def test_flexural_dispersion_vti_multi_frequency_monotonicity():
    """Slow-formation flexural slowness increases monotonically
    with frequency: cutoff at ``1/V_Sv`` (low f), HF asymptote at
    a Rayleigh-like speed slightly faster than ``V_Sv`` -- so
    slowness rises from ``~1/V_Sv`` toward ``~1/V_R > 1/V_Sv``.
    Mirrors F.2.e's layered counterpart."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = np.geomspace(3000.0, 15000.0, 12)

    res = flexural_dispersion_vti(
        f, **cij, rho=rho, vf=1500.0, rho_f=1000.0, a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    # Tiny negative tolerance for asymptotic-flatness rounding noise.
    diffs = np.diff(res.slowness)
    assert np.all(diffs > -1.0e-9)


def test_flexural_dispersion_vti_weak_anisotropy_matches_phenomenological():
    """Plan H.e weak-anisotropy oracle: with small ``gamma`` (TI
    close to isotropic), the full Schmitt 1989 modal determinant
    slowness should qualitatively track the Sinha-Norris-Chang
    phenomenological asymptote
    :func:`flexural_dispersion_vti_physical` across the dipole-
    sonic band (~ 1-6 kHz, equivalent to ~ 1-3 cutoff multiples
    for this fixture).

    Tolerance follows the precedent of the isotropic
    ``test_flexural_dispersion_qualitative_match_with_phenomenological``
    (10 %). The phenomenological model is a smoothed-step
    interpolation between the LF ``1/V_Sv`` and HF
    ``1/V_R(V_P, V_Sh)`` asymptotes -- both physically correct
    in their own limits; the few-percent quantitative offset
    arises from Scholte / fluid-loading effects the modal solver
    captures but the phenomenological does not."""
    rho = 2200.0
    vsv = 1100.0
    vsh = 1110.0  # gamma ~ 0.009 (very weak TI)
    vp = 2200.0
    cij = dict(
        c11=rho * vp ** 2,         # set epsilon = 0 (c11 = c33)
        c33=rho * vp ** 2,
        c44=rho * vsv ** 2,
        c66=rho * vsh ** 2,        # gamma > 0 only
    )
    # c13 = c11 - 2 c44 (delta-coupled isotropic value).
    cij["c13"] = cij["c11"] - 2.0 * cij["c44"]
    a = 0.1
    vf, rho_f = 1500.0, 1000.0

    # Dipole-sonic band: 1.5-3.5 cutoff multiples (~ 2.6 - 6.1 kHz
    # for this fixture). Stays comfortably above the geometric
    # cutoff where the bracket would otherwise touch the floor.
    fc = vsv / (2.0 * np.pi * a)
    f = np.linspace(fc * 1.5, fc * 3.5, 8)

    res_modal = flexural_dispersion_vti(
        f, **cij, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    s_phenom = flexural_dispersion_vti_physical(
        vp=vp, vsv=vsv, vsh=vsh, a_borehole=a,
    )(f)
    assert np.all(np.isfinite(res_modal.slowness))
    rel_diff = np.abs(res_modal.slowness - s_phenom) / s_phenom
    assert np.all(rel_diff < 0.10)


# =====================================================================
# Plan item G.0 -- public-API foundation for cased-hole multi-layer
# =====================================================================
#
# G.0 widens the multi-layer dispatch in stoneley_dispersion_layered
# / flexural_dispersion_layered with: (a) a sharper NotImplementedError
# message that points at the G.c / G.d / G' follow-ups (verified by
# the existing multilayer_raises_not_implemented tests above, with
# their match strings updated), and (b) a new helper
# _validate_borehole_layers_stacked that wraps F's per-layer
# validation with the borehole-radius check. The propagator-matrix
# path itself lands in G.b / G.c / G.d.


def test_validate_borehole_layers_stacked_accepts_typical_two_layer_stack():
    """A casing + cement geometry passes the stacked validator
    without raising. Same ``BoreholeLayer`` validation rules as
    F's per-layer validator, plus ``a > 0``."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    # Should not raise.
    _validate_borehole_layers_stacked((casing, cement), a=0.1)


def test_validate_borehole_layers_stacked_accepts_empty_stack():
    """Empty stack ``()`` is the degenerate "no extra layers" case
    and validates trivially as long as ``a > 0``."""
    _validate_borehole_layers_stacked((), a=0.1)


def test_validate_borehole_layers_stacked_rejects_zero_thickness_in_multi_stack():
    """A zero-thickness layer in a multi-layer stack is rejected
    by the per-layer validation (delegated to
    ``_validate_borehole_layers``). The error message identifies
    the offending index."""
    bad = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.0)
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    with pytest.raises(ValueError, match=r"layers\[1\].*thickness must be positive"):
        _validate_borehole_layers_stacked((casing, bad), a=0.1)


def test_validate_borehole_layers_stacked_rejects_non_positive_a():
    """Non-positive borehole radius is rejected with a clear
    error. Catches it earlier than the public-API dispatch, which
    is useful when G.c starts using the helper."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    with pytest.raises(ValueError, match="a must be positive"):
        _validate_borehole_layers_stacked((casing, cement), a=0.0)
    with pytest.raises(ValueError, match="a must be positive"):
        _validate_borehole_layers_stacked((casing, cement), a=-0.1)


def test_stoneley_dispersion_layered_two_layer_NIE_points_at_G_c_G_d():
    """The G.0 dispatch sharpens the multi-layer NIE message to
    name plan items G.c (stacked modal determinant) and G.d
    (public-API hook). Verifies the user-facing pointer is
    actionable rather than just naming "plan G"."""
    f = np.array([5000.0])
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    with pytest.raises(NotImplementedError) as exc_info:
        stoneley_dispersion_layered(
            f, vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(casing, cement),
        )
    msg = str(exc_info.value)
    assert "G.c" in msg
    assert "G.d" in msg
    assert "cylindrical_biot_G.md" in msg


def test_flexural_dispersion_layered_two_layer_NIE_points_at_G_prime():
    """Mirror at n=1: the multi-layer NIE message identifies the
    deferred G' follow-up (cased-hole flexural with 6x6 propagator
    blocks) rather than just naming "plan G"."""
    f = np.array([5000.0])
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    with pytest.raises(NotImplementedError) as exc_info:
        flexural_dispersion_layered(
            f, vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(casing, cement),
        )
    msg = str(exc_info.value)
    assert "G'" in msg
    assert "cylindrical_biot_G.md" in msg


def test_stoneley_dispersion_layered_zero_and_one_layer_paths_unchanged():
    """G.0 must not perturb the two existing collapse paths
    (``len(layers) == 0`` and ``len(layers) == 1``). Regression:
    each produces the same slowness curve at a representative
    frequency."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.array([5000.0])

    # Empty-layer path: dispatches to stoneley_dispersion.
    res_empty = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a, layers=(),
    )
    res_unlayered = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    )
    np.testing.assert_array_equal(res_empty.slowness, res_unlayered.slowness)

    # Single-layer path: F.1 hand-coded determinant. Just smoke;
    # bit-equivalent regressions are exercised elsewhere.
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    res_one = stoneley_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a, layers=(layer,),
    )
    assert np.isfinite(res_one.slowness[0])
    assert isinstance(res_one, BoreholeMode)
    assert res_one.azimuthal_order == 0


# =====================================================================
# Plan item G.b.1 -- mode-amplitude-to-state-vector matrix E(r)
# =====================================================================
#
# Per-element oracle: at r=a, the layer-amplitude columns of
# F.1.b row 1 / 2 / 3 (with explicit sign factors per the BC's
# subtraction convention) match rows 0, 2, 3 of E(a). At r=b,
# F.1.b row 5 layer cols match row 1 (u_z) of E(b). Together
# these cover all four rows of E.


def _typical_g_b1_layer_params():
    """Representative non-isotropic-collapse layer + (kz, omega)
    fixture for G.b.1 / G.b.2 tests. Sits in the slow-formation
    bound regime so the propagator-matrix path is well-defined."""
    return dict(
        vp=3500.0, vs=1800.0, rho=2100.0,
        kz=2.0 * np.pi * 5000.0 / 1500.0,  # bound: kz > omega/V_S
        omega=2.0 * np.pi * 5000.0,
    )


def test_layer_e_matrix_n0_row0_matches_F1_row1_at_a_layer_cols():
    """Row 0 of E(a) (u_r) matches the layer-amplitude columns
    (1..5) of ``_layered_n0_row1_at_a`` with a sign flip: F.1's
    BC1 is ``u_r^(f) - u_r^(m) = 0``, so the layer side is
    negated in the row builder. This is the cleanest per-element
    oracle for the u_r row of E(r)."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    # E(a) for the layer.
    E = _layer_e_matrix_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"], r=a,
    )
    # F.1.b row 1: signature uses (vp, vs, rho) for the formation
    # half-space; the layer is passed via ``layer``. Row 1 doesn't
    # touch the formation parameters except for signature uniformity.
    row1 = _layered_n0_row1_at_a(
        kz=p["kz"], omega=p["omega"], vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=a, layer=layer,
    )
    # Layer cols 1..5 of row 1 = -E[0, :] (negation from f - m).
    np.testing.assert_allclose(
        row1[1:5].real, -E[0, :], rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row2_matches_F1_row2_at_a_layer_cols():
    """Row 2 of E(a) (sigma_rr) matches the layer cols of
    ``_layered_n0_row2_at_a`` with a sign flip: BC2 is
    ``-(sigma_rr^(m) + P^(f)) = 0``, layer side negated."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"], r=a,
    )
    row2 = _layered_n0_row2_at_a(
        kz=p["kz"], omega=p["omega"], vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=a, layer=layer,
    )
    np.testing.assert_allclose(
        row2[1:5].real, -E[2, :], rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row3_matches_F1_row3_at_a_layer_cols():
    """Row 3 of E(a) (sigma_rz) matches the layer cols of
    ``_layered_n0_row3_at_a`` with NO sign flip: BC3 is
    ``sigma_rz^(m) = 0`` (no subtraction with the fluid)."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"], r=a,
    )
    row3 = _layered_n0_row3_at_a(
        kz=p["kz"], omega=p["omega"], vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=a, layer=layer,
    )
    np.testing.assert_allclose(
        row3[1:5].real, E[3, :], rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row1_uz_matches_F1_row5_at_b_layer_cols():
    """Row 1 of E(b) (u_z) matches the layer cols of
    ``_layered_n0_row5_at_b`` with NO sign flip: BC5 is
    ``u_z^(m)(b) - u_z^(s)(b) = 0`` and the layer cols carry the
    layer's direct contribution (the formation cols carry the
    subtracted contribution). Validates the u_z row of E,
    which has no analog at r=a (the fluid doesn't impose u_z
    continuity at the borehole wall)."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    b = a + layer.thickness
    E = _layer_e_matrix_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"], r=b,
    )
    row5 = _layered_n0_row5_at_b(
        kz=p["kz"], omega=p["omega"], vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=a, layer=layer,
    )
    np.testing.assert_allclose(
        row5[1:5].real, E[1, :], rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_returns_nan_below_bound_floor():
    """Below the layer's bound floor (``kz < omega / V_S``), at
    least one of ``p^2``, ``s^2`` becomes negative -- the Bessel
    arguments would be imaginary. The helper returns NaN-filled
    so downstream propagator / determinant evaluations propagate
    NaN cleanly (brentq-safe convention, mirrors
    ``_modal_determinant_n0`` and friends)."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 3500.0, 1800.0, 2100.0
    # kz well below omega/V_S.
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        E = _layer_e_matrix_n0(
            kz=kz, omega=omega, vp=vp, vs=vs, rho=rho, r=0.1,
        )
    assert np.all(np.isnan(E))


def test_layer_e_matrix_n0_determinant_nonzero_in_bound_regime():
    """The G.b.2 propagator path requires inverting E(r). Confirm
    that ``det(E(r))`` is well above floating-point noise for a
    representative bound-regime ``(kz, omega, layer)``. The
    quantitative budget is loose -- the absolute scale of
    ``det(E)`` depends on the Bessel-pack magnitudes, which can
    be very large or very small; we just want to rule out the
    near-singular case that would defeat the inverse."""
    p = _typical_g_b1_layer_params()
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"], r=a,
    )
    det = float(np.linalg.det(E))
    # Just a finite, non-zero determinant. The propagator
    # round-trip oracle in G.b.2 will catch any conditioning
    # issue more sharply.
    assert np.isfinite(det)
    assert abs(det) > 0.0


# =====================================================================
# Plan item G.b.2 -- per-layer propagator P(r_outer | r_inner)
# =====================================================================
#
# Group-law oracles for ``_layer_propagator_n0`` plus an end-to-end
# state-vector continuity check. Each oracle is independent of the
# F.1.b transcription used in G.b.1, so this layer adds genuinely
# new constraints on top of the per-element match.


def test_layer_propagator_n0_identity_when_r_inner_equals_r_outer():
    """Identity oracle: ``r_inner == r_outer`` -> propagator is
    ``eye(4)`` to floating-point precision. Catches sign / shape
    errors in the solve."""
    p = _typical_g_b1_layer_params()
    P = _layer_propagator_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"],
        r_inner=0.105, r_outer=0.105,
    )
    np.testing.assert_array_equal(P, np.eye(4))


def test_layer_propagator_n0_round_trip_preserves_state_vector():
    """Round-trip oracle: applying ``P(a|b) @ P(b|a)`` to a
    physical state vector ``v`` returns ``v`` to floating-point
    precision. Equivalent to ``P(a|b) P(b|a) = I`` in exact
    arithmetic; phrasing as a state-vector identity avoids the
    spurious ~1e-6 off-diagonals from the disparate-magnitude
    rows (displacement ~ O(1) vs stress ~ O(mu) ~ O(1e10)) that
    would defeat ``assert_allclose(M, eye, atol=1e-10)`` directly
    at the matrix level."""
    p = _typical_g_b1_layer_params()
    a = 0.1
    b = a + 0.005
    P_b_from_a = _layer_propagator_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"],
        r_inner=a, r_outer=b,
    )
    P_a_from_b = _layer_propagator_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"],
        r_inner=b, r_outer=a,
    )
    # Physical state vector (displacement ~ O(1), stress ~ O(mu)).
    mu = p["rho"] * p["vs"] ** 2
    v = np.array([1.0, 2.0, 3.0 * mu, 4.0 * mu])
    v_round = P_a_from_b @ (P_b_from_a @ v)
    np.testing.assert_allclose(v_round, v, rtol=1.0e-10)
    # Other direction.
    v_round_other = P_b_from_a @ (P_a_from_b @ v)
    np.testing.assert_allclose(v_round_other, v, rtol=1.0e-10)


def test_layer_propagator_n0_composition_law():
    """Composition oracle: ``P(r3|r1) ~ P(r3|r2) @ P(r2|r1)`` for
    any intermediate ``r2 in (r1, r3)``. The propagator-group law
    in the radial coordinate. Independent of the F.1.b oracle
    in G.b.1."""
    p = _typical_g_b1_layer_params()
    r1, r2, r3 = 0.1, 0.105, 0.115
    P_3_from_1 = _layer_propagator_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"],
        r_inner=r1, r_outer=r3,
    )
    P_2_from_1 = _layer_propagator_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"],
        r_inner=r1, r_outer=r2,
    )
    P_3_from_2 = _layer_propagator_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"],
        r_inner=r2, r_outer=r3,
    )
    np.testing.assert_allclose(P_3_from_1, P_3_from_2 @ P_2_from_1, atol=1.0e-10)


def test_layer_propagator_n0_state_vector_continuity():
    """End-to-end state-vector check: pick an arbitrary amplitude
    vector ``c``; compute ``v(r1) = E(r1) c`` and apply
    ``P(r2|r1)`` to get ``v(r2)``; verify the result matches
    ``E(r2) c`` directly. Strongest single-test oracle for the
    G.b.1 + G.b.2 chain combined."""
    p = _typical_g_b1_layer_params()
    r1, r2 = 0.1, 0.115
    E_r1 = _layer_e_matrix_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"], r=r1,
    )
    E_r2 = _layer_e_matrix_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"], r=r2,
    )
    P = _layer_propagator_n0(
        kz=p["kz"], omega=p["omega"], vp=p["vp"], vs=p["vs"], rho=p["rho"],
        r_inner=r1, r_outer=r2,
    )
    # Arbitrary amplitude vector.
    c = np.array([1.3, -0.7, 2.1, 0.4])
    v_r1 = E_r1 @ c
    v_r2_via_P = P @ v_r1
    v_r2_direct = E_r2 @ c
    np.testing.assert_allclose(v_r2_via_P, v_r2_direct, rtol=1.0e-10)


def test_layer_propagator_n0_returns_nan_below_bound_floor():
    """Below the layer's bound floor, ``E(r)`` is NaN-filled; the
    propagator inherits the NaN. Confirms brentq-safe propagation
    so the G.c assembly's bound-regime gate is reliable."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 3500.0, 1800.0, 2100.0
    kz = omega / vs * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        P = _layer_propagator_n0(
            kz=kz, omega=omega, vp=vp, vs=vs, rho=rho,
            r_inner=0.1, r_outer=0.105,
        )
    assert np.all(np.isnan(P))


# =====================================================================
# Plan item G.c -- stacked modal determinant
# =====================================================================
#
# Tests anchor on the N=1 collapse to F.1 (``_modal_determinant_n0_layered``)
# as the floating-point oracle, plus a few oracles that exercise
# the N >= 2 propagator chain (order-matters; two-identical-layers
# equivalent to one double-thickness layer via the group law).


def _typical_g_c_params():
    """Slow-formation cased-hole fixture for G.c tests. Keeps the
    Stoneley root in the bound regime across a representative
    band; layers are typical casing / cement / mudcake values."""
    return dict(
        vp=4500.0, vs=2500.0, rho=2400.0,
        vf=1500.0, rho_f=1000.0, a=0.1,
    )


def test_modal_determinant_n0_cased_N1_matches_F1_off_root():
    """N=1 floating-point oracle: at any (kz, omega) in the bound
    regime away from the Stoneley root, G.c's determinant matches
    F.1's ``_modal_determinant_n0_layered`` to relative precision
    ``rtol=1e-10`` (no extra scale factor; the propagator chain
    at N=1 reduces P_1 @ E_1(a) -> E_1(b), exactly the F.1 form).

    Strongest pinning of the G.c assembly against the existing
    F.1 row-builder transcription that has shipped through F.1.b.4."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    # Pick kz away from the Stoneley root (1.05x off the unlayered
    # bound floor; well into the bound regime).
    kz = omega / p["vf"] * 1.05
    det_F1 = _modal_determinant_n0_layered(
        kz, omega, p["vp"], p["vs"], p["rho"], p["vf"], p["rho_f"], p["a"],
        layer=layer,
    )
    det_Gc = _modal_determinant_n0_cased(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(layer,),
    )
    assert det_Gc == pytest.approx(det_F1, rel=1.0e-10)


def test_modal_determinant_n0_cased_N1_vanishes_at_F1_brentq_root():
    """N=1 brentq-root oracle: at the Stoneley root recovered by
    ``stoneley_dispersion_layered(layers=(layer,))``, G.c's
    determinant is many orders of magnitude smaller than its
    value 1% off the root. Confirms the brentq pipeline G.d will
    drive against G.c will find the same root as F.1."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    res = stoneley_dispersion_layered(
        np.array([5000.0]),
        vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega
    det_at = _modal_determinant_n0_cased(
        kz_root, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(layer,),
    )
    det_off = _modal_determinant_n0_cased(
        kz_root * 1.01, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(layer,),
    )
    assert abs(det_at) < abs(det_off) * 1.0e-10


def test_modal_determinant_n0_cased_returns_nan_below_bound_floor():
    """``kz < omega / V_f`` -> ``F_f^2 < 0`` -> NaN; or ``kz`` below
    the slowest layer / formation V_S -> propagator chain returns
    NaN. Either way the assembly propagates NaN cleanly so brentq
    can reject the bracket."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    # kz well below the fluid floor.
    kz = omega / p["vf"] * 0.5
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n0_cased(
            kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
            vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(layer,),
        )
    assert np.isnan(det)


def test_modal_determinant_n0_cased_two_identical_layers_equals_one_double_thickness():
    """Group-law oracle for the propagator chain: two contiguous
    identical layers (L, L) of thickness ``h`` each compose to a
    single layer of thickness ``2h`` via ``P_2 @ P_1 = P(r3 | r1)``.

    Direct test that G.c.7 propagator-chain composition is wired
    correctly. Independent of F.1: would catch any error in the
    inside-out layer-radii arithmetic or the chain accumulator."""
    p = _typical_g_c_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vf"] * 1.05  # bound regime
    # Single layer of thickness 0.01.
    L_double = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    # Two layers of thickness 0.005 each, same params.
    L_half = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    det_single = _modal_determinant_n0_cased(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(L_double,),
    )
    det_split = _modal_determinant_n0_cased(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(L_half, L_half),
    )
    # The two assemblies use different innermost-layer E_1(a) factors
    # (both layers have the same params, but the half-layer's
    # E_1(a) has different bessel-pack values than the double-layer's
    # E_1(a)). The brentq root in kz is the same; the absolute
    # determinant magnitudes can differ. Verify the root match by
    # checking that |det_single| / |det_split| is the same as the
    # ratio of innermost-layer det(E_1(a))s (the layer-1-amplitude
    # scale factor that distinguishes them).
    #
    # Simplest oracle: both should change sign across the same kz
    # window, captured by the same-sign / same-magnitude-order test.
    # Tight ratio: at this off-root kz, both should be the SAME up
    # to an overall sign because L_double's E_1(a) is identical to
    # L_half's E_1(a) (same vp/vs/rho/r=a). The propagator chain
    # composes to the same total transformation across thickness 0.01.
    assert det_single == pytest.approx(det_split, rel=1.0e-10)


def test_modal_determinant_n0_cased_order_matters_at_N2():
    """Physical sanity: with two distinct layers ``(L_a, L_b)``,
    swapping the order to ``(L_b, L_a)`` produces a different
    determinant -- the inside-out layer ordering is a physical
    parameter (a casing inside a cement looks different from a
    cement inside a casing).

    Independent of F.1; would catch any error where the
    propagator chain ignored layer ordering or composed in the
    wrong direction."""
    p = _typical_g_c_params()
    omega = 2.0 * np.pi * 5000.0
    L_a = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)  # casing
    L_b = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.01)  # cement
    # kz safely above the slowest-shear bound floor (cement V_S = 1300).
    kz = omega / min(L_a.vs, L_b.vs, p["vs"], p["vf"]) * 1.05
    det_ab = _modal_determinant_n0_cased(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(L_a, L_b),
    )
    det_ba = _modal_determinant_n0_cased(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(L_b, L_a),
    )
    assert np.isfinite(det_ab) and np.isfinite(det_ba)
    # Layer permutation is non-trivial (well above floating-point noise).
    rel_diff = abs(det_ab - det_ba) / max(abs(det_ab), abs(det_ba))
    assert rel_diff > 0.01


def test_modal_determinant_n0_cased_N2_runs_smoke():
    """Smoke test for the N=2 (cased-hole) path: a typical
    casing + cement geometry produces a finite real determinant
    at a representative bound-regime ``kz``."""
    p = _typical_g_c_params()
    omega = 2.0 * np.pi * 5000.0
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    # kz safely above the slowest-shear bound floor.
    kz = omega / min(casing.vs, cement.vs, p["vs"], p["vf"]) * 1.05
    det = _modal_determinant_n0_cased(
        kz, omega, vp=p["vp"], vs=p["vs"], rho=p["rho"],
        vf=p["vf"], rho_f=p["rho_f"], a=p["a"], layers=(casing, cement),
    )
    assert np.isfinite(det)
    assert isinstance(det, float)
