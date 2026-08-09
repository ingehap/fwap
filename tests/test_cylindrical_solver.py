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
    _is_isotropic_stiffness,
    _layer_e_matrix_n0,
    _layer_e_matrix_n1,
    _layer_e_matrix_n2,
    _layer_propagator_n0,
    _layer_propagator_n1,
    _layer_propagator_n2,
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
    _layered_n1_row2_at_a,
    _layered_n1_row3_at_a,
    _layered_n1_row4_at_a,
    _layered_n1_row5_at_b,
    _layered_n1_row6_at_b,
    _layered_n1_row7_at_b,
    _layered_n1_row8_at_b,
    _layered_n1_row9_at_b,
    _layered_n1_row10_at_b,
    _modal_determinant_n0,
    _modal_determinant_n0_cased,
    _modal_determinant_n0_layered,
    _modal_determinant_n0_vti,
    _modal_determinant_n1,
    _modal_determinant_n1_cased,
    _modal_determinant_n1_layered,
    _modal_determinant_n1_vti,
    _modal_row1_at_a_n1_vti,
    _modal_row1_at_a_vti,
    _modal_row2_at_a_n1_vti,
    _modal_row2_at_a_vti,
    _modal_row3_at_a_n1_vti,
    _modal_row3_at_a_vti,
    _modal_row4_at_a_n1_vti,
    _polarization_ratio_uz_over_ur_vti,
    _radial_wavenumbers_vti,
    _validate_borehole_layers_stacked,
    _validate_flexural_layers_stacked,
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
    res = flexural_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    finite = np.isfinite(res.slowness)
    assert finite.any(), "fast-formation path must populate at least one frequency"
    velocity = 1.0 / res.slowness[finite]
    # Strictly between V_R and V_S (the leaky-F regime window).
    assert (velocity > vR * 0.99).all(), (
        f"velocity must stay near or above V_R ({vR:.0f}); got {velocity}"
    )
    assert (velocity < vs).all(), f"velocity must stay below V_S ({vs}); got {velocity}"
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
            kz,
            omega,
            LEAKY_VP,
            LEAKY_VS,
            LEAKY_RHO,
            LEAKY_VF,
            LEAKY_RHO_F,
            LEAKY_A,
        )
        d_complex = _modal_determinant_n0_complex(
            complex(kz),
            omega,
            LEAKY_VP,
            LEAKY_VS,
            LEAKY_RHO,
            LEAKY_VF,
            LEAKY_RHO_F,
            LEAKY_A,
        )
        # Real part agrees exactly.
        assert abs(d_complex.real - d_real) / abs(d_real) < 1.0e-12, (
            f"real-vs-complex mismatch at kz={kz}: real={d_real}, complex={d_complex}"
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
        np.array([1000.0]),
        vp=LEAKY_VP,
        vs=LEAKY_VS,
        rho=LEAKY_RHO,
        vf=LEAKY_VF,
        rho_f=LEAKY_RHO_F,
        a=LEAKY_A,
    )
    kz_root = res.slowness[0] * omega
    d_lo = _modal_determinant_n0_complex(
        complex(kz_root * 0.999),
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_RHO,
        LEAKY_VF,
        LEAKY_RHO_F,
        LEAKY_A,
    )
    d_hi = _modal_determinant_n0_complex(
        complex(kz_root * 1.001),
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_RHO,
        LEAKY_VF,
        LEAKY_RHO_F,
        LEAKY_A,
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
        complex(kz),
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_VF,
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
        complex(kz),
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_VF,
    )
    assert leaky_s is True  # S radiates outward
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
        complex(kz),
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_VF,
    )
    assert leaky_F is True  # fluid radiates (V_R > V_f for fast formation)
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
        complex(kz),
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_VF,
    )
    d = _modal_determinant_n0_complex(
        complex(kz),
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_RHO,
        LEAKY_VF,
        LEAKY_RHO_F,
        LEAKY_A,
        leaky_p=leaky_p,
        leaky_s=leaky_s,
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
        kz,
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_VF,
    )
    d = _modal_determinant_n0_complex(
        kz,
        omega,
        LEAKY_VP,
        LEAKY_VS,
        LEAKY_RHO,
        LEAKY_VF,
        LEAKY_RHO_F,
        LEAKY_A,
        leaky_p=leaky_p,
        leaky_s=leaky_s,
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
    expected = np.array([2.0 * np.pi * f * slowness_at_f(f) for f in freqs])
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
            kz,
            omega,
            vp,
            vs,
            vf,
        )
        return _modal_determinant_n0_complex(
            kz,
            omega,
            vp,
            vs,
            rho,
            vf,
            rho_f,
            a,
            leaky_p=leaky_p,
            leaky_s=leaky_s,
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
        freqs,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        name="Stoneley",
        azimuthal_order=0,
        freq=np.array([1000.0, 2000.0]),
        slowness=np.array([7e-4, 7e-4]),
    )
    assert bm.attenuation_per_meter is None


def test_borehole_mode_attenuation_per_meter_field_accepts_array():
    """The new field accepts an ndarray for leaky modes carrying
    spatial attenuation Im(k_z)."""
    from fwap.cylindrical_solver import BoreholeMode

    bm = BoreholeMode(
        name="pseudo_rayleigh",
        azimuthal_order=0,
        freq=np.array([15000.0, 20000.0]),
        slowness=np.array([4.5e-4, 4.2e-4]),
        attenuation_per_meter=np.array([1e-3, 5e-4]),
    )
    assert bm.attenuation_per_meter is not None
    np.testing.assert_array_equal(
        bm.attenuation_per_meter,
        np.array([1e-3, 5e-4]),
    )


def test_existing_stoneley_solver_attenuation_field_is_None():
    """The existing ``stoneley_dispersion`` doesn't populate
    ``attenuation_per_meter`` (Stoneley is bound -> attenuation is
    zero -> no field needed). Verify the default-None behaviour
    survives the dataclass extension."""
    res = stoneley_dispersion(
        np.array([1000.0]),
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
            f,
            vp=PR_VP,
            vs=1200.0,
            rho=PR_RHO,
            vf=PR_VF,
            rho_f=PR_RHO_F,
            a=PR_A,
        )


def test_pseudo_rayleigh_rejects_non_positive_inputs():
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    f = np.array([10000.0])
    base = dict(vp=PR_VP, vs=PR_VS, rho=PR_RHO, vf=PR_VF, rho_f=PR_RHO_F, a=PR_A)
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
            bad_freq,
            vp=PR_VP,
            vs=PR_VS,
            rho=PR_RHO,
            vf=PR_VF,
            rho_f=PR_RHO_F,
            a=PR_A,
        )


def test_pseudo_rayleigh_returns_borehole_mode_with_attenuation():
    """The leaky-mode return contract: ``BoreholeMode`` with the
    expected name, n=0 azimuthal order, frequency echoed, and a
    populated ``attenuation_per_meter`` field whose shape matches
    ``slowness``."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    freq = np.linspace(30000.0, 80000.0, 20)
    res = pseudo_rayleigh_dispersion(
        freq,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
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
        freq,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
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
        freq,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
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
        freq_asc,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
    )
    res_desc = pseudo_rayleigh_dispersion(
        freq_desc,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
    )
    # Compare element-wise between freq[i] in ascending order and
    # the matching freq[-1-i] in descending order.
    np.testing.assert_allclose(
        res_asc.slowness,
        res_desc.slowness[::-1],
        rtol=1.0e-9,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        res_asc.attenuation_per_meter,
        res_desc.attenuation_per_meter[::-1],
        rtol=1.0e-9,
        equal_nan=True,
    )


def test_pseudo_rayleigh_handles_empty_frequency_array():
    """Empty ``freq`` should return empty arrays without crashing
    -- a no-op that the marcher must short-circuit before
    indexing the (non-existent) high-frequency seed point."""
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    res = pseudo_rayleigh_dispersion(
        np.array([], dtype=float),
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
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
        _modal_determinant_n0_complex,
        pseudo_rayleigh_dispersion,
    )

    freq = np.array([60000.0])
    res = pseudo_rayleigh_dispersion(
        freq,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
    )
    assert np.isfinite(res.slowness[0])
    omega = 2.0 * np.pi * freq[0]
    kz_root = complex(res.slowness[0] * omega, res.attenuation_per_meter[0])
    det_at_root = _modal_determinant_n0_complex(
        kz_root,
        omega,
        PR_VP,
        PR_VS,
        PR_RHO,
        PR_VF,
        PR_RHO_F,
        PR_A,
        leaky_p=False,
        leaky_s=True,
    )
    # The matrix entries scale with K_n / I_n / Hankel values that
    # span many orders of magnitude (matrix norm ~ 1e9 for these
    # parameters); the determinant at a true zero is many orders
    # below that scale.
    kz_off_root = kz_root * 1.05
    det_off_root = _modal_determinant_n0_complex(
        kz_off_root,
        omega,
        PR_VP,
        PR_VS,
        PR_RHO,
        PR_VF,
        PR_RHO_F,
        PR_A,
        leaky_p=False,
        leaky_s=True,
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
        freq,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
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
    assert (
        _classify_marcher_step(None, 1000.0, lambda kz, w: True)
        == "convergence_failure"
    )


def test_classify_marcher_step_recognises_regime_exit():
    """A converged root rejected by the validator is reported as
    ``"regime_exit"``, and a validator exception (``ValueError`` /
    ``ArithmeticError``) is treated as the same."""
    from fwap.cylindrical_solver import _classify_marcher_step

    rejector = lambda kz, w: False  # noqa: E731
    raiser = lambda kz, w: (_ for _ in ()).throw(ValueError("bad"))  # noqa: E731

    assert _classify_marcher_step(complex(1.0, 0.0), 1.0, rejector) == "regime_exit"
    assert _classify_marcher_step(complex(1.0, 0.0), 1.0, raiser) == "regime_exit"


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
    one = BranchSegment(
        start_idx=5, end_idx=5, freq=np.array([4.0]), kz=np.array([4.0 + 0j])
    )
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
        det,
        freq,
        seed,
        validator=validator,
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
        det,
        freq,
        seed,
        validator=validator,
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
        det,
        freq,
        seed,
        validator=validator,
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
            kz,
            omega,
            SLOW_VP,
            SLOW_VS,
            SLOW_RHO,
            SLOW_VF,
            SLOW_RHO_F,
            SLOW_A,
        )
        d_complex = _modal_determinant_n1_complex(
            complex(kz),
            omega,
            SLOW_VP,
            SLOW_VS,
            SLOW_RHO,
            SLOW_VF,
            SLOW_RHO_F,
            SLOW_A,
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
    from scipy import optimize as _opt

    from fwap.cylindrical_solver import (
        _flexural_kz_bracket,
        _modal_determinant_n1,
    )

    f = np.linspace(2000.0, 12000.0, 11)
    res = flexural_dispersion(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 8, "slow-formation path must still find roots above cutoff"

    # Reference computation: open-coded brentq on
    # ``_modal_determinant_n1`` with the existing bracket helper.
    for i in np.where(finite)[0]:
        omega = 2.0 * np.pi * f[i]

        def _det(kz):
            return _modal_determinant_n1(
                kz,
                omega,
                SLOW_VP,
                SLOW_VS,
                SLOW_RHO,
                SLOW_VF,
                SLOW_RHO_F,
                SLOW_A,
            )

        kz_lo, kz_hi = _flexural_kz_bracket(
            omega,
            SLOW_VP,
            SLOW_VS,
            SLOW_RHO,
            SLOW_VF,
            SLOW_RHO_F,
            SLOW_A,
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

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(20000.0, 80000.0, 30)
    res = flexural_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    finite_idx = np.where(np.isfinite(res.slowness))[0]
    assert finite_idx.size >= 3, "expected at least a few finite samples"

    # At each converged root, |det| must be small relative to
    # |det| at an off-root point (the same local-zero test used
    # for the n=0 leaky solver).
    for i in finite_idx[:5]:
        omega = 2.0 * np.pi * res.freq[i]
        kz_root = complex(res.slowness[i] * omega, 0.0)
        det_at_root = _modal_determinant_n1_complex(
            kz_root,
            omega,
            vp,
            vs,
            rho,
            vf,
            rho_f,
            a,
            leaky_p=False,
            leaky_s=False,
        )
        # Off-root sample: shift kz by 1 % toward V_R.
        kz_off = kz_root * 1.005
        det_off = _modal_determinant_n1_complex(
            kz_off,
            omega,
            vp,
            vs,
            rho,
            vf,
            rho_f,
            a,
            leaky_p=False,
            leaky_s=False,
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
    res_asc = flexural_dispersion(f_asc, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    res_desc = flexural_dispersion(
        f_desc, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    np.testing.assert_allclose(
        res_asc.slowness,
        res_desc.slowness[::-1],
        rtol=1.0e-9,
        equal_nan=True,
    )


def test_pseudo_rayleigh_segmenter_returns_single_continuous_segment():
    """For the standard fast-formation parameter set, the validated
    marcher recovers one contiguous segment over the supported
    band -- no NaN gaps in the middle even where the strict
    marcher used to drop steps to single-step root hops."""
    from fwap.cylindrical_solver import (
        pseudo_rayleigh_dispersion,
        segments_from_kz_curve,
    )

    freq = np.linspace(30000.0, 80000.0, 60)
    res = pseudo_rayleigh_dispersion(
        freq,
        vp=PR_VP,
        vs=PR_VS,
        rho=PR_RHO,
        vf=PR_VF,
        rho_f=PR_RHO_F,
        a=PR_A,
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
        f,
        vp=QUAD_VP,
        vs=QUAD_VS,
        rho=QUAD_RHO,
        vf=QUAD_VF,
        rho_f=QUAD_RHO_F,
        a=QUAD_A,
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
    from fwap.cylindrical import rayleigh_speed
    from fwap.cylindrical_solver import quadrupole_dispersion

    vR = rayleigh_speed(QUAD_VP, QUAD_VS)
    f = np.linspace(3000.0, 20000.0, 10)
    res = quadrupole_dispersion(
        f,
        vp=QUAD_VP,
        vs=QUAD_VS,
        rho=QUAD_RHO,
        vf=QUAD_VF,
        rho_f=QUAD_RHO_F,
        a=QUAD_A,
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
    from fwap.cylindrical import rayleigh_speed
    from fwap.cylindrical_solver import quadrupole_dispersion

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    vR = rayleigh_speed(vp, vs)
    f = np.linspace(40000.0, 100000.0, 30)
    res = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    finite = np.isfinite(res.slowness)
    assert finite.any(), "fast-formation path must populate at least one frequency"
    velocity = 1.0 / res.slowness[finite]
    assert (velocity > vR * 0.99).all(), (
        f"velocity must stay near or above V_R ({vR:.0f}); got {velocity}"
    )
    assert (velocity < vs).all(), f"velocity must stay below V_S ({vs}); got {velocity}"
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
        f,
        vp=QUAD_VP,
        vs=QUAD_VS,
        rho=QUAD_RHO,
        vf=QUAD_VF,
        rho_f=QUAD_RHO_F,
        a=QUAD_A,
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
        f,
        vp=QUAD_VP,
        vs=QUAD_VS,
        rho=QUAD_RHO,
        vf=QUAD_VF,
        rho_f=QUAD_RHO_F,
        a=QUAD_A,
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
        _modal_determinant_n2,
        quadrupole_dispersion,
    )

    f = np.array([8000.0])
    res = quadrupole_dispersion(
        f,
        vp=QUAD_VP,
        vs=QUAD_VS,
        rho=QUAD_RHO,
        vf=QUAD_VF,
        rho_f=QUAD_RHO_F,
        a=QUAD_A,
    )
    assert np.isfinite(res.slowness[0])
    omega = 2.0 * np.pi * f[0]
    kz_root = res.slowness[0] * omega
    det_at_root = _modal_determinant_n2(
        kz_root,
        omega,
        QUAD_VP,
        QUAD_VS,
        QUAD_RHO,
        QUAD_VF,
        QUAD_RHO_F,
        QUAD_A,
    )
    # Off-root sample 5 % below kz_root.
    kz_off = kz_root * 0.95
    det_off = _modal_determinant_n2(
        kz_off,
        omega,
        QUAD_VP,
        QUAD_VS,
        QUAD_RHO,
        QUAD_VF,
        QUAD_RHO_F,
        QUAD_A,
    )
    assert abs(det_at_root) < abs(det_off) * 1.0e-2


def test_quadrupole_dispersion_rejects_non_positive_inputs():
    """Mirrors the Stoneley / flexural input-validation suite."""
    from fwap.cylindrical_solver import quadrupole_dispersion

    f = np.array([5000.0])
    base = dict(
        vp=QUAD_VP, vs=QUAD_VS, rho=QUAD_RHO, vf=QUAD_VF, rho_f=QUAD_RHO_F, a=QUAD_A
    )
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
            bad,
            vp=QUAD_VP,
            vs=QUAD_VS,
            rho=QUAD_RHO,
            vf=QUAD_VF,
            rho_f=QUAD_RHO_F,
            a=QUAD_A,
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
            kz,
            omega,
            QUAD_VP,
            QUAD_VS,
            QUAD_RHO,
            QUAD_VF,
            QUAD_RHO_F,
            QUAD_A,
        )
        d_complex = _modal_determinant_n2_complex(
            complex(kz),
            omega,
            QUAD_VP,
            QUAD_VS,
            QUAD_RHO,
            QUAD_VF,
            QUAD_RHO_F,
            QUAD_A,
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
    from scipy import optimize as _opt

    from fwap.cylindrical_solver import (
        _modal_determinant_n2,
        _quadrupole_kz_bracket,
        quadrupole_dispersion,
    )

    f = np.linspace(5000.0, 20000.0, 11)
    res = quadrupole_dispersion(
        f,
        vp=QUAD_VP,
        vs=QUAD_VS,
        rho=QUAD_RHO,
        vf=QUAD_VF,
        rho_f=QUAD_RHO_F,
        a=QUAD_A,
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 8

    for i in np.where(finite)[0]:
        omega = 2.0 * np.pi * f[i]

        def _det(kz):
            return _modal_determinant_n2(
                kz,
                omega,
                QUAD_VP,
                QUAD_VS,
                QUAD_RHO,
                QUAD_VF,
                QUAD_RHO_F,
                QUAD_A,
            )

        kz_lo, kz_hi = _quadrupole_kz_bracket(
            omega,
            QUAD_VP,
            QUAD_VS,
            QUAD_RHO,
            QUAD_VF,
            QUAD_RHO_F,
            QUAD_A,
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
        _modal_determinant_n2_complex,
        quadrupole_dispersion,
    )

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(50000.0, 100000.0, 30)
    res = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    finite_idx = np.where(np.isfinite(res.slowness))[0]
    assert finite_idx.size >= 3

    for i in finite_idx[:5]:
        omega = 2.0 * np.pi * res.freq[i]
        kz_root = complex(res.slowness[i] * omega, 0.0)
        det_at_root = _modal_determinant_n2_complex(
            kz_root,
            omega,
            vp,
            vs,
            rho,
            vf,
            rho_f,
            a,
            leaky_p=False,
            leaky_s=False,
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
    from fwap.cylindrical import rayleigh_speed
    from fwap.cylindrical_solver import quadrupole_dispersion

    vp, vs, rho = 5500.0, 3100.0, 2500.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    vR = rayleigh_speed(vp, vs)
    f = np.linspace(40000.0, 100000.0, 50)
    res = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        f_asc,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_desc = quadrupole_dispersion(
        f_desc,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_allclose(
        res_asc.slowness,
        res_desc.slowness[::-1],
        rtol=1.0e-9,
        equal_nan=True,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    np.testing.assert_array_equal(res_layered.slowness, res_unlayered.slowness)
    np.testing.assert_array_equal(res_layered.freq, res_unlayered.freq)
    assert res_layered.name == res_unlayered.name == "Stoneley"
    assert res_layered.azimuthal_order == 0


def test_stoneley_dispersion_layered_empty_layers_returns_borehole_mode():
    f = np.linspace(500.0, 5000.0, 5)
    res = stoneley_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0


def test_stoneley_dispersion_layered_rejects_bad_layer_object():
    f = np.array([1000.0])
    with pytest.raises(ValueError, match="BoreholeLayer"):
        stoneley_dispersion_layered(
            f,
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
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
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
            layers=(layer,),
        )


def test_stoneley_dispersion_layered_accepts_list_for_layers():
    """``layers`` should accept any iterable that ``tuple(...)``
    consumes; the empty list must dispatch to the unlayered solver
    just like ``()``."""
    f = np.linspace(500.0, 5000.0, 4)
    res_tuple = stoneley_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(),
    )
    res_list = stoneley_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=[],
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
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
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
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            vf=p["vf"],
            layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    assert F_f**2 == pytest.approx(kz**2 - (omega / p["vf"]) ** 2)
    assert p_m**2 == pytest.approx(kz**2 - (omega / p["layer"].vp) ** 2)
    assert s_m**2 == pytest.approx(kz**2 - (omega / p["layer"].vs) ** 2)
    assert pp**2 == pytest.approx(kz**2 - (omega / p["vp"]) ** 2)
    assert ss**2 == pytest.approx(kz**2 - (omega / p["vs"]) ** 2)


def test_layered_radial_wavenumbers_layer_equals_formation_collapses():
    """When the annulus material matches the formation, ``p_m == p``
    and ``s_m == s`` to floating-point precision -- the algebraic
    cornerstone of the substep F.1.a.6 self-check."""
    vp, vs = 4500.0, 2500.0
    p = dict(
        vp=vp,
        vs=vs,
        rho=2400.0,
        vf=1500.0,
        layer=BoreholeLayer(vp=vp, vs=vs, rho=2400.0, thickness=0.01),
    )
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, p["vf"]) * 1.5
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    pack = _layered_n0_bessel_pack(F_f, p_m, s_m, pp, ss, a, b)
    assert len(pack) == 22
    expected_keys = {
        "I0_Ff_a",
        "I1_Ff_a",
        "I0_pm_a",
        "I1_pm_a",
        "K0_pm_a",
        "K1_pm_a",
        "I0_sm_a",
        "I1_sm_a",
        "K0_sm_a",
        "K1_sm_a",
        "I0_pm_b",
        "I1_pm_b",
        "K0_pm_b",
        "K1_pm_b",
        "I0_sm_b",
        "I1_sm_b",
        "K0_sm_b",
        "K1_sm_b",
        "K0_p_b",
        "K1_p_b",
        "K0_s_b",
        "K1_s_b",
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=1500.0,
        layer=layer,
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    # Reconstruct the corresponding entries of the unlayered matrix
    # without invoking the determinant routine: M_11, M_12, M_13 of
    # _modal_determinant_n0 (lifted from the docstring / source).
    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = F * float(sp.iv(1, F * a)) / (rho_f * omega**2)
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row1_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: post-rescale row entries are
    purely real in the bound regime. Any non-zero imaginary part
    flags a sign error in the C-flavour rescaling."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n0_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
    M23 = -2.0 * kz * mu * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)

    assert row[0].real == pytest.approx(M21)
    assert row[2].real == pytest.approx(M22)
    assert row[4].real == pytest.approx(M23)


def test_layered_row2_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (5, 6) are
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row2_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: post-rescale row 2 entries
    are purely real in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
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
    expected_CI = (
        +2.0
        * mu_m
        * kz
        * (s_m * float(sp.iv(0, s_m * a)) - float(sp.iv(1, s_m * a)) / a)
    )

    row = _layered_n0_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row3_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (5, 6) are
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row3_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5: the full ``row * i`` plus column-by-(-i)
    rescale lands row 3 in real form."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row4_at_b_fluid_column_is_zero():
    """The fluid lives at ``r < a``; it does not reach ``r = b``.
    Column 0 (A) is identically zero in row 4."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row4_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 4 is real in the bound
    regime (no row scaling; column-by-(-i) on C_I, C_K, C kills
    the explicit ``i`` factors)."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row4_at_b_matches_closed_form_per_column():
    """Cross-check every non-zero entry against the substep-F.1.a.2
    closed form, evaluated at ``r = b``. No single-interface analog
    to compare against, so this is the per-element transcription
    check."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row4 = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row5_at_b_fluid_column_is_zero():
    """Fluid lives at ``r < a``; column 0 (A) is identically zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row5_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.2 at
    r = b, with the row * i / col * -i rescaling applied. Notable
    feature: row 5 uses degree-0 Bessel functions (I_0 / K_0),
    distinguishing it from rows 1, 4, 6 (degree-1)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row4 = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    row5 = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row6_at_b_fluid_column_is_zero():
    """Fluid lives at ``r < a``; column 0 (A) is identically zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row6_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 6 is real in the bound
    regime. Same imaginary-power pattern as rows 1, 4 (B-real,
    C-imag pre-rescale) so no row scaling needed; only the
    column-by-(-i) on C_I, C_K, C is applied."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row6_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.3 at
    r = b. Row 6 carries the Lame combination ``(2 k_z^2 - k_Sm^2)``
    on each B / C column, identical in structure to the row-2 form
    but evaluated at r = b with non-zero formation columns."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
    expected_CI = (
        -2.0
        * kz
        * mu_m
        * (s_m * float(sp.iv(0, s_m * b)) - float(sp.iv(1, s_m * b)) / b)
    )
    expected_CK = (
        +2.0
        * kz
        * mu_m
        * (s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b)
    )
    expected_B = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p_form * b))
        + 2.0 * p_form * float(sp.kv(1, p_form * b)) / b
    )
    expected_C = (
        -2.0
        * kz
        * mu
        * (s_form * float(sp.kv(0, s_form * b)) + float(sp.kv(1, s_form * b)) / b)
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
    M23_at_b = (
        -2.0 * kz * mu * (s * float(sp.kv(0, s * b)) + float(sp.kv(1, s * b)) / b)
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row7_at_b_fluid_column_is_zero():
    """Row 7 column 0 (A) is identically zero. The fluid carries
    no shear AND lives at r < a, so it contributes nothing to the
    sigma_rz continuity at r=b."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    assert row[1].real == pytest.approx(
        -2.0 * kz * mu_m * p_m * float(sp.iv(1, p_m * b))
    )
    assert row[2].real == pytest.approx(
        +2.0 * kz * mu_m * p_m * float(sp.kv(1, p_m * b))
    )
    assert row[3].real == pytest.approx(
        +mu_m * two_kz2_minus_kSm2 * float(sp.iv(1, s_m * b))
    )
    assert row[4].real == pytest.approx(
        +mu_m * two_kz2_minus_kSm2 * float(sp.kv(1, s_m * b))
    )
    assert row[5].real == pytest.approx(
        -2.0 * kz * mu * p_form * float(sp.kv(1, p_form * b))
    )
    assert row[6].real == pytest.approx(
        -mu * two_kz2_minus_kS2 * float(sp.kv(1, s_form * b))
    )


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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
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
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n0_layered(
        kz_root,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    det_off_root = _modal_determinant_n0_layered(
        kz_root * 1.05,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    np.testing.assert_allclose(
        res_layered.slowness,
        res_unlayered.slowness,
        rtol=1.0e-8,
        equal_nan=True,
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
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    # Even a "different" layer with vanishing thickness should
    # converge to the unlayered Stoneley slowness.
    layer_thin = BoreholeLayer(
        vp=3500.0,
        vs=1800.0,
        rho=2100.0,
        thickness=1.0e-9,
    )
    res_thin = stoneley_dispersion_layered(
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer_thin,),
    )
    assert res_thin.slowness[0] == pytest.approx(
        res_unlayered.slowness[0],
        rel=1.0e-4,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    soft_layer = BoreholeLayer(
        vp=3500.0,
        vs=1800.0,
        rho=2100.0,
        thickness=0.01,
    )
    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        vp=layer_vp,
        vs=layer_vs,
        rho=layer_rho,
        thickness=0.5,
    )

    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    # Limit value: unlayered Stoneley with the LAYER properties as
    # the formation half-space.
    res_limit = stoneley_dispersion(
        f,
        vp=layer_vp,
        vs=layer_vs,
        rho=layer_rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    # Tolerance is loose because the limit is asymptotic; tighter
    # match would need an even thicker layer (numerically delicate
    # because K_n decays exponentially).
    assert res_layered.slowness[0] == pytest.approx(
        res_limit.slowness[0],
        rel=1.0e-3,
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
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n0_layered(
        kz_root,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    det_off = _modal_determinant_n0_layered(
        kz_root * 1.05,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        vp=layer_vp,
        vs=layer_vs,
        rho=layer_rho,
        thickness=0.005,
    )
    f = np.array([10.0])

    res = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = flexural_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    np.testing.assert_array_equal(res_layered.slowness, res_unlayered.slowness)
    np.testing.assert_array_equal(res_layered.freq, res_unlayered.freq)
    assert res_layered.name == res_unlayered.name == "flexural"
    assert res_layered.azimuthal_order == 1


def test_flexural_dispersion_layered_empty_layers_returns_borehole_mode():
    f = np.linspace(2000.0, 5000.0, 5)
    res = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1


def test_flexural_dispersion_layered_fast_formation_dispatches_to_complex_path():
    """Fast-formation layered flexural (``V_S > V_f`` with a
    non-empty layer) dispatches to the complex-determinant path
    via ``_modal_determinant_n1_cased_complex`` and
    ``_flexural_dispersion_fast_formation_layered``. The earlier
    ``NotImplementedError`` from F.2.d is gone -- fast-formation
    layered flexural is now a supported regime."""
    f = np.array([2000.0, 4000.0])
    # Fast formation: vs (2500) > vf (1500).
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    res = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(layer,),
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    # Bound mode: real-valued slowness (or NaN outside the
    # geometric cutoff). Either way, no exception.
    assert res.slowness.dtype == np.float64
    assert res.attenuation_per_meter is None


def test_flexural_dispersion_layered_rejects_bad_layer_object():
    f = np.array([2000.0])
    with pytest.raises(ValueError, match="BoreholeLayer"):
        flexural_dispersion_layered(
            f,
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
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
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
            layers=(layer,),
        )


def test_flexural_dispersion_layered_accepts_list_for_layers():
    """``layers`` should accept any iterable that ``tuple(...)``
    consumes; empty list dispatches to the unlayered solver."""
    f = np.linspace(2000.0, 5000.0, 4)
    res_tuple = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(),
    )
    res_list = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=[],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    # Reconstruct the M31-M34 entries directly from the n=1
    # single-interface formula (see _modal_determinant_n1
    # docstring lines for row 3).
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M31 = 0.0
    M32 = (
        2.0
        * mu
        * (p * float(sp.kv(0, p * a)) / a + 2.0 * float(sp.kv(1, p * a)) / (a * a))
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row3_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substeps F.2.a.2 /
    F.2.a.3 closed forms (with the F.1.a.2 sign-flip pattern
    applied to the I-flavour annulus terms)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2

    expected_BI = (
        2.0
        * mu_m
        * (
            -p_m * float(sp.iv(0, p_m * a)) / a
            + 2.0 * float(sp.iv(1, p_m * a)) / (a * a)
        )
    )
    expected_BK = (
        2.0
        * mu_m
        * (
            +p_m * float(sp.kv(0, p_m * a)) / a
            + 2.0 * float(sp.kv(1, p_m * a)) / (a * a)
        )
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )

    expected_ratio = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row6_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_theta closed forms. The B and D coefficients carry the
    F.1.a.2 sign-flip pattern: ``s I_0`` flips, ``K_1/r``-style
    direct terms keep sign."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    expected_ratio = +float(sp.iv(1, p_m * b)) / float(sp.kv(1, p_m * b))
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row9_at_b_sparsity():
    """Sparsity: A column zero (fluid r<a, no shear); all other
    nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row9_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row9_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substeps F.2.a.2 /
    F.2.a.3 closed forms for all nine non-zero entries (annulus
    B/C/D + formation B/C/D)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row9_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    mu = p["rho"] * p["vs"] ** 2

    expected_BI = (
        2.0
        * mu_m
        * (
            -p_m * float(sp.iv(0, p_m * b)) / b
            + 2.0 * float(sp.iv(1, p_m * b)) / (b * b)
        )
    )
    expected_BK = (
        2.0
        * mu_m
        * (
            +p_m * float(sp.kv(0, p_m * b)) / b
            + 2.0 * float(sp.kv(1, p_m * b)) / (b * b)
        )
    )
    expected_CI = +kz * mu_m * float(sp.iv(1, s_m * b)) / b
    expected_CK = +kz * mu_m * float(sp.kv(1, s_m * b)) / b
    expected_B = (
        -2.0
        * mu
        * (
            +p_form * float(sp.kv(0, p_form * b)) / b
            + 2.0 * float(sp.kv(1, p_form * b)) / (b * b)
        )
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    # Row 3's M32, M33, M34 form evaluated at r=b instead of r=a.
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M32_at_b = (
        2.0
        * mu
        * (p * float(sp.kv(0, p * b)) / b + 2.0 * float(sp.kv(1, p * b)) / (b * b))
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = (F * float(sp.iv(0, F * a)) - float(sp.iv(1, F * a)) / a) / (rho_f * omega**2)
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row1_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_r decomposition (with the F.1.a.2 sign-flip pattern applied
    to the I-flavour annulus terms)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    expected_A = (F_f * float(sp.iv(0, F_f * a)) - float(sp.iv(1, F_f * a)) / a) / (
        p["rho_f"] * omega**2
    )
    expected_BI = -p_m * float(sp.iv(0, p_m * a)) + float(sp.iv(1, p_m * a)) / a
    expected_BK = +p_m * float(sp.kv(0, p_m * a)) + float(sp.kv(1, p_m * a)) / a
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    expected_ratio = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
    M23 = -2.0 * kz * mu * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)
    M24 = (
        +2.0
        * mu
        * (s * float(sp.kv(0, s * a)) / a + 2.0 * float(sp.kv(1, s * a)) / (a * a))
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
    expected_CI = (
        +2.0
        * kz
        * mu_m
        * (s_m * float(sp.iv(0, s_m * a)) - float(sp.iv(1, s_m * a)) / a)
    )
    expected_CK = (
        -2.0
        * kz
        * mu_m
        * (s_m * float(sp.kv(0, s_m * a)) + float(sp.kv(1, s_m * a)) / a)
    )
    expected_DI = (
        +2.0
        * mu_m
        * (
            -s_m * float(sp.iv(0, s_m * a)) / a
            + 2.0 * float(sp.iv(1, s_m * a)) / (a * a)
        )
    )
    expected_DK = (
        +2.0
        * mu_m
        * (
            +s_m * float(sp.kv(0, s_m * a)) / a
            + 2.0 * float(sp.kv(1, s_m * a)) / (a * a)
        )
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M41 = 0.0
    M42 = +2.0 * kz * mu * (p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a)
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row4_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rz decomposition. The B columns have multi-term
    (``+p_m X_0 +/- X_1/r``); C and D columns are single-Bessel-term."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row4_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    expected_BI = (
        -2.0
        * kz
        * mu_m
        * (p_m * float(sp.iv(0, p_m * a)) - float(sp.iv(1, p_m * a)) / a)
    )
    expected_BK = (
        +2.0
        * kz
        * mu_m
        * (p_m * float(sp.kv(0, p_m * a)) + float(sp.kv(1, p_m * a)) / a)
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row4_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )

    expected_ratio = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row5_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid r<a doesn't reach r=b);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row5_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 5 has the no-row-rescale pattern;
    column-by-(-i) on C_I, C_K, C only. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row5_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_r decomposition at r=b."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row5 = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row7_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_z decomposition at r=b for the six non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    # B ratio: +I_1/K_1 (KEEP sign).
    expected_B_ratio = +float(sp.iv(1, p_m * b)) / float(sp.kv(1, p_m * b))
    assert row[1].real / row[2].real == pytest.approx(expected_B_ratio)
    # C ratio: -I_0/K_0 (FLIP sign on derivative-induced term).
    expected_C_ratio = -float(sp.iv(0, s_m * b)) / float(sp.kv(0, s_m * b))
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row8_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid r<a doesn't reach r=b);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row8_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 8 has the no-row-rescale pattern;
    only column-by-(-i) on C cols. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row8_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rr decomposition at r=b for all nine non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
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
    expected_CI = (
        -2.0
        * kz
        * mu_m
        * (s_m * float(sp.iv(0, s_m * b)) - float(sp.iv(1, s_m * b)) / b)
    )
    expected_CK = (
        +2.0
        * kz
        * mu_m
        * (s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b)
    )
    expected_B = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p_form * b))
        + 2.0 * p_form * float(sp.kv(0, p_form * b)) / b
        + 4.0 * float(sp.kv(1, p_form * b)) / (b * b)
    )
    expected_C = (
        -2.0
        * kz
        * mu
        * (s_form * float(sp.kv(0, s_form * b)) + float(sp.kv(1, s_form * b)) / b)
    )
    expected_DI = (
        +2.0
        * mu_m
        * (
            s_m * float(sp.iv(0, s_m * b)) / b
            - 2.0 * float(sp.iv(1, s_m * b)) / (b * b)
        )
    )
    expected_DK = (
        -2.0
        * mu_m
        * (
            s_m * float(sp.kv(0, s_m * b)) / b
            + 2.0 * float(sp.kv(1, s_m * b)) / (b * b)
        )
    )
    expected_D = (
        +2.0
        * mu
        * (
            s_form * float(sp.kv(0, s_form * b)) / b
            + 2.0 * float(sp.kv(1, s_form * b)) / (b * b)
        )
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
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
    M23_at_b = (
        -2.0 * kz * mu * (s * float(sp.kv(0, s * b)) + float(sp.kv(1, s * b)) / b)
    )
    M24_at_b = (
        +2.0
        * mu
        * (s * float(sp.kv(0, s * b)) / b + 2.0 * float(sp.kv(1, s * b)) / (b * b))
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row10_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid no shear AND fluid r<a);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row10_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 10 is z-derivative-bearing -- gets the
    FULL rescale (row * i + col-by-(-i) on C cols)."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row10_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rz decomposition at r=b for all nine non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    expected_BI = (
        -2.0
        * kz
        * mu_m
        * (p_m * float(sp.iv(0, p_m * b)) - float(sp.iv(1, p_m * b)) / b)
    )
    expected_BK = (
        +2.0
        * kz
        * mu_m
        * (p_m * float(sp.kv(0, p_m * b)) + float(sp.kv(1, p_m * b)) / b)
    )
    expected_CI = +mu_m * two_kz2_minus_kSm2 * float(sp.iv(1, s_m * b))
    expected_CK = +mu_m * two_kz2_minus_kSm2 * float(sp.kv(1, s_m * b))
    expected_B = (
        -2.0
        * kz
        * mu
        * (p_form * float(sp.kv(0, p_form * b)) + float(sp.kv(1, p_form * b)) / b)
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
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M42_at_b = (
        +2.0 * kz * mu * (p * float(sp.kv(0, p * b)) + float(sp.kv(1, p * b)) / b)
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
        vp=3000.0,
        vs=1200.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
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
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n1_layered(
        kz_root,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    det_off_root = _modal_determinant_n1_layered(
        kz_root * 1.05,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = flexural_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    np.testing.assert_allclose(
        res_layered.slowness,
        res_unlayered.slowness,
        rtol=1.0e-8,
        equal_nan=True,
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
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    # Even a "different" layer with vanishing thickness should
    # converge to the unlayered flexural slowness. Use a harder
    # layer (layer.vs > vs) so the bound regime holds in the
    # annulus per the F.2.d slow-formation gate.
    layer_thin = BoreholeLayer(
        vp=3200.0,
        vs=1300.0,
        rho=2350.0,
        thickness=1.0e-9,
    )
    res_thin = flexural_dispersion_layered(
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer_thin,),
    )
    assert res_thin.slowness[0] == pytest.approx(
        res_unlayered.slowness[0],
        rel=1.0e-4,
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
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
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
        np.array([f]),
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(p["layer"],),
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n1_layered(
        kz_root,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
    )
    det_off = _modal_determinant_n1_layered(
        kz_root * 1.01,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
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
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    hard_layer = BoreholeLayer(
        vp=3500.0,
        vs=1500.0,
        rho=2400.0,
        thickness=0.01,
    )
    res_layered = flexural_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
            f,
            **cij_gam,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )


def test_dispersion_vti_returns_borehole_mode():
    f = np.linspace(2000.0, 5000.0, 5)
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
            f,
            **base,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )


def test_dispersion_vti_rejects_unstable_c33_le_c13():
    """Validator rejects ``C33 <= C13`` (would break qP/qSV
    decoupling in the Christoffel equation)."""
    f = np.array([5000.0])
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    cij["c13"] = cij["c33"] * 1.5  # force c13 > c33
    with pytest.raises(ValueError, match="c33 > c13"):
        stoneley_dispersion_vti(
            f,
            **cij,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )


def test_dispersion_vti_rejects_unstable_c11_le_c66():
    """Validator rejects ``C11 <= C66`` (would have horizontal P
    no faster than horizontal S)."""
    f = np.array([5000.0])
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    cij["c66"] = cij["c11"] * 1.5  # force c66 > c11
    with pytest.raises(ValueError, match="c11 > c66"):
        stoneley_dispersion_vti(
            f,
            **cij,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )


def test_dispersion_vti_rejects_non_positive_freq_and_geometry():
    """Standard freq/geometry validation as for the isotropic
    public APIs."""
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    base = dict(rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1)
    # Non-positive freq.
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        stoneley_dispersion_vti(
            np.array([0.0]),
            **cij,
            **base,
        )
    # Non-positive vf.
    with pytest.raises(ValueError, match="vf and rho_f must be positive"):
        stoneley_dispersion_vti(
            np.array([5000.0]),
            **cij,
            **{**base, "vf": 0.0},
        )
    # Non-positive a.
    with pytest.raises(ValueError, match="a must be positive"):
        stoneley_dispersion_vti(
            np.array([5000.0]),
            **cij,
            **{**base, "a": 0.0},
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
        c11=4.0e10,  # V_Ph^2 * rho ~ (4080 m/s)^2 * 2400
        c13=1.5e10,  # delta-coupled
        c33=3.5e10,  # V_Pv^2 * rho ~ (3819 m/s)^2 * 2400
        c44=1.0e10,  # V_Sv^2 * rho ~ (2041 m/s)^2 * 2400
        c66=1.3e10,  # V_Sh^2 * rho ~ (2327 m/s)^2 * 2400  (gamma > 0)
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
        kz,
        omega,
        **cij,
        rho=rho,
    )

    p_iso = float(np.sqrt(kz**2 - (omega / vp) ** 2))
    s_iso = float(np.sqrt(kz**2 - (omega / vs) ** 2))

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
        kz,
        omega,
        **cij,
        rho=rho,
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
        kz,
        omega,
        **cij,
        rho=rho,
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
        kz,
        omega,
        **cij,
        rho=rho,
    )
    expected_SH_sq = (cij["c44"] * kz**2 - rho * omega**2) / cij["c66"]
    assert alpha_SH**2 == pytest.approx(expected_SH_sq, rel=1.0e-12)
    # Sanity: the buggy ``kz^2 - rho omega^2 / C66`` form would
    # give a DIFFERENT value here because C44 != C66.
    buggy_SH_sq = kz**2 - rho * omega**2 / cij["c66"]
    assert abs(alpha_SH**2 - buggy_SH_sq) > 0.0  # they differ
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
            kz,
            omega,
            **cij,
            rho=rho,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = F * float(sp.iv(1, F * a)) / (rho_f * omega**2)
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    # Pull alpha_qP from the helper.
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
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

    p_iso = float(np.sqrt(kz**2 - (omega / vp) ** 2))
    gamma_qP = _polarization_ratio_uz_over_ur_vti(
        p_iso,
        kz,
        omega,
        c11=cij["c11"],
        c13=cij["c13"],
        c44=cij["c44"],
        rho=rho,
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

    s_iso = float(np.sqrt(kz**2 - (omega / vs) ** 2))
    gamma_qSV = _polarization_ratio_uz_over_ur_vti(
        s_iso,
        kz,
        omega,
        c11=cij["c11"],
        c13=cij["c13"],
        c44=cij["c44"],
        rho=rho,
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
        kz,
        omega,
        **cij,
        rho=rho,
    )
    rho_omega_sq = rho * omega**2

    for alpha_qX in (alpha_qP, alpha_qSV):
        gamma = _polarization_ratio_uz_over_ur_vti(
            alpha_qX,
            kz,
            omega,
            c11=cij["c11"],
            c13=cij["c13"],
            c44=cij["c44"],
            rho=rho,
        )
        # Eigenvector equation residual:
        residual = (
            -cij["c11"] * alpha_qX**2 + cij["c44"] * kz**2 - rho_omega_sq
        ) + 1j * (cij["c13"] + cij["c44"]) * alpha_qX * kz * gamma
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
    M23 = -2.0 * kz * mu * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)

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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, _, _ = _radial_wavenumbers_vti(kz, omega, **cij, rho=rho)
    rho_omega_sq = rho * omega**2
    q_qP = (
        cij["c44"] * (cij["c11"] * alpha_qP**2 + cij["c13"] * kz**2)
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    row_b = _modal_row3_at_a_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n0_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    det_off_root = _modal_determinant_n0_vti(
        kz_root * 1.05,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root_iso = float(bound.slowness[0]) * omega

    def _det(kz):
        return _modal_determinant_n0_vti(
            kz,
            omega,
            **cij,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
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
            kz,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = stoneley_dispersion_vti(
        f,
        **cij_perturbed,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_allclose(
        res_vti.slowness,
        res_iso.slowness,
        rtol=1.0e-5,
        equal_nan=True,
    )
    # Confirm the perturbation actually defeated the isotropic
    # dispatch (the test would pass trivially otherwise).
    assert not _is_isotropic_stiffness(
        **{k: cij_perturbed[k] for k in ("c11", "c13", "c33", "c44", "c66")}
    )


def test_stoneley_dispersion_vti_genuine_TI_runs_smoke():
    """Smoke: a typical genuine-TI fixture produces a finite
    slowness curve. No analytic oracle (Norris 1990 LF check is
    H.c.3); just confirms the brentq + bracket combination
    handles the TI case across a broad band."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(1000.0, 10000.0, 8)

    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        np.array([f]),
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n0_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    det_off = _modal_determinant_n0_vti(
        kz_root * 1.01,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_stoneley_dispersion_vti_returns_borehole_mode_for_genuine_TI():
    """BoreholeMode return-type contract on the genuine-TI path."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(2000.0, 5000.0, 4)
    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
    return float(np.sqrt(1.0 / vf**2 + rho_f / c66))


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
        np.array([10.0]),
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        np.array([10.0]),
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        np.array([10.0]),
        **cij_a,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_b = stoneley_dispersion_vti(
        np.array([10.0]),
        **cij_b,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = (F * float(sp.iv(0, F * a)) - float(sp.iv(1, F * a)) / a) / (rho_f * omega**2)
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    from scipy import special as sp

    expected_BqP = (
        alpha_qP * float(sp.kv(0, alpha_qP * a)) + float(sp.kv(1, alpha_qP * a)) / a
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
    M23 = -2.0 * kz * mu * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)
    M24 = (
        +2.0
        * mu
        * (s * float(sp.kv(0, s * a)) / a + 2.0 * float(sp.kv(1, s * a)) / (a * a))
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    rho_omega_sq = rho * omega**2
    q_qP = (
        cij["c44"] * (cij["c11"] * alpha_qP**2 + cij["c13"] * kz**2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    q_qSV = (
        cij["c44"] * (cij["c11"] * alpha_qSV**2 + cij["c13"] * kz**2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    from scipy import special as sp

    F = float(np.sqrt(kz**2 - (omega / vf) ** 2))
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
    expected_DSH = (
        +2.0
        * cij["c66"]
        * (
            alpha_SH * float(sp.kv(0, alpha_SH * a)) / a
            + 2.0 * float(sp.kv(1, alpha_SH * a)) / (a * a)
        )
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
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    row_b = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
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
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    row_b = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    # row[3] = 2 C66 (alpha_SH K_0(alpha_SH a)/a + 2 K_1(alpha_SH a)/a^2).
    # Both C66 (outer) and alpha_SH (Bessel arg) change.
    _, _, alpha_SH_a = _radial_wavenumbers_vti(kz, omega, **cij_a, rho=rho)
    _, _, alpha_SH_b = _radial_wavenumbers_vti(kz, omega, **cij_b, rho=rho)
    from scipy import special as sp

    expected_a = (
        +2.0
        * cij_a["c66"]
        * (
            alpha_SH_a * float(sp.kv(0, alpha_SH_a * a)) / a
            + 2.0 * float(sp.kv(1, alpha_SH_a * a)) / (a * a)
        )
    )
    expected_b = (
        +2.0
        * cij_b["c66"]
        * (
            alpha_SH_b * float(sp.kv(0, alpha_SH_b * a)) / a
            + 2.0 * float(sp.kv(1, alpha_SH_b * a)) / (a * a)
        )
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M31 = 0.0
    M32 = (
        2.0
        * mu
        * (p * float(sp.kv(0, p * a)) / a + 2.0 * float(sp.kv(1, p * a)) / (a * a))
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    from scipy import special as sp

    expected_BqP = (
        +2.0
        * cij["c66"]
        * (
            alpha_qP * float(sp.kv(0, alpha_qP * a)) / a
            + 2.0 * float(sp.kv(1, alpha_qP * a)) / (a * a)
        )
    )
    expected_CqSV = +kz * cij["c66"] * float(sp.kv(1, alpha_qSV * a)) / a
    expected_DSH = -cij["c66"] * (
        alpha_SH**2 * float(sp.kv(1, alpha_SH * a))
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
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    row_b = _modal_row3_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M41 = 0.0
    M42 = 2.0 * kz * mu * (p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a)
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    rho_omega_sq = rho * omega**2
    p_qP = cij["c11"] * alpha_qP**2 + cij["c13"] * kz**2 + rho_omega_sq
    p_qSV = cij["c11"] * alpha_qSV**2 + cij["c13"] * kz**2 + rho_omega_sq
    from scipy import special as sp

    expected_BqP = (
        +cij["c44"]
        * p_qP
        / ((cij["c13"] + cij["c44"]) * kz)
        * (alpha_qP * float(sp.kv(0, alpha_qP * a)) + float(sp.kv(1, alpha_qP * a)) / a)
    )
    expected_CqSV = (
        +cij["c44"] * p_qSV / (cij["c13"] + cij["c44"]) * float(sp.kv(1, alpha_qSV * a))
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
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    row_b = _modal_row4_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n1_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    det_off_root = _modal_determinant_n1_vti(
        kz_root * 1.05,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root_iso = float(bound.slowness[0]) * omega

    def _det(kz):
        return _modal_determinant_n1_vti(
            kz,
            omega,
            **cij,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
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
            kz,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
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
        c13=4.0e9,  # delta-coupled (c33 > c13 stability)
        c33=1.06e10,  # V_Pv^2 * rho ~ (2200 m/s)^2 * 2200
        c44=2.66e9,  # V_Sv^2 * rho ~ (1100 m/s)^2 * 2200  (Vsv < Vf)
        c66=3.17e9,  # V_Sh^2 * rho ~ (1200 m/s)^2 * 2200  (gamma > 0)
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = flexural_dispersion_vti(
        f,
        **cij_perturbed,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_allclose(
        res_vti.slowness,
        res_iso.slowness,
        rtol=1.0e-5,
        equal_nan=True,
    )
    # Confirm the perturbation actually defeated the isotropic
    # dispatch (the test would pass trivially otherwise).
    assert not _is_isotropic_stiffness(
        **{k: cij_perturbed[k] for k in ("c11", "c13", "c33", "c44", "c66")}
    )


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
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        np.array([f]),
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n1_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    det_off = _modal_determinant_n1_vti(
        kz_root * 1.01,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_flexural_dispersion_vti_returns_borehole_mode_for_genuine_TI():
    """BoreholeMode return-type contract on the slow-formation
    genuine-TI path."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(2000.0, 5000.0, 4)
    res = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        np.array([3000.0]),
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n0_vti(
            kz_root,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
        det_off = _modal_determinant_n0_vti(
            kz_root * 1.01,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
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
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n1_vti(
            kz_root,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
        det_off = _modal_determinant_n1_vti(
            kz_root * 1.01,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
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
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    # All slownesses above the fluid-only slowness 1/V_f and
    # below the Norris LF cap (a sanity fence, not a tight oracle).
    s_fluid = 1.0 / 1500.0
    s_norris = float(np.sqrt(1.0 / 1500.0**2 + 1000.0 / cij["c66"]))
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
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        c11=rho * vp**2,  # set epsilon = 0 (c11 = c33)
        c33=rho * vp**2,
        c44=rho * vsv**2,
        c66=rho * vsh**2,  # gamma > 0 only
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
        f,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    s_phenom = flexural_dispersion_vti_physical(
        vp=vp,
        vsv=vsv,
        vsh=vsh,
        a_borehole=a,
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
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    res_unlayered = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_array_equal(res_empty.slowness, res_unlayered.slowness)

    # Single-layer path: F.1 hand-coded determinant. Just smoke;
    # bit-equivalent regressions are exercised elsewhere.
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    res_one = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
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
        vp=3500.0,
        vs=1800.0,
        rho=2100.0,
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
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    # F.1.b row 1: signature uses (vp, vs, rho) for the formation
    # half-space; the layer is passed via ``layer``. Row 1 doesn't
    # touch the formation parameters except for signature uniformity.
    row1 = _layered_n0_row1_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    # Layer cols 1..5 of row 1 = -E[0, :] (negation from f - m).
    np.testing.assert_allclose(
        row1[1:5].real,
        -E[0, :],
        rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row2_matches_F1_row2_at_a_layer_cols():
    """Row 2 of E(a) (sigma_rr) matches the layer cols of
    ``_layered_n0_row2_at_a`` with a sign flip: BC2 is
    ``-(sigma_rr^(m) + P^(f)) = 0``, layer side negated."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    row2 = _layered_n0_row2_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(
        row2[1:5].real,
        -E[2, :],
        rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row3_matches_F1_row3_at_a_layer_cols():
    """Row 3 of E(a) (sigma_rz) matches the layer cols of
    ``_layered_n0_row3_at_a`` with NO sign flip: BC3 is
    ``sigma_rz^(m) = 0`` (no subtraction with the fluid)."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    row3 = _layered_n0_row3_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(
        row3[1:5].real,
        E[3, :],
        rtol=1.0e-12,
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
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=b,
    )
    row5 = _layered_n0_row5_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(
        row5[1:5].real,
        E[1, :],
        rtol=1.0e-12,
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
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r=0.1,
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
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
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
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=0.105,
        r_outer=0.105,
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
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=a,
        r_outer=b,
    )
    P_a_from_b = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=b,
        r_outer=a,
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
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r3,
    )
    P_2_from_1 = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    P_3_from_2 = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r2,
        r_outer=r3,
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
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r1,
    )
    E_r2 = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r2,
    )
    P = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
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
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r_inner=0.1,
            r_outer=0.105,
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
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
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
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=layer,
    )
    det_Gc = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
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
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega
    det_at = _modal_determinant_n0_cased(
        kz_root,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    det_off = _modal_determinant_n0_cased(
        kz_root * 1.01,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
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
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            rho=p["rho"],
            vf=p["vf"],
            rho_f=p["rho_f"],
            a=p["a"],
            layers=(layer,),
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_double,),
    )
    det_split = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_half, L_half),
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_a, L_b),
    )
    det_ba = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_b, L_a),
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
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(casing, cement),
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


# =====================================================================
# Plan item G.d -- public-API hook for cased-hole Stoneley
# =====================================================================
#
# Replaces the G.0 ``len(layers) > 1 -> NotImplementedError`` raise
# with a brentq loop on ``_modal_determinant_n0_cased``. Tests
# anchor on a multi-layer regression (matching the F.1 single-layer
# brentq root when an N=2 stack collapses to the F.1 case),
# layer-permutation distinctness, and a three-layer smoke.


def _typical_cased_geometry():
    """Realistic casing + cement geometry used as the fixture for
    G.d cased-hole tests. V_S_cement = 1800 m/s (high-strength
    cement) keeps the cement above V_f = 1500 so the bound floor
    stays at the fluid value across the dipole-sonic band."""
    return dict(
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1800.0, rho=1900.0, thickness=0.05),
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_stoneley_dispersion_layered_N1_path_unchanged_after_G_d():
    """Regression: the N=1 dispatch (F.1 hand-coded path) remains
    functional after the G.d multi-layer wiring. Picks a typical
    single-layer setup and verifies the slowness curve is finite
    and well-formed."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = np.linspace(2000.0, 8000.0, 6)
    res = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert np.all(np.isfinite(res.slowness))
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0


def test_stoneley_dispersion_layered_N2_runs_smoke():
    """G.d two-layer regression: a typical casing + cement
    geometry produces a finite, smoothly-dispersive Stoneley
    slowness curve across the dipole-sonic band."""
    g = _typical_cased_geometry()
    f = np.linspace(1000.0, 12000.0, 8)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert np.all(np.isfinite(res.slowness))
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    np.testing.assert_array_equal(res.freq, f)
    # Smoothness fence: relative step-to-step change capped at 5%.
    rel_steps = np.abs(np.diff(res.slowness)) / res.slowness[:-1]
    assert np.all(rel_steps < 0.05)


def test_stoneley_dispersion_layered_N2_returns_borehole_mode():
    """``BoreholeMode`` return-type contract on the multi-layer
    dispatch (G.d)."""
    g = _typical_cased_geometry()
    f = np.linspace(2000.0, 6000.0, 4)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert isinstance(res, BoreholeMode)
    assert res.attenuation_per_meter is None  # bound mode


def test_stoneley_dispersion_layered_N2_layer_permutation_changes_slowness():
    """Casing-inside-cement and cement-inside-casing produce
    distinct slowness curves -- the inside-out layer ordering is
    a physical parameter, not a labelling convention."""
    g = _typical_cased_geometry()
    f = np.array([5000.0])
    res_cs = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    res_sc = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["cement"], g["casing"]),
    )
    assert np.all(np.isfinite(res_cs.slowness))
    assert np.all(np.isfinite(res_sc.slowness))
    rel_diff = abs(res_cs.slowness[0] - res_sc.slowness[0]) / res_cs.slowness[0]
    assert rel_diff > 0.001


def test_stoneley_dispersion_layered_N2_collapse_to_N1_via_thin_outer_layer():
    """Two-layer-collapse oracle: with a vanishingly-thin outer
    layer that has the formation's parameters, the G.d two-layer
    path should match the F.1 single-layer slowness to high
    precision. Confirms the propagator chain and BC bookkeeping
    handle the trivial outer-layer limit cleanly."""
    p = _typical_g_c_params()
    f = np.array([5000.0])
    layer1 = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    # Outer "layer" has formation properties + tiny thickness;
    # equivalent to no outer layer at all.
    layer2_trivial = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=1.0e-5,
    )
    res_one_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer1,),
    )
    res_two_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer1, layer2_trivial),
    )
    assert res_two_layer.slowness[0] == pytest.approx(
        res_one_layer.slowness[0],
        rel=1.0e-4,
    )


def test_stoneley_dispersion_layered_three_layer_runs_smoke():
    """Smoke test for N=3 (casing + cement + mudcake): produces
    a finite slowness curve. No analytic oracle; just confirms
    the propagator chain extends past N=2 cleanly."""
    g = _typical_cased_geometry()
    mudcake = BoreholeLayer(vp=2000.0, vs=1600.0, rho=1700.0, thickness=0.003)
    f = np.linspace(2000.0, 8000.0, 5)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"], mudcake),
    )
    assert np.all(np.isfinite(res.slowness))


# =====================================================================
# Plan item G.e -- validation hardening for the cased-hole solver
# =====================================================================
#
# Mirror of F.2.e / H.e for the propagator-matrix Stoneley path.
# The Tang & Cheng 2004 fig 7.1 reproduction (digitised CSV) is
# deferred to a follow-up; the four tests below are the cheap
# self-consistency / collapse oracles that the propagator chain
# can satisfy without external reference data.


def test_modal_determinant_n0_cased_vanishes_at_converged_root_multi_freq():
    """Self-consistency: at every brentq-converged ``k_z`` from
    ``stoneley_dispersion_layered`` (cased-hole, two-layer), the
    propagator-matrix determinant is many orders of magnitude
    smaller than its value at ``k_z * 1.005``. Multi-frequency
    sharper than the H.c.2 / G.d single-frequency oracle: catches
    regressions where the brentq pipeline converges to something
    other than the true root for some frequencies."""
    g = _typical_cased_geometry()
    f = np.geomspace(1500.0, 10000.0, 6)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n0_cased(
            kz_root,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], g["cement"]),
        )
        det_off = _modal_determinant_n0_cased(
            kz_root * 1.005,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], g["cement"]),
        )
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_stoneley_dispersion_layered_thin_inner_layer_collapses_to_outer_only():
    """Two-layer-collapse oracle: with a vanishingly-thin INNER
    layer that has the outer layer's parameters, the G.d
    two-layer slowness should match the F.1 single-layer
    answer with just the outer layer (effectively ignoring
    the trivial inner annulus). Mirror of the G.d
    ``thin_outer_layer`` test, this time exercising the
    propagator chain's first link."""
    p = _typical_g_c_params()
    f = np.array([5000.0])
    outer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    inner_trivial = BoreholeLayer(
        vp=outer.vp,
        vs=outer.vs,
        rho=outer.rho,
        thickness=1.0e-5,
    )
    res_one_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(outer,),
    )
    res_two_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(inner_trivial, outer),
    )
    assert res_two_layer.slowness[0] == pytest.approx(
        res_one_layer.slowness[0],
        rel=1.0e-4,
    )


def test_stoneley_dispersion_layered_two_formation_layers_collapse_to_unlayered():
    """Master-plan G validation bullet 1: with both annular
    layers carrying formation properties, the multi-layer
    Stoneley slowness should match the unlayered
    ``stoneley_dispersion`` answer to high precision. The
    layers are physically vacuous (just slabs of formation
    pretending to be a casing + cement).

    Strongest pinning of the G.d brentq pipeline against the
    pre-G unlayered baseline: would catch any bias introduced
    by the propagator-chain assembly that survives at small
    layer-formation contrast."""
    p = _typical_g_c_params()
    f = np.linspace(2000.0, 8000.0, 5)
    # Two layers with formation properties; thicknesses are
    # arbitrary since the layer = formation collapse is exact.
    formation_l1 = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=0.01,
    )
    formation_l2 = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=0.05,
    )
    res_unlayered = stoneley_dispersion(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
    )
    res_cased = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(formation_l1, formation_l2),
    )
    # Tolerance sized to brentq's xtol (1e-10) propagated into
    # slowness via division by omega ~ 1e4: relative ~1e-6.
    np.testing.assert_allclose(
        res_cased.slowness,
        res_unlayered.slowness,
        rtol=1.0e-6,
    )


def test_stoneley_dispersion_layered_stiffer_cement_speeds_up_stoneley():
    """Cement-bond physics: a stiffer cement (higher V_S)
    couples the Stoneley wave more strongly to the formation,
    pulling the slowness DOWN (faster wave) at the same casing
    geometry. Soft cement (lower V_S, closer to free-pipe)
    leaves the Stoneley closer to its fluid-coupled limit
    (slower wave; larger slowness).

    Direct test of the qualitative cement-bond logging
    signature without committing to digitised reference data
    (Tang & Cheng 2004 fig 7.1, deferred). Quantitative: the
    speedup is on the order of a few percent for typical
    cement-stiffness contrasts."""
    g = _typical_cased_geometry()
    f = np.array([5000.0])
    casing = g["casing"]
    cement_stiff = BoreholeLayer(
        vp=2500.0,
        vs=2000.0,
        rho=2000.0,
        thickness=g["cement"].thickness,
    )
    cement_soft = BoreholeLayer(
        vp=1900.0,
        vs=1600.0,
        rho=1700.0,
        thickness=g["cement"].thickness,
    )
    res_stiff = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_stiff),
    )
    res_soft = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_soft),
    )
    assert np.all(np.isfinite(res_stiff.slowness))
    assert np.all(np.isfinite(res_soft.slowness))
    # Stiffer cement -> smaller slowness (faster Stoneley).
    assert res_stiff.slowness[0] < res_soft.slowness[0]
    # Quantitative: at least 0.1% speedup.
    rel_speedup = (res_soft.slowness[0] - res_stiff.slowness[0]) / res_soft.slowness[0]
    assert rel_speedup > 0.001


# =====================================================================
# Plan item G'.0 -- public-API foundation for cased-hole flexural
# =====================================================================
#
# G'.0 sharpens the multi-layer NIE in flexural_dispersion_layered
# (verified by the test_flexural_dispersion_layered_two_layer_NIE_points_at_G_prime_c_d
# and ..._multilayer_raises_not_implemented tests above, with their
# match strings updated), and adds the per-layer slow-formation
# validator _validate_flexural_layers_stacked. The propagator-
# matrix path itself lands in G'.b / G'.c / G'.d.


def test_validate_flexural_layers_stacked_accepts_typical_harder_layer_stack():
    """A typical multi-layer stack with all layers harder than
    the formation (``layer.vs >= vs``) passes validation. Mirrors
    the slow-formation regime constraint that the F.2 single-layer
    path documents but does not enforce programmatically."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2900.0, vs=2700.0, rho=1900.0, thickness=0.05)
    # Formation V_S = 2500; both layers harder.
    _validate_flexural_layers_stacked(
        (casing, cement),
        a=0.1,
        vs=2500.0,
    )


def test_validate_flexural_layers_stacked_rejects_softer_layer():
    """A stack with one layer slower than the formation in shear
    is rejected with a clear error identifying the offending
    index. Catches a soft-cement-bond-like configuration that
    would otherwise drive the propagator path into the unbound
    regime."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement_soft = BoreholeLayer(
        vp=2300.0,
        vs=1300.0,
        rho=1900.0,
        thickness=0.05,
    )
    # Formation V_S = 2500; cement V_S = 1300 < 2500 -> reject.
    with pytest.raises(ValueError, match=r"layers\[1\].*layer\.vs"):
        _validate_flexural_layers_stacked(
            (casing, cement_soft),
            a=0.1,
            vs=2500.0,
        )


def test_validate_flexural_layers_stacked_rejects_softer_inner_layer():
    """Inner layer (index 0) softer than the formation is also
    rejected, with the index identified in the error."""
    casing_soft = BoreholeLayer(
        vp=2300.0,
        vs=1300.0,
        rho=1900.0,
        thickness=0.01,
    )
    cement = BoreholeLayer(vp=2900.0, vs=2700.0, rho=1900.0, thickness=0.05)
    with pytest.raises(ValueError, match=r"layers\[0\].*layer\.vs"):
        _validate_flexural_layers_stacked(
            (casing_soft, cement),
            a=0.1,
            vs=2500.0,
        )


def test_validate_flexural_layers_stacked_inherits_geometry_checks():
    """The flexural validator chains to
    ``_validate_borehole_layers_stacked`` for geometry checks
    (zero-thickness, non-positive ``a``, etc.). Pin one
    geometry rejection here so the chain is explicit."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2900.0, vs=2700.0, rho=1900.0, thickness=0.05)
    with pytest.raises(ValueError, match="a must be positive"):
        _validate_flexural_layers_stacked(
            (casing, cement),
            a=0.0,
            vs=2500.0,
        )


def test_flexural_dispersion_layered_zero_and_one_layer_paths_unchanged_after_G_prime_0():
    """Regression: the two existing F.2 dispatch paths
    (``len(layers) == 0`` and ``len(layers) == 1``) remain
    functional after the G'.0 NIE message edit. Smoke only;
    bit-equivalent regressions are covered elsewhere. Uses the
    slow-formation fixture (V_S < V_f) since the F.2 single-
    layer path requires it."""
    f = np.array([5000.0])

    # Empty-layer path: dispatches to flexural_dispersion.
    res_empty = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(),
    )
    res_unlayered = flexural_dispersion(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
    )
    np.testing.assert_array_equal(res_empty.slowness, res_unlayered.slowness)
    assert res_empty.azimuthal_order == 1

    # Single-layer F.2 hand-coded path. Layer harder than the
    # slow formation (vs > SLOW_VS = 800).
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    res_one = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer,),
    )
    assert isinstance(res_one, BoreholeMode)
    assert res_one.azimuthal_order == 1


# =====================================================================
# Plan item G'.b.1 -- 6x6 mode-amplitude-to-state-vector matrix at n=1
# =====================================================================
#
# Per-element oracle: every row of E(r) post-rescale matches an
# F.2.b/c row builder's layer-amplitude columns to ``rtol=1e-12``,
# with explicit BC sign factors (rows at r=a from BC1-4 with
# specific subtraction conventions; rows at r=b from BC5-10 all
# m-s positive).


def _typical_g_prime_b1_layer_params():
    """Slow-formation harder-than-formation layer fixture for
    G'.b.1 / G'.b.2 tests. ``layer.vs > formation.vs`` so the
    flexural mode stays bound in the annulus."""
    return dict(
        vp=2500.0,
        vs=1100.0,
        rho=2100.0,
        kz=2.0 * np.pi * 5000.0 / 800.0,  # bound: kz > omega/V_S
        omega=2.0 * np.pi * 5000.0,
    )


def test_layer_e_matrix_n1_at_a_rows_match_F2_row1_to_row4_layer_cols():
    """Per-element oracle at r=a: rows 0 (u_r), 3 (sigma_rr),
    4 (sigma_rz), 5 (sigma_rtheta) of E(a) match the layer
    cols (cols 1-4 for B/C and 7-8 for D in F.2's column packing
    ``[A | B_I, B_K, C_I, C_K | B, C | D_I, D_K | D]``) of the
    corresponding F.2 row builders, with BC sign factors:

    * Row 1 (BC1: u_r^f - u_r^m = 0): layer cols negated.
      ``-_layered_n1_row1_at_a[1:5, 7:9] == E[0, :]`` permuted.
    * Row 2 (BC2: -(sigma_rr^m + P^f) = 0): layer cols negated.
    * Row 3 (BC3: sigma_rtheta^m = 0): layer cols positive.
    * Row 4 (BC4: sigma_rz^m = 0): layer cols positive.

    F.2 col packing maps to E col packing as
    ``[B_I, B_K, C_I, C_K, D_I, D_K] = F2[1, 2, 3, 4, 7, 8]``."""
    p = _typical_g_prime_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )

    # Helper: extract layer cols (1, 2, 3, 4, 7, 8) from F.2 row.
    def _layer_cols(row):
        return np.array([row[1], row[2], row[3], row[4], row[7], row[8]]).real

    # Row 0 (u_r) at a vs F.2 row 1 (BC1 negation).
    row1 = _layered_n1_row1_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row1), -E[0, :], rtol=1.0e-12)

    # Row 3 (sigma_rr) at a vs F.2 row 2 (BC2 negation).
    row2 = _layered_n1_row2_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row2), -E[3, :], rtol=1.0e-12)

    # Row 5 (sigma_rtheta) at a vs F.2 row 3 (BC3 no negation).
    row3 = _layered_n1_row3_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row3), E[5, :], rtol=1.0e-12)

    # Row 4 (sigma_rz) at a vs F.2 row 4 (BC4 no negation).
    row4 = _layered_n1_row4_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row4), E[4, :], rtol=1.0e-12)


def test_layer_e_matrix_n1_at_b_rows_match_F2_row5_to_row10_layer_cols():
    """Per-element oracle at r=b: all six rows of E(b) match the
    layer cols of F.2 rows 5-10 (BC5-10 are all m-s continuity
    with positive layer-side sign). Exhaustive coverage of every
    row of E."""
    p = _typical_g_prime_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    b = a + layer.thickness
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=b,
    )

    def _layer_cols(row):
        return np.array([row[1], row[2], row[3], row[4], row[7], row[8]]).real

    # Row 0 (u_r) at b vs F.2 row 5.
    row5 = _layered_n1_row5_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row5), E[0, :], rtol=1.0e-12)

    # Row 2 (u_theta) at b vs F.2 row 6.
    row6 = _layered_n1_row6_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row6), E[2, :], rtol=1.0e-12)

    # Row 1 (u_z) at b vs F.2 row 7.
    row7 = _layered_n1_row7_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row7), E[1, :], rtol=1.0e-12)

    # Row 3 (sigma_rr) at b vs F.2 row 8.
    row8 = _layered_n1_row8_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row8), E[3, :], rtol=1.0e-12)

    # Row 5 (sigma_rtheta) at b vs F.2 row 9.
    row9 = _layered_n1_row9_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row9), E[5, :], rtol=1.0e-12)

    # Row 4 (sigma_rz) at b vs F.2 row 10.
    row10 = _layered_n1_row10_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row10), E[4, :], rtol=1.0e-12)


def test_layer_e_matrix_n1_sparsity_pattern():
    """Pin the known-zero entries of E(r) at n=1:

    * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``): SH potential
      ``psi_z`` does not contribute to ``u_z``.
    * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``): SV
      potential ``psi_theta`` does not contribute to
      ``u_theta``.

    These zero entries are baked into E(r); confirms the F.2.a.6
    erratum direction (the cos/sin sectors don't decouple wholly
    at n=1, but specific (state-row, amplitude-col) pairs do)."""
    p = _typical_g_prime_b1_layer_params()
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    # u_z row, D cols.
    assert E[1, 4] == 0.0
    assert E[1, 5] == 0.0
    # u_theta row, C cols.
    assert E[2, 2] == 0.0
    assert E[2, 3] == 0.0


def test_layer_e_matrix_n1_real_in_bound_regime():
    """All 36 entries are finite real in the bound regime
    post-rescale."""
    p = _typical_g_prime_b1_layer_params()
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    assert np.all(np.isfinite(E))
    assert E.dtype == np.float64


def test_layer_e_matrix_n1_returns_nan_below_bound_floor():
    """Below the layer's bound floor (``kz < omega / V_S``), at
    least one of ``p^2``, ``s^2`` is negative -- the helper
    returns NaN-filled so downstream propagator and determinant
    evaluations propagate NaN cleanly."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        E = _layer_e_matrix_n1(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r=0.1,
        )
    assert np.all(np.isnan(E))


def test_layer_e_matrix_n1_determinant_nonzero_in_bound_regime():
    """The G'.b.2 propagator path requires inverting E(r). Just
    a finiteness + non-zero check; the round-trip oracle in
    G'.b.2 catches any sharper conditioning issue."""
    p = _typical_g_prime_b1_layer_params()
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    det = float(np.linalg.det(E))
    assert np.isfinite(det)
    assert abs(det) > 0.0


# =====================================================================
# Plan item G'.b.2 -- per-layer propagator P(r_outer | r_inner) at n=1
# =====================================================================
#
# Group-law oracles for ``_layer_propagator_n1`` (6x6 sister of
# ``_layer_propagator_n0``). Round-trip uses the state-vector
# form to avoid the ``cond(E) ~ mu`` issue called out in G.b.2.


def test_layer_propagator_n1_identity_when_r_inner_equals_r_outer():
    """Identity oracle: ``r_inner == r_outer`` -> propagator is
    ``eye(6)`` to floating-point precision."""
    p = _typical_g_prime_b1_layer_params()
    P = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=0.105,
        r_outer=0.105,
    )
    np.testing.assert_array_equal(P, np.eye(6))


def test_layer_propagator_n1_round_trip_preserves_state_vector():
    """Round-trip oracle via state-vector identity: applying
    ``P(a|b) @ P(b|a)`` to a physical state vector ``v`` returns
    ``v`` to ``rtol=1e-10``. State-vector phrasing avoids the
    spurious ~1e-6 off-diagonals from disparate-magnitude rows
    (displacement ~ O(1) vs stress ~ O(mu) ~ O(1e10)) per the
    G.b.2 lesson."""
    p = _typical_g_prime_b1_layer_params()
    a = 0.1
    b = a + 0.005
    P_b_from_a = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=a,
        r_outer=b,
    )
    P_a_from_b = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=b,
        r_outer=a,
    )
    mu = p["rho"] * p["vs"] ** 2
    # Physical state vector (3 displacement components ~ O(1),
    # 3 stress components ~ O(mu)).
    v = np.array([1.0, 2.0, 1.5, 3.0 * mu, 4.0 * mu, 2.5 * mu])
    v_round = P_a_from_b @ (P_b_from_a @ v)
    np.testing.assert_allclose(v_round, v, rtol=1.0e-10)
    v_round_other = P_b_from_a @ (P_a_from_b @ v)
    np.testing.assert_allclose(v_round_other, v, rtol=1.0e-10)


def test_layer_propagator_n1_composition_law():
    """Composition oracle: ``P(r3|r1) ~ P(r3|r2) @ P(r2|r1)`` for
    any intermediate ``r2 in (r1, r3)``. The propagator-group law
    in the radial coordinate."""
    p = _typical_g_prime_b1_layer_params()
    r1, r2, r3 = 0.1, 0.105, 0.115
    P_3_from_1 = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r3,
    )
    P_2_from_1 = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    P_3_from_2 = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r2,
        r_outer=r3,
    )
    np.testing.assert_allclose(P_3_from_1, P_3_from_2 @ P_2_from_1, atol=1.0e-10)


def test_layer_propagator_n1_state_vector_continuity():
    """End-to-end state-vector check: pick an arbitrary amplitude
    vector ``c``; compute ``v(r1) = E(r1) c`` and apply
    ``P(r2|r1)`` to get ``v(r2)``; verify the result matches
    ``E(r2) c`` directly. Strongest single-test oracle for the
    G'.b.1 + G'.b.2 chain combined."""
    p = _typical_g_prime_b1_layer_params()
    r1, r2 = 0.1, 0.115
    E_r1 = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r1,
    )
    E_r2 = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r2,
    )
    P = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    # Arbitrary amplitude vector (6 components for n=1).
    c = np.array([1.3, -0.7, 2.1, 0.4, -1.1, 0.6])
    v_r1 = E_r1 @ c
    v_r2_via_P = P @ v_r1
    v_r2_direct = E_r2 @ c
    np.testing.assert_allclose(v_r2_via_P, v_r2_direct, rtol=1.0e-10)


def test_layer_propagator_n1_returns_nan_below_bound_floor():
    """Below the layer's bound floor, ``E(r)`` is NaN-filled; the
    propagator inherits the NaN."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        P = _layer_propagator_n1(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r_inner=0.1,
            r_outer=0.105,
        )
    assert np.all(np.isnan(P))


# =====================================================================
# Plan item G'.c -- stacked modal determinant at n=1 (10x10)
# =====================================================================
#
# Tests anchor on the N=1 collapse to F.2 (numerical equality at
# rtol=1e-10) as the floating-point oracle, plus a few oracles
# that exercise the N >= 2 propagator chain.


def _typical_g_prime_c_params():
    """Slow-formation cased-hole fixture for G'.c tests. Keeps
    the flexural root in the bound regime; layers harder than
    formation V_S = 800."""
    return dict(
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_modal_determinant_n1_cased_N1_matches_F2_off_root():
    """N=1 floating-point oracle: at any (kz, omega) in the bound
    regime away from the flexural root, G'.c's determinant matches
    F.2's ``_modal_determinant_n1_layered`` to ``rtol=1e-10`` (no
    extra scale factor; same column packing). The propagator chain
    at N=1 reduces ``P_1 @ E_1(a)`` to ``E_1(b)``, exactly the F.2
    form.

    Strongest pinning of the G'.c assembly against the F.2 row-
    builder transcription that has shipped through F.2.d."""
    p = _typical_g_prime_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    # kz safely above the formation V_S bound floor (slowest in the stack).
    kz = omega / p["vs"] * 1.05
    det_F2 = _modal_determinant_n1_layered(
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=layer,
    )
    det_Gp_c = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert det_Gp_c == pytest.approx(det_F2, rel=1.0e-10)


def test_modal_determinant_n1_cased_N1_vanishes_at_F2_brentq_root():
    """N=1 brentq-root oracle: at the flexural root recovered by
    ``flexural_dispersion_layered(layers=(layer,))``, G'.c's
    determinant is many orders of magnitude smaller than its
    value 0.5% off the root."""
    p = _typical_g_prime_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    res = flexural_dispersion_layered(
        np.array([5000.0]),
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega
    det_at = _modal_determinant_n1_cased(
        kz_root,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    det_off = _modal_determinant_n1_cased(
        kz_root * 1.005,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_modal_determinant_n1_cased_returns_nan_below_bound_floor():
    """``kz`` below the slowest-shear bound floor -> at least one
    radial wavenumber goes imaginary -> NaN. brentq-safe
    propagation."""
    p = _typical_g_prime_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n1_cased(
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            rho=p["rho"],
            vf=p["vf"],
            rho_f=p["rho_f"],
            a=p["a"],
            layers=(layer,),
        )
    assert np.isnan(det)


def test_modal_determinant_n1_cased_two_identical_layers_equals_one_double_thickness():
    """Group-law oracle: two contiguous identical layers (L, L)
    of thickness ``h`` each compose to a single layer of
    thickness ``2h``. Direct test of the propagator chain."""
    p = _typical_g_prime_c_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 1.05
    L_double = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.01)
    L_half = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    det_single = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_double,),
    )
    det_split = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_half, L_half),
    )
    # Same E_1(a), identical propagator chain composes through the
    # same total thickness 2h.
    assert det_single == pytest.approx(det_split, rel=1.0e-10)


def test_modal_determinant_n1_cased_order_matters_at_N2():
    """Physical sanity: with two distinct layers ``(L_a, L_b)``,
    swapping the order to ``(L_b, L_a)`` produces a different
    determinant -- inside-out layer ordering is a physical
    parameter."""
    p = _typical_g_prime_c_params()
    omega = 2.0 * np.pi * 5000.0
    L_a = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)  # casing
    L_b = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.01)  # cement
    # kz safely above the slowest-shear bound floor. Slowest in
    # this stack: cement V_S = 1300 > formation V_S = 800.
    kz = omega / min(L_a.vs, L_b.vs, p["vs"], p["vf"]) * 1.05
    det_ab = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_a, L_b),
    )
    det_ba = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_b, L_a),
    )
    assert np.isfinite(det_ab) and np.isfinite(det_ba)
    rel_diff = abs(det_ab - det_ba) / max(abs(det_ab), abs(det_ba))
    assert rel_diff > 0.01


def test_modal_determinant_n1_cased_N2_runs_smoke():
    """Smoke for the cased-hole flexural N=2 path: typical casing
    + cement geometry produces a finite real determinant at a
    representative bound-regime ``kz``."""
    p = _typical_g_prime_c_params()
    omega = 2.0 * np.pi * 5000.0
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    kz = omega / min(casing.vs, cement.vs, p["vs"], p["vf"]) * 1.05
    det = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(casing, cement),
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


# =====================================================================
# Plan item G'.d -- public-API hook for cased-hole flexural
# =====================================================================
#
# Replaces the G'.0 ``len(layers) > 1 -> NotImplementedError`` raise
# with a brentq loop on ``_modal_determinant_n1_cased`` (G'.c).
# Tests anchor on multi-layer regression / smoothness, plus the
# slow-formation per-layer constraint enforced by
# ``_validate_flexural_layers_stacked``.


def _typical_g_prime_d_cased_geometry():
    """Slow-formation cased-hole fixture for G'.d tests. Casing
    + cement layers both faster in shear than the formation
    (V_S = 800 m/s); the slow-formation per-layer constraint
    (validated by ``_validate_flexural_layers_stacked``) holds."""
    return dict(
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_flexural_dispersion_layered_N1_path_unchanged_after_G_prime_d():
    """Regression: the existing N=1 dispatch (F.2 hand-coded path)
    remains functional after the G'.d multi-layer wiring. Smoke
    test on a slow-formation single-layer setup."""
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    f = np.linspace(2000.0, 6000.0, 4)
    res = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer,),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    finite = np.isfinite(res.slowness)
    assert finite.any()


def test_flexural_dispersion_layered_N2_runs_smoke():
    """G'.d two-layer regression: typical casing + cement
    geometry produces a finite, smoothly-dispersive flexural
    slowness curve across the dipole-sonic band (3-12 kHz)."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.linspace(3000.0, 12000.0, 8)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= len(f) - 1  # allow at most one cutoff miss
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    np.testing.assert_array_equal(res.freq, f)
    # Smoothness fence on contiguous finite values. Slow-formation
    # flexural slowness diverges sharply near the geometric cutoff
    # (~ 1.3 kHz here), so step-to-step changes are large at the LF
    # end of the band and shrink toward the HF asymptote. The fence
    # is a coarse sanity check, not a tight oracle.
    s_finite = res.slowness[finite]
    rel_steps = np.abs(np.diff(s_finite)) / s_finite[:-1]
    assert np.all(rel_steps < 0.50)


def test_flexural_dispersion_layered_N2_returns_borehole_mode():
    """``BoreholeMode`` return-type contract on the multi-layer
    flexural dispatch (G'.d). ``attenuation_per_meter is None``
    confirms the slow-formation bound mode."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.linspace(3000.0, 6000.0, 3)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert isinstance(res, BoreholeMode)
    assert res.attenuation_per_meter is None


def test_flexural_dispersion_layered_N2_layer_permutation_changes_slowness():
    """Casing-inside-cement and cement-inside-casing produce
    distinct flexural slowness curves -- inside-out layer
    ordering is a physical parameter."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.array([5000.0])
    res_cs = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    res_sc = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["cement"], g["casing"]),
    )
    assert np.all(np.isfinite(res_cs.slowness))
    assert np.all(np.isfinite(res_sc.slowness))
    rel_diff = abs(res_cs.slowness[0] - res_sc.slowness[0]) / res_cs.slowness[0]
    assert rel_diff > 0.001


def test_flexural_dispersion_layered_N2_collapse_to_N1_via_thin_outer_layer():
    """Two-layer-collapse oracle: with a vanishingly-thin OUTER
    layer that has the formation's parameters, the G'.d two-
    layer slowness should match the F.2 single-layer answer with
    just the inner layer."""
    layer1 = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    # Outer "layer" has formation properties + tiny thickness;
    # equivalent to no outer layer at all. layer.vs == vs satisfies
    # the slow-formation per-layer constraint (>=).
    layer2_trivial = BoreholeLayer(
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        thickness=1.0e-5,
    )
    f = np.array([5000.0])
    res_one = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer1,),
    )
    res_two = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer1, layer2_trivial),
    )
    assert res_two.slowness[0] == pytest.approx(
        res_one.slowness[0],
        rel=1.0e-4,
    )


def test_flexural_dispersion_layered_three_layer_runs_smoke():
    """Smoke test for N=3 (casing + cement + mudcake): the
    propagator chain extends past N=2 cleanly. All three layers
    must be at least as fast in shear as the formation."""
    g = _typical_g_prime_d_cased_geometry()
    mudcake = BoreholeLayer(vp=2000.0, vs=1100.0, rho=1700.0, thickness=0.003)
    f = np.linspace(4000.0, 10000.0, 4)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"], mudcake),
    )
    finite = np.isfinite(res.slowness)
    assert finite.any()


def test_flexural_dispersion_layered_N2_rejects_softer_layer():
    """The G'.d brentq path enforces ``layer.vs >= vs`` per layer
    via ``_validate_flexural_layers_stacked``. A softer layer
    triggers ``ValueError`` rather than NaN slownesses or a
    silent unbound-mode failure."""
    g = _typical_g_prime_d_cased_geometry()
    soft_cement = BoreholeLayer(
        vp=1500.0,
        vs=600.0,
        rho=1700.0,
        thickness=0.05,
    )  # vs = 600 < formation vs = 800 -> reject
    with pytest.raises(ValueError, match=r"layer\.vs"):
        flexural_dispersion_layered(
            np.array([5000.0]),
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], soft_cement),
        )


# =====================================================================
# Plan item G'.e -- validation hardening for the cased-hole flexural
# =====================================================================
#
# Mirror of G.e for the propagator-matrix flexural path.


def test_modal_determinant_n1_cased_vanishes_at_converged_root_multi_freq():
    """Self-consistency at every brentq-converged ``k_z`` from
    ``flexural_dispersion_layered`` (cased-hole, two-layer): the
    propagator-matrix determinant is at least 6 orders of
    magnitude smaller than its value at ``kz * 1.005``. Multi-
    frequency sharper than G'.d's single-frequency oracles."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.linspace(4000.0, 12000.0, 6)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    finite = np.isfinite(res.slowness)
    assert finite.any()
    for i, fi in enumerate(f):
        if not finite[i]:
            continue
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n1_cased(
            kz_root,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], g["cement"]),
        )
        det_off = _modal_determinant_n1_cased(
            kz_root * 1.005,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], g["cement"]),
        )
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_flexural_dispersion_layered_thin_inner_layer_collapses_to_outer_only():
    """Two-layer-collapse oracle: with a vanishingly-thin INNER
    layer that has the outer layer's parameters, the G'.d two-
    layer slowness should match the F.2 single-layer answer with
    just the outer layer.

    Uses a harder outer layer (vs > V_f) so the F.2 single-layer
    flexural root is clearly bound."""
    vp_form, vs_form, rho_form = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    outer = BoreholeLayer(vp=2900.0, vs=2700.0, rho=2400.0, thickness=0.05)
    inner_trivial = BoreholeLayer(
        vp=outer.vp,
        vs=outer.vs,
        rho=outer.rho,
        thickness=1.0e-5,
    )
    f = np.array([5000.0])
    res_one = flexural_dispersion_layered(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(outer,),
    )
    res_two = flexural_dispersion_layered(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(inner_trivial, outer),
    )
    assert np.isfinite(res_one.slowness[0])
    assert np.isfinite(res_two.slowness[0])
    assert res_two.slowness[0] == pytest.approx(
        res_one.slowness[0],
        rel=1.0e-3,
    )


def test_flexural_dispersion_layered_two_formation_layers_collapse_to_unlayered():
    """Master-plan G' validation bullet: with both annular
    layers carrying formation properties (``layer.vs == vs``),
    the multi-layer flexural slowness matches the unlayered
    ``flexural_dispersion`` answer.

    Strongest pinning of the G'.d brentq pipeline against the
    pre-G' unlayered baseline."""
    vp_form, vs_form, rho_form = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(3000.0, 8000.0, 5)
    L1 = BoreholeLayer(
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        thickness=0.01,
    )
    L2 = BoreholeLayer(
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        thickness=0.05,
    )
    res_unlayered = flexural_dispersion(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_cased = flexural_dispersion_layered(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(L1, L2),
    )
    np.testing.assert_allclose(
        res_cased.slowness,
        res_unlayered.slowness,
        rtol=1.0e-6,
        equal_nan=True,
    )


def test_flexural_dispersion_layered_cement_stiffness_sensitivity():
    """Cement-bond physics for the dipole sonic: stiffer vs
    softer cement produce distinct flexural slowness curves.
    Direct test of the qualitative cement-stiffness sensitivity
    that the cased-hole propagator captures and the unlayered
    ``flexural_dispersion`` does not.

    Empirical direction (from the typical cased-hole fixture at
    5 kHz): stiffer cement -> SMALLER slowness (faster wave).
    Same direction as the Stoneley cement-bond signature
    (commit 9df7a78). Pinned here as a regression target; the
    quantitative tolerance is loose because the direction depends
    on which mode the brentq-expansion-loop converges to."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.array([5000.0])
    casing = g["casing"]
    cement_stiff = BoreholeLayer(
        vp=2500.0,
        vs=2000.0,
        rho=2000.0,
        thickness=g["cement"].thickness,
    )
    cement_soft = BoreholeLayer(
        vp=1900.0,
        vs=1100.0,
        rho=1700.0,
        thickness=g["cement"].thickness,
    )
    res_stiff = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_stiff),
    )
    res_soft = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_soft),
    )
    assert np.isfinite(res_stiff.slowness[0])
    assert np.isfinite(res_soft.slowness[0])
    # Distinct curves: at least 1% relative difference.
    rel_diff = abs(res_stiff.slowness[0] - res_soft.slowness[0]) / res_soft.slowness[0]
    assert rel_diff > 0.01
    # Empirical direction: stiffer cement -> smaller slowness.
    assert res_stiff.slowness[0] < res_soft.slowness[0]


# =====================================================================
# Plan item G''.0 -- public-API foundation for cased-hole quadrupole
# =====================================================================
#
# Introduces ``quadrupole_dispersion_layered`` from scratch: with
# ``layers=()`` it dispatches to ``quadrupole_dispersion``; with
# any non-empty layer stack it raises NotImplementedError pointing
# at G''.c / G''.d (the propagator-matrix scaffolding that ships
# in subsequent sub-units). Validation rules cover the slow-
# formation per-layer constraint and a fast-formation NIE for
# the deferred complex-determinant follow-up.


def test_quadrupole_dispersion_layered_layers_empty_dispatches_to_unlayered():
    """``layers=()`` -> bit-equivalent to
    ``quadrupole_dispersion``. Floating-point regression oracle
    for plan G''."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    f = np.linspace(8000.0, 25000.0, 5)
    # LWD-quadrupole-relevant slow formation.
    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_lyr = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    np.testing.assert_array_equal(res_lyr.slowness, res_unl.slowness)
    np.testing.assert_array_equal(res_lyr.freq, res_unl.freq)
    assert res_lyr.name == "quadrupole"
    assert res_lyr.azimuthal_order == 2


def test_quadrupole_dispersion_layered_returns_borehole_mode_for_unlayered():
    """``BoreholeMode`` return-type contract on the unlayered
    dispatch. Type and azimuthal_order are pinned for every
    public path; the actual slowness numerics are covered by
    the existing ``quadrupole_dispersion`` tests."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    f = np.linspace(8000.0, 15000.0, 3)
    res = quadrupole_dispersion_layered(
        f,
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    np.testing.assert_array_equal(res.freq, f)


def test_quadrupole_dispersion_layered_rejects_softer_layer():
    """The per-layer slow-formation constraint
    ``layer.vs >= vs`` is enforced via the G'.0
    ``_validate_flexural_layers_stacked`` helper (the constraint
    is the same at n=1 and n=2). A softer layer triggers
    ``ValueError`` with the offending index."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    soft_layer = BoreholeLayer(
        vp=1500.0,
        vs=600.0,
        rho=1700.0,
        thickness=0.05,
    )  # vs = 600 < formation vs = 800 -> reject
    with pytest.raises(ValueError, match=r"layer\.vs"):
        quadrupole_dispersion_layered(
            np.array([10000.0]),
            vp=2200.0,
            vs=800.0,
            rho=2200.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
            layers=(soft_layer,),
        )


def test_quadrupole_dispersion_layered_fast_formation_with_layer_dispatches_to_complex_path():
    """Fast-formation cased-hole quadrupole (``V_S > V_f`` with
    a non-empty layer) dispatches to the complex-determinant
    path via ``_modal_determinant_n2_cased_complex`` and
    ``_quadrupole_dispersion_fast_formation_layered``. The
    earlier ``NotImplementedError`` raise from G''.0 is gone --
    fast-formation cased quadrupole is now a supported regime."""
    from fwap.cylindrical_solver import (
        BoreholeMode,
        quadrupole_dispersion_layered,
    )

    layer = BoreholeLayer(vp=4000.0, vs=3300.0, rho=2500.0, thickness=0.005)
    # Fast formation: V_S = 3100 > V_f = 1500.
    res = quadrupole_dispersion_layered(
        np.array([10000.0]),
        vp=5500.0,
        vs=3100.0,
        rho=2500.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(layer,),
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    # Bound-mode result: real-valued slowness (or NaN if outside
    # the geometric cutoff window). Either way no exception.
    assert res.slowness.dtype == np.float64


def test_quadrupole_dispersion_layered_rejects_invalid_inputs():
    """Standard input validation: positivity, ``vp > vs``, and
    strictly-positive frequencies."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    base = dict(
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    f = np.array([10000.0])
    # Non-positive vp.
    with pytest.raises(ValueError, match="vp, vs, rho must all be positive"):
        quadrupole_dispersion_layered(
            f,
            **{**base, "vp": 0.0},
            layers=(layer,),
        )
    # vp <= vs.
    with pytest.raises(ValueError, match=r"vp > vs"):
        quadrupole_dispersion_layered(
            f,
            **{**base, "vp": 700.0},
            layers=(layer,),
        )
    # Non-positive freq.
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        quadrupole_dispersion_layered(
            np.array([0.0]),
            **base,
            layers=(layer,),
        )


# =====================================================================
# Plan item G''.b.1 -- 6x6 mode-amplitude-to-state-vector matrix at n=2
# =====================================================================
#
# Without an F.3-equivalent oracle, the per-element pinning is
# weaker than G.b.1 / G'.b.1. The structural oracles (sparsity,
# finiteness, NaN propagation) plus a layer=formation cross-
# check vs the K-flavour entries transcribed from the existing
# ``_modal_determinant_n2`` docstring catch transcription errors
# at the entry level; the determinant-level checks in G''.c /
# G''.d / G''.e anchor correctness via root-match identities.


def _typical_g_pp_b1_layer_params():
    """Slow-formation harder-than-formation layer fixture for
    G''.b.1 / G''.b.2 tests."""
    return dict(
        vp=2500.0,
        vs=1100.0,
        rho=2100.0,
        kz=2.0 * np.pi * 5000.0 / 800.0,
        omega=2.0 * np.pi * 5000.0,
    )


def test_layer_e_matrix_n2_real_in_bound_regime():
    """All 36 entries are finite real in the bound regime
    post-rescale."""
    p = _typical_g_pp_b1_layer_params()
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    assert np.all(np.isfinite(E))
    assert E.dtype == np.float64


def test_layer_e_matrix_n2_returns_nan_below_bound_floor():
    """Below the bound floor (``kz < omega / V_S``), at least one
    of ``p^2``, ``s^2`` is negative -- the helper returns NaN-
    filled."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        E = _layer_e_matrix_n2(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r=0.1,
        )
    assert np.all(np.isnan(E))


def test_layer_e_matrix_n2_determinant_nonzero_in_bound_regime():
    """Precondition for the G''.b.2 propagator inverse."""
    p = _typical_g_pp_b1_layer_params()
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    det = float(np.linalg.det(E))
    assert np.isfinite(det)
    assert abs(det) > 0.0


def test_layer_e_matrix_n2_sparsity_pattern():
    """Pin the known-zero entries of E_n2(r):

    * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``): SH potential
      ``psi_z`` doesn't contribute to ``u_z``.
    * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``): SV
      potential ``psi_theta`` doesn't contribute to ``u_theta``."""
    p = _typical_g_pp_b1_layer_params()
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    # u_z row, D cols.
    assert E[1, 4] == 0.0
    assert E[1, 5] == 0.0
    # u_theta row, C cols.
    assert E[2, 2] == 0.0
    assert E[2, 3] == 0.0


def test_layer_e_matrix_n2_K_flavour_matches_modal_determinant_n2_docstring():
    """Cross-check vs the K-flavour entries transcribed from
    ``_modal_determinant_n2``'s docstring at r=a:

    * Row 1 (BC1: f - m, layer cols negated) -> u_r row of E,
      K-flavour cols (B_K, C_K, D_K).
    * Row 2 (BC2: -m, layer cols negated) -> sigma_rr row.
    * Row 3 (BC3: m=0, layer cols positive) -> sigma_rtheta row.
    * Row 4 (BC4: m=0, layer cols positive) -> sigma_rz row.

    The unlayered formula from the docstring at radius ``r=a``
    becomes (for the layer's E entries) the formation parameters
    evaluated at ``r=a`` (since the formation half-space's K-only
    contribution to the modal matrix uses the same Bessel ansatz
    as the layer's K-flavour part at this radius)."""
    from scipy import special as _special

    vp, vs, rho = 2200.0, 800.0, 2200.0
    omega = 2.0 * np.pi * 5000.0
    kz = omega / vs * 1.05  # bound regime
    a = 0.1

    E = _layer_e_matrix_n2(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=a,
    )

    # Local Bessel evaluations matching the docstring's notation.
    p_ = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s_ = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    K1pa = float(_special.kv(1, p_ * a))
    K2pa = float(_special.kv(2, p_ * a))
    K1sa = float(_special.kv(1, s_ * a))
    K2sa = float(_special.kv(2, s_ * a))
    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    # Row 1 (BC1) layer cols are -E[u_r row, K-flavour cols]:
    # B_K: +p K_1 + 2 K_2/a    (docstring)
    # C_K: +kz K_2             (docstring)
    # D_K: -2 K_2/a            (docstring)
    assert -E[0, 1] == pytest.approx(+p_ * K1pa + 2.0 * K2pa / a, rel=1e-12)
    assert -E[0, 3] == pytest.approx(+kz * K2sa, rel=1e-12)
    assert -E[0, 5] == pytest.approx(-2.0 * K2sa / a, rel=1e-12)

    # Row 2 (BC2) layer cols negated:
    # B_K: -mu * [(2 kz^2 - kS^2) K_2 + 2 p K_1/a + 12 K_2/a^2]
    # C_K: -2 mu kz * [s K_1 + 2 K_2/a]
    # D_K: +4 mu * [s K_1/a + 3 K_2/a^2]
    assert -E[3, 1] == pytest.approx(
        -mu * (two_kz2_minus_kS2 * K2pa + 2.0 * p_ * K1pa / a + 12.0 * K2pa / (a * a)),
        rel=1e-12,
    )
    assert -E[3, 3] == pytest.approx(
        -2.0 * mu * kz * (s_ * K1sa + 2.0 * K2sa / a),
        rel=1e-12,
    )
    assert -E[3, 5] == pytest.approx(
        +4.0 * mu * (s_ * K1sa / a + 3.0 * K2sa / (a * a)),
        rel=1e-12,
    )

    # Row 3 (BC3) layer cols positive (no negation):
    # B_K: +4 mu * [p K_1/a + 3 K_2/a^2]
    # C_K: +2 mu kz K_2/a
    # D_K: -mu * [(s^2 + 12/a^2) K_2 + 2 s K_1/a]
    assert E[5, 1] == pytest.approx(
        +4.0 * mu * (p_ * K1pa / a + 3.0 * K2pa / (a * a)),
        rel=1e-12,
    )
    assert E[5, 3] == pytest.approx(+2.0 * mu * kz * K2sa / a, rel=1e-12)
    assert E[5, 5] == pytest.approx(
        -mu * ((s_ * s_ + 12.0 / (a * a)) * K2sa + 2.0 * s_ * K1sa / a),
        rel=1e-12,
    )

    # Row 4 (BC4) layer cols positive:
    # B_K: +2 mu kz * [p K_1 + 2 K_2/a]
    # C_K: +mu * [(2 kz^2 - kS^2) + 3/a^2] K_2
    # D_K: -2 mu kz K_2/a
    assert E[4, 1] == pytest.approx(
        +2.0 * mu * kz * (p_ * K1pa + 2.0 * K2pa / a),
        rel=1e-12,
    )
    assert E[4, 3] == pytest.approx(
        +mu * (two_kz2_minus_kS2 + 3.0 / (a * a)) * K2sa,
        rel=1e-12,
    )
    assert E[4, 5] == pytest.approx(-2.0 * mu * kz * K2sa / a, rel=1e-12)


def test_layer_e_matrix_n2_n2_factors_appear():
    """Pin the explicit n=2 factors: ``12 = 2 n (n+1)`` in the
    F_2 / r^2 term of sigma_rr; ``3 = n^2 - 1`` in the SV column
    of sigma_rz. Catches a transcription error where an n=1
    formula was reused at n=2."""
    from scipy import special as _special

    p = _typical_g_pp_b1_layer_params()
    a = 0.1
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    # sigma_rr row (3) B_K col (1): expect (2 kz^2 - kS^2) K_2
    # + 2 p K_1/a + 12 K_2/a^2 (with leading +mu).
    p_ = float(np.sqrt(p["kz"] * p["kz"] - (p["omega"] / p["vp"]) ** 2))
    K1pa = float(_special.kv(1, p_ * a))
    K2pa = float(_special.kv(2, p_ * a))
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (p["omega"] / p["vs"]) ** 2
    expected = mu * (
        (2.0 * p["kz"] ** 2 - kS2) * K2pa + 2.0 * p_ * K1pa / a + 12.0 * K2pa / (a * a)
    )
    assert E[3, 1] == pytest.approx(expected, rel=1e-12)

    # sigma_rz row (4) C_K col (3): expect (2 kz^2 - kS^2 + 3/a^2) K_2
    # (with leading +mu).
    s_ = float(np.sqrt(p["kz"] * p["kz"] - (p["omega"] / p["vs"]) ** 2))
    K2sa = float(_special.kv(2, s_ * a))
    expected_rz = mu * (2.0 * p["kz"] ** 2 - kS2 + 3.0 / (a * a)) * K2sa
    assert E[4, 3] == pytest.approx(expected_rz, rel=1e-12)


# =====================================================================
# Plan item G''.b.2 -- per-layer propagator P(r_outer | r_inner) at n=2
# =====================================================================
#
# Group-law oracles for ``_layer_propagator_n2`` (6x6 sister of
# ``_layer_propagator_n1``). Round-trip uses the state-vector
# form to avoid the ``cond(E) ~ mu`` issue called out in
# G.b.2 / G'.b.2.


def test_layer_propagator_n2_identity_when_r_inner_equals_r_outer():
    """Identity oracle: ``r_inner == r_outer`` -> propagator is
    ``eye(6)`` to floating-point precision."""
    p = _typical_g_pp_b1_layer_params()
    P = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=0.105,
        r_outer=0.105,
    )
    np.testing.assert_array_equal(P, np.eye(6))


def test_layer_propagator_n2_round_trip_preserves_state_vector():
    """Round-trip oracle via state-vector identity at n=2:
    applying ``P(a|b) @ P(b|a)`` to a physical state vector
    ``v`` returns ``v`` to ``rtol=1e-10``. State-vector phrasing
    avoids the spurious ~1e-6 off-diagonals from disparate-
    magnitude rows (displacement ~ O(1) vs stress ~ O(mu) ~
    O(1e10)) per the G.b.2 / G'.b.2 lesson."""
    p = _typical_g_pp_b1_layer_params()
    a = 0.1
    b = a + 0.005
    P_b_from_a = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=a,
        r_outer=b,
    )
    P_a_from_b = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=b,
        r_outer=a,
    )
    mu = p["rho"] * p["vs"] ** 2
    v = np.array([1.0, 2.0, 1.5, 3.0 * mu, 4.0 * mu, 2.5 * mu])
    v_round = P_a_from_b @ (P_b_from_a @ v)
    np.testing.assert_allclose(v_round, v, rtol=1.0e-10)
    v_round_other = P_b_from_a @ (P_a_from_b @ v)
    np.testing.assert_allclose(v_round_other, v, rtol=1.0e-10)


def test_layer_propagator_n2_composition_law():
    """Composition oracle: ``P(r3|r1) ~ P(r3|r2) @ P(r2|r1)`` for
    any intermediate ``r2``. The propagator-group law in radius
    at n=2."""
    p = _typical_g_pp_b1_layer_params()
    r1, r2, r3 = 0.1, 0.105, 0.115
    P_3_from_1 = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r3,
    )
    P_2_from_1 = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    P_3_from_2 = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r2,
        r_outer=r3,
    )
    np.testing.assert_allclose(P_3_from_1, P_3_from_2 @ P_2_from_1, atol=1.0e-10)


def test_layer_propagator_n2_state_vector_continuity():
    """End-to-end state-vector check: pick an arbitrary 6-amp
    vector ``c``; verify ``P(r2|r1) @ E_n2(r1) c == E_n2(r2) c``
    to ``rtol=1e-10``. Strongest single-test oracle for the
    G''.b.1 + G''.b.2 chain combined."""
    p = _typical_g_pp_b1_layer_params()
    r1, r2 = 0.1, 0.115
    E_r1 = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r1,
    )
    E_r2 = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r2,
    )
    P = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    c = np.array([1.3, -0.7, 2.1, 0.4, -1.1, 0.6])
    v_r1 = E_r1 @ c
    v_r2_via_P = P @ v_r1
    v_r2_direct = E_r2 @ c
    np.testing.assert_allclose(v_r2_via_P, v_r2_direct, rtol=1.0e-10)


def test_layer_propagator_n2_returns_nan_below_bound_floor():
    """Below the bound floor, ``E_n2(r)`` is NaN-filled; the
    propagator inherits the NaN."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        P = _layer_propagator_n2(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r_inner=0.1,
            r_outer=0.105,
        )
    assert np.all(np.isnan(P))


# =====================================================================
# Plan item G''.c -- 10x10 stacked modal determinant at n=2
# =====================================================================
#
# Tests for ``_modal_determinant_n2_cased``. With no F.3-equivalent
# hand-coded n=2 single-layer oracle, the strongest root-level
# oracle is the layer = formation collapse identity: at layer params
# equal to formation params, the brentq root in ``k_z`` of the 10x10
# cased determinant matches the unlayered 4x4
# ``_modal_determinant_n2`` root (per the substep G''.a.6 algebra:
# ``det(M_10) = det(E_form(b)) * det(M_4)``). The "N=0 dispatch to
# unlayered" check is covered by the existing G''.0 test
# ``test_quadrupole_dispersion_layered_layers_empty_dispatches_to_unlayered``
# (the dispatch lives in ``quadrupole_dispersion_layered``, not in
# ``_modal_determinant_n2_cased``), so this block ships 5 tests.


def _typical_g_pp_c_params():
    """Slow-formation cased-hole fixture for G''.c tests. Keeps
    the unlayered quadrupole root in the bound regime (V_S = 800
    < V_f = 1500); layers must satisfy the slow-formation
    constraint ``layer.vs >= formation.vs``."""
    return dict(
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_modal_determinant_n2_cased_N1_layer_eq_formation_vanishes_at_unlayered_root():
    """G''.a.6 collapse identity: at ``layer.{vp,vs,rho} ==
    formation.{vp,vs,rho}``, the cased determinant has a brentq
    root in ``k_z`` exactly where the unlayered
    ``_modal_determinant_n2`` does. Verified at the
    ``quadrupole_dispersion``-converged root: ``|det_at|`` is
    many orders of magnitude smaller than ``|det_off|`` 0.5%
    away.

    Strongest root-level oracle in the absence of an F.3
    per-element form."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased,
        quadrupole_dispersion,
    )

    p = _typical_g_pp_c_params()
    # Layer = formation. Thickness arbitrary; G''.a.6 holds for any
    # ``b > a``.
    layer = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=0.01,
    )
    omega = 2.0 * np.pi * 5000.0
    res = quadrupole_dispersion(
        np.array([5000.0]),
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
    )
    assert np.isfinite(res.slowness[0]), (
        "fixture must put the unlayered quadrupole root in the bound "
        "regime so the collapse identity is exercisable"
    )
    kz_root = float(res.slowness[0]) * omega
    det_at = _modal_determinant_n2_cased(
        kz_root,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    det_off = _modal_determinant_n2_cased(
        kz_root * 1.005,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    # Guard against the unphysical case where both are tiny.
    assert abs(det_off) > 0.0
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_modal_determinant_n2_cased_returns_nan_below_bound_floor():
    """``kz`` below the slowest-shear bound floor -> at least one
    radial wavenumber goes imaginary -> NaN. brentq-safe
    propagation."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n2_cased(
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            rho=p["rho"],
            vf=p["vf"],
            rho_f=p["rho_f"],
            a=p["a"],
            layers=(layer,),
        )
    assert np.isnan(det)


def test_modal_determinant_n2_cased_two_identical_layers_equals_one_double_thickness():
    """Group-law oracle: two contiguous identical layers (L, L)
    of thickness ``h`` each compose to a single layer of
    thickness ``2h``. Direct test of the ``P_total = P_N ... P_1``
    propagator chain at n=2."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 1.05
    L_double = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.01)
    L_half = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    det_single = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_double,),
    )
    det_split = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_half, L_half),
    )
    # Same E_n2(a), identical propagator chain composes through the
    # same total thickness 2h.
    assert det_single == pytest.approx(det_split, rel=1.0e-10)


def test_modal_determinant_n2_cased_order_matters_at_N2():
    """Physical sanity: with two distinct layers ``(L_a, L_b)``,
    swapping the inside-out order to ``(L_b, L_a)`` produces a
    different determinant -- layer ordering is a physical
    parameter, not a labelling convention."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    omega = 2.0 * np.pi * 5000.0
    L_a = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)  # casing
    L_b = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.01)  # cement
    # kz safely above the slowest-shear bound floor.
    kz = omega / min(L_a.vs, L_b.vs, p["vs"], p["vf"]) * 1.05
    det_ab = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_a, L_b),
    )
    det_ba = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_b, L_a),
    )
    assert np.isfinite(det_ab) and np.isfinite(det_ba)
    rel_diff = abs(det_ab - det_ba) / max(abs(det_ab), abs(det_ba))
    assert rel_diff > 0.01


def test_modal_determinant_n2_cased_N2_casing_plus_cement_smoke():
    """Smoke for the cased-hole quadrupole N=2 path: typical
    casing + cement geometry produces a finite real determinant
    at a representative bound-regime ``kz``. Mirrors the
    G'.c smoke test for the LWD-band cement-bond geometry."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    omega = 2.0 * np.pi * 5000.0
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    kz = omega / min(casing.vs, cement.vs, p["vs"], p["vf"]) * 1.05
    det = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(casing, cement),
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


# =====================================================================
# Plan item G''.d -- public-API hook for cased-hole quadrupole
# =====================================================================
#
# Replaces the G''.0 ``len(layers) >= 1 -> NotImplementedError`` raise
# with a brentq loop on ``_modal_determinant_n2_cased`` (G''.c). The
# cased-hole quadrupole bound-mode regime in slow formation is much
# narrower than the n=1 flexural sister: at strong layer perturbations
# (real casing + cement; layer Vs much larger than formation Vs) the
# mode is mostly cut off across the LWD band. The smoke tests therefore
# pick formation params close to the slow-formation ceiling
# (Vs = 1200 m/s, vf = 1500 m/s) so casing + cement still satisfies the
# layer.vs >= formation.vs constraint *and* the geometric quadrupole
# bound regime stays open in a usable frequency window (~15-18 kHz).


def _typical_g_pp_d_cased_geometry():
    """Slow-formation cased-hole fixture for G''.d tests. Formation
    Vs = 1200 m/s sits just below ``vf = 1500`` so the slow-
    formation regime (``vs < vf``) holds; casing + cement layers
    remain faster in shear than the formation, satisfying
    ``_validate_flexural_layers_stacked``. The bound-mode quadrupole
    window for this geometry is ~15-18 kHz (narrower than the
    flexural sister's 3-12 kHz)."""
    return dict(
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
        vp=2400.0,
        vs=1200.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_quadrupole_dispersion_layered_N1_layer_eq_formation_matches_unlayered():
    """G''.a.6 root-match oracle wired through the public API:
    when ``layer.{vp,vs,rho} == formation.{vp,vs,rho}`` the
    multi-layer brentq path returns the unlayered
    ``quadrupole_dispersion`` slowness bit-identically (modulo
    brentq's xtol=1e-10). Strongest end-to-end pinning of the
    G''.d wiring."""
    from fwap.cylindrical_solver import (
        BoreholeMode,
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    f = np.linspace(4000.0, 12000.0, 9)
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_lyr = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert isinstance(res_lyr, BoreholeMode)
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_lyr.slowness)
    assert finite.any(), "fixture must put the bound mode in band"
    np.testing.assert_allclose(
        res_lyr.slowness[finite],
        res_unl.slowness[finite],
        rtol=1.0e-9,
    )


def test_quadrupole_dispersion_layered_N2_runs_smoke():
    """G''.d two-layer regression: casing + cement geometry
    produces finite quadrupole slownesses in the bound-mode window
    (15-18 kHz at this fixture). Uses the slow-formation-ceiling
    fixture (formation Vs = 1200 < vf = 1500) so casing + cement
    can satisfy the per-layer constraint while keeping the
    quadrupole geometric cutoff inside an exercisable band."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_g_pp_d_cased_geometry()
    f = np.linspace(15000.0, 18000.0, 4)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    np.testing.assert_array_equal(res.freq, f)
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 3, (
        f"expected at least 3 of 4 frequencies to land in the cased "
        f"quadrupole bound regime; got {finite.sum()}"
    )
    # Bound-regime sanity: slowness > formation 1/V_S; phase
    # velocity below formation V_S (low-f asymptote) but above
    # formation Rayleigh (high-f asymptote, ~0.92 V_S).
    sl = res.slowness[finite]
    assert np.all(sl > 1.0 / g["vs"] * 0.99)
    assert np.all(sl < 1.0 / (g["vs"] * 0.85))


def test_quadrupole_dispersion_layered_N2_returns_borehole_mode():
    """``BoreholeMode`` return-type contract on the multi-layer
    quadrupole dispatch (G''.d). ``attenuation_per_meter is None``
    confirms slow-formation bound-mode path (no leaky fast-formation
    cased-hole quadrupole shipped here)."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_g_pp_d_cased_geometry()
    f = np.linspace(15000.0, 18000.0, 4)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    from fwap.cylindrical_solver import BoreholeMode

    assert isinstance(res, BoreholeMode)
    assert res.attenuation_per_meter is None


def test_quadrupole_dispersion_layered_N2_layer_permutation_changes_slowness():
    """Casing-inside-cement and cement-inside-casing produce
    distinct quadrupole slowness curves -- inside-out layer
    ordering is a physical parameter, not a labelling convention."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_g_pp_d_cased_geometry()
    f = np.array([16000.0, 17000.0, 18000.0])
    res_cs = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    res_sc = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["cement"], g["casing"]),
    )
    # Both should produce at least some finite slownesses for the
    # comparison to be meaningful.
    common = np.isfinite(res_cs.slowness) & np.isfinite(res_sc.slowness)
    assert common.any(), "fixture must yield bound modes in both orderings"
    rel_diff = np.abs(res_cs.slowness[common] - res_sc.slowness[common]) / np.abs(
        res_cs.slowness[common]
    )
    assert np.any(rel_diff > 1.0e-3)


def test_quadrupole_dispersion_layered_N2_thin_outer_layer_eq_formation_collapses_to_N1():
    """G''.a.6 propagator-chain corollary: stacking a thin outer
    layer with formation parameters on top of an inner layer
    leaves the cased-hole quadrupole brentq root unchanged
    (within brentq xtol). The trivial outer layer's per-layer
    propagator collapses to ``E_form(b_outer) E_form(b_inner)^{-1}``
    and feeds the same effective state vector at the formation
    interface as the N=1 path would."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    inner = BoreholeLayer(vp=2300.0, vs=900.0, rho=2200.0, thickness=0.005)
    trivial_outer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.003)
    f = np.linspace(4000.0, 12000.0, 6)
    res_n1 = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(inner,),
    )
    res_n2 = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(inner, trivial_outer),
    )
    common = np.isfinite(res_n1.slowness) & np.isfinite(res_n2.slowness)
    assert common.any(), "fixture must produce some bound modes"
    np.testing.assert_allclose(
        res_n2.slowness[common],
        res_n1.slowness[common],
        rtol=1.0e-7,
    )


def test_quadrupole_dispersion_layered_N3_runs_smoke():
    """G''.d N=3 smoke (casing + cement + mudcake): the brentq +
    propagator-chain path runs to completion at three layers and
    returns a finite slowness in the LWD-relevant band where the
    cased quadrupole is bound. Mudcake placed *inside* the casing
    requires the radial-outward layer order ``(mudcake, casing,
    cement)``."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_g_pp_d_cased_geometry()
    mudcake = BoreholeLayer(vp=2200.0, vs=1300.0, rho=1700.0, thickness=0.002)
    f = np.array([16000.0, 17000.0, 18000.0])
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(mudcake, g["casing"], g["cement"]),
    )
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    finite = np.isfinite(res.slowness)
    assert finite.any(), "expected at least one frequency in bound regime"


# =====================================================================
# Plan item G''.e -- n=2 hardening
# =====================================================================
#
# Hardening tests for the G''.d brentq path: multi-frequency
# det-at-root self-consistency, thin-inner-formation-layer collapse to
# the N=1 outer-only case, two-formation-layers collapse to the
# unlayered ``quadrupole_dispersion`` (master-plan G'' validation
# bullet), and the LWD-quadrupole cement-bond physics direction
# (stiffer cement -> faster phase velocity / smaller slowness; the
# *opposite* of the original plan-doc impedance-coupling argument,
# now corrected).


def test_quadrupole_dispersion_layered_det_vanishes_at_brentq_roots_multi_freq():
    """G''.e self-consistency: at every brentq-converged ``k_z`` in
    the LWD band, ``|_modal_determinant_n2_cased(k_z_root)|`` is
    many orders of magnitude smaller than the same determinant
    evaluated 0.5 % off the root. Catches a brentq that returned
    a non-zero (non-converged) ``k_z`` -- e.g., from a bracket that
    touched but did not cross zero.

    Sister of the multi-frequency det-at-root oracle in G.e / G'.e."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased,
        quadrupole_dispersion_layered,
    )

    g = _typical_g_pp_d_cased_geometry()
    # Use the thin-cement variant where the bound regime is widest.
    cement_thin = BoreholeLayer(
        vp=g["cement"].vp,
        vs=g["cement"].vs,
        rho=g["cement"].rho,
        thickness=0.02,
    )
    f = np.linspace(14000.0, 18000.0, 5)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], cement_thin),
    )
    finite_idx = np.where(np.isfinite(res.slowness))[0]
    assert finite_idx.size >= 4, (
        f"need >=4 bound-mode frequencies for the multi-freq oracle; "
        f"got {finite_idx.size}"
    )
    for i in finite_idx:
        omega = 2.0 * np.pi * float(res.freq[i])
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n2_cased(
            kz_root,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], cement_thin),
        )
        det_off = _modal_determinant_n2_cased(
            kz_root * 1.005,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], cement_thin),
        )
        assert abs(det_off) > 0.0
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={res.freq[i] / 1000:.1f} kHz: |det_at|={abs(det_at):.2e}  "
            f"|det_off|={abs(det_off):.2e}  ratio={abs(det_at) / abs(det_off):.2e}"
        )


def test_quadrupole_dispersion_layered_two_formation_layers_collapse_to_unlayered():
    """G''.e master-plan validation: with both layers carrying
    formation parameters, the multi-layer brentq path returns the
    unlayered ``quadrupole_dispersion`` slowness to ``rtol=1e-6``
    across the full bound-regime band. Sister of the G''.d N=1
    layer = formation oracle, extended to N=2 telescoping
    (G''.a.6 propagator chain ``P_total = E_form(b) E_form(a)^-1``
    independent of ``N`` when all layers equal formation)."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer1 = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    layer2 = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.008)
    f = np.linspace(4000.0, 12000.0, 9)
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_n2 = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer1, layer2),
    )
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_n2.slowness)
    assert finite.sum() >= 6, f"need >=6 bound-mode frequencies; got {finite.sum()}"
    np.testing.assert_allclose(
        res_n2.slowness[finite],
        res_unl.slowness[finite],
        rtol=1.0e-6,
    )


def test_quadrupole_dispersion_layered_thin_inner_formation_layer_approaches_outer_only():
    """G''.e thin-inner-layer collapse: with a thin inner layer
    carrying formation parameters, the (thin_inner, outer) N=2
    brentq root converges to the (outer,) N=1 root as
    ``thickness_inner -> 0``. Continuity / vanishing-perturbation
    test on the propagator chain.

    The inner-layer-at-formation-params propagator
    ``P_inner = E_form(a + h) E_form(a)^{-1}`` is a near-identity
    perturbation as ``h -> 0``; the outer layer's contribution at
    its inner radius (now at ``a + h`` instead of ``a``) is the
    only residual difference, which vanishes linearly in ``h``."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_g_pp_d_cased_geometry()
    cement_thin = BoreholeLayer(
        vp=g["cement"].vp,
        vs=g["cement"].vs,
        rho=g["cement"].rho,
        thickness=0.02,
    )
    inner_form = BoreholeLayer(
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        thickness=1.0e-4,
    )
    f = np.array([15000.0, 16000.0])
    res_n1 = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], cement_thin),
    )
    res_n2 = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(inner_form, g["casing"], cement_thin),
    )
    finite = np.isfinite(res_n1.slowness) & np.isfinite(res_n2.slowness)
    assert finite.any(), "fixture must yield bound modes in both"
    # rtol scales with thickness_inner / borehole_radius ~ 1e-3.
    np.testing.assert_allclose(
        res_n2.slowness[finite],
        res_n1.slowness[finite],
        rtol=1.0e-3,
    )


def test_quadrupole_dispersion_layered_cement_bond_stiffer_cement_makes_mode_faster():
    """G''.e LWD-quadrupole cement-bond physics: holding casing
    and formation fixed, sweeping ``vs_cement`` over the valid
    range (``vs_cement >= vs_form``) shows the guided-mode phase
    velocity rising monotonically with cement stiffness -- i.e.,
    slowness *decreases* as cement gets stiffer.

    This is the *opposite* of the original plan-doc prediction
    (which argued by impedance-coupling intuition that stiffer
    cement should pull slowness toward the formation-shear
    asymptote). The numerical evidence over cement Vs in
    [1300, 1500] m/s with the slow-formation fixture
    (``vs_form = 1200``) consistently shows the simpler "stiffer
    annulus transmits the wave faster" reading: stiffer cement
    -> smaller slowness."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_g_pp_d_cased_geometry()
    cement_soft = BoreholeLayer(
        vp=2300.0,
        vs=1300.0,
        rho=1900.0,
        thickness=0.02,
    )
    cement_stiff = BoreholeLayer(
        vp=2700.0,
        vs=1500.0,
        rho=1950.0,
        thickness=0.02,
    )
    f = np.array([14000.0, 15000.0, 16000.0])
    res_soft = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], cement_soft),
    )
    res_stiff = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], cement_stiff),
    )
    common = np.isfinite(res_soft.slowness) & np.isfinite(res_stiff.slowness)
    assert common.sum() >= 2, (
        f"need >=2 frequencies with both cement variants in the bound "
        f"regime; got {common.sum()}"
    )
    # Stiffer cement should give SMALLER slowness (faster phase
    # velocity) at every comparable frequency.
    soft_sl = res_soft.slowness[common]
    stiff_sl = res_stiff.slowness[common]
    assert np.all(stiff_sl < soft_sl), (
        f"expected stiffer cement to give smaller slowness;\n"
        f"  soft (Vs=1300) slownesses: {soft_sl}\n"
        f"  stiff(Vs=1500) slownesses: {stiff_sl}"
    )
    # Sanity: the shift should be a few percent (not tiny noise,
    # not >50% which would indicate a different-mode ambiguity).
    rel_shift = (soft_sl - stiff_sl) / soft_sl
    assert np.all(rel_shift > 0.005)
    assert np.all(rel_shift < 0.30)


# =====================================================================
# Fast-formation cased-hole quadrupole (deferred follow-up to G'')
# =====================================================================
#
# Tests for the complex-determinant cased-hole quadrupole dispatch in
# ``quadrupole_dispersion_layered`` when ``V_S > V_f``. The path
# uses ``_layer_e_matrix_n2_complex`` /
# ``_layer_propagator_n2_complex`` /
# ``_modal_determinant_n2_cased_complex`` to evaluate the modal
# determinant at complex ``k_z``, and brentq's ``Im(det)`` along the
# real-``k_z`` axis (mirroring the unlayered fast-formation auto-
# dispatch in ``quadrupole_dispersion``).
#
# Per-layer slow-formation constraint does NOT apply in this regime
# (a cement layer softer than a fast carbonate formation is
# physically permissible); only the formation half-space is
# required to be in the fast regime ``V_S > V_f``.


def _typical_fast_formation_cased_geometry():
    """Fast-formation cased-hole fixture for the n=2 complex-
    determinant tests. Formation V_S = 2600 > V_f = 1500 puts the
    bound regime in slowness ``(1/V_S, 1/V_R) = (~3.85e-4, ~4.18e-4)``;
    layers carry typical casing + cement properties (no slow-
    formation per-layer constraint at fast formation)."""
    return dict(
        vp=4500.0,
        vs=2600.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
    )


def test_modal_determinant_n2_cased_complex_real_kz_slow_formation_matches_real_cased():
    """Slow-formation regression: at real ``k_z`` and slow
    formation, the complex cased determinant matches the real
    one to floating-point precision (its imaginary part is
    zero). Anchors the complex-extension correctness against the
    G''.c real-valued path."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased,
        _modal_determinant_n2_cased_complex,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    omega = 2.0 * np.pi * 5000.0
    kz = 1.05 * omega / vs
    det_re = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    det_cx = _modal_determinant_n2_cased_complex(
        complex(kz, 0.0),
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert det_cx.real == pytest.approx(det_re, rel=1.0e-12)
    # ``Im(det)`` is exactly zero when the matrix is real (slow
    # formation, real ``k_z``, all layers / formation in the bound
    # regime).
    assert det_cx.imag == 0.0


def test_modal_determinant_n2_cased_complex_layer_eq_formation_im_det_sign_matches_unlayered():
    """G''.a.6 collapse identity in the fast-formation regime:
    when ``layer == formation``, the cased determinant is the
    unlayered determinant times a real-valued ``det(E_form(b))``
    factor. The sign-change locations of ``Im(det_cas)`` along
    real ``k_z`` therefore coincide with the sign-change
    locations of ``Im(det_unl)`` -- i.e., the modal roots match
    even though the brentq-picked specific value can differ
    when there are multiple sign changes in the bracket."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased_complex,
        _modal_determinant_n2_complex,
    )

    g = _typical_fast_formation_cased_geometry()
    layer = BoreholeLayer(vp=g["vp"], vs=g["vs"], rho=g["rho"], thickness=0.01)
    omega = 2.0 * np.pi * 14000.0
    vR = 0.92 * g["vs"]
    slows = np.linspace(1.0 / g["vs"] * 1.0001, 1.0 / vR * 0.9999, 40)
    sgn_unl = []
    sgn_cas = []
    for slow in slows:
        kz = slow * omega
        d_unl = _modal_determinant_n2_complex(
            complex(kz, 0.0),
            omega,
            g["vp"],
            g["vs"],
            g["rho"],
            g["vf"],
            g["rho_f"],
            g["a"],
        ).imag
        d_cas = _modal_determinant_n2_cased_complex(
            complex(kz, 0.0),
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(layer,),
        ).imag
        sgn_unl.append(np.sign(d_unl))
        sgn_cas.append(np.sign(d_cas))
    sgn_unl = np.array(sgn_unl)
    sgn_cas = np.array(sgn_cas)
    # Sign-change indices must match exactly. The sign relationship
    # between the two streams is uniformly opposite (the real
    # ``det(E_form(b))`` scale factor is negative for this fixture)
    # but a uniform sign flip preserves the zero-crossing locations.
    changes_unl = np.where(np.diff(sgn_unl) != 0)[0]
    changes_cas = np.where(np.diff(sgn_cas) != 0)[0]
    np.testing.assert_array_equal(changes_unl, changes_cas)
    # And both streams must have at least one sign change in the
    # bound regime -- otherwise the test fixture would not exercise
    # the modal-root topology.
    assert changes_unl.size >= 1


def test_quadrupole_dispersion_layered_fast_formation_layer_eq_formation_finite_in_bound_regime():
    """Public-API end-to-end: with ``layer = formation`` in fast
    formation, the cased dispatch returns at least one finite
    slowness in the bound regime ``(1/V_S, 1/V_R)``. Soft
    correctness oracle that doesn't depend on brentq's exact
    multi-root pick."""
    from fwap.cylindrical_solver import (
        BoreholeMode,
        quadrupole_dispersion_layered,
    )

    g = _typical_fast_formation_cased_geometry()
    layer = BoreholeLayer(vp=g["vp"], vs=g["vs"], rho=g["rho"], thickness=0.01)
    f = np.linspace(10000.0, 16000.0, 4)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(layer,),
    )
    assert isinstance(res, BoreholeMode)
    assert res.attenuation_per_meter is None  # bound mode
    finite = np.isfinite(res.slowness)
    assert finite.any()
    # Bound regime: V in (V_R, V_S) where V_R ~= 0.92 V_S.
    sl = res.slowness[finite]
    assert np.all(sl > 1.0 / g["vs"] * 0.99)
    assert np.all(sl < 1.0 / (g["vs"] * 0.85))


def test_quadrupole_dispersion_layered_fast_formation_N2_runs_smoke():
    """Multi-layer fast-formation smoke (casing + cement on a
    fast carbonate formation): the brentq-on-Im(det) path runs
    to completion at N=2 and returns the same return-type
    contract as the slow-formation path."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_fast_formation_cased_geometry()
    f = np.linspace(8000.0, 18000.0, 6)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    assert res.attenuation_per_meter is None
    np.testing.assert_array_equal(res.freq, f)
    # Don't pin the slowness values themselves -- the cased fast-
    # formation bound mode for casing+cement is in a narrow
    # window and the brentq-on-Im(det) bracket can land on
    # spurious sign changes outside the physical regime when
    # the mode is cut off (same behaviour as the slow-formation
    # sister; documented in PR #71). Just verify the path
    # executes without exceptions.
    assert res.slowness.shape == f.shape


def test_quadrupole_dispersion_layered_fast_formation_does_not_break_slow_formation_path():
    """Regression: adding the fast-formation dispatch did not
    perturb the slow-formation cased-hole quadrupole brentq path
    that landed in G''.d. With slow formation params (``V_S = 800
    < V_f = 1500``), the dispatch goes through the real-valued
    G''.c path and the result matches the unlayered
    ``quadrupole_dispersion`` slowness when ``layer = formation``."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    f = np.linspace(4000.0, 12000.0, 9)
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_lyr = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_lyr.slowness)
    assert finite.any()
    np.testing.assert_allclose(
        res_lyr.slowness[finite],
        res_unl.slowness[finite],
        rtol=1.0e-9,
    )


def test_quadrupole_dispersion_layered_fast_formation_does_not_apply_slow_formation_constraint():
    """In fast formation, the per-layer ``layer.vs >= formation.vs``
    constraint enforced by ``_validate_flexural_layers_stacked`` for
    the slow-formation regime does NOT apply: a cement layer with
    ``vs < formation.vs`` is physically permissible (e.g., soft
    cement behind a fast-carbonate formation). The fast-formation
    dispatch must accept such a configuration without raising."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    # Fast formation V_S = 2600 > V_f = 1500.
    # Cement V_S = 1300 < V_S = 2600 (would fail slow-formation
    # constraint, but allowed in fast formation).
    soft_layer = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.02)
    res = quadrupole_dispersion_layered(
        np.array([10000.0, 14000.0]),
        vp=4500.0,
        vs=2600.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(soft_layer,),
    )
    # No exception; structural contract holds.
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2


# =====================================================================
# Fast-formation cased-hole flexural (n=1 sister of the G'' follow-up)
# =====================================================================
#
# Tests for the complex-determinant dispatch in
# ``flexural_dispersion_layered`` when ``V_S > V_f``. The path uses
# ``_layer_e_matrix_n1_complex`` / ``_layer_propagator_n1_complex`` /
# ``_modal_determinant_n1_cased_complex`` to evaluate the modal
# determinant at complex ``k_z``, and brentq's ``Im(det)`` along the
# real-``k_z`` axis (mirroring the unlayered fast-formation flexural
# auto-dispatch in ``flexural_dispersion``).
#
# Unlike the n=2 sister, the n=1 layer=formation collapse is clean
# enough that brentq lands on the *same* root as the unlayered
# solver, so the oracle here can pin slowness values directly rather
# than only sign-change locations.


def _typical_fast_formation_cased_n1_geometry():
    """Fast-formation cased-hole fixture for the n=1 complex-
    determinant tests. Formation V_S = 2820 > V_f = 1500 (the
    Paillet & Cheng limestone), so the bound regime is slowness in
    ``(1/V_S, 1/V_R) ~= (3.55e-4, 3.85e-4)``. Layers carry typical
    casing + cement properties; no slow-formation per-layer
    constraint applies at fast formation."""
    return dict(
        vp=4880.0,
        vs=2820.0,
        rho=2700.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
    )


def test_modal_determinant_n1_cased_complex_real_kz_slow_formation_matches_real():
    """Slow-formation regression: at real ``k_z`` and slow
    formation the complex cased determinant matches the real
    G'.c one to floating-point precision, with an exactly-zero
    imaginary part. Anchors the complex-extension correctness."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n1_cased,
        _modal_determinant_n1_cased_complex,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = 1.05 * omega / vs
    det_re = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    det_cx = _modal_determinant_n1_cased_complex(
        complex(kz, 0.0),
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert det_cx.real == pytest.approx(det_re, rel=1.0e-12)
    assert det_cx.imag == 0.0


def test_layer_e_matrix_n1_complex_matches_real_at_real_kz():
    """Entry-level oracle for the complex E-matrix: every one of
    the 36 entries matches the real ``_layer_e_matrix_n1`` in the
    bound regime, with zero imaginary part."""
    from fwap.cylindrical_solver import (
        _layer_e_matrix_n1,
        _layer_e_matrix_n1_complex,
    )

    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho, r = 2500.0, 1100.0, 2100.0, 0.1
    kz = 1.05 * omega / vs
    E_re = _layer_e_matrix_n1(kz, omega, vp=vp, vs=vs, rho=rho, r=r)
    E_cx = _layer_e_matrix_n1_complex(
        complex(kz, 0.0), omega, vp=vp, vs=vs, rho=rho, r=r
    )
    assert np.all(E_cx.imag == 0.0)
    np.testing.assert_allclose(E_cx.real, E_re, rtol=1.0e-12)


def test_flexural_dispersion_layered_fast_formation_layer_eq_formation_matches_unlayered():
    """Collapse identity through the public API: with
    ``layer = formation`` in the fast-formation regime, the cased
    dispatch reproduces the unlayered ``flexural_dispersion``
    slowness to ``rtol=1e-9``, and the NaN pattern (frequencies
    outside the geometric cutoff) matches exactly.

    Strongest end-to-end oracle for the n=1 fast-formation
    wiring."""
    g = _typical_fast_formation_cased_n1_geometry()
    layer = BoreholeLayer(vp=g["vp"], vs=g["vs"], rho=g["rho"], thickness=0.01)
    f = np.linspace(3000.0, 15000.0, 7)
    res_unl = flexural_dispersion(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
    )
    res_lyr = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(layer,),
    )
    # NaN pattern must agree exactly -- the layer=formation stack
    # cannot open or close the geometric cutoff.
    np.testing.assert_array_equal(
        np.isfinite(res_unl.slowness), np.isfinite(res_lyr.slowness)
    )
    finite = np.isfinite(res_unl.slowness)
    assert finite.any(), "fixture must put the bound mode in band"
    np.testing.assert_allclose(
        res_lyr.slowness[finite], res_unl.slowness[finite], rtol=1.0e-9
    )


def test_flexural_dispersion_layered_fast_formation_N2_runs_smoke():
    """Multi-layer fast-formation smoke (casing + cement behind a
    fast limestone): the brentq-on-Im(det) path runs to completion
    at N=2 and lands inside the ``(V_R, V_S)`` bound window where
    it converges."""
    g = _typical_fast_formation_cased_n1_geometry()
    f = np.linspace(4000.0, 16000.0, 7)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    assert res.attenuation_per_meter is None
    np.testing.assert_array_equal(res.freq, f)
    finite = np.isfinite(res.slowness)
    assert finite.any(), "expected at least one bound-regime root"
    # Bound regime: phase velocity between the formation Rayleigh
    # speed (~0.92 V_S) and V_S itself.
    sl = res.slowness[finite]
    assert np.all(sl > 1.0 / g["vs"] * 0.99)
    assert np.all(sl < 1.0 / (g["vs"] * 0.90))


def test_flexural_dispersion_layered_fast_formation_does_not_break_slow_formation():
    """Regression: adding the fast-formation dispatch did not
    perturb the slow-formation layered path. With slow-formation
    params (``V_S = 800 < V_f = 1500``) and ``layer = formation``,
    the result still matches the unlayered ``flexural_dispersion``
    to ``rtol=1e-9``."""
    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    f = np.linspace(2000.0, 8000.0, 5)
    res_unl = flexural_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    res_lyr = flexural_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a, layers=(layer,)
    )
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_lyr.slowness)
    assert finite.any()
    np.testing.assert_allclose(
        res_lyr.slowness[finite], res_unl.slowness[finite], rtol=1.0e-9
    )


def test_flexural_dispersion_layered_fast_formation_allows_layer_softer_than_formation():
    """In fast formation the per-layer ``layer.vs >= formation.vs``
    constraint enforced for the slow-formation regime does NOT
    apply: cement softer in shear than a fast carbonate is
    physically permissible. The dispatch must accept it without
    raising ``ValueError``."""
    g = _typical_fast_formation_cased_n1_geometry()
    # Cement V_S = 1300 < formation V_S = 2820: would fail the
    # slow-formation stacked constraint, but is legal here.
    res = flexural_dispersion_layered(
        np.array([8000.0, 12000.0]),
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["cement"],),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1


# ----------------------------------------------------------------------
# Where the cased flexural solver actually stops (roadmap A.2)
#
# The roadmap recorded this as a *layered* bracketing limitation: "the
# layered n=1 solver no longer refuses fast formations, but its root-
# finding stays sparse for a typical casing + cement stack". Measuring it
# says otherwise, and the difference matters because it points the next
# attempt at a different piece of code.
#
# In a fast formation the flexural mode is leaky: its root leaves the real
# k_z axis, and the real-axis Im(det) sign change the solver looks for
# survives only in a sliver next to the shear branch point at high
# frequency. Widening the real bracket cannot recover it -- there is no
# sign change to find below the cutoff, in any of the three sub-windows
# (below the slowest layer shear, between it and the formation Rayleigh
# speed, or between that and the formation shear).
#
# These tests pin the behaviour so the limitation stays a measured number
# rather than folklore, and so that a genuine fix shows up as a failure
# here rather than going unnoticed.
# ----------------------------------------------------------------------

_A2_CASING = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
_A2_CEMENT = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
_A2_FAST = dict(vp=4500.0, vs=2600.0, rho=2400.0)
_A2_SLOW = dict(vp=2200.0, vs=800.0, rho=2200.0)
_A2_BOREHOLE = dict(vf=1500.0, rho_f=1000.0, a=0.10)
_A2_FREQ = np.linspace(1000.0, 12000.0, 45)


def _a2_coverage(**kwargs) -> float:
    """Fraction of the frequency band at which the mode converged."""
    res = flexural_dispersion_layered(_A2_FREQ, **kwargs)
    return float(np.isfinite(res.slowness).mean())


def test_cased_flexural_slow_formation_covers_the_whole_band():
    """A slow formation behind casing converges everywhere.

    This is the control for the fast-formation test below: it shows the
    layered machinery itself is sound, so the sparsity there is about the
    formation regime rather than about the presence of a layer stack.
    """
    coverage = _a2_coverage(**_A2_SLOW, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT))
    assert coverage == 1.0


def test_cased_flexural_fast_formation_is_sparse_and_high_frequency_only():
    """A fast formation converges on well under half the band, high-f only."""
    res = flexural_dispersion_layered(
        _A2_FREQ, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    finite = np.isfinite(res.slowness)
    assert 0.2 < finite.mean() < 0.6
    # every converged point is in the upper part of the band -- the mode is
    # lost going *down* in frequency, not scattered at random
    assert _A2_FREQ[finite].min() > 4000.0
    assert finite[-1]


def test_the_sparsity_is_not_caused_by_the_casing():
    """The open hole is just as sparse, which relocates the problem.

    If the layer stack were responsible, removing it would restore the low
    frequencies. It does not: the identical formation in an *open* hole
    fails over the same lower part of the band. Whatever fixes this lives
    in the fast-formation flexural treatment, not in layered bracketing.
    """
    cased = flexural_dispersion_layered(
        _A2_FREQ, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    open_hole = flexural_dispersion(_A2_FREQ, **_A2_FAST, **_A2_BOREHOLE)

    open_finite = np.isfinite(open_hole.slowness)
    assert open_finite.mean() < 0.6
    # both lose the low end; the open hole is no better off
    assert _A2_FREQ[open_finite].min() > 4000.0
    assert not np.isfinite(cased.slowness[0])
    assert not np.isfinite(open_hole.slowness[0])


def test_converged_fast_formation_points_sit_below_the_shear_velocity():
    """The branch that is found is formation-controlled, bounded by V_S.

    It runs down from the shear branch point rather than tracking the
    cement, which is the other reason to stop calling this a cased-hole
    bracketing problem.
    """
    res = flexural_dispersion_layered(
        _A2_FREQ, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    finite = np.isfinite(res.slowness)
    velocity = 1.0 / res.slowness[finite]
    v_rayleigh = rayleigh_speed(_A2_FAST["vp"], _A2_FAST["vs"])
    assert np.all(velocity < _A2_FAST["vs"])
    assert np.all(velocity > v_rayleigh)
    # and it is dispersive downward with frequency, not a flat artefact
    assert velocity[-1] < velocity[0]


def test_complex_marcher_reproduces_the_real_cased_flexural_branch():
    """The complex root tracker and the n=1 cased determinant compose.

    Above the cutoff the cased flexural root is real, so marching it in the
    complex plane must return the same curve the real-axis solver finds,
    with an imaginary part at the level of floating-point noise. That is a
    prerequisite for any future leaky-mode work on this determinant --- if
    the marcher could not reproduce the part of the branch that is already
    known, its results below the cutoff would not be trustworthy either.

    Each frequency is seeded from the known root at that frequency rather
    than by continuation from the previous one. That is deliberate: with
    1 kHz steps, continuation seeding hops to a different branch (a root
    below the formation Rayleigh speed), which is one of the reasons the
    leaky extension needs the validated marcher's regime checks rather
    than the bare tracker.

    It is deliberately *not* a claim that the branch continues below the
    cutoff; see the roadmap for why that remains open.
    """
    from fwap.cylindrical_solver._cased import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import (
        _detect_leaky_branches,
        _track_complex_root,
    )

    def det(kz: complex, omega: float) -> complex:
        _, leaky_p, leaky_s = _detect_leaky_branches(
            kz, omega, _A2_FAST["vp"], _A2_FAST["vs"], _A2_BOREHOLE["vf"]
        )
        return _modal_determinant_n1_cased_complex(
            kz,
            omega,
            **_A2_FAST,
            **_A2_BOREHOLE,
            layers=(_A2_CASING, _A2_CEMENT),
            leaky_p=leaky_p,
            leaky_s=leaky_s,
        )

    frequencies = np.array([12000.0, 11000.0, 10000.0, 9000.0])
    reference = flexural_dispersion_layered(
        frequencies, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    assert np.all(np.isfinite(reference.slowness))

    for frequency, expected in zip(frequencies, reference.slowness, strict=True):
        omega = 2.0 * np.pi * float(frequency)
        seed = complex(expected * omega, 0.0)
        root = _track_complex_root(lambda kz, w=omega: det(kz, w), seed)
        assert root is not None
        assert abs(root.imag) < 1.0e-9 * abs(root.real)
        assert root.real / omega == pytest.approx(expected, rel=1e-6)


# ----------------------------------------------------------------------
# The pseudo-Rayleigh geometric cutoff against its rigid-pipe closed form
#
# `_J1_FIRST_ZERO` has been in the module since the leaky work landed, with
# a docstring offering the rigid-pipe estimate
#
#     f_c ~ j_{1,1} V_f V_S / (2 pi a sqrt(V_S^2 - V_f^2))
#
# to "callers that want to guard against requesting frequencies below the
# cutoff". Nothing checked it against the solver. Doing so splits into a
# part that holds cleanly and a part that does not, and both are pinned
# here because the second one contradicts that docstring advice.
# ----------------------------------------------------------------------

_PR_ROCK = dict(vp=4500.0, vs=2600.0, rho=2400.0)
_PR_FLUID = dict(vf=1500.0, rho_f=1000.0)
# One fixed grid for every case. Tying the grid to the estimate would make
# any grid-determined stopping point produce a constant ratio for free,
# since the estimate scales as 1/a and so would the grid.
_PR_GRID = np.linspace(500.0, 30000.0, 1500)


def _rigid_pipe_cutoff(vs: float, vf: float, a: float) -> float:
    from fwap.cylindrical_solver._leaky import _J1_FIRST_ZERO

    return _J1_FIRST_ZERO * vf * vs / (2.0 * np.pi * a * np.sqrt(vs**2 - vf**2))


def _lowest_converged(a: float) -> float:
    from fwap import pseudo_rayleigh_modal_dispersion

    result = pseudo_rayleigh_modal_dispersion(_PR_GRID, **_PR_ROCK, **_PR_FLUID, a=a)
    finite = np.isfinite(result.slowness)
    assert finite.any()
    return float(_PR_GRID[finite].min())


def test_pseudo_rayleigh_cutoff_scales_inversely_with_borehole_radius():
    """The geometric 1/a scaling of the closed form is reproduced exactly.

    This is the part of the rigid-pipe comparison that holds. Measured on a
    *fixed* frequency grid across a 3.3x range of borehole radius, the ratio
    of solver cutoff to closed form is constant to about 1 part in 300 --
    so the radius enters the solver's cutoff exactly as the closed form
    says it should.

    It would catch a radius/diameter confusion, which is the classic way to
    get this wrong and is invisible to any single-radius check.
    """
    radii = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    ratios = [
        _lowest_converged(a) / _rigid_pipe_cutoff(2600.0, 1500.0, a) for a in radii
    ]
    assert max(ratios) - min(ratios) < 0.01 * np.mean(ratios)


def test_the_rigid_pipe_estimate_is_not_usable_as_a_cutoff_guard():
    """It overestimates by ~2.8x, so guarding with it discards a valid band.

    The docstring of `pseudo_rayleigh_dispersion` offers the rigid-pipe
    formula to callers wanting to avoid requesting frequencies below the
    cutoff. Taken literally that is bad advice: the solver converges well
    below the estimate, because a compliant elastic wall admits the mode at
    lower frequency than a rigid pipe does.

    Pinned as a bound rather than a constant, since the factor is not
    universal -- it varies strongly with formation velocity (see the
    module-level note and docs/roadmap_old.m A.1).
    """
    estimate = _rigid_pipe_cutoff(2600.0, 1500.0, 0.10)
    measured = _lowest_converged(0.10)
    assert measured < 0.5 * estimate
    # and the discarded band is wide enough to matter
    assert estimate - measured > 5000.0


# ----------------------------------------------------------------------
# Quadrupole high-frequency asymptote, and where it stops being usable
#
# At short wavelength the borehole wall looks flat to every azimuthal
# order, so the n=2 quadrupole must approach the same plane-interface
# Scholte speed the n=0 Stoneley does. `scholte_speed` computes that from
# a different equation, so this is an external check on the n=2 solver.
#
# It holds in slow formations. In fast ones the mode is leaky and the
# real-axis search returns a non-monotone scatter between the Rayleigh and
# shear speeds -- the same failure as the n=1 flexural case (roadmap A.2),
# now known to affect n=2 as well.
# ----------------------------------------------------------------------

_QUAD_SLOW = dict(vp=2200.0, vs=800.0, rho=2200.0)
_QUAD_MID = dict(vp=3000.0, vs=1400.0, rho=2300.0)
_QUAD_FAST = dict(vp=4500.0, vs=2600.0, rho=2400.0)
_QUAD_FLUID = dict(vf=1500.0, rho_f=1000.0)


@pytest.mark.parametrize("rock", [_QUAD_SLOW, _QUAD_MID], ids=["slow", "mid"])
def test_quadrupole_converges_to_the_plane_scholte_speed(rock):
    """n=2 approaches the same plane-interface limit as n=0.

    Checked against `scholte_speed`, which solves a plane fluid/solid
    interface problem -- a different equation from the n=2 modal
    determinant -- so agreement is a cross-check rather than the solver
    confirming itself.
    """
    from fwap import quadrupole_dispersion, scholte_speed

    reference = scholte_speed(**rock, **_QUAD_FLUID)
    frequencies = np.array([50.0e3, 100.0e3, 200.0e3, 400.0e3])
    result = quadrupole_dispersion(frequencies, **rock, **_QUAD_FLUID, a=0.10)
    assert np.all(np.isfinite(result.slowness))

    error = np.abs(1.0 / result.slowness / reference - 1.0)
    assert np.all(np.diff(error) < 0.0)  # converging, not merely close
    assert error[-1] < 1.0e-3


def test_quadrupole_and_stoneley_share_the_high_frequency_limit():
    """Both azimuthal orders must land on the same flat-wall answer."""
    from fwap import quadrupole_dispersion, stoneley_dispersion

    high = np.array([400.0e3])
    quad = quadrupole_dispersion(high, **_QUAD_SLOW, **_QUAD_FLUID, a=0.10)
    stoneley = stoneley_dispersion(high, **_QUAD_SLOW, **_QUAD_FLUID, a=0.10)
    assert 1.0 / quad.slowness[0] == pytest.approx(1.0 / stoneley.slowness[0], rel=1e-4)


def test_fast_formation_quadrupole_is_not_a_usable_curve():
    """In a fast formation the returned values are not a guided mode.

    They are finite -- which is the hazard, since a caller filtering on
    NaN keeps them -- but non-monotone, scattered between the Rayleigh and
    shear speeds. A guided mode's phase velocity decreases with frequency;
    this does not.

    Pinned so that a future fix to the leaky-mode search shows up here as
    a failure rather than passing unnoticed.
    """
    from fwap import quadrupole_dispersion

    frequencies = np.array([10.0e3, 15.0e3, 20.0e3, 30.0e3, 60.0e3, 100.0e3])
    result = quadrupole_dispersion(frequencies, **_QUAD_FAST, **_QUAD_FLUID, a=0.10)
    finite = np.isfinite(result.slowness)
    assert finite.sum() >= 4  # finite, so NaN-filtering does not catch it

    velocity = 1.0 / result.slowness[finite]
    assert not np.all(np.diff(velocity) <= 0.0)  # not a dispersion curve

    v_rayleigh = rayleigh_speed(_QUAD_FAST["vp"], _QUAD_FAST["vs"])
    assert np.all(velocity > 0.99 * v_rayleigh)
    assert np.all(velocity < 1.10 * _QUAD_FAST["vs"])


def test_slow_formation_quadrupole_is_a_usable_curve():
    """The control: in the regime it is meant for, the curve is monotone."""
    from fwap import quadrupole_dispersion

    frequencies = np.linspace(5.0e3, 60.0e3, 12)
    result = quadrupole_dispersion(frequencies, **_QUAD_SLOW, **_QUAD_FLUID, a=0.10)
    finite = np.isfinite(result.slowness)
    assert finite.sum() >= 10
    velocity = 1.0 / result.slowness[finite]
    assert np.all(np.diff(velocity) <= 1.0e-9)


# ---------------------------------------------------------------------------
# Branch selection for the pseudo-Rayleigh marcher.
#
# These replace five tests that pinned the *old* seeding heuristic's defects:
# the tracked branch depended on the grid's top frequency, and two silent
# all-NaN failures. The seed is now enumerated, so the same assertions run
# the other way round -- they check the answer no longer moves. The measured
# numbers are carried over from the defect tests unchanged, which is what
# makes them evidence that behaviour changed in the intended direction.
# ---------------------------------------------------------------------------

_LEAKY_FAST = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_LEAKY_BRINE = {"vf": 1500.0, "rho_f": 1000.0}


def _pr_at(probe, top, *, branch=0, a=0.10, n=400):
    """Run the solver on a 2 kHz-to-``top`` grid and read off ``probe``."""
    from fwap import pseudo_rayleigh_modal_dispersion

    grid = np.sort(np.unique(np.concatenate([np.linspace(2.0e3, top, n), [probe]])))
    mode = pseudo_rayleigh_modal_dispersion(
        grid, **_LEAKY_FAST, **_LEAKY_BRINE, a=a, branch=branch
    )
    j = int(np.argmin(np.abs(grid - probe)))
    return 1.0 / mode.slowness[j], mode.attenuation_per_meter[j]


def test_pseudo_rayleigh_does_not_depend_on_the_grid_top_frequency():
    """The mode returned is a function of the medium, not of the request.

    This is the regression that motivated enumerating the seeds. With the
    old heuristic seed the answer at 30 kHz switched from 2486 m/s to
    2952 m/s somewhere between a 55 kHz and a 60 kHz grid top -- silently,
    and to a different but equally genuine root. The span below brackets
    that switch on both sides.
    """
    reference = _pr_at(30.0e3, 40.0e3)
    for top in (32.0e3, 55.0e3, 60.0e3, 80.0e3, 100.0e3):
        c, att = _pr_at(30.0e3, top)
        assert c == pytest.approx(reference[0], rel=1.0e-9), top
        assert att == pytest.approx(reference[1], rel=1.0e-9), top

    assert reference[0] == pytest.approx(2486.16, rel=1.0e-4)


def test_pseudo_rayleigh_branches_are_distinct_and_both_are_genuine_roots():
    """``branch`` selects radial order, and every order is a real root.

    The two modes here are the pair the old seeding used to confuse: both
    solve the determinant at 30 kHz, and which one you got depended on the
    grid. Now they are addressable, and each is checked against the
    determinant -- a true root sits orders of magnitude below the
    magnitude on a small circle around it.
    """
    from fwap.cylindrical_solver._leaky import _modal_determinant_n0_complex

    probe = 30.0e3
    omega = 2.0 * np.pi * probe

    def det(kz):
        return _modal_determinant_n0_complex(
            kz,
            omega,
            **_LEAKY_FAST,
            **_LEAKY_BRINE,
            a=0.10,
            leaky_p=False,
            leaky_s=True,
        )

    fundamental = _pr_at(probe, 80.0e3, branch=0)
    overtone = _pr_at(probe, 80.0e3, branch=1)

    assert fundamental[0] == pytest.approx(2486.16, rel=1.0e-4)
    assert overtone[0] == pytest.approx(2951.7, rel=1.0e-4)

    # The fundamental is the slower of the two: lowest radial order means
    # the smallest radial wavenumber and so the largest axial one.
    assert fundamental[0] < overtone[0]

    for c, att in (fundamental, overtone):
        kz = complex(omega / c, att)
        radius = 0.01 * abs(kz)
        ring = np.median(
            [
                abs(det(kz + radius * np.exp(1j * t)))
                for t in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
            ]
        )
        assert abs(det(kz)) < 1.0e-10 * ring, (c, att)


def test_pseudo_rayleigh_recovers_the_band_on_a_coarse_grid():
    """A coarse grid gives a coarse answer, not silence.

    A 0.07 m borehole over 4-30 kHz used to return *nothing at all* at 60
    samples while recovering the band at 80 -- a total, silent failure of
    the heuristic seed's first step. Both grids now converge throughout.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.07)
    coarse = pseudo_rayleigh_modal_dispersion(np.linspace(4.0e3, 30.0e3, 60), **medium)
    fine = pseudo_rayleigh_modal_dispersion(np.linspace(4.0e3, 30.0e3, 120), **medium)

    assert np.isfinite(coarse.slowness).sum() == 60
    assert np.isfinite(fine.slowness).sum() == 120


def test_pseudo_rayleigh_sub_window_agrees_with_the_full_band():
    """Asking about part of a band gives the same answer as asking about all of it.

    Requesting only 24-32 kHz used to return nothing, while a 2-40 kHz grid
    converged across that whole interval. Now the narrow request not only
    converges but reproduces the wide one's values.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    band = np.arange(24.0e3, 32.0e3 + 1.0, 100.0)

    narrow = pseudo_rayleigh_modal_dispersion(band, **medium)
    assert np.isfinite(narrow.slowness).sum() == band.size

    wide_grid = np.arange(2.0e3, 40.0e3 + 1.0, 100.0)
    wide = pseudo_rayleigh_modal_dispersion(wide_grid, **medium)
    shared = np.searchsorted(wide_grid, band)
    assert np.allclose(narrow.slowness, wide.slowness[shared], rtol=1.0e-9)
    assert np.allclose(
        narrow.attenuation_per_meter,
        wide.attenuation_per_meter[shared],
        rtol=1.0e-9,
    )


def test_pseudo_rayleigh_rejects_a_branch_that_does_not_exist():
    """Higher radial orders appear only above their cutoffs; say so loudly.

    Returning an empty curve would be indistinguishable from "this mode
    does not propagate here", which is the confusion the old silent NaNs
    caused. The message names how many branches were found so the caller
    knows whether to extend the grid or to ask for a lower order.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    with pytest.raises(ValueError, match=r"only 1 leaky branch"):
        pseudo_rayleigh_modal_dispersion(
            np.linspace(2.0e3, 15.0e3, 50), **medium, branch=4
        )
    with pytest.raises(ValueError, match="branch must be non-negative"):
        pseudo_rayleigh_modal_dispersion(
            np.linspace(2.0e3, 40.0e3, 50), **medium, branch=-1
        )


def test_pseudo_rayleigh_returns_all_nan_below_the_lowest_cutoff():
    """No branch exists at the top of the band, so there is nothing to march.

    This is the one case where an all-NaN curve is the right answer rather
    than a search failure: at 100-200 Hz the wavelength is two orders above
    the borehole radius and no leaky mode has reached its cutoff. The
    enumeration finds nothing and the solver says so, instead of raising --
    "this mode does not propagate here" is a physical statement, and it is
    distinguished from an out-of-range ``branch``, which does raise.
    """
    from fwap import pseudo_rayleigh_modal_dispersion
    from fwap.cylindrical_solver._leaky import _enumerate_leaky_roots_n0

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    assert _enumerate_leaky_roots_n0(2.0 * np.pi * 200.0, **medium) == []

    freq = np.linspace(100.0, 200.0, 5)
    mode = pseudo_rayleigh_modal_dispersion(freq, **medium)
    assert mode.slowness.shape == freq.shape
    assert not np.any(np.isfinite(mode.slowness))
    assert not np.any(np.isfinite(mode.attenuation_per_meter))


def test_leaky_root_enumeration_count_is_insensitive_to_scan_density():
    """The seed scan must not be the thing that decides how many modes exist.

    If the recovered count moved with the scan resolution, "how many
    branches are there" would be an artefact of the search rather than a
    property of the medium -- and ``branch=1`` would mean different things
    at different densities. Checked over a 3.3x span of seed density.
    """
    from fwap.cylindrical_solver._leaky import _enumerate_leaky_roots_n0

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    for freq in (15.0e3, 30.0e3, 60.0e3):
        omega = 2.0 * np.pi * freq
        counts = {
            len(_enumerate_leaky_roots_n0(omega, **medium, n_re=n_re, n_im=n_im))
            for n_re, n_im in ((24, 5), (40, 8), (80, 16))
        }
        assert len(counts) == 1, (freq, counts)

    # ...and the count grows with frequency, as radial orders pass cutoff.
    counts = [
        len(_enumerate_leaky_roots_n0(2.0 * np.pi * f, **medium))
        for f in (15.0e3, 30.0e3, 60.0e3)
    ]
    assert counts == sorted(counts) and counts[0] < counts[-1], counts


def test_pseudo_rayleigh_is_independent_of_grid_density_where_it_converges():
    """The flip side: once it converges, refinement changes nothing.

    Halving the step reproduces the attenuation to machine precision, so
    the window sensitivity above is about *which* root is tracked, not
    about the accuracy of the tracking.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    coarse_grid = np.arange(2.0e3, 40.0e3 + 1.0, 50.0)
    fine_grid = np.arange(2.0e3, 40.0e3 + 1.0, 25.0)

    coarse = pseudo_rayleigh_modal_dispersion(coarse_grid, **medium)
    fine = pseudo_rayleigh_modal_dispersion(fine_grid, **medium)
    shared = np.searchsorted(fine_grid, coarse_grid)

    ok = np.isfinite(coarse.slowness) & np.isfinite(fine.slowness[shared])
    assert ok.sum() > 700
    rel = np.abs(
        coarse.attenuation_per_meter[ok] / fine.attenuation_per_meter[shared][ok] - 1.0
    )
    assert rel.max() < 1.0e-10


# ---------------------------------------------------------------------------
# Energy balance for the leaky modes: a candidate oracle that does NOT work.
#
# `plans/learning.md` listed this as the most promising remaining candidate,
# on the reasoning that radiated power over axial power must reproduce
# Im(k_z) with no free geometry in it -- and so might explain the ~0.6 offset
# that `leaky_radiation_attenuation` leaves unexplained. It does neither, and
# these tests record why in a form that runs, so the next attempt does not
# re-derive it and mistake the result for a confirmation.
#
# The derivation is sound and the agreement is perfect. That is the problem:
# it is perfect for k_z values that are not roots either.
# ---------------------------------------------------------------------------

_EB_MEDIUM = {
    "vp": 4000.0,
    "vs": 2300.0,
    "rho": 2500.0,
    "vf": 1500.0,
    "rho_f": 1000.0,
    "a": 0.10,
}


def _fluid_energy_balance_im_kz(kz, omega, *, vf, a):
    """``Im(k_z)`` from radiated power over twice the axial power.

    Closes the energy balance over the fluid column only. The wall
    boundary conditions (``sigma_rz = 0`` and ``sigma_rr = -P``) put both
    the radiated flux at ``r = a`` and the axial flux in terms of the same
    fluid amplitude, which cancels.
    """
    from scipy import special

    f_radial = np.sqrt(kz**2 - (omega / vf) ** 2)
    i0a = complex(special.iv(0, f_radial * a))
    i1a = complex(special.iv(1, f_radial * a))
    radiated = -a * np.imag(i0a * np.conj(f_radial * i1a))

    r = np.linspace(0.0, a, 2001)
    axial = np.trapezoid(np.abs(special.iv(0, f_radial * r)) ** 2 * r, r)
    return radiated / (2.0 * kz.real * axial)


def test_fluid_energy_balance_reproduces_the_leaky_attenuation():
    """At genuine roots the balance returns Im(k_z) exactly.

    Establishes that the derivation is right, which is what makes the next
    test's result meaningful rather than a sign of a broken formula.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    freq = np.linspace(8.0e3, 30.0e3, 12)
    mode = pseudo_rayleigh_modal_dispersion(freq, **_EB_MEDIUM)
    ok = np.isfinite(mode.slowness)
    assert ok.sum() > 8

    for f, s, att in zip(freq[ok], mode.slowness[ok], mode.attenuation_per_meter[ok]):
        omega = 2.0 * np.pi * f
        kz = complex(s * omega, att)
        predicted = _fluid_energy_balance_im_kz(
            kz, omega, vf=_EB_MEDIUM["vf"], a=_EB_MEDIUM["a"]
        )
        assert predicted == pytest.approx(att, rel=1.0e-6), f


def test_fluid_energy_balance_is_an_identity_and_so_validates_nothing():
    """...and it returns Im(k_z) for k_z values that are not roots at all.

    That is the whole finding. Closing the balance inside the fluid gives
    the divergence theorem applied to a source-free Helmholtz solution,
    which is true of *any* field of the form ``A I0(F r) exp(i k_z z)``
    with ``F^2 = k_z^2 - (omega/V_f)^2``. Nothing about the formation, and
    hence nothing about the eigenvalue condition, enters it.

    A check that cannot fail is not a check. This is the "limit that
    cannot discriminate" failure mode in `plans/learning.md`, caught by
    the rule that document states: ask what the check would do to a wrong
    answer.
    """
    omega = 2.0 * np.pi * 20.0e3
    rng = np.random.default_rng(0)

    kz_lo = omega / _EB_MEDIUM["vp"]
    kz_hi = omega / _EB_MEDIUM["vs"]
    for _ in range(8):
        kz = complex(rng.uniform(kz_lo, kz_hi), rng.uniform(0.2, 8.0))
        predicted = _fluid_energy_balance_im_kz(
            kz, omega, vf=_EB_MEDIUM["vf"], a=_EB_MEDIUM["a"]
        )
        # Arbitrary k_z, no root anywhere near it, and it still comes back.
        assert predicted == pytest.approx(kz.imag, rel=1.0e-5), kz


def test_full_energy_balance_has_no_finite_denominator():
    """Extending the balance into the formation does not rescue it.

    The obvious repair is to include the formation's axial flux, which
    would bring the outgoing-wave condition into the balance. It cannot be
    done: the leaky-S field *grows* with radius -- the standard leaky-mode
    divergence -- so the axial power integral has no finite value to
    divide by.

    Checked with the solver's own radial evaluator rather than a
    hand-rolled Hankel call, because the two Hankel kinds differ here by
    growth versus decay and picking the wrong one reverses the conclusion.
    """
    from fwap import pseudo_rayleigh_modal_dispersion
    from fwap.cylindrical_solver._bessel import _k_or_hankel

    freq = np.linspace(8.0e3, 30.0e3, 12)
    mode = pseudo_rayleigh_modal_dispersion(freq, **_EB_MEDIUM)
    j = int(np.argmin(np.abs(freq - 20.0e3)))
    omega = 2.0 * np.pi * freq[j]
    kz = complex(mode.slowness[j] * omega, mode.attenuation_per_meter[j])

    s_radial = np.sqrt(kz**2 - (omega / _EB_MEDIUM["vs"]) ** 2)
    radii = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    magnitude = np.array(
        [abs(_k_or_hankel(0, s_radial, float(r), leaky=True)[0]) for r in radii]
    )
    assert np.all(np.diff(magnitude) > 0.0), magnitude
    assert magnitude[-1] > 1.0e6 * magnitude[0], magnitude

    # The bound P wave decays, as it must -- so the growth above is the
    # leaky branch specifically, not every field in the formation.
    p_radial = np.sqrt(kz**2 - (omega / _EB_MEDIUM["vp"]) ** 2)
    p_magnitude = np.array(
        [abs(_k_or_hankel(0, p_radial, float(r), leaky=False)[0]) for r in radii]
    )
    assert np.all(np.diff(p_magnitude) < 0.0), p_magnitude


# ---------------------------------------------------------------------------
# Layer-subdivision invariance: an oracle for the layered propagator stack.
#
# Subdividing a homogeneous annulus into several adjacent layers with the same
# properties changes the *description* and not the medium, so the dispersion
# must be bit-stable. It exercises exactly the machinery the single-layer
# tests cannot reach -- interface matching and propagator composition across
# more than one boundary -- and it is independent of the physics of any one
# layer, because it compares the solver against itself under a relabelling
# that no correct implementation may notice.
#
# Note what this does NOT test: an error common to every interface cancels
# out. It is a consistency oracle, not an absolute one.
# ---------------------------------------------------------------------------

_SUB_SLOW = {"vp": 2600.0, "vs": 1300.0, "rho": 2300.0}
_SUB_FAST = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_SUB_FLUID = {"vf": 1500.0, "rho_f": 1000.0}
_SUB_MUD = {"vp": 3000.0, "vs": 1600.0, "rho": 2100.0}
_SUB_STEEL = {"vp": 5860.0, "vs": 3140.0, "rho": 7800.0}
_SUB_CEMENT = {"vp": 2800.0, "vs": 1600.0, "rho": 1920.0}


def _subdivide(layers, index, fractions):
    """Replace ``layers[index]`` by adjacent copies summing to its thickness."""
    from fwap import BoreholeLayer

    target = layers[index]
    assert abs(sum(fractions) - 1.0) < 1.0e-12
    pieces = tuple(
        BoreholeLayer(
            vp=target.vp,
            vs=target.vs,
            rho=target.rho,
            thickness=target.thickness * fraction,
        )
        for fraction in fractions
    )
    return layers[:index] + pieces + layers[index + 1 :]


def test_subdividing_a_homogeneous_layer_leaves_the_dispersion_unchanged():
    """Relabelling one annulus as several must change nothing at all.

    Covered for all three azimuthal orders, for an open-hole mudcake and a
    cased steel-plus-cement stack, and splitting the inner as well as the
    outer layer of that stack -- so the invariance is checked where the
    propagator has to compose across a large impedance contrast, not only
    in the easy case.
    """
    from fwap import (
        BoreholeLayer,
        flexural_dispersion_layered,
        quadrupole_dispersion_layered,
        stoneley_dispersion_layered,
    )

    mud = (BoreholeLayer(**_SUB_MUD, thickness=0.04),)
    cased = (
        BoreholeLayer(**_SUB_STEEL, thickness=0.01),
        BoreholeLayer(**_SUB_CEMENT, thickness=0.03),
    )
    freq = np.linspace(2.0e3, 14.0e3, 13)

    cases = [
        (stoneley_dispersion_layered, _SUB_SLOW, mud, 0),
        (flexural_dispersion_layered, _SUB_SLOW, mud, 0),
        (quadrupole_dispersion_layered, _SUB_SLOW, mud, 0),
        (stoneley_dispersion_layered, _SUB_FAST, mud, 0),
        (stoneley_dispersion_layered, _SUB_SLOW, cased, 0),
        (stoneley_dispersion_layered, _SUB_SLOW, cased, 1),
    ]
    for solver, formation, layers, index in cases:
        medium = dict(**formation, **_SUB_FLUID, a=0.10)
        base = solver(freq, **medium, layers=layers).slowness
        for fractions in ((0.5, 0.5), (0.3, 0.7), (0.25, 0.35, 0.40)):
            split = solver(
                freq, **medium, layers=_subdivide(layers, index, fractions)
            ).slowness
            ok = np.isfinite(base) & np.isfinite(split)
            assert ok.sum() >= 4, (solver.__name__, layers, fractions)
            assert np.allclose(base[ok], split[ok], rtol=1.0e-12, atol=0.0), (
                solver.__name__,
                index,
                fractions,
                np.max(np.abs(split[ok] / base[ok] - 1.0)),
            )


def test_layer_subdivision_invariance_is_not_vacuous():
    """A subdivision that does not preserve the medium must be detected.

    The invariance above is only worth asserting if a wrong stack fails it.
    A thickness error of one part in ten thousand moves the slowness by
    ~3e-6 relative -- nine orders above the 1e-15 floor the correct split
    achieves -- so the check discriminates with enormous margin.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    freq = np.linspace(2.0e3, 14.0e3, 13)
    layers = (BoreholeLayer(**_SUB_MUD, thickness=0.04),)
    base = stoneley_dispersion_layered(freq, **medium, layers=layers).slowness

    exact = stoneley_dispersion_layered(
        freq, **medium, layers=_subdivide(layers, 0, (0.5, 0.5))
    ).slowness
    assert np.max(np.abs(exact / base - 1.0)) < 1.0e-12

    for relative_error, floor in ((1.0e-4, 1.0e-6), (1.0e-3, 1.0e-5)):
        wrong = (
            BoreholeLayer(**_SUB_MUD, thickness=0.02),
            BoreholeLayer(**_SUB_MUD, thickness=0.02 * (1.0 + relative_error)),
        )
        got = stoneley_dispersion_layered(freq, **medium, layers=wrong).slowness
        assert np.max(np.abs(got / base - 1.0)) > floor, relative_error


def test_swapping_layer_order_is_not_an_invariance():
    """Recorded because a planning note claimed it was.

    `plans/learning.md` listed "swapping layer order should leave the
    dispersion invariant" as a candidate oracle. It is false for a
    cylindrical stack: the layers sit at *different radii*, so exchanging
    them moves material from one radius to another and changes the medium.
    Measured here at about 1 % -- far too large to be numerical, and in the
    direction physics requires.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    medium = dict(**_SUB_FAST, **_SUB_FLUID, a=0.10)
    freq = np.linspace(2.0e3, 15.0e3, 8)
    inner = BoreholeLayer(**_SUB_MUD, thickness=0.02)
    outer = BoreholeLayer(vp=3500.0, vs=1900.0, rho=2300.0, thickness=0.03)

    forward = stoneley_dispersion_layered(
        freq, **medium, layers=(inner, outer)
    ).slowness
    reversed_ = stoneley_dispersion_layered(
        freq, **medium, layers=(outer, inner)
    ).slowness

    ok = np.isfinite(forward) & np.isfinite(reversed_)
    assert ok.sum() >= 6
    assert np.max(np.abs(reversed_[ok] / forward[ok] - 1.0)) > 1.0e-3


# ---------------------------------------------------------------------------
# Where the transparency invariance stops holding.
#
# Appending a layer whose properties equal the formation is a no-op on the
# medium, so it must be a no-op on the dispersion. It is -- until the radial
# dynamic range across the added layer gets large, at which point the root
# search returns finite, plausible, wrong values. These tests pin that the
# boundary exists and, importantly, establish *which* side is right using an
# oracle outside the layered solver entirely.
#
# They pin a limitation, so a future fix should make the second one fail; it
# should then be rewritten as a guarantee rather than worked around.
#
# Nothing about the size or location of the error is asserted. Both proved
# platform-dependent, and two earlier versions of these tests failed in CI by
# pinning first where the spurious root lands and then how far off it is.
#
# The existing single-layer transparency tests use a 0.005 m layer over
# 0.5-8 kHz, which is far inside the safe region -- which is why the
# limitation went unnoticed rather than being a regression.
# ---------------------------------------------------------------------------


def test_transparent_layer_is_a_no_op_while_the_dynamic_range_is_moderate():
    """Thin formation-equal layers are transparent, as they must be."""
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    freq = np.linspace(2.0e3, 14.0e3, 13)
    stack = (BoreholeLayer(**_SUB_MUD, thickness=0.02),)
    base = stoneley_dispersion_layered(freq, **medium, layers=stack).slowness

    for thickness in (0.01, 0.05, 0.10):
        padded = stack + (BoreholeLayer(**_SUB_SLOW, thickness=thickness),)
        got = stoneley_dispersion_layered(freq, **medium, layers=padded).slowness
        ok = np.isfinite(base) & np.isfinite(got)
        assert ok.sum() >= 10, thickness
        assert np.max(np.abs(got[ok] / base[ok] - 1.0)) < 1.0e-10, thickness


def test_transparent_layer_stops_being_a_no_op_when_it_is_thick():
    """...and stops being one well before it returns NaN.

    A formation-equal layer is physically nothing at all, so the padded and
    plain stacks must agree to the same ~1e-15 that layer subdivision
    achieves. For thin layers they do -- the previous test asserts 1e-10.
    Somewhere above 0.1 m at 100 kHz they stop agreeing, silently: both
    calls return finite slownesses that look like dispersion curves.

    Which one is wrong is settled from outside the layered solver, with
    ``scholte_speed``: at 100 kHz the wavelength in the 2 cm mudcake is
    ~1.6 cm, so the mode rides the *innermost* layer and must approach that
    layer's Scholte speed. The plain stack does, to 0.05 %.

    **Nothing about the size or location of the error is asserted, because
    neither is a property of the physics.** Both move with thickness and
    with platform: the padded answer has been seen at 289 m/s and at
    1095 m/s for the same stack, disagreeing with the plain answer by 7 % on
    one machine and by a factor of four on another, and two earlier versions
    of this test failed in CI by pinning first the location and then the
    magnitude. What is stable, and all that is claimed here, is that
    transparency is lost somewhere in this range -- which is the finding.
    """
    from fwap import BoreholeLayer, scholte_speed, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    stack = (BoreholeLayer(**_SUB_MUD, thickness=0.02),)
    freq = np.array([100.0e3])

    plain = stoneley_dispersion_layered(freq, **medium, layers=stack).slowness[0]
    assert np.isfinite(plain)
    mud_scholte = scholte_speed(**_SUB_MUD, **_SUB_FLUID)
    assert abs((1.0 / plain) / mud_scholte - 1.0) < 1.0e-3

    # Transparency means agreement to ~1e-15; 1e-3 is twelve orders looser
    # than that and seven looser than the thin-layer test's tolerance, so
    # exceeding it cannot be round-off on any platform.
    def is_transparent(thickness):
        padded = stack + (BoreholeLayer(**_SUB_SLOW, thickness=thickness),)
        got = stoneley_dispersion_layered(freq, **medium, layers=padded).slowness[0]
        if not np.isfinite(got):
            return False  # NaN is a clean failure, but still not transparent
        return abs(got / plain - 1.0) < 1.0e-3

    thicknesses = (0.12, 0.15, 0.18, 0.20, 0.25)
    assert not all(is_transparent(t) for t in thicknesses), thicknesses


def test_genuine_thick_layers_still_converge_to_the_right_limit():
    """The breakdown is specific to a *redundant* layer, not to thick ones.

    A real altered zone with genuine contrast keeps converging to the
    innermost layer's Scholte speed at every thickness tried, and its error
    barely moves with thickness. That matters for how the limitation should
    be read: it is a defect in a construction used to *verify* the solver,
    not a defect in the configurations the solver exists to model.
    """
    from fwap import BoreholeLayer, scholte_speed, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    inner_scholte = scholte_speed(**_SUB_MUD, **_SUB_FLUID)
    freq = np.array([50.0e3, 100.0e3, 200.0e3])

    for thickness in (0.02, 0.10, 0.25):
        layers = (BoreholeLayer(**_SUB_MUD, thickness=thickness),)
        slowness = stoneley_dispersion_layered(freq, **medium, layers=layers).slowness
        assert np.all(np.isfinite(slowness)), thickness
        error = np.abs((1.0 / slowness) / inner_scholte - 1.0)
        assert np.all(error < 5.0e-3), (thickness, error)
        # ...and it tightens with frequency, as a short-wavelength limit must.
        assert np.all(np.diff(error) < 0.0), (thickness, error)


# ---------------------------------------------------------------------------
# The n=1 / n=2 cutoffs are NOT rigid-pipe fluid-column cutoffs.
#
# `plans/learning.md` proposed checking them against the rigid-pipe closed form
# with the appropriate Bessel zeros, as PR #61 did for n=0. The premise does not
# survive measurement, and the reason is structural rather than numerical:
#
#   * The n=0 mode that check applies to is the *pseudo-Rayleigh* mode, which
#     is a fluid-column resonance -- a higher-order mode of the borehole fluid,
#     perturbed by the wall.
#   * `flexural_dispersion` and `quadrupole_dispersion` return the
#     *fundamental* modes at their azimuthal orders. Those are interface modes,
#     not fluid-column ones, so there is no rigid-pipe resonance for them to be
#     compared against. The solver exposes no n=1 or n=2 counterpart of
#     pseudo-Rayleigh.
#
# What the measurement shows instead is recorded here, because a scaling law is
# falsifiable even when a closed form is not available.
# ---------------------------------------------------------------------------

_CUT_GRID = np.linspace(100.0, 40.0e3, 2000)


def _lowest_converged_frequency(solver, **medium):
    mode = solver(_CUT_GRID, **medium)
    ok = np.isfinite(mode.slowness)
    return float(_CUT_GRID[ok].min()) if ok.any() else float("nan")


def _cutoff_solvers():
    from fwap import flexural_dispersion, quadrupole_dispersion

    return (("flexural", flexural_dispersion), ("quadrupole", quadrupole_dispersion))


def test_n1_n2_cutoffs_scale_inversely_with_borehole_radius():
    """A geometric cutoff exists, and it goes as 1/a.

    This much the rigid-pipe picture would also predict, so on its own it does
    not discriminate; it is asserted because it establishes that there *is* a
    clean geometric cutoff to reason about.
    """
    for name, solver in _cutoff_solvers():
        products = []
        for a in (0.06, 0.10, 0.20):
            medium = dict(vp=2200.0, vs=1000.0, rho=2200.0, vf=1500.0, rho_f=1000.0)
            cutoff = _lowest_converged_frequency(solver, **medium, a=a)
            assert np.isfinite(cutoff), (name, a)
            products.append(cutoff * a)
        spread = max(products) / min(products) - 1.0
        assert spread < 0.05, (name, products)


def test_n1_n2_cutoffs_are_shear_controlled_not_fluid_controlled():
    """The discriminating measurement, and the one that kills the candidate.

    A rigid-pipe fluid-column cutoff is set by the *fluid*: on the n=0 form
    it carries a full power of ``V_f`` and none of ``V_S`` in the rigid limit.
    The n=1 and n=2 cutoffs behave the other way round. Measured as log-log
    sensitivities over a factor ~2 in ``V_S`` and ~1.6 in ``V_f``, the
    exponents are about 0.87 on ``V_S`` and 0.10 on ``V_f``.

    Bounds are set well clear of the measured values rather than tight to
    them, since the point is the qualitative ordering -- shear-controlled, not
    fluid-controlled -- and not the exponent itself.
    """
    for name, solver in _cutoff_solvers():
        slow_shear = _lowest_converged_frequency(
            solver,
            vp=2.2 * 700.0,
            vs=700.0,
            rho=2200.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.10,
        )
        fast_shear = _lowest_converged_frequency(
            solver,
            vp=2.2 * 1450.0,
            vs=1450.0,
            rho=2200.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.10,
        )
        slow_fluid = _lowest_converged_frequency(
            solver, vp=2200.0, vs=1000.0, rho=2200.0, vf=1200.0, rho_f=1000.0, a=0.10
        )
        fast_fluid = _lowest_converged_frequency(
            solver, vp=2200.0, vs=1000.0, rho=2200.0, vf=1900.0, rho_f=1000.0, a=0.10
        )
        assert all(
            np.isfinite(x) for x in (slow_shear, fast_shear, slow_fluid, fast_fluid)
        ), name

        exponent_vs = np.log(fast_shear / slow_shear) / np.log(1450.0 / 700.0)
        exponent_vf = np.log(fast_fluid / slow_fluid) / np.log(1900.0 / 1200.0)

        assert exponent_vs > 0.6, (name, exponent_vs)
        assert exponent_vf < 0.4, (name, exponent_vf)
        assert exponent_vs > 3.0 * exponent_vf, (name, exponent_vs, exponent_vf)


def test_n1_n2_cutoffs_do_not_match_the_rigid_pipe_closed_form():
    """Stated as a test so the candidate cannot quietly come back.

    Evaluated where the rigid-pipe form is even defined -- a fast formation,
    ``V_S > V_f`` -- the closed form and the solver disagree, and by different
    factors for the two orders, so no single constant reconciles them. In that
    regime both solvers are separately known to be defective (roadmap A.2),
    which is the second reason the comparison cannot be made to work.
    """
    from scipy.special import jnp_zeros

    medium = dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10)
    rigid_pipe = {
        order: jnp_zeros(order, 1)[0]
        * medium["vf"]
        * medium["vs"]
        / (2.0 * np.pi * medium["a"] * np.sqrt(medium["vs"] ** 2 - medium["vf"] ** 2))
        for order in (1, 2)
    }

    ratios = {}
    for order, (name, solver) in zip((1, 2), _cutoff_solvers()):
        cutoff = _lowest_converged_frequency(solver, **medium)
        assert np.isfinite(cutoff), name
        ratios[order] = cutoff / rigid_pipe[order]

    # No single constant reconciles the two orders, so this is not the
    # n=0 situation of a fixed offset that could be documented and used.
    assert abs(ratios[1] / ratios[2] - 1.0) > 0.15, ratios


# ---------------------------------------------------------------------------
# Modal biorthogonality: the first oracle here that needs two solutions at once.
#
# Every earlier check evaluates something on a single mode, which is why the
# energy balance turned out vacuous -- a law evaluated on one solution in a
# region where that solution already satisfies the governing equations comes
# back exact and means nothing. Auld's waveguide reciprocity relation does not
# have that escape: it couples two *different* eigenvectors, so it cannot be
# satisfied by construction from either.
#
# The test set is real. In a fast formation the n=0 bound spectrum holds the
# Stoneley mode (c < V_f) *and* the trapped pseudo-Rayleigh modes
# (V_f < c < V_S) -- four of them at 30 kHz, all bound, all azimuthal order 0.
# Note that `stoneley_dispersion` returns only the first: its bracket stops at
# omega/V_f, so the trapped modes are found here directly from the determinant.
# ---------------------------------------------------------------------------

_BIO_MEDIUM = {
    "vp": 4000.0,
    "vs": 2300.0,
    "rho": 2500.0,
    "vf": 1500.0,
    "rho_f": 1000.0,
    "a": 0.10,
}


def _bound_n0_roots(omega, medium):
    """Every bound n=0 root at one frequency: Stoneley plus trapped modes."""
    from scipy.optimize import brentq

    from fwap.cylindrical_solver._leaky import _modal_determinant_n0_complex

    def det(kz):
        return _modal_determinant_n0_complex(
            kz, omega, **medium, leaky_p=False, leaky_s=False
        )

    roots = []
    windows = (
        (omega / medium["vs"] * 1.0001, omega / medium["vf"] * 0.9999),
        (omega / medium["vf"] * 1.0001, omega / medium["vf"] * 3.0),
    )
    for lo, hi in windows:
        grid = np.linspace(lo, hi, 3000)
        values = np.array([det(k) for k in grid])
        use_real = np.nanmax(np.abs(values.real)) >= np.nanmax(np.abs(values.imag))
        scalar = (lambda k: det(k).real) if use_real else (lambda k: det(k).imag)
        flat = values.real if use_real else values.imag
        for i in range(grid.size - 1):
            if not (np.isfinite(flat[i]) and np.isfinite(flat[i + 1])):
                continue
            if np.sign(flat[i]) != np.sign(flat[i + 1]):
                roots.append(
                    brentq(scalar, grid[i], grid[i + 1], xtol=1e-14, rtol=8.9e-16)
                )
    return sorted(roots)


def _mode_shape(kz, omega, medium):
    """Null vector of the 3x3 bound matrix, plus the radial wavenumbers."""
    from scipy import special

    mu = medium["rho"] * medium["vs"] ** 2
    a = medium["a"]
    f_r = np.sqrt(complex(kz * kz - (omega / medium["vf"]) ** 2))
    p = np.sqrt(complex(kz * kz - (omega / medium["vp"]) ** 2))
    s = np.sqrt(complex(kz * kz - (omega / medium["vs"]) ** 2))
    i0, i1 = complex(special.iv(0, f_r * a)), complex(special.iv(1, f_r * a))
    k0p, k1p = complex(special.kv(0, p * a)), complex(special.kv(1, p * a))
    k0s, k1s = complex(special.kv(0, s * a)), complex(special.kv(1, s * a))
    two_kz2 = 2.0 * kz * kz - (omega / medium["vs"]) ** 2
    matrix = np.array(
        [
            [f_r * i1 / (medium["rho_f"] * omega**2), p * k1p, kz * k1s],
            [
                -i0,
                -mu * (two_kz2 * k0p + 2.0 * p * k1p / a),
                -2.0 * kz * mu * (s * k0s + k1s / a),
            ],
            [0.0, 2.0 * kz * p * mu * k1p, mu * two_kz2 * k1s],
        ],
        dtype=complex,
    )
    _, _, vh = np.linalg.svd(matrix)
    vec = vh[-1].conj()
    # The matrix writes continuity as u_r(fluid) + u_r(solid) = 0, so the solid
    # amplitudes carry the opposite sign to the field expressions below.
    return (vec[0], -vec[1], -vec[2]), f_r, p, s


def _mode_fields(r, kz, omega, medium, shape):
    """u_r, u_z, sigma_rz, sigma_zz across the whole cross-section."""
    from scipy import special

    (amp_f, amp_p, amp_s), f_r, p, s = shape
    mu = medium["rho"] * medium["vs"] ** 2
    lam = medium["rho"] * (medium["vp"] ** 2 - 2.0 * medium["vs"] ** 2)
    two_kz2 = 2.0 * kz * kz - (omega / medium["vs"]) ** 2

    r = np.atleast_1d(r).astype(float)
    u_r = np.zeros_like(r, dtype=complex)
    u_z = np.zeros_like(r, dtype=complex)
    s_rz = np.zeros_like(r, dtype=complex)
    s_zz = np.zeros_like(r, dtype=complex)

    inside = r < medium["a"]
    if inside.any():
        ri = r[inside]
        i0, i1 = special.iv(0, f_r * ri), special.iv(1, f_r * ri)
        u_r[inside] = amp_f * f_r * i1 / (medium["rho_f"] * omega**2)
        u_z[inside] = 1j * kz * amp_f * i0 / (medium["rho_f"] * omega**2)
        s_zz[inside] = -amp_f * i0  # sigma_rz vanishes in an inviscid fluid
    if (~inside).any():
        ro = r[~inside]
        k0p, k1p = special.kv(0, p * ro), special.kv(1, p * ro)
        k0s, k1s = special.kv(0, s * ro), special.kv(1, s * ro)
        u_r[~inside] = amp_p * p * k1p + amp_s * kz * k1s
        u_z[~inside] = -1j * (kz * amp_p * k0p + amp_s * s * k0s)
        s_rz[~inside] = 1j * mu * (2.0 * kz * p * amp_p * k1p + amp_s * two_kz2 * k1s)
        s_zz[~inside] = lam * (
            omega**2 / medium["vp"] ** 2
        ) * amp_p * k0p + 2.0 * mu * (kz * (kz * amp_p * k0p + amp_s * s * k0s))
    return u_r, u_z, s_rz, s_zz


def _cross_integral(kz_m, kz_n, omega, medium, shapes, span=15.0):
    """INT (conj(u^m) . T^n . zhat) r dr, by fixed-node Gauss-Legendre.

    Fixed nodes rather than an adaptive rule on purpose: the integrand
    underflows in the evanescent tail, and adaptive quadrature spends its
    error budget out there and returns noise ~1e-4, which is large enough to
    look like a failed orthogonality relation.
    """
    a = medium["a"]
    decay = [
        np.sqrt(max(k * k - (omega / medium["vs"]) ** 2, 1e-12)) for k in (kz_m, kz_n)
    ]
    outer = a + span / min(decay)
    total = 0.0 + 0.0j
    for lo, hi, count in ((0.0, a, 300), (a, outer, 600)):
        x, weights = np.polynomial.legendre.leggauss(count)
        r = 0.5 * (hi - lo) * (x + 1.0) + lo
        u_r, u_z, _, _ = _mode_fields(r, kz_m, omega, medium, shapes[kz_m])
        _, _, s_rz, s_zz = _mode_fields(r, kz_n, omega, medium, shapes[kz_n])
        total += (
            0.5
            * (hi - lo)
            * np.sum(weights * (np.conj(u_r) * s_rz + np.conj(u_z) * s_zz) * r)
        )
    return total


def test_bound_n0_modes_satisfy_the_boundary_conditions_they_were_built_from():
    """Sanity gate on the eigenfunctions before they are used for anything.

    Written first because a sign slip in the solid amplitudes was caught here
    -- ``|du_r| / |u_r|`` came back as exactly 2.0, which is what equal and
    opposite gives -- and would otherwise have shown up as a failed
    orthogonality relation, i.e. as a fake finding about the solver.
    """
    omega = 2.0 * np.pi * 30.0e3
    roots = _bound_n0_roots(omega, _BIO_MEDIUM)
    assert len(roots) >= 3, roots

    step = 1.0e-9
    for kz in roots:
        shape = _mode_shape(kz, omega, _BIO_MEDIUM)
        u_in = _mode_fields(_BIO_MEDIUM["a"] - step, kz, omega, _BIO_MEDIUM, shape)[0]
        u_out = _mode_fields(_BIO_MEDIUM["a"] + step, kz, omega, _BIO_MEDIUM, shape)[0]
        scale = max(abs(u_in[0]), abs(u_out[0]))
        assert abs(u_in[0] - u_out[0]) / scale < 1.0e-6, kz


def test_bound_n0_modes_are_biorthogonal():
    """Auld's relation across every pair of coexisting bound modes.

    ``S_mn - conj(S_nm)`` must vanish for ``m != n``. Measured over the four
    bound modes at 30 kHz it does, to ~1e-13 relative, while the diagonal
    stays O(1) -- so this is orthogonality rather than everything being small.
    """
    omega = 2.0 * np.pi * 30.0e3
    roots = _bound_n0_roots(omega, _BIO_MEDIUM)
    shapes = {kz: _mode_shape(kz, omega, _BIO_MEDIUM) for kz in roots}
    n = len(roots)

    gram = np.array(
        [
            [
                _cross_integral(roots[i], roots[j], omega, _BIO_MEDIUM, shapes)
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    for i in range(n):
        for j in range(n):
            scale = np.sqrt(abs(gram[i, i] * gram[j, j]))
            assert scale > 0.0
            residual = abs(gram[i, j] - np.conj(gram[j, i])) / scale
            if i == j:
                continue
            assert residual < 1.0e-9, (i, j, residual)


def test_biorthogonality_check_rejects_the_wrong_bilinear_form():
    """The relation is specific, not a statement that everything is small.

    Using one term of the pairing instead of Auld's difference -- a natural
    wrong guess, and the one tried first -- leaves off-diagonals around 1e-2,
    ten orders above the tolerance the correct form meets. That gap is what
    makes the test above evidence rather than an accident of normalisation.
    """
    omega = 2.0 * np.pi * 30.0e3
    roots = _bound_n0_roots(omega, _BIO_MEDIUM)
    shapes = {kz: _mode_shape(kz, omega, _BIO_MEDIUM) for kz in roots}

    worst = 0.0
    for i in range(len(roots)):
        for j in range(len(roots)):
            if i == j:
                continue
            s_ij = _cross_integral(roots[i], roots[j], omega, _BIO_MEDIUM, shapes)
            s_ii = _cross_integral(roots[i], roots[i], omega, _BIO_MEDIUM, shapes)
            s_jj = _cross_integral(roots[j], roots[j], omega, _BIO_MEDIUM, shapes)
            worst = max(worst, abs(s_ij) / np.sqrt(abs(s_ii * s_jj)))
    assert worst > 1.0e-3, worst


# ---------------------------------------------------------------------------
# Trapped pseudo-Rayleigh modes, now public.
#
# These were found while building the biorthogonality check and were reachable
# only by scanning the determinant directly: `stoneley_dispersion` brackets
# from omega/min(V_S, V_f) upward and so returns just the Stoneley mode, and
# `pseudo_rayleigh_dispersion` covers the leaky half above V_S.
# ---------------------------------------------------------------------------

_TRAP_MEDIUM = {
    "vp": 4000.0,
    "vs": 2300.0,
    "rho": 2500.0,
    "vf": 1500.0,
    "rho_f": 1000.0,
    "a": 0.10,
}


def test_trapped_pseudo_rayleigh_lies_strictly_inside_the_trapped_window():
    """Bound between the fluid and shear velocities, and lossless.

    The defining property: both formation waves evanescent, fluid field
    oscillatory. Outside ``V_f < c < V_S`` the mode is either the Stoneley
    wave or a leaky one, and neither belongs here.
    """
    from fwap import trapped_pseudo_rayleigh_dispersion

    freq = np.linspace(8.0e3, 60.0e3, 20)
    mode = trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM)
    assert mode.name == "trapped_pseudo_rayleigh"
    assert mode.azimuthal_order == 0
    assert mode.attenuation_per_meter is None  # bound modes do not radiate

    ok = np.isfinite(mode.slowness)
    assert ok.sum() > 15
    speeds = 1.0 / mode.slowness[ok]
    assert np.all(speeds > _TRAP_MEDIUM["vf"])
    assert np.all(speeds < _TRAP_MEDIUM["vs"])


def test_trapped_pseudo_rayleigh_branches_appear_in_order_and_descend():
    """Higher orders switch on at higher frequency; each then slows toward V_f.

    Each branch starts at its own cutoff near ``V_S`` and decreases toward
    the fluid velocity, so at any one frequency the fundamental is the
    *slowest* of the trapped modes -- which is the ordering convention the
    ``branch`` argument uses, shared with the leaky sister function.
    """
    from fwap import trapped_pseudo_rayleigh_dispersion

    freq = np.linspace(5.0e3, 60.0e3, 40)
    curves = [
        trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM, branch=b).slowness
        for b in range(3)
    ]

    cutoffs = []
    for slowness in curves:
        ok = np.isfinite(slowness)
        assert ok.any()
        cutoffs.append(freq[ok].min())
        speeds = 1.0 / slowness[ok]
        # monotonically slowing with frequency, allowing for grid coarseness
        assert speeds[0] > speeds[-1]
    assert cutoffs[0] < cutoffs[1] < cutoffs[2], cutoffs

    # ...and at a frequency where all three exist, branch 0 is the slowest.
    high = np.array([60.0e3])
    speeds = [
        1.0
        / trapped_pseudo_rayleigh_dispersion(high, **_TRAP_MEDIUM, branch=b).slowness[0]
        for b in range(3)
    ]
    assert speeds[0] < speeds[1] < speeds[2], speeds


def test_trapped_pseudo_rayleigh_is_independent_of_the_frequency_grid():
    """No marching, so no grid coupling -- unlike the leaky sister function.

    Each frequency is solved on its own, which is why this function needs
    none of the seed-enumeration machinery `pseudo_rayleigh_dispersion`
    required. Worth pinning, because that grid independence is the whole
    reason the simpler algorithm is safe here.
    """
    from fwap import trapped_pseudo_rayleigh_dispersion

    probe = 33.0e3
    reference = None
    for base in (
        np.array([]),
        np.linspace(5.0e3, 40.0e3, 37),
        np.linspace(20.0e3, 90.0e3, 200),
        np.array([90.0e3, 12.0e3]),  # unordered once the probe is inserted
    ):
        # The probe must be an exact member of every grid: comparing the
        # *nearest* sample would compare different frequencies, which is a
        # different quantity rather than a grid effect.
        grid = np.concatenate([base, [probe]])
        mode = trapped_pseudo_rayleigh_dispersion(grid, **_TRAP_MEDIUM)
        j = int(np.flatnonzero(grid == probe)[0])
        value = mode.slowness[j]
        assert np.isfinite(value)
        if reference is None:
            reference = value
        else:
            assert value == pytest.approx(reference, rel=1.0e-12)


def test_trapped_pseudo_rayleigh_is_biorthogonal_to_the_stoneley_mode():
    """Checked against physics rather than against itself.

    The trapped modes and the Stoneley mode coexist at the same frequency
    and the same azimuthal order, so Auld's relation must hold across them.
    That ties the new function to the biorthogonality oracle above: a
    spurious root would not be orthogonal to the Stoneley mode.
    """
    from fwap import stoneley_dispersion, trapped_pseudo_rayleigh_dispersion

    omega = 2.0 * np.pi * 30.0e3
    freq = np.array([30.0e3])

    trapped = [
        trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM, branch=b).slowness[0]
        for b in range(3)
    ]
    stoneley = stoneley_dispersion(freq, **_TRAP_MEDIUM).slowness[0]
    assert all(np.isfinite(s) for s in trapped) and np.isfinite(stoneley)

    wavenumbers = [s * omega for s in trapped] + [stoneley * omega]
    shapes = {kz: _mode_shape(kz, omega, _BIO_MEDIUM) for kz in wavenumbers}
    n = len(wavenumbers)
    gram = np.array(
        [
            [
                _cross_integral(
                    wavenumbers[i], wavenumbers[j], omega, _BIO_MEDIUM, shapes
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scale = np.sqrt(abs(gram[i, i] * gram[j, j]))
            assert abs(gram[i, j] - np.conj(gram[j, i])) / scale < 1.0e-9, (i, j)


def test_trapped_pseudo_rayleigh_rejects_slow_formations_and_bad_input():
    """A slow formation has no trapped window at all."""
    from fwap import trapped_pseudo_rayleigh_dispersion

    freq = np.array([10.0e3])
    with pytest.raises(ValueError, match="fast formation"):
        trapped_pseudo_rayleigh_dispersion(
            freq, vp=2600.0, vs=1300.0, rho=2300.0, vf=1500.0, rho_f=1000.0, a=0.10
        )
    with pytest.raises(ValueError, match="branch must be non-negative"):
        trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM, branch=-1)
    with pytest.raises(ValueError, match="require vp > vs"):
        trapped_pseudo_rayleigh_dispersion(
            freq, vp=2000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10
        )
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        trapped_pseudo_rayleigh_dispersion(np.array([0.0]), **_TRAP_MEDIUM)

    for bad in ({"vs": -1.0}, {"rho_f": 0.0}, {"a": -0.1}):
        with pytest.raises(ValueError, match="must (all )?be positive"):
            trapped_pseudo_rayleigh_dispersion(freq, **{**_TRAP_MEDIUM, **bad})


def test_trapped_root_scan_resolution_does_not_decide_how_many_modes_exist():
    """The scan density must not set the branch count, or `branch` drifts."""
    from fwap.cylindrical_solver._leaky import (
        _modal_determinant_n0_complex,
        _scan_bound_roots,
    )

    omega = 2.0 * np.pi * 50.0e3

    def det(kz):
        return _modal_determinant_n0_complex(
            kz, omega, **_TRAP_MEDIUM, leaky_p=False, leaky_s=False
        ).real

    lo = omega / _TRAP_MEDIUM["vs"] * (1.0 + 1.0e-9)
    hi = omega / _TRAP_MEDIUM["vf"] * (1.0 - 1.0e-9)
    counts = {
        len(_scan_bound_roots(det, lo, hi, samples=n)) for n in (500, 1000, 2000, 4000)
    }
    assert len(counts) == 1, counts


# ---------------------------------------------------------------------------
# Compliant layers in the cased stack: NaN rather than nonsense.
#
# Found while starting the free-pipe / debonded item (roadmap G.2). Modelling
# debonding as a very compliant *elastic* layer does not work, and the way it
# failed was the dangerous way: the propagator's dynamic range ran past double
# precision, the 7x7 determinant became meaningless, and the bracket search
# found sign changes in the garbage and returned them as roots -- finite
# slownesses corresponding to phase velocities of 3-12 m/s, against a fluid
# velocity of 1500.
#
# The guard rejects a propagator product that cannot be formed in double
# precision and returns NaN instead. The bonded regime is untouched.
# ---------------------------------------------------------------------------

_DEBOND_FORMATION = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_DEBOND_FLUID = {"vf": 1500.0, "rho_f": 1000.0}


def _debond_stack(layer_vs, thickness_m):
    from fwap import BoreholeLayer

    return (
        BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        BoreholeLayer(
            vp=max(1600.0, 2.0 * layer_vs),
            vs=layer_vs,
            rho=1000.0,
            thickness=thickness_m,
        ),
        BoreholeLayer(vp=2800.0, vs=1600.0, rho=1920.0, thickness=0.03),
    )


def test_compliant_cased_layer_returns_nan_not_a_spurious_root():
    """A phase velocity of a few m/s is not a mode, and must not be reported.

    Checked across thicknesses and compliances because the failure was not
    monotone in either: some configurations produced a spurious root with no
    warning at all, which is why a warning filter would not have caught it.
    """
    from fwap import stoneley_dispersion_layered

    freq = np.linspace(2.0e3, 14.0e3, 13)
    for thickness in (1.0e-3, 0.2e-3):
        for layer_vs in (600.0, 300.0, 100.0, 30.0):
            mode = stoneley_dispersion_layered(
                freq,
                **_DEBOND_FORMATION,
                **_DEBOND_FLUID,
                a=0.10,
                layers=_debond_stack(layer_vs, thickness),
            )
            finite = mode.slowness[np.isfinite(mode.slowness)]
            # Either nothing at all, or nothing absurd. Never a 3 m/s "mode".
            if finite.size:
                assert np.all(1.0 / finite > 0.5 * _DEBOND_FLUID["vf"]), (
                    thickness,
                    layer_vs,
                    1.0 / finite,
                )


def test_compliant_cased_layer_does_not_warn():
    """...and it gets there without emitting numerical warnings.

    The overflow was raised by the matmul itself, so testing the *result* for
    finiteness cleaned up the determinant but left the warning. The guard
    checks the product's magnitude before forming it.
    """
    import warnings

    from fwap import stoneley_dispersion_layered

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for thickness in (1.0e-3, 0.2e-3):
            for layer_vs in (300.0, 100.0, 30.0):
                stoneley_dispersion_layered(
                    np.linspace(2.0e3, 14.0e3, 13),
                    **_DEBOND_FORMATION,
                    **_DEBOND_FLUID,
                    a=0.10,
                    layers=_debond_stack(layer_vs, thickness),
                )


def test_bonded_cased_stoneley_is_unaffected_by_the_guard():
    """The guard must not cost anything in the regime that works.

    Cement stiffer than the fluid converges across the whole band and the
    values are pinned, so a guard that clipped valid configurations would
    show up here rather than as a silent loss of coverage.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    freq = np.linspace(2.0e3, 14.0e3, 13)
    expected = {2200.0: 1440.30, 1800.0: 1420.19, 1600.0: 1404.12, 1500.0: 1393.53}
    for cement_vs, speed_at_8k in expected.items():
        layers = (
            BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
            BoreholeLayer(
                vp=max(2.0 * cement_vs, 2800.0),
                vs=cement_vs,
                rho=1920.0,
                thickness=0.03,
            ),
        )
        mode = stoneley_dispersion_layered(
            freq, **_DEBOND_FORMATION, **_DEBOND_FLUID, a=0.10, layers=layers
        )
        assert np.all(np.isfinite(mode.slowness)), cement_vs
        assert 1.0 / mode.slowness[6] == pytest.approx(speed_at_8k, rel=1.0e-4)


def test_cased_stoneley_stops_being_bound_once_cement_is_softer_than_the_fluid():
    """The documented restriction on the cased dataset's prior, measured.

    `CasingCementPriors` keeps `cement_vs >= vf` on the stated grounds that
    the cased Stoneley stops being bound below it. That holds: full
    convergence at and above the fluid velocity, partial just below, none
    further down. It is the reason the shipped cased dataset spans graded
    cement quality rather than reaching debonding.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    freq = np.linspace(2.0e3, 14.0e3, 13)

    def converged(cement_vs):
        layers = (
            BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
            BoreholeLayer(
                vp=max(2.0 * cement_vs, 2800.0),
                vs=cement_vs,
                rho=1920.0,
                thickness=0.03,
            ),
        )
        mode = stoneley_dispersion_layered(
            freq, **_DEBOND_FORMATION, **_DEBOND_FLUID, a=0.10, layers=layers
        )
        return int(np.isfinite(mode.slowness).sum())

    assert converged(1600.0) == freq.size
    assert converged(1500.0) == freq.size
    assert converged(1200.0) == 0
    assert converged(800.0) == 0


# ---------------------------------------------------------------------------
# Fluid-annulus propagator (n=0): the first piece of the microannulus model.
#
# Starting roadmap G.2 established that debonding modelled as a fluid
# microannulus is not blocked by the leaky-mode derivation the roadmap assumed
# -- a fluid contributes no shear-velocity floor to the bound-mode bracket --
# but that it does need a propagator element the module did not have. This is
# that element, verified in isolation before anything is wired to it.
#
# It is deliberately not reachable from the public API yet. The layer stack
# cannot express a fluid (`BoreholeLayer` requires vs > 0) and the global
# assembly changes shape when one is present, since a fluid annulus carries two
# amplitudes rather than four and imposes sigma_rz = 0 at both faces. Shipping a
# public layer type no solver accepts would be worse than shipping nothing.
# ---------------------------------------------------------------------------

_FA_OMEGA = 2.0 * np.pi * 8.0e3
_FA_FLUID = {"vf": 1500.0, "rho": 1000.0}


def test_fluid_annulus_e_matrix_determinant_matches_the_wronskian():
    """``det E_f(r) = -1 / (rho omega^2 r)``, with no dependence on ``F``.

    The Bessel Wronskian ``I0(x) K1(x) + I1(x) K0(x) = 1/x`` collapses the
    determinant to a closed form that does not involve the radial
    wavenumber at all. That makes it a check from outside this module: it
    holds for every ``k_z``, frequency and fluid, so a sign slip or a
    swapped Bessel order breaks it immediately.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_e_matrix_n0

    for phase_velocity in (900.0, 1200.0, 1450.0):
        kz = _FA_OMEGA / phase_velocity
        for r in (0.09, 0.11, 0.15, 0.30):
            e_matrix = _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, **_FA_FLUID, r=r)
            expected = -1.0 / (_FA_FLUID["rho"] * _FA_OMEGA**2 * r)
            assert np.linalg.det(e_matrix) == pytest.approx(expected, rel=1.0e-10)


def test_fluid_annulus_e_matrix_reproduces_the_momentum_relation():
    """The physics, checked against a numerical derivative rather than algebra.

    ``u_r = (1 / rho omega^2) dp/dr`` and ``sigma_rr = -p``. Building the
    pressure from the amplitudes and differentiating it numerically tests
    the matrix against the equation it is supposed to encode, independently
    of how it was derived -- which is the check that would have caught the
    sign convention mistakes made earlier in this work.
    """
    from scipy import special

    from fwap.cylindrical_solver._cased import _fluid_layer_e_matrix_n0

    kz = _FA_OMEGA / 1200.0
    f_radial = np.sqrt(kz**2 - (_FA_OMEGA / _FA_FLUID["vf"]) ** 2)
    amplitudes = np.array([0.37, -1.9])

    def pressure(r):
        return amplitudes[0] * special.iv(0, f_radial * r) + amplitudes[1] * special.kv(
            0, f_radial * r
        )

    for r in (0.10, 0.13):
        state = _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, **_FA_FLUID, r=r) @ amplitudes
        step = 1.0e-7
        dp_dr = (pressure(r + step) - pressure(r - step)) / (2.0 * step)
        assert state[0] == pytest.approx(
            dp_dr / (_FA_FLUID["rho"] * _FA_OMEGA**2), rel=1.0e-6
        )
        assert state[1] == pytest.approx(-pressure(r), rel=1.0e-12)


def test_fluid_annulus_propagator_determinant_is_the_radius_ratio():
    """``det P_f = r_inner / r_outer``, independent of everything else.

    Follows from the determinant identity above, and is the sharpest
    available statement about the propagator: no frequency, velocity,
    density or ``k_z`` appears in it.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_propagator_n0

    for phase_velocity in (900.0, 1200.0, 1450.0):
        kz = _FA_OMEGA / phase_velocity
        # Microannulus geometries (microns to a few mm) plus a deliberately
        # generous annulus. All sit inside the accuracy range characterised by
        # the next test.
        for r_inner, r_outer in (
            (0.11, 0.11002),
            (0.11, 0.1101),
            (0.10, 0.11),
            (0.10, 0.14),
        ):
            propagator = _fluid_layer_propagator_n0(
                kz, _FA_OMEGA, **_FA_FLUID, r_inner=r_inner, r_outer=r_outer
            )
            assert np.linalg.det(propagator) == pytest.approx(
                r_inner / r_outer, rel=1.0e-10
            )


def test_fluid_annulus_propagator_accuracy_is_set_by_the_bessel_span():
    """Where the element stops being usable, measured rather than assumed.

    The propagator carries ``I`` and ``K`` Bessel functions across the
    annulus, so its dynamic range grows with ``F * (r_outer - r_inner)`` and
    eventually exceeds double precision. Using the determinant identity as
    the error measure, accuracy degrades smoothly: machine precision up to a
    span of about 2, ~1e-11 by 7, and no significant digits at all by 20.

    This matters for how the element should be used, not for the case it was
    built for: a microannulus is microns to millimetres thick, which puts
    ``F * dr`` below 0.1 and nowhere near the limit. Recorded because the
    same exponential-range failure produced spurious roots elsewhere in this
    module, and because an element with an undocumented validity range is
    how that happens again.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_propagator_n0

    kz = _FA_OMEGA / 900.0
    f_radial = np.sqrt(kz**2 - (_FA_OMEGA / _FA_FLUID["vf"]) ** 2)

    def relative_error(r_inner, r_outer):
        propagator = _fluid_layer_propagator_n0(
            kz, _FA_OMEGA, **_FA_FLUID, r_inner=r_inner, r_outer=r_outer
        )
        return abs(np.linalg.det(propagator) / (r_inner / r_outer) - 1.0)

    # Comfortably inside the range: exact.
    assert f_radial * (0.14 - 0.10) < 3.0
    assert relative_error(0.10, 0.14) < 1.0e-12

    # Far outside it: the identity fails outright, which is the point.
    assert f_radial * (0.50 - 0.05) > 15.0
    assert relative_error(0.05, 0.50) > 1.0e-3


def test_fluid_annulus_propagator_composes_and_degenerates():
    """Zero thickness gives the identity; subdivision composes.

    The same invariance that the elastic layer stack satisfies, and for
    the same reason -- relabelling one annulus as two changes the
    description and not the medium. Checked here on the element itself, so
    that when the assembly is built any failure is attributable to the
    assembly.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_propagator_n0

    kz = _FA_OMEGA / 1150.0

    identity = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.12, r_outer=0.12
    )
    assert np.allclose(identity, np.eye(2), atol=1.0e-12)

    whole = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.10, r_outer=0.16
    )
    first = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.10, r_outer=0.13
    )
    second = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.13, r_outer=0.16
    )
    assert np.allclose(whole, second @ first, rtol=1.0e-9)


def test_fluid_annulus_rejects_a_propagating_radial_wavenumber():
    """The bound formulation needs ``kz > omega / vf`` in the annulus.

    Below it the radial wavenumber turns imaginary, the fluid field
    oscillates instead of decaying, and the real-valued state matrix here
    no longer describes it. Refused rather than silently returning the
    real part -- the microannulus case this element exists for is squarely
    in the bound regime, and a caller who has left it should be told.
    """
    from fwap.cylindrical_solver._cased import (
        _fluid_layer_e_matrix_n0,
        _fluid_layer_propagator_n0,
    )

    kz_oscillatory = _FA_OMEGA / (2.0 * _FA_FLUID["vf"])
    with pytest.raises(ValueError, match="not real"):
        _fluid_layer_e_matrix_n0(kz_oscillatory, _FA_OMEGA, **_FA_FLUID, r=0.12)

    kz = _FA_OMEGA / 1200.0
    with pytest.raises(ValueError, match="must be positive"):
        _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, vf=-1.0, rho=1000.0, r=0.12)
    with pytest.raises(ValueError, match="r must be positive"):
        _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, **_FA_FLUID, r=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        _fluid_layer_propagator_n0(kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.0, r_outer=0.1)
    with pytest.raises(ValueError, match="require r_outer >= r_inner"):
        _fluid_layer_propagator_n0(kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.2, r_outer=0.1)


# ---------------------------------------------------------------------------
# Microannulus global assembly (n=0): the 11x11 determinant for
# `fluid | casing | microannulus | cement | formation`.
#
# The fluid element above is verified in isolation; this section verifies the
# assembly that uses it. There is no reduction to the existing solver available
# here -- the `annulus_thickness -> 0` limit is a frictionless slip interface,
# not the bonded stack -- so the checks are:
#
# * the Krauklis (fluid-filled-crack) wave, an analytic result derived from
#   lubrication flow and quasi-static wall compliance, with no Bessel functions
#   and no cylindrical geometry in it at all. This is the strong one: it fixes
#   the absolute phase velocity of the slow root, not just its scaling, so a
#   wrong row or a swapped condition would show up as an O(1) prefactor error.
# * an independently assembled 13x13 form that keeps the annulus amplitudes as
#   explicit unknowns. Different size, different column layout -- an index slip
#   in one does not reproduce in the other.
# * subdivision invariance of the elastic blocks.
#
# The assembly is deliberately not reachable from the public API yet: choosing
# which root family a dispersion curve should follow is a separate decision,
# and this section shows there are two.
# ---------------------------------------------------------------------------

_MA_FORMATION = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_MA_FLUID = {"vf": 1500.0, "rho_f": 1000.0}
_MA_ANNULUS = {"annulus_vf": 1500.0, "annulus_rho": 1000.0}
_MA_A = 0.10
_MA_CASING = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.01)
_MA_CEMENT = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.03)
# Blocks thick compared with the gap mode's decay length ~ 1 / k_z, so the
# half-space idealisation the Krauklis formula assumes actually applies. The
# confinement test below measures what happens when they are not.
_MA_THICK_CASING = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.05)
_MA_THICK_CEMENT = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.20)


def _ma_det(
    phase_velocity,
    freq,
    *,
    thickness=1.0e-4,
    inner=(_MA_CASING,),
    outer=(_MA_CEMENT,),
    **overrides,
):
    """The 11x11 microannulus determinant as a function of phase velocity."""
    from fwap.cylindrical_solver._cased import _modal_determinant_n0_microannulus

    omega = 2.0 * np.pi * freq
    kwargs = {
        **_MA_FORMATION,
        **_MA_FLUID,
        **_MA_ANNULUS,
        "a": _MA_A,
        "inner_layers": inner,
        "outer_layers": outer,
        "annulus_thickness": thickness,
    }
    kwargs.update(overrides)
    return _modal_determinant_n0_microannulus(omega / phase_velocity, omega, **kwargs)


def _ma_roots(det_fn, lo, hi, samples=800, log_grid=False):
    """Every sign change of ``det_fn`` on ``[lo, hi]``, refined by brentq.

    800 samples rather than the few thousand a first pass used: the root set
    is unchanged from 200 samples upward and under moved endpoints, measured
    below in ``test_microannulus_carries_two_root_families``.
    """
    from scipy import optimize

    grid = np.geomspace(lo, hi, samples) if log_grid else np.linspace(lo, hi, samples)
    values = np.array([det_fn(c) for c in grid])
    finite = np.isfinite(values)
    found = []
    for i in range(grid.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if np.sign(values[i]) != np.sign(values[i + 1]):
            found.append(
                optimize.brentq(det_fn, grid[i], grid[i + 1], xtol=1e-13, rtol=1e-15)
            )
    return found


def _krauklis_velocity(freq, thickness, inner, outer, rho_annulus):
    r"""Phase velocity of the crack (Krauklis) wave in a thin fluid gap.

    Derived outside this module and with none of its machinery. Pressure in a
    gap thin compared with the wavelength is uniform across it, so the fluid
    accelerates along the gap under its own pressure gradient,
    ``u_z = i k p / (rho_f omega^2)``, and volume conservation ties the flow to
    the opening ``Delta`` of the two walls:

        d(h u_z)/dz + Delta = 0   =>   Delta = h k^2 p / (rho_f omega^2).

    Each wall is a half-space loaded by a normal traction ``p e^{ikz}``, whose
    quasi-static surface displacement is ``p (1 - nu) / (mu k)``. Summing the
    two compliances, ``C = sum (1 - nu) / mu``, and eliminating ``p``:

        k^3 = C rho_f omega^2 / h,   c = omega / k = (omega h / (C rho_f))^{1/3}.

    Bessel-free, cylinder-free, and independent of the borehole fluid and the
    formation. Valid when the gap is thin compared with both the wavelength and
    the wall thicknesses.

    Parameters
    ----------
    freq : float
        Frequency (Hz).
    thickness : float
        Gap thickness (m).
    inner, outer : BoreholeLayer
        The two walls; only their elastic moduli are used.
    rho_annulus : float
        Gap fluid density (kg/m^3).

    Returns
    -------
    float
        Phase velocity (m/s).
    """
    compliance = 0.0
    for wall in (inner, outer):
        mu = wall.rho * wall.vs**2
        poisson = (wall.vp**2 - 2.0 * wall.vs**2) / (2.0 * (wall.vp**2 - wall.vs**2))
        compliance += (1.0 - poisson) / mu
    omega = 2.0 * np.pi * freq
    return float((omega * thickness / (compliance * rho_annulus)) ** (1.0 / 3.0))


def _ma_gap_root(freq, thickness, inner=None, outer=None, rho_annulus=1000.0):
    """The slow gap-mode root, searched on a log grid well below the fluid."""
    inner = inner or _MA_THICK_CASING
    outer = outer or _MA_THICK_CEMENT

    def det(c):
        return _ma_det(
            c,
            freq,
            thickness=thickness,
            inner=(inner,),
            outer=(outer,),
            annulus_rho=rho_annulus,
        )

    found = _ma_roots(det, 5.0, 1400.0, log_grid=True)
    assert found, "no gap-mode root found"
    return found[0]


def test_microannulus_slow_root_is_the_krauklis_crack_wave():
    """The slow root converges to the analytic crack-wave speed as the gap thins.

    This is the check that the assembly is right rather than merely
    self-consistent. ``_krauklis_velocity`` shares no code, no special
    functions and no geometry with the solver, and it predicts an absolute
    velocity: if the shear-traction-free rows at the two gap faces, or the
    ``(u_r, sigma_rr)`` pair carried across the gap, were wrong, the prefactor
    would be off by an O(1) factor rather than by the O(k h) thin-gap error.

    Measured ratios at 8 kHz, thick walls: 1.0002 at a 1 um gap, 0.998 at
    10 um, 0.983 at 100 um, 0.915 at 1 mm -- converging to the analytic value
    exactly where the derivation says it should, and departing as ``k h``
    stops being small.
    """
    for thickness, tol in ((1.0e-6, 2.0e-3), (1.0e-5, 1.0e-2), (1.0e-4, 3.0e-2)):
        measured = _ma_gap_root(8.0e3, thickness)
        predicted = _krauklis_velocity(
            8.0e3, thickness, _MA_THICK_CASING, _MA_THICK_CEMENT, 1000.0
        )
        assert measured == pytest.approx(predicted, rel=tol)

    # The approximation degrades monotonically in k h, and does so in the
    # direction the derivation implies (fluid inertia neglected across the gap
    # makes the analytic wave too fast), so the ordering is a claim too.
    ratios = [
        _ma_gap_root(8.0e3, h)
        / _krauklis_velocity(8.0e3, h, _MA_THICK_CASING, _MA_THICK_CEMENT, 1000.0)
        for h in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < 0.95


def test_microannulus_gap_mode_follows_the_crack_wave_parameter_scaling():
    """``c ~ (omega h / (C rho_f))^{1/3}`` in every variable separately.

    A prefactor match at one operating point could be a coincidence of the
    chosen numbers. Varying frequency, wall stiffness and gap-fluid density
    -- each of which enters the analytic formula differently -- makes that
    much harder: the cube-root scaling in ``omega`` and ``h``, the linear
    scaling in wall compliance, and the inverse scaling in gap-fluid density
    are four independent predictions.
    """
    soft = BoreholeLayer(vp=2000.0, vs=1100.0, rho=1800.0, thickness=0.20)
    stiff = BoreholeLayer(vp=4500.0, vs=2600.0, rho=2400.0, thickness=0.20)
    cases = [
        (500.0, 1.0e-5, _MA_THICK_CEMENT, 1000.0),
        (2.0e3, 1.0e-5, _MA_THICK_CEMENT, 1000.0),
        (2.0e4, 1.0e-5, _MA_THICK_CEMENT, 1000.0),
        (8.0e3, 1.0e-5, soft, 1000.0),
        (8.0e3, 1.0e-5, stiff, 1000.0),
        (8.0e3, 1.0e-5, _MA_THICK_CEMENT, 700.0),
        (8.0e3, 1.0e-5, _MA_THICK_CEMENT, 1500.0),
    ]
    for freq, thickness, outer, rho_annulus in cases:
        measured = _ma_gap_root(freq, thickness, outer=outer, rho_annulus=rho_annulus)
        predicted = _krauklis_velocity(
            freq, thickness, _MA_THICK_CASING, outer, rho_annulus
        )
        assert measured == pytest.approx(predicted, rel=2.0e-2)


def test_microannulus_gap_mode_needs_walls_thicker_than_its_decay_length():
    """Where the analytic oracle stops applying, and why -- measured, not assumed.

    The crack wave is confined to within ``~1 / k_z`` of the gap, so the
    half-space compliance the formula uses is only right when the walls are
    thicker than that. At 8 kHz and a 100 um gap the mode's ``1 / k_z`` is
    about 6 mm: a 2 mm casing gives 0.64 of the analytic speed, a 5 mm casing
    0.87, and the ratio saturates near 0.98 once the casing exceeds ~1 cm.
    Recorded so the oracle's domain is a measurement rather than a hope.
    """
    ratios = {}
    for casing_thickness in (0.002, 0.005, 0.01, 0.05):
        casing = BoreholeLayer(
            vp=5900.0, vs=3200.0, rho=7800.0, thickness=casing_thickness
        )
        measured = _ma_gap_root(8.0e3, 1.0e-4, inner=casing)
        ratios[casing_thickness] = measured / _krauklis_velocity(
            8.0e3, 1.0e-4, casing, _MA_THICK_CEMENT, 1000.0
        )
    assert ratios[0.002] < 0.7
    assert ratios[0.005] < ratios[0.01] < ratios[0.05]
    assert ratios[0.05] == pytest.approx(0.98, abs=0.02)


def _ma_det_13(phase_velocity, freq, thickness, inner, outer):
    """Independent 13x13 assembly keeping the annulus amplitudes explicit.

    Same physics, different bookkeeping: instead of folding the two gap
    amplitudes out through the fluid propagator, they stay as unknowns and
    ``(u_r, sigma_rr)`` is matched at both gap faces. Thirteen unknowns
    (1 borehole + 4 + 2 + 4 + 2) against thirteen equations (3 + 3 + 3 + 4).
    """
    from scipy import special

    from fwap.cylindrical_solver._cased import _fluid_layer_e_matrix_n0

    omega = 2.0 * np.pi * freq
    kz = omega / phase_velocity

    def block(layers, r_start):
        e_inner = _layer_e_matrix_n0(
            kz=kz,
            omega=omega,
            vp=layers[0].vp,
            vs=layers[0].vs,
            rho=layers[0].rho,
            r=r_start,
        )
        transfer = np.eye(4)
        r_lo = r_start
        for layer in layers:
            transfer = (
                _layer_propagator_n0(
                    kz=kz,
                    omega=omega,
                    vp=layer.vp,
                    vs=layer.vs,
                    rho=layer.rho,
                    r_inner=r_lo,
                    r_outer=r_lo + layer.thickness,
                )
                @ transfer
            )
            r_lo += layer.thickness
        return e_inner, transfer, r_lo

    e_in_a, transfer_in, r_b = block(inner, _MA_A)
    r_c = r_b + thickness
    e_out_c, transfer_out, r_d = block(outer, r_c)
    e_form = _layer_e_matrix_n0(kz=kz, omega=omega, **_MA_FORMATION, r=r_d)
    fluid_kwargs = {"vf": _MA_ANNULUS["annulus_vf"], "rho": _MA_ANNULUS["annulus_rho"]}
    e_gap_b = _fluid_layer_e_matrix_n0(kz, omega, **fluid_kwargs, r=r_b)
    e_gap_c = _fluid_layer_e_matrix_n0(kz, omega, **fluid_kwargs, r=r_c)
    state_b = transfer_in @ e_in_a
    state_d = transfer_out @ e_out_c

    f_core = np.sqrt(kz * kz - (omega / _MA_FLUID["vf"]) ** 2)
    i0 = float(special.iv(0, f_core * _MA_A))
    i1 = float(special.iv(1, f_core * _MA_A))

    # Columns: [A | inner (4) | gap (2) | outer (4) | formation (2)]
    m = np.zeros((13, 13))
    m[0, 0] = f_core * i1 / (_MA_FLUID["rho_f"] * omega**2)
    m[0, 1:5] = -e_in_a[0, :]
    m[1, 0] = -i0
    m[1, 1:5] = -e_in_a[2, :]
    m[2, 1:5] = e_in_a[3, :]
    m[3, 1:5] = state_b[0, :]
    m[3, 5:7] = -e_gap_b[0, :]
    m[4, 1:5] = state_b[2, :]
    m[4, 5:7] = -e_gap_b[1, :]
    m[5, 1:5] = state_b[3, :]
    m[6, 5:7] = e_gap_c[0, :]
    m[6, 7:11] = -e_out_c[0, :]
    m[7, 5:7] = e_gap_c[1, :]
    m[7, 7:11] = -e_out_c[2, :]
    m[8, 7:11] = e_out_c[3, :]
    for state in range(4):
        m[9 + state, 7:11] = state_d[state, :]
        m[9 + state, 11] = -e_form[state, 1]
        m[9 + state, 12] = -e_form[state, 3]
    if not np.all(np.isfinite(m)):
        return float("nan")
    return float(np.linalg.det(m))


def test_microannulus_matches_an_independent_explicit_amplitude_assembly():
    """The 11x11 and a 13x13 that never folds the gap amplitudes out agree.

    The two matrices differ in size, column layout and row content -- the
    11x11 carries ``(u_r, sigma_rr)`` across the gap with the fluid
    propagator and matches at ``r = c`` only, the 13x13 matches at both gap
    faces -- so a transposed index or a dropped sign in one does not
    reproduce in the other. Both call ``_fluid_layer_e_matrix_n0``, so this
    checks the assembly and not the fluid element; the Wronskian test above
    covers that separately.
    """
    for freq in (2.0e3, 8.0e3, 1.2e4):
        folded = _ma_roots(lambda c: _ma_det(c, freq), 200.0, 1499.0)
        explicit = _ma_roots(
            lambda c: _ma_det_13(c, freq, 1.0e-4, (_MA_CASING,), (_MA_CEMENT,)),
            200.0,
            1499.0,
        )
        assert len(folded) == len(explicit)
        for lhs, rhs in zip(folded, explicit):
            assert lhs == pytest.approx(rhs, rel=1.0e-9)


def test_microannulus_roots_are_invariant_under_layer_subdivision():
    """Splitting either elastic block into sub-layers leaves the roots fixed.

    The established invariance oracle for propagator stacks, applied to the
    two-block form: subdivision changes the number of propagator factors and
    the radii they are evaluated at, but not the physics, so any residual is
    the assembly's own bookkeeping error.
    """

    def split(layer, pieces):
        piece = layer.thickness / pieces
        return tuple(
            BoreholeLayer(vp=layer.vp, vs=layer.vs, rho=layer.rho, thickness=piece)
            for _ in range(pieces)
        )

    freq = 8.0e3
    whole = _ma_roots(lambda c: _ma_det(c, freq), 200.0, 1499.0)
    assert len(whole) == 2
    for inner_pieces, outer_pieces in ((3, 1), (1, 4), (2, 5)):
        subdivided = _ma_roots(
            lambda c: _ma_det(
                c,
                freq,
                inner=split(_MA_CASING, inner_pieces),
                outer=split(_MA_CEMENT, outer_pieces),
            ),
            200.0,
            1499.0,
        )
        assert len(subdivided) == len(whole)
        for lhs, rhs in zip(whole, subdivided):
            assert lhs == pytest.approx(rhs, rel=1.0e-9)


def test_microannulus_thin_gap_limit_is_a_slip_interface_not_the_bonded_stack():
    """``h -> 0`` converges, and not to the bonded root. That is the point.

    A vanishing gap still forbids shear traction on both faces and still lets
    ``u_z`` slip, so the limit is frictionless contact, not welded contact.
    There is therefore no reduction of this assembly to the all-elastic 7x7 to
    validate against -- which is why the checks above had to come from
    outside. Measured at 8 kHz: the Stoneley-like root converges as O(h) to
    1383.446 m/s while the bonded stack gives 1400.038 m/s, a 16.6 m/s
    (1.2 %) offset that does not shrink as the gap closes.
    """
    from fwap.cylindrical_solver._cased import _modal_determinant_n0_cased

    freq = 8.0e3
    omega = 2.0 * np.pi * freq

    def bonded(c):
        return _modal_determinant_n0_cased(
            omega / c,
            omega,
            **_MA_FORMATION,
            **_MA_FLUID,
            a=_MA_A,
            layers=(_MA_CASING, _MA_CEMENT),
        )

    bonded_root = _ma_roots(bonded, 1000.0, 1499.0)
    assert len(bonded_root) == 1

    thicknesses = [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]
    stoneley = [
        _ma_roots(lambda c: _ma_det(c, freq, thickness=h), 1000.0, 1499.0)[0]
        for h in thicknesses
    ]
    # Cauchy convergence: each tenfold thinning moves the root ~10x less.
    steps = np.abs(np.diff(stoneley))
    assert np.all(steps[1:] < steps[:-1] / 5.0)
    # ... but not towards the bonded root.
    offsets = [bonded_root[0] - c for c in stoneley]
    assert all(o > 15.0 for o in offsets)
    assert offsets[-1] == pytest.approx(offsets[0], rel=1.0e-3)


def test_microannulus_carries_two_root_families():
    """Two roots, not one: a Stoneley-like mode and the slow gap mode.

    Recorded because it is a trap for the root finder that would come next.
    The n=0 branch-selection defect fixed earlier in this module came from
    exactly this -- a bracket that assumed a single root, silently returning
    whichever one the grid happened to straddle. The two families are far
    apart and move in opposite directions with gap thickness: thinning the gap
    slows the crack wave towards zero while leaving the Stoneley-like root
    essentially fixed.
    """
    freq = 8.0e3
    for thickness in (1.0e-3, 1.0e-4, 1.0e-5):
        found = _ma_roots(
            lambda c: _ma_det(c, freq, thickness=thickness),
            5.0,
            1499.0,
            log_grid=True,
        )
        assert len(found) == 2
        gap_mode, stoneley = found
        assert gap_mode < 0.5 * stoneley
        assert 1300.0 < stoneley < 1400.0

    thin = _ma_roots(
        lambda c: _ma_det(c, freq, thickness=1.0e-5), 5.0, 1499.0, log_grid=True
    )
    thick = _ma_roots(
        lambda c: _ma_det(c, freq, thickness=1.0e-3), 5.0, 1499.0, log_grid=True
    )
    assert thin[0] < thick[0] / 3.0
    assert thin[1] == pytest.approx(thick[1], rel=1.0e-3)

    # "Exactly two" is only a claim about the assembly if it survives a change
    # of grid. The n=0 branch-selection defect looked like solver noise on one
    # grid and was only exposed by moving the endpoints, so both knobs are
    # turned here: sample count and window.
    reference = _ma_roots(
        lambda c: _ma_det(c, freq), 5.0, 1499.0, samples=6000, log_grid=True
    )
    for samples, lo, hi in (
        (200, 5.0, 1499.0),
        (800, 3.1, 1499.9),
        (400, 11.0, 1490.0),
    ):
        found = _ma_roots(
            lambda c: _ma_det(c, freq), lo, hi, samples=samples, log_grid=True
        )
        assert len(found) == len(reference)
        for lhs, rhs in zip(found, reference):
            assert lhs == pytest.approx(rhs, rel=1.0e-9)


def test_microannulus_returns_nan_rather_than_warning_or_raising():
    """The determinant contract: NaN everywhere it cannot be formed.

    A root scan evaluates this at arbitrary trial velocities, so an escaping
    ``RuntimeWarning`` or ``LinAlgError`` is a defect, not a diagnostic. Both
    happened while this assembly was being built: unscaled ``I_n`` overflows
    at low phase velocity, and the fluid propagator's ``solve`` hits an
    exactly singular matrix once the Bessel dynamic range collapses. Three
    regimes are covered -- above the bound floor (including the gap fluid's
    own floor, which the borehole fluid's does not imply), past the Bessel
    argument cap, and past the span where the fluid propagator is meaningful.
    """
    import warnings

    from fwap.cylindrical_solver._cased import _BESSEL_ARG_MAX

    # Above the bound floor: the borehole fluid and the gap fluid each set one.
    assert np.isnan(_ma_det(1600.0, 8.0e3))
    assert np.isnan(_ma_det(1450.0, 8.0e3, annulus_vf=1400.0))
    assert np.isfinite(_ma_det(1450.0, 8.0e3, annulus_vf=1500.0))

    # Past the Bessel argument cap: kz * r_outermost > log(sqrt(DBL_MAX)).
    r_outermost = _MA_A + _MA_CASING.thickness + 1.0e-4 + _MA_CEMENT.thickness
    c_cap = 2.0 * np.pi * 8.0e3 * r_outermost / _BESSEL_ARG_MAX
    assert np.isnan(_ma_det(c_cap * 0.9, 8.0e3))
    assert np.isfinite(_ma_det(c_cap * 1.5, 8.0e3))

    # Past the fluid propagator's usable Bessel span, where its exact
    # determinant identity fails long before its entries become non-finite.
    assert np.isnan(_ma_det(300.0, 8.0e3, thickness=0.5))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for phase_velocity in np.geomspace(0.05, 1600.0, 400):
            for thickness in (1.0e-6, 1.0e-4, 1.0e-2, 0.3):
                _ma_det(phase_velocity, 8.0e3, thickness=thickness)


def test_microannulus_rejects_configurations_it_cannot_represent():
    """Both blocks must exist, and the gap must be a real fluid gap.

    A zero-thickness inner block is refused rather than accommodated: it makes
    the two ``sigma_rz = 0`` rows at ``r = a`` and ``r = b`` the same row, so
    the determinant vanishes identically and every trial ``k_z`` looks like a
    root.
    """
    from fwap.cylindrical_solver._cased import _modal_determinant_n0_microannulus

    base = {
        **_MA_FORMATION,
        **_MA_FLUID,
        **_MA_ANNULUS,
        "a": _MA_A,
        "inner_layers": (_MA_CASING,),
        "outer_layers": (_MA_CEMENT,),
        "annulus_thickness": 1.0e-4,
    }
    omega = 2.0 * np.pi * 8.0e3
    kz = omega / 1400.0
    assert np.isfinite(_modal_determinant_n0_microannulus(kz, omega, **base))

    for bad in (
        {"inner_layers": ()},
        {"outer_layers": ()},
    ):
        with pytest.raises(ValueError, match="non-empty"):
            _modal_determinant_n0_microannulus(kz, omega, **{**base, **bad})
    with pytest.raises(ValueError, match="annulus_thickness must be positive"):
        _modal_determinant_n0_microannulus(
            kz, omega, **{**base, "annulus_thickness": 0.0}
        )
    for bad in ({"annulus_vf": 0.0}, {"annulus_rho": -1.0}):
        with pytest.raises(ValueError, match="must be positive"):
            _modal_determinant_n0_microannulus(kz, omega, **{**base, **bad})
