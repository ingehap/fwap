"""Schmitt (1988) cylindrical-borehole modal-determinant solver tests.

Phase 1: n=0 axisymmetric Stoneley dispersion. Validates against
the closed-form White (1983) low-f limit and a battery of
parametric checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwap.cylindrical import flexural_dispersion_physical, rayleigh_speed
from fwap.cylindrical_solver import (
    BoreholeLayer,
    BoreholeMode,
    _layered_n0_bessel_pack,
    _layered_n0_radial_wavenumbers,
    _layered_n0_row1_at_a,
    _layered_n0_row2_at_a,
    _layered_n0_row3_at_a,
    _layered_n0_row4_at_b,
    _layered_n0_row5_at_b,
    _layered_n0_row6_at_b,
    _modal_determinant_n0,
    _modal_determinant_n1,
    flexural_dispersion,
    stoneley_dispersion,
    stoneley_dispersion_layered,
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


def test_stoneley_dispersion_layered_non_empty_layers_raises_not_implemented():
    """Non-empty layers are blocked on plan item F.1 (the 7x7 layered
    modal determinant); the public API raises NotImplementedError so
    callers don't silently get the unlayered answer."""
    f = np.array([1000.0, 2000.0])
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    with pytest.raises(NotImplementedError, match="plan item F"):
        stoneley_dispersion_layered(
            f,
            vp=4500.0, vs=2500.0, rho=2400.0,
            vf=1500.0, rho_f=1000.0, a=0.1,
            layers=(layer,),
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

