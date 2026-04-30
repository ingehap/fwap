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
    BoreholeMode,
    _modal_determinant_n0,
    _modal_determinant_n1,
    flexural_dispersion,
    stoneley_dispersion,
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


def test_flexural_returns_nan_in_fast_formation():
    """Fast formations (V_S > V_f) put the flexural mode above
    the fluid speed, making F^2 < 0 and the bound-mode solver
    inapplicable. ``flexural_dispersion`` returns NaN throughout
    rather than raising; the leaky-flexural regime is a roadmap
    follow-up."""
    f = np.array([2000.0, 5000.0, 10000.0])
    res = flexural_dispersion(
        f, vp=4500.0, vs=2500.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1
    )
    assert np.all(np.isnan(res.slowness))


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

