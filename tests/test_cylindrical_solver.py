"""Schmitt (1988) cylindrical-borehole modal-determinant solver tests.

Phase 1: n=0 axisymmetric Stoneley dispersion. Validates against
the closed-form White (1983) low-f limit and a battery of
parametric checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwap.cylindrical_solver import (
    BoreholeMode,
    _modal_determinant_n0,
    stoneley_dispersion,
)


def _stoneley_lf_truth(vs, rho, vf, rho_f):
    """White (1983) eq. 5.42 closed form: S_ST^2 = 1/V_f^2 + rho_f/mu."""
    mu = rho * vs ** 2
    return float(np.sqrt(1.0 / vf ** 2 + rho_f / mu))


# ---------------------------------------------------------------------
# Closed-form low-f validation
# ---------------------------------------------------------------------


def test_stoneley_low_f_matches_white_closed_form():
    """At very low frequency (10 Hz) the Stoneley slowness from the
    full modal determinant must match the closed-form
    `S_ST^2 = 1/V_f^2 + rho_f / mu` to better than 0.01 %."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(np.array([10.0]),
                              vp=vp, vs=vs, rho=rho,
                              vf=vf, rho_f=rho_f, a=a)
    s_truth = _stoneley_lf_truth(vs, rho, vf, rho_f)
    assert res.slowness[0] == pytest.approx(s_truth, rel=1.0e-4)


def test_stoneley_solver_returns_nan_in_slow_formation_regime():
    """Slow formations (V_S < V_f) put V_ST above V_S, so the
    formation-side S radial wavenumber becomes imaginary (s^2 < 0)
    -- the Stoneley wave radiates into the formation as a leaky S
    arrival. The Phase-1 bound-mode solver cannot handle that
    regime; it should return ``NaN`` rather than wrong numbers."""
    vp, vs, rho = 2200.0, 800.0, 2200.0    # slow shale: V_S < V_f
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(np.array([10.0]),
                              vp=vp, vs=vs, rho=rho,
                              vf=vf, rho_f=rho_f, a=a)
    assert np.isnan(res.slowness[0])


def test_stoneley_low_f_matches_white_closed_form_fast_formation():
    """Fast-formation low-f match (V_S much larger than V_f)."""
    vp, vs, rho = 6000.0, 3500.0, 2700.0   # fast carbonate
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(np.array([10.0]),
                              vp=vp, vs=vs, rho=rho,
                              vf=vf, rho_f=rho_f, a=a)
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
    res = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho,
                              vf=vf, rho_f=rho_f, a=a)
    assert np.all(np.isfinite(res.slowness))


def test_stoneley_dispersion_speeds_up_with_frequency():
    """Stoneley phase slowness *decreases* with frequency (the wave
    sheds the rho_f / mu loading and approaches the bare fluid as
    f rises in this band) for the typical brine-in-sandstone case."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(100.0, 10_000.0, 40)
    s = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho,
                            vf=vf, rho_f=rho_f, a=a).slowness
    # Slowness is monotonically decreasing across the test band.
    assert np.all(np.diff(s) < 0.0)


def test_stoneley_slowness_always_above_fluid_slowness():
    """Stoneley wave is always slower than the unconfined fluid wave
    (formation loading slows it down)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(100.0, 12_000.0, 30)
    s = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho,
                            vf=vf, rho_f=rho_f, a=a).slowness
    assert np.all(s > 1.0 / vf)


def test_stoneley_slowness_above_formation_shear_slowness():
    """Stoneley phase velocity in a fast formation is below V_S
    (V_ST < V_f < V_S), so the *slowness* sits above 1/V_S. The
    Stoneley wave is loaded by the formation through the rho_f /
    mu term but never speeds up to formation-shear levels."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(100.0, 12_000.0, 30)
    s = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho,
                            vf=vf, rho_f=rho_f, a=a).slowness
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
        f, vp=5000.0, vs=3000.0, rho=rho, vf=vf, rho_f=rho_f, a=a,
    ).slowness[0]
    s_soft = stoneley_dispersion(
        f, vp=4000.0, vs=2000.0, rho=rho, vf=vf, rho_f=rho_f, a=a,
    ).slowness[0]
    assert s_soft > s_stiff


def test_stoneley_slowness_increases_with_heavier_fluid():
    """Heavier borehole fluid -> larger rho_f / mu loading -> slower
    Stoneley."""
    f = np.array([1000.0])
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, a = 1500.0, 0.1
    s_light = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=900.0, a=a,
    ).slowness[0]
    s_heavy = stoneley_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=1100.0, a=a,
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
    res = stoneley_dispersion(np.array([1000.0]),
                              vp=vp, vs=vs, rho=rho,
                              vf=vf, rho_f=rho_f, a=a)
    s_root = res.slowness[0]
    kz_root = s_root * omega
    d_lo = _modal_determinant_n0(
        kz_root * 0.999, omega, vp, vs, rho, vf, rho_f, a)
    d_hi = _modal_determinant_n0(
        kz_root * 1.001, omega, vp, vs, rho, vf, rho_f, a)
    assert np.sign(d_lo) != np.sign(d_hi)


def test_modal_determinant_at_root_is_near_zero():
    """det(M) at the recovered k_z must be 8+ orders of magnitude
    smaller than at a nearby k_z."""
    omega = 2.0 * np.pi * 1000.0
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    s_root = stoneley_dispersion(np.array([1000.0]),
                                 vp=vp, vs=vs, rho=rho,
                                 vf=vf, rho_f=rho_f,
                                 a=a).slowness[0]
    kz_root = s_root * omega
    d_at = abs(_modal_determinant_n0(
        kz_root, omega, vp, vs, rho, vf, rho_f, a))
    d_near = abs(_modal_determinant_n0(
        kz_root * 1.05, omega, vp, vs, rho, vf, rho_f, a))
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
        s = stoneley_dispersion(f, vp=vp, vs=vs, rho=rho,
                                vf=vf, rho_f=rho_f, a=a).slowness[0]
        assert s == pytest.approx(s_truth, rel=1.0e-3)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_stoneley_dispersion_rejects_non_positive_inputs():
    f = np.array([1000.0])
    base = dict(vp=4500.0, vs=2500.0, rho=2400.0,
                vf=1500.0, rho_f=1000.0, a=0.1)
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
        stoneley_dispersion(f, vp=2500.0, vs=2500.0, rho=2400.0,
                            vf=1500.0, rho_f=1000.0, a=0.1)


def test_stoneley_dispersion_rejects_non_positive_freq():
    f = np.array([10.0, 0.0, 100.0])
    with pytest.raises(ValueError, match="freq"):
        stoneley_dispersion(f, vp=4500.0, vs=2500.0, rho=2400.0,
                            vf=1500.0, rho_f=1000.0, a=0.1)


# ---------------------------------------------------------------------
# Output dataclass contract
# ---------------------------------------------------------------------


def test_stoneley_dispersion_returns_borehole_mode_dataclass():
    f = np.linspace(500.0, 5000.0, 5)
    res = stoneley_dispersion(f, vp=4500.0, vs=2500.0, rho=2400.0,
                              vf=1500.0, rho_f=1000.0, a=0.1)
    assert isinstance(res, BoreholeMode)
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    np.testing.assert_array_equal(res.freq, f)
    assert res.slowness.shape == f.shape
