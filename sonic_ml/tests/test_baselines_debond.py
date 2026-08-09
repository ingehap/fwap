"""
Tests for the closed-form microannulus-thickness baseline (G.2).

The bundles here are **planted**, not generated: a debonded sample costs
~14 s of modal root-finding, so a fixture of any useful size would dominate
the suite. Planting an exact Krauklis curve also makes the arithmetic
testable to floating-point rather than to solver tolerance, which is what
these tests are for. The estimator's behaviour against real solver output
is a measurement rather than an assertion, and lives in `CHANGELOG.md` and
`plans/roadmap.md` G.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml.baselines.debond import (
    CrackWaveThicknessBaseline,
    crack_compliance,
    krauklis_thickness,
    solid_compliance,
)
from sonic_ml.loader import DatasetBundle

# Steel casing and a typical set cement, as MicroannulusPriors draws them.
_CASING = (5850.0, 3140.0, 7800.0, 0.010)
_CEMENT = (2300.0, 1725.0, 1850.0, 0.045)
_RHO_F = 1000.0


def _planted_bundle(gaps, freq=None, *, n_freq=32):
    """A debonded bundle whose crack-wave rows follow the Krauklis law exactly."""
    freq = np.linspace(1000.0, 12000.0, n_freq) if freq is None else freq
    gaps = np.asarray(gaps, dtype=float)
    n = gaps.size

    layer_params = np.empty((n, 3, 4), dtype=float)
    layer_params[:, 0] = _CASING
    layer_params[:, 1] = (1500.0, 0.0, _RHO_F, 0.0)
    layer_params[:, 2] = _CEMENT
    layer_params[:, 1, 3] = gaps

    compliance = solid_compliance(*_CASING[:3]) + solid_compliance(*_CEMENT[:3])
    omega = 2.0 * np.pi * freq
    slowness = np.empty((n, 2, freq.size), dtype=float)
    slowness[:, 0] = 1.0 / 1400.0  # a stand-in Stoneley row
    for i, h in enumerate(gaps):
        velocity = (omega * h / (compliance * _RHO_F)) ** (1.0 / 3.0)
        slowness[i, 1] = 1.0 / velocity

    params = np.tile(np.array([4000.0, 2200.0, 2400.0, 1500.0, _RHO_F, 0.1]), (n, 1))
    return DatasetBundle(
        params=params,
        slowness=slowness,
        gather=np.zeros((n, 8, 64)),
        mode_in_gather=np.tile(np.array([True, False]), (n, 1)),
        freq=freq,
        param_names=("vp", "vs", "rho", "vf", "rho_f", "a"),
        mode_names=("Stoneley", "crack_wave"),
        schema_version=4,
        layer_params=layer_params,
        layer_names=("casing", "microannulus", "cement"),
        bond_index=np.linspace(1.0, 0.0, n),
    )


def test_solid_compliance_matches_the_closed_form():
    vp, vs, rho = _CEMENT[:3]
    mu = rho * vs**2
    nu = (vp**2 - 2 * vs**2) / (2 * (vp**2 - vs**2))
    assert solid_compliance(vp, vs, rho) == pytest.approx((1 - nu) / mu, rel=1e-12)


def test_solid_compliance_of_a_fluid_is_infinite():
    """A fluid has no shear stiffness, so it offers unbounded compliance.

    Returning ``inf`` rather than raising means a mis-specified stack gives a
    ridiculous gap estimate instead of a plausible one -- the failure stays
    visible.
    """
    assert solid_compliance(1500.0, 0.0, 1000.0) == float("inf")


def test_solid_compliance_rejects_an_impossible_solid():
    with pytest.raises(ValueError, match="vp must exceed vs"):
        solid_compliance(2000.0, 2500.0, 2000.0)


def test_crack_compliance_sums_both_bounding_solids():
    bundle = _planted_bundle([1e-4])
    total = crack_compliance(bundle.layer_params[0], bundle.layer_names)
    expected = solid_compliance(*_CASING[:3]) + solid_compliance(*_CEMENT[:3])
    assert total == pytest.approx(expected, rel=1e-12)


def test_crack_compliance_needs_a_gap_with_two_neighbours():
    params = np.array([[*_CASING], [1500.0, 0.0, 1000.0, 1e-4]])
    with pytest.raises(ValueError, match="edge of the stack"):
        crack_compliance(params, ("casing", "microannulus"))
    with pytest.raises(ValueError, match="not a debonded stack"):
        crack_compliance(params, ("casing", "cement"))


def test_krauklis_inversion_recovers_a_planted_gap_exactly():
    """The law inverted against itself: the point is that it is closed form.

    Every frequency must return the same thickness, because a true Krauklis
    crack has no frequency-dependent residual. Spread across the band is
    exactly the diagnostic the estimator reports on real curves.
    """
    freq = np.linspace(1000.0, 12000.0, 32)
    compliance = solid_compliance(*_CASING[:3]) + solid_compliance(*_CEMENT[:3])
    h = 1.37e-4
    velocity = (2 * np.pi * freq * h / (compliance * _RHO_F)) ** (1.0 / 3.0)

    estimates = krauklis_thickness(freq, velocity, rho_f=_RHO_F, compliance=compliance)
    assert np.allclose(estimates, h, rtol=1e-12)


def test_baseline_recovers_gaps_across_two_decades():
    gaps = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    bundle = _planted_bundle(gaps)
    baseline = CrackWaveThicknessBaseline()

    assert np.allclose(baseline.predict(bundle), gaps, rtol=1e-10)
    assert np.allclose(baseline.truth(bundle), gaps, rtol=1e-12)

    score = baseline.score(bundle)
    assert score["median_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert score["log_rmse"] < 1e-9
    assert score["rank_correlation"] == pytest.approx(1.0)
    assert score["n_scored"] == gaps.size


def test_baseline_needs_no_fit():
    """The distinguishing property against the bonded bond baseline.

    `StoneleyBondBaseline` spends the training split on a calibration; this
    one has none to spend, so it is a stricter bar rather than a laxer one.
    """
    assert not hasattr(CrackWaveThicknessBaseline(), "fit")


def test_baseline_reports_a_bias_rather_than_absorbing_it():
    """A systematically wrong compliance must show up as a ratio, not vanish.

    Scoring in the ratio domain is what separates "off by a constant factor"
    -- which a recalibration would fix -- from genuine scatter, which it
    would not. Doubling every gap leaves the ranking perfect and the bias
    obvious.
    """
    gaps = np.array([1e-5, 1e-4, 1e-3])
    bundle = _planted_bundle(gaps)
    bundle.layer_params[:, 1, 3] = gaps / 2.0  # truth halved; curves unchanged

    score = CrackWaveThicknessBaseline().score(bundle)
    assert score["median_ratio"] == pytest.approx(2.0, rel=1e-9)
    assert score["ratio_iqr"] < 1e-9, "a pure bias must leave no spread"
    assert score["rank_correlation"] == pytest.approx(1.0)


def test_baseline_survives_an_all_nan_curve():
    bundle = _planted_bundle([1e-4, 2e-4])
    bundle.slowness[0, 1] = np.nan

    baseline = CrackWaveThicknessBaseline()
    predicted = baseline.predict(bundle)
    assert not np.isfinite(predicted[0])
    assert np.isfinite(predicted[1])
    assert baseline.score(bundle)["n_scored"] == 1.0


def test_baseline_rejects_a_bundle_without_a_crack_wave_row():
    bundle = _planted_bundle([1e-4])
    object.__setattr__(bundle, "mode_names", ("Stoneley", "flexural"))
    with pytest.raises(ValueError, match="--debonded"):
        CrackWaveThicknessBaseline().predict(bundle)
