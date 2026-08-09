"""
Tests for the microannulus gap-width benchmark harness (G.6).

Bundles are **planted** rather than generated, for the same reason
``test_baselines_debond.py`` plants its own: a debonded sample costs ~14 s of
modal root-finding. Planting an exact Krauklis curve also means the harness
can be checked against arithmetic that is right to floating point, rather than
to solver tolerance.

Everything here is torch-free. That is a property worth keeping: the classical
predictor and the harness must be usable in an environment with no ML stack,
which is why :class:`~sonic_ml.bench.debond.KrauklisThicknessPredictor` lives
in ``bench`` and the learned adapter does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml.baselines.debond import (
    CrackWaveThicknessBaseline,
    gap_thickness,
    solid_compliance,
)
from sonic_ml.bench import (
    GAP_REGIME_EDGE_M,
    GAP_ROW_ORDER,
    KrauklisThicknessPredictor,
    MeanThicknessPredictor,
    Scorecard,
    ThicknessPredictor,
    decades_to_percent,
    evaluate_thickness,
    format_thickness_scorecard,
    gap_regime_labels,
)
from sonic_ml.loader import DatasetBundle

_CASING = (5850.0, 3140.0, 7800.0, 0.010)
_CEMENT = (2300.0, 1725.0, 1850.0, 0.045)
_RHO_F = 1000.0
_COMPLIANCE = solid_compliance(*_CASING[:3]) + solid_compliance(*_CEMENT[:3])


def _planted_bundle(gaps, *, bias=1.0, n_freq=32):
    """A debonded bundle whose crack-wave rows follow the Krauklis law.

    ``bias`` scales the velocity, so the closed-form estimate comes out wrong
    by a known constant factor -- which is how a test can assert an exact
    error rather than a tolerance.
    """
    gaps = np.asarray(gaps, dtype=float)
    n = gaps.size
    freq = np.linspace(1000.0, 12000.0, n_freq)
    omega = 2.0 * np.pi * freq

    layer_params = np.empty((n, 3, 4), dtype=float)
    layer_params[:, 0] = _CASING
    layer_params[:, 1] = (1500.0, 0.0, _RHO_F, 0.0)
    layer_params[:, 2] = _CEMENT
    layer_params[:, 1, 3] = gaps

    slowness = np.empty((n, 2, n_freq), dtype=float)
    slowness[:, 0] = 1.0 / 1400.0
    for i, h in enumerate(gaps):
        velocity = (omega * h / (_COMPLIANCE * _RHO_F)) ** (1.0 / 3.0) * bias
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


def _log_gaps(n, seed=0):
    rng = np.random.default_rng(seed)
    return 10.0 ** rng.uniform(np.log10(1e-5), np.log10(1e-3), size=n)


# --------------------------------------------------------------------------
# regimes
# --------------------------------------------------------------------------


def test_regime_edge_is_fixed_not_data_dependent():
    """The property that lets two scorecards be compared.

    A median split would put the same gap in different regimes depending on
    what else was in the dataset. Both bundles here have wildly different
    distributions and the same gap must land in the same row.
    """
    narrow = _planted_bundle([1.1e-4, 1.2e-4, 1.3e-4, 5.0e-4])
    wide = _planted_bundle([1.1e-4, 9.0e-4, 9.5e-4, 9.9e-4])
    assert gap_regime_labels(narrow)[0] == gap_regime_labels(wide)[0] == 1


def test_regimes_split_at_the_documented_edge():
    just_under = GAP_REGIME_EDGE_M * 0.999
    just_over = GAP_REGIME_EDGE_M * 1.001
    labels = gap_regime_labels(_planted_bundle([just_under, just_over]))
    assert labels.tolist() == [0, 1]


def test_regime_labels_reject_a_bundle_with_no_gap_layer():
    bundle = _planted_bundle([1e-4])
    object.__setattr__(bundle, "layer_names", ("casing", "cement", "formation"))
    with pytest.raises(ValueError, match="--debonded"):
        gap_regime_labels(bundle)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_exact_predictor_scores_zero_error():
    bundle = _planted_bundle(_log_gaps(24))
    sc = evaluate_thickness(KrauklisThicknessPredictor(), bundle, n_boot=100)
    assert isinstance(sc, Scorecard)
    assert sc.parameter == "log10_thickness"
    assert sc.predictor == "krauklis_closed_form"
    assert sc.per_regime["all"].median_abs_error == pytest.approx(0.0, abs=1e-12)
    assert sc.per_regime["all"].n_finite == 24


def test_a_known_bias_is_reported_as_the_right_number_of_decades():
    """The units check, and the reason it is worth having.

    Scaling the crack-wave velocity by ``b`` scales the estimated gap by
    ``b**3`` -- the law is cubic. So the medAE must be exactly
    ``3*log10(b)`` decades, and no other convention gives that.
    """
    bias = 1.2
    bundle = _planted_bundle(_log_gaps(32), bias=bias)
    sc = evaluate_thickness(KrauklisThicknessPredictor(), bundle, n_boot=100)
    expected = 3.0 * np.log10(bias)
    assert sc.per_regime["all"].median_abs_error == pytest.approx(expected, rel=1e-9)
    assert decades_to_percent(expected) == pytest.approx(
        100.0 * (bias**3 - 1.0), rel=1e-9
    )


def test_regimes_partition_the_scored_samples():
    bundle = _planted_bundle(_log_gaps(40, seed=1))
    sc = evaluate_thickness(KrauklisThicknessPredictor(), bundle, n_boot=50)
    rows = [r for r in ("tight", "wide") if r in sc.per_regime]
    assert sum(sc.per_regime[r].n for r in rows) == sc.per_regime["all"].n == 40


def test_indices_score_a_held_out_subset_only():
    bundle = _planted_bundle(_log_gaps(30, seed=2))
    idx = np.arange(0, 30, 3)
    sc = evaluate_thickness(
        KrauklisThicknessPredictor(), bundle, indices=idx, n_boot=50
    )
    assert sc.per_regime["all"].n == idx.size


def test_subset_scores_match_the_matching_rows_of_a_full_pass():
    """A stateless predictor must not care whether it was asked for a subset.

    This is what makes a test-split scorecard trustworthy: it has to be the
    same arithmetic as the full pass, restricted, not a different one.
    """
    bundle = _planted_bundle(_log_gaps(24, seed=4), bias=1.15)
    idx = np.arange(0, 24, 2)
    pred = KrauklisThicknessPredictor()
    full = pred.predict_log_thickness(bundle)
    assert np.array_equal(pred.predict_log_thickness(bundle, idx), full[idx])


def test_bootstrap_ci_is_reproducible_and_brackets_the_median():
    bundle = _planted_bundle(_log_gaps(48, seed=3), bias=1.3)
    a = evaluate_thickness(KrauklisThicknessPredictor(), bundle, seed=7, n_boot=200)
    b = evaluate_thickness(KrauklisThicknessPredictor(), bundle, seed=7, n_boot=200)
    for name in a.per_regime:
        assert a.per_regime[name].ci_low == b.per_regime[name].ci_low
        assert a.per_regime[name].ci_high == b.per_regime[name].ci_high
    row = a.per_regime["all"]
    assert row.ci_low <= row.median_abs_error <= row.ci_high


def test_nan_predictions_are_counted_not_crashed():
    bundle = _planted_bundle(_log_gaps(12, seed=6))
    bundle.slowness[:6, 1] = np.nan
    sc = evaluate_thickness(KrauklisThicknessPredictor(), bundle, n_boot=50)
    assert sc.per_regime["all"].n == 12
    assert sc.per_regime["all"].n_finite == 6


def test_predictor_that_ignores_indices_is_rejected():
    """A silent shape mismatch here would score the wrong samples.

    Returning one estimate per *bundle* sample when a subset was asked for
    would line predictions up against the wrong truths and still produce a
    plausible-looking scorecard, so it has to be an error rather than
    broadcasting.
    """

    class IgnoresIndices:
        name = "sloppy"

        def predict_log_thickness(self, bundle, indices=None):
            return np.zeros(bundle.n_samples)

    bundle = _planted_bundle(_log_gaps(10, seed=8))
    with pytest.raises(ValueError, match="honour"):
        evaluate_thickness(IgnoresIndices(), bundle, indices=np.arange(4), n_boot=20)


def test_evaluate_rejects_a_bundle_that_is_not_debonded():
    bundle = _planted_bundle(_log_gaps(4))
    object.__setattr__(bundle, "layer_names", ("casing", "cement", "formation"))
    with pytest.raises(ValueError, match="--debonded"):
        evaluate_thickness(KrauklisThicknessPredictor(), bundle, n_boot=20)


# --------------------------------------------------------------------------
# the no-skill reference
# --------------------------------------------------------------------------


def test_mean_predictor_scores_about_half_a_decade_on_a_log_uniform_prior():
    """The yardstick that makes the other numbers mean something.

    Against a log-uniform prior two decades wide, always answering the mean
    is off by half a decade in the median -- ~216 %. Anything quoted as a
    skill number has to be read against this, not against zero.
    """
    bundle = _planted_bundle(_log_gaps(200, seed=9))
    train = np.arange(0, 200, 2)
    pred = MeanThicknessPredictor.from_train_split(bundle, train)
    sc = evaluate_thickness(pred, bundle, indices=np.arange(1, 200, 2), n_boot=200)
    assert sc.predictor == "mean_thickness"
    assert 0.4 < sc.per_regime["all"].median_abs_error < 0.6
    assert decades_to_percent(sc.per_regime["all"].median_abs_error) > 100.0


def test_mean_predictor_uses_only_the_split_it_was_given():
    """A "no skill" reference that peeked at the test split would have some."""
    bundle = _planted_bundle(np.array([1e-5] * 10 + [1e-3] * 10))
    low = MeanThicknessPredictor.from_train_split(bundle, np.arange(10))
    high = MeanThicknessPredictor.from_train_split(bundle, np.arange(10, 20))
    assert low.value == pytest.approx(-5.0)
    assert high.value == pytest.approx(-3.0)


def test_both_predictors_satisfy_the_protocol():
    assert isinstance(KrauklisThicknessPredictor(), ThicknessPredictor)
    assert isinstance(MeanThicknessPredictor(-4.0), ThicknessPredictor)


def test_the_wrapped_baseline_is_the_one_that_was_passed_in():
    baseline = CrackWaveThicknessBaseline(gap_name="microannulus")
    pred = KrauklisThicknessPredictor(baseline, name="custom")
    assert pred.baseline is baseline
    assert pred.name == "custom"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_report_shows_decades_and_percent_and_names_its_units():
    bundle = _planted_bundle(_log_gaps(30, seed=11), bias=1.2)
    sc = evaluate_thickness(KrauklisThicknessPredictor(), bundle, n_boot=100)
    text = format_thickness_scorecard(sc)
    lines = text.splitlines()

    assert "errors in decades" in lines[0]
    assert "medAE %" in lines[1]
    printed = {line.split()[0] for line in lines[3:]}
    assert printed <= set(GAP_ROW_ORDER) and "all" in printed
    # The percentage column carries the number worth quoting: 1.2**3 - 1.
    assert "72.8" in text


def test_report_skips_rows_the_scorecard_does_not_have():
    bundle = _planted_bundle([2e-4, 3e-4, 5e-4])  # all "wide"
    sc = evaluate_thickness(KrauklisThicknessPredictor(), bundle, n_boot=20)
    text = format_thickness_scorecard(sc)
    assert "tight" not in text
    assert "wide" in text


# --------------------------------------------------------------------------
# the shared truth reader
# --------------------------------------------------------------------------


def test_truth_reader_agrees_with_the_baselines_own():
    bundle = _planted_bundle(_log_gaps(8, seed=12))
    assert np.array_equal(
        gap_thickness(bundle), CrackWaveThicknessBaseline().truth(bundle)
    )
