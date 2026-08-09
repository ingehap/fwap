"""
Tests for the learned microannulus-thickness inverse (G.2, torch-gated).

Bundles are planted rather than generated: a debonded sample costs ~14 s of
modal root-finding. The measured comparison against real solver output is a
measurement, not an assertion, and lives in `CHANGELOG.md` and
`plans/roadmap.md` G.2.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.baselines.debond import solid_compliance  # noqa: E402
from sonic_ml.loader import DatasetBundle  # noqa: E402
from sonic_ml.models.debond import (  # noqa: E402
    DEBOND_FEATURE_NAMES,
    debond_features,
    debond_scores,
    debond_targets,
    train_debond_inverse,
)

_CASING = (5850.0, 3140.0, 7800.0, 0.010)
_CEMENT = (2300.0, 1725.0, 1850.0, 0.045)
_RHO_F = 1000.0
_COMPLIANCE = solid_compliance(*_CASING[:3]) + solid_compliance(*_CEMENT[:3])


def _planted_bundle(gaps, *, n_freq=32, bias=1.0, drift=0.0, seed=0):
    """
    A debonded bundle with a controllable departure from the Krauklis law.

    ``bias`` scales the crack-wave velocity so the closed-form estimate comes
    out wrong by a known constant factor; ``drift`` tilts it with frequency,
    which is what a finite-layer wall does. Both give the residual model
    something real to learn, and give the test a known right answer.
    """
    rng = np.random.default_rng(seed)
    gaps = np.asarray(gaps, dtype=float)
    n = gaps.size
    freq = np.linspace(1000.0, 12000.0, n_freq)
    omega = 2.0 * np.pi * freq

    layer_params = np.empty((n, 3, 4), dtype=float)
    layer_params[:, 0] = _CASING
    layer_params[:, 1] = (1500.0, 0.0, _RHO_F, 0.0)
    layer_params[:, 2] = _CEMENT
    layer_params[:, 1, 3] = gaps
    # Vary the casing thickness: it is what the half-space law cannot see,
    # so it is the feature a residual model has to use.
    thickness = rng.uniform(0.008, 0.012, size=n)
    layer_params[:, 0, 3] = thickness

    slowness = np.empty((n, 2, n_freq), dtype=float)
    slowness[:, 0] = 1.0 / 1400.0
    tilt = 1.0 + drift * (np.log10(freq) - np.log10(freq).mean())
    for i, h in enumerate(gaps):
        # A thicker casing wall stiffens the crack, so the departure from the
        # half-space law is a deterministic function of a visible feature.
        scale = bias * (1.0 + 4.0 * (thickness[i] - 0.010))
        velocity = (omega * h / (_COMPLIANCE * _RHO_F)) ** (1.0 / 3.0) * scale * tilt
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


def test_features_never_carry_the_gap_thickness():
    """The leak guard, and the most important test in this file.

    The answer sits in `layer_params` one column away from features that are
    legitimately used. Perturbing the stored gap while holding the dispersion
    curve fixed must leave every feature untouched -- if it does not, the
    model is reading the label.
    """
    bundle = _planted_bundle(_log_gaps(16))
    before, names, baseline_before = debond_features(bundle)

    bundle.layer_params[:, 1, 3] *= 7.3  # move only the stored answer
    after, _, baseline_after = debond_features(bundle)

    assert np.array_equal(before, after)
    assert np.array_equal(baseline_before, baseline_after)
    assert "microannulus_thickness" not in names


def test_feature_names_line_up_with_the_matrix():
    bundle = _planted_bundle(_log_gaps(8))
    features, names, _ = debond_features(bundle)
    assert features.shape[1] == len(names)
    assert names[: len(DEBOND_FEATURE_NAMES)] == DEBOND_FEATURE_NAMES
    # The layer properties the baseline never sees are present ...
    assert "casing_thickness" in names
    assert "cement_thickness" in names
    # ... and every other gap column is too, only its thickness is gone.
    assert "microannulus_vp" in names


def test_features_reject_a_bundle_without_the_crack_wave():
    bundle = _planted_bundle(_log_gaps(4))
    object.__setattr__(bundle, "mode_names", ("Stoneley", "flexural"))
    with pytest.raises(ValueError, match="crack_wave"):
        debond_features(bundle)


def test_targets_are_log10_of_the_drawn_gap():
    gaps = _log_gaps(12)
    bundle = _planted_bundle(gaps)
    assert np.allclose(debond_targets(bundle), np.log10(gaps), rtol=1e-12)


def test_untrained_model_reproduces_the_classical_baseline_exactly():
    """Zero-initialised output head: the model starts *at* the baseline.

    That is what makes any later gain attributable rather than mysterious,
    and it means a broken training run degrades to the classical answer
    instead of to noise.
    """
    from sonic_ml.models.debond import DebondResidualNet, TrainedDebondInverse
    from sonic_ml.normalize import Standardizer

    bundle = _planted_bundle(_log_gaps(24), bias=1.4)
    features, names, baseline = debond_features(bundle)
    std = Standardizer.fit(features, names)
    trained = TrainedDebondInverse(
        DebondResidualNet(std.transform(features).shape[1]), std, names
    )
    assert np.allclose(trained.predict_log_thickness(bundle), baseline, atol=1e-6)


def test_residual_model_beats_the_closed_form_it_starts_from():
    """The point of the exercise: learn the correction the law cannot express.

    The planted curves depart from the Krauklis law by a factor that depends
    on casing thickness -- exactly the finite-layer effect a half-space law
    throws away, and exactly what the extra features expose. The model has to
    close most of that gap on *held-out* samples.
    """
    bundle = _planted_bundle(_log_gaps(240, seed=3), bias=1.25, drift=0.05, seed=3)
    trained, history = train_debond_inverse(bundle, epochs=250, seed=0)

    from sonic_ml.split import stratified_split

    split = stratified_split(bundle, seed=0)
    truth = debond_targets(bundle)
    _, _, baseline = debond_features(bundle)
    learned = trained.predict_log_thickness(bundle)

    test = split.test
    classical_score = debond_scores(baseline[test], truth[test])
    learned_score = debond_scores(learned[test], truth[test])

    assert history[-1][0] < history[0][0], "training did not reduce its own loss"
    assert learned_score["log_rmse"] < 0.5 * classical_score["log_rmse"], (
        f"learned {learned_score['log_rmse']:.4f} vs "
        f"classical {classical_score['log_rmse']:.4f}"
    )
    # And the bias the closed form carries is what got removed.
    assert abs(learned_score["median_ratio"] - 1.0) < abs(
        classical_score["median_ratio"] - 1.0
    )


def test_training_is_reproducible():
    bundle = _planted_bundle(_log_gaps(64, seed=5), bias=1.2, seed=5)
    a, _ = train_debond_inverse(bundle, epochs=30, seed=1)
    b, _ = train_debond_inverse(bundle, epochs=30, seed=1)
    assert np.allclose(a.predict_log_thickness(bundle), b.predict_log_thickness(bundle))


def test_scores_report_bias_and_spread_separately():
    truth = np.log10(np.array([1e-5, 1e-4, 1e-3]))
    doubled = truth + np.log10(2.0)
    score = debond_scores(doubled, truth)
    assert score["median_ratio"] == pytest.approx(2.0, rel=1e-9)
    assert score["ratio_iqr"] < 1e-9
    assert score["pct_error"] == pytest.approx(100.0, rel=1e-6)


def test_scores_survive_an_empty_comparison():
    empty = debond_scores(np.array([np.nan]), np.array([np.nan]))
    assert empty["n_scored"] == 0.0
    assert not np.isfinite(empty["log_rmse"])
