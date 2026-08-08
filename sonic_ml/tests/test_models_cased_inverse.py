"""Tests for the cement-bond inverse and its scoring harness (M5d, torch-gated).

A note on what is *not* asserted here. The M3 open-hole inverse beat classical
processing by ~25x, so a head-to-head ordering was a safe CI assertion. The
cased-hole bond problem is far less identifiable -- a forward sweep moves the
Stoneley curve ~7% across the cement-stiffness prior, against ~1.5% for
formation Vs -- and the CI fixture trains on ~32 samples, so an ordering
assertion would be flaky rather than informative. These tests therefore pin
*correctness* (shapes, ranges, protocol conformance, harness arithmetic,
anti-leakage, reproducibility); the measured head-to-head numbers live in the
CHANGELOG, quoted with the sample count they came from.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.baselines.bond import (  # noqa: E402
    StoneleyBondBaseline,
    stoneley_peak_slowness,
)
from sonic_ml.bench import format_scorecard  # noqa: E402
from sonic_ml.bench.bond import (  # noqa: E402
    BOND_ROW_ORDER,
    BondPredictor,
    MeanBondPredictor,
    bond_regime_labels,
    evaluate_bond,
)
from sonic_ml.models.cased_inverse import (  # noqa: E402
    DEFAULT_CASED_TARGETS,
    CasedInverseDataset,
    NeuralBondPredictor,
    cased_residual_zscore_std,
    cased_target_mae,
    cased_targets,
    train_cased_inverse,
)
from sonic_ml.models.inverse import TrainedInverseNet  # noqa: E402
from sonic_ml.normalize import Standardizer  # noqa: E402
from sonic_ml.split import stratified_split  # noqa: E402


@pytest.fixture(scope="module")
def cased_split(cased_bundle):
    return stratified_split(cased_bundle, seed=0)


@pytest.fixture(scope="module")
def cased_net(cased_bundle, cased_split):
    """A briefly-trained bond inverse (module-scoped: training is the cost)."""
    return train_cased_inverse(cased_bundle, split=cased_split, epochs=15, seed=0)


class _PerfectBond:
    """Bond predictor that returns the truth -- validates the harness itself."""

    name = "perfect"

    def predict_bond(self, bundle, geom, indices):
        return np.asarray(bundle.bond_index, dtype=float)[np.asarray(indices, int)]


# ------------------------------------------------------------------
# cased_targets
# ------------------------------------------------------------------


def test_cased_targets_default_shape_and_values(cased_bundle):
    targets, names = cased_targets(cased_bundle)
    assert names == DEFAULT_CASED_TARGETS == ("vs", "bond_index")
    assert targets.shape == (cased_bundle.n_samples, 2)
    np.testing.assert_array_equal(targets[:, 0], cased_bundle.param("vs"))
    np.testing.assert_array_equal(targets[:, 1], cased_bundle.bond_index)


def test_cased_targets_custom_names(cased_bundle):
    targets, names = cased_targets(cased_bundle, ("bond_index", "vp", "a"))
    assert names == ("bond_index", "vp", "a")
    np.testing.assert_array_equal(targets[:, 1], cased_bundle.param("vp"))


def test_cased_targets_rejects_open_hole(bundle):
    with pytest.raises(ValueError, match="cased-hole"):
        cased_targets(bundle)


def test_cased_targets_rejects_unknown_name(cased_bundle):
    with pytest.raises(ValueError, match="unknown cased target"):
        cased_targets(cased_bundle, ("not_a_target",))


# ------------------------------------------------------------------
# CasedInverseDataset
# ------------------------------------------------------------------


def test_cased_inverse_dataset_items(cased_bundle, cased_split):
    targets, names = cased_targets(cased_bundle)
    std = Standardizer.fit(targets[cased_split.train], names)
    ds = CasedInverseDataset(cased_bundle, cased_split.train, std)
    assert len(ds) == cased_split.train.size
    x, y = ds[0]
    assert x.shape == cased_bundle.gather.shape[1:]
    assert y.shape == (std.n_active,)
    assert x.dtype == torch.float32


def test_cased_inverse_input_is_gather_only(cased_bundle, cased_split):
    """Anti-leakage: garbling the dispersion label must not change the input."""
    import dataclasses

    targets, names = cased_targets(cased_bundle)
    std = Standardizer.fit(targets[cased_split.train], names)
    ref = CasedInverseDataset(cased_bundle, cased_split.train, std).x
    garbled = dataclasses.replace(
        cased_bundle, slowness=np.zeros_like(cased_bundle.slowness)
    )
    assert torch.equal(CasedInverseDataset(garbled, cased_split.train, std).x, ref)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def test_train_cased_inverse_learns_and_names_targets(cased_net):
    trained, history = cased_net
    assert len(history) == 15
    assert all(np.isfinite(h) for h in history)
    assert history[-1] < history[0]
    assert set(trained.active_names) <= set(DEFAULT_CASED_TARGETS)
    assert "bond_index" in trained.active_names


def test_train_cased_inverse_is_reproducible(cased_bundle, cased_split):
    a, ha = train_cased_inverse(cased_bundle, split=cased_split, epochs=3, seed=0)
    b, hb = train_cased_inverse(cased_bundle, split=cased_split, epochs=3, seed=0)
    assert ha == hb
    pa, sa = a.predict(cased_bundle.gather[:3])
    pb, sb = b.predict(cased_bundle.gather[:3])
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_array_equal(sa, sb)


def test_train_cased_inverse_rejects_open_hole(bundle):
    with pytest.raises(ValueError, match="cased-hole"):
        train_cased_inverse(bundle, epochs=1)


def test_cased_checkpoint_round_trips(cased_net, cased_bundle, tmp_path):
    trained, _ = cased_net
    ref_p, ref_s = trained.predict(cased_bundle.gather[:3])
    path = tmp_path / "bond.pt"
    trained.save(str(path))
    loaded = TrainedInverseNet.load(str(path))
    got_p, got_s = loaded.predict(cased_bundle.gather[:3])
    np.testing.assert_array_equal(got_p, ref_p)
    np.testing.assert_array_equal(got_s, ref_s)
    assert loaded.active_names == trained.active_names


def test_cased_target_mae_is_finite(cased_net, cased_bundle, cased_split):
    trained, _ = cased_net
    for name in trained.active_names:
        mae = cased_target_mae(trained, cased_bundle, cased_split.test, name)
        assert np.isfinite(mae) and mae >= 0.0


def test_cased_residual_zscore_std_reaches_bond_index(
    cased_net, cased_bundle, cased_split
):
    """The M3 z-score helper resolves truth via ``bundle.param()`` and so cannot
    see ``bond_index``; the cased variant must."""
    from sonic_ml.models.inverse_train import residual_zscore_std

    trained, _ = cased_net
    for name in trained.active_names:
        z = cased_residual_zscore_std(trained, cased_bundle, cased_split.test, name)
        assert np.isfinite(z) and z > 0.0
    # The formation-parameter helper agrees on a formation column ...
    if "vs" in trained.active_names:
        np.testing.assert_allclose(
            cased_residual_zscore_std(trained, cased_bundle, cased_split.test, "vs"),
            residual_zscore_std(trained, cased_bundle, cased_split.test, "vs"),
        )
    # ... but raises on the bond target, which is why the cased variant exists.
    with pytest.raises(KeyError):
        residual_zscore_std(trained, cased_bundle, cased_split.test, "bond_index")


def test_uncertainty_is_positive_and_per_target(cased_net, cased_bundle):
    trained, _ = cased_net
    params, std = trained.predict(cased_bundle.gather[:6])
    assert params.shape == std.shape == (6, len(trained.active_names))
    assert np.all(std > 0.0)


# ------------------------------------------------------------------
# NeuralBondPredictor
# ------------------------------------------------------------------


def test_neural_bond_predictor_conforms_and_clips(cased_net, cased_bundle, geom):
    trained, _ = cased_net
    predictor = NeuralBondPredictor(trained)
    assert isinstance(predictor, BondPredictor)
    out = predictor.predict_bond(cased_bundle, geom, np.arange(5))
    assert out.shape == (5,)
    assert np.all((out >= 0.0) & (out <= 1.0))  # bond index is a [0, 1] quantity
    sigma = predictor.predict_bond_uncertainty(cased_bundle, np.arange(5))
    assert sigma.shape == (5,) and np.all(sigma > 0.0)


def test_neural_bond_predictor_rejects_model_without_bond(cased_bundle, cased_split):
    trained, _ = train_cased_inverse(
        cased_bundle, split=cased_split, target_names=("vs", "vp"), epochs=2, seed=0
    )
    with pytest.raises(ValueError, match="bond_index"):
        NeuralBondPredictor(trained)


# ------------------------------------------------------------------
# Bond scoring harness
# ------------------------------------------------------------------


def test_bond_regime_labels_use_fixed_thirds(cased_bundle):
    labels = bond_regime_labels(cased_bundle)
    bond = cased_bundle.bond_index
    assert labels.shape == bond.shape
    np.testing.assert_array_equal(labels == 0, bond < 1 / 3)
    np.testing.assert_array_equal(labels == 2, bond >= 2 / 3)


def test_bond_regime_labels_rejects_open_hole(bundle):
    with pytest.raises(ValueError, match="cased-hole"):
        bond_regime_labels(bundle)


def test_evaluate_bond_scores_a_perfect_predictor_at_zero(cased_bundle, geom):
    card = evaluate_bond(_PerfectBond(), cased_bundle, geom, n_boot=50)
    assert card.parameter == "bond_index"
    assert card.per_regime["all"].median_abs_error == 0.0
    assert card.per_regime["all"].n == cased_bundle.n_samples
    assert card.per_regime["all"].n_finite == cased_bundle.n_samples


def test_evaluate_bond_splits_by_bond_regime(cased_bundle, geom):
    card = evaluate_bond(
        MeanBondPredictor(float(cased_bundle.bond_index.mean())),
        cased_bundle,
        geom,
        n_boot=50,
    )
    present = set(card.per_regime) - {"all"}
    assert present  # at least one bond regime is populated
    assert present <= {"poor", "medium", "good"}
    assert sum(card.per_regime[r].n for r in present) == cased_bundle.n_samples


def test_evaluate_bond_rejects_open_hole(bundle, geom):
    with pytest.raises(ValueError, match="cased-hole"):
        evaluate_bond(_PerfectBond(), bundle, geom)


def test_format_scorecard_renders_bond_rows_at_higher_precision(cased_bundle, geom):
    card = evaluate_bond(MeanBondPredictor(0.5), cased_bundle, geom, n_boot=50)
    table = format_scorecard(card, row_order=BOND_ROW_ORDER, precision=3)
    assert "bond_index" in table
    # A [0, 1] target needs 3 decimals to be legible at all.
    assert any(line.count(".") >= 2 for line in table.splitlines()[3:])
    # The default Vs row order would have printed no data rows at all.
    assert len(table.splitlines()) > 3


def test_format_scorecard_default_behaviour_is_unchanged(bundle, geom):
    """The Vs scorecard rendering must not shift under the new parameters."""
    from sonic_ml.bench import StubPredictor, evaluate

    card = evaluate(StubPredictor(2000.0), bundle, geom, n_boot=20)
    assert format_scorecard(card) == format_scorecard(
        card, row_order=("all", "slow", "fast"), precision=1
    )


# ------------------------------------------------------------------
# Classical Stoneley bond baseline
# ------------------------------------------------------------------


def test_stoneley_peak_slowness_lands_in_the_search_window(cased_bundle, geom):
    s = stoneley_peak_slowness(cased_bundle.gather[0], geom)
    assert np.isfinite(s)
    assert 5.0e-4 <= s <= 1.1e-3


def test_bond_baseline_requires_fit_first(cased_bundle, geom):
    baseline = StoneleyBondBaseline()
    assert not baseline.is_fitted
    with pytest.raises(RuntimeError, match="fit"):
        baseline.predict_bond(cased_bundle, geom, np.arange(2))


def test_bond_baseline_fits_and_predicts_in_range(cased_bundle, geom, cased_split):
    # STC is ~0.3 s per gather, so score only a handful of samples here.
    train_idx = cased_split.train[:10]
    test_idx = cased_split.test[:4]
    baseline = StoneleyBondBaseline().fit(cased_bundle, geom, train_idx)
    assert baseline.is_fitted
    assert isinstance(baseline, BondPredictor)
    out = baseline.predict_bond(cased_bundle, geom, test_idx)
    assert out.shape == (test_idx.size,)
    finite = out[np.isfinite(out)]
    assert finite.size  # the STC found a peak for at least one gather
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_bond_baseline_rejects_open_hole(bundle, geom):
    with pytest.raises(ValueError, match="cased-hole"):
        StoneleyBondBaseline().fit(bundle, geom, np.arange(2))
