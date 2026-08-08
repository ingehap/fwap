"""Tests for the cased-hole forward operator (M5c, torch-gated)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.cased import (  # noqa: E402
    BACKBONES,
    CasedForwardOperator,
    TrainedCasedOperator,
    cased_features,
    normalized_coords,
)
from sonic_ml.models.cased_train import (  # noqa: E402
    CasedDataset,
    resolution_transfer_error,
    slowness_mae_us_per_ft,
    train_cased_operator,
)
from sonic_ml.models.dataset import SlownessNormalizer  # noqa: E402
from sonic_ml.normalize import Standardizer  # noqa: E402
from sonic_ml.split import stratified_split  # noqa: E402

_M_PER_FT = 0.3048


@pytest.fixture(scope="module")
def split(cased_bundle):
    return stratified_split(cased_bundle, seed=0)


@pytest.fixture(scope="module")
def trained(cased_bundle, split):
    """One trained operator per backbone (module-scoped: training is the cost)."""
    return {
        backbone: train_cased_operator(
            cased_bundle, split=split, epochs=40, backbone=backbone, seed=0
        )
        for backbone in BACKBONES
    }


def _mean_curve_baseline_mae(bundle, split) -> float:
    """MAE (us/ft) of predicting the training-mean curve for every sample."""
    train_mean = np.nanmean(bundle.slowness[split.train], axis=0)
    truth = bundle.slowness[split.test]
    mask = np.isfinite(truth)
    base = np.broadcast_to(train_mean, truth.shape)
    return float(np.abs(base[mask] - truth[mask]).mean() * _M_PER_FT * 1.0e6)


# ------------------------------------------------------------------
# cased_features
# ------------------------------------------------------------------


def test_cased_features_shape_and_names(cased_bundle):
    features, names = cased_features(cased_bundle)
    # 6 formation + (2 layers x 4 properties) + bond index
    assert features.shape == (cased_bundle.n_samples, 15)
    assert len(names) == 15
    assert names[: len(cased_bundle.param_names)] == cased_bundle.param_names
    assert names[6:14] == (
        "casing_vp",
        "casing_vs",
        "casing_rho",
        "casing_thickness",
        "cement_vp",
        "cement_vs",
        "cement_rho",
        "cement_thickness",
    )
    assert names[-1] == "bond_index"
    assert np.all(np.isfinite(features))


def test_cased_features_columns_match_bundle_arrays(cased_bundle):
    features, names = cased_features(cased_bundle)
    np.testing.assert_array_equal(features[:, :6], cased_bundle.params)
    # Layer block is the (L, 4) stack flattened in radial order.
    np.testing.assert_array_equal(
        features[:, 6:14], cased_bundle.layer_params.reshape(cased_bundle.n_samples, -1)
    )
    np.testing.assert_array_equal(features[:, -1], cased_bundle.bond_index)


def test_cased_features_rejects_open_hole_bundle(bundle):
    # The open-hole fixture is schema v4 but carries no annulus.
    assert not bundle.is_cased
    with pytest.raises(ValueError, match="cased-hole"):
        cased_features(bundle)


def test_standardizer_drops_constant_feature_columns(cased_bundle):
    features, names = cased_features(cased_bundle)
    std = Standardizer.fit(features, names)
    # Fluid properties and the fixed casing density carry no information.
    for constant in ("vf", "rho_f", "casing_rho"):
        assert constant not in std.active_names
    assert "bond_index" in std.active_names
    assert "cement_vs" in std.active_names
    assert std.n_active == len(names) - 3


# ------------------------------------------------------------------
# normalized_coords
# ------------------------------------------------------------------


def test_normalized_coords_maps_training_band_to_unit_interval():
    freq = np.linspace(1000.0, 10000.0, 5)
    coords = normalized_coords(freq, 1000.0, 10000.0)
    np.testing.assert_allclose(coords, np.linspace(0.0, 1.0, 5))


def test_normalized_coords_uses_training_band_not_query_endpoints():
    # A sub-band query must land inside [0, 1], not be rescaled to span it.
    sub = np.linspace(4000.0, 6000.0, 3)
    coords = normalized_coords(sub, 1000.0, 10000.0)
    assert coords[0] > 0.0 and coords[-1] < 1.0
    np.testing.assert_allclose(coords, [1 / 3, 4 / 9, 5 / 9])


def test_normalized_coords_rejects_degenerate_band():
    with pytest.raises(ValueError):
        normalized_coords(np.array([1.0]), 5000.0, 5000.0)


# ------------------------------------------------------------------
# CasedForwardOperator
# ------------------------------------------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_operator_output_shape(backbone):
    model = CasedForwardOperator(
        6, 2, backbone=backbone, width=16, n_fourier_modes=6, depth=2, latent=16
    )
    out = model(torch.zeros(3, 6), torch.linspace(0, 1, 32))
    assert out.shape == (3, 2, 32)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_operator_accepts_a_regridded_coordinate_axis(backbone):
    model = CasedForwardOperator(
        4, 1, backbone=backbone, width=16, n_fourier_modes=6, depth=2, latent=16
    )
    for n_grid in (16, 48, 97):
        out = model(torch.zeros(2, 4), torch.linspace(0, 1, n_grid))
        assert out.shape == (2, 1, n_grid)


def test_operator_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="backbone"):
        CasedForwardOperator(4, backbone="transformer")


def test_operator_hparams_recorded_for_model_card():
    model = CasedForwardOperator(7, 3, backbone="fno", width=16, depth=2)
    assert model.hparams["n_features"] == 7
    assert model.hparams["n_modes_out"] == 3
    assert model.hparams["backbone"] == "fno"


# ------------------------------------------------------------------
# CasedDataset
# ------------------------------------------------------------------


def test_cased_dataset_items(cased_bundle, split):
    features, names = cased_features(cased_bundle)
    std = Standardizer.fit(features[split.train], names)
    slow_norm = SlownessNormalizer.fit(cased_bundle.slowness[split.train])
    ds = CasedDataset(cased_bundle, split.train, std, slow_norm)
    assert len(ds) == split.train.size
    x, target, mask = ds[0]
    assert x.shape == (std.n_active,)
    assert target.shape == (cased_bundle.n_modes, cased_bundle.n_freq)
    assert mask.shape == target.shape
    assert x.dtype == torch.float32 and mask.dtype == torch.bool
    # Masked entries are zeroed, never NaN.
    assert torch.isfinite(target).all()


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_training_reduces_loss(trained, backbone):
    _, history = trained[backbone]
    assert len(history) == 40
    assert all(np.isfinite(h) for h in history)
    assert history[-1] < 0.5 * history[0]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_beats_predict_the_mean_baseline(trained, cased_bundle, split, backbone):
    """Skill is measured against a real baseline, not an absolute MAE."""
    model, _ = trained[backbone]
    mae = slowness_mae_us_per_ft(model, cased_bundle, split.test)
    baseline = _mean_curve_baseline_mae(cased_bundle, split)
    assert np.isfinite(mae)
    assert mae < baseline


def test_train_cased_operator_is_reproducible(cased_bundle, split):
    a, ha = train_cased_operator(cased_bundle, split=split, epochs=3, seed=0)
    b, hb = train_cased_operator(cased_bundle, split=split, epochs=3, seed=0)
    assert ha == hb
    features, _ = cased_features(cased_bundle)
    np.testing.assert_array_equal(a.predict(features[:3]), b.predict(features[:3]))


def test_train_rejects_open_hole_bundle(bundle):
    with pytest.raises(ValueError, match="cased-hole"):
        train_cased_operator(bundle, epochs=1)


# ------------------------------------------------------------------
# Checkpoint round-trip
# ------------------------------------------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_checkpoint_round_trips_exactly(trained, cased_bundle, tmp_path, backbone):
    model, _ = trained[backbone]
    features, _ = cased_features(cased_bundle)
    ref = model.predict(features[:4])
    path = tmp_path / f"{backbone}.pt"
    model.save(str(path))
    loaded = TrainedCasedOperator.load(str(path))
    np.testing.assert_array_equal(loaded.predict(features[:4]), ref)
    assert loaded.model.hparams["backbone"] == backbone
    assert loaded.feature_names == model.feature_names
    assert loaded.mode_names == model.mode_names
    np.testing.assert_array_equal(loaded.freq, model.freq)


# ------------------------------------------------------------------
# The operator payoff: evaluation on an unseen frequency grid
# ------------------------------------------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_predicts_on_a_grid_it_was_never_trained_on(trained, cased_bundle, backbone):
    model, _ = trained[backbone]
    features, _ = cased_features(cased_bundle)
    fine = np.linspace(model.f_min, model.f_max, 3 * cased_bundle.n_freq)
    out = model.predict(features[:5], freq=fine)
    assert out.shape == (5, cased_bundle.n_modes, fine.size)
    assert np.all(np.isfinite(out))
    # Predictions stay on the physical scale of the training data.
    truth = cased_bundle.slowness[np.isfinite(cased_bundle.slowness)]
    assert out.min() > 0.5 * truth.min()
    assert out.max() < 2.0 * truth.max()


@pytest.mark.parametrize("backbone", BACKBONES)
def test_resolution_transfer_is_self_consistent(trained, cased_bundle, split, backbone):
    model, _ = trained[backbone]
    for factor in (2, 3):
        rel = resolution_transfer_error(model, cased_bundle, split.test, factor=factor)
        assert np.isfinite(rel)
        assert rel < 0.05


def test_deeponet_resolution_transfer_beats_fno_by_orders_of_magnitude(
    trained, cased_bundle, split
):
    """A pointwise trunk makes re-gridding consistent to float precision.

    The DeepONet evaluates each coordinate independently, so a shared node's
    value cannot depend on the rest of the grid. The FNO couples the whole grid
    through the FFT, so its transfer error is small but genuinely non-zero --
    the contrast is the point, and it is the reason to keep both backbones.
    """
    deeponet, _ = trained["deeponet"]
    fno, _ = trained["fno"]
    pointwise = resolution_transfer_error(deeponet, cased_bundle, split.test)
    global_ = resolution_transfer_error(fno, cased_bundle, split.test)
    assert pointwise < 1e-6
    assert pointwise < 0.01 * global_


def test_deeponet_prediction_is_slice_invariant(trained, cased_bundle):
    """Querying a sub-grid equals slicing the full-grid prediction (DeepONet).

    Exact in exact arithmetic; compared at float32 tolerance because a
    different query length can select a different BLAS kernel in the trunk's
    ``Linear`` layers (observed drift is ~1e-9 relative).
    """
    model, _ = trained["deeponet"]
    features, _ = cased_features(cased_bundle)
    full = model.predict(features[:3])
    subset = model.predict(features[:3], freq=cased_bundle.freq[:5])
    np.testing.assert_allclose(subset, full[:, :, :5], rtol=1e-6, atol=0)


def test_resolution_transfer_rejects_factor_below_two(trained, cased_bundle, split):
    model, _ = trained["fno"]
    with pytest.raises(ValueError):
        resolution_transfer_error(model, cased_bundle, split.test, factor=1)
