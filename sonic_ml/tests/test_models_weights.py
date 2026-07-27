"""Tests for sonic_ml.models.weights -- model cards (torch-gated).

The card itself is torch-free, but importing ``sonic_ml.models`` pulls torch,
so the whole module is gated behind ``importorskip`` for parity with the other
``test_models_*`` files.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import fwap
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.dataset import SlownessNormalizer  # noqa: E402
from sonic_ml.models.forward import (  # noqa: E402
    ForwardSurrogate,
    TrainedForwardSurrogate,
)
from sonic_ml.models.inverse import InverseNet, TrainedInverseNet  # noqa: E402
from sonic_ml.models.weights import (  # noqa: E402
    CARD_SUFFIX,
    ModelCard,
    card_for,
    read_card,
    save_with_card,
)
from sonic_ml.normalize import Standardizer  # noqa: E402
from sonic_ml.provenance import content_hash  # noqa: E402


def _trained_inverse(bundle) -> TrainedInverseNet:
    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    model = InverseNet(bundle.gather.shape[1], param_std.n_active, channels=(8, 16))
    return TrainedInverseNet(model, param_std)


def _trained_forward(bundle) -> TrainedForwardSurrogate:
    param_std = Standardizer.fit(bundle.params, bundle.param_names)
    slow_norm = SlownessNormalizer.fit(bundle.slowness)
    model = ForwardSurrogate(
        param_std.n_active, bundle.n_modes, bundle.n_freq, width=16, depth=1
    )
    return TrainedForwardSurrogate(model, param_std, slow_norm, bundle.mode_names)


# ------------------------------------------------------------------
# ModelCard dataclass
# ------------------------------------------------------------------


def test_model_card_json_round_trip_restores_tuples():
    card = ModelCard(
        model_type="TrainedInverseNet",
        fwap_version="9.9.9",
        fwap_git_sha="deadbeef",
        sonic_ml_version="0.0.1",
        hparams={"n_rec": 8, "channels": [8, 16]},
        metrics={"vs_mae_m_s": 63.7},
        active_names=("vp", "vs", "rho", "a"),
        mode_names=None,
        data_content_hash="abc123",
        note="unit test",
    )
    back = ModelCard.from_json(card.to_json())
    assert back == card
    # JSON has no tuple type; the loader must restore it.
    assert isinstance(back.active_names, tuple)


def test_model_card_json_is_sorted_and_stable():
    card = ModelCard(
        model_type="M",
        fwap_version="1",
        fwap_git_sha=None,
        sonic_ml_version="0.0.1",
        hparams={},
        metrics={},
    )
    # sort_keys=True -> byte-stable sidecar across runs.
    assert card.to_json() == card.to_json()
    assert '"model_type": "M"' in card.to_json()


# ------------------------------------------------------------------
# card_for: introspection of trained wrappers
# ------------------------------------------------------------------


def test_card_for_inverse_captures_provenance_and_hparams(bundle):
    trained = _trained_inverse(bundle)
    card = card_for(trained, metrics={"vs_mae_m_s": 42})
    assert card.model_type == "TrainedInverseNet"
    assert card.hparams == dict(trained.model.hparams)
    assert card.active_names == tuple(trained.active_names)
    assert card.mode_names is None  # an inverse net has no mode axis
    assert card.fwap_version == getattr(fwap, "__version__", "unknown")
    # metrics values are coerced to float.
    assert card.metrics == {"vs_mae_m_s": 42.0}
    assert isinstance(card.metrics["vs_mae_m_s"], float)
    # git sha degrades gracefully.
    assert card.fwap_git_sha is None or isinstance(card.fwap_git_sha, str)
    assert card.sonic_ml_version  # non-empty


def test_card_for_forward_captures_mode_and_active_names(bundle):
    trained = _trained_forward(bundle)
    card = card_for(trained, metrics={"rmse": 1.0})
    assert card.model_type == "TrainedForwardSurrogate"
    assert card.mode_names == tuple(bundle.mode_names)
    # active_names is recovered from the standardizer for a forward surrogate.
    assert card.active_names == tuple(trained.param_std.active_names)
    assert card.hparams == dict(trained.model.hparams)


def test_card_for_binds_data_content_hash(bundle, stacked):
    trained = _trained_inverse(bundle)
    h = content_hash(stacked)
    card = card_for(trained, metrics={}, data_content_hash=h, note="bound")
    assert card.data_content_hash == h
    assert card.note == "bound"


# ------------------------------------------------------------------
# save_with_card / read_card
# ------------------------------------------------------------------


def test_save_with_card_writes_checkpoint_and_sidecar(bundle, tmp_path):
    trained = _trained_inverse(bundle)
    ref_p, ref_s = trained.predict(bundle.gather[:3])
    path = tmp_path / "inv.pt"
    ckpt_path, card_path = save_with_card(
        trained, str(path), metrics={"vs_mae_m_s": 63.7}
    )
    # Both files exist and the card sidecar name derives from the checkpoint.
    assert ckpt_path == path and path.exists()
    assert card_path == Path(str(path) + CARD_SUFFIX)
    assert card_path.exists()

    # The checkpoint still loads and predicts identically (card is a no-op on it).
    loaded = TrainedInverseNet.load(str(path))
    got_p, got_s = loaded.predict(bundle.gather[:3])
    np.testing.assert_allclose(got_p, ref_p, rtol=0, atol=0)
    np.testing.assert_allclose(got_s, ref_s, rtol=0, atol=0)

    # The sidecar round-trips to an equal card.
    card = read_card(str(path))
    assert card.model_type == "TrainedInverseNet"
    assert card.metrics == {"vs_mae_m_s": 63.7}
    assert card == ModelCard.from_json(card_path.read_text())


def test_read_card_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_card(str(tmp_path / "nope.pt"))


# ------------------------------------------------------------------
# .gitignore keeps checkpoints (but not cards) out of git
# ------------------------------------------------------------------


def _repo_root() -> Path:
    # fwap/__init__.py -> fwap/ -> repo root.
    assert fwap.__file__ is not None
    return Path(fwap.__file__).resolve().parent.parent


def test_gitignore_excludes_checkpoints_not_cards():
    root = _repo_root()
    try:
        ignored = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "model.pt", "model.pt.card.json"],
            capture_output=True,
            text=True,
        )
    except OSError:
        pytest.skip("git not available")
    if not (root / ".gitignore").exists():  # pragma: no cover - defensive
        pytest.skip("no .gitignore at repo root")
    out = ignored.stdout
    # The large binary checkpoint is ignored; its small JSON card is committed.
    assert "model.pt" in out
    assert "model.pt.card.json" not in out
