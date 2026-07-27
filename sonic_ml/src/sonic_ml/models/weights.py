"""
Model cards: bind a trained checkpoint to its provenance (M4d).

A ``.save()`` checkpoint (see :class:`~sonic_ml.models.forward.TrainedForwardSurrogate`
and :class:`~sonic_ml.models.inverse.TrainedInverseNet`) stores weights and the
normalizers needed to run the model, but *nothing about how it came to be* --
which fwap physics produced its training data, which config, or how well it
scored. A :class:`ModelCard` is the small JSON sidecar that answers that: the
model type and hyper-parameters, the fwap version + git SHA at training time
(reused from :mod:`sonic_ml.provenance`), the held-out metrics, and -- crucially
-- the ``content_hash`` of the training dataset, so a checkpoint can be tied
back to the exact ``.npz`` (and thus the exact fwap solver output) it learned
from. Without that binding, a re-labelled dataset silently invalidates a
checkpoint with no way to detect it.

This module is deliberately torch-free: it duck-types the trained wrapper
(``.model.hparams`` + ``.save(path)``), so the spine invariant is preserved and
a card can be read on a machine without PyTorch. It lives under ``models/`` only
because that is where the checkpoints it describes are produced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import fwap

import sonic_ml
from sonic_ml import provenance

#: Suffix for the model-card sidecar written next to a checkpoint.
CARD_SUFFIX = ".card.json"


class _TrainedModel(Protocol):
    """Structural type for the trained wrappers a card can describe.

    Both :class:`TrainedForwardSurrogate` and :class:`TrainedInverseNet`
    satisfy this: they expose the network's ``hparams`` and can serialize
    themselves. Duck-typed on purpose -- no torch import here.
    """

    def save(self, path: str) -> None: ...


@dataclass(frozen=True)
class ModelCard:
    """
    Provenance + evaluation record for a trained checkpoint.

    Attributes
    ----------
    model_type : str
        The trained-wrapper class name (e.g. ``"TrainedInverseNet"``).
    fwap_version : str
        ``fwap.__version__`` at training time.
    fwap_git_sha : str or None
        Git commit of the fwap checkout, if available.
    sonic_ml_version : str
        ``sonic_ml.__version__`` at training time.
    hparams : dict
        The network's hyper-parameters (``model.hparams``). JSON-serializable.
    metrics : dict of str to float
        Held-out evaluation metrics (e.g. ``{"vs_mae_m_s": 63.7}``).
    active_names : tuple of str or None
        Predicted (active) parameter columns, for an inverse model.
    mode_names : tuple of str or None
        Mode labels, for a forward surrogate.
    data_content_hash : str or None
        :func:`sonic_ml.provenance.content_hash` of the training dataset, so
        the checkpoint can be tied back to the exact ``.npz`` it learned from.
    note : str or None
        Free-form human note.
    """

    model_type: str
    fwap_version: str
    fwap_git_sha: str | None
    sonic_ml_version: str
    hparams: dict[str, Any]
    metrics: dict[str, float]
    active_names: tuple[str, ...] | None = None
    mode_names: tuple[str, ...] | None = None
    data_content_hash: str | None = None
    note: str | None = None

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a JSON string (keys sorted for a stable sidecar)."""
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ModelCard:
        """Rebuild a :class:`ModelCard` from :meth:`to_json` output.

        JSON has no tuples, so ``active_names`` / ``mode_names`` come back as
        lists and are restored to tuples here.
        """
        d = json.loads(text)
        active = d.get("active_names")
        modes = d.get("mode_names")
        return cls(
            model_type=d["model_type"],
            fwap_version=d["fwap_version"],
            fwap_git_sha=d.get("fwap_git_sha"),
            sonic_ml_version=d["sonic_ml_version"],
            hparams=d.get("hparams", {}),
            metrics=d.get("metrics", {}),
            active_names=tuple(active) if active is not None else None,
            mode_names=tuple(modes) if modes is not None else None,
            data_content_hash=d.get("data_content_hash"),
            note=d.get("note"),
        )

    def write(self, checkpoint_path: str) -> Path:
        """
        Write ``<checkpoint_path>.card.json`` next to the checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the ``.pt`` checkpoint this card describes.

        Returns
        -------
        pathlib.Path
            The written card path.
        """
        card_path = Path(str(checkpoint_path) + CARD_SUFFIX)
        card_path.write_text(self.to_json())
        return card_path

    @classmethod
    def read(cls, checkpoint_path: str) -> ModelCard:
        """Read the model card written for ``checkpoint_path``."""
        card_path = Path(str(checkpoint_path) + CARD_SUFFIX)
        return cls.from_json(card_path.read_text())


def card_for(
    trained: _TrainedModel,
    *,
    metrics: dict[str, float],
    data_content_hash: str | None = None,
    note: str | None = None,
) -> ModelCard:
    """
    Build a :class:`ModelCard` describing a trained model.

    The wrapper is introspected by duck-typing -- ``model.hparams`` plus the
    optional ``active_names`` (inverse) / ``mode_names`` (forward) attributes --
    so no concrete model type or torch import is required.

    Parameters
    ----------
    trained : object
        A trained wrapper exposing ``.model.hparams`` (e.g.
        ``TrainedInverseNet`` or ``TrainedForwardSurrogate``).
    metrics : dict of str to float
        Held-out evaluation metrics; values are coerced to ``float``.
    data_content_hash : str or None
        :func:`sonic_ml.provenance.content_hash` of the training dataset.
    note : str or None
        Optional human note.

    Returns
    -------
    ModelCard
    """
    model = getattr(trained, "model", None)
    hparams = dict(getattr(model, "hparams", {}))

    active_names = getattr(trained, "active_names", None)
    if active_names is None:
        # A forward surrogate keeps its active columns on the standardizer.
        param_std = getattr(trained, "param_std", None)
        active_names = getattr(param_std, "active_names", None)
    mode_names = getattr(trained, "mode_names", None)

    return ModelCard(
        model_type=type(trained).__name__,
        fwap_version=getattr(fwap, "__version__", "unknown"),
        fwap_git_sha=provenance.fwap_git_sha(),
        sonic_ml_version=getattr(sonic_ml, "__version__", "unknown"),
        hparams=hparams,
        metrics={str(k): float(v) for k, v in metrics.items()},
        active_names=tuple(active_names) if active_names is not None else None,
        mode_names=tuple(mode_names) if mode_names is not None else None,
        data_content_hash=data_content_hash,
        note=note,
    )


def save_with_card(
    trained: _TrainedModel,
    path: str,
    *,
    metrics: dict[str, float],
    data_content_hash: str | None = None,
    note: str | None = None,
) -> tuple[Path, Path]:
    """
    Serialize a trained model and write its provenance card beside it.

    Calls ``trained.save(path)`` (the model's own weights-only-safe checkpoint)
    and then writes ``<path>.card.json``. The two files travel together: the
    ``.pt`` is large and git-ignored, the card is small and meant to be
    committed as the durable record of what the checkpoint is.

    Parameters
    ----------
    trained : object
        A trained wrapper with a ``.save(path)`` method.
    path : str
        Destination for the checkpoint (conventionally ``*.pt``).
    metrics : dict of str to float
        Held-out evaluation metrics.
    data_content_hash : str or None
        Content hash of the training dataset.
    note : str or None
        Optional human note.

    Returns
    -------
    (pathlib.Path, pathlib.Path)
        ``(checkpoint_path, card_path)``.
    """
    trained.save(path)
    card = card_for(
        trained,
        metrics=metrics,
        data_content_hash=data_content_hash,
        note=note,
    )
    card_path = card.write(path)
    return Path(path), card_path


def read_card(checkpoint_path: str) -> ModelCard:
    """Read the :class:`ModelCard` sidecar for ``checkpoint_path``."""
    return ModelCard.read(checkpoint_path)
