"""
Torch models for sonic_ml (issue #22, M2+).

Importing this subpackage requires PyTorch (a hard dependency of ``sonic_ml``
but imported only here, so the pure-NumPy spine stays torch-free). M2 ships the
forward dispersion surrogate.
"""

from __future__ import annotations

from sonic_ml.models.dataset import ForwardDataset, SlownessNormalizer
from sonic_ml.models.forward import ForwardSurrogate, TrainedForwardSurrogate
from sonic_ml.models.losses import masked_slowness_loss, presence_bce
from sonic_ml.models.train import (
    presence_auc,
    slowness_rmse,
    train_forward,
)

__all__ = [
    "ForwardSurrogate",
    "TrainedForwardSurrogate",
    "ForwardDataset",
    "SlownessNormalizer",
    "masked_slowness_loss",
    "presence_bce",
    "train_forward",
    "slowness_rmse",
    "presence_auc",
]
