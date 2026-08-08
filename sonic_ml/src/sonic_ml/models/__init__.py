"""
Torch models for sonic_ml (issue #22, M2+).

Importing this subpackage requires PyTorch (a hard dependency of ``sonic_ml``
but imported only here, so the pure-NumPy spine stays torch-free). M2 ships the
forward dispersion surrogate; M3 the DL-FWI inverse net.
"""

from __future__ import annotations

from sonic_ml.models.augment import GatherAugmentation
from sonic_ml.models.cased import (
    CasedForwardOperator,
    TrainedCasedOperator,
    cased_features,
    normalized_coords,
)
from sonic_ml.models.cased_train import (
    CasedDataset,
    resolution_transfer_error,
    slowness_mae_us_per_ft,
    train_cased_operator,
)
from sonic_ml.models.dataset import ForwardDataset, SlownessNormalizer
from sonic_ml.models.forward import ForwardSurrogate, TrainedForwardSurrogate
from sonic_ml.models.inverse import InverseNet, TrainedInverseNet
from sonic_ml.models.inverse_dataset import InverseDataset, normalize_gathers
from sonic_ml.models.inverse_train import (
    InversePredictor,
    parameter_mae,
    residual_zscore_std,
    train_inverse,
)
from sonic_ml.models.losses import gaussian_nll, masked_slowness_loss, presence_bce
from sonic_ml.models.lwd import (
    LatencyAccuracy,
    build_lwd_inverse_net,
    count_parameters,
    format_latency_accuracy,
    latency_accuracy_report,
    measure_latency_ms,
    train_lwd_inverse,
)
from sonic_ml.models.operator import (
    DeepONet,
    FNO1d,
    SpectralConv1d,
    params_on_grid,
)
from sonic_ml.models.train import (
    presence_auc,
    slowness_rmse,
    train_forward,
)
from sonic_ml.models.weights import (
    ModelCard,
    card_for,
    read_card,
    save_with_card,
)

__all__ = [
    # forward surrogate (M2)
    "ForwardSurrogate",
    "TrainedForwardSurrogate",
    "ForwardDataset",
    "SlownessNormalizer",
    "masked_slowness_loss",
    "presence_bce",
    "train_forward",
    "slowness_rmse",
    "presence_auc",
    # inverse net (M3)
    "InverseNet",
    "TrainedInverseNet",
    "InverseDataset",
    "normalize_gathers",
    "gaussian_nll",
    "train_inverse",
    "InversePredictor",
    "parameter_mae",
    "residual_zscore_std",
    # augmentation (M4)
    "GatherAugmentation",
    # model cards / weight storage (M4d)
    "ModelCard",
    "card_for",
    "save_with_card",
    "read_card",
    # low-latency LWD inverse variant (M4f)
    "build_lwd_inverse_net",
    "train_lwd_inverse",
    "count_parameters",
    "measure_latency_ms",
    "LatencyAccuracy",
    "latency_accuracy_report",
    "format_latency_accuracy",
    # operator-learning primitives (M5b)
    "SpectralConv1d",
    "FNO1d",
    "DeepONet",
    "params_on_grid",
    # cased-hole forward operator (M5c)
    "CasedForwardOperator",
    "TrainedCasedOperator",
    "cased_features",
    "normalized_coords",
    "CasedDataset",
    "train_cased_operator",
    "slowness_mae_us_per_ft",
    "resolution_transfer_error",
]
