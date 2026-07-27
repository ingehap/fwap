"""
Forward dispersion surrogate: formation parameters -> per-mode slowness.

A small residual MLP that maps the varying standardized formation parameters
to two heads on the shared frequency grid: a per-mode slowness curve
``(M, F)`` and a per-mode presence logit ``(M)``. It is a fast, differentiable
stand-in for the modal-determinant root-finding -- framed honestly as a
plumbing / validation baseline (the unlayered path it learns is already cheap),
and a prerequisite for surrogate-in-the-loop inverse training later.

:class:`TrainedForwardSurrogate` bundles the network with the parameter and
slowness normalizers so it can predict in raw physical units and round-trip
through a checkpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from sonic_ml.models.dataset import SlownessNormalizer
from sonic_ml.normalize import Standardizer


class _ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ForwardSurrogate(nn.Module):
    """
    Residual-MLP forward surrogate.

    Parameters
    ----------
    n_features : int
        Number of (active) input parameters.
    n_modes : int
        Number of modes ``M`` (slowness/presence output width).
    n_freq : int
        Number of frequency samples ``F`` per mode.
    width : int, default 256
        Hidden width.
    depth : int, default 4
        Number of residual blocks.
    dropout : float, default 0.0
        Dropout probability inside the residual blocks (also enables
        MC-dropout uncertainty later).
    """

    def __init__(
        self,
        n_features: int,
        n_modes: int,
        n_freq: int,
        *,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hparams: dict[str, int | float] = {
            "n_features": n_features,
            "n_modes": n_modes,
            "n_freq": n_freq,
            "width": width,
            "depth": depth,
            "dropout": dropout,
        }
        self.n_modes = n_modes
        self.n_freq = n_freq
        self.input_proj = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(width, dropout) for _ in range(depth)]
        )
        self.head_norm = nn.LayerNorm(width)
        self.slowness_head = nn.Linear(width, n_modes * n_freq)
        self.presence_head = nn.Linear(width, n_modes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map parameters to (slowness, presence-logits).

        Parameters
        ----------
        x : Tensor, shape (B, n_features)

        Returns
        -------
        slowness : Tensor, shape (B, M, F)
            Normalized per-mode slowness.
        presence_logits : Tensor, shape (B, M)
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        slowness = self.slowness_head(h).reshape(x.shape[0], self.n_modes, self.n_freq)
        presence = self.presence_head(h)
        return slowness, presence


class TrainedForwardSurrogate:
    """
    A trained :class:`ForwardSurrogate` with its normalizers.

    Predicts in raw physical units and serializes to a checkpoint.

    Parameters
    ----------
    model : ForwardSurrogate
    param_std : Standardizer
        Fitted parameter standardizer.
    slowness_norm : SlownessNormalizer
        Fitted slowness normalizer.
    mode_names : tuple of str
        Mode labels aligned with the model's output axis.
    """

    def __init__(
        self,
        model: ForwardSurrogate,
        param_std: Standardizer,
        slowness_norm: SlownessNormalizer,
        mode_names: tuple[str, ...],
    ) -> None:
        self.model = model
        self.param_std = param_std
        self.slowness_norm = slowness_norm
        self.mode_names = mode_names

    @torch.no_grad()
    def predict(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict raw slowness curves and presence probabilities.

        Parameters
        ----------
        params : ndarray, shape (N, P)
            Raw formation parameters (full ``PARAM_NAMES`` columns).

        Returns
        -------
        slowness : ndarray, shape (N, M, F), float64
            Predicted per-mode slowness (s/m), de-normalized.
        presence : ndarray, shape (N, M), float64
            Per-mode presence probabilities (sigmoid of the logits).
        """
        self.model.eval()
        x = self.param_std.transform(np.asarray(params, dtype=float))
        xt = torch.as_tensor(x, dtype=torch.float32)
        s_norm, p_logits = self.model(xt)
        slowness = self.slowness_norm.inverse(s_norm.cpu().numpy().astype(float))
        presence = torch.sigmoid(p_logits).cpu().numpy().astype(float)
        return slowness, presence

    def save(self, path: str) -> None:
        """Serialize the model + normalizers to ``path`` (weights-only safe)."""
        checkpoint: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "hparams": dict(self.model.hparams),
            "param_mean": torch.as_tensor(self.param_std.mean),
            "param_std": torch.as_tensor(self.param_std.std),
            "param_active": torch.as_tensor(self.param_std.active),
            "param_names": list(self.param_std.names),
            "slow_mean": torch.as_tensor(self.slowness_norm.mean),
            "slow_std": torch.as_tensor(self.slowness_norm.std),
            "mode_names": list(self.mode_names),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, *, map_location: str = "cpu") -> TrainedForwardSurrogate:
        """Load a checkpoint written by :meth:`save`."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        hp = ckpt["hparams"]
        model = ForwardSurrogate(
            int(hp["n_features"]),
            int(hp["n_modes"]),
            int(hp["n_freq"]),
            width=int(hp["width"]),
            depth=int(hp["depth"]),
            dropout=float(hp["dropout"]),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        param_std = Standardizer(
            mean=ckpt["param_mean"].numpy(),
            std=ckpt["param_std"].numpy(),
            active=ckpt["param_active"].numpy().astype(bool),
            names=tuple(ckpt["param_names"]),
        )
        slowness_norm = SlownessNormalizer(
            mean=ckpt["slow_mean"].numpy(),
            std=ckpt["slow_std"].numpy(),
        )
        return cls(model, param_std, slowness_norm, tuple(ckpt["mode_names"]))
