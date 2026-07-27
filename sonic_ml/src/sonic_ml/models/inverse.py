"""
DL-FWI inverse net: waveform gather -> formation parameters.

A 1-D CNN over the multi-receiver gather (receivers as channels, time as the
convolved axis) with a **heteroscedastic** head that predicts a mean *and* a
log-variance per parameter. Trained with a Gaussian NLL, so weakly identifiable
parameters (vp / rho) are reported as wide-but-calibrated error bars rather
than falsely precise point estimates.

The net's input is only the gather (never the dispersion label), so it must
close the modal-vs-processing gap the classical baseline cannot -- the whole
justification for a learned inverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from sonic_ml.models.inverse_dataset import normalize_gathers
from sonic_ml.normalize import Standardizer


def _num_groups(channels: int) -> int:
    """Largest group count in {8,4,2,1} that divides ``channels``."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
    """Strided Conv1d -> GroupNorm -> GELU -> Dropout (halves the time axis)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=2, padding=kernel // 2)
        self.norm = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class InverseNet(nn.Module):
    """
    1-D CNN inverse net with a heteroscedastic parameter head.

    Parameters
    ----------
    n_rec : int
        Number of receivers (input channels).
    n_targets : int
        Number of (active) formation parameters to predict.
    channels : tuple of int, default (32, 64, 128)
        Output channels of each strided conv block.
    kernel : int, default 7
        Conv kernel size.
    hidden : int, default 128
        Width of the post-pool MLP.
    dropout : float, default 0.0
        Dropout in the conv blocks and the MLP.
    """

    def __init__(
        self,
        n_rec: int,
        n_targets: int,
        *,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel: int = 7,
        hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hparams: dict[str, Any] = {
            "n_rec": n_rec,
            "n_targets": n_targets,
            "channels": list(channels),
            "kernel": kernel,
            "hidden": hidden,
            "dropout": dropout,
        }
        self.n_targets = n_targets
        blocks: list[nn.Module] = []
        in_ch = n_rec
        for out_ch in channels:
            blocks.append(_ConvBlock(in_ch, out_ch, kernel, dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.mean_head = nn.Linear(hidden, n_targets)
        self.logvar_head = nn.Linear(hidden, n_targets)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map a gather batch to (mean, log-variance) per target.

        Parameters
        ----------
        x : Tensor, shape (B, n_rec, T)

        Returns
        -------
        mean : Tensor, shape (B, n_targets)
        log_var : Tensor, shape (B, n_targets)
        """
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        h = self.mlp(h)
        return self.mean_head(h), self.logvar_head(h)


class TrainedInverseNet:
    """
    A trained :class:`InverseNet` with its parameter standardizer.

    Predicts raw formation parameters (the active columns) with per-parameter
    uncertainty, and serializes to a checkpoint.

    Parameters
    ----------
    model : InverseNet
    param_std : Standardizer
        Fitted parameter standardizer (defines the active columns / order).
    """

    def __init__(self, model: InverseNet, param_std: Standardizer) -> None:
        self.model = model
        self.param_std = param_std
        self.active_names = param_std.active_names

    @torch.no_grad()
    def predict(self, gather: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict raw active parameters and their (raw-unit) std.

        Parameters
        ----------
        gather : ndarray, shape (N, R, T)
            Raw waveforms (amplitude-normalized internally).

        Returns
        -------
        params : ndarray, shape (N, n_active), float64
            Predicted active formation parameters, in
            :attr:`active_names` order.
        std : ndarray, shape (N, n_active), float64
            Per-parameter standard deviation in raw units.
        """
        self.model.eval()
        x = normalize_gathers(np.asarray(gather, dtype=float))
        xt = torch.as_tensor(x, dtype=torch.float32)
        mean, log_var = self.model(xt)
        params = self.param_std.inverse_transform(mean.cpu().numpy().astype(float))
        active_std = self.param_std.std[self.param_std.active]
        std = np.exp(0.5 * log_var.cpu().numpy().astype(float)) * active_std[None, :]
        return params, std

    def save(self, path: str) -> None:
        """Serialize the model + standardizer to ``path`` (weights-only safe)."""
        checkpoint: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "hparams": dict(self.model.hparams),
            "param_mean": torch.as_tensor(self.param_std.mean),
            "param_std": torch.as_tensor(self.param_std.std),
            "param_active": torch.as_tensor(self.param_std.active),
            "param_names": list(self.param_std.names),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, *, map_location: str = "cpu") -> TrainedInverseNet:
        """Load a checkpoint written by :meth:`save`."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        hp = ckpt["hparams"]
        model = InverseNet(
            int(hp["n_rec"]),
            int(hp["n_targets"]),
            channels=tuple(int(c) for c in hp["channels"]),
            kernel=int(hp["kernel"]),
            hidden=int(hp["hidden"]),
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
        return cls(model, param_std)
