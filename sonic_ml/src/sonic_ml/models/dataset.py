"""
Torch dataset adapter for the forward surrogate.

Turns a :class:`~sonic_ml.loader.DatasetBundle` into standardized tensors:
formation parameters (via a fitted :class:`~sonic_ml.normalize.Standardizer`,
which drops the constant ``vf``/``rho_f`` columns) as the input, and per-mode
slowness curves (per-mode standardized) plus the finite mask and the
``mode_in_gather`` presence labels as targets. The finite mask travels with
each sample so the loss is only ever computed on real slowness values, never on
the ``NaN`` sentinels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sonic_ml.loader import DatasetBundle
from sonic_ml.normalize import Standardizer


@dataclass(frozen=True)
class SlownessNormalizer:
    """
    Per-mode standardization of slowness curves over finite entries.

    Attributes
    ----------
    mean, std : ndarray, shape (M,), float64
        Per-mode mean and standard deviation of the finite slowness values.
        Modes with no finite samples get ``mean = 0``, ``std = 1``.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, slowness: np.ndarray, *, tol: float = 1e-30) -> SlownessNormalizer:
        """
        Fit per-mode statistics over finite slowness entries.

        Parameters
        ----------
        slowness : ndarray, shape (N, M, F)
            Per-mode slowness curves (``NaN`` where a mode is absent).
        tol : float
            Standard deviations ``<= tol`` are replaced by ``1.0`` (a mode
            with a single finite value, or none).

        Returns
        -------
        SlownessNormalizer
        """
        slowness = np.asarray(slowness, dtype=float)
        n_modes = slowness.shape[1]
        mean = np.zeros(n_modes, dtype=float)
        std = np.ones(n_modes, dtype=float)
        for m in range(n_modes):
            finite = slowness[:, m, :][np.isfinite(slowness[:, m, :])]
            if finite.size:
                mean[m] = float(finite.mean())
                s = float(finite.std())
                std[m] = s if s > tol else 1.0
        return cls(mean=mean, std=std)

    def transform(self, slowness: np.ndarray) -> np.ndarray:
        """Standardize ``(N, M, F)`` slowness per mode (``NaN`` propagates)."""
        return (slowness - self.mean[None, :, None]) / self.std[None, :, None]

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform` back to raw slowness (s/m)."""
        return z * self.std[None, :, None] + self.mean[None, :, None]


class ForwardDataset(Dataset):
    """
    Tensor view of a bundle for training the forward surrogate.

    Each item is ``(x, target, mask, presence)``:

    * ``x`` -- standardized active parameters, shape ``(n_active,)``, float32.
    * ``target`` -- per-mode standardized slowness, shape ``(M, F)``, float32
      (masked entries zeroed).
    * ``mask`` -- finite-slowness mask, shape ``(M, F)``, bool.
    * ``presence`` -- ``mode_in_gather`` labels, shape ``(M,)``, float32.

    Parameters
    ----------
    bundle : DatasetBundle
    indices : ndarray of int
        Sample indices to include (e.g. a train or val split).
    param_std : Standardizer
        Fitted on the *training* parameters.
    slowness_norm : SlownessNormalizer
        Fitted on the *training* slowness.
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray,
        param_std: Standardizer,
        slowness_norm: SlownessNormalizer,
    ) -> None:
        idx = np.asarray(indices, dtype=int)
        x = param_std.transform(bundle.params[idx])
        slow = bundle.slowness[idx]
        mask = np.isfinite(slow)
        target = np.nan_to_num(slowness_norm.transform(slow), nan=0.0)
        presence = bundle.mode_in_gather[idx].astype(np.float32)

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.presence = torch.as_tensor(presence, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.target[index], self.mask[index], self.presence[index]
