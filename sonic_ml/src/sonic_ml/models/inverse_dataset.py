"""
Torch dataset adapter for the inverse net (gather -> formation parameters).

Input is the raw multi-receiver waveform ``gather`` (the inverse-net input),
per-gather amplitude-normalized so the network keys on moveout/phase rather
than absolute amplitude (which the modal law does not encode). Targets are the
*varying* standardized formation parameters (the constant ``vf``/``rho_f`` are
dropped by the :class:`~sonic_ml.normalize.Standardizer`).

Crucially the slowness label is **never** provided here -- the inverse net sees
only the gather, so it must close the gather-to-parameters gap itself rather
than reading the dispersion curve.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from sonic_ml.loader import DatasetBundle
from sonic_ml.normalize import Standardizer


def normalize_gathers(gather: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """
    Per-gather amplitude standardization.

    Parameters
    ----------
    gather : ndarray, shape (N, R, T)
        Raw waveforms.
    eps : float
        Guard against a zero-variance (silent) gather.

    Returns
    -------
    ndarray, shape (N, R, T), float64
        Each gather shifted to zero mean and scaled to unit std over its
        ``(R, T)`` samples.
    """
    gather = np.asarray(gather, dtype=float)
    mean = gather.mean(axis=(1, 2), keepdims=True)
    std = gather.std(axis=(1, 2), keepdims=True)
    return (gather - mean) / (std + eps)


class InverseDataset(Dataset):
    """
    Tensor view of a bundle for training the inverse net.

    Each item is ``(x, y)``:

    * ``x`` -- amplitude-normalized gather, shape ``(R, T)``, float32
      (receivers as channels for a 1-D CNN over the time axis).
    * ``y`` -- standardized active formation parameters, shape ``(n_active,)``,
      float32.

    Parameters
    ----------
    bundle : DatasetBundle
    indices : ndarray of int
        Sample indices to include.
    param_std : Standardizer
        Fitted on the *training* parameters.
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray,
        param_std: Standardizer,
    ) -> None:
        idx = np.asarray(indices, dtype=int)
        x = normalize_gathers(bundle.gather[idx])
        y = param_std.transform(bundle.params[idx])
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
