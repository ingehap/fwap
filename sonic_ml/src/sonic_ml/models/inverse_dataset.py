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
from sonic_ml.models.augment import Augmentation
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
    augment : Augmentation or None, default None
        If set, each ``__getitem__`` returns a freshly augmented+normalized
        gather (stochastic per access); ``None`` returns the deterministic
        normalized gather. Use only for the *training* split.
    augment_seed : int, default 0
        Seed for the augmentation RNG (reproducible given a fixed access
        order, e.g. a seeded ``DataLoader``).
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray,
        param_std: Standardizer,
        *,
        augment: Augmentation | None = None,
        augment_seed: int = 0,
    ) -> None:
        idx = np.asarray(indices, dtype=int)
        self._raw = np.asarray(bundle.gather[idx], dtype=float)
        # Precompute the non-augmented normalized view (fast path + ``.x`` API).
        self.x = torch.as_tensor(normalize_gathers(self._raw), dtype=torch.float32)
        self.y = torch.as_tensor(
            param_std.transform(bundle.params[idx]), dtype=torch.float32
        )
        self.augment = augment
        self._rng = np.random.default_rng(augment_seed)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment is None:
            return self.x[index], self.y[index]
        g = self.augment.apply(self._raw[index], self._rng)
        g = normalize_gathers(g[None, ...])[0]
        return torch.as_tensor(g, dtype=torch.float32), self.y[index]
