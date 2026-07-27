"""Losses for the forward surrogate."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_slowness_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Smooth-L1 (Huber) slowness loss over finite entries only.

    Parameters
    ----------
    pred, target : Tensor, shape (B, M, F)
        Predicted and target (normalized) slowness.
    mask : Tensor, shape (B, M, F), bool
        ``True`` where the target slowness is finite. The loss is averaged
        over these entries only, so ``NaN`` sentinels never contribute.
    beta : float, default 1.0
        Huber transition point.

    Returns
    -------
    Tensor
        Scalar loss; ``0`` if the mask selects nothing.
    """
    if not bool(mask.any()):
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[mask], target[mask], beta=beta)


def presence_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy for the per-mode presence head.

    Parameters
    ----------
    logits : Tensor, shape (B, M)
        Raw presence logits.
    target : Tensor, shape (B, M), float
        ``mode_in_gather`` labels (0/1).

    Returns
    -------
    Tensor
        Scalar loss.
    """
    return F.binary_cross_entropy_with_logits(logits, target)
