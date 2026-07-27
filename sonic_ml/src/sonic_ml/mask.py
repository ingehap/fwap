"""
Masking and class-imbalance helpers.

Two easy-to-conflate quantities live here, kept deliberately separate:

* :func:`finite_mask` -- per-(sample, mode, frequency) validity of the
  ``slowness`` label, for masking the regression loss so the network never
  trains on ``NaN`` sentinels.
* :func:`presence_labels` -- per-(sample, mode) whether a mode was actually
  injected into the gather, read straight from ``mode_in_gather``. This is the
  authoritative presence target; deriving it from ``isfinite(slowness).any()``
  is WRONG because a partly-finite curve with fewer than the generator's
  ``min_finite`` points is still absent from the waveform.
"""

from __future__ import annotations

import numpy as np


def finite_mask(slowness: np.ndarray) -> np.ndarray:
    """
    Boolean mask of finite slowness samples.

    Parameters
    ----------
    slowness : ndarray, shape (N, M, F)
        Per-mode slowness curves (``NaN`` where a mode is absent).

    Returns
    -------
    ndarray, shape (N, M, F), bool
    """
    return np.isfinite(slowness)


def presence_labels(mode_in_gather: np.ndarray) -> np.ndarray:
    """
    Authoritative per-mode presence labels.

    This is a named passthrough of ``mode_in_gather`` -- use it (never
    ``isfinite(slowness).any(axis=-1)``) to supervise a presence head.

    Parameters
    ----------
    mode_in_gather : ndarray, shape (N, M)
        The generator's per-mode injection flags.

    Returns
    -------
    ndarray, shape (N, M), bool
    """
    return np.asarray(mode_in_gather, dtype=bool)


def inverse_frequency_weights(
    labels: np.ndarray,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    """
    Inverse-frequency class weights for imbalance reweighting.

    Weight of class ``c`` is ``n_total / (n_classes * count[c])`` (the standard
    balanced weighting), so rarer classes weigh more and the weights average to
    ``1``. Absent classes get weight ``0``.

    Parameters
    ----------
    labels : ndarray of int, shape (N,)
        Integer class labels in ``[0, num_classes)``.
    num_classes : int or None
        Number of classes. ``None`` infers ``labels.max() + 1``.

    Returns
    -------
    ndarray, shape (num_classes,), float64
        Per-class weights.

    Raises
    ------
    ValueError
        If ``labels`` is empty or contains negative values.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("labels must be non-empty")
    if labels.min() < 0:
        raise ValueError("labels must be non-negative integers")
    k = int(labels.max()) + 1 if num_classes is None else int(num_classes)
    counts = np.bincount(labels, minlength=k).astype(float)
    weights = np.zeros(k, dtype=float)
    nonzero = counts > 0
    weights[nonzero] = labels.size / (k * counts[nonzero])
    return weights


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean of ``values`` over ``True`` positions of ``mask``.

    Parameters
    ----------
    values : ndarray
        Values to average.
    mask : ndarray of bool
        Same shape as ``values``.

    Returns
    -------
    float
        ``nan`` if the mask selects nothing.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != np.shape(values):
        raise ValueError("values and mask must have the same shape")
    if not mask.any():
        return float("nan")
    return float(np.asarray(values)[mask].mean())
