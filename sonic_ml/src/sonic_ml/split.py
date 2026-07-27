"""
Regime-stratified train / validation / test splitting.

The generator's default prior straddles the borehole-fluid velocity, so a
dataset mixes *slow* (``vs < vf``) and *fast* (``vs > vf``) formations -- the
two flexural regimes with different dispersion physics. A naive random split
can under-represent a regime in the held-out set and quietly bias every
reported metric. :func:`stratified_split` splits *within* each regime and
concatenates, so train/val/test carry the same slow/fast mix, and stores the
integer index arrays so the exact partition is reproducible and bindable to a
provenance record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sonic_ml.loader import DatasetBundle


def regime_labels(bundle: DatasetBundle) -> np.ndarray:
    """
    Per-sample slow/fast regime label.

    ``0`` = slow (``vs < vf``), ``1`` = fast (``vs >= vf``), using each
    sample's own ``vf`` column (constant under default priors, but read
    per-sample so it stays correct if the prior varies it).

    Parameters
    ----------
    bundle : DatasetBundle

    Returns
    -------
    ndarray, shape (N,), int
    """
    vs = bundle.param("vs")
    vf = bundle.param("vf")
    return (vs >= vf).astype(int)


@dataclass(frozen=True)
class DataSplit:
    """
    Disjoint train / validation / test index arrays into a dataset.

    Attributes
    ----------
    train, val, test : ndarray of int
        Sorted, disjoint indices whose union is ``range(N)``.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        """Return the split as plain Python lists (JSON-friendly)."""
        return {
            "train": self.train.tolist(),
            "val": self.val.tolist(),
            "test": self.test.tolist(),
        }

    def to_json(self) -> str:
        """Serialize the split indices to a JSON string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, text: str) -> DataSplit:
        """Rebuild a :class:`DataSplit` from :meth:`to_json` output."""
        d = json.loads(text)
        return cls(
            train=np.asarray(d["train"], dtype=int),
            val=np.asarray(d["val"], dtype=int),
            test=np.asarray(d["test"], dtype=int),
        )

    def write(self, path: str) -> Path:
        """Write the split to ``path`` as JSON; returns the path."""
        p = Path(path)
        p.write_text(self.to_json())
        return p

    @classmethod
    def read(cls, path: str) -> DataSplit:
        """Read a split JSON written by :meth:`write`."""
        return cls.from_json(Path(path).read_text())


def stratified_split(
    bundle: DatasetBundle,
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> DataSplit:
    """
    Regime-stratified train/val/test split.

    Splits the slow and fast subsets independently in the given proportions
    and concatenates, so every partition preserves the slow/fast balance. The
    partition is deterministic given ``seed``.

    Parameters
    ----------
    bundle : DatasetBundle
    fractions : (float, float, float), default (0.8, 0.1, 0.1)
        Train / val / test fractions; must be positive and sum to ~1.
    seed : int, default 0
        Seed for the shuffling RNG.

    Returns
    -------
    DataSplit
        Sorted, disjoint index arrays covering every sample exactly once.

    Raises
    ------
    ValueError
        If ``fractions`` are non-positive or do not sum to 1.
    """
    f_train, f_val, f_test = fractions
    if min(fractions) <= 0:
        raise ValueError(f"fractions must be positive, got {fractions}")
    if not np.isclose(f_train + f_val + f_test, 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

    labels = regime_labels(bundle)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for regime in np.unique(labels):
        idx = np.flatnonzero(labels == regime)
        rng.shuffle(idx)
        n = idx.size
        n_train = int(round(f_train * n))
        n_val = int(round(f_val * n))
        # Give any rounding remainder to train (the largest split).
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])

    return DataSplit(
        train=np.sort(np.concatenate(train_parts)) if train_parts else _empty(),
        val=np.sort(np.concatenate(val_parts)) if val_parts else _empty(),
        test=np.sort(np.concatenate(test_parts)) if test_parts else _empty(),
    )


def _empty() -> np.ndarray:
    """An empty int index array."""
    return np.empty(0, dtype=int)
