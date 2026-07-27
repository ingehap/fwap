"""
Formation-parameter standardization with a zero-variance guard.

Under the generator's default priors ``vf`` and ``rho_f`` are constants
(``FormationPriors`` fixes the borehole-fluid velocity and density), so their
column standard deviation is exactly zero. Naive ``(x - mean) / std`` would
divide by zero and emit ``NaN``/``inf`` features. :class:`Standardizer`
detects those columns, marks them inactive, and drops them from the
transformed output -- the network sees only the genuinely varying parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    """
    Fitted per-column mean/std standardizer over formation parameters.

    Attributes
    ----------
    mean : ndarray, shape (P,), float64
        Per-column means (all columns, including inactive ones).
    std : ndarray, shape (P,), float64
        Per-column standard deviations, with inactive (zero-variance)
        columns replaced by ``1.0`` so division is always safe.
    active : ndarray, shape (P,), bool
        ``True`` for columns with non-zero variance. Inactive columns are
        dropped by :meth:`transform`.
    names : tuple of str, length P
        Column names in ``params`` order.
    """

    mean: np.ndarray
    std: np.ndarray
    active: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        params: np.ndarray,
        names: tuple[str, ...],
        *,
        tol: float = 0.0,
    ) -> Standardizer:
        """
        Fit a standardizer to a ``(N, P)`` parameter matrix.

        Parameters
        ----------
        params : ndarray, shape (N, P), float
            Formation parameters.
        names : tuple of str, length P
            Column names (from ``DatasetBundle.param_names``).
        tol : float, default 0.0
            A column is treated as constant (inactive) when its standard
            deviation is ``<= tol``. The default keys off exact zero
            variance; raise it to also drop near-constant columns.

        Returns
        -------
        Standardizer

        Raises
        ------
        ValueError
            If ``params`` is not 2-D or its column count disagrees with
            ``names``.
        """
        params = np.asarray(params, dtype=float)
        if params.ndim != 2:
            raise ValueError(f"params must be 2-D, got shape {params.shape}")
        if params.shape[1] != len(names):
            raise ValueError(
                f"params has {params.shape[1]} columns but {len(names)} names"
            )
        mean = params.mean(axis=0)
        std = params.std(axis=0)
        active = std > tol
        safe_std = np.where(active, std, 1.0)
        return cls(mean=mean, std=safe_std, active=active, names=tuple(names))

    @property
    def active_names(self) -> tuple[str, ...]:
        """Names of the kept (active) columns, in order."""
        return tuple(n for n, a in zip(self.names, self.active, strict=True) if a)

    @property
    def n_active(self) -> int:
        """Number of active (kept) columns."""
        return int(self.active.sum())

    def transform(self, params: np.ndarray) -> np.ndarray:
        """
        Standardize and drop inactive columns.

        Parameters
        ----------
        params : ndarray, shape (N, P), float

        Returns
        -------
        ndarray, shape (N, n_active), float64
            ``(params - mean) / std`` restricted to the active columns.
        """
        params = np.asarray(params, dtype=float)
        if params.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"expected {self.mean.shape[0]} columns, got {params.shape[1]}"
            )
        z = (params - self.mean) / self.std
        return z[:, self.active]

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Map standardized active columns back to raw parameter values.

        Parameters
        ----------
        z : ndarray, shape (N, n_active), float

        Returns
        -------
        ndarray, shape (N, n_active), float64
            Raw values of the active columns (inactive constants are not
            reconstructed -- they carry no information).
        """
        z = np.asarray(z, dtype=float)
        if z.shape[1] != self.n_active:
            raise ValueError(f"expected {self.n_active} columns, got {z.shape[1]}")
        return z * self.std[self.active] + self.mean[self.active]
