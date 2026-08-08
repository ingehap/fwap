"""
Cement-bond scoring harness -- the bond-index counterpart of
:mod:`sonic_ml.bench.harness`.

The Vs harness is shear-velocity specific: it scores ``bundle.param("vs")`` and
splits by the slow/fast flexural regime. Cement-bond evaluation needs a
different target and a different stratification -- what matters is whether the
error is uniform across *bond quality*, since an estimator that only works on
well-bonded cement is useless for the job bond logs exist to do.

The :class:`~sonic_ml.bench.harness.Scorecard` / ``RegimeScore`` dataclasses and
the bootstrap machinery are reused unchanged; only the target and the regime
definition differ.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from fwap import ArrayGeometry

from sonic_ml.bench.harness import RegimeScore, Scorecard, _bootstrap_median_ci
from sonic_ml.loader import DatasetBundle

#: Bond-quality regime code -> human name. Thresholds are fixed thirds of the
#: ``[0, 1]`` bond index, so a regime means the same thing across datasets.
BOND_REGIME_NAMES: dict[int, str] = {0: "poor", 1: "medium", 2: "good"}

#: Row order used by :func:`format_bond_scorecard`.
BOND_ROW_ORDER: tuple[str, ...] = ("all", "poor", "medium", "good")


@runtime_checkable
class BondPredictor(Protocol):
    """
    Structural type for a cement-bond-index predictor.

    Implementations expose a ``name`` and a ``predict_bond`` returning a bond
    index in ``[0, 1]`` per requested sample, with ``NaN`` where the estimate
    is unavailable.
    """

    name: str

    def predict_bond(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray: ...


def bond_regime_labels(bundle: DatasetBundle) -> np.ndarray:
    """
    Per-sample bond-quality regime label.

    ``0`` = poor (``bond < 1/3``), ``1`` = medium, ``2`` = good
    (``bond >= 2/3``). Fixed thirds rather than data-dependent tertiles, so the
    same bond value always lands in the same regime.

    Parameters
    ----------
    bundle : DatasetBundle
        A cased-hole dataset carrying ``bond_index``.

    Returns
    -------
    ndarray, shape (N,), int

    Raises
    ------
    ValueError
        If the bundle is not a cased-hole dataset. Note an *open-hole* schema-v4
        bundle does carry a ``bond_index`` key -- it is simply all-``NaN`` -- so
        the check is on :attr:`~sonic_ml.loader.DatasetBundle.is_cased`, not on
        the key's presence; otherwise this would silently score ``NaN``.
    """
    if not bundle.is_cased or bundle.bond_index is None:
        raise ValueError(
            "bond_regime_labels requires a cased-hole dataset (schema v4 with a "
            "non-empty layer stack); got an open-hole bundle"
        )
    bond = np.asarray(bundle.bond_index, dtype=float)
    return np.digitize(bond, [1.0 / 3.0, 2.0 / 3.0]).astype(int)


def evaluate_bond(
    predictor: BondPredictor,
    bundle: DatasetBundle,
    geom: ArrayGeometry,
    *,
    indices: np.ndarray | None = None,
    seed: int = 0,
    n_boot: int = 1000,
) -> Scorecard:
    """
    Score a bond predictor against the stored truth, split by bond quality.

    Parameters
    ----------
    predictor : BondPredictor
        Any object satisfying the :class:`BondPredictor` protocol.
    bundle : DatasetBundle
        A cased-hole dataset holding the ``bond_index`` truth.
    geom : fwap.ArrayGeometry
        Acquisition geometry the gathers were built with.
    indices : ndarray of int or None
        Samples to score (e.g. a held-out split). ``None`` scores all.
    seed : int, default 0
        Bootstrap seed (reproducible CIs).
    n_boot : int, default 1000
        Bootstrap resample count.

    Returns
    -------
    Scorecard
        ``parameter`` is ``"bond_index"``; rows are ``"all"`` plus whichever of
        ``"poor"`` / ``"medium"`` / ``"good"`` are present.
    """
    if not bundle.is_cased or bundle.bond_index is None:
        raise ValueError(
            "evaluate_bond requires a cased-hole dataset (schema v4 with a "
            "non-empty layer stack); got an open-hole bundle"
        )
    idx = (
        np.arange(bundle.n_samples)
        if indices is None
        else np.asarray(indices, dtype=int)
    )

    pred = np.asarray(predictor.predict_bond(bundle, geom, idx), dtype=float)
    truth = np.asarray(bundle.bond_index, dtype=float)[idx]
    labels = bond_regime_labels(bundle)[idx]
    abs_err = np.abs(pred - truth)

    groups: dict[str, np.ndarray] = {"all": np.ones(idx.size, dtype=bool)}
    for code, name in BOND_REGIME_NAMES.items():
        mask = labels == code
        if mask.any():
            groups[name] = mask

    rng = np.random.default_rng(seed)
    per_regime: dict[str, RegimeScore] = {}
    for name, mask in groups.items():
        err = abs_err[mask]
        finite = err[np.isfinite(err)]
        median, lo, hi = _bootstrap_median_ci(err, rng, n_boot=n_boot)
        per_regime[name] = RegimeScore(
            regime=name,
            n=int(mask.sum()),
            n_finite=int(finite.size),
            median_abs_error=median,
            mean_abs_error=float(finite.mean()) if finite.size else float("nan"),
            ci_low=lo,
            ci_high=hi,
        )

    return Scorecard(
        predictor=predictor.name,
        parameter="bond_index",
        per_regime=per_regime,
    )


class MeanBondPredictor:
    """
    Predict a constant bond index -- the "no skill" reference.

    A bond estimator that cannot beat the training mean has learned nothing, so
    every reported skill number is quoted against this.

    Parameters
    ----------
    value : float
        The constant to predict (typically the training-split mean).
    name : str
        Predictor name shown on the scorecard.
    """

    def __init__(self, value: float, *, name: str = "mean_bond") -> None:
        self.value = float(value)
        self.name = name

    def predict_bond(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Return the constant bond value for every requested sample."""
        return np.full(np.asarray(indices, dtype=int).size, self.value)
