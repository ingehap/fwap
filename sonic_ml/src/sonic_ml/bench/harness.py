"""
Model-agnostic scoring harness for shear-velocity (Vs) predictors.

A :class:`Predictor` is anything that maps a dataset (+ tool geometry) to a Vs
estimate per sample -- a classical STC baseline today, a trained network later.
:func:`evaluate` scores one against the stored truth, split by slow/fast
regime, and reports the median absolute error with a bootstrap confidence
interval. Because the classical baseline and a future ML model implement the
same protocol, the scorecard compares them on identical held-out samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from fwap import ArrayGeometry

from sonic_ml.loader import DatasetBundle
from sonic_ml.split import regime_labels

#: Regime code -> human name (matches :func:`sonic_ml.split.regime_labels`).
REGIME_NAMES: dict[int, str] = {0: "slow", 1: "fast"}


@runtime_checkable
class Predictor(Protocol):
    """
    Structural type for a Vs predictor.

    Implementations expose a ``name`` and a ``predict_vs`` returning a
    shear-velocity estimate (m/s) per requested sample index, with ``NaN``
    where the estimate is unavailable.
    """

    name: str

    def predict_vs(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class RegimeScore:
    """
    Error summary for one regime (or ``"all"``).

    Attributes
    ----------
    regime : str
        ``"slow"``, ``"fast"`` or ``"all"``.
    n : int
        Samples in this group.
    n_finite : int
        Samples with a finite prediction (the rest failed / returned ``NaN``).
    median_abs_error : float
        Median ``|Vs_pred - Vs_true|`` (m/s) over finite predictions; ``nan``
        if none are finite.
    mean_abs_error : float
        Mean absolute error (m/s) over finite predictions.
    ci_low, ci_high : float
        Bootstrap 95% confidence interval on the median absolute error.
    """

    regime: str
    n: int
    n_finite: int
    median_abs_error: float
    mean_abs_error: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class Scorecard:
    """
    A predictor's per-regime error report for one parameter.

    Attributes
    ----------
    predictor : str
        The predictor's ``name``.
    parameter : str
        The scored parameter (``"vs"``).
    per_regime : dict of str to RegimeScore
        Keys ``"all"``, ``"slow"``, ``"fast"`` (regimes present in the data).
    """

    predictor: str
    parameter: str
    per_regime: dict[str, RegimeScore]


def _bootstrap_median_ci(
    errors: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Median of ``errors`` plus a bootstrap ``(1 - alpha)`` CI on the median."""
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    medians = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(finite, size=finite.size, replace=True)
        medians[b] = np.median(sample)
    lo, hi = np.percentile(medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(finite)), float(lo), float(hi)


def evaluate(
    predictor: Predictor,
    bundle: DatasetBundle,
    geom: ArrayGeometry,
    *,
    indices: np.ndarray | None = None,
    seed: int = 0,
    n_boot: int = 1000,
) -> Scorecard:
    """
    Score a predictor's Vs estimates against the dataset truth, by regime.

    Parameters
    ----------
    predictor : Predictor
        Any object satisfying the :class:`Predictor` protocol.
    bundle : DatasetBundle
        The dataset holding truth (``params``) and inputs (``gather``).
    geom : fwap.ArrayGeometry
        Acquisition geometry the gathers were built with (see
        :func:`sonic_ml.geometry.default_geometry`).
    indices : ndarray of int or None
        Sample indices to score (e.g. a held-out test split). ``None`` scores
        all samples.
    seed : int, default 0
        Seed for the bootstrap resampling (reproducible CIs).
    n_boot : int, default 1000
        Bootstrap resample count.

    Returns
    -------
    Scorecard
    """
    n = bundle.n_samples
    idx = np.arange(n) if indices is None else np.asarray(indices, dtype=int)

    vs_pred = np.asarray(predictor.predict_vs(bundle, geom, idx), dtype=float)
    vs_true = bundle.param("vs")[idx]
    labels = regime_labels(bundle)[idx]
    abs_err = np.abs(vs_pred - vs_true)

    groups: dict[str, np.ndarray] = {"all": np.ones(idx.size, dtype=bool)}
    for code, name in REGIME_NAMES.items():
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

    return Scorecard(predictor=predictor.name, parameter="vs", per_regime=per_regime)


class StubPredictor:
    """
    A trivial predictor, for exercising the harness independently of physics.

    Parameters
    ----------
    vs_value : float or None
        If a float, predicts that constant Vs for every sample. If ``None``,
        returns the ground-truth Vs (a "perfect" predictor -> zero error),
        which lets tests assert the harness scores predictions faithfully.
    name : str
        Predictor name shown on the scorecard.
    """

    def __init__(self, vs_value: float | None = None, *, name: str = "stub") -> None:
        self.vs_value = vs_value
        self.name = name

    def predict_vs(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray:
        idx = np.asarray(indices, dtype=int)
        if self.vs_value is None:
            return bundle.param("vs")[idx].astype(float).copy()
        return np.full(idx.size, float(self.vs_value))
