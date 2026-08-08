"""
Honest evaluation of an operator's re-gridding claim (M5f).

M5c reports that the cased forward operator is *self-consistent* under
re-gridding: evaluated on a finer grid, it agrees with itself at the shared
nodes to a fraction of a percent. That number is true, and it is not evidence
that the re-gridded prediction is **accurate** -- a model can be smoothly,
confidently wrong at every new frequency and still agree with itself perfectly.
Self-consistency is a necessary condition, not the claim anyone actually wants.

This module measures the claim people want, against the control that makes it
meaningful:

* :func:`true_curves_on_grid` re-runs fwap's layered solver at *arbitrary*
  frequencies, giving ground truth on a grid the model never trained on.
* :func:`evaluate_regridding` scores three things on that grid -- the operator's
  zero-shot prediction, the same operator's training-grid prediction simply
  **interpolated** onto the fine grid, and the operator's accuracy on its own
  training grid.

The middle one is the control, and it is the whole point. "The operator can be
evaluated at new frequencies" is only interesting if doing so beats the boring
alternative of predicting on the training grid and interpolating. For smooth
dispersion curves it often does not, and a library that ships the
super-resolution claim should ship the measurement that can refute it.

.. note::

   :func:`true_curves_on_grid` reconstructs the solver call from the stored
   formation parameters and annulus, so it is exact by construction -- called on
   the dataset's own grid it reproduces the stored labels bit for bit, which is
   what the accompanying test asserts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fwap import BoreholeLayer, stoneley_dispersion_layered

from sonic_ml.loader import DatasetBundle
from sonic_ml.models.cased import TrainedCasedOperator, cased_features

#: Slowness in s/m -> us/ft.
_US_PER_FT: float = 0.3048e6


def true_curves_on_grid(
    bundle: DatasetBundle,
    indices: np.ndarray,
    freq: np.ndarray,
) -> np.ndarray:
    """
    Recompute ground-truth cased Stoneley curves at arbitrary frequencies.

    Rebuilds each sample's solver call from its stored formation parameters and
    annulus, then evaluates :func:`fwap.stoneley_dispersion_layered` on ``freq``.
    This is what makes an off-grid accuracy claim checkable at all: without it,
    a re-gridded prediction has nothing to be compared against.

    Parameters
    ----------
    bundle : DatasetBundle
        A cased-hole dataset whose single mode is the Stoneley curve.
    indices : ndarray of int
        Samples to evaluate.
    freq : ndarray, shape (F',)
        Frequency grid (Hz) -- need not be the dataset's own.

    Returns
    -------
    ndarray, shape (len(indices), 1, F'), float64
        True per-mode slowness (s/m); ``NaN`` where the mode is unbound.

    Raises
    ------
    ValueError
        If the bundle is not a cased single-mode Stoneley dataset (the only
        case whose solver call can be reconstructed unambiguously).
    """
    if not bundle.is_cased or bundle.layer_params is None:
        raise ValueError(
            "true_curves_on_grid requires a cased-hole dataset (schema v4 with "
            "a non-empty layer stack); got an open-hole bundle"
        )
    if bundle.mode_names != ("Stoneley",):
        raise ValueError(
            "true_curves_on_grid can only reconstruct a single-mode cased "
            f"Stoneley dataset; got modes {bundle.mode_names}"
        )
    idx = np.asarray(indices, dtype=int)
    grid = np.asarray(freq, dtype=float)
    out = np.empty((idx.size, 1, grid.size), dtype=float)
    for k, i in enumerate(idx):
        layers = tuple(
            BoreholeLayer(vp=row[0], vs=row[1], rho=row[2], thickness=row[3])
            for row in bundle.layer_params[i]
        )
        params = {
            name: float(bundle.params[i, j])
            for j, name in enumerate(bundle.param_names)
        }
        out[k, 0] = stoneley_dispersion_layered(grid, **params, layers=layers).slowness
    return out


@dataclass(frozen=True)
class RegridScore:
    """
    Accuracy of an operator on a frequency grid it never trained on.

    Attributes
    ----------
    n_train_grid, n_eval_grid : int
        Frequency-sample counts of the training and evaluation grids.
    training_grid_mae : float
        Operator MAE (us/ft) on its own training grid -- the reference for how
        good this model is at all.
    zero_shot_mae : float
        Operator MAE (us/ft) evaluated natively on the evaluation grid.
    interpolated_mae : float
        MAE (us/ft) of the operator's *training-grid* prediction linearly
        interpolated onto the evaluation grid. The control: if this is no worse
        than :attr:`zero_shot_mae`, native re-gridding bought nothing.
    """

    n_train_grid: int
    n_eval_grid: int
    training_grid_mae: float
    zero_shot_mae: float
    interpolated_mae: float

    @property
    def zero_shot_beats_interpolation(self) -> bool:
        """``True`` if native evaluation is more accurate than interpolating."""
        return self.zero_shot_mae < self.interpolated_mae


def _mae_us_per_ft(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute slowness error (us/ft) over jointly finite entries."""
    mask = np.isfinite(pred) & np.isfinite(truth)
    if not mask.any():
        return float("nan")
    return float(np.abs(pred[mask] - truth[mask]).mean() * _US_PER_FT)


def evaluate_regridding(
    trained: TrainedCasedOperator,
    bundle: DatasetBundle,
    indices: np.ndarray,
    freq: np.ndarray,
) -> RegridScore:
    """
    Score an operator's off-grid accuracy against the interpolation control.

    Parameters
    ----------
    trained : TrainedCasedOperator
        A trained cased forward operator.
    bundle : DatasetBundle
        The cased-hole dataset it was trained on.
    indices : ndarray of int
        Held-out samples to score.
    freq : ndarray, shape (F',)
        Evaluation grid (Hz), typically finer than the training grid.

    Returns
    -------
    RegridScore
    """
    idx = np.asarray(indices, dtype=int)
    grid = np.asarray(freq, dtype=float)
    features, _ = cased_features(bundle)

    truth_eval = true_curves_on_grid(bundle, idx, grid)
    truth_train = bundle.slowness[idx]

    pred_train = trained.predict(features[idx])
    pred_eval = trained.predict(features[idx], freq=grid)

    # Control: predict on the training grid, then interpolate per sample/mode.
    interpolated = np.empty_like(pred_eval)
    for k in range(pred_train.shape[0]):
        for m in range(pred_train.shape[1]):
            interpolated[k, m] = np.interp(grid, trained.freq, pred_train[k, m])

    return RegridScore(
        n_train_grid=int(trained.freq.size),
        n_eval_grid=int(grid.size),
        training_grid_mae=_mae_us_per_ft(pred_train, truth_train),
        zero_shot_mae=_mae_us_per_ft(pred_eval, truth_eval),
        interpolated_mae=_mae_us_per_ft(interpolated, truth_eval),
    )


def format_regrid_score(score: RegridScore) -> str:
    """
    Render a :class:`RegridScore` as a short verdict table.

    Parameters
    ----------
    score : RegridScore

    Returns
    -------
    str
        Multi-line summary ending in an explicit verdict on whether native
        re-gridding beat the interpolation control.
    """
    lines = [
        f"Re-gridding accuracy: {score.n_train_grid} -> {score.n_eval_grid} "
        "frequency samples",
        f"  {'operator on its training grid':<38} {score.training_grid_mae:7.3f} us/ft",
        f"  {'operator zero-shot on eval grid':<38} {score.zero_shot_mae:7.3f} us/ft",
        f"  {'training-grid prediction, interpolated':<38} "
        f"{score.interpolated_mae:7.3f} us/ft  (control)",
    ]
    if score.zero_shot_beats_interpolation:
        lines.append("  -> native re-gridding beat the interpolation control")
    else:
        lines.append(
            "  -> native re-gridding did NOT beat interpolating the "
            "training-grid prediction"
        )
    return "\n".join(lines)
