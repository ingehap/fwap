"""
Microannulus gap-width scoring harness -- the debonded-regime counterpart of
:mod:`sonic_ml.bench.harness` and :mod:`sonic_ml.bench.bond` (G.6).

Two things differ from its siblings, and both are deliberate.

**It scores in log10 metres, not metres.** The gap spans two decades by
construction (``MicroannulusPriors`` draws it log-uniformly over 10 um-1 mm),
so a median absolute error in metres would be set almost entirely by the
widest samples and would say nothing about the narrow ones -- which are the
operationally interesting end, where a microannulus is a hair-line gap rather
than a frank channel. Errors are therefore reported in **decades**, and
:func:`format_thickness_scorecard` also prints them as a percentage of
thickness, which is the form worth quoting. This matches the ratio-domain
scoring
:meth:`~sonic_ml.baselines.debond.CrackWaveThicknessBaseline.score` already
uses; what the harness adds is bootstrap confidence intervals, per-regime
rows, and a report -- the same treatment every other predictor in this layer
gets.

**It takes no acquisition geometry.** :class:`~sonic_ml.bench.harness.Predictor`
and :class:`~sonic_ml.bench.bond.BondPredictor` both receive an
:class:`fwap.ArrayGeometry` because they read the gather. A gap-width
estimator does not: the cased Stoneley mode moves 0.05 % across the whole
prior, so there is nothing in the waveform to read, and both estimators work
from the crack-wave *dispersion curve* instead (see
:mod:`sonic_ml.models.debond`). Passing a geometry they cannot use would
misdescribe what is being measured, so :class:`ThicknessPredictor` omits it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from sonic_ml.baselines.debond import (
    GAP_LAYER_NAME,
    CrackWaveThicknessBaseline,
    gap_thickness,
)
from sonic_ml.bench.harness import RegimeScore, Scorecard, _bootstrap_median_ci
from sonic_ml.loader import DatasetBundle

#: Gap width (m) separating the two scored regimes. A decade boundary rather
#: than a tuned threshold: it bisects the generator's log-uniform prior, and
#: decades are the unit both the prior and the scoring work in.
GAP_REGIME_EDGE_M: float = 1.0e-4

#: Gap-regime code -> human name.
GAP_REGIME_NAMES: dict[int, str] = {0: "tight", 1: "wide"}

#: Row order used by :func:`format_thickness_scorecard`.
GAP_ROW_ORDER: tuple[str, ...] = ("all", "tight", "wide")


@runtime_checkable
class ThicknessPredictor(Protocol):
    """
    Structural type for a microannulus gap-width predictor.

    Implementations expose a ``name`` and a ``predict_log_thickness``
    returning ``log10`` of the gap width in metres per requested sample, with
    ``NaN`` where the estimate is unavailable.

    Log10 rather than metres is part of the protocol, not a formatting choice:
    it is the scale the target is drawn on and scored on, and it keeps a
    predictor from having to round-trip through a quantity that spans two
    decades.
    """

    name: str

    def predict_log_thickness(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray | None = None,
    ) -> np.ndarray: ...


def gap_regime_labels(
    bundle: DatasetBundle, *, gap_name: str = GAP_LAYER_NAME
) -> np.ndarray:
    """
    Per-sample gap-width regime label.

    ``0`` = tight (``h < 100 um``), ``1`` = wide. A fixed edge rather than a
    data-dependent median, so the same gap always lands in the same regime and
    two scorecards from different datasets can be read side by side.

    Parameters
    ----------
    bundle : DatasetBundle
        A debonded dataset carrying a ``gap_name`` layer.
    gap_name : str, default ``"microannulus"``

    Returns
    -------
    ndarray, shape (N,), int

    Raises
    ------
    ValueError
        If the bundle is not a debonded dataset, via
        :func:`~sonic_ml.baselines.debond.gap_thickness`.
    """
    return np.digitize(
        gap_thickness(bundle, gap_name=gap_name), [GAP_REGIME_EDGE_M]
    ).astype(int)


def evaluate_thickness(
    predictor: ThicknessPredictor,
    bundle: DatasetBundle,
    *,
    indices: np.ndarray | None = None,
    gap_name: str = GAP_LAYER_NAME,
    seed: int = 0,
    n_boot: int = 1000,
) -> Scorecard:
    """
    Score a gap-width predictor against the drawn truth, split by gap width.

    Parameters
    ----------
    predictor : ThicknessPredictor
        Any object satisfying the :class:`ThicknessPredictor` protocol --
        :class:`KrauklisThicknessPredictor`,
        :class:`MeanThicknessPredictor`, or a trained
        :class:`~sonic_ml.models.debond.TrainedDebondInverse`.
    bundle : DatasetBundle
        A debonded dataset (``gen_surrogate_dataset.py --debonded``).
    indices : ndarray of int or None
        Samples to score (e.g. a held-out split). ``None`` scores all.
    gap_name : str, default ``"microannulus"``
        Which layer is the fluid gap.
    seed : int, default 0
        Bootstrap seed (reproducible CIs).
    n_boot : int, default 1000
        Bootstrap resample count.

    Returns
    -------
    Scorecard
        ``parameter`` is ``"log10_thickness"`` and every error column is in
        **decades**; rows are ``"all"`` plus whichever of ``"tight"`` /
        ``"wide"`` are present.

    Raises
    ------
    ValueError
        If the bundle is not a debonded dataset, or the predictor returns one
        estimate per *bundle* sample when a subset was requested.
    """
    truth_h = gap_thickness(bundle, gap_name=gap_name)
    idx = (
        np.arange(bundle.n_samples)
        if indices is None
        else np.asarray(indices, dtype=int)
    )

    pred = np.asarray(predictor.predict_log_thickness(bundle, idx), dtype=float)
    if pred.shape != idx.shape:
        raise ValueError(
            f"{predictor.name!r} returned {pred.shape} estimates for "
            f"{idx.shape} requested samples; a ThicknessPredictor must honour "
            "`indices`"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        truth = np.log10(truth_h[idx])
    labels = gap_regime_labels(bundle, gap_name=gap_name)[idx]
    abs_err = np.abs(pred - truth)

    groups: dict[str, np.ndarray] = {"all": np.ones(idx.size, dtype=bool)}
    for code, name in GAP_REGIME_NAMES.items():
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
        parameter="log10_thickness",
        per_regime=per_regime,
    )


def decades_to_percent(decades: float) -> float:
    """
    Convert a log10 error to a percentage of thickness.

    Parameters
    ----------
    decades : float
        An error in ``log10`` metres, as the scorecard reports it.

    Returns
    -------
    float
        ``100 * (10**decades - 1)``, i.e. the same error read as a fractional
        overshoot. 0.0107 decades is 2.5 %. ``nan`` propagates.
    """
    return float(100.0 * (10.0**decades - 1.0))


def format_thickness_scorecard(
    scorecard: Scorecard,
    *,
    row_order: Sequence[str] = GAP_ROW_ORDER,
    precision: int = 4,
) -> str:
    """
    Render a gap-width scorecard, in decades and as a percentage.

    :func:`~sonic_ml.bench.report.format_scorecard` would print this
    scorecard correctly but only in decades, which almost nobody reads
    fluently. This adds the percentage column and nothing else.

    Parameters
    ----------
    scorecard : Scorecard
        From :func:`evaluate_thickness`.
    row_order : sequence of str, default :data:`GAP_ROW_ORDER`
        Rows to print, in order; rows absent from the scorecard are skipped.
    precision : int, default 4
        Decimal places for the decade columns.

    Returns
    -------
    str
        A multi-line table: one row per regime with sample counts, median and
        mean absolute error in decades, the same median as a percentage, and
        the bootstrap 95 % CI on the median (in decades).
    """
    header = (
        f"Scorecard: {scorecard.predictor}  "
        f"(parameter: {scorecard.parameter}, errors in decades)"
    )
    cols = (
        f"{'regime':>6}  {'n':>5}  {'finite':>6}  "
        f"{'medAE':>8}  {'meanAE':>8}  {'medAE %':>8}  {'95% CI (median)':>20}"
    )
    lines = [header, cols, "-" * len(cols)]
    for name in row_order:
        row = scorecard.per_regime.get(name)
        if row is None:
            continue
        ci = f"[{row.ci_low:.{precision}f}, {row.ci_high:.{precision}f}]"
        pct = decades_to_percent(row.median_abs_error)
        lines.append(
            f"{row.regime:>6}  {row.n:>5}  {row.n_finite:>6}  "
            f"{row.median_abs_error:>8.{precision}f}  "
            f"{row.mean_abs_error:>8.{precision}f}  {pct:>8.1f}  {ci:>20}"
        )
    return "\n".join(lines)


class KrauklisThicknessPredictor:
    """
    The closed-form baseline, as a :class:`ThicknessPredictor`.

    A thin adapter over
    :class:`~sonic_ml.baselines.debond.CrackWaveThicknessBaseline`: it
    converts metres to log10 and honours ``indices``. Kept here rather than on
    the baseline itself so :mod:`sonic_ml.baselines` stays a description of
    the physics and this module stays the only place that knows the harness's
    conventions.

    Parameters
    ----------
    baseline : CrackWaveThicknessBaseline or None
        The estimator to wrap; ``None`` builds a default one.
    name : str
        Predictor name shown on the scorecard.
    """

    def __init__(
        self,
        baseline: CrackWaveThicknessBaseline | None = None,
        *,
        name: str = "krauklis_closed_form",
    ) -> None:
        self.baseline = CrackWaveThicknessBaseline() if baseline is None else baseline
        self.name = name

    def predict_log_thickness(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        ``log10`` gap width (m) for the requested samples.

        The baseline has no fitted state, so it is evaluated over the whole
        bundle and subset afterwards -- scoring a subset gives exactly the
        same numbers as scoring all of it and reading those rows.
        """
        estimate = self.baseline.predict(bundle)
        idx = np.arange(estimate.size) if indices is None else np.asarray(indices, int)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(estimate[idx])


class MeanThicknessPredictor:
    """
    Predict a constant ``log10`` gap width -- the "no skill" reference.

    The direct analogue of :class:`~sonic_ml.bench.bond.MeanBondPredictor`,
    and it matters more here than it looks: against a log-uniform prior over
    two decades, always answering the mean scores ~0.5 decades, so it is the
    yardstick that makes "18 %" and "2.5 %" mean something rather than being
    bare numbers.

    Parameters
    ----------
    value : float
        The constant to predict, in ``log10`` metres (typically the
        training-split mean; see :meth:`from_train_split`).
    name : str
        Predictor name shown on the scorecard.
    """

    def __init__(self, value: float, *, name: str = "mean_thickness") -> None:
        self.value = float(value)
        self.name = name

    @classmethod
    def from_train_split(
        cls,
        bundle: DatasetBundle,
        train: np.ndarray,
        *,
        gap_name: str = GAP_LAYER_NAME,
        name: str = "mean_thickness",
    ) -> MeanThicknessPredictor:
        """
        Build the reference from a training split's mean ``log10`` thickness.

        Parameters
        ----------
        bundle : DatasetBundle
        train : ndarray of int
            Training indices -- the held-out split must not be used, or the
            "no skill" reference quietly acquires some.
        gap_name : str, default ``"microannulus"``
        name : str

        Returns
        -------
        MeanThicknessPredictor
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            log_h = np.log10(gap_thickness(bundle, gap_name=gap_name))
        finite = log_h[np.asarray(train, dtype=int)]
        finite = finite[np.isfinite(finite)]
        return cls(float(finite.mean()) if finite.size else float("nan"), name=name)

    def predict_log_thickness(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the constant ``log10`` thickness for every requested sample."""
        n = bundle.n_samples if indices is None else np.asarray(indices, dtype=int).size
        return np.full(n, self.value)
