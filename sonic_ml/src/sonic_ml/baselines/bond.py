"""
Classical cement-bond indicator from the cased Stoneley arrival -- the
baseline the M5d learned inverse has to beat.

Why *not* a CBL amplitude indicator
-----------------------------------
Field cement-bond logging keys off the **casing-ring amplitude**: a debonded
(free) pipe rings loudly, a well-bonded one leaks energy into the formation and
rings quietly. That indicator cannot be run here, and pretending otherwise would
make a strawman baseline: the M5a cased-hole gathers are synthesized from the
**cased Stoneley dispersion curve alone**, so they contain no casing-ring
arrival for an amplitude gate to measure. A CBL-amplitude baseline would be
measuring noise, and beating it would prove nothing.

The honest classical analogue for *this* dataset uses the signal that is
actually present. Cement stiffness is the dominant control on the cased Stoneley
curve -- sweeping the cement shear velocity across the prior moves the curve by
~7%, against ~1.5% for the formation shear velocity -- so a slowness-time
coherence (STC) pick of the Stoneley arrival, mapped to bond quality through a
calibration fitted on the training split, is a fair and genuinely informative
reference.

:class:`StoneleyBondBaseline` is that estimator. It is deliberately given every
advantage the learned model does not get: its calibration is fitted on the same
training samples, and the fit is closed-form rather than stochastic.
"""

from __future__ import annotations

import numpy as np
from fwap import ArrayGeometry, stc

from sonic_ml.loader import DatasetBundle

#: STC search window (s/m) bracketing the cased Stoneley arrival under the
#: ``CasingCementPriors`` defaults (~200-260 us/ft).
DEFAULT_SLOWNESS_RANGE: tuple[float, float] = (5.0e-4, 1.1e-3)


def stoneley_peak_slowness(
    gather: np.ndarray,
    geom: ArrayGeometry,
    *,
    slowness_range: tuple[float, float] = DEFAULT_SLOWNESS_RANGE,
    n_slowness: int = 121,
) -> float:
    """
    Peak-coherence Stoneley slowness of one cased gather.

    Parameters
    ----------
    gather : ndarray, shape (n_rec, n_samples)
        One cased-hole waveform gather.
    geom : fwap.ArrayGeometry
        Acquisition geometry (sampling interval and receiver offsets).
    slowness_range : (float, float), default :data:`DEFAULT_SLOWNESS_RANGE`
        STC trial-slowness window (s/m).
    n_slowness : int, default 121
        Slowness-axis resolution.

    Returns
    -------
    float
        Slowness (s/m) of the global coherence peak; ``nan`` if the STC fails.
    """
    try:
        result = stc(
            gather,
            dt=geom.dt,
            offsets=geom.offsets,
            slowness_range=slowness_range,
            n_slowness=n_slowness,
        )
    except (ValueError, ZeroDivisionError):
        return float("nan")
    coherence = np.nan_to_num(np.asarray(result.coherence))
    slowness = np.asarray(result.slowness)
    row = int(np.unravel_index(int(np.argmax(coherence)), coherence.shape)[0])
    return float(slowness[row])


class StoneleyBondBaseline:
    """
    Classical bond-index estimator: STC Stoneley slowness -> linear calibration.

    Picks the Stoneley arrival's apparent slowness with
    :func:`fwap.stc`, then maps it to a bond index with a straight line fitted
    by least squares on the *training* split. Predictions are clipped to
    ``[0, 1]``, the range the bond index is defined on.

    Satisfies the :class:`~sonic_ml.bench.bond.BondPredictor` protocol, so it is
    scored by the same harness as the learned inverse.

    Parameters
    ----------
    slowness_range : (float, float)
        STC trial-slowness window (s/m).
    n_slowness : int, default 121
        Slowness-axis resolution.
    """

    name = "stoneley_stc_bond"

    def __init__(
        self,
        *,
        slowness_range: tuple[float, float] = DEFAULT_SLOWNESS_RANGE,
        n_slowness: int = 121,
    ) -> None:
        self.slowness_range = slowness_range
        self.n_slowness = n_slowness
        self._coeffs: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """``True`` once :meth:`fit` has established a calibration."""
        return self._coeffs is not None

    def fit(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> StoneleyBondBaseline:
        """
        Fit the slowness -> bond calibration on the given (training) samples.

        Parameters
        ----------
        bundle : DatasetBundle
            A cased-hole dataset carrying ``bond_index``.
        geom : fwap.ArrayGeometry
        indices : ndarray of int
            Training sample indices.

        Returns
        -------
        StoneleyBondBaseline
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the bundle is not a cased-hole dataset, or no sample yields a
            finite STC pick (nothing to calibrate against). An *open-hole*
            schema-v4 bundle carries an all-``NaN`` ``bond_index``, so the check
            is on ``is_cased`` rather than on the key's presence.
        """
        if not bundle.is_cased or bundle.bond_index is None:
            raise ValueError(
                "StoneleyBondBaseline requires a cased-hole dataset (schema v4 "
                "with a non-empty layer stack); got an open-hole bundle"
            )
        idx = np.asarray(indices, dtype=int)
        slowness = self._slowness(bundle, geom, idx)
        bond = np.asarray(bundle.bond_index, dtype=float)[idx]
        good = np.isfinite(slowness) & np.isfinite(bond)
        if not good.any():
            raise ValueError("no finite STC picks to calibrate the bond baseline")
        self._coeffs = np.polyfit(slowness[good], bond[good], 1)
        return self

    def predict_bond(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the bond index per requested sample.

        Parameters
        ----------
        bundle : DatasetBundle
        geom : fwap.ArrayGeometry
        indices : ndarray of int

        Returns
        -------
        ndarray, shape (len(indices),), float64
            Bond-index estimates clipped to ``[0, 1]``; ``nan`` where the STC
            pick failed.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._coeffs is None:
            raise RuntimeError("call fit() before predict_bond()")
        idx = np.asarray(indices, dtype=int)
        slowness = self._slowness(bundle, geom, idx)
        out = np.polyval(self._coeffs, slowness)
        out = np.clip(out, 0.0, 1.0)
        return np.where(np.isfinite(slowness), out, np.nan)

    def _slowness(
        self, bundle: DatasetBundle, geom: ArrayGeometry, idx: np.ndarray
    ) -> np.ndarray:
        """Peak-coherence Stoneley slowness (s/m) for each requested sample."""
        return np.array(
            [
                stoneley_peak_slowness(
                    bundle.gather[i],
                    geom,
                    slowness_range=self.slowness_range,
                    n_slowness=self.n_slowness,
                )
                for i in idx
            ],
            dtype=float,
        )
