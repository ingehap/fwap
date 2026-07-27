"""
Classical shear-velocity estimators from fwap -- the baseline to beat.

Two :class:`~sonic_ml.bench.harness.Predictor` implementations, both estimating
formation Vs from the waveform ``gather`` using only fwap's slowness-time /
dispersion processing (no ML):

* :class:`ClassicalSTCBaseline` -- regime-split semblance. Slow formations
  (``vs < vf``) use dispersion-corrected STC over the dipole-flexural family;
  fast formations (``vs >= vf``) use pseudo-Rayleigh STC. The estimate is the
  peak-coherence slowness inverted to a velocity.
* :class:`FKDispersionBaseline` -- a regime-agnostic estimator: an f-k phase
  slowness curve reduced to a low-frequency shear slowness.

Both are honest, imperfect references. In particular the default two-mode
dataset (Stoneley + flexural) contains no clean pseudo-Rayleigh / shear-head
arrival for a *fast* formation, so classical fast-regime Vs recovery is weak --
which is precisely the gap a gather-to-parameters inverse net is meant to
close. Failures are returned as ``NaN`` (never raised).

.. note::

   The regime split uses the *true* ``vs`` (via
   :func:`sonic_ml.split.regime_labels`) to pick the estimator, which gives the
   classical baseline the benefit of knowing the regime -- a deliberately
   conservative (baseline-favouring) choice, so "beating the baseline" means
   more.
"""

from __future__ import annotations

import numpy as np
from fwap import ArrayGeometry, dipole_flexural_dispersion
from fwap.dispersion import (
    dispersive_pseudo_rayleigh_stc,
    dispersive_stc,
    phase_slowness_from_f_k,
    shear_slowness_from_dispersion,
)

from sonic_ml.loader import DatasetBundle
from sonic_ml.split import regime_labels


def _peak_slowness(result: object) -> float:
    """Peak-coherence slowness (s/m) of an ``STCResult``."""
    coherence = np.nan_to_num(np.asarray(result.coherence))  # type: ignore[attr-defined]
    slowness = np.asarray(result.slowness)  # type: ignore[attr-defined]
    flat = int(np.argmax(coherence))
    row = int(np.unravel_index(flat, coherence.shape)[0])
    return float(slowness[row])


class ClassicalSTCBaseline:
    """
    Regime-split dispersion-corrected STC shear-velocity estimator.

    Parameters
    ----------
    v_fluid : float, default 1500.0
        Borehole-fluid velocity (m/s), used by the pseudo-Rayleigh branch and
        as the slow/fast regime boundary.
    a_borehole : float, default 0.1
        Borehole radius (m).
    slow_slowness_range, fast_slowness_range : (float, float)
        Trial shear-slowness search ranges (s/m) for the slow and fast
        branches. Defaults span the generator's default prior
        (``vs`` in ~1200-3200 m/s); the fast range stays below ``1 / v_fluid``
        as the pseudo-Rayleigh STC requires.
    n_slowness : int, default 61
        Slowness-axis resolution.
    slow_f_range, fast_f_range : (float, float)
        Processing frequency bands (Hz) for each branch.
    """

    name = "classical_stc"

    def __init__(
        self,
        *,
        v_fluid: float = 1500.0,
        a_borehole: float = 0.1,
        slow_slowness_range: tuple[float, float] = (400e-6, 900e-6),
        fast_slowness_range: tuple[float, float] = (280e-6, 650e-6),
        n_slowness: int = 61,
        slow_f_range: tuple[float, float] = (1000.0, 4000.0),
        fast_f_range: tuple[float, float] = (3000.0, 10000.0),
    ) -> None:
        self.v_fluid = v_fluid
        self.a_borehole = a_borehole
        self.slow_slowness_range = slow_slowness_range
        self.fast_slowness_range = fast_slowness_range
        self.n_slowness = n_slowness
        self.slow_f_range = slow_f_range
        self.fast_f_range = fast_f_range

    def predict_vs(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Estimate Vs (m/s) per index; ``NaN`` where the estimator fails."""
        idx = np.asarray(indices, dtype=int)
        labels = regime_labels(bundle)
        out = np.full(idx.size, np.nan, dtype=float)
        for k, i in enumerate(idx):
            out[k] = self._estimate_one(bundle.gather[i], geom, int(labels[i]))
        return out

    def _estimate_one(
        self, gather: np.ndarray, geom: ArrayGeometry, regime: int
    ) -> float:
        try:
            if regime == 0:  # slow: dipole-flexural dispersion-corrected STC

                def family(s_shear: float):
                    return dipole_flexural_dispersion(
                        vs=1.0 / s_shear, a_borehole=self.a_borehole
                    )

                result = dispersive_stc(
                    gather,
                    dt=geom.dt,
                    offsets=geom.offsets,
                    dispersion_family=family,
                    shear_slowness_range=self.slow_slowness_range,
                    n_slowness=self.n_slowness,
                    f_range=self.slow_f_range,
                )
            else:  # fast: pseudo-Rayleigh STC
                result = dispersive_pseudo_rayleigh_stc(
                    gather,
                    dt=geom.dt,
                    offsets=geom.offsets,
                    v_fluid=self.v_fluid,
                    a_borehole=self.a_borehole,
                    shear_slowness_range=self.fast_slowness_range,
                    n_slowness=self.n_slowness,
                    f_range=self.fast_f_range,
                )
            s = _peak_slowness(result)
            return 1.0 / s if s > 0.0 else float("nan")
        except (ValueError, ZeroDivisionError):
            return float("nan")


class FKDispersionBaseline:
    """
    Regime-agnostic Vs from an f-k phase-slowness curve.

    Estimates a dispersion curve with :func:`fwap.dispersion.phase_slowness_from_f_k`
    and reduces it to a low-frequency shear slowness with
    :func:`fwap.dispersion.shear_slowness_from_dispersion` (the flexural
    ``f -> 0`` limit approaches ``1 / vs``).

    Parameters
    ----------
    f_range : (float, float), default (1000.0, 6000.0)
        Band for the f-k slowness estimate (Hz).
    f_lo, f_hi : float
        Low-frequency window (Hz) whose quality-weighted mean slowness is
        taken as the shear slowness.
    quality_threshold : float, default 0.3
        Minimum per-frequency quality to include. The fwap default of 0.8 is
        deliberately loosened -- it rejects nearly everything on these gathers.
    """

    name = "fk_dispersion"

    def __init__(
        self,
        *,
        f_range: tuple[float, float] = (1000.0, 6000.0),
        f_lo: float = 1000.0,
        f_hi: float = 2500.0,
        quality_threshold: float = 0.3,
    ) -> None:
        self.f_range = f_range
        self.f_lo = f_lo
        self.f_hi = f_hi
        self.quality_threshold = quality_threshold

    def predict_vs(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Estimate Vs (m/s) per index; ``NaN`` where the estimator fails."""
        idx = np.asarray(indices, dtype=int)
        out = np.full(idx.size, np.nan, dtype=float)
        for k, i in enumerate(idx):
            try:
                curve = phase_slowness_from_f_k(
                    bundle.gather[i],
                    dt=geom.dt,
                    offsets=geom.offsets,
                    f_range=self.f_range,
                )
                s = shear_slowness_from_dispersion(
                    curve,
                    f_lo=self.f_lo,
                    f_hi=self.f_hi,
                    quality_threshold=self.quality_threshold,
                )
                out[k] = 1.0 / s if s > 0.0 else float("nan")
            except (ValueError, ZeroDivisionError):
                out[k] = float("nan")
        return out
