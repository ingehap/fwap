"""
Classical microannulus-thickness estimator -- the bar a learned debonding
model has to clear.

Why this baseline is a real one, unlike a CBL amplitude gate
------------------------------------------------------------
Field cement-bond logging keys off casing-ring amplitude, and a debonded pipe
rings louder. That indicator still cannot be run on these synthetics and
still would be a strawman: the cased gathers carry no casing-ring arrival,
and :class:`~sonic_ml.models.augment.CasingRingAugmentation` deliberately
draws its ring amplitude *independently* of bond so that no model can recover
a relationship that was planted rather than modelled.

The debonded dataset supplies something better. Its crack-wave branch is
governed by the Krauklis dispersion relation

.. math::

    c = \\left(\\frac{\\omega h}{C \\rho_f}\\right)^{1/3},
    \\qquad C = \\sum \\frac{1 - \\nu}{\\mu},

summed over the two solids bounding the gap -- so a *closed-form* inversion
for the gap width exists:

.. math::

    h = \\frac{c^3 C \\rho_f}{\\omega}.

That is this module. It is a genuinely independent estimator rather than a
circular one: the dataset's curves come from a numerical root of the full
modal determinant, and this law is the analytic asymptote that was used to
validate that determinant to 0.02 % (``plans/roadmap.md`` A.5). It also needs
**no fitted calibration**, unlike :class:`~sonic_ml.baselines.bond.StoneleyBondBaseline`,
so it is a harder bar: it spends none of the training split.

What it assumes, and where that bites
-------------------------------------
The Krauklis law treats the bounding solids as half-spaces. Here they are
finite layers -- roughly 10 mm of casing and 45 mm of cement -- against a
crack wavelength of the same order over the 1-12 kHz band. The estimator is
therefore expected to carry a systematic bias, and
:class:`CrackWaveThicknessBaseline` reports it rather than hiding it: see
:meth:`CrackWaveThicknessBaseline.score`, which returns the median ratio
alongside the scatter so a bias and a spread cannot be confused.

References
----------
* Krauklis, P. V. (1962). On some low-frequency oscillations of a liquid
  layer in an elastic medium. *PMM* 26(6), 1111-1115.
* Korneev, V. (2008). Slow waves in fractures filled with viscous fluid.
  *Geophysics* 73(1), N1-N7.
"""

from __future__ import annotations

import numpy as np

from sonic_ml.loader import LAYER_PARAM_NAMES, DatasetBundle

#: Name the generator gives the fluid-gap layer.
GAP_LAYER_NAME: str = "microannulus"

#: Name the generator gives the crack-wave dispersion row.
CRACK_WAVE_MODE_NAME: str = "crack_wave"


def solid_compliance(vp: float, vs: float, rho: float) -> float:
    """
    The ``(1 - nu) / mu`` term one bounding solid contributes.

    Parameters
    ----------
    vp, vs : float
        Compressional and shear velocity (m/s).
    rho : float
        Density (kg/m^3).

    Returns
    -------
    float
        Compliance contribution (Pa^-1). ``inf`` for a fluid (``vs == 0``),
        which is the physically right limit -- a fluid offers no shear
        stiffness -- and which makes a mis-specified layer stack blow up
        loudly rather than return a plausible number.

    Raises
    ------
    ValueError
        If ``vp <= vs``, which no elastic solid satisfies.
    """
    if vp <= vs:
        raise ValueError(f"vp must exceed vs; got vp={vp!r}, vs={vs!r}")
    if vs <= 0.0:
        return float("inf")
    mu = rho * vs**2
    nu = (vp**2 - 2.0 * vs**2) / (2.0 * (vp**2 - vs**2))
    return (1.0 - nu) / mu


def crack_compliance(
    layer_params: np.ndarray,
    layer_names: tuple[str, ...],
    *,
    gap_name: str = GAP_LAYER_NAME,
) -> float:
    """
    Total compliance ``C`` of the two solids bounding the fluid gap.

    Parameters
    ----------
    layer_params : ndarray, shape (n_layer, 4)
        One sample's layer rows, in :data:`~sonic_ml.loader.LAYER_PARAM_NAMES`
        order.
    layer_names : tuple of str
        Labels aligned with axis 0 of ``layer_params``.
    gap_name : str, default ``"microannulus"``
        Label of the fluid gap.

    Returns
    -------
    float
        ``C`` in Pa^-1.

    Raises
    ------
    ValueError
        If no layer carries ``gap_name``, or the gap sits at the edge of the
        stack so that one bounding solid is missing from it.
    """
    if gap_name not in layer_names:
        raise ValueError(
            f"no {gap_name!r} layer in {layer_names!r}; this is not a debonded stack"
        )
    idx = layer_names.index(gap_name)
    if idx == 0 or idx == len(layer_names) - 1:
        raise ValueError(
            f"the {gap_name!r} layer is at the edge of the stack, so one "
            "bounding solid is outside it; the crack-wave law needs both"
        )
    vp_i, vs_i, rho_i, _ = layer_params[idx - 1]
    vp_o, vs_o, rho_o, _ = layer_params[idx + 1]
    return solid_compliance(vp_i, vs_i, rho_i) + solid_compliance(vp_o, vs_o, rho_o)


def krauklis_thickness(
    freq: np.ndarray,
    phase_velocity: np.ndarray,
    *,
    rho_f: float,
    compliance: float,
) -> np.ndarray:
    """
    Invert the Krauklis law for the gap width, frequency by frequency.

    Parameters
    ----------
    freq : ndarray, shape (n_freq,)
        Frequencies (Hz).
    phase_velocity : ndarray, shape (n_freq,)
        Crack-wave phase velocity (m/s). ``NaN`` propagates.
    rho_f : float
        Density of the fluid in the gap (kg/m^3).
    compliance : float
        ``C`` from :func:`crack_compliance` (Pa^-1).

    Returns
    -------
    ndarray, shape (n_freq,)
        Gap width (m) implied at each frequency. A perfect measurement of a
        perfect Krauklis crack would give the same value at every frequency;
        the spread across the band is a direct read-out of how well the
        half-space assumption holds, so it is returned rather than reduced.
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    c = np.asarray(phase_velocity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return c**3 * compliance * rho_f / omega


class CrackWaveThicknessBaseline:
    """
    Closed-form gap-width estimator from the crack-wave dispersion curve.

    Takes the crack-wave row of a debonded bundle, inverts the Krauklis law
    at every frequency, and reduces the band to one number with a median --
    which is robust to the few frequencies where the numerical root wanders.

    Unlike :class:`~sonic_ml.baselines.bond.StoneleyBondBaseline` this needs
    no ``fit``: the law supplies the calibration. That makes it a stricter
    bar, because it spends none of the training split to reach its answer.

    Parameters
    ----------
    mode_name : str, default ``"crack_wave"``
        Which dispersion row to read.
    gap_name : str, default ``"microannulus"``
        Which layer is the fluid gap.
    """

    name = "krauklis_crack_wave_thickness"

    def __init__(
        self,
        *,
        mode_name: str = CRACK_WAVE_MODE_NAME,
        gap_name: str = GAP_LAYER_NAME,
    ) -> None:
        self.mode_name = mode_name
        self.gap_name = gap_name

    def predict(self, bundle: DatasetBundle) -> np.ndarray:
        """
        Estimated gap width for every sample in the bundle.

        Parameters
        ----------
        bundle : DatasetBundle
            A debonded dataset: a layer stack carrying ``gap_name`` and a
            dispersion row named ``mode_name``.

        Returns
        -------
        ndarray, shape (n_samples,)
            Gap width (m); ``NaN`` where the crack-wave curve had no finite
            frequency.

        Raises
        ------
        ValueError
            If the bundle carries no such mode or no layer stack.
        """
        if self.mode_name not in bundle.mode_names:
            raise ValueError(
                f"bundle has no {self.mode_name!r} mode; it carries "
                f"{bundle.mode_names!r}. A debonded dataset is generated by "
                "gen_surrogate_dataset.py --debonded"
            )
        if bundle.layer_params is None:
            raise ValueError("bundle carries no layer stack")

        row = bundle.mode_names.index(self.mode_name)
        rho_f = self._fluid_density(bundle)
        out = np.full(bundle.n_samples, np.nan, dtype=float)
        for i in range(bundle.n_samples):
            compliance = crack_compliance(
                bundle.layer_params[i], bundle.layer_names, gap_name=self.gap_name
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                velocity = 1.0 / np.asarray(bundle.slowness[i, row], dtype=float)
            estimates = krauklis_thickness(
                bundle.freq, velocity, rho_f=rho_f[i], compliance=compliance
            )
            finite = estimates[np.isfinite(estimates)]
            if finite.size:
                out[i] = float(np.median(finite))
        return out

    def truth(self, bundle: DatasetBundle) -> np.ndarray:
        """
        The gap width the generator drew, read back out of ``layer_params``.

        Parameters
        ----------
        bundle : DatasetBundle

        Returns
        -------
        ndarray, shape (n_samples,)
            Gap width (m).
        """
        if bundle.layer_params is None:
            raise ValueError("bundle carries no layer stack")
        idx = bundle.layer_names.index(self.gap_name)
        thickness = LAYER_PARAM_NAMES.index("thickness")
        return np.asarray(bundle.layer_params[:, idx, thickness], dtype=float)

    def score(self, bundle: DatasetBundle) -> dict[str, float]:
        """
        Score the estimate against the drawn gap width.

        Reported in the ratio domain rather than as an absolute error,
        because the gap spans two decades and an RMS in metres would be
        dominated by the widest samples alone.

        Parameters
        ----------
        bundle : DatasetBundle

        Returns
        -------
        dict
            ``median_ratio`` -- the systematic bias (1.0 is unbiased);
            ``ratio_iqr`` -- the spread of that ratio, i.e. how much is left
            after a single constant correction;
            ``log_rmse`` -- RMS error in ``log10`` of the gap, the scale the
            prior samples on;
            ``rank_correlation`` -- Spearman correlation with the truth,
            which a constant bias cannot flatter;
            ``n_scored``.
        """
        predicted = self.predict(bundle)
        actual = self.truth(bundle)
        ok = np.isfinite(predicted) & np.isfinite(actual) & (actual > 0.0)
        if not ok.any():
            return {
                "median_ratio": float("nan"),
                "ratio_iqr": float("nan"),
                "log_rmse": float("nan"),
                "rank_correlation": float("nan"),
                "n_scored": 0.0,
            }
        ratio = predicted[ok] / actual[ok]
        log_err = np.log10(np.abs(predicted[ok])) - np.log10(actual[ok])
        order_p = np.argsort(np.argsort(predicted[ok]))
        order_a = np.argsort(np.argsort(actual[ok]))
        rank = (
            float(np.corrcoef(order_p, order_a)[0, 1]) if ok.sum() > 2 else float("nan")
        )
        return {
            "median_ratio": float(np.median(ratio)),
            "ratio_iqr": float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
            "log_rmse": float(np.sqrt(np.mean(log_err**2))),
            "rank_correlation": rank,
            "n_scored": float(ok.sum()),
        }

    @staticmethod
    def _fluid_density(bundle: DatasetBundle) -> np.ndarray:
        """Per-sample borehole-fluid density, the gap fluid by construction."""
        return np.asarray(
            bundle.params[:, bundle.param_names.index("rho_f")], dtype=float
        )
