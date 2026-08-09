"""
Scoring fwap dispersion curves against digitised reference figures.

The validation notebook (``docs/notebooks/cylindrical_biot_validation.ipynb``)
reproduces published borehole-acoustic dispersion figures and overlays the fwap
curve on a digitised copy of the reference. Plotting the two together is the
easy half. This module is the other half: turning "they look close" into a
number with a budget attached, so a regression in the solver fails a check
instead of subtly changing a picture nobody re-reads.

Why this is mostly input validation
-----------------------------------
The reference data is *hand-digitised* --- someone traces a curve out of a
figure with a tool like WebPlotDigitizer and saves the points. That process has
a small number of failure modes, and every one of them produces a file that
looks perfectly reasonable:

* axes read in the units printed on the figure (us/ft, us/m, kHz) rather than
  the SI the loader expects;
* a *velocity* axis digitised as though it were slowness;
* points emitted in click order rather than sorted by frequency;
* a trace that covers a different frequency band than the model curve.

The first two matter most, because they are the ones that can make a **wrong
solver look validated** --- a curve compared against a reference in the wrong
units will disagree by a factor of a thousand, which is loud, but a curve
compared against a reference that has been silently rescaled to fit would agree
for no reason at all. So :func:`load_reference_curve` refuses ambiguous input
with a specific diagnosis rather than guessing, and it never converts units on
the caller's behalf. Being told "this looks like us/ft, not s/m" is useful;
being quietly given the answer you wanted is not.

What the score means
--------------------
:func:`score_against_reference` interpolates the model curve onto the
reference's frequencies --- the model is dense and smooth, the reference sparse
and noisy, so that is the direction that does not invent detail --- and reports
RMS *relative* deviation over the overlapping band. Relative rather than
absolute, because slowness spans a factor of several across a dispersion curve
and an absolute budget would be dominated by whichever end of the band happens
to be slowest.

The default 5% budget is the figure from Plan I. It is loose on purpose:
digitisation error alone runs to a couple of percent on a printed log-axis
figure, so a tighter budget would be measuring the tracing rather than the
solver.

References
----------
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes.*
  CRC Press.
* Schmitt, D. P. (1988). Shear wave logging in elastic formations.
  *J. Acoust. Soc. Am.*, 84(6), 2215-2229.
* Tang, X. M., & Cheng, C. H. (2004). *Quantitative Borehole Acoustic
  Methods.* Elsevier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Plausible band for borehole phase slowness in SI (s/m). Spans roughly
#: 100 m/s to 100 km/s, which is far wider than any real formation, so a value
#: outside it is a units mistake rather than an unusual rock.
_SLOWNESS_SI_RANGE: tuple[float, float] = (1.0e-5, 1.0e-2)

#: Plausible band for borehole-acoustic frequency (Hz).
_FREQ_HZ_RANGE: tuple[float, float] = (1.0e2, 1.0e6)


class ReferenceDataError(ValueError):
    """A digitised reference file is unusable, with a reason attached."""


@dataclass(frozen=True)
class ReferenceCurve:
    """
    One digitised curve traced out of a published figure.

    Attributes
    ----------
    freq : ndarray, shape (N,), float64
        Frequency (Hz), strictly increasing.
    slowness : ndarray, shape (N,), float64
        Phase slowness (s/m).
    source : str
        Where the curve came from --- by default the filename it was loaded
        from. Carried so a score can be reported against a named figure rather
        than against "the reference".
    """

    freq: np.ndarray
    slowness: np.ndarray
    source: str

    @property
    def n_points(self) -> int:
        """Number of digitised points."""
        return int(self.freq.size)

    @property
    def freq_range(self) -> tuple[float, float]:
        """Lowest and highest digitised frequency (Hz)."""
        return float(self.freq[0]), float(self.freq[-1])


def _diagnose_slowness_units(values: np.ndarray) -> str | None:
    """
    Name the likely unit mistake in a slowness column, or ``None`` if it is SI.

    Deliberately a *diagnosis*, not a conversion. Rescaling a reference to make
    it fit is the one operation that could turn a failing comparison into a
    passing one for no physical reason.
    """
    median = float(np.median(values))
    lo, hi = _SLOWNESS_SI_RANGE
    if lo <= median <= hi:
        return None
    if 30.0 <= median <= 1000.0:
        return (
            f"median {median:.3g} looks like slowness in us/ft or us/m, not "
            f"s/m (expected around {1.0 / 3000.0:.2e}); divide by 0.3048e6 or "
            "1e6 respectively"
        )
    if 100.0 < median <= 1.0e5:
        return (
            f"median {median:.3g} looks like a *velocity* in m/s, not a "
            "slowness; the reference column must be slowness (take 1/v)"
        )
    return (
        f"median {median:.3g} is outside the plausible slowness band "
        f"[{lo:.0e}, {hi:.0e}] s/m; check the units of the digitised y-axis"
    )


def load_reference_curve(
    path: str | Path, *, source: str | None = None
) -> ReferenceCurve:
    """
    Read a two-column digitised reference CSV, refusing anything suspicious.

    The file is ``freq_hz, slowness_s_per_m``, no header, one row per digitised
    point. Rows are sorted by frequency on load, because digitiser output
    follows click order rather than axis order and an unsorted curve would
    silently break interpolation.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV to read.
    source : str or None
        Label for the curve; defaults to the file name.

    Returns
    -------
    ReferenceCurve

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ReferenceDataError
        If the file is empty, is not two columns, contains non-finite or
        non-positive values, has duplicate frequencies, or carries values whose
        magnitude indicates the wrong units. The message names the suspected
        mistake --- no silent rescaling happens in any case.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no digitised reference at {p}")

    if not p.read_text().strip():
        raise ReferenceDataError(f"{p.name} is empty")
    raw = np.loadtxt(p, delimiter=",", ndmin=2)
    if raw.size == 0:
        raise ReferenceDataError(f"{p.name} is empty")
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ReferenceDataError(
            f"{p.name} must have exactly 2 columns (freq_hz, slowness_s_per_m), "
            f"got shape {raw.shape}"
        )
    if raw.shape[0] < 3:
        raise ReferenceDataError(
            f"{p.name} has {raw.shape[0]} point(s); a curve needs at least 3 "
            "to be worth comparing against"
        )
    if not np.all(np.isfinite(raw)):
        raise ReferenceDataError(f"{p.name} contains non-finite values")
    if np.any(raw <= 0.0):
        raise ReferenceDataError(
            f"{p.name} contains non-positive values; both frequency and "
            "slowness must be strictly positive"
        )

    order = np.argsort(raw[:, 0])
    freq, slowness = raw[order, 0], raw[order, 1]
    if np.any(np.diff(freq) == 0.0):
        raise ReferenceDataError(
            f"{p.name} has duplicate frequencies; deduplicate the digitised "
            "points before comparing"
        )

    median_freq = float(np.median(freq))
    if not _FREQ_HZ_RANGE[0] <= median_freq <= _FREQ_HZ_RANGE[1]:
        raise ReferenceDataError(
            f"{p.name}: median frequency {median_freq:.3g} is outside the "
            f"plausible band [{_FREQ_HZ_RANGE[0]:.0e}, {_FREQ_HZ_RANGE[1]:.0e}] "
            "Hz; a figure axis in kHz must be multiplied by 1e3 before saving"
        )
    problem = _diagnose_slowness_units(slowness)
    if problem is not None:
        raise ReferenceDataError(f"{p.name}: {problem}")

    return ReferenceCurve(freq=freq, slowness=slowness, source=source or p.name)


@dataclass(frozen=True)
class OverlayScore:
    """
    How far a model curve sits from a digitised reference.

    Attributes
    ----------
    source : str
        The reference this was scored against.
    rms_relative : float
        Root-mean-square of ``|model - reference| / reference`` over the
        overlapping frequencies. The headline number.
    max_relative : float
        Worst single-point relative deviation. Quoted alongside the RMS because
        a curve that matches everywhere except one end --- the usual signature of
        a wrong asymptote --- can pass on RMS alone.
    n_compared : int
        Reference points that fell inside the model's finite frequency range.
    n_reference : int
        Points in the reference curve. ``n_compared`` well below this means the
        two curves barely overlap and the score covers little of the figure.
    overlap_hz : tuple of float
        Frequency span actually compared (Hz).
    budget : float
        Relative-deviation budget the comparison was held to.
    """

    source: str
    rms_relative: float
    max_relative: float
    n_compared: int
    n_reference: int
    overlap_hz: tuple[float, float]
    budget: float

    @property
    def passed(self) -> bool:
        """``True`` if the RMS deviation is within budget."""
        return self.rms_relative <= self.budget

    @property
    def coverage(self) -> float:
        """Fraction of reference points that could be compared."""
        return self.n_compared / self.n_reference if self.n_reference else 0.0


def score_against_reference(
    freq: np.ndarray,
    slowness: np.ndarray,
    reference: ReferenceCurve,
    *,
    budget: float = 0.05,
    min_overlap: int = 3,
) -> OverlayScore:
    """
    Score a model dispersion curve against a digitised reference.

    The model curve is interpolated onto the reference frequencies, not the
    other way round: the model is dense and smooth while the reference is a
    sparse hand-traced sample, so interpolating the reference would invent
    structure that was never in the figure.

    Frequencies where the model is ``NaN`` --- a mode that does not exist there ---
    are excluded rather than treated as data, and reference points outside the
    model's finite band are excluded too. :attr:`OverlayScore.coverage` reports
    how much of the figure survived that, because a score computed over three
    points of a forty-point curve is not the check it appears to be.

    Parameters
    ----------
    freq : ndarray, shape (M,)
        Model frequencies (Hz).
    slowness : ndarray, shape (M,)
        Model phase slowness (s/m); ``NaN`` where the mode is absent.
    reference : ReferenceCurve
    budget : float, default 0.05
        Relative-deviation budget (Plan I's 5%).
    min_overlap : int, default 3
        Minimum reference points that must fall inside the model's band.

    Returns
    -------
    OverlayScore

    Raises
    ------
    ValueError
        If the inputs are not 1-D and the same length, or ``budget`` is not
        positive.
    ReferenceDataError
        If fewer than ``min_overlap`` reference points fall inside the model's
        finite frequency range --- an empty comparison must not be reportable as
        a pass.
    """
    freq = np.asarray(freq, dtype=float)
    slowness = np.asarray(slowness, dtype=float)
    if freq.ndim != 1 or slowness.ndim != 1:
        raise ValueError("freq and slowness must be 1-D")
    if freq.shape != slowness.shape:
        raise ValueError(
            f"freq and slowness must match: {freq.shape} vs {slowness.shape}"
        )
    if budget <= 0.0:
        raise ValueError(f"budget must be positive, got {budget}")

    finite = np.isfinite(freq) & np.isfinite(slowness)
    if finite.sum() < 2:
        raise ReferenceDataError(
            "model curve has fewer than 2 finite points; nothing to compare"
        )
    model_f, model_s = freq[finite], slowness[finite]
    order = np.argsort(model_f)
    model_f, model_s = model_f[order], model_s[order]

    inside = (reference.freq >= model_f[0]) & (reference.freq <= model_f[-1])
    n_compared = int(inside.sum())
    if n_compared < min_overlap:
        raise ReferenceDataError(
            f"{reference.source}: only {n_compared} reference point(s) fall in "
            f"the model band {model_f[0]:.4g}-{model_f[-1]:.4g} Hz; the curves "
            "do not overlap enough to score"
        )

    ref_f, ref_s = reference.freq[inside], reference.slowness[inside]
    predicted = np.interp(ref_f, model_f, model_s)
    relative = np.abs(predicted - ref_s) / ref_s

    return OverlayScore(
        source=reference.source,
        rms_relative=float(np.sqrt(np.mean(relative**2))),
        max_relative=float(np.max(relative)),
        n_compared=n_compared,
        n_reference=reference.n_points,
        overlap_hz=(float(ref_f[0]), float(ref_f[-1])),
        budget=budget,
    )


def format_overlay_score(score: OverlayScore) -> str:
    """
    One-line verdict, always stating the outcome rather than only the numbers.

    Parameters
    ----------
    score : OverlayScore

    Returns
    -------
    str
    """
    verdict = "PASS" if score.passed else "FAIL"
    coverage = 100.0 * score.coverage
    return (
        f"{verdict}  {score.source}: RMS {100.0 * score.rms_relative:.2f}% "
        f"(budget {100.0 * score.budget:.1f}%), worst point "
        f"{100.0 * score.max_relative:.2f}%, {score.n_compared}/"
        f"{score.n_reference} points compared ({coverage:.0f}% coverage) over "
        f"{score.overlap_hz[0] / 1e3:.2f}-{score.overlap_hz[1] / 1e3:.2f} kHz"
    )
