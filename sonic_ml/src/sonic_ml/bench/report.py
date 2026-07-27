"""Human-readable formatting for a :class:`~sonic_ml.bench.harness.Scorecard`."""

from __future__ import annotations

from sonic_ml.bench.harness import Scorecard

_ROW_ORDER = ("all", "slow", "fast")


def format_scorecard(scorecard: Scorecard) -> str:
    """
    Render a scorecard as a fixed-width text table.

    Parameters
    ----------
    scorecard : Scorecard

    Returns
    -------
    str
        A multi-line table: one row per regime with sample counts, median /
        mean absolute Vs error (m/s), and the bootstrap 95% CI on the median.
    """
    header = f"Scorecard: {scorecard.predictor}  (parameter: {scorecard.parameter})"
    cols = (
        f"{'regime':>6}  {'n':>5}  {'finite':>6}  "
        f"{'medAE':>8}  {'meanAE':>8}  {'95% CI (median)':>20}"
    )
    lines = [header, cols, "-" * len(cols)]
    for name in _ROW_ORDER:
        row = scorecard.per_regime.get(name)
        if row is None:
            continue
        ci = f"[{row.ci_low:.0f}, {row.ci_high:.0f}]"
        lines.append(
            f"{row.regime:>6}  {row.n:>5}  {row.n_finite:>6}  "
            f"{row.median_abs_error:>8.1f}  {row.mean_abs_error:>8.1f}  {ci:>20}"
        )
    return "\n".join(lines)
