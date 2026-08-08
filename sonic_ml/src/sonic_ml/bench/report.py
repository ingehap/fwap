"""Human-readable formatting for a :class:`~sonic_ml.bench.harness.Scorecard`."""

from __future__ import annotations

from collections.abc import Sequence

from sonic_ml.bench.harness import Scorecard

_ROW_ORDER = ("all", "slow", "fast")


def format_scorecard(
    scorecard: Scorecard,
    *,
    row_order: Sequence[str] = _ROW_ORDER,
    precision: int = 1,
) -> str:
    """
    Render a scorecard as a fixed-width text table.

    Parameters
    ----------
    scorecard : Scorecard
    row_order : sequence of str, default ``("all", "slow", "fast")``
        Regime rows to print, in order. Rows absent from the scorecard are
        skipped. Pass :data:`~sonic_ml.bench.bond.BOND_ROW_ORDER` for a
        cement-bond scorecard.
    precision : int, default 1
        Decimal places for the error columns. The Vs default suits m/s; a
        target on a ``[0, 1]`` scale (e.g. the bond index) wants ``3``.

    Returns
    -------
    str
        A multi-line table: one row per regime with sample counts, median /
        mean absolute error, and the bootstrap 95% CI on the median.
    """
    header = f"Scorecard: {scorecard.predictor}  (parameter: {scorecard.parameter})"
    cols = (
        f"{'regime':>6}  {'n':>5}  {'finite':>6}  "
        f"{'medAE':>8}  {'meanAE':>8}  {'95% CI (median)':>20}"
    )
    lines = [header, cols, "-" * len(cols)]
    for name in row_order:
        row = scorecard.per_regime.get(name)
        if row is None:
            continue
        ci = f"[{row.ci_low:.{precision}f}, {row.ci_high:.{precision}f}]"
        lines.append(
            f"{row.regime:>6}  {row.n:>5}  {row.n_finite:>6}  "
            f"{row.median_abs_error:>8.{precision}f}  "
            f"{row.mean_abs_error:>8.{precision}f}  {ci:>20}"
        )
    return "\n".join(lines)
