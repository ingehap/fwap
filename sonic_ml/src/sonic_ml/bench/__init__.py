"""Model-agnostic benchmark harness: score any Vs predictor against truth."""

from __future__ import annotations

from sonic_ml.bench.harness import (
    Predictor,
    RegimeScore,
    Scorecard,
    StubPredictor,
    evaluate,
)
from sonic_ml.bench.report import format_scorecard

__all__ = [
    "Predictor",
    "RegimeScore",
    "Scorecard",
    "StubPredictor",
    "evaluate",
    "format_scorecard",
]
