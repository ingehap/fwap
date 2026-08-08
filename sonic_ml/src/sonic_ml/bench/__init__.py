"""Model-agnostic benchmark harness: score any Vs or bond predictor."""

from __future__ import annotations

from sonic_ml.bench.bond import (
    BOND_REGIME_NAMES,
    BOND_ROW_ORDER,
    BondPredictor,
    MeanBondPredictor,
    bond_regime_labels,
    evaluate_bond,
)
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
    # cement-bond scoring (M5d)
    "BondPredictor",
    "evaluate_bond",
    "bond_regime_labels",
    "MeanBondPredictor",
    "BOND_REGIME_NAMES",
    "BOND_ROW_ORDER",
]
