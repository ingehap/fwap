"""Classical (non-ML) baselines -- the bar an ML model must beat."""

from __future__ import annotations

from sonic_ml.baselines.bond import (
    StoneleyBondBaseline,
    stoneley_peak_slowness,
)
from sonic_ml.baselines.classical import (
    ClassicalSTCBaseline,
    FKDispersionBaseline,
)

__all__ = [
    "ClassicalSTCBaseline",
    "FKDispersionBaseline",
    # cement-bond baseline (M5d)
    "StoneleyBondBaseline",
    "stoneley_peak_slowness",
]
