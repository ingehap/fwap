"""Classical (non-ML) shear-velocity baselines -- the bar an ML model must beat."""

from __future__ import annotations

from sonic_ml.baselines.classical import (
    ClassicalSTCBaseline,
    FKDispersionBaseline,
)

__all__ = [
    "ClassicalSTCBaseline",
    "FKDispersionBaseline",
]
