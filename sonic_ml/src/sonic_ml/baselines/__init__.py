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
from sonic_ml.baselines.debond import (
    CrackWaveThicknessBaseline,
    crack_compliance,
    krauklis_thickness,
    solid_compliance,
)

__all__ = [
    "ClassicalSTCBaseline",
    "FKDispersionBaseline",
    # cement-bond baseline (M5d)
    "StoneleyBondBaseline",
    "stoneley_peak_slowness",
    # debonded-regime gap-width baseline (G.2)
    "CrackWaveThicknessBaseline",
    "crack_compliance",
    "krauklis_thickness",
    "solid_compliance",
]
