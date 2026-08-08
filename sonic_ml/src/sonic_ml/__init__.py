"""
sonic_ml -- machine-learning surrogate & inverse models for fwap.

An in-repo sibling package (issue #22) that consumes the ``.npz`` produced by
``scripts/gen_surrogate_dataset.py`` and trains PyTorch surrogate / inverse
models on it. Deliberately separate from the pure-NumPy/SciPy ``fwap`` core so
no ML dependency ever enters the physics package.

This M0 layer is the pre-model spine: a defensive dataset loader, a bridge to
the core generator, provenance/reproducibility capture, regime-stratified
splitting, standardization, masking, and determinism helpers. Models arrive in
later milestones.
"""

from __future__ import annotations

from sonic_ml import baselines, bench, determinism, gen_shim, mask, oracles, provenance
from sonic_ml.baselines import (
    ClassicalSTCBaseline,
    FKDispersionBaseline,
    StoneleyBondBaseline,
)
from sonic_ml.bench import (
    BondPredictor,
    MeanBondPredictor,
    Predictor,
    RegimeScore,
    Scorecard,
    StubPredictor,
    bond_regime_labels,
    evaluate,
    evaluate_bond,
    format_scorecard,
)
from sonic_ml.geometry import default_geometry
from sonic_ml.loader import (
    SUPPORTED_SCHEMA_VERSIONS,
    DatasetBundle,
    SchemaError,
    UnsupportedSchemaVersionError,
    load_npz,
)
from sonic_ml.normalize import Standardizer
from sonic_ml.provenance import Provenance
from sonic_ml.split import DataSplit, regime_labels, stratified_split

__version__ = "0.0.1"

__all__ = [
    "__version__",
    # loader
    "DatasetBundle",
    "load_npz",
    "SchemaError",
    "UnsupportedSchemaVersionError",
    "SUPPORTED_SCHEMA_VERSIONS",
    # split
    "DataSplit",
    "stratified_split",
    "regime_labels",
    # normalize
    "Standardizer",
    # provenance
    "Provenance",
    # geometry
    "default_geometry",
    # benchmark harness (M1)
    "Predictor",
    "Scorecard",
    "RegimeScore",
    "StubPredictor",
    "evaluate",
    "format_scorecard",
    # classical baselines (M1)
    "ClassicalSTCBaseline",
    "FKDispersionBaseline",
    # cement-bond scoring + baseline (M5d)
    "BondPredictor",
    "evaluate_bond",
    "bond_regime_labels",
    "MeanBondPredictor",
    "StoneleyBondBaseline",
    # submodules
    "gen_shim",
    "mask",
    "provenance",
    "determinism",
    "oracles",
    "baselines",
    "bench",
]
