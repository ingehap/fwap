"""Pytest configuration for the fwap test suite."""

from __future__ import annotations

import os
import sys

# Make the package importable without requiring an editable install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Force a non-interactive matplotlib backend so the demo-regression
# tests never try to open a display. Has to be set before any
# ``import matplotlib.pyplot`` in the test process.
os.environ.setdefault("MPLBACKEND", "Agg")
