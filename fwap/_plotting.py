"""
Legacy module path for the plotting helpers.

``fwap._plotting`` is kept as a thin re-export of :mod:`fwap.plotting`
so that existing imports (``from fwap._plotting import _wiggle``) keep
working. New code should use :mod:`fwap.plotting` directly.
"""

from __future__ import annotations

from fwap.plotting import (
    _savefig,
    _wiggle,
    save_figure,
    wiggle_plot,
)

__all__ = ["_savefig", "_wiggle", "save_figure", "wiggle_plot"]
