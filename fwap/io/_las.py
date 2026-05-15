"""
LAS (Log ASCII Standard) reader and writer.

Wraps `lasio <https://lasio.rtfd.io>`_. References:
Canadian Well Logging Society (1991), *Log ASCII
Standard (LAS) Version 2.0*.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import lasio
import numpy as np

from fwap.io._common import _FWAP_UNITS


@dataclass
class LasCurves:
    """
    A LAS file loaded into memory.

    Attributes
    ----------
    depth : ndarray, shape (n_depth,)
        The depth axis (usually ``DEPT`` or ``DEPTH``), in the units
        declared by the LAS header (typically metres or feet).
    curves : dict[str, ndarray]
        One entry per non-depth curve, keyed by mnemonic (e.g.
        ``"GR"``, ``"DT"``, ``"RHOB"``). Arrays have the same length
        as ``depth``. ``NaN`` marks null values in the LAS file.
    units : dict[str, str]
        Per-curve units from the LAS header; empty string when the
        unit field was blank.
    well : dict[str, str]
        The ``~Well`` section as a flat dict (keys like ``"WELL"``,
        ``"COMPANY"``, ``"FLD"``, ``"SRVC"``).
    step : float
        Constant sampling step from the LAS header (``STEP``). May be
        ``NaN`` when the file does not use a uniform step.
    """

    depth: np.ndarray
    curves: dict[str, np.ndarray]
    units: dict[str, str]
    well: dict[str, str]
    step: float


def read_las(path: str) -> LasCurves:
    """
    Read a LAS file into a :class:`LasCurves` container.

    All curves are returned as ``float64`` NumPy arrays with null
    values replaced by ``NaN``.

    Parameters
    ----------
    path : str
        Filesystem path to a LAS file. LAS 2.0 and 3.0 are both
        supported via ``lasio``.

    Returns
    -------
    LasCurves
    """
    las = lasio.read(path)
    depth = np.asarray(las.index, dtype=float)

    curves: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    for i, curve in enumerate(las.curves):
        if i == 0:
            # The first curve is the depth axis, already extracted.
            continue
        curves[curve.mnemonic] = np.asarray(curve.data, dtype=float)
        units[curve.mnemonic] = str(curve.unit) if curve.unit else ""

    well: dict[str, str] = {}
    for item in las.well:
        well[item.mnemonic] = str(item.value) if item.value is not None else ""

    try:
        step = float(las.well.STEP.value)
    except (AttributeError, TypeError, ValueError):
        step = float("nan")

    return LasCurves(depth=depth, curves=curves, units=units, well=well, step=step)


def write_las(
    path: str,
    depth: np.ndarray,
    curves: Mapping[str, np.ndarray],
    *,
    depth_unit: str = "M",
    well_name: str = "",
    well: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
) -> None:
    """
    Write an fwap-derived log set out as a LAS file.

    Units for the common fwap curves (DTP, DTS, DTST, COH*, VPVS, Q,
    K, MU, E, NU) are filled in automatically from an internal table.
    Pass ``units`` to override or supply units for custom mnemonics.

    Parameters
    ----------
    path : str
        Output LAS path.
    depth : ndarray, shape (n_depth,)
        Depth axis. Must match the first dimension of every curve
        array.
    curves : mapping from str to ndarray
        ``{mnemonic: ndarray}``. Each array must have shape
        ``(n_depth,)``; ``NaN`` is written as the LAS null value.
    depth_unit : str, default ``"M"``
        Unit for the depth axis (LAS convention: ``"M"`` or ``"FT"``).
    well_name : str, default empty
        Value for the LAS ``WELL`` header entry.
    well : mapping from str to str, optional
        Additional ``~Well``-section entries. Keys must be the
        standard LAS 2.0 mnemonics (``COMP`` for company, ``FLD``
        for field, ``SRVC`` for service company, ``UWI``, etc.).
        Non-standard mnemonics are rejected by ``lasio``.
    units : mapping from str to str, optional
        Per-curve unit override, keyed by mnemonic. Missing entries
        fall back to the built-in table and then to empty.

    Raises
    ------
    ValueError
        If any curve has a length different from ``depth``.
    """
    depth = np.asarray(depth, dtype=float)
    n = depth.size
    for mnemonic, arr in curves.items():
        if np.asarray(arr).shape != (n,):
            raise ValueError(
                f"curve {mnemonic!r} has shape {np.asarray(arr).shape}, expected ({n},)"
            )

    las = lasio.LASFile()
    las.well["WELL"] = well_name
    if well is not None:
        for k, v in well.items():
            las.well[k] = v

    unit_table: dict[str, str] = dict(_FWAP_UNITS)
    if units:
        unit_table.update(units)

    las.append_curve("DEPT", depth, unit=depth_unit, descr="depth")
    for mnemonic, arr in curves.items():
        las.append_curve(
            mnemonic,
            np.asarray(arr, dtype=float),
            unit=unit_table.get(mnemonic, ""),
            descr="",
        )
    las.write(path, version=2.0)
