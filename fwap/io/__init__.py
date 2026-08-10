"""
I/O wrappers for the formats sonic data arrives in.

The package re-exports the public LAS / DLIS / SEG-Y entry points
from the per-format submodules so that ``from fwap.io import
read_las`` (and the existing top-level ``from fwap import read_las``
aliases) keep working unchanged.

Subpackage layout
-----------------
* :mod:`fwap.io._common` -- internal constants shared by the LAS
  and DLIS writers (``_FWAP_UNITS``).
* :mod:`fwap.io._las`    -- LAS via ``lasio``.
* :mod:`fwap.io._dlis`   -- DLIS via ``dlisio`` (read) +
  ``dliswriter`` (write).
* :mod:`fwap.io._segy`   -- SEG-Y via ``segyio``.

References
----------
* Canadian Well Logging Society (1991). *Log ASCII Standard (LAS)
  Version 2.0.*
* American Petroleum Institute (1991). *Recommended Practices for
  Exploration and Production Data Digital Interchange (RP66 v1).*
* Society of Exploration Geophysicists (2017). *SEG Y rev 2.0 Data
  Exchange Format.*
* Mari, J.-L., & Vergniault, C. (2018). *Well Seismic Surveying and
  Acoustic Logging*, Chapter 1 (log formats). EDP Open.
"""

from __future__ import annotations

from fwap.io._dlis import (
    DlisAxis,
    DlisCurves,
    DlisWaveforms,
    read_dlis,
    read_dlis_waveforms,
    write_dlis,
)
from fwap.io._las import LasCurves, read_las, write_las
from fwap.io._ldeo import (
    LDEO_MODE_NAMES,
    LDEO_TOOL_NAMES,
    LdeoWaveforms,
    read_ldeo_waveforms,
)
from fwap.io._segy import SegyGather, read_segy, write_segy

__all__ = [
    "DlisAxis",
    "DlisCurves",
    "DlisWaveforms",
    "LasCurves",
    "LdeoWaveforms",
    "LDEO_MODE_NAMES",
    "LDEO_TOOL_NAMES",
    "SegyGather",
    "read_dlis",
    "read_dlis_waveforms",
    "read_las",
    "read_ldeo_waveforms",
    "read_segy",
    "write_dlis",
    "write_las",
    "write_segy",
]
