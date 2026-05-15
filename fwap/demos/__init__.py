"""
Synthetic demonstrations of each fwap algorithm family.

One demo per chapter of Mari, Coppens, Gavin & Wicquart (1994),
*Full Waveform Acoustic Data Processing*, plus the I/O round-trip
and extension demos. The package re-exports every ``demo_*``
function from its themed submodule so callers can keep using
``from fwap.demos import demo_X`` (and ``fwap.cli`` keeps
dispatching via ``fwap.demos.demo_X``).

Subpackage layout
-----------------
* :mod:`fwap.demos._common`       -- canonical synthetic gather
  helper shared across the picking, separation, tau-p, and SEG-Y
  round-trip demos.
* :mod:`fwap.demos._signal`       -- Part 1 + 2 demos (STC
  picking; f-k / K-L / tau-p separation).
* :mod:`fwap.demos._inversion`    -- Part 3 + 4 demos (intercept-
  time inversion; dipole flexural dispersion; dip + azimuth).
* :mod:`fwap.demos._extensions`   -- Q / anisotropy / LWD demos.
* :mod:`fwap.demos._io_roundtrip` -- LAS / DLIS / SEG-Y round-trip
  demos.
"""

from __future__ import annotations

from fwap._common import logger
from fwap.demos._extensions import (
    demo_alford,
    demo_attenuation,
    demo_lwd,
)
from fwap.demos._inversion import (
    demo_dip,
    demo_dipole,
    demo_intercept_time,
)
from fwap.demos._io_roundtrip import (
    demo_dlis_roundtrip,
    demo_las_roundtrip,
    demo_segy_roundtrip,
)
from fwap.demos._signal import (
    demo_pseudo_rayleigh,
    demo_stc_picker,
    demo_tau_p_separation,
    demo_wave_separation,
)

__all__ = [
    "logger",
    "demo_alford",
    "demo_attenuation",
    "demo_dip",
    "demo_dipole",
    "demo_dlis_roundtrip",
    "demo_intercept_time",
    "demo_las_roundtrip",
    "demo_lwd",
    "demo_pseudo_rayleigh",
    "demo_segy_roundtrip",
    "demo_stc_picker",
    "demo_tau_p_separation",
    "demo_wave_separation",
]
