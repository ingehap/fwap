"""
Canonical synthetic test configuration shared by several demos.

The :data:`_CANONICAL_VP` / :data:`_CANONICAL_VS` /
:data:`_CANONICAL_VST` constants and the
:func:`_canonical_monopole_gather` helper centralise the
reference fast-formation monopole gather (Vp=4500, Vs=2500,
Vst=1400 m/s) used by the picker, wave-separation, tau-p,
and SEG-Y round-trip demos.
"""

from __future__ import annotations

import numpy as np

from fwap.synthetic import (
    ArrayGeometry,
    monopole_formation_modes,
    synthesize_gather,
)

_CANONICAL_VP = 4500.0
_CANONICAL_VS = 2500.0
_CANONICAL_VST = 1400.0


def _canonical_monopole_gather(
    seed: int = 42,
    noise: float = 0.05,
) -> tuple[ArrayGeometry, np.ndarray, float, float, float]:
    """
    Build the shared (geometry, gather, Vp, Vs, Vst) used by the
    ``demo_stc_picker`` and ``demo_wave_separation`` demos.

    Returns the geometry alongside the synthetic gather so callers can
    re-derive per-receiver offsets and time axes without recomputing.
    """
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    modes = monopole_formation_modes(
        vp=_CANONICAL_VP, vs=_CANONICAL_VS, v_stoneley=_CANONICAL_VST
    )
    data = synthesize_gather(geom, modes, noise=noise, seed=seed)
    return geom, data, _CANONICAL_VP, _CANONICAL_VS, _CANONICAL_VST
