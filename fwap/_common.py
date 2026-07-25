"""Constants and small helpers shared across the fwap package."""

from __future__ import annotations

import logging
from typing import overload

import numpy as np

# 1 microsecond per foot in seconds per metre (exact).
US_PER_FT: float = 1.0e-6 / 0.3048


# Overloaded so an array in yields an array out (and a scalar in a
# scalar out): callers that build ``dict[str, np.ndarray]`` log-curve
# maps from array velocities (the LAS/DLIS writers) then type-check
# cleanly. Without the overload the plain ``np.ndarray | float`` return
# leaks a ``float`` into those dicts, which recent numpy stubs reject
# against the writers' ``Mapping[str, np.ndarray]`` signature.
@overload
def m_per_s_to_us_per_ft(v: float) -> float: ...


@overload
def m_per_s_to_us_per_ft(v: np.ndarray) -> np.ndarray: ...


def m_per_s_to_us_per_ft(v: np.ndarray | float) -> np.ndarray | float:
    """Convert a velocity in m/s to a slowness in us/ft.

    Inverse of multiplying by :data:`US_PER_FT`. Used by the LAS / DLIS
    writers (CLI + demos) which emit slowness logs in the borehole-
    acoustic unit ``us/ft`` while every other interface in the package
    speaks SI seconds per metre.
    """
    return 1.0 / (v * US_PER_FT)


# Package-wide logger. Every module that wants to log should import
# this rather than calling ``logging.getLogger("fwap")`` directly, so
# that the name is fixed in one place.
logger: logging.Logger = logging.getLogger("fwap")


def _phase_shift(spec: np.ndarray, f: np.ndarray, tau: np.ndarray) -> np.ndarray:
    r"""
    Frequency-domain fractional time shift.

    A **positive** ``tau[i]`` advances trace ``i`` by ``tau[i]``
    seconds: a ``delta(t - t0)`` arrival on trace ``i`` becomes
    ``delta(t - (t0 - tau[i]))``. Under NumPy's forward-FFT convention
    :math:`X(f) = \sum_n x(n)\,e^{-2\pi i f n}`, a time advance by
    ``tau`` corresponds to multiplication by
    :math:`e^{+2\pi i f \tau}`.

    This sign convention is load-bearing for the whole package. It is
    relied on by:

    * :func:`fwap.coherence.stc` (STC moveout)
    * :func:`fwap.dispersion.dispersive_stc` (dispersion-corrected STC)
    * :func:`fwap.wavesep.apply_moveout` / ``unapply_moveout``
    * :func:`fwap.dip._coherence_after_detilt` (azimuthal detilt)

    A silent flip here would corrupt every one of those results, so
    the sign is pinned by ``tests/test_common.py``.
    """
    phase = np.exp(1j * 2.0 * np.pi * np.outer(tau, f))
    return spec * phase
