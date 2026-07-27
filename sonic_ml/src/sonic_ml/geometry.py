"""
Tool-geometry reconstruction for a loaded dataset.

The surrogate ``.npz`` (schema v1) stores the waveform ``gather`` but **not**
the acquisition geometry (sampling interval ``dt`` and receiver offsets) needed
to process it with fwap's slowness-time estimators. :func:`default_geometry`
reconstructs the generator's *default* :class:`fwap.ArrayGeometry` at the
stored gather's ``(n_rec, n_samples)``.

.. warning::

   This is only correct when the dataset was generated with the default
   :class:`~fwap.ArrayGeometry` (``dt = 1e-5 s``, ``dr = 0.1524 m``,
   ``tr_offset = 3.0 m``). If a custom geometry was used, pass that geometry
   explicitly to the baselines/harness instead. Persisting the geometry in the
   ``.npz`` (a schema v2 that makes gathers self-describing) is the clean
   long-term fix.
"""

from __future__ import annotations

from fwap import ArrayGeometry

from sonic_ml.loader import DatasetBundle


def default_geometry(bundle: DatasetBundle) -> ArrayGeometry:
    """
    Acquisition geometry for a bundle's gathers.

    Prefers the geometry persisted in the dataset (schema v2+, exposed as
    ``bundle.geometry``). Falls back to reconstructing the generator's
    *default* :class:`~fwap.ArrayGeometry` for legacy schema v1 files, which
    is only correct if that dataset used the default ``dt`` / ``dr`` /
    ``tr_offset`` (see the module warning).

    Parameters
    ----------
    bundle : DatasetBundle
        A dataset whose ``gather`` has shape ``(N, n_rec, n_samples)``.

    Returns
    -------
    fwap.ArrayGeometry
        The stored geometry (v2) or the default-reconstructed geometry (v1).

    Raises
    ------
    ValueError
        If the gather is not 3-D (only reachable for a v1 fallback).
    """
    if bundle.geometry is not None:
        return bundle.geometry
    if bundle.gather.ndim != 3:
        raise ValueError(
            f"expected a 3-D gather (N, n_rec, n_samples), got shape "
            f"{bundle.gather.shape}"
        )
    n_rec = int(bundle.gather.shape[1])
    n_samples = int(bundle.gather.shape[2])
    return ArrayGeometry(n_rec=n_rec, n_samples=n_samples)
