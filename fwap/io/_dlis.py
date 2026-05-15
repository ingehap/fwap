"""
DLIS (Digital Log Interchange Standard, API RP66 v1)
reader and writer.

Read uses `dlisio <https://dlisio.readthedocs.io>`_;
write uses `dliswriter
<https://dliswriter.readthedocs.io>`_. Reference: API
RP66 v1 (1991).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import dliswriter
import numpy as np
from dlisio import dlis as dlisio_dlis

from fwap.io._common import _FWAP_UNITS


@contextlib.contextmanager
def _suppress_fd(fd: int) -> Iterator[None]:
    """
    Redirect a file descriptor to ``/dev/null`` for the duration of
    the ``with`` block. Used to silence libraries that write directly
    to fd 1 / fd 2 (e.g. ``progressbar2`` inside ``dliswriter``)
    rather than through ``sys.stdout`` / ``sys.stderr``.
    """
    saved = os.dup(fd)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


_DLIS_TO_LAS_WELL: Mapping[str, str] = {
    "well_name": "WELL",
    "company": "COMP",
    "field_name": "FLD",
    "producer_name": "PROD",
    "well_id": "UWI",
}
_LAS_TO_DLIS_WELL: Mapping[str, str] = {v: k for k, v in _DLIS_TO_LAS_WELL.items()}


@dataclass
class DlisCurves:
    """
    A DLIS frame loaded into memory.

    Mirrors :class:`LasCurves` so curves can move between LAS and DLIS
    without the caller having to translate field names.

    Attributes
    ----------
    depth : ndarray, shape (n_depth,)
        The frame's index channel (typically borehole depth, in the
        unit declared by the channel header -- usually metres or feet).
    curves : dict[str, ndarray]
        One entry per non-index channel, keyed by channel name. Arrays
        have the same length as ``depth``. ``NaN`` round-trips natively
        as the IEEE-754 NaN bit pattern.
    units : dict[str, str]
        Per-channel units from the DLIS header; empty string when no
        unit was set.
    well : dict[str, str]
        Origin metadata, re-keyed to the LAS-2.0 mnemonics
        (``WELL``, ``COMP``, ``FLD``, ``PROD``, ``UWI``) so that the
        same dict can be passed to :func:`write_las`. Origin fields
        without a LAS analogue are dropped.
    step : float
        Frame index spacing (``Frame.spacing`` in DLIS); ``NaN`` if
        the file does not declare a constant spacing.
    frame_name : str
        Name of the DLIS frame the data was read from.
    index_type : str
        DLIS frame index type (e.g. ``"BOREHOLE-DEPTH"``,
        ``"VERTICAL-DEPTH"``, ``"TIME"``). Empty string when the frame
        has no declared index type.
    """

    depth: np.ndarray
    curves: dict[str, np.ndarray]
    units: dict[str, str]
    well: dict[str, str]
    step: float
    frame_name: str
    index_type: str


def read_dlis(
    path: str,
    *,
    logical_file_index: int = 0,
    frame_index: int = 0,
) -> DlisCurves:
    """
    Read one frame of a DLIS (RP66 v1) file into a :class:`DlisCurves`.

    A DLIS file is a container that may hold multiple Logical Files,
    each of which may hold multiple Frames; ``logical_file_index`` and
    ``frame_index`` select one. The defaults read the first frame of
    the first logical file, which covers the common single-pass /
    single-frame case.

    All channels are returned as ``float64`` NumPy arrays. ``NaN`` is
    preserved as IEEE-754 NaN. Multi-dimensional channels (e.g. array
    waveforms) are skipped: only scalar-per-sample curves are loaded
    -- the dataclass mirrors :class:`LasCurves`, which assumes scalar
    curves.

    Parameters
    ----------
    path : str
        Filesystem path to a DLIS file.
    logical_file_index : int, default 0
        Which logical file to read.
    frame_index : int, default 0
        Which frame inside that logical file to read.

    Returns
    -------
    DlisCurves

    Raises
    ------
    IndexError
        If ``logical_file_index`` or ``frame_index`` is out of range
        for the file.
    """
    with dlisio_dlis.load(path) as files:
        if logical_file_index >= len(files):
            raise IndexError(
                f"logical_file_index={logical_file_index} but file has "
                f"only {len(files)} logical file(s)"
            )
        lf = files[logical_file_index]
        if frame_index >= len(lf.frames):
            raise IndexError(
                f"frame_index={frame_index} but logical file has only "
                f"{len(lf.frames)} frame(s)"
            )
        frame = lf.frames[frame_index]

        # The frame's curves() method returns a structured ndarray with
        # one named field per channel, plus the implicit "FRAMENO"
        # column. The first non-FRAMENO channel is the index.
        rec = frame.curves()
        channel_names = [ch.name for ch in frame.channels]
        if not channel_names:
            raise ValueError(f"frame {frame.name!r} has no channels")
        depth_name = channel_names[0]
        depth = np.asarray(rec[depth_name], dtype=float)

        curves: dict[str, np.ndarray] = {}
        units: dict[str, str] = {}
        for ch in frame.channels:
            if ch.name == depth_name:
                continue
            arr = np.asarray(rec[ch.name])
            # Skip vector channels -- the LAS-style container expects
            # one sample per depth.
            if arr.ndim != 1:
                continue
            curves[ch.name] = arr.astype(float)
            units[ch.name] = str(ch.units) if ch.units else ""

        well: dict[str, str] = {}
        if lf.origins:
            origin = lf.origins[0]
            for dlis_attr, las_key in _DLIS_TO_LAS_WELL.items():
                value = getattr(origin, dlis_attr, None)
                if value is not None and value != "":
                    well[las_key] = str(value)

        spacing = frame.spacing
        step = float(spacing) if spacing is not None else float("nan")
        index_type = str(frame.index_type) if frame.index_type else ""

    return DlisCurves(
        depth=depth,
        curves=curves,
        units=units,
        well=well,
        step=step,
        frame_name=str(frame.name),
        index_type=index_type,
    )


def write_dlis(
    path: str,
    depth: np.ndarray,
    curves: Mapping[str, np.ndarray],
    *,
    depth_unit: str = "m",
    well_name: str = "",
    well: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    frame_name: str = "MAIN",
    index_type: str = "BOREHOLE-DEPTH",
    origin_name: str = "FWAP",
) -> None:
    """
    Write an fwap-derived log set out as a DLIS (RP66 v1) file.

    Mirror of :func:`write_las`: ``depth`` becomes the index channel
    of a single frame named ``frame_name`` inside a single logical
    file, and ``curves`` are added as non-index channels. Units for
    the standard fwap mnemonics (``DTP``, ``DTS``, ``DTST``,
    ``COH*``, ``VPVS``, ``Q``, ``K``, ``MU``, ``E``, ``NU``) are
    filled in from the same internal table used by :func:`write_las`.

    Parameters
    ----------
    path : str
        Output DLIS path.
    depth : ndarray, shape (n_depth,)
        Depth axis. Becomes the frame's index channel and must match
        the first dimension of every curve array.
    curves : mapping from str to ndarray
        ``{channel_name: ndarray}``. Each array must have shape
        ``(n_depth,)``; ``NaN`` is preserved as IEEE-754 NaN.
    depth_unit : str, default ``"m"``
        Unit for the depth axis.
    well_name : str, default empty
        Convenience for the ``"WELL"`` entry, identical to placing
        ``"WELL"`` in ``well``.
    well : mapping from str to str, optional
        Origin metadata. Keys use the LAS-2.0 mnemonics
        (``"WELL"``, ``"COMP"``, ``"FLD"``, ``"PROD"``, ``"UWI"``)
        so the same dict works with :func:`write_las`. Other keys
        are silently ignored.
    units : mapping from str to str, optional
        Per-channel unit override, keyed by channel name. Missing
        entries fall back to the built-in table and then to empty.
    frame_name : str, default ``"MAIN"``
        Name of the DLIS frame written.
    index_type : str, default ``"BOREHOLE-DEPTH"``
        DLIS frame index type. Common values:
        ``"BOREHOLE-DEPTH"``, ``"VERTICAL-DEPTH"``, ``"TIME"``.
    origin_name : str, default ``"FWAP"``
        Name of the Origin record. Identifies the producer of the
        logical file; rarely user-visible.

    Raises
    ------
    ValueError
        If any curve has a length different from ``depth``.

    Notes
    -----
    DLIS standard units are restricted to a small whitelist (see
    RP66 v1 Appendix B). Non-canonical units like ``us/ft`` are
    accepted by ``dliswriter`` but trigger an info-level log entry
    on its ``dliswriter.utils.internal.validator_enum`` logger; this
    is suppressed during the write so the file is produced silently.
    """
    depth = np.asarray(depth, dtype=float)
    n = depth.size
    for name, arr in curves.items():
        if np.asarray(arr).shape != (n,):
            raise ValueError(
                f"curve {name!r} has shape {np.asarray(arr).shape}, expected ({n},)"
            )

    # Resolve well metadata: well_name kwarg, then well mapping,
    # filtered through the LAS->DLIS key map.
    origin_kwargs: dict[str, str] = {}
    if well_name:
        origin_kwargs["well_name"] = well_name
    if well is not None:
        for las_key, value in well.items():
            dlis_attr = _LAS_TO_DLIS_WELL.get(las_key)
            if dlis_attr is not None and value:
                origin_kwargs[dlis_attr] = value

    unit_table: dict[str, str] = dict(_FWAP_UNITS)
    if units:
        unit_table.update(units)

    # ``dliswriter`` logs an info/warning for every unit and every
    # empty-unit channel that is not in the RP66 v1 Appendix B
    # whitelist (``us/ft`` and the empty string are not), and renders
    # a ``progressbar2`` bar straight to fd 2 during the write. The
    # file is still produced correctly; silence both for the duration
    # of the build + write so callers get a clean stream.
    import logging

    validator_logger = logging.getLogger("dliswriter.utils.internal.validator_enum")
    prev_level = validator_logger.level
    validator_logger.setLevel(logging.ERROR)
    try:
        with _suppress_fd(2):
            f = dliswriter.DLISFile()
            lf = f.add_logical_file()
            # ``add_origin`` declares each kwarg as an Optional[str|int|...]
            # union; mypy cannot infer from our static ``_LAS_TO_DLIS_WELL``
            # map that ``origin_kwargs`` only ever contains string-typed
            # parameters. Cast the unpacked mapping to silence that.
            lf.add_origin(origin_name, **cast(Any, origin_kwargs))

            depth_channel = lf.add_channel(name="DEPT", data=depth, units=depth_unit)
            channel_objs = [depth_channel]
            for name, arr in curves.items():
                channel_objs.append(
                    lf.add_channel(
                        name=name,
                        data=np.asarray(arr, dtype=float),
                        units=unit_table.get(name, ""),
                    )
                )
            lf.add_frame(
                frame_name,
                channels=tuple(channel_objs),
                index_type=index_type,
            )
            f.write(path)
    finally:
        validator_logger.setLevel(prev_level)
