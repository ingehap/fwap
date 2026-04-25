"""
I/O wrappers for the formats sonic data arrives in.

* :func:`read_las` / :func:`write_las` -- LAS (Log ASCII Standard),
  the most common exchange format for *log curves* (the Vp/Vs/Stoneley
  outputs of the processing chain). Wraps
  `lasio <https://lasio.rtfd.io>`_.

* :func:`read_dlis` / :func:`write_dlis` -- DLIS (Digital Log
  Interchange Standard, API RP66 v1), the binary equivalent of LAS
  used by modern wireline acquisition systems. Read uses
  `dlisio <https://dlisio.readthedocs.io>`_; write uses
  `dliswriter <https://dliswriter.readthedocs.io>`_.

* :func:`read_segy` -- SEG-Y is the standard format for *waveform*
  dumps (the inputs to the processing chain). Wraps
  `segyio <https://segyio.readthedocs.io/>`_.

All four underlying libraries (``lasio``, ``dlisio``, ``dliswriter``,
``segyio``) are core fwap dependencies, so the LAS / DLIS / SEG-Y
readers and writers are always available.

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

import contextlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import dliswriter
import lasio
import numpy as np
import segyio
from dlisio import dlis as dlisio_dlis


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

    return LasCurves(depth=depth, curves=curves, units=units,
                     well=well, step=step)


_FWAP_UNITS: Mapping[str, str] = {
    # Compressional / shear / Stoneley / pseudo-Rayleigh slowness
    # (us/ft is the borehole-acoustic convention; consumers can
    # convert on read).
    "DTP":   "us/ft",
    "DTS":   "us/ft",
    "DTST":  "us/ft",
    "DTPR":  "us/ft",
    # Per-mode coherence (unitless).
    "COHP":  "",
    "COHS":  "",
    "COHST": "",
    "COHPR": "",
    # Per-mode stack amplitude (input units; dimensionless when the
    # source gather was unit-amplitude).
    "AMPP":  "",
    "AMPS":  "",
    "AMPST": "",
    "AMPPR": "",
    # Per-mode pick time (window-start time, seconds).
    "TIMP":  "s",
    "TIMS":  "s",
    "TIMST": "s",
    "TIMPR": "s",
    # Vp / Vs ratio.
    "VPVS":  "",
    # Q (dimensionless).
    "QP":    "",
    "QS":    "",
    # Elastic moduli.
    "K":     "Pa",
    "MU":    "Pa",
    "E":     "Pa",
    "NU":    "",
}


def write_las(path: str,
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
                f"curve {mnemonic!r} has shape {np.asarray(arr).shape}, "
                f"expected ({n},)"
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


# ---------------------------------------------------------------------
# DLIS reader / writer
# ---------------------------------------------------------------------

# Map between the LAS-2.0 well-section mnemonics and the DLIS Origin
# attributes that carry the same information. Using the LAS keys on
# both sides means a curve set can move between the two formats
# without the caller having to re-key the well dict.
_DLIS_TO_LAS_WELL: Mapping[str, str] = {
    "well_name":     "WELL",
    "company":       "COMP",
    "field_name":    "FLD",
    "producer_name": "PROD",
    "well_id":       "UWI",
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


def read_dlis(path: str,
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


def write_dlis(path: str,
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
                f"curve {name!r} has shape {np.asarray(arr).shape}, "
                f"expected ({n},)"
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
    validator_logger = logging.getLogger(
        "dliswriter.utils.internal.validator_enum"
    )
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

            depth_channel = lf.add_channel(
                name="DEPT", data=depth, units=depth_unit
            )
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


# ---------------------------------------------------------------------
# SEG-Y reader
# ---------------------------------------------------------------------


@dataclass
class SegyGather:
    """
    A SEG-Y file loaded into memory as a single gather.

    Attributes
    ----------
    data : ndarray, shape (n_traces, n_samples)
        Trace data, one row per trace, in the file's original sample
        order.
    dt : float
        Sample interval (s). Read from the binary header's ``Interval``
        field, which is stored in microseconds; we convert to seconds.
    offsets : ndarray, shape (n_traces,) or None
        Source-to-receiver offsets (m) extracted from a trace header
        field (``offset`` by default). ``None`` if the header field
        was all zeros, which is common for synthetic files.
    n_traces : int
        Number of traces in the file.
    n_samples : int
        Samples per trace.
    textual_header : str
        The 3200-byte EBCDIC textual file header, decoded to ASCII.
    """
    data: np.ndarray
    dt: float
    offsets: np.ndarray | None
    n_traces: int
    n_samples: int
    textual_header: str


def read_segy(path: str,
              *,
              offset_header: str = "offset",
              ) -> SegyGather:
    """
    Read a SEG-Y file into a :class:`SegyGather`.

    This is the minimal reader needed to feed the
    :mod:`fwap.coherence` / :mod:`fwap.dispersion` processing chain
    from real sonic data. It assumes the file contains one gather
    (all traces share a source position); callers with multi-gather
    data should split on ``FieldRecord`` before reading with this
    function.

    Parameters
    ----------
    path : str
        Filesystem path to a SEG-Y rev 1 or rev 2 file.
    offset_header : str, default ``"offset"``
        Name of the :class:`segyio.TraceField` attribute from which
        to read per-trace source-to-receiver offsets. Common choices:
        ``"offset"`` (standard offset field, bytes 37-40),
        ``"GroupX"`` / ``"SourceX"`` for receiver / source positions.
        If the chosen header is all zero the returned ``offsets``
        attribute is ``None``.

    Returns
    -------
    SegyGather

    Raises
    ------
    AttributeError
        If ``offset_header`` is not a valid
        :class:`segyio.TraceField` name.
    """
    # strict=False accepts non-standard geometries (common for borehole
    # data where traces are indexed by receiver, not CDP).
    with segyio.open(path, mode="r", strict=False, ignore_geometry=True) as f:
        data = np.stack([np.asarray(tr, dtype=float) for tr in f.trace])
        n_traces, n_samples = data.shape
        # segyio stores ``dt`` in microseconds in the binary header;
        # segyio.dt(f) returns it as a float in microseconds.
        dt_us = segyio.dt(f)
        dt = float(dt_us) * 1.0e-6

        # Offsets: look up the named TraceField, read it from every
        # trace header, coerce to float, and fall back to None if the
        # column is all zero.
        field = getattr(segyio.TraceField, offset_header)
        raw = np.array(
            [int(f.header[i][field]) for i in range(n_traces)],
            dtype=float,
        )
        offsets = raw if np.any(raw != 0.0) else None

        # The 3200-byte EBCDIC header as ASCII. segyio exposes it on
        # the context-manager object.
        try:
            textual = segyio.tools.wrap(f.text[0].decode("ascii",
                                                         errors="replace"))
        except Exception:   # pragma: no cover - guard against oddly-encoded files
            textual = ""

    return SegyGather(
        data=data,
        dt=dt,
        offsets=offsets,
        n_traces=int(n_traces),
        n_samples=int(n_samples),
        textual_header=textual,
    )


def write_segy(path: str,
               data: np.ndarray,
               dt: float,
               offsets: np.ndarray | None = None,
               *,
               textual_header: str | None = None,
               ) -> None:
    """
    Write a multichannel gather to a SEG-Y rev 1 file (IEEE float).

    Mirror of :func:`read_segy` on the output side. Useful for piping
    processed synthetics back out as SEG-Y for interchange with other
    seismic software.

    Parameters
    ----------
    path : str
        Output file path.
    data : ndarray, shape (n_traces, n_samples)
        Trace data. Written as IEEE 32-bit float (SEG-Y format code 5).
    dt : float
        Sample interval (s). Converted to microseconds for the binary
        header's ``Interval`` field; values below 1 us are clipped.
    offsets : ndarray, shape (n_traces,), optional
        Source-to-receiver offsets written into the ``offset`` trace
        header field. Coerced to ``int`` because the SEG-Y header is
        a 32-bit integer. If ``None``, all-zero offsets are written.
    textual_header : str, optional
        Free-form text for the 3200-byte EBCDIC header. Truncated /
        padded to 3200 bytes as required by SEG-Y.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, ``offsets`` has the wrong length, or
        ``dt`` is non-positive.
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {data.shape}")
    n_traces, n_samples = data.shape
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")
    dt_us = max(1, int(round(dt * 1.0e6)))

    if offsets is None:
        offsets_arr = np.zeros(n_traces, dtype=int)
    else:
        offsets_arr = np.asarray(offsets)
        if offsets_arr.shape != (n_traces,):
            raise ValueError(
                f"offsets must have shape ({n_traces},); got "
                f"{offsets_arr.shape}"
            )
        offsets_arr = np.round(offsets_arr).astype(int)

    spec = segyio.spec()
    spec.format = 5                            # IEEE float
    spec.samples = np.arange(n_samples)
    spec.tracecount = n_traces
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        if textual_header is not None:
            # segyio.tools.create_text_header pads/truncates to the
            # required 40-line x 80-char layout.
            f.text[0] = segyio.tools.create_text_header(
                {i + 1: ""
                 for i in range(40)}   # placeholder: fill with blanks
            )
            # Overwrite with the caller's content, clipped to 3200 bytes.
            encoded = textual_header.encode("ascii", errors="replace")
            encoded = encoded[:3200].ljust(3200, b" ")
            f.text[0] = encoded
        for i in range(n_traces):
            f.header[i].update({
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.offset: int(offsets_arr[i]),
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
            })
            f.trace[i] = data[i]
