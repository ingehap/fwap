"""
SEG-Y reader and writer.

Wraps `segyio <https://segyio.readthedocs.io/>`_.
Reference: Society of Exploration Geophysicists
(2017), *SEG Y rev 2.0 Data Exchange Format*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import segyio


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


def read_segy(
    path: str,
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
            textual = segyio.tools.wrap(f.text[0].decode("ascii", errors="replace"))
        except Exception:  # pragma: no cover - guard against oddly-encoded files
            textual = ""

    return SegyGather(
        data=data,
        dt=dt,
        offsets=offsets,
        n_traces=int(n_traces),
        n_samples=int(n_samples),
        textual_header=textual,
    )


def write_segy(
    path: str,
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
                f"offsets must have shape ({n_traces},); got {offsets_arr.shape}"
            )
        offsets_arr = np.round(offsets_arr).astype(int)

    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = np.arange(n_samples)
    spec.tracecount = n_traces
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        if textual_header is not None:
            # segyio.tools.create_text_header pads/truncates to the
            # required 40-line x 80-char layout.
            f.text[0] = segyio.tools.create_text_header(
                {i + 1: "" for i in range(40)}  # placeholder: fill with blanks
            )
            # Overwrite with the caller's content, clipped to 3200 bytes.
            encoded = textual_header.encode("ascii", errors="replace")
            encoded = encoded[:3200].ljust(3200, b" ")
            f.text[0] = encoded
        for i in range(n_traces):
            f.header[i].update(
                {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.offset: int(offsets_arr[i]),
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                }
            )
            f.trace[i] = data[i]
