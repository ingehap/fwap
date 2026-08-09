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
from dataclasses import dataclass, field
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


# RP66 v1 declares a unit string on every attribute, so an axis says what
# its COORDINATES and SPACING mean rather than leaving the reader to guess.
# These tables are what turn that declaration into SI. Keys are compared
# case-insensitively; the entries cover what appears on array-sonic axes.
_TIME_UNIT_TO_S: Mapping[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "ms": 1.0e-3,
    "msec": 1.0e-3,
    "us": 1.0e-6,
    "usec": 1.0e-6,
    "ns": 1.0e-9,
}
_LENGTH_UNIT_TO_M: Mapping[str, float] = {
    "m": 1.0,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "in": 0.0254,
    "ft": 0.3048,
}


# Service-company PARAMETER records naming a digitiser sample interval.
# RP66 v1 puts acquisition geometry in AXIS objects, which carry a declared
# unit; this prefix is a *convention*, not a standard, and is used only as a
# fallback -- see ``DlisWaveforms.sample_interval``. Files that declare no
# AXIS at all are real: ODP Leg 157 Hole 952A carries none, and ``DSI0`` is
# the only place its 10 us sample interval appears.
_SAMPLE_INTERVAL_PARAM_PREFIX: str = "DSI"

# A digitiser sample interval is quoted in microseconds by every file seen,
# but the parameter carries no declared unit, so the assumption is checked
# rather than trusted: the implied record length must be sonic-plausible.
# 10 us x 500 samples = 5 ms lands inside this; the same 10 read as
# milliseconds (5 s) or nanoseconds (5 us) does not.
_PLAUSIBLE_RECORD_SECONDS: tuple[float, float] = (1.0e-4, 1.0e-1)


# ``dliswriter`` buffers file bytes before each write() syscall, and its
# default buffer is 2**32 bytes -- a 4 GiB allocation for a file of any
# size. Measured on a 7.5 kB output: 62.6 s and 8.3 GB peak RSS with the
# default, against 0.45 s and 89 MB at 1 MiB, byte-identical either way.
# 8 MiB keeps the syscall count sane without the allocation.
_DLIS_OUTPUT_CHUNK_BYTES: int = 8 * 1024 * 1024


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
    waveform_channels: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass
class DlisAxis:
    """
    One axis of a multi-dimensional DLIS channel (RP66 v1 AXIS object).

    An array channel -- a sonic waveform, say -- carries one AXIS per
    dimension beyond depth, and each AXIS declares what its numbers mean.
    That is what makes :class:`DlisWaveforms` able to report a sample
    interval in seconds and receiver offsets in metres without any
    knowledge of the tool or the service company that wrote the file.

    Attributes
    ----------
    axis_id : str
        The AXIS-ID string, e.g. ``"MICRO_TIME"`` or
        ``"SENSOR_OFFSET"``. Producer-defined, so it is reported but
        never interpreted -- the unit is what carries the meaning.
    coordinates : ndarray, shape (n_coordinate,)
        The COORDINATES attribute as written. RP66 permits either every
        coordinate of the axis, or a single starting coordinate to be
        read together with ``spacing``; both occur in practice. Use
        :meth:`values` rather than this field to get one number per
        sample.
    spacing : float
        The SPACING attribute, in ``units``. ``NaN`` when the file
        declares none (an irregular axis lists every coordinate
        instead).
    units : str
        Unit of ``coordinates`` and ``spacing`` as declared in the
        file; empty string when the file declares none.
    size : int
        Length of the channel dimension this axis describes.
    """

    axis_id: str
    coordinates: np.ndarray
    spacing: float
    units: str
    size: int

    def values(self) -> np.ndarray:
        """
        One coordinate per sample, in the axis's own ``units``.

        Returns
        -------
        ndarray, shape (size,)

        Raises
        ------
        ValueError
            If the axis lists neither ``size`` coordinates nor a
            starting coordinate plus a finite ``spacing``, so that no
            per-sample coordinate can be recovered.
        """
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.size == self.size:
            return coords
        if coords.size >= 1 and np.isfinite(self.spacing):
            return coords[0] + self.spacing * np.arange(self.size, dtype=float)
        raise ValueError(
            f"axis {self.axis_id!r} declares {coords.size} coordinate(s) for a "
            f"dimension of {self.size} and spacing={self.spacing!r}, so "
            "per-sample coordinates cannot be recovered"
        )


@dataclass
class DlisWaveforms:
    """
    One multi-dimensional DLIS channel -- an array-sonic waveform set.

    :func:`read_dlis` returns scalar curves and skips channels like this
    one; :func:`read_dlis_waveforms` returns this instead.

    Attributes
    ----------
    channel : str
        Name of the channel that was read, e.g. ``"PWF4"``.
    data : ndarray, shape (n_depth, ...)
        The channel, depth-major. The trailing dimensions are the
        channel's own, one per entry of ``axes`` -- for a monopole
        array-sonic waveform typically ``(n_depth, n_receiver,
        n_sample)``.
    depth : ndarray, shape (n_depth,)
        The frame's index channel, in ``depth_unit`` -- **not** converted.
    depth_unit : str
        Unit of ``depth`` as declared in the file; empty when none.
        Worth reading rather than assuming: service-company files often
        index in tenths of an inch (``"0.1 in"``), so treating the raw
        numbers as feet or metres is out by a factor of 120 or 254.
    axes : list of DlisAxis
        One per trailing dimension of ``data``, in the same order.
        Empty when the file declares no AXIS for the channel, which is
        legal and leaves :meth:`sample_interval` and :meth:`offsets`
        with nothing to work from.
    units : str
        Unit of the sample values; array-sonic waveforms are commonly
        unitless (raw counts) and report an empty string.
    frame_name : str
        Name of the DLIS frame the channel was read from.
    """

    channel: str
    data: np.ndarray
    depth: np.ndarray
    depth_unit: str
    axes: list[DlisAxis]
    units: str
    frame_name: str
    sample_interval_parameters: dict[str, float] = field(default_factory=dict)

    def _sole_axis(
        self, table: Mapping[str, float], kind: str
    ) -> tuple[DlisAxis, float]:
        """Return the one axis whose declared unit is of ``kind``, and its scale."""
        matches = [
            (ax, table[ax.units.strip().lower()])
            for ax in self.axes
            if ax.units.strip().lower() in table
        ]
        if len(matches) == 1:
            return matches[0]
        declared = (
            ", ".join(f"{ax.axis_id!r} ({ax.units!r})" for ax in self.axes) or "none"
        )
        raise ValueError(
            f"channel {self.channel!r} has {len(matches)} axes with a {kind} unit, "
            f"expected exactly 1; declared axes: {declared}"
        )

    def _from_axis(self) -> float:
        """Sample interval from the single time-unit AXIS, in seconds."""
        axis, scale = self._sole_axis(_TIME_UNIT_TO_S, "time")
        spacing = axis.spacing
        if not np.isfinite(spacing):
            values = axis.values()
            if values.size < 2:
                raise ValueError(
                    f"time axis {axis.axis_id!r} of channel {self.channel!r} "
                    "declares no spacing and has fewer than two coordinates"
                )
            spacing = float(values[1] - values[0])
        return float(spacing) * scale

    def _from_parameter(self) -> tuple[float, str]:
        """Sample interval from the vendor parameter fallback, in seconds."""
        found = self.sample_interval_parameters
        if not found:
            raise ValueError(
                f"channel {self.channel!r} declares no AXIS and the file "
                f"carries no {_SAMPLE_INTERVAL_PARAM_PREFIX}* parameter to "
                "fall back on, so its sample interval is not recorded anywhere"
            )
        distinct = sorted({round(v, 9) for v in found.values()})
        if len(distinct) != 1:
            listed = ", ".join(f"{k}={v:g}" for k, v in sorted(found.items()))
            raise ValueError(
                f"channel {self.channel!r} declares no AXIS, and the file's "
                f"{_SAMPLE_INTERVAL_PARAM_PREFIX}* parameters disagree "
                f"({listed}) -- which one belongs to this channel is a "
                "vendor-specific question this reader will not guess at. "
                "Pass the interval explicitly."
            )
        seconds = distinct[0] * _TIME_UNIT_TO_S["us"]
        record = seconds * float(self.data.shape[-1])
        low, high = _PLAUSIBLE_RECORD_SECONDS
        if not low <= record <= high:
            raise ValueError(
                f"the {_SAMPLE_INTERVAL_PARAM_PREFIX}* fallback for channel "
                f"{self.channel!r} implies a {record:g} s record over "
                f"{self.data.shape[-1]} samples, which is not a sonic "
                "waveform; the microsecond convention does not hold here"
            )
        name = ", ".join(sorted(found))
        return seconds, f"parameter {name} (microseconds assumed)"

    def sample_interval(self) -> float:
        """
        Time between samples, in seconds, ready for :func:`fwap.stc`.

        Taken from the SPACING of the single axis whose declared unit is
        a time unit. Selecting on the *unit* rather than on the AXIS-ID
        string is deliberate: AXIS-ID values are producer-defined, units
        are not.

        **Falls back to a vendor parameter when the file declares no AXIS
        at all**, which happens: ODP Leg 157 Hole 952A carries zero AXIS
        objects, and its ``DSI0`` parameter is the only record of the 10 us
        interval. The fallback is deliberately timid, because a parameter
        carries no declared unit and guessing wrong is a factor-of-1000
        error:

        * it fires only when the file's ``DSI*`` parameters agree on one
          value -- if they disagree, as in a file carrying a separate
          interval per waveform, deciding which belongs to this channel is
          a vendor-specific question and this raises instead;
        * the microsecond convention is *checked*, not assumed: the implied
          record length must be sonic-plausible.

        Use :meth:`sample_interval_source` to see which route answered.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If neither route yields an unambiguous, plausible interval.
        """
        return self._sample_interval_with_source()[0]

    def sample_interval_source(self) -> str:
        """
        Where :meth:`sample_interval` got its number.

        Returns
        -------
        str
            e.g. ``"axis 'MICRO_TIME' (us)"`` or
            ``"parameter DSI0 (microseconds assumed)"``. Worth recording
            alongside any result derived from it: the first is read from a
            unit-bearing standard record, the second rests on a convention.
        """
        return self._sample_interval_with_source()[1]

    def _sample_interval_with_source(self) -> tuple[float, str]:
        """
        Sample interval and the provenance of the number.

        The fallback fires only when *no* axis carries a time unit. Several
        that do is a different situation -- the file said something, and it
        said too much -- so that keeps the AXIS diagnostics rather than
        reporting the file as silent when it is not.
        """
        matches = [
            ax for ax in self.axes if ax.units.strip().lower() in _TIME_UNIT_TO_S
        ]
        if len(matches) > 1:
            self._sole_axis(_TIME_UNIT_TO_S, "time")  # raises, naming them all
        if not matches:
            return self._from_parameter()
        axis = matches[0]
        return self._from_axis(), f"axis {axis.axis_id!r} ({axis.units})"

    def offsets(self) -> np.ndarray:
        """
        Source-to-receiver offsets, in metres, ready for :func:`fwap.stc`.

        Taken from the single axis whose declared unit is a length unit,
        one value per receiver. Note that these are the offsets the tool
        recorded, which is not the same as counting receiver spacings
        from an assumed first offset: slowness estimates depend only on
        the spacing, but arrival *times* depend on the absolute value.

        Returns
        -------
        ndarray, shape (n_receiver,)

        Raises
        ------
        ValueError
            If the channel does not have exactly one length-unit axis,
            or that axis cannot produce one coordinate per receiver.
        """
        axis, scale = self._sole_axis(_LENGTH_UNIT_TO_M, "length")
        return axis.values() * scale


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
        waveform_channels: dict[str, tuple[int, ...]] = {}
        for ch in frame.channels:
            if ch.name == depth_name:
                continue
            arr = np.asarray(rec[ch.name])
            # Skip vector channels -- the LAS-style container expects
            # one sample per depth -- but record their shapes, so a
            # caller can see what is in the file and hand the name to
            # ``read_dlis_waveforms``.
            if arr.ndim != 1:
                waveform_channels[ch.name] = tuple(int(n) for n in arr.shape[1:])
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
        waveform_channels=waveform_channels,
    )


def _axis_units(axis: Any, attribute: str) -> str:
    """Declared unit of one AXIS attribute, or empty string if absent."""
    try:
        attr = axis.attic[attribute]
    except (KeyError, TypeError, AttributeError):  # pragma: no cover - dlisio shape
        return ""
    return str(getattr(attr, "units", "") or "")


def _read_axes(channel: Any, dimensions: tuple[int, ...]) -> list[DlisAxis]:
    """
    Build one :class:`DlisAxis` per trailing dimension of a channel.

    A file may declare fewer AXIS objects than the channel has
    dimensions, or none at all. Rather than guess which dimension an
    incomplete list describes, this returns an empty list unless the
    counts agree -- ``DlisWaveforms.axes`` is then empty and the
    derived accessors say so rather than reporting a number from the
    wrong axis.
    """
    axes = list(getattr(channel, "axis", []) or [])
    if len(axes) != len(dimensions):
        return []
    out: list[DlisAxis] = []
    for axis, size in zip(axes, dimensions):
        coords = axis.coordinates
        spacing = axis.spacing
        # SPACING and COORDINATES carry the same unit in every file seen;
        # prefer SPACING's and fall back, since either may be absent.
        units = _axis_units(axis, "SPACING") or _axis_units(axis, "COORDINATES")
        out.append(
            DlisAxis(
                axis_id=str(axis.axis_id) if axis.axis_id else "",
                coordinates=np.asarray(
                    list(coords) if coords is not None else [], dtype=float
                ),
                spacing=float(spacing) if spacing is not None else float("nan"),
                units=units,
                size=int(size),
            )
        )
    return out


def _sample_interval_parameters(logical_file: Any) -> dict[str, float]:
    """
    Numeric ``DSI*`` PARAMETER records, the sample-interval fallback source.

    Read whether or not they are needed, so that
    :meth:`DlisWaveforms.sample_interval` can explain itself either way and
    a caller can inspect what the file offered. Non-numeric records are
    skipped: files carry companion entries like ``DSIN='DS10'`` that name a
    mode rather than a value.
    """
    out: dict[str, float] = {}
    for param in getattr(logical_file, "parameters", []) or []:
        name = str(param.name)
        if not name.upper().startswith(_SAMPLE_INTERVAL_PARAM_PREFIX):
            continue
        values = list(param.values) if param.values is not None else []
        if len(values) != 1:
            continue
        try:
            out[name] = float(values[0])
        except (TypeError, ValueError):
            continue  # a mode string such as 'DS10', not an interval
    return out


def read_dlis_waveforms(
    path: str,
    channel: str,
    *,
    logical_file_index: int = 0,
    frame_index: int = 0,
) -> DlisWaveforms:
    """
    Read one multi-dimensional DLIS channel -- an array-sonic waveform set.

    :func:`read_dlis` returns the scalar curves of a frame and skips
    channels with more than one value per depth, because its
    :class:`DlisCurves` container mirrors :class:`LasCurves`. Those
    skipped channels are where a full-waveform sonic record actually
    lives, and this reads one of them. The names available in a file are
    reported by :attr:`DlisCurves.waveform_channels`.

    Only the requested channel and the frame's index channel are read,
    so this does not pay for the rest of the frame. That matters: a
    single monopole waveform channel of a logged interval runs to
    hundreds of megabytes.

    The returned object carries the acquisition geometry the file
    declares -- see :meth:`DlisWaveforms.sample_interval` and
    :meth:`DlisWaveforms.offsets`, which are what :func:`fwap.stc`
    needs.

    Parameters
    ----------
    path : str
        Filesystem path to a DLIS file.
    channel : str
        Name of the channel to read, e.g. ``"PWF4"``.
    logical_file_index : int, default 0
        Which logical file to read.
    frame_index : int, default 0
        Which frame inside that logical file to read.

    Returns
    -------
    DlisWaveforms

    Raises
    ------
    IndexError
        If ``logical_file_index`` or ``frame_index`` is out of range.
    KeyError
        If the frame has no channel of that name.
    ValueError
        If the named channel is scalar -- one value per depth -- and so
        belongs to :func:`read_dlis` rather than here.

    Examples
    --------
    Process a real array-sonic record without assuming its geometry::

        curves = read_dlis(path)
        curves.waveform_channels        # {'PWF1': (8, 512), ...}

        wf = read_dlis_waveforms(path, "PWF4")
        surface = stc(wf.data[depth_index], wf.sample_interval(), wf.offsets())
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

        by_name = {ch.name: ch for ch in frame.channels}
        if channel not in by_name:
            raise KeyError(
                f"frame {frame.name!r} has no channel {channel!r}; "
                f"it has {sorted(by_name)}"
            )
        if not frame.channels:  # pragma: no cover - guarded by the lookup above
            raise ValueError(f"frame {frame.name!r} has no channels")

        target = by_name[channel]
        index_channel = frame.channels[0]

        data = np.asarray(target.curves())
        if data.ndim < 2:
            raise ValueError(
                f"channel {channel!r} has one value per depth, so it is a curve "
                "rather than a waveform set; read it with read_dlis()"
            )
        depth = np.asarray(index_channel.curves(), dtype=float)
        depth_unit = str(index_channel.units) if index_channel.units else ""
        axes = _read_axes(target, tuple(int(n) for n in data.shape[1:]))
        units = str(target.units) if target.units else ""
        frame_name = str(frame.name)
        interval_params = _sample_interval_parameters(lf)

    return DlisWaveforms(
        channel=channel,
        data=data,
        depth=depth,
        depth_unit=depth_unit,
        axes=axes,
        units=units,
        frame_name=frame_name,
        sample_interval_parameters=interval_params,
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
            f.write(path, output_chunk_size=_DLIS_OUTPUT_CHUNK_BYTES)
    finally:
        validator_logger.setLevel(prev_level)
