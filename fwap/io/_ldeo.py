"""
Reader for the LDEO scientific-drilling sonic-waveform binary format.

Why a fourth format, when the package already reads DLIS
--------------------------------------------------------
The Lamont-Doherty Borehole Research Group distributes every scientific
ocean-drilling sonic run twice: as the original service-company DLIS, and as a
plain binary export produced by loading that DLIS into GeoFrame and dumping the
waveform arrays. The export is roughly a fifth the size of the DLIS it came
from, needs no DLIS library to read, and -- unlike a vendor's DLIS -- carries a
short self-describing header that states the array size, the sample interval
and the tool outright. For a test fixture that is the better of the two files.

Layout, which the archive documents in its own ``*_info-swf*.html`` page
-----------------------------------------------------------------------
Fixed-length records of ``4 * (1 + n_receiver * n_sample)`` bytes. Record 0 is
the header; record ``k + 1`` holds depth ``k`` followed by that depth's whole
gather, receiver-major. Everything is **big-endian** -- the conversion ran on
Sun hardware -- which is the one thing a reader can get wrong and still produce
plausible-looking numbers, so :func:`read_ldeo_waveforms` checks rather than
assumes (see below).

Header fields, in order: ``n_depth``, ``n_sample``, ``n_receiver``, ``tool``,
``mode`` as 4-byte integers, then ``depth_increment``, ``depth_scale`` and
``sample_interval`` as 4-byte IEEE floats.

What this format does *not* record
----------------------------------
**Transmitter-to-receiver offsets.** They are in neither the header nor the
DLIS the export came from, so :class:`LdeoWaveforms` does not invent them: it
reports ``n_receiver`` and leaves building an
:class:`~fwap.synthetic.ArrayGeometry` to the caller, who must get the spacing
from the tool specification. That is not merely a formality -- ``plans/roadmap.md``
F.5 records how it was settled for one archive, by taking the offsets from the
tool spec and then testing them against first-break moveout across the array.

References
----------
* Lamont-Doherty Earth Observatory, Borehole Research Group. *Sonic Waveform
  Data* file descriptions, published per hole alongside the data.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

#: Header ``tool`` code -> tool name, as the archive's file descriptions define
#: them. Codes outside this table are kept as-is rather than rejected: a newer
#: tool is not a corrupt file.
LDEO_TOOL_NAMES: dict[int, str] = {
    0: "DSI",
    1: "SonicVISION",
    2: "SonicScope",
    3: "Sonic Scanner",
    4: "XBAT",
    5: "MCS",
    6: "SDT",
    7: "LSS",
    8: "SST",
    9: "BHC",
    10: "QL40",
    11: "2PSA",
}

#: Header ``mode`` code -> firing mode.
LDEO_MODE_NAMES: dict[int, str] = {
    1: "Lower Dipole",
    2: "Upper Dipole",
    3: "Stoneley",
    4: "Monopole",
}

#: Byte order of every numeric field. Not configurable on purpose -- see
#: :func:`read_ldeo_waveforms` for why a wrong guess must fail loudly.
_ENDIAN: str = ">"

#: Sample intervals (s) a borehole sonic record can plausibly have. Used to
#: reject a header parsed under the wrong byte order.
_PLAUSIBLE_SAMPLE_INTERVAL_S: tuple[float, float] = (1.0e-7, 1.0e-3)


@dataclass(frozen=True)
class LdeoWaveforms:
    """
    One LDEO sonic-waveform export: a depth-ordered stack of gathers.

    Attributes
    ----------
    data : ndarray, shape (n_depth, n_receiver, n_sample)
        Waveform amplitudes in the archive's own arbitrary units. Empty along
        axis 0 when read with ``max_depths=0``.
    depth : ndarray, shape (n_depth,)
        Depth of each gather, **in metres**, with the header's ``depth_scale``
        already applied. Empty when read with ``max_depths=0``.
    sample_interval : float
        Time between samples (s).
    depth_increment : float
        Nominal distance between successive gathers (m), scaled like ``depth``.
    n_depth : int
        Depths in the *file*, which exceeds ``len(depth)`` when ``max_depths``
        truncated the read. Kept so a truncated read still reports the whole.
    n_receiver, n_sample : int
        Array size and record length.
    tool_code, mode_code : int
        Raw header codes; see :attr:`tool` and :attr:`mode` for names.
    depth_scale : float
        The header's depth scaling: ``1.0`` when depths are stored in metres,
        ``0.3048`` when in feet.
    path : str
        File the data was read from.

    Notes
    -----
    Transmitter-to-receiver offsets are **not** part of this format and are not
    guessed here; see the module docstring.
    """

    data: np.ndarray
    depth: np.ndarray
    sample_interval: float
    depth_increment: float
    n_depth: int
    n_receiver: int
    n_sample: int
    tool_code: int
    mode_code: int
    depth_scale: float
    path: str

    @property
    def tool(self) -> str:
        """Tool name for :attr:`tool_code`, or ``"unknown (<code>)"``."""
        return LDEO_TOOL_NAMES.get(self.tool_code, f"unknown ({self.tool_code})")

    @property
    def mode(self) -> str:
        """Firing mode for :attr:`mode_code`, or ``"unknown (<code>)"``."""
        return LDEO_MODE_NAMES.get(self.mode_code, f"unknown ({self.mode_code})")

    @property
    def record_length(self) -> float:
        """Duration of one waveform (s): ``n_sample * sample_interval``."""
        return self.n_sample * self.sample_interval

    def gather(self, index: int) -> np.ndarray:
        """
        One depth's gather, shape ``(n_receiver, n_sample)``.

        Parameters
        ----------
        index : int
            Index into :attr:`depth`, not a depth in metres.

        Returns
        -------
        ndarray, shape (n_receiver, n_sample)
        """
        return self.data[index]


def _read_header(
    handle: BinaryIO, size: int, path: Path
) -> tuple[int, int, int, int, int, float, float, float]:
    """Parse and sanity-check record 0. See :func:`read_ldeo_waveforms`."""
    raw = handle.read(32)
    if len(raw) < 32:
        raise ValueError(f"{path} is too short to hold an LDEO waveform header")
    n_depth, n_sample, n_receiver, tool, mode = struct.unpack(f"{_ENDIAN}5i", raw[:20])
    dz, scale, dt_us = struct.unpack(f"{_ENDIAN}3f", raw[20:32])

    if min(n_depth, n_sample, n_receiver) <= 0:
        raise ValueError(
            f"{path}: header gives n_depth={n_depth}, n_sample={n_sample}, "
            f"n_receiver={n_receiver}; all must be positive. This format is "
            "big-endian, and a little-endian file would fail here."
        )

    record = 4 * (1 + n_receiver * n_sample)
    expected = record * (1 + n_depth)
    if expected != size:
        raise ValueError(
            f"{path}: header implies {expected} bytes "
            f"({record}-byte records x {1 + n_depth}) but the file is {size}. "
            "Either it is truncated or it is not an LDEO waveform export."
        )

    lo, hi = _PLAUSIBLE_SAMPLE_INTERVAL_S
    if not lo <= dt_us * 1e-6 <= hi:
        raise ValueError(
            f"{path}: header sample interval {dt_us} us is outside the "
            f"plausible range {lo * 1e6:g}-{hi * 1e6:g} us for a sonic record"
        )
    if scale <= 0.0:
        raise ValueError(f"{path}: header depth scale {scale} is not positive")

    return n_depth, n_sample, n_receiver, tool, mode, dz, scale, dt_us


def read_ldeo_waveforms(
    path: str | Path, *, max_depths: int | None = None
) -> LdeoWaveforms:
    """
    Read an LDEO sonic-waveform export.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``*.bin`` file from a scientific-drilling logging archive's
        "Sonic Waveform Data" directory.
    max_depths : int or None
        Read at most this many depths from the top of the file. ``None`` reads
        all of them; ``0`` reads the header only, which is how to inspect a
        150 MB file cheaply. :attr:`LdeoWaveforms.n_depth` always reports the
        file's full count regardless.

    Returns
    -------
    LdeoWaveforms

    Raises
    ------
    ValueError
        If the header is not self-consistent with the file size, or its sample
        interval is not sonic-plausible. Both checks exist for one reason: this
        format is big-endian, and read little-endian its header decodes to
        enormous garbage rather than to nothing. A reader that trusted the
        header would then allocate wildly or return silently wrong data, so the
        arithmetic is verified before a single sample is read.

    Examples
    --------
    >>> wf = read_ldeo_waveforms("324-U1347A_mono_p1.bin")  # doctest: +SKIP
    >>> wf.tool, wf.mode, wf.n_receiver, wf.sample_interval  # doctest: +SKIP
    ('DSI', 'Monopole', 8, 1e-05)
    """
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as handle:
        header = _read_header(handle, size, path)
        n_depth, n_sample, n_receiver, tool, mode, dz, scale, dt_us = header

        count = n_depth if max_depths is None else min(max_depths, n_depth)
        if count < 0:
            raise ValueError(f"max_depths must be non-negative, got {max_depths}")

        record = 4 * (1 + n_receiver * n_sample)
        depth = np.empty(count, dtype=float)
        data = np.empty((count, n_receiver, n_sample), dtype=float)
        for k in range(count):
            handle.seek(record * (k + 1))
            row = np.frombuffer(handle.read(record), dtype=f"{_ENDIAN}f4")
            depth[k] = float(row[0]) * scale
            data[k] = row[1:].reshape(n_receiver, n_sample)

    return LdeoWaveforms(
        data=data,
        depth=depth,
        sample_interval=float(dt_us) * 1e-6,
        depth_increment=float(dz) * float(scale),
        n_depth=int(n_depth),
        n_receiver=int(n_receiver),
        n_sample=int(n_sample),
        tool_code=int(tool),
        mode_code=int(mode),
        depth_scale=float(scale),
        path=str(path),
    )
