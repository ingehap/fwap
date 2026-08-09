"""LAS / DLIS / SEG-Y I/O round-trip tests.

``lasio``, ``dlisio``, ``dliswriter``, and ``segyio`` are all core
fwap dependencies, so every suite below runs unconditionally.
"""

from __future__ import annotations

import lasio
import numpy as np
import pytest

from fwap.io import LasCurves, read_las, write_las


def _synthetic_curves(n=200):
    """A small Vp/Vs/Stoneley log set with a few NaNs thrown in."""
    depth = np.linspace(1000.0, 1100.0, n)
    dtp = 70.0 + 5.0 * np.sin(depth / 10.0)
    dts = 140.0 + 10.0 * np.sin(depth / 10.0)
    dts[0] = np.nan  # null at the top
    dts[-1] = np.nan  # null at the bottom
    cohp = 0.85 + 0.05 * np.cos(depth / 5.0)
    return depth, {"DTP": dtp, "DTS": dts, "COHP": cohp}


def test_read_write_round_trip(tmp_path):
    """Writing then reading a LAS file reproduces the curves."""
    depth, curves = _synthetic_curves()
    path = tmp_path / "out.las"
    write_las(
        str(path), depth, curves, well_name="FWAP_TEST", well={"COMP": "fwap"}
    )  # LAS uses COMP (not COMPANY)
    assert path.exists()

    loaded = read_las(str(path))
    assert isinstance(loaded, LasCurves)
    assert loaded.well["WELL"] == "FWAP_TEST"
    assert loaded.well["COMP"] == "fwap"

    # Depth axis preserved (LAS writes in fixed-decimal so tolerate rounding).
    assert np.allclose(loaded.depth, depth, atol=1.0e-6)

    # Every written curve reappears with matching values, NaNs preserved.
    for name, arr in curves.items():
        assert name in loaded.curves
        out = loaded.curves[name]
        mask = ~np.isnan(arr)
        assert np.allclose(out[mask], arr[mask], atol=1.0e-6)
        # NaN positions round-trip as NaN.
        assert np.isnan(out[np.isnan(arr)]).all()


def test_write_rejects_shape_mismatch(tmp_path):
    """A curve whose length doesn't match ``depth`` raises ValueError."""
    depth = np.linspace(0.0, 10.0, 50)
    bad = {"DTP": np.zeros(49)}
    with pytest.raises(ValueError, match="shape"):
        write_las(str(tmp_path / "bad.las"), depth, bad)


def test_write_fills_fwap_curve_units(tmp_path):
    """fwap's standard mnemonics get their canonical units applied."""
    depth = np.linspace(0.0, 10.0, 5)
    curves = {
        "DTP": np.full(5, 70.0),
        "DTS": np.full(5, 140.0),
        "COHP": np.full(5, 0.9),
        "VPVS": np.full(5, 1.7),
    }
    path = tmp_path / "units.las"
    write_las(str(path), depth, curves)
    las = lasio.read(str(path))
    unit_by_mnemonic = {c.mnemonic: c.unit for c in las.curves}
    assert unit_by_mnemonic["DTP"] == "us/ft"
    assert unit_by_mnemonic["DTS"] == "us/ft"
    # Dimensionless curves write as empty unit strings.
    assert unit_by_mnemonic["COHP"] == ""
    assert unit_by_mnemonic["VPVS"] == ""


def test_write_custom_units_override(tmp_path):
    """Passing ``units=`` overrides the internal table for custom mnemonics."""
    depth = np.linspace(0.0, 10.0, 5)
    curves = {"CUSTOM": np.full(5, 1.5)}
    path = tmp_path / "custom.las"
    write_las(str(path), depth, curves, units={"CUSTOM": "m/s"})
    las = lasio.read(str(path))
    unit = {c.mnemonic: c.unit for c in las.curves}["CUSTOM"]
    assert unit == "m/s"


def test_read_las_exposes_units_dict(tmp_path):
    """read_las returns a units dict keyed by mnemonic."""
    depth, curves = _synthetic_curves(n=30)
    path = tmp_path / "units.las"
    write_las(str(path), depth, curves)
    loaded = read_las(str(path))
    assert loaded.units["DTP"] == "us/ft"
    assert loaded.units["DTS"] == "us/ft"
    assert loaded.units["COHP"] == ""


def test_read_las_exposes_step(tmp_path):
    """Uniformly-spaced depth axis is reported via LasCurves.step."""
    depth, curves = _synthetic_curves(n=50)
    path = tmp_path / "step.las"
    write_las(str(path), depth, curves)
    loaded = read_las(str(path))
    expected_step = depth[1] - depth[0]
    # STEP is written as a text field with a few decimals, so we only
    # need a fractional match here.
    assert abs(loaded.step - expected_step) / expected_step < 1.0e-5


# ---------------------------------------------------------------------
# DLIS reader / writer
# ---------------------------------------------------------------------

from fwap.io import (
    DlisAxis,
    DlisCurves,
    DlisWaveforms,
    read_dlis,
    read_dlis_waveforms,
    write_dlis,
)


def test_read_write_dlis_round_trip(tmp_path):
    """write_dlis + read_dlis reproduces the curves and well metadata."""
    depth, curves = _synthetic_curves()
    path = tmp_path / "out.dlis"
    write_dlis(
        str(path),
        depth,
        curves,
        well_name="FWAP_TEST",
        well={"COMP": "fwap", "FLD": "TestField"},
    )
    assert path.exists()

    loaded = read_dlis(str(path))
    assert isinstance(loaded, DlisCurves)
    assert loaded.frame_name == "MAIN"
    assert loaded.index_type == "BOREHOLE-DEPTH"
    assert loaded.well["WELL"] == "FWAP_TEST"
    assert loaded.well["COMP"] == "fwap"
    assert loaded.well["FLD"] == "TestField"

    # Depth axis preserved (DLIS uses IEEE float so the round-trip is
    # exact for the float64 values we wrote).
    assert np.allclose(loaded.depth, depth, atol=0.0)

    # Every written curve reappears with matching values; NaN
    # round-trips natively as the IEEE-754 NaN bit pattern.
    for name, arr in curves.items():
        assert name in loaded.curves
        out = loaded.curves[name]
        mask = ~np.isnan(arr)
        assert np.allclose(out[mask], arr[mask], atol=0.0)
        assert np.isnan(out[np.isnan(arr)]).all()


def test_write_dlis_rejects_shape_mismatch(tmp_path):
    """A curve whose length doesn't match ``depth`` raises ValueError."""
    depth = np.linspace(0.0, 10.0, 50)
    bad = {"DTP": np.zeros(49)}
    with pytest.raises(ValueError, match="shape"):
        write_dlis(str(tmp_path / "bad.dlis"), depth, bad)


def test_read_dlis_exposes_units_and_step(tmp_path):
    """Per-channel units and frame spacing come back from read_dlis."""
    depth, curves = _synthetic_curves(n=40)
    path = tmp_path / "units.dlis"
    write_dlis(str(path), depth, curves)
    loaded = read_dlis(str(path))
    assert loaded.units["DTP"] == "us/ft"
    assert loaded.units["DTS"] == "us/ft"
    expected_step = depth[1] - depth[0]
    assert abs(loaded.step - expected_step) / expected_step < 1.0e-10


def test_write_dlis_custom_units_override(tmp_path):
    """Passing ``units=`` overrides the internal table for custom mnemonics."""
    depth = np.linspace(0.0, 10.0, 5)
    curves = {"CUSTOM": np.full(5, 1.5)}
    path = tmp_path / "custom.dlis"
    write_dlis(str(path), depth, curves, units={"CUSTOM": "m/s"})
    loaded = read_dlis(str(path))
    assert loaded.units["CUSTOM"] == "m/s"


def test_read_dlis_out_of_range_indices_raise(tmp_path):
    """Asking for a non-existent logical file or frame raises IndexError."""
    depth = np.linspace(0.0, 10.0, 5)
    path = tmp_path / "small.dlis"
    write_dlis(str(path), depth, {"DTP": np.full(5, 70.0)})
    with pytest.raises(IndexError, match="logical_file_index"):
        read_dlis(str(path), logical_file_index=5)
    with pytest.raises(IndexError, match="frame_index"):
        read_dlis(str(path), frame_index=5)


def test_write_dlis_custom_frame_and_index_type(tmp_path):
    """frame_name and index_type are preserved through the round-trip."""
    depth = np.linspace(0.0, 10.0, 5)
    path = tmp_path / "time.dlis"
    write_dlis(
        str(path),
        depth,
        {"X": np.zeros(5)},
        frame_name="WAVEFORM",
        index_type="TIME",
        depth_unit="s",
    )
    loaded = read_dlis(str(path))
    assert loaded.frame_name == "WAVEFORM"
    assert loaded.index_type == "TIME"


# ---------------------------------------------------------------------
# SEG-Y reader
# ---------------------------------------------------------------------

import segyio


def _synth_segy(path, n_traces=8, n_samples=512, dt_us=100, offsets=None):
    """Create a small SEG-Y file with an ascending sinusoid per trace."""
    if offsets is None:
        offsets = np.arange(n_traces) * 500  # segyio stores as int
    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = np.arange(n_samples)
    spec.tracecount = n_traces
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING

    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_us)
        for i in range(n_traces):
            f.header[i].update(
                {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.offset: int(offsets[i]),
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(dt_us),
                }
            )
            t = np.arange(n_samples, dtype=np.float32)
            f.trace[i] = np.sin(2 * np.pi * 0.02 * (t - i * 5.0)).astype(np.float32)


def test_read_segy_basic_round_trip(tmp_path):
    """read_segy returns the same shape and dt we wrote."""
    from fwap.io import SegyGather, read_segy

    path = tmp_path / "simple.sgy"
    _synth_segy(path, n_traces=8, n_samples=512, dt_us=100)

    g = read_segy(str(path))
    assert isinstance(g, SegyGather)
    assert g.data.shape == (8, 512)
    assert g.n_traces == 8
    assert g.n_samples == 512
    assert abs(g.dt - 100.0e-6) < 1e-12
    assert g.offsets is not None
    assert g.offsets.shape == (8,)
    assert np.allclose(g.offsets, np.arange(8) * 500)


def test_read_segy_zero_offset_header_returns_none(tmp_path):
    """If the offset header is all zeros, offsets comes back as None."""
    from fwap.io import read_segy

    path = tmp_path / "no_offsets.sgy"
    _synth_segy(
        path, n_traces=4, n_samples=128, dt_us=50, offsets=np.zeros(4, dtype=int)
    )
    g = read_segy(str(path))
    assert g.offsets is None


def test_read_segy_alternate_offset_header(tmp_path):
    """offset_header= can point at a different TraceField."""
    from fwap.io import read_segy

    path = tmp_path / "srcx.sgy"
    # Write offsets into SourceX instead of offset.
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(128)
    spec.tracecount = 4
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = 100
        for i in range(4):
            f.header[i].update(
                {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.SourceX: 1000 + i * 250,
                    segyio.TraceField.TRACE_SAMPLE_COUNT: 128,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: 100,
                }
            )
            f.trace[i] = np.zeros(128, dtype=np.float32)

    g = read_segy(str(path), offset_header="SourceX")
    assert g.offsets is not None
    assert np.allclose(g.offsets, 1000 + np.arange(4) * 250)


def test_read_segy_rejects_bad_offset_header(tmp_path):
    """An unknown offset_header raises AttributeError."""
    from fwap.io import read_segy

    path = tmp_path / "simple.sgy"
    _synth_segy(path, n_traces=4, n_samples=128, dt_us=100)
    with pytest.raises(AttributeError):
        read_segy(str(path), offset_header="not_a_field")


def test_segy_round_trip_via_fwap_write(tmp_path):
    """write_segy + read_segy reproduces the gather and dt."""
    from fwap.io import read_segy, write_segy

    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 128)).astype(np.float32)
    dt = 2.5e-4
    offsets = np.linspace(3.0, 4.0, 6) * 1000  # in integer "metres * 1000"
    path = tmp_path / "rt.sgy"
    write_segy(str(path), data, dt=dt, offsets=offsets)

    g = read_segy(str(path))
    assert g.data.shape == data.shape
    assert abs(g.dt - dt) < 1.0e-9
    # IEEE-float round-trip is exact for float32 input.
    assert np.allclose(g.data, data, atol=0.0)
    # Offsets round to int on the way out.
    assert np.allclose(g.offsets, np.round(offsets).astype(int))


def test_write_segy_rejects_non_2d_data(tmp_path):
    """Data that is not 2-D raises ValueError."""
    from fwap.io import write_segy

    with pytest.raises(ValueError, match="2-D"):
        write_segy(
            str(tmp_path / "bad.sgy"), np.zeros(100, dtype=np.float32), dt=1.0e-4
        )


def test_write_segy_rejects_bad_offsets_length(tmp_path):
    """Offsets with the wrong length raise ValueError."""
    from fwap.io import write_segy

    with pytest.raises(ValueError, match="shape"):
        write_segy(
            str(tmp_path / "bad.sgy"),
            np.zeros((4, 32), dtype=np.float32),
            dt=1.0e-4,
            offsets=np.array([1, 2, 3]),
        )


def test_write_segy_rejects_non_positive_dt(tmp_path):
    """dt <= 0 raises ValueError."""
    from fwap.io import write_segy

    with pytest.raises(ValueError, match="dt must be positive"):
        write_segy(
            str(tmp_path / "bad.sgy"), np.zeros((2, 16), dtype=np.float32), dt=0.0
        )


# ---------------------------------------------------------------------
# DLIS waveform channels (array-sonic records)
#
# `read_dlis` returns one value per depth and skips everything else, so
# for most of this package's life the per-receiver waveforms in a real
# sonic DLIS -- the only thing the processing chain actually consumes --
# were unreachable from the public API. `read_dlis_waveforms` reads them.
#
# Two kinds of test below, because `dliswriter` cannot produce the shape
# the real files use. It caps datasets at two dimensions and overrides a
# declared `dimension` from the data, so a written channel is always
# `(n_depth, n)` with a single axis. The round-trip tests use that and
# are honest end-to-end; the `(n_depth, n_receiver, n_sample)` case is
# covered against a stub whose shape the round-trip tests pin.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def waveform_dlis(tmp_path_factory):
    """A DLIS carrying one vector channel with a declared time axis.

    Module-scoped because writing is the expensive half of these tests
    and every one of them wants the same file.
    """
    import dliswriter

    from fwap.io._dlis import _DLIS_OUTPUT_CHUNK_BYTES, _suppress_fd

    n_depth, n_sample, dt_us = 6, 32, 10.0
    f = dliswriter.DLISFile()
    lf = f.add_logical_file()
    lf.add_origin("FWAP")
    axis = lf.add_axis(
        "TAX",
        axis_id="MICRO_TIME",
        # dliswriter requires one coordinate per sample; the regular
        # start+spacing form real files use is exercised separately
        # against DlisAxis directly.
        coordinates={"value": [dt_us * i for i in range(n_sample)], "units": "us"},
        spacing={"value": dt_us, "units": "us"},
    )
    depth = np.arange(n_depth, dtype=float) * 0.1524
    data = np.arange(n_depth * n_sample, dtype=np.float32).reshape(n_depth, n_sample)
    channels = (
        lf.add_channel("DEPT", data=depth, units="m"),
        lf.add_channel("WFRM", data=data, axis=axis, units=""),
        lf.add_channel("DTCO", data=np.linspace(50.0, 60.0, n_depth), units="us/ft"),
    )
    lf.add_frame("MAIN", channels=channels, index_type="BOREHOLE-DEPTH")

    path = tmp_path_factory.mktemp("dlis_wf") / "wf.dlis"
    with _suppress_fd(2):
        f.write(str(path), output_chunk_size=_DLIS_OUTPUT_CHUNK_BYTES)
    return str(path), depth, data


def test_read_dlis_reports_the_waveform_channels_it_skips(waveform_dlis):
    """The skipped channels are at least visible, with their shapes.

    Before this, a caller reading a real sonic DLIS got a `DlisCurves`
    with no indication that the waveforms existed at all.
    """
    path, _, _ = waveform_dlis

    loaded = read_dlis(path)
    assert "WFRM" not in loaded.curves, "vector channels stay out of curves"
    assert "DTCO" in loaded.curves, "scalar channels are unaffected"
    assert loaded.waveform_channels == {"WFRM": (32,)}


def test_read_dlis_waveforms_round_trip(waveform_dlis):
    """The samples come back exactly, depth-major, with the frame metadata."""
    path, depth, data = waveform_dlis

    wf = read_dlis_waveforms(path, "WFRM")
    assert isinstance(wf, DlisWaveforms)
    assert wf.channel == "WFRM"
    assert wf.frame_name == "MAIN"
    assert wf.data.shape == (6, 32)
    assert np.allclose(wf.data, data, atol=0.0)
    assert np.allclose(wf.depth, depth, atol=0.0)
    assert wf.depth_unit == "m"


def test_read_dlis_waveforms_recovers_the_sample_interval_from_the_file(waveform_dlis):
    """10 us declared on the axis comes back as 1e-5 s, not as 10.

    The unit conversion is the point: `fwap.stc` takes seconds, and the
    file says microseconds. Getting this from the file rather than from
    a hard-coded constant is what the whole reader exists for.
    """
    path, _, _ = waveform_dlis

    wf = read_dlis_waveforms(path, "WFRM")
    assert len(wf.axes) == 1
    assert wf.axes[0].axis_id == "MICRO_TIME"
    assert wf.axes[0].units == "us"
    assert wf.sample_interval() == pytest.approx(1.0e-5, rel=1e-12)


def test_read_dlis_waveforms_rejects_a_scalar_channel(waveform_dlis):
    """A curve is not a waveform set, and the error says where to go."""
    path, _, _ = waveform_dlis

    with pytest.raises(ValueError, match="read_dlis"):
        read_dlis_waveforms(path, "DTCO")


def test_read_dlis_waveforms_unknown_channel_lists_what_is_there(waveform_dlis):
    """A typo should not require opening the file by hand to diagnose."""
    path, _, _ = waveform_dlis

    with pytest.raises(KeyError, match="WFRM"):
        read_dlis_waveforms(path, "PWF4")


def test_read_dlis_waveforms_out_of_range_indices_raise(waveform_dlis):
    path, _, _ = waveform_dlis

    with pytest.raises(IndexError, match="logical_file_index"):
        read_dlis_waveforms(path, "WFRM", logical_file_index=3)
    with pytest.raises(IndexError, match="frame_index"):
        read_dlis_waveforms(path, "WFRM", frame_index=3)


# --- DlisAxis: the two coordinate conventions, and the unit selection ---
#
# Real files use the regular form (one starting coordinate plus SPACING);
# `dliswriter` can only write the exhaustive form (every coordinate), so
# the round-trip tests above cover one and these cover both.


def test_dlis_axis_uses_coordinates_when_they_span_the_dimension():
    axis = DlisAxis(
        axis_id="MICRO_TIME",
        coordinates=np.array([0.0, 7.0, 30.0, 31.0]),
        spacing=float("nan"),
        units="us",
        size=4,
    )
    # Irregular on purpose: the listed coordinates win, they are not
    # re-derived from a spacing that does not exist.
    assert np.allclose(axis.values(), [0.0, 7.0, 30.0, 31.0])


def test_dlis_axis_expands_a_regular_axis_from_start_and_spacing():
    """The form the Schlumberger file uses: one coordinate, plus SPACING."""
    axis = DlisAxis(
        axis_id="SENSOR_OFFSET",
        coordinates=np.array([7.874]),
        spacing=0.1524,
        units="m",
        size=8,
    )
    values = axis.values()
    assert values.shape == (8,)
    assert values[0] == pytest.approx(7.874)
    assert values[-1] == pytest.approx(7.874 + 7 * 0.1524)


def test_dlis_axis_refuses_to_invent_coordinates():
    axis = DlisAxis(
        axis_id="ORDINAL",
        coordinates=np.array([]),
        spacing=float("nan"),
        units="",
        size=4,
    )
    with pytest.raises(ValueError, match="cannot be recovered"):
        axis.values()


def _waveforms(axes, *, shape=(3, 8, 512)):
    return DlisWaveforms(
        channel="PWF4",
        data=np.zeros(shape),
        depth=np.arange(shape[0], dtype=float),
        depth_unit="0.1 in",
        axes=axes,
        units="",
        frame_name="60B",
    )


def test_waveform_geometry_is_selected_by_unit_not_by_axis_name():
    """AXIS-ID strings are producer-defined; units are not.

    These axes are named nonsensically on purpose. A reader keying off
    "MICRO_TIME" / "SENSOR_OFFSET" would fail here; one keying off the
    declared unit does not.
    """
    wf = _waveforms(
        [
            DlisAxis("CHANNEL_B", np.array([2.7432]), 0.1524, "m", 8),
            DlisAxis("CHANNEL_A", np.array([0.0]), 40.0, "us", 512),
        ]
    )
    assert wf.sample_interval() == pytest.approx(4.0e-5)
    assert np.allclose(wf.offsets(), 2.7432 + 0.1524 * np.arange(8))


def test_waveform_geometry_converts_from_whatever_unit_the_file_declares():
    """Feet and milliseconds come back as metres and seconds."""
    wf = _waveforms(
        [
            DlisAxis("OFF", np.array([9.0]), 0.5, "ft", 8),
            DlisAxis("T", np.array([0.0]), 0.01, "ms", 512),
        ]
    )
    assert wf.sample_interval() == pytest.approx(1.0e-5)
    assert wf.offsets()[0] == pytest.approx(9.0 * 0.3048)
    assert wf.offsets()[1] - wf.offsets()[0] == pytest.approx(0.5 * 0.3048)


def test_waveform_geometry_ignores_a_unitless_axis():
    """A 3-axis channel with an unitless ORDINAL axis still resolves."""
    wf = _waveforms(
        [
            DlisAxis("ORDINAL", np.array([1.0]), 1.0, "", 4),
            DlisAxis("SENSOR_OFFSET", np.array([7.874]), 0.1524, "m", 8),
            DlisAxis("MICRO_TIME", np.array([0.0]), 40.0, "us", 512),
        ],
        shape=(3, 4, 8, 512),
    )
    assert wf.sample_interval() == pytest.approx(4.0e-5)
    assert wf.offsets().shape == (8,)


@pytest.mark.parametrize(
    "axes, kind",
    [
        ([], "time"),
        (
            [
                DlisAxis("A", np.array([0.0]), 10.0, "us", 8),
                DlisAxis("B", np.array([0.0]), 40.0, "ms", 512),
            ],
            "time",
        ),
    ],
)
def test_waveform_sample_interval_refuses_when_ambiguous(axes, kind):
    """Zero or two time axes is not a number this can report."""
    with pytest.raises(ValueError, match=f"{kind} unit"):
        _waveforms(axes).sample_interval()


def test_waveform_offsets_refuse_when_no_length_axis_exists():
    wf = _waveforms([DlisAxis("T", np.array([0.0]), 10.0, "us", 512)])
    with pytest.raises(ValueError, match="length unit"):
        wf.offsets()


def test_waveform_sample_interval_falls_back_to_the_coordinate_step():
    """An axis with coordinates but no SPACING still yields an interval."""
    wf = _waveforms(
        [DlisAxis("T", np.array([0.0, 10.0, 20.0, 30.0]), float("nan"), "us", 4)],
        shape=(3, 4),
    )
    assert wf.sample_interval() == pytest.approx(1.0e-5)


# --- The real files' shape, which `dliswriter` cannot produce ---------
#
# A Schlumberger monopole waveform channel is (n_depth, 8, 512) with two
# axes. `dliswriter` caps datasets at two dimensions and overrides a
# declared `dimension` from the data shape, so no writable file reaches
# this case. These stub the dlisio object graph instead; what keeps the
# stub honest is that the round-trip tests above exercise the same code
# against a real file, so any drift in the attribute names dlisio
# exposes breaks those rather than passing silently here.


class _StubAttr:
    def __init__(self, units):
        self.units = units


class _StubAxis:
    def __init__(self, axis_id, coordinates, spacing, units):
        self.axis_id = axis_id
        self.coordinates = coordinates
        self.spacing = spacing
        self.attic = {"SPACING": _StubAttr(units), "COORDINATES": _StubAttr(units)}


class _StubChannel:
    def __init__(self, name, data, units="", axis=()):
        self.name = name
        self.units = units
        self.axis = list(axis)
        self._data = data

    def curves(self):
        return self._data


class _StubFrame:
    def __init__(self, name, channels):
        self.name = name
        self.channels = channels


class _StubFiles(list):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_sonic_file(n_depth=4, n_rec=8, n_sample=512, n_axes=2):
    """The object graph dlisio hands back for a monopole array-sonic frame."""
    data = np.arange(n_depth * n_rec * n_sample, dtype=np.int16).reshape(
        n_depth, n_rec, n_sample
    )
    axes = [
        _StubAxis("SENSOR_OFFSET", [7.874], 0.1524, "m"),
        _StubAxis("MICRO_TIME", [0.0], 10.0, "us"),
    ][:n_axes]
    frame = _StubFrame(
        "60B",
        [
            _StubChannel("TDEP", np.arange(n_depth, dtype=float) * 6.0, units="0.1 in"),
            _StubChannel("PWF4", data, units="", axis=axes),
        ],
    )
    logical_file = type("LF", (), {"frames": [frame]})()
    return _StubFiles([logical_file]), data


def test_read_dlis_waveforms_reads_the_real_array_sonic_shape(monkeypatch):
    """(n_depth, n_receiver, n_sample) with both axes resolved.

    This is the case the whole reader exists for, and the numbers are
    the ones the Utah FORGE DSI file declares: 8 receivers 6 in apart
    starting 7.874 m from the source, sampled every 10 us.
    """
    from fwap.io import _dlis

    files, data = _stub_sonic_file()
    monkeypatch.setattr(_dlis.dlisio_dlis, "load", lambda _path: files)

    wf = read_dlis_waveforms("ignored.dlis", "PWF4")

    assert wf.data.shape == (4, 8, 512)
    assert np.array_equal(wf.data, data)
    assert wf.frame_name == "60B"
    assert wf.depth_unit == "0.1 in", "the index unit is reported, not converted"

    assert [a.axis_id for a in wf.axes] == ["SENSOR_OFFSET", "MICRO_TIME"]
    assert wf.sample_interval() == pytest.approx(1.0e-5)
    offsets = wf.offsets()
    assert offsets.shape == (8,)
    assert offsets[0] == pytest.approx(7.874)
    # 6 inches between receivers, which is what sets the slowness scale.
    assert np.allclose(np.diff(offsets), 0.1524)


def test_read_dlis_waveforms_declines_a_partial_axis_list(monkeypatch):
    """Fewer AXIS objects than dimensions is legal, and unassignable.

    Guessing which dimension a lone axis describes would put a receiver
    spacing where a sample interval belongs. Reporting no axes at all
    makes the accessors raise instead.
    """
    from fwap.io import _dlis

    files, _ = _stub_sonic_file(n_axes=1)
    monkeypatch.setattr(_dlis.dlisio_dlis, "load", lambda _path: files)

    wf = read_dlis_waveforms("ignored.dlis", "PWF4")

    assert wf.data.shape == (4, 8, 512)
    assert wf.axes == []
    with pytest.raises(ValueError, match="time unit"):
        wf.sample_interval()


def test_waveform_sample_interval_refuses_a_single_coordinate_with_no_spacing():
    """One coordinate and no SPACING pins nothing, and says so."""
    wf = _waveforms(
        [DlisAxis("T", np.array([0.0]), float("nan"), "us", 1)], shape=(3, 1)
    )
    with pytest.raises(ValueError, match="fewer than two coordinates"):
        wf.sample_interval()
