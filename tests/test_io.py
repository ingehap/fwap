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

from fwap.io import DlisCurves, read_dlis, write_dlis


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
