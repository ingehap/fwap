"""CLI entry-point tests."""

from __future__ import annotations

import numpy as np
import pytest

from fwap.cli import main
from fwap.io import write_segy
from fwap.synthetic import (
    ArrayGeometry,
    monopole_formation_modes,
    synthesize_gather,
)


def _prepare_sgy(path, Vp=4500.0, Vs=2500.0, Vst=1400.0, seed=7):
    geom = ArrayGeometry.schlumberger_array_sonic()
    modes = monopole_formation_modes(vp=Vp, vs=Vs, v_stoneley=Vst)
    data = synthesize_gather(geom, modes, noise=0.05, seed=seed)
    # Offsets written as integer millimetres so int round-trip through
    # the SEG-Y ``offset`` trace header is lossless.
    offsets_mm = np.round(geom.offsets * 1000.0).astype(int)
    write_segy(str(path), data.astype(np.float32), dt=geom.dt, offsets=offsets_mm)


def test_fwap_process_prints_picks(tmp_path, capsys):
    """fwap process <file.sgy> prints P / S / Stoneley picks to stdout."""
    sgy = tmp_path / "gather.sgy"
    _prepare_sgy(sgy)

    rc = main(["process", str(sgy), "--offset-scale", "1000", "--quiet"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "mode" in out
    # Each mode row has the form ``<name>  <us/ft>  <m/s>  <coh>``.
    for mode, _v_true in [("P", 4500.0), ("S", 2500.0), ("Stoneley", 1400.0)]:
        assert mode in out, f"missing {mode} row:\n{out}"
    # Parse the velocity column for P and check it's within 2% of truth.
    for line in out.splitlines():
        if line.startswith("P"):
            parts = line.split()
            v = float(parts[2])
            assert abs(v - 4500.0) / 4500.0 < 0.02, f"Vp={v}"
            break


def test_fwap_process_requires_offsets(tmp_path, capsys):
    """Non-zero offsets are required for processing; all-zero -> exit 2."""
    sgy = tmp_path / "no_offsets.sgy"
    geom = ArrayGeometry.schlumberger_array_sonic()
    modes = monopole_formation_modes()
    data = synthesize_gather(geom, modes, noise=0.05, seed=0)
    write_segy(
        str(sgy),
        data.astype(np.float32),
        dt=geom.dt,
        offsets=np.zeros(geom.n_rec, dtype=int),
    )

    rc = main(["process", str(sgy), "--quiet"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "offsets" in err


def test_fwap_process_unknown_args_errors(tmp_path):
    """--not-a-real-flag exits with a non-zero status."""
    sgy = tmp_path / "g.sgy"
    _prepare_sgy(sgy)
    with pytest.raises(SystemExit):
        main(["process", str(sgy), "--definitely-not-a-flag"])


def test_fwap_process_multi_gather_table(tmp_path, capsys):
    """Multi-file input without --output prints a per-depth summary table."""
    paths = []
    for i, vs in enumerate([2400.0, 2500.0, 2600.0]):
        p = tmp_path / f"{i:03d}.sgy"
        _prepare_sgy(p, Vs=vs)
        paths.append(str(p))
    rc = main(
        [
            "process",
            *paths,
            "--offset-scale",
            "1000",
            "--depth-start",
            "1000.0",
            "--depth-step",
            "0.1524",
            "--quiet",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "depth_m" in out
    lines = [line for line in out.splitlines() if line and line[0].isdigit()]
    assert len(lines) == 3, f"expected 3 rows, got {lines!r}"


def test_fwap_process_multi_gather_writes_las(tmp_path):
    """Multi-file input with --output writes a LAS file with expected curves."""
    paths = []
    for i, vs in enumerate([2400.0, 2500.0, 2600.0]):
        p = tmp_path / f"{i:03d}.sgy"
        _prepare_sgy(p, Vs=vs)
        paths.append(str(p))
    out = tmp_path / "logs.las"
    rc = main(
        [
            "process",
            *paths,
            "--offset-scale",
            "1000",
            "--depth-start",
            "1000.0",
            "--depth-step",
            "0.1524",
            "--output",
            str(out),
            "--quiet",
        ]
    )
    assert rc == 0
    assert out.exists()

    from fwap.io import read_las

    loaded = read_las(str(out))
    for mnemonic in (
        "DTP",
        "DTS",
        "DTST",
        "VPVS",
        "COHP",
        "COHS",
        "COHST",
        "K",
        "MU",
        "E",
        "NU",
    ):
        assert mnemonic in loaded.curves
    # DTS should decrease monotonically as Vs increases across depths.
    dts = loaded.curves["DTS"]
    assert len(dts) == 3
    assert dts[0] > dts[1] > dts[2]
