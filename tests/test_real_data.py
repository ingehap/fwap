"""
Integration tests against real-world files written by other software.

Every other test in this suite runs on synthetics with planted answers. That is
what makes them assertable, and also what bounds them: a synthetic file is
produced by the same assumptions the reader holds, so it can never catch a
convention the reader failed to anticipate. These tests can.

The files are **not** in the repository -- they are third-party and fetched on
demand (see ``scripts/fetch_real_data.py`` for the registry, provenance and
licensing). Without them every test here skips with a pointer to that script, so
a normal ``pytest`` run is unaffected and CI stays hermetic.

Run them with::

    python scripts/fetch_real_data.py --fetch all
    pytest tests/test_real_data.py -v

What these tests do *not* cover: neither file is a full-waveform sonic gather,
because no openly redistributable one is known to exist. The sonic processing
chain -- and every quantitative claim built on it -- is still validated only
against synthetics. See the module docstring of the fetch script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_real_data.py"
_spec = importlib.util.spec_from_file_location("fetch_real_data", _SCRIPT)
assert _spec is not None and _spec.loader is not None
fetch_real_data = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fetch_real_data
_spec.loader.exec_module(fetch_real_data)


def _require(name: str) -> Path:
    """Return the local path of a registered dataset, or skip the test."""
    dataset = fetch_real_data.find(name)
    path = fetch_real_data.local_path(dataset)
    if not path.is_file():
        pytest.skip(
            f"real-data file {dataset.filename!r} not present; fetch it with "
            f"`python scripts/fetch_real_data.py --fetch {name}`"
        )
    if not fetch_real_data.verify(dataset):
        pytest.fail(
            f"{path} is present but its SHA-256 does not match the registry; "
            "the assertions below were written against different bytes"
        )
    return path


# ------------------------------------------------------------------
# The registry itself (no downloads required)
# ------------------------------------------------------------------


def test_registry_entries_are_well_formed():
    """Every entry must carry the metadata that makes it auditable."""
    assert fetch_real_data.DATASETS
    names = [d.name for d in fetch_real_data.DATASETS]
    assert len(names) == len(set(names)), "dataset names must be unique"
    for dataset in fetch_real_data.DATASETS:
        assert dataset.kind in {"las", "segy"}
        assert dataset.url.startswith("https://")
        assert len(dataset.sha256) == 64
        # Provenance and licence are not optional: a fixture whose origin is
        # unrecorded cannot be audited later.
        assert dataset.provenance.strip()
        assert dataset.licence.strip()
        assert dataset.what_it_tests.strip()


def test_find_rejects_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        fetch_real_data.find("not_a_dataset")


def test_registry_listing_mentions_licences():
    text = fetch_real_data.format_registry()
    for dataset in fetch_real_data.DATASETS:
        assert dataset.name in text
    assert "licence" in text
    assert "git-ignored" in text


def test_corrupted_file_is_refused_not_silently_used(tmp_path):
    """A checksum mismatch must be an error, and must not trigger a download.

    The failure this guards against is the quiet one: testing against different
    bytes than the assertions were written for, and reporting a pass. No network
    is needed to check it -- a wrong-content file is refused on inspection.
    """
    dataset = fetch_real_data.DATASETS[0]
    (tmp_path / dataset.filename).write_bytes(b"not the real file")

    assert fetch_real_data.verify(dataset, tmp_path) is False
    with pytest.raises(ValueError, match="checksum"):
        fetch_real_data.fetch(dataset, tmp_path)


def test_verify_reports_false_for_a_missing_file(tmp_path):
    assert fetch_real_data.verify(fetch_real_data.DATASETS[0], tmp_path) is False


def test_data_dir_honours_the_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv(fetch_real_data.DATA_DIR_ENV, str(tmp_path))
    assert fetch_real_data.data_dir() == tmp_path


def test_fetched_data_directory_is_git_ignored():
    """The no-redistribution guarantee is enforced, not merely intended."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "tests/data/real/example.las"],
            capture_output=True,
            text=True,
        )
    except OSError:  # pragma: no cover - git absent
        pytest.skip("git not available")
    assert result.stdout.strip(), (
        "tests/data/real/ must be git-ignored so third-party files fetched by "
        "scripts/fetch_real_data.py can never be committed"
    )


# ------------------------------------------------------------------
# Real LAS: a file our own writer would never produce
# ------------------------------------------------------------------


def test_reads_a_real_wrapped_las():
    from fwap import read_las

    las = read_las(str(_require("kgs_las")))

    # Depth is monotonic and in the header's stated range (feet).
    assert las.depth.ndim == 1 and las.depth.size > 0
    assert np.all(np.diff(las.depth) > 0)
    assert 1780.0 < las.depth.min() < 1790.0

    # A real service-company curve set, not the tidy DTP/DTS/DTST fwap writes.
    assert len(las.curves) > 20
    for expected in ("GSGR", "DLDN", "ACTC"):
        assert expected in las.curves, f"missing real-world curve {expected}"

    # Every curve is aligned with the depth axis.
    for name, values in las.curves.items():
        assert values.shape == las.depth.shape, f"{name} is depth-misaligned"


def test_real_las_preserves_well_identity_and_units():
    from fwap import read_las

    las = read_las(str(_require("kgs_las")))
    # Header metadata survives parsing -- this is what ties a curve to a well.
    assert las.well.get("UWI") == "15-187-20743"
    assert "STAT" in las.well
    # Units come through from the curve section.
    assert las.units.get("ACTC")


def test_real_las_curves_are_finite_or_null_never_garbage():
    from fwap import read_las

    las = read_las(str(_require("kgs_las")))
    for name, values in las.curves.items():
        finite = values[np.isfinite(values)]
        if finite.size:
            # LAS null sentinels (-999.25 and friends) must not survive as data.
            assert finite.min() > -900.0, f"{name} looks like an unhandled null"


# ------------------------------------------------------------------
# Foreign-written SEG-Y: a reader/writer disagreement a round-trip can't catch
# ------------------------------------------------------------------


def test_reads_a_segy_written_by_other_software():
    from fwap import read_segy

    gather = read_segy(str(_require("segyio_small")))

    assert gather.data.ndim == 2
    assert gather.data.shape == (gather.n_traces, gather.n_samples)
    assert gather.n_traces > 0 and gather.n_samples > 0
    assert np.all(np.isfinite(gather.data))
    # Sampling interval decoded from the binary header, not assumed.
    assert gather.dt > 0.0


def test_foreign_segy_round_trips_through_our_writer():
    """Read a foreign file, write it with fwap, read it back unchanged.

    This is the composition that matters: if our reader mis-decodes a foreign
    convention, or our writer emits something we alone can read, the values
    diverge here even though a write-then-read test would pass.
    """
    import tempfile

    from fwap import read_segy, write_segy

    original = read_segy(str(_require("segyio_small")))
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "roundtrip.sgy")
        write_segy(out, original.data, dt=original.dt)
        again = read_segy(out)

    assert again.data.shape == original.data.shape
    assert again.dt == pytest.approx(original.dt)
    # SEG-Y IBM/IEEE float conversion is lossy in the last bits; compare at
    # single-precision tolerance rather than demanding equality.
    np.testing.assert_allclose(again.data, original.data, rtol=1e-6, atol=1e-9)
