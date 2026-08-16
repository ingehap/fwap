"""
Integration tests against real-world files written by other software.

Every other test in this suite runs on synthetics with planted answers. That is
what makes them assertable, and also what bounds them: a synthetic file is
produced by the same assumptions the reader holds, so it can never catch a
convention the reader failed to anticipate. These tests can.

The files are **not** in the repository -- they are third-party and fetched on
demand (see ``scripts/fetch_real_data.py`` for the registry, provenance and
licensing). Without them every test here skips with a pointer to that script, so
a normal ``pytest`` run is unaffected and the ``ci.yml`` gate stays hermetic.

Run them with::

    python scripts/fetch_real_data.py --fetch all
    pytest tests/test_real_data.py -v

**Skipping is the danger, not the feature.** For a long time that skip was the
whole story: these were the strongest external check outside the cylindrical
solver and the only one that could catch a convention the readers failed to
anticipate, and they had never once run -- every skip in every CI run came from
this file. They now have their own ``.github/workflows/real-data.yml``, weekly
and post-merge, which fetches the registry and then runs
``fetch_real_data.py --verify`` *before* pytest. That verify step is the point:
without it, a run that downloaded nothing would skip everything and report
green.

This file used to say that no openly redistributable full-waveform sonic gather
was known to exist. ``iodp_u1347a_dsi`` is the counterexample -- an
eight-receiver DSI run published under CC0 -- so the registry now reaches real
waveforms and not only real *parsing* oddities. What is still true, and narrower,
is that one hole of one tool bounds how wrong the processing can be without
turning ``sonic_ml``'s synthetic identifiability results into field
measurements. See the module docstring of the fetch script.
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
        assert dataset.kind in {"las", "segy", "dlis", "ldeo"}
        assert dataset.url.startswith("https://")
        assert len(dataset.sha256) == 64
        # A member is only meaningful with its own digest: the archive's hash
        # cannot vouch for it, because recompressing a zip changes one and not
        # the other.
        if dataset.member is not None:
            assert dataset.member_sha256 is not None
            assert len(dataset.member_sha256) == 64
        else:
            assert dataset.member_sha256 is None
        # Provenance and licence are not optional: a fixture whose origin is
        # unrecorded cannot be audited later.
        assert dataset.provenance.strip()
        assert dataset.licence.strip()
        assert dataset.what_it_tests.strip()


#: The entries whose digests have never been checked against bytes that came
#: down their own URL. Pinned rather than derived, so that confirming one is a
#: deliberate edit here as well as in the registry -- roadmap F.4.
_UNCONFIRMED_CHECKSUMS = ("forge_dsi_las", "iodp_u1347a_dsi")


def test_the_registry_says_the_same_thing_twice_about_its_checksums():
    """A fixture registry must not assert what it has not verified.

    Two entries carry digests computed from copies that did not come down
    their canonical URL, because the hosts were unreachable from the sessions
    that added them -- and still are, refused at the network gateway. That is
    recorded in two places: a ``checksum_confirmed`` flag for programs and a
    ``CHECKSUM CAVEAT`` paragraph for people.

    Recording it twice is only useful if the two cannot drift, which is what
    this checks. The failure it is aimed at is the quiet one: someone
    confirms a digest, clears the flag, leaves the paragraph -- or the
    reverse -- and the registry then looks verified where it is not.
    """
    assert fetch_real_data.check_registry_caveats() == _UNCONFIRMED_CHECKSUMS

    for dataset in fetch_real_data.DATASETS:
        if dataset.name in _UNCONFIRMED_CHECKSUMS:
            assert not dataset.checksum_confirmed
            assert fetch_real_data.CAVEAT_MARKER in dataset.provenance
        else:
            assert dataset.checksum_confirmed
            assert fetch_real_data.CAVEAT_MARKER not in dataset.provenance


def test_an_unconfirmed_entry_cannot_lose_half_its_record():
    """The drift this guards against, made to happen.

    Clearing the flag while leaving the prose -- or the reverse -- must be an
    error rather than a quietly weaker claim.
    """
    import dataclasses

    original = fetch_real_data.find("forge_dsi_las")
    assert not original.checksum_confirmed

    for mutation in (
        dataclasses.replace(original, checksum_confirmed=True),
        dataclasses.replace(
            original,
            provenance=original.provenance.replace(
                fetch_real_data.CAVEAT_MARKER, "note"
            ),
        ),
    ):
        patched = tuple(
            mutation if d.name == original.name else d for d in fetch_real_data.DATASETS
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fetch_real_data, "DATASETS", patched)
            with pytest.raises(ValueError, match="checksum_confirmed"):
                fetch_real_data.check_registry_caveats()


def test_the_registry_listing_admits_the_unconfirmed_entries():
    """Whoever runs ``--list`` should see it, not only whoever reads the code.

    An unverified claim recorded where only a maintainer will find it is
    barely recorded at all, so the header line says it outright.
    """
    lines = fetch_real_data.format_registry().splitlines()
    headers = {
        line.split()[0]: line for line in lines if line and not line.startswith(" ")
    }
    for dataset in fetch_real_data.DATASETS:
        header = headers[dataset.name]
        flagged = "CHECKSUM UNCONFIRMED AGAINST URL" in header
        assert flagged == (dataset.name in _UNCONFIRMED_CHECKSUMS), header


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


def test_reads_a_real_schlumberger_sonic_log():
    """The first *sonic* log in the registry, and the first vendor picks.

    The KGS log above tests LAS parsing against real service-company oddities.
    This one tests something the package could not test at all before: whether
    its answers agree with a vendor's on the same rock. It is a Schlumberger
    DSI run whose curve set is the tool's own slowness-time processing output --
    compressional and shear slowness plus the per-mode coherence peaks.
    """
    from fwap import read_las

    las = read_las(str(_require("forge_dsi_las")))

    assert las.well.get("WELL") == "MU-ESW1"
    assert las.well.get("SRVC") == "Schlumberger"
    # 0.5 ft sampling over the logged interval.
    assert las.depth.size > 10_000
    assert 2150.0 < las.depth.min() < 2151.0
    assert 7568.0 < las.depth.max() < 7570.0

    # The reference picks, and the coherence peaks that produced them.
    for expected in ("DTCO", "DTSM", "CHRP", "CHR1", "CHR2"):
        assert expected in las.curves, f"missing DSI output {expected}"
    assert las.units.get("DTCO") == "US/F"

    for name, values in las.curves.items():
        assert values.shape == las.depth.shape, f"{name} is depth-misaligned"


def test_real_sonic_reference_picks_are_physically_ordered():
    """Shear is slower than compressional at every depth where both are picked.

    A property of the rock rather than of the tool, so it holds for any honest
    log and fails loudly for a mis-mapped curve. It is also the invariant
    ``quality_control_picks`` enforces on our own picks, checked here against a
    vendor's -- which is what makes it evidence rather than a tautology.
    """
    from fwap import read_las

    las = read_las(str(_require("forge_dsi_las")))
    dtco, dtsm = las.curves["DTCO"], las.curves["DTSM"]
    both = np.isfinite(dtco) & np.isfinite(dtsm) & (dtco > 0) & (dtsm > 0)
    assert both.sum() > 10_000

    assert np.all(dtsm[both] > dtco[both]), "shear must be slower than P"
    # Vp/Vs stays in the range real rock occupies; a units slip would break it.
    vp_vs = dtsm[both] / dtco[both]
    assert 1.4 < np.median(vp_vs) < 2.4
    assert np.all(vp_vs < 5.0)


# ------------------------------------------------------------------
# IODP U1347A -- a real eight-receiver DSI gather (F.2)
# ------------------------------------------------------------------


def _require_member(name: str) -> Path:
    """Return a zip-archived dataset's extracted member, or skip."""
    dataset = fetch_real_data.find(name)
    _require(name)  # archive present and its own checksum good
    try:
        return fetch_real_data.extract_member(dataset)
    except ValueError as exc:  # pragma: no cover - depends on local files
        pytest.fail(f"{name}: {exc}")


def test_reads_a_real_dsi_waveform_gather():
    """The registry's first real waveforms, read by the package's own reader.

    Every field here is asserted against the archive's published file
    description rather than against what the reader happened to return, so a
    reader that silently changed convention would fail rather than agree with
    itself.
    """
    from fwap import read_ldeo_waveforms

    path = _require_member("iodp_u1347a_dsi")
    wf = read_ldeo_waveforms(path, max_depths=0)

    assert wf.tool == "DSI"
    assert wf.mode == "Monopole"
    assert (wf.n_depth, wf.n_receiver, wf.n_sample) == (1307, 8, 512)
    assert wf.sample_interval == pytest.approx(1.0e-5)
    assert wf.depth_increment == pytest.approx(0.1524, rel=1e-6)
    assert wf.record_length == pytest.approx(5.12e-3)


def test_real_dsi_depths_are_monotonic_and_regular():
    """A depth axis that is not monotonic would invalidate every depth-indexed
    result downstream, and no synthetic in this suite can drift the way a real
    wireline depth channel can."""
    from fwap import read_ldeo_waveforms

    wf = read_ldeo_waveforms(_require_member("iodp_u1347a_dsi"))
    step = np.diff(wf.depth)
    assert (step > 0).all(), "depths must increase"
    assert np.median(step) == pytest.approx(wf.depth_increment, rel=1e-3)
    assert wf.depth[-1] - wf.depth[0] == pytest.approx(199.0, abs=0.5)


def test_real_dsi_gathers_are_signal_not_padding():
    """Guards the failure that would make every other assertion vacuous.

    A misread offset or a wrong record length would still yield an array of the
    right shape -- full of zeros, or of one receiver repeated. Both are checked
    directly: every receiver must carry energy, and no two receivers may be
    identical.
    """
    from fwap import read_ldeo_waveforms

    wf = read_ldeo_waveforms(_require_member("iodp_u1347a_dsi"), max_depths=200)
    rms = np.sqrt(np.mean(wf.data**2, axis=2))

    assert np.isfinite(wf.data).all()
    assert (rms > 0).all(), "a receiver with no energy means a misread record"
    for i in range(wf.n_receiver - 1):
        same = np.isclose(wf.data[:, i], wf.data[:, i + 1]).all(axis=1)
        assert not same.any(), f"receivers {i} and {i + 1} are identical"


def test_stc_on_real_dsi_gathers_finds_coherent_arrivals():
    """The point of the whole exercise: the processing meets data it did not
    generate.

    The assertion is deliberately loose on the *value* -- U1347A logs chert,
    chalk and basalt over this interval, so slowness genuinely ranges over a
    factor of four and pinning a number would be pinning the lithology. What is
    asserted instead is that coherent arrivals exist at all, and that none of
    them sits on an edge of the search band. The second is the more useful
    check: a band too narrow for the formation returns picks pinned to its own
    boundary, which look like measurements and are not. The first band tried
    here was (5e-5, 6e-4) and 10 % of its picks came back pinned at 6e-4.
    """
    from fwap import read_ldeo_waveforms, stc

    # Receiver spacing is 0.1524 m, from the archive's file description. It
    # happens to equal the depth increment -- both are six inches -- but they
    # are different quantities, so it is written out rather than reused.
    receiver_spacing = 0.1524
    band = (5.0e-5, 1.0e-3)

    wf = read_ldeo_waveforms(_require_member("iodp_u1347a_dsi"), max_depths=400)
    offsets = 3.0 + np.arange(wf.n_receiver) * receiver_spacing

    peaks, slownesses = [], []
    for k in range(0, wf.data.shape[0], 8):
        result = stc(
            wf.gather(k),
            wf.sample_interval,
            offsets,
            slowness_range=band,
            n_slowness=221,
            window_length=6.0e-4,
        )
        i, _ = np.unravel_index(np.argmax(result.coherence), result.coherence.shape)
        peaks.append(result.coherence.max())
        slownesses.append(result.slowness[i])

    peaks = np.asarray(peaks)
    slownesses = np.asarray(slownesses)
    assert np.median(peaks) > 0.8, f"median peak coherence {np.median(peaks):.3f}"
    assert (peaks > 0.6).mean() > 0.8

    good = slownesses[peaks > 0.6]
    pinned = np.isclose(good, band[0]) | np.isclose(good, band[1])
    assert not pinned.any(), (
        f"{pinned.sum()} of {good.size} picks sit on a search-band edge, so the "
        "band clipped the formation instead of bracketing it"
    )
