"""Tests for sonic_ml.provenance."""

from __future__ import annotations

import numpy as np

from sonic_ml import provenance
from sonic_ml.provenance import Provenance, capture, content_hash


def test_content_hash_is_deterministic(stacked):
    assert content_hash(stacked) == content_hash(dict(stacked))


def test_content_hash_changes_on_data_change(stacked):
    h0 = content_hash(stacked)
    perturbed = dict(stacked)
    g = perturbed["gather"].copy()
    g[0, 0, 0] += 1.0
    perturbed["gather"] = g
    assert content_hash(perturbed) != h0


def test_content_hash_changes_on_key_set_change(stacked):
    h0 = content_hash(stacked)
    fewer = {k: v for k, v in stacked.items() if k != "freq"}
    assert content_hash(fewer) != h0


def test_capture_records_version_and_hash(stacked):
    pv = capture({"n": 40, "seed": 0}, stacked, note="unit test")
    assert pv.fwap_version  # non-empty
    assert pv.content_hash == content_hash(stacked)
    assert pv.config == {"n": 40, "seed": 0}
    assert pv.note == "unit test"
    # git sha is either a hex-ish string or None (graceful when not a checkout)
    assert pv.fwap_git_sha is None or isinstance(pv.fwap_git_sha, str)


def test_json_round_trip(stacked):
    pv = capture(
        {"n": 3}, stacked, split_indices={"train": [0, 1], "val": [2], "test": []}
    )
    back = Provenance.from_json(pv.to_json())
    assert back == pv


def test_sidecar_round_trip(tmp_path, stacked):
    npz = tmp_path / "d.npz"
    npz.write_bytes(b"")  # the sidecar path derives from this name only
    pv = capture({"n": 40}, stacked)
    sidecar = pv.write_sidecar(str(npz))
    assert sidecar.name == "d.npz" + provenance.SIDECAR_SUFFIX
    assert Provenance.read_sidecar(str(npz)) == pv


def test_git_sha_is_none_safe():
    # Should never raise, whatever the environment.
    sha = provenance.fwap_git_sha()
    assert sha is None or isinstance(sha, str)
