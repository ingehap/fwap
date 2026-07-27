"""
Reproducibility provenance for surrogate datasets.

Dataset labels (``slowness``, ``gather``) ARE fwap solver output, so a change
to the physics engine -- not just to the generator config -- silently
re-labels a dataset and invalidates any model trained on it. A :class:`Provenance`
binds the essentials for exact reproduction and for detecting drift: the fwap
version and git SHA, the generating configuration, and a content hash of the
stacked arrays. Persisted as a JSON sidecar next to the ``.npz`` (and later
bound to trained weights) it answers "which fwap and which config produced
this data, and is the data on disk still the data I trained on?".
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fwap
import numpy as np

SIDECAR_SUFFIX = ".provenance.json"


def fwap_git_sha() -> str | None:
    """
    Current git commit of the fwap checkout, or ``None``.

    Returns ``None`` when fwap is not installed from a git checkout or ``git``
    is unavailable -- provenance degrades gracefully rather than failing.
    """
    fwap_file = fwap.__file__
    if fwap_file is None:  # pragma: no cover - a namespace-only fwap is unexpected
        return None
    repo = Path(fwap_file).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = out.stdout.strip()
    return sha or None


def content_hash(stacked: dict[str, np.ndarray]) -> str:
    """
    Deterministic SHA-256 over a stacked-dataset dict.

    Hashes each key (sorted), its dtype, shape, and raw bytes, so any change to
    a value, dtype, shape, or the key set changes the digest.

    Parameters
    ----------
    stacked : dict of str to ndarray
        The mapping returned by ``gen_surrogate_dataset.stack_dataset``.

    Returns
    -------
    str
        Hex SHA-256 digest.
    """
    h = hashlib.sha256()
    for key in sorted(stacked):
        arr = np.ascontiguousarray(stacked[key])
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(repr(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Provenance:
    """
    Reproducibility record for a generated dataset.

    Attributes
    ----------
    fwap_version : str
        ``fwap.__version__`` at generation time.
    fwap_git_sha : str or None
        Git commit of the fwap checkout, if available.
    config : dict
        The generating configuration (e.g. ``n``, ``seed``, frequency-grid
        spec, prior fields, mode names, ``noise_max``, ``min_finite``). Must be
        JSON-serializable.
    content_hash : str
        SHA-256 of the stacked dataset (:func:`content_hash`).
    split_indices : dict or None
        Optional stored ``{"train": [...], "val": [...], "test": [...]}``
        binding the exact partition to this dataset.
    note : str or None
        Free-form human note.
    """

    fwap_version: str
    fwap_git_sha: str | None
    config: dict[str, Any]
    content_hash: str
    split_indices: dict[str, list[int]] | None = None
    note: str | None = None

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Provenance:
        """Rebuild a :class:`Provenance` from :meth:`to_json` output."""
        d = json.loads(text)
        return cls(
            fwap_version=d["fwap_version"],
            fwap_git_sha=d.get("fwap_git_sha"),
            config=d.get("config", {}),
            content_hash=d["content_hash"],
            split_indices=d.get("split_indices"),
            note=d.get("note"),
        )

    def write_sidecar(self, npz_path: str) -> Path:
        """
        Write ``<npz_path>.provenance.json`` next to the dataset.

        Parameters
        ----------
        npz_path : str
            Path to the ``.npz`` this record describes.

        Returns
        -------
        pathlib.Path
            The written sidecar path.
        """
        sidecar = Path(str(npz_path) + SIDECAR_SUFFIX)
        sidecar.write_text(self.to_json())
        return sidecar

    @classmethod
    def read_sidecar(cls, npz_path: str) -> Provenance:
        """Read the provenance sidecar written for ``npz_path``."""
        sidecar = Path(str(npz_path) + SIDECAR_SUFFIX)
        return cls.from_json(sidecar.read_text())


def capture(
    config: dict[str, Any],
    stacked: dict[str, np.ndarray],
    *,
    split_indices: dict[str, list[int]] | None = None,
    note: str | None = None,
) -> Provenance:
    """
    Build a :class:`Provenance` for a freshly generated dataset.

    Parameters
    ----------
    config : dict
        JSON-serializable generating configuration.
    stacked : dict of str to ndarray
        The stacked dataset (from ``stack_dataset``) to hash.
    split_indices : dict or None
        Optional stored split partition.
    note : str or None
        Optional human note.

    Returns
    -------
    Provenance
    """
    return Provenance(
        fwap_version=getattr(fwap, "__version__", "unknown"),
        fwap_git_sha=fwap_git_sha(),
        config=config,
        content_hash=content_hash(stacked),
        split_indices=split_indices,
        note=note,
    )
