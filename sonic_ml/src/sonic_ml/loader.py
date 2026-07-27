"""
Reader for the surrogate-dataset ``.npz`` written by
``scripts/gen_surrogate_dataset.py``.

The ``.npz`` is the sole contract between the (pure NumPy/SciPy) fwap data
generator and this ML layer. This module loads it defensively -- with
``allow_pickle=False`` (never execute pickled objects from a data file),
asserting the ``schema_version`` is one we understand, and reading the sample
count ``N``, mode count ``M`` and frequency count ``F`` from the stored
metadata (``param_names`` / ``mode_names`` / ``freq``) rather than hard-coding
them. See ``tests/test_npz_schema_contract.py`` in the core repo for the frozen
layout this mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Schema versions this loader understands. Extend (do not silently widen)
#: when the generator bumps ``SCHEMA_VERSION`` in a backward-compatible way.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})

#: Keys every conformant ``.npz`` must contain.
REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "params",
        "slowness",
        "gather",
        "mode_in_gather",
        "freq",
        "param_names",
        "mode_names",
        "schema_version",
    }
)


class SchemaError(ValueError):
    """The ``.npz`` does not match the expected surrogate-dataset layout."""


class UnsupportedSchemaVersionError(SchemaError):
    """The ``.npz`` schema version is not in :data:`SUPPORTED_SCHEMA_VERSIONS`."""


@dataclass(frozen=True)
class DatasetBundle:
    """
    An in-memory surrogate dataset.

    Attributes
    ----------
    params : ndarray, shape (N, P), float64
        Formation parameters, columns in :attr:`param_names` order.
    slowness : ndarray, shape (N, M, F), float64
        Per-mode phase-slowness curves (s/m); ``NaN`` where a mode is absent
        at a frequency. This is the forward-surrogate label.
    gather : ndarray, shape (N, R, T), float64
        Synthetic multi-receiver waveforms (R receivers, T samples). The
        inverse-net input.
    mode_in_gather : ndarray, shape (N, M), bool
        ``True`` where a mode was injected into the gather. This -- not
        ``isfinite(slowness).any()`` -- is the authoritative mode-presence
        label (a partly-finite curve with too few points is still "absent").
    freq : ndarray, shape (F,), float64
        Shared frequency grid (Hz).
    param_names : tuple of str, length P
        Formation-parameter column names.
    mode_names : tuple of str, length M
        Mode labels aligned with axis 1 of :attr:`slowness` /
        :attr:`mode_in_gather`.
    schema_version : int
        The on-disk contract version this bundle was loaded from.
    """

    params: np.ndarray
    slowness: np.ndarray
    gather: np.ndarray
    mode_in_gather: np.ndarray
    freq: np.ndarray
    param_names: tuple[str, ...]
    mode_names: tuple[str, ...]
    schema_version: int

    @property
    def n_samples(self) -> int:
        """Number of samples ``N``."""
        return int(self.params.shape[0])

    @property
    def n_modes(self) -> int:
        """Number of modes ``M`` (read from :attr:`mode_names`)."""
        return len(self.mode_names)

    @property
    def n_freq(self) -> int:
        """Number of frequency samples ``F``."""
        return int(self.freq.shape[0])

    def param(self, name: str) -> np.ndarray:
        """
        Return one formation-parameter column by name.

        Parameters
        ----------
        name : str
            A member of :attr:`param_names` (e.g. ``"vs"``).

        Returns
        -------
        ndarray, shape (N,), float64

        Raises
        ------
        KeyError
            If ``name`` is not a known parameter column.
        """
        try:
            idx = self.param_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"unknown parameter {name!r}; have {self.param_names}"
            ) from exc
        return self.params[:, idx]

    def finite_mask(self) -> np.ndarray:
        """Boolean ``(N, M, F)`` mask of finite slowness samples.

        Use this to mask any per-frequency slowness loss; never train on the
        ``NaN`` sentinels.
        """
        return np.isfinite(self.slowness)


def load_npz(path: str) -> DatasetBundle:
    """
    Load a surrogate-dataset ``.npz`` into a :class:`DatasetBundle`.

    Parameters
    ----------
    path : str
        Path to a ``.npz`` written by ``gen_surrogate_dataset.save_npz``.

    Returns
    -------
    DatasetBundle

    Raises
    ------
    SchemaError
        If a required key is missing or array shapes are mutually
        inconsistent.
    UnsupportedSchemaVersionError
        If the file's ``schema_version`` is not supported.
    """
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = REQUIRED_KEYS - keys
        if missing:
            raise SchemaError(
                f"{path}: missing required keys {sorted(missing)}; found {sorted(keys)}"
            )

        schema_version = int(data["schema_version"])
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaVersionError(
                f"{path}: schema_version {schema_version} not supported "
                f"(this loader understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}); "
                "regenerate the dataset or upgrade sonic_ml"
            )

        params = np.asarray(data["params"])
        slowness = np.asarray(data["slowness"])
        gather = np.asarray(data["gather"])
        mode_in_gather = np.asarray(data["mode_in_gather"])
        freq = np.asarray(data["freq"])
        param_names = tuple(str(s) for s in data["param_names"].tolist())
        mode_names = tuple(str(s) for s in data["mode_names"].tolist())

    n = params.shape[0]
    n_params = len(param_names)
    n_modes = len(mode_names)
    n_freq = freq.shape[0]

    _check(params.ndim == 2, f"params must be 2-D, got shape {params.shape}")
    _check(
        params.shape[1] == n_params,
        f"params has {params.shape[1]} columns but param_names has {n_params}",
    )
    _check(
        slowness.shape == (n, n_modes, n_freq),
        f"slowness shape {slowness.shape} != (N={n}, M={n_modes}, F={n_freq})",
    )
    _check(
        mode_in_gather.shape == (n, n_modes),
        f"mode_in_gather shape {mode_in_gather.shape} != (N={n}, M={n_modes})",
    )
    _check(
        gather.ndim == 3 and gather.shape[0] == n,
        f"gather must be (N, R, T) with N={n}, got shape {gather.shape}",
    )
    _check(freq.ndim == 1, f"freq must be 1-D, got shape {freq.shape}")

    return DatasetBundle(
        params=params,
        slowness=slowness,
        gather=gather,
        mode_in_gather=mode_in_gather,
        freq=freq,
        param_names=param_names,
        mode_names=mode_names,
        schema_version=schema_version,
    )


def _check(condition: bool, message: str) -> None:
    """Raise :class:`SchemaError` with ``message`` unless ``condition``."""
    if not condition:
        raise SchemaError(message)
