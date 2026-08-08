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
from fwap import ArrayGeometry

#: Schema versions this loader understands. Extend (do not silently widen)
#: when the generator bumps ``SCHEMA_VERSION`` in a backward-compatible way.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2, 3, 4})

#: Keys every conformant ``.npz`` must contain (all schema versions).
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

#: Acquisition-geometry scalars added in schema v2 (required for v2+).
GEOMETRY_KEYS: frozenset[str] = frozenset({"dt", "tr_offset", "dr"})

#: Leaky-mode attenuation label added in schema v3 (required for v3+).
ATTENUATION_KEY: str = "attenuation"

#: Cased-hole annulus + bond-evaluation keys added in schema v4 (required
#: for v4+). Open-hole v4 datasets carry an empty ``(N, 0, 4)`` layer stack
#: and an all-``NaN`` ``bond_index``.
CASED_KEYS: frozenset[str] = frozenset({"layer_params", "layer_names", "bond_index"})

#: Column order of the last axis of ``layer_params`` (schema v4). Mirrors
#: ``LAYER_PARAM_NAMES`` in ``scripts/gen_surrogate_dataset.py``; the core
#: ``tests/test_npz_schema_contract.py`` pins both.
LAYER_PARAM_NAMES: tuple[str, ...] = ("vp", "vs", "rho", "thickness")


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
    geometry : fwap.ArrayGeometry or None
        Acquisition geometry the gathers were synthesized with,
        reconstructed from the stored ``dt`` / ``tr_offset`` / ``dr``
        scalars plus the gather's ``(n_rec, n_samples)``. ``None`` for
        schema v1 files (which did not persist geometry).
    attenuation : ndarray or None, shape (N, M, F)
        Per-mode spatial attenuation rate (1/m) for leaky modes; ``NaN``
        for bound modes / absent frequencies. ``None`` for schema v1/v2
        files (which did not persist attenuation).
    layer_params : ndarray or None, shape (N, L, 4)
        Per-annular-layer ``[vp, vs, rho, thickness]`` for a cased-hole
        dataset (schema v4); an empty ``(N, 0, 4)`` array for an open-hole
        v4 dataset, and ``None`` for schema v1/v2/v3 files.
    layer_names : tuple of str, length L
        Layer labels aligned with axis 1 of :attr:`layer_params` (e.g.
        ``("casing", "cement")``); empty for open-hole / pre-v4 files.
    bond_index : ndarray or None, shape (N,)
        Cement-bond-quality proxy in ``[0, 1]`` for a cased-hole dataset
        (schema v4); ``NaN`` per-sample for open-hole draws, and ``None``
        for schema v1/v2/v3 files. The cement-bond-evaluation label.
    """

    params: np.ndarray
    slowness: np.ndarray
    gather: np.ndarray
    mode_in_gather: np.ndarray
    freq: np.ndarray
    param_names: tuple[str, ...]
    mode_names: tuple[str, ...]
    schema_version: int
    geometry: ArrayGeometry | None = None
    attenuation: np.ndarray | None = None
    layer_params: np.ndarray | None = None
    layer_names: tuple[str, ...] = ()
    bond_index: np.ndarray | None = None

    @property
    def is_cased(self) -> bool:
        """``True`` if this dataset carries a non-empty casing/cement stack."""
        return self.layer_params is not None and self.layer_params.shape[1] > 0

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

        geometry: ArrayGeometry | None = None
        if schema_version >= 2:
            missing_geom = GEOMETRY_KEYS - keys
            if missing_geom:
                raise SchemaError(
                    f"{path}: schema_version {schema_version} requires geometry "
                    f"keys but is missing {sorted(missing_geom)}"
                )
            if gather.ndim != 3:
                raise SchemaError(
                    f"{path}: schema v2 gather must be (N, n_rec, n_samples), "
                    f"got shape {gather.shape}"
                )
            geometry = ArrayGeometry(
                n_rec=int(gather.shape[1]),
                tr_offset=float(data["tr_offset"]),
                dr=float(data["dr"]),
                dt=float(data["dt"]),
                n_samples=int(gather.shape[2]),
            )

        attenuation: np.ndarray | None = None
        if schema_version >= 3:
            if ATTENUATION_KEY not in keys:
                raise SchemaError(
                    f"{path}: schema_version {schema_version} requires an "
                    f"'{ATTENUATION_KEY}' key"
                )
            attenuation = np.asarray(data[ATTENUATION_KEY])

        layer_params: np.ndarray | None = None
        layer_names: tuple[str, ...] = ()
        bond_index: np.ndarray | None = None
        if schema_version >= 4:
            missing_cased = CASED_KEYS - keys
            if missing_cased:
                raise SchemaError(
                    f"{path}: schema_version {schema_version} requires cased-hole "
                    f"keys but is missing {sorted(missing_cased)}"
                )
            layer_params = np.asarray(data["layer_params"])
            layer_names = tuple(str(s) for s in data["layer_names"].tolist())
            bond_index = np.asarray(data["bond_index"])

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
    if attenuation is not None:
        _check(
            attenuation.shape == (n, n_modes, n_freq),
            f"attenuation shape {attenuation.shape} != slowness shape "
            f"(N={n}, M={n_modes}, F={n_freq})",
        )
    if layer_params is not None:
        n_layers = len(layer_names)
        _check(
            layer_params.ndim == 3 and layer_params.shape[0] == n,
            f"layer_params must be (N, L, 4) with N={n}, got {layer_params.shape}",
        )
        _check(
            layer_params.shape[1] == n_layers,
            f"layer_params has {layer_params.shape[1]} layers but layer_names "
            f"has {n_layers}",
        )
        _check(
            layer_params.shape[2] == 4,
            f"layer_params last axis must be 4 (vp, vs, rho, thickness), "
            f"got {layer_params.shape[2]}",
        )
        assert bond_index is not None  # v4 reads both together
        _check(
            bond_index.shape == (n,),
            f"bond_index shape {bond_index.shape} != (N={n},)",
        )

    return DatasetBundle(
        params=params,
        slowness=slowness,
        gather=gather,
        mode_in_gather=mode_in_gather,
        freq=freq,
        param_names=param_names,
        mode_names=mode_names,
        schema_version=schema_version,
        geometry=geometry,
        attenuation=attenuation,
        layer_params=layer_params,
        layer_names=layer_names,
        bond_index=bond_index,
    )


def _check(condition: bool, message: str) -> None:
    """Raise :class:`SchemaError` with ``message`` unless ``condition``."""
    if not condition:
        raise SchemaError(message)
