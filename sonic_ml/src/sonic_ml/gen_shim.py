"""
Thin bridge to the core fwap surrogate-dataset generator.

``scripts/gen_surrogate_dataset.py`` is a repository *script*, not part of the
installed ``fwap`` package, so it cannot be imported normally. This module
path-imports it once (mirroring ``tests/test_gen_surrogate_dataset.py`` in the
core repo) and exposes thin wrappers so sonic_ml code can build datasets with
custom modes / geometry / priors without re-implementing the discovery dance.

Because ``sonic_ml`` is an in-repo sibling of ``fwap``, the generator is
located by walking up from the installed fwap package to the repository root;
an explicit path or the ``SONIC_ML_GENERATOR`` environment variable overrides
the search.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import fwap

_ENV_VAR = "SONIC_ML_GENERATOR"
_SCRIPT_RELPATH = ("scripts", "gen_surrogate_dataset.py")


def _default_generator_path() -> Path:
    """
    Locate ``scripts/gen_surrogate_dataset.py`` in the fwap repository.

    Resolution order: the ``SONIC_ML_GENERATOR`` environment variable, then a
    walk up from the installed ``fwap`` package (the editable install keeps
    ``fwap.__file__`` inside the repo checkout).

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        If the generator script cannot be found.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{_ENV_VAR}={override!r} is not a file")
        return p

    # fwap/__init__.py -> parents[1] is the repo root of an editable checkout.
    fwap_file = fwap.__file__
    if fwap_file is None:  # pragma: no cover - a namespace-only fwap is unexpected
        raise FileNotFoundError(
            "fwap has no __file__ to locate scripts/ from; set "
            f"{_ENV_VAR} to the generator path"
        )
    fwap_pkg = Path(fwap_file).resolve()
    for base in fwap_pkg.parents:
        candidate = base.joinpath(*_SCRIPT_RELPATH)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "could not locate scripts/gen_surrogate_dataset.py; set the "
        f"{_ENV_VAR} environment variable to its path"
    )


@cache
def generator(path: str | None = None) -> ModuleType:
    """
    Path-import the generator module (cached).

    Parameters
    ----------
    path : str or None
        Explicit path to ``gen_surrogate_dataset.py``. ``None`` uses
        :func:`_default_generator_path`.

    Returns
    -------
    module
        The imported ``gen_surrogate_dataset`` module, exposing
        ``FormationPriors``, ``ModeSpec``, ``DEFAULT_MODES``,
        ``default_freq_grid``, ``generate_dataset``, ``save_npz``,
        ``PARAM_NAMES`` and ``SCHEMA_VERSION``.
    """
    script = Path(path).resolve() if path is not None else _default_generator_path()
    spec = importlib.util.spec_from_file_location("gen_surrogate_dataset", script)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not create import spec for {script}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses resolve the module by name under
    # ``from __future__ import annotations``.
    import sys

    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def generate(n: int, **kwargs: Any) -> list[Any]:
    """
    Generate ``n`` accepted samples via the core generator.

    Parameters
    ----------
    n : int
        Number of samples to accept.
    **kwargs
        Forwarded verbatim to ``gen_surrogate_dataset.generate_dataset``
        (e.g. ``seed``, ``geom``, ``freq``, ``priors``, ``modes``,
        ``noise_max``, ``min_finite``, ``max_attempts``).

    Returns
    -------
    list
        ``SurrogateSample`` instances.
    """
    return list(generator().generate_dataset(n, **kwargs))


def save(path: str, samples: Sequence[Any]) -> None:
    """Write ``samples`` to a ``.npz`` via ``gen_surrogate_dataset.save_npz``."""
    generator().save_npz(path, samples)


def build_npz(path: str, n: int, **kwargs: Any) -> None:
    """
    Generate a dataset and write it to ``path`` in one call.

    Convenience for tests and small runs: ``generate(n, **kwargs)`` followed by
    :func:`save`.
    """
    save(path, generate(n, **kwargs))
