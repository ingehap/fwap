"""
Best-effort determinism for reproducible training.

CPU PyTorch is not bit-reproducible by default (thread-dependent reduction
order), so bit-identical training requires seeding every RNG *and* constraining
threading and algorithm selection. These helpers centralize that so a training
script is one :func:`seed_everything` + :func:`pin_threads` call away from
reproducibility.

torch is an optional import here: the pure-NumPy sonic_ml spine (loader, split,
normalize, mask, provenance) must be usable and testable without torch
installed, so the torch-specific steps are skipped when it is absent. Torch is
imported lazily inside each function (guarded by :data:`_HAS_TORCH`) so this
module type-checks and imports cleanly whether or not torch is present.
"""

from __future__ import annotations

import importlib.util
import os
import random

import numpy as np

#: Whether torch is importable in this environment (a full sonic_ml install).
_HAS_TORCH = importlib.util.find_spec("torch") is not None


def torch_available() -> bool:
    """Return whether torch could be imported (True in a full install)."""
    return _HAS_TORCH


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy and (if present) torch RNGs.

    Parameters
    ----------
    seed : int
        The seed applied to ``random``, ``numpy.random`` (legacy global),
        ``PYTHONHASHSEED`` and torch (CPU + any CUDA devices). torch is also
        switched to deterministic algorithms (``warn_only`` so an op without a
        deterministic implementation warns rather than raises).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def pin_threads(n: int = 1) -> None:
    """
    Pin CPU threading so reductions are order-stable.

    Parameters
    ----------
    n : int, default 1
        Thread count for OpenMP/MKL and (if present) torch. ``1`` is the
        safest choice for bit-reproducibility at some throughput cost.
    """
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    if _HAS_TORCH:
        import torch

        torch.set_num_threads(n)


def worker_init_fn(worker_id: int) -> None:
    """
    Deterministic per-worker seeding for a torch ``DataLoader``.

    Pass as ``DataLoader(worker_init_fn=...)`` so each worker derives a
    distinct-but-reproducible seed from the base torch seed.

    Parameters
    ----------
    worker_id : int
        The DataLoader-provided worker index.
    """
    if _HAS_TORCH:
        import torch

        base = int(torch.initial_seed()) % (2**32)
    else:
        base = 0
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
