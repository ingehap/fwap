"""
Low-latency LWD inverse net: a compact, depthwise-separable variant + benchmark.

Logging-while-drilling (LWD) acoustic inversion runs under a hard latency and
power budget -- the shear-slowness answer is wanted in real time from a downhole
tool, not in post-processing. This module packages a *compact* configuration of
the M3 :class:`~sonic_ml.models.inverse.InverseNet` (fewer, depthwise-separable
conv blocks and a narrower head) and a benchmark that **measures** the resulting
parameter-count / inference-latency / accuracy trade-off against the full net
rather than asserting it.

The compact net is a plain ``InverseNet`` built with ``separable=True``, so it
round-trips through :class:`~sonic_ml.models.inverse.TrainedInverseNet` and is
scored by the same :class:`~sonic_ml.bench.harness.Predictor` harness and
:class:`~sonic_ml.models.inverse_train.InversePredictor` adapter as every other
model -- no parallel class hierarchy. Because the net pools over the whole time
axis (``AdaptiveAvgPool1d``), it also accepts a *truncated* record, the other
lever for cutting while-drilling latency.

The hardware-independent win is the **parameter / FLOP** reduction -- what
translates to power and latency on a memory- and compute-bound downhole DSP.
Desktop-CPU wall-clock latency for models this small is a different quantity: it
is dominated by kernel-launch overhead and BLAS vectorization, so a
depthwise-separable net can even measure *slower* per forward pass than its
dense twin despite having fewer weights. :func:`latency_accuracy_report`
therefore reports both numbers side by side and lets the reader judge, rather
than pretending fewer parameters always means lower wall-clock latency.

Collar-mode contamination -- the other defining LWD problem -- is handled by the
phenomenological ``fwap.lwd`` processing layer (collar-notch filtering /
quadrupole stacking) and is orthogonal to the latency question addressed here;
pairing a collar-contamination augmentation with this compact net is future
work.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from sonic_ml.loader import DatasetBundle
from sonic_ml.models.augment import GatherAugmentation
from sonic_ml.models.inverse import InverseNet, TrainedInverseNet
from sonic_ml.models.inverse_dataset import normalize_gathers
from sonic_ml.models.inverse_train import parameter_mae, train_inverse
from sonic_ml.split import DataSplit

#: Compact conv-channel widths for the LWD variant (two blocks vs the full
#: net's three).
LWD_COMPACT_CHANNELS: tuple[int, ...] = (16, 32)
#: Compact conv kernel size.
LWD_COMPACT_KERNEL: int = 5
#: Compact post-pool MLP width.
LWD_COMPACT_HIDDEN: int = 32


def build_lwd_inverse_net(n_rec: int, n_targets: int) -> InverseNet:
    """
    Construct the compact LWD inverse net (a depthwise-separable ``InverseNet``).

    Parameters
    ----------
    n_rec : int
        Number of receivers (input channels).
    n_targets : int
        Number of active formation parameters to predict.

    Returns
    -------
    InverseNet
        A ``separable=True`` net with the compact channel / kernel / hidden
        preset. Uninitialised weights -- train it with :func:`train_lwd_inverse`
        (or wrap an already-trained one in ``TrainedInverseNet``).
    """
    return InverseNet(
        n_rec,
        n_targets,
        channels=LWD_COMPACT_CHANNELS,
        kernel=LWD_COMPACT_KERNEL,
        hidden=LWD_COMPACT_HIDDEN,
        dropout=0.0,
        separable=True,
    )


def train_lwd_inverse(
    bundle: DatasetBundle,
    *,
    split: DataSplit | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1.0e-3,
    augment: GatherAugmentation | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[TrainedInverseNet, list[float]]:
    """
    Train the compact LWD inverse net.

    Thin wrapper over :func:`~sonic_ml.models.inverse_train.train_inverse` that
    pins the low-latency preset (depthwise-separable blocks, compact channels /
    kernel / hidden). All other arguments behave exactly as in ``train_inverse``.

    Parameters
    ----------
    bundle : DatasetBundle
    split : DataSplit or None
        Train/val/test split; ``None`` builds a regime-stratified one.
    epochs, batch_size, lr : training hyperparameters.
    augment : GatherAugmentation or None, default None
        Optional training-time waveform augmentation.
    seed : int, default 0
    device : str, default "cpu"

    Returns
    -------
    trained : TrainedInverseNet
    history : list of float
        Mean training NLL per epoch.
    """
    return train_inverse(
        bundle,
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        channels=LWD_COMPACT_CHANNELS,
        kernel=LWD_COMPACT_KERNEL,
        hidden=LWD_COMPACT_HIDDEN,
        dropout=0.0,
        separable=True,
        augment=augment,
        seed=seed,
        device=device,
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters in ``model``."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def measure_latency_ms(
    trained: TrainedInverseNet,
    gather: np.ndarray,
    *,
    batch_size: int = 1,
    repeats: int = 20,
    warmup: int = 3,
) -> float:
    """
    Median wall-clock inference latency **per sample** (ms) on CPU.

    Runs the model's forward pass on a fixed batch ``repeats`` times after
    ``warmup`` untimed passes and returns the median, divided by the batch size.
    This is a *measurement helper*, not a CI assertion -- wall-clock budgets are
    machine-dependent and belong in a dedicated bench job, not the default test
    suite (mirroring the core repo's ``tests/test_bench.py`` policy).

    Parameters
    ----------
    trained : TrainedInverseNet
        The model to time (its ``.model`` forward pass).
    gather : ndarray, shape (N, R, T)
        Raw waveforms; the first ``batch_size`` rows are timed (amplitude-
        normalized internally, exactly as in inference).
    batch_size : int, default 1
        Records per forward pass. ``1`` reflects the streaming while-drilling
        case (one depth frame at a time).
    repeats : int, default 20
        Timed forward passes (median is reported).
    warmup : int, default 3
        Untimed passes first, to exclude one-off allocation costs.

    Returns
    -------
    float
        Median per-sample latency in milliseconds.
    """
    trained.model.eval()
    x = normalize_gathers(np.asarray(gather[:batch_size], dtype=float))
    xt = torch.as_tensor(x, dtype=torch.float32)
    for _ in range(max(warmup, 0)):
        trained.model(xt)
    samples = np.empty(max(repeats, 1), dtype=float)
    for i in range(samples.size):
        t0 = time.perf_counter()
        trained.model(xt)
        samples[i] = time.perf_counter() - t0
    per_sample = float(np.median(samples)) / max(batch_size, 1)
    return per_sample * 1.0e3


@dataclass(frozen=True)
class LatencyAccuracy:
    """
    One model's size / speed / accuracy row in an LWD trade-off report.

    Attributes
    ----------
    name : str
        Model label (e.g. ``"full"`` / ``"lwd_compact"``).
    n_params : int
        Trainable parameter count.
    latency_ms_per_sample : float
        Median CPU inference latency per sample (ms).
    vs_mae_m_s : float
        Mean absolute Vs error (m/s) on the scored indices.
    """

    name: str
    n_params: int
    latency_ms_per_sample: float
    vs_mae_m_s: float


def latency_accuracy_report(
    models: dict[str, TrainedInverseNet],
    bundle: DatasetBundle,
    indices: np.ndarray,
    *,
    batch_size: int = 1,
    repeats: int = 20,
) -> list[LatencyAccuracy]:
    """
    Build a size / latency / accuracy row per model on a held-out split.

    Parameters
    ----------
    models : dict of str to TrainedInverseNet
        Named trained models to compare, e.g.
        ``{"full": full_net, "lwd_compact": compact_net}``.
    bundle : DatasetBundle
        Dataset holding the truth (``params``) and inputs (``gather``).
    indices : ndarray of int
        Held-out sample indices to score / time on.
    batch_size, repeats : forwarded to :func:`measure_latency_ms`.

    Returns
    -------
    list of LatencyAccuracy
        One row per model, in input (insertion) order.
    """
    idx = np.asarray(indices, dtype=int)
    rows: list[LatencyAccuracy] = []
    for name, trained in models.items():
        rows.append(
            LatencyAccuracy(
                name=name,
                n_params=count_parameters(trained.model),
                latency_ms_per_sample=measure_latency_ms(
                    trained, bundle.gather[idx], batch_size=batch_size, repeats=repeats
                ),
                vs_mae_m_s=parameter_mae(trained, bundle, idx, "vs"),
            )
        )
    return rows


def format_latency_accuracy(rows: Sequence[LatencyAccuracy]) -> str:
    """
    Render a latency/accuracy trade-off report as a fixed-width table.

    Parameters
    ----------
    rows : sequence of LatencyAccuracy

    Returns
    -------
    str
        A multi-line table: one row per model with parameter count, per-sample
        latency (ms), and Vs MAE (m/s).
    """
    header = f"{'model':>12}  {'params':>9}  {'latency/ms':>11}  {'vs MAE m/s':>11}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.name:>12}  {row.n_params:>9,}  "
            f"{row.latency_ms_per_sample:>11.3f}  {row.vs_mae_m_s:>11.1f}"
        )
    return "\n".join(lines)
