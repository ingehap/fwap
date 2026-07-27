"""
Training loop, metrics, and CLI for the forward dispersion surrogate.

Fits parameter/slowness normalizers on the training split, trains the residual
MLP with a masked slowness loss + presence BCE, and returns a
:class:`~sonic_ml.models.forward.TrainedForwardSurrogate`. Determinism is
engaged up front so a run is reproducible from ``(seed, config, split)``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from sonic_ml.determinism import seed_everything
from sonic_ml.loader import DatasetBundle, load_npz
from sonic_ml.models.dataset import ForwardDataset, SlownessNormalizer
from sonic_ml.models.forward import ForwardSurrogate, TrainedForwardSurrogate
from sonic_ml.models.losses import masked_slowness_loss, presence_bce
from sonic_ml.normalize import Standardizer
from sonic_ml.split import DataSplit, stratified_split


def train_forward(
    bundle: DatasetBundle,
    *,
    split: DataSplit | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1.0e-3,
    width: int = 256,
    depth: int = 4,
    dropout: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[TrainedForwardSurrogate, list[float]]:
    """
    Train a forward surrogate on a dataset.

    Parameters
    ----------
    bundle : DatasetBundle
    split : DataSplit or None
        Train/val/test split; ``None`` builds a regime-stratified one.
    epochs, batch_size, lr, width, depth, dropout : int / float
        Training and architecture hyperparameters.
    seed : int, default 0
        Master seed (RNGs + weight init + shuffling).
    device : str, default "cpu"
        Torch device.

    Returns
    -------
    trained : TrainedForwardSurrogate
    history : list of float
        Mean training loss per epoch.
    """
    seed_everything(seed)
    if split is None:
        split = stratified_split(bundle, seed=seed)

    param_std = Standardizer.fit(bundle.params[split.train], bundle.param_names)
    slowness_norm = SlownessNormalizer.fit(bundle.slowness[split.train])
    train_ds = ForwardDataset(bundle, split.train, param_std, slowness_norm)

    dev = torch.device(device)
    model = ForwardSurrogate(
        param_std.n_active,
        bundle.n_modes,
        bundle.n_freq,
        width=width,
        depth=depth,
        dropout=dropout,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    history: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for x, target, mask, presence in loader:
            x, target, mask, presence = (
                x.to(dev),
                target.to(dev),
                mask.to(dev),
                presence.to(dev),
            )
            optimizer.zero_grad()
            slowness, presence_logits = model(x)
            loss = masked_slowness_loss(slowness, target, mask) + presence_bce(
                presence_logits, presence
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        history.append(total / max(n_batches, 1))

    trained = TrainedForwardSurrogate(
        model.eval(), param_std, slowness_norm, bundle.mode_names
    )
    return trained, history


def slowness_rmse(
    trained: TrainedForwardSurrogate,
    bundle: DatasetBundle,
    indices: np.ndarray,
) -> float:
    """
    Masked RMSE of predicted vs true slowness (raw units, s/m).

    Computed over finite target entries only.
    """
    idx = np.asarray(indices, dtype=int)
    pred, _ = trained.predict(bundle.params[idx])
    target = bundle.slowness[idx]
    mask = np.isfinite(target)
    if not mask.any():
        return float("nan")
    err = pred[mask] - target[mask]
    return float(np.sqrt(np.mean(err**2)))


def presence_auc(
    trained: TrainedForwardSurrogate,
    bundle: DatasetBundle,
    indices: np.ndarray,
) -> float:
    """
    ROC-AUC of the presence head against ``mode_in_gather`` (all modes pooled).

    Returns ``nan`` if a class is absent (AUC undefined).
    """
    idx = np.asarray(indices, dtype=int)
    _, presence = trained.predict(bundle.params[idx])
    labels = bundle.mode_in_gather[idx].astype(int).ravel()
    scores = presence.ravel()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the forward dispersion surrogate on a surrogate .npz."
    )
    parser.add_argument("--npz", required=True, help="dataset .npz path")
    parser.add_argument("--out", default="forward_surrogate.pt", help="checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: train and checkpoint a forward surrogate."""
    args = _build_parser().parse_args(argv)
    bundle = load_npz(args.npz)
    split = stratified_split(bundle, seed=args.seed)
    trained, history = train_forward(
        bundle,
        split=split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
    )
    trained.save(args.out)
    val = split.val if split.val.size else split.train
    print(f"trained {args.epochs} epochs; final train loss {history[-1]:.4f}")
    print(f"  val slowness RMSE = {slowness_rmse(trained, bundle, val):.3e} s/m")
    print(f"  val presence AUC  = {presence_auc(trained, bundle, val):.3f}")
    print(f"  checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
