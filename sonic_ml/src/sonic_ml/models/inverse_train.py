"""
Training loop, metrics, harness adapter, and CLI for the inverse net.

``InversePredictor`` adapts a trained net to the M1
:class:`~sonic_ml.bench.harness.Predictor` protocol, so the inverse net is
scored by the *same* harness as the classical baselines -- the head-to-head
comparison that justifies the learned inverse.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import torch
from fwap import ArrayGeometry
from torch.utils.data import DataLoader

from sonic_ml.determinism import seed_everything
from sonic_ml.loader import DatasetBundle, load_npz
from sonic_ml.models.augment import Augmentation
from sonic_ml.models.inverse import InverseNet, TrainedInverseNet
from sonic_ml.models.inverse_dataset import InverseDataset
from sonic_ml.models.losses import gaussian_nll
from sonic_ml.normalize import Standardizer
from sonic_ml.split import DataSplit, stratified_split


def train_inverse(
    bundle: DatasetBundle,
    *,
    split: DataSplit | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1.0e-3,
    channels: tuple[int, ...] = (32, 64, 128),
    kernel: int = 7,
    hidden: int = 128,
    dropout: float = 0.0,
    separable: bool = False,
    augment: Augmentation | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[TrainedInverseNet, list[float]]:
    """
    Train the inverse net (gather -> parameters) with a Gaussian NLL.

    Parameters
    ----------
    bundle : DatasetBundle
    split : DataSplit or None
        Train/val/test split; ``None`` builds a regime-stratified one.
    epochs, batch_size, lr, channels, kernel, hidden, dropout : hyperparameters
    separable : bool, default False
        Use depthwise-separable conv blocks (the low-latency LWD variant; see
        :mod:`sonic_ml.models.lwd`).
    augment : Augmentation or None, default None
        Training-time waveform augmentation (SNR sweep / amplitude jitter) for
        sim-to-real robustness. Applied to the *training* split only.
    seed : int, default 0
        Master seed.
    device : str, default "cpu"

    Returns
    -------
    trained : TrainedInverseNet
    history : list of float
        Mean training NLL per epoch.
    """
    seed_everything(seed)
    if split is None:
        split = stratified_split(bundle, seed=seed)

    param_std = Standardizer.fit(bundle.params[split.train], bundle.param_names)
    train_ds = InverseDataset(
        bundle, split.train, param_std, augment=augment, augment_seed=seed
    )

    dev = torch.device(device)
    model = InverseNet(
        int(bundle.gather.shape[1]),
        param_std.n_active,
        channels=channels,
        kernel=kernel,
        hidden=hidden,
        dropout=dropout,
        separable=separable,
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
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            mean, log_var = model(x)
            loss = gaussian_nll(mean, log_var, y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        history.append(total / max(n_batches, 1))

    return TrainedInverseNet(model.eval(), param_std), history


class InversePredictor:
    """
    Adapt a :class:`TrainedInverseNet` to the M1 ``Predictor`` protocol.

    Lets the inverse net be scored by :func:`sonic_ml.bench.evaluate` alongside
    the classical baselines. The gather is the only input -- geometry is
    accepted for protocol compatibility but unused (offsets are fixed per
    dataset and learned implicitly).
    """

    name = "inverse_net"

    def __init__(self, trained: TrainedInverseNet) -> None:
        self.trained = trained
        self._vs_idx = trained.active_names.index("vs")

    def predict_vs(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Predict Vs (m/s) per index from the gather."""
        idx = np.asarray(indices, dtype=int)
        params, _ = self.trained.predict(bundle.gather[idx])
        return params[:, self._vs_idx]


def parameter_mae(
    trained: TrainedInverseNet,
    bundle: DatasetBundle,
    indices: np.ndarray,
    name: str,
) -> float:
    """Mean absolute error (raw units) of a named active parameter."""
    idx = np.asarray(indices, dtype=int)
    params, _ = trained.predict(bundle.gather[idx])
    col = trained.active_names.index(name)
    true = bundle.param(name)[idx]
    return float(np.mean(np.abs(params[:, col] - true)))


def residual_zscore_std(
    trained: TrainedInverseNet,
    bundle: DatasetBundle,
    indices: np.ndarray,
    name: str,
) -> float:
    """
    Standard deviation of the standardized residuals for a parameter.

    ``(pred - true) / pred_std`` should be ~unit-variance if the predicted
    heteroscedastic uncertainty is calibrated; values >> 1 mean the model is
    over-confident, << 1 under-confident.
    """
    idx = np.asarray(indices, dtype=int)
    params, std = trained.predict(bundle.gather[idx])
    col = trained.active_names.index(name)
    true = bundle.param(name)[idx]
    z = (params[:, col] - true) / np.maximum(std[:, col], 1e-12)
    return float(np.std(z))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the DL-FWI inverse net on a surrogate .npz."
    )
    parser.add_argument("--npz", required=True, help="dataset .npz path")
    parser.add_argument("--out", default="inverse_net.pt", help="checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: train and checkpoint an inverse net."""
    args = _build_parser().parse_args(argv)
    bundle = load_npz(args.npz)
    split = stratified_split(bundle, seed=args.seed)
    trained, history = train_inverse(
        bundle,
        split=split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
    )
    trained.save(args.out)
    val = split.val if split.val.size else split.train
    print(f"trained {args.epochs} epochs; final train NLL {history[-1]:.4f}")
    for name in trained.active_names:
        print(f"  val MAE {name:>4} = {parameter_mae(trained, bundle, val, name):.3g}")
    print(f"  val Vs z-std = {residual_zscore_std(trained, bundle, val, 'vs'):.2f}")
    print(f"  checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
