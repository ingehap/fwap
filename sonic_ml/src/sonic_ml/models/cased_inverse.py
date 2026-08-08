"""
Cement-bond inverse: cased waveform gather -> (behind-casing Vs, bond index).

The M5 headline. Where M3 inverts an *open-hole* gather for formation
parameters, this inverts a *cased-hole* gather for the two quantities a cement
evaluation job actually wants: how well the cement is bonded, and what the
formation behind the casing is doing.

It deliberately reuses the M3 machinery rather than forking it -- the network
(:class:`~sonic_ml.models.inverse.InverseNet`, a 1-D CNN over receivers with a
heteroscedastic mean/log-variance head) and its trained wrapper are unchanged.
Only the *targets* differ, so :func:`cased_targets` assembles them and a
:class:`~sonic_ml.normalize.Standardizer` handles the rest.

Why the uncertainty head earns its keep here
--------------------------------------------
A forward sensitivity sweep over the M5a priors shows the two targets are *not*
equally identifiable from a single-mode cased Stoneley gather: moving the cement
shear velocity across its prior shifts the dispersion curve by ~7%, while moving
the formation shear velocity across its prior shifts it by only ~1.5% -- less
than the ~2% contributed by the nuisance cement-thickness variation. So bond
quality is genuinely recoverable and behind-casing Vs is genuinely weak, and an
honest model has to *say so* rather than emit a confident number. That is what
the per-target ``sigma`` is for, and
:func:`~sonic_ml.models.inverse_train.residual_zscore_std` is what checks it is
calibrated rather than decorative.

Scope
-----
The M5a dataset spans the **bonded** regime only (stiff cement, where the cased
Stoneley mode stays bound), so this inverts *graded bond quality*. It is not a
free-pipe detector -- the fully-debonded casing-ring signature is outside the
bound-mode dataset entirely. Any claim about detecting a free pipe would be
unsupported by this training data.
"""

from __future__ import annotations

import numpy as np
import torch
from fwap import ArrayGeometry
from torch.utils.data import DataLoader, Dataset

from sonic_ml.determinism import seed_everything
from sonic_ml.loader import DatasetBundle
from sonic_ml.models.augment import GatherAugmentation
from sonic_ml.models.inverse import InverseNet, TrainedInverseNet
from sonic_ml.models.inverse_dataset import normalize_gathers
from sonic_ml.models.losses import gaussian_nll
from sonic_ml.normalize import Standardizer
from sonic_ml.split import DataSplit, stratified_split

#: Default inversion targets: the cement-evaluation deliverables.
DEFAULT_CASED_TARGETS: tuple[str, ...] = ("vs", "bond_index")


def cased_targets(
    bundle: DatasetBundle,
    names: tuple[str, ...] = DEFAULT_CASED_TARGETS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    Assemble the inverse-net target matrix from a cased-hole bundle.

    Parameters
    ----------
    bundle : DatasetBundle
        A cased-hole dataset (schema v4).
    names : tuple of str, default :data:`DEFAULT_CASED_TARGETS`
        Target column names. Each is either ``"bond_index"`` or a formation
        parameter present in ``bundle.param_names``.

    Returns
    -------
    targets : ndarray, shape (N, K), float64
    names : tuple of str, length K
        Echoed back, so callers can pair the matrix with its labels.

    Raises
    ------
    ValueError
        If the bundle is not cased, or a requested name is unknown.
    """
    if not bundle.is_cased or bundle.bond_index is None:
        raise ValueError(
            "cased_targets requires a cased-hole dataset (schema v4 with a "
            "non-empty layer stack); got an open-hole bundle"
        )
    columns: list[np.ndarray] = []
    for name in names:
        if name == "bond_index":
            columns.append(np.asarray(bundle.bond_index, dtype=float))
        elif name in bundle.param_names:
            columns.append(bundle.param(name))
        else:
            raise ValueError(
                f"unknown cased target {name!r}; expected 'bond_index' or one "
                f"of {bundle.param_names}"
            )
    return np.stack(columns, axis=1), tuple(names)


class CasedInverseDataset(Dataset):
    """
    Tensor view of a cased-hole bundle for the bond inverse.

    Each item is ``(x, y)``:

    * ``x`` -- per-gather amplitude-normalized waveforms, shape ``(R, T)``.
    * ``y`` -- standardized active targets, shape ``(n_active,)``.

    The gather is the **only** input: the dispersion label and the annulus
    properties never enter, so the network cannot shortcut the inversion.

    Parameters
    ----------
    bundle : DatasetBundle
    indices : ndarray of int
        Sample indices to include.
    target_std : Standardizer
        Fitted on the *training* targets.
    target_names : tuple of str
        Target columns to assemble (must match ``target_std.names``).
    augment : GatherAugmentation or None
        Optional training-time waveform augmentation (applied per access).
    augment_seed : int, default 0
        Seed for the augmentation RNG.
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray,
        target_std: Standardizer,
        target_names: tuple[str, ...] = DEFAULT_CASED_TARGETS,
        *,
        augment: GatherAugmentation | None = None,
        augment_seed: int = 0,
    ) -> None:
        idx = np.asarray(indices, dtype=int)
        targets, _ = cased_targets(bundle, target_names)
        self._raw_gather = np.asarray(bundle.gather[idx], dtype=float)
        self.x = torch.as_tensor(
            normalize_gathers(self._raw_gather), dtype=torch.float32
        )
        self.y = torch.as_tensor(
            target_std.transform(targets[idx]), dtype=torch.float32
        )
        self._augment = augment
        self._rng = np.random.default_rng(augment_seed)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._augment is None:
            return self.x[index], self.y[index]
        noisy = self._augment.apply(self._raw_gather[index], self._rng)
        x = normalize_gathers(noisy[None, ...])[0]
        return torch.as_tensor(x, dtype=torch.float32), self.y[index]


def train_cased_inverse(
    bundle: DatasetBundle,
    *,
    split: DataSplit | None = None,
    target_names: tuple[str, ...] = DEFAULT_CASED_TARGETS,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1.0e-3,
    channels: tuple[int, ...] = (32, 64, 128),
    kernel: int = 7,
    hidden: int = 128,
    dropout: float = 0.0,
    separable: bool = False,
    augment: GatherAugmentation | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[TrainedInverseNet, list[float]]:
    """
    Train the cement-bond inverse on a cased-hole bundle.

    Parameters
    ----------
    bundle : DatasetBundle
        A cased-hole dataset (schema v4).
    split : DataSplit or None
        Train/val/test split; ``None`` builds a regime-stratified one. Cased
        datasets are fast-formation only, so that reduces to a reproducible
        random split.
    target_names : tuple of str, default :data:`DEFAULT_CASED_TARGETS`
        Quantities to invert for.
    epochs, batch_size, lr, channels, kernel, hidden, dropout, separable :
        Network and optimization hyperparameters (see
        :func:`~sonic_ml.models.inverse_train.train_inverse`).
    augment : GatherAugmentation or None
        Training-time waveform augmentation, applied to the training split only.
    seed : int, default 0
    device : str, default "cpu"

    Returns
    -------
    trained : TrainedInverseNet
        Its ``active_names`` are the (non-constant) target columns.
    history : list of float
        Mean training NLL per epoch.
    """
    seed_everything(seed)
    if split is None:
        split = stratified_split(bundle, seed=seed)

    targets, names = cased_targets(bundle, target_names)
    target_std = Standardizer.fit(targets[split.train], names)
    train_ds = CasedInverseDataset(
        bundle,
        split.train,
        target_std,
        target_names,
        augment=augment,
        augment_seed=seed,
    )

    dev = torch.device(device)
    model = InverseNet(
        int(bundle.gather.shape[1]),
        target_std.n_active,
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

    return TrainedInverseNet(model.eval(), target_std), history


class NeuralBondPredictor:
    """
    Adapt a trained cased inverse to the :mod:`sonic_ml.bench.bond` protocol.

    Lets the learned inverse be scored by the *same* harness as
    :class:`~sonic_ml.baselines.bond.StoneleyBondBaseline`. Geometry is accepted
    for protocol compatibility but unused (the net reads the gather only).

    Parameters
    ----------
    trained : TrainedInverseNet
        A model whose ``active_names`` include ``"bond_index"``.
    name : str
        Predictor name shown on the scorecard.

    Raises
    ------
    ValueError
        If the model does not predict ``bond_index``.
    """

    def __init__(
        self, trained: TrainedInverseNet, *, name: str = "cased_inverse_net"
    ) -> None:
        if "bond_index" not in trained.active_names:
            raise ValueError(
                "NeuralBondPredictor needs a model predicting 'bond_index'; "
                f"got targets {trained.active_names}"
            )
        self.trained = trained
        self.name = name
        self._bond_idx = trained.active_names.index("bond_index")

    def predict_bond(
        self,
        bundle: DatasetBundle,
        geom: ArrayGeometry,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Predict the bond index per index, clipped to ``[0, 1]``."""
        idx = np.asarray(indices, dtype=int)
        params, _ = self.trained.predict(bundle.gather[idx])
        return np.clip(params[:, self._bond_idx], 0.0, 1.0)

    def predict_bond_uncertainty(
        self, bundle: DatasetBundle, indices: np.ndarray
    ) -> np.ndarray:
        """Predicted bond-index standard deviation per index (raw units)."""
        idx = np.asarray(indices, dtype=int)
        _, std = self.trained.predict(bundle.gather[idx])
        return std[:, self._bond_idx]


def cased_target_mae(
    trained: TrainedInverseNet,
    bundle: DatasetBundle,
    indices: np.ndarray,
    name: str,
    target_names: tuple[str, ...] = DEFAULT_CASED_TARGETS,
) -> float:
    """
    Mean absolute error of one named cased target (raw units).

    Parameters
    ----------
    trained : TrainedInverseNet
    bundle : DatasetBundle
    indices : ndarray of int
        Samples to score.
    name : str
        Target column, e.g. ``"bond_index"`` or ``"vs"``.
    target_names : tuple of str
        The target set the model was trained on.

    Returns
    -------
    float
        MAE in the target's own units.
    """
    idx = np.asarray(indices, dtype=int)
    targets, names = cased_targets(bundle, target_names)
    pred, _ = trained.predict(bundle.gather[idx])
    col = trained.active_names.index(name)
    truth = targets[idx][:, names.index(name)]
    return float(np.mean(np.abs(pred[:, col] - truth)))
