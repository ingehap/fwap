"""
Learned microannulus-thickness inverse for the debonded regime (G.2).

What this model is asked, and why that is the honest question
-------------------------------------------------------------
Not "predict the gap from the waveform". Measured, the cased Stoneley mode --
the only debonded branch that reaches the receivers at all -- moves 0.05 %
when the gap goes from 10 um to 1 mm, against 1.0-1.5 % for the formation
shear velocity alone. There is no gap information in the gather to learn, and
a model trained to find some would be fitting noise. The crack wave carries
the width, and it arrives 4.8-47.6 ms out on a 5.12 ms record, so it is a
*dispersion-curve* product rather than a waveform one.

So the input is the crack-wave curve, and the competition is against
:class:`~sonic_ml.baselines.debond.CrackWaveThicknessBaseline`, which inverts
the Krauklis law in closed form and reaches ~18 % on that curve with no
fitting at all.

Residual learning, so the win has a physical name
-------------------------------------------------
The model predicts the **residual** ``log10(h_true) - log10(h_krauklis)``
rather than the thickness itself. Two reasons, and both are about being able
to say what was learned:

* The Krauklis law assumes half-space walls. The dataset has ~10 mm of casing
  and ~45 mm of cement against a crack wavelength of the same order, so the
  residual *is* the finite-layer correction. A model that beats the baseline
  has learned that correction and nothing more mysterious.
* It cannot win by accident. With a zero output the model reproduces the
  classical baseline exactly, so any improvement is attributable and any
  regression is visible immediately.

The features give it strictly more than the baseline has: the baseline uses
the bounding moduli only, through the compliance ``C``, and never sees the
layer *thicknesses* -- which is precisely what the half-space assumption
throws away. That asymmetry is the intended explanation for a win, and
:func:`debond_features` is written to make it checkable.

**No leakage.** The gap thickness sits in ``layer_params`` next to the layers
the features do use, so :func:`debond_features` drops that column explicitly
and ``tests/test_models_debond.py`` pins it: perturbing the stored gap while
holding the dispersion curve fixed must not move a single feature.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sonic_ml.baselines.debond import (
    CRACK_WAVE_MODE_NAME,
    GAP_LAYER_NAME,
    crack_compliance,
    krauklis_thickness,
)
from sonic_ml.determinism import seed_everything
from sonic_ml.loader import LAYER_PARAM_NAMES, DatasetBundle
from sonic_ml.normalize import Standardizer
from sonic_ml.split import DataSplit, stratified_split

#: Names of the features :func:`debond_features` builds, in column order.
#: The three curve summaries come first because they are what the baseline
#: also sees; everything after them is information the baseline does not have.
DEBOND_FEATURE_NAMES: tuple[str, ...] = (
    "log10_krauklis_h",
    "krauklis_log_slope",
    "krauklis_log_iqr",
)


def _krauklis_per_frequency(bundle: DatasetBundle, index: int, row: int) -> np.ndarray:
    """Per-frequency Krauklis thickness estimates for one sample."""
    if bundle.layer_params is None:  # pragma: no cover - guarded by callers
        raise ValueError("bundle carries no layer stack")
    compliance = crack_compliance(bundle.layer_params[index], bundle.layer_names)
    rho_f = float(bundle.params[index, bundle.param_names.index("rho_f")])
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = 1.0 / np.asarray(bundle.slowness[index, row], dtype=float)
    return krauklis_thickness(bundle.freq, velocity, rho_f=rho_f, compliance=compliance)


def debond_features(
    bundle: DatasetBundle,
    *,
    mode_name: str = CRACK_WAVE_MODE_NAME,
    gap_name: str = GAP_LAYER_NAME,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """
    Build the inverse's feature matrix, its column names, and the baseline.

    Columns, in order:

    1. ``log10_krauklis_h`` -- the closed-form baseline estimate, in log10 m.
    2. ``krauklis_log_slope`` -- least-squares slope of ``log10(h_est)``
       against ``log10(freq)``. A true Krauklis crack gives zero at every
       frequency; a finite-layer one drifts, so this is a direct read-out of
       how badly the half-space assumption fits *this* sample.
    3. ``krauklis_log_iqr`` -- the interquartile spread of the same estimates.
    4. Every layer property **except the gap thickness**, plus the formation
       parameters.

    Parameters
    ----------
    bundle : DatasetBundle
        A debonded dataset carrying ``mode_name`` and a ``gap_name`` layer.
    mode_name : str, default ``"crack_wave"``
    gap_name : str, default ``"microannulus"``

    Returns
    -------
    features : ndarray, shape (N, P)
    names : tuple of str, length P
    baseline : ndarray, shape (N,)
        ``log10`` of the closed-form estimate, i.e. column 1 -- returned
        separately because it is the residual's origin, not just a feature.

    Raises
    ------
    ValueError
        If the bundle carries no such mode, or no layer stack.
    """
    if mode_name not in bundle.mode_names:
        raise ValueError(
            f"bundle has no {mode_name!r} mode; it carries {bundle.mode_names!r}"
        )
    if bundle.layer_params is None:
        raise ValueError("bundle carries no layer stack")

    row = bundle.mode_names.index(mode_name)
    gap_index = bundle.layer_names.index(gap_name)
    thickness_col = LAYER_PARAM_NAMES.index("thickness")
    n = bundle.n_samples

    summaries = np.zeros((n, 3), dtype=float)
    log_freq = np.log10(np.asarray(bundle.freq, dtype=float))
    for i in range(n):
        estimates = _krauklis_per_frequency(bundle, i, row)
        ok = np.isfinite(estimates) & (estimates > 0.0)
        if not ok.any():
            summaries[i] = (np.nan, 0.0, 0.0)
            continue
        log_h = np.log10(estimates[ok])
        summaries[i, 0] = float(np.median(log_h))
        summaries[i, 1] = (
            float(np.polyfit(log_freq[ok], log_h, 1)[0]) if ok.sum() > 1 else 0.0
        )
        summaries[i, 2] = float(np.percentile(log_h, 75) - np.percentile(log_h, 25))

    # Every layer column except the one that holds the answer.
    layer_cols: list[np.ndarray] = []
    layer_names: list[str] = []
    for layer_index, layer in enumerate(bundle.layer_names):
        for prop_index, prop in enumerate(LAYER_PARAM_NAMES):
            if layer_index == gap_index and prop_index == thickness_col:
                continue  # the target; see the module docstring
            layer_cols.append(bundle.layer_params[:, layer_index, prop_index])
            layer_names.append(f"{layer}_{prop}")

    features = np.concatenate(
        [
            summaries,
            np.stack(layer_cols, axis=1),
            np.asarray(bundle.params, dtype=float),
        ],
        axis=1,
    )
    names = DEBOND_FEATURE_NAMES + tuple(layer_names) + tuple(bundle.param_names)
    return features, names, summaries[:, 0].copy()


def debond_targets(
    bundle: DatasetBundle, *, gap_name: str = GAP_LAYER_NAME
) -> np.ndarray:
    """
    ``log10`` of the drawn gap width (m), the quantity being inverted for.

    Parameters
    ----------
    bundle : DatasetBundle
    gap_name : str, default ``"microannulus"``

    Returns
    -------
    ndarray, shape (N,)
    """
    if bundle.layer_params is None:
        raise ValueError("bundle carries no layer stack")
    index = bundle.layer_names.index(gap_name)
    thickness = LAYER_PARAM_NAMES.index("thickness")
    return np.log10(np.asarray(bundle.layer_params[:, index, thickness], dtype=float))


class DebondResidualNet(nn.Module):
    """
    Small MLP predicting the log-residual of the closed-form estimate.

    Deliberately small. The quantity being learned is a smooth geometric
    correction in a handful of parameters, not a waveform, and the dataset
    costs ~14 s a sample -- so capacity is the wrong thing to spend here.

    Parameters
    ----------
    n_features : int
        Input width from :func:`debond_features`.
    hidden : tuple of int, default ``(32, 32)``
        Hidden layer widths.
    """

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (32, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SiLU()]
            width = size
        # Held as a typed local: reading ``layers[-1].weight`` off a
        # ``list[nn.Module]`` gives ``Tensor | Module``, which ``zeros_``
        # rejects.
        head = nn.Linear(width, 1)
        # Start at exactly the classical baseline: a zero residual reproduces
        # it, so training can only move away from a known-good answer.
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted log-residual, shape ``(n, 1)``."""
        return self.net(x)


@dataclass
class TrainedDebondInverse:
    """
    A trained residual inverse plus the standardizer its features expect.

    Attributes
    ----------
    model : DebondResidualNet
    feature_std : Standardizer
        Fitted on the training split; drops constant columns.
    feature_names : tuple of str
    """

    model: DebondResidualNet
    feature_std: Standardizer
    feature_names: tuple[str, ...]

    def predict_log_thickness(self, bundle: DatasetBundle) -> np.ndarray:
        """
        Predicted ``log10`` gap width (m) for every sample.

        Returns
        -------
        ndarray, shape (N,)
        """
        features, _, baseline = debond_features(bundle)
        x = torch.as_tensor(
            self.feature_std.transform(np.nan_to_num(features)), dtype=torch.float32
        )
        self.model.eval()
        with torch.no_grad():
            residual = self.model(x).squeeze(-1).numpy()
        return baseline + residual

    def predict_thickness(self, bundle: DatasetBundle) -> np.ndarray:
        """Predicted gap width in metres."""
        return 10.0 ** self.predict_log_thickness(bundle)


def train_debond_inverse(
    bundle: DatasetBundle,
    *,
    split: DataSplit | None = None,
    epochs: int = 400,
    batch_size: int = 32,
    lr: float = 3.0e-3,
    weight_decay: float = 1.0e-4,
    hidden: tuple[int, ...] = (32, 32),
    seed: int = 0,
) -> tuple[TrainedDebondInverse, list[tuple[float, float]]]:
    """
    Train the residual gap-width inverse.

    Parameters
    ----------
    bundle : DatasetBundle
        A debonded dataset (``gen_surrogate_dataset.py --debonded``).
    split : DataSplit or None
        Train/val/test split; ``None`` builds a reproducible stratified one.
    epochs, batch_size, lr, weight_decay, hidden : hyperparameters.
    seed : int, default 0

    The validation split selects the returned weights. That is not a
    formality here: the feature count is comparable to the sample count a
    debonded dataset can afford (~14 s a sample), so the training loss reaches
    zero long before the model generalises, and the last epoch is the wrong
    one to keep.

    Returns
    -------
    trained : TrainedDebondInverse
        Carrying the weights that scored best on the validation split.
    history : list of (float, float)
        ``(train_mse, val_mse)`` per epoch, in log10-thickness units.
    """
    seed_everything(seed)
    if split is None:
        split = stratified_split(bundle, seed=seed)

    features, names, baseline = debond_features(bundle)
    targets = debond_targets(bundle)
    residual = targets - baseline

    features = np.nan_to_num(features)
    feature_std = Standardizer.fit(features[split.train], names)
    x_train = torch.as_tensor(
        feature_std.transform(features[split.train]), dtype=torch.float32
    )
    y_train = torch.as_tensor(residual[split.train], dtype=torch.float32)[:, None]

    model = DebondResidualNet(x_train.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    x_val = torch.as_tensor(
        feature_std.transform(features[split.val]), dtype=torch.float32
    )
    y_val = torch.as_tensor(residual[split.val], dtype=torch.float32)[:, None]

    history: list[tuple[float, float]] = []
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        model.eval()
        with torch.no_grad():
            val = (
                float(nn.functional.mse_loss(model(x_val), y_val))
                if x_val.shape[0]
                else float("nan")
            )
        history.append((total / max(n_batches, 1), val))
        if x_val.shape[0] and val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return TrainedDebondInverse(model.eval(), feature_std, names), history


def debond_scores(
    predicted_log_h: np.ndarray, true_log_h: np.ndarray
) -> dict[str, float]:
    """
    Score a gap-width prediction, in the same terms as the classical baseline.

    Parameters
    ----------
    predicted_log_h, true_log_h : ndarray, shape (N,)
        ``log10`` gap width (m).

    Returns
    -------
    dict
        ``median_ratio``, ``ratio_iqr``, ``log_rmse``, ``pct_error`` (the
        log RMSE expressed as a percentage of thickness, which is the number
        worth quoting), and ``n_scored``.
    """
    ok = np.isfinite(predicted_log_h) & np.isfinite(true_log_h)
    if not ok.any():
        return {
            "median_ratio": float("nan"),
            "ratio_iqr": float("nan"),
            "log_rmse": float("nan"),
            "pct_error": float("nan"),
            "n_scored": 0.0,
        }
    error = predicted_log_h[ok] - true_log_h[ok]
    ratio = 10.0**error
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "median_ratio": float(np.median(ratio)),
        "ratio_iqr": float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
        "log_rmse": rmse,
        "pct_error": float(100.0 * (10.0**rmse - 1.0)),
        "n_scored": float(ok.sum()),
    }
