"""
Cased-hole forward operator: annulus + formation -> Stoneley dispersion (M5c).

Wires the M5b operator primitives to the M5a cased-hole dataset. The learned map
is

    (formation, casing, cement, bond index)  ->  Stoneley slowness curve s(f)

which is a *surrogate for the layered modal-determinant root-finding* in
:func:`fwap.stoneley_dispersion_layered`. Framed honestly, as with the M2
open-hole surrogate: this is a methodology and validation baseline, not a speed
claim -- the reason to build it is that an operator over the annulus is the
right object for the cased-hole path, where the modal picture is far messier
than the open hole and the M5d bond inverse needs a differentiable forward map.

What an *operator* buys over the M2 point surrogate
---------------------------------------------------
The frequency axis enters as a **coordinate**, not an array index, so a trained
:class:`TrainedCasedOperator` can be evaluated on a frequency grid it never saw
-- finer, coarser, or a sub-band -- via ``predict(..., freq=...)``. The stored
training band is what normalizes any query grid, so predictions on a new grid
stay on the same coordinate system.

Both M5b backbones are selectable: the ``"fno"`` default lifts the parameters
onto the grid (:func:`~sonic_ml.models.operator.params_on_grid`) and runs a
:class:`~sonic_ml.models.operator.FNO1d`; ``"deeponet"`` feeds the parameters to
the branch net and the coordinates to the trunk. Both return ``(B, M, F)``, so
everything downstream is backbone-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from sonic_ml.loader import LAYER_PARAM_NAMES, DatasetBundle
from sonic_ml.models.dataset import SlownessNormalizer
from sonic_ml.models.operator import DeepONet, FNO1d, params_on_grid
from sonic_ml.normalize import Standardizer

#: Selectable operator backbones.
BACKBONES: tuple[str, ...] = ("fno", "deeponet")


def cased_features(bundle: DatasetBundle) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    Build the operator's input feature matrix from a cased-hole bundle.

    Concatenates the formation parameters, the flattened per-layer annulus
    properties, and the bond index into one ``(N, P)`` matrix.

    The bond index is *redundant* -- it is a normalized function of the cement
    shear velocity, which is already a column -- but it is included on purpose:
    it makes the operator's bond sensitivity directly probeable (hold the
    formation fixed, sweep the bond) and gives the M5d inverse a named target
    that lines up with this operator's input. A :class:`Standardizer` fitted on
    the result drops any genuinely constant column (e.g. the fixed casing
    density and the fluid properties), so redundancy costs at most one feature.

    Parameters
    ----------
    bundle : DatasetBundle
        A cased-hole dataset (schema v4, ``is_cased`` true).

    Returns
    -------
    features : ndarray, shape (N, P), float64
        Formation columns, then ``<layer>_<property>`` columns in radial
        order, then ``bond_index``.
    names : tuple of str, length P
        Column names aligned with ``features``.

    Raises
    ------
    ValueError
        If the bundle carries no cased-hole annulus.
    """
    if not bundle.is_cased or bundle.layer_params is None:
        raise ValueError(
            "cased_features requires a cased-hole dataset (schema v4 with a "
            "non-empty layer stack); got an open-hole bundle"
        )
    if bundle.bond_index is None:  # pragma: no cover - v4 reads both together
        raise ValueError("cased bundle is missing bond_index")

    n_samples = bundle.n_samples
    layer_flat = bundle.layer_params.reshape(n_samples, -1)
    layer_names = tuple(
        f"{layer}_{prop}" for layer in bundle.layer_names for prop in LAYER_PARAM_NAMES
    )
    features = np.concatenate(
        [
            np.asarray(bundle.params, dtype=float),
            np.asarray(layer_flat, dtype=float),
            np.asarray(bundle.bond_index, dtype=float)[:, None],
        ],
        axis=1,
    )
    names = (*bundle.param_names, *layer_names, "bond_index")
    return features, names


def normalized_coords(freq: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    """
    Map a frequency grid to the operator's ``[0, 1]`` coordinate axis.

    Normalizing against the *training* band (rather than each grid's own
    endpoints) is what keeps a re-gridded query on the same coordinate system as
    training -- without it, a sub-band query would be silently rescaled.

    Parameters
    ----------
    freq : ndarray, shape (F,)
        Frequencies (Hz).
    f_min, f_max : float
        Training-band edges (Hz). Must satisfy ``f_max > f_min``.

    Returns
    -------
    ndarray, shape (F,), float64
        ``(freq - f_min) / (f_max - f_min)``; values outside the training band
        fall outside ``[0, 1]`` (extrapolation, flagged by the caller).
    """
    if not f_max > f_min:
        raise ValueError(f"require f_max > f_min, got {f_min} and {f_max}")
    return (np.asarray(freq, dtype=float) - f_min) / (f_max - f_min)


class CasedForwardOperator(nn.Module):
    """
    Operator network mapping cased-hole parameters to per-mode dispersion.

    Parameters
    ----------
    n_features : int
        Number of (active) input features, from a fitted
        :class:`~sonic_ml.normalize.Standardizer`.
    n_modes_out : int, default 1
        Number of output curves ``M`` (one for the cased Stoneley dataset).
    backbone : {"fno", "deeponet"}, default "fno"
        Which M5b primitive to use.
    width, n_fourier_modes, depth : int
        ``FNO1d`` hyper-parameters (ignored by the DeepONet backbone).
    latent : int, default 64
        DeepONet basis size (ignored by the FNO backbone).
    """

    def __init__(
        self,
        n_features: int,
        n_modes_out: int = 1,
        *,
        backbone: str = "fno",
        width: int = 32,
        n_fourier_modes: int = 16,
        depth: int = 4,
        latent: int = 64,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
        self.hparams: dict[str, Any] = {
            "n_features": n_features,
            "n_modes_out": n_modes_out,
            "backbone": backbone,
            "width": width,
            "n_fourier_modes": n_fourier_modes,
            "depth": depth,
            "latent": latent,
        }
        self.backbone = backbone
        self.net: nn.Module
        if backbone == "fno":
            # +1 input channel for the coordinate appended by params_on_grid.
            self.net = FNO1d(
                n_features + 1,
                n_modes_out,
                width=width,
                n_modes=n_fourier_modes,
                depth=depth,
            )
        else:
            self.net = DeepONet(
                n_features,
                1,
                latent=latent,
                out_channels=n_modes_out,
            )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Predict normalized dispersion curves at the given coordinates.

        Parameters
        ----------
        x : Tensor, shape (B, n_features)
            Standardized cased-hole features.
        coords : Tensor, shape (F,)
            Normalized frequency coordinates (see :func:`normalized_coords`).

        Returns
        -------
        Tensor, shape (B, M, F)
            Normalized per-mode slowness.
        """
        if self.backbone == "fno":
            return self.net(params_on_grid(x, coords))
        return self.net(x, coords)


class TrainedCasedOperator:
    """
    A trained :class:`CasedForwardOperator` with its normalizers and band.

    Predicts raw slowness (s/m) on any frequency grid inside -- or, with a
    warning-free but flagged extrapolation, outside -- the training band, and
    round-trips through a ``weights_only``-safe checkpoint.

    Parameters
    ----------
    model : CasedForwardOperator
    feature_std : Standardizer
        Fitted on the *training* cased features.
    slowness_norm : SlownessNormalizer
        Fitted on the *training* slowness curves.
    feature_names : tuple of str
        Full (pre-drop) feature column names, so raw inputs can be assembled.
    mode_names : tuple of str
        Mode labels aligned with the model's output axis.
    freq : ndarray, shape (F,)
        Training frequency grid (Hz); its endpoints define the coordinate
        normalization used for every query.
    """

    def __init__(
        self,
        model: CasedForwardOperator,
        feature_std: Standardizer,
        slowness_norm: SlownessNormalizer,
        feature_names: tuple[str, ...],
        mode_names: tuple[str, ...],
        freq: np.ndarray,
    ) -> None:
        self.model = model
        self.feature_std = feature_std
        self.slowness_norm = slowness_norm
        self.feature_names = feature_names
        self.mode_names = mode_names
        self.freq = np.asarray(freq, dtype=float)

    @property
    def f_min(self) -> float:
        """Lower edge of the training band (Hz)."""
        return float(self.freq[0])

    @property
    def f_max(self) -> float:
        """Upper edge of the training band (Hz)."""
        return float(self.freq[-1])

    @torch.no_grad()
    def predict(
        self, features: np.ndarray, *, freq: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Predict raw slowness curves, optionally on a different frequency grid.

        Parameters
        ----------
        features : ndarray, shape (N, P)
            Raw cased-hole features (full column set, as from
            :func:`cased_features`).
        freq : ndarray, shape (F',) or None
            Frequency grid to evaluate on (Hz). ``None`` uses the training
            grid. **Any** grid is admissible -- this is the operator payoff --
            and is normalized against the stored training band.

        Returns
        -------
        ndarray, shape (N, M, F'), float64
            Predicted per-mode slowness (s/m).
        """
        self.model.eval()
        grid = self.freq if freq is None else np.asarray(freq, dtype=float)
        x = self.feature_std.transform(np.asarray(features, dtype=float))
        coords = normalized_coords(grid, self.f_min, self.f_max)
        xt = torch.as_tensor(x, dtype=torch.float32)
        ct = torch.as_tensor(coords, dtype=torch.float32)
        z = self.model(xt, ct).cpu().numpy().astype(float)
        return self.slowness_norm.inverse(z)

    def save(self, path: str) -> None:
        """Serialize model + normalizers to ``path`` (weights-only safe)."""
        checkpoint: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "hparams": dict(self.model.hparams),
            "feature_mean": torch.as_tensor(self.feature_std.mean),
            "feature_std": torch.as_tensor(self.feature_std.std),
            "feature_active": torch.as_tensor(self.feature_std.active),
            "feature_std_names": list(self.feature_std.names),
            "slow_mean": torch.as_tensor(self.slowness_norm.mean),
            "slow_std": torch.as_tensor(self.slowness_norm.std),
            "feature_names": list(self.feature_names),
            "mode_names": list(self.mode_names),
            "freq": torch.as_tensor(self.freq),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, *, map_location: str = "cpu") -> TrainedCasedOperator:
        """Load a checkpoint written by :meth:`save`."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        hp = ckpt["hparams"]
        model = CasedForwardOperator(
            int(hp["n_features"]),
            int(hp["n_modes_out"]),
            backbone=str(hp["backbone"]),
            width=int(hp["width"]),
            n_fourier_modes=int(hp["n_fourier_modes"]),
            depth=int(hp["depth"]),
            latent=int(hp["latent"]),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        feature_std = Standardizer(
            mean=ckpt["feature_mean"].numpy(),
            std=ckpt["feature_std"].numpy(),
            active=ckpt["feature_active"].numpy().astype(bool),
            names=tuple(ckpt["feature_std_names"]),
        )
        slowness_norm = SlownessNormalizer(
            mean=ckpt["slow_mean"].numpy(),
            std=ckpt["slow_std"].numpy(),
        )
        return cls(
            model,
            feature_std,
            slowness_norm,
            tuple(ckpt["feature_names"]),
            tuple(ckpt["mode_names"]),
            ckpt["freq"].numpy(),
        )
