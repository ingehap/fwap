"""
Operator-learning primitives: a minimal FNO and a DeepONet (M5b).

The M2 forward surrogate and M3 inverse net are *point* networks: they map a
fixed-length parameter vector to a fixed-length curve, so the frequency grid is
baked into the layer shapes. An **operator** learner instead approximates a map
between *functions*, which buys the property those nets cannot have -- the
learned map is defined on the coordinate, not on the array index, so it
transfers to a grid it was never trained on.

That matters for the cased-hole path (M5c/M5d): the annular stack
(fluid -> casing -> cement -> formation) makes the modal picture far messier
than the open hole, and the dispersion surface is what we want to learn as an
object in its own right.

Two complementary primitives, both **implemented in-house on plain PyTorch**
(no ``neuraloperator`` or other new dependency -- see the scope rule in
``CLAUDE.md``):

* :class:`FNO1d` -- a Fourier neural operator over the frequency axis. Each
  block multiplies the lowest retained Fourier modes by learned complex weights
  (:class:`SpectralConv1d`) and adds a pointwise channel mixer. Truncating to a
  fixed **mode count** rather than a fixed grid length is what makes the block
  resolution-agnostic: the same weights act on any grid long enough to carry
  those modes.
* :class:`DeepONet` -- a branch/trunk factorization. The branch net encodes the
  parameter vector, the trunk net encodes a *query coordinate*, and the output
  is their inner product, so the operator can be evaluated at arbitrary
  frequencies rather than on a stored grid.

Neither class owns any physics; they are wired to the cased-hole dataset in a
later slice.

References
----------
* Li, Z., et al. (2021). Fourier neural operator for parametric partial
  differential equations. *ICLR.*
* Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning
  nonlinear operators via DeepONet. *Nature Machine Intelligence* 3, 218-229.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def params_on_grid(params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Build an FNO input by broadcasting parameters along a coordinate grid.

    An operator layer consumes a *function on a grid*, so a scalar parameter
    vector is lifted by repeating it at every grid point and appending the
    coordinate itself as one extra channel (the standard parametric-operator
    encoding -- without it the network cannot tell grid positions apart).

    Parameters
    ----------
    params : Tensor, shape (B, P)
        Per-sample parameter vector (already standardized).
    coords : Tensor, shape (T,)
        Grid coordinates, e.g. a normalized frequency axis. Normalize these to
        a modest range (``[0, 1]``) before calling.

    Returns
    -------
    Tensor, shape (B, P + 1, T)
        Channel-first operator input: ``P`` broadcast parameter channels plus a
        trailing coordinate channel.
    """
    if params.ndim != 2:
        raise ValueError(f"params must be (B, P), got shape {tuple(params.shape)}")
    if coords.ndim != 1:
        raise ValueError(f"coords must be (T,), got shape {tuple(coords.shape)}")
    batch, n_params = params.shape
    n_grid = coords.shape[0]
    broadcast = params[:, :, None].expand(batch, n_params, n_grid)
    grid = (
        coords.to(params.dtype)
        .to(params.device)[None, None, :]
        .expand(batch, 1, n_grid)
    )
    return torch.cat([broadcast, grid], dim=1)


class SpectralConv1d(nn.Module):
    """
    1-D spectral convolution: learned complex weights on the lowest modes.

    Transforms to the Fourier domain along the last axis, multiplies the lowest
    ``n_modes`` coefficients by a learned complex ``(C_in, C_out, n_modes)``
    tensor, zeroes the rest, and transforms back. Because only a **mode count**
    is learned -- never a grid length -- the same weights apply to any input
    length; modes beyond what a short grid can represent are simply skipped.

    The complex weights are stored as a real tensor with a trailing size-2
    ``(real, imag)`` axis so the parameter is an ordinary real tensor: plain
    optimizers and ``weights_only=True`` checkpoints work unchanged.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel widths.
    n_modes : int
        Number of low-frequency Fourier modes to retain (a low-pass cutoff).
        Must be positive.
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        # He-style scaling keeps activations O(1) through a stack of blocks.
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the spectral convolution.

        Parameters
        ----------
        x : Tensor, shape (B, C_in, T)

        Returns
        -------
        Tensor, shape (B, C_out, T)
            Same grid length as the input.
        """
        n_grid = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        # A grid shorter than the trained mode count carries fewer modes; use
        # what is available rather than erroring (resolution-agnostic by design).
        keep = min(self.n_modes, x_ft.shape[-1])
        weight = torch.view_as_complex(self.weight)[:, :, :keep]
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x_ft.shape[-1],
            dtype=x_ft.dtype,
            device=x.device,
        )
        out_ft[:, :, :keep] = torch.einsum("bit,iot->bot", x_ft[:, :, :keep], weight)
        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)


class FNO1d(nn.Module):
    """
    Fourier neural operator over a 1-D grid.

    Lifts the input channels to ``width``, applies ``depth`` blocks of
    ``GELU(spectral(h) + pointwise(h))``, and projects to ``out_channels``. The
    spectral path carries global (low-frequency) structure while the pointwise
    ``1x1`` path carries the local residual -- the standard FNO decomposition.

    Every layer is either a spectral conv (mode-truncated) or a ``1x1``
    convolution, so the model has **no grid-length-dependent weights** and
    evaluates on any grid length.

    Parameters
    ----------
    in_channels : int
        Input channels, e.g. ``P + 1`` from :func:`params_on_grid`.
    out_channels : int, default 1
        Output channels (one per predicted curve).
    width : int, default 32
        Hidden channel width.
    n_modes : int, default 16
        Retained Fourier modes per spectral conv.
    depth : int, default 4
        Number of spectral blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        width: int = 32,
        n_modes: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.hparams: dict[str, Any] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "width": width,
            "n_modes": n_modes,
            "depth": depth,
        }
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, n_modes) for _ in range(depth)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(depth)]
        )
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map an input function to an output function on the same grid.

        Parameters
        ----------
        x : Tensor, shape (B, in_channels, T)

        Returns
        -------
        Tensor, shape (B, out_channels, T)
        """
        h = self.lift(x)
        for spectral, pointwise in zip(self.spectral, self.pointwise, strict=True):
            h = torch.nn.functional.gelu(spectral(h) + pointwise(h))
        return self.project(h)


def _mlp(n_in: int, hidden: Sequence[int], n_out: int) -> nn.Sequential:
    """Build a GELU MLP ``n_in -> hidden... -> n_out`` (no output activation)."""
    layers: list[nn.Module] = []
    prev = n_in
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.GELU())
        prev = width
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """
    Branch/trunk operator network.

    The *branch* net encodes the parameter vector into ``out_channels x latent``
    coefficients; the *trunk* net encodes each query coordinate into ``latent``
    basis values. Their inner product is the operator's value at that
    coordinate:

    .. math::

        u(\\xi) \\;\\approx\\; \\sum_{k=1}^{L} b_k(\\text{params})\\, t_k(\\xi)
                              \\; + \\; b_0 .

    The trunk is evaluated pointwise, so the operator can be queried at
    **arbitrary coordinates** -- including frequencies absent from the training
    grid -- rather than only at stored grid indices. That makes the learned
    basis, not the sampling, the object being fit.

    Parameters
    ----------
    n_branch_features : int
        Parameter-vector width ``P``.
    n_trunk_features : int, default 1
        Query-coordinate width (``1`` for a frequency axis).
    latent : int, default 64
        Number of learned basis functions ``L``.
    branch_hidden, trunk_hidden : sequence of int
        Hidden widths of the two MLPs.
    out_channels : int, default 1
        Number of output curves.
    """

    def __init__(
        self,
        n_branch_features: int,
        n_trunk_features: int = 1,
        *,
        latent: int = 64,
        branch_hidden: Sequence[int] = (128, 128),
        trunk_hidden: Sequence[int] = (128, 128),
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.hparams: dict[str, Any] = {
            "n_branch_features": n_branch_features,
            "n_trunk_features": n_trunk_features,
            "latent": latent,
            "branch_hidden": list(branch_hidden),
            "trunk_hidden": list(trunk_hidden),
            "out_channels": out_channels,
        }
        self.latent = latent
        self.out_channels = out_channels
        self.branch = _mlp(n_branch_features, branch_hidden, out_channels * latent)
        self.trunk = _mlp(n_trunk_features, trunk_hidden, latent)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the operator at the given query coordinates.

        Parameters
        ----------
        params : Tensor, shape (B, P)
            Per-sample parameter vector.
        coords : Tensor, shape (T,) or (T, n_trunk_features)
            Query coordinates. A 1-D tensor is treated as ``(T, 1)``.

        Returns
        -------
        Tensor, shape (B, out_channels, T)
        """
        if params.ndim != 2:
            raise ValueError(f"params must be (B, P), got shape {tuple(params.shape)}")
        if coords.ndim == 1:
            coords = coords[:, None]
        branch = self.branch(params).reshape(
            params.shape[0], self.out_channels, self.latent
        )
        trunk = self.trunk(coords.to(params.dtype))  # (T, latent)
        return torch.einsum("bol,tl->bot", branch, trunk) + self.bias[None, :, None]
