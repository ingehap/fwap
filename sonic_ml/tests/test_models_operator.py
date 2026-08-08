"""Tests for sonic_ml.models.operator -- FNO / DeepONet primitives (torch-gated).

These are pure-torch unit tests: no dataset fixture, so they stay fast. The
exact-mathematics tests (DC-only convolution, low-pass rejection, pointwise
trunk) pin the operator semantics; the short training loops only assert that
learning *happens*, never a specific accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonic_ml.models.operator import (  # noqa: E402
    DeepONet,
    FNO1d,
    SpectralConv1d,
    params_on_grid,
)


def _target(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A smooth band-limited reference operator: params -> curve on ``c``."""
    curve = p[:, 0:1] * torch.sin(2 * torch.pi * c[None, :]) + p[:, 1:2] * torch.cos(
        4 * torch.pi * c[None, :]
    )
    return curve[:, None, :]


# ------------------------------------------------------------------
# SpectralConv1d -- exact Fourier semantics
# ------------------------------------------------------------------


def test_spectral_conv_dc_only_equals_broadcast_mean():
    """Retaining only mode 0 with unit gain is exactly the grid mean."""
    conv = SpectralConv1d(1, 1, n_modes=1)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 0, 0] = 1.0  # real part = 1 on the DC mode
    x = torch.randn(2, 1, 64)
    out = conv(x)
    expected = x.mean(dim=-1, keepdim=True).expand_as(out)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_spectral_conv_rejects_modes_above_cutoff():
    """Energy above the retained mode count is filtered out (low-pass)."""
    conv = SpectralConv1d(1, 1, n_modes=4)
    t = torch.arange(64, dtype=torch.float32)
    high = torch.cos(2 * torch.pi * 20 * t / 64)[None, None, :]  # mode 20 > 4
    assert float(conv(high).abs().max().detach()) < 1e-5


def test_spectral_conv_preserves_grid_length_and_sets_channels():
    conv = SpectralConv1d(3, 5, n_modes=6)
    for n_grid in (16, 33, 128):
        out = conv(torch.randn(2, 3, n_grid))
        assert out.shape == (2, 5, n_grid)


def test_spectral_conv_handles_grid_shorter_than_mode_count():
    # A grid too short to carry every trained mode uses those it can.
    conv = SpectralConv1d(2, 2, n_modes=32)
    out = conv(torch.randn(1, 2, 8))
    assert out.shape == (1, 2, 8)
    assert torch.isfinite(out).all()


def test_spectral_conv_rejects_nonpositive_modes():
    with pytest.raises(ValueError):
        SpectralConv1d(1, 1, n_modes=0)


def test_spectral_conv_weight_is_real_for_weights_only_checkpoints():
    # Stored as a real (..., 2) tensor so plain optimizers / weights_only load
    # work; complex Parameters would break both.
    conv = SpectralConv1d(2, 3, n_modes=4)
    assert conv.weight.shape == (2, 3, 4, 2)
    assert not conv.weight.is_complex()


# ------------------------------------------------------------------
# params_on_grid
# ------------------------------------------------------------------


def test_params_on_grid_shape_and_coordinate_channel():
    params = torch.randn(5, 3)
    coords = torch.linspace(0.0, 1.0, 40)
    grid = params_on_grid(params, coords)
    assert grid.shape == (5, 4, 40)  # P broadcast channels + 1 coordinate
    # Every parameter channel is constant along the grid axis.
    for p in range(3):
        torch.testing.assert_close(grid[:, p, :], params[:, p : p + 1].expand(5, 40))
    # The trailing channel is the coordinate itself.
    torch.testing.assert_close(grid[0, -1, :], coords)


@pytest.mark.parametrize(
    ("params", "coords"),
    [
        (torch.randn(4), torch.linspace(0, 1, 8)),  # params not 2-D
        (torch.randn(4, 2), torch.zeros(8, 2)),  # coords not 1-D
    ],
)
def test_params_on_grid_validates_shapes(params, coords):
    with pytest.raises(ValueError):
        params_on_grid(params, coords)


# ------------------------------------------------------------------
# FNO1d
# ------------------------------------------------------------------


def test_fno_output_shape_follows_input_grid():
    model = FNO1d(4, out_channels=1, width=16, n_modes=8, depth=2)
    for n_grid in (7, 32, 48, 128):
        out = model(torch.randn(3, 4, n_grid))
        assert out.shape == (3, 1, n_grid)


def test_fno_has_no_grid_length_dependent_weights():
    """The same state_dict runs on any grid -- the resolution-agnostic claim."""
    model = FNO1d(3, width=16, n_modes=8, depth=2)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    twin = FNO1d(3, width=16, n_modes=8, depth=2)
    twin.load_state_dict(state)  # no shape mismatch
    twin.eval()
    with torch.no_grad():
        assert twin(torch.randn(1, 3, 32)).shape == (1, 1, 32)
        assert twin(torch.randn(1, 3, 96)).shape == (1, 1, 96)


def test_fno_rejects_nonpositive_depth():
    with pytest.raises(ValueError):
        FNO1d(2, depth=0)


def test_fno_hparams_recorded_for_model_card():
    model = FNO1d(5, out_channels=2, width=16, n_modes=6, depth=3)
    assert model.hparams == {
        "in_channels": 5,
        "out_channels": 2,
        "width": 16,
        "n_modes": 6,
        "depth": 3,
    }


def test_fno_deterministic_given_seed():
    from sonic_ml.determinism import seed_everything

    seed_everything(0)
    a = FNO1d(3, width=16, n_modes=6, depth=2).eval()
    seed_everything(0)
    b = FNO1d(3, width=16, n_modes=6, depth=2).eval()
    x = torch.randn(2, 3, 32)
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x))


def test_fno_checkpoint_round_trips_weights_only(tmp_path):
    model = FNO1d(3, width=16, n_modes=6, depth=2).eval()
    x = torch.randn(2, 3, 40)
    with torch.no_grad():
        ref = model(x)
    path = tmp_path / "fno.pt"
    torch.save(model.state_dict(), path)
    loaded = FNO1d(3, width=16, n_modes=6, depth=2)
    loaded.load_state_dict(torch.load(str(path), weights_only=True))
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), ref)


def _train_fno(steps: int = 150) -> tuple[FNO1d, torch.Tensor, float, float]:
    """Fit the reference operator on a coarse grid; return the loss endpoints."""
    torch.manual_seed(0)
    coarse = torch.linspace(0.0, 1.0, 32)
    params = torch.rand(128, 2) * 2 - 1
    model = FNO1d(3, width=24, n_modes=10, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = last = float("nan")
    for step in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(params_on_grid(params, coarse)), _target(params, coarse)
        )
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if step == 0:
            first = last
    return model, coarse, first, last


def test_fno_learns_a_simple_operator():
    _, _, first, last = _train_fno()
    assert np.isfinite(last)
    assert last < 0.2 * first  # loss falls by well over 5x


def test_fno_zero_shot_transfers_to_a_finer_grid():
    """A grid-trained FNO evaluated on a 2x finer grid agrees at shared nodes.

    This is the capability a point network structurally cannot have: the
    frequency axis is a coordinate, not an array index.
    """
    model, coarse, _, _ = _train_fno()
    model.eval()
    fine = torch.linspace(0.0, 1.0, 2 * coarse.numel() - 1)
    torch.testing.assert_close(fine[::2], coarse)  # nodes really do coincide
    params = torch.rand(32, 2) * 2 - 1
    with torch.no_grad():
        on_fine = model(params_on_grid(params, fine))
        on_coarse = model(params_on_grid(params, coarse))
    rel = (on_fine[:, :, ::2] - on_coarse).norm() / on_coarse.norm()
    # Loose bound: FFT edge effects differ between resolutions, so this is
    # consistency, not equality.
    assert float(rel) < 0.25


# ------------------------------------------------------------------
# DeepONet
# ------------------------------------------------------------------


def test_deeponet_output_shape_for_arbitrary_coords():
    model = DeepONet(3, latent=16, branch_hidden=(32,), trunk_hidden=(32,))
    params = torch.randn(5, 3)
    for n_query in (1, 10, 55):
        out = model(params, torch.rand(n_query))
        assert out.shape == (5, 1, n_query)


def test_deeponet_accepts_2d_coords_and_multiple_outputs():
    model = DeepONet(
        3,
        n_trunk_features=2,
        latent=8,
        branch_hidden=(16,),
        trunk_hidden=(16,),
        out_channels=4,
    )
    out = model(torch.randn(2, 3), torch.rand(7, 2))
    assert out.shape == (2, 4, 7)


def test_deeponet_trunk_is_pointwise_in_the_query():
    """Each coordinate's value is independent of the others in the batch.

    This is what lets the operator be queried at arbitrary frequencies -- if a
    query's value depended on its neighbours, the trunk would be a grid net.
    """
    model = DeepONet(3, latent=16, branch_hidden=(32,), trunk_hidden=(32,)).eval()
    params = torch.randn(4, 3)
    coords = torch.rand(20)
    with torch.no_grad():
        full = model(params, coords)
        subset = model(params, coords[:5])
        shuffled = model(params, coords.flip(0))
    torch.testing.assert_close(subset, full[:, :, :5])
    torch.testing.assert_close(shuffled, full.flip(-1))


def test_deeponet_hparams_recorded_for_model_card():
    model = DeepONet(3, latent=8, branch_hidden=(16, 16), trunk_hidden=(16,))
    assert model.hparams["n_branch_features"] == 3
    assert model.hparams["latent"] == 8
    assert model.hparams["branch_hidden"] == [16, 16]
    assert model.hparams["out_channels"] == 1


def test_deeponet_validates_params_shape():
    model = DeepONet(3, latent=8, branch_hidden=(16,), trunk_hidden=(16,))
    with pytest.raises(ValueError):
        model(torch.randn(3), torch.rand(5))  # params must be (B, P)


def test_deeponet_learns_and_queries_off_grid():
    torch.manual_seed(0)
    coarse = torch.linspace(0.0, 1.0, 32)
    params = torch.rand(128, 2) * 2 - 1
    model = DeepONet(2, latent=32, branch_hidden=(64, 64), trunk_hidden=(64, 64))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = float("nan")
    for step in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(params, coarse), _target(params, coarse)
        )
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if step == 0:
            first = last
    assert last < 0.2 * first
    # Queried at coordinates never seen during training, the output stays
    # finite and on the right scale (accuracy itself is not asserted).
    model.eval()
    off_grid = torch.rand(37)
    with torch.no_grad():
        out = model(torch.rand(8, 2) * 2 - 1, off_grid)
    assert out.shape == (8, 1, 37)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) < 10.0
