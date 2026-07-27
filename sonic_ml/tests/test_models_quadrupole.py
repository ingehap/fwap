"""The models are mode-count-agnostic: verify they handle a 3-mode dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sonic_ml import gen_shim, load_npz, stratified_split  # noqa: E402
from sonic_ml.models import train_forward, train_inverse  # noqa: E402


@pytest.fixture(scope="module")
def q_bundle(tmp_path_factory):
    """A 3-mode dataset (default Stoneley + flexural, plus quadrupole)."""
    g = gen_shim.generator()
    freq = g.default_freq_grid(n_freq=48)
    modes = (*g.DEFAULT_MODES, g.QUADRUPOLE_MODE)
    path = tmp_path_factory.mktemp("q") / "q.npz"
    gen_shim.build_npz(str(path), n=36, seed=0, freq=freq, modes=modes)
    return load_npz(str(path))


def test_three_mode_bundle_loads(q_bundle):
    assert q_bundle.n_modes == 3
    assert q_bundle.mode_names == ("Stoneley", "flexural", "quadrupole")


def test_forward_surrogate_handles_three_modes(q_bundle):
    split = stratified_split(q_bundle, seed=0)
    trained, _ = train_forward(
        q_bundle, split=split, epochs=5, width=32, depth=2, seed=0
    )
    slowness, presence = trained.predict(q_bundle.params[:3])
    # One curve + one presence logit per mode -- M read from the data, not hard-coded.
    assert slowness.shape == (3, 3, q_bundle.n_freq)
    assert presence.shape == (3, 3)


def test_inverse_net_is_mode_count_agnostic(q_bundle):
    # The inverse target is the parameters, so more modes just mean richer
    # gathers -- the output stays the active-parameter vector.
    split = stratified_split(q_bundle, seed=0)
    trained, _ = train_inverse(
        q_bundle, split=split, epochs=4, channels=(8, 16), seed=0
    )
    params, std = trained.predict(q_bundle.gather[:3])
    assert params.shape == (3, trained.param_std.n_active)
    assert std.shape == (3, trained.param_std.n_active)
