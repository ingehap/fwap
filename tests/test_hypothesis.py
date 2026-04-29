"""Property-based tests for algebraic invariants of the fwap core.

Catches edge cases (near-zero, very large, degenerate aspect ratios)
that the hand-crafted unit tests do not systematically cover.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays  # noqa: E402

from fwap.coherence import semblance  # noqa: E402
from fwap.rockphysics import elastic_moduli  # noqa: E402
from fwap.wavesep import fk_forward, fk_inverse  # noqa: E402

# Keep the test suite fast: 30 examples per test covers most branches
# without bloating CI.
_FAST = settings(max_examples=30, deadline=1000)


# ------------------------------------------------------------------
# semblance invariants
# ------------------------------------------------------------------

_finite_floats = st.floats(
    min_value=-1.0e6,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
)


@_FAST
@given(
    window=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=2, max_value=10),
            st.integers(min_value=2, max_value=64),
        ),
        elements=_finite_floats,
    ),
)
def test_semblance_in_unit_interval(window):
    """semblance(w) is always in [0, 1] for any finite real input."""
    rho = semblance(window)
    if np.isnan(rho):
        return  # all-zero window; NaN is documented behaviour
    assert -1.0e-9 <= rho <= 1.0 + 1.0e-9


@_FAST
@given(
    window=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=2, max_value=8),
            st.integers(min_value=4, max_value=32),
        ),
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    alpha=st.floats(
        min_value=1.0e-3,
        max_value=1.0e3,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_semblance_scale_invariant(window, alpha):
    """semblance(alpha * x) == semblance(x) for any non-zero alpha.

    Semblance is a ratio of stack power to trace power; multiplying
    every trace by the same scalar cancels out.
    """
    rho_a = semblance(window)
    rho_b = semblance(alpha * window)
    if np.isnan(rho_a) or np.isnan(rho_b):
        return
    assert abs(rho_a - rho_b) < 1.0e-9


@_FAST
@given(
    trace=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=4, max_value=128),
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    n_rec=st.integers(min_value=2, max_value=8),
)
def test_semblance_of_identical_traces_is_one(trace, n_rec):
    """N perfectly coherent traces have semblance exactly 1."""
    # Skip the degenerate all-zero case where semblance returns NaN.
    if np.sum(trace * trace) < 1.0e-12:
        return
    window = np.tile(trace, (n_rec, 1))
    rho = semblance(window)
    assert abs(rho - 1.0) < 1.0e-10


# ------------------------------------------------------------------
# f-k round trip
# ------------------------------------------------------------------


@_FAST
@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=2, max_value=16),
            st.integers(min_value=8, max_value=256),
        ),
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
)
def test_fk_round_trip_is_identity(data):
    """fk_forward o fk_inverse reproduces the input to machine precision."""
    # dt/dx are irrelevant to the round-trip: they only scale the
    # returned frequency / wavenumber axes, not the spectrum itself.
    spec, _, _ = fk_forward(data, dt=1.0e-5, dx=0.1)
    back = fk_inverse(spec, n_samples=data.shape[1])
    # Absolute tolerance scales with the input magnitude; a lax
    # relative tol catches numerical drift without flagging tiny
    # inputs as failures.
    assert np.allclose(back, data, atol=1.0e-8, rtol=1.0e-8)


# ------------------------------------------------------------------
# Rock physics
# ------------------------------------------------------------------


@_FAST
@given(
    vs=st.floats(min_value=500.0, max_value=5000.0),
    vp_ratio=st.floats(min_value=1.5, max_value=3.0),  # Vp/Vs
    rho=st.floats(min_value=1000.0, max_value=3500.0),
)
def test_elastic_moduli_poisson_in_physical_range(vs, vp_ratio, rho):
    """Poisson's ratio is in (-1, 0.5] for any physically valid input."""
    vp = vp_ratio * vs
    out = elastic_moduli(vp=vp, vs=vs, rho=rho)
    nu = float(out.poisson)
    assert nu > -1.0
    assert nu <= 0.5 + 1.0e-12


@_FAST
@given(
    vs=st.floats(min_value=500.0, max_value=5000.0),
    vp_ratio=st.floats(min_value=1.5, max_value=3.0),
    rho=st.floats(min_value=1000.0, max_value=3500.0),
)
def test_elastic_moduli_are_non_negative(vs, vp_ratio, rho):
    """Bulk, shear, and Young's moduli are non-negative."""
    vp = vp_ratio * vs
    out = elastic_moduli(vp=vp, vs=vs, rho=rho)
    assert float(out.k) > 0.0
    assert float(out.mu) > 0.0
    assert float(out.young) > 0.0


@_FAST
@given(
    vs=st.floats(min_value=500.0, max_value=5000.0),
    vp_ratio=st.floats(min_value=1.5, max_value=3.0),
    rho=st.floats(min_value=1000.0, max_value=3500.0),
)
def test_elastic_moduli_young_from_bulk_shear(vs, vp_ratio, rho):
    """Young's modulus satisfies E = 9 K mu / (3 K + mu) for isotropic media."""
    vp = vp_ratio * vs
    out = elastic_moduli(vp=vp, vs=vs, rho=rho)
    k, mu, young = float(out.k), float(out.mu), float(out.young)
    expected = 9.0 * k * mu / (3.0 * k + mu)
    assert abs(young - expected) / expected < 1.0e-10


# ------------------------------------------------------------------
# STC and moveout invariants
# ------------------------------------------------------------------


@_FAST
@given(
    n_rec=st.integers(min_value=2, max_value=8),
    n_samples=st.integers(min_value=64, max_value=512),
    n_slowness=st.integers(min_value=2, max_value=64),
)
def test_stc_returns_expected_shape(n_rec, n_samples, n_slowness):
    """stc output slowness / time / coherence arrays have the advertised shapes."""
    from fwap.coherence import stc

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_rec, n_samples))
    offsets = np.arange(n_rec) * 0.1524
    result = stc(
        data,
        dt=1.0e-5,
        offsets=offsets,
        slowness_range=(50e-6, 400e-6),
        n_slowness=n_slowness,
        window_length=max(1e-5, n_samples * 0.05 * 1e-5),
        time_step=max(1, n_samples // 32),
    )
    assert result.slowness.shape == (n_slowness,)
    assert result.time.ndim == 1
    assert result.coherence.shape == (n_slowness, result.time.size)


@_FAST
@given(
    n_rec=st.integers(min_value=2, max_value=8),
    n_samples=st.integers(min_value=64, max_value=512),
    slowness=st.floats(min_value=50.0e-6, max_value=300.0e-6, allow_nan=False),
)
def test_apply_moveout_preserves_energy(n_rec, n_samples, slowness):
    """``apply_moveout`` is an approximate isometry in the rFFT sense.

    The frequency-domain phase shift modifies only the phase of each
    bin, so the total spectral magnitude is preserved. The time-domain
    RMS therefore matches to within the Nyquist-bin floor (real-only
    constraint forces a tiny amplitude change on even-length inputs).
    Tolerance 1% relative RMS -- same budget as the one pinned in
    ``tests/test_wavesep.py``.
    """
    from fwap.wavesep import apply_moveout

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_rec, n_samples))
    offsets = np.arange(n_rec) * 0.1524
    shifted = apply_moveout(data, dt=1.0e-5, offsets=offsets, slowness=slowness)
    rms_in = np.sqrt(np.mean(data**2))
    rms_out = np.sqrt(np.mean(shifted**2))
    if rms_in < 1.0e-12:
        return
    assert abs(rms_out - rms_in) / rms_in < 1.0e-2


@_FAST
@given(
    n=st.integers(min_value=1, max_value=20),
    vs_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_elastic_moduli_array_broadcasts(n, vs_seed):
    """Array-of-N inputs produce same-shape outputs with finite moduli."""
    rng = np.random.default_rng(vs_seed)
    vs = rng.uniform(1000.0, 3500.0, size=n)
    vp = vs * rng.uniform(1.5, 2.8, size=n)
    rho = rng.uniform(1500.0, 3000.0, size=n)
    out = elastic_moduli(vp=vp, vs=vs, rho=rho)
    assert out.k.shape == (n,)
    assert out.mu.shape == (n,)
    assert out.young.shape == (n,)
    assert out.poisson.shape == (n,)
    assert np.all(np.isfinite(out.k))
    assert np.all(np.isfinite(out.mu))
    assert np.all(np.isfinite(out.young))
    assert np.all(np.isfinite(out.poisson))


@_FAST
@given(
    n=st.integers(min_value=1, max_value=8),
    raw=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=8),
        elements=st.floats(
            min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_voigt_always_above_or_equal_reuss(n, raw):
    """For any non-degenerate mixture, Voigt >= Reuss."""
    from fwap.rockphysics import reuss_average, voigt_average

    moduli = raw.astype(float)
    # Build non-negative fractions summing to 1.
    fractions = np.full_like(moduli, 1.0 / moduli.size)
    v = voigt_average(moduli, fractions)
    r = reuss_average(moduli, fractions)
    assert v >= r - 1.0e-12


# ------------------------------------------------------------------
# Picker robustness on synthetic STC surfaces
# ------------------------------------------------------------------


@_FAST
@given(
    vp=st.floats(min_value=3500.0, max_value=6000.0),
    vs_ratio=st.floats(min_value=1.6, max_value=2.2),  # Vp/Vs
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_pick_modes_recovers_planted_p_slowness(vp, vs_ratio, seed):
    """Synthesise a clean monopole gather, run STC + picker, recover Vp.

    For any physically reasonable formation (Vp 3.5-6 km/s, Vp/Vs in
    a sandstone-to-carbonate range) the picker should land within
    5% of the planted P slowness.
    """
    from fwap.coherence import stc
    from fwap.picker import pick_modes
    from fwap.synthetic import (
        ArrayGeometry,
        monopole_formation_modes,
        synthesize_gather,
    )

    vs = vp / vs_ratio
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    data = synthesize_gather(
        geom,
        monopole_formation_modes(vp=vp, vs=vs, v_stoneley=1400.0),
        noise=0.05,
        seed=seed,
    )
    surface = stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(50e-6, 800e-6),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )
    picks = pick_modes(surface, threshold=0.4)
    # P should always be picked on a clean synthetic.
    assert "P" in picks, f"P missing from picks {list(picks)}"
    s_planted = 1.0 / vp
    s_recovered = picks["P"].slowness
    rel = abs(s_recovered - s_planted) / s_planted
    assert rel < 0.05, (
        f"P slowness off by {rel * 100:.1f}% "
        f"(recovered {s_recovered:.3e}, truth {s_planted:.3e})"
    )
