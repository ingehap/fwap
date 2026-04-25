"""Performance regression tests.

These benchmarks lock in the per-frequency vectorisation speedups
landed in the 0.4.0 rewrite. Each test runs one hot-path function on
a realistic problem size and asserts a hard wall-clock budget
generous enough to survive CI-runner variance but tight enough to
catch a catastrophic regression (e.g. accidentally reintroducing a
Python loop).

The budgets are generous: they are set at ~5x the observed median on
a developer laptop, so CI runners with different hardware profiles
still pass. Lower them if you want tighter regression bars.

Run just these tests with::

    pytest tests/test_bench.py

To collect detailed timing stats for comparison across commits::

    pytest tests/test_bench.py --benchmark-autosave
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from fwap.dispersion import (  # noqa: E402
    dispersive_stc,
    phase_slowness_from_f_k,
    phase_slowness_matrix_pencil,
)
from fwap.synthetic import (  # noqa: E402
    ArrayGeometry,
    Mode,
    dipole_flexural_dispersion,
    monopole_formation_modes,
    synthesize_gather,
)

# ------------------------------------------------------------------
# Shared fixtures -- small enough to keep each bench < 1s on CI.
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def flex_gather():
    Vs = 2500.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=2.0e-5, n_samples=2048)
    disp = dipole_flexural_dispersion(vs=Vs, a_borehole=0.1)
    mode = Mode(name="Flex", slowness=1.0 / Vs, f0=4000.0,
                amplitude=1.0, dispersion=disp)
    return geom, synthesize_gather(geom, [mode], noise=0.03, seed=7)


@pytest.fixture(scope="module")
def monopole_gather():
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=1.0e-5, n_samples=2048)
    data = synthesize_gather(geom, monopole_formation_modes(),
                             noise=0.05, seed=0)
    return geom, data


# ------------------------------------------------------------------
# Benchmarks + hard-bound regression assertions.
#
# Budgets (seconds, one call): set to ~5x the developer-laptop median
# measured at fwap 0.4.0 on an 8-core x86_64 laptop with numpy 1.26.
# ------------------------------------------------------------------

def test_bench_dispersive_stc(benchmark, flex_gather):
    """Dispersive STC on a 81-slowness x 8-receiver grid."""
    geom, data = flex_gather

    def disp_family(s_shear: float):
        return dipole_flexural_dispersion(vs=1.0 / s_shear,
                                          a_borehole=0.1)

    result = benchmark.pedantic(
        dispersive_stc,
        args=(data,),
        kwargs=dict(
            dt=geom.dt, offsets=geom.offsets,
            dispersion_family=disp_family,
            shear_slowness_range=(200e-6, 600e-6),
            n_slowness=81, f_range=(500.0, 4000.0),
            window_length=1.5e-3, time_step=4,
        ),
        rounds=3, iterations=1,
    )
    assert result.slowness.size == 81
    # Budget: 500 ms for an n_slowness=81 scan (inner loop is
    # vectorised to ~90 ms on laptop; 5x headroom).
    assert benchmark.stats.stats.mean < 0.5


def test_bench_phase_slowness_freq_unwrap(benchmark, flex_gather):
    """Frequency-unwrap dispersion estimator on a single gather."""
    geom, data = flex_gather
    result = benchmark.pedantic(
        phase_slowness_from_f_k,
        args=(data,),
        kwargs=dict(
            dt=geom.dt, offsets=geom.offsets,
            f_range=(500.0, 8000.0),
            method="frequency_unwrap",
        ),
        rounds=5, iterations=1,
    )
    assert result.slowness.size > 0
    # Budget 100 ms (vectorised ~1 ms on laptop; loops-era was ~13 ms).
    assert benchmark.stats.stats.mean < 0.1


def test_bench_phase_slowness_spatial_unwrap(benchmark, flex_gather):
    """Spatial-unwrap dispersion estimator (vectorised np.unwrap)."""
    geom, data = flex_gather
    result = benchmark.pedantic(
        phase_slowness_from_f_k,
        args=(data,),
        kwargs=dict(
            dt=geom.dt, offsets=geom.offsets,
            f_range=(500.0, 8000.0),
            method="spatial_unwrap",
        ),
        rounds=5, iterations=1,
    )
    assert result.slowness.size > 0
    assert benchmark.stats.stats.mean < 0.1


def test_bench_matrix_pencil(benchmark, flex_gather):
    """Matrix-pencil estimator over the full band."""
    geom, data = flex_gather
    result = benchmark.pedantic(
        phase_slowness_matrix_pencil,
        args=(data,),
        kwargs=dict(
            dt=geom.dt, offsets=geom.offsets,
            f_range=(500.0, 8000.0),
        ),
        rounds=5, iterations=1,
    )
    assert result.slowness.size > 0
    assert benchmark.stats.stats.mean < 0.05


def test_bench_synthesize_dispersive_gather(benchmark):
    """Synthesise one dipole-flexural gather with 8 receivers."""
    Vs = 2500.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524,
                         dt=2.0e-5, n_samples=2048)
    disp = dipole_flexural_dispersion(vs=Vs, a_borehole=0.1)
    mode = Mode(name="Flex", slowness=1.0 / Vs, f0=4000.0,
                amplitude=1.0, dispersion=disp)
    data = benchmark.pedantic(
        synthesize_gather,
        args=(geom, [mode]),
        kwargs=dict(noise=0.03, seed=7),
        rounds=5, iterations=1,
    )
    assert data.shape == (geom.n_rec, geom.n_samples)
    # Budget 50 ms (vectorised ~1 ms on laptop).
    assert benchmark.stats.stats.mean < 0.05
