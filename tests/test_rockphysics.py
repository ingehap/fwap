"""Rock-physics helper tests.

The closed-form relations have exact known answers for standard
reference rocks; we check a few and the scalar/array broadcasting.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwap.rockphysics import ElasticModuli, elastic_moduli, vp_vs_ratio


def test_isotropic_moduli_known_values():
    """Berea-sandstone-like values: Vp=3500, Vs=2000, rho=2200.

    The numbers below are hand-derived from the closed-form relations;
    they have no physical magic, just algebra.
    """
    vp, vs, rho = 3500.0, 2000.0, 2200.0
    mu = rho * vs**2  # 8.80e9
    lam = rho * vp**2 - 2.0 * mu  # 9.34e9
    k = lam + 2.0 / 3.0 * mu  # 15.21e9
    young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)  # 21.35e9
    nu = lam / (2.0 * (lam + mu))  # 0.258
    out = elastic_moduli(vp, vs, rho)
    assert isinstance(out, ElasticModuli)
    for actual, expected, name in [
        (out.mu, mu, "mu"),
        (out.lambda_, lam, "lambda"),
        (out.k, k, "k"),
        (out.young, young, "young"),
        (out.poisson, nu, "poisson"),
    ]:
        assert abs(float(actual) - expected) / expected < 1.0e-10, name


def test_moduli_broadcast_over_arrays():
    """Arrays of inputs produce arrays of outputs with matching shape."""
    n = 5
    vp = np.linspace(3000.0, 4500.0, n)
    vs = np.linspace(1700.0, 2600.0, n)
    rho = np.full(n, 2400.0)
    out = elastic_moduli(vp, vs, rho)
    for arr in (out.k, out.mu, out.young, out.poisson, out.lambda_):
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr))


def test_poisson_in_physical_range():
    """Poisson's ratio lies in (-1, 0.5] for every physically valid input."""
    rng = np.random.default_rng(0)
    n = 50
    vs = rng.uniform(1200.0, 3000.0, size=n)
    vp = vs * rng.uniform(1.4, 2.5, size=n)  # Vp > Vs always
    rho = rng.uniform(1800.0, 2700.0, size=n)
    out = elastic_moduli(vp, vs, rho)
    assert np.all(out.poisson > -1.0)
    assert np.all(out.poisson <= 0.5)


def test_fluid_limit_gives_half_poisson():
    """As Vs -> 0 (true fluid) Poisson's ratio -> 0.5.

    We approach the limit rather than sitting on it, because the
    solver rejects vs == 0 outright.
    """
    out = elastic_moduli(vp=1500.0, vs=1.0, rho=1000.0)
    assert abs(float(out.poisson) - 0.5) < 1.0e-6


def test_elastic_moduli_rejects_non_positive_inputs():
    """Zero or negative Vp / Vs / rho raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        elastic_moduli(vp=0.0, vs=2000.0, rho=2200.0)
    with pytest.raises(ValueError, match="positive"):
        elastic_moduli(vp=3500.0, vs=-10.0, rho=2200.0)
    with pytest.raises(ValueError, match="positive"):
        elastic_moduli(vp=3500.0, vs=2000.0, rho=0.0)


def test_elastic_moduli_rejects_vs_ge_vp():
    """``vs >= vp`` is unphysical and raises ValueError."""
    with pytest.raises(ValueError, match="vs < vp"):
        elastic_moduli(vp=2000.0, vs=2500.0, rho=2200.0)
    with pytest.raises(ValueError, match="vs < vp"):
        elastic_moduli(vp=2500.0, vs=2500.0, rho=2200.0)


def test_vp_vs_ratio_is_just_division():
    """vp_vs_ratio is the obvious elementwise quotient."""
    vp = np.array([3500.0, 4000.0, 4500.0])
    vs = np.array([1800.0, 2000.0, 2500.0])
    ratio = vp_vs_ratio(vp, vs)
    assert np.allclose(ratio, vp / vs)


# ---------------------------------------------------------------------
# Reuss / Voigt / Hill mixing laws
# ---------------------------------------------------------------------


def test_voigt_reuss_bounds_ordered():
    """Reuss <= Hill <= Voigt for any non-degenerate mixture."""
    from fwap.rockphysics import hill_average, reuss_average, voigt_average

    # Two-phase quartz (K = 37 GPa) + clay (K = 25 GPa), equal volumes.
    moduli = np.array([37.0e9, 25.0e9])
    fractions = np.array([0.5, 0.5])
    r = reuss_average(moduli, fractions)
    v = voigt_average(moduli, fractions)
    h = hill_average(moduli, fractions)
    assert r < h < v
    assert abs(h - 0.5 * (r + v)) < 1.0e-6


def test_voigt_reuss_single_component_equals_modulus():
    """A one-component mixture gives back the modulus itself."""
    from fwap.rockphysics import hill_average, reuss_average, voigt_average

    moduli = np.array([37.0e9])
    fractions = np.array([1.0])
    assert voigt_average(moduli, fractions) == 37.0e9
    assert reuss_average(moduli, fractions) == 37.0e9
    assert hill_average(moduli, fractions) == 37.0e9


def test_voigt_known_value_two_phase_equal():
    """Voigt of equal volumes of 37 GPa and 25 GPa equals 31 GPa exactly."""
    from fwap.rockphysics import voigt_average

    moduli = np.array([37.0e9, 25.0e9])
    fractions = np.array([0.5, 0.5])
    assert abs(voigt_average(moduli, fractions) - 31.0e9) < 1.0e-3


def test_reuss_known_value_two_phase_equal():
    """Reuss harmonic mean of equal volumes of 37 GPa + 25 GPa.

    Hand-derived: 1 / (0.5/37e9 + 0.5/25e9) = 29.84 GPa to 3 sig figs.
    """
    from fwap.rockphysics import reuss_average

    moduli = np.array([37.0e9, 25.0e9])
    fractions = np.array([0.5, 0.5])
    expected = 1.0 / (0.5 / 37.0e9 + 0.5 / 25.0e9)
    assert abs(reuss_average(moduli, fractions) - expected) / expected < 1.0e-12


def test_mixing_law_rejects_fraction_sum_mismatch():
    """Fractions not summing to 1.0 raise ValueError."""
    import pytest

    from fwap.rockphysics import reuss_average

    with pytest.raises(ValueError, match="sum to 1"):
        reuss_average(np.array([10.0, 20.0]), np.array([0.3, 0.3]))


def test_mixing_law_rejects_non_positive_modulus():
    """Zero or negative modulus raises ValueError."""
    import pytest

    from fwap.rockphysics import voigt_average

    with pytest.raises(ValueError, match="positive"):
        voigt_average(np.array([10.0, 0.0]), np.array([0.5, 0.5]))


def test_mixing_law_rejects_negative_fraction():
    """Negative volume fractions raise ValueError."""
    import pytest

    from fwap.rockphysics import hill_average

    with pytest.raises(ValueError, match="non-negative"):
        hill_average(np.array([10.0, 20.0]), np.array([1.2, -0.2]))


def test_mixing_law_rejects_shape_mismatch():
    """Mismatched moduli / fractions shapes raise ValueError."""
    import pytest

    from fwap.rockphysics import reuss_average

    with pytest.raises(ValueError, match="shape"):
        reuss_average(np.array([10.0, 20.0, 30.0]), np.array([0.5, 0.5]))


# ---------------------------------------------------------------------
# Stoneley permeability indicator
# ---------------------------------------------------------------------


def test_stoneley_indicator_zero_on_tight_reference():
    """Observed == reference gives indicator 0."""
    from fwap.rockphysics import stoneley_permeability_indicator

    s_ref = 1.0 / 1400.0
    ind = stoneley_permeability_indicator(s_ref, s_ref)
    assert float(ind) == 0.0


def test_stoneley_indicator_positive_for_slower_stoneley():
    """Permeable zone (larger observed slowness) gives positive indicator."""
    from fwap.rockphysics import stoneley_permeability_indicator

    s_ref = 1.0 / 1400.0
    s_perm = 1.05 * s_ref  # 5% slower = more permeable
    ind = stoneley_permeability_indicator(s_perm, s_ref)
    assert abs(float(ind) - 0.05) < 1.0e-12


def test_stoneley_indicator_vector_input():
    """Array input produces a same-shape array output."""
    from fwap.rockphysics import stoneley_permeability_indicator

    s_ref = 1.0 / 1400.0
    observed = np.array([s_ref, 1.02 * s_ref, 1.1 * s_ref])
    ind = stoneley_permeability_indicator(observed, s_ref)
    assert ind.shape == observed.shape
    assert np.all(np.diff(ind) > 0)


def test_stoneley_indicator_per_depth_reference():
    """A per-depth reference baseline broadcasts elementwise."""
    from fwap.rockphysics import stoneley_permeability_indicator

    observed = np.array([7.0e-4, 7.1e-4, 7.2e-4])
    reference = np.array([7.0e-4, 7.0e-4, 7.0e-4])
    ind = stoneley_permeability_indicator(observed, reference)
    assert np.allclose(ind, [0.0, 0.1 / 7.0, 0.2 / 7.0], atol=1e-10)


def test_stoneley_indicator_rejects_non_positive():
    """Zero or negative slownesses raise ValueError."""
    import pytest

    from fwap.rockphysics import stoneley_permeability_indicator

    with pytest.raises(ValueError, match="observed"):
        stoneley_permeability_indicator(0.0, 7.0e-4)
    with pytest.raises(ValueError, match="reference"):
        stoneley_permeability_indicator(7.0e-4, -1.0)


# ---------------------------------------------------------------------
# stoneley_permeability_tang_cheng (Tang-Cheng-Toksoz 1991)
# ---------------------------------------------------------------------


# Standard Tang-Cheng-Toksoz parameter set used across the tests below.
# Water-filled borehole at typical Stoneley logging frequency, limestone
# matrix with K_phi = 30 GPa.
TC_K_F = 2.2e9  # Pa, water bulk modulus
TC_ETA = 1.0e-3  # Pa s, water viscosity
TC_RHO_F = 1000.0  # kg/m^3, water density
TC_FREQ = 1500.0  # Hz, typical Stoneley band
TC_K_PHI = 3.0e10  # Pa, limestone frame modulus
TC_DARCY = 9.869233e-13  # m^2 per darcy


def _tc_forward(
    kappa, phi, K_f=TC_K_F, K_phi=TC_K_PHI, eta=TC_ETA, rho_f=TC_RHO_F, freq=TC_FREQ
):
    """Forward model: predict alpha_ST from a known kappa array.

    Inverts the same closed form ``stoneley_permeability_tang_cheng``
    inverts, used here to round-trip-check the inversion.
    """
    omega = 2.0 * np.pi * freq
    omega_c = eta * phi / (kappa * rho_f)
    x2 = (omega / omega_c) ** 2
    A = K_f / (2.0 * K_phi)
    return A * x2 / (1.0 + x2)


# Round-trip recovery (the validation that anchors against
# Tang & Cheng 2004 fig 5.3)
# ---------------------------------------------------------------------


def test_tc_round_trip_recovers_known_permeability():
    """Forward-model alpha_ST from a known kappa profile, synthesise
    observed slowness, run the inversion, and confirm round-trip
    recovery to high precision. Profile mimics Tang & Cheng (2004)
    fig 5.3: tight limestone (~0.01-0.1 mD) bracketing a permeable
    bed (~1-2 darcy)."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_ref = 1.0 / 1500.0
    phi = np.array([0.05, 0.05, 0.20, 0.30, 0.20, 0.05, 0.05])
    kappa_true = np.array(
        [
            1.0e-17,  # tight, ~0.01 mD
            1.0e-16,  # tight, ~0.1 mD
            5.0e-13,  # ~0.5 darcy
            2.0e-12,  # ~2 darcy (peak)
            5.0e-13,  # ~0.5 darcy
            1.0e-16,  # tight
            1.0e-17,  # tight
        ]
    )
    alpha = _tc_forward(kappa_true, phi)
    s_obs = s_ref * (1.0 + alpha)
    K_phi = np.full_like(phi, TC_K_PHI)
    kappa_inv = stoneley_permeability_tang_cheng(
        s_obs,
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=phi,
        frame_bulk_modulus=K_phi,
    )
    # Round-trip relative error well under 0.1% across the band
    rel_err = np.abs(kappa_inv - kappa_true) / kappa_true
    assert np.all(rel_err < 1.0e-3)


def test_tc_recovers_tang_cheng_fig_5_3_orders_of_magnitude():
    """Recovered permeabilities span the expected darcy ranges from
    Tang & Cheng (2004) fig 5.3: tight ~0.01-0.1 mD, permeable
    ~1-2 darcy."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_ref = 1.0 / 1500.0
    phi = np.array([0.05, 0.30, 0.05])
    kappa_true = np.array([5.0e-17, 1.5e-12, 5.0e-17])  # 0.05 mD, 1.5 darcy, 0.05 mD
    alpha = _tc_forward(kappa_true, phi)
    s_obs = s_ref * (1.0 + alpha)
    K_phi = np.full_like(phi, TC_K_PHI)
    kappa_inv = stoneley_permeability_tang_cheng(
        s_obs,
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=phi,
        frame_bulk_modulus=K_phi,
    )
    kappa_md = kappa_inv / 9.869233e-16  # convert m^2 -> mD
    # Tight zones in 0.01 - 1.0 mD
    assert 0.01 < kappa_md[0] < 1.0
    assert 0.01 < kappa_md[2] < 1.0
    # Permeable zone in 0.5 - 5 darcy = 500-5000 mD
    assert 500.0 < kappa_md[1] < 5000.0


# Edge cases (zero / negative shift, out-of-model)
# ---------------------------------------------------------------------


def test_tc_zero_shift_gives_zero_permeability():
    """Tight zone (s_obs = s_ref, alpha_ST = 0) gives kappa = 0."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_ref = 1.0 / 1500.0
    kappa = stoneley_permeability_tang_cheng(
        np.array([s_ref]),
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=np.array([0.10]),
        frame_bulk_modulus=np.array([TC_K_PHI]),
    )
    assert kappa[0] == 0.0


def test_tc_negative_shift_clipped_to_zero():
    """Negative slowness shift (observed faster than reference;
    noise-driven or imperfect tight-reference) clipped to kappa = 0
    rather than raising or returning negative permeability."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_ref = 1.0 / 1500.0
    s_obs = s_ref * 0.99  # 1% faster than reference
    kappa = stoneley_permeability_tang_cheng(
        np.array([s_obs]),
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=np.array([0.10]),
        frame_bulk_modulus=np.array([TC_K_PHI]),
    )
    assert kappa[0] == 0.0


def test_tc_out_of_model_returns_nan():
    """Slowness shift exceeding the model upper bound A = K_f/(2 K_phi)
    returns NaN (typical cause: open fractures requiring the Hornby
    aperture model rather than the Biot-Rosenbaum matrix model)."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_ref = 1.0 / 1500.0
    A = TC_K_F / (2.0 * TC_K_PHI)
    s_obs = s_ref * (1.0 + 2.0 * A)  # well above A
    kappa = stoneley_permeability_tang_cheng(
        np.array([s_obs]),
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=np.array([0.10]),
        frame_bulk_modulus=np.array([TC_K_PHI]),
    )
    assert np.isnan(kappa[0])


# Monotonicity (more shift -> more permeability)
# ---------------------------------------------------------------------


def test_tc_recovered_kappa_monotonic_in_shift():
    """Larger fractional slowness shift produces larger recovered
    permeability across a representative band (within the model's
    valid alpha_ST range)."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_ref = 1.0 / 1500.0
    A = TC_K_F / (2.0 * TC_K_PHI)
    # Five increasing alpha_ST values, all below A
    alpha_grid = A * np.array([0.01, 0.1, 0.3, 0.6, 0.9])
    s_obs = s_ref * (1.0 + alpha_grid)
    phi = np.full_like(alpha_grid, 0.10)
    K_phi = np.full_like(alpha_grid, TC_K_PHI)
    kappa = stoneley_permeability_tang_cheng(
        s_obs,
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=phi,
        frame_bulk_modulus=K_phi,
    )
    assert np.all(np.diff(kappa) > 0)


# Input validation
# ---------------------------------------------------------------------


def test_tc_rejects_non_positive_scalar_inputs():
    """Each of frequency, K_f, eta, rho_f rejected when non-positive."""
    import pytest

    from fwap.rockphysics import stoneley_permeability_tang_cheng

    base = dict(
        slowness_observed=np.array([1.0e-3]),
        slowness_reference=1.0 / 1500.0,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=np.array([0.10]),
        frame_bulk_modulus=np.array([TC_K_PHI]),
    )
    with pytest.raises(ValueError, match="frequency"):
        stoneley_permeability_tang_cheng(**{**base, "frequency": 0.0})
    with pytest.raises(ValueError, match="fluid_bulk_modulus"):
        stoneley_permeability_tang_cheng(**{**base, "fluid_bulk_modulus": -1.0})
    with pytest.raises(ValueError, match="fluid_viscosity"):
        stoneley_permeability_tang_cheng(**{**base, "fluid_viscosity": 0.0})
    with pytest.raises(ValueError, match="fluid_density"):
        stoneley_permeability_tang_cheng(**{**base, "fluid_density": 0.0})


def test_tc_rejects_unphysical_porosity():
    """Porosity must be strictly in (0, 1)."""
    import pytest

    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_obs = np.array([1.0e-3])
    s_ref = 1.0 / 1500.0
    K_phi = np.array([TC_K_PHI])
    with pytest.raises(ValueError, match="porosity"):
        stoneley_permeability_tang_cheng(
            s_obs,
            s_ref,
            frequency=TC_FREQ,
            fluid_bulk_modulus=TC_K_F,
            fluid_viscosity=TC_ETA,
            fluid_density=TC_RHO_F,
            porosity=np.array([0.0]),  # zero not allowed
            frame_bulk_modulus=K_phi,
        )
    with pytest.raises(ValueError, match="porosity"):
        stoneley_permeability_tang_cheng(
            s_obs,
            s_ref,
            frequency=TC_FREQ,
            fluid_bulk_modulus=TC_K_F,
            fluid_viscosity=TC_ETA,
            fluid_density=TC_RHO_F,
            porosity=np.array([1.0]),  # 1.0 not allowed
            frame_bulk_modulus=K_phi,
        )


def test_tc_rejects_non_positive_slowness():
    """Slowness inputs must be strictly positive."""
    import pytest

    from fwap.rockphysics import stoneley_permeability_tang_cheng

    base_kwargs = dict(
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=np.array([0.10]),
        frame_bulk_modulus=np.array([TC_K_PHI]),
    )
    with pytest.raises(ValueError, match="slowness_observed"):
        stoneley_permeability_tang_cheng(np.array([0.0]), 1.0 / 1500.0, **base_kwargs)
    with pytest.raises(ValueError, match="slowness_reference"):
        stoneley_permeability_tang_cheng(np.array([1.0e-3]), -1.0, **base_kwargs)


# Output contract
# ---------------------------------------------------------------------


def test_tc_output_shape_matches_input():
    """Output array shape matches the (broadcast) input shape."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_obs = np.array([1.05, 1.10, 1.15, 1.20]) / 1500.0
    s_ref = 1.0 / 1500.0
    phi = np.array([0.10, 0.15, 0.20, 0.25])
    K_phi = np.full(4, TC_K_PHI)
    kappa = stoneley_permeability_tang_cheng(
        s_obs,
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=phi,
        frame_bulk_modulus=K_phi,
    )
    assert kappa.shape == (4,)


def test_tc_returns_si_units_m_squared():
    """Output should be in m^2 (SI), with magnitudes in the 1e-17
    to 1e-12 range for typical sonic Stoneley measurements."""
    from fwap.rockphysics import stoneley_permeability_tang_cheng

    s_ref = 1.0 / 1500.0
    A = TC_K_F / (2.0 * TC_K_PHI)
    s_obs = np.array([s_ref * (1.0 + 0.5 * A)])  # mid-range alpha
    kappa = stoneley_permeability_tang_cheng(
        s_obs,
        s_ref,
        frequency=TC_FREQ,
        fluid_bulk_modulus=TC_K_F,
        fluid_viscosity=TC_ETA,
        fluid_density=TC_RHO_F,
        porosity=np.array([0.10]),
        frame_bulk_modulus=np.array([TC_K_PHI]),
    )
    # Mid-range alpha gives a permeability in the millidarcy-to-darcy band
    assert 1.0e-17 < kappa[0] < 1.0e-9


# ---------------------------------------------------------------------
# stoneley_amplitude_fracture_indicator
# ---------------------------------------------------------------------


def test_stoneley_amplitude_indicator_zero_at_reference_amplitude():
    """A_obs == A_ref gives indicator = 0 (no fracture/permeability flagged)."""
    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    a_ref = 1.2
    ind = stoneley_amplitude_fracture_indicator(a_ref, a_ref)
    assert ind == 0.0


def test_stoneley_amplitude_indicator_positive_when_observed_attenuated():
    """Lower observed amplitude -> positive indicator (fractured / permeable)."""
    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    ind = stoneley_amplitude_fracture_indicator(
        amplitude_observed=0.6, amplitude_reference=1.0
    )
    # 1 - 0.6 / 1.0 = 0.4
    assert ind == pytest.approx(0.4)


def test_stoneley_amplitude_indicator_negative_when_amplified():
    """A_obs > A_ref gives a negative indicator (rare; resonance / SNR)."""
    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    ind = stoneley_amplitude_fracture_indicator(
        amplitude_observed=1.2, amplitude_reference=1.0
    )
    assert ind == pytest.approx(-0.2)


def test_stoneley_amplitude_indicator_handles_zero_observed():
    """A_obs = 0 (total attenuation) -> indicator = 1 (max fracture flag)."""
    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    ind = stoneley_amplitude_fracture_indicator(0.0, 1.0)
    assert ind == 1.0


def test_stoneley_amplitude_indicator_vector_input():
    """Vectorised: same shape, element-wise formula."""
    import numpy as np

    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    observed = np.array([1.0, 0.8, 0.5, 0.2])
    ref = 1.0
    ind = stoneley_amplitude_fracture_indicator(observed, ref)
    np.testing.assert_allclose(ind, [0.0, 0.2, 0.5, 0.8])


def test_stoneley_amplitude_indicator_per_depth_reference():
    """A per-depth reference baseline gives an element-wise fractional deficit."""
    import numpy as np

    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    observed = np.array([1.0, 0.7, 0.4])
    reference = np.array([1.0, 1.0, 0.8])
    ind = stoneley_amplitude_fracture_indicator(observed, reference)
    np.testing.assert_allclose(ind, [0.0, 0.3, 0.5])


def test_stoneley_amplitude_indicator_rejects_negative_observed():
    """Negative observed amplitude is not physical."""
    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    with pytest.raises(ValueError, match="observed"):
        stoneley_amplitude_fracture_indicator(-0.1, 1.0)


def test_stoneley_amplitude_indicator_rejects_non_positive_reference():
    """Zero or negative reference would divide-by-zero or flip the sign."""
    from fwap.rockphysics import stoneley_amplitude_fracture_indicator

    with pytest.raises(ValueError, match="reference"):
        stoneley_amplitude_fracture_indicator(0.5, 0.0)
    with pytest.raises(ValueError, match="reference"):
        stoneley_amplitude_fracture_indicator(0.5, -1.0)


# ---------------------------------------------------------------------
# Stoneley reflection-coefficient fracture-aperture inversion (Hornby 1989)
# ---------------------------------------------------------------------


def test_stoneley_reflection_coefficient_basic():
    """|R| = |A_r| / |A_i|, clipped to [0, 1]."""
    from fwap.rockphysics import stoneley_reflection_coefficient

    R = stoneley_reflection_coefficient(amplitude_incident=1.0, amplitude_reflected=0.3)
    assert R == pytest.approx(0.3)


def test_stoneley_reflection_coefficient_clips_to_unit_interval():
    """Noisy estimates can drift > 1; clip to keep |R| physical."""
    from fwap.rockphysics import stoneley_reflection_coefficient

    R = stoneley_reflection_coefficient(1.0, 1.05)
    assert R == 1.0


def test_stoneley_reflection_coefficient_takes_absolute_value():
    """Sign of incident / reflected is irrelevant."""
    import numpy as np

    from fwap.rockphysics import stoneley_reflection_coefficient

    R = stoneley_reflection_coefficient(np.array([-1.0, 1.0]), np.array([0.4, -0.4]))
    np.testing.assert_allclose(R, [0.4, 0.4])


def test_stoneley_reflection_coefficient_rejects_zero_incident():
    """Division by zero is not allowed."""
    from fwap.rockphysics import stoneley_reflection_coefficient

    with pytest.raises(ValueError, match="amplitude_incident"):
        stoneley_reflection_coefficient(0.0, 0.5)


def test_hornby_aperture_zero_R_gives_zero_aperture():
    """A non-reflecting depth has no fracture, so aperture = 0."""
    from fwap.rockphysics import hornby_fracture_aperture

    L = hornby_fracture_aperture(
        reflection_coefficient=0.0, frequency_hz=2000.0, stoneley_velocity=1400.0
    )
    assert L == 0.0


def test_hornby_aperture_round_trips_through_forward_model():
    """Plant L0, build |R|, recover L0 to floating-point precision."""
    import numpy as np

    from fwap.rockphysics import hornby_fracture_aperture

    f = 2000.0
    Vt = 1400.0
    omega = 2.0 * np.pi * f
    L0_planted = 1.0e-3  # 1 mm aperture
    # Forward: |R| = omega L / sqrt(V^2 + omega^2 L^2)
    R_forward = omega * L0_planted / np.sqrt(Vt**2 + omega**2 * L0_planted**2)
    L0_recovered = hornby_fracture_aperture(R_forward, f, Vt)
    assert L0_recovered == pytest.approx(L0_planted, rel=1.0e-12)


def test_hornby_aperture_small_amplitude_matches_full_for_small_R():
    """Small-amplitude approximation < 5% off the full form for |R| <= 0.3."""
    import numpy as np

    from fwap.rockphysics import hornby_fracture_aperture

    R = np.linspace(0.01, 0.30, 10)
    f, Vt = 2000.0, 1400.0
    L_full = hornby_fracture_aperture(R, f, Vt, small_amplitude_approx=False)
    L_small = hornby_fracture_aperture(R, f, Vt, small_amplitude_approx=True)
    # Full form is L = V|R|/(omega sqrt(1-R^2)); small-amp drops the
    # radical, so L_small / L_full = sqrt(1 - R^2) <= 1.
    rel_err = np.abs(L_full - L_small) / L_full
    assert rel_err.max() < 0.05


def test_hornby_aperture_diverges_at_unit_reflection():
    """|R| -> 1 saturates the inversion at +inf."""
    import numpy as np

    from fwap.rockphysics import hornby_fracture_aperture

    L = hornby_fracture_aperture(1.0, 2000.0, 1400.0)
    assert np.isinf(L)


def test_hornby_aperture_vector_inputs_broadcast():
    """Per-fracture |R| array gives a per-fracture aperture array."""
    import numpy as np

    from fwap.rockphysics import hornby_fracture_aperture

    R = np.array([0.0, 0.1, 0.2, 0.3])
    L = hornby_fracture_aperture(R, frequency_hz=2000.0, stoneley_velocity=1400.0)
    assert L.shape == (4,)
    # Aperture is monotonic in |R|.
    assert np.all(np.diff(L) > 0)
    assert L[0] == 0.0


def test_hornby_aperture_rejects_R_out_of_range():
    """Reflection coefficient outside [0, 1] is unphysical."""
    from fwap.rockphysics import hornby_fracture_aperture

    with pytest.raises(ValueError, match="reflection_coefficient"):
        hornby_fracture_aperture(-0.1, 2000.0, 1400.0)
    with pytest.raises(ValueError, match="reflection_coefficient"):
        hornby_fracture_aperture(1.5, 2000.0, 1400.0)


def test_hornby_aperture_rejects_non_positive_frequency_or_velocity():
    """f and V_T must be strictly positive."""
    from fwap.rockphysics import hornby_fracture_aperture

    with pytest.raises(ValueError, match="frequency_hz"):
        hornby_fracture_aperture(0.3, 0.0, 1400.0)
    with pytest.raises(ValueError, match="stoneley_velocity"):
        hornby_fracture_aperture(0.3, 2000.0, -1.0)


def test_hornby_aperture_round_trips_through_write_las(tmp_path):
    """RFRAC and FAPER mnemonics carry correct units in LAS round-trip."""
    import numpy as np

    from fwap.io import read_las, write_las
    from fwap.rockphysics import hornby_fracture_aperture

    n = 5
    depth = np.linspace(1000.0, 1004.0, n)
    R = np.array([0.0, 0.1, 0.25, 0.05, 0.0])
    L = hornby_fracture_aperture(R, frequency_hz=2000.0, stoneley_velocity=1400.0)
    curves = {"RFRAC": R, "FAPER": L}
    path = str(tmp_path / "fracture.las")
    write_las(path, depth, curves, well_name="FRACWELL")
    loaded = read_las(path)
    assert loaded.units["RFRAC"] == ""
    assert loaded.units["FAPER"] == "m"


def test_stoneley_amplitude_indicator_complements_slowness_indicator():
    """Both indicators agree on which depths are fractured / permeable."""
    import numpy as np

    from fwap.rockphysics import (
        stoneley_amplitude_fracture_indicator,
        stoneley_permeability_indicator,
    )

    # Synthetic model: depths 0, 2 are tight; 1, 3 are progressively
    # fractured. Slowness shifts up and amplitude drops at the
    # fractured depths.
    s_ref = 7.0e-4
    a_ref = 1.0
    slowness = np.array([s_ref, 7.7e-4, s_ref, 8.4e-4])
    amplitude = np.array([a_ref, 0.7, a_ref, 0.4])
    perm = stoneley_permeability_indicator(slowness, s_ref)
    frac = stoneley_amplitude_fracture_indicator(amplitude, a_ref)
    # Both indicators are zero at the tight depths and positive (and
    # increasing) at the fractured depths.
    np.testing.assert_allclose(perm[[0, 2]], 0.0)
    np.testing.assert_allclose(frac[[0, 2]], 0.0)
    assert perm[1] > 0 and perm[3] > perm[1]
    assert frac[1] > 0 and frac[3] > frac[1]
    # The two flag the *same* depths; their rank order matches.
    np.testing.assert_array_equal(np.argsort(perm), np.argsort(frac))


# ---------------------------------------------------------------------
# Gassmann fluid substitution
# ---------------------------------------------------------------------


def test_gassmann_known_value_brine_sand():
    """Brine-saturated quartz sandstone: Mavko-handbook-style numbers.

    K_dry=15 GPa, mu_dry=10 GPa, K_s=37 GPa (quartz), K_f=2.2 GPa
    (brine), phi=0.2. Hand-derived from the closed form:

        K_sat = 15 + (1 - 15/37)^2 / (0.2/2.2 + 0.8/37 - 15/37^2) GPa
              ~ 18.481 GPa.
    """
    from fwap.rockphysics import GassmannResult, gassmann_fluid_substitution

    out = gassmann_fluid_substitution(
        k_dry=15.0e9,
        mu_dry=10.0e9,
        k_mineral=37.0e9,
        k_fluid=2.2e9,
        porosity=0.2,
    )
    assert isinstance(out, GassmannResult)
    assert abs(float(out.k_sat) - 18.4806e9) / 18.4806e9 < 1.0e-4
    # Gassmann's central fact: shear modulus is fluid-insensitive.
    assert float(out.mu_sat) == 10.0e9


def test_gassmann_mu_sat_equals_mu_dry_exactly():
    """mu_sat must equal mu_dry bit-for-bit across arbitrary inputs."""
    from fwap.rockphysics import gassmann_fluid_substitution

    rng = np.random.default_rng(0)
    n = 10
    mu_dry = rng.uniform(5.0e9, 20.0e9, size=n)
    out = gassmann_fluid_substitution(
        k_dry=rng.uniform(5.0e9, 30.0e9, size=n),
        mu_dry=mu_dry,
        k_mineral=37.0e9,
        k_fluid=2.2e9,
        porosity=rng.uniform(0.05, 0.3, size=n),
    )
    np.testing.assert_array_equal(out.mu_sat, mu_dry)


def test_gassmann_zero_porosity_recovers_mineral_modulus():
    """phi = 0 collapses Gassmann to K_sat = K_mineral regardless of fluid."""
    from fwap.rockphysics import gassmann_fluid_substitution

    out = gassmann_fluid_substitution(
        k_dry=20.0e9,
        mu_dry=15.0e9,
        k_mineral=37.0e9,
        k_fluid=2.2e9,
        porosity=0.0,
    )
    assert abs(float(out.k_sat) - 37.0e9) / 37.0e9 < 1.0e-12


def test_gassmann_k_dry_equals_k_mineral_gives_mineral():
    """A solid mineral with no compliant pore space: K_sat = K_s."""
    from fwap.rockphysics import gassmann_fluid_substitution

    out = gassmann_fluid_substitution(
        k_dry=37.0e9,
        mu_dry=44.0e9,
        k_mineral=37.0e9,
        k_fluid=2.2e9,
        porosity=0.15,
    )
    assert abs(float(out.k_sat) - 37.0e9) / 37.0e9 < 1.0e-12


def test_gassmann_soft_fluid_limit_approaches_k_dry():
    """As K_f -> 0, K_sat -> K_dry (vacuum limit)."""
    from fwap.rockphysics import gassmann_fluid_substitution

    k_dry = 15.0e9
    out = gassmann_fluid_substitution(
        k_dry=k_dry,
        mu_dry=10.0e9,
        k_mineral=37.0e9,
        k_fluid=1.0e3,  # ~ gas at low pressure
        porosity=0.2,
    )
    assert abs(float(out.k_sat) - k_dry) / k_dry < 1.0e-3


def test_gassmann_stiffer_fluid_gives_stiffer_rock():
    """Replacing gas with brine must increase K_sat (monotone)."""
    from fwap.rockphysics import gassmann_fluid_substitution

    common = dict(k_dry=15.0e9, mu_dry=10.0e9, k_mineral=37.0e9, porosity=0.2)
    k_gas = gassmann_fluid_substitution(k_fluid=0.05e9, **common).k_sat
    k_oil = gassmann_fluid_substitution(k_fluid=1.0e9, **common).k_sat
    k_brine = gassmann_fluid_substitution(k_fluid=2.2e9, **common).k_sat
    assert float(k_gas) < float(k_oil) < float(k_brine)


def test_gassmann_broadcast_over_depth():
    """Arrays in one input, scalars in the others: broadcast cleanly."""
    from fwap.rockphysics import gassmann_fluid_substitution

    n = 7
    k_dry = np.linspace(10.0e9, 20.0e9, n)
    out = gassmann_fluid_substitution(
        k_dry=k_dry,
        mu_dry=10.0e9,
        k_mineral=37.0e9,
        k_fluid=2.2e9,
        porosity=0.2,
    )
    assert out.k_sat.shape == (n,)
    assert out.mu_sat.shape == (n,)
    # Monotone in K_dry: stiffer frame -> stiffer saturated rock.
    assert np.all(np.diff(out.k_sat) > 0)


def test_gassmann_rejects_non_positive_moduli():
    """Non-positive moduli raise ValueError with a useful message."""
    import pytest

    from fwap.rockphysics import gassmann_fluid_substitution

    base = dict(mu_dry=10.0e9, k_mineral=37.0e9, k_fluid=2.2e9, porosity=0.2)
    with pytest.raises(ValueError, match="k_dry"):
        gassmann_fluid_substitution(k_dry=0.0, **base)
    with pytest.raises(ValueError, match="mu_dry"):
        gassmann_fluid_substitution(
            k_dry=15.0e9,
            mu_dry=-1.0,
            k_mineral=37.0e9,
            k_fluid=2.2e9,
            porosity=0.2,
        )
    with pytest.raises(ValueError, match="k_mineral"):
        gassmann_fluid_substitution(
            k_dry=15.0e9,
            mu_dry=10.0e9,
            k_mineral=0.0,
            k_fluid=2.2e9,
            porosity=0.2,
        )
    with pytest.raises(ValueError, match="k_fluid"):
        gassmann_fluid_substitution(
            k_dry=15.0e9,
            mu_dry=10.0e9,
            k_mineral=37.0e9,
            k_fluid=0.0,
            porosity=0.2,
        )


def test_gassmann_rejects_bad_porosity():
    """Porosity outside [0, 1] raises ValueError."""
    import pytest

    from fwap.rockphysics import gassmann_fluid_substitution

    base = dict(k_dry=15.0e9, mu_dry=10.0e9, k_mineral=37.0e9, k_fluid=2.2e9)
    with pytest.raises(ValueError, match="porosity"):
        gassmann_fluid_substitution(porosity=-0.01, **base)
    with pytest.raises(ValueError, match="porosity"):
        gassmann_fluid_substitution(porosity=1.01, **base)


def test_gassmann_rejects_dry_stiffer_than_mineral():
    """K_dry > K_mineral is unphysical and raises ValueError."""
    import pytest

    from fwap.rockphysics import gassmann_fluid_substitution

    with pytest.raises(ValueError, match="mineral"):
        gassmann_fluid_substitution(
            k_dry=40.0e9,
            mu_dry=10.0e9,
            k_mineral=37.0e9,
            k_fluid=2.2e9,
            porosity=0.2,
        )


# ---------------------------------------------------------------------
# vs_from_stoneley_slow_formation
# ---------------------------------------------------------------------


def test_vs_from_stoneley_round_trips_through_white_formula():
    """Plant V_S, build matching Stoneley slowness, recover V_S exactly."""
    from fwap.rockphysics import vs_from_stoneley_slow_formation

    rho_f, v_f = 1000.0, 1500.0
    rho = 2200.0
    Vs_planted = 1100.0  # slow formation: V_S < V_fluid
    mu = rho * Vs_planted**2
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / mu)
    Vs = vs_from_stoneley_slow_formation(s_st, rho, rho_fluid=rho_f, v_fluid=v_f)
    assert Vs == pytest.approx(Vs_planted, rel=1.0e-12)


def test_vs_from_stoneley_works_in_fast_formation_too():
    """Formula is general; the 'slow_formation' name is a use-case label.
    A planted fast-formation V_S round-trips just as cleanly.
    """
    from fwap.rockphysics import vs_from_stoneley_slow_formation

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vs_planted = 2500.0  # fast formation
    mu = rho * Vs_planted**2
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / mu)
    Vs = vs_from_stoneley_slow_formation(s_st, rho, rho_fluid=rho_f, v_fluid=v_f)
    assert Vs == pytest.approx(Vs_planted, rel=1.0e-12)


def test_vs_from_stoneley_vector_input_broadcasts():
    """Per-depth Stoneley slowness + density -> per-depth V_S."""
    from fwap.rockphysics import vs_from_stoneley_slow_formation

    rho_f, v_f = 1000.0, 1500.0
    Vs_planted = np.array([800.0, 1100.0, 1400.0])
    rho = np.full_like(Vs_planted, 2200.0)
    mu = rho * Vs_planted**2
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / mu)
    Vs = vs_from_stoneley_slow_formation(s_st, rho, rho_fluid=rho_f, v_fluid=v_f)
    assert Vs.shape == (3,)
    np.testing.assert_allclose(Vs, Vs_planted, rtol=1.0e-12)


def test_vs_from_stoneley_consistent_with_c66_helper():
    """Vs == sqrt(C66 / rho) with C66 from anisotropy.stoneley_horizontal_shear_modulus."""
    from fwap.anisotropy import stoneley_horizontal_shear_modulus
    from fwap.rockphysics import vs_from_stoneley_slow_formation

    rho_f, v_f = 1000.0, 1500.0
    rho = 2200.0
    Vs_planted = 1200.0
    mu = rho * Vs_planted**2
    s_st = np.sqrt(1.0 / v_f**2 + rho_f / mu)
    Vs = vs_from_stoneley_slow_formation(s_st, rho, rho_fluid=rho_f, v_fluid=v_f)
    c66 = stoneley_horizontal_shear_modulus(s_st, rho_fluid=rho_f, v_fluid=v_f)
    np.testing.assert_allclose(Vs, np.sqrt(c66 / rho), rtol=1.0e-12)


def test_vs_from_stoneley_rejects_slowness_below_fluid_slowness():
    """Stoneley slowness <= fluid slowness is unphysical."""
    from fwap.rockphysics import vs_from_stoneley_slow_formation

    rho_f, v_f = 1000.0, 1500.0
    s_f = 1.0 / v_f
    with pytest.raises(ValueError, match="v_fluid"):
        vs_from_stoneley_slow_formation(s_f, 2200.0, rho_fluid=rho_f, v_fluid=v_f)
    with pytest.raises(ValueError, match="v_fluid"):
        vs_from_stoneley_slow_formation(0.5 * s_f, 2200.0, rho_fluid=rho_f, v_fluid=v_f)


def test_vs_from_stoneley_rejects_non_positive_inputs():
    """All inputs must be strictly positive."""
    from fwap.rockphysics import vs_from_stoneley_slow_formation

    with pytest.raises(ValueError, match="rho_fluid"):
        vs_from_stoneley_slow_formation(8.0e-4, 2200.0, rho_fluid=0.0, v_fluid=1500.0)
    with pytest.raises(ValueError, match="v_fluid"):
        vs_from_stoneley_slow_formation(8.0e-4, 2200.0, rho_fluid=1000.0, v_fluid=-1.0)
    with pytest.raises(ValueError, match="slowness_stoneley"):
        vs_from_stoneley_slow_formation(0.0, 2200.0, rho_fluid=1000.0, v_fluid=1500.0)
    with pytest.raises(ValueError, match="rho"):
        vs_from_stoneley_slow_formation(8.0e-4, 0.0, rho_fluid=1000.0, v_fluid=1500.0)


def test_vs_from_stoneley_lower_when_formation_softer():
    """At fixed Stoneley slowness, lower density -> higher Vs (mu fixed)."""
    from fwap.rockphysics import vs_from_stoneley_slow_formation

    rho_f, v_f = 1000.0, 1500.0
    s_st = 8.0e-4  # 1250 m/s tube wave -- typical slow formation
    Vs_high_rho = vs_from_stoneley_slow_formation(
        s_st, rho=2600.0, rho_fluid=rho_f, v_fluid=v_f
    )
    Vs_low_rho = vs_from_stoneley_slow_formation(
        s_st, rho=2000.0, rho_fluid=rho_f, v_fluid=v_f
    )
    # Lower formation density and the same modulus gives higher Vs.
    assert Vs_low_rho > Vs_high_rho


# ---------------------------------------------------------------------
# stoneley_fracture_density (unified four-indicator combiner)
# ---------------------------------------------------------------------


def test_fd_zero_indicators_give_zero_score():
    """Tight zone (all indicators at zero) gives score = 0."""
    from fwap.rockphysics import stoneley_fracture_density

    fi = stoneley_fracture_density(
        slowness_indicator=np.zeros(5),
        amplitude_indicator=np.zeros(5),
    )
    assert np.all(fi == 0.0)


def test_fd_score_in_unit_interval():
    """Score is always clipped to [0, 1]."""
    from fwap.rockphysics import stoneley_fracture_density

    rng = np.random.default_rng(0)
    n = 30
    alpha_s = rng.uniform(-0.05, 0.5, n)  # mix of negative and large positives
    alpha_a = rng.uniform(-0.1, 1.2, n)
    fi = stoneley_fracture_density(alpha_s, alpha_a)
    assert np.all(fi >= 0.0)
    assert np.all(fi <= 1.0)


def test_fd_slowness_only_contributes_with_default_weights():
    """With default weights and only slowness_indicator supplied,
    the score reduces to ``clip(0.5 * alpha_s / 0.1, 0, 1)``."""
    from fwap.rockphysics import stoneley_fracture_density

    alpha_s = np.array([0.0, 0.05, 0.10, 0.20, 0.30])
    fi = stoneley_fracture_density(alpha_s)
    expected = np.clip(0.5 * alpha_s / 0.1, 0.0, 1.0)
    assert np.allclose(fi, expected)


def test_fd_amplitude_only_path():
    """Score is ``0.5 * alpha_a`` clipped when only amplitude
    indicator is supplied (slowness defaulted to zeros)."""
    from fwap.rockphysics import stoneley_fracture_density

    alpha_a = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    fi = stoneley_fracture_density(
        slowness_indicator=np.zeros_like(alpha_a),
        amplitude_indicator=alpha_a,
    )
    expected = np.clip(0.5 * alpha_a, 0.0, 1.0)
    assert np.allclose(fi, expected)


def test_fd_matrix_partitioning_suppresses_matrix_only_zones():
    """When matrix_permeability is supplied, depths with finite
    kappa (matrix-explained) get the slowness contribution
    suppressed; depths with NaN kappa (matrix model failed) keep
    the full slowness contribution."""
    from fwap.rockphysics import stoneley_fracture_density

    alpha_s = np.array([0.05, 0.05, 0.05])
    alpha_a = np.array([0.0, 0.0, 0.0])
    # Three zones: matrix-explained (finite), out-of-model (NaN),
    # matrix-explained (finite). Kappa values are arbitrary so long
    # as their finite/NaN status is correct.
    kappa = np.array([1.0e-13, np.nan, 1.0e-13])
    fi = stoneley_fracture_density(alpha_s, alpha_a, matrix_permeability=kappa)
    # Matrix-explained: slowness contribution suppressed -> 0.
    assert fi[0] == 0.0
    assert fi[2] == 0.0
    # Fracture-suspected: full slowness contribution.
    expected = 0.5 * 0.05 / 0.1
    assert abs(fi[1] - expected) < 1.0e-12


def test_fd_aperture_term_saturates_with_tanh():
    """Aperture contribution saturates via tanh: a 1 mm aperture
    contributes ~0.5*tanh(1) ~ 0.38 with default settings; 5 mm
    saturates near 0.5*tanh(5) ~ 0.5."""
    from fwap.rockphysics import stoneley_fracture_density

    alpha_s = np.zeros(3)
    alpha_a = np.zeros(3)
    apertures = np.array([0.0, 1.0e-3, 5.0e-3])
    fi = stoneley_fracture_density(
        alpha_s,
        alpha_a,
        fracture_aperture=apertures,
        aperture_weight=0.5,
    )
    expected = np.clip(0.5 * np.tanh(apertures / 1.0e-3), 0.0, 1.0)
    assert np.allclose(fi, expected)


def test_fd_aperture_default_weight_zero_means_no_contribution():
    """Default ``aperture_weight=0.0`` means the aperture term
    contributes zero, even when ``fracture_aperture`` is supplied."""
    from fwap.rockphysics import stoneley_fracture_density

    alpha_s = np.zeros(2)
    alpha_a = np.zeros(2)
    fi = stoneley_fracture_density(
        alpha_s,
        alpha_a,
        fracture_aperture=np.array([1.0e-3, 5.0e-3]),
    )
    assert np.all(fi == 0.0)


def test_fd_aperture_with_nan_treated_as_no_contribution():
    """NaN apertures (no Hornby reflection coefficient available)
    contribute zero; finite apertures contribute via tanh."""
    from fwap.rockphysics import stoneley_fracture_density

    fi = stoneley_fracture_density(
        slowness_indicator=np.zeros(2),
        amplitude_indicator=np.zeros(2),
        fracture_aperture=np.array([np.nan, 2.0e-3]),
        aperture_weight=0.5,
    )
    assert fi[0] == 0.0
    expected_1 = 0.5 * np.tanh(2.0)
    assert abs(fi[1] - expected_1) < 1.0e-12


def test_fd_combined_score_increases_with_each_indicator():
    """Increasing any one indicator (others held fixed) must not
    decrease the score. Partial-monotonicity check."""
    from fwap.rockphysics import stoneley_fracture_density

    base = stoneley_fracture_density(
        slowness_indicator=np.array([0.05]),
        amplitude_indicator=np.array([0.20]),
    )
    higher_s = stoneley_fracture_density(
        slowness_indicator=np.array([0.10]),
        amplitude_indicator=np.array([0.20]),
    )
    higher_a = stoneley_fracture_density(
        slowness_indicator=np.array([0.05]),
        amplitude_indicator=np.array([0.40]),
    )
    assert higher_s[0] >= base[0]
    assert higher_a[0] >= base[0]


def test_fd_rejects_negative_weights():
    """Negative weights raise ValueError."""
    import pytest

    from fwap.rockphysics import stoneley_fracture_density

    alpha_s = np.array([0.05])
    with pytest.raises(ValueError, match="slowness_weight"):
        stoneley_fracture_density(alpha_s, slowness_weight=-0.1)
    with pytest.raises(ValueError, match="amplitude_weight"):
        stoneley_fracture_density(alpha_s, amplitude_weight=-0.1)
    with pytest.raises(ValueError, match="aperture_weight"):
        stoneley_fracture_density(alpha_s, aperture_weight=-0.1)


def test_fd_rejects_non_positive_scales():
    """Zero or negative scales raise ValueError."""
    import pytest

    from fwap.rockphysics import stoneley_fracture_density

    alpha_s = np.array([0.05])
    with pytest.raises(ValueError, match="slowness_scale"):
        stoneley_fracture_density(alpha_s, slowness_scale=0.0)
    with pytest.raises(ValueError, match="aperture_scale"):
        stoneley_fracture_density(alpha_s, aperture_scale=0.0)


def test_fd_rejects_shape_mismatch():
    """Optional arrays with shape mismatching slowness_indicator
    raise ValueError."""
    import pytest

    from fwap.rockphysics import stoneley_fracture_density

    alpha_s = np.array([0.05, 0.10])
    with pytest.raises(ValueError, match="amplitude_indicator"):
        stoneley_fracture_density(
            alpha_s,
            amplitude_indicator=np.array([0.20]),  # wrong shape
        )
    with pytest.raises(ValueError, match="matrix_permeability"):
        stoneley_fracture_density(
            alpha_s,
            matrix_permeability=np.array([1.0e-13]),  # wrong shape
        )
    with pytest.raises(ValueError, match="fracture_aperture"):
        stoneley_fracture_density(
            alpha_s,
            aperture_weight=0.1,
            fracture_aperture=np.array([1.0e-3]),  # wrong shape
        )
