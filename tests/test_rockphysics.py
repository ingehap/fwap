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
    mu = rho * vs ** 2                       # 8.80e9
    lam = rho * vp ** 2 - 2.0 * mu           # 9.34e9
    k = lam + 2.0 / 3.0 * mu                 # 15.21e9
    young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)   # 21.35e9
    nu = lam / (2.0 * (lam + mu))            # 0.258
    out = elastic_moduli(vp, vs, rho)
    assert isinstance(out, ElasticModuli)
    for actual, expected, name in [
        (out.mu,      mu,    "mu"),
        (out.lambda_, lam,   "lambda"),
        (out.k,       k,     "k"),
        (out.young,   young, "young"),
        (out.poisson, nu,    "poisson"),
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
    vp = vs * rng.uniform(1.4, 2.5, size=n)   # Vp > Vs always
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
    s_perm = 1.05 * s_ref        # 5% slower = more permeable
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
        amplitude_observed=0.6, amplitude_reference=1.0)
    # 1 - 0.6 / 1.0 = 0.4
    assert ind == pytest.approx(0.4)


def test_stoneley_amplitude_indicator_negative_when_amplified():
    """A_obs > A_ref gives a negative indicator (rare; resonance / SNR)."""
    from fwap.rockphysics import stoneley_amplitude_fracture_indicator
    ind = stoneley_amplitude_fracture_indicator(
        amplitude_observed=1.2, amplitude_reference=1.0)
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
    R = stoneley_reflection_coefficient(amplitude_incident=1.0,
                                        amplitude_reflected=0.3)
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
    R = stoneley_reflection_coefficient(np.array([-1.0, 1.0]),
                                        np.array([0.4, -0.4]))
    np.testing.assert_allclose(R, [0.4, 0.4])


def test_stoneley_reflection_coefficient_rejects_zero_incident():
    """Division by zero is not allowed."""
    from fwap.rockphysics import stoneley_reflection_coefficient
    with pytest.raises(ValueError, match="amplitude_incident"):
        stoneley_reflection_coefficient(0.0, 0.5)


def test_hornby_aperture_zero_R_gives_zero_aperture():
    """A non-reflecting depth has no fracture, so aperture = 0."""
    from fwap.rockphysics import hornby_fracture_aperture
    L = hornby_fracture_aperture(reflection_coefficient=0.0,
                                 frequency_hz=2000.0,
                                 stoneley_velocity=1400.0)
    assert L == 0.0


def test_hornby_aperture_round_trips_through_forward_model():
    """Plant L0, build |R|, recover L0 to floating-point precision."""
    import numpy as np
    from fwap.rockphysics import hornby_fracture_aperture
    f = 2000.0
    Vt = 1400.0
    omega = 2.0 * np.pi * f
    L0_planted = 1.0e-3   # 1 mm aperture
    # Forward: |R| = omega L / sqrt(V^2 + omega^2 L^2)
    R_forward = (omega * L0_planted
                 / np.sqrt(Vt ** 2 + omega ** 2 * L0_planted ** 2))
    L0_recovered = hornby_fracture_aperture(R_forward, f, Vt)
    assert L0_recovered == pytest.approx(L0_planted, rel=1.0e-12)


def test_hornby_aperture_small_amplitude_matches_full_for_small_R():
    """Small-amplitude approximation < 5% off the full form for |R| <= 0.3."""
    import numpy as np
    from fwap.rockphysics import hornby_fracture_aperture
    R = np.linspace(0.01, 0.30, 10)
    f, Vt = 2000.0, 1400.0
    L_full  = hornby_fracture_aperture(R, f, Vt, small_amplitude_approx=False)
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
    L = hornby_fracture_aperture(R, frequency_hz=2000.0,
                                 stoneley_velocity=1400.0)
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
    L = hornby_fracture_aperture(R, frequency_hz=2000.0,
                                 stoneley_velocity=1400.0)
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
    np.testing.assert_array_equal(
        np.argsort(perm), np.argsort(frac)
    )


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
        k_dry=15.0e9, mu_dry=10.0e9,
        k_mineral=37.0e9, k_fluid=2.2e9, porosity=0.2,
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
        k_dry=20.0e9, mu_dry=15.0e9,
        k_mineral=37.0e9, k_fluid=2.2e9, porosity=0.0,
    )
    assert abs(float(out.k_sat) - 37.0e9) / 37.0e9 < 1.0e-12


def test_gassmann_k_dry_equals_k_mineral_gives_mineral():
    """A solid mineral with no compliant pore space: K_sat = K_s."""
    from fwap.rockphysics import gassmann_fluid_substitution
    out = gassmann_fluid_substitution(
        k_dry=37.0e9, mu_dry=44.0e9,
        k_mineral=37.0e9, k_fluid=2.2e9, porosity=0.15,
    )
    assert abs(float(out.k_sat) - 37.0e9) / 37.0e9 < 1.0e-12


def test_gassmann_soft_fluid_limit_approaches_k_dry():
    """As K_f -> 0, K_sat -> K_dry (vacuum limit)."""
    from fwap.rockphysics import gassmann_fluid_substitution
    k_dry = 15.0e9
    out = gassmann_fluid_substitution(
        k_dry=k_dry, mu_dry=10.0e9,
        k_mineral=37.0e9, k_fluid=1.0e3,   # ~ gas at low pressure
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
        k_dry=k_dry, mu_dry=10.0e9,
        k_mineral=37.0e9, k_fluid=2.2e9, porosity=0.2,
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
            k_dry=15.0e9, mu_dry=-1.0,
            k_mineral=37.0e9, k_fluid=2.2e9, porosity=0.2,
        )
    with pytest.raises(ValueError, match="k_mineral"):
        gassmann_fluid_substitution(
            k_dry=15.0e9, mu_dry=10.0e9,
            k_mineral=0.0, k_fluid=2.2e9, porosity=0.2,
        )
    with pytest.raises(ValueError, match="k_fluid"):
        gassmann_fluid_substitution(
            k_dry=15.0e9, mu_dry=10.0e9,
            k_mineral=37.0e9, k_fluid=0.0, porosity=0.2,
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
            k_dry=40.0e9, mu_dry=10.0e9,
            k_mineral=37.0e9, k_fluid=2.2e9, porosity=0.2,
        )
