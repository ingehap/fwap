"""
Sonic-derived geomechanics indices: brittleness, fracability,
unconfined compressive strength, sand stability, closure stress,
plus a one-call :func:`geomechanics_indices` bundler that wraps them
into a :class:`GeomechanicsIndices` result.

All inputs are raw elastic-moduli / stress arrays in SI units; the
caller is responsible for applying any dynamic-to-static correction
upstream (sonic-derived Young's modulus is the *dynamic* value while
the published correlations were calibrated against *static* core
measurements; see the module-level caveat in :mod:`fwap.geomechanics`).

**What that costs, since "the caller is responsible" is not a
magnitude.** Static ``E`` runs 20-40 % below dynamic for shales
(Rybacki et al. 2016) and a factor 1.5-3 below it for porous rocks
(Mavko et al. 2009). Skipping the correction overstates
:func:`brittleness_index_rickman` by 8-17 index points and makes
:func:`unconfined_compressive_strength` an upper bound. Both functions
carry a *Static vs dynamic* section with the worked numbers, and both
sets of numbers are executed in ``tests/test_geomechanics.py``.

References
----------
* Rickman, R., Mullen, M. J., Petre, J. E., Grieser, W. V., &
  Kundert, D. (2008). SPE 115258 (brittleness and fracability).
* Rybacki, E., Meier, T., & Dresen, G. (2016). What controls the
  mechanical properties of shale rocks? -- Part II: Brittleness.
  *J. Petroleum Science and Engineering* 144, 39-58. Table A3 records
  Rickman's normalisation bounds and that they are calibrated on
  *static* moduli; sect. 4 gives the 20-40 % static-dynamic gap. Used
  here because SPE 115258 itself is paywalled.
* Lacy, L. L. (1997). SPE 38716. Cited by `unconfined_compressive_
  strength` for its sandstone UCS form; **unverified** -- the
  attribution chain that named it was checked and broke (see that
  function). Paywalled.
* Chang, C., Zoback, M. D., & Khaksar, A. (2006). Empirical relations
  between rock strength and physical properties in sedimentary rocks.
  *J. Petroleum Science and Engineering* 51, 223-237. Its Table 1 is
  the compilation the UCS docstring used to claim; it does not contain
  that formula.
* Eaton, B. A. (1969). *J. Petroleum Technology* 21(10), 1353-1360
  (uniaxial-strain closure stress).
* Bratli, R. K., & Risnes, R. (1981). *SPE J.* 21(2), 236-248 (sand
  stability heuristic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fwap.rockphysics import ElasticModuli

UCSModel = Literal["lacy_sandstone"]

# Default Rickman normalisation bounds, in SI units. Rickman et al.
# (2008) use E in 1-8 Mpsi and nu in 0.15-0.40; converted at
# 1 Mpsi = 6.894757 GPa.
#
# **These were wrong until the source was checked**, by a factor of
# 1.450 on both E bounds. They read 1.0e10 and 8.0e10 -- the paper's
# *numerals* with "Mpsi" swapped for "1e10 Pa", so the conversion was
# written down beside them and then not performed. The mistake shipped
# because everything around it was self-consistent: the index stayed
# monotone and inside [0, 1], and the trailing comments correctly
# described the wrong numbers.
#
# The bounds are confirmed three ways, none of them SPE 115258 itself
# (paywalled). Rybacki et al. (2016) state them twice, once as
# "E max = 8 Mpsi (~55 GPa), nu min = 0.15, E min = 1 Mpsi (~7 GPa),
# nu max = 0.4 (Rickman et al., 2008)" and again as "the limits
# (E min = 7 GPa, E max = 55 GPa) suggested by Rickman et al. (2008)".
# Independently, these bounds are the *only* ones that reproduce the
# widely-quoted linear form ``B = 7.14 E - 200 nu + 72.9`` (E in Mpsi):
# the E coefficient fixes the span at 7 Mpsi, the nu coefficient fixes
# it at 0.25, and the constant then fixes ``E_min = 1``. That identity
# is executed in `test_brittleness_reproduces_the_published_linear_form`
# -- the first external tie geomechanics has.
RICKMAN_E_MIN_PA: float = 6.894757e9  # 1 Mpsi
RICKMAN_E_MAX_PA: float = 5.5158056e10  # 8 Mpsi
RICKMAN_NU_MIN: float = 0.15
RICKMAN_NU_MAX: float = 0.40

# Bratli & Risnes (1981) / Schlumberger field guideline: shear
# modulus below ~5 GPa flags poorly consolidated formations prone to
# sand production.
SAND_STABILITY_SHEAR_THRESHOLD_PA: float = 5.0e9


def brittleness_index_rickman(
    young_pa: np.ndarray,
    poisson: np.ndarray,
    *,
    e_min_pa: float = RICKMAN_E_MIN_PA,
    e_max_pa: float = RICKMAN_E_MAX_PA,
    nu_min: float = RICKMAN_NU_MIN,
    nu_max: float = RICKMAN_NU_MAX,
) -> np.ndarray:
    r"""
    Rickman brittleness index from Young's modulus and Poisson's ratio.

    Per Rickman et al. (2008, SPE 115258), the brittleness index is
    the average of two normalised, clipped attributes:

    .. math::

        \mathrm{BI} \;=\;
        \tfrac{1}{2}\,(\hat E + \hat\nu),

    .. math::

        \hat E   \;=\; \mathrm{clip}\!\left(
            \tfrac{E - E_{\min}}{E_{\max} - E_{\min}},\;0,\;1\right),

    .. math::

        \hat\nu \;=\; \mathrm{clip}\!\left(
            \tfrac{\nu_{\max} - \nu}{\nu_{\max} - \nu_{\min}},\;0,\;1\right).

    High Young's modulus and low Poisson's ratio map to high
    brittleness; the result is in ``[0, 1]``.

    Static vs dynamic
    -----------------
    Rickman calibrated on **static** moduli -- Rybacki et al. (2016)
    table A3 records the inputs to this equation as "``E`` = static
    Young's modulus, ``nu`` = static Poisson's ratio" -- while
    sonic-derived ``young_pa`` is the **dynamic** modulus. Static
    values "may be **20-40 % lower** than (undrained) dynamic
    ``E``-moduli" (Rybacki et al. 2016, sect. 4, after Yale & Jamieson
    1994; Britt & Schoeffler 2009; Sone & Zoback 2013a).

    Passing a dynamic modulus therefore **overstates** brittleness.
    At ``E = 40 GPa``, ``nu = 0.25`` -- an ordinary sonic-derived pair
    -- this returns 64.3 %, against 56.0 / 51.9 / 47.7 % for the same
    rock at a static modulus 20 / 30 / 40 % lower: an overstatement of
    **8.3 to 16.6 index points**.

    That range widened when the normalisation bounds were corrected to
    Rickman's published 1-8 Mpsi. The window is narrower than the
    values that shipped before, so the same dynamic modulus now
    normalises higher -- 0.686 against 0.429 on the ``E`` term at
    40 GPa. The correction is right and it raises the cost of skipping
    the conversion.

    The depth-by-depth *ranking* survives, since the bias is
    one-signed; absolute values do not. Apply a dynamic-to-static
    correction upstream if you need them. The worked numbers above are
    executed by
    ``test_dynamic_moduli_overstate_brittleness_by_the_documented_amount``
    in ``tests/test_geomechanics.py``.

    Parameters
    ----------
    young_pa : scalar or ndarray
        Young's modulus (Pa). Static -- see above.
    poisson : scalar or ndarray
        Poisson's ratio (dimensionless).
    e_min_pa, e_max_pa : float
        Normalisation bounds for Young's modulus (Pa). Defaults are
        :data:`RICKMAN_E_MIN_PA` and :data:`RICKMAN_E_MAX_PA`, the
        **1 to 8 Mpsi** (6.89 to 55.2 GPa) of Rickman et al. (2008).
        These were 1.45-11.60 Mpsi until the source was checked -- an
        unperformed unit conversion, 1.450x on both bounds, which
        moved every value this function returns.
    nu_min, nu_max : float
        Normalisation bounds for Poisson's ratio. Defaults are
        :data:`RICKMAN_NU_MIN` and :data:`RICKMAN_NU_MAX` (0.15-0.40).

    Returns
    -------
    ndarray
        Brittleness index in ``[0, 1]``, broadcast to the common
        shape of the inputs.

    Raises
    ------
    ValueError
        If ``e_max_pa <= e_min_pa`` or ``nu_max <= nu_min``.
    """
    if not (e_max_pa > e_min_pa):
        raise ValueError("require e_max_pa > e_min_pa")
    if not (nu_max > nu_min):
        raise ValueError("require nu_max > nu_min")
    e = np.asarray(young_pa, dtype=float)
    nu = np.asarray(poisson, dtype=float)
    e_norm = np.clip((e - e_min_pa) / (e_max_pa - e_min_pa), 0.0, 1.0)
    nu_norm = np.clip((nu_max - nu) / (nu_max - nu_min), 0.0, 1.0)
    return 0.5 * (e_norm + nu_norm)


def fracability_index(
    young_pa: np.ndarray,
    poisson: np.ndarray,
    *,
    e_min_pa: float = RICKMAN_E_MIN_PA,
    e_max_pa: float = RICKMAN_E_MAX_PA,
    nu_min: float = RICKMAN_NU_MIN,
    nu_max: float = RICKMAN_NU_MAX,
) -> np.ndarray:
    """
    Fracability index from Young's modulus and Poisson's ratio.

    For sonic-only inputs the fracability index used in the hydraulic-
    fracture-design literature reduces to the Rickman brittleness
    index (Rickman et al. 2008): a stiff, low-Poisson formation
    fractures more readily and supports a propped fracture better
    than a ductile, high-Poisson one. Other published fracability
    formulations layer in mineralogy, in-situ stress contrast or
    fracture toughness; without those auxiliary inputs the brittleness
    index is the standard sonic-derived proxy.

    Parameters and return value identical to
    :func:`brittleness_index_rickman`; this function is provided as
    a named alias so the call site documents whether the index is
    being used as a *brittleness* (rock-property) measure or a
    *fracability* (completion-design) measure.
    """
    return brittleness_index_rickman(
        young_pa,
        poisson,
        e_min_pa=e_min_pa,
        e_max_pa=e_max_pa,
        nu_min=nu_min,
        nu_max=nu_max,
    )


def closure_stress(
    poisson: np.ndarray,
    sigma_v_pa: np.ndarray,
    *,
    pore_pressure_pa: float | np.ndarray = 0.0,
    biot_alpha: float = 1.0,
) -> np.ndarray:
    r"""
    Minimum horizontal (closure) stress under a uniaxial-strain assumption.

    Eaton (1969) gives the closure stress in a tectonically relaxed
    basin as

    .. math::

        \sigma_h \;-\; \alpha P_p \;=\;
        \frac{\nu}{1 - \nu}\,(\sigma_v \;-\; \alpha P_p),

    i.e. the effective horizontal stress is a Poisson-fraction of the
    effective vertical stress. Solving for the absolute closure stress
    :math:`\sigma_h`:

    .. math::

        \sigma_h \;=\; \frac{\nu}{1 - \nu}\,
                       (\sigma_v - \alpha P_p) \;+\; \alpha P_p.

    This is the standard first-order closure-stress estimate used to
    seed hydraulic-fracture design (Mavko et al. 2009, sect. 8.7);
    in tectonically active basins the result is a lower bound and a
    fault-and-stress correction is required.

    Parameters
    ----------
    poisson : scalar or ndarray
        Poisson's ratio at each depth (dimensionless). Must satisfy
        ``0 <= poisson < 1``. The lower bound rules out auxetic
        materials, which are physically valid but produce negative
        effective horizontal stresses under this formula and are not
        the use case the Eaton model targets. The upper bound is the
        formula's removable singularity at :math:`\nu = 1`.
    sigma_v_pa : scalar or ndarray
        Vertical (overburden) stress at each depth (Pa). Use
        :func:`overburden_stress` to compute this from a density log,
        or pass an externally measured value.
    pore_pressure_pa : scalar or ndarray, default 0.0
        Pore pressure at each depth (Pa). The formula is calibrated in
        effective-stress terms, so a non-zero pore pressure is needed
        whenever the formation is over- or under-pressured relative to
        hydrostatic. Default 0.0 collapses the formula to the dry-rock
        case.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient. ``1.0`` (default) is the
        textbook upper bound for a soft frame; tight rocks may be
        as low as 0.7-0.8. Carried as a scalar because the constant
        is dominated by lithology rather than depth-by-depth
        variability.

    Returns
    -------
    ndarray
        Closure stress (Pa), broadcast to the common shape of the
        inputs.

    Raises
    ------
    ValueError
        If ``poisson >= 1`` or ``poisson < 0`` anywhere.
    """
    nu = np.asarray(poisson, dtype=float)
    sigma_v = np.asarray(sigma_v_pa, dtype=float)
    pp = np.asarray(pore_pressure_pa, dtype=float)
    if np.any(nu >= 1.0):
        raise ValueError("require poisson < 1 everywhere")
    if np.any(nu < 0.0):
        raise ValueError(
            "require poisson >= 0 everywhere; the Eaton uniaxial-strain "
            "closure-stress formula is calibrated for the positive-"
            "Poisson regime of typical sedimentary rocks (auxetic "
            "materials are out of scope)"
        )
    eff_v = sigma_v - biot_alpha * pp
    return (nu / (1.0 - nu)) * eff_v + biot_alpha * pp


def unconfined_compressive_strength(
    young_pa: np.ndarray,
    *,
    model: UCSModel = "lacy_sandstone",
) -> np.ndarray:
    r"""
    Empirical UCS estimate from Young's modulus.

    Available models
    ----------------
    ``"lacy_sandstone"`` (default)

        .. math::

            \mathrm{UCS}\,[\mathrm{MPa}] \;=\;
            0.278\,E^2 \;+\; 2.458\,E,

        with :math:`E` in GPa. Inputs and outputs are converted to
        Pa internally so the API stays SI.

        ⚠ **The provenance of this formula is unverified, and the
        citation it used to carry was wrong.** It said "Lacy (1997,
        SPE 38716) ... in the form compiled by Chang et al. (2006,
        eq. 7)". Chang et al. was checked and none of that holds:

        * Chang's Table 1 contains **no** quadratic in :math:`E`. Its
          only two :math:`E`-based sandstone relations are
          eq. (8) ``UCS = 46.2 exp(0.027 E)`` and eq. (9)
          ``UCS = 2.28 + 4.1089 E`` (Bradford et al. 1998).
        * Chang's **eq. (7)** is ``UCS = 3.87 exp(1.14e-10 rho Vp^2)``
          for the Gulf of Mexico — a density-and-velocity relation,
          not a modulus one.
        * **"Lacy" does not appear in Chang et al. at all**, in any
          table or reference.

        The formula may still be Lacy's own from SPE 38716, which is
        paywalled and unchecked; what is certain is that it did not
        come from where it said. It is **left in place and pinned**
        rather than changed, per ``plans/learning.md`` — replacing a
        shipped correlation needs the right one confirmed, not just
        the wrong citation removed.

        Against Chang's two published :math:`E` relations it runs
        **1.9x to 3.9x high above 20 GPa** (2.58x at 30, 3.26x at 40),
        agreeing with them only near 10 GPa. Treat values from the
        stiff end of a log as unsupported.

    Static vs dynamic
    -----------------
    The Lacy correlation was fit on **static** core-derived Young's
    moduli, while sonic-log-derived ``young_pa`` is the **dynamic**
    modulus, which is generally larger by a factor of 1.5 to 3 for
    porous rocks (Mavko et al. 2009, sect. 5.5).

    **The error is superlinear, not a scale factor**, because the
    correlation is quadratic: halving ``E`` from 40 to 20 GPa divides
    the returned UCS by **3.4**, not by 2. So an uncorrected dynamic
    modulus does not shift the answer, it changes its order of
    magnitude.

    At ordinary sonic-derived moduli the result also leaves the range
    of the rock type it is for. This returns 324 MPa at 30 GPa and
    **543 MPa at 40 GPa**, against roughly 168 MPa for a *strong*
    sandstone in the standard engineering classification (Mansour
    et al. 2020, after Hoek; sandstone UCS is typically quoted over
    ~20-170 MPa). A value in the hundreds of MPa is the signature of
    an uncorrected dynamic modulus rather than a strong rock, and is
    worth treating as a units-style error rather than a conservative
    estimate.

    The depth-by-depth *ranking* does survive -- the correlation is
    monotone in ``E`` -- so the profile shape remains usable even
    where the absolute values are not.

    Parameters
    ----------
    young_pa : scalar or ndarray
        Young's modulus (Pa). Must be non-negative.
    model : ``"lacy_sandstone"``, default ``"lacy_sandstone"``
        Empirical model to use. New models may be added in future
        versions; existing model names are preserved.

    Returns
    -------
    ndarray
        UCS in Pa, broadcast to the shape of ``young_pa``.

    Raises
    ------
    ValueError
        If ``model`` is unknown or ``young_pa`` is negative.
    """
    e = np.asarray(young_pa, dtype=float)
    if np.any(e < 0):
        raise ValueError("young_pa must be non-negative")
    if model == "lacy_sandstone":
        # Lacy (1997) sandstone form (Chang et al. 2006, eq. 7):
        # UCS [MPa] = 0.278 E^2 + 2.458 E, with E in GPa.
        e_gpa = e / 1.0e9
        ucs_mpa = 0.278 * e_gpa**2 + 2.458 * e_gpa
        return ucs_mpa * 1.0e6
    raise ValueError(f"unknown UCS model {model!r}; supported: 'lacy_sandstone'")


def sand_stability_indicator(
    shear_pa: np.ndarray,
    *,
    threshold_pa: float = SAND_STABILITY_SHEAR_THRESHOLD_PA,
) -> np.ndarray:
    """
    Boolean sand-stability flag from the formation shear modulus.

    Bratli & Risnes (1981) -- and the practical Schlumberger field
    guideline that grew out of it -- treats a shear modulus below
    ~5 GPa as a strong indicator that grain-arch failure (and
    therefore sand production) is plausible at typical drawdowns.
    The default :data:`SAND_STABILITY_SHEAR_THRESHOLD_PA` encodes
    that 5 GPa rule of thumb.

    The flag returned here is a soft binary indicator -- callers
    that want a smooth ranking can take the shear modulus itself and
    threshold downstream.

    Parameters
    ----------
    shear_pa : scalar or ndarray
        Shear modulus (Pa).
    threshold_pa : float, default 5e9
        Threshold below which the formation is flagged sand-prone.

    Returns
    -------
    ndarray of bool
        ``True`` where the formation is **stable** (shear modulus at
        or above the threshold), ``False`` where it is sand-prone.
        Boundary convention: ``mu == threshold_pa`` is treated as
        stable. ``True`` for sand-prone is *not* the convention here
        so the flag composes naturally with other "is OK" gates.
    """
    mu = np.asarray(shear_pa, dtype=float)
    return mu >= threshold_pa


@dataclass
class GeomechanicsIndices:
    """
    Per-sample geomechanics indices derived from elastic moduli.

    Output of :func:`geomechanics_indices`. All arrays are aligned
    on the same depth axis as the input ``ElasticModuli``; closure
    stress is ``None`` when the caller did not supply an overburden
    profile.

    Attributes
    ----------
    brittleness : ndarray
        Rickman brittleness index in ``[0, 1]``.
    fracability : ndarray
        Fracability index (alias of brittleness for sonic-only inputs;
        see :func:`fracability_index`).
    ucs : ndarray
        Estimated unconfined compressive strength (Pa).
    sand_stability : ndarray of bool
        ``True`` where the shear modulus exceeds the sanding
        threshold (formation is stable).
    closure_stress : ndarray or None
        Minimum horizontal (closure) stress (Pa). ``None`` when
        ``sigma_v_pa`` was not passed to :func:`geomechanics_indices`.
    """

    brittleness: np.ndarray
    fracability: np.ndarray
    ucs: np.ndarray
    sand_stability: np.ndarray
    closure_stress: np.ndarray | None = None


def geomechanics_indices(
    moduli: ElasticModuli,
    *,
    sigma_v_pa: np.ndarray | None = None,
    pore_pressure_pa: float | np.ndarray = 0.0,
    biot_alpha: float = 1.0,
    ucs_model: UCSModel = "lacy_sandstone",
    sand_threshold_pa: float = SAND_STABILITY_SHEAR_THRESHOLD_PA,
    e_min_pa: float = RICKMAN_E_MIN_PA,
    e_max_pa: float = RICKMAN_E_MAX_PA,
    nu_min: float = RICKMAN_NU_MIN,
    nu_max: float = RICKMAN_NU_MAX,
) -> GeomechanicsIndices:
    """
    One-call geomechanics layer on top of :class:`ElasticModuli`.

    Computes the four sonic-derivable Workflow-3 deliverables --
    brittleness / fracability, sand-stability flag, UCS, and (when
    overburden is supplied) closure stress -- in a single pass and
    returns them as a :class:`GeomechanicsIndices` bundle.

    Parameters
    ----------
    moduli : ElasticModuli
        Output of :func:`fwap.rockphysics.elastic_moduli` (or any
        equivalent dataclass instance).
    sigma_v_pa : ndarray, optional
        Vertical (overburden) stress at each depth (Pa). When
        omitted the closure-stress field on the result is ``None``;
        all other indices are independent of overburden.
    pore_pressure_pa : scalar or ndarray, default 0.0
        Pore pressure at each depth (Pa). Forwarded to
        :func:`closure_stress`.
    biot_alpha : float, default 1.0
        Biot coefficient. Forwarded to :func:`closure_stress`.
    ucs_model : str, default ``"lacy_sandstone"``
        Empirical UCS correlation. Forwarded to
        :func:`unconfined_compressive_strength`.
    sand_threshold_pa : float, default 5 GPa
        Shear-modulus threshold for the sand-stability flag.
    e_min_pa, e_max_pa, nu_min, nu_max : float
        Rickman normalisation bounds. Defaults match
        :func:`brittleness_index_rickman`.

    Returns
    -------
    GeomechanicsIndices
    """
    bi = brittleness_index_rickman(
        moduli.young,
        moduli.poisson,
        e_min_pa=e_min_pa,
        e_max_pa=e_max_pa,
        nu_min=nu_min,
        nu_max=nu_max,
    )
    fi = bi.copy()  # alias today; decoupled in the API for future divergence
    ucs = unconfined_compressive_strength(moduli.young, model=ucs_model)
    stable = sand_stability_indicator(moduli.mu, threshold_pa=sand_threshold_pa)
    sh = None
    if sigma_v_pa is not None:
        sh = closure_stress(
            moduli.poisson,
            np.asarray(sigma_v_pa, dtype=float),
            pore_pressure_pa=pore_pressure_pa,
            biot_alpha=biot_alpha,
        )
    return GeomechanicsIndices(
        brittleness=bi,
        fracability=fi,
        ucs=ucs,
        sand_stability=stable,
        closure_stress=sh,
    )
