"""
Geomechanics indices derived from elastic moduli.

Closes the workflow gap between the elastic-moduli output of
:func:`fwap.rockphysics.elastic_moduli` and the Workflow-3 deliverables
listed in Mari et al. (1994), Part 3: the petrophysical / completion-
engineering indices that drive **sanding prediction**, **hydraulic-
fracture design**, and **fracture-reservoir characterisation**.

The functions in this module are deliberately thin: each takes raw
arrays (Young's modulus in Pa, Poisson's ratio, etc.) and applies a
single well-cited formula, with the parameter defaults pinned to the
original publications. A one-call wrapper :func:`geomechanics_indices`
operates on a :class:`fwap.rockphysics.ElasticModuli` and returns the
full :class:`GeomechanicsIndices` bundle.

Caveats
-------
Sonic-derived Young's modulus is the **dynamic** modulus; published
correlations for unconfined compressive strength, brittleness windows
and similar geomechanics quantities were calibrated against **static**
core measurements. The two differ by a lithology- and stress-dependent
factor (Mavko, Mukerji & Dvorkin 2009, sect. 5.5; Eissa & Kazi 1988).
A dynamic-to-static correction should be applied upstream of this
module when absolute numbers matter; the indices here remain useful as
relative depth-by-depth rankings without it.

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 3 (dipole-sonic
  reservoir characterisation as input for HF design / sanding
  prediction). Editions Technip, Paris. ISBN 978-2-7108-0664-6.
* Rickman, R., Mullen, M. J., Petre, J. E., Grieser, W. V., &
  Kundert, D. (2008). A practical use of shale petrophysics for
  stimulation design optimization: All shale plays are not clones of
  the Barnett Shale. *SPE Annual Technical Conference and Exhibition*,
  SPE 115258.
* Lacy, L. L. (1997). Dynamic rock mechanics testing for optimized
  fracture designs. *SPE Annual Technical Conference and Exhibition*,
  SPE 38716.
* Eaton, B. A. (1969). Fracture gradient prediction and its
  application in oilfield operations. *J. Petroleum Technology*
  21(10), 1353-1360.
* Bratli, R. K., & Risnes, R. (1981). Stability and failure of sand
  arches. *Society of Petroleum Engineers Journal* 21(2), 236-248.
* Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock Physics
  Handbook*, 2nd ed., Chapter 5 (static vs dynamic moduli). Cambridge
  University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fwap.rockphysics import ElasticModuli

UCSModel = Literal["lacy_sandstone"]

# Default Rickman normalisation bounds, in SI units. The original
# paper uses 1-8 Mpsi for E and 0.15-0.40 for nu; converted at
# 1 Mpsi = 6.8948 GPa.
RICKMAN_E_MIN_PA: float = 1.0e10  # ~1.45 Mpsi
RICKMAN_E_MAX_PA: float = 8.0e10  # ~11.6 Mpsi
RICKMAN_NU_MIN: float = 0.15
RICKMAN_NU_MAX: float = 0.40

# Bratli & Risnes (1981) / Schlumberger field guideline: shear
# modulus below ~5 GPa flags poorly consolidated formations prone to
# sand production.
SAND_STABILITY_SHEAR_THRESHOLD_PA: float = 5.0e9

# Standard gravity for overburden integration.
_STANDARD_G: float = 9.80665


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

    Parameters
    ----------
    young_pa : scalar or ndarray
        Young's modulus (Pa).
    poisson : scalar or ndarray
        Poisson's ratio (dimensionless).
    e_min_pa, e_max_pa : float
        Normalisation bounds for Young's modulus (Pa). Defaults are
        :data:`RICKMAN_E_MIN_PA` and :data:`RICKMAN_E_MAX_PA`,
        matching Rickman et al. (2008) Table 1 (1-8 Mpsi).
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
    pore_pressure_pa: np.ndarray = 0.0,
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
        Lacy (1997, SPE 38716) sandstone correlation, in the form
        compiled by Chang et al. (2006, *J. Petr. Sci. Eng.* 51, eq. 7):

        .. math::

            \mathrm{UCS}\,[\mathrm{MPa}] \;=\;
            0.278\,E^2 \;+\; 2.458\,E,

        with :math:`E` in GPa. Inputs and outputs are converted to
        Pa internally so the API stays SI.

    Static vs dynamic
    -----------------
    The Lacy correlation was fit on **static** core-derived Young's
    moduli, while sonic-log-derived ``young_pa`` is the **dynamic**
    modulus, which is generally larger by a factor of 1.5 to 3 for
    porous rocks (Mavko et al. 2009, sect. 5.5). Without a dynamic-to-
    static correction the returned UCS is an upper bound; the
    depth-by-depth ranking is still informative.

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


def overburden_stress(
    depth: np.ndarray,
    density: np.ndarray,
    *,
    surface_value_pa: float = 0.0,
    g: float = _STANDARD_G,
) -> np.ndarray:
    r"""
    Overburden stress :math:`\sigma_v(z)` by trapezoidal integration of
    a density log.

    .. math::

        \sigma_v(z) \;=\; \sigma_v(z_0) \;+\;
        g \int_{z_0}^{z} \rho(z')\,dz'.

    Trapezoidal integration is exact for piecewise-linear density
    interpolation; for typical 0.1-0.5 m sample spacing the error
    is well below the density-log measurement uncertainty.

    Parameters
    ----------
    depth : ndarray, shape (n_depth,)
        Depth (m), strictly increasing.
    density : ndarray, shape (n_depth,)
        Bulk density (kg/m^3) at each depth. Must be non-negative.
    surface_value_pa : float, default 0.0
        Boundary condition :math:`\sigma_v(z_0)`. Use a non-zero
        value to seed the integration with the overburden above the
        first logged sample (e.g. obtained from a regional
        density-depth model).
    g : float, default 9.80665
        Standard gravity (m/s^2).

    Returns
    -------
    ndarray, shape (n_depth,)
        Overburden stress (Pa) at each depth.

    Raises
    ------
    ValueError
        If ``depth`` is not strictly increasing, if ``density`` is
        negative, or if the two arrays have different length.
    """
    z = np.asarray(depth, dtype=float)
    rho = np.asarray(density, dtype=float)
    if z.shape != rho.shape:
        raise ValueError("depth and density must have the same shape")
    if z.ndim != 1:
        raise ValueError("depth and density must be 1-D")
    if z.size > 1 and np.any(np.diff(z) <= 0):
        raise ValueError("depth must be strictly increasing")
    if np.any(rho < 0):
        raise ValueError("density must be non-negative")
    if z.size == 0:
        return np.empty(0, dtype=float)
    # Trapezoidal cumulative integral of rho(z) along z.
    dz = np.diff(z)
    avg_rho = 0.5 * (rho[:-1] + rho[1:])
    increments = g * avg_rho * dz
    sigma = np.empty_like(z)
    sigma[0] = surface_value_pa
    if z.size > 1:
        sigma[1:] = surface_value_pa + np.cumsum(increments)
    return sigma


# ---------------------------------------------------------------------
# Pore-pressure prediction (Eaton 1975 sonic method)
# ---------------------------------------------------------------------


def hydrostatic_pressure(
    depth: np.ndarray,
    *,
    fluid_density: float = 1000.0,
    g: float = _STANDARD_G,
) -> np.ndarray:
    r"""
    Hydrostatic pressure :math:`P_\mathrm{hydro}(z) = \rho_w \, g \, z`.

    The reference for "normal compaction" pore pressure: a connected
    water column from the surface to depth ``z`` exerts this much
    pressure on the formation. Use as the baseline for
    :func:`pore_pressure_eaton`.

    Parameters
    ----------
    depth : ndarray
        Depth below datum (m), non-negative. Datum is typically the
        sea floor for offshore wells, KB (Kelly bushing) for onshore.
    fluid_density : float, default 1000.0
        Connate water density (kg/m^3). Default is fresh-water; use
        ~1030-1080 for typical seawater / brine. Salinity correction
        is the main reason to override the default.
    g : float, default 9.80665
        Standard gravity (m/s^2).

    Returns
    -------
    ndarray
        Hydrostatic pressure at each depth (Pa).

    Raises
    ------
    ValueError
        If ``fluid_density`` is non-positive or ``depth`` is negative.

    See Also
    --------
    pore_pressure_eaton : Eaton's pore-pressure prediction. Uses
        this hydrostatic baseline as ``P_hydro`` if no explicit
        pressure is supplied.
    overburden_stress : Companion vertical-stress integral; the two
        bracket the formation pressure in the dry-rock case.
    """
    if fluid_density <= 0:
        raise ValueError("fluid_density must be positive")
    z = np.asarray(depth, dtype=float)
    if np.any(z < 0):
        raise ValueError("depth must be non-negative")
    return fluid_density * g * z


def pore_pressure_eaton(
    sigma_v_pa: np.ndarray,
    slowness_observed: np.ndarray,
    slowness_normal: np.ndarray,
    *,
    hydrostatic_pressure_pa: np.ndarray | float | None = None,
    depth: np.ndarray | None = None,
    eaton_exponent: float = 3.0,
    fluid_density: float = 1000.0,
    g: float = _STANDARD_G,
) -> np.ndarray:
    r"""
    Eaton (1975) sonic pore-pressure prediction.

    Closed-form pore-pressure log from a sonic slowness log, an
    overburden-stress log, and a normal-compaction-trend slowness:

    .. math::

        P_p(z) \;=\; \sigma_v(z)
                  \;-\; \big[\sigma_v(z) - P_\mathrm{hydro}(z)\big]
                       \cdot
                       \left(\frac{\Delta t_\mathrm{normal}(z)}
                                  {\Delta t_\mathrm{observed}(z)}
                       \right)^{n},

    where :math:`n` is the Eaton exponent (default 3.0; the standard
    sonic value).

    The ratio
    :math:`\Delta t_\mathrm{normal} / \Delta t_\mathrm{observed}`
    measures how "fast" the rock is at depth ``z`` relative to its
    normal-compaction trend at that depth:

    * Normally compacted rocks (no overpressure):
      ratio :math:`\approx 1`, so :math:`P_p \to P_\mathrm{hydro}`.
    * Overpressured rocks (undercompacted; slower than the trend):
      ratio :math:`< 1`, so the second term shrinks and
      :math:`P_p` rises toward :math:`\sigma_v`.
    * Sub-hydrostatic / depleted rocks (faster than the trend):
      ratio :math:`> 1`, the second term grows, and
      :math:`P_p < P_\mathrm{hydro}`.

    The normal-compaction trend ``slowness_normal`` is typically a
    log-linear fit to known-normal intervals, e.g.

    .. math::

        \Delta t_\mathrm{normal}(z) \;=\; \Delta t_0 \,\exp(-k z),

    with the constants :math:`(\Delta t_0, k)` chosen by least
    squares against the observed slowness in a presumed-normal
    section. Fitting is the caller's responsibility; this function
    just consumes the trend as an array.

    Parameters
    ----------
    sigma_v_pa : ndarray
        Vertical (overburden) stress at each depth (Pa). Use
        :func:`overburden_stress` to compute from a density log,
        or pass an externally measured value.
    slowness_observed : ndarray
        Observed sonic slowness at each depth (s/m). Same shape
        as ``sigma_v_pa``. Strictly positive.
    slowness_normal : ndarray
        Normal-compaction-trend sonic slowness at each depth (s/m).
        Same shape and units as ``slowness_observed``.
    hydrostatic_pressure_pa : ndarray, float, or None, optional
        Hydrostatic pressure at each depth (Pa). If omitted,
        ``depth`` must be supplied so the function can compute
        ``P_hydro = fluid_density * g * depth``.
    depth : ndarray, optional
        Depth (m) for the hydrostatic-pressure computation. Only
        used when ``hydrostatic_pressure_pa`` is None.
    eaton_exponent : float, default 3.0
        The Eaton exponent ``n``. Standard sonic value is 3.0
        (Eaton 1975); resistivity-based variants use 1.2. Higher
        ``n`` increases sensitivity of :math:`P_p` to the
        slowness-ratio departure from 1.
    fluid_density : float, default 1000.0
        Connate-water density (kg/m^3) for the hydrostatic
        computation. Only used when ``depth`` is supplied.
    g : float, default 9.80665
        Standard gravity (m/s^2). Only used when ``depth`` is
        supplied.

    Returns
    -------
    ndarray
        Estimated pore pressure (Pa) at each depth, broadcast to
        the input shape.

    Raises
    ------
    ValueError
        If neither ``hydrostatic_pressure_pa`` nor ``depth`` is
        supplied; if any slowness is non-positive; if
        ``eaton_exponent`` is non-positive; if ``sigma_v_pa`` is
        negative anywhere; or if ``fluid_density`` is non-positive.

    Notes
    -----
    Pore-pressure prediction has known limitations the Eaton
    method does not address:

    * **Sand vs shale**: the Eaton sonic method is calibrated for
      shales (where undercompaction is the dominant overpressure
      mechanism). For sands, a different exponent (or a different
      method like Bowers) is appropriate.
    * **Unloading mechanisms**: gas generation, aquathermal
      pressurisation, and clay diagenesis cause "unloading"
      overpressure that the Eaton equation underestimates;
      Bowers' method is the standard alternative for those.
    * **Tectonic overpressure**: the Eaton equation assumes the
      vertical stress is the maximum principal stress and the
      basin is tectonically relaxed. In active basins the result
      is a starting estimate and should be refined with stress-
      direction information.

    The function deliberately does not clip negative results.
    A negative :math:`P_p` typically indicates a misspecified
    normal-trend ``slowness_normal``; surfacing the sign error
    rather than silently clipping is more useful for diagnosis.

    See Also
    --------
    hydrostatic_pressure : The :math:`P_\mathrm{hydro}` baseline
        used inside the Eaton formula.
    overburden_stress : The :math:`\sigma_v` input to this
        function from a density log.
    closure_stress : Once :math:`P_p` is known, feeds directly
        into the closure-stress estimate.

    References
    ----------
    * Eaton, B. A. (1975). The equation for geopressure prediction
      from well logs. *SPE Annual Fall Meeting*, paper SPE-5544.
    * Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock
      Physics Handbook*, 2nd ed., Section 8.6 (effective-stress
      pore-pressure prediction).
    * Bowers, G. L. (1995). Pore pressure estimation from velocity
      data: Accounting for overpressure mechanisms besides
      undercompaction. *SPE Drilling & Completion* 10(2), 89-95
      (the alternative method for unloading-driven overpressure;
      not implemented here).
    """
    if eaton_exponent <= 0:
        raise ValueError("eaton_exponent must be positive")

    sigma_v = np.asarray(sigma_v_pa, dtype=float)
    s_obs = np.asarray(slowness_observed, dtype=float)
    s_normal = np.asarray(slowness_normal, dtype=float)

    if np.any(s_obs <= 0) or np.any(s_normal <= 0):
        raise ValueError(
            "slowness_observed and slowness_normal must be strictly positive"
        )
    if np.any(sigma_v < 0):
        raise ValueError("sigma_v_pa must be non-negative")

    if hydrostatic_pressure_pa is None:
        if depth is None:
            raise ValueError(
                "either hydrostatic_pressure_pa or depth must be supplied"
            )
        P_hydro = hydrostatic_pressure(
            depth, fluid_density=fluid_density, g=g
        )
    else:
        P_hydro = np.asarray(hydrostatic_pressure_pa, dtype=float)

    ratio = s_normal / s_obs
    return sigma_v - (sigma_v - P_hydro) * ratio**eaton_exponent


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
    pore_pressure_pa: np.ndarray = 0.0,
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


# ---------------------------------------------------------------------
# Wellbore stability: Kirsch wall stresses + Mohr-Coulomb breakout
# ---------------------------------------------------------------------


def kirsch_wall_stresses(
    sigma_v: np.ndarray,
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    *,
    azimuth_deg: np.ndarray,
    mud_pressure: np.ndarray = 0.0,
    poisson: np.ndarray = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Kirsch (1898) borehole-wall stresses for a vertical well.

    Stress concentration around a circular hole drilled vertically
    through a homogeneous, isotropic, elastic medium under far-field
    horizontal stresses :math:`\sigma_H` (max), :math:`\sigma_h`
    (min), and vertical stress :math:`\sigma_v`. At the borehole
    wall (``r = a``), at azimuth :math:`\theta` measured from the
    :math:`\sigma_H` direction:

    .. math::

        \sigma_{\theta\theta}(\theta) &=
            \sigma_H + \sigma_h
            - 2 (\sigma_H - \sigma_h)\cos(2\theta) - P_w,
        \\
        \sigma_{zz}(\theta) &=
            \sigma_v - 2\nu(\sigma_H - \sigma_h)\cos(2\theta),
        \\
        \sigma_{rr} &= P_w.

    The shear stresses :math:`\sigma_{r\theta}, \sigma_{rz},
    \sigma_{\theta z}` vanish at the wall when the well axis is
    aligned with one of the principal stress directions, which is
    the convention used here. The three stresses returned are then
    principal stresses of the local stress tensor at the wall.

    Special azimuths:

    * :math:`\theta = 0\degree` (in the :math:`\sigma_H` direction):
      :math:`\sigma_{\theta\theta} = 3\sigma_h - \sigma_H - P_w`
      (least compressive; tensile-failure / fracture-initiation
      azimuth).
    * :math:`\theta = 90\degree` (in the :math:`\sigma_h`
      direction): :math:`\sigma_{\theta\theta} = 3\sigma_H -
      \sigma_h - P_w` (most compressive; shear-failure / breakout
      azimuth).

    Parameters
    ----------
    sigma_v : scalar or ndarray
        Vertical (overburden) stress (Pa). Use
        :func:`overburden_stress`.
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa). Convention:
        ``sigma_H >= sigma_h``; the function does not enforce this
        because the user may pass either as the "long" or "short"
        principal direction.
    azimuth_deg : scalar or ndarray
        Azimuth (degrees) measured from the :math:`\sigma_H`
        direction.
    mud_pressure : scalar or ndarray, default 0.0
        Wellbore (mud) pressure :math:`P_w` (Pa). Default 0.0
        models a dry hole.
    poisson : scalar or ndarray, default 0.25
        Poisson's ratio (dimensionless). Enters the
        :math:`\sigma_{zz}` formula via the plane-strain
        coupling between horizontal stress deviator and axial
        stress. Default 0.25 is typical for sandstones.

    Returns
    -------
    (sigma_theta, sigma_z, sigma_r) : tuple of ndarrays
        Hoop, axial, and radial stresses at the wall (Pa),
        broadcast to the common shape of the inputs. All three
        are total stresses (no pore-pressure subtraction); pass
        through ``sigma - alpha * P_p`` for effective stresses.

    See Also
    --------
    mohr_coulomb_breakout_pressure : Critical mud pressure for
        shear breakout, derived from the Kirsch hoop stress at the
        breakout azimuth.

    References
    ----------
    * Kirsch, E. G. (1898). Die Theorie der Elastizitaet und die
      Beduerfnisse der Festigkeitslehre. *Z. Verein. Deutsch.
      Ing.* 42, 797-807.
    * Jaeger, J. C., Cook, N. G. W., & Zimmerman, R. W. (2007).
      *Fundamentals of Rock Mechanics*, 4th ed., Chapter 8
      (borehole-stress analysis).
    """
    theta = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    sH = np.asarray(sigma_H, dtype=float)
    sh = np.asarray(sigma_h, dtype=float)
    sv = np.asarray(sigma_v, dtype=float)
    Pw = np.asarray(mud_pressure, dtype=float)
    nu = np.asarray(poisson, dtype=float)

    cos2 = np.cos(2.0 * theta)
    deviator = sH - sh
    sigma_theta = sH + sh - 2.0 * deviator * cos2 - Pw
    sigma_z = sv - 2.0 * nu * deviator * cos2
    sigma_r = np.broadcast_to(Pw, sigma_theta.shape).astype(float).copy()
    return sigma_theta, sigma_z, sigma_r


def mohr_coulomb_breakout_pressure(
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    pore_pressure: np.ndarray,
    ucs: np.ndarray,
    *,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
) -> np.ndarray:
    r"""
    Mohr-Coulomb shear-breakout mud pressure for a vertical well.

    Returns the minimum mud pressure :math:`P_w^{\,\mathrm{crit}}`
    below which Mohr-Coulomb shear failure initiates at the
    breakout azimuth (perpendicular to :math:`\sigma_H`). For
    :math:`P_w < P_w^{\,\mathrm{crit}}` the wellbore wall fails in
    shear, leading to wellbore breakout / enlargement /
    eventually collapse.

    Derivation
    ----------
    At the breakout azimuth, the Kirsch hoop stress is
    :math:`\sigma_{\theta\theta} = 3\sigma_H - \sigma_h - P_w`
    (most compressive); the radial stress is
    :math:`\sigma_{rr} = P_w` (least compressive). For a vertical
    well in a normal-fault stress regime where
    :math:`\sigma_{\theta\theta} > \sigma_{zz} > \sigma_{rr}`,
    these are the maximum and minimum principal stresses at the
    wall. Pass through effective stresses by subtracting
    :math:`\alpha P_p`:

    .. math::

        \sigma_1' &= \sigma_{\theta\theta} - \alpha P_p
                  = 3\sigma_H - \sigma_h - P_w - \alpha P_p,
        \\
        \sigma_3' &= \sigma_{rr} - \alpha P_p
                  = P_w - \alpha P_p.

    Apply the Mohr-Coulomb failure criterion in principal-stress
    form
    :math:`\sigma_1' = q\,\sigma_3' + \mathrm{UCS}` where
    :math:`q = (1+\sin\phi)/(1-\sin\phi)` for friction angle
    :math:`\phi`. Solving for :math:`P_w`:

    .. math::

        P_w^{\,\mathrm{crit}} \;=\;
            \frac{3\sigma_H \;-\; \sigma_h
                  \;+\; (q - 1)\,\alpha P_p
                  \;-\; \mathrm{UCS}}{1 + q}.

    Sensitivity (typical regimes with :math:`q > 1`):

    * Higher horizontal stress anisotropy
      (:math:`3\sigma_H - \sigma_h`): higher
      :math:`P_w^{\,\mathrm{crit}}` (more support needed).
    * Higher pore pressure: higher
      :math:`P_w^{\,\mathrm{crit}}` (the rock is weaker in
      effective stress).
    * Higher UCS or friction angle (stronger rock): lower
      :math:`P_w^{\,\mathrm{crit}}`.

    Parameters
    ----------
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa).
    pore_pressure : scalar or ndarray
        Pore pressure :math:`P_p` (Pa). Use
        :func:`pore_pressure_eaton` to estimate from sonic data.
    ucs : scalar or ndarray
        Unconfined compressive strength :math:`\mathrm{UCS}` (Pa).
        Use :func:`unconfined_compressive_strength` from a sonic
        log, or pass a measured value.
    friction_angle_deg : float, default 30.0
        Internal friction angle :math:`\phi` (degrees). Typical
        ranges: 25-35 for shales, 30-40 for sandstones, 35-45 for
        limestones. Set to 0 for the cohesion-only (Tresca) limit.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient. ``1.0`` is the textbook
        upper bound for a soft frame; tight rocks may be 0.7-0.8.

    Returns
    -------
    ndarray
        Critical mud pressure :math:`P_w^{\,\mathrm{crit}}` (Pa)
        for shear breakout, broadcast to the common shape of the
        inputs. Negative values indicate the rock is strong
        enough to remain stable even with negative wellbore
        pressure (i.e. shear-failure-free); in practice, the
        actual mud pressure should be at least
        :math:`P_p` to balance pore pressure regardless of the
        Mohr-Coulomb result.

    Raises
    ------
    ValueError
        If ``friction_angle_deg`` is not in the open interval
        ``(-90, 90)`` (which keeps :math:`\cos\phi > 0` so
        :math:`q` is finite and positive).

    Notes
    -----
    The formula assumes:

    * Vertical well, vertical principal stress (normal-fault
      regime). Strike-slip and reverse-fault regimes need a
      different :math:`\sigma_1, \sigma_3` identification at
      the wall and are not handled here.
    * :math:`\sigma_{\theta\theta} > \sigma_{zz}` at the
      breakout azimuth, which is the typical case but can fail
      in regimes where :math:`\sigma_v` greatly exceeds
      :math:`\sigma_H`. Callers in non-typical regimes should
      use :func:`kirsch_wall_stresses` directly and apply the
      Mohr-Coulomb criterion to the actual maximum principal
      stress.
    * No tensile failure (fracture initiation, the upper bound
      of the safe mud-weight window). The companion tensile-
      breakdown calculation is a planned follow-up.

    See Also
    --------
    kirsch_wall_stresses : Underlying primitive that gives the
        wall stresses at any azimuth.
    unconfined_compressive_strength : Sonic-derived UCS estimate
        suitable as the ``ucs`` input.
    pore_pressure_eaton : Sonic-derived pore-pressure estimate
        suitable as the ``pore_pressure`` input.
    closure_stress : Closure stress (the lower bound of the safe
        mud-weight window when the limiting failure is tensile).

    References
    ----------
    * Mohr, O. (1900). Welche Umstaende bedingen die
      Elastizitaetsgrenze und den Bruch eines Materiales? *Z.
      Verein. Deutsch. Ing.* 44, 1524-1530.
    * Coulomb, C. A. (1776). Essai sur une application des
      regles de maximis et minimis a quelques problemes de
      statique relatifs a l'architecture. *Mem. Acad. Sci. Paris*
      7, 343-382.
    * Zoback, M. D. (2007). *Reservoir Geomechanics.* Cambridge
      University Press, Chapter 6.
    * Jaeger, J. C., Cook, N. G. W., & Zimmerman, R. W. (2007).
      *Fundamentals of Rock Mechanics*, 4th ed., Section 8.6.
    """
    if not (-90.0 < friction_angle_deg < 90.0):
        raise ValueError("friction_angle_deg must be in (-90, 90)")

    sH = np.asarray(sigma_H, dtype=float)
    sh = np.asarray(sigma_h, dtype=float)
    Pp = np.asarray(pore_pressure, dtype=float)
    UCS = np.asarray(ucs, dtype=float)

    phi = np.deg2rad(friction_angle_deg)
    sin_phi = np.sin(phi)
    q = (1.0 + sin_phi) / (1.0 - sin_phi)

    return (3.0 * sH - sh + (q - 1.0) * biot_alpha * Pp - UCS) / (1.0 + q)

