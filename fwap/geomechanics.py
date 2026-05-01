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
from scipy.optimize import brentq

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


def tensile_breakdown_pressure(
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    pore_pressure: np.ndarray,
    *,
    tensile_strength: np.ndarray = 0.0,
    biot_alpha: float = 1.0,
) -> np.ndarray:
    r"""
    Tensile-failure mud pressure (fracture initiation) for a vertical well.

    Returns the maximum mud pressure :math:`P_w^{\,\mathrm{break}}`
    above which tensile failure (fracture initiation) starts at the
    breakdown azimuth (in the :math:`\sigma_H` direction). Mud
    pressures above this limit open hydraulic fractures at the wall
    -- the standard "lost circulation" / "leak-off" scenario.

    Derivation
    ----------
    At the breakdown azimuth (:math:`\theta = 0` from
    :math:`\sigma_H`), the Kirsch hoop stress is
    :math:`\sigma_{\theta\theta} = 3\sigma_h - \sigma_H - P_w`
    (the LEAST compressive of the three wall stresses). Tensile
    failure occurs when the effective hoop stress drops below
    :math:`-T` (negative = tension; :math:`T` = tensile strength):

    .. math::

        \sigma_{\theta\theta} - \alpha P_p \;\le\; -T,

    so the maximum mud pressure that keeps the wall in compression is

    .. math::

        P_w^{\,\mathrm{break}} \;=\;
            3\sigma_h - \sigma_H + T - \alpha P_p.

    This is the Hubbert-Willis (1957) fracture-initiation pressure
    for a vertical well aligned with the principal stress axes.

    Sensitivity:

    * Higher :math:`\sigma_h` (stronger wall confinement): higher
      :math:`P_w^{\,\mathrm{break}}` (more pressure needed before
      tension overcomes compression).
    * Higher :math:`\sigma_H`: lower
      :math:`P_w^{\,\mathrm{break}}` (the wall is already pre-
      tensioned by horizontal stress anisotropy).
    * Higher :math:`P_p`: lower
      :math:`P_w^{\,\mathrm{break}}` (effective stress is reduced).
    * Higher tensile strength :math:`T`: higher
      :math:`P_w^{\,\mathrm{break}}` (the rock can carry some
      tension before failing).

    Parameters
    ----------
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa).
    pore_pressure : scalar or ndarray
        Pore pressure :math:`P_p` (Pa).
    tensile_strength : scalar or ndarray, default 0.0
        Tensile strength :math:`T` (Pa). Default 0.0 is the
        conservative case (no tensile strength); typical sandstones
        carry 1-5% of UCS in tension. Many petroleum-engineering
        treatments stick with the default zero because a single
        crack can dominate even when the bulk rock has tensile
        strength.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.

    Returns
    -------
    ndarray
        Tensile-breakdown mud pressure :math:`P_w^{\,\mathrm{break}}`
        (Pa). Mud pressures above this limit open fractures at the
        wall.

    See Also
    --------
    mohr_coulomb_breakout_pressure : The lower bound of the safe
        mud-weight window (shear-failure threshold).
    safe_mud_weight_window : Convenience wrapper returning both
        bounds plus a drillability flag.

    Notes
    -----
    Same vertical-well, principal-stress-aligned, drained-elastic
    assumptions as :func:`kirsch_wall_stresses` and
    :func:`mohr_coulomb_breakout_pressure`. In strike-slip and
    reverse-fault stress regimes the breakdown azimuth and formula
    change; this function does not handle those cases.

    References
    ----------
    * Hubbert, M. K., & Willis, D. G. (1957). Mechanics of
      hydraulic fracturing. *Trans. AIME* 210, 153-168.
    * Zoback, M. D. (2007). *Reservoir Geomechanics.* Cambridge
      University Press, Section 6.6.
    """
    sH = np.asarray(sigma_H, dtype=float)
    sh = np.asarray(sigma_h, dtype=float)
    Pp = np.asarray(pore_pressure, dtype=float)
    T = np.asarray(tensile_strength, dtype=float)
    return 3.0 * sh - sH + T - biot_alpha * Pp


@dataclass
class MudWeightWindow:
    r"""
    Output of :func:`safe_mud_weight_window`.

    The two pressure bounds that frame the safe mud-weight window
    for a vertical well, plus convenience properties for the
    window width and a per-depth drillability flag.

    Attributes
    ----------
    breakout_pressure : ndarray
        Lower bound (Pa); mud pressures below this trigger
        Mohr-Coulomb shear breakout at the borehole wall. Output
        of :func:`mohr_coulomb_breakout_pressure`.
    breakdown_pressure : ndarray
        Upper bound (Pa); mud pressures above this trigger tensile
        failure / fracture initiation. Output of
        :func:`tensile_breakdown_pressure`.

    Properties
    ----------
    width : ndarray
        ``breakdown_pressure - breakout_pressure`` (Pa). The mud-
        weight margin available for drilling.
    is_drillable : ndarray of bool
        ``True`` where ``width > 0`` (a non-empty safe window
        exists). ``False`` where the breakout limit exceeds the
        breakdown limit -- the well cannot be drilled in the
        chosen geometry without casing or stress-state
        intervention.
    """

    breakout_pressure: np.ndarray
    breakdown_pressure: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.breakdown_pressure - self.breakout_pressure

    @property
    def is_drillable(self) -> np.ndarray:
        return self.width > 0


def safe_mud_weight_window(
    sigma_H: np.ndarray,
    sigma_h: np.ndarray,
    pore_pressure: np.ndarray,
    ucs: np.ndarray,
    *,
    tensile_strength: np.ndarray = 0.0,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
) -> MudWeightWindow:
    r"""
    Both mud-weight bounds (shear breakout + tensile breakdown).

    Convenience wrapper that calls
    :func:`mohr_coulomb_breakout_pressure` and
    :func:`tensile_breakdown_pressure` with consistent inputs and
    returns the two pressures bundled in a :class:`MudWeightWindow`
    dataclass.

    The "safe" mud-weight window is the closed interval
    ``[breakout_pressure, breakdown_pressure]``: mud pressures in
    this range avoid both shear failure (collapse) at the borehole
    wall and tensile failure (lost circulation). Pressures outside
    either bound trigger the corresponding failure mode.

    Per-depth diagnostic: if the window has zero or negative width
    at a particular depth (``breakout > breakdown``), the well
    cannot be drilled in the supplied geometry without casing,
    drilling-fluid-additive intervention, or a different well
    trajectory.

    Parameters
    ----------
    sigma_H, sigma_h : scalar or ndarray
        Maximum and minimum horizontal stresses (Pa).
    pore_pressure : scalar or ndarray
        Pore pressure (Pa).
    ucs : scalar or ndarray
        Unconfined compressive strength (Pa). Drives the breakout
        bound.
    tensile_strength : scalar or ndarray, default 0.0
        Tensile strength (Pa). Drives the breakdown bound.
    friction_angle_deg : float, default 30.0
        Internal friction angle (degrees) for the Mohr-Coulomb
        breakout calculation.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.

    Returns
    -------
    MudWeightWindow
        Dataclass with ``breakout_pressure`` and
        ``breakdown_pressure`` arrays plus ``width`` and
        ``is_drillable`` properties.

    See Also
    --------
    mohr_coulomb_breakout_pressure : The lower-bound primitive.
    tensile_breakdown_pressure : The upper-bound primitive.
    """
    P_breakout = mohr_coulomb_breakout_pressure(
        sigma_H, sigma_h, pore_pressure, ucs,
        friction_angle_deg=friction_angle_deg,
        biot_alpha=biot_alpha,
    )
    P_breakdown = tensile_breakdown_pressure(
        sigma_H, sigma_h, pore_pressure,
        tensile_strength=tensile_strength,
        biot_alpha=biot_alpha,
    )
    return MudWeightWindow(
        breakout_pressure=np.asarray(P_breakout, dtype=float),
        breakdown_pressure=np.asarray(P_breakdown, dtype=float),
    )


# ---------------------------------------------------------------------
# Bowers (1995) sonic pore-pressure with unloading branch
# ---------------------------------------------------------------------


def pore_pressure_bowers(
    sigma_v_pa: np.ndarray,
    sonic_velocity: np.ndarray,
    *,
    mudline_velocity: float = 1524.0,
    bowers_A: float = 14.02,
    bowers_B: float = 0.673,
    sigma_max_pa: np.ndarray | float | None = None,
    unloading_exponent: float = 3.13,
) -> np.ndarray:
    r"""
    Bowers (1995) sonic pore-pressure prediction with optional unloading.

    Velocity-effective-stress closed form
    :math:`V = V_\mathrm{ml} + A\,\sigma'{}^B`, inverted for the
    effective stress :math:`\sigma' = \sigma_v - P_p` and then for
    the pore pressure. Two branches:

    * **Loading (virgin curve)**: when the rock has never
      experienced a higher effective stress, the velocity is on
      the loading curve and

      .. math::

          \sigma' \;=\; \left(\frac{V - V_\mathrm{ml}}{A}\right)^{1/B},
          \qquad
          P_p \;=\; \sigma_v - \sigma'.

      Selected when ``sigma_max_pa`` is ``None``.

    * **Unloading**: when the rock has been unloaded from a
      previous peak effective stress :math:`\sigma_\mathrm{max}`
      (e.g. by overpressure generation post-burial), the velocity
      is on the unloading curve

      .. math::

          \sigma' \;=\; \sigma_\mathrm{max} \cdot
              \left(
                  \frac{V - V_\mathrm{ml}}
                       {A\,\sigma_\mathrm{max}^B}
              \right)^{U/B},
          \qquad
          P_p \;=\; \sigma_v - \sigma'.

      Selected when ``sigma_max_pa`` is supplied. The unloading
      exponent :math:`U > B` is what makes the unloading curve
      steeper than the loading curve for a given velocity drop --
      the physical signature of unloading-driven overpressure that
      Eaton's method (which assumes loading-only behaviour)
      misses.

    Why Bowers vs Eaton (per the geophysics literature)
    ---------------------------------------------------
    Eaton's method (:func:`pore_pressure_eaton`) assumes the rock
    is on a normal-compaction trend, i.e. on the loading curve.
    For overpressure that arises *during* burial from
    undercompaction (the most common mechanism), Eaton works well.

    For overpressure caused by post-burial unloading mechanisms --
    gas generation, clay-diagenetic dehydration, hydrocarbon
    expulsion, lateral fluid migration -- the rock has been
    unloaded from a higher peak effective stress and now sits on a
    different (steeper) velocity-stress curve. Eaton applied to
    such a rock under-estimates pore pressure (predicts lower than
    actual). Bowers' unloading branch is the standard correction.

    Parameters
    ----------
    sigma_v_pa : ndarray
        Vertical (overburden) stress at each depth (Pa). Use
        :func:`overburden_stress`.
    sonic_velocity : ndarray
        Sonic compressional-wave velocity at each depth (m/s).
        Same shape as ``sigma_v_pa``. Must be strictly greater
        than ``mudline_velocity``.
    mudline_velocity : float, default 1524.0
        Sonic velocity at the mudline / surface (m/s). Default
        1524 m/s (5000 ft/s) is Bowers' (1995) Gulf of Mexico
        shale calibration.
    bowers_A : float, default 14.02
        Bowers' velocity-stress coefficient :math:`A`. Units
        ``(m/s) / MPa^B``. Default 14.02 is a commonly cited
        SI conversion of the Gulf of Mexico shale calibration; for
        production work this should be re-calibrated against well
        data.
    bowers_B : float, default 0.673
        Bowers' velocity-stress exponent :math:`B`. Default 0.673
        for Gulf of Mexico shales.
    sigma_max_pa : ndarray, float, or None, optional
        Per-depth maximum effective stress (Pa) the rock has
        previously experienced. When supplied, the unloading
        branch is used. ``None`` (default) selects the loading
        (virgin) branch.
    unloading_exponent : float, default 3.13
        The unloading exponent :math:`U`. Default 3.13 is
        Bowers' (1995) GoM shale fit. Values 3 to 8 are
        typical for clay-rich shales; only used when
        ``sigma_max_pa`` is supplied.

    Returns
    -------
    ndarray
        Estimated pore pressure (Pa) at each depth.

    Raises
    ------
    ValueError
        If ``mudline_velocity``, ``bowers_A``, ``bowers_B``,
        or ``unloading_exponent`` is non-positive; if any
        ``sonic_velocity <= mudline_velocity`` (the formula
        becomes complex / undefined); if ``sigma_max_pa``
        contains non-positive values.

    Notes
    -----
    **Unit convention**: the formula uses :math:`\sigma'` in
    *megapascals* internally (so ``A`` has units
    ``(m/s)/MPa^B``). All function inputs and outputs are in SI:
    pressures in Pa, velocity in m/s. The Pa <-> MPa conversion
    happens internally; users see a clean SI interface.

    **Calibration is basin-specific**. The default ``A``, ``B``,
    ``V_ml``, ``U`` come from Bowers' original Gulf of Mexico
    shale fit. Other basins (Caspian, North Sea, Bohai Bay) need
    different calibrations -- typically derived by least-squares
    fitting on a presumed-normal section of the well. See Sayers
    (2010) for the procedure.

    **Loading vs unloading selection** is the user's call: this
    function does not auto-detect which branch to use, because the
    correct choice depends on the rock's burial history (which the
    function does not have access to). A pragmatic workflow:

    1. Compute Eaton-style P_p first, identify the depths where
       it predicts overpressure.
    2. For those depths, check if the lithology is gas-bearing
       or has experienced clay diagenesis (use mineralogy logs).
    3. Where unloading mechanisms are plausible, re-compute with
       Bowers' unloading branch using ``sigma_max_pa`` set to the
       loading-curve effective stress at peak burial depth (often
       the maximum of the loading-curve P_p across the well).

    See Also
    --------
    pore_pressure_eaton : The undercompaction-based alternative.
        Suitable for the loading branch of compaction-driven
        overpressure.
    overburden_stress : Companion vertical-stress integral.
    closure_stress : Once P_p is known, feeds directly into
        the closure-stress estimate.

    References
    ----------
    * Bowers, G. L. (1995). Pore pressure estimation from velocity
      data: Accounting for overpressure mechanisms besides
      undercompaction. *SPE Drilling & Completion* 10(2), 89-95.
    * Sayers, C. M. (2010). *Geophysics Under Stress.*
      Distinguished Instructor Series, SEG, Section 5.4
      (calibration procedure for the Bowers parameters).
    * Zhang, J. (2011). Pore pressure prediction from well logs.
      *Earth-Science Reviews* 108(1-2), 50-63 (review of
      Bowers + Eaton + competing methods).
    """
    if mudline_velocity <= 0:
        raise ValueError("mudline_velocity must be positive")
    if bowers_A <= 0:
        raise ValueError("bowers_A must be positive")
    if bowers_B <= 0:
        raise ValueError("bowers_B must be positive")
    if unloading_exponent <= 0:
        raise ValueError("unloading_exponent must be positive")

    sigma_v = np.asarray(sigma_v_pa, dtype=float)
    V = np.asarray(sonic_velocity, dtype=float)

    if np.any(sigma_v < 0):
        raise ValueError("sigma_v_pa must be non-negative")
    if np.any(V < mudline_velocity):
        raise ValueError(
            "sonic_velocity must be >= mudline_velocity (Bowers' formula "
            "becomes complex / non-physical below the mudline value; "
            "V == mudline_velocity is allowed and gives sigma_eff = 0)"
        )

    # Convert Pa <-> MPa for the velocity-stress relation.
    PA_PER_MPA = 1.0e6

    if sigma_max_pa is None:
        # Loading (virgin) branch.
        sigma_eff_MPa = ((V - mudline_velocity) / bowers_A) ** (1.0 / bowers_B)
    else:
        sigma_max = np.asarray(sigma_max_pa, dtype=float)
        if np.any(sigma_max <= 0):
            raise ValueError("sigma_max_pa must be strictly positive")
        sigma_max_MPa = sigma_max / PA_PER_MPA
        # Unloading branch: sigma_eff = sigma_max * ratio^(U/B)
        # where ratio = (V - V_ml) / (A * sigma_max^B).
        ratio = (V - mudline_velocity) / (bowers_A * sigma_max_MPa**bowers_B)
        if np.any(ratio < 0):
            raise ValueError(
                "Bowers unloading-branch ratio went negative; check that "
                "V > mudline_velocity and that sigma_max is reasonable."
            )
        sigma_eff_MPa = sigma_max_MPa * ratio ** (unloading_exponent / bowers_B)

    P_p = sigma_v - sigma_eff_MPa * PA_PER_MPA
    return P_p


# ---------------------------------------------------------------------
# Inclined-wellbore stability (generalized Kirsch + Mohr-Coulomb scan)
# ---------------------------------------------------------------------


def _stress_rotation_to_well_frame(
    inclination_deg: float, azimuth_deg: float
) -> np.ndarray:
    r"""
    3x3 rotation matrix from principal-stress (PS) to well-aligned frame.

    PS axes: :math:`X = \sigma_H` direction, :math:`Y = \sigma_h`
    direction (both horizontal), :math:`Z = \sigma_v` direction
    (vertical, downward positive). Well axes: :math:`Z_W` along the
    well axis (down the hole). The well inclination ``iota`` is the
    angle from :math:`\sigma_v` (vertical) to the well axis; the
    well azimuth ``phi`` is the angle from :math:`\sigma_H` to the
    horizontal projection of the well axis (clockwise looking down).

    Standard parameterisation (Fjaer et al. 2008, Box 4.1): the
    high-side direction :math:`X_W` lies in the plane containing
    the well axis and the vertical axis, pointing toward the
    high side of the inclined wellbore.

    Returns the rotation matrix ``R`` such that the stress tensor
    transforms as ``sigma_W = R @ sigma_PS @ R.T``.

    For a vertical well (``iota = 0``), ``R`` reduces to the
    identity rotated by ``phi`` about the vertical axis -- the
    standard "well aligned with horizontal stress directions" case.
    """
    iota = np.deg2rad(inclination_deg)
    phi = np.deg2rad(azimuth_deg)
    cos_i, sin_i = np.cos(iota), np.sin(iota)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    # X_W (high side): in the (Z_W, Z_PS) plane.
    # Y_W: perpendicular to that plane, completing right-handed.
    # Z_W (well axis): down the hole.
    R = np.array([
        [cos_i * cos_p, cos_i * sin_p, -sin_i],
        [-sin_p, cos_p, 0.0],
        [sin_i * cos_p, sin_i * sin_p, cos_i],
    ])
    return R


def inclined_wellbore_wall_stresses(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    azimuth_around_wall_deg: np.ndarray,
    mud_pressure: float = 0.0,
    poisson: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Generalized Kirsch wall stresses for an arbitrarily inclined well.

    Returns the four wall stress components at each azimuth around
    the borehole (measured from the high side of the inclined well)
    after rotating the principal-stress tensor into well-aligned
    coordinates. Generalises :func:`kirsch_wall_stresses` (which
    is the vertical-well special case) to wells of any orientation.

    Two-step computation:

    1. **Stress rotation**: build the rotation matrix from
       principal-stress (PS) to well-aligned (W) coordinates from
       the well inclination and azimuth, then transform the
       diagonal :math:`\sigma_\mathrm{PS} =
       \mathrm{diag}(\sigma_H, \sigma_h, \sigma_v)` into the
       generally-non-diagonal :math:`\sigma_W` with components
       :math:`\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy},
       \sigma_{xz}, \sigma_{yz}`.

    2. **Generalized Kirsch (Hiramatsu-Oka 1962, Fairhurst 1968)**:
       at the wall, parameterised by azimuth :math:`\theta` from
       the high-side :math:`X_W` axis,

       .. math::

           \sigma_{\theta\theta}(\theta) &=
               \sigma_{xx} + \sigma_{yy}
               - 2(\sigma_{xx} - \sigma_{yy})\cos 2\theta
               - 4\sigma_{xy}\sin 2\theta
               - P_w,
           \\
           \sigma_{zz}(\theta) &=
               \sigma_{zz}^{(W)}
               - 2\nu(\sigma_{xx} - \sigma_{yy})\cos 2\theta
               - 4\nu\sigma_{xy}\sin 2\theta,
           \\
           \sigma_{\theta z}(\theta) &=
               2(-\sigma_{xz}\sin\theta + \sigma_{yz}\cos\theta),
           \\
           \sigma_{rr} &= P_w.

       The :math:`\sigma_{\theta z}` shear component is the new
       feature at non-vertical wells: it is identically zero for
       a vertical well (because the off-diagonal :math:`\sigma_{xz}`
       and :math:`\sigma_{yz}` rotation components vanish) and the
       formulas reduce to the standard Kirsch in
       :func:`kirsch_wall_stresses`.

    The four returned stresses are NOT all principal stresses --
    :math:`\sigma_{rr}` is, but :math:`\sigma_{\theta\theta},
    \sigma_{zz}, \sigma_{\theta z}` are coupled by the off-
    diagonal shear. Use the eigenvalues of the
    :math:`\begin{pmatrix}\sigma_{\theta\theta} & \sigma_{\theta z}
    \\ \sigma_{\theta z} & \sigma_{zz}\end{pmatrix}` 2x2 sub-block
    (or the helper inside :func:`inclined_breakout_pressure`) to
    extract the principal stresses.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Vertical and the two horizontal principal stresses (Pa).
    well_inclination_deg : float
        Angle from vertical to the well axis (deg). 0 = vertical
        well; 90 = horizontal well.
    well_azimuth_deg : float
        Angle from sigma_H direction to the horizontal projection
        of the well axis (deg, clockwise looking down). Only
        relevant when ``well_inclination_deg > 0``.
    azimuth_around_wall_deg : ndarray
        Azimuth grid around the borehole (deg, measured from the
        high-side X_W axis). Use ``np.linspace(0, 360, n,
        endpoint=False)`` for a full sweep.
    mud_pressure : float, default 0.0
        Wellbore pressure :math:`P_w` (Pa).
    poisson : float, default 0.25
        Poisson's ratio.

    Returns
    -------
    (sigma_theta, sigma_z, sigma_theta_z, sigma_r) : tuple of ndarrays
        Wall-stress components at each azimuth (Pa). Each array
        has the shape of ``azimuth_around_wall_deg``.

    See Also
    --------
    kirsch_wall_stresses : The vertical-well special case.
    inclined_breakout_pressure : The minimum-mud-pressure
        consumer of this primitive.

    References
    ----------
    * Hiramatsu, Y., & Oka, Y. (1962). Stress around a shaft or
      level excavated in ground with a three-dimensional stress
      state. *Memoirs of the Faculty of Engineering, Kyoto Univ.*
      24, 56-76.
    * Fairhurst, C. (1968). Methods of determining in-situ rock
      stresses at great depths. *Tech. Rept. ARSE 1-68*, US Army
      Corps of Engineers.
    * Fjaer, E., Holt, R. M., Horsrud, P., Raaen, A. M., & Risnes,
      R. (2008). *Petroleum Related Rock Mechanics*, 2nd ed.
      Elsevier, Chapter 4 and Box 4.1 (the rotation matrix
      parameterisation used here).
    """
    sigma_PS = np.diag([float(sigma_H), float(sigma_h), float(sigma_v)])
    R = _stress_rotation_to_well_frame(well_inclination_deg, well_azimuth_deg)
    sigma_W = R @ sigma_PS @ R.T

    s_xx = sigma_W[0, 0]
    s_yy = sigma_W[1, 1]
    s_zz_W = sigma_W[2, 2]
    s_xy = sigma_W[0, 1]
    s_xz = sigma_W[0, 2]
    s_yz = sigma_W[1, 2]

    theta = np.deg2rad(np.asarray(azimuth_around_wall_deg, dtype=float))
    cos_2t = np.cos(2.0 * theta)
    sin_2t = np.sin(2.0 * theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    sigma_theta = (
        s_xx + s_yy
        - 2.0 * (s_xx - s_yy) * cos_2t
        - 4.0 * s_xy * sin_2t
        - mud_pressure
    )
    sigma_z = (
        s_zz_W
        - 2.0 * poisson * (s_xx - s_yy) * cos_2t
        - 4.0 * poisson * s_xy * sin_2t
    )
    sigma_theta_z = 2.0 * (-s_xz * sin_t + s_yz * cos_t)
    sigma_r = np.full_like(sigma_theta, float(mud_pressure))

    return sigma_theta, sigma_z, sigma_theta_z, sigma_r


def _wall_principal_stresses(
    sigma_theta: np.ndarray,
    sigma_z: np.ndarray,
    sigma_theta_z: np.ndarray,
    sigma_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three principal stresses at each wall azimuth.

    The radial stress ``sigma_r`` is already principal (no shear in
    the r direction at the wall). The other two principal stresses
    are eigenvalues of the (theta, z) 2x2 sub-block:

        | sigma_theta   sigma_theta_z |
        | sigma_theta_z sigma_z       |

    Returns ``(sigma_r, lambda_plus, lambda_minus)`` where
    ``lambda_plus >= lambda_minus`` are the eigenvalues.
    """
    half_sum = 0.5 * (sigma_theta + sigma_z)
    half_diff = 0.5 * (sigma_theta - sigma_z)
    disc = np.sqrt(half_diff**2 + sigma_theta_z**2)
    return sigma_r, half_sum + disc, half_sum - disc


def inclined_breakout_pressure(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    pore_pressure: float,
    ucs: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
    poisson: float = 0.25,
    n_azimuth: int = 180,
) -> float:
    r"""
    Mohr-Coulomb breakout pressure for an arbitrarily inclined well.

    Generalises :func:`mohr_coulomb_breakout_pressure` (which
    closes the formula at the vertical-well breakout azimuth) to
    inclined wells where the failure azimuth around the borehole
    must be searched numerically. Returns the minimum mud pressure
    :math:`P_w^\mathrm{crit}` such that the Mohr-Coulomb failure
    condition :math:`\sigma_1' \le q\,\sigma_3' + \mathrm{UCS}` is
    satisfied at every azimuth around the wall.

    Algorithm:

    1. For a candidate :math:`P_w`, compute the four wall stresses
       (:func:`inclined_wellbore_wall_stresses`) at
       ``n_azimuth`` points around the borehole.
    2. Diagonalise the 2x2 :math:`(\theta, z)` sub-block to get
       the principal stresses at each azimuth (plus the trivial
       :math:`\sigma_{rr} = P_w` principal stress).
    3. Subtract :math:`\alpha P_p` for effective stresses.
    4. Compute the Mohr-Coulomb failure margin
       :math:`\Phi(\theta, P_w) = \sigma_1' - q\,\sigma_3' -
       \mathrm{UCS}` at each azimuth; take the maximum over
       :math:`\theta`. :math:`\Phi_\mathrm{max}(P_w) > 0` means
       the wall fails somewhere; :math:`\Phi_\mathrm{max}(P_w)
       \le 0` means stable.
    5. Bisect (``scipy.optimize.brentq``) for the smallest
       :math:`P_w` such that :math:`\Phi_\mathrm{max}(P_w) = 0`.

    For a vertical well (``well_inclination_deg = 0``), the result
    must agree with :func:`mohr_coulomb_breakout_pressure` to
    within the azimuth-grid resolution.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Principal stresses (Pa).
    pore_pressure : float
        Pore pressure (Pa).
    ucs : float
        Unconfined compressive strength (Pa).
    well_inclination_deg : float
        Well inclination from vertical (deg).
    well_azimuth_deg : float
        Well azimuth from sigma_H direction (deg).
    friction_angle_deg : float, default 30.0
        Internal friction angle (deg).
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.
    poisson : float, default 0.25
        Poisson's ratio (used for the sigma_zz wall-stress
        component, which depends on it).
    n_azimuth : int, default 180
        Number of azimuth-around-wall sample points (1-deg
        resolution by default). Increase for higher precision in
        the worst-azimuth search.

    Returns
    -------
    float
        Critical mud pressure (Pa). Mud pressures below this
        value trigger Mohr-Coulomb shear breakout somewhere on
        the wall.

    Raises
    ------
    ValueError
        If ``friction_angle_deg`` is not in ``(-90, 90)``; if
        ``n_azimuth`` is less than 8; or if the bracket search
        cannot find a sign change for ``brentq`` (typical cause:
        the wall is unconditionally stable or unconditionally
        unstable in the chosen geometry).

    See Also
    --------
    inclined_wellbore_wall_stresses : The wall-stress primitive.
    mohr_coulomb_breakout_pressure : The vertical-well closed-form
        special case.
    """
    if not (-90.0 < friction_angle_deg < 90.0):
        raise ValueError("friction_angle_deg must be in (-90, 90)")
    if n_azimuth < 8:
        raise ValueError("n_azimuth must be at least 8")

    phi = np.deg2rad(friction_angle_deg)
    sin_phi = np.sin(phi)
    q = (1.0 + sin_phi) / (1.0 - sin_phi)
    alpha_Pp = biot_alpha * pore_pressure
    azimuths = np.linspace(0.0, 360.0, n_azimuth, endpoint=False)

    def phi_max(P_w: float) -> float:
        s_t, s_z, s_tz, s_r = inclined_wellbore_wall_stresses(
            sigma_v, sigma_H, sigma_h,
            well_inclination_deg=well_inclination_deg,
            well_azimuth_deg=well_azimuth_deg,
            azimuth_around_wall_deg=azimuths,
            mud_pressure=P_w,
            poisson=poisson,
        )
        sr, l_p, l_m = _wall_principal_stresses(s_t, s_z, s_tz, s_r)
        principal = np.stack([sr - alpha_Pp, l_p - alpha_Pp, l_m - alpha_Pp])
        sigma_1 = np.max(principal, axis=0)
        sigma_3 = np.min(principal, axis=0)
        violation = sigma_1 - q * sigma_3 - ucs
        return float(np.max(violation))

    # Coarse-grid scan: phi_max(P_w) is positive in two regimes
    # (low P_w: shear breakout; very high P_w: tensile breakdown
    # -- our criterion treats negative sigma_3 as a Mohr-Coulomb
    # violation too) and negative in the stable window in between.
    # We want the FIRST sign change going from low to high P_w
    # -- the shear breakout boundary.
    P_max_scan = 1.5 * max(sigma_v, sigma_H, sigma_h)
    P_grid = np.linspace(0.0, P_max_scan, 50)
    phi_grid = np.array([phi_max(P) for P in P_grid])
    sign_changes = np.where(np.diff(np.sign(phi_grid)))[0]
    if len(sign_changes) == 0:
        if phi_grid[0] <= 0.0:
            # Wall is unconditionally stable at P_w = 0; no breakout.
            return 0.0
        raise ValueError(
            "Could not bracket a sign change of the Mohr-Coulomb failure "
            f"margin in P_w in [0, {P_max_scan:.2e}]. The wall is "
            "unconditionally unstable in the chosen geometry; the well "
            "cannot be drilled without casing or stress-state intervention."
        )
    i = int(sign_changes[0])
    return float(brentq(phi_max, P_grid[i], P_grid[i + 1], xtol=1.0e-3))


def inclined_breakdown_pressure(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    pore_pressure: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    tensile_strength: float = 0.0,
    biot_alpha: float = 1.0,
    poisson: float = 0.25,
    n_azimuth: int = 180,
) -> float:
    r"""
    Tensile-breakdown mud pressure for an arbitrarily inclined well.

    Generalises :func:`tensile_breakdown_pressure` (closed form at
    the vertical-well breakdown azimuth) to inclined wells where
    the failure azimuth must be searched numerically. Returns the
    maximum mud pressure :math:`P_w^\mathrm{break}` such that the
    minimum effective principal stress at every azimuth around the
    wall stays above :math:`-T` (no tensile failure).

    Algorithm:

    1. For a candidate :math:`P_w`, compute the four wall stresses
       (:func:`inclined_wellbore_wall_stresses`) at ``n_azimuth``
       points around the borehole.
    2. Diagonalise the 2x2 :math:`(\theta, z)` sub-block to get
       the two non-radial principal stresses (the smaller is
       :math:`\lambda_-`); the radial principal stress is
       :math:`\sigma_{rr} = P_w` itself.
    3. Subtract :math:`\alpha P_p` for effective stresses and take
       the minimum over all azimuths and over the three
       principal-stress candidates -- this is the worst-azimuth
       most-tensile effective stress.
    4. Tensile failure occurs when this minimum is below
       :math:`-T`. The breakdown pressure is the smallest
       :math:`P_w` at which that condition is just satisfied.

    Increasing :math:`P_w` always decreases :math:`\sigma_{\theta
    \theta}`, which can only decrease :math:`\lambda_-` (the
    smaller eigenvalue), so the worst-azimuth minimum eigenvalue
    is monotonically non-increasing in :math:`P_w`. This makes the
    breakdown bisection well-conditioned.

    For a vertical well (``well_inclination_deg = 0``), the
    result must agree with :func:`tensile_breakdown_pressure` to
    within the azimuth-grid resolution.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Principal stresses (Pa).
    pore_pressure : float
        Pore pressure (Pa).
    well_inclination_deg : float
        Well inclination from vertical (deg).
    well_azimuth_deg : float
        Well azimuth from sigma_H direction (deg).
    tensile_strength : float, default 0.0
        Tensile strength :math:`T` (Pa). Default 0 is the
        conservative (no tensile strength) case.
    biot_alpha : float, default 1.0
        Biot poro-elastic coefficient.
    poisson : float, default 0.25
        Poisson's ratio.
    n_azimuth : int, default 180
        Number of azimuth-around-wall sample points.

    Returns
    -------
    float
        Tensile-breakdown mud pressure (Pa). Mud pressures above
        this value trigger tensile failure / fracture initiation
        somewhere on the wall. May be negative if the wall is in
        tension at zero mud pressure -- in which case no positive
        mud pressure is "safe" and the well is undrillable in the
        chosen geometry without intervention.

    See Also
    --------
    inclined_wellbore_wall_stresses : The wall-stress primitive.
    inclined_breakout_pressure : The shear-failure lower bound.
    tensile_breakdown_pressure : The vertical-well closed-form
        special case.
    """
    if n_azimuth < 8:
        raise ValueError("n_azimuth must be at least 8")

    alpha_Pp = biot_alpha * pore_pressure
    azimuths = np.linspace(0.0, 360.0, n_azimuth, endpoint=False)

    def margin(P_w: float) -> float:
        """Margin against tensile failure: positive = stable.

        Convention follows the vertical-well
        :func:`tensile_breakdown_pressure`: tensile failure
        criterion applies to the smallest eigenvalue of the
        :math:`(\\theta, z)` 2x2 sub-block at the wall, NOT to
        the radial principal stress :math:`\\sigma_{rr} = P_w`.
        Including :math:`\\sigma_{rr}` would always trigger
        failure under positive pore pressure (since
        :math:`\\sigma_{rr,\\mathrm{eff}} = P_w - \\alpha P_p` is
        easily negative) and would not match the standard
        Hubbert-Willis fracture-initiation interpretation.
        """
        s_t, s_z, s_tz, s_r = inclined_wellbore_wall_stresses(
            sigma_v, sigma_H, sigma_h,
            well_inclination_deg=well_inclination_deg,
            well_azimuth_deg=well_azimuth_deg,
            azimuth_around_wall_deg=azimuths,
            mud_pressure=P_w,
            poisson=poisson,
        )
        _, _, l_m = _wall_principal_stresses(s_t, s_z, s_tz, s_r)
        # Worst-azimuth most-tensile eigenvalue of the (theta, z)
        # sub-block, after effective-stress correction.
        sigma_3 = float(np.min(l_m)) - alpha_Pp
        return sigma_3 + tensile_strength

    # margin(0) is typically positive (no tension at zero mud);
    # margin(very_high) is negative (tensile failure dominates).
    P_low = 0.0
    P_high = 2.0 * max(sigma_v, sigma_H, sigma_h, abs(pore_pressure))
    f_low = margin(P_low)
    f_high = margin(P_high)
    n_expand = 0
    while f_low * f_high > 0.0 and n_expand < 8:
        P_high *= 1.5
        f_high = margin(P_high)
        n_expand += 1
    if f_low <= 0.0:
        # Wall is in tension at zero mud pressure; no positive
        # mud weight prevents tensile failure. Return P_low (= 0)
        # as the boundary; callers should detect this via
        # MudWeightWindow.is_drillable when paired with a breakout
        # pressure.
        return 0.0
    if f_low * f_high > 0.0:
        raise ValueError(
            "Could not bracket a sign change of the tensile-failure "
            f"margin between P_w={P_low} and P_w={P_high}. The wall is "
            "either unconditionally stable in tension (no tensile "
            "breakdown possible) or there is a numerical issue."
        )
    return float(brentq(margin, P_low, P_high, xtol=1.0e-3))


def inclined_safe_mud_weight_window(
    sigma_v: float,
    sigma_H: float,
    sigma_h: float,
    pore_pressure: float,
    ucs: float,
    *,
    well_inclination_deg: float,
    well_azimuth_deg: float,
    tensile_strength: float = 0.0,
    friction_angle_deg: float = 30.0,
    biot_alpha: float = 1.0,
    poisson: float = 0.25,
    n_azimuth: int = 180,
) -> MudWeightWindow:
    r"""
    Both mud-weight bounds for an arbitrarily inclined well.

    Convenience wrapper that calls
    :func:`inclined_breakout_pressure` and
    :func:`inclined_breakdown_pressure` with consistent inputs and
    returns the two pressures bundled in a :class:`MudWeightWindow`
    (the same dataclass used by the vertical-well counterpart
    :func:`safe_mud_weight_window`).

    The "safe" mud-weight window is the closed interval
    ``[breakout_pressure, breakdown_pressure]``: pressures in this
    range avoid both shear failure (collapse) and tensile failure
    (lost circulation) at every azimuth around the inclined
    borehole. ``MudWeightWindow.is_drillable`` flags whether such
    a window exists.

    Parameters
    ----------
    sigma_v, sigma_H, sigma_h : float
        Principal stresses (Pa).
    pore_pressure : float
        Pore pressure (Pa).
    ucs : float
        Unconfined compressive strength (Pa). Drives the breakout
        bound.
    well_inclination_deg, well_azimuth_deg : float
        Well orientation.
    tensile_strength : float, default 0.0
        Tensile strength (Pa). Drives the breakdown bound.
    friction_angle_deg : float, default 30.0
        Internal friction angle (deg).
    biot_alpha : float, default 1.0
        Biot coefficient.
    poisson : float, default 0.25
        Poisson's ratio.
    n_azimuth : int, default 180
        Azimuth-around-wall scan resolution.

    Returns
    -------
    MudWeightWindow
        Dataclass with ``breakout_pressure`` and
        ``breakdown_pressure`` (both 0-D arrays of float for
        scalar inputs) plus ``width`` and ``is_drillable``
        properties. Note that for inclined wells the bounds are
        always scalar; vector inputs are not supported here
        because the per-orientation worst-azimuth scan does not
        vectorise across depths cleanly. Loop in the caller for
        per-depth inclined-well analysis.

    See Also
    --------
    safe_mud_weight_window : The vertical-well counterpart, which
        does vectorise across depth arrays.
    inclined_breakout_pressure : The lower-bound primitive.
    inclined_breakdown_pressure : The upper-bound primitive.
    """
    P_breakout = inclined_breakout_pressure(
        sigma_v, sigma_H, sigma_h, pore_pressure, ucs,
        well_inclination_deg=well_inclination_deg,
        well_azimuth_deg=well_azimuth_deg,
        friction_angle_deg=friction_angle_deg,
        biot_alpha=biot_alpha,
        poisson=poisson,
        n_azimuth=n_azimuth,
    )
    P_breakdown = inclined_breakdown_pressure(
        sigma_v, sigma_H, sigma_h, pore_pressure,
        well_inclination_deg=well_inclination_deg,
        well_azimuth_deg=well_azimuth_deg,
        tensile_strength=tensile_strength,
        biot_alpha=biot_alpha,
        poisson=poisson,
        n_azimuth=n_azimuth,
    )
    return MudWeightWindow(
        breakout_pressure=np.asarray(P_breakout, dtype=float),
        breakdown_pressure=np.asarray(P_breakdown, dtype=float),
    )


# ---------------------------------------------------------------------
# Tensile strength from UCS (rule-of-thumb correlation)
# ---------------------------------------------------------------------


def tensile_strength_from_ucs(
    ucs: np.ndarray,
    *,
    ratio: float = 0.10,
) -> np.ndarray:
    r"""
    Tensile strength :math:`T` as a fixed fraction of UCS.

    Standard petroleum-engineering rule-of-thumb correlation
    :math:`T = \mathrm{ratio} \cdot \mathrm{UCS}`. The default
    ratio of 0.10 is appropriate for typical sandstones; published
    ranges are roughly:

    * Sandstones / siltstones: ratio :math:`\sim` 0.07 - 0.12
    * Shales: ratio :math:`\sim` 0.04 - 0.08 (rocks with bedding-
      plane weaknesses fail in tension at smaller stresses than
      the MC linear extrapolation predicts)
    * Clean limestones / dolomites: ratio :math:`\sim` 0.08 - 0.15
    * Crystalline / massive rocks: ratio :math:`\sim` 0.10 - 0.20

    Why a fixed ratio rather than the Mohr-Coulomb extrapolation
    :math:`T_\mathrm{MC} = \mathrm{UCS} / q` (where
    :math:`q = (1+\sin\phi)/(1-\sin\phi)`)? The MC envelope, when
    extended into the tensile regime, gives :math:`T \sim
    \mathrm{UCS}/3` for :math:`\phi = 30^\circ` -- substantially
    higher than what laboratory direct-tension or Brazilian-disc
    tests measure on real rocks. The rule-of-thumb ratio (Hoek-
    Brown style "tension cutoff") matches the empirical
    measurements; the MC linear extrapolation is a commonly-
    flagged geomechanical pitfall to avoid in production work.

    Parameters
    ----------
    ucs : scalar or ndarray
        Unconfined compressive strength (Pa). Must be non-negative;
        zero UCS is allowed and gives T = 0.
    ratio : float, default 0.10
        Tensile-to-UCS ratio. Must be in :math:`(0, 1)`.

    Returns
    -------
    ndarray
        Estimated tensile strength (Pa), same shape as ``ucs``.

    Raises
    ------
    ValueError
        If ``ratio`` is outside ``(0, 1)`` or any ``ucs`` is
        negative.

    See Also
    --------
    unconfined_compressive_strength : Sonic-derived UCS estimate
        suitable as the input to this function.
    tensile_breakdown_pressure : The downstream consumer; uses
        the tensile strength as its ``tensile_strength`` argument.

    References
    ----------
    * Hoek, E., & Brown, E. T. (1980). *Underground Excavations
      in Rock.* Institution of Mining and Metallurgy, Section 6
      (the Hoek-Brown empirical failure criterion with a tension
      cutoff at :math:`T \approx \mathrm{UCS} / m_i` where
      :math:`m_i \sim 8-25` for typical lithologies, supporting
      the 0.04-0.13 ratio range).
    * Sheorey, P. R. (1997). *Empirical Rock Failure Criteria.*
      Balkema, Chapter 4 (literature review of T/UCS ratios from
      laboratory tests on a range of lithologies).
    * Zoback, M. D. (2007). *Reservoir Geomechanics.* Cambridge
      University Press, Section 5.3 (the rule-of-thumb 0.10
      default for petroleum-engineering work).
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must be in (0, 1)")
    UCS = np.asarray(ucs, dtype=float)
    if np.any(UCS < 0):
        raise ValueError("ucs must be non-negative")
    return ratio * UCS

