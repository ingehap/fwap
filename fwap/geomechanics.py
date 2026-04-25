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
RICKMAN_E_MIN_PA: float = 1.0e10   # ~1.45 Mpsi
RICKMAN_E_MAX_PA: float = 8.0e10   # ~11.6 Mpsi
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
        young_pa, poisson,
        e_min_pa=e_min_pa, e_max_pa=e_max_pa,
        nu_min=nu_min, nu_max=nu_max,
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
        ucs_mpa = 0.278 * e_gpa ** 2 + 2.458 * e_gpa
        return ucs_mpa * 1.0e6
    raise ValueError(
        f"unknown UCS model {model!r}; supported: 'lacy_sandstone'"
    )


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
        moduli.young, moduli.poisson,
        e_min_pa=e_min_pa, e_max_pa=e_max_pa,
        nu_min=nu_min, nu_max=nu_max,
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
