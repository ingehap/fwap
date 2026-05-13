"""
Stress and pore-pressure helpers: overburden integration, hydrostatic
baseline, Eaton sonic pore-pressure (with optional unloading branch
via Bowers), and the rule-of-thumb tensile-strength-from-UCS
correlation.

These functions produce the inputs to the wellbore-stability
calculations in :mod:`fwap.geomechanics.vertical` and
:mod:`fwap.geomechanics.inclined` (sigma_v, P_p, T).

References
----------
* Eaton, B. A. (1975). The equation for geopressure prediction from
  well logs. *SPE Annual Fall Meeting*, SPE 5544.
* Bowers, G. L. (1995). Pore pressure estimation from velocity data:
  Accounting for overpressure mechanisms besides undercompaction.
  *SPE Drilling and Completion* 10(2), 89-95.
* Zoback, M. D. (2007). *Reservoir Geomechanics*, Section 5.3.
  Cambridge University Press (T/UCS rule-of-thumb).
"""

from __future__ import annotations

import numpy as np

# Standard gravity for overburden integration.
_STANDARD_G: float = 9.80665


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
            raise ValueError("either hydrostatic_pressure_pa or depth must be supplied")
        P_hydro = hydrostatic_pressure(depth, fluid_density=fluid_density, g=g)
    else:
        P_hydro = np.asarray(hydrostatic_pressure_pa, dtype=float)

    ratio = s_normal / s_obs
    return sigma_v - (sigma_v - P_hydro) * ratio**eaton_exponent


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
