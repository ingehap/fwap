"""
fwap -- Full-Waveform Acoustic Processing.

Python implementation of the algorithms described in

    Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
    *Full Waveform Acoustic Data Processing.*
    Translated by Gillian Harvey-Bletsas. Editions Technip, Paris, 136 pp.
    ISBN 978-2-7108-0664-6. (Originally published in French as
    *Traitement des diagraphies acoustiques.*)

Chapter-to-module map
---------------------
* Part 1 -- AI picking of waves on full-waveform acoustic data
                                          ->  :mod:`fwap.coherence` + :mod:`fwap.picker`
* Part 2 -- Wave separation in acoustic well logging
                                          ->  :mod:`fwap.wavesep`
* Part 3 -- Reservoir characterization with the dipole sonic imaging tool
            (intercept-time inversion + dipole-flexural processing)
                                          ->  :mod:`fwap.tomography` +
                                              :mod:`fwap.dispersion`
* Part 4 -- Dip measurement based on acoustic data
                                          ->  :mod:`fwap.dip`

Modules outside the scope of the 1994 book (added for completeness):

* :mod:`fwap.anisotropy`  -- cross-dipole Alford rotation (Alford, 1986)
* :mod:`fwap.attenuation` -- Q from array sonic (Quan & Harris, 1997;
                              Bath, 1974)
* :mod:`fwap.rockphysics` -- elastic moduli (K, mu, E, nu) from
                              Vp, Vs, rho
* :mod:`fwap.cylindrical` -- Rayleigh-speed surface-wave
                              calculation and a physics-grounded
                              flexural-mode dispersion law
* :mod:`fwap.geomechanics` -- brittleness / fracability / closure
                              stress / UCS / sand-stability indices
                              on top of :class:`ElasticModuli`
                              (Rickman 2008; Eaton 1969; Lacy 1997)
* :mod:`fwap.lwd`         -- LWD (logging-while-drilling)
                              phenomenological layer: steel-collar
                              :class:`Mode` factory, slowness-band
                              notch for collar rejection, quadrupole-
                              source ring synthesis, and m=2
                              receiver-side stacker
                              (Tang & Cheng 2004 sect. 2.4-2.5)
* :mod:`fwap.io`          -- LAS reader/writer (``lasio``), DLIS
                              reader/writer (``dlisio`` +
                              ``dliswriter``), and SEG-Y reader/writer
                              (``segyio``) -- all core deps

Recommended companion references
--------------------------------
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes.*
  CRC Press.
* Mari, J.-L., Glangeaud, F., & Coppens, F. (1999). *Signal Processing
  for Geologists and Geophysicists.* Editions Technip, Paris.
  ISBN 2-7108-0752-1.
* Mari, J.-L., & Vergniault, C. (2018). *Well Seismic Surveying and
  Acoustic Logging.* EDP Open.
* Coppens, F., & Mari, J.-L. (1995). Application of the intercept time
  method to full waveform acoustic data. *First Break* 13(1), 11-20.
* Coppens, F., & Mari, J.-L. (1995). Imagerie par refraction en
  diagraphie acoustique. *Revue de l'Institut Francais du Petrole*
  50(2), 143.

Top-level usage::

    from fwap import stc, pick_modes, solve_intercept_time, alford_rotation
"""

from __future__ import annotations

import logging

__version__ = "0.4.0"

# Constants + shared logger (see fwap._common). Silent by default;
# application code attaches handlers.
from fwap._common import US_PER_FT, logger
logger.addHandler(logging.NullHandler())
from fwap.picker import DEFAULT_PRIORS

# Synthetic
from fwap.synthetic import (
    ArrayGeometry,
    Mode,
    dipole_flexural_dispersion,
    gabor,
    monopole_formation_modes,
    pseudo_rayleigh_dispersion,
    ricker,
    synthesize_gather,
)

# STC
from fwap.coherence import STCResult, find_peaks, semblance, stc

# Picker
from fwap.picker import (
    DepthPicks,
    ModePick,
    PickQualityFlags,
    PosteriorPick,
    filter_picks_by_shape,
    filter_track_by_shape,
    onset_polarity,
    pick_modes,
    quality_control_picks,
    quality_control_track,
    track_modes,
    track_to_log_curves,
    viterbi_pick,
    viterbi_pick_joint,
    viterbi_posterior_marginals,
    wavelet_shape_score,
)

# Wave separation
from fwap.wavesep import (
    apply_moveout,
    fk_filter,
    fk_forward,
    fk_inverse,
    sequential_kl_separation,
    svd_project,
    tau_p_adjoint,
    tau_p_filter,
    tau_p_forward,
    tau_p_inverse,
    unapply_moveout,
)

# Intercept-time
from fwap.tomography import (
    AlteredZoneEstimate,
    InterceptTimeResult,
    altered_zone_estimate,
    assemble_observations_from_picks,
    build_design_matrix,
    build_design_matrix_segmented,
    delay_to_altered_zone_thickness,
    delay_to_altered_zone_velocity_contrast,
    solve_intercept_time,
)

# Dispersion
from fwap.dispersion import (
    DispersionCurve,
    FlexuralDispersionDiagnosis,
    bandpass,
    classify_flexural_anisotropy,
    dispersive_pseudo_rayleigh_stc,
    dispersive_stc,
    narrow_band_stc,
    phase_slowness_from_f_k,
    phase_slowness_matrix_pencil,
    shear_slowness_from_dispersion,
)

# Dip
from fwap.dip import (
    AzimuthalGather,
    DipResult,
    estimate_dip,
    synthesize_azimuthal_arrival,
)

# Attenuation
from fwap.attenuation import (
    AttenuationResult,
    centroid_frequency_shift_Q,
    spectral_ratio_Q,
)

# Cross-dipole
from fwap.anisotropy import (
    AlfordResult,
    StressAnisotropyEstimate,
    ThomsenEpsilonDeltaResult,
    ThomsenGammaResult,
    VtiModuli,
    alford_rotation,
    alford_rotation_from_tensor,
    c33_from_p_pick,
    stoneley_horizontal_shear_modulus,
    stoneley_horizontal_shear_modulus_corrected,
    stress_anisotropy_from_alford,
    thomsen_epsilon_delta_from_walkaway_vsp,
    thomsen_gamma,
    thomsen_gamma_from_logs,
    vti_moduli_from_logs,
)

# Rock physics
from fwap.rockphysics import (
    ElasticModuli,
    GassmannResult,
    elastic_moduli,
    gassmann_fluid_substitution,
    hill_average,
    hornby_fracture_aperture,
    reuss_average,
    stoneley_amplitude_fracture_indicator,
    stoneley_permeability_indicator,
    stoneley_reflection_coefficient,
    voigt_average,
    vp_vs_ratio,
    vs_from_stoneley_slow_formation,
)

# Geomechanics indices (brittleness, fracability, UCS, closure stress,
# sand-stability) on top of ElasticModuli.
from fwap.geomechanics import (
    GeomechanicsIndices,
    RICKMAN_E_MAX_PA,
    RICKMAN_E_MIN_PA,
    RICKMAN_NU_MAX,
    RICKMAN_NU_MIN,
    SAND_STABILITY_SHEAR_THRESHOLD_PA,
    brittleness_index_rickman,
    closure_stress,
    fracability_index,
    geomechanics_indices,
    overburden_stress,
    sand_stability_indicator,
    unconfined_compressive_strength,
)

# Cylindrical / surface-wave speeds
from fwap.cylindrical import (
    flexural_dispersion_physical,
    flexural_dispersion_vti_physical,
    rayleigh_speed,
)
from fwap.cylindrical_solver import (
    BoreholeMode,
    stoneley_dispersion,
)

# LWD (logging-while-drilling) phenomenological layer
from fwap.lwd import (
    DEFAULT_COLLAR_FREQUENCY_HZ,
    DEFAULT_COLLAR_GABOR_SIGMA_S,
    DEFAULT_COLLAR_SLOWNESS_S_PER_M,
    QuadrupoleRingGather,
    lwd_collar_mode,
    lwd_quadrupole_priors,
    notch_slowness_band,
    quadrupole_stack,
    synthesize_lwd_gather,
    synthesize_quadrupole_lwd_gather,
)

# File I/O (optional dependencies imported lazily inside each function)
from fwap.io import (
    DlisCurves,
    LasCurves,
    SegyGather,
    read_dlis,
    read_las,
    read_segy,
    write_dlis,
    write_las,
    write_segy,
)

# Plotting helpers (public; require matplotlib at call time)
from fwap.plotting import save_figure, wiggle_plot

__all__ = [
    # Constants + logger
    "US_PER_FT", "DEFAULT_PRIORS", "logger",
    # Synthetic
    "ricker", "gabor", "ArrayGeometry", "Mode", "synthesize_gather",
    "monopole_formation_modes", "dipole_flexural_dispersion",
    "pseudo_rayleigh_dispersion",
    # STC
    "STCResult", "semblance", "stc", "find_peaks",
    # Picker
    "ModePick", "DepthPicks", "PosteriorPick",
    "pick_modes", "track_modes",
    "viterbi_pick", "viterbi_pick_joint",
    "viterbi_posterior_marginals",
    "onset_polarity", "wavelet_shape_score",
    "filter_picks_by_shape", "filter_track_by_shape",
    "PickQualityFlags",
    "quality_control_picks", "quality_control_track",
    "track_to_log_curves",
    # Wave separation
    "fk_forward", "fk_inverse", "fk_filter",
    "tau_p_forward", "tau_p_adjoint", "tau_p_inverse", "tau_p_filter",
    "apply_moveout", "unapply_moveout",
    "svd_project", "sequential_kl_separation",
    # Intercept-time
    "InterceptTimeResult", "build_design_matrix",
    "build_design_matrix_segmented", "solve_intercept_time",
    "assemble_observations_from_picks",
    "delay_to_altered_zone_thickness",
    "delay_to_altered_zone_velocity_contrast",
    "AlteredZoneEstimate", "altered_zone_estimate",
    # Dispersion
    "bandpass", "narrow_band_stc", "DispersionCurve",
    "phase_slowness_from_f_k", "phase_slowness_matrix_pencil",
    "shear_slowness_from_dispersion", "dispersive_stc",
    "dispersive_pseudo_rayleigh_stc",
    "FlexuralDispersionDiagnosis", "classify_flexural_anisotropy",
    # Dip
    "DipResult", "estimate_dip", "synthesize_azimuthal_arrival",
    "AzimuthalGather",
    # Attenuation
    "AttenuationResult", "centroid_frequency_shift_Q", "spectral_ratio_Q",
    # Cross-dipole + VTI Thomsen gamma + vertical-well VTI moduli
    "AlfordResult", "alford_rotation", "alford_rotation_from_tensor",
    "StressAnisotropyEstimate", "stress_anisotropy_from_alford",
    "ThomsenGammaResult", "stoneley_horizontal_shear_modulus",
    "stoneley_horizontal_shear_modulus_corrected",
    "thomsen_gamma", "thomsen_gamma_from_logs",
    "VtiModuli", "c33_from_p_pick", "vti_moduli_from_logs",
    "ThomsenEpsilonDeltaResult",
    "thomsen_epsilon_delta_from_walkaway_vsp",
    # Rock physics
    "ElasticModuli", "elastic_moduli", "vp_vs_ratio",
    "reuss_average", "voigt_average", "hill_average",
    "stoneley_permeability_indicator",
    "stoneley_amplitude_fracture_indicator",
    "stoneley_reflection_coefficient",
    "hornby_fracture_aperture",
    "vs_from_stoneley_slow_formation",
    "GassmannResult", "gassmann_fluid_substitution",
    # Surface-wave speeds / cylindrical
    "rayleigh_speed", "flexural_dispersion_physical",
    "flexural_dispersion_vti_physical",
    # Cylindrical-borehole modal-determinant solver (Schmitt 1988)
    "BoreholeMode", "stoneley_dispersion",
    # LWD phenomenological layer
    "lwd_collar_mode", "synthesize_lwd_gather", "notch_slowness_band",
    "DEFAULT_COLLAR_SLOWNESS_S_PER_M",
    "DEFAULT_COLLAR_FREQUENCY_HZ",
    "DEFAULT_COLLAR_GABOR_SIGMA_S",
    "QuadrupoleRingGather",
    "synthesize_quadrupole_lwd_gather", "quadrupole_stack",
    "lwd_quadrupole_priors",
    # Geomechanics
    "GeomechanicsIndices", "brittleness_index_rickman",
    "fracability_index", "closure_stress",
    "unconfined_compressive_strength", "sand_stability_indicator",
    "overburden_stress", "geomechanics_indices",
    "RICKMAN_E_MIN_PA", "RICKMAN_E_MAX_PA",
    "RICKMAN_NU_MIN", "RICKMAN_NU_MAX",
    "SAND_STABILITY_SHEAR_THRESHOLD_PA",
    # I/O (optional deps)
    "LasCurves", "DlisCurves", "SegyGather",
    "read_las", "write_las",
    "read_dlis", "write_dlis",
    "read_segy", "write_segy",
    # Plotting
    "wiggle_plot", "save_figure",
]
