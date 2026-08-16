"""
Frozen public-API guard.

Imports :mod:`fwap` and asserts every name in the sealed list below is
exposed as a top-level attribute. The list mirrors ``fwap.__all__`` at
the moment Step 0 of the refactor plan was committed.

The point of this script is to catch accidental public-API drift during
the upcoming module-splitting PRs: each split moves code into a new
submodule and re-exports it from the package, and it is easy to drop a
name from the re-export by mistake. CI runs this script on every PR;
when a public name is added or removed intentionally, update the list
in the same commit so the rationale lives in the diff.

Run locally with::

    python scripts/check_public_api.py
"""

from __future__ import annotations

import sys

import fwap

FROZEN_PUBLIC_API: tuple[str, ...] = (
    # Constants + logger
    "US_PER_FT",
    "DEFAULT_PRIORS",
    "logger",
    # Synthetic
    "ricker",
    "gabor",
    "ArrayGeometry",
    "Mode",
    "synthesize_gather",
    "monopole_formation_modes",
    "dipole_flexural_dispersion",
    "pseudo_rayleigh_dispersion",
    # STC
    "STCResult",
    "semblance",
    "stc",
    "find_peaks",
    # Picker
    "ModePick",
    "DepthPicks",
    "PosteriorPick",
    "pick_modes",
    "track_modes",
    "viterbi_pick",
    "viterbi_pick_joint",
    "viterbi_posterior_marginals",
    "onset_polarity",
    "wavelet_shape_score",
    "filter_picks_by_shape",
    "filter_track_by_shape",
    "PickQualityFlags",
    "quality_control_picks",
    "quality_control_track",
    "track_to_log_curves",
    # Wave separation
    "fk_forward",
    "fk_inverse",
    "fk_filter",
    "tau_p_forward",
    "tau_p_adjoint",
    "tau_p_inverse",
    "tau_p_filter",
    "apply_moveout",
    "unapply_moveout",
    "svd_project",
    "sequential_kl_separation",
    # Intercept-time
    "InterceptTimeResult",
    "build_design_matrix",
    "build_design_matrix_segmented",
    "solve_intercept_time",
    "assemble_observations_from_picks",
    "delay_to_altered_zone_thickness",
    "delay_to_altered_zone_velocity_contrast",
    "AlteredZoneEstimate",
    "altered_zone_estimate",
    # Dispersion
    "bandpass",
    "narrow_band_stc",
    "DispersionCurve",
    "phase_slowness_from_f_k",
    "phase_slowness_matrix_pencil",
    "shear_slowness_from_dispersion",
    "dispersive_stc",
    "dispersive_pseudo_rayleigh_stc",
    "FlexuralDispersionDiagnosis",
    "classify_flexural_anisotropy",
    # Dip
    "DipResult",
    "estimate_dip",
    "synthesize_azimuthal_arrival",
    "AzimuthalGather",
    # Attenuation
    "AttenuationResult",
    "centroid_frequency_shift_Q",
    "spectral_ratio_Q",
    # Cross-dipole + VTI Thomsen gamma + vertical-well VTI moduli
    "AlfordResult",
    "alford_rotation",
    "alford_rotation_from_tensor",
    "StressAnisotropyEstimate",
    "stress_anisotropy_from_alford",
    "ThomsenGammaResult",
    "stoneley_horizontal_shear_modulus",
    "stoneley_horizontal_shear_modulus_corrected",
    "thomsen_gamma",
    "thomsen_gamma_from_logs",
    "VtiModuli",
    "c33_from_p_pick",
    "vti_moduli_from_logs",
    "ThomsenEpsilonDeltaResult",
    "thomsen_epsilon_delta_from_walkaway_vsp",
    "BackusResult",
    "backus_average",
    "vti_phase_velocities",
    "VtiGroupVelocities",
    "vti_group_velocities",
    # Rock physics
    "ElasticModuli",
    "elastic_moduli",
    "vp_vs_ratio",
    "reuss_average",
    "voigt_average",
    "hill_average",
    "stoneley_permeability_indicator",
    "stoneley_permeability_tang_cheng",
    "stoneley_fracture_density",
    "stoneley_amplitude_fracture_indicator",
    "stoneley_reflection_coefficient",
    "hornby_fracture_aperture",
    "vs_from_stoneley_slow_formation",
    "GassmannResult",
    "gassmann_fluid_substitution",
    # Surface-wave speeds / cylindrical
    "rayleigh_speed",
    "scholte_speed",
    "tube_wave_speed",
    "leaky_radiation_attenuation",
    "flexural_dispersion_physical",
    "flexural_dispersion_vti_physical",
    # Cylindrical-borehole modal-determinant solver (Schmitt 1988)
    "BoreholeLayer",
    "BoreholeMode",
    "BranchSegment",
    "FluidAnnulus",
    "crack_wave_dispersion",
    "stoneley_dispersion",
    "stoneley_dispersion_layered",
    "stoneley_dispersion_microannulus",
    "stoneley_dispersion_vti",
    "flexural_dispersion",
    "flexural_dispersion_layered",
    "flexural_dispersion_layered_vti",
    "flexural_dispersion_vti",
    "pseudo_rayleigh_modal_dispersion",
    "leaky_compressional_dispersion",
    "leaky_quadrupole_dispersion",
    "trapped_pseudo_rayleigh_dispersion",
    "trapped_pseudo_rayleigh_dispersion_layered",
    "quadrupole_dispersion",
    "quadrupole_dispersion_layered",
    "segments_from_kz_curve",
    # LWD phenomenological layer
    "lwd_collar_mode",
    "synthesize_lwd_gather",
    "notch_slowness_band",
    "DEFAULT_COLLAR_SLOWNESS_S_PER_M",
    "DEFAULT_COLLAR_FREQUENCY_HZ",
    "DEFAULT_COLLAR_GABOR_SIGMA_S",
    "QuadrupoleRingGather",
    "synthesize_quadrupole_lwd_gather",
    "quadrupole_stack",
    "lwd_quadrupole_priors",
    # Geomechanics
    "GeomechanicsIndices",
    "brittleness_index_rickman",
    "fracability_index",
    "closure_stress",
    "unconfined_compressive_strength",
    "tensile_strength_from_ucs",
    "sand_stability_indicator",
    "overburden_stress",
    "hydrostatic_pressure",
    "pore_pressure_eaton",
    "pore_pressure_bowers",
    "kirsch_wall_stresses",
    "mohr_coulomb_breakout_pressure",
    "inclined_wellbore_wall_stresses",
    "inclined_breakout_pressure",
    "inclined_breakdown_pressure",
    "inclined_safe_mud_weight_window",
    "tensile_breakdown_pressure",
    "MudWeightWindow",
    "safe_mud_weight_window",
    "geomechanics_indices",
    "RICKMAN_E_MIN_PA",
    "RICKMAN_E_MAX_PA",
    "RICKMAN_NU_MIN",
    "RICKMAN_NU_MAX",
    "SAND_STABILITY_SHEAR_THRESHOLD_PA",
    # I/O (optional deps)
    "LasCurves",
    "DlisCurves",
    "DlisAxis",
    "DlisWaveforms",
    "LdeoWaveforms",
    "LDEO_TOOL_NAMES",
    "LDEO_MODE_NAMES",
    "SegyGather",
    "read_las",
    "write_las",
    "read_dlis",
    "read_dlis_waveforms",
    "write_dlis",
    "read_ldeo_waveforms",
    "read_segy",
    "write_segy",
    # Plotting
    "wiggle_plot",
    "save_figure",
    # Validation against digitised reference figures
    "ReferenceCurve",
    "ReferenceDataError",
    "load_reference_curve",
    "OverlayScore",
    "score_against_reference",
    "format_overlay_score",
)


def main() -> int:
    missing = [name for name in FROZEN_PUBLIC_API if not hasattr(fwap, name)]
    if missing:
        print(
            "ERROR: the following names are in the frozen public-API list "
            "but are not exposed by `fwap`:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nIf the name was removed intentionally, drop it from "
            "FROZEN_PUBLIC_API in this script in the same commit and "
            "note the change in CHANGELOG.md.",
            file=sys.stderr,
        )
        return 1

    also_exported = set(getattr(fwap, "__all__", ())) - set(FROZEN_PUBLIC_API)
    if also_exported:
        print(
            "Note: `fwap.__all__` exposes names not in the frozen list:",
            file=sys.stderr,
        )
        for name in sorted(also_exported):
            print(f"  + {name}", file=sys.stderr)
        print(
            "If these are intentional new public names, add them to "
            "FROZEN_PUBLIC_API to seal them.",
            file=sys.stderr,
        )

    print(f"OK: {len(FROZEN_PUBLIC_API)} public names present on `fwap`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
