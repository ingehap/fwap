"""
Shared internal constants for the LAS / DLIS / SEG-Y
I/O submodules.

Currently holds only :data:`_FWAP_UNITS`, the
mnemonic-to-unit map used by both :func:`write_las`
and :func:`write_dlis` to auto-fill the curve unit
strings for the fwap-specific log mnemonics.
"""

from __future__ import annotations

from collections.abc import Mapping

_FWAP_UNITS: Mapping[str, str] = {
    # Compressional / shear / Stoneley / pseudo-Rayleigh slowness
    # (us/ft is the borehole-acoustic convention; consumers can
    # convert on read).
    "DTP": "us/ft",
    "DTS": "us/ft",
    "DTST": "us/ft",
    "DTPR": "us/ft",
    # Per-mode coherence (unitless).
    "COHP": "",
    "COHS": "",
    "COHST": "",
    "COHPR": "",
    # Per-mode stack amplitude (input units; dimensionless when the
    # source gather was unit-amplitude).
    "AMPP": "",
    "AMPS": "",
    "AMPST": "",
    "AMPPR": "",
    # Per-mode pick time (window-start time, seconds).
    "TIMP": "s",
    "TIMS": "s",
    "TIMST": "s",
    "TIMPR": "s",
    # Vp / Vs ratio.
    "VPVS": "",
    # Q (dimensionless).
    "QP": "",
    "QS": "",
    # Elastic moduli.
    "K": "Pa",
    "MU": "Pa",
    "E": "Pa",
    "NU": "",
    # Geomechanics indices (fwap.geomechanics).
    "BRIT": "",  # Rickman brittleness index, [0, 1]
    "FRAC": "",  # Fracability index, [0, 1]
    "UCS": "Pa",  # Unconfined compressive strength
    "SH": "Pa",  # Minimum horizontal (closure) stress
    "SV": "Pa",  # Vertical (overburden) stress
    "SAND": "",  # Sand-stability flag (0 = sand-prone, 1 = stable)
    # VTI elastic moduli (fwap.anisotropy.vti_moduli_from_logs /
    # thomsen_gamma_from_logs).
    "C33": "Pa",  # Vertical P-wave modulus (rho * Vp^2)
    "C44": "Pa",  # Vertical shear modulus (rho * Vsv^2)
    "C66": "Pa",  # Horizontal shear modulus (Stoneley inversion)
    "GAMMA": "",  # Thomsen shear-anisotropy parameter
    "VP": "m/s",  # Vertical P-wave velocity
    "VSV": "m/s",  # Vertical shear velocity
    "VSH": "m/s",  # Horizontal shear velocity (Stoneley-derived)
    # Stoneley reflection / fracture-aperture inversion
    # (fwap.stoneley.hornby_fracture_aperture).
    "RFRAC": "",  # Stoneley reflection coefficient |R|
    "FAPER": "m",  # Hornby et al. 1989 fracture aperture
}
