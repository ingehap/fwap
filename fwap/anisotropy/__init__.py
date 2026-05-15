"""
Shear-wave anisotropy: cross-dipole Alford rotation plus VTI
(vertical-symmetry-axis) parameter inversion and Christoffel
dispersion.

Cross-dipole Alford rotation is **not** treated in Mari et al.
(1994); it complements the dipole-flexural processing of
:mod:`fwap.dispersion` by rotating a four-component (XX, XY, YX,
YY) shear measurement into the (fast, slow) shear frame. The VTI
helpers cover the Thomsen-parameter family (gamma from
dipole+Stoneley; epsilon, delta from walkaway-VSP) plus the
Christoffel phase/group velocities and Backus averaging of
isotropic stacks.

Subpackage layout
-----------------
* :mod:`fwap.anisotropy._alford`         -- cross-dipole rotation
  and petrophysical relabelling (:class:`AlfordResult`,
  :func:`alford_rotation`, :func:`alford_rotation_from_tensor`,
  :class:`StressAnisotropyEstimate`,
  :func:`stress_anisotropy_from_alford`).
* :mod:`fwap.anisotropy._thomsen`        -- Thomsen gamma plus the
  Stoneley low-frequency tube wave -> C66 inversion
  (:class:`ThomsenGammaResult`, :func:`thomsen_gamma`,
  :func:`thomsen_gamma_from_logs`,
  :func:`stoneley_horizontal_shear_modulus[_corrected]`).
* :mod:`fwap.anisotropy._vti_inversion`  -- vertical-well VTI moduli
  summary and walkaway-VSP epsilon/delta inversion
  (:func:`c33_from_p_pick`, :class:`VtiModuli`,
  :func:`vti_moduli_from_logs`, :class:`ThomsenEpsilonDeltaResult`,
  :func:`thomsen_epsilon_delta_from_walkaway_vsp`).
* :mod:`fwap.anisotropy._vti_dispersion` -- Backus averaging and
  Christoffel phase/group velocities (:class:`BackusResult`,
  :func:`backus_average`, :func:`vti_phase_velocities`,
  :class:`VtiGroupVelocities`, :func:`vti_group_velocities`).

References
----------
* Alford, R. M. (1986). Shear data in the presence of azimuthal
  anisotropy. *56th SEG Annual International Meeting, Expanded
  Abstracts*, 476-479.
* Esmersoy, C., Koster, K., Williams, M., Boyd, A., & Kane, M.
  (1994). Dipole shear anisotropy logging. *64th SEG Annual
  International Meeting, Expanded Abstracts*, 1139-1142.
* Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics* 51(10),
  1954-1966.
"""

from __future__ import annotations

from fwap.anisotropy._alford import (
    AlfordResult,
    StressAnisotropyEstimate,
    alford_rotation,
    alford_rotation_from_tensor,
    stress_anisotropy_from_alford,
)
from fwap.anisotropy._thomsen import (
    ThomsenGammaResult,
    stoneley_horizontal_shear_modulus,
    stoneley_horizontal_shear_modulus_corrected,
    thomsen_gamma,
    thomsen_gamma_from_logs,
)
from fwap.anisotropy._vti_dispersion import (
    BackusResult,
    VtiGroupVelocities,
    backus_average,
    vti_group_velocities,
    vti_phase_velocities,
)
from fwap.anisotropy._vti_inversion import (
    ThomsenEpsilonDeltaResult,
    VtiModuli,
    c33_from_p_pick,
    thomsen_epsilon_delta_from_walkaway_vsp,
    vti_moduli_from_logs,
)

__all__ = [
    "AlfordResult",
    "BackusResult",
    "StressAnisotropyEstimate",
    "ThomsenEpsilonDeltaResult",
    "ThomsenGammaResult",
    "VtiGroupVelocities",
    "VtiModuli",
    "alford_rotation",
    "alford_rotation_from_tensor",
    "backus_average",
    "c33_from_p_pick",
    "stoneley_horizontal_shear_modulus",
    "stoneley_horizontal_shear_modulus_corrected",
    "stress_anisotropy_from_alford",
    "thomsen_epsilon_delta_from_walkaway_vsp",
    "thomsen_gamma",
    "thomsen_gamma_from_logs",
    "vti_group_velocities",
    "vti_moduli_from_logs",
    "vti_phase_velocities",
]
