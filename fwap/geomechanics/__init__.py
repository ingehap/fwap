"""
Geomechanics indices and wellbore-stability calculations on top of
sonic-derived elastic moduli.

Closes the workflow gap between the elastic-moduli output of
:func:`fwap.rockphysics.elastic_moduli` and the Workflow-3
deliverables listed in Mari et al. (1994), Part 3: petrophysical
and completion-engineering quantities that drive **sanding
prediction**, **hydraulic-fracture design**, and **drilling /
wellbore-stability** decisions.

Caveats
-------
Sonic-derived Young's modulus is the **dynamic** modulus; published
correlations for unconfined compressive strength, brittleness
windows and similar geomechanics quantities were calibrated against
**static** core measurements. The two differ by a lithology- and
stress-dependent factor (Mavko, Mukerji & Dvorkin 2009, sect. 5.5;
Eissa & Kazi 1988). A dynamic-to-static correction should be applied
upstream of this module when absolute numbers matter; the indices
here remain useful as relative depth-by-depth rankings without it.

Subpackage layout
-----------------
* :mod:`fwap.geomechanics.indices`   -- per-sample petrophysical
                                        indices (brittleness,
                                        fracability, UCS, sand
                                        stability, closure stress)
                                        plus the
                                        :func:`geomechanics_indices`
                                        bundler.
* :mod:`fwap.geomechanics.pressures` -- stress / pore-pressure
                                        helpers (overburden,
                                        hydrostatic, Eaton, Bowers,
                                        tensile-from-UCS).
* :mod:`fwap.geomechanics.vertical`  -- Kirsch wall stresses,
                                        Mohr-Coulomb breakout,
                                        Hubbert-Willis breakdown,
                                        safe-mud-weight window
                                        (vertical wells).
* :mod:`fwap.geomechanics.inclined`  -- the four ``inclined_*``
                                        counterparts for
                                        arbitrarily oriented wells
                                        (generalised Kirsch +
                                        worst-azimuth brentq scan).

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 3. Editions Technip,
  Paris. ISBN 978-2-7108-0664-6.
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
* Zoback, M. D. (2007). *Reservoir Geomechanics.* Cambridge
  University Press (the standard wellbore-stability reference).
* Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock Physics
  Handbook*, 2nd ed., Chapter 5 (static vs dynamic moduli).
  Cambridge University Press.
"""

from __future__ import annotations

from fwap.geomechanics.inclined import (
    inclined_breakdown_pressure,
    inclined_breakout_pressure,
    inclined_safe_mud_weight_window,
    inclined_wellbore_wall_stresses,
)
from fwap.geomechanics.indices import (
    RICKMAN_E_MAX_PA,
    RICKMAN_E_MIN_PA,
    RICKMAN_NU_MAX,
    RICKMAN_NU_MIN,
    SAND_STABILITY_SHEAR_THRESHOLD_PA,
    GeomechanicsIndices,
    UCSModel,
    brittleness_index_rickman,
    closure_stress,
    fracability_index,
    geomechanics_indices,
    sand_stability_indicator,
    unconfined_compressive_strength,
)
from fwap.geomechanics.pressures import (
    hydrostatic_pressure,
    overburden_stress,
    pore_pressure_bowers,
    pore_pressure_eaton,
    tensile_strength_from_ucs,
)
from fwap.geomechanics.vertical import (
    MudWeightWindow,
    kirsch_wall_stresses,
    mohr_coulomb_breakout_pressure,
    safe_mud_weight_window,
    tensile_breakdown_pressure,
)

__all__ = [
    "GeomechanicsIndices",
    "MudWeightWindow",
    "RICKMAN_E_MAX_PA",
    "RICKMAN_E_MIN_PA",
    "RICKMAN_NU_MAX",
    "RICKMAN_NU_MIN",
    "SAND_STABILITY_SHEAR_THRESHOLD_PA",
    "UCSModel",
    "brittleness_index_rickman",
    "closure_stress",
    "fracability_index",
    "geomechanics_indices",
    "hydrostatic_pressure",
    "inclined_breakdown_pressure",
    "inclined_breakout_pressure",
    "inclined_safe_mud_weight_window",
    "inclined_wellbore_wall_stresses",
    "kirsch_wall_stresses",
    "mohr_coulomb_breakout_pressure",
    "overburden_stress",
    "pore_pressure_bowers",
    "pore_pressure_eaton",
    "safe_mud_weight_window",
    "sand_stability_indicator",
    "tensile_breakdown_pressure",
    "tensile_strength_from_ucs",
    "unconfined_compressive_strength",
]
