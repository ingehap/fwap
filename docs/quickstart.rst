Quick start
===========

Installation
------------

.. code-block:: bash

   pip install -e .

The package requires Python >= 3.9. Core dependencies are NumPy,
SciPy, Matplotlib (Matplotlib only for the demos and CLI), and the
log-format libraries ``lasio`` (LAS), ``dlisio`` + ``dliswriter``
(DLIS), and ``segyio`` (SEG-Y).

Documentation builds
--------------------

Pre-built PDF snapshots of this manual live in :file:`docs/`:

* :file:`docs/fwap.pdf` -- all-in-one reference, ~140 pages.
  GitHub's blob viewer lazy-loads PDFs and only renders the first
  few pages inline; download the file (right-click *Download* on
  the GitHub blob page, or use the ``raw`` URL) to read all of it.
* :file:`docs/fwap-quickstart.pdf` -- Quick-start section only.
* :file:`docs/fwap-chapter-map.pdf` -- Chapter-to-module map.
* :file:`docs/fwap-roadmap.pdf` -- Roadmap.
* :file:`docs/fwap-changelog.pdf` -- Changelog.

The four per-section PDFs are short enough (3-9 pages each) that
GitHub's blob viewer renders them inline; the all-in-one PDF is
the reference for offline reading.

To regenerate the HTML or PDFs locally:

.. code-block:: bash

   pip install -e .[docs]
   sphinx-build -b html docs docs/_build/html       # HTML

   # PDFs: needs a TeX Live install (xelatex + xindy + poppler-utils):
   sphinx-build -b latex docs docs/_build/latex
   make -C docs/_build/latex                        # builds all PDFs

``make`` produces five PDFs in ``docs/_build/latex/``: the all-in-one
``fwap.pdf`` plus four per-section PDFs configured via
``latex_documents`` in :file:`docs/conf.py`. The LaTeX builder is
pinned to ``xelatex`` in :file:`docs/conf.py` because some of the
docstrings carry Unicode math glyphs that ``pdflatex`` cannot
typeset.

Run the demos
-------------

.. code-block:: bash

   python -m fwap

or equivalently ``fwap all``. This writes a diagnostic figure per
algorithm to ``figures/``.

To run a specific demo:

.. code-block:: bash

   fwap stc            # Part 1: STC + rule-based picker
   fwap pseudorayleigh # Part 1: 4-mode picker incl. pseudo-Rayleigh
   fwap wavesep        # Part 2: f-k filter + SVD / Karhunen-Loeve
   fwap taup           # Part 2: tau-p / slant-stack / linear Radon
   fwap intercept      # Part 3: Coppens & Mari intercept-time inversion
   fwap dipole         # Part 3: dipole flexural dispersion
   fwap dip            # Part 4: dip / azimuth from azimuthal array
   fwap alford         # Extension: cross-dipole Alford rotation
   fwap attenuation    # Extension: Q by centroid shift and spectral ratio
   fwap lwd            # Extension: LWD collar rejection + quadrupole stack
   fwap las            # Extension: LAS I/O round-trip
   fwap dlis           # Extension: DLIS I/O round-trip
   fwap segy           # Extension: SEG-Y I/O round-trip

Programmatic use
----------------

.. code-block:: python

   from fwap import (
       ArrayGeometry, monopole_formation_modes, synthesize_gather,
       stc, pick_modes,
   )

   geom = ArrayGeometry.schlumberger_array_sonic()
   data = synthesize_gather(geom, monopole_formation_modes())
   surface = stc(data, dt=geom.dt, offsets=geom.offsets,
                 window_length=4.0e-4)
   picks = pick_modes(surface)   # {"P": ..., "S": ..., "Stoneley": ...}

Each :class:`~fwap.picker.ModePick` carries the picked slowness,
arrival time, semblance coherence, **and per-mode amplitude** (the
RMS of the per-trace stack contribution at the picked cell), so the
two halves of the book's Workflow-1 deliverable -- amplitude *and*
coherence logs -- come straight off the picker:

.. code-block:: python

   for name, p in picks.items():
       print(f"{name}: slow={p.slowness * 1e6 / 3.281:.2f} us/ft  "
             f"coh={p.coherence:.3f}  amp={p.amplitude:.3f}")

Cross-mode consistency QC flags depths where the picks aren't
internally consistent (Vp/Vs out of band, or canonical mode time
ordering violated):

.. code-block:: python

   from fwap import quality_control_picks

   qc = quality_control_picks(picks, depth=1000.0)
   if qc.flagged:
       print(f"flagged at {qc.depth} m: {qc.reasons}")

Rock-physics moduli from the recovered logs:

.. code-block:: python

   from fwap import elastic_moduli

   m = elastic_moduli(vp=4500.0, vs=2500.0, rho=2400.0)
   print(m.young / 1e9, "GPa")     # Young's modulus
   print(m.poisson)                # Poisson's ratio

Geomechanics indices on top of :class:`~fwap.rockphysics.ElasticModuli`
for the Workflow-3 deliverables (Rickman brittleness / fracability,
Eaton closure stress, Lacy sandstone UCS, Bratli-Risnes sand-stability
flag):

.. code-block:: python

   from fwap import geomechanics_indices, overburden_stress

   sigma_v = overburden_stress(depth, density)             # density log
   geo = geomechanics_indices(
       m, sigma_v_pa=sigma_v, pore_pressure_pa=10.0e6,
   )
   print(geo.brittleness, geo.fracability, geo.ucs / 1e6, "MPa",
         geo.closure_stress / 1e6, "MPa", geo.sand_stability)

Picker -> log-curve bridge converts a per-depth pick track straight
into the ``{mnemonic: ndarray}`` dict the LAS / DLIS writers consume:

.. code-block:: python

   from fwap import (
       track_modes, track_to_log_curves, write_las,
   )

   track = track_modes(stc_results, depths)
   depths_arr, curves = track_to_log_curves(track)
   write_las("output.las", depths_arr, curves)
   # curves carries DTP / DTS / DTST / COHP / COHS / COHST / AMP* /
   # VPVS in the standard fwap mnemonics; slowness in us/ft.

VTI Thomsen-:math:`\gamma` from the combined dipole shear (:math:`C_{44}`)
and Stoneley low-frequency tube-wave inversion (:math:`C_{66}`):

.. code-block:: python

   from fwap import thomsen_gamma_from_logs

   res = thomsen_gamma_from_logs(
       slowness_dipole=dts_curve,         # s/m
       slowness_stoneley=dtst_curve,      # s/m
       rho=density_log,                   # kg/m^3
       rho_fluid=1000.0, v_fluid=1500.0,
   )
   # res.c44, res.c66 in Pa; res.gamma is the Thomsen shear-anisotropy
   # parameter (0 for isotropic, positive for typical VTI shales).

Slow-formation Vs from the low-frequency Stoneley phase velocity, the
primary sonic-only V_S route when the formation has no S head wave on
a monopole gather and pseudo-Rayleigh does not exist:

.. code-block:: python

   from fwap import vs_from_stoneley_slow_formation

   vs = vs_from_stoneley_slow_formation(
       slowness_stoneley=dtst_curve, rho=density_log,
       rho_fluid=1000.0, v_fluid=1500.0,
   )

Drilling-decision stress-state pipeline -- from a density + sonic
acquisition to a per-depth safe-mud-weight window for a vertical
well:

.. code-block:: python

   from fwap import (
       overburden_stress, hydrostatic_pressure,
       pore_pressure_eaton, closure_stress,
       unconfined_compressive_strength, tensile_strength_from_ucs,
       safe_mud_weight_window,
   )

   # 1. Vertical stress from the density log.
   sigma_v = overburden_stress(depth, density)
   P_hydro = hydrostatic_pressure(depth)

   # 2. Pore pressure from sonic via Eaton's normal-trend method.
   #    Use ``pore_pressure_bowers`` instead when unloading
   #    overpressure mechanisms (gas, diagenesis) are suspected.
   P_p = pore_pressure_eaton(
       sigma_v, slowness_observed, slowness_normal, depth=depth,
   )

   # 3. Minimum horizontal stress (Eaton 1969 closure).
   sigma_h = closure_stress(poisson, sigma_v, pore_pressure_pa=P_p)

   # 4. Rock strength from sonic + density (Lacy 1997 / Chang 2006).
   ucs = unconfined_compressive_strength(vp, rho)
   T = tensile_strength_from_ucs(ucs)            # ~10% rule

   # 5. Safe mud-weight window: shear-breakout floor (Mohr-Coulomb)
   #    and tensile-breakdown ceiling (Hubbert-Willis).
   sigma_H = sigma_h + 0.4 * (sigma_v - P_p)     # generic anisotropy
   window = safe_mud_weight_window(
       sigma_H, sigma_h, P_p, ucs, tensile_strength=T,
   )
   # window.breakout_pressure (Pa); window.breakdown_pressure (Pa)
   # window.width = breakdown - breakout
   # window.is_drillable: True where the window has positive width

For inclined / horizontal wells, swap to
``inclined_safe_mud_weight_window(sigma_v, sigma_H, sigma_h, P_p,
ucs, well_inclination_deg=..., well_azimuth_deg=...)`` -- same
:class:`MudWeightWindow` return type, but the bounds come from a
worst-azimuth scan around the wall after rotating the principal
stresses into well-aligned coordinates.

VTI forward modelling -- effective elastic tensor from a layered
isotropic stack (Backus averaging) plus the qP / qSV / SH velocity
surfaces:

.. code-block:: python

   from fwap import (
       backus_average, vti_phase_velocities, vti_group_velocities,
   )
   import numpy as np

   # Backus-average a thinly-bedded shale / sand interval.
   b = backus_average(
       thickness=np.array([0.5, 0.5, 0.3]),    # m
       vp=np.array([2500.0, 3500.0, 2400.0]),
       vs=np.array([1200.0, 2000.0, 1100.0]),
       rho=np.array([2300.0, 2200.0, 2350.0]),
   )
   # b is a BackusResult with c11, c13, c33, c44, c66 (Pa) + rho.

   # Phase velocities of qP / qSV / SH at 0-90 deg from the
   # symmetry axis.
   theta = np.linspace(0.0, np.pi / 2, 91)
   v_qP, v_qSV, v_SH = vti_phase_velocities(
       b.c11, b.c13, b.c33, b.c44, b.c66, b.rho,
       phase_angle_rad=theta,
   )

   # Group velocities + group angles (the wavefront surface).
   g = vti_group_velocities(
       b.c11, b.c13, b.c33, b.c44, b.c66, b.rho,
       phase_angle_rad=theta,
   )
   # Cartesian wavefront for the qP mode, unit-time:
   x = g.v_qP * np.sin(g.psi_qP)
   z = g.v_qP * np.cos(g.psi_qP)

Cylindrical-Biot modal solver -- Stoneley and dipole-flexural
dispersion curves directly from the Schmitt (1988) modal
determinants:

.. code-block:: python

   from fwap import stoneley_dispersion, flexural_dispersion
   import numpy as np

   freq = np.linspace(500.0, 15000.0, 60)

   # n=0 axisymmetric Stoneley mode (works in fast formations).
   st = stoneley_dispersion(
       freq, vp=4500.0, vs=2500.0, rho=2400.0,
       vf=1500.0, rho_f=1000.0, a=0.1,
   )

   # n=1 dipole flexural mode. Slow-formation only (V_S < V_f);
   # in fast formations the mode is narrowly leaky and outside
   # the bound-mode solver scope.
   fl = flexural_dispersion(
       freq, vp=2200.0, vs=800.0, rho=2200.0,
       vf=1500.0, rho_f=1000.0, a=0.1,
   )
   # st.slowness, fl.slowness arrays in s/m, NaN where the
   # bracket failed (out-of-regime depths).

LWD (logging-while-drilling) phenomenological layer -- collar
contamination synthesis + slowness-band rejection:

.. code-block:: python

   from fwap import (
       monopole_formation_modes, synthesize_lwd_gather,
       notch_slowness_band, DEFAULT_COLLAR_SLOWNESS_S_PER_M,
       stc, pick_modes,
   )

   # Synthesize a monopole gather contaminated by the LWD collar.
   data = synthesize_lwd_gather(
       geom, monopole_formation_modes(),
       collar_slowness=DEFAULT_COLLAR_SLOWNESS_S_PER_M,
   )
   # Notch a +/- 15 % band around the known collar slowness.
   c = DEFAULT_COLLAR_SLOWNESS_S_PER_M
   cleaned = notch_slowness_band(
       data, dt=geom.dt, offsets=geom.offsets,
       slow_min=c * 0.85, slow_max=c * 1.15,
   )
   # Run STC + pick_modes on the cleaned record as usual.
