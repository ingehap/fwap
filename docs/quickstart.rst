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
