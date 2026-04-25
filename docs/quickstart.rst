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
