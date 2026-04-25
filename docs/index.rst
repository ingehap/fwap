fwap -- Full-Waveform Acoustic Processing
==========================================

Python implementation of the algorithms described in

  Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing.*
  Editions Technip, Paris, 136 pp. ISBN 978-2-7108-0664-6.
  (Originally published in French as *Traitement des diagraphies
  acoustiques.*)

The book picks four borehole-acoustic problems and works each one
through from a raw multichannel waveform to a log curve a
petrophysicist can actually use. This package provides a modern
NumPy/SciPy implementation of the four chapter algorithms plus a
small extension layer:

* cross-dipole Alford rotation, with a petrophysical labelling that
  exposes the max-horizontal-stress azimuth, the shear-wave
  splitting time, and a flexural-fracture indicator;
* Q attenuation from the array sonic;
* elastic moduli, Reuss / Voigt / Hill mixing, Gassmann fluid
  substitution, and a Stoneley permeability indicator
  (:mod:`fwap.rockphysics`);
* a Rayleigh-speed and physics-grounded flexural dispersion law
  (:mod:`fwap.cylindrical`);
* log-format I/O for LAS, DLIS and SEG-Y (:mod:`fwap.io`).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   chapter_map
   notebooks/workflow
   api
   roadmap
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
