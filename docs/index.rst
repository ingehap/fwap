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
NumPy/SciPy implementation of the four chapter algorithms plus an
extension layer that covers the post-1994 borehole-acoustic
literature:

* cross-dipole Alford rotation, with a petrophysical labelling that
  exposes the max-horizontal-stress azimuth, the shear-wave
  splitting time, and a flexural-fracture indicator; the VTI
  Thomsen-:math:`\gamma` shear-anisotropy parameter from the
  combined dipole shear (:math:`C_{44}`) and Stoneley
  (:math:`C_{66}` via the White / Norris tube-wave inversion); the
  Backus (1962) layered-medium average for upscaling layered
  isotropic stacks to an effective VTI tensor; and the Tsvankin-
  2001 phase- and group-velocity surfaces for forward-modelling
  qP / qSV / SH wavefronts in VTI media (:mod:`fwap.anisotropy`);
* Q attenuation from the array sonic (:mod:`fwap.attenuation`);
* elastic moduli, Reuss / Voigt / Hill mixing, Gassmann fluid
  substitution, a four-tool Stoneley fracture / permeability suite
  (slowness indicator, amplitude indicator, Tang-Cheng-Toksoz
  quantitative matrix-permeability inversion, Hornby et al. (1989)
  reflection-coefficient fracture-aperture inversion) plus a
  :func:`~fwap.rockphysics.stoneley_fracture_density` combiner that
  mixes them into a unified per-depth fracture-intensity score, and
  a slow-formation Vs estimator from the low-frequency Stoneley
  phase velocity (:mod:`fwap.rockphysics`);
* a complete geomechanics layer on top of
  :class:`~fwap.rockphysics.ElasticModuli`: Rickman brittleness /
  fracability index, Eaton uniaxial-strain closure stress, Lacy
  sandstone UCS, Bratli-Risnes sand-stability flag, density-log
  overburden integration, hydrostatic-pressure helper, sonic-
  derived pore-pressure prediction (Eaton 1975 for undercompaction
  and Bowers 1995 for unloading), tensile-strength rule-of-thumb
  correlation, and a full **drilling-decision pipeline** -- Kirsch
  wall stresses + Mohr-Coulomb shear breakout + Hubbert-Willis
  tensile breakdown + safe-mud-weight window for both vertical and
  arbitrarily inclined wells; plus the one-call
  :func:`~fwap.geomechanics.geomechanics_indices` bundle for the
  brittleness / fracability / UCS / closure-stress quartet
  (:mod:`fwap.geomechanics`);
* a Rayleigh-speed and physics-grounded flexural dispersion law
  (:mod:`fwap.cylindrical`), and a real cylindrical-Biot modal
  solver: Schmitt 1988 n=0 Stoneley (3x3) and n=1 dipole flexural
  (4x4) determinants, root-finding via :func:`scipy.optimize.brentq`
  in the bound-mode regime (:mod:`fwap.cylindrical_solver`);
* a stress-vs-intrinsic anisotropy classifier from a fast / slow
  cross-dipole flexural-dispersion-curve crossover (Sinha & Kostek
  1996; :func:`~fwap.dispersion.classify_flexural_anisotropy`),
  plus a dispersion-corrected STC for the pseudo-Rayleigh / guided
  trapped mode (:func:`~fwap.dispersion.dispersive_pseudo_rayleigh_stc`);
* a phenomenological LWD (logging-while-drilling) layer
  (:mod:`fwap.lwd`): a steel-collar :class:`~fwap.synthetic.Mode`
  factory, a slowness-band notch filter for collar rejection, and
  a quadrupole-source ring synthesizer + m=2 receiver-side stacker
  (Tang & Cheng 2004 sect. 2.4–2.5);
* a picker → log-curve bridge :func:`~fwap.picker.track_to_log_curves`
  that converts a track of :class:`~fwap.picker.DepthPicks` into the
  ``{mnemonic: ndarray}`` dict the LAS / DLIS writers consume
  directly;
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
