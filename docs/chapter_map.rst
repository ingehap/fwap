Chapter-to-module map
=====================

Each algorithm module of :mod:`fwap` implements one Part of Mari,
Coppens, Gavin & Wicquart (1994). Several additional modules
implement downstream workflows and log-format I/O that are not in
the book.

+---------------------+----------------------------------------------------+--------------------------------+
| Book part           | Topic                                              | Module(s)                      |
+=====================+====================================================+================================+
| Part 1              | AI picking of P / S / pseudo-Rayleigh / Stoneley   | :mod:`fwap.coherence`,         |
|                     | (slowness windows + STC + Viterbi continuity +     | :mod:`fwap.picker`             |
|                     | wavelet-shape and onset-polarity expert rules)     |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| Part 2              | Wave separation: f-k, tau-p / slant-stack, and     | :mod:`fwap.wavesep`            |
|                     | SVD / Karhunen-Loeve                               |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| Part 3              | Intercept-time inversion (with altered-zone        | :mod:`fwap.tomography`,        |
|                     | thickness + velocity-contrast pair) and            | :mod:`fwap.dispersion`         |
|                     | dipole-flexural processing                         |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| Part 4              | Dip measurement based on acoustic data             | :mod:`fwap.dip`                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Cross-dipole Alford rotation + petrophysical       | :mod:`fwap.anisotropy`         |
|                     | labelling (max-H stress azimuth, splitting time,   |                                |
|                     | fracture indicator); VTI Thomsen-gamma from        |                                |
|                     | dipole C_44 and Stoneley C_66 (White / Norris      |                                |
|                     | tube-wave inversion); Backus (1962) layered-       |                                |
|                     | medium average producing the 5-parameter VTI       |                                |
|                     | elastic tensor; Tsvankin-2001 phase- and group-    |                                |
|                     | velocity surfaces for qP / qSV / SH wavefronts     |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Q from array sonic                                 | :mod:`fwap.attenuation`        |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Elastic moduli from Vp, Vs, rho;                   | :mod:`fwap.rockphysics`        |
|                     | Reuss / Voigt / Hill mixing; Gassmann fluid        |                                |
|                     | substitution; **four-tool Stoneley fracture /      |                                |
|                     | permeability suite**: slowness indicator,          |                                |
|                     | amplitude indicator, Tang-Cheng-Toksoz quantitative|                                |
|                     | matrix-permeability inversion, Hornby (1989)       |                                |
|                     | aperture inversion + a unified                     |                                |
|                     | :func:`stoneley_fracture_density` combiner;        |                                |
|                     | slow-formation Vs from low-frequency Stoneley      |                                |
|                     | phase velocity                                     |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | **Geomechanics drilling-decision pipeline** on top | :mod:`fwap.geomechanics`       |
|                     | of :class:`~fwap.rockphysics.ElasticModuli`:       |                                |
|                     | Rickman brittleness / fracability;                 |                                |
|                     | Lacy sandstone UCS + tensile-strength rule of      |                                |
|                     | thumb; Bratli-Risnes sand stability;               |                                |
|                     | density-log overburden + hydrostatic helper;       |                                |
|                     | sonic-derived pore-pressure (Eaton 1975 for        |                                |
|                     | undercompaction; Bowers 1995 for unloading);       |                                |
|                     | Eaton uniaxial-strain closure stress;              |                                |
|                     | Kirsch wall stresses + Mohr-Coulomb shear-         |                                |
|                     | breakout + Hubbert-Willis tensile-breakdown +      |                                |
|                     | safe-mud-weight window for **vertical and          |                                |
|                     | inclined** wells; one-call                         |                                |
|                     | :func:`geomechanics_indices` bundle                |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Cylindrical-borehole surface-wave speeds (Rayleigh | :mod:`fwap.cylindrical`,       |
|                     | + physics-grounded flexural dispersion law) and    | :mod:`fwap.cylindrical_solver` |
|                     | the Schmitt (1988) cylindrical-Biot modal-         |                                |
|                     | determinant solver: n=0 Stoneley (3x3) and n=1     |                                |
|                     | dipole flexural (4x4) in the bound-mode regime,    |                                |
|                     | with :func:`stoneley_dispersion` and               |                                |
|                     | :func:`flexural_dispersion` public APIs            |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | LWD phenomenological layer: collar Mode factory,   | :mod:`fwap.lwd`                |
|                     | slowness-band notch for collar rejection,          |                                |
|                     | quadrupole-source ring synthesis, m=2 receiver-    |                                |
|                     | side stacker, LWD-tuned picker priors              |                                |
|                     | (Tang & Cheng 2004 sect. 2.4-2.5)                  |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Dispersion-curve / dispersive-STC additions:       | :mod:`fwap.dispersion`         |
|                     | dispersive STC for the pseudo-Rayleigh /           |                                |
|                     | guided trapped mode; stress-vs-intrinsic           |                                |
|                     | anisotropy classifier from a fast / slow flexural  |                                |
|                     | dispersion-curve crossover (Sinha & Kostek 1996)   |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Picker -> log-curve bridge:                        | :mod:`fwap.picker`             |
|                     | :func:`track_to_log_curves` converts a per-depth   |                                |
|                     | pick track into a LAS/DLIS-ready                   |                                |
|                     | ``{mnemonic: ndarray}`` dict                       |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | LAS / DLIS / SEG-Y log-format I/O                  | :mod:`fwap.io`                 |
+---------------------+----------------------------------------------------+--------------------------------+

Supporting modules
------------------

* :mod:`fwap.synthetic` -- canonical P/S/Stoneley/pseudo-Rayleigh
  test gathers, plus the phenomenological flexural and pseudo-
  Rayleigh dispersion laws used by the demos and tests.
* :mod:`fwap.demos`     -- one worked example per chapter (plus
  one per extension).
* :mod:`fwap.cli`       -- the ``fwap`` command-line demo runner.

Recommended companion references
--------------------------------

* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes.*
  CRC Press. (Forward / mode-theory companion to Mari 1994.)
* Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole Acoustic
  Methods.* Elsevier. (Inverse / processing companion; underpins the
  Stoneley-permeability, Stoneley-fracture, Thomsen-:math:`\gamma`
  and LWD layers.)
* Mari, J.-L., Glangeaud, F., & Coppens, F. (1999). *Signal Processing
  for Geologists and Geophysicists.* Editions Technip, Paris.
  ISBN 2-7108-0752-1.
* Mari, J.-L., & Vergniault, C. (2018). *Well Seismic Surveying and
  Acoustic Logging.* EDP Open.
* Coppens, F., & Mari, J.-L. (1995). Application of the intercept time
  method to full waveform acoustic data. *First Break* 13(1), 11-20.
* Hornby, B. E., Johnson, D. L., Winkler, K. W., & Plumb, R. A.
  (1989). Fracture evaluation using reflected Stoneley-wave
  arrivals. *Geophysics* 54(10), 1274-1288.
* Rickman, R., Mullen, M. J., Petre, J. E., Grieser, W. V., &
  Kundert, D. (2008). A practical use of shale petrophysics for
  stimulation design optimization. *SPE 115258*.
* Sinha, B. K., & Kostek, S. (1996). Stress-induced azimuthal
  anisotropy in borehole flexural waves. *Geophysics* 61(6),
  1899-1907.
