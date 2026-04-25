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
| (extension)         | Cross-dipole Alford rotation, plus a               | :mod:`fwap.anisotropy`         |
|                     | petrophysical labelling (max-H stress azimuth,     |                                |
|                     | splitting-time delay, fracture indicator)          |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Q from array sonic                                 | :mod:`fwap.attenuation`        |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Elastic moduli from Vp, Vs, rho;                   | :mod:`fwap.rockphysics`        |
|                     | Reuss / Voigt / Hill mixing; Gassmann fluid        |                                |
|                     | substitution; Stoneley permeability indicator      |                                |
+---------------------+----------------------------------------------------+--------------------------------+
| (extension)         | Cylindrical-borehole surface-wave speeds and a     | :mod:`fwap.cylindrical`        |
|                     | physics-grounded flexural dispersion law           |                                |
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
  CRC Press.
* Mari, J.-L., Glangeaud, F., & Coppens, F. (1999). *Signal Processing
  for Geologists and Geophysicists.* Editions Technip, Paris.
  ISBN 2-7108-0752-1.
* Mari, J.-L., & Vergniault, C. (2018). *Well Seismic Surveying and
  Acoustic Logging.* EDP Open.
* Coppens, F., & Mari, J.-L. (1995). Application of the intercept time
  method to full waveform acoustic data. *First Break* 13(1), 11-20.
