# fwap -- Full-Waveform Acoustic Processing

[![CI](https://github.com/ingehap/B_Mari_Full-Waveform-Acoustic-Data-Processing/actions/workflows/ci.yml/badge.svg)](https://github.com/ingehap/B_Mari_Full-Waveform-Acoustic-Data-Processing/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ingehap/B_Mari_Full-Waveform-Acoustic-Data-Processing/branch/main/graph/badge.svg)](https://codecov.io/gh/ingehap/B_Mari_Full-Waveform-Acoustic-Data-Processing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python implementation of the algorithms described in

> Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
> *Full Waveform Acoustic Data Processing.*
> Editions Technip, Paris, 136 pp. ISBN 978-2-7108-0664-6.
> (Originally published in French as *Traitement des diagraphies
> acoustiques.*)

The book picks four borehole-acoustic problems and works each one
through from a raw multichannel waveform to a log curve a petrophysicist
can actually use. This repository provides a modern NumPy/SciPy
implementation of the four chapter algorithms, plus two extensions
(cross-dipole anisotropy and Q attenuation).

## Chapter-to-module map

| Book part | Topic | Module(s) |
|-----------|-------|-----------|
| Part 1 | AI picking of waves on full-waveform acoustic data | [`fwap.coherence`](fwap/coherence.py), [`fwap.picker`](fwap/picker.py) |
| Part 2 | Wave separation in acoustic well logging            | [`fwap.wavesep`](fwap/wavesep.py) |
| Part 3 | Intercept-time inversion + dipole-flexural processing | [`fwap.tomography`](fwap/tomography.py), [`fwap.dispersion`](fwap/dispersion.py) |
| Part 4 | Dip measurement based on acoustic data               | [`fwap.dip`](fwap/dip.py) |
| (extension) | Cross-dipole Alford rotation                    | [`fwap.anisotropy`](fwap/anisotropy.py) |
| (extension) | Q from array sonic                              | [`fwap.attenuation`](fwap/attenuation.py) |
| (extension) | Elastic moduli from Vp, Vs, rho                 | [`fwap.rockphysics`](fwap/rockphysics.py) |
| (extension) | LAS read / write                                | [`fwap.io.read_las` / `write_las`](fwap/io.py) |
| (extension) | DLIS read / write                               | [`fwap.io.read_dlis` / `write_dlis`](fwap/io.py) |
| (extension) | SEG-Y waveform read                             | [`fwap.io.read_segy`](fwap/io.py) |

Helpers: [`fwap.synthetic`](fwap/synthetic.py) (canonical test gathers),
[`fwap.demos`](fwap/demos.py) (one worked example per chapter),
[`fwap.cli`](fwap/cli.py) (command-line demo runner).

## Installation

```bash
pip install -e .
```

The package requires Python >= 3.9. Core dependencies are NumPy,
SciPy, Matplotlib (Matplotlib only for the demos and CLI), and the
log-format libraries `lasio` (LAS), `dlisio` + `dliswriter` (DLIS),
and `segyio` (SEG-Y).

## Quick start

Run every demo and write diagnostic figures to `figures/`:

```bash
python -m fwap
# or
fwap all
```

Run a specific demo:

```bash
fwap stc          # Part 1: STC + rule-based picker (P / S / Stoneley)
fwap pseudorayleigh # Part 1: 4-mode picker incl. pseudo-Rayleigh
fwap wavesep      # Part 2: f-k filter + SVD / Karhunen-Loeve
fwap taup         # Part 2: tau-p / slant-stack / linear Radon
fwap intercept    # Part 3: Coppens & Mari intercept-time inversion
fwap dipole       # Part 3: dipole flexural dispersion
fwap dip          # Part 4: dip / azimuth from azimuthal array
fwap alford       # Extension: cross-dipole Alford rotation
fwap attenuation  # Extension: Q by centroid shift and spectral ratio
fwap las          # Extension: LAS I/O round-trip
fwap dlis         # Extension: DLIS I/O round-trip
fwap segy         # Extension: SEG-Y I/O round-trip
```

Process a real SEG-Y gather and print the P / S / Stoneley picks:

```bash
fwap process gather.sgy --offset-scale 1000
```

Programmatic use:

```python
from fwap import (
    ArrayGeometry, monopole_formation_modes, synthesize_gather,
    stc, pick_modes,
)

geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5)
data = synthesize_gather(geom, monopole_formation_modes())
surface = stc(data, dt=geom.dt, offsets=geom.offsets,
              window_length=4.0e-4)
picks = pick_modes(surface)           # {"P": ..., "S": ..., "Stoneley": ...}
```

## Documentation

Build the full API reference locally:

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

The docs include an end-to-end Jupyter notebook
([`docs/notebooks/workflow.ipynb`](docs/notebooks/workflow.ipynb))
that walks through synthesise → process → pick → derive moduli →
write LAS → read back in ~30 lines of library code.

The CI workflow builds the same docs on every push; a `docs-html`
artifact is uploaded on the Actions page. A `.readthedocs.yaml` is
included so the repository can be connected to ReadTheDocs without
further configuration.

## Tests

```bash
pip install pytest
pytest
```

The suite exercises one end-to-end path per algorithm family against
synthetic data with known ground truth.

## Recommended companion references

* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes.*
  CRC Press.
* Mari, J.-L., Glangeaud, F., & Coppens, F. (1999). *Signal Processing
  for Geologists and Geophysicists.* Editions Technip, Paris.
  ISBN 2-7108-0752-1.
* Mari, J.-L., & Vergniault, C. (2018). *Well Seismic Surveying and
  Acoustic Logging.* EDP Open.
* Coppens, F., & Mari, J.-L. (1995). Application of the intercept time
  method to full waveform acoustic data. *First Break* 13(1), 11-20.
* Coppens, F., & Mari, J.-L. (1995). Imagerie par refraction en
  diagraphie acoustique. *Revue de l'Institut Francais du Petrole*
  50(2), 143.

## License

See [LICENSE](LICENSE).
