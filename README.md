# fwap -- Full-Waveform Acoustic Processing

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
implementation of the four chapter algorithms plus an extension layer
covering the post-1994 borehole-acoustic literature: cross-dipole and
VTI anisotropy, Stoneley-derived petrophysics, a geomechanics
drilling-decision pipeline, the cylindrical-Biot modal solver, an LWD
acquisition layer, and LAS / DLIS / SEG-Y log-format I/O.

## Chapter-to-module map

| Book part | Topic | Module(s) |
|-----------|-------|-----------|
| Part 1 | AI picking of waves on full-waveform acoustic data | [`fwap.coherence`](fwap/coherence.py), [`fwap.picker`](fwap/picker/) |
| Part 2 | Wave separation in acoustic well logging            | [`fwap.wavesep`](fwap/wavesep.py) |
| Part 3 | Intercept-time inversion + dipole-flexural processing | [`fwap.tomography`](fwap/tomography.py), [`fwap.dispersion`](fwap/dispersion.py) |
| Part 4 | Dip measurement based on acoustic data               | [`fwap.dip`](fwap/dip.py) |
| (extension) | Cross-dipole Alford rotation, Thomsen γ, Backus average, qP / qSV / SH velocity surfaces | [`fwap.anisotropy`](fwap/anisotropy/) |
| (extension) | Q from array sonic                              | [`fwap.attenuation`](fwap/attenuation.py) |
| (extension) | Elastic moduli (K, μ, E, ν), Reuss / Voigt / Hill mixing, Gassmann fluid substitution | [`fwap.rockphysics`](fwap/rockphysics.py) |
| (extension) | Stoneley-wave permeability / fracture indicators, Tang-Cheng-Toksoz inversion, Hornby aperture, slow-formation V<sub>s</sub> | [`fwap.stoneley`](fwap/stoneley.py) |
| (extension) | Geomechanics drilling-decision pipeline: indices, pore pressure (Eaton / Bowers), vertical and inclined wellbore stability | [`fwap.geomechanics`](fwap/geomechanics/) |
| (extension) | Cylindrical-Biot modal solver: n=0 Stoneley, n=1 flexural, n=2 quadrupole (bound + leaky), layered / cased hole, VTI, and the debonded regime (fluid microannulus + crack wave) | [`fwap.cylindrical`](fwap/cylindrical.py), [`fwap.cylindrical_solver`](fwap/cylindrical_solver/) |
| (extension) | LWD collar rejection + quadrupole-ring stack    | [`fwap.lwd`](fwap/lwd.py) |
| (extension) | LAS / DLIS / SEG-Y log-format I/O               | [`fwap.io`](fwap/io/) |

Helpers: [`fwap.synthetic`](fwap/synthetic.py) (canonical test gathers),
[`fwap.demos`](fwap/demos/) (one worked example per chapter),
[`fwap.cli`](fwap/cli.py) (command-line demo runner).

## `sonic_ml` — the optional machine-learning layer

[`sonic_ml/`](sonic_ml/) is an **in-repo sibling package** that treats the
`fwap` forward solver as a labelled-data factory and learns surrogate and
inverse models on top of it. It is deliberately separate: `fwap` stays pure
NumPy/SciPy, and no ML dependency ever enters the physics package. Installing
or importing `fwap` never pulls in PyTorch.

| Layer | What it does | Module |
|-------|--------------|--------|
| Data | Cylindrical-Biot forward solves → a versioned `.npz` of (parameters, dispersion, waveform) triples, open-hole and cased-hole | [`scripts/gen_surrogate_dataset.py`](scripts/gen_surrogate_dataset.py) |
| Spine (torch-free) | Defensive loader, provenance capture, regime-stratified splits, standardization, determinism | [`sonic_ml.loader`](sonic_ml/src/sonic_ml/loader.py), [`sonic_ml.provenance`](sonic_ml/src/sonic_ml/provenance.py), [`sonic_ml.split`](sonic_ml/src/sonic_ml/split.py) |
| Benchmark | Model-agnostic scoring of any Vs or cement-bond predictor, with bootstrap CIs and per-regime rows | [`sonic_ml.bench`](sonic_ml/src/sonic_ml/bench/) |
| Baselines | The classical bar an ML model must clear (STC, f-k dispersion, Stoneley bond calibration) | [`sonic_ml.baselines`](sonic_ml/src/sonic_ml/baselines/) |
| Models | Forward surrogate, DL-FWI inverse net with calibrated uncertainty, low-latency LWD variant, FNO / DeepONet operators, cased-hole operator + cement-bond inverse | [`sonic_ml.models`](sonic_ml/src/sonic_ml/models/) |

Two headline results, reported with the gap between them intact:

* **Open hole** — the learned inverse recovers V<sub>s</sub> roughly an order of
  magnitude more accurately than classical slowness-time processing on identical
  held-out gathers, including the fast-formation regime where the two-mode
  gather starves classical STC.
* **Cased hole** — the cement-bond inverse reaches only ~2× the skill of
  predicting the mean. That is not a modelling failure: a forward sensitivity
  sweep shows cement stiffness moves the cased Stoneley curve ~7% across its
  prior while formation V<sub>s</sub> moves it ~1.5%, so the problem is only
  partially identifiable — and the heteroscedastic uncertainty head reports
  calibrated error bars that say so.

```bash
pip install -e ".[dev]"          # core fwap
pip install torch                # CPU wheel is fine
pip install -e "./sonic_ml[dev]" # the ML layer
```

`sonic_ml` is excluded from the core wheel and the core CI gate; it runs in its
own non-required workflow, so an ML failure can never block a physics change.

## Tutorial notebooks

| Notebook | Topic |
|----------|-------|
| [`open_hole_processing`](docs/notebooks/open_hole_processing.ipynb) | Parts 1–2: synthesize → STC → pick P/S/Stoneley → wave separation → log curves |
| [`open_hole_petrophysics`](docs/notebooks/open_hole_petrophysics.ipynb) | Part 3 + extensions: dispersion bias, moduli, Gassmann, Stoneley permeability, mud-weight window |
| [`sonic_ml_tutorial`](docs/notebooks/sonic_ml_tutorial.ipynb) | The ML loop: generate → train a DL-FWI inverse → beat the classical baseline |
| [`cased_hole_tutorial`](docs/notebooks/cased_hole_tutorial.ipynb) | Cement-bond evaluation, and an honest account of what is recoverable behind casing |
| [`cylindrical_biot_validation`](docs/notebooks/cylindrical_biot_validation.ipynb) | The modal solver checked against published oracle values |

The first two need only `fwap`; the middle two additionally need `sonic_ml` and
PyTorch. All are re-executed in CI, so the numbers in them are reproducible
rather than decorative.

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
fwap lwd          # Extension: LWD collar rejection + quadrupole stack
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
picks = pick_modes(surface)   # dict keyed by mode name (P / S / PseudoRayleigh / Stoneley)
```

## Documentation

Build the full API reference locally:

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

Pre-built PDF snapshots of the full manual and per-section subsets
live in [`docs/`](docs/); see [`docs/quickstart.rst`](docs/quickstart.rst)
for the list. The rendered manual covers the chapter-to-module map, the
[`sonic_ml`](docs/sonic_ml.rst) layer, the full API reference, and every
tutorial notebook listed above.

## Tests

```bash
pip install -e ".[dev]"
pytest                    # core suite; the perf bench is excluded by default
pytest tests/test_bench.py  # wall-clock perf suite, run deliberately
```

The suite exercises one end-to-end path per algorithm family against
synthetic data with known ground truth. Two guards are worth knowing about:

* [`scripts/check_public_api.py`](scripts/check_public_api.py) freezes the
  public surface — adding or renaming an exported name requires updating
  `fwap/__init__.py`, `docs/api.rst` and the guard together.
* [`tests/test_npz_schema_contract.py`](tests/test_npz_schema_contract.py)
  freezes the on-disk surrogate-dataset layout, so a core change that would
  silently mislabel downstream training data fails here instead.

The ML layer has its own suite, run from `sonic_ml/`:

```bash
cd sonic_ml && pytest
```

### Real-data integration tests

The suite above runs entirely on synthetics with planted answers. That is what
makes it assertable, and also what bounds it: a synthetic file is produced by
the same assumptions the reader holds, so it cannot catch a convention the
reader failed to anticipate. `tests/test_real_data.py` covers that gap using
files written by *other* software — a real Kansas Geological Survey well log
(wrapped LAS, 26 service-company curves), a SEG-Y written by `segyio`, and a
Schlumberger DSI sonic log from Utah FORGE carrying the tool's own
compressional and shear picks.

Those files are **not** in this repository. They are third-party, published
under their own terms, and are fetched on demand into a git-ignored directory:

```bash
python scripts/fetch_real_data.py --list        # registry, provenance, licences
python scripts/fetch_real_data.py --fetch all   # download + verify SHA-256
pytest tests/test_real_data.py
```

Without them those tests skip, so a normal `pytest` run and CI are unaffected.

#### What the real sonic log establishes, and what it does not

An earlier version of this section said no openly redistributable sonic log was
known to exist. That was wrong and is withdrawn: the Utah FORGE data is CC BY
4.0, and one of its logs is now registered.

The registered LAS carries Schlumberger's processed picks. The companion DLIS in
the same submission — 808 MB, not a viable test fixture — carries the
per-receiver waveforms those picks were derived from: eight receivers, 512
samples, monopole and both dipoles. Running fwap's own slowness-time coherence
and mode tracking over 400 contiguous frames and comparing against the vendor's
picks at the same depths:

| mode | median error | within 10 % |
|------|--------------|-------------|
| shear vs `DTSM` | **+0.12 %** | 96 % |
| compressional vs `DTCO` | −0.94 % | **95 %** |

The shear result is the strongest external evidence this package has. The
compressional result is the same log **after** a defect the comparison exposed
was found and fixed; before it, compressional agreed on only 62 % of depths.
That is exactly the kind of thing no synthetic test could have found — the
synthetic gathers are produced by the same forward model the picker is scored
against, so it could not disagree with itself.

The defect was mode confusion rather than imprecision: at 143 of the 400 depths
`track_modes` assigned the *same* STC peak to P and to S, reporting shear
slowness as compressional. The greedy loop ordered modes on arrival time only,
and equal times satisfy that trivially. It now refuses to give one arrival two
labels (`resolve_mode_collisions`, on by default), which moved 129 of those
depths to the right answer and left the shear pick bit-identical at all 400.

Which of the two labels is wrong turns out not to be decidable in general —
both directions occur — so the rule moves the faster-labelled mode only when it
has somewhere admissible to go, and otherwise changes nothing. It can never
leave a depth worse than before. See the `track_modes` docstring for the full
comparison, including against `viterbi_pick_joint`, which reaches 89 % on the
same surfaces by a different route and is still the better tool on confusions
that are not exact collisions.

The whole comparison now runs through fwap's own API.
`fwap.io.read_dlis_waveforms` reads the per-receiver waveforms a DLIS carries
and recovers the acquisition geometry from the file's RP66 AXIS records — 10 µs
sampling and eight receivers 6 in apart starting at 7.874 m, on this tool —
rather than from constants at the call site:

```python
curves = read_dlis(path)
curves.waveform_channels                 # {'PWF4': (8, 512), ...}

wf = read_dlis_waveforms(path, "PWF4")
surface = stc(wf.data[i], wf.sample_interval(), wf.offsets())
```

Two limitations remain. The waveform comparison is not part of CI, because the
fixture is a 471 MB archive containing an 808 MB file — so what defends the
picker fix in CI is a seeded synthetic, not the log that found it. And the
registered LAS's SHA-256 was computed from a mirror copy, because
`gdr.openei.org` was unreachable from the session that added it; the entry's
`provenance` says so. Until the first is addressed, `sonic_ml`'s headline
numbers are still measured against the same forward model that generated their
training data.

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
