# Digitised reference data for the validation notebook

`cylindrical_biot_validation.ipynb` overlays `fwap` dispersion
output on top of digitised reference curves. The reference CSVs
live in this directory.

## Status

**Five curves are shipped** and all five pass, every one at the
digitisation floor:

| File | Solver | Score |
|------|--------|-------|
| `schmitt_cheng_1987_fig8a_flexural_slow.csv` | `flexural_dispersion` | **0.04 %** RMS, worst 0.15 %, 55/55 pts |
| `schmitt_cheng_1987_fig2_flexural_fast.csv`  | `flexural_dispersion` | **0.37 %** RMS, worst 1.37 %, 61/89 pts |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_vti_soft.csv` | `flexural_dispersion_vti` | **0.30 %** RMS, worst 1.29 %, 70/73 pts |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_iso_soft.csv` | `flexural_dispersion` | **0.17 %** RMS, worst 0.34 %, 73/73 pts |
| `ellefsen_cheng_schmitt_1988_fig2_flexural_iso_hard.csv` | `flexural_dispersion` | **0.45 %** RMS, worst 0.91 %, 17/73 pts |

The third row is the **first external check `flexural_dispersion_vti` has
ever had**.

**Two of them are scored thinly, and say so.** The fast-formation path
returns `NaN` outside a narrow band rather than a wrong root, so
`..._fig2_flexural_iso_hard.csv` is scored over only 2.0-6.0 kHz (17 of 73
points) and `schmitt_cheng_1987_fig2_flexural_fast.csv` over 2.75-17.75 kHz
(61 of 89). Outside those bands **the overlay is silent, not green**.

These are deliberately *independent* of the reads recorded under roadmap
A.1, which live as constants in `tests/test_cylindrical_solver.py` rather
than as CSVs — different session, different resolution (400 dpi here),
different tracer. They agree: this trace puts fig 2(a) at 1494 m/s at
24.5 kHz against A.1's 1493 m/s at 24.9 kHz. Exporting A.1's constants
into this directory instead would have made the overlay a restatement of
a check that already exists.

The rest are still wanted. Each un-digitised section of the validation
notebook ships with the `fwap` curve only and prints a clearly-marked
`TODO: digitise <FIGURE>` line.

**The scoring is already wired.** Dropping a CSV here needs no notebook
edit: the section's `check_overlay(...)` call switches from printing
the TODO to plotting the overlay, printing an RMS verdict, and
asserting the 5 % budget. The machinery lives in `fwap.validation` and
is covered by `tests/test_validation.py`, so it is known to work before
any real reference exists.

## Schema

One CSV per published figure, two columns, no header rows.

```
freq_hz, slowness_s_per_m
1000.0,  0.000597
2000.0,  0.000601
...
```

Suggested filenames (matching the notebook section titles):

| File                                  | Reference                                                    | Mode                       |
|---------------------------------------|--------------------------------------------------------------|----------------------------|
| `paillet_cheng_1991_fig4_5_stoneley.csv`   | Paillet & Cheng 1991 fig 4.5                          | Stoneley, limestone        |
| `paillet_cheng_1991_fig4_5_pseudo_rayleigh.csv` | Paillet & Cheng 1991 fig 4.5                     | pseudo-Rayleigh, limestone |
| `schmitt_cheng_1987_fig8a_flexural_slow.csv` | Schmitt & Cheng 1987 fig 8(a) *(shipped)*             | flexural, slow sandstone   |
| `schmitt_cheng_1987_fig2_flexural_fast.csv` | Schmitt & Cheng 1987 fig 2(a) *(shipped)*             | flexural, fast sandstone   |
| *(none — reference withdrawn)*        | quadrupole slow + fast                                       | see note below             |
| *(none — reference withdrawn)*        | cased-hole Stoneley                                          | see note below             |
| `ellefsen_cheng_schmitt_1988_fig2_flexural_vti_hard.csv` | Ellefsen/Cheng/Schmitt 1988 fig 2 | elastic VTI flexural, hard |
| `ellefsen_cheng_schmitt_1988_fig2_flexural_iso_hard.csv` | Ellefsen/Cheng/Schmitt 1988 fig 2 | equivalent isotropic, hard |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_vti_soft.csv` | Ellefsen/Cheng/Schmitt 1988 fig 4 | elastic VTI flexural, soft |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_iso_soft.csv` | Ellefsen/Cheng/Schmitt 1988 fig 4 | equivalent isotropic, soft |

The two flexural rows were `schmitt_1988_fig4_flexural_{slow,fast}.csv`
until the reference was actually opened. The source consulted is the
open-access MIT ERL precursor, Schmitt & Cheng (1987) report 1987.8
([DSpace](https://dspace.mit.edu/handle/1721.1/121148)), whose **fig 4
is a time-domain shot gather, not a dispersion curve**. Its flexural
dispersion lives in fig 2(a) (fast sandstone) and fig 8(a) (slow
sandstone). The 1988 JASA article is paywalled and was not consulted, so
its own figure numbering is unverified — hence filenames that name the
document actually traced.

Two further figures in that report are worth digitising and are not in
the table above because no notebook section covers them yet: **fig 7**
(flexural and screw for granite, limestone and fast sandstone — three
fast formations in one figure) and **figs 20/21** (well-bonded cased-hole
flexural and screw, which would give section 4 a cased-hole reference
without needing Tang & Cheng).

The VTI rows were `schmitt_1989_fig5_flexural_vti_{qP,qSV}.csv` until the
reference was opened. In the open-access ERL precursor (Schmitt 1988.13)
**fig 5 is a monopole microseismogram**, not a dispersion curve; that
report's flexural dispersion is fig 22, and it is *poroelastic* while
`flexural_dispersion_vti` is elastic. There is also no qP/qSV branch pair
and no flexural splitting — a vertical borehole in a VTI medium with a
vertical symmetry axis is azimuthally isotropic. The `_qSV` row was an
orphan no notebook cell read.

The elastic reference is **Ellefsen, Cheng & Schmitt (1988)**, MIT ERL
([DSpace](https://dspace.mit.edu/handle/1721.1/75100)), figs 2 (hard) and
4 (soft). Each plots the TI formation against its *equivalent isotropic*
formation, so one figure ties both `flexural_dispersion_vti` and
`flexural_dispersion`.

That report states no numbers — constants deferred to Thomsen (1986), no
borehole radius, no fluid properties, its fig 1 a labelled schematic — so
the geometry had to be assembled from elsewhere and **checked before use**:

* **The rocks.** Thomsen (1986) table 1: Green River shale (Schock et al.
  1974 row) `V_P0` 3292, `V_S0` 1768 m/s, `rho` 2075, `eps` 0.195,
  `delta` -0.220, `gamma` 0.180; shale (5000) (Jones & Wang 1981 row)
  `V_P0` 3048, `V_S0` 1490 m/s, `rho` 2420, `eps` 0.255, `delta` -0.050,
  `gamma` 0.480.
* **How the rows were confirmed.** The curves were traced *first*, and
  their low-frequency limits — where the flexural branch tends to `V_S` —
  read **1775** and **1488** m/s against the table's 1768 and 1490:
  **+0.4 %** and **+0.13 %**. That also settles which of table 1's two
  Green River shale entries is meant, since the other (Podio et al.) has
  `V_S0` = 2432 m/s. Tsvankin's figure 1.12 gives the same row's
  `eps` = 0.195 and `delta` = -0.22 independently.
* **The radius was a prediction, not a fit.** `a` = 0.10 m is Schmitt's
  value in the companion ERL reports; it was used untuned, and all three
  scoreable overlays then landed under 0.5 % RMS. A wrong radius shifts the
  knee in frequency and would have shown up at once. Deriving it from the
  figures instead would have been the silent refit this directory exists to
  prevent.

**One of the four branches is still unscored**, and the blocker is the
solver rather than the reference: `flexural_dispersion_vti` raises
`NotImplementedError` for fast-formation TI, which is exactly Green River
shale at `V_Sv` 1768 > `V_f` 1500. That curve waits in
[`pending/`](pending/README.md) with its geometry already verified.

**The Tang & Cheng 2004 rows are withdrawn, not renamed.** Sections 3 and
4 cited that book for figures it does not contain:

* **"fig 7.1 — cased-hole Stoneley"** and the geometry quoted from
  **"sect. 7.2"** cannot exist: the book has **six chapters** (1 Overview,
  2 Elastic Wave Propagation in Boreholes, 3 Velocity and Attenuation
  Estimation, 4 Permeability Estimation, 5 Anisotropic Formations,
  6 Summary). So section 4's formation, casing, cement and radius are all
  unsourced — and its casing (5860/3140/7800) matches neither real casing
  row in the Schmitt ERL reports (6098/3354/7500, 6096/3352/7500).
* **"figs 3.7 / 3.10 — quadrupole slow + fast"**: both figures exist and
  neither is a dispersion curve. Fig 3.7 is waveform matching, fig 3.10 is
  acoustic time delay; chapter 3 is a *processing* chapter.

Both were confirmed against a physical copy. No replacement figure numbers
are asserted, because nobody has read the relevant chapters — inventing a
plausible-looking pointer is how this started.

The repository already contained the correct chapter list, in
[`docs/ideas/Tang2004.md`](../../ideas/Tang2004.md): *"The book is
organized into six chapters."* That is the second time the right answer was
already in the tree while several other files carried a wrong one — the
Schmitt 1988 page range was the first, correct in `fwap/validation.py` and
wrong in thirteen other places.

**Cased-hole Stoneley is now the one cased mode with no external tie.**
Roadmap A.1 tied cased flexural and screw from Schmitt & Cheng figs 20/21;
the monopole case is not in that report, and the cased Stoneley figures in
Schmitt 1988.13 (figs 59, 66) are transversely isotropic *poroelastic*,
which `stoneley_dispersion_layered` cannot reproduce.

## Workflow for adding an overlay

1. Digitise the figure (e.g. WebPlotDigitizer) into a CSV with the
   schema above.
2. Drop the CSV here under the documented name.
3. Re-run the notebook. That is the whole procedure — there is no code
   to edit.

## Units are checked, not assumed

The loader (`fwap.load_reference_curve`) refuses input whose magnitude
indicates a units mistake, and names the suspected error:

| Symptom in the CSV                | What it usually means            |
|-----------------------------------|----------------------------------|
| slowness around 60–700            | axis read in µs/ft or µs/m       |
| slowness around 1500–5000         | a *velocity* axis traced instead |
| frequency around 1–30             | axis left in kHz                 |

It also sorts click-order output by frequency, and rejects duplicate
frequencies, non-positive values and curves shorter than three points.

It never rescales the data to make it fit. That is deliberate: a
reference quietly adjusted to match would agree with a wrong solver
exactly as readily as with a right one, which would turn this whole
directory into decoration.

## Validation gate

`pytest --nbval-lax docs/notebooks/cylindrical_biot_validation.ipynb`
re-executes every cell and fails on any error — including the
`assert score.passed` inside `check_overlay`. So once a reference CSV
is present here, a solver regression that moves the curve by more than
the 5 % RMS budget fails the notebook.

With no CSVs present the notebook still runs end to end, and its
closing cell states plainly that nothing in it is validated against
literature yet. Green plots are not evidence.
