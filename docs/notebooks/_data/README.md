# Digitised reference data for the validation notebook

`cylindrical_biot_validation.ipynb` overlays `fwap` dispersion
output on top of digitised reference curves. The reference CSVs
live in this directory.

## Status

**Two curves are shipped**, both traced from Schmitt & Cheng (1987):

| File | Scores |
|------|--------|
| `schmitt_cheng_1987_fig8a_flexural_slow.csv` | **PASS** — 1.27 % RMS against `flexural_dispersion` |
| `schmitt_cheng_1987_fig2_flexural_fast.csv`  | **FAIL** — 36.8 % RMS; expected, see below |

The rest are still wanted. Each un-digitised section of the validation
notebook ships with the `fwap` curve only and prints a clearly-marked
`TODO: digitise <FIGURE>` line.

The fast-formation failure is roadmap A.2, not a surprise: above roughly
10 kHz the solver's `(V_R, V_S)` search window no longer contains the
fundamental flexural branch and the returned root is an overtone. The
overlay is marked `known_defect=` in the notebook, which **inverts** the
assertion rather than relaxing it — the cell fails if that curve starts
passing, so the fix trips the marker instead of leaving a stale exemption
behind. Raising the budget to accommodate it would have hidden the
eventual fix as effectively as it hides the bug.

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
| `tang_cheng_2004_fig3_7_quadrupole_slow.csv` | Tang & Cheng 2004 fig 3.7                             | quadrupole, slow formation |
| `tang_cheng_2004_fig3_10_quadrupole_fast.csv` | Tang & Cheng 2004 fig 3.10                           | quadrupole, fast formation |
| `tang_cheng_2004_fig7_1_stoneley_cased.csv` | Tang & Cheng 2004 fig 7.1                              | cased-hole Stoneley        |
| `schmitt_1989_fig5_flexural_vti_qP.csv` | Schmitt 1989 fig 5                                         | VTI flexural, qP branch    |
| `schmitt_1989_fig5_flexural_vti_qSV.csv` | Schmitt 1989 fig 5                                        | VTI flexural, qSV branch   |

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
