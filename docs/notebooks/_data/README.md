# Digitised reference data for the validation notebook

`cylindrical_biot_validation.ipynb` overlays `fwap` dispersion
output on top of digitised reference curves. The reference CSVs
live in this directory.

## Status

Empty until reference curves are digitised. Each section of the
validation notebook ships with the `fwap` curve only and prints a
clearly-marked `TODO: digitise <FIGURE>` line.

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
| `schmitt_1988_fig4_flexural_slow.csv` | Schmitt 1988 fig 4                                           | flexural, slow formation   |
| `schmitt_1988_fig4_flexural_fast.csv` | Schmitt 1988 fig 4                                           | flexural, fast formation   |
| `tang_cheng_2004_fig3_7_quadrupole_slow.csv` | Tang & Cheng 2004 fig 3.7                             | quadrupole, slow formation |
| `tang_cheng_2004_fig3_10_quadrupole_fast.csv` | Tang & Cheng 2004 fig 3.10                           | quadrupole, fast formation |
| `tang_cheng_2004_fig7_1_stoneley_cased.csv` | Tang & Cheng 2004 fig 7.1                              | cased-hole Stoneley        |
| `schmitt_1989_fig5_flexural_vti_qP.csv` | Schmitt 1989 fig 5                                         | VTI flexural, qP branch    |
| `schmitt_1989_fig5_flexural_vti_qSV.csv` | Schmitt 1989 fig 5                                        | VTI flexural, qSV branch   |

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
