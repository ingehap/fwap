# Digitised reference data for the validation notebook

`cylindrical_biot_validation.ipynb` overlays `fwap` dispersion
output on top of digitised reference curves. The reference CSVs
live in this directory.

## Status

Empty until reference curves are digitised. Each section of the
validation notebook ships with the `fwap` curve only; the overlay
cells are stubbed to a clearly-marked `TODO: digitise <FIGURE>` block
that becomes a real `pandas.read_csv(...)` + plot once the CSV lands.

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
2. Drop the CSV here.
3. In the corresponding notebook section, replace the `TODO`
   placeholder cell with:

   ```python
   ref = pd.read_csv("_data/<filename>.csv", names=["freq", "slowness"])
   plt.plot(ref["freq"], ref["slowness"], "k:", label="reference")
   ```

4. Re-run the notebook. The overlay panel renders.

## Validation gate

`pytest --nbval-lax docs/notebooks/cylindrical_biot_validation.ipynb`
re-executes every cell and fails on errors. A tighter gate
(per-curve RMS deviation < 5 %) is part of the eventual full Plan I
deliverable; it activates only once at least one reference CSV is
present (the notebook's `OVERLAY_AVAILABLE` flag controls the
assertion block).
