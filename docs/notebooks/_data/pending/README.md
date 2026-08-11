# Traced but not yet scoreable

CSVs here are finished digitisations that **cannot be scored yet** because
the model parameters needed to compute the `fwap` curve are not known.
They are deliberately *not* in `_data/` proper: a CSV there is picked up by
`check_overlay` on sight and asserted against the 5 % budget, and scoring
against a geometry nobody has verified would produce a number that looks
like validation and is not.

Nothing in the notebook reads this directory.

## Contents

Four flexural **phase**-velocity curves from Ellefsen, K. J., Cheng, C. H.,
& Schmitt, D. P. (1988), *Acoustic Logging Guided Waves In Transversely
Isotropic Formations*, MIT Earth Resources Laboratory
([DSpace](https://dspace.mit.edu/handle/1721.1/75100)).

| File | Figure | Branch |
|------|--------|--------|
| `..._fig2_flexural_vti_hard.csv` | fig 2 | transversely isotropic, hard |
| `..._fig2_flexural_iso_hard.csv` | fig 2 | equivalent isotropic, hard |
| `..._fig4_flexural_vti_soft.csv` | fig 4 | transversely isotropic, soft |
| `..._fig4_flexural_iso_soft.csv` | fig 4 | equivalent isotropic, soft |

Each figure plots the TI formation (solid) against an *equivalent
isotropic* one (dashed) defined to have the same vertical P- and S-wave
velocities. So each pair ties two solvers from one trace: the TI branch
scores `flexural_dispersion_vti`, the isotropic branch scores
`flexural_dispersion`.

73 points each, 1.5-19.5 kHz, traced at 400 dpi. Axis calibration is from
the panel frame and ticks only — velocity 1.2-1.8 km/s (fig 2, hard) and
1.1-1.5 km/s (fig 4, soft), frequency 0-20 kHz — converted as
`slowness = 1 / (v_km_s * 1000)`.

## What is missing

The report states **no numbers at all**. Its fig 1 is a labelled schematic;
the elastic constants are deferred to Thomsen (1986) — Green River shale
(hard) and shale (5000) (soft) — and no borehole radius or fluid properties
appear anywhere in it.

To promote these files into `_data/`:

1. Get **Thomsen, L. (1986), *Weak elastic anisotropy*, Geophysics 51(10),
   1954-1966**, table 1, rows "Green River shale" and "shale (5000)":
   `V_P0`, `V_S0`, `rho`, `epsilon`, `delta`, `gamma`.
2. Convert to the stiffnesses `flexural_dispersion_vti` takes:
   ```
   C33 = rho * V_P0**2          C11 = C33 * (1 + 2*epsilon)
   C44 = rho * V_S0**2          C66 = C44 * (1 + 2*gamma)
   C13 = sqrt(2*C33*(C33 - C44)*delta + (C33 - C44)**2) - C44
   ```
3. Settle the borehole radius. `a` = 0.10 m is Schmitt's own value in the
   companion ERL reports and is the obvious first guess — but treat it as a
   *prediction*: if the curve then matches, that corroborates it, and if it
   does not, report the mismatch rather than tuning `a` until it fits.
4. Move the CSVs up one directory and point section 5's `check_overlay`
   calls at them.

## Two anchors that identify the right Thomsen rows

Read straight off the figures, and **not** used in the tracing:

* **Hard formation: `V_S0` = 1775 m/s.** Both fig 2 branches converge to
  1775.0 and 1775.1 m/s at low frequency.
* **Soft formation: `V_S0` = 1488 m/s.** Both fig 4 branches converge to
  1488.0 m/s.

That the TI and equivalent-isotropic curves — traced independently, one
solid and one dashed — agree to 0.1 m/s and 0.0 m/s is the internal check
on the digitising, since the equivalent isotropic formation is *defined* to
share the TI formation's vertical S-wave velocity.

If Thomsen's Green River shale row does not give `V_S0` ≈ 1775 m/s, the
hard formation is not Green River shale and something upstream is wrong;
same for shale (5000) at 1488 m/s. Tsvankin's figure 1.12 supplies a second
check for the hard rock only: `epsilon` = 0.195, `delta` = -0.22.
