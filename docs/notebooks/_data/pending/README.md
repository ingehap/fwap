# Traced but not yet scoreable

CSVs here are finished digitisations that **cannot be scored yet**. They
are deliberately not in `_data/` proper: a CSV there is picked up by
`check_overlay` on sight and asserted against the 5 % budget, so parking a
curve the notebook cannot evaluate would fail the notebook rather than
record a gap.

Nothing in the notebook reads this directory.

## Contents

One curve, from Ellefsen, K. J., Cheng, C. H., & Schmitt, D. P. (1988),
*Acoustic Logging Guided Waves In Transversely Isotropic Formations*, MIT
Earth Resources Laboratory
([DSpace](https://dspace.mit.edu/handle/1721.1/75100)).

| File | Figure | Branch |
|------|--------|--------|
| `..._fig2_flexural_vti_hard.csv` | fig 2 | transversely isotropic, hard |

73 points, 1.5-19.5 kHz, traced at 400 dpi. Its three siblings — fig 2's
equivalent-isotropic branch and both of fig 4's — were promoted to `_data/`
and now score at 0.45 %, 0.17 % and 0.30 % RMS.

## What is missing is the solver, not the reference

**The reference is complete and verified.** The formation is Green River
shale, from Thomsen (1986) table 1 (Schock et al. 1974 row): `V_P0` 3292,
`V_S0` 1768 m/s, `rho` 2075 kg/m^3, `eps` 0.195, `delta` -0.220,
`gamma` 0.180; water at 1500 m/s and 1000 kg/m^3; `a` = 0.10 m. That
geometry is confirmed — the three sibling curves score under 0.5 % with it.

What blocks this one is that **`flexural_dispersion_vti` raises
`NotImplementedError` for fast-formation TI**:

> `flexural_dispersion_vti` for fast-formation TI (`V_Sv` = 1768 m/s >
> `V_f` = 1500 m/s) is not implemented yet. The real-valued VTI modal
> determinant requires `F_f^2 = k_z^2 - (omega/V_f)^2 > 0`, i.e.
> `V_Sv < V_f`. The complex-determinant path mirroring the isotropic
> `_flexural_dispersion_fast_formation` is deferred.

Green River shale is fast at 1768 m/s against a 1500 m/s borehole fluid, so
this figure lands squarely on the unimplemented path. It is tracked as the
H.d follow-up in `docs/plans/cylindrical_biot_H.md`.

## To promote it

Once fast-formation TI is implemented, move the CSV up one directory and
add a fourth `check_overlay` call to section 5 with
`flexural_dispersion_vti` on the hard rock. No new digitising, no new
reference hunting — the curve and the geometry are both already here.

Worth knowing before that work starts: the *isotropic* fast-formation path
is itself sparse. Fig 2's equivalent-isotropic branch scores 0.45 % RMS but
over only **17 of 73** points, 2.0-6.0 kHz, because `flexural_dispersion`
returns `NaN` outside that band for this rock. A fast-formation TI path
that inherits the same behaviour would be scored just as thinly, and that
is a property of the solver rather than of the figure.
