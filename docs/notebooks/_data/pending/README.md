# Traced but not yet scoreable

CSVs here are finished digitisations that **cannot be scored yet**. They
are deliberately not in `_data/` proper: a CSV there is picked up by
`check_overlay` on sight and asserted against the 5 % budget, so parking a
curve the notebook cannot evaluate would fail the notebook rather than
record a gap.

Nothing in the notebook reads this directory.

**In every case here the reference is finished and the solver is what is
missing.** None of these is waiting on more digitising or a better scan.

## Contents

| File | Source | Blocked on |
|------|--------|------------|
| `ellefsen_..._fig2_flexural_vti_hard.csv` | Ellefsen/Cheng/Schmitt 1988 fig 2 | fast-formation TI in `flexural_dispersion_vti` |
| `tubman_..._fig4b_pseudo_rayleigh1_cased.csv` | Tubman/Cheng/Toksoz 1984 fig 4(b) | no cased pseudo-Rayleigh API |
| `tubman_..._fig4b_pseudo_rayleigh2_cased.csv` | Tubman/Cheng/Toksoz 1984 fig 4(b) | no cased pseudo-Rayleigh API |
| `sinha_..._fig11a_leaky_compressional_slow.csv` | Sinha & Asvadurov 2004 fig 11(a) | no slow-formation leaky-compressional solver |

## 1. Fast-formation TI flexural

From Ellefsen, K. J., Cheng, C. H., & Schmitt, D. P. (1988), *Acoustic
Logging Guided Waves In Transversely Isotropic Formations*, MIT Earth
Resources Laboratory ([DSpace](https://dspace.mit.edu/handle/1721.1/75100)).

73 points, 1.5-19.5 kHz, traced at 400 dpi. Its three siblings — fig 2's
equivalent-isotropic branch and both of fig 4's — were promoted to `_data/`
and now score at 0.45 %, 0.17 % and 0.30 % RMS.

**The reference is complete and verified.** The formation is Green River
shale, from Thomsen (1986) table 1 (Schock et al. 1974 row): `V_P0` 3292,
`V_S0` 1768 m/s, `rho` 2075 kg/m^3, `eps` 0.195, `delta` -0.220,
`gamma` 0.180; water at 1500 m/s and 1000 kg/m^3; `a` = 0.10 m. That
geometry is confirmed — the three sibling curves score under 0.5 % with it.

What blocks it is that **`flexural_dispersion_vti` raises
`NotImplementedError` for fast-formation TI**:

> `flexural_dispersion_vti` for fast-formation TI (`V_Sv` = 1768 m/s >
> `V_f` = 1500 m/s) is not implemented yet. The real-valued VTI modal
> determinant requires `F_f^2 = k_z^2 - (omega/V_f)^2 > 0`, i.e.
> `V_Sv < V_f`. The complex-determinant path mirroring the isotropic
> `_flexural_dispersion_fast_formation` is deferred.

Green River shale is fast at 1768 m/s against a 1500 m/s borehole fluid, so
this figure lands squarely on the unimplemented path. It is tracked as the
H.d follow-up in `docs/plans/cylindrical_biot_H.md`.

**To promote it:** move the CSV up one directory and add a fourth
`check_overlay` call to section 5 with `flexural_dispersion_vti` on the hard
rock. No new digitising, no new reference hunting.

Worth knowing before that work starts: the *isotropic* fast-formation path
is itself sparse. Fig 2's equivalent-isotropic branch scores 0.45 % RMS but
over only **17 of 73** points, 2.0-6.0 kHz, because `flexural_dispersion`
returns `NaN` outside that band for this rock. A fast-formation TI path that
inherits the same behaviour would be scored just as thinly, and that is a
property of the solver rather than of the figure.

## 2. Cased-hole pseudo-Rayleigh, two branches

From Tubman, K. M., Cheng, C. H., & Toksoz, M. N. (1984), *Geophysics*
**49**(7), 1051-1059, fig 4(b). Same figure and same verified geometry as
the cased Stoneley curve that scores 2.34 % in `_data/` — fluid radius
1.85 in, 0.4 in steel, 1.75 in cement, formation contact at 4.0 in.

**The package has no cased pseudo-Rayleigh entry point.**
`stoneley_dispersion_layered`, `flexural_dispersion_layered` and
`quadrupole_dispersion_layered` cover n = 0, 1 and 2 for cased geometries,
but `trapped_pseudo_rayleigh_dispersion` takes no `layers` argument, so
there is nothing to call. Promoting these needs a new public function, not
a new argument to an existing one — so it is API work, and belongs behind
an issue rather than being bolted on.

## 3. Slow-formation leaky compressional mode

From Sinha & Asvadurov (2004) fig 11(a), curve 3 — 107 points, 2.2-15.0
kHz, extracted from vector paths rather than traced, on the same Table 1
geometry as the six scored Sinha curves (slow formation B: 1890 / 508 /
2054, water 1500 / 1000, `a` = 0.1016 m).

The paper identifies it directly: *"This mode is also referred to as a
leaky compressional mode as it exists between the formation compressional
and borehole-liquid slownesses."* The traced curve runs 529.0 to 639.1
us/m against a C line at 529.1 and an L line at 666.7 — inside that window
throughout, which is the check that it is the right curve.

**No fwap function computes this mode.** `pseudo_rayleigh_dispersion`
tracks the n=0 leaky root of a **fast** formation, between `1/V_P` and
`1/V_S`, and requires `V_S > V_f`; formation B is slow. The two are
different modes in different regimes, and scoring one against the other
would repeat the trapped-versus-leaky category error recorded in
`../README.md`.

It is parked here because the reference is finished and verified, so if a
slow-formation leaky-compressional solver is ever written, the tie is
already in the tree.
