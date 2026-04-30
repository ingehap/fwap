# Plan F.2: n=1 (flexural) layered modal determinant

Sister of plan F.1 at azimuthal order 1. Implements the
single-extra-layer flexural dispersion in
[`fwap.cylindrical_solver`](../../fwap/cylindrical_solver.py) on
top of the F.1 scaffolding, with one major structural addition:
the cos / sin sector decomposition at n=1.

## Status of plan F overall

- ✅ F.1 — n=0 (Stoneley) layered, 7×7 — landed in PR #45
- ⏳ F.2 — n=1 (flexural) layered, 10×10 — this document
- ⏳ F.3 — cross-cutting docs / plan-doc cleanup

## Mathematical structure

### Unknowns (10 total)

- 1 fluid amplitude: ``A`` (cos sector)
- 6 annulus amplitudes (three potentials × two flavours):
    - ``B_I, B_K`` — annulus P (cos)
    - ``C_I, C_K`` — annulus SV theta (cos)
    - ``D_I, D_K`` — annulus SH z (sin)
- 3 formation amplitudes:
    - ``B`` — formation P (cos)
    - ``C`` — formation SV theta (cos)
    - ``D`` — formation SH z (sin)

### Boundary conditions (10 total)

At ``r = a`` (fluid-annulus, 4 BCs):

1. ``u_r^{(f)} = u_r^{(m)}`` (cos)
2. ``sigma_rr^{(m)} = -P^{(f)}`` (cos)
3. ``sigma_rtheta^{(m)} = 0`` (sin) — fluid no shear
4. ``sigma_rz^{(m)} = 0`` (cos) — fluid no shear

At ``r = b`` (annulus-formation, 6 BCs):

5. ``u_r^{(m)} = u_r^{(s)}`` (cos)
6. ``u_theta^{(m)} = u_theta^{(s)}`` (sin)
7. ``u_z^{(m)} = u_z^{(s)}`` (cos)
8. ``sigma_rr^{(m)} = sigma_rr^{(s)}`` (cos)
9. ``sigma_rtheta^{(m)} = sigma_rtheta^{(s)}`` (sin)
10. ``sigma_rz^{(m)} = sigma_rz^{(s)}`` (cos)

### Cos / sin sector decomposition (block diagonal)

The cos and sin azimuthal sectors decouple:

- **Cos sector**: 7 amplitudes ``[A | B_I, B_K, C_I, C_K | B, C]``
  paired with 7 BCs (rows 1, 2, 4, 5, 7, 8, 10 — every BC
  involving u_r, u_z, sigma_rr, or sigma_rz).
- **Sin sector**: 3 amplitudes ``[D_I, D_K, D]`` paired with 3 BCs
  (rows 3, 6, 9 — every BC involving u_theta or sigma_rtheta).

The 10×10 matrix is **block-diagonal**:

```
          cos     sin
        [ M_7^cos    0    ]
M_10 =  [                 ]
        [   0    M_3^sin  ]

det(M_10) = det(M_7^cos) × det(M_3^sin)
```

This is the primary structural safety net for F.2: any
sector-mixing error (a cos amplitude appearing in a sin row, or
vice versa) is immediately visible as a non-block-diagonal matrix.

## Already shipped (PR #45)

The F.1 chain landed an analogous 7×7 modal determinant
(``_modal_determinant_n0_layered``) plus row-by-row builders, the
public ``stoneley_dispersion_layered``, the ``BoreholeLayer``
dataclass, and 51 layered-specific tests. F.2 reuses:

- ``BoreholeLayer`` + ``_validate_borehole_layers`` — same
  parameter shape.
- ``_layered_n0_radial_wavenumbers`` — the radial-wavenumber
  helper is identical at n=1; same five wavenumbers
  ``(F_f, p_m, s_m, p, s)``.
- ``BoreholeMode`` return type with ``name="flexural"``,
  ``azimuthal_order=1``.
- The single-interface n=1 solver ``_modal_determinant_n1`` (for
  the layer=formation per-element regression oracle).

## Decomposition into ~13 mergeable units

### F.2.0 — Public-API foundation

Pure plumbing, zero risk. Mirrors the F.1 PR #43 pattern.

```python
def flexural_dispersion_layered(
    freq, *, vp, vs, rho, vf, rho_f, a, layers=(),
) -> BoreholeMode
```

- ``layers=()`` dispatches to existing ``flexural_dispersion``
  (bit-equivalent to the unlayered solver).
- Non-empty layers raise ``NotImplementedError`` referencing this
  plan.
- Add to ``fwap/__init__.py`` exports.
- Tests: empty-layers regression, NotImplementedError sentinel,
  parametric malformed-layer rejection, list/tuple
  interchangeability.

Scope: ~30 lines code + 6 tests. Quarter day.

### F.2.a — Math scaffolding (substep blocks, comments only)

Mirrors F.1.a inline in the module. Each substep block lands as
``#`` comments next to where the implementation will go.

- **F.2.a.1** — sign conventions + field ansatz + regime gate +
  cos/sin sector partition. Field representation:

  ```
  Fluid:        P^{(f)} = A I_1(F_f r) cos(theta)
  Annulus P:    phi^{(m)} = (B_I I_1(p_m r) + B_K K_1(p_m r)) cos(theta)
  Annulus SV:   psi_theta^{(m)} = (C_I I_1(s_m r) + C_K K_1(s_m r)) cos(theta)
  Annulus SH:   psi_z^{(m)} = (D_I I_1(s_m r) + D_K K_1(s_m r)) sin(theta)
  Formation P:  phi^{(s)} = B K_1(p r) cos(theta)
  Formation SV: psi_theta^{(s)} = C K_1(s r) cos(theta)
  Formation SH: psi_z^{(s)} = D K_1(s r) sin(theta)
  ```

  Compare to F.1.a.1: the n=0 forms used ``I_0, K_0`` for the
  P potential and the formula was indexed by ``cos(0*theta) = 1``
  (no azimuthal factor). At n=1, every annulus / formation field
  picks up its azimuthal factor and the Bessel index shifts to 1
  (with derivative-induced shifts to 0 and 2 in the displacement
  / stress formulas).

- **F.2.a.2** — per-region displacements. Reuse the
  single-interface n=1 derivation in the existing module
  (substeps 1.1-1.6.e at lines ~401-2356) for the formation /
  fluid forms; extend with the I-flavour annulus terms.
  Bessel-derivative identities for ``I_1''(x)`` and
  ``K_1''(x)`` are needed (already in the module per substep
  1.3.b).

- **F.2.a.3** — per-region stresses with the Lame reduction
  ``-lambda_m k_Pm^2 + 2 mu_m p_m^2 = mu_m (2 k_z^2 - k_Sm^2)``
  carrying through from F.1.a.3.

- **F.2.a.4** — BC row layout for the 10×10 with the cos / sin
  block decomposition made explicit. Both blocks' sparsity
  patterns drawn out (cos block: A in rows 1, 2 only; B, C in
  rows 5-10 only; sin block: D appears in all three rows).

- **F.2.a.5** — phase rescaling: same recipe as F.1.a.5.
  ``z``-derivative-bearing rows scaled by ``i`` (cos block:
  rows 4, 7, 10; sin block: rows 6 sometimes, 9 — depending on
  which carry the i kz factor); C / D columns scaled by
  ``-i``. Net factor preserves the determinant root.

- **F.2.a.6** — block-diagonal property:
  ``det(M_10) = det(M_7^cos) × det(M_3^sin)``. This is the
  single most important structural identity: any sector cross-
  talk in the implementation is immediately visible.

- **F.2.a.7** — self-check protocol (layer=formation collapse,
  thickness → 0 limit, etc.). Mirrors F.1.a.6.

Scope: ~400 lines of inline math comments. One day.

### F.2.b — Cos sector 7×7 row builders (7 commits)

Each row builder returns a shape-(7,) complex array (post-rescale,
real in bound regime) covering only the cos-sector amplitudes.

- **F.2.b.1.a** — row 1: ``u_r`` continuity at ``r = a`` (cos)
- **F.2.b.1.b** — row 2: ``sigma_rr`` balance at ``r = a`` (cos)
- **F.2.b.1.c** — row 4: ``sigma_rz = 0`` at ``r = a`` (cos)
- **F.2.b.2.a** — row 5: ``u_r`` continuity at ``r = b`` (cos)
- **F.2.b.2.b** — row 7: ``u_z`` continuity at ``r = b`` (cos)
- **F.2.b.2.c** — row 8: ``sigma_rr`` continuity at ``r = b`` (cos)
- **F.2.b.2.d** — row 10: ``sigma_rz`` continuity at ``r = b`` (cos)

**Per-row tests** (4-5 tests each):

- Layer=formation per-element match against
  ``_modal_determinant_n1`` rows for the r=a builders (F.2.b.1.*),
  K-flavour cancellation for the r=b builders (F.2.b.2.*).
- Sparsity (formation columns zero at r=a; fluid column zero at
  r=b; sin-sector columns implicitly absent because the cos
  builder returns shape-(7,)).
- Bound-regime imaginary part is zero.
- Closed-form per-column transcription against F.2.a.2/3.
- Cross-row consistency identities (rows 4 vs row 1, row 5 vs row
  1, row 7 vs row 5 Bessel-index, etc.).

Scope: 7 rows × ~70 lines + ~30 tests = ~500 lines code, ~600
lines tests. 4-5 days.

### F.2.c — Sin sector 3×3 row builders (3 commits)

Each row builder returns shape-(3,) complex array covering only the
sin-sector amplitudes ``[D_I, D_K, D]``.

- **F.2.c.1** — row 3: ``sigma_rtheta = 0`` at ``r = a`` (sin).
  Fluid carries no shear → ``A`` column doesn't appear at all
  (matches the cos/sin block-diagonal structure).
- **F.2.c.2** — row 6: ``u_theta`` continuity at ``r = b`` (sin).
- **F.2.c.3** — row 9: ``sigma_rtheta`` continuity at ``r = b``
  (sin).

This sector is **the simplest part of F.2** because:

- Only 3 amplitudes per row (not 7).
- Only the SH potential ``psi_z`` contributes — the P and SV
  potentials live in the cos sector.
- The 3×3 sin block has no formation-side B / C amplitudes; only
  D_I, D_K (annulus) and D (formation) appear.
- The single-interface ``_modal_determinant_n1`` provides ``M33,
  M34`` for the layer=formation per-element oracle.

**Per-row tests** (4-5 tests each):

- Layer=formation per-element match (F.2.c.1) or K-cancellation
  (F.2.c.2, F.2.c.3).
- Sparsity (varies per row).
- Bound-regime imaginary part is zero.
- Closed-form per-column transcription.

Scope: 3 rows × ~50 lines + ~15 tests = ~150 lines code, ~250
lines tests. 1.5 days.

### F.2.d — Assembly + dispatch (1 commit)

```python
def _modal_determinant_n1_layered(
    kz, omega, vp, vs, rho, vf, rho_f, a, *, layer
) -> float
```

Implementation strategy: build the cos block and the sin block
**separately** (taking advantage of the block-diagonal
decomposition), compute each determinant, multiply.

```python
M_cos = np.vstack([row1_cos, row2_cos, row4_cos, row5_cos,
                    row7_cos, row8_cos, row10_cos])
M_sin = np.vstack([row3_sin, row6_sin, row9_sin])
return float(np.linalg.det(M_cos.real) * np.linalg.det(M_sin.real))
```

The full 10×10 form ``np.vstack([cos_padded, sin_padded])`` with
zero-padded blocks would be equivalent but is wasteful; the
block-by-block product is both cleaner and structurally validates
the block-diagonal property at every evaluation.

Replace the ``NotImplementedError`` in
``flexural_dispersion_layered`` (from F.2.0) with a brentq loop
mirroring ``stoneley_dispersion_layered``.

Bracket helper: ``_flexural_kz_bracket_layered`` parallel to
``_stoneley_kz_bracket_layered`` from F.1.b.4, with the lower
bound at ``omega / min(V_S, V_S_m, V_f) * (1 + 1e-6)``.

Scope: ~150 lines code + ~6 tests (mirrors F.1.b.4). One day.

### F.2.e — Validation hardening (1 commit)

Mirrors F.1.d. Tests:

- **Layer=formation regression** vs ``flexural_dispersion`` to
  ``rtol=1e-8``. Floating-point oracle for the entire chain.
- Thickness → 0 limit recovers unlayered.
- Thickness → ∞ limit matches ``flexural_dispersion(formation =
  layer_props)``.
- Determinant vanishes at converged root.
- Multi-frequency monotonicity smoke.
- **Schmitt 1988 fig 6 quantitative match** — the headline
  validation target for plan item F (altered zone with reduced
  V_S shows the characteristic flexural slow-down at low
  frequency).
- Optional: low-f layer-shift bracketing (sister of the F.1.d
  test).

Scope: ~150 lines tests. Half a day.

## Risk concentration

Ranked by per-unit risk:

1. **F.2.b.2.c (row 8: sigma_rr at r=b)** — highest. The Lame-
   reduction row at the second interface, with both annulus and
   formation parameters in play. Largest single row.
2. **F.2.b.1.b (row 2: sigma_rr at r=a)** — high. Lame-reduction
   row at the first interface. Conceptually identical to F.1.b.2.b
   but with n=1 Bessel functions.
3. **F.2.b.2.b (row 7: u_z at r=b)** — high. New BC type (no
   r=a analog), and the easy-to-miss F.1.a.5 ``row × i`` rescale
   error — same trap as F.1.b.3.b.
4. **F.2.c.1 (row 3: sigma_rtheta at r=a)** — medium. New stress
   type that didn't appear in F.1; the n=1 sigma_rtheta formula
   needs careful derivation. Small (3 entries) but novel.
5. **F.2.c.{2,3} (sin sector at r=b)** — low-medium. Simpler
   than the cos sector but sigma_rtheta / u_theta forms are
   structurally novel.
6. **F.2.b.{1,2}.a / F.2.b.{1,2}.{c,d} (cos u_r and sigma_rz
   rows)** — low-medium. Most structurally similar to F.1.

The **block-diagonal cross-check** (F.2.a.6) catches sector-
mixing errors at every evaluation (the assembled matrix should
have zero entries off the cos/sin blocks). The **layer=formation
per-element match** against ``_modal_determinant_n1`` catches
transcription errors immediately for r=a rows. The
**K-cancellation identity** catches them for r=b rows.

## Suggested execution order

**Recommended: sin-sector-first ("warm-up" approach)**:

1. F.2.0 — public-API foundation (PR-able alone)
2. F.2.a — math scaffolding (PR-able alone, comments only)
3. **F.2.c — sin sector 3×3 (3 commits)** — small, isolated,
   builds confidence with the new ``sigma_rtheta`` derivation
   before committing to the larger cos block.
4. F.2.b — cos sector 7×7 (7 commits)
5. F.2.d — assembly + dispatch
6. F.2.e — validation hardening

**Alternative: cos-sector-first** (matches F.1's flow). Slightly
faster end-to-end because the assembly's integration test is the
floating-point oracle, and reaching that oracle quickly catches
upstream errors. But starts with the larger / more complex
sector.

Either order is viable; the block-diagonal property means the two
sectors are genuinely independent.

## Total scope

| Stage | Commits | Solver lines | Tests |
|-------|---------|--------------|-------|
| F.2.0 | 1 | ~30 | 6 |
| F.2.a | 1 | ~400 (comments) | 0 |
| F.2.b | 7 | ~500 | ~30 |
| F.2.c | 3 | ~150 | ~15 |
| F.2.d | 1 | ~150 | ~6 |
| F.2.e | 1 | ~150 (tests) | ~7 |

**~14 commits**, **~1200 lines solver + ~1100 lines tests**,
**~64 new tests** on top of the 149 currently in
``test_cylindrical_solver.py``. Estimated end-to-end: **8-10
days** of focused work, comparable to F.1's actual cadence.

If F.2.b runs over budget on the cos sector, F.2.c (sin sector)
is independently mergeable as a small intermediate product
(layered ``u_theta`` continuity is occasionally useful in
pedagogical contexts even without the full flexural dispersion).
