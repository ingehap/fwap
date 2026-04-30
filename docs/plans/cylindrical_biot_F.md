# Plan: cylindrical-Biot item F continuation

Detailed expansion of [item F in
`docs/plans/cylindrical_biot.md`](cylindrical_biot.md) (single
extra annular layer between fluid and formation: mudcake or
altered zone).

## Already landed

PR #43 lands the public-API foundation:

- `BoreholeLayer` frozen dataclass `(vp, vs, rho, thickness)`
  with `_validate_borehole_layers` for parameter checking.
- `stoneley_dispersion_layered(...)` public API. With
  `layers=()` it dispatches bit-equivalently to
  `stoneley_dispersion`; with non-empty layers it raises
  `NotImplementedError` referencing this plan.
- 12 tests including the floating-point regression oracle
  (`layers=()` ↔ `stoneley_dispersion`) that anchors every
  follow-up.

What this lets follow-up work assume:

- Parameter shape, validation rules, and return type
  (`BoreholeMode`) are fixed.
- The follow-up only needs to swap the dispatch branch in
  `stoneley_dispersion_layered`.
- The empty-layer regression test stays as the floating-point
  oracle for the non-trivial case.

---

## F.1 — n=0 (Stoneley) layered modal determinant

The 7×7 matrix is the highest-impact next piece. Plan-doc
"six BCs instead of three" is approximate; the actual count is
seven (3 fluid-annulus BCs + 4 annulus-formation BCs).

### Unknowns and BCs (n=0)

Total unknowns 7 = 1 (fluid) + 4 (annulus) + 2 (formation).

- Fluid (`r < a`): 1 amplitude `A` for the regular-at-axis
  pressure `P = A I_0(F r)`.
- Annulus (`a < r < b`, `b = a + thickness`): 4 amplitudes
  `(B_I, B_K, C_I, C_K)` for the two body-wave potentials,
  each carrying both regular (I_n) and singular (K_n) parts:
    - P potential: `phi = B_I I_0(p_m r) + B_K K_0(p_m r)`
    - SV potential, theta component:
      `psi_theta = C_I I_1(s_m r) + C_K K_1(s_m r)`
- Formation half-space (`r > b`): 2 amplitudes `(B, C)` for
  the decaying-at-infinity outer solutions `B K_0(p r)` and
  `C K_1(s r)`.

Total BCs 7 = 3 + 4:

- At `r = a` (fluid–annulus interface):
    1. `u_r^{(f)} = u_r^{(m)}` (continuity of radial
       displacement)
    2. `sigma_rr^{(m)} = -P^{(f)}` (normal-stress balance)
    3. `sigma_rz^{(m)} = 0` (fluid carries no shear)

- At `r = b` (annulus–formation interface):
    4. `u_r^{(m)} = u_r^{(s)}`
    5. `u_z^{(m)} = u_z^{(s)}`
    6. `sigma_rr^{(m)} = sigma_rr^{(s)}`
    7. `sigma_rz^{(m)} = sigma_rz^{(s)}`

### F.1.a — Math scaffolding (substep blocks)

Mirror the existing n=1 substep style in
`fwap/cylindrical_solver.py` (substeps 1.1 through 1.6.e
inside the n=1 derivation block, lines ~401–2356). Land each
substep as inline comments next to the implementation, not as
a separate document.

- **F.1.a.1** Sign conventions and field ansatz for the
  layered case. The bound regime is now `kz > omega / V_S`
  in *both* the annulus and the formation (otherwise the
  annulus `(p_m, s_m)` go imaginary and the I/K Bessel
  decomposition needs a J/Y sister; out of scope here, flag
  it explicitly in a `ValueError`).
- **F.1.a.2** Per-region displacements from the seven
  amplitudes. The annulus contributes the I_n column on top
  of the K_n column already used in
  `_modal_determinant_n0`; the I_n derivative identities
  mirror the K_n ones with sign flips:
    - `I_0'(x) = I_1(x)` (vs `K_0'(x) = -K_1(x)`)
    - `(1/r) d_r [r I_1(s r)] = +s I_0(s r)` (vs `-s K_0(s r)`)
- **F.1.a.3** Per-region stresses with the same Lamé
  reduction `-lambda k_P^2 + 2 mu p^2 = mu (2 kz^2 - kS^2)`
  that the n=0 / n=1 single-interface forms already use.
  The annulus stresses use the layer's own `lambda_m, mu_m`.
- **F.1.a.4** Boundary-condition row layout (7×7).
  Recommended column order
  `[A | B_I, B_K, C_I, C_K | B, C]` so the three BCs at
  `r = a` only touch columns 1-5 and the four BCs at
  `r = b` only touch columns 2-7. That gives a sparse
  block-bidiagonal structure that will pay off in F.2 / G.
- **F.1.a.5** Phase rescaling. Same trick as the
  single-interface n=0: multiply BC row 3 by `i` and the C
  columns by `-i` so every entry of M is real in the
  bound regime.
- **F.1.a.6** Self-check: in the limit
  `(vp_m, vs_m, rho_m) → (vp, vs, rho)` (annulus material
  identical to formation), the determinant must vanish at
  the same `kz` as `_modal_determinant_n0`. The two
  determinants are *not* numerically equal (different
  matrix size, different overall scalar prefactor) — the
  invariant is *same root set*. Captured in F.1.d below.

### F.1.b — `_modal_determinant_n0_layered` implementation

Signature:

```python
def _modal_determinant_n0_layered(
    kz: float,
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    *,
    layer: BoreholeLayer,
) -> float
```

Returns a real scalar after the F.1.a.5 row/column rescaling.
Single-layer only here; the multi-layer signature
(`layers: tuple[BoreholeLayer, ...]`) is plan item G.

Scope: ~120 lines (the matrix is 7×7 with ~25 non-zero
entries; transcribed mechanically from F.1.a.4).

### F.1.c — Hook into `stoneley_dispersion_layered`

Replace the `NotImplementedError` branch with a brentq loop
that mirrors the existing `stoneley_dispersion`. Bracket
helper: reuse `_stoneley_kz_bracket` driven by
`min(vs, layer.vs)` so the lower bound stays above the
slowest body-wave `kz` floor in the entire stack.

Edge cases to surface explicitly (raise `ValueError` for
now, schedule a fix in plan G):

- `layer.vs < phase velocity`: annulus radial wavenumbers
  go imaginary, the I_n / K_n basis no longer spans the
  solution space. Detect via
  `(omega/layer.vs)^2 > kz^2` and raise.
- `layer.vp < layer.vs` is already caught by the validator
  in PR #43.

### F.1.d — n=0 validation tests (~7 tests)

1. **Layer=formation regression** (the floating-point
   oracle): with `BoreholeLayer(vp=vp, vs=vs, rho=rho,
   thickness=t)` for any `t > 0`, the converged Stoneley
   slowness must match `stoneley_dispersion` to `rtol=1e-8`
   across a 16-point frequency grid. *This is the primary
   correctness check; any algebra error in F.1.b shows up
   here immediately.*
2. **Thickness → 0 regression**: shrinking thickness with
   non-trivial `(vp_m, vs_m, rho_m)` continuously approaches
   the single-interface answer. (Asserts the limit, not
   bit-equivalence.)
3. **Thickness → ∞ regression**: with thickness much larger
   than the wavelength, the converged slowness approaches
   the single-interface Stoneley computed *with the layer
   properties as the formation*. Cross-checks the BC
   bookkeeping at `r = b` is correct.
4. **Schmitt 1988 fig 6 qualitative match**: altered zone
   with reduced V_S produces the characteristic
   low-frequency slowdown (slower phase velocity at low f).
   Tolerance ~5 % per the parent plan's notebook bar.
5. Bound-regime guard: if the supplied layer puts the
   phase velocity into the layer-leaky regime, raises
   `ValueError`.
6. Determinant-at-root: at the converged `kz`,
   `_modal_determinant_n0_layered` evaluates to within
   `1e-9` of zero. (Same shape as
   `test_modal_determinant_at_root_is_near_zero`.)
7. Per-frequency NaN handling: a frequency outside the
   bound regime returns NaN slowness instead of raising.

### F.1 — scope estimate

~150 lines of code + ~150 lines of derivation comments + 7
tests. Two days. Risk concentrated in F.1.a.4 (matrix
layout) and surfaces in F.1.d.1.

---

## F.2 — n=1 (flexural) layered modal determinant

The 10×10 sister of F.1, structurally heavier because the
n=1 single-interface case is already 4×4 (vs 3×3 for n=0).

### Unknowns and BCs (n=1)

Total unknowns 10 = 1 (fluid) + 6 (annulus) + 3 (formation):

- Fluid: `A` for `P = A I_1(F r) cos(theta)`.
- Annulus: 6 amplitudes for the three potentials each in I/K
  flavour (`B_I, B_K, C_I, C_K, D_I, D_K`).
- Formation: 3 amplitudes `(B, C, D)` for K_1(p r) /
  K_1(s r) on the P / SV / SH potentials.

Total BCs 10 = 4 + 6:

- At `r = a` (fluid–annulus, same four BCs as the n=1
  single-interface):
    1. `u_r` continuity (cos sector)
    2. `sigma_rr + P = 0` (cos sector)
    3. `sigma_r_theta = 0` (sin sector)
    4. `sigma_rz = 0` (cos sector)

- At `r = b` (annulus–formation, six continuity BCs):
    5. `u_r` continuity (cos)
    6. `u_theta` continuity (sin)
    7. `u_z` continuity (cos)
    8. `sigma_rr` continuity (cos)
    9. `sigma_r_theta` continuity (sin)
    10. `sigma_rz` continuity (cos)

The cos / sin azimuthal split fixed in n=1 substep 1.1
carries through unchanged: the cos sector contributes
columns / rows 1, 2, 4, 5, 7, 8, 10 (seven entries) and the
sin sector contributes 3, 6, 9 (three entries). The 10×10
matrix is block-decomposable into a 7×7 cos block and a 3×3
sin block — a useful sanity check at implementation time.

### F.2.a — Math scaffolding

Mirror F.1.a, with two structural additions:

- The SH potential `psi_z` doubles in the annulus
  (`D_I I_1(s_m r)` + `D_K K_1(s_m r)`) and contributes
  one row each at `r = a` (BC 3, sin sector) and at `r = b`
  (BC 6, BC 9; both sin sector).
- The 10×10 → 7×7 + 3×3 block decomposition above. The
  block determinant identity
  `det(M_10) = det(M_7^cos) × det(M_3^sin)` (when the cos
  and sin sectors share no rows or columns, which is the
  case here) makes F.2.b ~30 % cheaper to write.

### F.2.b — `_modal_determinant_n1_layered` implementation

Signature mirrors F.1.b. ~200 lines (the cos block alone is
twice the size of F.1.b's 7×7).

### F.2.c — Public `flexural_dispersion_layered`

Same dispatch pattern as F.1.c. With `layers=()` dispatch
to `flexural_dispersion`. Public-API plumbing first
(parallel to PR #43): land the API skeleton with
`NotImplementedError` for non-empty layers ahead of F.2.b
landing, so the regression oracle is in place when the
matrix lands.

### F.2.d — n=1 validation tests (~7 tests)

Mirror F.1.d:

1. Layer=formation bit-equivalence with
   `flexural_dispersion`.
2. Thickness → 0 limit.
3. Thickness → ∞ limit (layer becomes the formation).
4. Schmitt 1988 fig 6 quantitative match: altered zone
   slow-down at low f (the parent plan's headline
   validation target for F).
5. Bound-regime guard.
6. Determinant at root.
7. NaN propagation.

### F.2 — scope estimate

~250 lines + 7 tests. Three days. Risk concentrated in the
sin-sector / cos-sector bookkeeping; the F.2.a block-
decomposition cross-check is the primary safety net.

---

## F.3 — Cross-cutting deliverables

- Mark item F as "done" in `docs/plans/cylindrical_biot.md`
  with a pointer to PR #43 plus the F.1 / F.2 PRs.
- Add `BoreholeLayer` and the two `*_dispersion_layered`
  functions to `docs/api.rst`.
- Add a one-cell entry to the validation notebook
  (`docs/notebooks/cylindrical_biot_validation.ipynb`,
  scheduled in plan item I) reproducing Schmitt 1988 fig 6
  once F.2 lands.
- Update `fwap/cylindrical_solver.py` module docstring
  ("Scope" section, lines 11-30) to reflect the layered
  capability.

Half a day.

---

## Execution order

1. **F.1.a** (math scaffolding inline) — establishes the
   substep skeleton other work checks against.
2. **F.1.b + F.1.d** (n=0 matrix + tests) — first
   end-to-end shippable piece. Smallest viable layered
   product.
3. **F.1.c** (hook the n=0 layered solver into the public
   API) — flips `NotImplementedError` to live behaviour.
4. **F.2.c skeleton** (`flexural_dispersion_layered` API
   raising `NotImplementedError`, mirroring PR #43 for
   n=0). Lands the regression oracle ahead of the matrix.
5. **F.2.a + F.2.b + F.2.d** (n=1 end-to-end).
6. **F.3** (docs + plan-doc cleanup).

Each numbered step is a separately mergeable PR.

If F.2 turns out larger than budgeted, F.1 alone is already
a shippable "layered Stoneley" product; the n=1 layered
case can defer to plan item G (cased-hole multi-layer with
the propagator-matrix machinery), where its bookkeeping is
done generically.

## Total scope

~7 days end-to-end (F.1: 2d, F.2: 3d, F.3: 0.5d, plus
review / iteration buffer). The parent-plan estimate
("Three days. Care needed in the matrix-block bookkeeping;
the physics is identical.") is optimistic for the n=1 half;
n=0 alone fits the original budget.
