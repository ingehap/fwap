# Plan H: VTI formation (transversely isotropic, vertical symmetry axis)

Detailed expansion of [item H in
`docs/plans/cylindrical_biot.md`](cylindrical_biot.md). Implements
the full Schmitt 1989 isotropic-collapse-equivalent modal-
determinant solver for a VTI formation, on top of the existing
isotropic scaffolding in `fwap.cylindrical_solver`.

## Status of plan H overall

- ⏳ H.0 — public-API foundation
- ⏳ H.a — math scaffolding (substep blocks, comments only)
- ⏳ H.b — radial-wavenumber helper (Christoffel-equation roots)
- ⏳ H.c — n=0 (Stoneley) VTI modal determinant + public API
- ⏳ H.d — n=1 (flexural) VTI modal determinant + public API
- ⏳ H.e — validation hardening
- ⏳ H.f — cross-cutting docs

Plan H is independent of plans F (layered) and G (multi-layer
cased-hole); F's per-row builder pattern transfers conceptually
but the matrix entries change at every position because the C-
matrix replaces Lamé (λ, μ).

## Parameterisation

Direct C-matrix entries (5 stiffness coefficients + ρ), matching
the Schmitt 1989 conventions:

```python
def stoneley_dispersion_vti(
    freq: np.ndarray,
    *,
    c11: float, c13: float, c33: float, c44: float, c66: float,
    rho: float,
    vf: float, rho_f: float, a: float,
) -> BoreholeMode
```

Same shape for `flexural_dispersion_vti`. Thomsen parameters are
derived if needed (γ = (C66 − C44) / (2 C44); ε = (C11 − C33) /
(2 C33); δ encoded in C13). The isotropic-collapse condition is

    C11 == C33  (and C44 == C66)  and  C13 == C11 − 2 C44

which the validator surfaces as a clean dispatch: if the C-matrix
is isotropic, dispatch to `stoneley_dispersion` /
`flexural_dispersion` with `vp = sqrt(C33/ρ), vs = sqrt(C44/ρ)`.
This dispatch is the **floating-point oracle** for the entire
H chain.

## H.0 — public-API foundation (~120 lines + 8 tests)

Lands the public API surface and the isotropic-collapse oracle
ahead of the modal-determinant work. Mirrors the F.2.0 / F.1.0
foundation pattern.

* `stoneley_dispersion_vti` and `flexural_dispersion_vti`
  signatures + input validation.
* `_validate_vti_stiffness(c11, c13, c33, c44, c66, rho)` —
  positivity + Thomsen-stability checks.
* `_is_isotropic_stiffness(c11, c13, c33, c44, c66) -> bool` —
  detects the degenerate isotropic case.
* When `_is_isotropic_stiffness` returns `True`, dispatches to
  the isotropic API; otherwise raises `NotImplementedError`
  pointing at the H.c / H.d follow-ups.

**Tests** (8):
- Isotropic-collapse bit-equivalent regression for both Stoneley
  and flexural across a frequency grid (the floating-point
  oracle).
- BoreholeMode return-type contract.
- NotImplementedError for genuine TI (`c11 != c33`, etc.).
- Positivity / stability rejection cases.

## H.a — math scaffolding (~250 lines comments-only)

Inline substep blocks in `cylindrical_solver.py` mirroring F.2.a.
- **H.a.1** — TI stiffness tensor + Thomsen parameters; sign
  conventions (same time / axial / azimuthal as isotropic).
- **H.a.2** — Christoffel-equation derivation in cylindrical
  geometry. Vertical symmetry axis means the borehole-wall
  normal is horizontal, so the Christoffel roots stratify by
  azimuthal order.
- **H.a.3** — qP / qSV decoupling and root selection. The
  Christoffel determinant gives a quadratic in the squared
  radial wavenumber `α²`; both roots `α_qP²`, `α_qSV²` are real
  positive in the bound regime.
- **H.a.4** — SH decoupling at n=0 and n=1: for the symmetric
  field at azimuthal order n, the SH polarization decouples from
  qP/qSV and contributes only via the n=2 quadrupole (out of
  scope for H -- defers to plan E for VTI quadrupole).
- **H.a.5** — Modal-determinant entries with C-matrix instead
  of Lamé. The Lamé reduction `−λ k_P² + 2μ p² = μ(2 k_z² − k_S²)`
  is replaced by analogous C-matrix combinations specific to
  qP / qSV (see Schmitt 1989 eqs. 17-22).
- **H.a.6** — Phase rescaling: same row × i / col × −i pattern
  as the isotropic forms.
- **H.a.7** — Self-check protocol: isotropic-collapse identity at
  every level (radial wavenumbers, modal-matrix entries,
  determinant root).

## H.b — Radial-wavenumber helper (~150 lines + 5 tests)

```python
def _radial_wavenumbers_vti(
    kz: float, omega: float,
    *,
    c11: float, c13: float, c33: float, c44: float, c66: float,
    rho: float,
) -> tuple[float, float]:
    """Returns (alpha_qP, alpha_qSV) -- the two radial-wavenumber
    roots from the Christoffel-equation quadratic. For SH the
    decoupling means alpha_SH = sqrt(kz^2 - rho omega^2 / C66)."""
```

**Tests**:
- Isotropic-collapse: `alpha_qP -> p`, `alpha_qSV -> s` to
  floating-point precision.
- Both roots real positive in the bound regime.
- Christoffel-determinant identity holds numerically (substitute
  back).
- SH decoupled and matches the simple isotropic-in-C66 form.
- NaN for kz below the bound floor.

## H.c — n=0 (Stoneley) VTI modal determinant + public API (~250 lines + 8 tests)

`_modal_determinant_n0_vti(...)` mirrors `_modal_determinant_n0`
with C-matrix entries (one qP and one qSV term instead of one P
and one S). Stoneley public API replaces `NotImplementedError`
with brentq loop.

**Tests**:
- **Isotropic-collapse regression** vs `stoneley_dispersion`
  across a 16-point frequency grid (`rtol=1e-8`).
- **Norris 1990 LF closed-form match**: at f → 0, slowness
  approaches `sqrt(1/V_f^2 + ρ_f / C66)`. *Note*: this is the
  closed-form oracle that's truly TI-specific (depends on C66,
  not C44). Strong validation of the qSH decoupling at n=0.
- Real-valued post-rescale, determinant-at-root vanishes,
  multi-frequency monotonicity smoke.

## H.d — n=1 (flexural) VTI modal determinant + public API (~300 lines + 8 tests)

Mirror of H.c at n=1.

**Tests**:
- Isotropic-collapse regression vs `flexural_dispersion`.
- Sanity vs `flexural_dispersion_vti_physical`: at low f the
  full VTI flexural slowness should approach `1/V_Sv` (matching
  the phenomenological model's LF asymptote).
- High-f asymptote toward Rayleigh in the C66-dominated
  pseudo-isotropic limit.
- Same imaginary-power / determinant-at-root checks as H.c.

## H.e — Hardening tests (~80 lines)

Mirror of F.2.e: determinant-at-root self-consistency for non-
trivial TI parameters; multi-frequency monotonicity; weak-
anisotropy regression vs the phenomenological model from
`fwap.cylindrical`.

## H.f — Cross-cutting docs (~30 lines)

- Mark item H done in `docs/plans/cylindrical_biot.md`.
- Update module docstring scope.
- PR pointers in this plan doc.

## Total scope

~1000 lines of solver code + ~800 lines of tests + ~50 tests,
distributed across ~7 mergeable PRs. Conservative estimate:
4-6 days of focused work. Risk concentrated in:
- The Christoffel-equation root selection (H.b) -- qP and qSV
  must be picked unambiguously (not all roots are physical for
  bound modes).
- The C-matrix matrix-entry transcriptions (H.c, H.d) -- the
  Lamé reduction at multiple positions in each row gets
  replaced by C-matrix-specific Lamé-like combinations from
  Schmitt 1989 eqs. 17-22.
- The Norris 1990 LF closed-form match (H.c) -- the strongest
  TI-specific oracle but requires getting the C66 coupling
  right at azimuthal order n=0.

## References

- Schmitt, D. P. (1989). Acoustic multipole logging in
  transversely isotropic poroelastic formations. *J. Acoust.
  Soc. Am.* 86(6), 2397-2421.
- Norris, A. N. (1990). The speed of a tube wave. *J. Acoust.
  Soc. Am.* 87(1), 414-417.
- Ellefsen, Cheng & Toksöz (1991). Effects of anisotropy on the
  shear-wave logging in a TI formation. *J. Acoust. Soc. Am.*
  89(5), 2197-2210.
- Sinha, Norris & Chang (1994). Borehole flexural modes in
  anisotropic formations. *Geophysics* 59(7), 1037-1052.

## Execution order

1. **H.0** (foundation + isotropic-collapse oracle) — establishes
   API and floating-point regression test.
2. **H.a** (math scaffolding) — comments only; pins the
   Christoffel-equation derivation and C-matrix conventions.
3. **H.b** (radial-wavenumber helper) — pure plumbing with
   strong unit oracle (Christoffel-determinant identity).
4. **H.c** (n=0 + Stoneley public API + Norris 1990 LF check) —
   smallest viable VTI product; the Norris match is a TI-
   specific oracle.
5. **H.d** (n=1 + flexural public API) — full dipole flexural
   in TI media.
6. **H.e** (hardening) — multi-frequency / det-at-root.
7. **H.f** (docs) — close out plan H.
