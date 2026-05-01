# Changelog

All notable changes to this project are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/), and
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **``quadrupole_dispersion`` now auto-dispatches to a fast-formation
  path when ``V_S > V_f``** (Roadmap A, plan item E in
  ``docs/plans/cylindrical_biot.md``). Direct n=2 sister of the
  plan item B work on ``flexural_dispersion``: previously
  fast-formation inputs returned NaN throughout (with a
  documented "plan item E follow-up" caveat), now they dispatch
  to a new private ``_quadrupole_dispersion_fast_formation`` that
  brentq's the imaginary part of
  :func:`_modal_determinant_n2_complex` along the real-``k_z``
  axis in the ``(omega/V_S, omega/V_R)`` bracket, with
  continuation across frequencies. Slow-formation behaviour is
  unchanged bit-for-bit.

  As with the n=1 case, the converged ``k_z`` is real to
  floating-point precision: the formation P/S branches stay
  bound, so the mode is bound; the only effect of ``F^2 < 0`` is
  an overall ``i^k`` phase that makes the determinant
  predominantly imaginary at real ``k_z``, reducing the root
  condition to ``Im(det) = 0``. The n=2 determinant magnitudes
  are about 15 orders larger than the n=1 sister, so the
  absolute residual at the converged root sits at ~10^8 rather
  than ~10^4; the relative residual ``|Im(det)|/|det|`` is at
  machine precision in both cases.

### Added
- **``_modal_determinant_n2_complex``** (Roadmap A, plan item E
  scaffolding). Complex-``k_z`` n=2 quadrupole modal determinant
  with optional ``leaky_p`` / ``leaky_s`` flags, structurally
  identical to :func:`_modal_determinant_n2` with K-Bessel
  evaluations swapped for the Hankel analytic continuation in
  the leaky regime via :func:`_k_or_hankel`. Fluid I-Bessel
  handles complex ``F`` transparently via ``scipy.special.iv``.
  In the fully-bound regime (real ``kz``, both flags False) the
  result agrees with the real-only sister to floating-point
  precision -- the regression invariant tested in
  ``tests/test_cylindrical_solver.py::test_complex_n2_matches_real_in_bound_regime``.

  Four new tests added for the leaky-quadrupole deliverable:
  bound-regime regression, slow-formation bit-identical guard
  (``quadrupole_dispersion`` reproduces an open-coded brentq +
  bracket-helper reference value), fast-formation regime sanity
  (velocities in ``(V_R, V_S)``, attenuation_per_meter is None),
  ``Im(det)/|det|`` machine-precision check at converged
  fast-formation roots, and frequency-order invariance.

- **``quadrupole_dispersion`` public API** (Roadmap A, plan item D
  in ``docs/plans/cylindrical_biot.md``). Real-valued n=2 modal-
  determinant solver for the slow-formation (``V_S < V_f``) bound
  regime, the LWD-quadrupole mode framed in Tang & Cheng 2004
  sect. 2.5. Tracks the lowest-``k_z`` zero of a 4x4
  :func:`_modal_determinant_n2` across the input frequency grid
  with the same ``brentq`` + bracket-expansion pattern as
  ``stoneley_dispersion`` and ``flexural_dispersion``. Returns
  ``BoreholeMode(name="quadrupole", azimuthal_order=2)``.

  Implementation: extends the existing n=1 derivation by the rules
  ``(I_0, I_1, K_0, K_1) -> (I_{n-1}, I_n, K_{n-1}, K_n)`` and
  ``azimuthal-derivative factor 1 -> n``, with two structural
  factors that are zero at n=1 but finite at n>=2: a
  ``2 n(n+1)`` overall-rank coefficient that turns the
  ``+ 4 K_1(pa)/a^2`` 1/r^2 correction in M22 into
  ``+ 12 K_2(pa)/a^2`` at n=2, and an ``(n^2-1)/a^2`` correction
  to the sigma_rz C-coefficient that vanishes at n=1 but adds
  a ``+ 3/a^2`` term to M43 at n=2. Specialised to n=2 the
  matrix uses the ``(K_1, K_2)`` and ``(I_1, I_2)`` Bessel
  index pairs.

  **Slow-formation only in this release**: the fast-formation
  (``V_S > V_f``) leaky-quadrupole regime needs the same complex-
  modal-determinant scaffolding that plan item B used for
  fast-flexural and is plan item E. Fast formations return
  all-NaN.

  ``fwap.lwd.lwd_quadrupole_priors`` now points at the new
  ``quadrupole_dispersion`` for callers that have full formation
  properties; the rectangular-window prior factory is retained
  as a Viterbi seed for the rough-V_S case where the full set
  of formation parameters is not available.

  8 new tests cover the dataclass contract, slow-formation
  finite-output + velocity-window sanity, fast-formation
  all-NaN guard, below-cutoff NaN, the ``slowness > 1/V_S``
  invariant, the local-zero property of the modal determinant
  at converged roots, and input validation
  (non-positive scalars, ``vp <= vs``, non-positive freq).

### Changed
- **``flexural_dispersion`` now auto-dispatches to a fast-formation
  path when ``V_S > V_f``** (Roadmap A, plan item B in
  ``docs/plans/cylindrical_biot.md``). Previously the public
  ``flexural_dispersion`` returned NaN throughout for any fast
  formation -- a documented limitation. The function now detects
  ``V_S > V_f`` at call time and dispatches to a new private
  ``_flexural_dispersion_fast_formation`` that brentq's the
  imaginary part of :func:`_modal_determinant_n1_complex` along
  the real-``k_z`` axis. Slow-formation behaviour is unchanged
  bit-for-bit (the dispatch is purely additive).

  **Empirical finding that informed the implementation**: in the
  canonical ``(V_R, V_S)`` velocity window the converged ``k_z``
  is real to floating-point precision rather than complex. The
  earlier "fast-formation flexural is leaky and needs complex-
  ``k_z`` Mueller iteration" framing in the Roadmap-A comments
  was over-stated for this particular root: the formation P/S
  branches stay bound in this regime, so the mode is also bound.
  The complex modal determinant is needed only because ``F^2 < 0``
  introduces an overall ``i^k`` phase that makes the determinant
  predominantly imaginary at real ``k_z``; the root condition
  reduces to ``Im(det) = 0`` and brentq along the real axis is
  the natural tool. Truly leaky n=1 modes with non-trivial
  ``Im(k_z) > 0`` (higher-order leaky flexural, fast-formation
  pseudo-flexural) need the complex marcher and remain out of
  scope for this routine.

  Backward compatibility note: callers that explicitly relied on
  ``flexural_dispersion`` returning all-NaN for fast formations
  must now check ``np.isfinite`` per element. The previous
  "all-NaN sentinel" was documented as a stop-gap pending plan
  item B, so the change is in the spirit of the original API
  rather than against it.

### Added
- **``_modal_determinant_n1_complex``** (Roadmap A, plan item B
  scaffolding). Complex-``k_z`` n=1 dipole modal determinant with
  optional ``leaky_p`` / ``leaky_s`` flags, structurally
  identical to the real-valued :func:`_modal_determinant_n1`
  with K-Bessel evaluations swapped for the Hankel analytic
  continuation in the leaky regime. Fluid I-Bessel handles
  complex ``F`` transparently via ``scipy.special.iv``. In the
  fully-bound regime (real ``kz``, both flags False) the result
  agrees with the real-only sister to floating-point precision
  -- the regression invariant tested in
  ``tests/test_cylindrical_solver.py::test_complex_n1_matches_real_in_bound_regime``.

  Five new tests added for the leaky-flexural deliverable:
  bound-regime regression, slow-formation bit-identical guard
  (``flexural_dispersion`` reproduces an open-coded brentq +
  bracket-helper reference value), fast-formation finite-output
  + velocity-window check (``V_R < v < V_S``), local-zero
  property of ``Im(det)`` at converged fast-formation roots,
  and frequency-order invariance of the fast-formation marcher
  (ascending and descending input grids produce identical output).

- **Cutoff handling + branch tracker** (Roadmap A, plan item C
  in ``docs/plans/cylindrical_biot.md``). Adds a validator-aware
  marcher that distinguishes a converged-but-out-of-regime root
  from a root-finder failure, tolerates a small budget of
  consecutive bad steps before giving up, and recovers from
  one-off branch hops by resuming the march from the last good
  step. Three new symbols in ``fwap.cylindrical_solver``:

  * ``_classify_marcher_step(kz_root, omega, validator) -> str`` --
    private classifier returning ``"ok"``, ``"regime_exit"``, or
    ``"convergence_failure"``. Validator exceptions
    (``ValueError`` / ``ArithmeticError``) collapse to
    ``"regime_exit"`` so a numerically ill-conditioned step does
    not abort the march.

  * ``BranchSegment`` -- public dataclass representing a
    contiguous stretch of finite samples in a dispersion curve
    (``start_idx``, ``end_idx``, ``freq``, ``kz``). Re-exported
    at top level. ``len(segment)`` returns the inclusive sample
    count.

  * ``segments_from_kz_curve(freq_grid, kz_curve)
    -> list[BranchSegment]`` -- public splitter that walks a
    marcher output and emits one ``BranchSegment`` per maximal
    run of finite ``kz``. Re-exported at top level.

  * ``_march_complex_dispersion_validated(det_fn, freq_grid,
    kz_start, *, validator, max_consecutive_invalid, xtol)`` --
    private validator-aware marcher. ``validator(kz, omega) ->
    bool`` says whether a converged step belongs to the regime
    the caller wants to track; failed steps stay NaN, do not
    update the continuation seed, and count against
    ``max_consecutive_invalid``. Setting that to ``0`` recovers
    the strict-stop semantics of the original
    ``_march_complex_dispersion``.

  ``pseudo_rayleigh_dispersion`` is refactored to drive the new
  marcher with a leaky-S-regime validator (``Im(k_z) > 0`` and
  ``1/V_P < slowness < 1/V_S``). On the standard fast-formation
  parameter set the refactor recovers steps that the previous
  step-by-step loop dropped to single-step root hops, returning
  one contiguous segment over the supported band. 12 new tests
  cover the classifier verdicts (each return value, plus
  exception-as-regime-exit), the dataclass contract,
  ``segments_from_kz_curve`` (NaN-gap split, all-NaN, mismatched
  inputs), the validated marcher (skip-and-continue, budget
  exhaustion, empty grid, zero-budget = strict semantics), and
  the pseudo-Rayleigh single-segment regression.

- **``pseudo_rayleigh_dispersion`` public API** (Roadmap A,
  plan item A in ``docs/plans/cylindrical_biot.md``). First
  leaky-mode product on top of the L1-L3 scaffolding. Tracks the
  n=0 leaky root with the formation S wave radiating outward
  (``s``-branch leaky) while the fluid pressure and the formation
  P wave stay bound. Mode exists in fast formations only
  (``V_S > V_f``) above a low-frequency cutoff where it merges
  with the body S head wave.

  Implementation: walks the input frequency grid from high to low
  internally. Seeds at the highest frequency with slowness
  ``0.95 / V_S`` (5% inside the leaky-S regime) plus a small
  positive imaginary part; subsequent steps use the previous
  step's converged ``k_z`` rescaled to the next ``omega`` as the
  seed (constant-slowness extrapolation). The marcher stops as
  soon as

  1. ``scipy.optimize.root`` fails to converge, or
  2. the converged ``Im(k_z)`` is non-positive (mode merged with
     the bound regime, or root finder drifted to a non-physical
     growing branch), or
  3. the converged slowness falls outside ``(1/V_P, 1/V_S)``
     (root hopped to a different physical regime).

  Remaining low-frequency samples stay NaN; branch-stitching
  across the cutoff is plan item C. Returns
  :class:`BoreholeMode` with ``slowness = Re(k_z)/omega`` and the
  newly-populated ``attenuation_per_meter = Im(k_z)`` field.

  Re-exported at top level as ``pseudo_rayleigh_modal_dispersion``
  to disambiguate from the existing
  :func:`fwap.synthetic.pseudo_rayleigh_dispersion`
  phenomenological callable-factory model
  (kept unchanged for backward compatibility). Both names remain
  accessible by their fully-qualified module paths.

  10 tests cover input validation (slow-formation rejection,
  non-positive inputs, invalid frequencies), output contract
  (``BoreholeMode`` shape and ``attenuation_per_meter``
  population), regime sanity (slowness strictly inside
  ``(1/V_P, 1/V_S)``; ``Im(k_z) > 0`` everywhere finite;
  velocity strictly between ``V_S`` and ``V_P``), the
  frequency-order invariance of the marcher (ascending and
  descending input grids produce identical per-frequency
  output), the empty-frequency-array no-op, and the local-zero
  property of the determinant at converged roots
  (``|det(root)| < 1% * |det(off-root)|``).

- **``BoreholeMode.attenuation_per_meter`` field** (Roadmap A
  continuation, dataclass extension for upcoming leaky-mode
  solvers). Adds an optional
  ``attenuation_per_meter: ndarray | None`` field to the
  :class:`BoreholeMode` dataclass, default ``None`` for backward
  compatibility with the existing bound-mode solvers (Stoneley
  and slow-formation flexural). Future leaky-mode solvers will
  populate the field with ``Im(k_z)`` to expose the spatial
  attenuation rate in 1/m. 3 tests cover the field contract:
  default-None, accepts an ndarray, and the existing Stoneley
  solver continues to return ``None`` (bound mode -> no
  attenuation).

- **Complex-``k_z`` root finder + frequency-marching tracker for
  the leaky-mode solver** (Roadmap A continuation, phase L3). Two
  new private helpers in ``fwap.cylindrical_solver``:

  * ``_track_complex_root(det_fn, kz_start, *, xtol=1e-12)`` --
    single-frequency complex root finder. Wraps
    ``scipy.optimize.root(method='hybr')`` on the (Re, Im) split
    of the complex residual. Catches det-evaluation exceptions
    and converts them to large penalty residuals so the iterator
    backs off rather than aborting. Returns the converged
    complex ``k_z`` or ``None`` on failure.

  * ``_march_complex_dispersion(det_fn, freq_grid, kz_start, *,
    xtol)`` -- frequency-marching loop with **scale-invariant
    continuation**: the next step's initial guess is
    ``k_z_prev * (f / f_prev)``, which keeps the seed on the
    constant-slowness extrapolation of the previous step. This
    handles the multiplicative ``k_z`` jumps typical of bound-
    mode dispersion (Stoneley ``k_z`` doubles when frequency
    doubles) without losing the local-quadratic convergence of
    the per-step solver. Returns a NaN-padded complex array;
    once a step fails, the remaining steps stay NaN.

  Branch tracking across leaky-vs-bound transitions is the
  caller's responsibility (the marcher just walks the grid as
  given). Standard pattern: ``det_fn`` internally calls
  ``_detect_leaky_branches`` from L2 to re-classify the regime
  at each evaluation, OR the caller splits the frequency grid
  at the cutoff and calls the marcher separately on each side.

  This phase is purely the root-finding mechanics. The leaky-
  mode public APIs (``pseudo_rayleigh_dispersion``, fast-
  formation flexural, quadrupole) build on top in phases L4-L6.

  7 new tests cover: linear synthetic root (exact recovery);
  closest-root selection on a quadratic; exception-safety of the
  tracker; synthetic linear dispersion (constant complex
  slowness); large-multiplicative-frequency-jump continuation
  (Stoneley-like ``k_z`` scaling); smoothly drifting complex
  dispersion (both Re and Im of slowness drifting with
  frequency); end-to-end regression -- the marcher composed with
  ``_modal_determinant_n0_complex`` recovers the existing
  ``stoneley_dispersion`` result to ~1e-10 relative precision.

- **Leaky-mode scaffolding for the cylindrical-Biot solver**
  (Roadmap A continuation, phases L1 + L2). Mathematical
  scaffolding -- complex-``k_z`` sign conventions, Hankel-
  function ansatz for outgoing-wave BCs, branch-cut handling --
  plus a complex-aware n=0 modal determinant
  ``_modal_determinant_n0_complex(kz, omega, vp, vs, rho, vf,
  rho_f, a, *, leaky_p=False, leaky_s=False)`` that supports
  complex ``k_z`` and switchable K-Bessel / Hankel evaluators
  per radial branch. Plus two helpers:
  ``_detect_leaky_branches(kz, omega, vp, vs, vf)`` classifies
  ``(F, p, s)`` as bound or leaky from the sign of
  ``Re(alpha^2)``;
  ``_k_or_hankel(n, alpha, r, *, leaky)`` returns
  ``(K_n, K_{n+1})`` either as standard modified Bessels (bound)
  or as the Hankel-via-analytic-continuation
  ``(pi/2) i^{n+1} H_n^{(2)}(i alpha r)`` (leaky). The whole
  family is private (underscore-prefixed) because the public
  leaky-mode APIs (pseudo-Rayleigh, fast-formation flexural,
  quadrupole) require the L3 complex root finder which is the
  next planned PR. Regression invariant: in the bound regime
  (real ``kz``, both leaky flags ``False``) the complex
  evaluator agrees with the existing real ``_modal_determinant_n0``
  to floating-point precision (rel < 1e-12; imaginary part
  identically zero) -- this is the test guard that lets future
  L3+ work refactor confidently. 9 new tests cover: real-vs-
  complex agreement at multiple ``kz``; sign-change preservation
  across the Stoneley root; branch-detector classification in
  three regimes (bound, pseudo-Rayleigh, fast-flexural); Bessel-
  vs-Hankel helper agreement on the bound branch; finiteness of
  Hankel-branch evaluations and complex-``kz`` evaluations. The
  ``[Unreleased]`` section in ``docs/roadmap.md`` (item A) gets
  the L1-L7 sequencing detail; the bound-mode half of A remains
  shipped, the leaky-mode half is now mid-flight.

- **Tensile-strength rock-physics correlation**
  (``fwap.geomechanics.tensile_strength_from_ucs``). One-line
  convenience function returning ``T = ratio * UCS`` with default
  ``ratio = 0.10`` (typical sandstones). Documented as a Hoek-Brown-
  style "tension cutoff" rather than the Mohr-Coulomb linear
  extrapolation -- the latter overestimates real-rock tensile
  strength by ~3x and is a commonly-flagged geomechanical pitfall.
  Provides published lithology-specific ratio ranges (sandstones
  0.07-0.12, shales 0.04-0.08, limestones 0.08-0.15, crystalline
  rocks 0.10-0.20) so users can re-tune. Closes the last item on
  the original session-1 list of possible extensions; round-trip
  use is documented (compute UCS via
  ``unconfined_compressive_strength``, T via this function, feed
  T into ``tensile_breakdown_pressure``). 7 new tests cover the
  closed-form linearity, broadcasting, zero-UCS edge case,
  round-trip into the breakdown pressure, and input validation.

- **Inclined tensile-breakdown pressure + inclined safe mud-weight
  window** -- completes the wellbore-stability symmetry between
  vertical and inclined wells. Two new public functions in
  ``fwap.geomechanics``:

  * ``inclined_breakdown_pressure(...)``: Mohr-style tensile-
    failure scan around the wall of an inclined well. Diagonalises
    the (theta, z) 2x2 sub-block at each azimuth, finds the
    smallest eigenvalue lambda_-(theta, P_w), and bisects on the
    worst-azimuth tensile-failure margin
    ``min_theta lambda_- - alpha P_p + T``. Convention follows the
    vertical ``tensile_breakdown_pressure``: ignores the radial
    principal stress sigma_rr (which would always be most tensile
    under positive pore pressure and would not match the standard
    Hubbert-Willis fracture-initiation interpretation).

  * ``inclined_safe_mud_weight_window(...)``: convenience wrapper
    that combines ``inclined_breakout_pressure`` and
    ``inclined_breakdown_pressure`` and returns the same
    :class:`MudWeightWindow` dataclass used by the vertical
    counterpart, with ``width`` and ``is_drillable`` properties.

  Vertical-well consistency: at ``well_inclination_deg = 0`` both
  functions match the vertical closed forms to within the
  azimuth-grid resolution (verified by test). For a typical
  drillable scenario, the safe window narrows from 31.25 MPa
  (vertical) to 13.75 MPa (horizontal) -- breakout rises, breakdown
  falls, net width drops; the well remains drillable but with
  much less mud-weight margin. 10 new tests cover: vertical
  consistency for both bounds; monotonicity in inclination,
  tensile strength, pore pressure; the not-drillable-in-tension-
  at-zero-mud edge case; ``MudWeightWindow`` dataclass contract;
  vertical-window equivalence; window narrowing with inclination;
  horizontal-well drillability; input validation.

- **Inclined-wellbore stability** -- generalized Kirsch wall
  stresses (Hiramatsu-Oka 1962, Fairhurst 1968) and Mohr-Coulomb
  shear-breakout pressure for arbitrarily oriented wells in
  ``fwap.geomechanics``. Two new functions:
  ``inclined_wellbore_wall_stresses(sigma_v, sigma_H, sigma_h, *,
  well_inclination_deg, well_azimuth_deg,
  azimuth_around_wall_deg, mud_pressure, poisson)`` returns the
  four wall stress components ``(sigma_theta, sigma_z,
  sigma_theta_z, sigma_r)`` after rotating the principal-stress
  tensor into well-aligned coordinates;
  ``inclined_breakout_pressure(...)`` finds the critical mud
  pressure by scanning over wall azimuth, computing principal
  stresses (via 2x2 eigenvalue decomposition of the (theta, z)
  sub-block plus the trivial radial principal stress), applying
  Mohr-Coulomb at each azimuth, and bisecting on the worst-
  azimuth failure margin.

  Vertical-well consistency: at ``well_inclination_deg = 0`` the
  wall-stress formulas reduce exactly to the existing
  ``kirsch_wall_stresses``, and the breakout pressure agrees
  with the closed-form ``mohr_coulomb_breakout_pressure`` to
  within the azimuth grid resolution. Inclined wells in normal-
  fault stress regimes need progressively more mud-pressure
  support (verified by test on a drillable scenario: 33.75 MPa
  vertical -> ~40 MPa horizontal).

  Documented assumptions: principal-stress-aligned far-field
  stresses (sigma_v vertical, sigma_H/sigma_h horizontal); no
  shear stresses in the un-rotated frame (the rotation introduces
  them in the well frame). The wall is assumed to fail in shear
  per Mohr-Coulomb; tensile-breakdown for inclined wells is a
  follow-up. The function raises informatively when the wall is
  unconditionally unstable (no mud pressure can stabilise the
  geometry) or when ``friction_angle_deg`` is out of range.
  10 new tests cover: vertical-well wall-stress and breakout-
  pressure consistency with the closed forms; inclination
  monotonicity; horizontal-well azimuth dependence; periodicity
  and symmetry of the wall stresses; sigma_r = mud pressure
  identity; isotropic-horizontal-stress vertical-well limit;
  input validation; not-drillable-geometry error message.

- **Bowers (1995) sonic pore-pressure with unloading branch**
  (``fwap.geomechanics.pore_pressure_bowers``). Closes the
  Bowers-method follow-up flagged in PR #27's CHANGELOG.
  Velocity-effective-stress closed form
  ``V = V_ml + A * sigma_eff^B`` with two branches:

  * **Loading (virgin curve)**: pore pressure from
    ``sigma_eff = ((V - V_ml) / A)^(1/B)``. Selected when
    ``sigma_max_pa`` is None.
  * **Unloading**: pore pressure from
    ``sigma_eff = sigma_max * ((V - V_ml) / (A * sigma_max^B))^(U/B)``,
    selected when ``sigma_max_pa`` is supplied. The unloading
    exponent ``U > B`` makes the curve steeper than loading,
    which is the physical signature of unloading-driven
    overpressure (gas generation, clay diagenesis,
    hydrocarbon expulsion) that Eaton's method
    (``pore_pressure_eaton``) under-estimates.

  Both branches close in closed form with no numerical inversion.
  Default calibration ``(V_ml, A, B, U) = (1524, 14.02, 0.673,
  3.13)`` is Bowers' (1995) Gulf of Mexico shale fit; users
  should re-calibrate against well data for other basins.
  Unit convention: SI throughout (Pa for stresses, m/s for
  velocity); ``A`` is in (m/s) / MPa^B with the Pa↔MPa conversion
  internal. Loading/unloading branch selection is the user's
  responsibility -- the function does not auto-detect the regime
  because that requires burial-history information not on the log.
  11 new tests cover: round-trip recovery on both branches; mudline-
  velocity edge case (V = V_ml gives sigma_eff = 0); monotonicity
  in V; unloading > loading prediction at the same V (the
  Eaton-fix signature); end-to-end pipeline with closure_stress;
  and input validation (V < mudline, non-positive calibration
  constants and sigma_max).

- **VTI group velocities** (``fwap.anisotropy.vti_group_velocities``,
  ``VtiGroupVelocities``). Closes the wavefront-modelling deliverable
  flagged as a follow-up in PR #30: group velocity (the speed of
  energy / wavefront propagation) and group angle (the direction of
  energy propagation, generally different from the phase-angle
  direction in anisotropic media) for the three VTI modes
  (qP, qSV, SH). Tsvankin (2001) sect. 1.3 closed forms:

      v_g_x = v_p sin(theta) + (dv_p/dtheta) cos(theta)
      v_g_z = v_p cos(theta) - (dv_p/dtheta) sin(theta)
      |v_g| = sqrt(v_p^2 + (dv_p/dtheta)^2)
      tan(psi) = v_g_x / v_g_z

  The dv_p/dtheta derivative is computed numerically via
  np.gradient (central differences in the interior, one-sided at
  the grid endpoints); avoids the algebraic complexity of the
  closed-form Tsvankin derivatives at minimal accuracy cost. Output
  is a ``VtiGroupVelocities`` dataclass with three velocities and
  three group angles. Wavefront-plotting use:
  ``x = v_g * sin(psi); z = v_g * cos(psi)``. 12 new tests cover:
  isotropic limit (group exactly equals phase, psi exactly equals
  theta to floating-point precision); dataclass contract; group =
  phase at theta = 0 and pi/2; psi = 0 at theta = 0 and psi = pi/2
  at theta = pi/2 (symmetry-axis-aligned wavefronts); group angle
  differs from phase angle off-axis (qSV refracts toward symmetry
  for the Berea-VTI fixture, SH refracts away); all velocities
  positive and qP > qSV everywhere; input validation rejecting
  one-point and non-increasing grids; end-to-end Cartesian-
  wavefront monotonicity check on a Backus-derived medium.

- **VTI phase velocities** (``fwap.anisotropy.vti_phase_velocities``).
  Christoffel-determinant solution for the three plane-wave modes
  (quasi-P, quasi-SV, SH) in a transversely-isotropic medium with
  vertical symmetry axis, propagating at phase angle ``theta`` from
  the symmetry axis. Tsvankin (2001) eq. 1.41 in standard form;
  closed-form quadratic with the standard ``+/- sqrt(D)``
  discriminant for qP/qSV plus the decoupled SH formula
  ``v_SH^2 = (C44 cos^2 + C66 sin^2)/rho``. Natural consumer of
  ``backus_average`` output (the new function takes the five VTI
  elastic constants directly). Useful for forward-modelling
  wavefronts, qP/qSV crossover analysis, and Thomsen-anisotropy
  consistency checks. 12 new tests cover: vertical and horizontal
  limits (each velocity recovers the corresponding C-modulus
  square root); isotropic limit (constant in theta, qSV=SH);
  qSV / SH degeneracy at vertical; Thomsen-anisotropy signatures
  (epsilon > 0 -> v_qP at horizontal > vertical; gamma > 0 ->
  v_SH at horizontal > vertical); shape and broadcasting; round-
  trip with ``backus_average``; input validation. Pure phase
  velocity; group velocity is a planned follow-up.

- **Tensile-breakdown pressure + safe mud-weight window**
  (``fwap.geomechanics.tensile_breakdown_pressure``,
  ``MudWeightWindow``, ``safe_mud_weight_window``). Closes the
  drilling-decision pipeline by adding the upper bound of the
  safe mud-weight range -- the Hubbert-Willis (1957) fracture-
  initiation pressure
  ``P_w_break = 3 sigma_h - sigma_H + T - alpha P_p`` -- to
  complement the Mohr-Coulomb breakout (lower bound) shipped in
  the same release. The ``MudWeightWindow`` dataclass packages
  both bounds plus convenience properties ``width``
  (= breakdown - breakout) and ``is_drillable`` (= width > 0)
  for diagnostic output. Strong negative-width result on the
  PR #28 smoke-test scenario flagged it as "not drillable in
  this geometry without intervention" -- exactly the kind of
  immediate diagnostic this combiner is meant to produce. 11
  new tests cover: Hubbert-Willis closed-form match;
  at-critical-pressure inverse check (Kirsch hoop stress at
  theta=0 equals -T after effective-stress correction);
  monotonicity in tensile strength, horizontal-stress
  anisotropy, pore pressure; biot_alpha=0 limit; window
  dataclass contract; pure-pass-through equivalence with the
  individual primitives; per-depth drillability flag on vector
  input.

- **Wellbore-stability analysis** — Kirsch (1898) wall-stress
  primitive plus Mohr-Coulomb shear-breakout pressure
  (``fwap.geomechanics.kirsch_wall_stresses`` and
  ``mohr_coulomb_breakout_pressure``). Extends the geomechanics
  module from indices to a drilling-decision deliverable: the
  critical mud pressure below which the borehole wall fails in
  shear at the breakout azimuth (perpendicular to the maximum
  horizontal stress). Combined with the existing
  ``overburden_stress`` -> ``pore_pressure_eaton`` ->
  ``closure_stress`` -> ``unconfined_compressive_strength``
  pipeline, the geomechanics module now produces the full
  drilling stress-state log from a sonic + density acquisition
  alone. Closed-form derivation:
  ``P_w_crit = (3 sigma_H - sigma_h + (q-1) alpha P_p - UCS)
  / (1 + q)`` with ``q = (1 + sin phi) / (1 - sin phi)``.
  Documented assumptions: vertical well, normal-fault stress
  regime (sigma_v as the intermediate principal stress is
  assumed to be safe), no tensile-failure check
  (the upper bound of the safe mud-weight window is a planned
  follow-up). 14 new tests cover: Kirsch hand-derived values at
  the breakout and breakdown azimuths, isotropic-horizontal-
  stress azimuth-independence, mud-pressure linearity,
  Poisson-and-deviator coupling for sigma_z; MC at-critical-
  pressure inverse check, monotonicity in UCS / friction angle /
  pore pressure / horizontal-stress anisotropy, Tresca
  (zero-friction) and dry-rock (alpha=0) limits, friction-angle
  validation; and an end-to-end pipeline test that chains
  overburden -> pore pressure -> closure -> breakout from a
  synthetic 30-depth log.

- **Eaton (1975) sonic pore-pressure prediction**
  (``fwap.geomechanics.pore_pressure_eaton``). Closed-form
  pore-pressure log from a sonic-slowness log, an overburden-
  stress log, and a normal-compaction-trend slowness:
  ``P_p = sigma_v - (sigma_v - P_hydro) * (Dt_normal / Dt_obs)^n``
  with the standard Eaton exponent ``n = 3.0``. Plus a helper
  ``hydrostatic_pressure(depth, fluid_density=1000.0)`` that
  computes :math:`P_\mathrm{hydro} = \rho_w \, g \, z`. Closes
  a missing-input gap in the existing ``closure_stress``
  function: callers can now produce the full
  ``overburden -> pore -> closure`` stress-state pipeline from
  a sonic + density log alone. Documented limitations: the
  sonic Eaton method is calibrated for shales and undercompaction-
  driven overpressure; unloading mechanisms (gas generation,
  diagenesis) need Bowers' method, which is left as a follow-up.
  17 new tests cover: hydrostatic linearity and density scaling;
  Eaton's normal-compaction reduction to ``P_hydro``; severe-
  overpressure approach to ``sigma_v``; sub-hydrostatic /
  depleted-zone case; depth-vs-explicit-hydrostatic agreement;
  Eaton-exponent sensitivity; round-trip test with
  ``overburden_stress`` on a synthetic 30-depth log; input
  validation.

- **Unified Stoneley fracture-density log**
  (``fwap.rockphysics.stoneley_fracture_density``). Pure combiner
  that mixes the four primitive Stoneley indicators
  (``stoneley_permeability_indicator``,
  ``stoneley_amplitude_fracture_indicator``,
  ``stoneley_permeability_tang_cheng``,
  ``hornby_fracture_aperture``) into a single per-depth fracture-
  intensity score in ``[0, 1]``. The matrix-permeability output
  is used as a binary partitioning flag: depths where the TCT
  inversion returned NaN (out-of-model = simplified Biot-Rosenbaum
  cannot account for the observed slowness shift) keep the full
  slowness contribution; depths with finite kappa (matrix-explained)
  have the slowness contribution suppressed. Aperture term uses
  a tanh saturation with a 1 mm reference scale; weights and
  scales are tunable via keyword arguments. Heuristic combiner,
  not a calibrated geomechanical fracture density -- documented
  as such. 12 new tests cover: zero-indicator tight zone, score
  clipping to [0, 1], default-weight slowness-only and
  amplitude-only paths, matrix-partitioning logic, tanh-saturated
  aperture, NaN-aperture handling, partial monotonicity, and
  input validation.

### Added
- **Backus (1962) layered-medium averaging**
  (``fwap.anisotropy.backus_average``). Long-wavelength
  homogenisation of a stack of N isotropic layers into a single
  effective transversely-isotropic (VTI) elastic tensor with
  vertical symmetry axis. Returns a ``BackusResult`` with the
  five independent VTI elastic constants
  ``c11, c13, c33, c44, c66`` (Pa) plus the volume-weighted
  effective density. Layer-parallel components ``c11, c66`` are
  Voigt-like arithmetic averages; layer-perpendicular
  ``c33, c44`` are Reuss-like harmonic averages; ``c13`` is the
  standard Backus combination of ``lambda / (lambda + 2 mu)``
  averages. Useful for upscaling thinly-bedded sonic-log
  intervals to seismic resolution. 12 new tests cover: isotropic-
  limit recovery (single layer or uniform stack -> exact
  per-layer moduli), thickness-scale invariance (only volume
  fractions matter), Voigt-Reuss inequalities (``c66 >= c44`` and
  ``c11 >= c33`` always hold), positive-definiteness of the
  resulting tensor, hand-derived two-layer numerical check, and
  input validation.

### Changed
- **Repository-wide ``ruff format`` sweep**: 42 files reformatted to
  ruff-format defaults. Behaviour-preserving (full test suite still
  passes; same 433 / 1 skipped count as before the sweep). Closes
  Open Item E on the roadmap.
- **``ruff check`` lint debt cleanup**: 56 pre-existing lint
  warnings cleared from the tree. 52 ``I001`` (import-block
  ordering) auto-fixed by ``ruff check --fix`` -- mostly local
  ``import pytest`` blocks in tests that needed a blank line
  between ``import pytest`` and the subsequent ``from fwap.x
  import y``. 2 ``B023`` (loop-variable-not-bound) instances in
  ``stoneley_dispersion`` and ``flexural_dispersion`` fixed by
  binding ``omega`` as a default argument
  (``def _det(kz, omega=omega): ...``); the closure was always
  safe (the inner function was only called within the same loop
  iteration via ``brentq``) but explicit binding silences the
  warning and removes a footgun. 1 ``B007`` (unused loop
  variable ``lbl`` in ``demos.py``) renamed to ``_lbl``. 1
  ``F841`` (unused ``Vst`` in a Stoneley-omitted test) removed.

### Added
- **Pre-commit config** (``.pre-commit-config.yaml``) with both
  the ``ruff-check`` (with ``--fix``) and ``ruff-format`` hooks.
  Run ``pre-commit install`` after cloning to prevent format and
  lint drift on future commits.
- **Variable candidate budget for joint Viterbi picker**: when the
  raw per-depth tuple count ``prod(n_i + 1)`` would exceed
  ``max_triples_per_depth``, the trellis builder now automatically
  tightens per-mode top-K (preferring high-coherence candidates
  within each mode) to fit the budget rather than raising. Replaces
  the earlier hard-fail-on-overflow with graceful degradation;
  pathological peak-heavy STC surfaces no longer kill the sweep.
- **4-mode joint Viterbi**: ``viterbi_pick_joint`` and
  ``viterbi_posterior_marginals`` are now N-mode generic. Default
  priors changed from the (P, S, Stoneley) subset to the full
  ``DEFAULT_PRIORS`` (4 modes including PseudoRayleigh). Pass an
  explicit 3-mode subset to ``priors=`` if the previous default
  behavior is preferred. The 4-mode trellis is kept tractable by
  the variable-candidate-budget machinery above. Closes Open Item
  C on the roadmap (both sub-items).

### Changed
- ``viterbi_pick_joint`` and ``viterbi_posterior_marginals`` no
  longer reject 4-mode prior dicts. Empty prior dicts now raise
  ``ValueError`` with a clear message instead of silently producing
  empty picks.

### Added
- **Quantitative Stoneley permeability** via the Tang-Cheng-Toksoz
  (1991) simplified Biot-Rosenbaum closed form
  (`fwap.rockphysics.stoneley_permeability_tang_cheng`).
  Calibrated complement to the dimensionless rank-ordering
  returned by `stoneley_permeability_indicator`: takes the
  observed Stoneley slowness, a tight reference, and the standard
  set of Biot / fluid parameters (frequency, K_f, eta, rho_f,
  porosity, frame K_phi); returns absolute formation permeability
  in m^2 (multiply by ~9.87e-13 for darcies). Real-valued
  inversion (slowness shift only); the imaginary-part
  (attenuation) inversion is a follow-up. Out-of-model handling:
  `alpha_ST <= 0` (tight or noise-driven negative) clipped to
  `kappa = 0`; `alpha_ST >= K_f / (2 K_phi)` returns NaN
  (typical cause: open fractures requiring the
  `hornby_fracture_aperture` model rather than the matrix-flow
  Biot-Rosenbaum model). 11 new tests including a round-trip
  check against a Tang & Cheng (2004) fig 5.3-style synthetic
  (1-2 darcy permeable bed bracketed by 0.01-0.1 mD tight
  limestone). Closes Open Item B on the roadmap.
- **n=1 dipole flexural modal-determinant solver**
  (`fwap.cylindrical_solver.flexural_dispersion`). Companion to the
  existing n=0 Stoneley solver (Schmitt 1988); root-finds the
  zeros of the 4x4 isotropic-elastic dipole modal determinant in
  the bound-mode regime to produce the dipole flexural dispersion
  curve directly from the underlying boundary-value problem,
  replacing the rational-interpolation phenomenology in
  `fwap.cylindrical.flexural_dispersion_physical` with the
  cylindrical-Biot physics. Public surface: `flexural_dispersion(
  freq, *, vp, vs, rho, vf, rho_f, a)` returns a `BoreholeMode`
  with `name="flexural"` and `azimuthal_order=1`. Bound-mode
  regime only -- requires slow formations (`V_S < V_f`); fast
  formations and below-cutoff frequencies return NaN, matching
  the existing `stoneley_dispersion` out-of-regime convention.
  In a typical slow formation the recovered slowness sits at
  `1 / V_S` just above the geometric cutoff and rises toward
  slightly above `1 / V_R` at high frequency (the few-percent
  Scholte / fluid-loading offset that the phenomenological model
  does not capture). 12 new tests (slow-formation asymptotes,
  monotonicity, qualitative agreement with the phenomenological
  model, fast-formation NaN behavior, modal-determinant zero
  structure, input validation, dataclass contract). The full
  algebraic derivation -- field ansatz, displacements, stresses
  with the Lame reduction, BC strip, phase rescaling to a real
  4x4, low-f and high-f asymptotic cross-checks -- is documented
  in line in `cylindrical_solver.py` (substeps 1.1 through
  1.6.e). Closes the dipole half of the Open Item A
  ("Full cylindrical-Biot dispersion solver") on the roadmap;
  leaky-mode pseudo-Rayleigh and quadrupole n=2 remain open for
  follow-up via complex-`k_z` Mueller iteration with outgoing
  Hankel-function boundary conditions.
- **LWD (logging-while-drilling) phenomenological layer**
  (`fwap.lwd`). Models the steel-drill-collar contamination Tang &
  Cheng (2004) sect. 2.4-2.5 frame as the defining processing
  problem of the LWD era, plus the two practical responses:
  collar-band notching and quadrupole-source / receiver geometry.
  Public surface: `lwd_collar_mode(...)` returns a pre-configured
  Gabor `Mode` at the published 80-130 us/ft collar band;
  `synthesize_lwd_gather(...)` plants the collar on top of the
  formation modes; `notch_slowness_band(...)` is a slowness-band-
  stop filter via tau-p forward + cosine-tapered band-pass mask +
  adjoint **+ subtract-the-in-band** (the subtraction route
  preserves signals at slownesses outside the tau-p grid, e.g.
  Stoneley at ~217 us/ft survives a notch at ~92 us/ft); on the
  quadrupole side, `QuadrupoleRingGather`,
  `synthesize_quadrupole_lwd_gather(...)` builds a ring of n_rec
  >= 4 receivers with a `cos(2(theta - phi))` source pattern and
  `quadrupole_stack(data, azimuths, ...)` projects the ring onto
  m=2, rejecting m=0 / m=1 patterns by orthogonality;
  `lwd_quadrupole_priors()` returns a tool-aware picker priors
  dict. `fwap.demos.demo_lwd` and `fwap lwd` CLI provide a
  worked-example end-to-end. **Not** a layered cylindrical-Biot
  solver -- that is research-grade work and remains future work.
- **Stress-vs-intrinsic anisotropy classifier from a fast / slow
  flexural dispersion-curve crossover.** Sinha & Kostek (1996)
  showed that the two cross-dipole flexural dispersion curves of a
  stress-anisotropic formation cross over in frequency: the
  low-frequency mode samples the far-field rock fabric and the
  high-frequency mode samples the near-wellbore stress
  concentration, so a Δs(f) sign flip between the bands flags
  stress-induced anisotropy (intrinsic anisotropy shows no such
  crossover). New `classify_flexural_anisotropy(curve_a, curve_b,
  ...)` returns a `FlexuralDispersionDiagnosis` with a
  classification in `{"isotropic", "intrinsic", "stress_induced",
  "ambiguous"}`, the per-band Δs averages, the interpolated
  crossover frequency (when present, restricted to the bracket
  spanning the two band means), and a tuple of human-readable
  reasons for QC.
- **Slow-formation Vs from low-frequency Stoneley slowness**
  (`fwap.rockphysics.vs_from_stoneley_slow_formation`). Inverts the
  White (1983) tube-wave formula `S_ST^2 = 1/V_f^2 + rho_f / mu`
  for the formation shear velocity. Primary sonic-only Vs
  estimator for slow formations (V_S < V_fluid) where the
  formation has no critically-refracted S head wave on a monopole
  gather (Paillet & Cheng 1991, Ch. 3) and pseudo-Rayleigh does
  not exist. Same physics as
  `stoneley_horizontal_shear_modulus` (which returns C_66 for
  VTI), divided by rho and square-rooted; the difference is
  interpretation.
- **Hornby et al. (1989) Stoneley reflection-coefficient fracture-
  aperture inversion.** Quantitative complement to the existing
  Stoneley indicators. New `stoneley_reflection_coefficient(A_inc,
  A_refl)` builds `|R|` from incident / reflected pulse
  amplitudes; new `hornby_fracture_aperture(R, frequency_hz,
  V_T, ...)` inverts the rigid-frame, low-frequency, single-
  fracture closed form
  `|R(omega)| = omega L_0 / sqrt(V_T^2 + omega^2 L_0^2)` for the
  fracture aperture L_0 (m). Saturates at +inf for `|R| -> 1`; an
  optional small-amplitude approximation `L_0 ~ V_T |R| / omega`
  is < 5 % off for |R| <= 0.3.
- **Stoneley amplitude fracture indicator**
  (`fwap.rockphysics.stoneley_amplitude_fracture_indicator`).
  Companion to the existing slowness-shift permeability indicator.
  Returns `1 - A_obs / A_ref` -- the fractional Stoneley amplitude
  deficit relative to a tight reference. Detects the same
  fractures / permeable zones via the loss of acoustic energy
  rather than via the dynamic-poroelastic delay; complementary
  noise characteristics, so a coincidence flag is more robust than
  either indicator alone.
- **Dispersive STC for the pseudo-Rayleigh / guided trapped mode**
  (`fwap.dispersion.dispersive_pseudo_rayleigh_stc`). Direct
  pseudo-Rayleigh analogue of `dispersive_stc`: scans formation
  shear slowness, applies the per-frequency phase-slowness
  correction from `pseudo_rayleigh_dispersion`, returns an
  `STCResult` whose slowness axis is the formation `1 / V_S`.
  Removes the high-frequency bias that plain STC produces on
  guided arrivals. Enforces the fast-formation existence
  constraint (`shear_slowness_range[1] < 1 / v_fluid`).
- **Geomechanics indices on top of `ElasticModuli`** (new module
  `fwap.geomechanics`). Closes the gap between the elastic-moduli
  output of `fwap.rockphysics.elastic_moduli` and the
  Workflow-3 deliverables Mari et al. (1994) Part 3 lists --
  *sanding prediction* and *hydraulic-fracture design*. Public
  surface: `brittleness_index_rickman(E, nu, ...)` (Rickman et al.
  2008 BI in `[0, 1]`); `fracability_index(...)` (alias of BI for
  HF-design call sites); `closure_stress(nu, sigma_v, P_p,
  alpha)` (Eaton 1969 uniaxial-strain closure stress, with
  validation rejecting both `nu >= 1` singularity and `nu < 0`
  auxetic regime); `unconfined_compressive_strength(E,
  model='lacy_sandstone')` (Lacy 1997 / Chang et al. 2006);
  `sand_stability_indicator(mu, threshold)` (Bratli & Risnes 1981
  / 5 GPa shear-modulus rule); `overburden_stress(z, rho)`
  (trapezoidal density-log integration). One-call wrapper
  `geomechanics_indices(moduli, ...)` returns a
  `GeomechanicsIndices` dataclass with `brittleness`,
  `fracability`, `ucs`, `sand_stability` and (when
  `sigma_v_pa` is supplied) `closure_stress`. Module-level
  constants `RICKMAN_E_MIN_PA`, `RICKMAN_E_MAX_PA`,
  `RICKMAN_NU_MIN`, `RICKMAN_NU_MAX`,
  `SAND_STABILITY_SHEAR_THRESHOLD_PA` expose the published
  defaults. Six new LAS / DLIS mnemonics in `_FWAP_UNITS`:
  `BRIT`, `FRAC`, `UCS`, `SH`, `SV`, `SAND`.
- **Thomsen-gamma from combined dipole + Stoneley sonic logs**
  (`fwap.anisotropy`). VTI shear-anisotropy parameter
  `gamma = (C_66 - C_44) / (2 C_44)` from two complementary
  measurements: the dipole shear log gives `C_44 = rho * V_Sv^2`,
  and the Stoneley low-frequency tube-wave inversion (White 1983
  / Norris 1990) gives `C_66 = rho_f / (S_ST^2 - 1/V_f^2)`. New
  `stoneley_horizontal_shear_modulus(s_ST, rho_fluid, v_fluid)`,
  `thomsen_gamma(c44, c66)`, and a one-call
  `thomsen_gamma_from_logs(s_dipole, s_stoneley, rho, ...)`
  returning a `ThomsenGammaResult` with `c44`, `c66`, `gamma`.
  New LAS mnemonics: `C44`, `C66`, `GAMMA`.
- **Picker -> log-curve bridge**
  (`fwap.picker.track_to_log_curves`). Converts a
  `Sequence[DepthPicks]` from `track_modes` / `viterbi_pick` /
  `viterbi_pick_joint` into a `(depths, curves)` tuple where
  `curves` is a `{mnemonic: ndarray}` dict suitable to pass
  straight to `write_las` / `write_dlis`. Standard fwap mnemonics
  (`DTP`, `DTS`, `DTST`, `DTPR` / `COHP`, `COHS`, `COHST`,
  `COHPR` / `AMP*` / `TIM*` / `VPVS`); slowness in us/ft;
  missing picks become NaN by default with an optional numeric
  sentinel via `null_value`. `_FWAP_UNITS` extended to carry the
  new mnemonics.

### Fixed
- **WLS in attenuation Q estimators** (`centroid_frequency_shift_Q`
  and `spectral_ratio_Q`): the previous implementations multiplied
  both `A` and `y` by `W = diag(w)` before passing to `lstsq`,
  which makes the solver minimise `sum(w_i^2 * r_i^2)` instead of
  the documented `sum(w_i * r_i^2)`. Switched to `sqrt(W) @ A`,
  `sqrt(W) @ y` so the per-trace weights match the docstring intent
  ("weights = total power") and the residual variance / standard
  error formulas now match the system actually being solved.
- **`read_las` depth-curve detection**: skipped the depth curve
  via mnemonic equality with `las.curves[0].mnemonic`, which was
  fragile when a non-depth curve happened to share the depth
  mnemonic. Now skips by index instead (always first).
- **`viterbi_pick` doc bug**: the comment claimed the
  no-previous-mode sentinel was `+inf`; the code (correctly) uses
  `-inf`. Comment fixed.
- **`anisotropy_strength` docstring**: claimed the metric reaches 1
  for orthogonal waveforms, but actually reaches `1 / sqrt(2)` for
  orthogonal equal-energy waveforms and only saturates at 1 as
  `s -> -f`. Updated formula and verbal description; the metric
  itself was correct.

### Changed
- **`closure_stress` validation tightened** to reject negative
  Poisson's ratio. The Eaton uniaxial-strain model is calibrated
  for the positive-Poisson regime of typical sedimentary rocks;
  negative inputs would produce negative effective horizontal
  stresses. Auxetic materials are out of scope.
- **`classify_flexural_anisotropy`**: band-overlap guard tightened
  from `>` to `>=` (touching bands now rejected); crossover-
  frequency search restricted to the bracket
  `[f_low_band[0], f_high_band[1]]` so a spurious noise zero-
  crossing outside the bands is not reported as the band-to-band
  transition.
- **`track_to_log_curves`**: float-coerce `null_value` at function
  entry (`null_value = float(null_value)`) so passing `None` or
  any non-float type raises `TypeError` cleanly instead of
  slipping through the NaN check and producing object-dtype
  curves.
- **`dispersive_stc`**: rename misleading internal variable
  `tau_of_f` -> `s_of_f`. The variable holds slowness, not a time
  τ; no behaviour change.
- **`sand_stability_indicator`**: docstring now explicit that the
  `mu == threshold_pa` boundary is treated as stable.

- **Cross-mode consistency QC for the picker.** Closes the soft
  Workflow-3 gap flagged in the closing paragraph of the docx
  review: the book's QC philosophy is *"log continuity AND
  cross-consistency between modes"*; continuity was already
  enforced inside the Viterbi pickers but no API surfaced the
  cross-consistency layer. New ``quality_control_picks(picks,
  depth=, *, vp_vs_min=1.4, vp_vs_max=2.6, require_time_order=
  True)`` returns a ``PickQualityFlags`` with two checks: the
  Vp/Vs ratio (``s_S / s_P``) is gated against the canonical
  sedimentary-rock band, and the canonical time ordering
  ``t_P <= t_S <= t_PseudoRayleigh <= t_Stoneley`` is verified
  over the modes that were picked. The function only flags --
  callers decide whether to drop, mark, or human-review the
  flagged depths. ``quality_control_track`` is the multi-depth
  analogue. Picks whose Vp or S is missing have ``vp_vs=None``
  and ``vp_vs_in_band=True`` (the gate is skipped, not failed).
- **Stress-direction / flexural-fracture-indicator API on top of the
  cross-dipole Alford rotation.** The book frames the Workflow-3
  dipole-sonic deliverable as "shear anisotropy, mechanical
  properties and fracture indicators from the flexural wave" plus
  "stress-direction estimation"; the numerics already lived inside
  ``alford_rotation`` (the fast-shear angle, the cross-energy
  ratio) but no API surfaced them in those petrophysical terms.
  New ``stress_anisotropy_from_alford(alford, dt)`` returns a
  ``StressAnisotropyEstimate`` carrying ``max_horizontal_stress
  _azimuth`` (= fast-shear angle, with a docstring caveat that the
  conventional stress-direction interpretation depends on whether
  the anisotropy is stress- or fracture-induced),
  ``min_horizontal_stress_azimuth`` (orthogonal, folded into
  ``(-pi/2, pi/2]``), ``splitting_time_delay`` (cross-correlation
  lag of slow vs fast), ``anisotropy_strength`` (relative L2 norm
  in ``[0, 1]``), ``rotation_quality`` (= ``1 - cross_energy_ratio``)
  and a heuristic ``fracture_indicator`` (their product). The
  underlying ``alford`` is kept on the result so callers that need
  the rotated waveforms can still reach them.
- **Wavelet-shape + onset-polarity expert rules in the picker.** The
  book (Mari et al. 1994, Part 1) lists *"expert rules on slowness
  range, **wavelet shape, onset polarity**, coherence across the
  receiver array, depth-to-depth continuity"* as the picker's
  knowledge-based discriminator set; the package only had the
  bolded subset (slowness windows + STC coherence + Viterbi
  continuity). Two opt-in expert rules now run as post-pick gates:
  ``polarity`` (``+1``/``-1``/``0``) checks the sign of the
  stacked window's largest-absolute sample, and ``shape_match_min``
  gates picks against the absolute Pearson correlation between the
  stacked window and a Ricker template at the prior's ``f0``. Both
  are exposed via ``filter_picks_by_shape(picks, data, dt,
  offsets, *, priors)`` and ``filter_track_by_shape(track, datas,
  dt, offsets, *, priors)``; the underlying ``onset_polarity`` and
  ``wavelet_shape_score`` primitives are also public. Default
  priors leave both gates disabled (``polarity=0``,
  ``shape_match_min=0.0``) so existing callers are unaffected.
- **Per-mode amplitude logs in the Workflow-1 picker pipeline.** The
  book (Mari et al. 1994, Part 1) frames the rule-based picker's
  deliverable as *"continuous Vp, Vs and Stoneley slowness curves
  together with per-mode amplitude **and coherence** logs"*; the
  pipeline only carried the coherence half. ``STCResult`` now has an
  ``amplitude`` ``ndarray | None`` field of the same
  ``(n_slowness, n_time)`` shape as ``coherence``, populated by both
  ``coherence.stc`` and ``dispersion.dispersive_stc`` with the RMS
  of the per-trace stack contribution at each cell (a unit-amplitude
  sine on every trace gives ``amplitude = 1/sqrt(2)``).
  ``find_peaks`` returns a 4-column ``[slowness, time, coherence,
  amplitude]`` table when the input STC carries amplitude, and
  ``ModePick`` gained an ``amplitude: float | None`` field that
  ``pick_modes`` / ``track_modes`` / ``viterbi_pick`` /
  ``viterbi_pick_joint`` / ``viterbi_posterior_marginals`` now
  populate from the picked cell. Existing 3-column / amplitude=None
  call sites continue to work unchanged.
- **Altered-zone velocity contrast as a Workflow-2 deliverable.**
  The book (Mari et al. 1994, Part 3) frames the intercept-time
  workflow's altered-zone product as the *(thickness, velocity-
  contrast)* pair, but `fwap.tomography` only had
  `delay_to_altered_zone_thickness` -- which forces the caller to
  supply the altered-zone slowness as an input. The single
  refraction-geometry equation `delay = 2 * h * (s_altered -
  s_virgin)` is one constraint in two unknowns, so the package now
  exposes both directions plus a joint helper:
  `delay_to_altered_zone_velocity_contrast(delay, thickness)` is
  the algebraic dual; `altered_zone_estimate(delay, s_virgin,
  thickness=...)` or `altered_zone_estimate(delay, s_virgin,
  slowness_altered=...)` returns an `AlteredZoneEstimate` dataclass
  carrying thickness, absolute altered slowness, and slowness
  contrast at every depth, with the helper rejecting calls that
  pin both or neither anchor.
- **τ-p (slant-stack / linear Radon) wave separation.** The book
  (Mari et al. 1994, Part 2) lists the τ-p domain alongside f-k as
  a textbook multichannel velocity-filter, but `fwap.wavesep`
  previously offered only `fk_filter` and SVD/K-L. New
  `tau_p_forward`, `tau_p_adjoint`, `tau_p_inverse`, and
  `tau_p_filter` mirror the f-k API: forward stacks a (t, x) gather
  into a (τ, p) panel, the LSQR-style `tau_p_inverse` is a true
  per-frequency pseudoinverse (round-trip identity to ~0.1 % on a
  clean monopole gather), and the convenience `tau_p_filter` does
  forward → cosine-tapered slowness mask → adjoint for band-pass
  separation. Unlike `fk_filter`, τ-p tolerates non-uniform
  receiver spacings. New `demos.demo_tau_p_separation` and CLI
  subcommand `fwap taup` exercise the pipeline on the canonical
  P/S/Stoneley monopole gather.
- **Pseudo-Rayleigh / guided-mode picking.** The book (Mari et al.
  1994, Part 1) lists pseudo-Rayleigh alongside P, S, and Stoneley as
  one of the arrivals the rule-based picker must identify in fast
  formations; it was the only mode in that list missing from the
  package. `fwap.picker.DEFAULT_PRIORS` now carries a
  `"PseudoRayleigh"` entry (130-200 us/ft), and Stoneley's lower
  bound has been tightened from 180 to 200 us/ft so the four
  windows are non-overlapping. New phenomenological dispersion law
  `fwap.synthetic.pseudo_rayleigh_dispersion(vs, v_fluid, a_borehole)`
  matches the formation shear slowness at the low-frequency cutoff
  and asymptotes to the borehole-fluid slowness at high frequency.
  `monopole_formation_modes(...)` gained a `f_pr=` kwarg that
  appends a fourth mode at the band-centre slowness predicted by
  that law. New `demos.demo_pseudo_rayleigh` and CLI subcommand
  `fwap pseudorayleigh` exercise the four-mode pipeline end-to-end.
  `viterbi_pick_joint` and `viterbi_posterior_marginals` remain
  hardcoded to the (P, S, Stoneley) triple (extending the trellis
  to 4 modes squares its width); both now subset the default priors
  for backward compatibility and raise on a 4-mode prior dict.
- **DLIS read / write** — `fwap.io.read_dlis`, `fwap.io.write_dlis`,
  and `DlisCurves` mirror the existing LAS API for the binary RP66 v1
  format. Wraps `dlisio` for reading and `dliswriter` for writing.
  Well metadata is re-keyed to LAS-2.0 mnemonics (`WELL`, `COMP`,
  `FLD`, `PROD`, `UWI`) so the same dict can be passed to either
  writer.

### Changed
- **All log-format libraries are now core dependencies.** `lasio`,
  `dlisio`, `dliswriter`, and `segyio` are folded into the base
  `dependencies` list; the `[io]`, `[dlis]`, and `[segy]` extras
  are gone. The corresponding lazy-import helpers (`_require_lasio`,
  `_require_dlisio`, `_require_dliswriter`, `_require_segyio`) and
  their friendly-error guards have been removed.

## [0.4.0] - 2026-04-22

First formally-versioned release. Promotes the port of the 1994 Mari
et al. algorithms from a prototype into a tested, documented Python
package.

### Added
- **Package infrastructure**
  - Repository renamed `src/` to `fwap/` so the `from fwap.X import Y`
    imports throughout the codebase actually resolve.
  - `pyproject.toml` with runtime dependencies (NumPy, SciPy,
    Matplotlib) and optional `dev` / `docs` / `io` extras.
  - `LICENSE` (MIT).
  - `CITATION.cff` citing both the software and the 1994 book.
  - `README.md` expanded from a title-only placeholder to a
    chapter-to-module map, install + quick-start, companion
    references, and links to tests and docs.
  - `CHANGELOG.md` (this file).
- **Tests** (`tests/`, 83 cases, ~9 s run):
  - One file per algorithm module plus edge cases, sign-convention
    invariants, and demo regression tests that assert on the numerics
    logged by each `demo_*` function.
  - `.github/workflows/ci.yml` runs pytest on Python 3.9 / 3.11 /
    3.12 and smoke-tests `python -m fwap --quiet`.
- **Documentation** (`docs/`):
  - Sphinx skeleton (`conf.py`, `index.rst`, `quickstart.rst`,
    `chapter_map.rst`, `api.rst`, `changelog.rst`) that autogenerates
    an API reference from the docstrings.
  - `.readthedocs.yaml` so the repo can be connected to ReadTheDocs
    without further configuration.
  - CI `docs` job builds the site and uploads the rendered HTML as
    an artifact.
- **New algorithms / modules**:
  - `fwap.rockphysics.elastic_moduli(vp, vs, rho)` -> bulk / shear /
    Young's modulus and Poisson's ratio, closing the loop from
    raw-waveform Vp/Vs to geomechanical curves.
  - `fwap.rockphysics.vp_vs_ratio` (lithology / fluid indicator).
  - `fwap.io.read_las` / `fwap.io.write_las` via the optional
    `lasio` dependency.
- **API extensions**:
  - `fwap.anisotropy.alford_rotation_from_tensor` accepts a packed
    `(2, 2, n_samples)` cross-dipole tensor.
  - `fwap.dip.AzimuthalGather` NamedTuple returned by
    `synthesize_azimuthal_arrival` (tuple unpacking still works).
  - `fwap.synthetic.ArrayGeometry.schlumberger_array_sonic()`
    classmethod documenting the canonical 8/10 ft/6 in reference
    geometry.
  - `fwap.logger` is the shared package logger; every submodule
    imports it from `fwap._common`.
  - `fwap.picker.track_modes` gained `continuity_tol_cap_factor`
    (default 3.0) bounding the effective jump tolerance across long
    runs of missed picks.
- **Performance**: per-frequency inner loops vectorised in
  `dispersive_stc`, `phase_slowness_from_f_k` (both methods),
  `phase_slowness_matrix_pencil`, and `synthesize_gather` (5x-42x
  speedups on the reference benchmarks).
- **Docstrings**: every algorithm module now carries a book reference
  and a chapter map linking back to Mari, Coppens, Gavin & Wicquart
  (1994); every public symbol has Parameters / Returns sections with
  units and array shapes.

### Changed
- `solve_intercept_time` with `mean_delay_zero=True` now emits two
  separate zero-sum constraint rows (one per delay block). The
  previous single joint row left a
  `(d_src, d_rec) -> (d_src + c, d_rec - c)` gauge unresolved.
- `centroid_frequency_shift_Q` / `spectral_ratio_Q`: internal
  variable renamed `t_arr -> t_travel` to match Quan & Harris (1997);
  the docstring now flags the Gaussian-source assumption.
- `shear_slowness_from_dispersion` emits `logging.warning` when it
  has to fall back from the quality-weighted set to the unweighted
  set.
- `_coherence_after_detilt` now calls `fwap.coherence.semblance`
  instead of reimplementing the ratio; the two sites can no longer
  diverge on the semblance definition.
- `dispersion_family` (argument of `dispersive_stc`) and
  `Mode.dispersion` now document an array-in / array-out contract;
  `dipole_flexural_dispersion`'s type hint reflects it.

### Fixed
- Alford rotation docstring formula was off by a factor of 2:
  `tan(4 theta) = (A + B) / (C - D) = 2 A / (C - D)`. The code was
  already correct.
- `fk_filter` sign convention (`S = -k/f`) is now documented in the
  function's docstring rather than only an inline comment.

### Removed
- `fwap.coherence.semblance` no longer takes a `min_energy`
  parameter. The default was effectively dead (it fired only on a
  bit-exact zero sum). Callers needing an energy floor should filter
  windows upstream.
- `fwap.picker.pick_modes` / `track_modes`:
  `selection_rule="earliest"` was carrying a legacy fwap03 picker
  rule with no tests exercising it. Removed; use
  `"max_coherence"` or `"scored"` (the default).
- `fwap.picker.track_modes`: the deprecated `max_slow_jump_per_depth`
  keyword alias was removed. Callers must use `max_slow_jump`; the
  old name now raises `TypeError` with the standard "unexpected
  keyword argument" message.
- `fwap.tomography.solve_intercept_time`: the deprecated
  `smoothing` scalar was removed. Use the explicit per-block
  weights `smooth_s`, `smooth_src`, `smooth_rec` (all default 0.0).
- Internal references to legacy version tags ("fwap01", "fwap02",
  "v1", "v2", "testing_fwap03.py") have been stripped from
  docstrings and user-facing comments. Those were porting notes
  that did not belong in the published API documentation.
