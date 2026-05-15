"""
Cross-mode consistency QC and the picker-to-log-curve bridge.

Two end-of-pipeline concerns grouped together:

* **QC**: :func:`quality_control_picks` and
  :func:`quality_control_track` flag depths where the picked
  modes violate canonical sedimentary-rock Vp/Vs / Thomsen-gamma
  bands or canonical time-ordering. Covers the
  *cross-consistency between modes* layer of the QC philosophy
  in Mari et al. (1994), Part 1 closing paragraph -- the
  *log continuity* layer is enforced inside the Viterbi pickers.
* **Log curve bridge**: :func:`track_to_log_curves` converts a
  per-depth pick track into the standard fwap LAS / DLIS mnemonic
  set (``DTP``, ``DTS``, ``COHP``, ``AMP*``, ``VPVS``, optional
  VTI columns), which is what :func:`fwap.io.write_las` /
  :func:`fwap.io.write_dlis` consume directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fwap._common import US_PER_FT
from fwap.picker._types import DepthPicks, ModePick

# Canonical sedimentary-rock Vp/Vs band, used as the default gate in
# :func:`quality_control_picks`. Sources: gas-charged sandstones
# bottom out around ~1.4; high-clay shales and saturated carbonates
# top out around ~2.5-2.6 (Castagna et al. 1985; Mavko, Mukerji &
# Dvorkin, *Rock Physics Handbook*, 2nd ed., chap. 7).
_DEFAULT_VP_VS_MIN = 1.4
_DEFAULT_VP_VS_MAX = 2.6

# Physically-reasonable Thomsen gamma band, used as the default gate
# in :func:`quality_control_picks` when a per-depth gamma is supplied.
# The canonical *VTI shale* window is the tighter ``[0.05, 0.30]``
# (Thomsen 1986; Tang & Cheng 2004 sect. 5.4); the defaults here are
# wider so the gate catches obvious mispicks (gamma < -0.05 is
# unusual; gamma > 0.50 almost always means bad picks or a violated
# VTI assumption) without false-positive-ing isotropic carbonates or
# clean sands at gamma ~ 0.
_DEFAULT_GAMMA_MIN = -0.05
_DEFAULT_GAMMA_MAX = 0.50

# Canonical mode time ordering: P first, then S, then the guided
# pseudo-Rayleigh, then Stoneley last. Modes not in this list (or
# absent from the picks dict) are silently skipped.
_CANONICAL_MODE_TIME_ORDER = ("P", "S", "PseudoRayleigh", "Stoneley")

# Mnemonic suffix per canonical mode name, used by
# :func:`track_to_log_curves` to build LAS/DLIS-friendly column names
# (DTP / DTS / DTST / DTPR, COHP / COHS / COHST / COHPR, etc.). Modes
# outside this map fall through to ``mode_name.upper()``.
_MODE_MNEMONIC_SUFFIX: dict[str, str] = {
    "P": "P",
    "S": "S",
    "Stoneley": "ST",
    "PseudoRayleigh": "PR",
}


@dataclass(frozen=True)
class PickQualityFlags:
    """
    Per-depth cross-mode consistency QC for a multi-mode pick set.

    Returned by :func:`quality_control_picks` (and the multi-depth
    :func:`quality_control_track`). Covers the *cross-consistency
    between modes* layer of the book's QC philosophy (Mari et al.
    1994, Part 1, closing paragraph) -- the *log continuity* layer
    is enforced inside the Viterbi pickers.

    Attributes
    ----------
    depth : float
        Depth (m) the QC was computed at.
    vp_vs : float or None
        Vp/Vs ratio derived from the P and S picks
        (= ``s_S / s_P``). ``None`` when either pick is missing or
        when ``s_P`` is zero.
    vp_vs_in_band : bool
        True when ``vp_vs`` lies inside the configured
        ``[vp_vs_min, vp_vs_max]`` physical band, *or* when no
        Vp/Vs could be computed (a missing Vp/Vs is not flagged as
        inconsistent -- it just isn't checked).
    time_order_ok : bool
        True when the picked arrival times respect the canonical
        ordering ``t_P <= t_S <= t_PseudoRayleigh <= t_Stoneley``,
        skipping modes that weren't picked. Useful when the
        upstream picker was run with a soft time-order constraint
        (``viterbi_pick(time_order_slack > 0)`` /
        ``viterbi_pick_joint(soft_time_order=...)``) where the
        ordering can be deliberately violated.
    gamma : float or None
        Thomsen shear-anisotropy parameter for this depth, as passed
        in via the ``gamma`` keyword to :func:`quality_control_picks`.
        ``None`` when not supplied (the gate is then skipped).
    gamma_in_band : bool
        True when ``gamma`` lies inside the configured
        ``[gamma_min, gamma_max]`` band, *or* when no ``gamma`` was
        supplied (a missing gamma is not flagged -- it just isn't
        checked).
    flagged : bool
        True when any check failed at this depth.
    reasons : tuple of str
        Human-readable per-check failure descriptions; empty when
        ``flagged`` is False.
    """

    depth: float
    vp_vs: float | None
    vp_vs_in_band: bool
    time_order_ok: bool
    flagged: bool
    reasons: tuple[str, ...]
    gamma: float | None = None
    gamma_in_band: bool = True


def quality_control_picks(
    picks: dict[str, ModePick] | DepthPicks,
    *,
    depth: float | None = None,
    vp_vs_min: float = _DEFAULT_VP_VS_MIN,
    vp_vs_max: float = _DEFAULT_VP_VS_MAX,
    require_time_order: bool = True,
    gamma: float | None = None,
    gamma_min: float = _DEFAULT_GAMMA_MIN,
    gamma_max: float = _DEFAULT_GAMMA_MAX,
) -> PickQualityFlags:
    """
    Cross-mode consistency QC at one depth.

    Three checks (all opt-out / opt-in):

    * **Vp/Vs in physical band.** Computed as ``s_S / s_P`` (which
      equals Vp/Vs since slowness is the reciprocal of velocity).
      Flagged when outside ``[vp_vs_min, vp_vs_max]``. Skipped --
      and reported as ``vp_vs_in_band=True`` -- when either P or S
      is missing.
    * **Canonical time ordering.** Flagged when the picked arrival
      times do not satisfy ``t_P <= t_S <= t_PseudoRayleigh <=
      t_Stoneley`` over the modes that were picked. Disable by
      passing ``require_time_order=False``.
    * **Thomsen gamma in physical band** (opt-in). Flagged when the
      Thomsen shear-anisotropy parameter (computed externally via
      :func:`fwap.anisotropy.thomsen_gamma_from_logs` or
      :func:`fwap.anisotropy.vti_moduli_from_logs` and passed in
      via the ``gamma`` keyword) lies outside
      ``[gamma_min, gamma_max]``. Skipped -- and reported as
      ``gamma_in_band=True`` -- when ``gamma`` is ``None``. The
      default band is wider than the canonical VTI shale window
      (Thomsen 1986; Tang & Cheng 2004 sect. 5.4 give shales at
      ``[0.05, 0.30]``); the wider default catches mispicks
      (negative gamma is unusual; gamma > 0.50 almost always
      indicates bad picks or a violated VTI assumption) without
      false-positive-ing isotropic carbonates or clean sands at
      gamma ~ 0. Tighten to ``gamma_min=0.05, gamma_max=0.30`` if
      you specifically want a "this depth is in a VTI shale" gate.

    The function only **flags** -- it never modifies the picks. The
    caller decides what to do with flagged depths (drop, mark in
    plots, hand to a human for review).

    Parameters
    ----------
    picks : dict from str to ModePick, or DepthPicks
        Per-mode picks at one depth. When a :class:`DepthPicks` is
        passed, its ``depth`` field is used unless overridden by
        the explicit ``depth`` keyword.
    depth : float, optional
        Override depth value. Required when ``picks`` is a plain
        dict; ignored otherwise unless given.
    vp_vs_min, vp_vs_max : float
        Inclusive Vp/Vs band. Defaults span the canonical
        sedimentary-rock range from gas-charged sandstones (~1.4)
        to clay-rich shales / fluid-saturated carbonates (~2.6).
    require_time_order : bool, default True
        Disable to skip the canonical-ordering check entirely.
    gamma : float, optional
        Thomsen shear-anisotropy parameter for this depth. When
        supplied the function checks it against
        ``[gamma_min, gamma_max]``. ``None`` (default) skips the
        gate.
    gamma_min, gamma_max : float
        Inclusive Thomsen-gamma band. Defaults
        ``[-0.05, 0.50]`` flag obvious mispicks without
        false-positive-ing isotropic samples at gamma ~ 0.

    Returns
    -------
    PickQualityFlags
    """
    if isinstance(picks, DepthPicks):
        if depth is None:
            depth = picks.depth
        picks_dict = picks.picks
    else:
        if depth is None:
            depth = float("nan")
        picks_dict = picks

    reasons: list[str] = []

    # Vp/Vs gate
    vp_vs: float | None = None
    vp_vs_in_band = True
    if "P" in picks_dict and "S" in picks_dict:
        s_p = float(picks_dict["P"].slowness)
        s_s = float(picks_dict["S"].slowness)
        if s_p > 0.0:
            vp_vs = s_s / s_p
            if not (vp_vs_min <= vp_vs <= vp_vs_max):
                vp_vs_in_band = False
                reasons.append(
                    f"Vp/Vs={vp_vs:.2f} outside band [{vp_vs_min:.2f}, {vp_vs_max:.2f}]"
                )

    # Canonical time-ordering gate
    time_order_ok = True
    if require_time_order:
        present = [m for m in _CANONICAL_MODE_TIME_ORDER if m in picks_dict]
        times = [float(picks_dict[m].time) for m in present]
        if times != sorted(times):
            time_order_ok = False
            order_str = ", ".join(
                f"t_{m}={picks_dict[m].time * 1.0e3:.2f}ms" for m in present
            )
            reasons.append(f"canonical time order violated: {order_str}")

    # Thomsen-gamma gate (opt-in)
    gamma_value: float | None = None
    gamma_in_band = True
    if gamma is not None:
        gamma_value = float(gamma)
        if not (gamma_min <= gamma_value <= gamma_max):
            gamma_in_band = False
            reasons.append(
                f"Thomsen gamma={gamma_value:.3f} outside band "
                f"[{gamma_min:.2f}, {gamma_max:.2f}]"
            )

    flagged = (not vp_vs_in_band) or (not time_order_ok) or (not gamma_in_band)
    return PickQualityFlags(
        depth=float(depth),
        vp_vs=vp_vs,
        vp_vs_in_band=vp_vs_in_band,
        time_order_ok=time_order_ok,
        flagged=flagged,
        reasons=tuple(reasons),
        gamma=gamma_value,
        gamma_in_band=gamma_in_band,
    )


def quality_control_track(
    track: Sequence[DepthPicks],
    *,
    vp_vs_min: float = _DEFAULT_VP_VS_MIN,
    vp_vs_max: float = _DEFAULT_VP_VS_MAX,
    require_time_order: bool = True,
    gammas: Sequence[float] | np.ndarray | None = None,
    gamma_min: float = _DEFAULT_GAMMA_MIN,
    gamma_max: float = _DEFAULT_GAMMA_MAX,
) -> list[PickQualityFlags]:
    """
    Apply :func:`quality_control_picks` per-depth across a track.

    Parameters
    ----------
    track : sequence of DepthPicks
        Output of :func:`track_modes`, :func:`viterbi_pick`,
        :func:`viterbi_pick_joint`, or any other multi-depth picker.
    vp_vs_min, vp_vs_max, require_time_order : as in
        :func:`quality_control_picks`.
    gammas : sequence of float or ndarray, optional
        Per-depth Thomsen-gamma values (one per entry in ``track``).
        When supplied, each is forwarded to the per-depth
        :func:`quality_control_picks` call as the ``gamma`` keyword,
        enabling the gamma-band gate. ``None`` (default) skips the
        gate everywhere. Pass ``np.nan`` for individual depths
        where gamma is unavailable -- the per-depth call treats NaN
        as "skip the gamma gate at this depth".
    gamma_min, gamma_max : float
        Inclusive Thomsen-gamma band, forwarded to every per-depth
        call. See :func:`quality_control_picks` for the convention.

    Returns
    -------
    list of PickQualityFlags
        One entry per depth, in the order of ``track``.

    Raises
    ------
    ValueError
        If ``gammas`` is supplied with a different length than
        ``track``.
    """
    if gammas is None:
        per_depth_gammas: list[float | None] = [None] * len(track)
    else:
        gammas_arr = np.asarray(gammas, dtype=float)
        if gammas_arr.size != len(track):
            raise ValueError(
                "gammas must have the same length as track; got "
                f"len(gammas)={gammas_arr.size}, len(track)={len(track)}"
            )
        # Treat NaN as "skip the gamma gate at this depth".
        per_depth_gammas = [
            None if not np.isfinite(g) else float(g) for g in gammas_arr
        ]
    return [
        quality_control_picks(
            dp,
            vp_vs_min=vp_vs_min,
            vp_vs_max=vp_vs_max,
            require_time_order=require_time_order,
            gamma=per_depth_gammas[i],
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        for i, dp in enumerate(track)
    ]


# ---------------------------------------------------------------------
# Track -> log-curve bridge (picker output -> LAS/DLIS writer input)
# ---------------------------------------------------------------------


def track_to_log_curves(
    track: Sequence[DepthPicks],
    *,
    modes: Sequence[str] | None = None,
    include_amplitude: bool = True,
    include_coherence: bool = True,
    include_vp_vs: bool = True,
    include_time: bool = False,
    include_vti: bool = False,
    rho: float | np.ndarray | None = None,
    rho_fluid: float | None = None,
    v_fluid: float | None = None,
    correct_for_p_modulus: bool = True,
    null_value: float = float("nan"),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Convert a per-depth pick track into LAS/DLIS-ready log curves.

    Bridges the picker output (:func:`track_modes`,
    :func:`viterbi_pick`, :func:`viterbi_pick_joint`) and the I/O
    writers (:func:`fwap.io.write_las`, :func:`fwap.io.write_dlis`)
    by building one fixed-length ``(n_depth,)`` array per
    (mode, attribute) pair, keyed by the standard fwap mnemonics.
    Slownesses are converted to **us/ft** (the borehole-acoustic
    unit used by the LAS/DLIS unit table); coherences and amplitudes
    are kept dimensionless / in their input units.

    The Workflow-1 deliverable per Mari et al. (1994), Part 1 is a
    set of continuous Vp / Vs / Stoneley slowness curves with
    matching coherence (and amplitude) tracks. This function is the
    last mile: it produces the dict that
    :func:`fwap.io.write_las(path, depth, curves)` and
    :func:`fwap.io.write_dlis(path, depth, curves)` consume directly.

    Mnemonic conventions
    --------------------
    Canonical mode -> suffix mapping:
      ``P`` -> ``P``, ``S`` -> ``S``, ``Stoneley`` -> ``ST``,
      ``PseudoRayleigh`` -> ``PR``. Modes outside this set use
      ``mode_name.upper()`` as the suffix.

    Per mode, the columns produced are:

    =========  ===================  ========
    Mnemonic   Quantity              Unit
    =========  ===================  ========
    DT*        slowness              us/ft
    COH*       coherence             (-)
    AMP*       per-cell amplitude    (-)
    TIM*       pick time             s
    =========  ===================  ========

    plus a single ``VPVS`` (= ``s_S / s_P`` = ``Vp / Vs``) when both
    P and S are picked.

    With ``include_vti=True`` (and the required ``rho`` /
    ``rho_fluid`` / ``v_fluid`` inputs) the function additionally
    emits the seven VTI columns:

    =========  ============================================  =====
    Mnemonic   Quantity                                       Unit
    =========  ============================================  =====
    C33        :math:`\\rho V_P^2`                            Pa
    C44        :math:`\\rho V_{Sv}^2`                         Pa
    C66        Stoneley-derived horizontal shear modulus      Pa
    GAMMA      Thomsen :math:`\\gamma = (C_{66}-C_{44})/(2 C_{44})` (-)
    VP         :math:`\\sqrt{C_{33}/\\rho}`                   m/s
    VSV        :math:`\\sqrt{C_{44}/\\rho}`                   m/s
    VSH        :math:`\\sqrt{C_{66}/\\rho}`                   m/s
    =========  ============================================  =====

    Each VTI cell is computed only at depths where the underlying
    pick(s) are present:

    * C33 / VP need ``"P"``,
    * C44 / VSV need ``"S"``,
    * C66 / VSH need ``"Stoneley"`` (with a Stoneley slowness above
      :math:`1/V_f`),
    * GAMMA needs both C44 and C66.

    Cells where the relevant pick is missing receive ``null_value``.
    With ``correct_for_p_modulus=True`` (default) C66 uses the Tang
    & Cheng (2004) §5.4 finite-impedance correction at depths where
    the P pick is *also* present; depths where the P pick is
    missing fall back to the literal White (1983) reading
    transparently. This per-depth fall-back is the right
    operational choice for a track that is dense in S/Stoneley but
    sparse in P -- the resulting C66/GAMMA log uses the best
    physics available cell-by-cell rather than dropping out
    entirely on every missed P pick.

    Parameters
    ----------
    track : sequence of DepthPicks
        Output of :func:`track_modes`, :func:`viterbi_pick`,
        :func:`viterbi_pick_joint`, or any other multi-depth picker.
    modes : sequence of str, optional
        Restrict output to these mode names. Defaults to every mode
        that appears anywhere in ``track`` (preserving first-seen
        order).
    include_amplitude : bool, default True
        Emit ``AMP*`` columns. Skipped per mode if no pick of that
        mode carries an amplitude.
    include_coherence : bool, default True
        Emit ``COH*`` columns.
    include_vp_vs : bool, default True
        Emit a ``VPVS`` column when both ``P`` and ``S`` columns
        exist in the output.
    include_time : bool, default False
        Emit ``TIM*`` columns (pick time in seconds). Off by default
        because pick times are intermediate diagnostics rather than
        published log curves.
    include_vti : bool, default False
        Emit the seven VTI columns (``C33``, ``C44``, ``C66``,
        ``GAMMA``, ``VP``, ``VSV``, ``VSH``). Requires ``rho``,
        ``rho_fluid``, ``v_fluid``; raises if any of those is
        ``None``.
    rho : float or ndarray, optional
        Formation bulk density (kg/m^3). Either a scalar (constant
        density across the track) or a length-``n_depth`` per-depth
        array. Required when ``include_vti=True``; ignored
        otherwise.
    rho_fluid : float, optional
        Borehole-fluid density (kg/m^3). Required when
        ``include_vti=True``; ignored otherwise.
    v_fluid : float, optional
        Borehole-fluid acoustic velocity (m/s). Required when
        ``include_vti=True``; ignored otherwise.
    correct_for_p_modulus : bool, default True
        With ``include_vti=True``, apply the Tang & Cheng (2004)
        §5.4 finite-impedance correction to the Stoneley → C66
        inversion at depths where the P pick is also present.
        Depths without a P pick fall back to the literal White
        (1983) reading regardless of this flag. Pass ``False`` to
        force the legacy White path everywhere.
    null_value : float, default ``NaN``
        Fill value at depths where a mode was not picked. ``NaN`` is
        the LAS / DLIS native null marker; pass a numeric sentinel
        like ``-999.25`` if a downstream consumer requires that.

    Returns
    -------
    depths : ndarray, shape (n_depth,)
        Depth axis pulled from ``DepthPicks.depth``, in the same unit
        the picker was called with (typically metres).
    curves : dict[str, ndarray]
        Mnemonic -> ``(n_depth,)`` array. All arrays are aligned on
        ``depths``. Suitable to pass to :func:`fwap.io.write_las` or
        :func:`fwap.io.write_dlis` directly.

    Examples
    --------
    >>> from fwap import (
    ...     track_modes, track_to_log_curves, write_las,
    ... )
    >>> track = track_modes(stc_results, depths)
    >>> depths, curves = track_to_log_curves(track)
    >>> write_las("output.las", depths, curves)
    """
    # Coerce ``null_value`` to a float up-front so a wrong type (e.g.
    # ``None``) raises ``TypeError`` cleanly rather than slipping
    # through the NaN check and producing object-dtype curves.
    null_value = float(null_value)

    n_depth = len(track)
    if n_depth == 0:
        return np.empty(0, dtype=float), {}

    depths = np.array([float(dp.depth) for dp in track], dtype=float)

    if modes is None:
        seen: list[str] = []
        for dp in track:
            for name in dp.picks:
                if name not in seen:
                    seen.append(name)
        modes = seen

    # Build the per-mode columns with NaN as the internal "missing"
    # marker so VPVS arithmetic propagates correctly even when the
    # caller passes a numeric ``null_value``. NaNs are remapped to
    # the requested null_value at the very end.
    nan = float("nan")
    curves: dict[str, np.ndarray] = {}
    for mode in modes:
        suffix = _MODE_MNEMONIC_SUFFIX.get(mode, mode.upper())
        slow_arr: np.ndarray = np.full(n_depth, nan, dtype=float)
        coh_arr: np.ndarray = np.full(n_depth, nan, dtype=float)
        amp_arr: np.ndarray = np.full(n_depth, nan, dtype=float)
        time_arr: np.ndarray = np.full(n_depth, nan, dtype=float)
        any_amp = False
        any_pick = False
        for d, dp in enumerate(track):
            pick = dp.picks.get(mode)
            if pick is None:
                continue
            any_pick = True
            slow_arr[d] = float(pick.slowness) / US_PER_FT
            coh_arr[d] = float(pick.coherence)
            time_arr[d] = float(pick.time)
            if pick.amplitude is not None:
                amp_arr[d] = float(pick.amplitude)
                any_amp = True
        if not any_pick:
            # Mode never appeared in this track -- skip rather than
            # emit an all-null column.
            continue
        curves[f"DT{suffix}"] = slow_arr
        if include_coherence:
            curves[f"COH{suffix}"] = coh_arr
        if include_amplitude and any_amp:
            curves[f"AMP{suffix}"] = amp_arr
        if include_time:
            curves[f"TIM{suffix}"] = time_arr

    if include_vp_vs and "DTP" in curves and "DTS" in curves:
        # s_S / s_P = (1/v_S) / (1/v_P) = v_P / v_S = Vp/Vs.
        # Both columns are us/ft, so the unit cancels. Compute on the
        # NaN-marked internals so a missing P or S at any depth gives
        # NaN here; that NaN is converted to ``null_value`` below.
        with np.errstate(divide="ignore", invalid="ignore"):
            vpvs = curves["DTS"] / curves["DTP"]
        vpvs = np.where(np.isfinite(vpvs), vpvs, nan)
        curves["VPVS"] = vpvs

    if include_vti:
        if rho is None:
            raise ValueError(
                "include_vti=True requires `rho` (formation density "
                "in kg/m^3, scalar or per-depth array)"
            )
        if rho_fluid is None or v_fluid is None:
            raise ValueError(
                "include_vti=True requires `rho_fluid` and `v_fluid` "
                "(borehole-fluid density in kg/m^3 and acoustic "
                "velocity in m/s)"
            )
        if rho_fluid <= 0.0 or v_fluid <= 0.0:
            raise ValueError("rho_fluid and v_fluid must be strictly positive")
        rho_arr = np.asarray(rho, dtype=float)
        if rho_arr.ndim == 0:
            rho_arr = np.full(n_depth, float(rho_arr), dtype=float)
        elif rho_arr.shape != (n_depth,):
            raise ValueError(
                "rho must be a scalar or a length-n_depth array; got "
                f"shape {rho_arr.shape} for n_depth={n_depth}"
            )
        if np.any(rho_arr <= 0):
            raise ValueError("rho must be strictly positive everywhere")

        # Per-depth slownesses in s/m (NaN where the pick is missing).
        s_p_arr: np.ndarray = np.full(n_depth, nan, dtype=float)
        s_s_arr: np.ndarray = np.full(n_depth, nan, dtype=float)
        s_st_arr: np.ndarray = np.full(n_depth, nan, dtype=float)
        for d, dp in enumerate(track):
            p = dp.picks.get("P")
            if p is not None:
                s_p_arr[d] = float(p.slowness)
            s = dp.picks.get("S")
            if s is not None:
                s_s_arr[d] = float(s.slowness)
            st = dp.picks.get("Stoneley")
            if st is not None:
                s_st_arr[d] = float(st.slowness)

        with np.errstate(divide="ignore", invalid="ignore"):
            c33 = rho_arr / (s_p_arr * s_p_arr)
            c44 = rho_arr / (s_s_arr * s_s_arr)
            # White (1983) C66 forward inversion at every Stoneley-
            # picked depth.
            s_f2 = 1.0 / (v_fluid * v_fluid)
            diff = s_st_arr * s_st_arr - s_f2
            c66_white = np.where(diff > 0.0, rho_fluid / diff, np.nan)
            if correct_for_p_modulus:
                # Tang & Cheng (2004) §5.4 correction at depths where
                # the P pick is *also* present (and the resulting
                # correction factor stays positive). Depths without a
                # P pick keep the literal White reading -- documented
                # in the docstring.
                rho_vp2 = rho_arr / (s_p_arr * s_p_arr)
                rho_f_vf2 = rho_fluid * v_fluid * v_fluid
                factor = 1.0 - rho_f_vf2 / rho_vp2
                use_corrected = (
                    np.isfinite(factor) & (factor > 0.0) & np.isfinite(c66_white)
                )
                c66 = np.where(use_corrected, c66_white / factor, c66_white)
            else:
                c66 = c66_white

            gamma = (c66 - c44) / (2.0 * c44)
            vp = np.sqrt(c33 / rho_arr)
            vsv = np.sqrt(c44 / rho_arr)
            vsh = np.sqrt(c66 / rho_arr)

        # Replace any +/- inf or other non-finite values with NaN so
        # the null_value substitution downstream catches them.
        for arr in (c33, c44, c66, gamma, vp, vsv, vsh):
            np.copyto(arr, nan, where=~np.isfinite(arr))

        curves["C33"] = c33
        curves["C44"] = c44
        curves["C66"] = c66
        curves["GAMMA"] = gamma
        curves["VP"] = vp
        curves["VSV"] = vsv
        curves["VSH"] = vsh

    if not (isinstance(null_value, float) and np.isnan(null_value)):
        # Caller wants a numeric sentinel instead of NaN; remap.
        for name, arr in curves.items():
            curves[name] = np.where(np.isnan(arr), null_value, arr)

    return depths, curves
