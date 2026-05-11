"""Rule-based picker tests."""

from __future__ import annotations

import numpy as np
import pytest

from fwap._common import US_PER_FT
from fwap.coherence import stc
from fwap.picker import DEFAULT_PRIORS, pick_modes, track_modes
from fwap.synthetic import (
    ArrayGeometry,
    monopole_formation_modes,
    synthesize_gather,
)


def _make_stc(seed=0, Vp=4500.0, Vs=2500.0, Vst=1400.0):
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    data = synthesize_gather(
        geom, monopole_formation_modes(Vp, Vs, Vst), noise=0.05, seed=seed
    )
    return stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )


def test_pick_modes_returns_p_s_stoneley():
    """The default priors recover all three monopole modes."""
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    picks = pick_modes(_make_stc(Vp=Vp, Vs=Vs, Vst=Vst), threshold=0.4)
    assert set(picks) == {"P", "S", "Stoneley"}
    # Slowness within 10 us/ft of truth.
    tol = 10.0 * US_PER_FT
    assert abs(picks["P"].slowness - 1.0 / Vp) < tol
    assert abs(picks["S"].slowness - 1.0 / Vs) < tol
    assert abs(picks["Stoneley"].slowness - 1.0 / Vst) < tol


def test_pick_modes_respects_ordering():
    """P is required to arrive no later than S, S no later than Stoneley."""
    picks = pick_modes(_make_stc(), threshold=0.4)
    assert picks["P"].time <= picks["S"].time <= picks["Stoneley"].time


def test_pick_modes_populates_amplitude():
    """ModePick.amplitude is populated and positive for the planted modes."""
    picks = pick_modes(_make_stc(), threshold=0.4)
    for name in ("P", "S", "Stoneley"):
        assert picks[name].amplitude is not None
        assert picks[name].amplitude > 0.0


def test_track_modes_max_slow_jump_per_depth_alias_is_gone():
    """The deprecated ``max_slow_jump_per_depth`` alias was removed.

    Pre-0.5.0 breaking removal: callers that relied on the alias must
    migrate to ``max_slow_jump``. Using the old name now raises
    ``TypeError`` (unexpected keyword argument).
    """
    r = _make_stc()
    with pytest.raises(TypeError, match="max_slow_jump_per_depth"):
        track_modes(
            [r, r], depths=np.array([0.0, 0.15]), max_slow_jump_per_depth=1.0e-4
        )


def test_track_modes_continuity_cap_holds():
    """The effective tolerance cap never exceeds cap_factor * max_slow_jump."""
    # We can't directly probe the internal value, but we can verify that
    # setting cap_factor=1 (no growth) matches the max_slow_jump=0
    # behaviour: any pick must lie within max_slow_jump of the previous.
    r = _make_stc()
    picks = track_modes(
        [r, r, r, r],
        depths=np.array([0.0, 0.15, 0.3, 0.45]),
        max_slow_jump=5.0 * US_PER_FT,
        continuity_tol_cap_factor=1.0,
        continuity_tol_growth=0.0,
    )
    p_depths = [dp.picks["P"].slowness for dp in picks if "P" in dp.picks]
    if len(p_depths) > 1:
        jumps = np.abs(np.diff(p_depths))
        assert np.all(jumps <= 5.0 * US_PER_FT + 1e-12)


def test_default_priors_units_are_s_per_m():
    """DEFAULT_PRIORS stores s/m values (sanity check)."""
    # 40 us/ft ~= 1.31e-4 s/m -- if the units were ever mixed this would
    # catch it.
    assert DEFAULT_PRIORS["P"]["slow_min"] == 40.0 * US_PER_FT
    assert DEFAULT_PRIORS["P"]["slow_max"] > DEFAULT_PRIORS["P"]["slow_min"]


# ---------------------------------------------------------------------
# Viterbi picker
# ---------------------------------------------------------------------


def _stc_sequence(n_depth=10, Vp=4500.0, Vs=2500.0, Vst=1400.0, seed=0):
    """Build a sequence of STC surfaces along a depth profile."""
    from fwap.coherence import stc
    from fwap.synthetic import ArrayGeometry

    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    stcs = []
    for d in range(n_depth):
        data = synthesize_gather(
            geom, monopole_formation_modes(Vp, Vs, Vst), noise=0.05, seed=seed + d
        )
        stcs.append(
            stc(
                data,
                dt=geom.dt,
                offsets=geom.offsets,
                slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
                n_slowness=121,
                window_length=4.0e-4,
                time_step=2,
            )
        )
    return stcs


def test_viterbi_pick_recovers_all_three_modes():
    """Viterbi finds P, S, Stoneley on every depth of a clean sweep."""
    from fwap.picker import viterbi_pick

    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    stcs = _stc_sequence(n_depth=8, Vp=Vp, Vs=Vs, Vst=Vst)
    depths = np.arange(8) * 0.1524
    picks = viterbi_pick(stcs, depths=depths, threshold=0.4)
    assert len(picks) == 8
    # Every depth should have all three modes.
    for dp in picks:
        assert set(dp.picks) == {"P", "S", "Stoneley"}
    # Slownesses within 10 us/ft of truth on every depth.
    tol = 10.0 * US_PER_FT
    for dp in picks:
        assert abs(dp.picks["P"].slowness - 1.0 / Vp) < tol
        assert abs(dp.picks["S"].slowness - 1.0 / Vs) < tol
        assert abs(dp.picks["Stoneley"].slowness - 1.0 / Vst) < tol
    # Amplitude is populated by the Viterbi path too.
    for dp in picks:
        for name in ("P", "S", "Stoneley"):
            assert dp.picks[name].amplitude is not None
            assert dp.picks[name].amplitude > 0.0


def test_viterbi_pick_enforces_time_ordering_by_default():
    """Default slack=0 keeps P time <= S time <= Stoneley time."""
    from fwap.picker import viterbi_pick

    stcs = _stc_sequence(n_depth=5, seed=3)
    depths = np.arange(5) * 0.1524
    picks = viterbi_pick(stcs, depths=depths, time_order_slack=0.0)
    for dp in picks:
        t = [dp.picks[n].time for n in ("P", "S", "Stoneley") if n in dp.picks]
        assert t == sorted(t)


def test_viterbi_pick_absence_is_allowed():
    """Modes with no in-prior-window candidates are marked absent, not mispicked."""
    from fwap.picker import viterbi_pick

    # Force every mode's prior window to a slowness range no real
    # sonic mode falls into, so Viterbi should declare all modes
    # absent rather than reaching for out-of-window noise peaks.
    stcs = _stc_sequence(n_depth=4, seed=2)
    depths = np.arange(4) * 0.1524
    impossible_priors = {
        "P": dict(slow_min=1.0e-7, slow_max=5.0e-7, coherence_min=0.4, order=0),
        "S": dict(slow_min=1.0e-7, slow_max=5.0e-7, coherence_min=0.4, order=1),
        "Stoneley": dict(slow_min=1.0e-7, slow_max=5.0e-7, coherence_min=0.4, order=2),
    }
    picks = viterbi_pick(stcs, depths=depths, priors=impossible_priors, threshold=0.4)
    for dp in picks:
        assert dp.picks == {}


def test_viterbi_pick_rejects_length_mismatch():
    """stc_results and depths must have the same length."""
    import pytest

    from fwap.picker import viterbi_pick

    stcs = _stc_sequence(n_depth=3)
    with pytest.raises(ValueError, match="same length"):
        viterbi_pick(stcs, depths=np.array([0.0, 0.1]))


def test_viterbi_pick_empty_input():
    """Empty input returns an empty list."""
    from fwap.picker import viterbi_pick

    assert viterbi_pick([], depths=np.array([])) == []


def test_viterbi_pick_joint_recovers_all_three_modes():
    """Joint 3-mode Viterbi recovers P, S, Stoneley on a clean sweep."""
    from fwap.picker import viterbi_pick_joint

    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    stcs = _stc_sequence(n_depth=6, Vp=Vp, Vs=Vs, Vst=Vst)
    depths = np.arange(6) * 0.1524
    picks = viterbi_pick_joint(stcs, depths=depths, threshold=0.4)
    assert len(picks) == 6
    for dp in picks:
        assert set(dp.picks) == {"P", "S", "Stoneley"}
    tol = 10.0 * US_PER_FT
    for dp in picks:
        assert abs(dp.picks["P"].slowness - 1.0 / Vp) < tol
        assert abs(dp.picks["S"].slowness - 1.0 / Vs) < tol
        assert abs(dp.picks["Stoneley"].slowness - 1.0 / Vst) < tol


def test_viterbi_pick_joint_enforces_time_order():
    """Default slack=0 keeps P <= S <= Stoneley in every output depth."""
    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=5, seed=11)
    depths = np.arange(5) * 0.1524
    picks = viterbi_pick_joint(stcs, depths=depths, time_order_slack=0.0)
    for dp in picks:
        times = [dp.picks[n].time for n in ("P", "S", "Stoneley") if n in dp.picks]
        assert times == sorted(times)


def test_viterbi_pick_joint_agrees_with_sequential_on_clean_data():
    """On a clean synthetic the two picker flavours agree on Vp/Vs/Vst."""
    from fwap.picker import viterbi_pick, viterbi_pick_joint

    stcs = _stc_sequence(n_depth=4, seed=13)
    depths = np.arange(4) * 0.1524
    seq = viterbi_pick(stcs, depths=depths)
    jnt = viterbi_pick_joint(stcs, depths=depths)
    # Per-mode per-depth slownesses match to within 1 us/ft.
    tol = 1.0 * US_PER_FT
    for s, j in zip(seq, jnt):
        for mode in ("P", "S", "Stoneley"):
            if mode in s.picks and mode in j.picks:
                assert abs(s.picks[mode].slowness - j.picks[mode].slowness) < tol


def test_viterbi_pick_joint_empty_input():
    """Empty input returns an empty list."""
    from fwap.picker import viterbi_pick_joint

    assert viterbi_pick_joint([], depths=np.array([])) == []


def test_viterbi_pick_joint_rejects_length_mismatch():
    """stc_results / depths length mismatch raises ValueError."""
    import pytest

    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=3)
    with pytest.raises(ValueError, match="same length"):
        viterbi_pick_joint(stcs, depths=np.array([0.0, 0.1]))


def test_viterbi_pick_joint_auto_fallback_on_tight_budget():
    """An unreasonably loose threshold + tight max_triples_per_depth
    used to raise ValueError; now the auto-fallback variable-candidate-
    budget tightens per-mode top-K to fit the budget and the call
    succeeds. Confirms graceful degradation rather than a hard raise."""
    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=2, seed=17)
    # max_triples_per_depth=1 forces auto_K = 0 (every mode reduced
    # to "absent only"), so the trellis collapses to 1 row per
    # depth. Picks remain a list of length n_depth even when every
    # mode is absent.
    picks = viterbi_pick_joint(
        stcs,
        depths=np.array([0.0, 0.15]),
        threshold=0.1,
        max_triples_per_depth=1,
    )
    assert len(picks) == 2


def test_viterbi_pick_joint_top_k_keeps_overflow_working():
    """top_k_per_mode bounds the trellis even when threshold is permissive."""
    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=2, seed=17)
    # Without top_k this would blow up (many peaks pass threshold 0.1);
    # with top_k_per_mode=3 we only enumerate 4^3 = 64 triples per
    # depth, well within the default cap.
    picks = viterbi_pick_joint(
        stcs,
        depths=np.array([0.0, 0.15]),
        threshold=0.1,
        top_k_per_mode=3,
    )
    assert len(picks) == 2


def test_viterbi_pick_joint_top_k_agrees_with_full_on_clean_data():
    """On clean data, top_k_per_mode=large has no effect."""
    from fwap.picker import viterbi_pick_joint

    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    stcs = _stc_sequence(n_depth=5, Vp=Vp, Vs=Vs, Vst=Vst, seed=19)
    depths = np.arange(5) * 0.1524
    full = viterbi_pick_joint(stcs, depths=depths)
    trimmed = viterbi_pick_joint(stcs, depths=depths, top_k_per_mode=10)
    for a, b in zip(full, trimmed):
        for mode in ("P", "S", "Stoneley"):
            if mode in a.picks and mode in b.picks:
                assert abs(a.picks[mode].slowness - b.picks[mode].slowness) < 1.0e-12


def test_auto_fallback_k_finds_largest_fitting_K():
    """_auto_fallback_k finds the largest K such that
    prod(min(n_i, K) + 1) <= budget. Spot-check on a few configurations."""
    from fwap.picker import _auto_fallback_k

    # n=[10, 10, 10] (3-mode, each 10 candidates), budget 1000:
    # K=9 -> 10*10*10 = 1000 (fits exactly)
    # K=10 -> 11*11*11 = 1331 (over)
    assert _auto_fallback_k([10, 10, 10], 1000) == 9
    # 4-mode, each 10 candidates, budget 2000:
    # K=5 -> 6^4 = 1296 (fits); K=6 -> 7^4 = 2401 (over)
    assert _auto_fallback_k([10, 10, 10, 10], 2000) == 5
    # Asymmetric n: smaller modes are not over-tightened
    # K=2: min(3,2)+1=3; min(20,2)+1=3; min(20,2)+1=3 -> 27 (fits 100)
    # K=3: min(3,3)+1=4; min(20,3)+1=4; min(20,3)+1=4 -> 64 (fits)
    # K=4: min(3,4)+1=4; min(20,4)+1=5; min(20,4)+1=5 -> 100 (fits exactly)
    # K=5: 4*6*6 = 144 (over)
    assert _auto_fallback_k([3, 20, 20], 100) == 4
    # Edge: empty input
    assert _auto_fallback_k([], 1000) == 0
    # Edge: budget tight enough that K=0 (only "absent")
    # K=0: prod(1) = 1 fits any budget >= 1
    assert _auto_fallback_k([10, 10, 10], 1) == 0


def test_viterbi_posterior_marginals_returns_valid_probabilities():
    """Each depth/mode posterior sums (over candidates + absent) to 1."""
    from fwap.picker import viterbi_posterior_marginals

    stcs = _stc_sequence(n_depth=4, seed=23)
    depths = np.arange(4) * 0.1524
    map_picks, posteriors = viterbi_posterior_marginals(
        stcs, depths=depths, threshold=0.4
    )
    assert len(map_picks) == 4
    assert len(posteriors) == 4
    for d_post in posteriors:
        for mode in ("P", "S", "Stoneley"):
            entry = d_post[mode]
            total = float(entry.probabilities.sum()) + entry.p_absent
            assert abs(total - 1.0) < 1.0e-9
            assert np.all(entry.probabilities >= 0.0)
            assert 0.0 <= entry.p_absent <= 1.0


def test_viterbi_posterior_marginals_agrees_with_map_on_clean_data():
    """The MAP picks from forward-backward match viterbi_pick_joint."""
    from fwap.picker import viterbi_pick_joint, viterbi_posterior_marginals

    stcs = _stc_sequence(n_depth=5, seed=25)
    depths = np.arange(5) * 0.1524
    direct = viterbi_pick_joint(stcs, depths=depths)
    map_picks, _ = viterbi_posterior_marginals(stcs, depths=depths)
    assert len(direct) == len(map_picks)
    for a, b in zip(direct, map_picks):
        for mode in ("P", "S", "Stoneley"):
            if mode in a.picks and mode in b.picks:
                assert abs(a.picks[mode].slowness - b.picks[mode].slowness) < 1.0e-12


def test_viterbi_posterior_marginals_shapes_match_candidates():
    """PosteriorPick arrays line up with the per-mode candidate lists.

    A structural check: for every mode at every depth, the
    ``slownesses`` / ``times`` / ``coherences`` / ``probabilities``
    arrays all have the same length, equal to the number of
    candidates that passed the prior-window + threshold filter.
    (The ``probabilities`` array does NOT include the absent state;
    that's reported separately as ``p_absent``.)
    """
    from fwap.picker import viterbi_posterior_marginals

    stcs = _stc_sequence(n_depth=3, seed=29)
    depths = np.arange(3) * 0.1524
    _, posteriors = viterbi_posterior_marginals(stcs, depths=depths, threshold=0.4)
    for d_post in posteriors:
        for entry in d_post.values():
            assert (
                entry.slownesses.shape
                == entry.times.shape
                == entry.coherences.shape
                == entry.probabilities.shape
            )


def test_viterbi_posterior_marginals_empty_input():
    """Empty input returns two empty lists."""
    from fwap.picker import viterbi_posterior_marginals

    map_picks, posteriors = viterbi_posterior_marginals([], depths=np.array([]))
    assert map_picks == []
    assert posteriors == []


def test_viterbi_posterior_marginals_rejects_length_mismatch():
    """stc_results / depths length mismatch raises ValueError."""
    import pytest

    from fwap.picker import viterbi_posterior_marginals

    stcs = _stc_sequence(n_depth=3)
    with pytest.raises(ValueError, match="same length"):
        viterbi_posterior_marginals(stcs, depths=np.array([0.0, 0.1]))


def test_viterbi_pick_joint_soft_time_order_allows_violation():
    """With soft_time_order set, triples that violate the order are
    kept with a penalty rather than filtered out.

    We don't try to induce a real S-before-P synthetic (hard); we
    only verify that ``soft_time_order=None`` (strict, default) and
    ``soft_time_order=very_large`` produce the same results on a
    clean sweep where no violations happen, and that an extremely
    small ``soft_time_order`` doesn't crash.
    """
    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=4, seed=21)
    depths = np.arange(4) * 0.1524

    strict = viterbi_pick_joint(stcs, depths=depths, soft_time_order=None)
    soft_inf = viterbi_pick_joint(stcs, depths=depths, soft_time_order=1.0e12)
    # Very large penalty converges to the strict constraint.
    for a, b in zip(strict, soft_inf):
        for mode in ("P", "S", "Stoneley"):
            if mode in a.picks and mode in b.picks:
                assert abs(a.picks[mode].slowness - b.picks[mode].slowness) < 1.0e-10

    # A tiny penalty allows any ordering; the call should succeed.
    loose = viterbi_pick_joint(stcs, depths=depths, soft_time_order=1.0e-6)
    assert len(loose) == 4


def test_viterbi_pick_continuity_penalty_smooths_slowness():
    """Adjacent-depth slowness jumps are smaller than with pick_modes alone.

    On a clean sweep the Viterbi path should never make an abrupt
    slowness jump: the per-depth slowness should vary within the same
    narrow band the underlying truth varies in (effectively zero here).
    """
    from fwap.picker import viterbi_pick

    Vs = 2500.0
    stcs = _stc_sequence(n_depth=10, Vs=Vs, seed=7)
    depths = np.arange(10) * 0.1524
    picks = viterbi_pick(stcs, depths=depths, slow_jump_sigma=10 * US_PER_FT)
    s_series = np.array([dp.picks["S"].slowness for dp in picks if "S" in dp.picks])
    if s_series.size > 1:
        max_jump = np.max(np.abs(np.diff(s_series)))
        assert max_jump < 5.0 * US_PER_FT, (
            f"max adjacent-depth S slowness jump {max_jump / US_PER_FT:.2f} "
            f"us/ft -- Viterbi continuity penalty not doing its job"
        )


# ---------------------------------------------------------------------
# Pseudo-Rayleigh / guided mode
# ---------------------------------------------------------------------


def _make_stc_with_pseudo_rayleigh(
    seed=0, Vp=4500.0, Vs=2500.0, Vst=1400.0, v_fluid=1500.0
):
    """STC of a 4-mode gather: P + S + pseudo-Rayleigh + Stoneley."""
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    modes = monopole_formation_modes(
        Vp, Vs, Vst, v_fluid=v_fluid, f_pr=8_000.0, pr_amp=2.0
    )
    data = synthesize_gather(geom, modes, noise=0.05, seed=seed)
    return stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )


def test_pick_modes_recovers_pseudo_rayleigh_when_planted():
    """When a pseudo-Rayleigh arrival is planted, pick_modes finds it."""
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    picks = pick_modes(
        _make_stc_with_pseudo_rayleigh(Vp=Vp, Vs=Vs, Vst=Vst), threshold=0.4
    )
    assert "PseudoRayleigh" in picks
    pr = picks["PseudoRayleigh"]
    # Pseudo-Rayleigh phase slowness lives between the formation
    # shear slowness (low-f cutoff) and the borehole-fluid slowness
    # (high-f asymptote). For Vs=2500 m/s, v_fluid=1500 m/s that's
    # roughly 122-200 us/ft.
    assert 1.0 / Vs <= pr.slowness <= 1.0 / 1500.0 + 5e-6


def test_pick_modes_pseudo_rayleigh_absent_on_3_mode_gather():
    """The 3-mode default gather has no peak in the PseudoRayleigh window."""
    picks = pick_modes(_make_stc(), threshold=0.4)
    assert "PseudoRayleigh" not in picks
    # P / S / Stoneley still recovered.
    assert {"P", "S", "Stoneley"} <= set(picks)


def test_pseudo_rayleigh_dispersion_rejects_slow_formations():
    """vs <= v_fluid: pseudo-Rayleigh has no cutoff, factory must raise."""
    from fwap.synthetic import pseudo_rayleigh_dispersion

    with pytest.raises(ValueError, match="fast formation"):
        pseudo_rayleigh_dispersion(vs=1200.0, v_fluid=1500.0)


def test_pseudo_rayleigh_dispersion_endpoints():
    """At f=0 phase slowness equals 1/vs; at f -> inf it asymptotes to 1/v_fluid."""
    from fwap.synthetic import pseudo_rayleigh_dispersion

    Vs = 2500.0
    Vf = 1500.0
    s_of_f = pseudo_rayleigh_dispersion(vs=Vs, v_fluid=Vf, a_borehole=0.1)
    f = np.array([0.0, 1.0e8])
    s = s_of_f(f)
    assert abs(s[0] - 1.0 / Vs) < 1e-12
    # The Lorentzian shape ensures we get within 0.1% of 1/v_fluid by
    # 1e8 Hz (well above any physical sonic frequency).
    assert abs(s[1] - 1.0 / Vf) / (1.0 / Vf) < 1.0e-3


def test_viterbi_pick_joint_accepts_4_mode_priors():
    """Joint Viterbi is now N-mode generic; passing the full
    DEFAULT_PRIORS (4 modes including PseudoRayleigh) no longer raises.
    Confirms the relaxation of the old (P, S, Stoneley)-only hardcoding."""
    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=3)
    depths = np.arange(3) * 0.1524
    picks = viterbi_pick_joint(stcs, depths=depths, priors=DEFAULT_PRIORS)
    assert len(picks) == 3
    # Picks may include any subset of the 4 modes (including
    # PseudoRayleigh) depending on which peaks survive the priors.
    for dp in picks:
        assert set(dp.picks) <= set(DEFAULT_PRIORS)


def test_viterbi_pick_joint_default_priors_now_4_modes():
    """The default-priors path now uses the full DEFAULT_PRIORS
    (4 modes). Picks are drawn from any of {P, S, PseudoRayleigh,
    Stoneley}, not just the (P, S, Stoneley) subset."""
    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=4)
    depths = np.arange(4) * 0.1524
    picks = viterbi_pick_joint(stcs, depths=depths, threshold=0.4)
    assert len(picks) == 4
    for dp in picks:
        assert set(dp.picks) <= set(DEFAULT_PRIORS)


def test_viterbi_pick_joint_subset_priors_still_work():
    """Explicitly passing a 3-mode subset preserves the old behavior;
    PseudoRayleigh stays out of the trellis."""
    from fwap.picker import viterbi_pick_joint

    subset = {m: DEFAULT_PRIORS[m] for m in ("P", "S", "Stoneley")}
    stcs = _stc_sequence(n_depth=4)
    depths = np.arange(4) * 0.1524
    picks = viterbi_pick_joint(stcs, depths=depths, priors=subset, threshold=0.4)
    for dp in picks:
        assert set(dp.picks) <= {"P", "S", "Stoneley"}


def test_viterbi_pick_joint_rejects_empty_priors():
    """Empty priors dict raises ValueError -- nothing to pick."""
    from fwap.picker import viterbi_pick_joint

    stcs = _stc_sequence(n_depth=2)
    depths = np.arange(2) * 0.1524
    with pytest.raises(ValueError, match="at least one mode"):
        viterbi_pick_joint(stcs, depths=depths, priors={})


# ---------------------------------------------------------------------
# Wavelet-shape + onset-polarity expert rules
# ---------------------------------------------------------------------


def test_onset_polarity_basic_signs():
    """Sign of the largest-absolute sample wins."""
    from fwap.picker import onset_polarity

    assert onset_polarity(np.array([0.0, 0.5, 1.0, 0.5, 0.0])) == +1
    assert onset_polarity(np.array([0.0, -0.5, -1.0, -0.5, 0.0])) == -1
    assert onset_polarity(np.zeros(8)) == 0
    # Tie-breaks toward whichever extremum np.argmax picks first.
    assert onset_polarity(np.array([])) == 0


def test_wavelet_shape_score_perfect_match_is_one():
    """A pure Ricker at f0 correlates 1.0 with itself."""
    from fwap.picker import wavelet_shape_score

    f0 = 8_000.0
    dt = 1.0e-5
    n = 64
    # Build a centered Ricker.
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f0 * t) ** 2
    ricker = (1.0 - 2.0 * a) * np.exp(-a)
    score = wavelet_shape_score(ricker, dt, f0)
    assert score > 0.999


def test_wavelet_shape_score_orthogonal_pulse_is_low():
    """A pulse at a very different frequency correlates poorly."""
    from fwap.picker import wavelet_shape_score

    dt = 1.0e-5
    n = 64
    t = (np.arange(n) - n // 2) * dt
    # Ricker at 1 kHz vs template at 15 kHz: very different shapes.
    a = (np.pi * 1_000.0 * t) ** 2
    low_f_ricker = (1.0 - 2.0 * a) * np.exp(-a)
    score = wavelet_shape_score(low_f_ricker, dt, 15_000.0)
    assert score < 0.5


def test_wavelet_shape_score_polarity_blind():
    """The score uses |corr|, so a flipped wavelet still matches."""
    from fwap.picker import wavelet_shape_score

    dt = 1.0e-5
    n = 64
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * 8_000.0 * t) ** 2
    ricker = (1.0 - 2.0 * a) * np.exp(-a)
    score_pos = wavelet_shape_score(ricker, dt, 8_000.0)
    score_neg = wavelet_shape_score(-ricker, dt, 8_000.0)
    assert abs(score_pos - score_neg) < 1.0e-12


def test_filter_picks_by_shape_default_priors_is_passthrough():
    """With ``polarity=0`` and ``shape_match_min=0`` (the defaults),
    the filter is a no-op and returns every input pick unchanged."""
    from fwap.picker import filter_picks_by_shape

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    out = filter_picks_by_shape(picks, data, geom.dt, geom.offsets)
    assert set(out) == set(picks)
    for name in picks:
        assert out[name] is picks[name]  # not mutated, same object


def test_filter_picks_by_shape_polarity_drop():
    """Setting ``polarity=-1`` on the canonical positive-Ricker P
    drops the P pick (and only the P pick)."""
    from fwap.picker import DEFAULT_PRIORS, filter_picks_by_shape

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    priors = dict(DEFAULT_PRIORS)
    priors["P"] = dict(DEFAULT_PRIORS["P"], polarity=-1)
    out = filter_picks_by_shape(picks, data, geom.dt, geom.offsets, priors=priors)
    assert "P" not in out
    assert {"S", "Stoneley"} <= set(out)


def test_filter_picks_by_shape_shape_drop_when_min_too_high():
    """A shape_match_min of 0.999 against the Stoneley f0 drops Stoneley
    (Gabor has the same envelope but isn't an exact Ricker)."""
    from fwap.picker import DEFAULT_PRIORS, filter_picks_by_shape

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    priors = dict(DEFAULT_PRIORS)
    priors["Stoneley"] = dict(
        DEFAULT_PRIORS["Stoneley"], shape_match_min=0.999, f0=3_000.0
    )
    out = filter_picks_by_shape(picks, data, geom.dt, geom.offsets, priors=priors)
    assert "Stoneley" not in out


def test_filter_picks_by_shape_requires_f0_when_shape_enabled():
    """shape_match_min > 0 without an f0 in the prior is a
    configuration error and must raise."""
    import pytest

    from fwap.picker import DEFAULT_PRIORS, filter_picks_by_shape

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    priors = dict(DEFAULT_PRIORS)
    priors["P"] = dict(DEFAULT_PRIORS["P"], shape_match_min=0.5)
    with pytest.raises(ValueError, match="f0"):
        filter_picks_by_shape(picks, data, geom.dt, geom.offsets, priors=priors)


def test_filter_track_by_shape_runs_per_depth():
    """Multi-depth filter applies per-depth using the matching gather."""
    from fwap.picker import (
        DEFAULT_PRIORS,
        filter_track_by_shape,
        viterbi_pick,
    )

    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    n = 4
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    datas = [
        synthesize_gather(
            geom, monopole_formation_modes(Vp, Vs, Vst), noise=0.05, seed=s
        )
        for s in range(n)
    ]
    stcs = [_stc_from_gather(geom, d) for d in datas]
    depths = np.arange(n) * 0.1524
    track = viterbi_pick(stcs, depths=depths, threshold=0.4)

    # Polarity=-1 on P should drop P at every depth.
    priors = dict(DEFAULT_PRIORS)
    priors["P"] = dict(DEFAULT_PRIORS["P"], polarity=-1)
    out = filter_track_by_shape(track, datas, geom.dt, geom.offsets, priors=priors)
    assert len(out) == n
    for dp in out:
        assert "P" not in dp.picks
        assert {"S", "Stoneley"} <= set(dp.picks)


def test_filter_track_by_shape_rejects_length_mismatch():
    """track_picks and datas must have matching length."""
    import pytest

    from fwap.picker import filter_track_by_shape

    geom, data, *_ = _gather()
    track = []  # empty
    datas = [data]
    with pytest.raises(ValueError, match="same length"):
        filter_track_by_shape(track, datas, geom.dt, geom.offsets)


def _gather(seed=0):
    """One canonical 3-mode monopole gather + its geometry."""
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    data = synthesize_gather(
        geom, monopole_formation_modes(4500.0, 2500.0, 1400.0), noise=0.05, seed=seed
    )
    return geom, data, 4500.0, 2500.0, 1400.0


def _stc_from_gather(geom, data):
    """Run STC with the canonical settings used elsewhere in this file."""
    from fwap.coherence import stc as _stc

    return _stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )


# ---------------------------------------------------------------------
# Cross-mode consistency QC
# ---------------------------------------------------------------------


def test_quality_control_picks_clean_passes():
    """Canonical Vp=4500 / Vs=2500 / Vst=1400 picks (Vp/Vs=1.8) pass
    every check."""
    from fwap.picker import PickQualityFlags, quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    qc = quality_control_picks(picks, depth=1000.0)
    assert isinstance(qc, PickQualityFlags)
    assert qc.depth == 1000.0
    assert qc.vp_vs is not None
    assert 1.7 < qc.vp_vs < 1.9
    assert qc.vp_vs_in_band
    assert qc.time_order_ok
    assert not qc.flagged
    assert qc.reasons == ()


def test_quality_control_picks_vp_vs_out_of_band_flags():
    """An S pick that drives Vp/Vs above the 2.6 cap is flagged."""
    from fwap.picker import ModePick, quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    # Make S 1.6x slower to push Vp/Vs to ~2.8 (well above the 2.6 cap).
    picks["S"] = ModePick(
        name="S",
        slowness=picks["S"].slowness * 1.6,
        time=picks["S"].time,
        coherence=0.5,
        amplitude=0.1,
    )
    qc = quality_control_picks(picks)
    assert not qc.vp_vs_in_band
    assert qc.flagged
    assert any("Vp/Vs" in r for r in qc.reasons)


def test_quality_control_picks_time_order_flags():
    """Picks whose times violate the canonical t_P <= t_S <= ... order
    are flagged."""
    from fwap.picker import ModePick, quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    # Force S to arrive before P.
    picks["S"] = ModePick(
        name="S",
        slowness=picks["S"].slowness,
        time=picks["P"].time - 1.0e-3,
        coherence=picks["S"].coherence,
        amplitude=picks["S"].amplitude,
    )
    qc = quality_control_picks(picks)
    assert not qc.time_order_ok
    assert qc.flagged
    assert any("time order" in r for r in qc.reasons)


def test_quality_control_picks_skips_vp_vs_when_either_missing():
    """No Vp/Vs computed when only one of P/S is picked."""
    from fwap.picker import quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    del picks["P"]
    qc = quality_control_picks(picks)
    assert qc.vp_vs is None
    # vp_vs_in_band stays True (no check applied), so flagged depends
    # only on the time-order check (which still passes).
    assert qc.vp_vs_in_band
    assert not qc.flagged


def test_quality_control_picks_accepts_depth_picks():
    """Passing a DepthPicks uses its .depth field by default."""
    from fwap.picker import DepthPicks, quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    dp = DepthPicks(depth=2025.5, picks=picks)
    qc = quality_control_picks(dp)
    assert qc.depth == 2025.5
    # Explicit depth kwarg overrides.
    qc2 = quality_control_picks(dp, depth=999.0)
    assert qc2.depth == 999.0


def test_quality_control_picks_require_time_order_disable():
    """``require_time_order=False`` skips the ordering check."""
    from fwap.picker import ModePick, quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    picks["S"] = ModePick(
        name="S",
        slowness=picks["S"].slowness,
        time=picks["P"].time - 1.0e-3,  # violates ordering
        coherence=picks["S"].coherence,
    )
    qc = quality_control_picks(picks, require_time_order=False)
    assert qc.time_order_ok  # not actually checked
    # And so the only remaining gate is the Vp/Vs one (which passes).
    assert not qc.flagged


def test_quality_control_picks_reports_both_failures_simultaneously():
    """A pick set that fails both gates lists both reasons."""
    from fwap.picker import ModePick, quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    # Vp/Vs violation AND time-order violation in one go.
    picks["S"] = ModePick(
        name="S",
        slowness=picks["S"].slowness * 1.6,
        time=picks["P"].time - 1.0e-3,
        coherence=0.5,
    )
    qc = quality_control_picks(picks)
    assert qc.flagged
    assert not qc.vp_vs_in_band
    assert not qc.time_order_ok
    assert len(qc.reasons) == 2


def test_quality_control_track_runs_per_depth():
    """quality_control_track returns one PickQualityFlags per depth."""
    from fwap.picker import (
        DepthPicks,
        ModePick,
        PickQualityFlags,
        quality_control_track,
    )

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    bad_picks = dict(picks)
    bad_picks["S"] = ModePick(
        name="S",
        slowness=picks["S"].slowness * 1.6,  # Vp/Vs violation
        time=picks["S"].time,
        coherence=0.5,
    )
    track = [
        DepthPicks(depth=1000.0, picks=picks),
        DepthPicks(depth=1001.0, picks=bad_picks),
        DepthPicks(depth=1002.0, picks=picks),
    ]
    qc = quality_control_track(track)
    assert len(qc) == 3
    assert all(isinstance(q, PickQualityFlags) for q in qc)
    assert [q.flagged for q in qc] == [False, True, False]
    assert [q.depth for q in qc] == [1000.0, 1001.0, 1002.0]


# ---------------------------------------------------------------------
# Thomsen-gamma band gate (Tier 1 VTI QC)
# ---------------------------------------------------------------------


def test_quality_control_picks_skips_gamma_when_not_supplied():
    """Default (gamma=None) skips the gate; gamma_in_band stays True
    and no gamma reason is appended."""
    import pytest

    from fwap.picker import quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    qc = quality_control_picks(picks, depth=1000.0)
    assert qc.gamma is None
    assert qc.gamma_in_band is True
    assert all("gamma" not in r.lower() for r in qc.reasons)


def test_quality_control_picks_gamma_in_band_passes():
    """A typical-shale gamma value (0.15) passes the default band."""
    from fwap.picker import quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    qc = quality_control_picks(picks, depth=1000.0, gamma=0.15)
    assert qc.gamma == 0.15
    assert qc.gamma_in_band is True
    assert qc.flagged is False
    assert all("gamma" not in r.lower() for r in qc.reasons)


def test_quality_control_picks_gamma_above_band_flags():
    """gamma > gamma_max (default 0.50) flags with a descriptive reason."""
    from fwap.picker import quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    qc = quality_control_picks(picks, depth=1000.0, gamma=0.75)
    assert qc.gamma == 0.75
    assert qc.gamma_in_band is False
    assert qc.flagged is True
    assert any("Thomsen gamma" in r and "0.75" in r for r in qc.reasons)


def test_quality_control_picks_gamma_below_band_flags():
    """gamma < gamma_min (default -0.05) flags with a descriptive reason."""
    from fwap.picker import quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    qc = quality_control_picks(picks, depth=1000.0, gamma=-0.20)
    assert qc.gamma_in_band is False
    assert qc.flagged is True
    assert any("Thomsen gamma" in r for r in qc.reasons)


def test_quality_control_picks_gamma_isotropic_passes_default_band():
    """gamma == 0 (isotropic carbonate / clean sand) is in the default
    band -- no false positive."""
    from fwap.picker import quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    qc = quality_control_picks(picks, depth=1000.0, gamma=0.0)
    assert qc.gamma_in_band is True
    assert qc.flagged is False


def test_quality_control_picks_gamma_custom_band_for_shale_only():
    """Caller can tighten to the canonical VTI shale window
    [0.05, 0.30] to flag non-shale depths."""
    from fwap.picker import quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    # Sandstone-like gamma 0.02 -- inside the wide default but outside
    # the tight shale band.
    qc_wide = quality_control_picks(picks, depth=1000.0, gamma=0.02)
    qc_shale = quality_control_picks(
        picks, depth=1000.0, gamma=0.02, gamma_min=0.05, gamma_max=0.30
    )
    assert qc_wide.gamma_in_band is True
    assert qc_shale.gamma_in_band is False


def test_quality_control_track_per_depth_gammas():
    """quality_control_track forwards a per-depth gamma array; NaN
    entries skip the gate at that depth only."""
    from fwap.picker import (
        DepthPicks,
        ModePick,
        quality_control_track,
    )

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    track = [
        DepthPicks(depth=1000.0, picks=picks),
        DepthPicks(depth=1001.0, picks=picks),
        DepthPicks(depth=1002.0, picks=picks),
        DepthPicks(depth=1003.0, picks=picks),
    ]
    # In-band, out-of-band, NaN (skip), in-band.
    gammas = np.array([0.15, 0.80, np.nan, 0.20])
    qc = quality_control_track(track, gammas=gammas)
    assert qc[0].gamma_in_band is True and qc[0].flagged is False
    assert qc[1].gamma_in_band is False and qc[1].flagged is True
    assert qc[2].gamma is None and qc[2].gamma_in_band is True
    assert qc[3].gamma_in_band is True and qc[3].flagged is False


def test_quality_control_track_rejects_gammas_length_mismatch():
    """Length mismatch is a caller error."""
    import pytest

    from fwap.picker import DepthPicks, ModePick, quality_control_track

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    track = [DepthPicks(depth=1000.0 + i, picks=picks) for i in range(3)]
    with pytest.raises(ValueError, match="length"):
        quality_control_track(track, gammas=[0.1, 0.2])


def test_quality_control_track_default_gammas_none_skips_gate():
    """Without gammas, no QC entry has the gate active."""
    from fwap.picker import DepthPicks, ModePick, quality_control_track

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    track = [DepthPicks(depth=1000.0 + i, picks=picks) for i in range(3)]
    qc = quality_control_track(track)
    for q in qc:
        assert q.gamma is None
        assert q.gamma_in_band is True


def test_quality_control_picks_gamma_combines_with_other_reasons():
    """A gamma violation appears alongside a Vp/Vs violation in
    the reasons tuple."""
    from fwap.picker import ModePick, quality_control_picks

    geom, data, *_ = _gather()
    picks = pick_modes(_stc_from_gather(geom, data), threshold=0.4)
    bad = dict(picks)
    bad["S"] = ModePick(
        name="S",
        slowness=picks["S"].slowness * 1.6,
        time=picks["S"].time,
        coherence=0.5,
    )
    qc = quality_control_picks(bad, depth=1000.0, gamma=0.80)
    assert qc.flagged is True
    assert qc.vp_vs_in_band is False
    assert qc.gamma_in_band is False
    # Both reasons present.
    assert any("Vp/Vs" in r for r in qc.reasons)
    assert any("Thomsen gamma" in r for r in qc.reasons)


# ---------------------------------------------------------------------
# track_to_log_curves: picker -> LAS/DLIS bridge
# ---------------------------------------------------------------------


def _three_depth_track() -> list:
    """Hand-built three-depth, three-mode track for bridge tests."""
    from fwap.picker import DepthPicks, ModePick

    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    return [
        DepthPicks(
            depth=1000.0,
            picks={
                "P": ModePick("P", 1.0 / Vp, 2.0e-3, 0.92, amplitude=0.7),
                "S": ModePick("S", 1.0 / Vs, 3.0e-3, 0.85, amplitude=0.9),
                "Stoneley": ModePick(
                    "Stoneley", 1.0 / Vst, 5.0e-3, 0.80, amplitude=1.1
                ),
            },
        ),
        DepthPicks(
            depth=1001.0,
            picks={
                "P": ModePick("P", 1.0 / (Vp + 50), 2.05e-3, 0.91, amplitude=0.71),
                # S missing at this depth.
                "Stoneley": ModePick(
                    "Stoneley", 1.0 / Vst, 5.05e-3, 0.79, amplitude=1.05
                ),
            },
        ),
        DepthPicks(
            depth=1002.0,
            picks={
                "P": ModePick("P", 1.0 / (Vp - 30), 2.10e-3, 0.93, amplitude=0.72),
                "S": ModePick("S", 1.0 / (Vs + 20), 3.10e-3, 0.86, amplitude=0.91),
                "Stoneley": ModePick(
                    "Stoneley", 1.0 / Vst, 5.10e-3, 0.81, amplitude=1.10
                ),
            },
        ),
    ]


def test_track_to_log_curves_basic_mnemonics_and_shapes():
    """Default call emits DT/COH/AMP per mode plus VPVS, all (n_depth,)."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    depths, curves = track_to_log_curves(track)
    assert depths.shape == (3,)
    np.testing.assert_array_equal(depths, [1000.0, 1001.0, 1002.0])
    expected = {
        "DTP",
        "DTS",
        "DTST",
        "COHP",
        "COHS",
        "COHST",
        "AMPP",
        "AMPS",
        "AMPST",
        "VPVS",
    }
    assert set(curves) == expected
    for arr in curves.values():
        assert arr.shape == (3,)


def test_track_to_log_curves_slowness_unit_is_us_per_ft():
    """DT* columns carry slowness in us/ft (LAS/DLIS unit)."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(track)
    Vp = 4500.0
    expected_dtp_d0 = (1.0 / Vp) / US_PER_FT
    np.testing.assert_allclose(curves["DTP"][0], expected_dtp_d0, rtol=1e-12)


def test_track_to_log_curves_missing_picks_become_nan():
    """A mode missing at a depth fills that depth with NaN."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(track)
    # S is absent at depth index 1.
    assert np.isnan(curves["DTS"][1])
    assert np.isnan(curves["COHS"][1])
    assert np.isnan(curves["AMPS"][1])
    # ...and present at indices 0 and 2.
    assert np.all(np.isfinite(curves["DTS"][[0, 2]]))


def test_track_to_log_curves_vpvs_propagates_nan_when_p_or_s_missing():
    """VPVS requires both P and S; missing either yields NaN at that depth."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(track)
    assert "VPVS" in curves
    # Depth 1 has no S -> VPVS is NaN there.
    assert np.isnan(curves["VPVS"][1])
    # At depth 0, VPVS = s_S / s_P = Vp / Vs.
    np.testing.assert_allclose(curves["VPVS"][0], 4500.0 / 2500.0, rtol=1e-12)


def test_track_to_log_curves_skips_amplitude_when_all_none():
    """AMP* columns are dropped for modes whose picks all have amplitude=None."""
    from fwap.picker import DepthPicks, ModePick, track_to_log_curves

    track = [
        DepthPicks(
            depth=1000.0,
            picks={
                "P": ModePick("P", 2.0e-4, 1.0e-3, 0.9, amplitude=None),
            },
        ),
        DepthPicks(
            depth=1001.0,
            picks={
                "P": ModePick("P", 2.0e-4, 1.0e-3, 0.9, amplitude=None),
            },
        ),
    ]
    _, curves = track_to_log_curves(track)
    assert "DTP" in curves
    assert "AMPP" not in curves


def test_track_to_log_curves_include_time_emits_tim_columns():
    """include_time=True adds TIM* columns (seconds)."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(track, include_time=True)
    assert "TIMP" in curves and "TIMS" in curves and "TIMST" in curves
    np.testing.assert_allclose(curves["TIMP"][0], 2.0e-3, rtol=1e-12)


def test_track_to_log_curves_modes_filter():
    """Passing modes=['P'] drops S/Stoneley columns and the VPVS column."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(track, modes=["P"])
    assert "DTP" in curves
    assert "DTS" not in curves and "DTST" not in curves
    assert "VPVS" not in curves


def test_track_to_log_curves_custom_null_value():
    """A numeric sentinel replaces NaN at missing-pick depths."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(track, null_value=-999.25)
    # S missing at depth 1.
    assert curves["DTS"][1] == -999.25
    # VPVS at depth 1 also gets the sentinel via the np.where cleanup.
    assert curves["VPVS"][1] == -999.25


def test_track_to_log_curves_rejects_non_float_null_value():
    """Passing ``None`` (or any non-float) for null_value raises
    cleanly instead of silently producing object-dtype columns."""
    import pytest

    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    with pytest.raises(TypeError):
        track_to_log_curves(track, null_value=None)  # type: ignore[arg-type]


def test_track_to_log_curves_pseudo_rayleigh_uses_pr_suffix():
    """Mode 'PseudoRayleigh' maps to the DTPR / COHPR / AMPPR family."""
    from fwap.picker import DepthPicks, ModePick, track_to_log_curves

    track = [
        DepthPicks(
            depth=1000.0,
            picks={
                "PseudoRayleigh": ModePick(
                    "PseudoRayleigh", 1.5e-4, 4.0e-3, 0.7, amplitude=0.5
                ),
            },
        )
    ]
    _, curves = track_to_log_curves(track)
    assert "DTPR" in curves
    assert "COHPR" in curves
    assert "AMPPR" in curves


def test_track_to_log_curves_unknown_mode_falls_back_to_uppercase_suffix():
    """A non-canonical mode name uses ``mode.upper()`` as the suffix."""
    from fwap.picker import DepthPicks, ModePick, track_to_log_curves

    track = [
        DepthPicks(
            depth=1000.0,
            picks={
                "leaky": ModePick("leaky", 2.0e-4, 1.0e-3, 0.6),
            },
        )
    ]
    _, curves = track_to_log_curves(track)
    assert "DTLEAKY" in curves


def test_track_to_log_curves_empty_track():
    """An empty track yields an empty depth axis and an empty curves dict."""
    from fwap.picker import track_to_log_curves

    depths, curves = track_to_log_curves([])
    assert depths.shape == (0,)
    assert curves == {}


def test_track_to_log_curves_round_trips_through_write_las(tmp_path):
    """track_to_log_curves -> write_las -> read_las preserves columns."""
    from fwap.io import read_las, write_las
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    depths, curves = track_to_log_curves(track)
    path = str(tmp_path / "track.las")
    write_las(path, depths, curves, well_name="TEST")
    loaded = read_las(path)
    np.testing.assert_allclose(loaded.depth, depths, rtol=0, atol=1e-9)
    for name, arr in curves.items():
        assert name in loaded.curves
        # LAS writes ASCII with finite precision; the units round-trip
        # exactly.
        mask = np.isfinite(arr) & np.isfinite(loaded.curves[name])
        np.testing.assert_allclose(
            loaded.curves[name][mask], arr[mask], rtol=0, atol=1e-3
        )
    # Slowness columns carry the us/ft unit from _FWAP_UNITS.
    assert loaded.units["DTP"] == "us/ft"
    assert loaded.units["DTS"] == "us/ft"
    assert loaded.units["DTST"] == "us/ft"


# ---------------------------------------------------------------------
# include_vti=True path: C33 / C44 / C66 / GAMMA / VP / VSV / VSH
# ---------------------------------------------------------------------


def test_track_to_log_curves_include_vti_off_by_default():
    """Default (include_vti=False) emits no VTI columns."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(track)
    for mnemonic in ("C33", "C44", "C66", "GAMMA", "VP", "VSV", "VSH"):
        assert mnemonic not in curves


def test_track_to_log_curves_include_vti_emits_seven_columns():
    """Default-on path: emits C33/C44/C66/GAMMA/VP/VSV/VSH."""
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    _, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=2400.0,
        rho_fluid=1000.0,
        v_fluid=1500.0,
    )
    for mnemonic in ("C33", "C44", "C66", "GAMMA", "VP", "VSV", "VSH"):
        assert mnemonic in curves
        assert curves[mnemonic].shape == (3,)


def test_track_to_log_curves_include_vti_planted_values():
    """Plant a uniform formation; recovered VTI columns match the
    closed-form analytical values cell by cell."""
    import pytest

    from fwap.picker import DepthPicks, ModePick, track_to_log_curves

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    track = [
        DepthPicks(
            depth=1000.0,
            picks={
                "P": ModePick("P", 1.0 / Vp, 2.0e-3, 0.92, amplitude=0.7),
                "S": ModePick("S", 1.0 / Vs, 3.0e-3, 0.85, amplitude=0.9),
                "Stoneley": ModePick(
                    "Stoneley", 1.0 / Vst, 5.0e-3, 0.80, amplitude=1.1
                ),
            },
        ),
    ]
    _, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    # C33, C44 follow rho * V^2 exactly.
    assert curves["C33"][0] == pytest.approx(rho * Vp**2, rel=1.0e-12)
    assert curves["C44"][0] == pytest.approx(rho * Vs**2, rel=1.0e-12)
    # C66 via the corrected (Tang & Cheng) inversion since P is picked.
    s_st = 1.0 / Vst
    factor = 1.0 - rho_f * v_f**2 / (rho * Vp**2)
    c66_expected = (rho_f / (s_st**2 - 1.0 / v_f**2)) / factor
    assert curves["C66"][0] == pytest.approx(c66_expected, rel=1.0e-12)
    # Velocities follow sqrt(C / rho).
    np.testing.assert_allclose(
        curves["VP"][0], np.sqrt(curves["C33"][0] / rho), rtol=1.0e-12
    )
    np.testing.assert_allclose(
        curves["VSV"][0], np.sqrt(curves["C44"][0] / rho), rtol=1.0e-12
    )
    np.testing.assert_allclose(
        curves["VSH"][0], np.sqrt(curves["C66"][0] / rho), rtol=1.0e-12
    )
    # GAMMA matches (C66 - C44) / (2 C44).
    np.testing.assert_allclose(
        curves["GAMMA"][0],
        (curves["C66"][0] - curves["C44"][0]) / (2.0 * curves["C44"][0]),
        rtol=1.0e-12,
    )


def test_track_to_log_curves_include_vti_per_pick_nan_propagation():
    """Cells where the underlying picks are missing become NaN."""
    from fwap.picker import DepthPicks, ModePick, track_to_log_curves

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    track = [
        # Depth 0: full picks -> all VTI cells finite.
        DepthPicks(
            depth=1000.0,
            picks={
                "P": ModePick("P", 1.0 / Vp, 2.0e-3, 0.92),
                "S": ModePick("S", 1.0 / Vs, 3.0e-3, 0.85),
                "Stoneley": ModePick("Stoneley", 1.0 / Vst, 5.0e-3, 0.80),
            },
        ),
        # Depth 1: P missing -> C33/VP NaN; C66 falls back to White.
        DepthPicks(
            depth=1001.0,
            picks={
                "S": ModePick("S", 1.0 / Vs, 3.0e-3, 0.85),
                "Stoneley": ModePick("Stoneley", 1.0 / Vst, 5.0e-3, 0.80),
            },
        ),
        # Depth 2: S missing -> C44/VSV/GAMMA NaN.
        DepthPicks(
            depth=1002.0,
            picks={
                "P": ModePick("P", 1.0 / Vp, 2.0e-3, 0.92),
                "Stoneley": ModePick("Stoneley", 1.0 / Vst, 5.0e-3, 0.80),
            },
        ),
        # Depth 3: Stoneley missing -> C66/VSH/GAMMA NaN.
        DepthPicks(
            depth=1003.0,
            picks={
                "P": ModePick("P", 1.0 / Vp, 2.0e-3, 0.92),
                "S": ModePick("S", 1.0 / Vs, 3.0e-3, 0.85),
            },
        ),
    ]
    _, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    # Depth 0: all finite.
    for mn in ("C33", "C44", "C66", "GAMMA", "VP", "VSV", "VSH"):
        assert np.isfinite(curves[mn][0])
    # Depth 1: P missing.
    assert np.isnan(curves["C33"][1])
    assert np.isnan(curves["VP"][1])
    # C66 is finite (White-fallback), C44 finite.
    assert np.isfinite(curves["C44"][1])
    assert np.isfinite(curves["C66"][1])
    assert np.isfinite(curves["GAMMA"][1])
    # Depth 2: S missing.
    assert np.isnan(curves["C44"][2])
    assert np.isnan(curves["VSV"][2])
    assert np.isnan(curves["GAMMA"][2])
    # Depth 3: Stoneley missing.
    assert np.isnan(curves["C66"][3])
    assert np.isnan(curves["VSH"][3])
    assert np.isnan(curves["GAMMA"][3])


def test_track_to_log_curves_include_vti_white_fallback_on_missing_p():
    """At depths where P is missing, C66 uses the White (1983) reading
    transparently -- the operational benefit of the per-depth
    fall-back."""
    import pytest

    from fwap.picker import DepthPicks, ModePick, track_to_log_curves

    rho_f, v_f = 1000.0, 1500.0
    rho = 2400.0
    Vs, Vst = 2500.0, 1400.0
    track = [
        DepthPicks(
            depth=1000.0,
            picks={
                "S": ModePick("S", 1.0 / Vs, 3.0e-3, 0.85),
                "Stoneley": ModePick("Stoneley", 1.0 / Vst, 5.0e-3, 0.80),
            },
        ),
    ]
    _, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
    )
    # White expected: C66 = rho_f / (s_ST^2 - 1/V_f^2).
    s_st = 1.0 / Vst
    c66_white = rho_f / (s_st**2 - 1.0 / v_f**2)
    assert curves["C66"][0] == pytest.approx(c66_white, rel=1.0e-12)


def test_track_to_log_curves_include_vti_correct_for_p_modulus_false():
    """correct_for_p_modulus=False forces White everywhere even when
    P is picked."""
    import pytest

    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    rho_f, v_f, rho = 1000.0, 1500.0, 2400.0
    _, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=rho,
        rho_fluid=rho_f,
        v_fluid=v_f,
        correct_for_p_modulus=False,
    )
    # Hand-derive White C66 at depth 0 (Stoneley slowness 1/1400).
    s_st = 1.0 / 1400.0
    c66_white = rho_f / (s_st**2 - 1.0 / v_f**2)
    assert curves["C66"][0] == pytest.approx(c66_white, rel=1.0e-12)


def test_track_to_log_curves_include_vti_per_depth_density():
    """rho can be a length-n_depth ndarray; each cell uses its own."""
    import pytest

    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    rho_per_depth = np.array([2300.0, 2400.0, 2500.0])
    _, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=rho_per_depth,
        rho_fluid=1000.0,
        v_fluid=1500.0,
    )
    Vp = 4500.0  # depth 0's planted Vp
    assert curves["C33"][0] == pytest.approx(2300.0 * Vp**2, rel=1.0e-12)


def test_track_to_log_curves_include_vti_requires_rho_and_fluid():
    """include_vti=True without rho/rho_fluid/v_fluid raises."""
    import pytest

    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    with pytest.raises(ValueError, match="rho"):
        track_to_log_curves(track, include_vti=True, rho_fluid=1000.0, v_fluid=1500.0)
    with pytest.raises(ValueError, match="rho_fluid"):
        track_to_log_curves(track, include_vti=True, rho=2400.0, v_fluid=1500.0)
    with pytest.raises(ValueError, match="rho_fluid"):
        track_to_log_curves(track, include_vti=True, rho=2400.0, rho_fluid=1000.0)


def test_track_to_log_curves_include_vti_rejects_bad_rho_shape():
    """rho must be a scalar or length-n_depth array."""
    import pytest

    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    with pytest.raises(ValueError, match="length-n_depth"):
        track_to_log_curves(
            track,
            include_vti=True,
            rho=np.array([2400.0, 2500.0]),  # wrong length (2 vs 3)
            rho_fluid=1000.0,
            v_fluid=1500.0,
        )


def test_track_to_log_curves_include_vti_round_trips_through_write_las(tmp_path):
    """All seven VTI columns + the standard pick columns round-trip
    through write_las / read_las with the right units."""
    from fwap.io import read_las, write_las
    from fwap.picker import track_to_log_curves

    track = _three_depth_track()
    depths, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=2400.0,
        rho_fluid=1000.0,
        v_fluid=1500.0,
    )
    path = str(tmp_path / "vti_track.las")
    write_las(path, depths, curves, well_name="TRACK_VTI")
    loaded = read_las(path)
    for mn, unit in (
        ("C33", "Pa"),
        ("C44", "Pa"),
        ("C66", "Pa"),
        ("GAMMA", ""),
        ("VP", "m/s"),
        ("VSV", "m/s"),
        ("VSH", "m/s"),
    ):
        assert mn in loaded.curves
        assert loaded.units[mn] == unit


def test_track_to_log_curves_include_vti_with_numeric_null_value():
    """null_value sentinel applies to VTI columns too."""
    from fwap.picker import DepthPicks, ModePick, track_to_log_curves

    Vs = 2500.0  # Stoneley deliberately omitted to test null_value path
    track = [
        # Stoneley missing -> C66/VSH/GAMMA should land on -999.25.
        DepthPicks(
            depth=1000.0,
            picks={
                "S": ModePick("S", 1.0 / Vs, 3.0e-3, 0.85),
            },
        ),
    ]
    _, curves = track_to_log_curves(
        track,
        include_vti=True,
        rho=2400.0,
        rho_fluid=1000.0,
        v_fluid=1500.0,
        null_value=-999.25,
    )
    assert curves["C66"][0] == -999.25
    assert curves["VSH"][0] == -999.25
    assert curves["GAMMA"][0] == -999.25
    # C44 / VSV are finite (S is picked).
    assert curves["C44"][0] != -999.25
    assert curves["VSV"][0] != -999.25
