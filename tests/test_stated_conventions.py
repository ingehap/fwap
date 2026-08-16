"""
Conventions the package states in prose, executed rather than read.

An ordering, a direction or a unit is the class of claim this package
gets wrong most quietly. "``alpha_qP`` is the smaller root", "row 3 is
``sigma_rz``", "index 0 is the fundamental", "converted at 1 Mpsi =
6.8948 GPa" -- each of these can be false while every numerical test
still passes, because the code is self-consistent and only the *label*
is wrong. Nothing in a determinant, a residual, a dispersion comparison
or a monotonicity check touches a label.

Five such claims were found wrong by hand across roadmap A.11-A.13,
and by hand is the whole problem: reading a docstring cannot fail. So
each test here quotes one stated claim, names where it lives, and
executes it. Where a claim has a plausible opposite -- a swap of two
rows, a reversed sort -- the test also checks that the opposite is
*detectably* wrong, since a test that only confirms the code works
does not pin which of two labels belongs to it.

Two of the claims in this file were false when it was written, and
both had already survived a hand fix:

* The substep H.b header in ``_bessel.py`` said ``alpha_qP`` was the
  smaller root, and offered ``p < s`` as the reason. Both halves are
  backwards. A.11 phase 1 corrected this same error in the function
  docstring 200 lines below and reported it fixed "in two places";
  there was a third.
* ``_modal_row3_at_a_n1_vti`` said all three non-fluid columns "scale
  linearly with C66". The qP and qSV columns do. The SH column does
  not, because ``c66`` also sets the SH Christoffel root.

A third was found by the W1 oracle survey, outside the solver: the
Rickman brittleness bounds in ``fwap/geomechanics/indices.py`` write
down a unit conversion and then do not perform it. That one is a
**defect pinned as a defect**, per the house rule in
``plans/learning.md`` -- fixing it moves a shipped output, so it waits
on the paper being confirmed. The test below fails if the constants
move, which is the intended behaviour when someone fixes them.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti
from fwap.cylindrical_solver._vti import (
    _formation_displacements_n1_vti,
    _formation_state_vector_n1_vti,
    _modal_row3_at_a_n1_vti,
    _modal_row4_at_a_n1_vti,
)

# A VTI shale, and the isotropic medium it collapses to.
_RHO = 2500.0
_VP = 3800.0
_VS = 2200.0
_C33 = _RHO * _VP**2
_C44 = _RHO * _VS**2
_VTI = dict(
    c11=_C33 * 1.25,
    c13=0.85 * (_C33 - 2.0 * _C44),
    c33=_C33,
    c44=_C44,
    c66=_C44 * 1.18,
    rho=_RHO,
)
_ISO = dict(
    c11=_C33,
    c13=_C33 - 2.0 * _C44,
    c33=_C33,
    c44=_C44,
    c66=_C44,
    rho=_RHO,
)
_OMEGA = 2.0 * np.pi * 6000.0
_KZ = _OMEGA / 1600.0
_A = 0.10
_FLUID = dict(vf=1500.0, rho_f=1000.0)


# ----------------------------------------------------------------------
# Root ordering: which of the Christoffel pair is qP
# ----------------------------------------------------------------------


def test_alpha_qP_is_the_larger_root_where_three_places_now_say_so():
    """``_bessel.py``, substep H.b header and ``_radial_wavenumbers_vti``.

    "``alpha_qP`` is the LARGER root ... the faster wave decays
    *faster* in ``r``."

    The claim is not decorative. Above ``V_Sv`` the two roots go
    complex conjugate and there is no ordering left to fall back on,
    so whichever rule is written here is the only thing keeping qP
    labelled qP. Ordering on ``Re(alpha)`` instead of ``Re(alpha^2)``
    broke this twice during A.11, each time silently.
    """
    for speed in (1200.0, 1600.0, 2000.0):
        kz = _OMEGA / speed
        alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(kz, _OMEGA, **_VTI)
        assert alpha_qP > alpha_qSV, (speed, alpha_qP, alpha_qSV)


def test_the_isotropic_collapse_puts_alpha_qP_at_p_and_p_above_s():
    """The justification the H.b header offers, not just its conclusion.

    "The isotropic limit is ``alpha_qP -> p > s -> alpha_qSV``, since
    ``V_P > V_S`` makes ``k_z^2 - (omega/V_P)^2`` the larger of the
    two."

    That last inequality is the part the old text had inverted, so it
    is checked as a separate fact from the ordering itself: a correct
    conclusion reached through a wrong reason is one edit away from a
    wrong conclusion.
    """
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(_KZ, _OMEGA, **_ISO)
    p = np.sqrt(_KZ**2 - (_OMEGA / _VP) ** 2)
    s = np.sqrt(_KZ**2 - (_OMEGA / _VS) ** 2)

    assert p > s, (p, s)
    assert alpha_qP == pytest.approx(p, rel=1e-12)
    assert alpha_qSV == pytest.approx(s, rel=1e-12)


# ----------------------------------------------------------------------
# Column and row identity in the VTI formation block
# ----------------------------------------------------------------------


def test_the_sh_column_carries_no_axial_displacement():
    """``_formation_displacements_n1_vti``.

    "Rows are ``(u_r, u_theta, u_z)``; columns are ``(qP, qSV, SH)``"
    and "SH is decoupled and carries ``u = curl(psi z)``, so ``u_z``
    is identically zero for it."

    An exact zero in one corner is the cheapest handle there is on a
    ``(3, 3)`` block's orientation: it pins which index is the SH
    column and which is the ``u_z`` row simultaneously, and a
    transpose or a column permutation moves it.
    """
    block = _formation_displacements_n1_vti(_KZ, _OMEGA, **_VTI, r=_A)

    assert block.shape == (3, 3)
    assert block[2, 2] == 0.0
    # And nowhere else, so the zero identifies a corner rather than
    # being one of several.
    assert np.count_nonzero(block == 0.0) == 1


def test_row3_is_sigma_rtheta_and_row4_is_sigma_rz_under_the_constitutive_law():
    """``_formation_state_vector_n1_vti``, and the reason it gives.

    "the cased block's ``sigma_rz`` row is fed by
    ``_modal_row4_at_a_n1_vti`` and its ``sigma_r_theta`` row by
    ``_modal_row3_at_a_n1_vti``, which is the opposite of what the
    open-hole row numbering suggests. Both were checked against the
    constitutive law, not against the numbering."

    That check is what runs here rather than being taken on trust.
    Each row builder is compared against the stress computed directly
    from the displacement field --

        ``sigma_rz    = c44 (i k_z u_r + du_z/dr)``
        ``sigma_rtheta = c66 (du_theta/dr - u_theta/r + (i n / r) u_r)``

    -- with ``d/dr`` by central difference. A row builder that is the
    stress it claims agrees up to *one* constant across all three
    columns, since the per-column normalisation is shared. The wrong
    pairing does not, and both are asserted: the layered calibration
    that first placed these rows matched values into slots, which
    passes just as happily with the two names exchanged.
    """
    h = 1.0e-6
    here = _formation_displacements_n1_vti(_KZ, _OMEGA, **_VTI, r=_A)
    derivative = (
        _formation_displacements_n1_vti(_KZ, _OMEGA, **_VTI, r=_A + h)
        - _formation_displacements_n1_vti(_KZ, _OMEGA, **_VTI, r=_A - h)
    ) / (2.0 * h)

    u_r, u_theta, _ = here
    _, du_theta, du_z = derivative
    sigma_rz = _VTI["c44"] * (1j * _KZ * u_r + du_z)
    sigma_rtheta = _VTI["c66"] * (du_theta - u_theta / _A + (1j / _A) * u_r)

    kwargs = dict(**_VTI, **_FLUID, a=_A)
    row3 = _modal_row3_at_a_n1_vti(_KZ, _OMEGA, **kwargs)[1:4]
    row4 = _modal_row4_at_a_n1_vti(_KZ, _OMEGA, **kwargs)[1:4]

    def column_spread(row, stress):
        """How far from "one constant across all three columns"."""
        ratio = row / stress
        return float(np.max(np.abs(ratio - ratio[0])) / np.max(np.abs(ratio)))

    assert column_spread(row3, sigma_rtheta) < 1e-8
    assert column_spread(row4, sigma_rz) < 1e-8
    # The exchange the numbering invites is decisively wrong, not
    # marginally so.
    assert column_spread(row3, sigma_rz) > 0.5
    assert column_spread(row4, sigma_rtheta) > 0.5


def test_the_shear_rows_have_an_identically_zero_fluid_column():
    """``_modal_row3_at_a_n1_vti``: "A column is identically zero
    because the fluid carries no shear."

    Identically, not nearly: an exact zero is a structural statement
    about the assembly and is worth holding to `== 0.0`. The same
    holds of row 4, which is the other shear traction.
    """
    kwargs = dict(**_VTI, **_FLUID, a=_A)

    assert _modal_row3_at_a_n1_vti(_KZ, _OMEGA, **kwargs)[0] == 0.0
    assert _modal_row4_at_a_n1_vti(_KZ, _OMEGA, **kwargs)[0] == 0.0


def test_c66_is_an_outer_factor_on_qp_and_qsv_but_not_on_sh():
    """``_modal_row3_at_a_n1_vti``, corrected by this test.

    The docstring said all three non-fluid columns "scale linearly
    with C66". Two of them do. The SH column does not, because
    ``c66`` is not only the outer factor -- it also sets the SH
    Christoffel root, ``alpha_SH^2 = (c44 k_z^2 - rho omega^2) /
    c66``, so doubling it moves that column by 3.03x rather than 2x.

    Kept as an assertion in both directions: the two that scale must
    scale exactly, and the one that does not must be visibly outside
    the claim rather than merely imprecise.
    """
    kwargs = dict(c11=_VTI["c11"], c13=_VTI["c13"], c33=_VTI["c33"])

    def row(c66):
        return _modal_row3_at_a_n1_vti(
            _KZ,
            _OMEGA,
            **kwargs,
            c44=_VTI["c44"],
            c66=c66,
            rho=_RHO,
            **_FLUID,
            a=_A,
        )

    base = row(_VTI["c66"])
    doubled = row(2.0 * _VTI["c66"])
    ratio = doubled[1:4] / base[1:4]

    assert ratio[0] == pytest.approx(2.0, rel=1e-12)  # qP
    assert ratio[1] == pytest.approx(2.0, rel=1e-12)  # qSV
    assert abs(ratio[2] - 2.0) > 0.5, ratio[2]  # SH: 3.03x
    assert ratio[2].real == pytest.approx(3.0325924, rel=1e-5)


def test_the_cased_state_vector_rows_are_in_the_order_documented():
    """``_formation_state_vector_n1_vti``.

    "rows ``(u_r, u_z, u_theta, sigma_rr, sigma_rz, sigma_r_theta)``,
    columns ``(qP, qSV, SH)``" with "per-row factors ``(-1, -i, -i,
    -1, 1, 1)``".

    Note that the state-vector order is **not** the displacement
    helper's: ``u_z`` and ``u_theta`` are exchanged between the two.
    That exchange is the kind of thing that stays invisible while both
    orders are only written down, so the three displacement rows are
    read back through it here, and the swap is checked to be
    detectable rather than a symmetry of the block.
    """
    displacements = _formation_displacements_n1_vti(_KZ, _OMEGA, **_VTI, r=_A)
    block = _formation_state_vector_n1_vti(_KZ, _OMEGA, **_VTI, r=_A, **_FLUID)

    assert block.shape == (6, 3)
    assert np.allclose(block[0], -1.0 * displacements[0])  # u_r
    assert np.allclose(block[1], -1.0j * displacements[2])  # u_z
    assert np.allclose(block[2], -1.0j * displacements[1])  # u_theta
    # The two are genuinely different rows, so the order is a claim
    # and not a convention that happens either way.
    assert not np.allclose(block[1], block[2])


# ----------------------------------------------------------------------
# Direction: which end of an ordered result is index 0
# ----------------------------------------------------------------------


def test_leaky_n0_roots_come_back_fundamental_first():
    """``_enumerate_leaky_roots_n0``.

    "Ordering is by **descending** ``Re(k_z)``, which is ascending
    radial order ... So index 0 is the fundamental, index 1 the first
    overtone."

    Needs a frequency holding more than one root, or the claim is
    vacuous where it is tested: at 30 kHz this configuration has one
    root and any ordering passes. 90 kHz has two.
    """
    from fwap.cylindrical_solver._leaky import _enumerate_leaky_roots_n0

    omega = 2.0 * np.pi * 90000.0
    roots = _enumerate_leaky_roots_n0(
        omega, vp=4500.0, vs=2600.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.10
    )

    assert len(roots) >= 2, len(roots)
    real_parts = [z.real for z in roots]
    assert real_parts == sorted(real_parts, reverse=True), real_parts
    # Descending Re(k_z) is ascending phase velocity, so index 0 is
    # the slowest -- which is what "the fundamental" means here.
    speeds = [omega / z.real for z in roots]
    assert speeds[0] == min(speeds), speeds


def test_the_microannulus_filter_returns_roots_fastest_first():
    """``_microannulus_stable_roots``: "Phase-velocity roots that
    survive a change of scan grid, **fastest first**."

    The filter's existing test uses a determinant with one surviving
    root, which cannot see an ordering. Two well-separated roots can,
    and the direction matters downstream: the caller numbers modes off
    this list, so a reversed sort renames every branch at once without
    changing a single velocity.
    """
    from fwap.cylindrical_solver import _microannulus_stable_roots

    def det(c: float) -> float:
        return (c - 120.0) * (c - 640.0)

    roots = _microannulus_stable_roots(det, 5.0, 1000.0, 240)

    assert len(roots) == 2, roots
    assert roots[0] == pytest.approx(640.0, rel=1e-6)
    assert roots[1] == pytest.approx(120.0, rel=1e-6)


def test_reference_curves_are_sorted_by_frequency_on_load(tmp_path):
    """``load_reference_curve``: "Rows are sorted by frequency on
    load, because digitiser output follows click order rather than
    axis order and an unsorted curve would silently break
    interpolation."

    Stated as a reason, which makes it testable by supplying exactly
    the input the reason describes -- click order.
    """
    from fwap.validation import load_reference_curve

    path = tmp_path / "scrambled.csv"
    path.write_text("9000,0.00042\n3000,0.00051\n15000,0.00038\n6000,0.00046\n")

    curve = load_reference_curve(path)

    assert list(curve.freq) == sorted(curve.freq)
    assert list(curve.freq) == [3000.0, 6000.0, 9000.0, 15000.0]
    # The values travelled with their frequencies rather than being
    # sorted independently.
    assert curve.slowness[0] == pytest.approx(0.00051)
    assert curve.slowness[-1] == pytest.approx(0.00038)


# ----------------------------------------------------------------------
# Units: a stated conversion that was written down and not performed
# ----------------------------------------------------------------------


def test_the_rickman_bounds_disagree_with_their_own_stated_conversion():
    """``fwap/geomechanics/indices.py``, pinned as a defect.

    The constants carry their own premise:

        "The original paper uses 1-8 Mpsi for E and 0.15-0.40 for nu;
        converted at 1 Mpsi = 6.8948 GPa."

    Apply that factor to 1 and 8 Mpsi and you get 6.895e9 and 5.516e10
    Pa. The shipped values are 1.0e10 and 8.0e10 -- the paper's
    *numerals* with "Mpsi" swapped for "1e10 Pa". The conversion is
    stated and then not performed, and the bottom of the window is
    45 % high as a result.

    Nothing else could catch this. The constants are self-consistent,
    the brittleness index they feed is monotone and lands in [0, 1],
    every geomechanics test passes, and the trailing comments
    (``# 1.45 Mpsi``, ``# 11.60 Mpsi``) correctly describe the *wrong*
    values -- someone computed what the constants are without noticing
    they contradict the source named two lines above. Only checking a
    number against the citation beside it fails here.

    **Pinned, not fixed**, per ``plans/learning.md``: "defects found by
    an oracle should be pinned as defects, not fixed silently". Fixing
    moves every ``brittleness_index`` output, so it needs Rickman et
    al. (2008) Table 1 confirmed first -- if the paper is really
    1-8 Mpsi the constants are wrong; if it is 1.45-11.6 Mpsi the
    stated premise is. That is a W1 task.

    **When it is fixed this test must fail**, and should then be
    rewritten to assert the conversion holds rather than that it does
    not.
    """
    from fwap.geomechanics import (
        RICKMAN_E_MAX_PA,
        RICKMAN_E_MIN_PA,
        RICKMAN_NU_MAX,
        RICKMAN_NU_MIN,
    )

    pa_per_mpsi = 6.8948e9  # the factor the module states

    # What the module says the paper is, carried through its own factor.
    assert 1.0 * pa_per_mpsi == pytest.approx(6.895e9, rel=1e-4)
    assert 8.0 * pa_per_mpsi == pytest.approx(5.516e10, rel=1e-4)

    # What is actually shipped, and by how much it misses.
    assert RICKMAN_E_MIN_PA == 1.0e10
    assert RICKMAN_E_MAX_PA == 8.0e10
    assert RICKMAN_E_MIN_PA / (1.0 * pa_per_mpsi) == pytest.approx(1.450, rel=1e-3)
    assert RICKMAN_E_MAX_PA / (8.0 * pa_per_mpsi) == pytest.approx(1.450, rel=1e-3)

    # Both bounds are off by the same factor, which is the tell that
    # this is a units slip and not a recalibration: a deliberate change
    # of window would not preserve the ratio exactly.
    assert (RICKMAN_E_MIN_PA / 1.0e10) == (RICKMAN_E_MAX_PA / 8.0e10)

    # Poisson's ratio is dimensionless, needs no conversion, and is
    # right -- which is why the slip is only in the E pair.
    assert (RICKMAN_NU_MIN, RICKMAN_NU_MAX) == (0.15, 0.40)
