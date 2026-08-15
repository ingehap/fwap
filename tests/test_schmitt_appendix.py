"""The Schmitt & Cheng appendix as an oracle for the layered solvers.

Provenance. Schmitt, D. P., & Cheng, C. H., "Shear Wave Logging In
(Multilayered) Elastic Formations: An Overview", MIT Earth Resources
Laboratory, pp. 213-268. The appendix on **pp. 235-236** prints all 36
elements of the layer matrix ``T(n, j, r)`` in closed form, under the
heading "It has to be noted that the following formulae are the ones
numerically used". ``_schmitt_appendix_T`` below is that matrix
transcribed entry for entry.

This is a different kind of reference from the digitised figures in
``test_cylindrical_solver.py``. Those constrain the solver's *output*;
this constrains its *formulation*, term by term, with no root-finding,
no digitisation and no tolerance to argue about.

What it establishes, in order:

1. **The transcription is right.** ``T`` is checked against Hooke's law
   using its own displacement rows, differentiated numerically: the
   stress rows follow to ~1e-11 at n=1 and n=2. Nothing in fwap is used.
2. **Four of fwap's six layer columns are exact.** Written in the
   appendix basis, the P and SH columns of ``_layer_e_matrix_n1`` and
   ``_layer_e_matrix_n2`` have r-independent coefficients to ~1e-13,
   so they are the same solutions.
3. **The two SV columns are not solutions at all.** Their coefficients
   in the same basis drift by 24-86 % across a 4 cm change in radius. A
   solution of the elastodynamic equations is a fixed combination of any
   basis for the solution space; these are not.
4. **It costs about 1.5 % on the public output.** Across figure 15's
   eight published invaded-zone frequencies a determinant built only
   from the appendix reads **0.24 % rms, median -0.22 %, errors of both
   signs** -- the digitisation's own resolution. On the same eight,
   ``flexural_dispersion_layered`` reads **1.49 % rms, median -1.49 %,
   low at every single one**. A one-sided residual six times the size
   of an unbiased one is a model error, not noise.

The mechanism is in the ansatz rather than the algebra. fwap represents
the SV field as a vector potential with only an azimuthal component,
``psi = psi_theta(r) e^{i n theta} e^{i k z} e_theta``. The vector
Laplacian in cylindrical coordinates couples the radial and azimuthal
components,

    (lap psi)_r = lap(psi_r) - psi_r / r^2 - (2 / r^2) d_theta psi_theta,

so with ``psi_r = 0`` the Laplacian acquires a radial component
proportional to ``n`` that the ansatz has no term to cancel; the radial
equation for ``psi_theta`` becomes Bessel of order ``sqrt(n^2 + 1)``
rather than ``n``. At ``n = 0`` the coupling term vanishes, which is
exactly why ``_layer_e_matrix_n0`` passes the same test cleanly and why
the Stoneley path ties figure 8a at 0.04 % rms. Schmitt uses the Hansen
form instead -- a scalar ``chi`` entering through ``curl curl (chi z)``
-- which is a solution at every order.

Recorded as roadmap A.8. Not fixed here: substituting the correct SV
column touches four layer matrices, both propagators, the cased
determinants at n=1 and n=2 and the single-layer path, and wants the
cased-hole figures (20, 21) digitised to validate it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg as sla
from scipy import special

from fwap import BoreholeLayer, flexural_dispersion_layered
from fwap.cylindrical_solver._cased import (
    _layer_e_matrix_n0,
    _layer_e_matrix_n1,
    _layer_e_matrix_n1_complex,
    _layer_e_matrix_n2,
    _layer_e_matrix_n2_complex,
)

# ----------------------------------------------------------------------
# The appendix, transcribed
# ----------------------------------------------------------------------


def _schmitt_appendix_T(
    n: int,
    *,
    k: complex,
    omega: float,
    kp: complex,
    ks: complex,
    rho: float,
    beta: float,
    r: float,
) -> np.ndarray:
    """Schmitt & Cheng pp. 235-236, ``T(n, j, r)``, verbatim.

    ``k`` is the axial wavenumber, ``kp`` / ``ks`` the radial P / S
    wavenumbers in the layer, ``beta`` its shear velocity and ``rho`` its
    density. Rows are ``(u_r, u_theta, u_z, sigma_rr, sigma_r_theta,
    sigma_rz)``; columns are ``(P-I, P-K, SH-I, SH-K, SV-I, SV-K)``. The
    common factor ``(1/n!)(kp_1 R_0 / 2)^n e^{ikz} e^{-i omega t}`` and
    the theta dependence are omitted, as the appendix states.
    """
    iv, kv = special.iv, special.kv
    Ip, Ip1 = iv(n, kp * r), iv(n + 1, kp * r)
    Kp, Kp1 = kv(n, kp * r), kv(n + 1, kp * r)
    Is, Is1 = iv(n, ks * r), iv(n + 1, ks * r)
    Ks, Ks1 = kv(n, ks * r), kv(n + 1, ks * r)
    mu = rho * beta * beta
    T = np.zeros((6, 6), dtype=complex)

    T[0, 0] = n / r * Ip + kp * Ip1
    T[0, 1] = n / r * Kp - kp * Kp1
    T[0, 2] = n / r * Is
    T[0, 3] = n / r * Ks
    T[0, 4] = 1j * k * (n / r * Is + ks * Is1)
    T[0, 5] = 1j * k * (n / r * Ks - ks * Ks1)

    T[1, 0] = -n / r * Ip
    T[1, 1] = -n / r * Kp
    T[1, 2] = -n / r * Is - ks * Is1
    T[1, 3] = -n / r * Ks + ks * Ks1
    T[1, 4] = -1j * k * n / r * Is
    T[1, 5] = -1j * k * n / r * Ks

    T[2, 0] = 1j * k * Ip
    T[2, 1] = 1j * k * Kp
    T[2, 2] = 0.0
    T[2, 3] = 0.0
    T[2, 4] = -ks * ks * Is
    T[2, 5] = -ks * ks * Ks

    bulk = rho * (2.0 * beta**2 * k**2 - omega**2)
    corner = 2.0 * n * (n - 1) / r**2 * mu
    T[3, 0] = Ip * (bulk + corner) - 2.0 / r * mu * kp * Ip1
    T[3, 1] = Kp * (bulk + corner) + 2.0 / r * mu * kp * Kp1
    T[3, 2] = 2.0 * mu * (n * (n - 1) / r**2 * Is + n / r * ks * Is1)
    T[3, 3] = 2.0 * mu * (n * (n - 1) / r**2 * Ks - n / r * ks * Ks1)
    T[3, 4] = 2j * k * mu * (Is * (n * (n - 1) / r**2 + ks**2) - ks * Is1 / r)
    T[3, 5] = 2j * k * mu * (Ks * (n * (n - 1) / r**2 + ks**2) + ks * Ks1 / r)

    T[4, 0] = -2.0 * n / r * mu * ((n - 1) / r * Ip + kp * Ip1)
    T[4, 1] = -2.0 * n / r * mu * ((n - 1) / r * Kp - kp * Kp1)
    T[4, 2] = mu * (Is * (2.0 * n * (1 - n) / r**2 - ks**2) + 2.0 / r * ks * Is1)
    T[4, 3] = mu * (Ks * (2.0 * n * (1 - n) / r**2 - ks**2) - 2.0 / r * ks * Ks1)
    T[4, 4] = 2j * k * n / r * mu * (Is * (1 - n) / r - ks * Is1)
    T[4, 5] = 2j * k * n / r * mu * (Ks * (1 - n) / r + ks * Ks1)

    T[5, 0] = 2j * k * mu * (n / r * Ip + kp * Ip1)
    T[5, 1] = 2j * k * mu * (n / r * Kp - kp * Kp1)
    T[5, 2] = 1j * k * n / r * mu * Is
    T[5, 3] = 1j * k * n / r * mu * Ks
    T[5, 4] = (omega**2 / beta**2 - 2.0 * k**2) * mu * (n / r * Is + ks * Is1)
    T[5, 5] = (omega**2 / beta**2 - 2.0 * k**2) * mu * (n / r * Ks - ks * Ks1)
    return T


#: fwap orders its state vector ``(u_r, u_z, u_theta, sigma_rr,
#: sigma_rz, sigma_r_theta)``; these index into the appendix's
#: ``(u_r, u_theta, u_z, sigma_rr, sigma_r_theta, sigma_rz)``.
_ROW = [0, 2, 1, 3, 5, 4]

#: fwap's columns ``(B_I, B_K, C_I, C_K, D_I, D_K)`` are P, P, SV, SV,
#: SH, SH; these index into the appendix's P, P, SH, SH, SV, SV.
_COL = [0, 1, 4, 5, 2, 3]

#: fwap rescales the ``u_z`` and ``sigma_rz`` rows by ``i`` so the
#: bound-regime matrix comes out real; the appendix does not.
_ROW_PHASE = np.array([1.0, 1j, 1.0, 1.0, 1j, 1.0])

#: The appendix writes ``u_theta`` and ``sigma_r_theta`` in the
#: quadrature angular sector, which is a factor ``i`` relative to the
#: plain ``e^{i n theta}`` form. Fitted, not assumed: see
#: ``test_the_appendix_matrix_obeys_hookes_law``.
_THETA_SECTOR = np.array([1.0, 1j, 1.0, 1.0, 1j, 1.0])

_CEMENT = dict(vp=2823.0, vs=1729.0, rho=1920.0)  # Table 1, cement 1
_OMEGA = 2.0 * np.pi * 8000.0
_C_TRIAL = 1400.0  # below the layer shear speed -> bound regime


def _appendix_in_fwap_layout(n: int, **kw: object) -> np.ndarray:
    T = _schmitt_appendix_T(n, **kw)  # type: ignore[arg-type]
    return (T[np.ix_(_ROW, _COL)].T * _ROW_PHASE).T


def _radial(kz: complex, omega: float, v: float) -> complex:
    return np.sqrt(complex(kz) ** 2 - (omega / v) ** 2)


def _layer_kwargs(
    kz: complex, medium: dict[str, float], omega: float = _OMEGA
) -> dict[str, object]:
    return dict(
        k=kz,
        omega=omega,
        kp=_radial(kz, omega, medium["vp"]),
        ks=_radial(kz, omega, medium["vs"]),
        rho=medium["rho"],
        beta=medium["vs"],
    )


# ----------------------------------------------------------------------
# 1. The transcription checks out against Hooke's law, using nothing else
# ----------------------------------------------------------------------

_STENCIL = ((-2, 1.0 / 12.0), (-1, -2.0 / 3.0), (1, 2.0 / 3.0), (2, -1.0 / 12.0))


def _hooke_residual(cols_at, r0: float, *, n: int, k: complex, medium) -> float:
    """Largest relative mismatch between a matrix's stress rows and the
    stresses implied by its own displacement rows."""
    mu = medium["rho"] * medium["vs"] ** 2
    lam = medium["rho"] * medium["vp"] ** 2 - 2.0 * mu
    h = 1.0e-6
    u0, s0 = cols_at(r0)
    du = np.zeros_like(u0)
    for off, w in _STENCIL:
        du += w * cols_at(r0 + off * h)[0]
    du /= h
    ur, uth, uz = u0
    dur, duth, duz = du
    div = (ur / r0 + dur) + (1j * n / r0) * uth + 1j * k * uz
    pred = np.vstack(
        [
            lam * div + 2.0 * mu * dur,
            mu * ((1j * n / r0) * ur + duth - uth / r0),
            mu * (1j * k * ur + duz),
        ]
    )
    scale = np.maximum(np.abs(pred), np.abs(s0)).max()
    return float(np.abs(pred - s0).max() / max(scale, 1e-300))


@pytest.mark.parametrize("n", [1, 2])
def test_the_appendix_matrix_obeys_hookes_law(n: int) -> None:
    """Validate the transcription before anything is built on it.

    The stress rows must be the stresses of the displacement rows. The
    displacements are differentiated in ``r`` numerically, so this uses
    only the transcription itself -- if a term were mis-keyed the
    residual would not collapse.

    The ``u_theta`` and ``sigma_r_theta`` rows sit in the quadrature
    angular sector, a factor ``i`` relative to ``e^{i n theta}``; that
    factor is applied via ``_THETA_SECTOR`` and is the only convention
    the appendix leaves implicit.
    """
    kz = _OMEGA / _C_TRIAL
    kw = _layer_kwargs(kz, _CEMENT)

    def cols(r: float):
        T = (_schmitt_appendix_T(n, r=r, **kw).T / _THETA_SECTOR).T
        return T[[0, 1, 2], :], T[[3, 4, 5], :]

    residual = _hooke_residual(cols, 0.12, n=n, k=kz, medium=_CEMENT)
    assert residual < 1.0e-8, f"n={n}: transcription residual {residual:.2e}"


# ----------------------------------------------------------------------
# 2 & 3. Which of fwap's layer columns are solutions
# ----------------------------------------------------------------------


def _coefficient_drift(matrix_at, n: int, radii, medium) -> np.ndarray:
    """Per-column drift of the coefficients expressing ``matrix_at(r)``
    in the appendix basis.

    A column that solves the elastodynamic equations is a **fixed**
    combination of any basis for the solution space, so its coefficients
    do not vary with ``r``. This allows an arbitrary mixing of the
    appendix's own columns, so it does not depend on identifying which
    fwap column is which.
    """
    kz = _OMEGA / _C_TRIAL
    kw = _layer_kwargs(kz, medium)
    coeffs = []
    for r in radii:
        E = np.asarray(matrix_at(r), dtype=complex)
        T = _appendix_in_fwap_layout(n, r=r, **kw)
        coeffs.append(np.linalg.solve(T, E))
    coeffs = np.array(coeffs)
    out = np.zeros(coeffs.shape[-1])
    for c in range(coeffs.shape[-1]):
        col = coeffs[:, :, c]
        scale = max(float(np.abs(col).max()), 1e-300)
        out[c] = float(np.abs(col - col[0]).max() / scale)
    return out


_RADII = (0.10, 0.11, 0.12, 0.13, 0.14)


@pytest.mark.parametrize(
    ("n", "matrix"),
    [
        (1, _layer_e_matrix_n1),
        (2, _layer_e_matrix_n2),
        (1, _layer_e_matrix_n1_complex),
        (2, _layer_e_matrix_n2_complex),
    ],
)
def test_the_p_and_sh_layer_columns_are_exact(n: int, matrix) -> None:
    """Four of the six columns are the appendix's own solutions.

    Columns 0-1 (P) and 4-5 (SH) express in the appendix basis with
    coefficients constant to ~1e-13 across a 4 cm change in radius. That
    is agreement at the level of the formulation, not of a fitted curve,
    and it is what makes the SV result in the next test a finding rather
    than a units mismatch.
    """
    kz = _OMEGA / _C_TRIAL
    if "complex" in matrix.__name__:

        def at(r: float, matrix=matrix):
            return matrix(complex(kz, 0.0), _OMEGA, **_CEMENT, r=r)

    else:

        def at(r: float, matrix=matrix):
            return matrix(kz, _OMEGA, **_CEMENT, r=r)

    drift = _coefficient_drift(at, n, _RADII, _CEMENT)
    for c in (0, 1, 4, 5):
        assert drift[c] < 1.0e-8, f"{matrix.__name__} col {c}: drift {drift[c]:.2e}"


@pytest.mark.parametrize(
    ("n", "matrix"),
    [
        (1, _layer_e_matrix_n1),
        (2, _layer_e_matrix_n2),
        (1, _layer_e_matrix_n1_complex),
        (2, _layer_e_matrix_n2_complex),
    ],
)
def test_the_sv_layer_columns_are_solutions(n: int, matrix) -> None:
    """Roadmap A.8, fixed: the SV columns are solutions again.

    This test was written to fail. It found that the two SV columns
    could not be written in the appendix basis with r-independent
    coefficients -- they drifted by 24 to 86 % over a 4 cm change in
    radius -- and since the appendix basis spans the whole solution
    space, a column that is a solution *must* have constant
    coefficients.

    The cause was the ansatz. fwap built the SV field from a vector
    potential with only an azimuthal component; the cylindrical vector
    Laplacian couples the radial and azimuthal components through a term
    proportional to ``n``, which that ansatz has nothing to cancel. At
    ``n = 0`` the term vanishes -- hence the companion test on
    ``_layer_e_matrix_n0``.

    The columns are now the Hansen form ``curl curl(chi z)``, which is
    Schmitt & Cheng's appendix column, and the drift has collapsed from
    tens of percent to ~1e-12: the same bound the P and SH columns meet
    in the test above.
    """
    kz = _OMEGA / _C_TRIAL
    if "complex" in matrix.__name__:

        def at(r: float, matrix=matrix):
            return matrix(complex(kz, 0.0), _OMEGA, **_CEMENT, r=r)

    else:

        def at(r: float, matrix=matrix):
            return matrix(kz, _OMEGA, **_CEMENT, r=r)

    drift = _coefficient_drift(at, n, _RADII, _CEMENT)
    for c in (2, 3):
        assert drift[c] < 1.0e-8, (
            f"{matrix.__name__} col {c}: drift {drift[c]:.2e} -- the SV "
            "columns must express in the appendix basis with constant "
            "coefficients, like the P and SH columns"
        )


def test_the_n0_layer_matrix_is_unaffected() -> None:
    """The axisymmetric matrix is clean, which locates the defect.

    At ``n = 0`` the vector Laplacian's radial/azimuthal coupling term
    carries a factor ``n`` and vanishes, so the azimuthal-only potential
    is a solution after all. All four columns express in the appendix
    basis with constant coefficients -- consistent with
    ``stoneley_dispersion`` tying figure 8a at 0.04 % rms.
    """
    kz = _OMEGA / _C_TRIAL
    kw = _layer_kwargs(kz, _CEMENT)
    coeffs = []
    for r in _RADII:
        E = np.asarray(_layer_e_matrix_n0(kz, _OMEGA, **_CEMENT, r=r), dtype=complex)
        T = _schmitt_appendix_T(0, r=r, **kw)
        sub = T[np.ix_([0, 2, 3, 5], [0, 1, 4, 5])]
        sub = (sub.T * np.array([1.0, 1j, 1.0, 1j])).T
        coeffs.append(np.linalg.solve(sub, E))
    coeffs = np.array(coeffs)
    for c in range(4):
        col = coeffs[:, :, c]
        drift = float(
            np.abs(col - col[0]).max() / max(float(np.abs(col).max()), 1e-300)
        )
        assert drift < 1.0e-8, f"n=0 col {c}: drift {drift:.2e}"


# ----------------------------------------------------------------------
# 4. What it costs on the public output
# ----------------------------------------------------------------------

#: Schmitt & Cheng figure 15(a), 16 cm invaded zone in the slow
#: sandstone (Hz, m/s). Same digitisation as
#: ``tests/test_cylindrical_solver.py::_FIG15A_PHASE``.
_FIG15A_INVADED_16CM = (
    (5.0e3, 985.0),
    (6.0e3, 970.1),
    (8.0e3, 952.5),
)

_SLOW = dict(vp=2751.0, vs=1201.0, rho=2100.0)
_INVADED = dict(vp=2338.0, vs=1081.0, rho=2000.0)
_FLUID = dict(vf=1500.0, rho_f=1000.0)
_HOLE = 0.10
_THICKNESS = 0.16


#: Cased-hole stack: API 5.5" steel and Schmitt & Cheng Table 1 cement 1.
_STEEL = dict(vp=5940.0, vs=3220.0, rho=7500.0)
_FAST = dict(vp=4500.0, vs=2600.0, rho=2500.0)
_CASED_STACK = ((_STEEL, 0.01), (_CEMENT, 0.03))
_CASED_HOLE = 0.10


def _bound_radial(kz: complex, omega: float, v: float) -> complex:
    return np.sqrt(complex(kz) ** 2 - (omega / v) ** 2)


def _outgoing_radial(kz: complex, omega: float, v: float) -> complex:
    """The radiating branch, fixed by the Bessel identity rather than
    by which sign happens to agree.

    ``K_nu(z) = (pi/2) i^(nu+1) H^(1)_nu(i z)``, so ``K_nu(ks r)`` -- the
    appendix's decaying column -- is an *outgoing* wave exactly when
    ``ks = -i k_r`` with ``k_r = sqrt((omega/V)^2 - k_z^2)`` taken with
    positive real part. Getting this wrong is not a small error: it
    leaves the assembly with no root at all where fwap has one, which
    reads as a disagreement rather than as a bug (PR #129).
    """
    kr = np.sqrt((omega / v) ** 2 - complex(kz) ** 2)
    if kr.real < 0.0:
        kr = -kr
    return -1j * kr


def _appendix_sigma_min_stack(
    kz: complex,
    omega: float,
    *,
    n: int,
    fluid: dict[str, float],
    a: float,
    stack: tuple[tuple[dict[str, float], float], ...],
    half: dict[str, float],
    leaky_s: bool = False,
) -> float:
    """Smallest singular value of the 4x4 boundary-condition matrix for
    a fluid core, ``N`` elastic annuli and an elastic half-space,
    assembled only from the appendix.

    Generalises :func:`_appendix_sigma_min` from one annulus to a stack
    and from real to complex ``k_z``, which is what a cased hole needs:
    behind steel the dipole runs faster than a slow formation's shear
    speed, so the mode radiates and ``k_z`` is complex.

    The branch rule is **per wave**: the formation P stays bound while
    its S radiates, so only ``ks`` moves to the outgoing branch. Columns
    are normalised at every stage; the raw determinant is numerically
    useless for the reason roadmap A.7 records, and the singular value
    is the conditioning-robust stand-in.

    Parameters
    ----------
    kz : complex
        Axial wavenumber (rad/m).
    omega : float
        Angular frequency (rad/s).
    n : int
        Azimuthal order.
    fluid : dict
        ``vf`` (m/s) and ``rho_f`` (kg/m^3).
    a : float
        Borehole (fluid-side) radius (m).
    stack : tuple of (dict, float)
        ``(medium, thickness)`` inside-out; ``vp``/``vs`` in m/s,
        ``rho`` in kg/m^3, thickness in m.
    half : dict
        Formation half-space medium.
    leaky_s : bool, default False
        Put the half-space shear wave on the radiating branch.

    Returns
    -------
    float
        Smallest singular value, or ``inf`` where the algebra fails.
    """

    def T_nat(medium: dict[str, float], r: float, radiating: bool) -> np.ndarray:
        kw = dict(
            k=complex(kz),
            omega=omega,
            kp=_bound_radial(kz, omega, medium["vp"]),
            ks=(
                _outgoing_radial(kz, omega, medium["vs"])
                if radiating
                else _bound_radial(kz, omega, medium["vs"])
            ),
            rho=medium["rho"],
            beta=medium["vs"],
        )
        T = _schmitt_appendix_T(n, r=r, **kw)  # type: ignore[arg-type]
        return (T.T / _THETA_SECTOR).T

    def normalise(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.abs(x).max(axis=0), 1e-300)

    radii = [a]
    for _, thickness in stack:
        radii.append(radii[-1] + thickness)

    try:
        V = normalise(T_nat(half, radii[-1], leaky_s)[:, [1, 3, 5]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sla.LinAlgWarning)
            for i in range(len(stack) - 1, -1, -1):
                medium = stack[i][0]
                V = normalise(
                    T_nat(medium, radii[i], False)
                    @ sla.solve(T_nat(medium, radii[i + 1], False), V)
                )
        F = _bound_radial(kz, omega, fluid["vf"])
        i_n = special.iv(n, F * a)
        i_n1 = special.iv(n + 1, F * a)
        M = np.zeros((4, 4), dtype=complex)
        M[0, :3] = V[0]
        M[0, 3] = -(F * i_n1 + n / a * i_n) / (fluid["rho_f"] * omega**2)
        M[1, :3] = V[3]
        M[1, 3] = +i_n
        M[2, :3] = V[4]
        M[3, :3] = V[5]
        M = normalise(M)
        if not np.isfinite(M).all():
            return np.inf
        return float(np.linalg.svd(M, compute_uv=False)[-1])
    except (np.linalg.LinAlgError, sla.LinAlgError, ValueError):
        return np.inf


def _appendix_root_sharpness(kz: complex, omega: float, **kw: object) -> float:
    """How far ``sigma_min`` rises off the root, as a ratio.

    The conditioning-robust stand-in for the log-decade depth used
    against Sinha's open-hole matrix: a normalised 4x4 bottoms out near
    1e-16 at a true root, so the discriminating quantity is how much
    *higher* the neighbours sit.
    """
    at_root = _appendix_sigma_min_stack(kz, omega, **kw)  # type: ignore[arg-type]
    if not np.isfinite(at_root) or at_root <= 0.0:
        return 0.0
    nearby = np.median(
        [
            _appendix_sigma_min_stack(kz * g, omega, **kw)  # type: ignore[arg-type]
            for g in (0.97, 0.98, 1.02, 1.03)
        ]
    )
    return float(nearby / at_root)


def _appendix_sigma_min(c: float, omega: float, n: int = 1) -> float:
    """Smallest singular value of the 4x4 boundary-condition matrix
    assembled only from the appendix.

    Fluid core, one elastic annulus, elastic half-space. Columns are
    normalised at every stage: the raw determinant of this system is
    numerically useless for the same reason roadmap A.7 records, and the
    singular value is the conditioning-robust stand-in.
    """
    kz = complex(omega / c)
    r_in, r_out = _HOLE, _HOLE + _THICKNESS

    def T_nat(medium, r):
        kw = _layer_kwargs(kz, medium, omega)
        T = _schmitt_appendix_T(n, r=r, **kw)  # type: ignore[arg-type]
        return (T.T / _THETA_SECTOR).T

    def normalise(a):
        return a / np.maximum(np.abs(a).max(axis=0), 1e-300)

    try:
        V = normalise(T_nat(_SLOW, r_out)[:, [1, 3, 5]])  # decaying columns
        # A direct solve, not lstsq: the layer matrix is ill-conditioned
        # here (that is what A.7 is about) and lstsq's rank truncation
        # discards real information rather than noise, moving the root.
        # Samples where the algebra genuinely fails return inf below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sla.LinAlgWarning)
            V = normalise(T_nat(_INVADED, r_in) @ sla.solve(T_nat(_INVADED, r_out), V))
        F = _radial(kz, omega, _FLUID["vf"])
        i_n = special.iv(n, F * r_in)
        i_n1 = special.iv(n + 1, F * r_in)
        M = np.zeros((4, 4), dtype=complex)
        M[0, :3] = V[0]
        M[0, 3] = -(F * i_n1 + n / r_in * i_n) / (_FLUID["rho_f"] * omega**2)
        M[1, :3] = V[3]
        M[1, 3] = +i_n
        M[2, :3] = V[4]
        M[3, :3] = V[5]
        M = normalise(M)
        if not np.isfinite(M).all():
            return np.inf
        return float(np.linalg.svd(M, compute_uv=False)[-1])
    except (np.linalg.LinAlgError, sla.LinAlgError, ValueError):
        return np.inf


def test_the_appendix_and_fwap_now_agree_on_the_published_invaded_curve() -> None:
    """The end-to-end check that closed A.8, against figure 15.

    This test was the measurement that made A.8 a finding. A determinant
    assembled only from the appendix landed within about 0.4 % of the
    published curve with no systematic sign, while
    ``flexural_dispersion_layered`` was low at every frequency by
    1.5-2 %. Figure 15's "1.47 % rms" had been read, when it was
    measured, as the layered propagator being sound; it was nearly
    sound, and the remaining 1.5 % was the SV column.

    With the column corrected the two formulations are the same answer:

        f (kHz)   published   appendix          fwap
          5.0        985.0      985.4 (+0.05%)    985.5 (+0.05%)
          6.0        970.1      968.2 (-0.19%)    968.3 (-0.18%)
          8.0        952.5      951.0 (-0.16%)    951.0 (-0.16%)

    They agree with each other to 0.011 % and with the figure to 0.19 %,
    and fwap's one-sided bias is gone. Kept as a tie: an independent
    assembly of the same boundary-value problem now reproduces the
    solver, which is the strongest form this check can take.
    """
    appendix_err, fwap_err = [], []
    for freq, published in _FIG15A_INVADED_16CM:
        omega = 2.0 * np.pi * freq
        # Search strictly below the *invaded* shear speed. The branch
        # figure 15 plots descends below it (985 -> 952 over 5-8 kHz),
        # while a second, unrelated minimum sits up near the *virgin*
        # shear speed. Over a bracket wide enough to hold both, which
        # one is the global minimum is a near-tie that different SciPy
        # builds resolve differently -- so the bracket excludes the
        # competitor rather than relying on the tie going one way.
        grid = np.linspace(900.0, _INVADED["vs"] - 3.0, 601)
        sv = np.array([_appendix_sigma_min(c, omega) for c in grid])
        sv = np.where(np.isfinite(sv), sv, np.inf)
        idx = int(np.argmin(sv))
        assert 0 < idx < grid.size - 1, (
            f"{freq / 1e3:.0f} kHz: minimum at a grid edge; widen the bracket"
        )
        assert sv[idx] < 0.05 * float(np.median(sv[np.isfinite(sv)])), (
            f"{freq / 1e3:.0f} kHz: minimum is not sharply separated "
            f"({sv[idx]:.2e} against a median of "
            f"{float(np.median(sv[np.isfinite(sv)])):.2e})"
        )
        c_appendix = float(grid[idx])
        c_fwap = (
            1.0
            / flexural_dispersion_layered(
                np.array([freq]),
                **_SLOW,
                **_FLUID,
                a=_HOLE,
                layers=(BoreholeLayer(**_INVADED, thickness=_THICKNESS),),
            ).slowness[0]
        )
        appendix_err.append(c_appendix / published - 1.0)
        fwap_err.append(c_fwap / published - 1.0)

    appendix = np.array(appendix_err)
    fwap = np.array(fwap_err)
    assert np.abs(appendix).max() < 0.005, f"appendix {100 * appendix}"
    assert np.abs(fwap).max() < 0.005, f"fwap {100 * fwap}"
    # The two assemblies agree with each other far more tightly than
    # either agrees with the digitised curve.
    assert np.abs(fwap - appendix).max() < 5.0e-4, (
        f"appendix {100 * appendix} vs fwap {100 * fwap}"
    )
    # And fwap's bias is gone: it is no longer low at every frequency.
    assert not np.all(fwap < 0.0), f"fwap {100 * fwap}"


# ----------------------------------------------------------------------
# 5. A.8 is not confined to the layered paths
# ----------------------------------------------------------------------
#
# The section above measures the SV defect through the layer matrices.
# The same ansatz appears in the **open-hole** determinants, and the
# tests below measure it there, where there is no propagator and no
# conditioning question to muddy the result.
#
# Assembling the open-hole boundary-value problem from the appendix and
# comparing both against figure 8a's three published curves for the same
# rock:
#
#     mode              n    appendix    fwap (before)   fwap (now)
#     Stoneley          0    0.03 %      0.03 %          0.032 %
#     flexural          1    **0.08 %**  1.30 %          **0.075 %**
#     screw             2    **0.06 %**  0.43 %          **0.057 %**
#
# Two things follow. The published curves are reproducible to **0.06 %**,
# so the "about +-1 %" that this project has been calling the
# digitisation floor for the n >= 1 modes was never the digitisation --
# it was fwap's model error, and the digitisation is an order of
# magnitude better than assumed. And `n = 0` agreeing with the appendix
# to 0.03 % is the control: where the azimuthal-only ansatz *is* a
# solution, the two formulations agree exactly.
#
# **The correction was derived, applied and measured before being
# reverted.** For the open-hole determinants the corrected SV column is,
# at general ``n`` with ``A = s K_{n-1} + n K_n / a``:
#
#     M13 = kz * A
#     M23 = -2 kz mu (s^2 K_n + s K_{n-1} / a + n(n+1) K_n / a^2)
#     M33 = 2 kz mu (n / a) (s K_{n-1} + (n+1) K_n / a)
#     M43 = (2 kz^2 - kS^2) mu * A
#
# verified against the appendix column to 1e-16 at both orders. Applied
# to `_modal_determinant_n1` and `_modal_determinant_n2` (real and
# complex), it moved figure 8a's flexural from **1.29 % to 0.063 % rms**
# and its screw from **0.43 % to 0.058 %**, and the fast-formation
# flexural of figure 2a from 0.78 % to **0.16 %**.
#
# **It is now landed everywhere**, including the layered paths, the
# cased propagator E-matrices (real and complex), the hand-coded 10x10
# single-layer determinant and the VTI qSV column, so the
# `layer == formation` and `N=1 vs F.2` invariants hold against the
# corrected forms rather than the old ones. The tests in this file that
# were written to fail while the defect stood have been inverted; they
# now pin the agreement.


def test_the_open_hole_determinants_match_the_appendix() -> None:
    """A.8 reached further than the layer matrices, and is closed there.

    Figure 8a plots three modes for one slow sandstone. Rebuilt from the
    appendix, all three land within 0.1 % of the published curve. fwap
    used to match at ``n = 0`` and be an order of magnitude worse at
    ``n = 1`` and ``n = 2`` -- the orders where the azimuthal-only SV
    ansatz stops being a solution:

        mode         n    appendix    fwap (before)   fwap (now)
        Stoneley     0    0.03 %      0.03 %          0.032 %
        flexural     1    0.08 %      1.30 %          0.075 %
        screw        2    0.06 %      0.43 %          0.057 %

    All three now sit at the appendix's own agreement with the figure,
    and the n >= 1 / n = 0 split is gone.
    """
    from fwap import flexural_dispersion, stoneley_dispersion
    from fwap.cylindrical_solver import quadrupole_dispersion

    rock = dict(vp=2751.0, vs=1201.0, rho=2100.0)
    published = {
        0: ((3.0e3, 1076.7), (5.0e3, 1059.6), (8.0e3, 1046.3), (12.0e3, 1037.2)),
        1: ((3.0e3, 1163.0), (5.0e3, 1099.5), (8.0e3, 1061.5), (12.0e3, 1044.8)),
        2: ((8.0e3, 1104.8), (10.0e3, 1081.3), (12.0e3, 1066.8), (14.0e3, 1057.3)),
    }
    solver = {0: stoneley_dispersion, 1: flexural_dispersion, 2: quadrupole_dispersion}

    scores = {}
    for n, table in published.items():
        freq = np.array([f for f, _ in table])
        ref = np.array([v for _, v in table])
        got = 1.0 / solver[n](freq, **rock, **_FLUID, a=_HOLE).slowness
        finite = np.isfinite(got)
        assert finite.all(), f"n={n}: {got}"
        scores[n] = float(np.sqrt((((got - ref) / ref) ** 2).mean()))

    # n = 0 is the control: the ansatz was always a solution there.
    assert scores[0] < 0.002, f"n=0 rms {scores[0]:.2%}"
    # n >= 1 now sit at the same level rather than an order of magnitude
    # above it.
    assert scores[1] < 0.002, f"n=1 rms {scores[1]:.2%}"
    assert scores[2] < 0.002, f"n=2 rms {scores[2]:.2%}"
    assert scores[1] < 5.0 * scores[0]


# ----------------------------------------------------------------------
# 5. The appendix as a cased-hole oracle
#
# Sections 1-4 check the appendix's *columns* and one published curve.
# This section uses the assembled matrix the way PR #129 used Sinha's:
# as an independent statement of where the roots are. It reaches what
# Sinha's 4x4 cannot -- a real cased stack, steel and cement, where the
# dipole is leaky because behind steel it outruns a slow formation's
# shear speed.
#
# Scale: a normalised 4x4 bottoms out near 1e-16 at a true root, so the
# measured quantity is the *ratio* by which sigma_min rises off it. All
# thresholds below are far under what is observed (99-5239x), and the
# contrast against the wrong branch is the point: there the ratio is
# 1.0-1.9, i.e. no root at all.
# ----------------------------------------------------------------------


def test_the_appendix_locates_the_cased_leaky_modes() -> None:
    """fwap's cased leaky dipole and screw modes are roots of the
    appendix matrix, across the band.

    This is the check the open-hole oracle cannot make. Behind steel the
    n=1 mode runs at 1.11-1.22 times the formation shear speed, so it
    radiates and ``k_z`` is complex -- the regime roadmap A.9 opened and
    PR #122 narrowed. An independently published layer matrix puts the
    root where fwap puts it, to within a singular value of 1e-16.
    """
    fluid, a = _FLUID, _CASED_HOLE
    layers = tuple(
        BoreholeLayer(**medium, thickness=t)  # type: ignore[arg-type]
        for medium, t in _CASED_STACK
    )
    for n, solver in ((1, flexural_dispersion_layered), (2, _quadrupole_layered())):
        freq = np.arange(5000.0, 14000.0, 1000.0)
        mode = solver(freq, **_SLOW, **fluid, a=a, layers=layers)  # type: ignore[arg-type]
        damping = mode.attenuation_per_meter
        assert damping is not None, n
        live = np.isfinite(mode.slowness)
        assert live.sum() >= 6, (n, live.sum())
        checked = 0
        for i in np.flatnonzero(live):
            omega = 2.0 * np.pi * freq[i]
            kz = omega * mode.slowness[i] + 1j * damping[i]
            # The mode is genuinely leaky here, which is why the
            # radiating branch is the right one to ask for.
            assert 1.0 / mode.slowness[i] > _SLOW["vs"], (n, freq[i])
            assert damping[i] > 0.0, (n, freq[i])
            sharpness = _appendix_root_sharpness(
                kz,
                omega,
                n=n,
                fluid=fluid,
                a=a,
                stack=_CASED_STACK,
                half=_SLOW,
                leaky_s=True,
            )
            assert sharpness > 25.0, (n, freq[i], sharpness)
            checked += 1
        assert checked >= 6, (n, checked)


def test_the_cased_oracle_needs_the_radiating_branch() -> None:
    """The same trap as PR #129, on a different paper's matrix.

    Asked with the *bound* half-space shear column, the assembly has no
    root at fwap's cased leaky dipole -- ``sigma_min`` sits at 1e-13 and
    barely moves (ratio about 1-2) where the radiating branch bottoms
    out at 1e-16 with a ratio in the hundreds. Nothing about the steel
    or the conditioning causes that; it is the branch alone.

    Kept because the failure is silent: a bound-branch transcription
    reproduces every trapped mode and then reports a clean disagreement
    on the leaky ones.
    """
    fluid, a = _FLUID, _CASED_HOLE
    layers = tuple(
        BoreholeLayer(**medium, thickness=t)  # type: ignore[arg-type]
        for medium, t in _CASED_STACK
    )
    freq = np.array([5000.0, 8000.0, 11000.0])
    mode = flexural_dispersion_layered(freq, **_SLOW, **fluid, a=a, layers=layers)
    damping = mode.attenuation_per_meter
    assert damping is not None
    common = dict(n=1, fluid=fluid, a=a, stack=_CASED_STACK, half=_SLOW)
    for i, f in enumerate(freq):
        if not np.isfinite(mode.slowness[i]):
            continue
        omega = 2.0 * np.pi * f
        kz = omega * mode.slowness[i] + 1j * damping[i]
        radiating = _appendix_root_sharpness(kz, omega, **common, leaky_s=True)
        bound = _appendix_root_sharpness(kz, omega, **common, leaky_s=False)
        assert radiating > 25.0, (f, radiating)
        assert bound < 5.0, (f, bound)
        assert radiating > 20.0 * bound, (f, radiating, bound)


def test_the_appendix_locates_the_cased_bound_modes() -> None:
    """Behind the same casing, the bound sub-fluid modes too.

    Fast formation, so the Stoneley and the flexural below ``V_f`` are
    trapped rather than radiating and ``k_z`` is real. Together with the
    leaky test above this puts both cased regimes, n=0 through n=2,
    under one published layer matrix.
    """
    fluid, a = _FLUID, _CASED_HOLE
    layers = tuple(
        BoreholeLayer(**medium, thickness=t)  # type: ignore[arg-type]
        for medium, t in _CASED_STACK
    )
    freq = np.arange(3000.0, 14000.0, 1500.0)
    mode = _stoneley_layered()(freq, **_FAST, **fluid, a=a, layers=layers)
    checked = 0
    for i, f in enumerate(freq):
        if not np.isfinite(mode.slowness[i]):
            continue
        c = 1.0 / mode.slowness[i]
        if c >= fluid["vf"]:
            continue  # the assembly's fluid core is the bound form
        omega = 2.0 * np.pi * f
        sharpness = _appendix_root_sharpness(
            omega * mode.slowness[i],
            omega,
            n=0,
            fluid=fluid,
            a=a,
            stack=_CASED_STACK,
            half=_FAST,
            leaky_s=False,
        )
        assert sharpness > 20.0, (f, sharpness)
        checked += 1
    assert checked >= 6, checked


def _quadrupole_layered():  # type: ignore[no-untyped-def]
    from fwap import quadrupole_dispersion_layered

    return quadrupole_dispersion_layered


def _stoneley_layered():  # type: ignore[no-untyped-def]
    from fwap import stoneley_dispersion_layered

    return stoneley_dispersion_layered
