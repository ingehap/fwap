"""
Wave separation -- f-k, tau-p, and SVD / Karhunen-Loeve projection.

Implements the multichannel velocity-filter family described in Part 2
of the book ("Wave separation in acoustic well logging"). Three
complementary tools are provided:

* :func:`fk_filter`              -- frequency-wavenumber pass-band
* :func:`tau_p_forward`,
  :func:`tau_p_inverse`,
  :func:`tau_p_filter`            -- linear Radon (slant-stack)
                                     pass-band
* :func:`svd_project`,
  :func:`sequential_kl_separation` -- moveout-corrected SVD / K-L

The book lists *both* f-k and tau-p as the textbook domains for
multichannel velocity filtering ("working in the frequency-wavenumber
(f-k) or tau-p domain"). f-k is unitary and exact-round-trip but
requires uniform offset spacing; tau-p accepts arbitrary offsets and
keeps a sharper localisation of linear-moveout events at the cost of
needing the rho filter (|f| in the temporal frequency domain) for an
accurate inverse.

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 2. Editions Technip,
  Paris. ISBN 978-2-7108-0664-6.
* Mari, J.-L., & Glangeaud, F. (1990). *Wave Separation.* Editions
  Technip, Paris.
* Freire, S. L. M., & Ulrych, T. J. (1988). Application of singular
  value decomposition to vertical seismic profiling. *Geophysics*
  53(6), 778-785.
* Yilmaz, O. (2001). *Seismic Data Processing*, 2nd ed., Section 1.2
  (f-k domain conventions) and Section 6.3 (tau-p / linear Radon).
  SEG, Tulsa.
* Beylkin, G. (1987). Discrete radon transform. *IEEE Transactions on
  Acoustics, Speech, and Signal Processing* 35(2), 162-172
  (rho-filter inverse).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fwap._common import _phase_shift


def fk_forward(
    data: np.ndarray, dt: float, dx: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward 2-D FFT of a regularly-sampled gather into the f-k domain.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
        Real-valued gather, receivers uniformly spaced by ``dx``.
    dt : float
        Time sampling interval (s).
    dx : float
        Receiver spacing (m).

    Returns
    -------
    spec : ndarray, shape (n_rec, n_samples // 2 + 1), complex
        2-D spectrum. Axis 0 is wavenumber (``k``), fft-shifted so
        that ``k`` increases monotonically from most negative to most
        positive. Axis 1 is temporal frequency (``f``), non-negative
        rFFT layout with ``f[0] = 0`` and ``f[-1] = Nyquist``.
    f : ndarray, shape (n_samples // 2 + 1,)
        Frequency axis (Hz).
    k : ndarray, shape (n_rec,)
        Wavenumber axis (1/m), fft-shifted to match ``spec``.
    """
    n_rec, n_samp = data.shape
    spec_t = np.fft.rfft(data, axis=1)
    spec = np.fft.fftshift(np.fft.fft(spec_t, axis=0), axes=0)
    f = np.fft.rfftfreq(n_samp, d=dt)
    k = np.fft.fftshift(np.fft.fftfreq(n_rec, d=dx))
    return spec, f, k


def fk_inverse(spec: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Inverse 2-D FFT back to the time-offset domain.

    The inverse of :func:`fk_forward`: consumes a spectrum in the
    fft-shifted (wavenumber, rFFT-frequency) layout and returns the
    real-valued time-offset gather.

    Parameters
    ----------
    spec : ndarray, shape (n_rec, n_samples // 2 + 1), complex
        Spectrum in the layout returned by :func:`fk_forward`.
    n_samples : int
        Original number of time samples; required because the rFFT
        output does not by itself distinguish an even- from an odd-
        length input.

    Returns
    -------
    ndarray, shape (n_rec, n_samples)
    """
    spec_t = np.fft.ifft(np.fft.ifftshift(spec, axes=0), axis=0)
    return np.fft.irfft(spec_t, n=n_samples, axis=1)


def fk_filter(
    data: np.ndarray,
    dt: float,
    dx: float,
    slow_min: float,
    slow_max: float,
    positive_only: bool = True,
    taper_width: float = 0.1,
) -> np.ndarray:
    """
    f-k (apparent-slowness) bandpass filter.

    Keeps only energy whose apparent slowness ``s = -k/f`` lies in
    ``[slow_min, slow_max]``. When ``positive_only=False`` the same
    passband in ``|s|`` is kept on both sides of the ``k`` axis
    (up-going and down-going), with the cosine taper applied
    symmetrically.

    Sign convention
    ---------------
    Under the NumPy FFT convention ``X(f) = sum_n x(n) e^{-2 pi i f n}``
    a forward-propagating wave of physical slowness ``s > 0`` -- that
    is, a source record ``x(t, i) = src(t - i * dx * s)`` -- produces
    energy at wavenumber ``k = -f * s``. The physical apparent
    slowness recovered from the f-k plane is therefore ``S = -k / f``.
    See Yilmaz (2001), *Seismic Data Processing*, Section 1.2.
    """
    if not (slow_max > slow_min > 0):
        raise ValueError("require 0 < slow_min < slow_max")
    spec, f, k = fk_forward(data, dt, dx)
    F, K = np.meshgrid(f, k)

    # See the sign-convention note in the docstring: S = -k/f.
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(F > 0, -K / F, np.nan)

    # ``ref_S`` is the quantity compared against [slow_min, slow_max]:
    # signed apparent slowness when ``positive_only`` (passes only the
    # forward-propagating side), or |S| when both sides are kept.
    ref_S = S if positive_only else np.abs(S)
    inside = (ref_S >= slow_min) & (ref_S <= slow_max)
    mask = inside.astype(float)

    if taper_width > 0:
        w = taper_width * (slow_max - slow_min)
        lo = (ref_S >= slow_min - w) & (ref_S < slow_min)
        hi = (ref_S > slow_max) & (ref_S <= slow_max + w)
        # Half-cosine ramps in a single vectorised shot per side.
        mask[lo] = 0.5 * (1.0 - np.cos(np.pi * (ref_S[lo] - (slow_min - w)) / w))
        mask[hi] = 0.5 * (1.0 + np.cos(np.pi * (ref_S[hi] - slow_max) / w))
    mask[np.isnan(S)] = 0.0
    return fk_inverse(spec * mask, n_samples=data.shape[1])


# ---------------------------------------------------------------------
# Tau-p / slant-stack / linear Radon
# ---------------------------------------------------------------------


def tau_p_forward(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slownesses: np.ndarray,
) -> np.ndarray:
    r"""
    Forward tau-p (slant-stack / linear Radon) transform.

    For each candidate apparent slowness :math:`p`, sums the trace
    amplitudes along the linear-moveout line :math:`t = \tau + p \cdot
    x_i` (with :math:`x_i = \text{offsets}[i] - \text{offsets}[0]`)::

        panel(\tau, p) = \sum_i d(\tau + p \cdot x_i, x_i)

    Implemented in the frequency domain via per-trace fractional time
    shifts of :func:`fwap._common._phase_shift`, which keeps the
    transform exact for non-integer ``p * dx / dt`` and avoids the
    sample-snapping error of a naive time-domain stack.

    Unlike :func:`fk_forward`, the offsets need *not* be uniformly
    spaced.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
        Real-valued gather.
    dt : float
        Time sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m). Need not be uniform; the
        slant-stack uses ``offsets - offsets[0]`` as the lever arm.
    slownesses : ndarray, shape (n_slowness,)
        Apparent-slowness grid (s/m). For wave separation pick
        ``n_slowness`` >= 2 * n_rec to keep the linear Radon frame
        well conditioned, and span a band centred on the moveout you
        want to isolate.

    Returns
    -------
    ndarray, shape (n_slowness, n_samples)
        ``panel[k, j]`` is the slant-stacked amplitude at slowness
        ``slownesses[k]`` and intercept time ``j * dt``.
    """
    n_rec, n_samp = data.shape
    offsets = np.asarray(offsets, dtype=float)
    slownesses = np.asarray(slownesses, dtype=float)
    if offsets.size != n_rec:
        raise ValueError("offsets must have length n_rec")
    spec = np.fft.rfft(data, axis=1)  # (n_rec, n_f)
    f = np.fft.rfftfreq(n_samp, d=dt)  # (n_f,)
    rel_off = offsets - offsets[0]  # (n_rec,)
    # phase[k, i, f] = exp(2 pi i f * slownesses[k] * rel_off[i]) is
    # the per-trace, per-slowness fractional time advance that flattens
    # an event at apparent slowness slownesses[k] before stacking.
    phase = np.exp(
        2j
        * np.pi
        * f[None, None, :]
        * slownesses[:, None, None]
        * rel_off[None, :, None]
    )  # (n_p, n_rec, n_f)
    panel_spec = (spec[None, :, :] * phase).sum(axis=1)  # (n_p, n_f)
    panel = np.fft.irfft(panel_spec, n=n_samp, axis=1)
    return panel


def tau_p_adjoint(
    panel: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slownesses: np.ndarray,
) -> np.ndarray:
    r"""
    Adjoint of :func:`tau_p_forward` (smeared back-projection).

    Sums each slowness slice of the panel back along its corresponding
    moveout line in (t, x)::

        \hat d(t, x_i) = \frac{1}{N_{slowness}} \sum_p
                         panel(t - p \cdot x_i, p)

    The :math:`1/N_{slowness}` factor is a Riemann-sum approximation of
    a slowness integral; it keeps the adjoint output's order of
    magnitude comparable to the input data rather than scaling with
    the slowness-grid density.

    The adjoint is *not* the inverse of :func:`tau_p_forward` --
    Radon is non-unitary, and adjoint-of-forward smears each event
    by the rectangular-aperture sinc kernel. Use it as the back-end
    of a slowness-band filter (:func:`tau_p_filter`), where the
    smearing is acceptable; for an exact round-trip use
    :func:`tau_p_inverse` instead.

    Parameters
    ----------
    panel : ndarray, shape (n_slowness, n_samples)
    dt : float
    offsets : ndarray, shape (n_rec,)
    slownesses : ndarray, shape (n_slowness,)

    Returns
    -------
    ndarray, shape (n_rec, n_samples)
    """
    n_p, n_samp = panel.shape
    offsets = np.asarray(offsets, dtype=float)
    slownesses = np.asarray(slownesses, dtype=float)
    if slownesses.size != n_p:
        raise ValueError("slownesses must have length n_slowness")
    M = np.fft.rfft(panel, axis=1)  # (n_p, n_f)
    f = np.fft.rfftfreq(n_samp, d=dt)  # (n_f,)
    rel_off = offsets - offsets[0]  # (n_rec,)
    phase = np.exp(
        -2j
        * np.pi
        * f[None, None, :]
        * slownesses[:, None, None]
        * rel_off[None, :, None]
    )  # (n_p, n_rec, n_f)
    rec_spec = (M[:, None, :] * phase).sum(axis=0) / n_p  # (n_rec, n_f)
    return np.fft.irfft(rec_spec, n=n_samp, axis=1)


def tau_p_inverse(
    panel: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slownesses: np.ndarray,
    *,
    ridge: float = 1.0e-10,
) -> np.ndarray:
    r"""
    Inverse tau-p (slant-stack / linear Radon) transform.

    Solves the per-frequency least-squares problem

    .. math::

        \hat d(f, \cdot) =
        \arg\min_{d'} \, \| A_f \, d' - panel(f, \cdot) \|_2^2

    where :math:`A_f[p, i] = \exp(2\pi i f \, s_p \, x_i)` is the
    forward stacking matrix at frequency :math:`f`. Because each
    temporal frequency is independent in tau-p, the inverse problem
    decouples into ``n_f`` small linear systems of size
    ``n_slowness x n_rec``, each solved exactly via the
    Moore-Penrose pseudoinverse. With ``n_slowness > n_rec`` (the
    standard over-determined case for a sonic array) and the panel
    coming from :func:`tau_p_forward` the round-trip is identity to
    floating-point precision.

    Parameters
    ----------
    panel : ndarray, shape (n_slowness, n_samples)
        tau-p panel from :func:`tau_p_forward` (or any same-shape
        array, e.g. a slowness-band-masked panel for use as a
        band-pass filter back-end).
    dt : float
        Time sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m). Must match the offsets used
        in the forward transform.
    slownesses : ndarray, shape (n_slowness,)
        Slowness grid used in the forward transform.
    ridge : float, default 1e-10
        Tikhonov regularisation added to the per-frequency normal-
        equation diagonal. Prevents catastrophic blow-up when the
        slowness grid is too coarse to resolve all ``n_rec`` traces
        at a given frequency (which happens at very low frequencies
        where the rows of :math:`A_f` are nearly identical). Default
        is small enough to be invisible on well-conditioned cases
        but large enough to keep low-f bins finite.

    Returns
    -------
    ndarray, shape (n_rec, n_samples)
        Reconstructed gather.
    """
    n_p, n_samp = panel.shape
    offsets = np.asarray(offsets, dtype=float)
    slownesses = np.asarray(slownesses, dtype=float)
    if slownesses.size != n_p:
        raise ValueError("slownesses must have length n_slowness")
    n_rec = offsets.size
    M = np.fft.rfft(panel, axis=1)  # (n_p, n_f)
    f = np.fft.rfftfreq(n_samp, d=dt)  # (n_f,)
    rel_off = offsets - offsets[0]  # (n_rec,)

    # A[p, i, f] = exp(2 pi i f s_p x_i). Per-frequency this is the
    # forward stacking matrix; we batch the per-frequency systems and
    # solve them via the regularised normal equations.
    A = np.exp(
        2j
        * np.pi
        * f[None, None, :]
        * slownesses[:, None, None]
        * rel_off[None, :, None]
    )  # (n_p, n_rec, n_f)
    A_h = np.conj(A)  # (n_p, n_rec, n_f)
    # AhA[i, j, f] = sum_p A_h[p, i, f] * A[p, j, f]
    AhA = np.einsum("pif,pjf->fij", A_h, A)  # (n_f, n_rec, n_rec)
    # AhM[f, i] = sum_p A_h[p, i, f] * M[p, f]
    AhM = np.einsum("pif,pf->fi", A_h, M)  # (n_f, n_rec)

    # Tikhonov-regularised batch solve: x = (A^H A + eps I)^{-1} A^H M.
    # ``np.linalg.solve`` wants the rhs as an (..., n_rec, k) column-
    # vector batch; we add a trailing axis and squeeze it back.
    eye = np.eye(n_rec) * ridge
    rec_spec_t = np.linalg.solve(
        AhA + eye[None, :, :],
        AhM[..., None],
    ).squeeze(-1)  # (n_f, n_rec)
    rec_spec = rec_spec_t.T  # (n_rec, n_f)

    return np.fft.irfft(rec_spec, n=n_samp, axis=1)


def tau_p_filter(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slow_min: float,
    slow_max: float,
    *,
    n_slowness: int = 181,
    slowness_pad_factor: float = 1.5,
    taper_width: float = 0.1,
) -> np.ndarray:
    """
    Apparent-slowness band-pass filter via tau-p forward / mask /
    inverse.

    Mirror of :func:`fk_filter` in the linear-Radon domain: the
    multichannel record is forward-transformed to (tau, p), a
    cosine-tapered mask keeps slownesses in ``[slow_min, slow_max]``
    (with a wider grid around it for clean edge behaviour), and
    :func:`tau_p_adjoint` projects back to (t, x).

    Differences vs :func:`fk_filter`
    --------------------------------
    * tau-p does **not** require uniform offsets; ``fk_filter`` does.
    * tau-p uses the adjoint as the back-end (the LSQR-inverse from
      :func:`tau_p_inverse` does not commute with masking, so
      using it here would amplify the masked panel rather than
      band-pass it). Out-of-band rejection is set by the slowness-
      grid aperture and the cosine taper rather than by a strict
      orthogonality, and amplitudes are smeared by the slant-stack
      point-spread function (event amplitude is preserved up to a
      modest aperture-dependent constant, *not* exactly).

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
        Real-valued gather.
    dt : float
        Time sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m). Need not be uniform.
    slow_min, slow_max : float
        Pass-band edges in apparent slowness (s/m). Must satisfy
        ``0 < slow_min < slow_max``.
    n_slowness : int, default 181
        Number of slowness samples in the tau-p grid.
    slowness_pad_factor : float, default 1.5
        Extra slowness coverage outside the pass-band, expressed as
        a multiple of the pass-band width. The slowness grid spans
        ``[slow_min - pad, slow_max + pad]`` with
        ``pad = slowness_pad_factor * (slow_max - slow_min)``,
        clipped at zero on the lower side. Wider padding gives a
        better-conditioned linear Radon frame at the cost of a
        denser slowness grid.
    taper_width : float, default 0.1
        Width of the half-cosine ramp on each pass-band edge,
        expressed as a fraction of the pass-band width. ``0.0``
        gives a hard rectangular mask.

    Returns
    -------
    ndarray, shape (n_rec, n_samples)
    """
    if not (slow_max > slow_min > 0):
        raise ValueError("require 0 < slow_min < slow_max")
    pad = slowness_pad_factor * (slow_max - slow_min)
    s_lo = max(slow_min - pad, 1.0e-12)
    s_hi = slow_max + pad
    slownesses = np.linspace(s_lo, s_hi, n_slowness)

    panel = tau_p_forward(data, dt, offsets, slownesses)

    # Cosine-tapered band-pass mask along the slowness axis.
    mask: np.ndarray = np.zeros(n_slowness, dtype=float)
    in_band = (slownesses >= slow_min) & (slownesses <= slow_max)
    mask[in_band] = 1.0
    if taper_width > 0:
        w = taper_width * (slow_max - slow_min)
        lo = (slownesses >= slow_min - w) & (slownesses < slow_min)
        hi = (slownesses > slow_max) & (slownesses <= slow_max + w)
        mask[lo] = 0.5 * (1.0 - np.cos(np.pi * (slownesses[lo] - (slow_min - w)) / w))
        mask[hi] = 0.5 * (1.0 + np.cos(np.pi * (slownesses[hi] - slow_max) / w))

    return tau_p_adjoint(panel * mask[:, None], dt, offsets, slownesses)


def apply_moveout(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slowness: float,
    reference: int = 0,
) -> np.ndarray:
    r"""
    Flatten a gather on a trial slowness using frequency-domain
    fractional shifts.

    Sign convention
    ---------------
    Trace ``i`` carries a mode with arrival at ``t_i = offset_i * s``
    (plus a constant). To flatten the gather at the reference offset
    we need to *advance* trace ``i`` by ``(offset_i - offset_ref) * s``
    (earlier-arriving traces are left alone; later-arriving traces are
    shifted earlier). In FFT convention :math:`X(f)=\sum x(n)e^{-2\pi
    i fn}`, a time advance by ``tau`` corresponds to multiplication by
    :math:`e^{+2\pi i f \tau}` (see :func:`fwap._common._phase_shift`).
    """
    n_rec, n_samp = data.shape
    if offsets.size != n_rec:
        raise ValueError("offsets must have length n_rec")
    spec = np.fft.rfft(data, axis=1)
    f = np.fft.rfftfreq(n_samp, d=dt)
    # tau[i] is the per-trace time advance (seconds) needed to flatten
    # an arrival of the given slowness onto the reference offset.
    tau = (offsets - offsets[reference]) * slowness  # (n_rec,)
    shifted = _phase_shift(spec, f, tau)
    return np.fft.irfft(shifted, n=n_samp, axis=1)


def unapply_moveout(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slowness: float,
    reference: int = 0,
) -> np.ndarray:
    """Inverse of ``apply_moveout`` -- restore the original moveout."""
    return apply_moveout(data, dt, offsets, -slowness, reference=reference)


def svd_project(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slowness: float,
    rank: int = 1,
    reference: int = 0,
    *,
    n_keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SVD / Karhunen-Loeve separation.

    Keeps the rank-``rank`` coherent part of ``data`` that travels at
    the given apparent slowness; returns ``(coherent, residual)``.

    The ``rank`` parameter is the number of singular components kept
    (the "K" in K-L decomposition); for borehole sonic data a single
    component captures the dominant coherent arrival.

    ``n_keep`` is accepted as a deprecated alias for ``rank`` to keep
    older call sites working. Pass it as a keyword if you must;
    ``rank`` takes precedence when both are supplied.

    Because ``apply_moveout`` now correctly flattens the gather, the
    largest singular vector of the moveout-corrected matrix genuinely
    represents the coherent arrival at that slowness (Freire & Ulrych,
    1988, *Geophysics* 53(6), 778-785).
    """
    if n_keep is not None and rank == 1:
        rank = int(n_keep)
    flat = apply_moveout(data, dt, offsets, slowness, reference=reference)
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    S_k = np.zeros_like(S)
    S_k[:rank] = S[:rank]
    coh_flat = (U * S_k) @ Vt
    coh = unapply_moveout(coh_flat, dt, offsets, slowness, reference=reference)
    return coh, data - coh


def sequential_kl_separation(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slownesses: Sequence[float],
    rank: int = 1,
    reference: int = 0,
    *,
    n_keep: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Peel off coherent modes one slowness at a time.

    ``rank`` is forwarded to :func:`svd_project`; ``n_keep`` is the
    deprecated alias kept for backward compatibility.
    """
    if n_keep is not None and rank == 1:
        rank = int(n_keep)
    work = data.copy()
    comps: list[np.ndarray] = []
    for s in slownesses:
        c, work = svd_project(work, dt, offsets, s, rank=rank, reference=reference)
        comps.append(c)
    return comps, work
