"""
Intercept-time (delay) inversion for borehole acoustic logs.

Implements the algorithmic spine of Part 3 of the book: the borehole
adaptation of refraction-seismic intercept-time analysis. The least-
squares system jointly solves for a "virgin" formation slowness log and
per-depth source/receiver delays, giving access to the near-borehole
altered zone.

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 3. Editions Technip,
  Paris. ISBN 978-2-7108-0664-6.
* Coppens, F., & Mari, J.-L. (1995). Application of the intercept time
  method to full waveform acoustic data. *First Break* 13(1), 11-20.
* Coppens, F., & Mari, J.-L. (1995). Imagerie par refraction en
  diagraphie acoustique. *Revue de l'Institut Francais du Petrole*
  50(2), 143.
* Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model
  Parameter Estimation*, Chapter 3. SIAM.
* Aster, R. C., Borchers, B., & Thurber, C. H. (2018). *Parameter
  Estimation and Inverse Problems*, 3rd ed., Chapter 4. Elsevier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

InterceptTimeMethod = Literal["midpoint", "segmented"]


@dataclass
class InterceptTimeResult:
    """
    Output of :func:`solve_intercept_time`.

    Attributes
    ----------
    depths, slowness, delay_src, delay_rec
        Depth axis and the three unknown columns.
    rms_residual
        RMS of ``G @ m - d`` on the observation rows only (i.e. not
        counting regularisation rows).
    sigma_slowness, sigma_delay_src, sigma_delay_rec
        Marginal 1-sigma posterior standard errors, computed from
        ``sigma_d**2 * diag((G_aug.T @ G_aug)^-1)``, where
        ``sigma_d = rms_residual``. A principled Bayesian treatment
        would carry independent prior variances and a full covariance
        (see Tarantola, 2005, sect. 3.2, or Aster, Borchers & Thurber,
        2018, chap. 4).
    method
        Design matrix style used (``"midpoint"`` or ``"segmented"``).
    """

    depths: np.ndarray
    slowness: np.ndarray
    delay_src: np.ndarray
    delay_rec: np.ndarray
    rms_residual: float
    sigma_slowness: np.ndarray
    sigma_delay_src: np.ndarray
    sigma_delay_rec: np.ndarray
    method: InterceptTimeMethod = "midpoint"

    def __repr__(self) -> str:
        n = self.depths.size
        if n == 0:
            slow_summary = "no depths"
        else:
            mean_slow = float(np.nanmean(self.slowness)) * 1.0e6 * 0.3048
            slow_summary = f"<s>={mean_slow:.1f} us/ft"
        return (
            f"InterceptTimeResult(method={self.method!r}, "
            f"n_depths={n}, {slow_summary}, "
            f"rms={self.rms_residual * 1e6:.2f} us)"
        )


def build_design_matrix(
    travel_times: np.ndarray,
    offsets: np.ndarray,
    src_depth_idx: np.ndarray,
    rec_depth_idx: np.ndarray,
    n_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Midpoint design matrix.

    Unknown vector
    ``m = [ S(z_0..N-1) | d_src(z_0..N-1) | d_rec(z_0..N-1) ]``.
    Each observation row assigns the full offset length to the single
    midpoint cell between source and receiver, and picks one delay from
    the source and one from the receiver block.

    This is fast and requires only the pick indices, but biases the
    slowness estimate when offsets are not small compared to the cell
    thickness ``dz``; see :func:`build_design_matrix_segmented` for the
    ray-path form.

    Parameters
    ----------
    travel_times : ndarray, shape (n_obs,)
        Picked travel times (s).
    offsets : ndarray, shape (n_obs,)
        Source-to-receiver offset (m) for each observation.
    src_depth_idx, rec_depth_idx : ndarray, shape (n_obs,)
        Integer indices into the depth cell grid for the source and
        receiver endpoints.
    n_depth : int
        Number of depth cells.

    Returns
    -------
    G : ndarray, shape (n_obs, 3 * n_depth)
    d : ndarray, shape (n_obs,)
        Copy of ``travel_times``.
    """
    n_obs = travel_times.size
    G: np.ndarray = np.zeros((n_obs, 3 * n_depth), dtype=float)
    mid = np.clip(((src_depth_idx + rec_depth_idx) // 2).astype(int), 0, n_depth - 1)
    r = np.arange(n_obs)
    G[r, mid] = offsets
    G[r, n_depth + src_depth_idx.astype(int)] = 1.0
    G[r, 2 * n_depth + rec_depth_idx.astype(int)] = 1.0
    return G, travel_times.copy()


# Target roughly 1 MB (~L2 cache) per broadcast chunk at float64 in
# ``build_design_matrix_segmented``. Keeps the hot temporary in cache
# and minimises memory-bandwidth stalls on large problems.
_CHUNK_TARGET_BYTES: int = 1 << 20  # 1 MiB


def _default_chunk_size(n_cells: int) -> int:
    """Pick a chunk size that keeps the (chunk, n_cells) temp near L2."""
    elems = max(1, _CHUNK_TARGET_BYTES // 8)  # float64
    return max(1, elems // max(1, n_cells))


def build_design_matrix_segmented(
    travel_times: np.ndarray,
    src_depths: np.ndarray,
    rec_depths: np.ndarray,
    cell_depths: np.ndarray,
    *,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segmented (1-D tomography) design matrix.

    Unknown vector layout (identical to :func:`build_design_matrix`)::

        m = [ S(z_0..N-1) | d_src(z_0..N-1) | d_rec(z_0..N-1) ]

    For each observation the ray path from ``src_depths[i]`` to
    ``rec_depths[i]`` is split across every depth cell it traverses;
    each cell contributes ``s_cell * path_length``. Source and receiver
    delay columns are indicator columns at the nearest cell to each
    endpoint.

    This is the correct form when offsets are not small compared to the
    cell thickness ``dz``. Source and receiver delays are still picked
    at the nearest cell (the delay stack is separate from the slowness
    stack -- cf. Coppens & Mari, 1995, *First Break* 13(1), 14-16).

    Parameters
    ----------
    travel_times : ndarray, shape (n_obs,)
        Picked travel-times (s).
    src_depths, rec_depths : ndarray, shape (n_obs,)
        Source and receiver depths (m). Ordering within a pair is
        irrelevant; the shallower endpoint is taken as the start of the
        ray.
    cell_depths : ndarray, shape (n_cells,)
        Cell centres (m). Must be strictly increasing and uniformly
        spaced to within 1e-6 relative tolerance on the first
        difference.
    chunk_size : int, optional
        Number of observations processed per broadcast chunk. Defaults
        to a cache-friendly size chosen from :data:`_CHUNK_TARGET_BYTES`
        and ``n_cells``.

    Returns
    -------
    G : ndarray, shape (n_obs, 3 * n_cells)
    d : ndarray, shape (n_obs,)
        Copy of ``travel_times``.

    Raises
    ------
    ValueError
        If any input has an incompatible shape, if the cell grid is not
        strictly increasing, or if the grid is not uniformly spaced.

    Notes
    -----
    * A depth exactly on a cell boundary is assigned to the **higher**
      of the two adjacent cells.
    * Rays whose endpoints fall outside the cell grid contribute zero
      to the slowness block for the out-of-grid portion; their delay
      indicators are clipped to the nearest in-grid cell.
    """
    travel_times = np.ascontiguousarray(travel_times, dtype=float)
    src_depths = np.ascontiguousarray(src_depths, dtype=float)
    rec_depths = np.ascontiguousarray(rec_depths, dtype=float)
    cell_depths = np.ascontiguousarray(cell_depths, dtype=float)

    n_obs = travel_times.size
    n_cells = cell_depths.size
    if n_cells < 2:
        raise ValueError("cell_depths must have >= 2 samples")
    if src_depths.size != n_obs or rec_depths.size != n_obs:
        raise ValueError(
            "travel_times, src_depths, rec_depths must have the same length"
        )

    diffs = np.diff(cell_depths)
    if np.any(diffs <= 0):
        raise ValueError("cell_depths must be strictly increasing")
    dz = float(diffs.mean())
    if not np.allclose(diffs, dz, rtol=1e-6, atol=0.0):
        raise ValueError("cell_depths must be uniformly spaced")

    z_lo = cell_depths - 0.5 * dz  # (n_cells,)
    z_hi = cell_depths + 0.5 * dz  # (n_cells,)
    edges = np.concatenate([z_lo, z_hi[-1:]])  # (n_cells + 1,)

    G: np.ndarray = np.zeros((n_obs, 3 * n_cells), dtype=float)

    # --- Slowness block: overlap of each ray with each cell ------------
    z1 = np.minimum(src_depths, rec_depths)
    z2 = np.maximum(src_depths, rec_depths)

    if chunk_size is None:
        chunk_size = _default_chunk_size(n_cells)

    # Scratch buffer reused across chunks; holds max(z1, z_lo).
    scratch: np.ndarray = np.empty((min(chunk_size, n_obs), n_cells), dtype=float)
    z_lo_row = z_lo[None, :]  # (1, n_cells)
    z_hi_row = z_hi[None, :]  # (1, n_cells)
    for start in range(0, n_obs, chunk_size):
        stop = min(n_obs, start + chunk_size)
        k = stop - start
        z1_col = z1[start:stop, None]  # (k, 1)
        z2_col = z2[start:stop, None]  # (k, 1)
        out_view = G[start:stop, :n_cells]  # (k, n_cells)
        np.minimum(z2_col, z_hi_row, out=out_view)
        np.maximum(z1_col, z_lo_row, out=scratch[:k])
        np.subtract(out_view, scratch[:k], out=out_view)
        np.clip(out_view, 0.0, None, out=out_view)

    # --- Delay blocks: indicator columns at the nearest cell -----------
    # searchsorted(edges, x, side='right') - 1 returns the index of the
    # cell whose half-open interval [z_lo, z_hi) contains x. Depths on
    # an interior edge land in the higher cell, equivalent to
    # ((x - z0)/dz + 0.5).astype(int) rounding.
    src_idx = np.clip(
        np.searchsorted(edges, src_depths, side="right") - 1,
        0,
        n_cells - 1,
    )
    rec_idx = np.clip(
        np.searchsorted(edges, rec_depths, side="right") - 1,
        0,
        n_cells - 1,
    )
    r = np.arange(n_obs)
    G[r, n_cells + src_idx] = 1.0
    G[r, 2 * n_cells + rec_idx] = 1.0
    return G, travel_times.copy()


def solve_intercept_time(
    travel_times: np.ndarray,
    offsets: np.ndarray,
    src_depth_idx: np.ndarray,
    rec_depth_idx: np.ndarray,
    n_depth: int,
    depth_axis: np.ndarray | None = None,
    mean_delay_zero: bool = True,
    smooth_s: float = 0.0,
    smooth_src: float = 0.0,
    smooth_rec: float = 0.0,
    delay_l2: float = 0.0,
    method: InterceptTimeMethod = "midpoint",
) -> InterceptTimeResult:
    """
    Least-squares intercept-time inversion with regularisation and
    posterior standard errors.

    Parameters
    ----------
    method : {"midpoint", "segmented"}
        If ``"midpoint"``, the whole offset is assigned to the
        midpoint cell -- fast but biased at long offsets. If
        ``"segmented"``, the offset is split across every cell the
        ray traverses; in that case ``src_depth_idx`` and
        ``rec_depth_idx`` are reinterpreted as *depths* in metres
        (they are still named ``_idx`` for API continuity).
    smooth_s, smooth_src, smooth_rec : float, default 0.0
        Per-block first-difference regularisation weights. Setting any
        of these > 0 penalises abrupt depth-to-depth changes in the
        corresponding unknown (formation slowness, source delay, or
        receiver delay). Default 0 gives an unregularised
        least-squares fit.
    mean_delay_zero : bool, default True
        Pin the mean of the source-delay block and of the receiver-delay
        block to zero, independently. Two separate rows are added to the
        augmented system: one constraining ``sum(d_src) = 0`` and one
        constraining ``sum(d_rec) = 0``. This resolves the global shift
        gauge (``s -> s + c``, ``d_src -> d_src - c * offset_mean`` etc.)
        and the inter-block swap gauge
        (``d_src -> d_src + c``, ``d_rec -> d_rec - c``) that together
        leave the problem rank-deficient otherwise.
    delay_l2 : float
        Weak L2 prior on the two delay blocks, pulling them towards
        zero. Useful when the true delays are expected to be small
        (zero mean, low variance): even after ``mean_delay_zero`` fixes
        the gauges, a weak L2 prior helps condition the system when the
        delay structure is subtle.

    Returns
    -------
    InterceptTimeResult
        Including marginal posterior standard errors for each block.
    """
    if depth_axis is None:
        depth_axis = np.arange(n_depth, dtype=float)

    if method == "midpoint":
        G, d = build_design_matrix(
            travel_times,
            offsets,
            np.asarray(src_depth_idx, dtype=int),
            np.asarray(rec_depth_idx, dtype=int),
            n_depth,
        )
    elif method == "segmented":
        G, d = build_design_matrix_segmented(
            travel_times,
            np.asarray(src_depth_idx, dtype=float),
            np.asarray(rec_depth_idx, dtype=float),
            np.asarray(depth_axis, dtype=float),
        )
    else:
        raise ValueError("method must be 'midpoint' or 'segmented'")

    ws = smooth_s
    wr = smooth_src
    wc = smooth_rec

    extra_rows: list[np.ndarray] = []
    extra_d: list[float] = []
    if mean_delay_zero:
        # Zero the mean of the source-delay and receiver-delay blocks
        # *separately*. A single joint row (as in fwap <= 0.4.0) only
        # constrains the sum d_src + d_rec, leaving the gauge
        # (d_src, d_rec) -> (d_src + c, d_rec - c) unresolved. Splitting
        # the constraint matches Coppens & Mari (1995), First Break
        # 13(1), section 2.
        row_src = np.zeros(3 * n_depth)
        row_src[n_depth : 2 * n_depth] = 1.0
        extra_rows.append(row_src)
        extra_d.append(0.0)

        row_rec = np.zeros(3 * n_depth)
        row_rec[2 * n_depth : 3 * n_depth] = 1.0
        extra_rows.append(row_rec)
        extra_d.append(0.0)

    for block, w in enumerate([ws, wr, wc]):
        if w <= 0:
            continue
        for i in range(n_depth - 1):
            row = np.zeros(3 * n_depth)
            row[block * n_depth + i] = +w
            row[block * n_depth + i + 1] = -w
            extra_rows.append(row)
            extra_d.append(0.0)

    # Optional weak L2 prior on delays: helps break the (s, delay)
    # null space when the true delays are small. See Tarantola (2005),
    # sect. 3.3, on weighting priors against data.
    if delay_l2 > 0:
        for block in (1, 2):
            for i in range(n_depth):
                row = np.zeros(3 * n_depth)
                row[block * n_depth + i] = delay_l2
                extra_rows.append(row)
                extra_d.append(0.0)

    if extra_rows:
        G_aug = np.vstack([G, np.array(extra_rows)])
        d_aug = np.concatenate([d, np.array(extra_d)])
    else:
        G_aug, d_aug = G, d

    m, *_ = np.linalg.lstsq(G_aug, d_aug, rcond=None)
    slowness = m[:n_depth]
    delay_src = m[n_depth : 2 * n_depth]
    delay_rec = m[2 * n_depth : 3 * n_depth]

    residual = G @ m - d
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0

    # Marginal posterior standard errors.
    # cov = sigma_d**2 * (G_aug^T G_aug)^(-1), clipped for conditioning.
    try:
        GtG_inv = np.linalg.pinv(G_aug.T @ G_aug, rcond=1e-10)
        cov_diag = np.clip(np.diag(GtG_inv), 0.0, None)
        sigma = rms * np.sqrt(cov_diag)
    except np.linalg.LinAlgError:
        sigma = np.full(3 * n_depth, np.nan)

    return InterceptTimeResult(
        depths=np.asarray(depth_axis),
        slowness=slowness,
        delay_src=delay_src,
        delay_rec=delay_rec,
        rms_residual=rms,
        sigma_slowness=sigma[:n_depth],
        sigma_delay_src=sigma[n_depth : 2 * n_depth],
        sigma_delay_rec=sigma[2 * n_depth : 3 * n_depth],
        method=method,
    )


def assemble_observations_from_picks(
    tool_depths: np.ndarray, offsets: np.ndarray, first_arrivals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Convert an ``(n_depth, n_rec)`` first-arrival pick matrix into the
    ragged observation form consumed by :func:`solve_intercept_time`
    (``method="midpoint"``).

    Observations whose mapped receiver cell falls outside the depth
    grid are dropped rather than clipped; this keeps the assembled
    design matrix well-conditioned at the edges of the logged
    interval.

    Parameters
    ----------
    tool_depths : ndarray, shape (n_depth,)
        Tool (source) depth for each shot (m). Must have at least two
        entries and be approximately uniformly spaced -- ``dz`` is
        taken as the mean of ``np.diff(tool_depths)``.
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m), positive downhole.
    first_arrivals : ndarray, shape (n_depth, n_rec)
        First-break travel times (s), one row per tool depth.

    Returns
    -------
    travel_times : ndarray, shape (n_obs,)
        Selected first-arrival picks.
    offsets_out : ndarray, shape (n_obs,)
        Offset for each observation.
    src_idx, rec_idx : ndarray, shape (n_obs,) of int
        Indices into ``depth_axis`` for the source and receiver
        endpoints of each observation; ``rec_idx`` is computed from
        ``round((tool_depth + offset - z0) / dz)``.
    n_depth : int
    depth_axis : ndarray, shape (n_depth,)
        Copy of ``tool_depths``, returned as the cell-centre axis.
    """
    n_depth, n_rec = first_arrivals.shape
    if tool_depths.size != n_depth or offsets.size != n_rec:
        raise ValueError("shape mismatch")
    if n_depth < 2:
        raise ValueError("need at least two depths")
    dz = float(np.mean(np.diff(tool_depths)))
    z0 = float(tool_depths[0])
    depth_axis = tool_depths.copy()
    tt: list[float] = []
    xs: list[float] = []
    src: list[int] = []
    rec: list[int] = []
    for j in range(n_depth):
        for k, x in enumerate(offsets):
            z_rec = tool_depths[j] + x
            idx = int(round((z_rec - z0) / dz))
            if 0 <= idx < n_depth:
                tt.append(float(first_arrivals[j, k]))
                xs.append(x)
                src.append(j)
                rec.append(idx)
    return (
        np.asarray(tt),
        np.asarray(xs),
        np.asarray(src, dtype=int),
        np.asarray(rec, dtype=int),
        n_depth,
        depth_axis,
    )


def delay_to_altered_zone_thickness(
    delay: np.ndarray,
    slowness_virgin: float,
    slowness_altered: float,
) -> np.ndarray:
    r"""
    Convert an intercept-time delay to an altered-zone thickness.

    First-order borehole refraction geometry: the ray leaves the
    source, refracts into the damaged zone at slowness
    ``s_altered``, travels along the borehole wall, and refracts back
    out at the receiver. Relative to the bulk-formation (``s_virgin``)
    reference, the extra traveltime spent inside the altered layer is

    .. math::

        \Delta t = 2 \, h \, (s_\text{altered} - s_\text{virgin})

    for a single-sided altered-zone thickness ``h`` (metres). Solving
    for ``h``:

    .. math::

        h = \frac{\Delta t}{2 \, (s_\text{altered} - s_\text{virgin})}

    This is the standard first-order conversion behind the "damaged-
    halo thickness" log described in Coppens & Mari (1995), section 3.
    It assumes (a) the altered zone is thin compared to the source-
    receiver offset, (b) the altered-zone slowness is well-estimated
    from an ancillary log (e.g. a shallow-reading resistivity
    interpretation or an assumed fractional velocity reduction), and
    (c) the delay column is one-sided (either ``delay_src`` or
    ``delay_rec`` from :func:`solve_intercept_time`, not their sum).

    Single-delay under-determination
    --------------------------------
    Each per-depth delay :math:`\Delta t` is a single equation in two
    unknowns :math:`(h, \, s_\text{altered})`. This routine pins
    :math:`s_\text{altered}` and solves for :math:`h`; the dual
    direction -- pin :math:`h`, solve for the slowness contrast --
    is :func:`delay_to_altered_zone_velocity_contrast`. The book
    (Mari et al. 1994, Part 3) frames the workflow output as the
    full ``(thickness, velocity-contrast)`` pair; use
    :func:`altered_zone_estimate` to obtain both at once given
    either anchor.

    Parameters
    ----------
    delay : ndarray or float
        Per-depth delay time (s), typically
        :attr:`InterceptTimeResult.delay_src` or
        :attr:`InterceptTimeResult.delay_rec`.
    slowness_virgin : float
        Undisturbed-formation slowness (s/m).
    slowness_altered : float
        Damaged-zone slowness (s/m). Must be strictly greater than
        ``slowness_virgin`` (a damaged zone is slower, by definition).

    Returns
    -------
    ndarray or float
        Altered-zone thickness (m), same shape as ``delay``. Values
        are clipped to be non-negative -- negative delays (which can
        arise from inversion noise in the virgin zones) map to a zero
        thickness.

    Raises
    ------
    ValueError
        If ``slowness_altered <= slowness_virgin``.

    References
    ----------
    Coppens, F., & Mari, J.-L. (1995). Application of the intercept
    time method to full waveform acoustic data. *First Break* 13(1),
    11-20 (section 3: altered-zone characterisation).
    """
    if slowness_altered <= slowness_virgin:
        raise ValueError(
            "slowness_altered must be greater than slowness_virgin "
            "(a damaged zone is slower than the bulk formation)"
        )
    delay_arr = np.asarray(delay, dtype=float)
    thickness = delay_arr / (2.0 * (slowness_altered - slowness_virgin))
    return np.clip(thickness, 0.0, None)


def delay_to_altered_zone_velocity_contrast(
    delay: np.ndarray,
    thickness: float | np.ndarray,
) -> np.ndarray:
    r"""
    Convert an intercept-time delay to an altered-zone slowness
    contrast.

    Algebraic dual of :func:`delay_to_altered_zone_thickness`: pins
    the altered-zone thickness :math:`h` and solves for the slowness
    contrast :math:`\Delta s = s_\text{altered} - s_\text{virgin}`
    from the same first-order refraction relation

    .. math::

        \Delta t = 2 \, h \, \Delta s
        \quad\Longrightarrow\quad
        \Delta s = \frac{\Delta t}{2 h}.

    The absolute altered-zone slowness is recovered by adding back
    the virgin slowness:
    :math:`s_\text{altered} = s_\text{virgin} + \Delta s`.

    Parameters
    ----------
    delay : ndarray or float
        Per-depth delay time (s).
    thickness : float or ndarray
        Assumed altered-zone thickness (m). A scalar pins the same
        thickness at every depth; a same-shape array allows a
        depth-varying anchor (e.g. driven by an image-log halo
        thickness). Must be strictly positive at every depth.

    Returns
    -------
    ndarray
        Slowness contrast (s/m), same shape as ``delay``. Values are
        clipped to be non-negative -- negative delays (inversion
        noise in the virgin zones) map to a zero contrast.

    Raises
    ------
    ValueError
        If any ``thickness`` value is non-positive.
    """
    thickness_arr = np.asarray(thickness, dtype=float)
    if np.any(thickness_arr <= 0.0):
        raise ValueError("thickness must be strictly positive")
    delay_arr = np.asarray(delay, dtype=float)
    contrast = delay_arr / (2.0 * thickness_arr)
    return np.clip(contrast, 0.0, None)


@dataclass
class AlteredZoneEstimate:
    """
    Joint altered-zone characterisation produced by
    :func:`altered_zone_estimate`.

    Each attribute is a per-depth array (or a scalar if the input
    delay was scalar). The triple is internally consistent under
    :math:`\\Delta t = 2 h \\Delta s` at every depth.

    Attributes
    ----------
    thickness : ndarray
        Altered-zone thickness (m).
    slowness_altered : ndarray
        Absolute altered-zone slowness (s/m).
    slowness_contrast : ndarray
        Altered-zone minus virgin slowness (s/m).
    """

    thickness: np.ndarray
    slowness_altered: np.ndarray
    slowness_contrast: np.ndarray


def altered_zone_estimate(
    delay: np.ndarray,
    slowness_virgin: float,
    *,
    thickness: float | np.ndarray | None = None,
    slowness_altered: float | None = None,
) -> AlteredZoneEstimate:
    r"""
    Joint ``(thickness, velocity-contrast)`` altered-zone estimate.

    The Workflow-2 deliverable per Mari et al. (1994), Part 3 is the
    altered-zone *thickness* and *velocity contrast* together; the
    refraction-geometry relation
    :math:`\Delta t = 2 h \, (s_\text{altered} - s_\text{virgin})`
    is one equation in two unknowns, so to deliver the pair the
    caller must pin one of them and the routine derives the other.

    Pass exactly one of ``thickness`` (a scalar or per-depth array
    altered-zone thickness in metres) or ``slowness_altered`` (a
    scalar absolute altered-zone slowness in s/m).

    Parameters
    ----------
    delay : ndarray or float
        Per-depth delay time (s) -- typically
        :attr:`InterceptTimeResult.delay_src` or
        :attr:`InterceptTimeResult.delay_rec`.
    slowness_virgin : float
        Undisturbed-formation slowness (s/m).
    thickness : float or ndarray, optional
        Anchor: assumed altered-zone thickness (m).
    slowness_altered : float, optional
        Anchor: assumed absolute altered-zone slowness (s/m). Must
        be strictly greater than ``slowness_virgin``.

    Returns
    -------
    AlteredZoneEstimate
        ``thickness``, ``slowness_altered`` and ``slowness_contrast``
        per depth; consistent under
        ``delay = 2 * thickness * slowness_contrast``.

    Raises
    ------
    ValueError
        If both anchors or neither anchor is supplied; if the
        supplied ``thickness`` has any non-positive entry; or if
        ``slowness_altered <= slowness_virgin``.
    """
    if (thickness is None) == (slowness_altered is None):
        raise ValueError(
            "exactly one of `thickness` or `slowness_altered` must be "
            "given (the (h, Δs) pair is otherwise under-determined "
            "from a single delay measurement)"
        )
    delay_arr = np.asarray(delay, dtype=float)
    if thickness is not None:
        contrast = delay_to_altered_zone_velocity_contrast(delay_arr, thickness)
        thickness_arr = np.broadcast_to(
            np.asarray(thickness, dtype=float), delay_arr.shape
        ).copy()
        s_altered_arr = slowness_virgin + contrast
    else:
        # slowness_altered is not None per the XOR check above.
        assert slowness_altered is not None
        thickness_arr = delay_to_altered_zone_thickness(
            delay_arr, slowness_virgin, slowness_altered
        )
        contrast = np.broadcast_to(
            np.asarray(slowness_altered - slowness_virgin, dtype=float),
            delay_arr.shape,
        ).copy()
        s_altered_arr = np.broadcast_to(
            np.asarray(slowness_altered, dtype=float),
            delay_arr.shape,
        ).copy()
    return AlteredZoneEstimate(
        thickness=thickness_arr,
        slowness_altered=s_altered_arr,
        slowness_contrast=np.asarray(contrast, dtype=float),
    )
