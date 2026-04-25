"""
Plotting helpers for the fwap demos and for interactive use.

Two utilities:

* :func:`wiggle_plot` -- draw a wiggle plot of a multichannel gather
  (time axis in ms, receivers on the y-axis).
* :func:`save_figure` -- create the output directory if needed and
  write the figure to ``<figdir>/<name>``, with a log line recording
  where it went.

Both were previously available as ``fwap._plotting._wiggle`` and
``fwap._plotting._savefig``; those underscored names are kept as
aliases so third-party scripts importing them continue to work.
"""

from __future__ import annotations

import os

import numpy as np

from fwap._common import logger


def wiggle_plot(ax, data: np.ndarray, t: np.ndarray,
                scale: float = 1.5, xmax: float | None = None,
                title: str = "") -> None:
    """
    Draw a wiggle plot of a multichannel gather.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : ndarray, shape (n_rec, n_samples)
        Gather to plot. Each receiver is drawn as a trace offset
        vertically by its receiver index.
    t : ndarray, shape (n_samples,)
        Time axis (s). Displayed in milliseconds on the x-axis.
    scale : float, default 1.5
        Amplitude scale: each trace is scaled so that its peak is at
        most ``scale`` vertical units.
    xmax : float, optional
        If given, restrict the x-axis to ``[0, xmax * 1e3]`` ms.
    title : str, default empty
        Axes title.
    """
    g = scale / (np.max(np.abs(data)) + 1e-12)
    for i, tr in enumerate(data):
        ax.plot(t * 1e3, tr * g + i, "k", lw=0.7)
    if xmax is not None:
        ax.set_xlim(0, xmax * 1e3)
    ax.set_ylim(-0.7, data.shape[0] - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Receiver")
    ax.set_title(title)


def save_figure(fig, figdir: str, name: str, show: bool = False) -> None:
    """
    Save ``fig`` to ``<figdir>/<name>``, creating the directory if
    needed, and close or show the figure depending on ``show``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    figdir : str
        Output directory. Created with ``os.makedirs(..., exist_ok=True)``.
    name : str
        Filename under ``figdir``, including extension
        (typically ``.png``).
    show : bool, default False
        If ``True``, call ``plt.show()`` instead of ``plt.close(fig)``.
    """
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, name)
    fig.savefig(out, dpi=140)
    logger.info("  saved %s", out)
    import matplotlib.pyplot as plt
    if show:
        plt.show()
    else:
        plt.close(fig)


# Backwards-compatible underscore-prefixed aliases. The module used to
# be ``fwap._plotting`` with ``_wiggle`` / ``_savefig``; keep those
# names callable so external demo scripts do not break.
_wiggle = wiggle_plot
_savefig = save_figure
