"""
Plotting utility functions for neural data visualization.

This minimal version contains only the functions actually used in Fig2-6 scripts:
- AvgSem() - Plot average with standard error of the mean
- let() - Add panel letter labels to subplots
- set_xlim() - Set x-axis limits with proper tick labels
"""

import numpy as np

import Details


def AvgSem(arr, ax=None, c=None, ls=None, label=None, lw=None, do_sem=True,
           zorder=1, xrange=None, showsem=True, showleg=True, alpha=0.4):
    """
    Calculate and plot average with standard error of the mean.

    Parameters
    ----------
    arr : ndarray
        Input array (neurons x timepoints)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, only returns avg and sem without plotting
    c : str, optional
        Color for the line
    ls : str, optional
        Line style
    label : str, optional
        Label for legend
    lw : float, optional
        Line width
    do_sem : bool, default=True
        Whether to divide by sqrt(n) for standard error
    zorder : int, default=1
        Z-order for plotting layers
    xrange : array-like, optional
        X-axis values. If None, uses range(arr.shape[-1])
    showsem : bool, default=True
        Whether to show shaded SEM region
    showleg : bool, default=True
        Whether to show legend
    alpha : float, default=0.4
        Alpha transparency for SEM shading

    Returns
    -------
    avg : ndarray
        Mean across neurons
    sem : ndarray
        Standard error of the mean
    """
    if xrange is None:
        xrange = range(arr.shape[-1])

    arr = np.array(arr)
    avg = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0)

    # Convert to standard error of the mean
    if do_sem:
        for i in range(len(sem)):
            if np.sum(~np.isnan(arr[:, i])) > 0:
                sem[i] /= np.sqrt(np.sum(~np.isnan(arr[:, i])))

    # Plot if axes provided
    if ax is not None:
        ax.plot(xrange, avg, color=c, lw=lw, ls=ls, label=label, zorder=zorder)
        col = ax.get_lines()[-1].get_color()

        if showsem:
            ax.fill_between(xrange, avg - sem, avg + sem, color=col,
                            alpha=alpha, zorder=zorder - 1)

        if label is not None and showleg:
            ax.legend()

    return avg, sem


def let(ax, let, x=-0.2, y=1, fontsize=10):
    """
    Add panel letter labels (A, B, C, etc.) to subplot axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add label to
    let : int or str
        Panel letter. If int, converts to letter (0='A', 1='B', etc.)
        If str, uses directly
    x : float, default=-0.2
        X position in axes coordinates
    y : float, default=1
        Y position in axes coordinates
    fontsize : int, default=10
        Font size for the label
    """
    if not isinstance(let, str):
        let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'][let]

    ax.text(x, y, let,
            verticalalignment='top',
            horizontalalignment='right',
            transform=ax.transAxes,
            weight='bold',
            fontsize=fontsize)


def set_xlim(data, ax, minv=None, maxv=None, showTicks=True, y=False,
             res=500, offset=0):
    """
    Set x-axis limits and tick labels based on data properties.

    This function handles the conversion between data indices and actual
    time values (in milliseconds), accounting for pre-window smoothing.

    Parameters
    ----------
    data : object
        Data object with attributes:
        - res: resolution (bin size in ms)
        - smooth_prewindow: pre-window duration (ms)
        - smooth_postwindow: post-window duration (ms)
        - numTimepoints: number of timepoints
    ax : matplotlib.axes.Axes
        Axes to modify
    minv : int, optional
        Minimum time value (ms). If None, uses default from data
    maxv : int, optional
        Maximum time value (ms). If None, uses default from data
    showTicks : bool, default=True
        Whether to show tick labels
    y : bool, default=False
        If True, applies limits to y-axis instead of x-axis (inverted)
    res : int, default=500
        Resolution for tick spacing (ms)
    offset : int, default=0
        Time offset (ms). If 0, uses data.smooth_prewindow
    """
    if offset == 0:
        offset = data.smooth_prewindow

    # Set tick positions (in data indices)
    ax.set_xticks(range(offset // Details.smooth_step, data.numTimepoints + 1, res // Details.smooth_step))

    # Set tick labels (in milliseconds)
    if showTicks:
        ax.set_xticklabels(np.arange(-data.smooth_prewindow + offset,
                                     data.smooth_postwindow + 1, res))
    else:
        ax.set_xticklabels([''] * len(np.arange(-data.smooth_prewindow + offset,
                                                data.smooth_postwindow, res)))

    # Set axis limits if specified
    if minv is not None and maxv is not None:
        ax.set_xlim((minv + data.smooth_prewindow) // Details.smooth_step,
                    (maxv + data.smooth_prewindow) // Details.smooth_step)

    # Apply to y-axis if requested (inverted)
    if y:
        ax.set_ylim((maxv + data.smooth_prewindow) // Details.smooth_step,
                    (minv + data.smooth_prewindow) // Details.smooth_step)
