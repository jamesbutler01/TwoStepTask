"""
Mathematical and statistical functions for neural data analysis.
"""

import numpy as np


def nans(shape):
    """
    Create an array filled with NaN values.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the output array

    Returns
    -------
    out : ndarray
        Array filled with NaN values
    """
    out = np.empty(shape)
    out.fill(np.nan)
    return out


def sem(arr):
    """
    Calculate standard error of the mean.

    Parameters
    ----------
    arr : ndarray
        Input array (1D or 2D)

    Returns
    -------
    std : float or ndarray
        Standard error of the mean
    """
    if arr.ndim == 1:
        if sum(~np.isnan(arr)) == 0:
            return np.nan
        std = np.nanstd(arr)
        std /= np.sqrt(np.sum(~np.isnan(arr)))
    else:
        std = np.nanstd(arr, axis=0)
        std /= np.sqrt(np.sum(~np.isnan(arr[:, 0])))

    return std


def reg(x, y):
    """
    Perform linear regression using pseudoinverse.

    Parameters
    ----------
    x : ndarray
        Design matrix (n_samples x n_features)
    y : ndarray
        Response variable (n_samples,) or (n_samples, n_timepoints)

    Returns
    -------
    coefficients : ndarray
        Regression coefficients
    """
    return np.dot(np.linalg.pinv(x), y)


def cpd(x, y, i, adj=False):
    """
    Calculate coefficient of partial determination (CPD).

    CPD measures the proportion of variance in y explained by specific
    predictor(s) in x, controlling for all other predictors.

    Parameters
    ----------
    x : ndarray
        Design matrix (n_samples x n_features)
    y : ndarray
        Response variable (n_samples,) or (n_samples, n_timepoints)
    i : int or list of int
        Index or indices of predictor(s) to compute CPD for
    adj : bool, optional
        Whether to compute adjusted R-squared (not implemented in minimal version)

    Returns
    -------
    out : ndarray
        CPD values (as percentages) for each specified predictor
        Shape is (n_timepoints,) for single int i, or (len(i), n_timepoints) for list i
    """
    # Compute full model predictions
    bs = reg(x, y)
    pred = np.dot(x, bs)
    r = y - pred
    sse = np.sum(np.square(r), axis=0)

    # Track if input was a single integer
    single_index = isinstance(i, int)

    # Convert single index to list
    if single_index:
        i = [i]

    # Initialize output array
    if len(y.shape) == 1:
        out = np.empty((len(i)))
    else:
        out = np.empty((len(i), y.shape[1]))

    # Compute CPD for each specified predictor
    for iii, ii in enumerate(i):
        # Reduced model without predictor ii
        xred = np.delete(x, ii, axis=1)
        bs2 = reg(xred, y)
        predred = np.dot(xred, bs2)
        rred = y - predred

        # CPD = proportion of variance explained by including predictor ii
        rsqr = (np.sum(np.square(rred), axis=0) - sse) / np.sum(np.square(rred), axis=0)

        # Fill zero divides with NaN (near-zero threshold due to rounding errors)
        rsqr[np.sum(np.square(rred), axis=0) < 0.00001] = np.nan

        if adj:
            # Note: adjusted R-squared not implemented in minimal version
            # Would require rsqared_adj function
            pass

        # Convert to percentage
        out[iii] = rsqr * 100

    # If input was a single index, return 1D array
    if single_index and len(out.shape) > 1:
        out = out[0]

    return out
