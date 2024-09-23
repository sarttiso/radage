import numpy as np
import pandas as pd
import scipy.stats as stats

"""
Give, two x,y curves this gives intersection points,
autor: Sukhbinder
5 April 2017
Based on: http://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections
"""
def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


def patch_dict_validator(patch_dict, n):
    """
    Validate patch_dict and returns list for styling of plotted patches.

    Parameters:
    -----------
    patch_dict : dict or list
        If a dictionary, same style will be used for all patches. If a list of dictionaries, must have length equal to n, and each dictionary will be used for each patch.
    n : int
        Number of patches to style

    Returns:
    --------
    patch_dict : list 
        validated list
    """
    # set up a default style
    patch_dict_def = {'facecolor': 'lightgray',
                      'linewidth': 1,
                      'edgecolor': 'k',
                      'alpha': 0.3}
    if patch_dict is None:
        patch_dict = n * [patch_dict_def]
    elif type(patch_dict) is dict:
        patch_dict = patch_dict_def | patch_dict
        patch_dict = n * [patch_dict]
    else:
        assert len(patch_dict) == n, 'Need one style dictionary per age.'

    return patch_dict


def epa_kern(u):
    """Epanechnikov kernel function.

    Parameters
    ----------
    u : array_like
        Array of values at which to evaluate the kernel function.

    Returns
    -------
    array_like
        Values of the kernel function at the given points.
    """
    return np.where(u**2 <= 1, 0.75 * (1 - u**2), 0)


def gauss_kern(u):
    """Gaussian kernel function.

    Parameters
    ----------
    u : array_like
        Array of values at which to evaluate the kernel function.

    Returns
    -------
    array_like
        Values of the kernel function at the given points.
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * u**2)


def kde_base(x, x_eval, bw='adaptive', kernel='epa', w=None, n_steps=1):
    """Kernel density estimation.

    Parameters
    ----------
    x : array_like
        Observed data points.
    x_eval : array_like
        Points at which to evaluate the KDE.
    bw : str or float, optional
        Bandwidth, by default 'adaptive'. Valid strings are 'adaptive', 'scott'.
    kernel : str, optional
        Kernel function to use, by default 'epa'. Valid strings are 'epa', 'gauss'.
    w : array-like, optional
        Observation weights, by default None. If None, all weights are set to 1. Must be the same length as x.
    n_steps : int, optional
        Number of steps for adaptive bandwidth estimation, by default 1.

    Returns
    -------
    """
    
    # weights are ones if not specified
    if w is None:
        w = np.ones_like(x)

    n_eff = np.sum(w)**2 / np.sum(w**2)
    sig_eff = np.min([np.std(x), 
                      (np.diff(np.percentile(x, [25, 75]))/1.34)[0]])

    # set bandwidth using Scott's rule; use for first estimate for adaptive model
    if bw == 'scott' or bw == 'adaptive':
        h = 1.06 * sig_eff * n_eff**(-1/5)
    else:
        h = bw
    
    # set up kernel
    if kernel == 'epa':
        kern = epa_kern
    elif kernel == 'gauss':
        kern = gauss_kern
    else:
        raise ValueError('Invalid kernel function.')

    # if adaptive bandwidth
    if bw == 'adaptive':
        # determine localized bandwidths
        for ii in range(n_steps):
            u = (x - x[:, np.newaxis]) / h
            f_hat = 1/np.sum(w) * np.sum(kern(u) * w, axis=1) / h
            G = stats.gmean(f_hat)
            # get new localized bandwidths
            lam = np.sqrt(G/f_hat)
            h = h * lam

    # set up evaluation grid
    x_eval = np.atleast_1d(x_eval)
    u = (x - x_eval[:, np.newaxis]) / h

    # compute KDE
    if bw == 'adaptive':
        f_hat = 1/np.sum(w) * np.sum(kern(u) * w/h, axis=1)
    else:
        f_hat = 1/np.sum(w) * np.sum(kern(u) * w, axis=1) / h

    return f_hat