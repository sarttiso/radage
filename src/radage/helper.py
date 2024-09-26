import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import dct
from scipy.optimize import root_scalar

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


def botev(x, n=None):
    """Botev et al. bandwidth selection algorithm.

    See Botev, Grotowski, and Kroese (2010) for details, in particular page 2932
    doi: 10.1214/10-AOS799

    Largely drawn from the MATLAB implementation by Botev, available at:
    http://web1.maths.unsw.edu.au/~zdravkobotev/php/kde_m.php

    Parameters
    ----------
    x : array_like
        Data for which to estimate the bandwidth.
    n : int, optional
        Number of gridded coordinates, must be power of 2. If None, 1024. By default None.

    Returns
    -------
    float
        Bandwidth for the given data.
    """

    x = np.asarray(x).flatten()
    if n is None:
        n = 2**10
    n = int(2**np.ceil(np.log2(n)))  # Round up n to the next power of 2

    x_min = np.min(x)
    x_max = np.max(x)
    Dx = x_max - x_min

    # from Botev MATLAB, add buffer below and above min and max
    MIN = x_min - Dx/5
    MAX = x_max + Dx/5
    R = MAX - MIN
    # set up evaluation grid
    dx = R / n
    xmesh = MIN + np.arange(0, R+dx, dx)
    
    N = len(np.unique(x)) # check for duplicates?
    
    # Create a histogram
    initial_data = np.histogram(x, bins=xmesh)[0] / N
    initial_data = initial_data / np.sum(initial_data)
    
    # Discrete cosine transform of initial data
    a = dct(initial_data)

    I = np.arange(1, n)**2
    a2 = (a[1:] / 2)**2

    # Define the fixed point function
    def fixed_point(t, N, I, a2):

        # from https://github.com/tommyod/KDEpy/blob/ae1c23c2dc50b91b93dfb982030f0127ce83e447/KDEpy/bw_selection.py#L21
        I = np.asfarray(I, dtype=float)
        a2 = np.asfarray(a2, dtype=float)

        l = 7
        # Initial f calculation
        f = 2 * (np.pi**(2 * l)) * np.sum((I**l) * a2 * np.exp(-I * (np.pi**2) * t))
        
        # Loop for decreasing s values from l-1 to 2
        for s in range(l-1, 1, -1):
            K0 = np.prod(np.arange(1, 2 * s + 1, 2)) / np.sqrt(2 * np.pi)
            const = (1 + (1/2)**(s + 0.5)) / 3
            # print(s)
            time = (2 * const * K0 / (N * f))**(2 / (3 + 2 * s))            
            f = 2 * (np.pi**(2 * s)) * np.sum((I**s) * a2 * np.exp(-I * (np.pi**2) * time))
        
        # Final output calculation
        out = t - (2 * N * np.sqrt(np.pi) * f)**(-2/5)
        return out

    # Use root to solve the equation t = zeta * gamma^[5](t)
    result = root_scalar(lambda t: fixed_point(t, N, I, a2), 
                         bracket=[0, 0.01], method='brentq')
    t_star = result.root

    # compute bandwidth
    bandwidth = np.sqrt(t_star) * R
    
    return bandwidth



def kde_base(x, x_eval, bw='adaptive', kernel='epa', w=None, n_steps=1):
    """Kernel density estimation.

    Parameters
    ----------
    x : array_like
        Observed data points.
    x_eval : array_like
        Points at which to evaluate the KDE.
    bw : str or float, optional
        Bandwidth, by default 'adaptive'. Valid strings are 'adaptive', 'scott', 'botev'
    kernel : str, optional
        Kernel function to use, by default 'epa'. Valid strings are 'epa', 'gauss'.
    w : array-like, optional
        Observation weights, by default None. If None, all weights are set to 1. Must be the same length as x.
    n_steps : int, optional
        Number of steps for adaptive bandwidth estimation, by default 1.

    Returns
    -------
    f_hat : array_like
        Kernel density estimate at the given points.
    """
    
    # weights are ones if not specified
    if w is None:
        w = np.ones_like(x)

    n_eff = np.sum(w)**2 / np.sum(w**2)
    sig_eff = np.min([np.std(x), 
                      (np.diff(np.percentile(x, [25, 75]))/1.34)[0]])

    # set bandwidth using Scott's rule; use for first estimate for adaptive model
    if bw == 'botev' or bw == 'adaptive':
        h = botev(x)
    elif bw == 'scott':
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