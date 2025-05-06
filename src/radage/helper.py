import numpy as np
import scipy.stats as stats
from scipy.fft import dct
from scipy.optimize import root_scalar


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
    MIN = x_min - Dx/2
    MAX = x_max + Dx/2
    R = MAX - MIN
    # set up evaluation grid
    dx = R / n
    xmesh = MIN + np.arange(0, R+dx, dx)

    N = len(np.unique(x))  # check for duplicates?

    # Create a histogram
    initial_data = np.histogram(x, bins=xmesh)[0] / N
    initial_data = initial_data / np.sum(initial_data)

    # Discrete cosine transform of initial data
    a = dct(initial_data)

    I = np.arange(1, n, dtype=np.float64)**2
    a2 = (a[1:])**2

    # Define the fixed point function
    def fixed_point(t, N, I, a2):

        # from https://github.com/tommyod/KDEpy/blob/ae1c23c2dc50b91b93dfb982030f0127ce83e447/KDEpy/bw_selection.py#L21
        # https://github.com/tommyod/KDEpy/issues/95
        I = np.asfarray(I, dtype=np.float64)
        a2 = np.asfarray(a2, dtype=np.float64)

        l = 7
        # Initial f calculation
        f = 1/2 * (np.pi**(2 * l)) * np.sum((I**l) * a2 * np.exp(-I * (np.pi**2) * t))

        # Loop for decreasing s values from l-1 to 2
        for s in range(l-1, 1, -1):
            K0 = np.prod(np.arange(1, 2 * s + 1, 2),
                         dtype=np.float64) / np.sqrt(2 * np.pi)
            K1 = (1 + (1/2)**(s + 0.5)) / 3
            # print(s)
            time = (2 * K1 * K0 / (N * f))**(2 / (3 + 2 * s))
            f = 1/2 * (np.pi**(2 * s)) * np.sum((I**s) *
                                                a2 * np.exp(-I * (np.pi**2) * time))

        # Final output calculation
        out = t - (2 * N * np.sqrt(np.pi) * f)**(-2/5)
        return out

    # Use root to solve the equation t = zeta * gamma^[5](t)
    # find tolerance
    def fixed_point_t(t):
        t = np.atleast_1d(t)
        output = np.zeros_like(t)
        for ii in range(len(t)):
            output[ii] = fixed_point(t[ii], N, I, a2)
        return output
    converged = False
    N_eff = 50 * (N <= 50) + 1050 * (N >= 1050) + N*((N < 1050) & (N > 50))
    tol = 1e-12 + 0.01 * (N_eff - 50) / 1000
    while not converged:
        t_values = np.linspace(0, tol, 50)
        # find roots (approximately)
        root_approx = np.where(np.diff(np.sign(fixed_point_t(t_values))))[0]
        if len(root_approx) > 0 and len(root_approx) < 2:
            converged = True
        # if more than one root, decrease range
        elif len(root_approx) > 1:
            # take halfway between the first two roots
            tol = (t_values[root_approx[0]] + t_values[root_approx[1]]) / 2
            # tol = tol / 1.5
        else:
            tol = tol * 2

    try:
        result = root_scalar(lambda t: fixed_point(t, N, I, a2),
                             bracket=[0, tol], method='brentq')
        # if no root, expand the bracket
    except ValueError:
        raise ValueError('No root found.')

    t_star = result.root

    # compute bandwidth
    bandwidth = np.sqrt(t_star) * R

    return bandwidth


def kde_base(x, x_eval, bw='adaptive', kernel='gauss', w=None, n_steps=1):
    """Kernel density estimation.

    If number of data are fewer than 30, use Scott's rule for initial bandwidth estimation.

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

    # check that bw is valid if it's a string
    if type(bw) is str and bw not in ['adaptive', 'scott', 'botev']:
        raise ValueError('Invalid bandwidth method.')
    # set bandwidth using Scott's rule; use for first estimate for adaptive model
    n_thres = 30
    if (bw in ['botev', 'adaptive']) and (len(x) > n_thres):
        h = botev(x)
    elif bw == 'scott' or ((len(x) <= n_thres) and (bw in ['botev', 'adaptive'])):
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
