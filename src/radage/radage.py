import pandas as pd
import numpy as np

import scipy.stats as stats
from scipy.optimize import minimize_scalar

import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms

import warnings
# from helper import *
# from .plotting import *

import pdb

# in My
l238 = 1.55125e-10 * 1e6
l238_std = 0.5 * l238 * 0.107 / 100  # see Schoene 2014 pg. 359
l235 = 9.8485e-10 * 1e6
l235_std = 0.5 * l235 * 0.137 / 100
l232 = 0.049475e-9 * 1e6  # probably needs to be updated, from Stacey and Kramers 1975
u238u235 = 137.837


def concordia(t):
    """
    Wetherill concordia curve.

    207/235 and 206/238 ratios for given times.

    Parameters:
    -----------
    t : array-like
        time points in Myr

    Returns:
    --------
    r207_235 : array-like
        207/235 ratios for the given times
    r206_238 : array-like
        206/238 ratios for the given times

    """
    r206_238 = np.exp(l238 * t) - 1
    r207_235 = np.exp(l235 * t) - 1

    return r207_235, r206_238


def concordia_tw(t):
    """
    Tara-Wasserberg concordia curve.

    238/206 and 207/206 ratios for given times.

    Parameters:
    -----------
    t : array-like
        time points in Myr

    Returns:
    --------
    r238_206 : array-like
        238/206 ratios for the given times
    r207_206 : array-like
        207/206 ratios for the given times
    """
    r206_238 = np.exp(l238 * t) - 1
    r238_206 = 1 / r206_238
    r207_206 = (np.exp(l235 * t) - 1) / (np.exp(l238 * t) - 1) * 1 / u238u235

    return r238_206, r207_206


def concordia_confint(t, conf=0.95):
    """
    Confidence intervals on concordia.

    Function giving 206/238, 207/235 ratios for bounds on confidence region around concordia at a given t
    Returns upper bound, then lower bound

    Parameters:
    -----------
    t : array-like
        time points in Myr
    conf : float, optional
        confidence level for interval, defaults to 0.95

    Returns:
    --------
    lower_bound : numpy.ndarray
        Lower bound of confidence interval, first column is 207/235, second column is 206/238
    upper_bound : numpy.ndarray
        Upper bound of confidence interval, first column is 207/235, second column is 206/238

    """
    # slope of line tangent to concordia
    m = (l238 * np.exp(l238 * t)) / (l235 * np.exp(l235 * t))

    # rename coordinates (207/235 = x, 206/238 = y)
    x, y = concordia(t)

    # uncertainty ellipse for given confidence interval
    r = stats.chi2.ppf(conf, 2)

    # uncertainties in x, y at given time t
    sigy = t * np.exp(l238 * t) * l238_std
    sigx = t * np.exp(l235 * t) * l235_std

    # tangent points to uncertainty ellipse
    with np.errstate(divide='ignore', invalid='ignore'):
        ytan_1 = np.sqrt(r / (1 / (sigx**2) *
                              (-m * sigx**2 / sigy**2)**2 + 1 / sigy**2)) + y
        xtan_1 = -m * sigx**2 / sigy**2 * (ytan_1 - y) + x

        ytan_2 = -np.sqrt(r / (1 / (sigx**2) *
                               (-m * sigx**2 / sigy**2)**2 + 1 / sigy**2)) + y
        xtan_2 = -m * sigx**2 / sigy**2 * (ytan_2 - y) + x

    if np.any(t == 0):
        idx = t == 0
        xtan_1[idx] = 0
        ytan_1[idx] = 0
        xtan_2[idx] = 0
        ytan_2[idx] = 0

    lower_bound = np.vstack([xtan_1, ytan_1]).T
    upper_bound = np.vstack([xtan_2, ytan_2]).T

    return lower_bound, upper_bound


def t238(r38_06):
    """_summary_

    :param r38_06: _description_
    :type r38_06: _type_
    :return: _description_
    :rtype: _type_
    """
    t = np.log((1 / r38_06) + 1) / l238
    return t


def t235(r35_07):
    """_summary_

    :param r35_07: _description_
    :type r35_07: _type_
    :return: _description_
    :rtype: _type_
    """
    t = np.log((1 / r35_07) + 1) / l235
    return t


def t207(r207_206, u238u235=u238u235):
    """_summary_

    :param r207_206: _description_
    :type r207_206: _type_
    :return: _description_
    :rtype: _type_
    """
    # ignore warning that occurs sometimes during optimization
    warnings.filterwarnings(
        'ignore', message='invalid value encountered in double_scalars')

    def cost(t, cur207_206):
        """
        cost function for solving for t
        """
        S = (1 / u238u235 * (np.exp(l235 * t) - 1) / (np.exp(l238 * t) - 1) -
             cur207_206)**2
        return S

    # compute age
    age = minimize_scalar(cost, args=(r207_206), bounds=(0, 5000)).x

    return age


class UPb:
    """
    class for handling U-Pb ages, where data are given as isotopic ratios and their uncertainties
    """

    def __init__(self,
                 r206_238,
                 r206_238_std,
                 r207_235,
                 r207_235_std,
                 r207_206,
                 r207_206_std,
                 rho75_68,
                 rho86_76,
                 name=None):
        """
        create object, just need means, stds, and correlations
        """
        self.r206_238 = r206_238
        self.r206_238_std = r206_238_std
        self.r207_235 = r207_235
        self.r207_235_std = r207_235_std
        self.r207_206 = r207_206
        self.r207_206_std = r207_206_std

        self.rho75_68 = rho75_68  # rho1
        self.rho86_76 = rho86_76  # rho2

        self.name = name

        # compute eigen decomposition of covariance matrix for 206/238, 207/235
        self.cov_235_238 = np.array(
            [[
                self.r207_235_std**2,
                self.rho75_68 * self.r206_238_std * self.r207_235_std
            ],
                [
                self.rho75_68 * self.r206_238_std * self.r207_235_std,
                self.r206_238_std**2
            ]])
        self.eigval_235_238, self.eigvec_235_238 = np.linalg.eig(
            self.cov_235_238)
        idx = np.argsort(self.eigval_235_238)[::-1]  # sort descending
        self.eigval_235_238 = self.eigval_235_238[idx]
        self.eigvec_235_238 = self.eigvec_235_238.T[idx].T

        # compute eigen decomposition of covariance matrix for 238/206, 207/206
        self.r238_206_std = (self.r206_238_std /
                             self.r206_238) * (1 / self.r206_238)
        self.cov_238_207 = np.array(
            [[
                self.r238_206_std**2,
                self.rho86_76 * self.r238_206_std * self.r207_206_std
            ],
                [
                self.rho86_76 * self.r238_206_std * self.r207_206_std,
                self.r207_206_std**2
            ]])
        self.eigval_238_207, self.eigvec_238_207 = np.linalg.eig(
            self.cov_238_207)
        idx = np.argsort(self.eigval_238_207)[::-1]  # sort descending
        self.eigval_238_207 = self.eigval_238_207[idx]
        self.eigvec_238_207 = self.eigvec_238_207.T[idx].T

    def ellipse_68_75(self,
                      conf=0.95,
                      patch_dict=None):
        """
        Generate uncertainty ellipse for desired confidence level for $^{206}$Pb/$^{238}$U

        [more here](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval)
        """
        # set up a default stle
        if patch_dict is None:
            patch_dict = {'facecolor': 'wheat',
                          'linewidth': 0.5,
                          'edgecolor': 'k'}

        r = stats.chi2.ppf(conf, 2)
        a1 = np.sqrt(self.eigval_235_238[0] * r)
        a2 = np.sqrt(self.eigval_235_238[1] * r)

        # compute rotation from primary eigenvector
        rotdeg = np.rad2deg(np.arccos(self.eigvec_235_238[0, 0]))

        # create
        ell = Ellipse((self.r207_235, self.r206_238),
                      width=a1 * 2,
                      height=a2 * 2,
                      angle=rotdeg,
                      **patch_dict)

        return ell

    def ellipse_76_86(self,
                      conf=0.95,
                      patch_dict=None,):
        """
            [more here](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval)
        """
        # set up a default stle
        if patch_dict is None:
            patch_dict = {'facecolor': 'wheat',
                          'linewidth': 0.5,
                          'edgecolor': 'k'}

        r = stats.chi2.ppf(conf, 2)
        a1 = np.sqrt(self.eigval_238_207[0] * r)
        a2 = np.sqrt(self.eigval_238_207[1] * r)

        # compute rotation from primary eigenvector
        rotdeg = np.rad2deg(np.arccos(self.eigvec_238_207[0, 0]))

        # create
        ell = Ellipse((1 / self.r206_238, self.r207_206),
                      width=a1 * 2,
                      height=a2 * 2,
                      angle=rotdeg,
                      **patch_dict)

        return ell

    def discordance(self, method='relative'):
        """
        compute the discordance by some metric of this age

        Paramters:
        ----------
        method : String
            'SK': mixture model with Stacey and Kramers' (1975) common lead model
            'p_238_235': use the probability of fit from the concordia age
            'relative': relative age different, where the 206/238 vs 207/235 ages are
                used for 206/238 ages younger than 1 Ga, and the 207/206 vs 206/238
                ages are used for ages older than 1 Ga
            'relative_68_76': relative age difference using only the 206/238 and 206/207
                ages, computed as 1-(t68/t76)

        """
        if method == 'SK':
            t = Pb_mix_find_t(self.r207_206, 1 / self.r206_238)
            r238_206_rad = 1 / (np.exp(l238 * t) - 1)
            # r207_206_rad = 1/u238u235 * (np.exp(l235*t)-1)/(np.exp(l238*t)-1)
            # r207_206_cm = sk_pb(t)[1]/sk_pb(t)[0]
            # L = np.sqrt((r207_206_rad-r207_206_cm)**2 + (r238_206_rad)**2)
            # l = np.sqrt((1/self.r206_238-r238_206_rad)**2 + (self.r207_206-r207_206_rad)**2)
            d = 1 - (1 / self.r206_238) / r238_206_rad
            # d = l/L
        elif method == 'relative':
            cur68_age = self.age68(conf=None)
            # use 75 vs 68 for younger than 1 Ga
            if cur68_age < 1000:
                d = 1 - cur68_age/self.age75(conf=None)
            else:
                d = 1 - cur68_age / self.age76(conf=None)
        elif method == 'relative_76_68':
            d = 1 - self.age68(conf=None) / self.age76(conf=None)
        elif method == 'absolute_76_68':
            d = np.abs(self.age76(conf=None) - self.age68(conf=None))
        elif method == 'relative_68_75':
            d = 1 - self.age68(conf=None) / self.age75(conf=None)
        elif method == 'absolute_68_75':
            d = np.abs(self.age75(conf=None) - self.age68(conf=None))
        elif method == 'p_68_75':
            tc = self.age_235_238_concordia()[0]
            v = np.array([[
                self.r207_235 - np.exp(l235 * tc) + 1,
                self.r206_238 - np.exp(l238 * tc) + 1
            ]]).T
            C = np.array([[
                self.r207_235_std**2,
                self.rho75_68 * self.r207_235_std * self.r206_238_std
            ],
                [
                self.rho75_68 * self.r207_235_std *
                self.r206_238_std, self.r206_238_std**2
            ]])
            # chi-squared statistic is this quadratic
            S = np.matmul(v.T, np.matmul(np.linalg.inv(C), v))
            # probability of exceeding the statistic is just 1-cdf for a chi-squared distribution with 2 dof
            d = 1 - stats.chi2.cdf(S, 2)
            d = np.squeeze(d)[()]
        elif method == 'aitchison_76_68':

            def dx(t):
                return np.log(self.r206_238) - np.log(np.exp(l238 * t) - 1)

            def dy(t):
                return np.log(self.r207_206) - np.log(1 / u238u235 *
                                                      (np.exp(l235 * t) - 1) /
                                                      (np.exp(l238 * t) - 1))

            d = dx(self.age76(conf=None)) * np.sin(
                np.arctan(
                    dy(self.age68(conf=None)) / dx(self.age76(conf=None))))
        elif method == 'aitchison_68_75':

            def dx(t):
                return np.log(self.r207_235) - np.log(np.exp(l235 * t) - 1)

            def dy(t):
                return np.log(self.r206_238) - np.log(np.exp(l238 * t) - 1)

            d = dx(self.age68(conf=None)) * np.sin(
                np.arctan(
                    dy(self.age75(conf=None)) / dx(self.age68(conf=None))))

        return d

    def age_207_238_concordia(self):
        """
        Generate $^{207}$Pb/$^{206}$Pb - $^{206}$Pb/$^{238}$U concordia age as per Ludwig (1998), with uncertainty and MSWD.

        DOES NOT WORK
        """

        # get omega for a given t
        def get_omega(t):
            """
            note that this function uses the error transformations in Appendix A of Ludwig (1998)
            """
            # relative errors
            SX = self.r206_238_std / self.r206_238
            SY = self.r207_206_std / self.r207_206
            Sx = self.r207_235_std / self.r207_235
            Sy = self.r206_238_std / self.r206_238
            rhoXY = self.rho86_76
            sigx = self.r207_235 * np.sqrt(SX**2 + SY**2 - 2 * SX * SY * rhoXY)
            rhoxy = (SX**2 - SX * SY * rhoXY) / (Sx * Sy)
            sigy = self.r206_238_std

            P235 = t * np.exp(l235 * t)
            P238 = t * np.exp(l238 * t)

            cov_mod = np.array(
                [[sigx**2 + P235**2 * l235_std**2, rhoxy * sigx * sigy],
                 [rhoxy * sigx * sigy, sigy**2 + P238**2 * l238_std**2]])
            omega = np.linalg.inv(cov_mod)
            return omega

        def S_cost(t):
            omega = get_omega(t)

            R = self.r207_206 * u238u235 * self.r206_238 - (np.exp(l235 * t) -
                                                            1)
            r = self.r206_238 - np.exp(l238 * t) + 1

            S = omega[0, 0] * R**2 + omega[1, 1] * r**2 + 2 * R * r * omega[0,
                                                                            1]

            return S

        opt = minimize_scalar(S_cost, bounds=[0, 5000], method='bounded')
        t = opt.x

        omega = get_omega(t)

        Q235 = l235 * np.exp(l235 * t)
        Q238 = l238 * np.exp(l238 * t)

        t_std = np.sqrt((Q235**2 * omega[0, 0] + Q238**2 * omega[1, 1] +
                         2 * Q235 * Q238 * omega[0, 1])**(-1))

        MSWD = 1 - stats.chi2.cdf(opt.fun, 1)
        # pdb.set_trace()
        return t, t_std, MSWD

    def age_235_238_concordia(self):
        """
        Generate $^{207}$Pb/$^{235}$U - $^{206}$Pb/$^{238}$U concordia age as per Ludwig (1998), with uncertainty and MSWD.

        [![Ludwig 1998](https://img.shields.io/badge/DOI-10.1016%2FS0016--7037(98)00059--3-blue?link=http://doi.org/10.1016/S0016-7037(98)00059-3&style=flat-square)](http://doi.org/10.1016/S0016-7037(98)00059-3)

        Returns:
        - t: age
        - t_std: standard deviation of age
        - MSWD: exceedance probability of misfit for concordance, corresponds to a 1-MSWD confidence level for accepting the observed misfit. Lower values permit more discordant ages.
        """

        # get omega for a given t
        def get_omega(t):
            P235 = t * np.exp(l235 * t)
            P238 = t * np.exp(l238 * t)

            cov_mod = np.array(
                [[
                    self.r207_235_std**2 + P235**2 * l235_std**2,
                    self.rho75_68 * self.r206_238_std * self.r207_235_std
                ],
                    [
                    self.rho75_68 * self.r206_238_std * self.r207_235_std,
                    self.r206_238_std**2 + P238**2 * l238_std**2
                ]])
            omega = np.linalg.inv(cov_mod)
            return omega

        # cost function for concordia from Ludwig 1998, using their notation
        def S_cost(t):
            omega = get_omega(t)

            R = self.r207_235 - np.exp(l235 * t) + 1
            r = self.r206_238 - np.exp(l238 * t) + 1

            S = omega[0, 0] * R**2 + omega[1, 1] * r**2 + 2 * R * r * omega[0,
                                                                            1]

            return S

        opt = minimize_scalar(S_cost, bounds=[0, 5000], method='bounded')
        t = opt.x

        omega = get_omega(t)

        Q235 = l235 * np.exp(l235 * t)
        Q238 = l238 * np.exp(l238 * t)

        t_std = np.sqrt((Q235**2 * omega[0, 0] + Q238**2 * omega[1, 1] +
                         2 * Q235 * Q238 * omega[0, 1])**(-1))

        MSWD = 1 - stats.chi2.cdf(opt.fun, 1)

        return t, t_std, MSWD

    def age68(self, conf=0.95, n=1e5):
        """
        return 206/238 age with interval for desired confidence
        """
        n = int(n)
        age = np.log(self.r206_238 + 1) / l238
        if conf == None:
            return age
        else:
            # sig = np.std(
            #     np.log(
            #         stats.norm.rvs(self.r206_238, self.r206_238_std, size=n) +
            #         1) / l238)
            sig = (1/l238) * self.r206_238_std/(self.r206_238 + 1)
            conf = 1 - (1 - conf) / 2
            confint = stats.norm.ppf(conf, age, sig) - age
            return age, sig, confint

    def age75(self, conf=0.95, n=1e5):
        """
        return 207/235 age with interval for desired confidence
        """
        n = int(n)
        age = np.log(self.r207_235 + 1) / l235
        if conf == None:
            return age
        else:
            sig = np.std(
                np.log(
                    stats.norm.rvs(self.r207_235, self.r207_235_std, size=n) +
                    1) / l235)
            conf = 1 - (1 - conf) / 2
            confint = stats.norm.ppf(conf, age, sig) - age
            return age, sig, confint

    def age76(self, conf=0.95, n=1e3, u238u235=u238u235):
        """
        return 207/206 age with interval for desired confidence
        """
        # ignore warning that occurs sometimes during optimization
        warnings.filterwarnings(
            'ignore', message='invalid value encountered in double_scalars')

        n = int(n)

        def cost(t, cur207_206):
            """
            cost function for solving for t
            """
            S = (1 / u238u235 * (np.exp(l235 * t) - 1) /
                 (np.exp(l238 * t) - 1) - cur207_206)**2
            return S

        # compute age
        age = minimize_scalar(cost, args=(self.r207_206), bounds=(0, 4500)).x

        if conf == None:
            return age
        else:
            # now Monte Carlo solutions for t given uncertainty on the 207/206 ratio
            ages = np.zeros(n)
            r207_206_samp = stats.norm.rvs(self.r207_206,
                                           self.r207_206_std,
                                           size=n)
            for ii in range(n):
                res = minimize_scalar(cost,
                                      args=(r207_206_samp[ii]),
                                      bounds=(0, 4500))
                ages[ii] = res.x
            sig = np.std(ages)
            conf = 1 - (1 - conf) / 2
            confint = stats.norm.ppf(conf, age, sig) - age
            return age, sig, confint


def sk_pb(t, t0=4.57e3, t1=3.7e3, mu1=7.19, mu2=9.74, x0=9.307, y0=10.294):
    """
    common lead model from Stacey and Kramers (1975)

    Parameters:
    -----------
    t : 1d array like
        time (in Ma) for which to return isotopic ratios for the linear mixing model from concordia to mantle

    t0 : float
        age of earth in My

    t1 : float 
        time for transition between model stages

    x0 : float
        206Pb/204Pb ratio for troilite lead

    y0 : float
        207Pb/204Pb ratio for troilite lead

    mu1 : float

    mu2 : float
    """

    t = np.reshape(np.array([t]), -1)

    n = len(t)

    # output ratios
    r206_204 = np.zeros(n)  # x(t)
    r207_204 = np.zeros(n)  # y(t)

    # for times in first stage
    idx = t > t1
    r206_204[idx] = x0 + mu1 * (np.exp(l238 * t0) - np.exp(l238 * t[idx]))
    r207_204[idx] = y0 + mu1 / u238u235 * (np.exp(l235 * t0) -
                                           np.exp(l235 * t[idx]))

    # for times in second stage
    idx = t <= t1
    x1 = x0 + mu1 * (np.exp(l238 * t0) - np.exp(l238 * t1))
    y1 = y0 + mu1 / u238u235 * (np.exp(l235 * t0) - np.exp(l235 * t1))
    r206_204[idx] = x1 + mu2 * (np.exp(l238 * t1) - np.exp(l238 * t[idx]))
    r207_204[idx] = y1 + mu2 / u238u235 * (np.exp(l235 * t1) -
                                           np.exp(l235 * t[idx]))

    return r206_204.squeeze(), r207_204.squeeze()


def Pb_mix_find_t(r207_206, r238_206):
    """
    given a pair of 207/206 & 238/206 isotopic ratios, solve for the age such that one has a lead mixing model that 
    results in identical Stacey & Kramers' (1975) common lead model ages and radiogenic concordia ages
    while also passing through the observed ratios.
    """

    def cost(t):
        """
        defines a cost metric to search over appropriate times
        """
        r206_204, r207_204 = sk_pb(t)
        # intercept (common lead)
        r207_206_0 = r207_204 / r206_204
        r238_206_0 = 0

        # concordia (radiogenic component)
        r238_206_rad = 1 / (np.exp(l238 * t) - 1)
        r207_206_rad = 1 / u238u235 * (np.exp(l235 * t) -
                                       1) / (np.exp(l238 * t) - 1)

        # compute distance from line connecting these two points
        d = np.abs((r238_206_rad-r238_206_0)*(r207_206_0-r207_206) -
                   (r238_206_0-r238_206)*(r207_206_rad-r207_206_0)) / \
            np.sqrt((r238_206_rad-r238_206_0)**2 +
                    (r207_206_rad-r207_206_0)**2)

        return d

    res = minimize_scalar(cost, bounds=(0, 4500))
    t = res.x

    return t


def Pb_mix_plot(t, ax=None, **kwargs):
    """
    for given t, plot a linear common-radiogenic lead mixing model in TW space

    To Do: update the input validation for ax
    """
    r238_206_rad = 1 / (np.exp(l238 * t) - 1)
    r207_206_rad = 1 / u238u235 * (np.exp(l235 * t) - 1) / (np.exp(l238 * t) -
                                                            1)

    r206_204, r207_204 = sk_pb(t)
    r207_206_0 = r207_204 / r206_204
    r238_206_0 = 0

    if ax == None:
        ax = plt.axes()

    ax.plot(np.array([r238_206_0, r238_206_rad]),
            np.array([r207_206_0, r207_206_rad]))


def annotate_concordia(ages, tw=False, ax=None, ann_style=None):
    """
    use this function to annotate concordia plots with times

    ages: list of numbers in Ma to annotate and plot on concordia
    tw: tera wasserberg space?
    ann_style: dict
    """
    n_ages = len(ages)
    if tw:
        x_lab, y_lab = concordia_tw(ages)
    else:
        x_lab, y_lab = concordia(ages)

    if ax == None:
        ax = plt.gca()

    if ann_style is None:
        ann_style = {'color': 'red',
                     'marker': 'o',
                     'linestyle': ''}

    for ii in range(n_ages):
        if tw:
            offset = (0, -15)
            ha = 'center'
        else:
            offset = (0, 5)
            ha = 'right'

    # time labels
    ax.plot(x_lab, y_lab, **ann_style)

    for ii in range(n_ages):
        ax.annotate(int(ages[ii]),
                    xy=(x_lab[ii], y_lab[ii]),
                    xytext=offset,
                    textcoords='offset points',
                    ha=ha)


def discordance_filter(ages,
                       method='relative',
                       threshold=0.03,
                       system_threshold=False):
    """
    function to filter on discordance

    Parameters:
    -----------
    filter_method : string
        'relative', 'absolute', 'aitchison'
        discordance metric to use. Discordance is evaluated between
        207/206-238/206 ages for samples with 207/206 ages > 1 Ga, and 
        206/238-207/235 ages for sample with 207/206 ages < 1 Ga

    filter_threshold : float
        discordance value for chosen filtering method above which to flag
        ages as discordant

    system_threshold : boolean
        whether or not to switch discordance metrics between the 207/206-238/206 
        and 206/238-207/235 systems at 1 Ga. If false, then discordance is always
        computed between the 207/206-238/206 systems. If true, only works for 
        aithison, relative, and absolute discordance metrics.
    """
    ages_conc = []
    if system_threshold:
        for age in ages:
            if age.age76(conf=None) > 1000:
                cur_method = method + '_76_68'
            else:
                cur_method = method + '_68_75'
            d = np.abs(age.discordance(method=cur_method))
            if d < threshold:
                ages_conc.append(age)
    else:
        for age in ages:
            d = np.abs(age.discordance(method=method + '_76_68'))
            if d < threshold:
                ages_conc.append(age)
    return ages_conc


def discordia_age_76_86(m, b, precision=3):
    """give 207/206 vs 238/206 age for a line with slope m and intercept b

    :param m: _description_
    :type m: _type_
    :param b: _description_
    :type b: _type_
    """
    n = 1 * 10**precision
    t = np.linspace(2, 4500, n)
    r238_206_conc, r207_206_conc = concordia_tw(t)
    # line
    r238_206_ax = np.linspace(np.min(r238_206_conc), np.max(r238_206_conc), n)
    r207_206_ax = m * r238_206_ax + b

    # compute intersection
    x, y = intersection(r238_206_ax, r207_206_ax, r238_206_conc, r207_206_conc)

    # take larger x value
    idx = np.argmax(x)
    x = x[idx]
    y = y[idx]

    # get age for given ratios
    t_x = t238(x)
    t_y = t207(y)

    age = np.mean([t_x, t_y])

    return age


def kde(ages, t,
        kernel='gau',
        bw='scott',
        systems='auto'):
    """
    evaluate kde for ages at times t

    Parameters
    ----------
    ages : arraylike
        list of radages.
    t : arraylike
        times at which to evaluate the kde.
    kernel : str, optional
        Type of kernel. The default is 'gau'.
    bw : str, optional
        bandwidth. The default is 'scott'.
    systems : str, optional
        systems to use. The default is 'auto,' which uses 76 after 1 Ga and 68
        before

    Returns
    -------
    kde_est : TYPE
        DESCRIPTION.

    """
    # determine ages to use as input based on systems requested
    if systems == 'auto':
        ages76 = np.array([age.age76(conf=None) for age in ages])
        ages68 = np.array([age.age68(conf=None) for age in ages])

        ages_in = np.concatenate([ages68[ages68 < 1000],
                                  ages76[ages76 > 1000]])

    cur_kde = sm.nonparametric.KDEUnivariate(ages_in).fit(kernel=kernel, bw=bw)

    kde_est = cur_kde.evaluate(t)

    return kde_est


def kdes(ages,
         kernel='gau',
         bw='scott',
         systems=['r68', 'r76', 'r75']):
    """
    generate kde's for a given list of ages

    Parameters:
    -----------
    ages : 1d array like of UPb objects
        ages from which to estimate KDE

    kernel : string
        statsmodels.nonparametric.kde.KDEUnivariate.fit() kernel

    bw : string or float
        statsmodels.nonparametric.kde.KDEUnivariate.fit() bw

    systems : list containing some combination of 'r68', 'r76', 'r75'
        Which isotopic systems to plot KDE's for

    Returns:
    --------
    kdes_by_system : list
        list of kdes objects, one for every requested system, but ordered as 'r68',
        'r76', 'r75'

    """
    kdes_by_system = []
    if 'r68' in systems:
        ages68 = [age.age68(conf=None) for age in ages]
        # ages_by_system.append(ages68)
        kdes_by_system.append(
            sm.nonparametric.KDEUnivariate(ages68).fit(kernel=kernel, bw=bw))
    if 'r76' in systems:
        ages76 = [age.age76(conf=None) for age in ages]
        # ages_by_system.append(ages76)
        kdes_by_system.append(
            sm.nonparametric.KDEUnivariate(ages76).fit(kernel=kernel, bw=bw))
    if 'r75' in systems:
        ages75 = [age.age75(conf=None) for age in ages]
        # ages_by_system.append(ages75)
        kdes_by_system.append(
            sm.nonparametric.KDEUnivariate(ages75).fit(kernel=kernel, bw=bw))

    # index isotopic systems in order provided by user
    # systems_def = ['r68', 'r76', 'r75']
    # idx = np.zeros(len(systems)).astype(int)
    # for ii, system in enumerate(systems):
    #     idx[ii] = np.argwhere(np.array(systems_def)==system).squeeze() - (3-len(systems))

    # kdes_by_system = [kdes_by_system[x] for x in idx]
    # ages_by_system = [ages_by_system[x] for x in idx]

    return kdes_by_system


def propagate_standard_uncertainty():
    """
    for a dataframe export from iolite, propagate uncertainty into all observations such that each standard population has MSWD <= 1

    UNFINISHED
    """
    stand_strs = ['AusZ', 'GJ1', 'Plesovice', '9435', '91500', 'Temora']

    # files = glob.glob('exports/*run[0-9].xlsx')

    dfs = []
    for file in files:
        dfs.append(pd.read_excel(file, sheet_name='Data', index_col=0))

    # for each run, scale standard standard errors to enforce MSWD<=1
    for ii in range(len(files)):
        cur_scale = 1
        for jj in range(len(stand_strs)):
            # find standard analyses
            idx = dfs[ii].index.str.match(stand_strs[jj])
            curdat = dfs[ii][idx]
            mu = np.mean(curdat['Final Pb206/U238 age_mean'])
            mswd = np.sum((curdat['Final Pb206/U238 age_mean']-mu)**2 /
                          (curdat['Final Pb206/U238 age_2SE(prop)']/2*cur_scale)**2)/(np.sum(idx)-1)
            while mswd > 1:
                cur_scale = cur_scale + 0.01
                mswd = np.sum((curdat['Final Pb206/U238 age_mean']-mu)**2 /
                              (curdat['Final Pb206/U238 age_2SE(prop)']/2*cur_scale)**2)/(np.sum(idx)-1)
        # rescale all uncertainties
        dfs[ii][list(dfs[ii].filter(like='2SE'))] = cur_scale * dfs[ii][list(
            dfs[ii].filter(like='2SE'))]
        dfs[ii][list(dfs[ii].filter(like='2SD'))] = cur_scale * dfs[ii][list(
            dfs[ii].filter(like='2SD'))]

    # concatenate
    dat = pd.concat(dfs, axis=0)

    n_dat = len(dat)


def yorkfit(x, y, wx, wy, r, thres=1e-3):
    """
    Implementation of York 1969 10.1016/S0012-821X(68)80059-7

    IN:
    x: mean x-values
    y: mean y-values
    wx: weights for x-values (typically 1/sigma^2)
    wy: weights for y-values (typically 1/sigma^2)
    r: correlation coefficient between sigma_x and sigma_y
    OUT:
    b: maximum likelihood estimate for slope of line
    a: maximum likelihood estimate for intercept of line
    b_sig: standard deviation of slope for line
    a_sig: standard deviation of intercept for line
    mswd: reduced chi-squared statistic for residuals with respect to the maximum likelihood linear model
    """
    n = len(x)
    # get first guess for b
    b = stats.linregress(x, y)[0]

    # initialize various quantities
    alpha = np.sqrt(wx * wy)

    # now iterate as per manuscript to improve b
    delta = thres + 1
    count = 0
    count_thres = 50
    while (delta > thres) and count < count_thres:
        # update values from current value of b
        W = (wx * wy) / (wx + b**2 * wy - 2 * b * r * alpha)
        xbar = np.sum(x * W) / np.sum(W)
        ybar = np.sum(y * W) / np.sum(W)
        U = x - xbar
        V = y - ybar
        beta = W * (U / wy + b * V / wx - (b * U + V) * r / alpha)
        # update b
        b_new = np.sum(W * beta * V) / np.sum(W * beta * U)
        delta = np.abs(b_new - b)
        b = b_new
        count = count + 1

    # compute a
    a = ybar - b * xbar
    # compute adjusted x, y
    x_adj = xbar + beta
    x_adj_bar = np.sum(W * x_adj) / np.sum(W)
    u = x_adj - x_adj_bar
    # compute parameter uncertainties
    b_sig = 1 / np.sum(W * u**2)
    a_sig = 1 / np.sum(W) + x_adj_bar**2 * b_sig**2
    # compute goodness of fit (reduced chi-squared statistic)
    mswd = np.sum(W * (y - b * x - a)**2) / (n - 2)

    return b, a, b_sig, a_sig, mswd


def get_sample(dfs, sample_name):
    """
    get entries in dataframes corresponding to a sample
    """
    df_sample = pd.DataFrame(columns=list(dfs[0]))
    for df in dfs:
        cur_idx = df.iloc[:, 0].str.contains(sample_name)
        df_sample = pd.concat([df_sample, df.loc[cur_idx]])
    return df_sample


def weighted_mean(ages, ages_s):
    """weighted mean age computation

    Args:
        ages (array-like): age means
        ages_s (array-like): age standard deviations (same length as ages)

    Returns:
        _type_: _description_
    """
    # weights
    w = 1/(ages_s**2)
    mu = np.sum(ages*w)/np.sum(w)
    # naive
    # sig = np.sqrt(1/np.sum(w))
    # unbiased
    sig2 = np.sum(w)/(np.sum(w)**2-np.sum(w**2))*np.sum(w*(ages-mu)**2)
    # biased
    # sig2 = (np.sum(w*ages**2)*np.sum(w) - np.sum(w*ages)**2)/np.sum(w)**2
    sig = np.sqrt(sig2)
    mswd = np.sum(w)/(np.sum(w)**2-np.sum(w**2)) * \
        np.sum((w*(ages-mu)**2)/ages_s**2)
    return mu, sig, mswd