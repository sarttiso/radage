import numpy as np

import scipy.stats as stats
from scipy.optimize import minimize_scalar, root_scalar

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import warnings
from .helper import *

# in My
l238 = 1.55125e-10 * 1e6
l238_std = 0.5 * l238 * 0.107 / 100  # see Schoene 2014 pg. 359
l235 = 9.8485e-10 * 1e6
l235_std = 0.5 * l235 * 0.137 / 100
l232 = 0.049475e-9 * 1e6  # probably needs to be updated, from Stacey and Kramers 1975
u238u235 = 138.818  # Heiss 2012


def concordia(t):
    """Wetherill concordia curve.

    207/235 and 206/238 ratios for given times.

    Parameters
    ----------
    t : array-like
        time points in Myr

    Returns
    -------
    r207_235 : array-like
        207/235 ratios for the given times
    r206_238 : array-like
        206/238 ratios for the given times

    """
    r206_238 = np.exp(l238 * t) - 1
    r207_235 = np.exp(l235 * t) - 1

    return r207_235, r206_238


def concordia_tw(t):
    """Tara-Wasserberg concordia curve.

    238/206 and 207/206 ratios for given times.

    Parameters
    ----------
    t : array-like
        time points in Myr

    Returns
    -------
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
    """Confidence intervals on concordia.

    Function giving 206/238, 207/235 ratios for bounds on confidence region around concordia at a given t
    Returns upper bound, then lower bound

    Parameters
    ----------
    t : array-like
        time points in Myr
    conf : float, optional
        confidence level for interval, defaults to 0.95

    Returns
    -------
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


def t238(r06_38):
    """206/238 date

    Parameters
    ----------
    r06_38 : float or numpy.ndarray
        206/238 ratio

    Returns
    -------
    t: float
        Date in Myr
    """
    t = np.log(r06_38 + 1) / l238
    return t


def t235(r07_35):
    """207/235 date

    Parameters
    ----------
    r07_35 : float or numpy.ndarray
        207/235 ratio

    Returns
    -------
    t: float
        Date in Myr
    """
    t = np.log(r07_35 + 1) / l235
    return t


def t207(r207_206, u238u235=u238u235):
    """207/206 date

    Parameters
    ----------
    r207_206 : float or numpy.ndarray
        207/206 ratio(s)
    u238u235 : float, optional
        Ratio of U238 to U235, by default u238u235

    Returns
    -------
    date: float
        Date in Myr
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

    # compute date
    date = minimize_scalar(cost, args=(r207_206), bounds=(0, 5000)).x

    return date


class UPb:
    """
    class for handling U-Pb ages, where data are given as isotopic ratios and their uncertainties

    Parameters
    ----------
    r206_238 : float
        Pb206/U238 ratio
    r206_238_std : float
        Standard deviation of Pb206/U238 ratio
    r207_235 : float
        Pb207/U235 ratio
    r207_235_std : float
        Standard deviation of Pb207/U235 ratio
    r207_206 : float
        Pb207/Pb206 ratio
    r207_206_std : float
        Standard deviation of Pb207/Pb206 ratio
    rho75_68 : float
        Error correlation between 207/235 and 206/238 ratios. Must be between -1 and 1
    rho86_76 : float
        Error correlation between 238/206 and 207/206 ratios. Must be between -1 and 1
    name : str, optional
        Name for age, by default None. Useful for plotting

    Attributes
    ----------
    cov_235_238 : numpy.ndarray
        Covariance matrix for 206/238 and 207/235 ratios
    eigval_235_238 : numpy.ndarray
        Eigenvalues of covariance matrix for 206/238 and 207/235 ratios
    eigvec_235_238 : numpy.ndarray
        Eigenvectors of covariance matrix for 206/238 and 207/235 ratios
    cov_238_207 : numpy.ndarray
        Covariance matrix for 238/206 and 207/206 ratios
    eigval_238_207 : numpy.ndarray
        Eigenvalues of covariance matrix for 238/206 and 207/206 ratios
    eigvec_238_207 : numpy.ndarray
        Eigenvectors of covariance matrix for 238/206 and 207/206 ratios
    """

    def __init__(self,
                 r206_238, r206_238_std,
                 r207_235, r207_235_std,
                 r207_206, r207_206_std,
                 rho75_68, rho86_76,
                 name=None):
        """Create UPb object
        """
        self.r206_238 = r206_238
        self.r206_238_std = r206_238_std
        self.r207_235 = r207_235
        self.r207_235_std = r207_235_std
        self.r207_206 = r207_206
        self.r207_206_std = r207_206_std
        self.r238_206 = 1 / r206_238

        # check that correlation coefficients are between 0 and 1
        if rho75_68 < -1 or rho75_68 > 1:
            raise ValueError('rho75_68 must be between -1 and 1')
        if rho86_76 < -1 or rho86_76 > 1:
            print(rho86_76)
            raise ValueError('rho86_76 must be between -1 and 1')
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
                             self.r206_238) * (self.r238_206)
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
        """Uncertainty ellipse for 206Pb/238U-207Pb/235U date

        See `here <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval>`__ for more information.

        Parameters
        ----------
        conf : float, optional
            Confidence level for interval, by default 0.95
        patch_dict : dict, optional
            Dictionary of keyword arguments for the ellipse, by default None. If None, a default style is used.
        """
        # set up a default stle
        patch_dict = patch_dict_validator(patch_dict, 1)

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
                      **patch_dict[0])

        return ell

    def ellipse_76_86(self,
                      conf=0.95,
                      patch_dict=None,):
        """Uncertainty ellipse for 207Pb/206Pb-238U/206Pb date

        See `here <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval>`__ for more information.
        """
        # set up a default stle
        patch_dict = patch_dict_validator(patch_dict, 1)

        r = stats.chi2.ppf(conf, 2)
        a1 = np.sqrt(self.eigval_238_207[0] * r)
        a2 = np.sqrt(self.eigval_238_207[1] * r)

        # compute rotation from primary eigenvector
        rotdeg = np.rad2deg(np.arccos(self.eigvec_238_207[0, 0]))

        # create
        ell = Ellipse((self.r238_206, self.r207_206),
                      width=a1 * 2,
                      height=a2 * 2,
                      angle=-rotdeg,
                      **patch_dict[0])

        return ell

    def discordance(self, method='relative'):
        """Date discordance.

        Compute discordance according to a given method.

        Parameters
        ----------
        method : str, optional
            Method for computing discordance, by default 'relative'. Valid strings are:

            - 'SK': mixture model with Stacey and Kramers' (1975) common lead model
            - 'p_238_235': use the probability of fit from the concordia age
            - 'relative': relative age different, where the 206/238 vs 207/235 ages are
                used for 206/238 ages younger than 1 Ga, and the 207/206 vs 206/238
                ages are used for ages older than 1 Ga
            - 'relative_68_76': relative age difference using only the 206/238 and 206/207
                ages, computed as 1-(t68/t76)
            - 'concordia-distance': as defined in Equation 8 of `Vermeesch (2021) <http://doi.org/10.5194/gchron-3-247-2021>`__ (see corrigendum).

        Returns
        -------
        d : float
            discordance
        """
        if method == 'SK':
            t = Pb_mix_t(self.r207_206, 1 / self.r206_238)
            r238_206_rad = 1 / (np.exp(l238 * t) - 1)
            # r207_206_rad = 1/u238u235 * (np.exp(l235*t)-1)/(np.exp(l238*t)-1)
            # r207_206_cm = sk_pb(t)[1]/sk_pb(t)[0]
            # L = np.sqrt((r207_206_rad-r207_206_cm)**2 + (r238_206_rad)**2)
            # l = np.sqrt((1/self.r206_238-r238_206_rad)**2 + (self.r207_206-r207_206_rad)**2)
            d = 1 - (1 / self.r206_238) / r238_206_rad
            # d = l/L
        elif method == 'relative':
            cur68_age = self.date68(conf=None)
            # use 75 vs 68 for younger than 1 Ga
            if cur68_age < 1000:
                d = 1 - cur68_age/self.date75(conf=None)
            else:
                d = 1 - cur68_age / self.date76(conf=None)
        elif method == 'concordia-distance':
            tc = self.date_207_238_concordia()[0]
            R68tc = np.exp(l238 * tc) - 1
            R75tc = np.exp(l235 * tc) - 1
            dx = 1/np.sqrt(2) * (np.log(self.r238_206) + np.log(R68tc))
            dy = np.sqrt(2/3) * (np.log(self.r207_206) - np.log((1/u238u235)*R75tc/R68tc))
            d = 100*np.sqrt(dx**2 + dy**2)
        elif method == 'relative_76_68':
            d = 1 - self.date68(conf=None) / self.date76(conf=None)
        elif method == 'absolute_76_68':
            d = np.abs(self.date76(conf=None) - self.date68(conf=None))
        elif method == 'relative_68_75':
            d = 1 - self.date68(conf=None) / self.date75(conf=None)
        elif method == 'absolute_68_75':
            d = np.abs(self.date75(conf=None) - self.date68(conf=None))
        elif method == 'p_68_75':
            tc = self.date_235_238_concordia()[0]
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

            d = dx(self.date76(conf=None)) * np.sin(
                np.arctan(
                    dy(self.date68(conf=None)) / dx(self.date76(conf=None))))
        elif method == 'aitchison_68_75':

            def dx(t):
                return np.log(self.r207_235) - np.log(np.exp(l235 * t) - 1)

            def dy(t):
                return np.log(self.r206_238) - np.log(np.exp(l238 * t) - 1)

            d = dx(self.date68(conf=None)) * np.sin(
                np.arctan(
                    dy(self.date75(conf=None)) / dx(self.date68(conf=None))))

        return d

    def date_207_238_concordia(self):
        """207Pb/206Pb - 206Pb/238U concordia date 
        
        The date is calculate as per `Ludwig (1998) <http://doi.org/10.1016/S0016-7037(98)00059-3>`__, with uncertainty and MSWD.

        Utilizes the error transformation for 207Pb/235U computed from 206Pb/238U and 207Pb/206Pb as per Appendix A of Ludwig (1998).

        Returns
        -------
        t: float
            Concordia date (Myr) for 207Pb/206Pb - 206Pb/238U
        t_std: float
            Standard deviation of age
        poc: float
            Probability of concordance. 
        """
        # relative errors
        SX = self.r206_238_std / self.r206_238
        SY = self.r207_206_std / self.r207_206
        # Sx = self.r207_235_std / self.r207_235
        Sy = self.r206_238_std / self.r206_238
        rhoXY = self.rho86_76
        # error transformation for 207/235 computed from 206/238 and 207/206
        Sx = np.sqrt(SX**2 + SY**2 - 2 * SX * SY * rhoXY)
        rhoxy = (SX**2 - SX * SY * rhoXY) / (Sx * Sy)
        # 207/235 computed from 206/238 and 207/206
        x = self.r207_206 * self.r206_238 * u238u235
        y = self.r206_238
        sigx = x * Sx
        sigy = self.r206_238_std

        # get omega for a given t
        def get_omega(t):
            """
            This function uses the error transformations in Appendix A of Ludwig (1998)
            """
            Px = t * np.exp(l235 * t)
            Py = t * np.exp(l238 * t)

            cov_mod = np.array(
                [[sigx**2 + Px**2 * l235_std**2, rhoxy * sigx * sigy],
                 [rhoxy * sigx * sigy, sigy**2 + Py**2 * l238_std**2]])
            omega = np.linalg.inv(cov_mod)
            return omega

        def S_cost(t):
            omega = get_omega(t)

            R = x - np.exp(l235 * t) + 1
            r = y - np.exp(l238 * t) + 1

            S = omega[0,0] * R**2 + omega[1,1] * r**2 + 2 * R * r * omega[0,1]

            return S

        opt = minimize_scalar(S_cost, bounds=[0, 5000], method='bounded')
        t = opt.x

        omega = get_omega(t)

        Q235 = l235 * np.exp(l235 * t)
        Q238 = l238 * np.exp(l238 * t)

        t_std = np.sqrt((Q235**2 * omega[0, 0] + Q238**2 * omega[1, 1] +
                         2 * Q235 * Q238 * omega[0, 1])**(-1))

        # probability of concordance
        poc = 1 - stats.chi2.cdf(opt.fun, 1)

        return t, t_std, poc

    def date_235_238_concordia(self):
        """207Pb/235U - 206Pb/238U concordia date 
        
        The date is calculate as per `Ludwig (1998) <http://doi.org/10.1016/S0016-7037(98)00059-3>`__, with uncertainty and MSWD.

        Returns
        -------
        t : float
            Concordia date (Myr) for 207Pb/235U - 206Pb/238U
        t_std : float
            Standard deviation of age
        poc : float
            Probability of concordance.
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

            S = omega[0,0] * R**2 + omega[1,1] * r**2 + 2 * R * r * omega[0,1]

            return S

        opt = minimize_scalar(S_cost, bounds=[0, 5000], method='bounded')
        t = opt.x

        omega = get_omega(t)

        Q235 = l235 * np.exp(l235 * t)
        Q238 = l238 * np.exp(l238 * t)

        t_std = np.sqrt((Q235**2 * omega[0, 0] + Q238**2 * omega[1, 1] +
                         2 * Q235 * Q238 * omega[0, 1])**(-1))

        # probability of concordance
        poc = 1 - stats.chi2.cdf(opt.fun, 1)

        return t, t_std, poc

    def date68(self, conf=0.95):
        """206/238 date with uncertainty.

        Parameters
        ----------
        conf : float or None, optional
            Confidence level for interval, by default 0.95. If None, only the age is returned.

        Returns
        -------
        date : float
            Date in Myr
        sig : float
            Standard deviation of date
        confint : float
            Confidence interval at desired confidence level
        """
        date = t238(self.r206_238)
        if conf == None:
            return date
        else:
            sig = (1/l238) * self.r206_238_std/(self.r206_238 + 1)
            conf = 1 - (1 - conf) / 2
            confint = stats.norm.ppf(conf, date, sig) - date
            return date, sig, confint

    def date75(self, conf=0.95):
        """207/235 date with uncertainty.

        Parameters
        ----------
        conf : float or None, optional
            Confidence level for interval, by default 0.95. If None, only the age is returned.

        Returns
        -------
        date : float
            Date in Myr
        sig : float
            Standard deviation of date
        confint : float
            Confidence interval at desired confidence level.
        """
        date = t235(self.r207_235)
        if conf == None:
            return date
        else:
            sig = (1/l235) * self.r207_235_std/(self.r207_235 + 1)
            conf = 1 - (1 - conf) / 2
            confint = stats.norm.ppf(conf, date, sig) - date
            return date, sig, confint

    def date76(self, conf=0.95, u238u235=u238u235):
        """207/206 date with uncertainty.

        Parameters
        ----------
        conf : float or None, optional
            Confidence level for interval, by default 0.95. If None, only the age is returned.
        u238u235 : float, optional
            Ratio of U238 to U235, by default u238u235

        Returns
        -------
        date : float
            Date in Myr
        sig : float
            Standard deviation of date
        confint : float
            Confidence interval at desired confidence level.
        """
        date = t207(self.r207_206, u238u235)

        if conf == None:
            return date
        else:
            sig = t207(self.r207_206 + self.r207_206_std) - date
            conf = 1 - (1 - conf) / 2
            confint = stats.norm.ppf(conf, date, sig) - date
            return date, sig, confint


def sk_pb(t, t0=4.57e3, t1=3.7e3, mu1=7.19, mu2=9.74, x0=9.307, y0=10.294):
    """Common lead model from `Stacey and Kramers (1975) <https://doi.org/10.1016/0012-821X(75)90088-6>`__.

    This model implements a two stage process for the evolution of common lead.

    For times before :math:`t_1 = 3.7` Ga, the following equations describe the temporal evolution of common lead:

    .. math::

        \\begin{align}
            \\frac{^{206}\\text{Pb}}{^{204}\\text{Pb}}(t) &= \\left(\\frac{^{206}\\text{Pb}}{^{204}\\text{Pb}}\\right)_0 + \\mu_1\\,\\left(e^{\\lambda_{238}t_0}-e^{\\lambda_{238}t}\\right) \\\\
            \\frac{^{207}\\text{Pb}}{^{204}\\text{Pb}}(t) &= \\left(\\frac{^{207}\\text{Pb}}{^{204}\\text{Pb}}\\right)_0 + \\frac{^{235}\\text{U}}{^{238}\\text{U}} \\, \\mu_1\\, \\left(e^{\\lambda_{235}t_0}-e^{\\lambda_{235}t}\\right)
        \\end{align}

    For times after :math:`t_1 = 3.7` Ga, the following equations describe the temporal evolution of common lead:

    .. math::
            
        \\begin{align}
        \\frac{^{206}\\text{Pb}}{^{204}\\text{Pb}}(t) &= \\left(\\frac{^{206}\\text{Pb}}{^{204}\\text{Pb}}\\right)_1 + \\mu_2\\,\\left(e^{\\lambda_{238}t_1}-e^{\\lambda_{238}t}\\right) \\\\
        \\frac{^{207}\\text{Pb}}{^{204}\\text{Pb}}(t) &= \\left(\\frac{^{207}\\text{Pb}}{^{204}\\text{Pb}}\\right)_1 + \\frac{^{235}\\text{U}}{^{238}\\text{U}} \\, \\mu_2\\, \\left(e^{\\lambda_{235}t_1}-e^{\\lambda_{235}t}\\right)
        \\end{align}

    where :math:`\\left(\\frac{^{206}\\text{Pb}}{^{204}\\text{Pb}}\\right)_1` and :math:`\\left(\\frac{^{207}\\text{Pb}}{^{204}\\text{Pb}}\\right)_1` are the above equations evaluated at :math:`t=t_1`.

    Parameters
    ----------
    t : 1d array like
        Time(s) (in Ma) for which to return isotopic ratios for the linear mixing model from concordia to mantle
    t0 : float, optional
        Age of earth in Myr, by default 4570.
    t1 : float, optional
        Age of transition between model stages, by default 3700.
    mu1 : float, optional

    mu2 : float, optional
    x0 : float, optional
        206Pb/204Pb ratio for troilite lead, by default 9.307.
    y0 : float, optional
        207Pb/204Pb ratio for troilite lead, by default 10.294.

    Returns
    -------
    r206_204 : 1d array
        206/204 ratio(s) for the given time(s)
    r207_204 : 1d array
        207/204 ratio(s) for the given time(s)

    Examples
    --------
    .. plot::
        :include-source:

        import radage
        import numpy as np
        import matplotlib.pyplot as plt
        t = np.linspace(0, 4500, 50)
        r206_204, r207_204 = radage.sk_pb(t)
        plt.figure(figsize=(6, 4))
        plt.plot(r206_204, r207_204, linewidth=4)
        plt.grid()
        plt.xlabel(r'${}^{206}\\mathrm{Pb}/{}^{204}\\mathrm{Pb}$')
        plt.ylabel(r'${}^{207}\\mathrm{Pb}/{}^{204}\\mathrm{Pb}$')
        plt.title('Stacey and Kramers (1975) Common Lead Model')
        plt.show()

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


def Pb_mix_t(r207_206, r238_206, Pbc='SK'):
    """Solve for age given a pair of 207/206 & 238/206 isotopic ratios and a common lead model.

    Given a pair of 207/206 & 238/206 isotopic ratios, this function solves for the age such that one has a lead mixing model that results in identical Stacey & Kramers' (1975) common lead model and lower intercept concordia ages while also passing through the observed pair of ratios.

    Parameters
    ----------
    r207_206 : float
        207/206 ratio
    r238_206 : float
        238/206 ratio
    Pbc : str or float, optional
        Initial common lead model to use, by default 'SK'. If float, then the common lead model is defined by the input value.
    
    Returns
    -------
    t : float
        Age in Myr

    Examples
    --------
    .. plot::
        :include-source:

        import radage
    """

    if Pbc == 'SK':
        def Pbc0(t): 
            r206_204, r207_204 = sk_pb(t)
            r207_206_0 = r207_204 / r206_204
            return r207_206_0
    else:
        # check value is a float
        assert isinstance(Pbc, float), 'Pbc must be a float or "SK"'
        # make sure value is greater than zero and less than 2
        assert Pbc > 0 and Pbc < 2, 'Pbc must be between 0 and 2'
        def Pbc0(t):
            r207_206_0 = Pbc
            return r207_206_0

    def cost(t):
        """
        defines a cost metric to search over appropriate times
        """
        r207_206_0 = Pbc0(t)
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


def discordance_filter(ages,
                       method='relative',
                       threshold=0.03,
                       system_threshold=False):
    """
    function to filter on discordance

    Parameters
    ----------
    ages : list
        List of UPb.radage objects
    method : string
        'relative', 'absolute', 'aitchison'
        discordance metric to use. Discordance is evaluated between
        207/206-238/206 ages for samples with 207/206 ages > 1 Ga, and 
        206/238-207/235 ages for sample with 207/206 ages < 1 Ga
    threshold : float, optional
        discordance value for chosen filtering method above which to flag
        ages as discordant
    system_threshold : boolean, optional
        whether or not to switch discordance metrics between the 207/206-238/206 
        and 206/238-207/235 systems at 1 Ga. If false, then discordance is always
        computed between the 207/206-238/206 systems. If true, only works for 
        aithison, relative, and absolute discordance metrics.

    Returns
    -------
    ages_conc : list
        List of UPb.radage objects that pass the discordance filter
    idx : np.ndarray
        Indices of the ages that pass the discordance filter
    """
    ages_conc = []
    idx = np.zeros(len(ages), dtype=bool)

    if system_threshold:
        for ii, age in enumerate(ages):
            if age.date76(conf=None) > 1000:
                cur_method = method + '_76_68'
            else:
                cur_method = method + '_68_75'
            d = np.abs(age.discordance(method=cur_method))
            if d < threshold:
                ages_conc.append(age)
                idx[ii] = True

    else:
        for ii, age in enumerate(ages):
            d = np.abs(age.discordance(method=method + '_76_68'))
            if d < threshold:
                ages_conc.append(age)
                idx[ii] = True
    return ages_conc, idx


def root_fun_76_86(r238_206, m, b):
    """Root function for finding the lower intercept date in Tera-Wasserburg space
    Parameters
    ----------
    r238_206 : float
        238/206 ratio
    m : float
        Slope of the line in Tera-Wasserburg space
    b : float
        Intercept of the line in Tera-Wasserburg space
    Returns
    -------
    r207_206_conc - r207_206_disc : float
        Difference between the concordia and discordia ratios
    """
    r207_206_conc = concordia_tw(t238(1/r238_206))[1]
    r207_206_disc = m*r238_206 + b
    return r207_206_conc - r207_206_disc

def discordia_date_76_86(UPbs, conf=None, n_mc=500, Pbc=None):
    """Lower intercept date in Tera-Wasserburg space

    Parameters
    ----------
    UPbs : list
        List of UPb.radage objects
    conf : float, optional
        Confidence level for interval, by default None. If None, only the lower intercept date is returned. Otherwise, a range corresponding to the confidence level is returned.
        Valid values are between 0 and 1.
    n_mc : int, optional
        Number of Monte Carlo iterations to use for estimating uncertainty, by default 1000. This is only used if conf is not None.
    Pbc : float, optional
        Common lead ratio to use, by default None. If None, the common lead ratio is estimated from the data. If a float, the common lead ratio is fixed to this value. If 'SK', the common lead ratio is determined such that it is equal to the Stacey and Kramers (1975) common lead ratio at the lower intercept date.

    Returns
    -------
    result : dict
        Dictionary containing the following:
        date : float
            Date in Ma corresponding to the lower intercept of the line in Tera-Wasserburg space.
        slope : float
            Slope of the line in Tera-Wasserburg space.
        slope_std : float
            Standard deviation of the slope.
        intercept : float
            Common lead ratio determined in the calculation. Returns same value as input if Pbc is a float.
        intercept_std : float
            Standard deviation of the common lead ratio.
        mswd : float
            Mean square weighted deviation of the fit.
        confint : list
            List of lower and upper bounds of the confidence interval for the lower intercept date. This is only computed if conf is not None.
    """
    
    # gather relevant ratios
    r238_206 = np.array([x.r238_206 for x in UPbs])
    r238_206_std = np.array([x.r238_206_std for x in UPbs])
    r207_206 = np.array([x.r207_206 for x in UPbs])
    r207_206_std = np.array([x.r207_206_std for x in UPbs])
    rho = np.array([x.rho86_76 for x in UPbs])

    # helper function to append fixed common lead ratios to the data
    def Pbc_append(Pbc):
        """Append common lead ratio to the data"""
        r238_206_app = np.append(r238_206, 0)
        r238_206_std_app = np.append(r238_206_std, 0.00001)
        r207_206_app = np.append(r207_206, Pbc)
        r207_206_std_app = np.append(r207_206_std, 0.00001)
        rho_app = np.append(rho, 0.0)
        return r238_206_app, r238_206_std_app, r207_206_app, r207_206_std_app, rho_app

    # if anchoring to a common lead ratio, append it to the data
    if (Pbc is not None) and (Pbc != 'SK'):
        # check value is a float
        assert isinstance(Pbc, float), 'Pbc must be a float or "SK"'
        # make sure value is greater than zero and less than 2
        assert Pbc > 0 and Pbc < 2, 'Pbc must be between 0 and 2'
        # append common lead ratio to the data
        r238_206_fit, r238_206_std_fit, r207_206_fit, r207_206_std_fit, rho_fit = Pbc_append(Pbc)
    # if using SK, solve for slope and intercept such that the common lead ratio is equal to the Stacey and Kramers (1975) common lead ratio at the lower intercept date
    elif Pbc == 'SK':
        def cost(t):
            """Cost function for finding optimal t that fits data and common lead model"""
            r206_204, r207_204 = sk_pb(t)
            Pbc = r207_204 / r206_204
            x, x_sig, y, y_sig, r = Pbc_append(Pbc)
            wx = 1/x_sig**2
            wy = 1/y_sig**2
            m, b, _, _, _, _ = yorkfit(x, 
                                       y,
                                       wx,
                                       wy,
                                       r)
            # don't include fixed point in the fit
            mswd = line_mswd(m, b, x[0:-1], y[0:-1], wx[0:-1], wy[0:-1], r[0:-1])
            return mswd
        # find optimal t that fits data and common lead model
        t_opt = minimize_scalar(cost, bounds=(0, 5000), method='bounded').x
        Pbc = sk_pb(t_opt)[1] / sk_pb(t_opt)[0]
        # append Pbc at t_opt to the data
        r238_206_fit, r238_206_std_fit, r207_206_fit, r207_206_std_fit, rho_fit = Pbc_append(Pbc)
    else:
        r238_206_fit, r238_206_std_fit, r207_206_fit, r207_206_std_fit, rho_fit = r238_206, r238_206_std, r207_206, r207_206_std, rho

    # compute slope and intercept of line in Tera-Wasserburg space
    m, b, m_sig, b_sig, _, _ = yorkfit(r238_206_fit, 
                                    r207_206_fit,
                                    1/r238_206_std_fit**2,
                                    1/r207_206_std_fit**2, 
                                    rho_fit)
    # compute MSWD using just input data
    mswd = line_mswd(m, b, r238_206, r207_206, 1/r238_206_std**2, 1/r207_206_std**2, rho)

    
    # find root, initial 238/206 = 500
    if conf is not None:
        m_mc = np.random.normal(m, m_sig, n_mc)
        b_mc = np.random.normal(b, b_sig, n_mc)
        dates = np.zeros(n_mc)
        for ii in range(n_mc):
            sol = root_scalar(root_fun_76_86, args=(m_mc[ii], b_mc[ii]), x0=500, method='newton')
            dates[ii] = t238(1/sol.root)
        confint = np.quantile(dates, [(1-conf)/2, 1-(1-conf)/2])
        date = np.mean(dates)
    else:
        sol = root_scalar(root_fun_76_86, args=(m, b), x0=500, method='newton')
        date = t238(1/sol.root)
        confint = None

    result = {'date': date, 
              'slope': m,
              'slope_sig': m_sig,
              'intercept': b, 
              'intercept_sig': b_sig, 
              'x_bar': np.mean(r238_206),
              'mswd': mswd,
              'confint': confint}

    return result


def wc1_corr(wc1_UPbs):
    """Correct 238/206 ratios of WC1 analyses to achieve lower intercept age of 254 Ma with fixed common Pb 207/206 ratio of 0.85.

    Parameters
    ----------
    wc1_UPbs : list
        List of UPb.radage objects for WC1 analyses

    Returns
    -------
    factor: float
        Factor by which to multiply 238/206 ratios to achieve lower intercept age of 254 Ma with fixed common Pb 207/206 ratio of 0.85.
    """
    # compute slope with fixed common lead ratio of 0.85
    b = 0.85
    result = discordia_date_76_86(wc1_UPbs, Pbc=b)
    m = result['slope']

    def cost(factor):
        """Cost function for finding factor to multiply 238/206 ratios by"""
        # compute lower intercept date
        sol = root_scalar(root_fun_76_86, args=(m/factor, b), x0=500, method='newton')
        date = t238(1/sol.root)

        return (date - 254)**2
    
    # find factor that minimizes cost function
    factor = minimize_scalar(cost, bounds=(0, 2), method='bounded').x

    return factor


def kde(radages, t,
        kernel='gauss',
        bw='adaptive',
        weights='uncertainty',
        **kwargs):
    """Evaluate kde for radages at times t. 
    
    The 238/206 - 207/206 concordia ages are used for plotting, and the associated uncertainties are used as weights.

    Parameters
    ----------
    radages : arraylike
        list of radages.
    t : arraylike
        times at which to evaluate the kde.
    bw : str or float, optional
        Bandwidth, by default 'adaptive'. Valid strings are 'adaptive', 'scott', 'botev'. If float, specifies a fixed bandwidth directly.
    kernel : str, optional
        Kernel function to use, by default 'epa'. Valid strings are 'epa', 'gauss'.
    weights : arraylike or None or str, optional
        Weights for each age, by default 'uncertainty'. Valid values are 'uncertainty', None, or an array of floats of the same length as radages.
    **kwargs : dict, optional
        Additional arguments to pass to helper.kde_base

    Returns
    -------
    kde_est : arraylike
        KDE estimate at times t
    """
    # compute concordia ages
    dates_conc = np.array([age.date_207_238_concordia()[0:2] for age in radages])
    ages_in = dates_conc[:, 0]

    if weights is None:
        w = np.ones(len(ages_in))
    elif weights == 'uncertainty':
        w = 1 / dates_conc[:, 1]**2
    else:
        assert len(weights) == len(ages_in), 'Weights must be the same length as radages'

    # evaluate kde
    kde_est = kde_base(ages_in, t, kernel=kernel, bw=bw, w=w, **kwargs)

    return kde_est


def line_mswd(m, b, x, y, wx, wy, r):
    """Reduced chi-squared statistic for a linear fit to data with errors in both x and y.
    This

    Parameters
    ----------
    m : float
        Slope of line
    b : float
        Intercept of line
    x : array-like
        x-values
    y : array-like
        y-values
    wx : array-like
        Weights for x-values (typically 1/sigma^2)
    wy : array-like
        Weights for y-values (typically 1/sigma^2)
    r : array-like
        Correlation coefficients of errors in x and y, must be between -1 and 1

    Returns
    -------
    mswd : float
        Reduced chi-squared statistic for residuals with respect to the line
    """
    # for variable meanings, see yorkfit()
    n = len(x)
    assert len(x) == len(y), 'x and y must be the same length'
    alpha = np.sqrt(wx * wy)
    W = (wx * wy) / (wx + m**2 * wy - 2 * m * r * alpha)
    mswd = np.sum(W * (y - m * x - b)**2) / (n - 2)
    return mswd

def yorkfit(x, y, wx, wy, r, thres=1e-3):
    """
    Implementation of York 1969 10.1016/S0012-821X(68)80059-7

    Parameters
    ----------
    x : array-like 
        Mean x-values
    y : array-like
        Mean y-values
    wx : array-like
        Weights for x-values (typically 1/sigma^2)
    wy : array-like 
        Weights for y-values (typically 1/sigma^2)
    r : array-like
        Correlation coefficients of errors in x and y, must be between -1 and 1
    
    Returns
    -------
    b : float
        Maximum likelihood estimate for slope of line
    a : float 
        Maximum likelihood estimate for intercept of line
    b_sig : float
        Standard deviation of slope for line
    a_sig : float
        Standard deviation of intercept for line
    mswd : float
        Reduced chi-squared statistic for residuals with respect to the maximum likelihood linear model
    ab_cov : float
        Covariance of slope and intercept for line
    """
    assert len(x) == len(y), 'x and y must be the same length'
    assert len(wx) == len(x), 'wx must be the same length as x'
    assert len(wy) == len(y), 'wy must be the same length as y'
    assert np.all(np.abs(r) <= 1), 'r must be between -1 and 1'
    
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
    b_sig = np.sqrt(1 / np.sum(W * u**2))
    a_sig = np.sqrt(1 / np.sum(W) + x_adj_bar**2 * b_sig**2)
    ab_cov = -x_adj_bar * b_sig**2
    # compute goodness of fit (reduced chi-squared statistic)
    mswd = line_mswd(b, a, x, y, wx, wy, r)

    return b, a, b_sig, a_sig, mswd, ab_cov


def weighted_mean(ages, ages_s, sig_method='naive', standard_error=True):
    """Weighted-mean age computation

    Parameters
    ----------
    ages : array-like
        Age means
    ages_s : array-like
        Age standard deviations (same length as ages)
    sig_method : str, optional
        Method for computing the standard error of the weighted mean, by default 'naive'. Valid strings are 'naive', 'unbiased'
        For unbiased, see 
            https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights,
            https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf 
    standard_error : bool, optional
        Whether to return the standard error of the weighted mean, by default True. If sig_method is 'naive', then this parameter is True.

    Returns
    -------
    mu : float
        Weighted mean age
    sig : float
        Standard deviation or error of weighted mean age, depending on standard_error value
    mswd : float
        Reduced chi-squared statistic for residuals with respect to the weighted mean
    """
    # weights
    w = 1/(ages_s**2)
    # weighted mean
    mu = np.sum(ages*w)/np.sum(w)
    # verify sig_method is valid
    assert sig_method in ['naive', 'unbiased', 'biased'], 'sig_method must be one of naive, unbiased, biased'
    # ignore standard error if sig_method is naive
    if sig_method == 'naive':
        standard_error = True
    # naive
    if sig_method == 'naive':
        sig2 = 1/np.sum(w)
        n = 1  # hacky
    # unbiased
    elif sig_method == 'unbiased':
        sig2 = np.sum(w)/(np.sum(w)**2-np.sum(w**2))*np.sum(w*(ages-mu)**2)
        n = np.sum(w)**2/np.sum(w**2)   # effective sample size
    # biased
    elif sig_method == 'biased':
        n = len(ages)   # sample size
        sig2 = (np.sum(w*(ages-mu)**2)/np.sum(w)) * n / (n-1)
    
    if standard_error:
       sig = np.sqrt(sig2/n)
    else:
        sig = np.sqrt(sig2)

    mswd = np.sum(w)/(np.sum(w)**2-np.sum(w**2)) * \
        np.sum((w*(ages-mu)**2)/ages_s**2)
    
    return mu, sig, mswd


def get_ages(df):
    """Produce UPb age objects

    Create UPb age objects from data in GeochemDB. This function assumes that the following quantities are present in the GeochemDB database and form columns in the input DataFrame:
    - Pb206/U238
    - Pb207/U235
    - Pb207/Pb206
    - rho 206Pb/238U v 207Pb/235U
    - rho 207Pb/206Pb v 238U/206Pb
    These quantitues should have both mean and uncertainty values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of U-Pb measurements, ideally from GeochemDB.GeochemDB.measurements_by_sample(). Must have a hierarchical column index with the following levels: 'Pb206/U238', 'Pb207/U235', 'Pb207/Pb206', 'rho 206Pb/238U v 207Pb/235U', 'rho 207Pb/206Pb v 238U/206Pb'. Each level should have 'mean' and 'uncertainty' sublevels. 
    
    Returns
    -------
    ages : list
        List of UPb objects.
    """
    # cols
    cols = [('Pb206/U238', 'mean'), 
            ('Pb206/U238', 'uncertainty'), 
            ('Pb207/U235', 'mean'),
            ('Pb207/U235', 'uncertainty'),
            ('Pb207/Pb206', 'mean'),
            ('Pb207/Pb206', 'uncertainty'),
            ('rho 206Pb/238U v 207Pb/235U', 'mean'),
            ('rho 207Pb/206Pb v 238U/206Pb', 'mean')]
    
    ages = []
    for ii in range(df.shape[0]):
        ages.append(UPb(df.iloc[ii][cols[0]], 
                              df.iloc[ii][cols[1]]/2,
                              df.iloc[ii][cols[2]], 
                              df.iloc[ii][cols[3]]/2, 
                              df.iloc[ii][cols[4]], 
                              df.iloc[ii][cols[5]]/2, 
                              df.iloc[ii][cols[6]],
                              df.iloc[ii][cols[7]], name=df.index[ii]))
    return ages


###
### HAFNIUM
###

# Vervoort et al. 2018 values
# Lu_DM = 0.03976 
# Hf_DM = 0.283238

Hf_DM = 0.283225
Lu_DM = 0.0383

# Vervoort & Blichert-Toft 1999
Hf_CHUR = 0.282772
Lu_CHUR = 0.0332

lam_Lu = 1.867e-5 # My-1

def eHf(Hf176Hf177, Lu176Hf177, t, Hf_CHUR=Hf_CHUR, Lu_CHUR=Lu_CHUR):
    """epsilon hafnium 
    
    for a given set of ratios

    Parameters
    ----------
    Hf176Hf177 : array-like
        176Hf/177Hf ratio(s)
    Lu176Hf177 : array-like
        176Lu/177Hf ratio(s)
    t : array-like 
        age(s) in Myr. If float, then the same age is used for all ratios.
    Hf_CHUR : float, optional
        176Hf/177Hf ratio for CHUR, by default Hf_CHUR
    Lu_CHUR : float, optional
        176Lu/177Hf ratio for CHUR, by default Lu_CHUR

    Returns
    -------
    eHf : array-like
        epsilon hafnium value(s)
    """
    return 10000*((Hf176Hf177 - Lu176Hf177*(np.exp(lam_Lu*t)-1)) / \
                  (Hf_CHUR - Lu_CHUR*(np.exp(lam_Lu*t)-1)) - 1)
                  
def eHf_DM(t):
    """epsilon Hafnium for depleted mantle

    Parameters
    ----------
    t : array-like
        Age(s) in Myr

    Returns
    -------
    eHf : array-like
        epsilon hafnium value(s) for depleted mantle
    """
    return eHf(Hf_DM, Lu_DM, t)