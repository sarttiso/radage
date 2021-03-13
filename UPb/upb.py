import pandas as pd
import numpy as np

import scipy.stats as stats
from scipy.optimize import minimize_scalar

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# in My
l238 = 1.55125e-10*1e6
l238_std = 0.5*l238*0.107/100  # see Schoene 2014 pg. 359
l235 = 9.8485e-10*1e6
l235_std = 0.5*l235*0.137/100

"""
t in My
returns ratios and times corresponding to those ratios in 206-238 vs 207-235 space over a given time interval.
"""
def concordia(t):
    r206_238 = np.exp(l238*t)-1
    r207_235 = np.exp(l235*t)-1

    return r207_235, r206_238

"""
function giving 206/238, 207/235 ratios for bounds on confidence region around concordia at a given t
returns upper bound, then lower bound
"""
def concordia_confint(t, conf=0.95):
    # slope of line tangent to concordia
    m = (l238*np.exp(l238*t))/(l235*np.exp(l235*t))

    # rename coordinates (207/235 = x, 206/238 = y)
    x, y = concordia(t)

    # uncertainty ellipse for given confidence interval
    r = stats.chi2.ppf(conf, 2)

    # uncertainties in x, y at given time t
    sigy = t*np.exp(l238*t)*l238_std
    sigx = t*np.exp(l235*t)*l235_std

    # tangent points to uncertainty ellipse
    with np.errstate(divide='ignore', invalid='ignore'):
        ytan_1 = np.sqrt(r/(1/(sigx**2)*(-m*sigx**2/sigy**2)**2 + 1/sigy**2)) + y
        xtan_1 = -m*sigx**2/sigy**2*(ytan_1-y) + x

        ytan_2 = -np.sqrt(r/(1/(sigx**2)*(-m*sigx**2/sigy**2)**2 + 1/sigy**2)) + y
        xtan_2 = -m*sigx**2/sigy**2*(ytan_2-y) + x

    if np.any(t==0):
        idx = t == 0
        xtan_1[idx] = 0
        ytan_1[idx] = 0
        xtan_2[idx] = 0
        ytan_2[idx] = 0

    return np.vstack([xtan_1, ytan_1]).T, np.vstack([xtan_2, ytan_2]).T


"""
return 238-206 age
"""
def t238(r38_06):
    t = np.log((1/r38_06)+1)/l238
    return t


"""
return 235-207 age
"""
def t235(r35_07):
    t = np.log((1/r35_07)+1)/l235
    return t


"""
class for handling U-Pb ages, where data are given as isotopic ratios and their uncertainties
"""
class UPbAge:
    """
    create object, just need means, stds, and correlations
    """
    def __init__(self, r206_238, r206_238_std, r207_235, r207_235_std, r207_206, r207_206_std, rho238_235,rho207_238):
        self.r206_238 = r206_238
        self.r206_238_std = r206_238_std
        self.r207_235 = r207_235
        self.r207_235_std = r207_235_std
        self.r207_206 = r207_206
        self.r207_206_std = r207_206_std

        self.rho238_235 = rho238_235 # rho1
        self.rho207_238 = rho207_238 # rho2

        # compute eigen decomposition of covariance matrix
        self.cov_235_238 = np.array([[self.r207_235_std**2,
                                      self.rho238_235*self.r206_238_std*self.r207_235_std],
                                     [self.rho238_235*self.r206_238_std*self.r207_235_std,
                                      self.r206_238_std**2]])
        self.eigval, self.eigvec = np.linalg.eig(self.cov_235_238)
        idx = np.argsort(self.eigval)[::-1] # sort descending
        self.eigval = self.eigval[idx]
        self.eigvec = self.eigvec.T[idx].T


    """
    produce uncertainty/confidence ellipse (need to make explicitly the 207/235 vs 206/238 ellipse)

    see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval
    """
    def ellipse(self, conf=0.95, facecolor='wheat', edgecolor='k', linewidth=0.5, **kwargs):
        r = stats.chi2.ppf(conf, 2)
        a1 = np.sqrt(self.eigval[0]*r)
        a2 = np.sqrt(self.eigval[1]*r)

        # compute rotation from primary eigenvector
        rotdeg = np.arctan(self.eigvec[0, 0]/self.eigvec[1, 0])

        # create
        ell  = Ellipse((self.r207_235, self.r206_238), width=a1*2, height=a2*2, angle=rotdeg,
                       facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, **kwargs)

        return ell

    """
    at some confidence level, is the grain concordant (does concordia pass through confidence region)
    """
    def isconcordant(self, conf=0.95, n=1e3):
        n = int(n)
        t = np.linspace(0, int(4.6e3), n)
        r35, r38 = concordia(t)
        mu = np.vstack([self.r35, self.r38])

        exp_arg = np.zeros(n)
        for ii in range(n):
            exp_arg[ii] = np.matmul((np.vstack([r35[ii], r38[ii]])-mu).T, np.matmul(np.linalg.inv(self.cov_235_238), (np.vstack([r35[ii], r38[ii]])-mu)))

        return np.any(exp_arg <= stats.chi2.ppf(conf, 2))

    """
    give 235, 238 concordia age as per Ludwig 1998, with uncertainty and MSWD
    """
    def age_235_238_concordia(self):

        # get omega for a given t
        def get_omega(t):
            P235 = t*np.exp(l235*t)
            P238 = t*np.exp(l238*t)

            cov_mod = np.array([[self.r207_235_std**2 + P235**2*l235_std**2,
                                 self.rho238_235*self.r206_238_std*self.r207_235_std],
                                [self.rho238_235*self.r206_238_std*self.r207_235_std,
                                 self.r206_238_std**2 + P238**2*l238_std**2]])
            omega = np.linalg.inv(cov_mod)
            return omega

        # cost function for concordia from Ludwig 1998, using their notation
        def S_cost(t):
            omega = get_omega(t)

            R = self.r207_235 - np.exp(l235*t) + 1
            r = self.r206_238 - np.exp(l238*t) + 1

            S = omega[0, 0]*R**2 + omega[1, 1]*r**2 + 2*R*r*omega[0, 1]

            return S

        opt = minimize_scalar(S_cost, bounds=[0, 5000], method='bounded')
        t = opt.x

        omega = get_omega(t)

        Q235 = l235*np.exp(l235*t)
        Q238 = l238*np.exp(l238*t)

        t_std = np.sqrt((Q235**2*omega[0, 0] + Q238**2*omega[1, 1] + 2*Q235*Q238*omega[0,1])**(-1))

        MSWD = 1 - stats.chi2.cdf(opt.fun, 1)

        return t, t_std, MSWD

    """
    return 238 age with interval for desired confidence
    """
    def age238(self, conf=0.95, n=1e6):
        n = int(n)
        age = np.log(self.r206_238+1)/l238
        sig = np.std(np.log(stats.norm.rvs(self.r206_238, self.r206_238_std, size=n)+1)/l238)
        conf = 1 - (1-conf)/2
        confint = stats.norm.ppf(conf, age, sig)-age
        return age, sig, confint

    """
    return 238 age with interval for desired confidence
    """
    def age235(self, conf=0.95, n=1e6):
        n = int(n)
        age = np.log(self.r207_235+1)/l235
        sig = np.std(np.log(stats.norm.rvs(self.r207_235, self.r207_235_std, size=n)+1)/l235)
        conf = 1 - (1-conf)/2
        confint = stats.norm.ppf(conf, age, sig)-age
        return age, sig, confint


"""
draw intelligent concordia plot

ages: list of UPbAges to plot
"""
# def plotconcordia(ages):



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
a: maximum likelihood estimate for intercept of lin
b_sig: standard deviation of slope for line
a_sig: standard deviation of intercept for line
mswd: reduced chi-squared statistic for residuals with respect to the maximum likelihood linear model
"""
def yorkfit(x, y, wx, wy, r, thres=1e-3):
    n = len(x)
    # get first guess for b
    b = stats.linregress(x, y)[0]

    # initialize various quantities
    alpha = np.sqrt(wx*wy)

    # now iterate as per manuscript to improve b
    delta = thres+1
    while delta > thres:
        # update values from current value of b
        W = (wx*wy)/(wx + b**2*wy - 2*b*r*alpha)
        xbar = np.sum(x*W)/np.sum(W)
        ybar = np.sum(y*W)/np.sum(W)
        U = x - xbar
        V = y - ybar
        beta = W * (U/wy + b*V/wx - (b*U+V)*r/alpha)
        # update b
        b_new = np.sum(W*beta*V)/np.sum(W*beta*U)
        delta = np.abs(b_new - b)
        b = b_new

    # compute a
    a = ybar - b*xbar
    # compute adjusted x, y
    x_adj = xbar + beta
    x_adj_bar = np.sum(W*x_adj)/np.sum(W)
    u = x_adj - x_adj_bar
    # compute parameter uncertainties
    b_sig = 1/np.sum(W*u**2)
    a_sig = 1/np.sum(W) + x_adj_bar**2*b_sig**2
    # compute goodness of fit (reduced chi-squared statistic)
    mswd = np.sum(W*(y-b*x-a)**2)/(n-2)

    return b, a, b_sig, a_sig, mswd
