import pandas as pd
import numpy as np

import scipy.stats as stats


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
    ytan_1 = np.sqrt(r/(1/(sigx**2)*(-m*sigx**2/sigy**2)**2 + 1/sigy**2)) + y
    xtan_1 = -m*sigx**2/sigy**2*(ytan_1-y) + x
    
    ytan_2 = -np.sqrt(r/(1/(sigx**2)*(-m*sigx**2/sigy**2)**2 + 1/sigy**2)) + y
    xtan_2 = -m*sigx**2/sigy**2*(ytan_2-y) + x
    
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
    def __init__(self, r206_238, std206_238, r207_235, std207_235, r207_206, std207_206, rho207_238, rho238_235):
        self.r206_238 = r206_238
        self.std206_238 = std206_238
        self.r207_235 = r207_235
        self.std207_235 = std207_235
        self.r207_206 = r207_206
        self.std207_206 = std207_206
        
        self.rho207_238 = rho207_238 # rho2
        self.rho238_235 = rho238_235 # rho1
        
        # compute eigen decomposition of covariance matrix
        #
        #       | 207/235^2  rho1*207/235*206/238   rho2*207/235
        # COV = |
        #       |
        #
        self.cov = np.array([[self.std207_235**2, self.rho238_235*self.std206_238*self.std207_235],
                        [self.rho238_235*self.std206_238*self.std207_235, self.std206_238**2]])
        self.eigval, self.eigvec = np.linalg.eig(self.cov)
        idx = np.argsort(self.eigval)[::-1] # sort descending
        self.eigval = self.eigval[idx]
        self.eigvec = self.eigvec.T[idx].T
        
        
    """
    produce uncertainty/confidence ellipse
    
    see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval
    """
    def ellipse(self, conf=0.95, facecolor='wheat', edgecolor='k', linewidth=0.5, **kwargs):
        r = stats.chi2.ppf(conf, 2)
        a1 = np.sqrt(self.eigval[0]*r)
        a2 = np.sqrt(self.eigval[1]*r)
        
        # compute rotation from primary eigenvector
        rotdeg = np.arctan(self.eigvec[0, 0]/self.eigvec[1, 0])
        
        # create
        ell  = Ellipse((self.r35, self.r38), width=a1*2, height=a2*2, angle=rotdeg, 
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
            exp_arg[ii] = np.matmul((np.vstack([r35[ii], r38[ii]])-mu).T, np.matmul(np.linalg.inv(self.cov), (np.vstack([r35[ii], r38[ii]])-mu)))
            
        return np.any(exp_arg <= stats.chi2.ppf(conf, 2))
    
    """
    return 238 age with interval for desired confidence
    """
    def age238(self, conf=0.95, n=1e6):
        n = int(n)
        age = np.log(self.r38+1)/l238
        sig = np.std(np.log(stats.norm.rvs(self.r38, self.r206_238, size=n)+1)/l238)
        conf = 1 - (1-conf)/2
        confint = stats.norm.ppf(conf, age, sig)-age
        return age, sig, confint
    
    """
    return 238 age with interval for desired confidence
    """
    def age235(self, conf=0.95, n=1e6):
        n = int(n)
        age = np.log(self.r35+1)/l235
        sig = np.std(np.log(stats.norm.rvs(self.r35, self.r207_235, size=n)+1)/l235)
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

