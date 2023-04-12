import gzip, glob, math, random, pickle, time, subprocess, warnings
import numpy as np
# from settings_V0_5 import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance
from scipy import optimize
from scipy.stats import norm


def shift_func(x, a):
    '''
    Shift function
    :param x: [ndarray or float] data
    :param a: [float] shift
    :return: [ndarray] shifted data
    '''
    return x-a
def rescale_func(x, a):
    '''
    Rescaling function
    :param x: [ndarray or float] data
    :param a: [float] rescale factor
    :return: [ndarray] rescaled data
    '''
    return x/a

def exponential_decay(x, a):
    '''
    Exponential decay function
    :param x: [ndarray or float] data
    :param a: [float] parameter to optimize
    :return: [ndarray] exponential decay
    '''
    return np.exp(-a*x)
def power_law(x, a, b):
    '''
    Power law function
    :param x: [ndarray or float] data
    :param a: [float] parameter to optimize
    :param b: [float] parameter to optimize
    :return: [ndarray] power law
    '''
    return a*(x**b)
def power_law_shifted(x, a, b):
    '''
    Power law function shifted -> a(x+c)**b with condition f(0)=1 -> (x/a+1)**b
    :param x: [ndarray or float] data
    :param a: [float] parameter to optimize
    :param b: [float] parameter to optimize
    :return: [ndarray] shifted power law
    '''
    return ((x/a+1)**b)
def power_lawb(x, a, b,c ,d):
    '''
    Power law function
    :param x: [ndarray or float] data
    :param a: [float] parameter to optimize
    :param b: [float] parameter to optimize
    :return: [ndarray] power law
    '''
    return a*(((x+c)/d)**b)




class Distribution():
    '''
    Class used to fit a statistical distribution and evaluate the fitting
    :arg self.dist: distribution function [function]
    :arg self.dist_nargs: number of parameters of the distribution function [int]
    :arg self.apply_pre_shift_and_rescale: boolean specifying if shifting/rescaling should be tested before dist function [bool]
    :arg self.apply_post_shift_and_rescale: boolean specifying if shifting/rescaling should be tested after dist function [bool]
    :arg self.func: function to optimize (distribution with shifting/rescaling if applied) [function]
    :arg self.parameters: parameter names of the function to optimize [list of string]
    :arg self.fit_done: boolean specifying if the fitting has already been done [bool]
    :arg self.fit_it: maximum number of iterations for the fitting [int]
    :arg self.fit_p0: initial parameters used for the fitting [list]
    :arg self.popt: optimized parameters after the fitting [tuple]
    :arg self.pcov: covariance matrix of the fitting [numpy.ndarray 2D]
    :arg self.wasserstein_distance: wasserstein distance computed as a score of the fitting [float]
    '''

    def __init__(self, distribution_func, pre_shift_and_rescale=True, post_shift_and_rescale=True):
        '''
        Constructor of the class Distribution
        :param distribution_func: distribution function [function]
        :param pre_shift_and_rescale: True if shifting/rescaling should be tested before dist function [bool]
        :param post_shift_and_rescale: True if shifting/rescaling should be tested after dist function [bool]
        '''
        self.dist = distribution_func
        self.dist_nargs = self.dist.__code__.co_argcount - 1
        self.apply_pre_shift_and_rescale = pre_shift_and_rescale
        self.apply_post_shift_and_rescale = post_shift_and_rescale

        self.parameters = []
        if self.apply_pre_shift_and_rescale:
            self.func1 = lambda x, pre_shift, pre_rescaling, *args: self.dist(rescale_func(shift_func(x, pre_shift), pre_rescaling),*args)
            self.parameters = ["pre_shift","pre_rescaling"] + self.parameters
        else:
            self.func1 = lambda x, *args: self.dist(x,*args)
        self.parameters = self.parameters + list(self.dist.__code__.co_varnames[1:])
        if self.apply_post_shift_and_rescale:
            self.func = lambda x, post_shift, post_rescaling, *args: rescale_func(shift_func(self.func1(x,*args),post_shift),post_rescaling)
            self.parameters = ["post_shift","post_rescaling"] + self.parameters
        else:
            self.func = self.func1

        # if self.apply_shift_and_rescale:
        #     self.func = lambda x, pre_shift, pre_rescaling, post_shift, post_rescaling, *args:\
        #         rescale_func(shift_func(self.dist(rescale_func(shift_func(x, pre_shift), pre_rescaling),*args),post_shift),post_rescaling)
        # else:
        #     self.func = self.dist
        self.fit_done = False

    def fit(self, xdata, ydata, it=5000, p0=None):
        '''
        Fitting the distribution and computing the optimal parameters (popts)
        :param xdata: xdata
        :param ydata: ydata
        :param it: maximum number of calls to the main function [int]
        :param p0: initial parameters [list]
        :return: wasserstein distance between the data and the optimal distribution [float]
        '''
        # Initializing p0 (initial parameter guesses)
        if p0 is None:
            p0 = []
            if self.apply_pre_shift_and_rescale: p0 += [0, 1]
            if self.apply_post_shift_and_rescale: p0 += [0, 1]
            p0 += self.dist_nargs * [1]
        # fitting the data and computing optimal parameters (popt)
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        self.fit_it = it
        self.fit_p0 = p0
        self.popt, self.pcov, _, _, _ = optimize.curve_fit(self.func, xdata, ydata, p0=p0, maxfev=it, full_output=True)
        self.fit_done = True
        return self.compute_wasserstein_distance(xdata, ydata)

    def compute_wasserstein_distance(self, xdata, ydata):
        '''
        Computing the wasserstein distance between ydata and self.func(xdata,*popt)
        :param xdata: xdata
        :param ydata: ydata
        :return: wasserstein distance [float]
        '''
        if self.fit_done:
            ydata_predicted = self.func(xdata, *self.popt)
            self.wasserstein_distance = wasserstein_distance(ydata, ydata_predicted)
        else:
            warnings.warn("wasserstein_distance can't be computed if the fitting has not been done")
            self.wasserstein_distance = float('nan')
        return self.wasserstein_distance

    def __str__(self):
        '''
        String representation of the distribution with the optimal parameters
        :return: string representation [str]
        '''
        # if self.apply_shift_and_rescale:
        #     parameters = ["pre_shift", "pre_rescaling", "post_shift", "post_rescaling"] + list(self.dist.__code__.co_varnames[1:])
        # else:
        #     parameters = list(self.dist.__code__.co_varnames[1:])
        if self.fit_done:
            parameters_formatted = ', '.join([f"{self.parameters[i]}={round(self.popt[i],2)}" for i in range(len(self.parameters))])
        else:
            parameters_formatted = ', '.join([f"{self.parameters[i]}=?" for i in range(len(self.parameters))])
        func_name = self.dist.__name__
        return func_name + "(" + parameters_formatted + ")"


def plot_primer_random_scores(scores):
    # scores = scores / np.mean(scores)
    values, bin_edges = np.histogram(scores, bins=101)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    values, bin_centers = values[values > 0], bin_centers[values > 0]
    # plt.hist(scores, bins=101)
    plt.plot(bin_centers, values, '-')



def exponential_decayb(x, a,b,c,d,e):
    '''
    Exponential decay function
    :param x: [ndarray or float] data
    :param a: [float] parameter to optimize
    :return: [ndarray] exponential decay
    '''
    return (np.exp(-a*(x-b)/c)-d)/e


def normal(x, mu, sigma, a):
    '''
    Normal distribution
    :param x: [ndarray or float] data
    :param mu: [float] mean to optimize
    :param sigma: [float] standard deviation to optimize
    :return: [ndarray] normal distribution
    '''
    return a * np.exp(-0.5*np.square((x-mu)/sigma)) / (sigma*np.sqrt(2*np.pi))

def normal2(x, a):
    '''
    Normal distribution
    :param x: [ndarray or float] data
    :param mu: [float] mean to optimize
    :param sigma: [float] standard deviation to optimize
    :return: [ndarray] normal distribution
    '''
    return a*np.exp(-0.5*np.square(x)) / (np.sqrt(2*np.pi))

if __name__ == '__main__':
    np.random.seed(0)

    # Dist = Distribution(power_law, pre_shift_and_rescale=False, post_shift_and_rescale=True)
    # print(Dist)
    # score = Dist.fit([0,1,2,3,4,5,6,7,8,9,10,11,12], [10,6,3,2,1,0.5,0.2,0.15,0.1,0.08,0.06,0.05,0.04])
    # print(score)
    # print(Dist)

    x = 9 * np.sort(np.random.rand(500)) - 3
    y = normal(x, mu=1.6, sigma=2, a=1)
    Dist = Distribution(normal2, pre_shift_and_rescale=True, post_shift_and_rescale=False)
    print(Dist)
    score = Dist.fit(x, y)
    print(score)
    print(Dist)

    plt.plot(x, y)
    plt.show()