import  warnings
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from scipy import optimize


"""##############################
          SHIFT/RESCALE
##############################"""


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
    return x/(a+1e-8) # add 1e-8 to avoid "RuntimeWarning: divide by zero"


"""##############################
     DISTRIBUTION FUNCTIONS      
##############################"""


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
def normal(x, mu, sigma, A):
    '''
    Normal distribution
    :param x: [ndarray or float] data
    :param mu: [float] mean to optimize
    :param sigma: [float] standard deviation to optimize
    :param A: [float] amplitude to optimize
    :return: [ndarray] normal distribution
    '''
    return A * np.exp(-0.5*np.square((x-mu)/sigma)) / (sigma*np.sqrt(2*np.pi))


"""##############################
   DISTRIBUTION FITTING CLASS      
##############################"""


class Distribution():
    '''
    Class used to fit a statistical distribution and evaluate the fitting
    :arg self.dist: distribution function [function]
    :arg self.dist_nargs: number of parameters of the distribution function [int]
    :arg self.apply_pre_shift_and_rescale: boolean specifying if shifting/rescaling should be tested before dist function [bool]
    :arg self.apply_post_rescale: boolean specifying if rescaling should be tested after dist function [bool]
    :arg self.func: function to optimize (distribution with shifting/rescaling if applied) [function]
    :arg self.parameters: parameter names of the function to optimize [list of string]
    :arg self.fit_done: boolean specifying if the fitting has already been done [bool]
    :arg self.fit_it: maximum number of iterations for the fitting [int]
    :arg self.fit_p0: initial parameters used for the fitting [list]
    :arg self.popt: optimized parameters after the fitting [tuple]
    :arg self.pcov: covariance matrix of the fitting [numpy.ndarray 2D]
    :arg self.wasserstein_distance: wasserstein distance computed as a score of the fitting [float]
    '''

    def __init__(self, distribution_func, pre_shift_and_rescale=False, post_rescale=False):
        '''
        Constructor of the class Distribution
        :param distribution_func: distribution function [function]
        :param pre_shift_and_rescale: True if shifting/rescaling should be tested before dist function [bool]
        :param post_rescale: True if rescaling should be tested after dist function [bool]
        '''
        self.dist = distribution_func
        self.dist_nargs = self.dist.__code__.co_argcount - 1
        self.apply_pre_shift_and_rescale = pre_shift_and_rescale
        self.apply_post_rescale = post_rescale

        self.parameters = []
        if self.apply_pre_shift_and_rescale:
            self.func1 = lambda x, pre_shift, pre_rescaling, *args: self.dist(rescale_func(shift_func(x, pre_shift), pre_rescaling),*args)
            self.parameters = ["pre_shift","pre_rescaling"] + self.parameters
        else:
            self.func1 = lambda x, *args: self.dist(x,*args)
        self.parameters = self.parameters + list(self.dist.__code__.co_varnames[1:])
        if self.apply_post_rescale:
            self.func = lambda x, post_rescaling, *args: rescale_func(self.func1(x,*args),post_rescaling)
            self.parameters = ["post_rescaling"] + self.parameters
        else:
            self.func = self.func1

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
            if self.apply_pre_shift_and_rescale: p0 = [0, 1] + p0
            if self.apply_post_rescale: p0 = [1] + p0
            p0 += self.dist_nargs * [1]
        if len(p0) == self.dist_nargs:
            if self.apply_pre_shift_and_rescale: p0 = [0, 1] + p0
            if self.apply_post_rescale: p0 = [1] + p0
        # Fitting the data and computing optimal parameters (popt)
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
            ydata_predicted = self.apply(xdata)
            self.wasserstein_distance = wasserstein_distance(ydata, ydata_predicted)
        else:
            warnings.warn("Distribution.wasserstein_distance can't be computed if the fitting has not been done")
            self.wasserstein_distance = float('nan')
        return self.wasserstein_distance

    def apply(self, xdata):
        '''
        Apply the fitted function to xdata -> compute self.func(xdata,*popt)
        :param xdata: xdata
        :return: ydata_predicted using self.func with self.popt
        '''
        if self.fit_done:
            ydata_predicted = self.func(xdata, *self.popt)
        else:
            warnings.warn("Distribution.apply can't be computed if the fitting has not been done")
            ydata_predicted = None
        return ydata_predicted

    def __str__(self):
        '''
        String representation of the distribution with the optimal parameters
        :return: string representation [str]
        '''
        if self.fit_done:
            parameters_formatted = ', '.join([f"{self.parameters[i]}={round(self.popt[i],2)}" for i in range(len(self.parameters))])
        else:
            parameters_formatted = ', '.join([f"{self.parameters[i]}=?" for i in range(len(self.parameters))])
        func_name = self.dist.__name__
        return func_name + "(" + parameters_formatted + ")"


"""##############################
        TEST FUNCTIONS          
##############################"""


def test_normal_fitting():
    np.random.seed(0)
    def normal_test(x, F):
        F = 1  # fixed
        return F * np.exp(-0.5 * np.square(x)) / (np.sqrt(2 * np.pi))

    # test data
    x = 9 * np.sort(np.random.rand(500)) - 3
    y = 3 * normal(x, mu=1.6, sigma=2, A=1)

    # To test: play with the parameters: 'pre_shift_and_rescale' & 'post_rescale'
    Dist = Distribution(normal_test, pre_shift_and_rescale=True, post_rescale=False)
    print(Dist) # Distribution unfitted
    score = Dist.fit(x, y)
    print(score) # wasserstein score
    print(Dist) # Distribution fitted

    # Plotting (blue = target | green = fitted)
    plt.plot(x, y, "b")
    plt.plot(x, Dist.apply(x), "g")
    plt.show()


if __name__ == '__main__':
    test_normal_fitting()

