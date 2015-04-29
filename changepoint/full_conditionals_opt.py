import full_conditionals as cond
import numpy as np
import scipy as sp
import scipy.stats as st

Theta_conditional = cond.Theta_conditional # Not vectorizable

def Theta_sampling(Yn, Sn, model):
    '''
    Sample Theta from its full conditional distribution

    Args:
        Yn: n x 1 data vector
        Sn: n x 1 latent state vector
    
    Returns:
        a (m + 1) x 1 vector of parameter, one param for each regime
    '''
    number_of_regimes = len(np.unique(Sn))

    k = np.arange(1, number_of_regimes + 1)
    f = lambda k: Theta_conditional(k, Yn, Sn, model).rvs() # Lambda function for a single k
    f_array = np.frompyfunc(f, 1, 1) # Vectorize the function for an array of k's
    Theta = f_array(k)
    
    return Theta        

P_conditional = cond.P_conditional # Not vectorizable

def P_sampling(Sn, a, b):
    '''
    Sample P from its conditional

    Args
        Sn: n x 1 latent state vector
        a: Beta prior parameter
        b: Beta prior parameter

    Returns
        the (m + 1) x (m + 1) Markov transition matrix.
        Recall that (m + 1) is the number of regimes
    '''
    number_of_regimes = len(np.unique(Sn))

    i = np.arange(1, number_of_regimes)
    f = lambda i: P_conditional(i, Sn, a, b).rvs()
    f_array = np.frompyfunc(f, 1, 1)

    p_iis = np.append(f_array(i), [1])
    p_ijs = 1 - p_iis[:-1]
    P = np.diag(p_iis) + np.diag(p_ijs, k=1)

    return P

fy = cond.fy
S_conditional = cond.S_conditional
S_sampling = cond.S_sampling