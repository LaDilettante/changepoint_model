import numpy as np
import scipy as sp
import scipy.stats as st

def Theta_conditional(k, Yn, Sn):
    '''
    The full conditional of theta, given Yn, Sn, and P.

    Args
        k: indicates which regime's parameter we're sampling
        Yn: n x 1 data vector
        Sn: n x 1 latent state vector

    Returns
        the distribution that is the full conditional of theta.
        We can sample from this distribution using `.rvs()`
    '''
    Nk = np.sum(Sn == k)
    Uk = np.sum(Yn[Sn == k])

    return st.beta(2 + Uk, 2 + Nk - Uk)

def Theta_sampling(Yn, Sn):
    '''
    Sample Theta from its full conditional distribution

    Args:
        Yn: n x 1 data vector
        Sn: n x 1 latent state vector
    
    Returns:
        a (m + 1) x 1 vector of parameter, one param for each regime
    '''
    number_of_regimes = len(np.unique(Sn))
    thetas = np.empty(number_of_regimes)
    for k in range(1, number_of_regimes + 1):
        thetas[k - 1] = Theta_conditional(k, Yn, Sn).rvs()
    return thetas        

def P_conditional(i, Sn, a, b):
    '''
    The full conditional of p_ii, given Sn

    Args
        i: the index of state in p_ii
        Sn: n x 1 latent state vector
        a: Beta prior parameter
        b: Beta prior parameter

    Returns
        the distribution that is the full conditional of p_ii.
        We can sample from this distribution using `.rvs()`
    '''
    n_ii = np.sum(Sn == i) - 1

    return st.beta(a + n_ii, b + 1)

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
    P = np.empty((number_of_regimes, number_of_regimes))
    P[-1, -1] = 1
    
    for i in range(1, number_of_regimes):
        idx = i - 1 # i is index of state, starts at 1. idx starts at 0
        P[idx, idx] = P_conditional(i, Sn, a, b).rvs()
        P[idx, idx + 1] = 1 - P[idx, idx]

    return P

def S_conditional(Yn, Theta, P):
    pass