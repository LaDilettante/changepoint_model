import numpy as np
import scipy as sp
import scipy.stats as st

def Theta_conditional(k, Yn, Sn, P):
    '''
    The full conditional of theta, given Yn, Sn, and P.

    Args
        k: indicates which regime's parameter we're sampling
        Yn: n x 1 data vector
        Sn: n x 1 latent state vector
        P: m x m Markov transition matrix (m = number of change points)

    Returns
        the distribution that is the full conditional of theta.
        We can sample from this distribution using `.rvs()`
    '''
    Nk = np.sum(Sn == k)
    Uk = np.sum(Yn[Sn == k])

    return st.beta(2 + Uk, 2 + Nk - Uk)

def P_conditional(i, Sn, a, b):
    '''
    The full conditional of p_ii, given Sn

    Args
        Sn: n x 1 latent state vector
        a: Beta prior parameter
        b: Beta prior parameter

    Returns
        the distribution that is the full conditional of p_ii.
        We can sample from this distribution using `.rvs()`
    '''
    n_ii = np.sum(Sn == i) - 1

    return st.beta(a + n_ii, b + 1)

def S_conditional(Yn, Theta, P):
    pass