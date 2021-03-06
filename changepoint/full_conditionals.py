import numpy as np
import scipy as sp
import scipy.stats as st

def Theta_conditional(k, Yn, Sn, model):
    '''
    The full conditional of theta, given Yn, Sn.

    Args
        k: indicates which regime's parameter we're sampling
        Yn: n x 1 data vector
        Sn: n x 1 latent state vector

    Returns
        the distribution that is the full conditional of theta.
        We can sample from this distribution using `.rvs()`
    '''
    m = len(np.unique(Sn)) - 1 # number of breaks
    
    Nk = np.sum(Sn == k)
    Uk = np.sum(Yn[Sn == k])

    if model == "binary":
        return st.beta(2 + Uk, 2 + Nk - Uk)
    elif model == "poisson":
        # Different priors based on number of breaks
        if m == 1:
            a = 2 ; b = 1
        elif m == 2:
            a = 3 ; b = 1
        return st.gamma(a + Uk, scale=(1.0 / (b + Nk)))

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
    thetas = np.empty(number_of_regimes)
    
    for k in range(1, number_of_regimes + 1):
        thetas[k - 1] = Theta_conditional(k, Yn, Sn, model=model).rvs()
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

def fy(t, k, Yn, Theta, model):
    '''
    f(y_t | Y_{t - 1})
    '''
    if model == "binary":
        return st.bernoulli.pmf(Yn[t-1], Theta[k-1])
    elif model == "poisson":
        return (Theta[k-1] ** Yn[t-1]) * np.exp(-Theta[k-1]) / np.math.factorial(Yn[t-1])

def S_conditional(Yn, Theta, P, model):
    '''
    Create a grid of pmf for Prob(s_t = k | Yt, Theta, P)

    Args
        Yn: n x 1 data vector
        Theta: (m + 1) x 1 parameter vector. (m + 1) is no of regimes
        P: (m + 1) x (m + 1) Markov transition matrix

    Returns
        the matrix n x (m + 1) matrix F that stores a grid of pmf
    '''
    n = len(Yn)
    m = len(Theta) - 1

    F1 = np.zeros((n, m + 1)) # lag 1 posterior p(s_t = k | Y_{t-1}, Theta, P)
    F0 = np.zeros((n, m + 1)) # lag 0 posterior p(s_t = k | Y_{t}, Theta, P)
    F1[0, 0] = 1
    F0[0, 0] = 1  

    for t in range(2, n + 1): # Forward
        d_t = np.array([fy(t, k_, Yn, Theta, model) for k_ in range(1, m + 2)])
        F1[t - 1] = ( F0[t - 2].dot(P) )
        F0[t - 1] = F1[t - 1] * d_t

    # Normalize
    F1 = F1 / F1.sum(axis=1)[:, np.newaxis]
    F0 = F0 / F0.sum(axis=1)[:, np.newaxis]
    return F1, F0

def S_sampling(Yn, Theta, P, model):
    '''
    Sample S from the n x (m + 1) grid of pmfs
    '''
    n = len(Yn)
    m = len(Theta) - 1

    F1, F0 = S_conditional(Yn, Theta, P, model=model)
    
    F = np.zeros((n, m + 1))
    F[-1, -1] = 1
    S = np.zeros(n)
    S[-1] = m + 1

    for t in range(n - 1, 0, -1): # Backward
        pmfs = F0[t - 1] * P[:, S[t] - 1].T
        pmfs = (pmfs / pmfs.sum()).astype('float64')
        F[t - 1] = pmfs
        S[t - 1] = np.random.choice(np.arange(1, m + 2), p=pmfs)

    return S, F, F1