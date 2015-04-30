import full_conditionals as cond
import numpy as np
import scipy as sp
import scipy.stats as st

def Nk(Sn):
    '''
    Calculate the number of observations ascribed to regime k in Sn, for each k in Sn

    Args
        Sn: n x 1 vector of latent state

    Returns
        Nks: (m + 1) x 1 vector
    '''
    ks = np.unique(Sn)
    return np.array([Sn == k for k in ks]).sum(axis=1)

def Uk(Yn, Sn):
    '''
    Calculate the sum of y_t in regime k, for each k in Sn

    Args
        Sn: n x 1 vector of latent state

    Returns
        Uks: (m + 1) x 1 vector
    '''
    ks = np.unique(Sn)
    return np.array([ (Yn[Sn == k]).sum() for k in ks ])

def Theta_sampling(Yn, Sn, model):
    '''
    Sample Theta from its full conditional distribution. Vectorized.

    Args:
        Yn: n x 1 data vector
        Sn: n x 1 latent state vector
    
    Returns:
        a (m + 1) x 1 vector of parameter, one param for each regime
    '''
    number_of_regimes = len(np.unique(Sn))
    m = number_of_regimes - 1

    Nks = Nk(Sn)
    Uks = Uk(Yn, Sn)

    if model == "binary":
        return st.beta(2 + Uks, 2 + Nks - Uks).rvs()
    elif model == "poisson":
        # Different priors based on number of breaks
        if m == 1:
            a = 2 ; b = 1
        elif m == 2:
            a = 3 ; b = 1
        return st.gamma(a + Uks, scale=(1.0 / (b + Nks))).rvs()

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

    i_s = np.arange(1, number_of_regimes) # Skip the last regime
    n_iis = np.array([np.sum(Sn == i) - 1 for i in i_s])
    p_iis = np.append(st.beta(a + n_iis, b + 1).rvs(), [1])
    p_ijs = 1 - p_iis[:-1]

    return np.diag(p_iis) + np.diag(p_ijs, k=1)

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

    # f(y_t | Y_{t-1}, \theta_k) for all k's and t's
    D_t = np.array([[fy(t, k_, Yn, Theta, model) for k_ in range(1, m + 2)] 
                                                 for t in range(2, n + 1)])

    for t in range(2, n + 1): # Forward
        F1[t - 1] = ( F0[t - 2].dot(P) )
        F0[t - 1] = F1[t - 1] * D_t[t - 2]

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