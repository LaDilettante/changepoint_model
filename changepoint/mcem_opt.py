import full_conditionals_opt as cond_opt
import mcem as mcem
import numpy as np
import scipy.stats as st

def S_estep(N, Yn, Theta, P, model, cond=cond_opt):
    Sn, F, F1 = cond.S_sampling(Yn, Theta, P, model)
    Sns = np.zeros((Sn.shape[0], N))

    for N_ in range(N):
        Sn, F, F1 = cond.S_sampling(Yn, Theta, P, model)
        Sns[:, N_] = Sn

    return Sns

def theta_mstep(k, Yn, Sns, model):
    '''
    Calculate theta_k, using N samples of Sn. Vectorized.
    '''
    Uks = np.sum(Yn[:, np.newaxis] * (Sns == k), axis=0)
    Nks = np.sum(Sns == k, axis=0)

    if model == "binary":
        return 1.0 * Uks.sum() / Nks.sum()
    if model == "poisson":
        return 1.0 * Uks.sum() / Nks.sum()

def Theta_mstep(Yn, Sns, model):
    '''
    Update Theta in the M-step, using N samples of Sn

    Args
        Sns: n x N matrix, with each column being one draw of Sn
        model: a string, either "binary" or "poisson"

    Returns:
        a (m + 1) x 1 vector of parameter, one param for each regime
    '''
    number_of_regimes = len(np.unique(Sns[:, 0]))
    Theta = np.zeros(number_of_regimes)

    for k in range(1, number_of_regimes + 1):
        Theta[k - 1] = theta_mstep(k, Yn, Sns, model)

    return Theta

def p_mstep(i, Sns):
    '''
    Calculate p_ii = \frac{\sum_j n_ii,j}{\sum_j (n_ii,j + 1)} from Eq 13. Vectorized
    '''
    n_iis = np.sum(Sns == i, axis=0) - 1

    return 1.0 * n_iis.sum() / (n_iis.sum() + len(n_iis))

def P_mstep(Sns):
    '''
    Update P in the M-step

    Args
        Sn: n x 1 latent state vector, sampled in the E-step

    Returns
        the (m + 1) x (m + 1) Markov transition matrix.
        Recall that (m + 1) is the number of regimes
    '''
    number_of_regimes = len(np.unique(Sns[:, 0]))
    P = np.empty((number_of_regimes, number_of_regimes))
    P[-1, -1] = 1

    for i in range(1, number_of_regimes):
        idx = i - 1 # i is index of state, starts at 1. idx starts at 0
        P[idx, idx] = p_mstep(i, Sns)
        P[idx, idx + 1] = 1 - P[idx, idx]

    return P