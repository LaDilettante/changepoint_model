import init_functions as init
import mcem as mcem
import mcem_opt as mcem_opt
import numpy as np
import scipy.stats as st

def mcem_sampler(Yn, model, m, mcem_module, tol=None):
    '''
    Calculate the MLE of Theta and P, using Monte Carlo EM.
    
    We use EM because the latent state vector is not observable.
    We use Monte Carlo method because the Q function in the E-step is not available analytically.

    Args
        Yn: array, time series data
        model: string, e.g. "binary", "poisson"
        m: int, number of change points
        mcem: module, e.g. mcem, mcem_opt
            Use the regular of optimized version
    '''
    # Initialize
    n = len(Yn) ; m = m
    P = init.P(m + 1)

    P[0, 0] = 0.8 # Paper
    P[0, 1] = 1 - 0.8

    if (m == 2) and (model == "poisson"):
        # Without giving a high chance of staying at state 2, the MCEM gets stuck
        P[1, 1] = 0.9
        P[1, 2] = 1 - 0.9

    # Initialize Theta
    if model == "binary":
        Theta = np.repeat(0.5, m + 1)
    elif model == "poisson":
        #Theta = np.repeat(2, m + 1)
        Theta = 2 + np.random.rand(m + 1)

    # N is the number of Sn sample
    # According to paper, start N = 1 and increases over the MCEM iterations
    Ns = np.linspace(1, 300, 10).astype(np.int64)
    Thetas = np.empty((m + 1, 100))
    Ps = np.empty((m + 1, m + 1, 100))

    # Start 100 MCEM steps
    i = 0
    while i < 100:
        N = Ns[i / 10]

        # E-step
        Sns = mcem_module.S_estep(N, Yn, Theta, P, model=model)
        #if i <= 3:
        #    print Sns

        # M-step
        Theta_old = Theta
        Theta = mcem_module.Theta_mstep(Yn, Sns, model=model)
        P = mcem_module.P_mstep(Sns)

        # Store result across iterations
        Thetas[:, i] = Theta
        Ps[:, :, i] = P

        
        #Stop condition based on convergence
        if (tol is not None) and (np.allclose(Theta, Theta_old, atol=tol)):
            break

        i += 1

    return Thetas, Ps