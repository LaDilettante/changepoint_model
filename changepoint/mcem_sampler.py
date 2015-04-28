import init_functions as init
import mcem as mcem
import numpy as np
import scipy.stats as st

def binary_true_data():
    thetas = np.array([0.5, 0.75, 0.25])
    ns = np.array([50, 50, 50])

    Yn = np.array([st.bernoulli.rvs(theta_, size=n_) for theta_, n_ in zip(thetas, ns)]) \
        .ravel()

    return Yn

def mcem_binary_sampler(Yn):
    # Initialize
    n = 150 ; m = 2

    P = init.P(m + 1)
    Theta = np.array([0.5, 0.5, 0.5])
    
    # 100 MCEM steps
    Ns = np.linspace(1, 300, 10).astype(np.int64) # increasing N
    Thetas = np.zeros((m + 1, 100))
    Ps = np.zeros((m + 1, m + 1, 100))

    for i in range(100):
        N = Ns[i / 10]
        # E-step
        Sns = mcem.S_estep(N, Yn, Theta, P, model="binary")
        # M-step
        Theta = mcem.Theta_mstep(Yn, Sns, model="binary")
        P = mcem.P_mstep(Sns)
        # Store values to see if they converge
        Thetas[:, i] = Theta
        Ps[:, :, i] = P

    return Thetas, Ps

def mcem_poisson_sampler(Yn):
    # Initialize
    n = 112 ; m = 1

    Sn = init.S(n, m)
    P = init.P(m + 1)
    P[0, 0] = 0.9 # Paper's initialization
    Theta = np.array([2, 2]) # Paper's initialization

    # 100 MCEM steps
    Ns = np.linspace(1, 300, 10).astype(np.int64) # increasing N
    Thetas = np.zeros((m + 1, 100))
    Ps = np.zeros((m + 1, m + 1, 100))

    for i in range(100):
        N = Ns[i / 10]
        # E-step
        Sns = mcem.S_estep(N, Yn, Theta, P, model="poisson")
        # M-step
        Theta = mcem.Theta_mstep(Yn, Sns, model="poisson")
        P = mcem.P_mstep(Sns)
        # Store values to see if they converge
        Thetas[:, i] = Theta
        Ps[:, :, i] = P

    return Thetas, Ps    
