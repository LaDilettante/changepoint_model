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

def mcem_poisson_sampler(Yn, m, tol=1e-4):
    # Initialize
    n = len(Yn) ; m = m
    Sn = init.S(n, m)
    P = init.P(m + 1)

    # Initialized value according to paper
    P[0, 0] = 0.9
    Theta = np.repeat(2, m + 1)

    # N is the number of Sn sample
    # According to paper, start N = 1 and increases over the MCEM iterations
    Ns = np.linspace(1, 300, 10).astype(np.int64)
    
    # Pre-populate result array
    Thetas = np.zeros((m + 1, 100))

    # Start 100 MCEM steps
    i = 0
    while i < 100:
        N = Ns[i / 10]

        # E-step
        Sns = mcem.S_estep(N, Yn, Theta, P, model="poisson")

        # M-step
        Theta_old = Theta
        Theta = mcem.Theta_mstep(Yn, Sns, model="poisson")
        P = mcem.P_mstep(Sns)

        # Store Thetas across iterations
        Thetas[:, i] = Theta
        
        # Check convergence
        #if np.allclose(Theta, Theta_old, atol=tol):
        #    break

        i += 1

    return Thetas, Theta, P
