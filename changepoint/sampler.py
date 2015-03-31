import init_functions as init
import full_conditionals as cond
import numpy as np
import scipy.stats as st

def binary_true_data():
    thetas = np.array([0.5, 0.75, 0.25])
    ns = np.array([50, 50, 50])

    Yn = np.array([st.bernoulli.rvs(theta_, size=n_) for theta_, n_ in zip(thetas, ns)]) \
        .ravel()

    return Yn

def binary_sampler(Yn):
    # Initialize
    Sn = init.S(150, 2)
    P = init.P(3)
    Theta = np.array([0.5, 0.5, 0.5])
    delta = 1

    # Prior
    a = 8
    b = 0.1

    tol = 1e-6
    max_iter = 5000
    i = 0
    while ( delta > tol ) and (i < max_iter):
        Theta_old = Theta.copy()

        Sn, F = cond.S_sampling(Yn, Theta, P)
        Theta = cond.Theta_sampling(Yn, Sn)
        P = cond.P_sampling(Sn, a, b)

        delta = np.abs((Theta - Theta_old).sum())
        i += 1

        # print i, delta
        if delta < tol:
            print "Convergence reached"

    return Sn, F, Theta, P

if __name__ == "__main__":
    binary_sampler()