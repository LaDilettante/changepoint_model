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
    n = 150
    m = 2

    Sn = init.S(n, m)
    P = init.P(m + 1)
    Theta = np.array([0.5, 0.5, 0.5])
    delta = 1


    # Prior
    a = 8
    b = 0.1

    tol = 1e-6
    max_iter = 6000
    burn_iter = 1000
    i = 0

    F_mcmc = np.empty((n, m + 1, max_iter))
    Theta_mcmc = np.empty((m + 1, max_iter))

    while (i < max_iter):

        Sn, F = cond.S_sampling(Yn, Theta, P)
        Theta = cond.Theta_sampling(Yn, Sn)
        P = cond.P_sampling(Sn, a, b)

        F_mcmc[:, :, i] = F
        Theta_mcmc[:, i] = Theta
        i += 1
    return Sn, F_mcmc, Theta_mcmc, P 

if __name__ == "__main__":
    binary_sampler()