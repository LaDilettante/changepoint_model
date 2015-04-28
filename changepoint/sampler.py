import init_functions as init
import full_conditionals as cond
import full_conditionals_opt as cond_opt
import numpy as np
import scipy.stats as st

def binary_true_data():
    thetas = np.array([0.5, 0.75, 0.25])
    ns = np.array([50, 50, 50])

    Yn = np.array([st.bernoulli.rvs(theta_, size=n_) for theta_, n_ in zip(thetas, ns)]) \
        .ravel()

    return Yn

def binary_sampler(Yn, max_iter=6000, burn_iter=1000, cond=cond):
    # Initialize
    n = 150 ; m = 2
    Sn = init.S(n, m)
    P = init.P(m + 1)
    Theta = np.array([0.5, 0.5, 0.5])
    # Prior
    a = 8 ; b = 0.1

    F1_mcmc = np.empty((n, m + 1, max_iter))
    F_mcmc = np.empty((n, m + 1, max_iter))
    Theta_mcmc = np.empty((m + 1, max_iter))

    i = 0
    while (i < max_iter):

        Sn, F, F1 = cond.S_sampling(Yn, Theta, P, model="binary")
        Theta = cond.Theta_sampling(Yn, Sn, model="binary")
        P = cond.P_sampling(Sn, a, b)

        F1_mcmc[:, :, i] = F1
        F_mcmc[:, :, i] = F
        Theta_mcmc[:, i] = Theta
        i += 1

    return Sn, F1_mcmc, F_mcmc, Theta_mcmc, P

def poisson_sampler(Yn, max_iter=6000, burn_iter=1000):
    # Initialize
    n = 112
    m = 1

    Sn = init.S(n, m)
    P = init.P(m + 1)
    P[0, 0] = 0.9 # Paper's initialization
    Theta = np.array([2, 2]) # Paper's initialization
    delta = 1


    # Prior # according to paper
    a = 8
    b = 0.1

    tol = 1e-6
    max_iter = max_iter
    burn_iter = burn_iter
    i = 0

    F1_mcmc = np.empty((n, m + 1, max_iter))
    F_mcmc = np.empty((n, m + 1, max_iter))
    Theta_mcmc = np.empty((m + 1, max_iter))

    while (i < max_iter):

        Sn, F, F1 = cond.S_sampling(Yn, Theta, P, model="poisson")
        Theta = cond.Theta_sampling(Yn, Sn, model="poisson")
        P = cond.P_sampling(Sn, a, b)

        F1_mcmc[:, :, i] = F1
        F_mcmc[:, :, i] = F
        Theta_mcmc[:, i] = Theta
        i += 1

    return Sn, F1_mcmc, F_mcmc, Theta_mcmc, P    

# if __name__ == "__main__":
#     binary_sampler()