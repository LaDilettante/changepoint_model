import init_functions as init
import full_conditionals as cond
import full_conditionals_opt as cond_opt
import mcem_sampler as mcem
import ordinate as ordinate
import numpy as np
import scipy.stats as st

def binary_true_data():
    thetas = np.array([0.5, 0.75, 0.25])
    ns = np.array([50, 50, 50])

    Yn = np.array([st.bernoulli.rvs(theta_, size=n_) for theta_, n_ in zip(thetas, ns)]) \
        .ravel()

    return Yn

def binary_sampler(Yn, cond, max_iter=6000, burn_iter=1000):
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

def poisson_sampler(Yn, cond, max_iter=6000, burn_iter=1000):
    # Initialize
    n = 112 ; m = 1

    Sn = init.S(n, m)
    P = init.P(m + 1)
    P[0, 0] = 0.9 # Paper's initialization
    Theta = np.array([2, 2]) # Paper's initialization

    # Prior # according to paper
    a = 8 ; b = 0.1

    F1_mcmc = np.empty((n, m + 1, max_iter))
    F_mcmc = np.empty((n, m + 1, max_iter))
    Theta_mcmc = np.empty((m + 1, max_iter))

    i = 0
    while (i < max_iter):
        # print "Iteration" + str(i)

        Sn, F, F1 = cond.S_sampling(Yn, Theta, P, model="poisson")
        Theta = cond.Theta_sampling(Yn, Sn, model="poisson")
        P = cond.P_sampling(Sn, a, b)

        F1_mcmc[:, :, i] = F1
        F_mcmc[:, :, i] = F
        Theta_mcmc[:, i] = Theta
        i += 1

    return Sn, F1_mcmc, F_mcmc, Theta_mcmc, P

def poisson_sampler_with_mcem(Yn, cond, max_iter=6000, burn_iter=1000):
    Thetas, Theta, P = mcem.mcem_poisson_sampler(Yn, tol=1e-4)
    Theta_mle = Theta ; P_mle = P

    # Initialize
    n = 112 ; m = 1

    Sn = init.S(n, m)
    P = init.P(m + 1)
    P[0, 0] = 0.9 # Paper's initialization
    Theta = np.array([2, 2]) # Paper's initialization

    # Prior # according to paper
    a = 8 ; b = 0.1

    F1_mcmc = np.empty((n, m + 1, max_iter))
    F_mcmc = np.empty((n, m + 1, max_iter))
    Theta_mcmc = np.empty((m + 1, max_iter))
    Sn_mcmc = np.empty((n, max_iter))

    Sn_extra_mcmc = np.empty((n, max_iter))
    pii_extra_mcmc = np.empty((m + 1, max_iter))

    i = 0
    while (i < max_iter):

        Sn, F, F1 = cond.S_sampling(Yn, Theta, P, model="poisson")
        Theta = cond.Theta_sampling(Yn, Sn, model="poisson")
        P = cond.P_sampling(Sn, a, b)

        # Additional sampling of Sn for marginal likelihood
        Sn_extra, F, F1 = cond.S_sampling(Yn, Theta_mle, P, model="poisson")
        P_extra = cond.P_sampling(Sn_extra, a, b)

        F1_mcmc[:, :, i] = F1
        F_mcmc[:, :, i] = F
        Theta_mcmc[:, i] = Theta
        Sn_mcmc[:, i] = Sn
        Sn_extra_mcmc[:, i] = Sn_extra
        pii_extra_mcmc[:, i] = np.diag(P_extra)
        i += 1

    marg_lik = ordinate.log_likelihood(Yn, Theta_mle, P_mle, model="poisson") + \
        ordinate.log_prior_Theta(Theta_mle, model="poisson") + \
        ordinate.log_prior_P(P_mle, a, b) + \
        ordinate.log_posterior_Theta(Theta_mle, Yn, Sn_mcmc, model="poisson") + \
        ordinate.log_posterior_P(P_mle, Sn_extra_mcmc, a, b)

    return Sn, F1_mcmc, F_mcmc, Theta_mcmc, P, marg_lik