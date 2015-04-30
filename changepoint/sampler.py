import init_functions as init
import full_conditionals as cond
import full_conditionals_opt as cond_opt
import mcem_sampler as mcem
import ordinate as ordinate
import numpy as np
import scipy.stats as st

def sampler(Yn, model, m, cond_module, max_iter=6000, burn_iter=1000):
    '''
    Detect the location of change points

    Args
        Yn: array, time-series data
        model: string, e.g. "binary", "poisson"
        m: int, number of change points
        cond: module, e.g. cond, cond_opt
            Whether to use the regular or optimized version
        max_iter: int, number of iterations
        burn_iter: int, number of burn-in iterations
    '''
    # Initialize Sn and P
    n = len(Yn) ; m = m
    Sn = init.S(n, m)
    P = init.P(m + 1)

    # Initializing Theta
    if model == "binary":
        Theta = np.repeat(0.5, m + 1)
    elif model == "poisson":
        Theta = np.repeat(2, m + 1) # From paper
    
    # Prior on P
    if m == 1:
        a = 8 ; b = 0.1
    elif m == 2:
        a = 5 ; b = 0.1

    F1_mcmc = np.empty((n, m + 1, max_iter))
    F_mcmc = np.empty((n, m + 1, max_iter))
    Theta_mcmc = np.empty((m + 1, max_iter))

    i = 0
    while (i < max_iter):

        Sn, F, F1 = cond_module.S_sampling(Yn, Theta, P, model=model)
        Theta = cond_module.Theta_sampling(Yn, Sn, model=model)
        P = cond_module.P_sampling(Sn, a, b)

        F1_mcmc[:, :, i] = F1
        F_mcmc[:, :, i] = F
        Theta_mcmc[:, i] = Theta
        i += 1

    return Sn, F1_mcmc, F_mcmc, Theta_mcmc, P

def sampler_with_mcem(Yn, model, m, cond_module, mcem_module, ordinate_module, max_iter=6000, burn_iter=1000):
    '''
    Detect the location of change points and calculate marginal likelihood
    
    Using marginal likelihood, we can compare different models
    with different number of change points

    Args
        Yn: array, time-series data
        model: string, e.g. "binary", "poisson"
        m: int, number of change points
        cond: string, e.g. "cond", "cond_opt"
            Whether to use the regular or optimized version of sub-functions
        max_iter: int, number of iterations
        burn_iter: int, number of burn-in iterations

    Returns
        marg_lik: marginal likelihood of the input model
    '''
    
    # MCEM to get the MLE estimate
    Thetas, Theta, P = mcem_module.mcem_sampler(Yn, model, m)
    Theta_mle = Theta ; P_mle = P
    # print Theta_mle, P_mle

    # Initialize
    n = len(Yn) ; m = m
    Sn = init.S(n, m)
    P = init.P(m + 1)

    # Initializing Theta
    if model == "binary":
        Theta = np.repeat(0.5, m + 1)
    elif model == "poisson":
        Theta = np.repeat(2, m + 1) # From paper
    
    # Prior on P
    if m == 1:
        a = 8 ; b = 0.1
    elif m == 2:
        a = 5 ; b = 0.1

    # Pre-populate result arrays
    F1_mcmc = np.empty((n, m + 1, max_iter))
    F_mcmc = np.empty((n, m + 1, max_iter))
    Theta_mcmc = np.empty((m + 1, max_iter))
    Sn_mcmc = np.empty((n, max_iter))
    # Extra result arrays for marginal likelihood calculation
    Sn_extra_mcmc = np.empty((n, max_iter))
    pii_extra_mcmc = np.empty((m + 1, max_iter))

    # Start MCMC
    i = 0
    while (i < max_iter):
        # print "Iteration" + str(i)
        # print P, Theta

        Sn, F, F1 = cond_module.S_sampling(Yn, Theta, P, model=model)
        Theta = cond_module.Theta_sampling(Yn, Sn, model=model)
        P = cond_module.P_sampling(Sn, a, b)

        # Additional sampling of Sn for marginal likelihood
        Sn_extra, F, F1 = cond.S_sampling(Yn, Theta_mle, P, model=model)
        P_extra = cond.P_sampling(Sn_extra, a, b)

        F1_mcmc[:, :, i] = F1
        F_mcmc[:, :, i] = F
        Theta_mcmc[:, i] = Theta
        Sn_mcmc[:, i] = Sn
        Sn_extra_mcmc[:, i] = Sn_extra
        pii_extra_mcmc[:, i] = np.diag(P_extra)
        i += 1

    # Calculate marginal likelihood
    log_likelihood = ordinate_module.log_likelihood(Yn, Theta_mle, P_mle, model=model)
    log_prior_Theta = ordinate_module.log_prior_Theta(Theta_mle, model=model)
    log_prior_P = ordinate_module.log_prior_P(P_mle, a, b)
    log_posterior_Theta = ordinate_module.log_posterior_Theta(Theta_mle, Yn, Sn_mcmc, model=model)
    log_posterior_P = ordinate_module.log_posterior_P(P_mle, Sn_extra_mcmc, a, b)

    print log_likelihood, log_prior_Theta, log_prior_P, log_posterior_Theta, log_posterior_P
    marg_lik = log_likelihood + log_prior_Theta + log_prior_P - log_posterior_Theta - log_posterior_P

    return Sn, F1_mcmc, F_mcmc, Theta_mcmc, Theta_mle, P_mle, marg_lik