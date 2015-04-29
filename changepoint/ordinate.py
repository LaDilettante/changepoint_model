import numpy as np
import scipy as st
import full_conditionals as cond

def log_likelihood(Yn, Theta, P, model):
    '''
    Calculate ln f(Y_n | Theta, P) = \sum ln f(y_t | Y_{t - 1}, Theta, P)
    Calulation based on Eq (9)

    Returns
        the likelihood
    '''
    F1, F0 = cond.S_conditional(Yn, Theta, P, model)

    log_fys = np.zeros(len(Yn))
    for t in range(1, n + 1):
        d_t = np.array([cond.fy(t, k_, Yn, Theta, model) for k_ in range(1, m + 2)])
        log_fy = np.log((F1 * d_t).sum())
        log_fys[t - 1] = log_fy

    return log_fys.sum()

def log_prior_Theta(Theta, model):
    '''
    Calculate the prior ordinate at the given Theta

    The prior is different for each model.
    For binary, theta_k ~ Beta(2, 2)
    For poisson, lambda_k ~ Gamma(2, 1) or Gamma(3, 1). See page 235
    '''
    if model == "binary":
        log_pdf = lambda theta: st.beta(2, 2).logpdf(theta)
    if model == "poisson":
        if len(Theta) <= 2:
            log_pdf = lambda theta: st.gamma(2, scale=1.0/1).logpdf(theta)
        if len(Theta) == 3:
            log_pdf = lambda theta: st.gamma(3, scale=1.0/1).logpdf(theta)

    return np.apply_along_axis(log_pdf, axis=0, arr=Theta).sum()

def log_prior_P(P, a, b):
    '''
    Calculate the prior ordinate at the given P
    Note: p_ii ~ Beta(a, b) and ln pi(P) = \sum ln pi(p_ii) 
    '''
    p_iis = np.diag(P)
    log_pdf = lambda p_ii: st.beta(a, b).logpdf(p_ii)

    return np.apply_along_axis(log_pdf, axis=0, arr=p_iis).sum()

def log_posterior_Theta_single(Theta, Yn, Sn, model):
    '''
    Calculate ln pi(Theta* | Yn, Sn) = \sum ln pi(theta_k* | Yn, Sn)
    '''
    k = np.arange(1, len(Theta) + 1)
    f = lambda k: cond.Theta_conditional(k, Yn, Sn, model).logpdf(Theta[k - 1])
    
    return np.apply_along_axis(f, axis=0, arr=k).sum()

def log_posterior_Theta(Theta, Yn, Sns, model):
    '''
    Calculate ln pi(Theta* | Yn) = ln \mean pi(Theta* | Yn, Sn)
    '''
    f = lambda Sn: log_posterior_Theta_single(Theta, Yn, Sn, model)
    return np.log(np.exp(np.apply_along_axis(f, axis=0, arr=Sns)).mean())

def log_posterior_P(P, Theta, a, b):
    pass

def log_marginal_likelihood(Yn, Theta, P, a, b, model):
    '''
    Calculate the log marginal likelihood ln m(Y_n)

    ln m(Y_n) = ln f(Yn | Theta*, P*) + ln pi(Theta*) + ln pi(P*) -
        ln pi(Theta* | Yn) - ln pi(P* | Yn, Theta*)
    '''
    return log_likelihood(Yn, Theta, P, model) + \
        log_prior_Theta(Theta, model) + \
        log_prior_P(P, a, b) - \
        log_posterior_Theta(Yn, Theta, model) - \
        log_posterior_P(Theta, P, a, b)
