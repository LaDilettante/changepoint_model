import full_conditionals as cond
import numpy as np
import scipy as sp
import scipy.stats as st

def log_likelihood(Yn, Theta, P, model):
    '''
    Calculate ln f(Y_n | Theta, P) = \sum ln f(y_t | Y_{t - 1}, Theta, P)
    Calulation based on Eq (9)

    Returns
        the likelihood
    '''
    F1, F0 = cond.S_conditional(Yn, Theta, P, model)
    n = len(Yn)
    m = len(Theta) - 1

    log_fys_lag1 = np.zeros(len(Yn))
    for t in range(1, n + 1):
        d_t = np.array([cond.fy(t, k_, Yn, Theta, model) for k_ in range(1, m + 2)])
        log_fy_lag1 = np.log((F1[t - 1] * d_t).sum())
        log_fys_lag1[t - 1] = log_fy_lag1

    return log_fys_lag1.sum()

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
    p_iis = np.diag(P)[:-1]
    f = lambda p_ii: st.beta(a, b).logpdf(p_ii)
    f_array = np.frompyfunc(f, 1, 1)

    return f_array(p_iis).sum()

def log_posterior_Theta_single(Theta, Yn, Sn, model):
    '''
    Calculate ln pi(Theta* | Yn, Sn) = \sum ln pi(theta_k* | Yn, Sn)
    '''
    ks = np.arange(1, len(Theta) + 1)

    f = lambda k: cond.Theta_conditional(k, Yn, Sn, model).logpdf(Theta[k - 1])
    f_array = np.frompyfunc(f, 1, 1)

    return f_array(ks).sum()

def log_posterior_Theta(Theta, Yn, Sns, model):
    '''
    Calculate ln pi(Theta* | Yn) = ln \mean pi(Theta* | Yn, Sn)
    '''
    f = lambda Sn: log_posterior_Theta_single(Theta, Yn, Sn, model)
    return np.log(np.exp(np.apply_along_axis(f, axis=0, arr=Sns)).mean())

def log_posterior_pii_single(i, P, Sn, a, b):
    '''
    Calculate ln pi(p_ii | Sn)
    '''
    p_ii = P[i - 1, i - 1]
    n_ii = np.sum(Sn == i) - 1
    return st.beta(a + n_ii, b + 1).logpdf(p_ii)

def posterior_P_single(P, Sn, a, b):
    '''
    Calculate pi(P | Sn) = \prod pi(p_ii | Sn) = exp(\sum ln pi(p_ii | Sn))
    '''
    m = len(np.unique(Sn)) - 1
    i = np.arange(1, m + 1)
    f = lambda i: log_posterior_pii_single(i, P, Sn, a, b)
    f_array = np.frompyfunc(f, 1, 1)
    
    return np.exp(f_array(i).sum())

def log_posterior_P(P, Sns, a, b):
    '''
    Calculate ln p(P* | Yn, Theta*) = ln 1/G \sum_j pi(P | S_{n,j})
    '''
    f = lambda Sn: posterior_P_single(P, Sn, a, b)
    return np.log(np.apply_along_axis(f, axis=0, arr=Sns).mean())