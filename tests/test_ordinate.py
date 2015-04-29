from changepoint.ordinate import *

def test_log_likelihood():
    Yn = np.array([3, 4, 2, 3, 5, 4, 2, 4, 7, 3])
    Theta = np.array([3, 5])
    P = np.array([0.6, 0.4], [0, 1])

def test_log_prior_Theta():
    Theta = np.array([2, 3])

    true = st.gamma(2, scale=1).logpdf(Theta[0]) + st.gamma(2, scale=1).logpdf(Theta[1])
    calculated = log_prior_Theta(Theta, model="poisson")

    np.testing.assert_equal( true, calculated )

def test_log_prior_P():
    P = np.array([[0.2, 0.8, 0], [0, 0.3, 0.7], [0, 0, 1]])
    a = 8 ; b = 0.1
    p_iis = np.diag(P)

    # True prior density is sum_i ln pi(p_ii). Skip the last p_ii = 1
    true = st.beta(a, b).logpdf(p_iis[0]) + \
        st.beta(a, b).logpdf(p_iis[1])

    calculated = log_prior_P(P, a, b)
 
    np.testing.assert_equal( true, calculated )

def test_log_posterior_Theta_single():
    model = "poisson"
    Yn = np.array([3, 4, 2, 3, 5, 4, 2, 4, 7, 3])
    Sn = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    Theta = np.array([3, 5])

    true = 0
    for k in range(1, 3):
        true += cond.Theta_conditional(k, Yn, Sn, model).logpdf(Theta[k - 1])
    calculated = log_posterior_Theta_single(Theta, Yn, Sn, model)

    np.testing.assert_equal( true, calculated )

def test_log_posterior_pii_single():
    a = 8 ; b = 0.1 # Prior params
    Sn = np.array([1, 1, 1, 1, 2, 2, 2])
    P = np.array([[0.4, 0.6], [0, 1]])
    i = 1
    n_ii = np.sum(Sn == i) - 1

    true = st.beta(a + n_ii, b + 1).logpdf(P[i - 1, i - 1])
    calculated = log_posterior_pii_single(i, P, Sn, a, b)

    np.testing.assert_equal( true, calculated )

def test_log_posterior_P_single():
    a = 8 ; b = 0.1 # Prior params
    Sn = np.array([1, 1, 1, 1, 2, 2, 2, 3])
    P = np.array([[0.7, 0.3, 0], [0, 0.6, 0.4], [0, 0, 1]])
    
    true = 0 # Sum the log of pi(p_ii | Sn)
    for i in range(1, 3):
        n_ii = np.sum(Sn == i) - 1
        p_ii = P[i - 1, i - 1]
        true += st.beta(a + n_ii, b + 1).logpdf(p_ii)
    true = np.exp(true)
    calculated = posterior_P_single(P, Sn, a, b)

    np.testing.assert_equal( true, calculated )