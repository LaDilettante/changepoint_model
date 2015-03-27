import changepoint.init_functions as init
import changepoint.full_conditionals as cond
import numpy as np
import scipy.stats as st

# Mock real parameters
thetas = np.array([0.5, 0.75, 0.25])
ns = np.array([50, 50, 50])
n = np.sum(ns) # number of periods
m = 2 # number of change points

# Mock data
Yn = np.array([st.bernoulli.rvs(theta_, size=n_) for theta_, n_ in zip(thetas, ns)]) \
    .ravel()
# Mock parameters of full conditionals
Sn = init.S(n, m)
#P = init.P(m + 1)

def test_P_has_correct_structure():
    '''
    Test if P has the correct transition matrix structure for change point problem
    '''
    # Prior params
    a = 8
    b = 0.1
    
    P = cond.P_sampling(Sn, a, b)
    print P

    assert( P[-1, -1] == 1 )
    assert( (np.diag(P) != 0).all() )
    assert( (np.diag(P, k=1) != 0).all() )
    np.testing.assert_almost_equal( np.diag(P, k=-1), 0 )