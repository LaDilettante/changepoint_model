# import changepoint.init_functions as init
# import changepoint.full_conditionals as cond
# import numpy as np
# import scipy.stats as st

# # Mock real parameters
# thetas = np.array([0.5, 0.75, 0.25])
# ns = np.array([50, 50, 50])
# n = np.sum(ns) # number of periods
# m = 2 # number of change points

# # Mock data
# Yn = np.array([st.bernoulli.rvs(theta_, size=n_) for theta_, n_ in zip(thetas, ns)]) \
#     .ravel()
# # Mock parameters of full conditionals
# Sn = init.S(n, m)
# #P = init.P(m + 1)

# def test_theta_leq_than_1():
#     '''
#     All returned theta must be leq than 1.
#     '''
#     assert( (cond.Theta_sampling(Yn, Sn) <= 1).all() )

# def test_theta_correct_number():
#     '''
#     The number of returned thetas = (# of regime) = (# of change point + 1)
#     '''
#     assert( len(cond.Theta_sampling(Yn, Sn)) == m + 1 )