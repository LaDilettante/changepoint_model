import changepoint.init_functions as init
import changepoint.full_conditionals as cond
import changepoint.full_conditionals_opt as cond_opt
import numpy as np

def test_Nk():
    Sn = np.array([1, 2, 1, 2, 2])
    ks = np.array([1, 2])

    true = np.array([2, 3])
    calculated = cond_opt.Nk(Sn)

    np.testing.assert_equal( true, calculated )

def test_Uk():
    Sn = np.array([1, 2, 1, 2, 2])
    Yn = np.array([3, 1, 4, 2, 6])

    true = np.array([7, 9])
    calculated = cond_opt.Uk(Yn, Sn)

    np.testing.assert_equal( true, calculated )

def test_P_sampling_square():
    Sn = np.array([1, 2, 1, 2, 2])
    P = cond_opt.P_sampling(Sn, a=8, b=0.1)
    np.testing.assert_equal( P.shape[0], P.shape[1] )

def test_P_sampling_diagonal():
    Sn = init.S(12, 4)
    P = cond_opt.P_sampling(Sn, a=8, b=0.1)
    assert np.sum(np.diag(P) == 0) == 0

def test_P_sampling_diagonal_offset1():
    Sn = init.S(10, 3)
    P = cond_opt.P_sampling(Sn, a=8, b=0.1)
    assert np.sum(np.diag(P, k=1) == 0) == 0    

def test_S_sampling():
    pass

